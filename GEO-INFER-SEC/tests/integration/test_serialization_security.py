"""
Cross-module serialization security integration tests.

GEO-INFER-DATA, GEO-INFER-GIT, and GEO-INFER-OPS each persist serialized
payloads (pickle to disk/object storage, JSON and pickle to Redis). Each ships
a ``secure_serialization`` module implementing the same GISP1 authenticated
envelope, and every cache and storage read is required to verify that envelope
with HMAC-SHA256 *before* a deserializer sees the bytes.

These tests own the security contract for that boundary:

* **Envelope parity** — the three implementations agree on the wire format and
  key derivation, so a payload signed by one module verifies in another.
* **HMAC validation** — correct payloads round-trip; a payload signed for a
  different layer (context) or under a different key does not.
* **Tampering detection** — flipping a byte anywhere in the envelope (payload,
  MAC, declared length, key id, version, algorithm) is detected.
* **Reject on corrupt/unsigned** — truncated, empty, random, and legacy bare
  pickle/JSON payloads are refused, and a hostile pickle placed in any cache
  or storage layer never reaches ``pickle.loads``.
"""

import base64
import gzip
import hashlib
import hmac
import importlib
import json
import os
import pickle
import stat
from pathlib import Path
from typing import Any

import pytest

from geo_infer_data.utils import secure_serialization as data_ss
from geo_infer_git.utils import secure_serialization as git_ss
from geo_infer_ops.core import secure_serialization as ops_ss

pytestmark = pytest.mark.integration

MODULES = (data_ss, git_ss, ops_ss)
MODULE_IDS = ("data", "git", "ops")

KEY = b"integration-test-master-key-32byt"
OTHER_KEY = b"a-completely-different-master-key"


# --------------------------------------------------------------------------- #
# Hostile payload used to prove deserialization never happens on rejection.
# --------------------------------------------------------------------------- #

EXECUTED: list[str] = []


def _mark_executed(tag: str) -> str:
    """Record that a hostile pickle reached the unpickler."""
    EXECUTED.append(tag)
    return tag


class HostilePayload:
    """A picklable object whose reconstruction has a side effect.

    Unpickling this calls :func:`_mark_executed`, which is exactly what a
    malicious pickle would do. Any layer that leaves ``EXECUTED`` empty proved
    it rejected the payload before deserializing it.
    """

    def __init__(self, tag: str = "hostile") -> None:
        self.tag = tag

    def __reduce__(self):
        return (_mark_executed, (self.tag,))


@pytest.fixture(autouse=True)
def _clear_executed():
    """Ensure each test observes only its own hostile-payload executions."""
    EXECUTED.clear()
    yield
    EXECUTED.clear()


def hostile_pickle(tag: str = "hostile") -> bytes:
    """Return a bare (unsigned) pickle stream with a side effect on load."""
    return pickle.dumps(HostilePayload(tag))


def assert_not_executed() -> None:
    """Assert no hostile payload was ever handed to the unpickler."""
    assert EXECUTED == [], f"hostile payload was deserialized: {EXECUTED}"


# --------------------------------------------------------------------------- #
# Fakes for the shared infrastructure the storage layers talk to.
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Minimal in-memory Redis stand-in covering the cache call surface."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}

    def ping(self) -> bool:
        return True

    def get(self, key):
        return self.store.get(_as_key(key))

    def set(self, key, value, **kwargs):
        name = _as_key(key)
        if kwargs.get("xx") and name not in self.store:
            return False
        self.store[name] = value
        self.ttls.pop(name, None)
        return True

    def setnx(self, key, value):
        name = _as_key(key)
        if name in self.store:
            return False
        self.store[name] = value
        return True

    def setex(self, key, seconds, value):
        name = _as_key(key)
        self.store[name] = value
        self.ttls[name] = int(seconds)
        return True

    def ttl(self, key):
        name = _as_key(key)
        if name not in self.store:
            return -2
        return self.ttls.get(name, -1)

    def expire(self, key, seconds):
        name = _as_key(key)
        if name not in self.store:
            return False
        self.ttls[name] = int(seconds)
        return True

    def exists(self, *keys):
        return sum(1 for key in keys if _as_key(key) in self.store)

    def delete(self, *keys):
        removed = 0
        for key in keys:
            name = _as_key(key)
            if self.store.pop(name, None) is not None:
                self.ttls.pop(name, None)
                removed += 1
        return removed

    def mget(self, keys):
        return [self.store.get(_as_key(key)) for key in keys]

    def keys(self, pattern="*"):
        prefix = pattern.rstrip("*")
        return [key for key in self.store if key.startswith(prefix)]

    def scan(self, cursor=0, match="*"):
        return 0, self.keys(match)

    def pipeline(self):
        return FakeRedisPipeline(self)


class FakeRedisPipeline:
    """Buffered pipeline that applies queued writes on ``execute``."""

    def __init__(self, client: "FakeRedis") -> None:
        self.client = client
        self.queued: list[tuple] = []

    def __enter__(self) -> "FakeRedisPipeline":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def set(self, key, value, **kwargs):
        self.queued.append(("set", (key, value), kwargs))
        return self

    def setex(self, key, seconds, value):
        self.queued.append(("setex", (key, seconds, value), {}))
        return self

    def execute(self) -> list[Any]:
        results = []
        for name, args, kwargs in self.queued:
            results.append(getattr(self.client, name)(*args, **kwargs))
        self.queued.clear()
        return results


def _as_key(key: Any) -> str:
    """Normalise a Redis key to str so bytes and str keys collide correctly."""
    return key.decode("utf-8") if isinstance(key, bytes) else str(key)


class FakeMinioObject:
    """Response object returned by :meth:`FakeMinio.get_object`."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.closed = True


class FakeMinio:
    """Minimal MinIO client stand-in backed by a dict."""

    instances: list["FakeMinio"] = []

    def __init__(self, endpoint, access_key=None, secret_key=None, secure=False):
        self.endpoint = endpoint
        self.buckets: set = set()
        self.objects: dict[str, bytes] = FakeMinio.shared_objects
        FakeMinio.instances.append(self)

    shared_objects: dict[str, bytes] = {}

    def bucket_exists(self, bucket) -> bool:
        return bucket in self.buckets

    def make_bucket(self, bucket) -> None:
        self.buckets.add(bucket)

    def put_object(self, bucket, name, stream, length=None, content_type=None):
        self.buckets.add(bucket)
        self.objects[f"{bucket}/{name}"] = stream.read()

    def get_object(self, bucket, name):
        return FakeMinioObject(self.objects[f"{bucket}/{name}"])

    def stat_object(self, bucket, name):
        if f"{bucket}/{name}" not in self.objects:
            raise KeyError(name)
        return object()

    def remove_object(self, bucket, name):
        self.objects.pop(f"{bucket}/{name}", None)


@pytest.fixture
def fake_minio(monkeypatch):
    """Install :class:`FakeMinio` in place of the real MinIO client."""
    import minio

    FakeMinio.instances = []
    FakeMinio.shared_objects = {}
    monkeypatch.setattr(minio, "Minio", FakeMinio)
    return FakeMinio


@pytest.fixture
def signing_env(tmp_path, monkeypatch):
    """Point key resolution at a throwaway key file for this test."""
    key_file = tmp_path / "keys" / "serialization.key"
    monkeypatch.delenv(data_ss.ENV_KEY, raising=False)
    monkeypatch.setenv(data_ss.ENV_KEY_FILE, str(key_file))
    for module in MODULES:
        module.clear_signing_key_cache()
    yield key_file
    for module in MODULES:
        module.clear_signing_key_cache()


def dataset_metadata(title: str = "serialization security fixture"):
    """Build a minimal valid :class:`DatasetMetadata`."""
    from geo_infer_data.models.schemas import DatasetMetadata

    return DatasetMetadata(
        title=title,
        lineage={
            "source": "integration-test",
            "process": "test-fixture",
            "created_by": "GEO-INFER-SEC",
        },
    )


def flip_byte(blob: bytes, index: int) -> bytes:
    """Return ``blob`` with one bit flipped at ``index``."""
    mutable = bytearray(blob)
    mutable[index] ^= 0x01
    return bytes(mutable)


# --------------------------------------------------------------------------- #
# A. Envelope parity across modules
# --------------------------------------------------------------------------- #


class TestEnvelopeParity:
    """The three module copies must implement one interoperable format."""

    def test_wire_constants_identical(self):
        for module in MODULES[1:]:
            assert module.MAGIC == data_ss.MAGIC
            assert module.ENVELOPE_VERSION == data_ss.ENVELOPE_VERSION
            assert module.ALG_HMAC_SHA256 == data_ss.ALG_HMAC_SHA256
            assert module.DIGEST_SIZE == data_ss.DIGEST_SIZE
            assert module.TEXT_PREFIX == data_ss.TEXT_PREFIX
            assert module.KEY_DERIVATION_LABEL == data_ss.KEY_DERIVATION_LABEL
            assert module.MIN_KEY_BYTES == data_ss.MIN_KEY_BYTES
            assert module.ENV_KEY == data_ss.ENV_KEY
            assert module.ENV_KEY_FILE == data_ss.ENV_KEY_FILE

    @pytest.mark.parametrize("producer", MODULES, ids=MODULE_IDS)
    @pytest.mark.parametrize("consumer", MODULES, ids=MODULE_IDS)
    def test_envelope_is_portable_between_modules(self, producer, consumer):
        envelope = producer.sign_payload(b"shared-bytes", context="shared", key=KEY)
        assert consumer.verify_payload(envelope, context="shared", key=KEY) == (
            b"shared-bytes"
        )

    def test_key_derivation_matches_documented_construction(self):
        derived = data_ss.derive_context_key(KEY, "data.cache.entry")
        expected = hmac.new(
            KEY,
            b"geo-infer/serialization/v1/data.cache.entry",
            hashlib.sha256,
        ).digest()
        assert derived == expected
        assert len(derived) == data_ss.DIGEST_SIZE

    def test_envelope_layout_is_header_mac_payload(self):
        payload = b"layout-check"
        envelope = data_ss.sign_payload(
            payload, context="layout", key=KEY, key_id="kid1"
        )
        assert envelope.startswith(data_ss.MAGIC)
        offset = len(data_ss.MAGIC)
        assert envelope[offset] == data_ss.ENVELOPE_VERSION
        assert envelope[offset + 1] == data_ss.ALG_HMAC_SHA256
        key_id_len = envelope[offset + 2]
        assert key_id_len == len(b"kid1")
        header_len = offset + 3 + key_id_len + 8
        header = envelope[:header_len]
        assert header[offset + 3 : offset + 3 + key_id_len] == b"kid1"
        assert int.from_bytes(header[-8:], "big") == len(payload)
        mac = envelope[header_len : header_len + data_ss.DIGEST_SIZE]
        assert envelope[header_len + data_ss.DIGEST_SIZE :] == payload
        context_key = data_ss.derive_context_key(KEY, "layout")
        assert mac == hmac.new(context_key, header + payload, hashlib.sha256).digest()

    def test_context_isolation_between_layers(self):
        envelope = data_ss.sign_payload(
            b"cache-only", context=data_ss.CONTEXT_CACHE_ENTRY, key=KEY
        )
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(
                envelope, context=data_ss.CONTEXT_STORAGE_REDIS, key=KEY
            )

    def test_declared_contexts_are_distinct(self):
        declared = [
            data_ss.CONTEXT_CACHE_ENTRY,
            data_ss.CONTEXT_STORAGE_MINIO,
            data_ss.CONTEXT_STORAGE_REDIS,
            data_ss.CONTEXT_STORAGE_LOCAL_FILE,
            data_ss.CONTEXT_STORAGE_GENERIC,
            git_ss.CONTEXT_DISK_CACHE,
            git_ss.CONTEXT_REDIS_CACHE,
            ops_ss.CONTEXT_REDIS_CACHE,
        ]
        assert len(set(declared)) == len(declared)


# --------------------------------------------------------------------------- #
# B. HMAC validation
# --------------------------------------------------------------------------- #


class TestHmacValidation:
    """Correct envelopes verify; anything else does not."""

    @pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
    @pytest.mark.parametrize(
        "value",
        [
            None,
            0,
            "text",
            {"nested": {"list": [1, 2, 3], "flag": True}},
            [1, "two", None],
        ],
    )
    def test_json_round_trip(self, module, value):
        envelope = module.dumps_signed(
            value, context="json-ctx", key=KEY, serializer="json"
        )
        assert (
            module.loads_signed(
                envelope, context="json-ctx", key=KEY, serializer="json"
            )
            == value
        )

    @pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
    def test_pickle_round_trip(self, module):
        value = {"tuple": (1, 2), "bytes": b"\x00\xff"}
        envelope = module.dumps_signed(value, context="pickle-ctx", key=KEY)
        assert module.loads_signed(envelope, context="pickle-ctx", key=KEY) == value

    @pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
    def test_text_envelope_round_trip(self, module):
        envelope = module.dumps_signed_text({"a": 1}, context="text-ctx", key=KEY)
        assert envelope.startswith(module.TEXT_PREFIX)
        assert module.loads_signed_text(envelope, context="text-ctx", key=KEY) == {
            "a": 1
        }

    def test_wrong_key_is_rejected(self):
        envelope = data_ss.sign_payload(b"secret", context="ctx", key=KEY)
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(envelope, context="ctx", key=OTHER_KEY)

    def test_key_id_is_authenticated(self):
        envelope = data_ss.sign_payload(
            b"payload", context="ctx", key=KEY, key_id="rotated"
        )
        offset = len(data_ss.MAGIC) + 3
        forged = envelope[:offset] + b"rotatee" + envelope[offset + 7 :]
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(forged, context="ctx", key=KEY)

    def test_short_key_is_refused(self):
        with pytest.raises(data_ss.SigningKeyUnavailableError):
            data_ss.sign_payload(b"x", context="ctx", key=b"tooshort")

    def test_non_bytes_payload_refused(self):
        with pytest.raises(TypeError):
            data_ss.sign_payload("not-bytes", context="ctx", key=KEY)

    def test_empty_context_refused(self):
        with pytest.raises(ValueError):
            data_ss.sign_payload(b"x", context="", key=KEY)

    def test_is_signed_envelope_discriminates(self):
        assert data_ss.is_signed_envelope(
            data_ss.sign_payload(b"x", context="c", key=KEY)
        )
        assert data_ss.is_signed_envelope(
            data_ss.sign_payload_text(b"x", context="c", key=KEY)
        )
        assert not data_ss.is_signed_envelope(pickle.dumps({"a": 1}))
        assert not data_ss.is_signed_envelope(json.dumps({"a": 1}))
        assert not data_ss.is_signed_envelope(None)


# --------------------------------------------------------------------------- #
# C. Tampering detection
# --------------------------------------------------------------------------- #


class TestTamperingDetection:
    """Every byte of the envelope is covered by the MAC or a structural check."""

    @pytest.fixture
    def envelope(self) -> bytes:
        return data_ss.sign_payload(
            b"the-quick-brown-fox-jumped", context="tamper", key=KEY, key_id="kid"
        )

    def test_every_byte_position_is_protected(self, envelope):
        """Flipping any single bit of the envelope must be caught."""
        for index in range(len(envelope)):
            mutated = flip_byte(envelope, index)
            with pytest.raises(data_ss.PayloadSecurityError):
                data_ss.verify_payload(mutated, context="tamper", key=KEY)

    def test_payload_tampering_detected(self, envelope):
        mutated = flip_byte(envelope, len(envelope) - 1)
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(mutated, context="tamper", key=KEY)

    def test_mac_tampering_detected(self, envelope):
        header_len = len(data_ss.MAGIC) + 3 + len(b"kid") + 8
        mutated = flip_byte(envelope, header_len)
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(mutated, context="tamper", key=KEY)

    def test_declared_length_tampering_detected(self, envelope):
        header_len = len(data_ss.MAGIC) + 3 + len(b"kid") + 8
        mutated = flip_byte(envelope, header_len - 1)
        with pytest.raises(data_ss.MalformedEnvelopeError):
            data_ss.verify_payload(mutated, context="tamper", key=KEY)

    def test_version_downgrade_detected(self, envelope):
        index = len(data_ss.MAGIC)
        mutated = bytearray(envelope)
        mutated[index] = 99
        with pytest.raises(data_ss.MalformedEnvelopeError):
            data_ss.verify_payload(bytes(mutated), context="tamper", key=KEY)

    def test_algorithm_downgrade_detected(self, envelope):
        index = len(data_ss.MAGIC) + 1
        mutated = bytearray(envelope)
        mutated[index] = 0
        with pytest.raises(data_ss.MalformedEnvelopeError):
            data_ss.verify_payload(bytes(mutated), context="tamper", key=KEY)

    def test_payload_extension_detected(self, envelope):
        with pytest.raises(data_ss.MalformedEnvelopeError):
            data_ss.verify_payload(envelope + b"extra", context="tamper", key=KEY)

    def test_payload_substitution_detected(self, envelope):
        """Swapping in a same-length hostile payload does not verify."""
        header_len = len(data_ss.MAGIC) + 3 + len(b"kid") + 8
        prefix = envelope[: header_len + data_ss.DIGEST_SIZE]
        original = envelope[header_len + data_ss.DIGEST_SIZE :]
        substitute = b"X" * len(original)
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(prefix + substitute, context="tamper", key=KEY)

    def test_text_envelope_tampering_detected(self):
        envelope = data_ss.sign_payload_text(b"text-payload", context="t", key=KEY)
        raw = bytearray(base64.urlsafe_b64decode(envelope[len(data_ss.TEXT_PREFIX) :]))
        raw[-1] ^= 0x01
        mutated = data_ss.TEXT_PREFIX + base64.urlsafe_b64encode(bytes(raw)).decode()
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload_text(mutated, context="t", key=KEY)


# --------------------------------------------------------------------------- #
# D. Reject-on-corrupt / unsigned
# --------------------------------------------------------------------------- #


class TestRejectUntrustedPayloads:
    """Unsigned, truncated, and garbage payloads never reach a deserializer."""

    @pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
    def test_bare_pickle_is_refused(self, module):
        with pytest.raises(module.UnsignedPayloadError):
            module.loads_signed(hostile_pickle(), context="any", key=KEY)
        assert_not_executed()

    @pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
    def test_bare_json_is_refused(self, module):
        with pytest.raises(module.UnsignedPayloadError):
            module.loads_signed_text(json.dumps({"a": 1}), context="any", key=KEY)

    @pytest.mark.parametrize(
        "blob",
        [b"", b"G", b"GISP", b"GISP1", b"GISP1\x01", b"not-an-envelope-at-all"],
    )
    def test_truncated_or_short_blobs_refused(self, blob):
        with pytest.raises(data_ss.MalformedEnvelopeError):
            data_ss.verify_payload(blob, context="ctx", key=KEY)

    def test_truncated_valid_envelope_refused(self):
        envelope = data_ss.sign_payload(b"a" * 64, context="ctx", key=KEY)
        for cut in (1, 8, 16, 32, 48):
            with pytest.raises(data_ss.PayloadSecurityError):
                data_ss.verify_payload(envelope[:-cut], context="ctx", key=KEY)

    def test_random_bytes_refused(self):
        with pytest.raises(data_ss.UnsignedPayloadError):
            data_ss.verify_payload(os.urandom(256), context="ctx", key=KEY)

    def test_non_bytes_envelope_refused(self):
        for blob in (None, 42, ["a"], {"a": 1}):
            with pytest.raises(data_ss.MalformedEnvelopeError):
                data_ss.verify_payload(blob, context="ctx", key=KEY)

    def test_text_envelope_requires_prefix(self):
        with pytest.raises(data_ss.UnsignedPayloadError):
            data_ss.verify_payload_text("plain text", context="ctx", key=KEY)

    def test_text_envelope_rejects_bad_base64(self):
        with pytest.raises(data_ss.MalformedEnvelopeError):
            data_ss.verify_payload_text(
                data_ss.TEXT_PREFIX + "!!!not base64!!!", context="ctx", key=KEY
            )

    def test_unsupported_serializer_refused(self):
        with pytest.raises(ValueError):
            data_ss.dumps_signed({}, context="ctx", key=KEY, serializer="yaml")


# --------------------------------------------------------------------------- #
# E. Key management
# --------------------------------------------------------------------------- #


class TestKeyManagement:
    """Key resolution is explicit, ordered, and fails closed."""

    def test_explicit_key_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv(data_ss.ENV_KEY, OTHER_KEY.hex())
        assert data_ss.resolve_signing_key(KEY) == KEY

    def test_environment_hex_key_is_decoded(self, monkeypatch):
        monkeypatch.setenv(data_ss.ENV_KEY, KEY.hex())
        assert data_ss.resolve_signing_key() == KEY

    def test_environment_short_key_is_refused(self, monkeypatch):
        monkeypatch.setenv(data_ss.ENV_KEY, "abcd")
        with pytest.raises(data_ss.SigningKeyUnavailableError):
            data_ss.resolve_signing_key()

    def test_key_file_is_created_owner_only(self, signing_env):
        assert not signing_env.exists()
        resolved = data_ss.resolve_signing_key()
        assert signing_env.exists()
        assert len(resolved) >= data_ss.MIN_KEY_BYTES
        mode = stat.S_IMODE(signing_env.stat().st_mode)
        assert mode == 0o600
        # Second resolution reuses the persisted key, so envelopes survive a
        # process restart.
        assert data_ss.resolve_signing_key() == resolved

    def test_key_file_is_shared_across_modules(self, signing_env):
        envelope = data_ss.sign_payload(b"cross", context="ctx")
        assert git_ss.verify_payload(envelope, context="ctx") == b"cross"
        assert ops_ss.verify_payload(envelope, context="ctx") == b"cross"

    def test_short_key_file_is_refused(self, signing_env):
        signing_env.parent.mkdir(parents=True, exist_ok=True)
        signing_env.write_bytes(b"short")
        with pytest.raises(data_ss.SigningKeyUnavailableError):
            data_ss.resolve_signing_key()

    def test_rotated_key_file_invalidates_old_envelopes(self, signing_env):
        envelope = data_ss.sign_payload(b"before-rotation", context="ctx")
        signing_env.write_bytes(os.urandom(32))
        data_ss.clear_signing_key_cache()
        with pytest.raises(data_ss.SignatureMismatchError):
            data_ss.verify_payload(envelope, context="ctx")


# --------------------------------------------------------------------------- #
# F. GEO-INFER-DATA cache layer
# --------------------------------------------------------------------------- #


class TestDataCacheManagerPersistence:
    """``geo_infer_data.utils.caching.CacheManager`` file persistence."""

    @staticmethod
    def manager(path: Path):
        from geo_infer_data.utils.caching import CacheManager

        return CacheManager(
            max_size=32,
            default_ttl=None,
            enable_persistence=True,
            persistence_path=path,
            signing_key=KEY,
        )

    async def test_persisted_entry_is_signed_not_bare_pickle(self, tmp_path):
        cache = self.manager(tmp_path)
        await cache.set("alpha", {"value": 1})

        files = list(tmp_path.glob("*.pkl"))
        assert len(files) == 1
        blob = files[0].read_bytes()
        assert blob.startswith(data_ss.MAGIC)
        with pytest.raises(pickle.UnpicklingError):
            pickle.loads(blob)

        payload = data_ss.verify_payload(
            blob, context=data_ss.CONTEXT_CACHE_ENTRY, key=KEY
        )
        assert pickle.loads(payload)["data"] == {"value": 1}

    async def test_signed_entry_reloads_across_restart(self, tmp_path):
        cache = self.manager(tmp_path)
        await cache.set("alpha", {"value": 1})

        reloaded = self.manager(tmp_path)
        assert await reloaded.get("alpha") == {"value": 1}

    async def test_tampered_entry_is_rejected_and_quarantined(self, tmp_path, caplog):
        cache = self.manager(tmp_path)
        await cache.set("alpha", {"value": 1})
        cache_file = next(iter(tmp_path.glob("*.pkl")))
        cache_file.write_bytes(flip_byte(cache_file.read_bytes(), -1))

        with caplog.at_level("ERROR"):
            reloaded = self.manager(tmp_path)

        assert await reloaded.get("alpha") is None
        assert not cache_file.exists()
        assert "Rejected untrusted cache entry" in caplog.text

    async def test_unsigned_legacy_file_is_rejected(self, tmp_path):
        """A bare pickle dropped into the cache directory is never loaded."""
        legacy = tmp_path / f"{hashlib.sha256(b'evil').hexdigest()}.pkl"
        tmp_path.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(hostile_pickle("data-cache"))

        reloaded = self.manager(tmp_path)

        assert reloaded.cache == {}
        assert not legacy.exists()
        assert_not_executed()

    async def test_wrong_key_cannot_read_persisted_entries(self, tmp_path):
        from geo_infer_data.utils.caching import CacheManager

        cache = self.manager(tmp_path)
        await cache.set("alpha", {"value": 1})

        attacker = CacheManager(
            enable_persistence=True,
            persistence_path=tmp_path,
            signing_key=OTHER_KEY,
        )
        assert attacker.cache == {}


# --------------------------------------------------------------------------- #
# G. GEO-INFER-DATA storage backends
# --------------------------------------------------------------------------- #


class TestDataLocalFileBackend:
    """Local-file backend pickles are authenticated on disk."""

    @staticmethod
    def backend(tmp_path: Path):
        from geo_infer_data.core.storage import LocalFileBackend

        return LocalFileBackend({"base_path": str(tmp_path), "signing_key": KEY})

    async def test_round_trip(self, tmp_path):
        backend = self.backend(tmp_path)
        data_id = await backend.store({"payload": [1, 2, 3]}, dataset_metadata())
        assert await backend.retrieve(data_id, {}) == {"payload": [1, 2, 3]}

    async def test_stored_file_is_an_envelope(self, tmp_path):
        backend = self.backend(tmp_path)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        pkl = next(iter(tmp_path.rglob(f"{data_id}.pkl")))
        assert pkl.read_bytes().startswith(data_ss.MAGIC)

    async def test_tampered_file_rejected_before_unpickling(self, tmp_path):
        backend = self.backend(tmp_path)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        pkl = next(iter(tmp_path.rglob(f"{data_id}.pkl")))
        pkl.write_bytes(flip_byte(pkl.read_bytes(), -1))

        with pytest.raises(data_ss.SignatureMismatchError):
            await backend.retrieve(data_id, {})

    async def test_hostile_pickle_replacement_never_executes(self, tmp_path):
        backend = self.backend(tmp_path)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        pkl = next(iter(tmp_path.rglob(f"{data_id}.pkl")))
        pkl.write_bytes(hostile_pickle("local-file"))

        with pytest.raises(data_ss.UnsignedPayloadError):
            await backend.retrieve(data_id, {})
        assert_not_executed()

    async def test_cross_context_envelope_rejected(self, tmp_path):
        """An envelope signed for a different layer does not verify here."""
        backend = self.backend(tmp_path)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        pkl = next(iter(tmp_path.rglob(f"{data_id}.pkl")))
        pkl.write_bytes(
            data_ss.dumps_signed(
                {"payload": "swapped"},
                context=data_ss.CONTEXT_STORAGE_REDIS,
                key=KEY,
            )
        )
        with pytest.raises(data_ss.SignatureMismatchError):
            await backend.retrieve(data_id, {})


class TestDataMinioBackend:
    """MinIO payloads are verified before decompression or unpickling."""

    @staticmethod
    def backend():
        from geo_infer_data.core.storage import MinIOBackend

        return MinIOBackend(
            {
                "endpoint": "minio.invalid:9000",
                "access_key": "key",
                "secret_key": "secret",
                "bucket": "test-bucket",
                "signing_key": KEY,
            }
        )

    async def test_round_trip(self, fake_minio):
        backend = self.backend()
        data_id = await backend.store({"payload": [1, 2, 3]}, dataset_metadata())
        assert await backend.retrieve(data_id, {}) == {"payload": [1, 2, 3]}

    async def test_object_bytes_are_an_envelope(self, fake_minio):
        backend = self.backend()
        await backend.store({"payload": 1}, dataset_metadata())
        stored = list(fake_minio.shared_objects.values())
        assert len(stored) == 1
        assert stored[0].startswith(data_ss.MAGIC)

    async def test_tampered_object_rejected(self, fake_minio):
        backend = self.backend()
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        name = f"test-bucket/{data_id}.bin"
        fake_minio.shared_objects[name] = flip_byte(fake_minio.shared_objects[name], -1)
        with pytest.raises(data_ss.SignatureMismatchError):
            await backend.retrieve(data_id, {})

    async def test_hostile_object_never_deserialized(self, fake_minio):
        backend = self.backend()
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        name = f"test-bucket/{data_id}.bin"
        fake_minio.shared_objects[name] = gzip.compress(hostile_pickle("minio"))
        with pytest.raises(data_ss.UnsignedPayloadError):
            await backend.retrieve(data_id, {})
        assert_not_executed()

    async def test_verification_precedes_decompression(self, fake_minio):
        """A gzip bomb is rejected on its MAC, never inflated."""
        backend = self.backend()
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        name = f"test-bucket/{data_id}.bin"
        bomb = gzip.compress(b"\x00" * (8 * 1024 * 1024))
        assert len(bomb) < 100_000
        fake_minio.shared_objects[name] = bomb
        with pytest.raises(data_ss.UnsignedPayloadError):
            await backend.retrieve(data_id, {})


class TestDataRedisBackend:
    """Redis dataset payloads are authenticated."""

    @staticmethod
    def backend(monkeypatch):
        import redis

        from geo_infer_data.core.storage import RedisBackend

        monkeypatch.setattr(redis, "Redis", lambda **kwargs: FakeRedis())
        return RedisBackend({"signing_key": KEY})

    async def test_round_trip(self, monkeypatch):
        backend = self.backend(monkeypatch)
        data_id = await backend.store({"payload": [1, 2]}, dataset_metadata())
        assert await backend.retrieve(data_id, {}) == {"payload": [1, 2]}

    async def test_stored_value_is_an_envelope(self, monkeypatch):
        backend = self.backend(monkeypatch)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        assert backend.client.store[data_id].startswith(data_ss.MAGIC)

    async def test_tampered_value_rejected(self, monkeypatch):
        backend = self.backend(monkeypatch)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        backend.client.store[data_id] = flip_byte(backend.client.store[data_id], -1)
        with pytest.raises(data_ss.SignatureMismatchError):
            await backend.retrieve(data_id, {})

    async def test_hostile_value_never_deserialized(self, monkeypatch):
        backend = self.backend(monkeypatch)
        data_id = await backend.store({"payload": 1}, dataset_metadata())
        backend.client.store[data_id] = hostile_pickle("data-redis")
        with pytest.raises(data_ss.UnsignedPayloadError):
            await backend.retrieve(data_id, {})
        assert_not_executed()

    async def test_missing_key_still_raises_file_not_found(self, monkeypatch):
        backend = self.backend(monkeypatch)
        with pytest.raises(FileNotFoundError):
            await backend.retrieve("absent", {})


class TestDataGenericTablePayload:
    """The generic key-value table decodes JSON and signed pickle only."""

    @staticmethod
    def backend():
        from geo_infer_data.core.storage import PostgreSQLBackend

        return PostgreSQLBackend(
            {
                "user": "u",
                "password": "p",
                "host": "localhost",
                "port": 5432,
                "database": "d",
                "signing_key": KEY,
            }
        )

    def test_json_payload_decodes(self):
        assert self.backend().decode_generic_payload('{"a": 1}') == {"a": 1}

    def test_signed_pickle_payload_decodes(self):
        signed = data_ss.sign_payload_text(
            pickle.dumps({"a": (1, 2)}),
            context=data_ss.CONTEXT_STORAGE_GENERIC,
            key=KEY,
        )
        assert self.backend().decode_generic_payload(signed) == {"a": (1, 2)}

    def test_tampered_signed_payload_rejected(self):
        signed = data_ss.sign_payload_text(
            pickle.dumps({"a": 1}),
            context=data_ss.CONTEXT_STORAGE_GENERIC,
            key=KEY,
        )
        tampered = signed[:-2] + ("A" if signed[-2] != "A" else "B") + signed[-1]
        with pytest.raises(data_ss.PayloadSecurityError):
            self.backend().decode_generic_payload(tampered)

    def test_hex_pickle_row_is_not_decoded(self):
        """The legacy hex-pickle encoding is no longer accepted."""
        with pytest.raises(ValueError):
            self.backend().decode_generic_payload(hostile_pickle("generic").hex())
        assert_not_executed()

    def test_non_string_payload_rejected(self):
        with pytest.raises(ValueError):
            self.backend().decode_generic_payload(b"{}")


# --------------------------------------------------------------------------- #
# H. GEO-INFER-GIT cache layers
# --------------------------------------------------------------------------- #


class TestGitDiskCache:
    """SQLite-backed disk cache files are authenticated envelopes."""

    @staticmethod
    def cache(tmp_path: Path, compression: bool = True, key=KEY):
        from geo_infer_git.utils.advanced_cache import DiskCache

        return DiskCache(
            tmp_path, max_size_gb=0.01, compression=compression, signing_key=key
        )

    @staticmethod
    def cache_files(tmp_path: Path) -> list[Path]:
        return sorted(tmp_path.glob("*.cache"))

    @pytest.mark.parametrize("compression", [True, False])
    def test_round_trip(self, tmp_path, compression):
        cache = self.cache(tmp_path, compression=compression)
        cache.put("k", {"payload": [1, 2, 3]})
        assert cache.get("k") == {"payload": [1, 2, 3]}

    @pytest.mark.parametrize("compression", [True, False])
    def test_cache_file_is_an_envelope(self, tmp_path, compression):
        cache = self.cache(tmp_path, compression=compression)
        cache.put("k", {"payload": 1})
        blob = self.cache_files(tmp_path)[0].read_bytes()
        assert blob.startswith(git_ss.MAGIC)
        payload = git_ss.verify_payload(
            blob, context=git_ss.CONTEXT_DISK_CACHE, key=KEY
        )
        if compression:
            payload = gzip.decompress(payload)
        assert pickle.loads(payload) == {"payload": 1}

    def test_tampered_file_is_evicted_and_reported_as_miss(self, tmp_path, caplog):
        cache = self.cache(tmp_path)
        cache.put("k", {"payload": 1})
        path = self.cache_files(tmp_path)[0]
        path.write_bytes(flip_byte(path.read_bytes(), -1))

        with caplog.at_level("ERROR"):
            assert cache.get("k", default="MISS") == "MISS"

        assert "Rejected untrusted disk cache entry" in caplog.text
        assert not path.exists()
        assert cache.get("k", default="MISS") == "MISS"

    def test_hostile_file_never_unpickled(self, tmp_path):
        cache = self.cache(tmp_path)
        cache.put("k", {"payload": 1})
        path = self.cache_files(tmp_path)[0]
        path.write_bytes(gzip.compress(hostile_pickle("git-disk")))

        assert cache.get("k", default="MISS") == "MISS"
        assert_not_executed()

    def test_gzip_bomb_rejected_before_decompression(self, tmp_path):
        cache = self.cache(tmp_path)
        cache.put("k", {"payload": 1})
        path = self.cache_files(tmp_path)[0]
        bomb = gzip.compress(b"\x00" * (8 * 1024 * 1024))
        assert len(bomb) < 100_000
        path.write_bytes(bomb)
        assert cache.get("k", default="MISS") == "MISS"

    def test_foreign_key_cannot_read_entries(self, tmp_path):
        cache = self.cache(tmp_path)
        cache.put("k", {"payload": 1})
        attacker = self.cache(tmp_path, key=OTHER_KEY)
        assert attacker.get("k", default="MISS") == "MISS"

    def test_cache_filenames_use_sha256(self, tmp_path):
        cache = self.cache(tmp_path)
        cache.put("named-key", {"payload": 1})
        expected = hashlib.sha256(b"named-key").hexdigest()
        assert self.cache_files(tmp_path)[0].stem == expected


class TestGitRedisCache:
    """Redis cache documents carry a MAC that survives access updates."""

    @pytest.fixture
    def redis_cache(self, monkeypatch):
        import redis

        from geo_infer_git.utils.advanced_cache import RedisCache

        fake = FakeRedis()
        monkeypatch.setattr(redis, "ConnectionPool", lambda **kwargs: object())
        monkeypatch.setattr(redis, "Redis", lambda **kwargs: fake)
        cache = RedisCache(signing_key=KEY)
        cache.fake = fake
        return cache

    def test_round_trip(self, redis_cache):
        redis_cache.put("k", {"payload": [1, 2]})
        assert redis_cache.get("k") == {"payload": [1, 2]}

    def test_stored_document_is_a_text_envelope(self, redis_cache):
        redis_cache.put("k", {"payload": 1})
        assert redis_cache.fake.store["k"].startswith(git_ss.TEXT_PREFIX)
        with pytest.raises(json.JSONDecodeError):
            json.loads(redis_cache.fake.store["k"])

    def test_access_update_is_resigned(self, redis_cache):
        redis_cache.put("k", {"payload": 1})
        first = redis_cache.fake.store["k"]
        assert redis_cache.get("k") == {"payload": 1}
        second = redis_cache.fake.store["k"]
        assert second != first
        document = git_ss.loads_signed_text(
            second, context=git_ss.CONTEXT_REDIS_CACHE, key=KEY
        )
        assert document["access_count"] == 1

    def test_access_update_preserves_ttl(self, redis_cache):
        redis_cache.put("k", {"payload": 1}, ttl_seconds=120)
        assert redis_cache.fake.ttl("k") == 120
        assert redis_cache.get("k") == {"payload": 1}
        assert redis_cache.fake.ttl("k") == 120

    def test_tampered_document_is_evicted(self, redis_cache, caplog):
        redis_cache.put("k", {"payload": 1})
        raw = bytearray(
            base64.urlsafe_b64decode(
                redis_cache.fake.store["k"][len(git_ss.TEXT_PREFIX) :]
            )
        )
        raw[-1] ^= 0x01
        redis_cache.fake.store["k"] = (
            git_ss.TEXT_PREFIX + base64.urlsafe_b64encode(bytes(raw)).decode()
        )

        with caplog.at_level("ERROR"):
            assert redis_cache.get("k", default="MISS") == "MISS"

        assert "Rejected untrusted Redis cache entry" in caplog.text
        assert "k" not in redis_cache.fake.store

    def test_unsigned_document_is_evicted(self, redis_cache):
        redis_cache.put("k", {"payload": 1})
        redis_cache.fake.store["k"] = json.dumps(
            {"key": "k", "value": "injected", "access_count": 0}
        )
        assert redis_cache.get("k", default="MISS") == "MISS"
        assert "k" not in redis_cache.fake.store

    def test_foreign_key_cannot_read_documents(self, redis_cache, monkeypatch):
        redis_cache.put("k", {"payload": 1})
        redis_cache.signing_key = OTHER_KEY
        assert redis_cache.get("k", default="MISS") == "MISS"


# --------------------------------------------------------------------------- #
# I. GEO-INFER-OPS cache layer
# --------------------------------------------------------------------------- #


class TestOpsCacheManager:
    """OPS Redis cache signs both JSON and pickle payloads."""

    @staticmethod
    def manager(monkeypatch, serializer=None, **kwargs):
        from unittest.mock import Mock

        import geo_infer_ops.core.cache as ops_cache

        config = Mock()
        config.cache.redis.host = "localhost"
        config.cache.redis.port = 6379
        config.cache.redis.db = 0
        config.cache.redis.password = None
        monkeypatch.setattr(ops_cache, "get_config", lambda: config)

        fake = FakeRedis()
        monkeypatch.setattr(ops_cache.redis, "Redis", lambda **kw: fake)

        serializer = serializer or ops_cache.CacheSerializer.JSON
        manager = ops_cache.CacheManager(
            serializer=serializer, signing_key=KEY, **kwargs
        )
        manager.fake = fake
        return manager

    def test_json_payload_is_text_envelope(self, monkeypatch):
        manager = self.manager(monkeypatch)
        serialized = manager._serialize({"key": "value"})
        assert isinstance(serialized, str)
        assert serialized.startswith(ops_ss.TEXT_PREFIX)
        with pytest.raises(json.JSONDecodeError):
            json.loads(serialized)
        assert manager._deserialize(serialized) == {"key": "value"}

    def test_pickle_payload_is_binary_envelope(self, monkeypatch):
        import geo_infer_ops.core.cache as ops_cache

        manager = self.manager(monkeypatch, serializer=ops_cache.CacheSerializer.PICKLE)
        serialized = manager._serialize({"key": (1, 2)})
        assert isinstance(serialized, bytes)
        assert serialized.startswith(ops_ss.MAGIC)
        with pytest.raises(pickle.UnpicklingError):
            pickle.loads(serialized)
        assert manager._deserialize(serialized) == {"key": (1, 2)}

    def test_set_get_round_trip(self, monkeypatch):
        manager = self.manager(monkeypatch)
        assert manager.set("k", {"a": 1}) is True
        assert manager.get("k") == {"a": 1}
        assert manager.fake.store["geo_infer:k"].startswith(ops_ss.TEXT_PREFIX)

    def test_unsigned_value_rejected_and_evicted(self, monkeypatch, caplog):
        manager = self.manager(monkeypatch)
        manager.fake.store["geo_infer:k"] = json.dumps({"injected": True})

        with caplog.at_level("ERROR"):
            assert manager.get("k", default="MISS") == "MISS"

        assert "geo_infer:k" not in manager.fake.store

    def test_tampered_value_rejected_and_evicted(self, monkeypatch):
        manager = self.manager(monkeypatch)
        manager.set("k", {"a": 1})
        stored = manager.fake.store["geo_infer:k"]
        raw = bytearray(base64.urlsafe_b64decode(stored[len(ops_ss.TEXT_PREFIX) :]))
        raw[-1] ^= 0x01
        manager.fake.store["geo_infer:k"] = ops_ss.TEXT_PREFIX + (
            base64.urlsafe_b64encode(bytes(raw)).decode()
        )
        assert manager.get("k", default="MISS") == "MISS"
        assert "geo_infer:k" not in manager.fake.store

    def test_strict_mode_raises_on_untrusted(self, monkeypatch):
        manager = self.manager(monkeypatch, raise_on_untrusted=True)
        manager.fake.store["geo_infer:k"] = json.dumps({"injected": True})
        with pytest.raises(ops_ss.PayloadSecurityError):
            manager.get("k")
        assert "geo_infer:k" not in manager.fake.store

    def test_hostile_pickle_value_never_deserialized(self, monkeypatch):
        import geo_infer_ops.core.cache as ops_cache

        manager = self.manager(monkeypatch, serializer=ops_cache.CacheSerializer.PICKLE)
        manager.set("k", {"a": 1})
        manager.fake.store["geo_infer:k"] = hostile_pickle("ops-redis")
        assert manager.get("k", default="MISS") == "MISS"
        assert_not_executed()

    def test_get_many_skips_untrusted_entries(self, monkeypatch):
        manager = self.manager(monkeypatch)
        manager.set("good", {"a": 1})
        manager.fake.store["geo_infer:bad"] = json.dumps({"injected": True})

        result = manager.get_many(["good", "bad", "absent"])

        assert result == {"good": {"a": 1}}
        assert "geo_infer:bad" not in manager.fake.store

    def test_set_many_signs_every_value(self, monkeypatch):
        manager = self.manager(monkeypatch)
        manager.set_many({"a": 1, "b": 2})
        for key in ("geo_infer:a", "geo_infer:b"):
            assert manager.fake.store[key].startswith(ops_ss.TEXT_PREFIX)
        assert manager.get_many(["a", "b"]) == {"a": 1, "b": 2}

    def test_counters_stay_outside_the_envelope(self, monkeypatch):
        """INCRBY/DECRBY operate on native integers and never deserialize."""
        manager = self.manager(monkeypatch)
        manager.fake.store["geo_infer:counter"] = 0

        def incrby(key, amount):
            manager.fake.store[_as_key(key)] += amount
            return manager.fake.store[_as_key(key)]

        manager.fake.incrby = incrby
        assert manager.increment("counter", 5) == 5

    def test_cross_layer_envelope_rejected(self, monkeypatch):
        """A DATA cache envelope cannot be replayed into the OPS cache."""
        manager = self.manager(monkeypatch)
        manager.fake.store["geo_infer:k"] = data_ss.dumps_signed_text(
            {"injected": True},
            context=data_ss.CONTEXT_CACHE_ENTRY,
            key=KEY,
        )
        assert manager.get("k", default="MISS") == "MISS"


# --------------------------------------------------------------------------- #
# J. Whole-surface audit
# --------------------------------------------------------------------------- #


class TestSerializationSurfaceAudit:
    """No hardened layer may reach a deserializer outside the trust boundary."""

    HARDENED_SOURCES = (
        ("geo_infer_data.utils.caching", {"pickle.dumps"}),
        ("geo_infer_data.core.storage", {"pickle.dumps", "pickle.loads"}),
        ("geo_infer_git.utils.advanced_cache", {"pickle.dumps", "pickle.loads"}),
        ("geo_infer_ops.core.cache", set()),
    )

    @pytest.mark.parametrize(
        "module_name,allowed",
        HARDENED_SOURCES,
        ids=[name for name, _ in HARDENED_SOURCES],
    )
    def test_no_unreviewed_deserializer_calls(self, module_name, allowed):
        """Every direct pickle call site in the hardened modules is accounted for.

        ``pickle.load``/``pickle.loads`` may only appear where an envelope has
        already been verified; the allowlist records the reviewed call forms so
        a newly introduced bare ``pickle.load(file)`` fails this test.
        """
        source = Path(importlib.import_module(module_name).__file__).read_text(
            encoding="utf-8"
        )
        found = {
            call
            for call in (
                "pickle.load(",
                "pickle.loads(",
                "pickle.dump(",
                "pickle.dumps(",
            )
            if call in source
        }
        normalised = {call.rstrip("(") for call in found}
        assert normalised <= allowed, (
            f"{module_name} calls {sorted(normalised - allowed)} directly; "
            "route it through secure_serialization or add a reviewed exemption"
        )

    def test_hardened_modules_import_the_trust_boundary(self):
        for module_name, _ in self.HARDENED_SOURCES:
            source = Path(importlib.import_module(module_name).__file__).read_text(
                encoding="utf-8"
            )
            assert "secure_serialization" in source, module_name

    @pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
    def test_verify_is_the_only_route_to_payload_bytes(self, module):
        """``loads_signed`` never returns for a payload that fails the MAC."""
        envelope = bytearray(module.dumps_signed({"a": 1}, context="audit", key=KEY))
        envelope[-1] ^= 0x01
        with pytest.raises(module.SignatureMismatchError):
            module.loads_signed(bytes(envelope), context="audit", key=KEY)
