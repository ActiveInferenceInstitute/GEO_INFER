"""
Authenticated serialization envelopes for GEO-INFER-AI.

This module is the trust boundary for every persisted model payload in
GEO-INFER-AI: ``ModelTrainer._save_model``/``load_model`` in
:mod:`geo_infer_ai.core.training` (joblib and pickle branches).

GISP1 wire format (identical in every GEO-INFER module that persists
serialized payloads, so a payload written by one layer is verifiable by any
other layer holding the same key)::

    header  = b"GISP1" | version(1) | alg(1) | key_id_len(1) | key_id
              | payload_len(8, big-endian)
    mac     = HMAC-SHA256(derived_key, header || payload)
    envelope= header || mac || payload

The MAC covers the header, so the declared payload length, the MAC algorithm,
and the key identifier are all authenticated: truncation, algorithm
downgrade, and key-id substitution are all detected. ``derived_key`` is
``HMAC-SHA256(master_key, b"geo-infer/serialization/v1/" + context)``, which
domain-separates every cache and storage layer — an envelope minted for one
layer will not verify in another.

The master key resolves from an explicit argument, then
``GEO_INFER_SERIALIZATION_KEY``, then the key file named by
``GEO_INFER_SERIALIZATION_KEY_FILE`` (default
``~/.config/geo-infer/serialization.key``), created with owner-only
permissions on first use. Callers that pass a key explicitly never touch the
environment.

Trust boundary: ``verify_payload`` raises on every failure path and is the
only route to a deserializer. An unsigned, truncated, cross-context, or
tampered payload never reaches ``pickle.loads`` or ``json.loads``.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import pickle
import secrets
import stat
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAGIC = b"GISP1"
ENVELOPE_VERSION = 1
ALG_HMAC_SHA256 = 1
DIGEST_SIZE = 32
TEXT_PREFIX = "gisp1:"
MIN_KEY_BYTES = 16
DEFAULT_KEY_ID = "local"
KEY_DERIVATION_LABEL = b"geo-infer/serialization/v1/"

CONTEXT_MODEL_SAVE = "ai.training.model"

ENV_KEY = "GEO_INFER_SERIALIZATION_KEY"
ENV_KEY_FILE = "GEO_INFER_SERIALIZATION_KEY_FILE"
ENV_KEY_ID = "GEO_INFER_SERIALIZATION_KEY_ID"

_HEADER_PREFIX_LEN = len(MAGIC) + 3  # magic + version + alg + key_id length
_LENGTH_FIELD_LEN = 8

_key_cache_lock = threading.Lock()
_key_file_cache: dict[tuple[str, int, int], bytes] = {}


class PayloadSecurityError(Exception):
    """Base class for authenticated-serialization failures."""


class SigningKeyUnavailableError(PayloadSecurityError):
    """No usable signing key could be resolved."""


class MalformedEnvelopeError(PayloadSecurityError):
    """The payload is structurally not a valid GISP1 envelope."""


class UnsignedPayloadError(MalformedEnvelopeError):
    """The payload carries no GISP1 envelope at all."""


class SignatureMismatchError(PayloadSecurityError):
    """The envelope MAC does not match the payload under the trusted key."""


def clear_signing_key_cache() -> None:
    """Drop cached key-file reads so a rotated key file is picked up."""
    with _key_cache_lock:
        _key_file_cache.clear()


def default_key_path() -> Path:
    """Return the per-installation signing key path."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "geo-infer" / "serialization.key"


def _coerce_key(material: bytes | bytearray | memoryview | str) -> bytes:
    """Normalise supplied key material to raw bytes."""
    if isinstance(material, str):
        stripped = material.strip()
        try:
            decoded = bytes.fromhex(stripped)
        except ValueError:
            decoded = stripped.encode("utf-8")
        return decoded
    if isinstance(material, (bytearray, memoryview)):
        return bytes(material)
    if isinstance(material, bytes):
        return material
    raise SigningKeyUnavailableError(
        f"Signing key material must be bytes or str, got {type(material).__name__}"
    )


def _read_key_file(path: Path) -> bytes:
    """Read a signing key file, caching by (path, mtime, size)."""
    file_stat = path.stat()
    cache_key = (str(path), file_stat.st_mtime_ns, file_stat.st_size)
    with _key_cache_lock:
        cached = _key_file_cache.get(cache_key)
    if cached is not None:
        return cached
    material = _coerce_key(path.read_bytes())
    if len(material) < MIN_KEY_BYTES:
        raise SigningKeyUnavailableError(
            f"Signing key file {path} holds {len(material)} bytes; "
            f"at least {MIN_KEY_BYTES} are required"
        )
    if file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            "Signing key file %s is group/world accessible; tighten to 0600", path
        )
    with _key_cache_lock:
        _key_file_cache[cache_key] = material
    return material


def _create_key_file(path: Path) -> bytes:
    """Create a new random signing key file with owner-only permissions."""
    material = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, material)
    finally:
        os.close(descriptor)
    logger.info("Created serialization signing key at %s", path)
    return material


def resolve_signing_key(key: bytes | str | None = None) -> bytes:
    """
    Resolve the master signing key for authenticated serialization.

    Resolution order: explicit argument, ``GEO_INFER_SERIALIZATION_KEY``,
    then the key file named by ``GEO_INFER_SERIALIZATION_KEY_FILE`` (default
    ``~/.config/geo-infer/serialization.key``), which is created with
    owner-only permissions on first use.

    Args:
        key: Explicit key material, bypassing environment resolution.

    Returns:
        Master key bytes.

    Raises:
        SigningKeyUnavailableError: If no key of sufficient length is usable.
    """
    if key is not None:
        material = _coerce_key(key)
        if len(material) < MIN_KEY_BYTES:
            raise SigningKeyUnavailableError(
                f"Signing key must be at least {MIN_KEY_BYTES} bytes, got {len(material)}"
            )
        return material

    env_value = os.environ.get(ENV_KEY)
    if env_value:
        material = _coerce_key(env_value)
        if len(material) < MIN_KEY_BYTES:
            raise SigningKeyUnavailableError(
                f"{ENV_KEY} must decode to at least {MIN_KEY_BYTES} bytes, got {len(material)}"
            )
        return material

    key_file = os.environ.get(ENV_KEY_FILE)
    path = Path(key_file) if key_file else default_key_path()
    try:
        if path.exists():
            return _read_key_file(path)
        return _create_key_file(path)
    except FileExistsError:
        return _read_key_file(path)
    except PayloadSecurityError:
        raise
    except OSError as exc:
        raise SigningKeyUnavailableError(
            f"Could not resolve a signing key from {path}: {exc}"
        ) from exc


def derive_context_key(master_key: bytes, context: str) -> bytes:
    """
    Derive a context-scoped subkey so envelopes cannot cross trust domains.

    Args:
        master_key: Master signing key bytes.
        context: Domain-separation label, e.g. ``"data.cache.entry"``.

    Returns:
        32-byte derived key.
    """
    if not context:
        raise ValueError("context must be a non-empty string")
    return hmac.new(
        master_key, KEY_DERIVATION_LABEL + context.encode("utf-8"), hashlib.sha256
    ).digest()


def _resolve_key_id(key_id: str | None) -> bytes:
    """Return the encoded key identifier bound into the envelope header."""
    resolved = key_id or os.environ.get(ENV_KEY_ID) or DEFAULT_KEY_ID
    encoded = resolved.encode("utf-8")
    if not encoded or len(encoded) > 255:
        raise ValueError("key_id must encode to between 1 and 255 bytes")
    return encoded


def _build_header(key_id: bytes, payload_len: int) -> bytes:
    """Assemble the authenticated envelope header."""
    return (
        MAGIC
        + bytes((ENVELOPE_VERSION, ALG_HMAC_SHA256, len(key_id)))
        + key_id
        + payload_len.to_bytes(_LENGTH_FIELD_LEN, "big")
    )


def is_signed_envelope(blob: Any) -> bool:
    """Return True when ``blob`` starts with the GISP1 envelope magic."""
    if isinstance(blob, str):
        return blob.startswith(TEXT_PREFIX)
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return bytes(blob)[: len(MAGIC)] == MAGIC
    return False


def sign_payload(
    payload: bytes,
    *,
    context: str,
    key: bytes | str | None = None,
    key_id: str | None = None,
) -> bytes:
    """
    Wrap raw serialized bytes in an authenticated GISP1 envelope.

    Args:
        payload: Serialized bytes to protect.
        context: Domain-separation label for the producing layer.
        key: Optional explicit master key.
        key_id: Optional key identifier recorded in the header.

    Returns:
        ``header || hmac_sha256 || payload`` envelope bytes.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like, got {type(payload).__name__}")
    payload_bytes = bytes(payload)
    encoded_key_id = _resolve_key_id(key_id)
    header = _build_header(encoded_key_id, len(payload_bytes))
    context_key = derive_context_key(resolve_signing_key(key), context)
    mac = hmac.new(context_key, header + payload_bytes, hashlib.sha256).digest()
    return header + mac + payload_bytes


def verify_payload(
    envelope: bytes | bytearray | memoryview,
    *,
    context: str,
    key: bytes | str | None = None,
) -> bytes:
    """
    Verify a GISP1 envelope and return its payload.

    This is the trust boundary: it must succeed before any payload reaches a
    deserializer. Every failure mode raises rather than returning data.

    Args:
        envelope: Envelope bytes as produced by :func:`sign_payload`.
        context: Domain-separation label the envelope was signed under.
        key: Optional explicit master key.

    Returns:
        The verified payload bytes.

    Raises:
        UnsignedPayloadError: The blob carries no envelope magic.
        MalformedEnvelopeError: The envelope is truncated or inconsistent.
        SignatureMismatchError: The MAC does not match under the trusted key.
    """
    if not isinstance(envelope, (bytes, bytearray, memoryview)):
        raise MalformedEnvelopeError(
            f"Envelope must be bytes-like, got {type(envelope).__name__}"
        )
    blob = bytes(envelope)
    if len(blob) < _HEADER_PREFIX_LEN:
        raise UnsignedPayloadError(
            f"Payload is {len(blob)} bytes, too short to carry a GISP1 envelope"
        )
    if blob[: len(MAGIC)] != MAGIC:
        raise UnsignedPayloadError(
            "Payload is not GISP1-signed; refusing to deserialize"
        )

    version = blob[len(MAGIC)]
    algorithm = blob[len(MAGIC) + 1]
    key_id_len = blob[len(MAGIC) + 2]
    if version != ENVELOPE_VERSION:
        raise MalformedEnvelopeError(f"Unsupported envelope version {version}")
    if algorithm != ALG_HMAC_SHA256:
        raise MalformedEnvelopeError(f"Unsupported MAC algorithm {algorithm}")

    header_len = _HEADER_PREFIX_LEN + key_id_len + _LENGTH_FIELD_LEN
    if len(blob) < header_len + DIGEST_SIZE:
        raise MalformedEnvelopeError("Envelope header or MAC is truncated")

    header = blob[:header_len]
    declared_len = int.from_bytes(header[-_LENGTH_FIELD_LEN:], "big")
    mac = blob[header_len : header_len + DIGEST_SIZE]
    payload = blob[header_len + DIGEST_SIZE :]
    if len(payload) != declared_len:
        raise MalformedEnvelopeError(
            f"Envelope declares {declared_len} payload bytes but carries {len(payload)}"
        )

    context_key = derive_context_key(resolve_signing_key(key), context)
    expected = hmac.new(context_key, header + payload, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise SignatureMismatchError(
            f"HMAC verification failed for context {context!r}; refusing to deserialize"
        )
    return payload


def sign_payload_text(
    payload: bytes,
    *,
    context: str,
    key: bytes | str | None = None,
    key_id: str | None = None,
) -> str:
    """Return a text-safe GISP1 envelope for string-mode transports."""
    envelope = sign_payload(payload, context=context, key=key, key_id=key_id)
    return TEXT_PREFIX + base64.urlsafe_b64encode(envelope).decode("ascii")


def verify_payload_text(
    envelope: str,
    *,
    context: str,
    key: bytes | str | None = None,
) -> bytes:
    """Verify a text-mode GISP1 envelope and return its payload bytes."""
    if not isinstance(envelope, str):
        raise MalformedEnvelopeError(
            f"Text envelope must be str, got {type(envelope).__name__}"
        )
    if not envelope.startswith(TEXT_PREFIX):
        raise UnsignedPayloadError(
            "Payload is not a GISP1 text envelope; refusing to deserialize"
        )
    encoded = envelope[len(TEXT_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise MalformedEnvelopeError(
            f"Text envelope is not valid base64: {exc}"
        ) from exc
    return verify_payload(raw, context=context, key=key)


def _serialize_object(obj: Any, serializer: str) -> bytes:
    """Serialize ``obj`` with the named serializer."""
    if serializer == "json":
        return json.dumps(obj).encode("utf-8")
    if serializer == "pickle":
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    raise ValueError(f"Unsupported serializer {serializer!r}")


def _deserialize_object(payload: bytes, serializer: str) -> Any:
    """Deserialize verified ``payload`` with the named serializer."""
    if serializer == "json":
        return json.loads(payload.decode("utf-8"))
    if serializer == "pickle":
        return pickle.loads(payload)
    raise ValueError(f"Unsupported serializer {serializer!r}")


def dumps_signed(
    obj: Any,
    *,
    context: str,
    key: bytes | str | None = None,
    serializer: str = "pickle",
    key_id: str | None = None,
) -> bytes:
    """Serialize ``obj`` and return an authenticated binary envelope."""
    return sign_payload(
        _serialize_object(obj, serializer),
        context=context,
        key=key,
        key_id=key_id,
    )


def loads_signed(
    envelope: bytes | bytearray | memoryview,
    *,
    context: str,
    key: bytes | str | None = None,
    serializer: str = "pickle",
) -> Any:
    """Verify an authenticated binary envelope, then deserialize its payload."""
    return _deserialize_object(
        verify_payload(envelope, context=context, key=key), serializer
    )


def dumps_signed_text(
    obj: Any,
    *,
    context: str,
    key: bytes | str | None = None,
    serializer: str = "json",
    key_id: str | None = None,
) -> str:
    """Serialize ``obj`` and return an authenticated text envelope."""
    return sign_payload_text(
        _serialize_object(obj, serializer),
        context=context,
        key=key,
        key_id=key_id,
    )


def loads_signed_text(
    envelope: str,
    *,
    context: str,
    key: bytes | str | None = None,
    serializer: str = "json",
) -> Any:
    """Verify an authenticated text envelope, then deserialize its payload."""
    return _deserialize_object(
        verify_payload_text(envelope, context=context, key=key), serializer
    )
