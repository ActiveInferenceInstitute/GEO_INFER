"""
Tests for CacheManager and CacheEntry in geo_infer_data.utils.caching.
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta, UTC
from unittest import mock

import pytest
from geo_infer_data.utils.caching import CacheEntry, CacheManager
from geo_infer_data.utils.secure_serialization import MAGIC

# Explicit key so persistence tests never touch the per-installation key file.
SIGNING_KEY = b"data-cache-unit-test-signing-key!"


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_not_expired_without_ttl(self):
        entry = CacheEntry(key="k", data="v")
        assert entry.is_expired() is False

    def test_not_expired_within_ttl(self):
        entry = CacheEntry(key="k", data="v", ttl=3600)
        assert entry.is_expired() is False

    def test_expired_entry(self):
        entry = CacheEntry(
            key="k",
            data="v",
            ttl=1,
            created_at=datetime.now(UTC) - timedelta(seconds=5),
        )
        assert entry.is_expired() is True

    def test_update_access(self):
        entry = CacheEntry(key="k", data="v")
        assert entry.access_count == 0
        entry.update_access()
        assert entry.access_count == 1

    def test_legacy_naive_timestamp_is_normalised(self):
        entry = CacheEntry(
            key="k",
            data="v",
            ttl=1,
            created_at=datetime.now() - timedelta(seconds=5),
        )
        assert entry.is_expired() is True


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class TestCacheManager:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_set_and_get(self):
        cache = CacheManager(max_size=10)
        self._run(cache.set("key1", {"data": 42}))
        result = self._run(cache.get("key1"))
        assert result == {"data": 42}

    def test_zero_ttl_expires_immediately(self):
        cache = CacheManager(max_size=10, default_ttl=3600)
        self._run(cache.set("key1", "value", ttl=0))
        assert self._run(cache.get("key1")) is None

    def test_persistent_keys_stay_inside_cache_directory(self, tmp_path):
        cache = CacheManager(
            max_size=10,
            enable_persistence=True,
            persistence_path=tmp_path,
            signing_key=SIGNING_KEY,
        )
        self._run(cache.set("../../outside", "value"))
        files = list(tmp_path.glob("*.pkl"))
        assert len(files) == 1
        assert files[0].parent == tmp_path

    def test_persisted_entry_is_signed_and_reloads(self, tmp_path):
        cache = CacheManager(
            max_size=10,
            default_ttl=None,
            enable_persistence=True,
            persistence_path=tmp_path,
            signing_key=SIGNING_KEY,
        )
        self._run(cache.set("key1", {"value": 1}))

        cache_file = next(iter(tmp_path.glob("*.pkl")))
        assert cache_file.read_bytes().startswith(MAGIC)

        reloaded = CacheManager(
            max_size=10,
            default_ttl=None,
            enable_persistence=True,
            persistence_path=tmp_path,
            signing_key=SIGNING_KEY,
        )
        assert self._run(reloaded.get("key1")) == {"value": 1}

    def test_tampered_persisted_entry_is_rejected(self, tmp_path):
        cache = CacheManager(
            max_size=10,
            default_ttl=None,
            enable_persistence=True,
            persistence_path=tmp_path,
            signing_key=SIGNING_KEY,
        )
        self._run(cache.set("key1", {"value": 1}))

        cache_file = next(iter(tmp_path.glob("*.pkl")))
        blob = bytearray(cache_file.read_bytes())
        blob[-1] ^= 0x01
        cache_file.write_bytes(bytes(blob))

        reloaded = CacheManager(
            max_size=10,
            default_ttl=None,
            enable_persistence=True,
            persistence_path=tmp_path,
            signing_key=SIGNING_KEY,
        )
        assert reloaded.cache == {}
        assert not cache_file.exists()

    def test_invalid_max_size_is_rejected(self):
        try:
            CacheManager(max_size=0)
        except ValueError as exc:
            assert "max_size" in str(exc)
        else:
            raise AssertionError("CacheManager accepted max_size=0")

    def test_get_missing_key_returns_none(self):
        cache = CacheManager(max_size=10)
        result = self._run(cache.get("nonexistent"))
        assert result is None

    def test_delete_key(self):
        cache = CacheManager(max_size=10)
        self._run(cache.set("key1", "value"))
        deleted = self._run(cache.delete("key1"))
        assert deleted is True
        assert self._run(cache.get("key1")) is None

    def test_delete_nonexistent_returns_false(self):
        cache = CacheManager(max_size=10)
        assert self._run(cache.delete("missing")) is False

    def test_clear(self):
        cache = CacheManager(max_size=10)
        self._run(cache.set("a", 1))
        self._run(cache.set("b", 2))
        self._run(cache.clear())
        assert self._run(cache.get("a")) is None
        assert self._run(cache.get("b")) is None

    def test_expired_entry_returns_none(self):
        cache = CacheManager(max_size=10, default_ttl=1)
        self._run(cache.set("k", "v", ttl=1))
        # Manually expire
        cache.cache["k"].created_at = datetime.now(UTC) - timedelta(seconds=10)
        result = self._run(cache.get("k"))
        assert result is None

    def test_lru_eviction(self):
        cache = CacheManager(max_size=3)
        self._run(cache.set("a", 1))
        self._run(cache.set("b", 2))
        self._run(cache.set("c", 3))
        # Cache is full, adding one more should evict the oldest
        self._run(cache.set("d", 4))
        # At least one early key should be evicted
        remaining = sum(
            1 for k in ["a", "b", "c", "d"] if self._run(cache.get(k)) is not None
        )
        assert remaining <= 3

    def test_stats(self):
        cache = CacheManager(max_size=10)
        self._run(cache.set("k", "v"))
        self._run(cache.get("k"))
        self._run(cache.get("missing"))
        stats = cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["total_sets"] == 1
        assert stats["hit_rate"] == 0.5

    def test_generate_cache_key(self):
        cache = CacheManager()
        key = cache.generate_cache_key(
            spatial_bounds=[-122.0, 37.0, -121.0, 38.0],
            temporal_range=(datetime(2023, 1, 1), datetime(2023, 6, 1)),
        )
        assert "spatial_" in key
        assert "temporal_" in key

    def test_generate_cache_key_hashing_for_long_keys(self):
        cache = CacheManager()
        key = cache.generate_cache_key(
            spatial_bounds=[-122.0, 37.0, -121.0, 38.0],
            temporal_range=(datetime(2023, 1, 1), datetime(2023, 6, 1)),
            query_params={f"param_{i}": f"value_{i}" for i in range(20)},
        )
        # Long keys get hashed
        assert key.startswith("hash_") or len(key) <= 200

    def test_optimize_cache(self):
        cache = CacheManager(max_size=5, default_ttl=1)
        for i in range(5):
            self._run(cache.set(f"k{i}", i))
        # Expire all
        for entry in cache.cache.values():
            entry.created_at = datetime.now(UTC) - timedelta(seconds=10)
        cache.optimize_cache()
        assert len(cache.cache) == 0


# ---------------------------------------------------------------------------
# Memory accounting (PL-06)
# ---------------------------------------------------------------------------


class TestCacheMemoryAccounting:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_stats_expose_estimated_entries_for_unpicklable_data(self):
        cache = CacheManager(max_size=10)
        self._run(cache.set("picklable", {"data": 42}))
        self._run(cache.set("unpicklable", threading.Lock()))
        stats = cache.get_stats()
        assert stats["estimated_entries"] == 1
        assert stats["current_size"] == 2

    def test_unpicklable_entry_logs_warning(self, caplog):
        cache = CacheManager(max_size=10)
        self._run(cache.set("unpicklable", threading.Lock()))
        with caplog.at_level(logging.WARNING, logger="geo_infer_data.utils.caching"):
            cache.get_stats()
        assert any(
            "unpicklable cache entry" in record.getMessage()
            for record in caplog.records
        )

    def test_unexpected_pickle_errors_propagate(self):
        cache = CacheManager(max_size=10)
        self._run(cache.set("boom", object()))
        with mock.patch(
            "geo_infer_data.utils.caching.pickle.dumps",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                cache.get_stats()
