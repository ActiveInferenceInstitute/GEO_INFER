"""
Caching utilities for GEO-INFER-DATA.

This module provides caching capabilities for frequently accessed data
including in-memory caching, file-based caching, and distributed caching.

Security note (serialization trust boundary): persisted cache entries are
serialised with :mod:`pickle` and wrapped in an authenticated GISP1 envelope
(see :mod:`geo_infer_data.utils.secure_serialization`). ``pickle.loads`` can
execute arbitrary code during unpickling, so the persistence layer never
unpickles a file directly: every read passes HMAC-SHA256 verification under
the ``data.cache.entry`` context key first, and any file that is unsigned,
truncated, cross-context, or tampered with is rejected and quarantined before
deserialisation is attempted.
"""

import logging
from typing import Any
from datetime import datetime, timedelta, UTC
import hashlib
import pickle
from pathlib import Path

from .secure_serialization import (
    CONTEXT_CACHE_ENTRY,
    PayloadSecurityError,
    dumps_signed,
    loads_signed,
)


logger = logging.getLogger(__name__)


def _normalise_timestamp(value: datetime | None) -> datetime:
    """Return a timezone-aware UTC timestamp for cache entries."""
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


class CacheEntry:
    """Cache entry with metadata."""

    def __init__(
        self,
        key: str,
        data: Any,
        ttl: int | None = None,
        created_at: datetime | None = None,
        access_count: int = 0,
        last_accessed: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.key = key
        self.data = data
        self.ttl = ttl  # Time to live in seconds
        self.created_at = _normalise_timestamp(created_at)
        self.access_count = access_count
        self.last_accessed = _normalise_timestamp(last_accessed)
        self.metadata = metadata or {}

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return datetime.now(UTC) - self.created_at > timedelta(seconds=self.ttl)

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)


class CacheManager:
    """
    Comprehensive caching manager for geospatial data.

    This class provides caching capabilities including in-memory caching,
    file-based persistence, and cache optimization strategies.

    Args:
        max_size: Maximum cache size in entries
        default_ttl: Default time-to-live for cache entries
        enable_persistence: Whether to enable file-based persistence
        persistence_path: Path for cache persistence
        signing_key: Explicit HMAC master key for persisted entries. When
            omitted the key is resolved from ``GEO_INFER_SERIALIZATION_KEY``
            or the per-installation key file.

    Examples:
        >>> cache = CacheManager(max_size=1000, default_ttl=3600)
        >>>
        >>> # Cache data
        >>> await cache.set('key1', geodataframe, ttl=1800)
        >>>
        >>> # Retrieve data
        >>> data = await cache.get('key1')
        >>>
        >>> # Get cache statistics
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2f}")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = 3600,
        enable_persistence: bool = False,
        persistence_path: Path | None = None,
        signing_key: bytes | str | None = None,
    ):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if default_ttl is not None and default_ttl < 0:
            raise ValueError("default_ttl must be non-negative or None")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or Path("/tmp/geo_infer_cache")
        self.signing_key = signing_key

        self.cache: dict[str, CacheEntry] = {}
        self.access_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

        if self.enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()

        logger.info(
            f"Initialized CacheManager with max_size={max_size}, persistence={enable_persistence}"
        )

    async def set(
        self,
        key: str,
        data: Any,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Set cache entry.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            True if successful
        """
        if ttl is not None and ttl < 0:
            raise ValueError("ttl must be non-negative or None")

        if len(self.cache) >= self.max_size:
            # Remove expired entries first
            self._cleanup_expired()

            # If still full, remove least recently used
            if len(self.cache) >= self.max_size:
                self._evict_lru()

        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            ttl=self.default_ttl if ttl is None else ttl,
            metadata=metadata or {},
        )

        self.cache[key] = entry
        self.access_stats["sets"] += 1

        # Persist if enabled
        if self.enable_persistence:
            self._persist_entry(entry)

        logger.debug(f"Cached data for key: {key}")
        return True

    async def get(self, key: str) -> Any | None:
        """
        Get cache entry.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        if key in self.cache:
            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                await self.delete(key)
                self.access_stats["misses"] += 1
                return None

            # Update access statistics
            entry.update_access()
            self.access_stats["hits"] += 1

            logger.debug(f"Cache hit for key: {key}")
            return entry.data
        else:
            self.access_stats["misses"] += 1
            logger.debug(f"Cache miss for key: {key}")
            return None

    async def delete(self, key: str) -> bool:
        """
        Delete cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if key in self.cache:
            del self.cache[key]
            self.access_stats["deletes"] += 1

            # Remove from persistence
            if self.enable_persistence:
                cache_file = self._cache_file(key)
                if cache_file.exists():
                    cache_file.unlink()

            logger.debug(f"Deleted cache entry: {key}")
            return True

        return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

        if self.enable_persistence:
            for cache_file in self.persistence_path.glob("*.pkl"):
                cache_file.unlink()

        logger.info("Cache cleared")

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self.cache[key]
            if self.enable_persistence:
                cache_file = self._cache_file(key)
                if cache_file.exists():
                    cache_file.unlink()

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        # Sort by last accessed time
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].last_accessed)

        # Remove oldest 10% or at least 1 entry
        entries_to_remove = max(1, len(self.cache) // 10)

        for key, entry in sorted_entries[:entries_to_remove]:
            del self.cache[key]
            if self.enable_persistence:
                cache_file = self._cache_file(key)
                if cache_file.exists():
                    cache_file.unlink()

        logger.debug(f"Evicted {entries_to_remove} LRU entries")

    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist cache entry to file."""
        cache_file = self._cache_file(entry.key)

        # Serialize entry
        entry_data = {
            "key": entry.key,
            "data": entry.data,
            "ttl": entry.ttl,
            "created_at": entry.created_at,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
            "metadata": entry.metadata,
        }

        # Sign before writing: the on-disk bytes are an authenticated GISP1
        # envelope, never a bare pickle stream.
        envelope = dumps_signed(
            entry_data,
            context=CONTEXT_CACHE_ENTRY,
            key=self.signing_key,
            serializer="pickle",
        )
        cache_file.write_bytes(envelope)

    def _cache_file(self, key: str) -> Path:
        """Return a filesystem-safe path for a logical cache key.

        Cache keys may contain slashes, traversal segments, or characters that
        are invalid on a target filesystem. Hashing keeps the logical key in
        the serialized payload while guaranteeing that persistence stays
        inside ``persistence_path``.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.persistence_path / f"{digest}.pkl"

    def _load_persistent_cache(self) -> None:
        """Load persistent cache from files."""
        if not self.persistence_path.exists():
            return

        for cache_file in self.persistence_path.glob("*.pkl"):
            try:
                # Trust boundary: HMAC verification runs before unpickling, so
                # an unsigned or tampered file is rejected, not deserialised.
                entry_data = loads_signed(
                    cache_file.read_bytes(),
                    context=CONTEXT_CACHE_ENTRY,
                    key=self.signing_key,
                    serializer="pickle",
                )

                entry = CacheEntry(
                    key=entry_data["key"],
                    data=entry_data["data"],
                    ttl=entry_data["ttl"],
                    created_at=entry_data["created_at"],
                    access_count=entry_data["access_count"],
                    last_accessed=entry_data["last_accessed"],
                    metadata=entry_data.get("metadata", {}),
                )

                # Check if entry is still valid
                if not entry.is_expired():
                    self.cache[entry.key] = entry
                else:
                    # Remove expired file
                    cache_file.unlink()

            except PayloadSecurityError as e:
                logger.error(
                    "Rejected untrusted cache entry %s before deserialization: %s",
                    cache_file,
                    e,
                )
                cache_file.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to load cache entry {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.access_stats["hits"] + self.access_stats["misses"]

        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = self.access_stats["hits"] / total_requests

        # Calculate memory usage estimate
        memory_usage = 0
        estimated_entries = 0
        for key, entry in self.cache.items():
            try:
                # Size estimation only: this pickles in-memory data and never
                # deserialises, so it is outside the trust boundary.
                memory_usage += len(pickle.dumps(entry.data))
            except (TypeError, OSError, pickle.PicklingError) as e:
                logger.warning(
                    "Memory accounting: unpicklable cache entry %r (%s); "
                    "using the 1KB fallback estimate",
                    key,
                    type(e).__name__,
                )
                memory_usage += 1000  # Estimate 1KB per entry
                estimated_entries += 1

        return {
            "max_size": self.max_size,
            "current_size": len(self.cache),
            "hit_rate": hit_rate,
            "total_hits": self.access_stats["hits"],
            "total_misses": self.access_stats["misses"],
            "total_sets": self.access_stats["sets"],
            "total_deletes": self.access_stats["deletes"],
            "estimated_memory_usage_mb": memory_usage / (1024 * 1024),
            "estimated_entries": estimated_entries,
            "persistence_enabled": self.enable_persistence,
        }

    def generate_cache_key(
        self,
        spatial_bounds: list[float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate cache key from query parameters.

        Args:
            spatial_bounds: Spatial query bounds
            temporal_range: Temporal query range
            query_params: Additional query parameters

        Returns:
            Generated cache key
        """
        key_components = []

        if spatial_bounds:
            key_components.append(f"spatial_{'_'.join(map(str, spatial_bounds))}")

        if temporal_range:
            start_str = temporal_range[0].isoformat() if temporal_range[0] else "none"
            end_str = temporal_range[1].isoformat() if temporal_range[1] else "none"
            key_components.append(f"temporal_{start_str}_{end_str}")

        if query_params:
            # Sort parameters for consistent key generation
            sorted_params = sorted(query_params.items())
            param_str = "_".join(f"{k}_{v}" for k, v in sorted_params)
            key_components.append(f"params_{param_str}")

        # Generate hash for long keys
        key_string = "_".join(key_components)
        if len(key_string) > 100:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"hash_{key_hash}"
        else:
            return key_string

    def optimize_cache(self) -> None:
        """Optimize cache performance."""
        # Remove expired entries
        self._cleanup_expired()

        # If still too large, remove least recently used
        while len(self.cache) > self.max_size * 0.9:  # Keep 90% capacity
            self._evict_lru()

        logger.info(f"Cache optimized: {len(self.cache)}/{self.max_size} entries")
