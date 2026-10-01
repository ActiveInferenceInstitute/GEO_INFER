#!/usr/bin/env python3

"""
Advanced caching strategies for GEO-INFER-GIT.

This module provides sophisticated caching functionality including:
- Redis-based distributed caching
- Multi-level caching (memory, disk, distributed)
- Intelligent cache invalidation strategies
- Cache analytics and optimization
- Adaptive cache sizing and eviction policies

Security note (serialization trust boundary): :class:`DiskCache` persists
pickled values to disk and :class:`RedisCache` persists JSON documents to a
shared Redis instance. Both are untrusted input on the way back in — a disk
cache file or Redis value can be rewritten by anything with write access, and
``pickle.loads`` executes arbitrary code. Every payload is therefore written
as an authenticated GISP1 envelope (see
:mod:`geo_infer_git.utils.secure_serialization`) and every read verifies the
envelope's HMAC-SHA256 under a per-layer context key before deserialisation.
For the disk cache the envelope wraps the *compressed* bytes, so a hostile
file is rejected before it is even decompressed. :class:`MemoryCache` holds
live objects and never round-trips through a deserialiser, so it sits inside
the trust boundary.
"""

import json
import time
import hashlib
import threading
import queue
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta, UTC
import pickle
import sqlite3
from abc import ABC, abstractmethod

from ..utils.logging_utils import get_logger
from .secure_serialization import (
    CONTEXT_DISK_CACHE,
    CONTEXT_REDIS_CACHE,
    PayloadSecurityError,
    dumps_signed_text,
    loads_signed_text,
    sign_payload,
    verify_payload,
)

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=_utc_now)
    accessed_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime | None = None
    access_count: int = 0
    size_bytes: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    def access(self) -> None:
        """Mark the entry as accessed."""
        self.accessed_at = datetime.now(UTC)
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Get age of the cache entry in seconds."""
        return (datetime.now(UTC) - self.created_at).total_seconds()


@dataclass
class CacheStatistics:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    average_access_time_ms: float = 0.0
    last_reset: datetime = field(default_factory=_utc_now)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.last_reset = datetime.now(UTC)


class CachePolicy(ABC):
    """Base class for cache eviction policies."""

    def __init__(self, max_size: int, max_age_seconds: int | None = None):
        """
        Initialize cache policy.

        Args:
            max_size: Maximum number of entries
            max_age_seconds: Maximum age of entries in seconds
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds

    @abstractmethod
    def should_evict(self, entries: dict[str, CacheEntry]) -> list[str]:
        """
        Determine which entries should be evicted.

        Args:
            entries: Dictionary of cache entries

        Returns:
            List of keys to evict
        """
        ...


class LRUPolicy(CachePolicy):
    """Least Recently Used eviction policy."""

    def should_evict(self, entries: dict[str, CacheEntry]) -> list[str]:
        """Evict least recently used entries."""
        if len(entries) <= self.max_size:
            return []

        # Sort by access time (oldest first)
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].accessed_at)
        evict_count = len(entries) - self.max_size

        return [key for key, _ in sorted_entries[:evict_count]]


class LFUPolicy(CachePolicy):
    """Least Frequently Used eviction policy."""

    def should_evict(self, entries: dict[str, CacheEntry]) -> list[str]:
        """Evict least frequently used entries."""
        if len(entries) <= self.max_size:
            return []

        # Sort by access count (lowest first)
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].access_count)
        evict_count = len(entries) - self.max_size

        return [key for key, _ in sorted_entries[:evict_count]]


class TTLPolicy(CachePolicy):
    """Time To Live eviction policy."""

    def should_evict(self, entries: dict[str, CacheEntry]) -> list[str]:
        """Evict expired entries."""
        expired_keys = []

        for key, entry in entries.items():
            if entry.is_expired():
                expired_keys.append(key)

        return expired_keys


class AdaptivePolicy(CachePolicy):
    """Adaptive eviction policy that combines multiple strategies."""

    def __init__(
        self,
        max_size: int,
        max_age_seconds: int | None = None,
        lru_weight: float = 0.4,
        lfu_weight: float = 0.3,
        size_weight: float = 0.3,
    ):
        """
        Initialize adaptive policy.

        Args:
            max_size: Maximum number of entries
            max_age_seconds: Maximum age of entries
            lru_weight: Weight for LRU scoring
            lfu_weight: Weight for LFU scoring
            size_weight: Weight for size-based scoring
        """
        super().__init__(max_size, max_age_seconds)
        self.lru_weight = lru_weight
        self.lfu_weight = lfu_weight
        self.size_weight = size_weight

    def should_evict(self, entries: dict[str, CacheEntry]) -> list[str]:
        """Evict entries based on adaptive scoring."""
        if len(entries) <= self.max_size:
            return []

        # Calculate scores for each entry
        scores = {}
        now = datetime.now(UTC)

        for key, entry in entries.items():
            # LRU score (inverse of access time)
            lru_score = 1.0 / (1.0 + (now - entry.accessed_at).total_seconds() / 3600)

            # LFU score (access count normalized)
            lfu_score = min(1.0, entry.access_count / 10.0)

            # Size score (smaller is better)
            size_score = 1.0 / (
                1.0 + entry.size_bytes / (1024 * 1024)
            )  # Normalize to MB

            # Combined score
            total_score = (
                lru_score * self.lru_weight
                + lfu_score * self.lfu_weight
                + size_score * self.size_weight
            )

            scores[key] = total_score

        # Sort by score (lowest first) and evict
        sorted_entries = sorted(scores.items(), key=lambda x: x[1])
        evict_count = len(entries) - self.max_size

        return [key for key, _ in sorted_entries[:evict_count]]


class MemoryCache:
    """
    High-performance in-memory cache with multiple eviction policies.

    Provides:
    - Fast access to frequently used data
    - Configurable eviction policies
    - Thread-safe operations
    - Memory usage monitoring
    - Statistics collection
    """

    def __init__(
        self,
        max_size: int = 1000,
        policy: CachePolicy | None = None,
        enable_stats: bool = True,
    ) -> None:
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            policy: Eviction policy to use
            enable_stats: Whether to collect statistics
        """
        self.max_size = max_size
        self.policy = policy or LRUPolicy(max_size)
        self.enable_stats = enable_stats

        # Cache storage
        self.entries: dict[str, CacheEntry] = {}
        self.size_bytes = 0

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = CacheStatistics() if enable_stats else None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self.lock:
            if key not in self.entries:
                if self.stats:
                    self.stats.misses += 1
                return default

            entry = self.entries[key]

            # Check expiration
            if entry.is_expired():
                del self.entries[key]
                self.size_bytes -= entry.size_bytes
                if self.stats:
                    self.stats.misses += 1
                    self.stats.evictions += 1
                return default

            # Update access statistics
            entry.access()
            if self.stats:
                self.stats.hits += 1

            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for the entry
            metadata: Additional metadata
        """
        with self.lock:
            # Serialize value to calculate size. This never deserialises, so
            # it is outside the serialization trust boundary.
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
            except Exception:
                # If not serializable, estimate size
                size_bytes = len(str(value).encode("utf-8"))

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                tags=tags or [],
                metadata=metadata or {},
            )

            # Set expiration
            if ttl_seconds:
                entry.expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

            # Check if we need to evict entries
            if len(self.entries) >= self.max_size:
                evict_keys = self.policy.should_evict(self.entries)
                for evict_key in evict_keys:
                    if evict_key in self.entries:
                        self.size_bytes -= self.entries[evict_key].size_bytes
                        del self.entries[evict_key]
                        if self.stats:
                            self.stats.evictions += 1

            # Remove existing entry if present
            if key in self.entries:
                self.size_bytes -= self.entries[key].size_bytes
                del self.entries[key]

            # Add new entry
            self.entries[key] = entry
            self.size_bytes += size_bytes

    def evict_by_tag(self, tag: str) -> int:
        """
        Evict all entries with a specific tag.

        Args:
            tag: Tag to evict

        Returns:
            Number of entries evicted
        """
        with self.lock:
            evicted_count = 0
            keys_to_remove = []

            for key, entry in self.entries.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)
                    self.size_bytes -= entry.size_bytes
                    evicted_count += 1

            for key in keys_to_remove:
                del self.entries[key]

            if self.stats:
                self.stats.evictions += evicted_count

            return evicted_count

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.size_bytes = 0
            if self.stats:
                self.stats.reset()

    def get_stats(self) -> CacheStatistics:
        """Get cache statistics."""
        if not self.stats:
            return CacheStatistics()

        with self.lock:
            self.stats.total_size_bytes = self.size_bytes
            self.stats.entry_count = len(self.entries)

        return self.stats


class DiskCache:
    """
    Persistent disk-based cache for large data.

    Provides:
    - Persistent storage across application restarts
    - Compression for space efficiency
    - Automatic cleanup of expired entries
    - SQLite-based backend for reliability
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_size_gb: float = 1.0,
        compression: bool = True,
        signing_key: bytes | str | None = None,
    ):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            compression: Whether to compress cached data
            signing_key: Explicit HMAC master key for cache files. When
                omitted the key is resolved from
                ``GEO_INFER_SERIALIZATION_KEY`` or the per-installation key
                file.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.compression = compression
        self.signing_key = signing_key

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()

        # Statistics
        self.stats = CacheStatistics()

    def _init_database(self) -> None:
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    expires_at REAL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER NOT NULL,
                    tags TEXT,  -- JSON array
                    metadata TEXT  -- JSON object
                )
            """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from disk cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT data_path, expires_at, access_count, size_bytes
                    FROM cache_entries WHERE key = ?
                """,
                    (key,),
                )

                row = cursor.fetchone()

                if not row:
                    self.stats.misses += 1
                    return default

                data_path, expires_at, access_count, size_bytes = row

                # Check expiration
                if expires_at and time.time() > expires_at:
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return default

                # Load data
                if not Path(data_path).exists():
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return default

                # Trust boundary: verify the envelope before the bytes reach
                # gzip or pickle, so a rewritten cache file is rejected rather
                # than inflated or executed.
                try:
                    data = verify_payload(
                        Path(data_path).read_bytes(),
                        context=CONTEXT_DISK_CACHE,
                        key=self.signing_key,
                    )
                except PayloadSecurityError as exc:
                    logger.error(
                        f"Rejected untrusted disk cache entry {key!r} before "
                        f"deserialization: {exc}"
                    )
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return default

                try:
                    if self.compression:
                        import gzip

                        data = gzip.decompress(data)
                    value = pickle.loads(data)
                except Exception:
                    # Data corrupted, remove it
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return default

                # Update access statistics
                current_time = time.time()
                conn.execute(
                    """
                    UPDATE cache_entries
                    SET accessed_at = ?, access_count = ?
                    WHERE key = ?
                """,
                    (current_time, access_count + 1, key),
                )

                self.stats.hits += 1
                return value

        except Exception as e:
            logger.warning(f"Error reading from disk cache: {e}")
            self.stats.misses += 1
            return default

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Put value in disk cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for the entry
            metadata: Additional metadata
        """
        try:
            # Serialize, compress, then sign: the on-disk bytes are always an
            # authenticated envelope wrapping the compressed payload.
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if self.compression:
                import gzip

                data = gzip.compress(data)

            data = sign_payload(
                data,
                context=CONTEXT_DISK_CACHE,
                key=self.signing_key,
            )

            size_bytes = len(data)
            data_path = (
                self.cache_dir
                / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.cache"
            )

            # Check if we need to evict entries
            current_size = self._get_total_size()
            if current_size + size_bytes > self.max_size_bytes:
                self._evict_lru_entries(size_bytes)

            # Remove existing entry
            if data_path.exists():
                self._evict_entry(key)

            # Write data to disk
            with open(data_path, "wb") as f:
                f.write(data)

            # Store metadata in database
            current_time = time.time()
            expires_at = current_time + ttl_seconds if ttl_seconds else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, data_path, created_at, accessed_at, expires_at, access_count, size_bytes, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        key,
                        str(data_path),
                        current_time,
                        current_time,
                        expires_at,
                        0,
                        size_bytes,
                        json.dumps(tags or []),
                        json.dumps(metadata or {}),
                    ),
                )

        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")

    def _evict_entry(self, key: str) -> None:
        """Evict a single cache entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data_path FROM cache_entries WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row:
                    data_path = Path(row[0])
                    if data_path.exists():
                        data_path.unlink()

                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    self.stats.evictions += 1

        except Exception as e:
            logger.warning(f"Error evicting cache entry {key}: {e}")

    def _get_total_size(self) -> int:
        """Get total size of cache in bytes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                return cursor.fetchone()[0] or 0
        except Exception:
            return 0

    def _evict_lru_entries(self, needed_bytes: int) -> None:
        """Evict least recently used entries to free space."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get entries ordered by access time (oldest first)
                cursor = conn.execute(
                    """
                    SELECT key, size_bytes FROM cache_entries
                    ORDER BY accessed_at ASC
                """
                )

                freed_bytes = 0
                keys_to_evict = []

                for row in cursor:
                    key, size_bytes = row
                    keys_to_evict.append(key)
                    freed_bytes += size_bytes

                    if freed_bytes >= needed_bytes:
                        break

                # Evict entries
                for key in keys_to_evict:
                    self._evict_entry(key)

        except Exception as e:
            logger.warning(f"Error evicting LRU entries: {e}")

    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        try:
            current_time = time.time()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT key FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                    (current_time,),
                )

                expired_keys = [row[0] for row in cursor]

                for key in expired_keys:
                    self._evict_entry(key)

                return len(expired_keys)

        except Exception as e:
            logger.warning(f"Error cleaning up expired entries: {e}")
            return 0


class RedisCache:
    """
    Redis-based distributed cache for multi-node scenarios.

    Provides:
    - Distributed caching across multiple nodes
    - Automatic failover and redundancy
    - Advanced Redis features (pipelines, transactions)
    - Connection pooling and health monitoring
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        max_connections: int = 20,
        signing_key: bytes | str | None = None,
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connection pool size
            signing_key: Explicit HMAC master key for cache documents. When
                omitted the key is resolved from
                ``GEO_INFER_SERIALIZATION_KEY`` or the per-installation key
                file.
        """
        self.host = host
        self.port = port
        self.db = db
        self.password: str | None = password
        self.max_connections = max_connections
        self.signing_key = signing_key

        # Connection pool
        self.connection_pool: Any = None
        self.redis_client: Any = None

        # Statistics
        self.stats = CacheStatistics()

        # Initialize connection
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Redis."""
        try:
            import redis

            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=True,
            )

            self.redis_client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            self.redis_client.ping()

        except ImportError:
            logger.error(
                "Redis package not installed. Install with: uv pip install redis"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from Redis cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            # Get data from Redis
            data = self.redis_client.get(key)

            if data is None:
                self.stats.misses += 1
                return default

            # Trust boundary: verify the envelope HMAC before parsing the
            # document. A Redis value rewritten by another client is rejected.
            try:
                cache_data = loads_signed_text(
                    data,
                    context=CONTEXT_REDIS_CACHE,
                    key=self.signing_key,
                    serializer="json",
                )
            except PayloadSecurityError as exc:
                logger.error(
                    f"Rejected untrusted Redis cache entry {key!r} before "
                    f"deserialization: {exc}"
                )
                self.redis_client.delete(key)
                self.stats.misses += 1
                return default
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.stats.misses += 1
                return default

            if not isinstance(cache_data, dict):
                logger.error(
                    f"Redis cache entry {key!r} is not a cache document; discarding"
                )
                self.redis_client.delete(key)
                self.stats.misses += 1
                return default

            # Check expiration
            if cache_data.get("expires_at") and time.time() > cache_data["expires_at"]:
                self.redis_client.delete(key)
                self.stats.misses += 1
                return default

            # Update access statistics
            cache_data["access_count"] = cache_data.get("access_count", 0) + 1
            cache_data["accessed_at"] = time.time()

            # Re-sign: the rewritten document must carry a fresh MAC, and the
            # original TTL must survive the update.
            remaining_ttl = self.redis_client.ttl(key)
            envelope = dumps_signed_text(
                cache_data,
                context=CONTEXT_REDIS_CACHE,
                key=self.signing_key,
                serializer="json",
            )
            if isinstance(remaining_ttl, int) and remaining_ttl > 0:
                self.redis_client.setex(key, remaining_ttl, envelope)
            else:
                self.redis_client.set(key, envelope)

            self.stats.hits += 1
            return cache_data.get("value")

        except Exception as e:
            logger.warning(f"Error reading from Redis cache: {e}")
            self.stats.misses += 1
            return default

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Put value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for the entry
            metadata: Additional metadata
        """
        try:
            cache_data = {
                "key": key,
                "value": value,
                "created_at": time.time(),
                "accessed_at": time.time(),
                "access_count": 0,
                "tags": tags or [],
                "metadata": metadata or {},
            }

            if ttl_seconds:
                cache_data["expires_at"] = time.time() + ttl_seconds

            envelope = dumps_signed_text(
                cache_data,
                context=CONTEXT_REDIS_CACHE,
                key=self.signing_key,
                serializer="json",
            )

            if ttl_seconds:
                self.redis_client.setex(key, ttl_seconds, envelope)
            else:
                self.redis_client.set(key, envelope)

        except Exception as e:
            logger.error(f"Error writing to Redis cache: {e}")

    def evict_by_tag(self, tag: str) -> int:
        """
        Evict all entries with a specific tag.

        Args:
            tag: Tag to evict

        Returns:
            Number of entries evicted
        """
        try:
            # Redis tag index pattern: we store a SET at key "tag::<tag>" containing
            # all cache keys carrying that tag.  We maintain this index in put().
            tag_index_key = f"tag::{tag}"
            members = self.redis_client.smembers(tag_index_key)
            if not members:
                return 0
            evicted = 0
            for member_key in members:
                deleted = self.redis_client.delete(member_key)
                if deleted:
                    evicted += 1
                    self.stats.evictions += 1
            # Remove the tag index itself
            self.redis_client.delete(tag_index_key)
            return evicted

        except Exception as e:
            logger.warning(f"Error evicting by tag: {e}")
            return 0

    def get_stats(self) -> CacheStatistics:
        """Get cache statistics."""
        try:
            info = self.redis_client.info("memory")

            # Update our stats with Redis info
            self.stats.total_size_bytes = info.get("used_memory", 0)
            self.stats.entry_count = self.redis_client.dbsize()

        except Exception as exc:
            logger.warning(
                "Redis memory info unavailable; returning stale stats: %s", exc
            )

        return self.stats


class MultiLevelCache:
    """
    Multi-level cache combining memory, disk, and distributed caching.

    Provides:
    - L1: Fast in-memory cache
    - L2: Persistent disk cache
    - L3: Distributed Redis cache
    - Intelligent cache promotion/demotion
    - Cross-level consistency
    """

    def __init__(
        self,
        memory_cache: MemoryCache | None = None,
        disk_cache: DiskCache | None = None,
        redis_cache: RedisCache | None = None,
    ) -> None:
        """
        Initialize multi-level cache.

        Args:
            memory_cache: L1 memory cache instance
            disk_cache: L2 disk cache instance
            redis_cache: L3 Redis cache instance
        """
        self.memory_cache = memory_cache or MemoryCache(max_size=1000)
        self.disk_cache: DiskCache | None = disk_cache
        self.redis_cache: RedisCache | None = redis_cache

        # Cache hierarchy for write-through
        self.write_levels: list[Any] = [self.memory_cache]
        if self.disk_cache:
            self.write_levels.append(self.disk_cache)
        if self.redis_cache:
            self.write_levels.append(self.redis_cache)

        # Read hierarchy (checked in order)
        self.read_levels: list[Any] = [self.memory_cache]
        if self.disk_cache:
            self.read_levels.append(self.disk_cache)
        if self.redis_cache:
            self.read_levels.append(self.redis_cache)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from multi-level cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        # Check levels in order
        for level in self.read_levels:
            value = level.get(key, None)
            if value is not None:
                # Promote to higher levels
                if level != self.memory_cache:
                    self.memory_cache.put(key, value)
                return value

        return default

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Put value in all cache levels.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for the entry
            metadata: Additional metadata
        """
        for level in self.write_levels:
            try:
                level.put(key, value, ttl_seconds, tags, metadata)
            except Exception as e:
                logger.warning(f"Failed to write to cache level: {e}")

    def evict(self, key: str) -> None:
        """Evict entry from all cache levels."""
        for level in self.write_levels:
            try:
                # Memory cache has a different API
                if hasattr(level, "entries"):
                    if key in level.entries:
                        level.size_bytes -= level.entries[key].size_bytes
                        del level.entries[key]
            except Exception as e:
                logger.warning(f"Failed to evict from cache level: {e}")

    def clear(self) -> None:
        """Clear all cache levels."""
        for level in self.write_levels:
            level.clear()


class IntelligentCache:
    """
    Intelligent cache with adaptive strategies and optimization.

    Provides:
    - Automatic cache warming for frequently accessed data
    - Predictive prefetching based on access patterns
    - Adaptive TTL based on access frequency
    - Cache analytics and optimization recommendations
    """

    #: Sentinel distinguishing "absent" from a cached ``None``.
    _MISS = object()
    #: Accesses this far apart count as co-access.
    CO_ACCESS_WINDOW_SECONDS = 5.0
    #: Weight contributed by sharing a key namespace.
    NAMESPACE_SIBLING_WEIGHT = 1.0
    #: Minimum score before a key is treated as related.
    MIN_RELATEDNESS_SCORE = 0.5
    #: Upper bound on keys returned per prefetch pass.
    MAX_RELATED_KEYS = 10

    def __init__(self, cache: MultiLevelCache | None = None) -> None:
        """
        Initialize intelligent cache.

        Args:
            cache: Multi-level cache to optimize
        """
        self.cache = cache or MultiLevelCache()
        self.access_patterns: dict[str, list[float]] = {}
        self.prefetch_queue: queue.Queue = queue.Queue()
        self.warmup_list: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with intelligent caching behavior.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        value = self.cache.get(key, None)

        if value is not None:
            # Record access pattern
            self._record_access(key)

            # Trigger prefetching if appropriate
            self._trigger_prefetch(key)

        return value if value is not None else default

    def put(
        self,
        key: str,
        value: Any,
        adaptive_ttl: bool = True,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Put value with intelligent TTL calculation.

        Args:
            key: Cache key
            value: Value to cache
            adaptive_ttl: Whether to use adaptive TTL
            tags: Tags for the entry
            metadata: Additional metadata
        """
        # Calculate TTL based on access patterns
        ttl_seconds = None
        if adaptive_ttl:
            ttl_seconds = self._calculate_adaptive_ttl(key)

        self.cache.put(key, value, ttl_seconds, tags, metadata)

    def _record_access(self, key: str) -> None:
        """Record access pattern for a key."""
        current_time = time.time()

        if key not in self.access_patterns:
            self.access_patterns[key] = []

        # Keep last 10 access times
        self.access_patterns[key].append(current_time)

        # Keep only recent accesses (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [
            access_time
            for access_time in self.access_patterns[key]
            if access_time > cutoff_time
        ]

    def _calculate_adaptive_ttl(self, key: str) -> int | None:
        """Calculate adaptive TTL based on access patterns."""
        if key not in self.access_patterns:
            return None

        accesses = self.access_patterns[key]

        if len(accesses) < 2:
            return None

        # Calculate average time between accesses
        intervals = []
        for i in range(1, len(accesses)):
            intervals.append(accesses[i] - accesses[i - 1])

        if not intervals:
            return None

        avg_interval = sum(intervals) / len(intervals)

        # Set TTL to 3x average interval, with bounds
        ttl = min(max(int(avg_interval * 3), 300), 86400)  # 5 minutes to 24 hours

        return ttl

    def _trigger_prefetch(self, key: str) -> None:
        """Queue related keys that are not currently resident in the cache.

        Keys already held by the cache need no prefetch; keys that were seen
        before but have since been evicted are exactly the ones worth
        warming, so residency -- not whether the key was ever accessed --
        decides what is queued.

        Note:
            :meth:`MultiLevelCache.get` cannot distinguish a stored ``None``
            from a miss, so a key whose cached value is ``None`` is treated
            as absent and may be queued again. The sentinel below keeps this
            method correct if that layer later gains that distinction.

        Args:
            key: The key whose access triggered this prefetch pass.
        """
        for related_key in self._find_related_keys(key):
            if self.cache.get(related_key, self._MISS) is self._MISS:
                self.prefetch_queue.put(related_key)

    @staticmethod
    def _key_namespace(key: str) -> str:
        """Return the namespace of a cache key.

        Keys are conventionally structured as ``<namespace><sep><leaf>``
        using ``:`` or ``/``; the namespace is everything up to the final
        separator. Keys with no separator have no namespace.

        Args:
            key: The cache key to split.

        Returns:
            The namespace, or an empty string when the key has none.
        """
        cut = max(key.rfind(":"), key.rfind("/"))
        return key[:cut] if cut > 0 else ""

    def _find_related_keys(self, key: str) -> list[str]:
        """Find keys likely to be needed alongside *key*.

        Two signals are combined, both drawn from observed behaviour rather
        than a fixed table:

        * **Namespace siblings** -- keys sharing this key's namespace are
          usually facets of one entity and are fetched together.
        * **Temporal co-access** -- keys repeatedly accessed within
          ``CO_ACCESS_WINDOW_SECONDS`` of this key are used together
          regardless of naming.

        Results are ordered by co-access strength (strongest first) so a
        bounded prefetch queue spends its budget on the best candidates.

        Args:
            key: The key that was just accessed.

        Returns:
            Related keys, most strongly related first, excluding *key*.
        """
        own_accesses = self.access_patterns.get(key) or []
        namespace = self._key_namespace(key)

        scores: dict[str, float] = {}
        for candidate, accesses in self.access_patterns.items():
            if candidate == key:
                continue

            score = 0.0
            if namespace and self._key_namespace(candidate) == namespace:
                score += self.NAMESPACE_SIBLING_WEIGHT

            if own_accesses and accesses:
                co_accesses = sum(
                    1
                    for own_time in own_accesses
                    for other_time in accesses
                    if abs(own_time - other_time) <= self.CO_ACCESS_WINDOW_SECONDS
                )
                if co_accesses:
                    # Normalise by this key's access count so a hot candidate
                    # cannot dominate purely by being accessed often.
                    score += co_accesses / len(own_accesses)

            if score >= self.MIN_RELATEDNESS_SCORE:
                scores[candidate] = score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [candidate for candidate, _ in ranked[: self.MAX_RELATED_KEYS]]

    def warmup(self, keys: list[str]) -> None:
        """
        Warm up cache with frequently accessed keys.

        Args:
            keys: List of keys to warm up
        """
        self.warmup_list.extend(keys)

        # Trigger warmup in background
        warmup_thread = threading.Thread(target=self._warmup_worker, daemon=True)
        warmup_thread.start()

    def _warmup_worker(self) -> None:
        """Background worker for cache warming.

        Drains the prefetch_queue, then also preloads any keys in warmup_list
        that are not already cached (via a no-op get to allow higher layers to
        populate from disk/Redis).  Warmup targets are processed in FIFO order.
        """
        # Process explicit prefetch queue first
        while not self.prefetch_queue.empty():
            try:
                key = self.prefetch_queue.get(timeout=0.1)
                # Trigger a cache lookup — the multi-level cache will promote
                # the value from disk/Redis into memory if it exists
                _ = self.cache.get(key)
                logger.debug(f"Warmed cache for prefetch key: {key}")
            except Exception:
                break

        # Drain the warmup_list
        while self.warmup_list:
            key = self.warmup_list.pop(0)
            _ = self.cache.get(key)
            logger.debug(f"Warmed cache for warmup key: {key}")

    def get_analytics(self) -> dict[str, Any]:
        """Get cache analytics and optimization recommendations."""
        analytics = {
            "cache_stats": self.cache.memory_cache.get_stats(),
            "access_patterns": {
                "total_keys": len(self.access_patterns),
                "frequently_accessed": len(
                    [
                        key
                        for key, accesses in self.access_patterns.items()
                        if len(accesses) > 5
                    ]
                ),
            },
            "recommendations": self._generate_optimization_recommendations(),
        }

        return analytics

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        stats = self.cache.memory_cache.get_stats()

        # Hit rate recommendations
        if stats.hit_rate < 70:
            recommendations.append(
                "Consider increasing cache size or improving cache locality"
            )

        # Memory usage recommendations
        if stats.entry_count > self.cache.memory_cache.max_size * 0.9:
            recommendations.append(
                "Cache is near capacity, consider increasing max_size"
            )

        # Access pattern recommendations
        frequently_accessed = len(
            [key for key, accesses in self.access_patterns.items() if len(accesses) > 5]
        )

        if frequently_accessed > 10:
            recommendations.append(
                "Enable predictive prefetching for better performance"
            )

        return recommendations


def create_optimized_cache(
    memory_size: int = 1000,
    disk_size_gb: float = 1.0,
    redis_host: str | None = None,
    signing_key: bytes | str | None = None,
) -> MultiLevelCache:
    """
    Create an optimized multi-level cache configuration.

    Args:
        memory_size: Size of memory cache
        disk_size_gb: Size of disk cache in GB
        redis_host: Redis host (None to disable Redis)
        signing_key: Explicit HMAC master key for the persistent levels. When
            omitted the key is resolved from ``GEO_INFER_SERIALIZATION_KEY``
            or the per-installation key file.

    Returns:
        Configured MultiLevelCache instance
    """
    # Memory cache
    memory_cache = MemoryCache(
        max_size=memory_size, policy=AdaptivePolicy(max_size=memory_size)
    )

    # Disk cache
    disk_cache: DiskCache | None = None
    if disk_size_gb > 0:
        cache_dir = Path.home() / ".geo_infer_git" / "cache"
        disk_cache = DiskCache(cache_dir, disk_size_gb, signing_key=signing_key)

    # Redis cache
    redis_cache: RedisCache | None = None
    if redis_host:
        try:
            redis_cache = RedisCache(host=redis_host, signing_key=signing_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")

    return MultiLevelCache(memory_cache, disk_cache, redis_cache)


class CacheDecorator:
    """
    Decorator for automatic caching of function results.

    Provides:
    - Automatic cache key generation
    - TTL management
    - Cache invalidation
    - Performance monitoring
    """

    def __init__(
        self,
        cache: MultiLevelCache,
        ttl_seconds: int | None = None,
        key_prefix: str = "",
        include_args: bool = True,
    ):
        """
        Initialize cache decorator.

        Args:
            cache: Cache instance to use
            ttl_seconds: Default TTL for cached results
            key_prefix: Prefix for cache keys
            include_args: Whether to include function arguments in cache key
        """
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.include_args = include_args

    def __call__(self, func: Callable) -> Callable:
        """Apply caching decorator to function."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            key_parts = [self.key_prefix, func.__name__]

            if self.include_args:
                # Include function arguments in key (simplified)
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])

            cache_key = hashlib.sha256("::".join(key_parts).encode("utf-8")).hexdigest()

            # Check cache
            result = self.cache.get(cache_key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            self.cache.put(cache_key, result, self.ttl_seconds)

            return result

        return wrapper

    def invalidate(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Invalidate cache entries for a function.

        Args:
            func_name: Name of the function
            *args, **kwargs: Function arguments to match
        """
        key_parts = [self.key_prefix, func_name]

        if self.include_args:
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])

        cache_key = hashlib.sha256("::".join(key_parts).encode("utf-8")).hexdigest()
        self.cache.evict(cache_key)
