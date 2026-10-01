"""
Caching management for GEO-INFER-OPS.

Security note (serialization trust boundary): Redis is shared, mutable
infrastructure — any client with write access can replace a cached value, and
``pickle.loads`` executes arbitrary code during unpickling. Every value this
manager writes is therefore wrapped in an authenticated GISP1 envelope (see
:mod:`geo_infer_ops.core.secure_serialization`) and every value it reads is
verified against its HMAC-SHA256 under the ``ops.cache.redis`` context key
before ``json.loads`` or ``pickle.loads`` is called. Unsigned, truncated, or
tampered values are rejected and never deserialised.

The counter operations (:meth:`CacheManager.increment` and
:meth:`CacheManager.decrement`) deliberately stay outside the envelope: they
use Redis' native ``INCRBY``/``DECRBY`` on integer strings and never
deserialise a payload, so they carry no code-execution risk.
"""

from typing import Any, cast
from enum import Enum

import redis
from redis.exceptions import RedisError

from geo_infer_ops.core.config import get_config
from geo_infer_ops.core.logging import get_logger
from geo_infer_ops.core.secure_serialization import (
    CONTEXT_REDIS_CACHE,
    PayloadSecurityError,
    dumps_signed,
    dumps_signed_text,
    loads_signed,
    loads_signed_text,
)

logger = get_logger(__name__)


class CacheSerializer(Enum):
    """Cache serialization formats."""

    JSON = "json"
    PICKLE = "pickle"


class CacheManager:
    """Manages caching operations for GEO-INFER-OPS."""

    def __init__(
        self,
        serializer: CacheSerializer = CacheSerializer.JSON,
        prefix: str = "geo_infer:",
        signing_key: bytes | str | None = None,
        raise_on_untrusted: bool = False,
    ):
        """
        Initialize cache manager.

        Args:
            serializer: Serialization format to use
            prefix: Key prefix for all cache entries
            signing_key: Explicit HMAC master key for cached payloads. When
                omitted the key is resolved from
                ``GEO_INFER_SERIALIZATION_KEY`` or the per-installation key
                file.
            raise_on_untrusted: When True, a read that fails HMAC verification
                propagates :class:`PayloadSecurityError` to the caller. When
                False (the default) the poisoned key is evicted, the rejection
                is logged at error level, and the read reports a cache miss.
                Either way the payload is never deserialized.
        """
        self.config = get_config()
        self.serializer = serializer
        self.prefix = prefix
        self.signing_key = signing_key
        self.raise_on_untrusted = raise_on_untrusted
        self._connect()

    def _connect(self) -> None:
        """Connect to Redis server."""
        try:
            cache_cfg: Any = self.config.cache
            self.redis = redis.Redis(
                host=cache_cfg.redis.host,
                port=cache_cfg.redis.port,
                db=cache_cfg.redis.db,
                password=cache_cfg.redis.password,
                decode_responses=(self.serializer == CacheSerializer.JSON),
            )
            self.redis.ping()
            logger.info("cache_connected")
        except AttributeError as e:
            raise RuntimeError(
                "CacheManager requires a 'cache' section on the configuration. "
                "The shipped Config model provides it by default; if a custom "
                "config object was injected, add a CacheConfig with redis "
                "host/port/db/password."
            ) from e
        except RedisError as e:
            logger.error("cache_connection_failed", error=str(e))
            raise

    def _serialize(self, value: Any) -> str | bytes:
        """
        Serialize and sign a value for storage.

        JSON mode produces a text envelope (the Redis client runs with
        ``decode_responses=True``); pickle mode produces a binary envelope.

        Args:
            value: Value to serialize

        Returns:
            Authenticated GISP1 envelope carrying the serialized value.
        """
        try:
            if self.serializer == CacheSerializer.JSON:
                return dumps_signed_text(
                    value,
                    context=CONTEXT_REDIS_CACHE,
                    key=self.signing_key,
                    serializer="json",
                )
            return dumps_signed(
                value,
                context=CONTEXT_REDIS_CACHE,
                key=self.signing_key,
                serializer="pickle",
            )
        except Exception as e:
            logger.error("cache_serialization_failed", error=str(e))
            raise

    def _deserialize(self, value: str | bytes | None) -> Any:
        """
        Verify and deserialize a value read from storage.

        This is the trust boundary: HMAC verification runs before any
        deserializer touches the payload, so an unsigned or tampered Redis
        value raises instead of being decoded.

        Args:
            value: Raw value read from Redis

        Returns:
            Deserialized value, or ``None`` when ``value`` is ``None``.

        Raises:
            PayloadSecurityError: If the payload is not a valid, correctly
                signed envelope for this cache.
        """
        if value is None:
            return None
        try:
            if self.serializer == CacheSerializer.JSON:
                return loads_signed_text(
                    cast(str, value),
                    context=CONTEXT_REDIS_CACHE,
                    key=self.signing_key,
                    serializer="json",
                )
            return loads_signed(
                cast(bytes, value),
                context=CONTEXT_REDIS_CACHE,
                key=self.signing_key,
                serializer="pickle",
            )
        except PayloadSecurityError as e:
            logger.error("cache_untrusted_payload_rejected", error=str(e))
            raise
        except Exception as e:
            logger.error("cache_deserialization_failed", error=str(e))
            raise

    def _reject_untrusted(self, key: str, full_key: str) -> None:
        """
        Handle a payload that failed HMAC verification.

        The poisoned entry is evicted so a later read cannot re-trigger the
        rejection, and the event is logged at error level. When
        ``raise_on_untrusted`` is set the exception is re-raised to the caller.

        Args:
            key: Logical cache key
            full_key: Prefixed Redis key

        Raises:
            PayloadSecurityError: Re-raised when ``raise_on_untrusted`` is set.
        """
        logger.error("cache_untrusted_payload_evicted", key=key)
        try:
            self.redis.delete(full_key)
        except RedisError as e:
            logger.error("cache_untrusted_eviction_failed", key=key, error=str(e))
        if self.raise_on_untrusted:
            raise

    def _get_key(self, key: str) -> str:
        """
        Get full cache key with prefix.

        Args:
            key: Cache key

        Returns:
            Full cache key
        """
        return f"{self.prefix}{key}"

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            full_key = self._get_key(key)
            value: Any = self.redis.get(full_key)

            if value is None:
                return default

            try:
                return self._deserialize(value)
            except PayloadSecurityError:
                self._reject_untrusted(key, full_key)
                return default
        except RedisError as e:
            logger.error("cache_get_failed", key=key, error=str(e))
            return default

    def set(
        self,
        key: str,
        value: Any,
        expire: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            bool: True if successful
        """
        try:
            full_key = self._get_key(key)
            serialized = self._serialize(value)

            if expire is not None:
                return bool(self.redis.setex(full_key, expire, serialized))
            elif nx:
                return bool(self.redis.setnx(full_key, serialized))
            elif xx:
                return bool(self.redis.set(full_key, serialized, xx=True))
            else:
                return bool(self.redis.set(full_key, serialized))
        except RedisError as e:
            logger.error("cache_set_failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            bool: True if successful
        """
        try:
            full_key = self._get_key(key)
            return bool(self.redis.delete(full_key))
        except RedisError as e:
            logger.error("cache_delete_failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key exists
        """
        try:
            full_key = self._get_key(key)
            return bool(self.redis.exists(full_key))
        except RedisError as e:
            logger.error("cache_exists_failed", key=key, error=str(e))
            return False

    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time for key.

        Args:
            key: Cache key
            seconds: Expiration time in seconds

        Returns:
            bool: True if successful
        """
        try:
            full_key = self._get_key(key)
            return bool(self.redis.expire(full_key, seconds))
        except RedisError as e:
            logger.error("cache_expire_failed", key=key, error=str(e))
            return False

    def ttl(self, key: str) -> int | None:
        """
        Get time to live for key.

        Args:
            key: Cache key

        Returns:
            Time to live in seconds or None if key doesn't exist
        """
        try:
            full_key = self._get_key(key)
            ttl_raw: Any = self.redis.ttl(full_key)
            ttl = int(ttl_raw)
            return ttl if ttl >= 0 else None
        except RedisError as e:
            logger.error("cache_ttl_failed", key=key, error=str(e))
            return None

    def increment(self, key: str, amount: int = 1) -> int | None:
        """
        Increment value in cache.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value or None if operation failed
        """
        try:
            full_key = self._get_key(key)
            return cast(int, self.redis.incrby(full_key, amount))
        except RedisError as e:
            logger.error("cache_increment_failed", key=key, error=str(e))
            return None

    def decrement(self, key: str, amount: int = 1) -> int | None:
        """
        Decrement value in cache.

        Args:
            key: Cache key
            amount: Amount to decrement by

        Returns:
            New value or None if operation failed
        """
        try:
            full_key = self._get_key(key)
            return cast(int, self.redis.decrby(full_key, amount))
        except RedisError as e:
            logger.error("cache_decrement_failed", key=key, error=str(e))
            return None

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dict mapping keys to values (excluding missing keys)
        """
        try:
            full_keys = [self._get_key(key) for key in keys]
            values: Any = self.redis.mget(full_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is None:
                    continue
                try:
                    result[key] = self._deserialize(value)
                except PayloadSecurityError:
                    self._reject_untrusted(key, self._get_key(key))
                except Exception:
                    logger.warning("cache_deserialization_failed", key=key)
            return result
        except RedisError as e:
            logger.error("cache_get_many_failed", error=str(e))
            return {}

    def set_many(self, mapping: dict[str, Any], expire: int | None = None) -> bool:
        """
        Set multiple values in cache.

        Args:
            mapping: Dict mapping keys to values
            expire: Optional expiration time in seconds

        Returns:
            bool: True if all operations were successful
        """
        try:
            with self.redis.pipeline() as pipe:
                for key, value in mapping.items():
                    full_key = self._get_key(key)
                    serialized = self._serialize(value)

                    if expire is not None:
                        pipe.setex(full_key, expire, serialized)
                    else:
                        pipe.set(full_key, serialized)

                results = pipe.execute()
                return all(bool(result) for result in results)
        except RedisError as e:
            logger.error("cache_set_many_failed", error=str(e))
            return False

    def delete_many(self, keys: list[str]) -> bool:
        """
        Delete multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            bool: True if all operations were successful
        """
        try:
            full_keys = [self._get_key(key) for key in keys]
            deleted = self.redis.delete(*full_keys)
            return bool(deleted)
        except RedisError as e:
            logger.error("cache_delete_many_failed", error=str(e))
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries with prefix.

        Returns:
            bool: True if successful
        """
        try:
            pattern = f"{self.prefix}*"
            cursor = 0
            success = True

            while True:
                scan_result: Any = self.redis.scan(cursor=cursor, match=pattern)
                cursor, keys = scan_result
                if keys:
                    if not self.redis.delete(*keys):
                        success = False
                cursor = int(cursor)
                if cursor == 0:
                    break

            return success
        except RedisError as e:
            logger.error("cache_clear_failed", error=str(e))
            return False

    def get_size(self) -> int:
        """
        Get number of cache entries with prefix.

        Returns:
            Number of entries
        """
        try:
            pattern = f"{self.prefix}*"
            cursor = 0
            count = 0

            while True:
                scan_result: Any = self.redis.scan(cursor=cursor, match=pattern)
                cursor, keys = scan_result
                count += len(keys)
                cursor = int(cursor)
                if cursor == 0:
                    break

            return count
        except RedisError as e:
            logger.error("cache_size_failed", error=str(e))
            return 0
