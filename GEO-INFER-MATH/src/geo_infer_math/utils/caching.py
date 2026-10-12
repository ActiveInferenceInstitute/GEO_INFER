"""
Caching Utilities for Expensive Computations

This module provides caching utilities for expensive mathematical
computations to improve performance.
"""

import functools
import hashlib
import pickle
from typing import Any
from collections.abc import Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


def cache_result(maxsize: int = 128, ttl: float | None = None) -> Callable:
    """
    Decorator to cache function results.

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds (None for no expiration)

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable) -> Callable:
        cache: dict[str, Any] = {}
        cache_times: dict[str, float] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key
            key = _create_cache_key(args, kwargs)

            # Check cache
            if key in cache:
                if ttl is None:
                    return cache[key]
                else:
                    import time

                    if time.time() - cache_times[key] < ttl:
                        return cache[key]
                    else:
                        # Expired, remove
                        del cache[key]
                        del cache_times[key]

            # Compute result
            result = func(*args, **kwargs)

            # Check cache size limit
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]

            cache[key] = result
            if ttl is not None:
                import time

                cache_times[key] = time.time()

            return result

        def cache_clear() -> None:
            cache.clear()
            cache_times.clear()

        def cache_info() -> dict[str, Any]:
            return {
                "size": len(cache),
                "maxsize": maxsize,
                "hits": getattr(wrapper, "_hits", 0),
                "misses": getattr(wrapper, "_misses", 0),
            }

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info

        return wrapper

    return decorator


def _create_cache_key(args: tuple, kwargs: dict) -> str:
    """Create a cache key from function arguments."""

    # Handle numpy arrays specially
    def serialize_arg(arg: Any) -> Any:
        if isinstance(arg, np.ndarray):
            return hashlib.md5(arg.tobytes()).hexdigest()
        try:
            return pickle.dumps(arg)
        except Exception:
            return str(arg)

    key_parts = [serialize_arg(arg) for arg in args]
    key_parts.extend([f"{k}:{serialize_arg(v)}" for k, v in sorted(kwargs.items())])

    return hashlib.md5(b"".join(str(p).encode() for p in key_parts)).hexdigest()


class ComputationCache:
    """
    Cache manager for expensive computations.
    """

    def __init__(self, maxsize: int = 256) -> None:
        """
        Initialize computation cache.

        Args:
            maxsize: Maximum cache size
        """
        self.maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._access_times: dict[str, float] = {}

    def get(self, key: str) -> Any | None:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self._cache:
            import time

            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        if len(self._cache) >= self.maxsize:
            # Remove least recently used
            lru_key = min(
                self._access_times.keys(), key=lambda k: self._access_times[k]
            )
            del self._cache[lru_key]
            del self._access_times[lru_key]

        self._cache[key] = value
        import time

        self._access_times[key] = time.time()

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._access_times.clear()

    def info(self) -> dict:
        """Get cache information."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "keys": list(self._cache.keys()),
        }
