"""
Caching Utilities for Expensive Computations

This module provides caching utilities for expensive mathematical
computations to improve performance.
"""

import functools
import hashlib
import math
import pickle
import time
from collections import OrderedDict
from numbers import Integral, Real
from threading import RLock
from typing import Any
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


def _validate_maxsize(maxsize: int) -> None:
    """Validate bounded cache capacity; zero explicitly disables storage."""
    if isinstance(maxsize, bool) or not isinstance(maxsize, Integral) or maxsize < 0:
        raise ValueError("maxsize must be a non-negative integer")


def cache_result(maxsize: int = 128, ttl: float | None = None) -> Callable:
    """
    Decorator to cache results with bounded least-recently-used storage.

    Results retain their native identity, so callers should not mutate shared
    cached results. Keys include argument structure and NumPy array shape,
    dtype, and values. Unpickleable arguments bypass storage. State updates are
    thread-safe; concurrent misses may compute the same key independently.

    Args:
        maxsize: Maximum cache size (zero disables storage).
        ttl: Monotonic lifetime after computation, in seconds. None does not
            expire results; zero expires immediately.

    Returns:
        Decorated function with caching
    """

    _validate_maxsize(maxsize)
    if ttl is not None and (
        isinstance(ttl, bool)
        or not isinstance(ttl, Real)
        or not math.isfinite(ttl)
        or ttl < 0
    ):
        raise ValueError("ttl must be a finite non-negative number or None")

    def decorator(func: Callable) -> Callable:
        cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        lock = RLock()
        hits = 0
        misses = 0

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal hits, misses
            try:
                key = _create_cache_key(args, kwargs)
            except Exception:
                # Serialization is only an optimization. User-defined pickle
                # protocols can raise arbitrary ordinary exceptions; still run
                # the requested computation. BaseException propagates.
                with lock:
                    misses += 1
                return func(*args, **kwargs)

            with lock:
                if key in cache:
                    cached_result, created_at = cache[key]
                    if ttl is None or time.monotonic() - created_at < ttl:
                        cache.move_to_end(key)
                        hits += 1
                        return cached_result
                    del cache[key]
                misses += 1

            result = func(*args, **kwargs)
            if maxsize and ttl != 0:
                with lock:
                    cache[key] = (result, time.monotonic())
                    cache.move_to_end(key)
                    while len(cache) > maxsize:
                        cache.popitem(last=False)
            return result

        def cache_clear() -> None:
            nonlocal hits, misses
            with lock:
                cache.clear()
                hits = misses = 0

        def cache_info() -> dict[str, Any]:
            with lock:
                return {
                    "size": len(cache),
                    "maxsize": maxsize,
                    "hits": hits,
                    "misses": misses,
                }

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info

        return wrapper

    return decorator


def _create_cache_key(args: tuple, kwargs: dict) -> str:
    """Hash structured arguments without discarding array metadata or contents."""
    payload = pickle.dumps((args, tuple(sorted(kwargs.items()))), protocol=5)
    return hashlib.sha256(payload).hexdigest()


class ComputationCache:
    """
    Thread-safe LRU cache manager retaining native value identity.
    """

    def __init__(self, maxsize: int = 256) -> None:
        """
        Initialize computation cache.

        Args:
            maxsize: Maximum cache size (zero disables storage).
        """
        _validate_maxsize(maxsize)
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if not self.maxsize:
                return
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

    def info(self) -> dict:
        """Get cache information."""
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "keys": list(self._cache.keys()),
            }
