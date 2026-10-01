"""
Shared caching infrastructure for GEO-INFER-PLACE API wrappers.

Provides a CachedAPIWrapper base class that eliminates duplicated caching
logic across _CALFIREWrapper, _NOAAWrapper, _USGSWrapper, and future wrappers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".geo_infer_place" / "cache"


class CachedAPIWrapper:
    """Base class providing file-based caching with TTL for API wrappers.

    Subclasses only need to set ``cache_ttl`` and implement their
    domain-specific fetch methods.  All cache key generation, read/write,
    and expiry logic lives here.

    Args:
        cache_dir: Directory for cache files. Defaults to ``~/.geo_infer_place/cache``.
        cache_ttl: Time-to-live for cached entries.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl: timedelta = timedelta(hours=24),
    ) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, method_name: str, **kwargs: Any) -> str:
        """Generate a deterministic cache key from method name + params."""
        key_data = {"method": method_name, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _read_cache(self, cache_key: str) -> Any | None:
        """Return cached data if present and not expired, else ``None``."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                envelope = json.load(f)

            cached_at = datetime.fromisoformat(envelope["timestamp"])
            if datetime.now() - cached_at > self.cache_ttl:
                logger.debug("Cache expired for key %s", cache_key)
                cache_file.unlink(missing_ok=True)
                return None

            logger.debug("Cache hit for key %s", cache_key)
            return envelope["data"]

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Cache read error for key %s: %s", cache_key, exc)
            cache_file.unlink(missing_ok=True)
            return None

    def _write_cache(self, cache_key: str, data: Any) -> None:
        """Persist *data* with a timestamp envelope."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        envelope = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        try:
            with open(cache_file, "w") as f:
                json.dump(envelope, f, indent=2, default=str)
            logger.debug("Cached data for key %s", cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cache data for key %s: %s", cache_key, exc)

    def clear_cache(self) -> int:
        """Remove all cache files.  Returns the number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cleared %d cache entries", count)
        return count

    def cache_stats(self) -> dict:
        """Return basic statistics about the cache directory."""
        files = list(self.cache_dir.glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "entries": len(files),
            "total_bytes": total_bytes,
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.cache_ttl.total_seconds() / 3600,
        }
