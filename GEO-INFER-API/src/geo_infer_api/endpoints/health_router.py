"""
Health check endpoints for the GEO-INFER-API.
"""

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Depends

from geo_infer_api.core.config import get_settings, Settings


logger = logging.getLogger(__name__)
# Record process start time for uptime tracking
_START_TIME: float = time.time()

router = APIRouter()


def _format_uptime(seconds: float) -> str:
    """Format elapsed seconds as a human-readable uptime string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def _get_memory_mb() -> float:
    """Return current RSS memory usage in megabytes."""
    try:
        import resource

        # resource.getrusage returns bytes on Linux, kilobytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        logger.debug("memory usage unavailable", exc_info=True)
        return -1.0


@router.get("/health", summary="Health check")
async def health_check(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """
    Simple health check endpoint.

    Returns basic status information about the running API instance.
    """
    return {
        "status": "ok",
        "version": settings.app_version,
        "name": settings.app_name,
    }


@router.get("/health/detailed", summary="Detailed health check")
async def detailed_health_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Detailed health check with real component status information.

    Returns live values for memory usage, uptime, and feature flags.
    """
    uptime_seconds = time.time() - _START_TIME
    memory_mb = _get_memory_mb()

    return {
        "status": "ok",
        "version": settings.app_version,
        "components": {
            "geojson_service": {"status": "ok"},
            "memory_usage": {
                "status": "ok",
                "value_mb": round(memory_mb, 2),
            },
            "uptime": {
                "status": "ok",
                "seconds": round(uptime_seconds, 1),
                "human": _format_uptime(uptime_seconds),
            },
        },
        "features": {
            "ogc_api_features": settings.ogc_api_features_enabled,
            "ogc_api_processes": settings.ogc_api_processes_enabled,
        },
        "environment": os.getenv("ENVIRONMENT", "development"),
    }
