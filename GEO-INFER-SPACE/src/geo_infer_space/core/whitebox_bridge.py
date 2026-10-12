"""Optional WhiteboxTools bridge for terrain/hydrology analysis.

GeoLibre integrates the Whitebox toolbox (1,000+ geoprocessing tools) either
in-browser via WASM or through a managed Python sidecar running
``whitebox-workflows``. GEO-INFER mirrors the *Python sidecar* approach: the
``whitebox-workflows`` package is an optional dependency, and this module probes
for it at import time (``HAS_WHITEBOX``) so the rest of GEO-INFER-SPACE
imports cleanly without it. Domain modules (WATER, FOREST, MARINE, EMERGENCY)
can call the terrain/hydrology helpers here; when the optional dependency is
absent they fail fast with a clear, actionable ImportError — never a silent
no-op that fabricates results.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import probe
    import whitebox_workflows as _wbw  # type: ignore[import-not-found]

    HAS_WHITEBOX: bool = True
    _WBW = _wbw
except ImportError:  # pragma: no cover - exercised when whitebox is missing
    _WBW = None
    HAS_WHITEBOX = False


def whitebox_available() -> bool:
    """Return whether the optional ``whitebox-workflows`` runtime is present."""
    return HAS_WHITEBOX


def whitebox_version() -> str | None:
    """Return the installed Whitebox version string, if available."""
    if not HAS_WHITEBOX:
        return None
    return str(getattr(_WBW, "__version__", "unknown"))


def _require_whitebox() -> None:
    """Raise a clear ImportError when the optional runtime is absent.

    Raises:
        ImportError: Always when ``HAS_WHITEBOX`` is ``False``.
    """
    if not HAS_WHITEBOX:
        raise ImportError(
            "whitebox-workflows is required for this terrain/hydrology tool. "
            'Install it with `uv pip install -e "./GEO-INFER-SPACE[optional]"` '
            "or `pip install whitebox-workflows`."
        )


def flow_accumulation(
    dem_file: str | Path,
    output_file: str | Path,
    *,
    flow_type: int = 1,
    outlet_file: str | Path | None = None,
) -> Path:
    """Compute Flow-Accumulation on a digital elevation model.

    Delegates to WhiteboxTools' ``flow_accumulation`` (D8, flow type 1). This
    is a representative terrain helper for ``WATER``/``FOREST`` modules; the
    domain modules can add further WhiteboxTools wrappers on the same pattern.

    Args:
        dem_file: Path to the input DEM raster (GeoTIFF or supported format).
        output_file: Path to write the output raster.
        flow_type: Whitebox flow type (1 = D8 direction, others per Whitebox).
        outlet_file: Optional outlet/streams raster to constrain accumulation.

    Returns:
        The resolved output path.

    Raises:
        ImportError: If ``whitebox-workflows`` is not installed.
        FileNotFoundError: If ``dem_file`` does not exist.
    """
    _require_whitebox()
    dem = Path(dem_file)
    if not dem.exists():
        raise FileNotFoundError(f"DEM file not found: {dem}")
    if _WBW is None:  # pragma: no cover - guarded by _require_whitebox
        raise ImportError("whitebox-workflows is not installed")

    wbt = _WBW.WbEnvironment()
    wbt.verbose = False
    dem_raster = wbt.read_raster(dem.as_posix())
    outlet_raster = None
    if outlet_file is not None:
        outlet = Path(outlet_file)
        if not outlet.exists():
            raise FileNotFoundError(f"Outlet raster not found: {outlet}")
        outlet_raster = wbt.read_raster(outlet.as_posix())

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    wbt.flow_accumulation(
        dem_raster, out.as_posix(), flow_type=flow_type, outlet=outlet_raster
    )
    return out


def whitebox_status() -> str:
    """Return a human-readable availability status for diagnostics."""
    if HAS_WHITEBOX:
        return f"whitebox-workflows: available (version {whitebox_version()})"
    return "whitebox-workflows: unavailable (terrain tools disabled)"


__all__ = [
    "HAS_WHITEBOX",
    "flow_accumulation",
    "whitebox_available",
    "whitebox_status",
    "whitebox_version",
]
