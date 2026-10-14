"""
GEO-INFER-PLACE: Place-Based Geospatial Analysis Framework

Comprehensive place-based analysis for specific geographic locations,
integrating multiple GEO-INFER modules (SPACE, DATA, TIME) to create
deep, location-specific insights.

Key Features:
- Location-specific data integration and analysis
- Multi-domain analysis (forest, coastal, fire, seismic hazard)
- Real-time data from USGS, NOAA, CAL FIRE with retry + caching
- GEO-INFER-DATA quality management and provenance tracking
- GEO-INFER-TIME temporal trend detection and forecasting
- Interactive H3 visualization and dashboard generation

Supported Locations:
- Del Norte County, California (forest health, coastal resilience, fire risk, seismic hazard)
- Cascadia Bioregion (agricultural land analysis, subduction zone seismicity)

Quick Start::

    from geo_infer_place import PlaceInterface

    pi = PlaceInterface("del_norte")
    results = pi.run_full_analysis()
    print(pi.status())
"""

from __future__ import annotations

from importlib import import_module
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.place_interface import PlaceInterface
    from .core.visualization_engine import InteractiveVisualizationEngine
    from .core import CascadianAgriculturalH3Backend, BaseAnalysisModule
    from .core.module_bridge import PlaceDataManager, PlaceTemporalAnalyzer
    from .utils.caching import CachedAPIWrapper
    from .locations.del_norte_county.forest_health_monitor import ForestHealthMonitor
    from .locations.del_norte_county.coastal_resilience_analyzer import (
        CoastalResilienceAnalyzer,
    )
    from .locations.del_norte_county.fire_risk_assessor import FireRiskAssessor
    from .locations.del_norte_county.seismic_hazard_analyzer import (
        SeismicHazardAnalyzer,
    )
    from .utils.data_sources import CaliforniaDataSources
    from .utils.h3_operations import (
        latlng_to_cell,
        cell_to_latlng,
        cell_to_latlng_boundary,
        geo_to_cells,
        polygon_to_cells,
        grid_disk,
        grid_distance,
        grid_ring,
        cell_area,
        get_resolution,
        is_valid_cell,
        are_neighbor_cells,
        cells_to_geodataframe,
        cell_to_parent,
        cell_to_children,
        compact_cells,
        uncompact_cells,
        estimate_cell_count,
    )
    from .core.api_clients import (
        CaliforniaAPIManager,
        NOAAClient,
        CALFIREClient,
        USGSClient,
        USGSEarthquakeClient,
        CDECClient,
    )

_EXPORTS = {
    "PlaceInterface": "core.place_interface",
    "InteractiveVisualizationEngine": "core.visualization_engine",
    "CascadianAgriculturalH3Backend": "core.unified_backend",
    "BaseAnalysisModule": "core.base_module",
    "PlaceDataManager": "core.module_bridge",
    "PlaceTemporalAnalyzer": "core.module_bridge",
    "CachedAPIWrapper": "utils.caching",
    "ForestHealthMonitor": "locations.del_norte_county.forest_health_monitor",
    "CoastalResilienceAnalyzer": "locations.del_norte_county.coastal_resilience_analyzer",
    "FireRiskAssessor": "locations.del_norte_county.fire_risk_assessor",
    "SeismicHazardAnalyzer": "locations.del_norte_county.seismic_hazard_analyzer",
    "CaliforniaDataSources": "utils.data_sources",
    **dict.fromkeys(
        (
            "latlng_to_cell",
            "cell_to_latlng",
            "cell_to_latlng_boundary",
            "geo_to_cells",
            "polygon_to_cells",
            "grid_disk",
            "grid_distance",
            "grid_ring",
            "cell_area",
            "get_resolution",
            "is_valid_cell",
            "are_neighbor_cells",
            "cells_to_geodataframe",
            "cell_to_parent",
            "cell_to_children",
            "compact_cells",
            "uncompact_cells",
            "estimate_cell_count",
        ),
        "utils.h3_operations",
    ),
    **dict.fromkeys(
        (
            "CaliforniaAPIManager",
            "NOAAClient",
            "CALFIREClient",
            "USGSClient",
            "USGSEarthquakeClient",
            "CDECClient",
        ),
        "core.api_clients",
    ),
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: object = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"
__email__ = "geo-infer@activeinference.institute"

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Export public API
__all__ = [
    # Unified interface (primary entry point)
    "PlaceInterface",
    # Module bridges
    "PlaceDataManager",
    "PlaceTemporalAnalyzer",
    # Core components (local)
    "InteractiveVisualizationEngine",
    "CascadianAgriculturalH3Backend",
    "BaseAnalysisModule",
    # Infrastructure
    "CachedAPIWrapper",
    # Del Norte County analyzers
    "ForestHealthMonitor",
    "CoastalResilienceAnalyzer",
    "FireRiskAssessor",
    "SeismicHazardAnalyzer",
    # Utilities
    "CaliforniaDataSources",
    "latlng_to_cell",
    "cell_to_latlng",
    "cell_to_latlng_boundary",
    "geo_to_cells",
    "polygon_to_cells",
    "grid_disk",
    "grid_distance",
    "grid_ring",
    "cell_area",
    "get_resolution",
    "is_valid_cell",
    "are_neighbor_cells",
    "cells_to_geodataframe",
    "cell_to_parent",
    "cell_to_children",
    "compact_cells",
    "uncompact_cells",
    "estimate_cell_count",
    # API clients
    "CaliforniaAPIManager",
    "NOAAClient",
    "CALFIREClient",
    "USGSClient",
    "USGSEarthquakeClient",
    "CDECClient",
]


def get_supported_locations() -> list[str]:
    """Get list of supported analysis locations."""
    from .core.place_interface import LOCATION_PRESETS

    return list(LOCATION_PRESETS.keys())


def create_analyzer(
    location_code: str, config_path: str | None = None
) -> PlaceInterface:
    """Create a PlaceInterface for a specific location.

    Args:
        location_code: Code for the location to analyze (``"del_norte"`` or ``"cascadia"``).
        config_path: Ignored; reserved for future use.

    Returns:
        Configured PlaceInterface instance.

    Raises:
        ValueError: If location_code is not supported.
    """
    supported = get_supported_locations()
    if location_code not in supported:
        raise ValueError(
            f"Location '{location_code}' not supported. Available: {supported}"
        )
    from .core.place_interface import PlaceInterface

    return PlaceInterface(location=location_code)


def create_place_interface(
    location: str = "del_norte",
    output_dir: str | None = None,
) -> PlaceInterface:
    """Convenience factory for PlaceInterface.

    Args:
        location: Location key (``"del_norte"`` or ``"cascadia"``).
        output_dir: Optional output directory path.

    Returns:
        Configured PlaceInterface instance.
    """
    from .core.place_interface import PlaceInterface

    return PlaceInterface(location=location, output_dir=output_dir)
