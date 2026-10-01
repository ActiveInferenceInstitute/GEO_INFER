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

import logging

# --- Local Core Imports ---
from .core.visualization_engine import InteractiveVisualizationEngine
from .core import CascadianAgriculturalH3Backend, BaseAnalysisModule

# --- Unified Interface ---
from .core.place_interface import PlaceInterface

# --- Module Bridge (GEO-INFER-DATA / GEO-INFER-TIME integration) ---
from .core.module_bridge import PlaceDataManager, PlaceTemporalAnalyzer

# --- Caching infrastructure ---
from .utils.caching import CachedAPIWrapper

# --- Location-specific Imports ---
from .locations.del_norte_county.forest_health_monitor import ForestHealthMonitor
from .locations.del_norte_county.coastal_resilience_analyzer import (
    CoastalResilienceAnalyzer,
)
from .locations.del_norte_county.fire_risk_assessor import FireRiskAssessor
from .locations.del_norte_county.seismic_hazard_analyzer import SeismicHazardAnalyzer

# --- Utilities ---
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

# --- API Clients ---
from .core.api_clients import (
    CaliforniaAPIManager,
    NOAAClient,
    CALFIREClient,
    USGSClient,
    USGSEarthquakeClient,
    CDECClient,
)

__version__ = "0.3.0"
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
) -> "PlaceInterface":
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
    return PlaceInterface(location=location, output_dir=output_dir)
