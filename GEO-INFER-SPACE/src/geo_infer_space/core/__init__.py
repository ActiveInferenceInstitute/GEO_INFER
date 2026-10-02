"""
Generic spatial methods core layer for GEO-INFER-SPACE.

This module provides generic spatial operations that can dispatch to
different backends (H3, SRAI, etc.) based on configuration.
"""

__version__ = "0.4.0"

from .dispatcher import SpatialBackendDispatcher, get_backend_dispatcher
from .spatial_indexing import SpatialIndexingInterface
from .geometric_operations import GeometricOperationsInterface
from .analytics import SpatialAnalyticsInterface
from .statistics import SpatialStatistics
from .interfaces import UnsupportedSpatialOperationError
from .state_space import H3StateSpace
from .spatiotemporal import align_h3_observations
from .h3_policy import (
    H3_AVG_AREA_KM2,
    H3_DEFAULT_TARGET_CELLS,
    H3_DEFAULT_MAX_RES,
    H3_HARD_CELL_CAP,
    H3HardCapExceededError,
    ResolutionSuggestion,
    check_cell_budget,
    estimate_cell_count,
    suggest_h3_resolution,
    suggest_resolution_with_budget,
)
from .geolibre_projects import (
    GEOLIBRE_PROJECT_VERSION,
    DEFAULT_BASEMAP_STYLE_URL,
    DEFAULT_LAYER_STYLE,
    DEFAULT_PROJECT_PREFERENCES,
    build_h3_grid_project,
    build_project,
    default_map_view,
    dumps_project,
    geojson_layer,
    tile_layer,
    write_project,
)
from .algorithm_registry import (
    AlgorithmRegistry,
    ParameterSpec,
    ProcessingAlgorithm,
    ProcessingContext,
    build_reference_registry,
)
from .whitebox_bridge import (
    HAS_WHITEBOX,
    flow_accumulation,
    whitebox_available,
    whitebox_status,
    whitebox_version,
)

__all__ = [
    "SpatialBackendDispatcher",
    "get_backend_dispatcher",
    "SpatialIndexingInterface",
    "GeometricOperationsInterface",
    "SpatialAnalyticsInterface",
    "SpatialStatistics",
    "UnsupportedSpatialOperationError",
    "H3StateSpace",
    "align_h3_observations",
    "H3_AVG_AREA_KM2",
    "H3_DEFAULT_TARGET_CELLS",
    "H3_DEFAULT_MAX_RES",
    "H3_HARD_CELL_CAP",
    "H3HardCapExceededError",
    "ResolutionSuggestion",
    "check_cell_budget",
    "estimate_cell_count",
    "suggest_h3_resolution",
    "suggest_resolution_with_budget",
    "GEOLIBRE_PROJECT_VERSION",
    "DEFAULT_BASEMAP_STYLE_URL",
    "DEFAULT_LAYER_STYLE",
    "DEFAULT_PROJECT_PREFERENCES",
    "build_h3_grid_project",
    "build_project",
    "default_map_view",
    "dumps_project",
    "geojson_layer",
    "tile_layer",
    "write_project",
    "AlgorithmRegistry",
    "ParameterSpec",
    "ProcessingAlgorithm",
    "ProcessingContext",
    "build_reference_registry",
    "HAS_WHITEBOX",
    "flow_accumulation",
    "whitebox_available",
    "whitebox_status",
    "whitebox_version",
]
