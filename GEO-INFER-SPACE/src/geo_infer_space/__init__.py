"""
GEO-INFER-SPACE - Advanced geospatial methods for the GEO-INFER framework.

This module provides powerful spatial indexing, analytics, and integration
with external geospatial tools and libraries through a unified, backend-agnostic API.
"""

__version__ = "0.4.0"

# Import the generic spatial interfaces
from .core.spatial_indexing import (
    SpatialIndexingInterface,
    latlng_to_cell,
    cell_to_latlng,
    polygon_to_cells,
)

from .core.geometric_operations import GeometricOperationsInterface
from .core.analytics import SpatialAnalyticsInterface
from .core.dispatcher import get_backend_dispatcher, configure_backends
from .core.interfaces import UnsupportedSpatialOperationError
from .core.state_space import H3StateSpace
from .core.spatiotemporal import align_h3_observations


# RISK-style policy: internal components import only declared hard
# dependencies, so these imports must succeed; a failure is a real
# packaging bug and propagates instead of silently nulling the public API.
from .place_analyzer import PlaceAnalyzer

from .spatial_utils import SpatialUtils

# Import the GIS submodule facade
from .gis import GISManager

# Make core functionality easily accessible
__all__ = [
    # Generic spatial interfaces
    "SpatialIndexingInterface",
    "GeometricOperationsInterface",
    "SpatialAnalyticsInterface",
    "get_backend_dispatcher",
    "configure_backends",
    "UnsupportedSpatialOperationError",
    "H3StateSpace",
    "align_h3_observations",
    # Convenience functions
    "latlng_to_cell",
    "cell_to_latlng",
    "polygon_to_cells",
    # Internal components
    "PlaceAnalyzer",
    "SpatialUtils",
    "GISManager",
]
