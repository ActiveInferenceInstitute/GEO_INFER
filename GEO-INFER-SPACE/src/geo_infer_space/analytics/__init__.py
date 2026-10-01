"""
Advanced spatial analytics module for GEO-INFER-SPACE.

This module provides comprehensive spatial analysis capabilities including:
- Vector operations (overlay, buffer, proximity)
- Raster analysis (terrain, map algebra, focal statistics)
- Network analysis (routing, service areas)
- Geostatistics (interpolation, clustering, hotspot detection)
- Spatio-temporal analysis
"""

import logging

# Core analytics - always available
from .temporal import TemporalAnalyzer
from .spatiotemporal import SpatioTemporalAnalyzer

logger = logging.getLogger(__name__)

# Vector operations
try:
    from .vector import (
        buffer_and_intersect,
        overlay_analysis,
        proximity_analysis,
        spatial_join_analysis,
        geometric_calculations,
        topology_operations,
    )

    VECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector analytics not available: {e}")
    VECTOR_AVAILABLE = False

# Raster operations (requires rasterio)
try:
    from .raster import (
        terrain_analysis as terrain_analysis,
        map_algebra as map_algebra,
        focal_statistics as focal_statistics,
        zonal_statistics as zonal_statistics,
        raster_overlay as raster_overlay,
        image_processing as image_processing,
    )

    RASTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Raster analytics not available: {e}")
    RASTER_AVAILABLE = False

# Network operations
try:
    from .network import (
        shortest_path as shortest_path,
        service_area as service_area,
        network_connectivity as network_connectivity,
        routing_analysis as routing_analysis,
        accessibility_analysis as accessibility_analysis,
    )

    NETWORK_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Network analytics not available: {e}")
    NETWORK_AVAILABLE = False

# Geostatistics
try:
    from .geostatistics import (
        spatial_interpolation,
        clustering_analysis,
        hotspot_detection,
        spatial_autocorrelation,
        variogram_analysis,
    )

    GEOSTATISTICS_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Geostatistics not available: {e}")
    GEOSTATISTICS_AVAILABLE = False

# Point cloud
try:
    from .point_cloud import (
        point_cloud_filtering as point_cloud_filtering,
        feature_extraction as feature_extraction,
        classification as classification,
        surface_generation as surface_generation,
    )

    POINT_CLOUD_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Point cloud analytics not available: {e}")
    POINT_CLOUD_AVAILABLE = False

__all__ = [
    # Always available
    "TemporalAnalyzer",
    "SpatioTemporalAnalyzer",
    # Vector operations (if available)
    "buffer_and_intersect",
    "overlay_analysis",
    "proximity_analysis",
    "spatial_join_analysis",
    "geometric_calculations",
    "topology_operations",
    # Raster operations
    "terrain_analysis",
    "map_algebra",
    # Geostatistical functions
    "spatial_interpolation",
    "clustering_analysis",
    "hotspot_detection",
    "spatial_autocorrelation",
    "variogram_analysis",
]
