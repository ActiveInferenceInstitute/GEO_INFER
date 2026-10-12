"""
GEO-INFER-SPACE Integration Adapter

Provides spatial operations wrapper for economic analysis.
"""

from typing import Any, cast
import numpy as np
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

# Try to import GEO-INFER-SPACE modules
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
    from geo_infer_space.core.geometric_operations import GeometricOperationsInterface

    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    logger.warning(
        "GEO-INFER-SPACE not available. Spatial operations will be limited. "
        "Install geo-infer-space to enable full functionality."
    )


class SpaceIntegration:
    """
    Integration adapter for GEO-INFER-SPACE.

    Provides spatial operations for economic analysis including:
    - Spatial indexing (H3, SRAI)
    - Spatial analytics (hotspots, clustering, interpolation)
    - Geometric operations (buffers, intersections, distances)
    """

    def __init__(self, backend: str = "h3", config: dict[str, Any] | None = None):
        """
        Initialize space integration.

        Args:
            backend: Spatial backend to use ('h3' or 'srai')
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.backend = backend

        if not SPACE_AVAILABLE:
            logger.warning(
                "SpaceIntegration initialized but GEO-INFER-SPACE not available"
            )
            self.indexer = None
            self.analytics = None
            self.geometry = None
        else:
            try:
                self.indexer = SpatialIndexingInterface(backend=backend)
                self.analytics = SpatialAnalyticsInterface(
                    backend="srai" if backend == "srai" else "h3"
                )
                self.geometry = GeometricOperationsInterface()
                logger.info(f"SpaceIntegration initialized with backend: {backend}")
            except Exception as e:
                logger.error(f"Failed to initialize SpaceIntegration: {e}")
                self.indexer = None
                self.analytics = None
                self.geometry = None

    def latlng_to_cell(self, lat: float, lng: float, resolution: int = 9) -> str | None:
        """
        Convert lat/lng to spatial cell index.

        Args:
            lat: Latitude
            lng: Longitude
            resolution: Spatial resolution level

        Returns:
            Cell index string or None if unavailable
        """
        if not SPACE_AVAILABLE or self.indexer is None:
            logger.warning("Spatial indexing not available")
            return None

        try:
            cell = self.indexer.latlng_to_cell(lat, lng, resolution)
            if isinstance(cell, int):
                import h3

                return str(h3.int_to_str(cell))
            return str(cell)
        except Exception as e:
            logger.error(f"Failed to convert lat/lng to cell: {e}")
            return None

    def cell_to_latlng(self, cell: str | int) -> tuple[float, float] | None:
        """
        Convert spatial cell index to lat/lng.

        Args:
            cell: Cell index string

        Returns:
            (latitude, longitude) tuple or None if unavailable
        """
        if not SPACE_AVAILABLE or self.indexer is None:
            logger.warning("Spatial indexing not available")
            return None

        try:
            if isinstance(cell, int):
                import h3

                cell = h3.int_to_str(cell)
            return cast(tuple[float, float], self.indexer.cell_to_latlng(cell))
        except Exception as e:
            logger.error(f"Failed to convert cell to lat/lng: {e}")
            return None

    def calculate_distance(
        self, point1: tuple[float, float], point2: tuple[float, float]
    ) -> float | None:
        """
        Calculate distance between two points.

        Args:
            point1: (lat, lng) tuple
            point2: (lat, lng) tuple

        Returns:
            Distance in meters or None if unavailable
        """
        if not SPACE_AVAILABLE or self.geometry is None:
            # Fallback to simple calculation
            from math import radians, cos, sin, asin, sqrt

            lat1, lon1 = point1
            lat2, lon2 = point2
            R = 6371000  # Earth radius in meters
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = (
                sin(dlat / 2) ** 2
                + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
            )
            c = 2 * asin(sqrt(a))
            return R * c

        try:
            # Use geometry operations if available
            from shapely.geometry import Point

            p1 = Point(point1[1], point1[0])  # Note: shapely uses (x, y) = (lon, lat)
            p2 = Point(point2[1], point2[0])
            return float(p1.distance(p2)) * 111000  # Approximate conversion to meters
        except Exception as e:
            logger.error(f"Failed to calculate distance: {e}")
            return None

    def analyze_hotspots(
        self, gdf: gpd.GeoDataFrame, value_column: str, **kwargs: Any
    ) -> gpd.GeoDataFrame | None:
        """
        Analyze spatial hotspots in economic data.

        Args:
            gdf: GeoDataFrame with economic data
            value_column: Column name with values to analyze
            **kwargs: Additional parameters for hotspot analysis

        Returns:
            GeoDataFrame with hotspot analysis results or None if unavailable
        """
        if not SPACE_AVAILABLE or self.analytics is None:
            logger.warning("Spatial analytics not available for hotspot analysis")
            return None

        try:
            return self.analytics.analyze_hotspots(gdf, value_column, **kwargs)
        except Exception as e:
            logger.error(f"Failed to analyze hotspots: {e}")
            return None

    def spatial_interpolation(
        self,
        points: gpd.GeoDataFrame,
        values: np.ndarray,
        target_locations: gpd.GeoDataFrame,
        method: str = "idw",
        **kwargs: Any,
    ) -> np.ndarray | None:
        """
        Perform spatial interpolation of economic values.

        Args:
            points: GeoDataFrame with known point locations
            values: Array of values at known points
            target_locations: GeoDataFrame with target locations
            method: Interpolation method ('idw', 'kriging', etc.)
            **kwargs: Additional parameters for interpolation

        Returns:
            Interpolated values array or None if unavailable
        """
        if not SPACE_AVAILABLE or self.analytics is None:
            logger.warning("Spatial analytics not available for interpolation")
            return None

        try:
            return cast(
                np.ndarray,
                self.analytics.spatial_interpolation(
                    points, values, target_locations, method=method, **kwargs
                ),
            )
        except Exception as e:
            logger.error(f"Failed to perform spatial interpolation: {e}")
            return None

    def create_buffer(
        self, geometry: gpd.GeoDataFrame, distance: float, **kwargs: Any
    ) -> gpd.GeoDataFrame | None:
        """
        Create buffer zones around geometries.

        Args:
            geometry: GeoDataFrame with geometries
            distance: Buffer distance in meters
            **kwargs: Additional parameters for buffering

        Returns:
            GeoDataFrame with buffered geometries or None if unavailable
        """
        if not SPACE_AVAILABLE or self.geometry is None:
            # Fallback to geopandas buffer
            try:
                return geometry.buffer(distance / 111000)  # Approximate conversion
            except Exception as e:
                logger.error(f"Failed to create buffer: {e}")
                return None

        try:
            return self.geometry.buffer(geometry, distance, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create buffer: {e}")
            return None

    def is_available(self) -> bool:
        """Check if GEO-INFER-SPACE is available."""
        return SPACE_AVAILABLE and self.indexer is not None
