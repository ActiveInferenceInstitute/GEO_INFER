"""
Spatial Utilities - Utility functions for spatial operations in GEO-INFER-SPACE.

This module provides common spatial utility functions for coordinate transformations,
distance calculations, and spatial data processing.
"""

import logging
from typing import Any, cast
import numpy as np
from shapely.geometry import Point, Polygon
from pyproj import Transformer
from .core import SpatialIndexingInterface

logger = logging.getLogger(__name__)


class SpatialUtils:
    """
    Utility class for spatial operations and transformations.

    Provides common spatial functions including:
    - Coordinate transformations
    - Distance calculations
    - Spatial indexing utilities
    - Data validation and processing
    """

    def __init__(self) -> None:
        """Initialize SpatialUtils."""
        self.transformers: dict[str, Transformer] = {}
        self.indexer = SpatialIndexingInterface()
        logger.info("SpatialUtils initialized")

    def get_transformer(self, from_crs: str, to_crs: str) -> Transformer:
        """
        Get or create a coordinate transformer.

        Args:
            from_crs: Source coordinate reference system
            to_crs: Target coordinate reference system

        Returns:
            PyProj transformer object
        """
        key = f"{from_crs}_{to_crs}"
        if key not in self.transformers:
            self.transformers[key] = Transformer.from_crs(
                from_crs, to_crs, always_xy=True
            )
        return self.transformers[key]

    def transform_coordinates(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        from_crs: str = "EPSG:4326",
        to_crs: str = "EPSG:3857",
    ) -> tuple[float, float] | list[tuple[float, float]]:
        """
        Transform coordinates between coordinate reference systems.

        Args:
            coords: Single ``(x, y)`` tuple or list of tuples. For WGS84,
                coordinates are ``(longitude, latitude)`` because transformations
                use the conventional GIS axis order.
            from_crs: Source CRS (default: WGS84)
            to_crs: Target CRS (default: Web Mercator)

        Returns:
            Transformed coordinates
        """
        transformer = self.get_transformer(from_crs, to_crs)

        if isinstance(coords, tuple):
            x, y = transformer.transform(coords[0], coords[1])
            return (x, y)
        else:
            if not coords:
                return []

            # PyProj transform is vectorized and accepts arrays/lists natively.
            xs = [coordinate[0] for coordinate in coords]
            ys = [coordinate[1] for coordinate in coords]
            x, y = transformer.transform(xs, ys)

            # Pack back into list of tuples
            return list(zip(x, y))

    def calculate_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        method: str = "haversine",
    ) -> float:
        """
        Calculate distance between two points.

        Args:
            point1: First point (lat, lon)
            point2: Second point (lat, lon)
            method: Distance calculation method ('haversine', 'euclidean')

        Returns:
            Distance in kilometers
        """
        if method == "haversine":
            return self._haversine_distance(point1, point2)
        elif method == "euclidean":
            return self._euclidean_distance(point1, point2)
        else:
            raise ValueError(f"Unknown distance method: {method}")

    def _haversine_distance(
        self, point1: tuple[float, float], point2: tuple[float, float]
    ) -> float:
        """
        Calculate haversine distance between two points.

        Delegates to :func:`geo_infer_space.utils.h3_utils.haversine_distance`
        — the canonical single-point implementation.
        """
        from .utils.h3_utils import haversine_distance

        return haversine_distance(point1, point2)

    def _euclidean_distance(
        self, point1: tuple[float, float], point2: tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return cast(
            float,
            np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2),
        )

    def create_buffer(
        self, center: tuple[float, float], radius_km: float, resolution: int = 16
    ) -> Polygon:
        """
        Create a circular buffer around a point.

        Args:
            center: Center point (lat, lon)
            radius_km: Buffer radius in kilometers
            resolution: Polygon resolution (number of segments)

        Returns:
            Buffer polygon

        Note:
            Approximation: converts the radius to degrees with a flat
            111 km-per-degree factor, ignoring latitude-dependent
            convergence; results are rough away from the equator.
        """
        center_point = Point(center[1], center[0])
        return center_point.buffer(radius_km / 111.0, resolution=resolution)

    def get_h3_cells_in_polygon(
        self, polygon: Polygon, resolution: int = 7
    ) -> list[str]:
        """
        Get H3 cells that intersect with a polygon.

        Args:
            polygon: Shapely polygon
            resolution: H3 resolution

        Returns:
            List of H3 cell identifiers
        """
        # Use Unified Spatial Interface
        cells = self.indexer.polygon_to_cells(
            {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]},
            resolution,
        )

        return list(cells)

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate coordinate values.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if coordinates are valid
        """
        return -90 <= lat <= 90 and -180 <= lon <= 180

    def snap_to_h3_grid(
        self, lat: float, lon: float, resolution: int = 7
    ) -> tuple[float, float]:
        """
        Snap coordinates to H3 grid center.

        Args:
            lat: Latitude
            lon: Longitude
            resolution: H3 resolution

        Returns:
            Snapped coordinates (lat, lon)
        """
        cell = self.indexer.latlng_to_cell(lat, lon, resolution)
        center_lat, center_lon = self.indexer.cell_to_latlng(cell)
        return center_lat, center_lon

    def calculate_spatial_density(
        self, points: list[tuple[float, float]], area_km2: float
    ) -> float:
        """
        Calculate spatial density of points.

        Args:
            points: List of point coordinates
            area_km2: Area in square kilometers

        Returns:
            Points per square kilometer
        """
        if area_km2 <= 0:
            return 0.0
        return len(points) / area_km2

    def find_nearest_point(
        self, target: tuple[float, float], candidates: list[tuple[float, float]]
    ) -> tuple[int, float]:
        """
        Find the nearest point from a list of candidates.

        Args:
            target: Target point (lat, lon)
            candidates: List of candidate points

        Returns:
            Tuple of (index, distance_km)
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        # Vectorized Haversine distance
        cands = np.radians(np.array(candidates))
        cand_lats, cand_lons = cands[:, 0], cands[:, 1]
        target_lat, target_lon = np.radians(target)

        dlat = cand_lats - target_lat
        dlon = cand_lons - target_lon

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(target_lat) * np.cos(cand_lats) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        distances = c * 6371.0  # Earth radius in kilometers

        min_idx = int(np.argmin(distances))
        return min_idx, float(distances[min_idx])

    def create_spatial_index(
        self,
        points: list[tuple[float, float]],
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a spatial index for efficient point queries.

        Args:
            points: List of point coordinates
            labels: Optional labels for points

        Returns:
            Spatial index dictionary
        """
        if labels is None:
            labels = [f"point_{i}" for i in range(len(points))]

        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"label": labels},
            geometry=[Point(p[1], p[0]) for p in points],  # point expects (lon, lat)
        )

        rtree_index = None
        try:
            from rtree import index as rtree_ix

            rtree_index = rtree_ix.Index()
            for i, p in enumerate(points):
                # (minx, miny, maxx, maxy) -> lon, lat
                rtree_index.insert(i, (p[1], p[0], p[1], p[0]))
        except ImportError:
            logger.debug("rtree library not available, fallback to geopandas sindex.")

        index_data = {
            "points": points,
            "labels": labels,
            "bounds": self._calculate_bounds(points),
            "centroid": self._calculate_centroid(points),
            "sindex": gdf.sindex,
            "rtree": rtree_index,
        }

        return index_data

    def _calculate_bounds(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float, float, float]:
        """Calculate bounding box for points."""
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return min(lats), min(lons), max(lats), max(lons)

    def _calculate_centroid(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float]:
        """Calculate centroid of points."""
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return cast(tuple[float, float], (np.mean(lats), np.mean(lons)))

    def filter_points_by_distance(
        self,
        center: tuple[float, float],
        points: list[tuple[float, float]],
        max_distance_km: float,
    ) -> list[tuple[float, float]]:
        """
        Filter points within a maximum distance from center.

        Args:
            center: Center point (lat, lon)
            points: List of points to filter
            max_distance_km: Maximum distance in kilometers

        Returns:
            Filtered list of points
        """
        filtered = []
        for point in points:
            distance = self.calculate_distance(center, point)
            if distance <= max_distance_km:
                filtered.append(point)
        return filtered
