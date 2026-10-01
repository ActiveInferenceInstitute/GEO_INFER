"""
GIS Submodule - Unified Geographic Information System interface for GEO-INFER-SPACE.

This module provides a consolidated `GISManager` class that unifies access to
SpatialMethods, SpatialProcessor, and SpatialUtils, ensuring all generic and
implementation-specific Spatial methods are available through a single interface.
"""

import logging
from typing import Any

import geopandas as gpd

from ..core.spatial_methods import SpatialMethods
from ..core.spatial_processor import SpatialProcessor
from ..spatial_utils import SpatialUtils

logger = logging.getLogger(__name__)


class GISManager:
    """
    Unified Geographic Information System (GIS) Manager.

    This class consolidates all generic and implementation-specific spatial
    methods (vector, raster, indexing, spatial statistics) under a single facade.
    It orchestrates operations utilizing internal `SpatialMethods`, `SpatialProcessor`,
    and `SpatialUtils` instances.
    """

    def __init__(self, h3_backend: Any | None = None) -> None:
        """
        Initialize the GISManager and its underlying spatial components.

        Args:
            h3_backend: Optional H3Backend instance for spatial indexing.
        """
        self.methods = SpatialMethods(h3_backend=h3_backend)
        self.processor = SpatialProcessor()
        self.utils = SpatialUtils()

        logger.info("GISManager initialized")

    # =========================================================================
    # VECTOR OPERATIONS & GEOMETRY (via SpatialProcessor & SpatialUtils)
    # =========================================================================

    def buffer_analysis(
        self, gdf: "gpd.GeoDataFrame", buffer_distance: float
    ) -> "gpd.GeoDataFrame":
        """
        Create buffers around geometries in a GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame.
            buffer_distance: Distance in units of the CRS.

        Returns:
            GeoDataFrame with buffered geometries.
        """
        return self.processor.buffer_analysis(gdf, buffer_distance)

    def proximity_analysis(
        self, gdf1: "gpd.GeoDataFrame", gdf2: "gpd.GeoDataFrame"
    ) -> dict[str, Any]:
        """
        Calculate proximity between two sets of features.

        Args:
            gdf1: First GeoDataFrame.
            gdf2: Second GeoDataFrame.

        Returns:
            Dict containing min, max, and mean distance.
        """
        return self.processor.proximity_analysis(gdf1, gdf2)

    def perform_multi_overlay(
        self, spatial_datasets: dict[str, "gpd.GeoDataFrame"]
    ) -> "gpd.GeoDataFrame":
        """
        Perform multi-layer spatial overlay on multiple GeoDataFrames.

        Args:
            spatial_datasets: Dictionary mapping dataset names to GeoDataFrames.

        Returns:
            Single GeoDataFrame with overlaid geometries and attributes.
        """
        return self.processor.perform_multi_overlay(spatial_datasets)

    def transform_coordinates(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        from_crs: str = "EPSG:4326",
        to_crs: str = "EPSG:3857",
    ) -> tuple[float, float] | list[tuple[float, float]]:
        """
        Transform coordinates between coordinate reference systems.

        Args:
            coords: A single (lon, lat) tuple or a list of tuples.
            from_crs: The source Coordinate Reference System (default: EPSG:4326/WGS84).
            to_crs: The destination Coordinate Reference System (default: EPSG:3857/Web Mercator).

        Returns:
            The transformed coordinates in the same format as the input.
        """
        return self.utils.transform_coordinates(coords, from_crs, to_crs)

    def calculate_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        method: str = "haversine",
    ) -> float:
        """
        Calculate distance between two points.

        Args:
            point1: The first point as a (latitude, longitude) tuple.
            point2: The second point as a (latitude, longitude) tuple.
            method: The algorithm to use, either "haversine" or "euclidean" (default: "haversine").

        Returns:
            The calculated distance in kilometers.
        """
        return self.utils.calculate_distance(point1, point2, method)

    # =========================================================================
    # SPATIAL INDEXING & HEXAGONAL GRIDS (via SpatialMethods)
    # =========================================================================

    def cell_buffer_analysis(
        self, cells: list[str], buffer_rings: int = 1, include_center: bool = True
    ) -> dict[str, Any]:
        """
        Create buffer zones around H3 cells.

        Args:
            cells: List of H3 cell IDs indicating the origin points.
            buffer_rings: Number of grid rings to buffer out (default: 1).
            include_center: Whether to include original origin cells in output (default: True).

        Returns:
            Dictionary containing buffer counts and expanded cell lists.
        """
        return self.methods.buffer_analysis(cells, buffer_rings, include_center)

    def overlay_cells(
        self, cells_a: list[str], cells_b: list[str], operation: str = "intersection"
    ) -> dict[str, Any]:
        """
        Perform overlay operations between two cell sets.

        Args:
            cells_a: The first set of H3 cell IDs.
            cells_b: The second set of H3 cell IDs.
            operation: Logical operation ('intersection', 'union', 'difference', 'symmetric_difference').

        Returns:
            Dictionary containing operation metadata and the resulting cell list.
        """
        return self.methods.overlay_cells(cells_a, cells_b, operation)

    def aggregate_to_region(
        self,
        cells: list[str],
        values: list[float],
        target_resolution: int,
        aggregation: str = "mean",
    ) -> dict[str, Any]:
        """
        Aggregate cell values to a coarser resolution.

        Args:
            cells: List of H3 cell IDs to aggregate.
            values: List of corresponding numeric values.
            target_resolution: Target parent H3 resolution (must be <= current maximum resolution).
            aggregation: Aggregation method ('mean', 'sum', 'max', 'min', 'count') (default: 'mean').

        Returns:
            Dictionary grouping the values to their parent grid cells.
        """
        return self.methods.aggregate_to_region(
            cells, values, target_resolution, aggregation
        )

    def disaggregate_to_cells(
        self,
        parent_cells: list[str],
        values: list[float],
        target_resolution: int,
        method: str = "equal",
    ) -> dict[str, Any]:
        """
        Disaggregate values to finer resolution cells.

        Args:
            parent_cells: List of coarser parent H3 cell IDs.
            values: Values corresponding to each parent cell.
            target_resolution: Target child H3 resolution (must be >= current resolution).
            method: Method to distribute values, 'equal' (split payload) or 'proportional' (replicate) (default: 'equal').

        Returns:
            Dictionary mapping child cells to distributed values.
        """
        return self.methods.disaggregate_to_cells(
            parent_cells, values, target_resolution, method
        )

    def calculate_coverage(
        self, cells: list[str], region_cells: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Calculate coverage statistics for cell sets.

        Args:
            cells: The target H3 cell IDs to analyze for footprint coverage.
            region_cells: Optional bounding region cells to compute overlap ratios against.

        Returns:
            Dictionary containing areal measurements in km^2 and cell resolution distributions.
        """
        return self.methods.calculate_coverage(cells, region_cells)

    # =========================================================================
    # SPATIAL STATS, ANALYTICS & FILTERING (via SpatialMethods & SpatialProcessor)
    # =========================================================================

    def spatial_filter(
        self,
        cells: list[str],
        values: list[float],
        filter_type: str = "threshold",
        threshold: float | None = None,
        percentile: float | None = None,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """
        Filter cells based on spatial criteria.

        Args:
            cells: List of candidate H3 cell IDs.
            values: Corresponding list of values for each cell.
            filter_type: Method to filter by ('threshold', 'percentile', 'top_n', 'outliers').
            threshold: Value threshold (used if filter_type='threshold').
            percentile: Percentile threshold (used if filter_type='percentile').
            top_n: Number of top cells to retain (used if filter_type='top_n').

        Returns:
            Dictionary containing filtered arrays of cells and values.
        """
        return self.methods.spatial_filter(
            cells, values, filter_type, threshold, percentile, top_n
        )

    def find_spatial_outliers(
        self, cells: list[str], values: list[float], k: int = 1
    ) -> dict[str, Any]:
        """
        Find spatial outliers using Local Moran's I.

        Args:
            cells: List of interconnected H3 cell IDs.
            values: Values corresponding to each cell.
            k: Grid neighborhood ring size for analysis (default: 1).

        Returns:
            Dictionary classifying cells into significant outliers (HL, LH) and clusters (HH, LL).
        """
        return self.methods.find_spatial_outliers(cells, values, k)

    def calculate_spatial_correlation(
        self, gdf: "gpd.GeoDataFrame"
    ) -> dict[str, float]:
        """
        Calculate spatial correlation metrics for a GeoDataFrame.

        Args:
            gdf: A GeoDataFrame containing geometric features to evaluate.

        Returns:
            Dictionary mapping 'spatial_correlation' to the computed index value.
        """
        return self.processor.calculate_spatial_correlation(gdf)

    def calculate_spatial_weights(
        self, cells: list[str], weight_type: str = "queen", k: int = 1
    ) -> dict[str, Any]:
        """
        Calculate spatial weights matrix for cells.

        Args:
            cells: Target list of H3 cell IDs.
            weight_type: Typology strategy ('queen', 'rook', 'distance') (default: 'queen').
            k: Neighborhood ring constraint (default: 1).

        Returns:
            Weights matrix representing connectivity influence.
        """
        return self.methods.calculate_spatial_weights(cells, weight_type, k)

    # =========================================================================
    # NETWORK & ACCESSIBILITY
    # =========================================================================

    def compute_accessibility(
        self,
        origin_cells: list[str],
        destination_cells: list[str],
        max_distance: int = 10,
    ) -> dict[str, Any]:
        """
        Compute grid-based accessibility from origins to destinations.

        Args:
            origin_cells: Starting point H3 cell IDs.
            destination_cells: Endpoint H3 cell IDs.
            max_distance: Maximum travel distance permitted in grid units (default: 10).

        Returns:
            Dictionary mapping accessibility scores out of 1.0 for each origin cell.
        """
        return self.methods.compute_accessibility(
            origin_cells, destination_cells, max_distance
        )

    def find_nearest_point(
        self, target: tuple[float, float], candidates: list[tuple[float, float]]
    ) -> tuple[int, float]:
        """
        Find the nearest point from a list of candidates.

        Args:
            target: The query coordinates as a (lon, lat) tuple.
            candidates: List of candidate coordinates to evaluate as (lon, lat) tuples.

        Returns:
            Tuple containing the index of the nearest point in the candidates list and its distance in kilometers.
        """
        return self.utils.find_nearest_point(target, candidates)
