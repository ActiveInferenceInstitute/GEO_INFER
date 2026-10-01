"""
Spatial and temporal indexing utilities for GEO-INFER-DATA.

This module provides indexing utilities for efficient spatial and temporal
queries including H3 and R-tree indexing strategies.
"""

import logging
import math
from typing import Any, cast

import geopandas as gpd
import h3
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon
from rtree import index as rtree_index_module
from shapely.ops import transform as transform_geometry


logger = logging.getLogger(__name__)


def _require_h3() -> Any:
    """Return the h3-py module after enforcing the supported v4 range."""
    version = tuple(
        int(part.split("+")[0].split("-")[0])
        for part in h3.__version__.lstrip("v").split(".")[:3]
    )
    if version < (4, 5, 0) or version >= (5, 0, 0):
        raise RuntimeError(
            f"Unsupported h3-py version {h3.__version__}; "
            "GEO-INFER requires h3-py>=4.5.0,<5"
        )
    return h3


class SpatialIndexer:
    """
    Spatial indexing for efficient geospatial queries.

    This class provides spatial indexing capabilities using various
    strategies including H3 and R-tree indexing.

    Examples:
        >>> indexer = SpatialIndexer()
        >>>
        >>> # Create spatial index
        >>> index = indexer.create_spatial_index(geodataframe, 'h3')
        >>>
        >>> # Query by spatial bounds
        >>> results = indexer.query_by_bounds(index, bbox=[-122.5, 37.7, -122.3, 37.9])
        >>>
        >>> # Convert lat/lng to H3 cell
        >>> cell = indexer.latlng_to_cell(37.7749, -122.4194, resolution=9)
    """

    def __init__(self) -> None:
        self.indexes: dict[str, Any] = {}
        logger.info("Initialized SpatialIndexer")

    def create_spatial_index(self, data: gpd.GeoDataFrame, strategy: str = "h3") -> str:
        """
        Create spatial index for geospatial data.

        Args:
            data: GeoDataFrame to index
            strategy: Indexing strategy ('h3', 'rtree')

        Returns:
            Index identifier
        """
        logger.info(f"Creating spatial index with {strategy} strategy")

        index_id = f"spatial_{strategy}_{len(self.indexes)}"

        if strategy == "h3":
            self.indexes[index_id] = self._create_h3_index(data)
        elif strategy == "rtree":
            self.indexes[index_id] = self._create_rtree_index(data)
        else:
            raise ValueError(f"Unknown indexing strategy: {strategy}")

        logger.info(f"Created spatial index: {index_id}")
        return index_id

    def _create_h3_index(self, data: gpd.GeoDataFrame) -> dict[str, Any]:
        """Create H3 spatial index."""
        h3 = _require_h3()

        if data.crs is None:
            raise ValueError(
                "H3 indexing requires a declared CRS; provide EPSG:4326 or a "
                "projected CRS that can be transformed to WGS84"
            )
        try:
            if data.crs.to_string().upper() == "EPSG:4326":
                indexed_data = data
            else:
                transformer = Transformer.from_crs(
                    data.crs, "EPSG:4326", always_xy=True
                )
                indexed_data = data.copy()
                indexed_data["geometry"] = indexed_data.geometry.map(
                    lambda geometry: (
                        transform_geometry(transformer.transform, geometry)
                        if geometry is not None and not geometry.is_empty
                        else geometry
                    )
                )
                indexed_data = indexed_data.set_crs("EPSG:4326", allow_override=True)
        except Exception as exc:
            raise ValueError(
                f"H3 indexing could not transform data CRS {data.crs!s} to EPSG:4326"
            ) from exc

        h3_indexes = {}

        for idx, row in indexed_data.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty and geom.is_valid:
                # Get centroid
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
                if not (
                    math.isfinite(lat)
                    and math.isfinite(lon)
                    and -90 <= lat <= 90
                    and -180 <= lon <= 180
                ):
                    continue

                # Create H3 index at resolution 9 (city level)
                h3_index = h3.latlng_to_cell(lat, lon, 9)
                h3_indexes[str(idx)] = h3_index

        return {
            "type": "h3",
            "resolution": 9,
            "crs": "EPSG:4326",
            "indexes": h3_indexes,
            "data": data,
        }

    def _create_rtree_index(self, data: gpd.GeoDataFrame) -> dict[str, Any]:
        """Create R-tree spatial index."""
        idx = rtree_index_module.Index()

        for i, row in data.iterrows():
            geom = row.geometry
            if geom and geom.is_valid:
                bounds = geom.bounds  # (min_lon, min_lat, max_lon, max_lat)
                idx.insert(i, bounds)

        return {"type": "rtree", "index": idx, "data": data}

    def query_by_bounds(self, index_id: str, bbox: list[float]) -> gpd.GeoDataFrame:
        """
        Query spatial index by bounding box.

        Args:
            index_id: Index identifier
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]

        Returns:
            Filtered GeoDataFrame
        """
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")

        index_data = self.indexes[index_id]
        index_type = index_data["type"]

        if index_type == "h3":
            return self._query_h3_bounds(index_data, bbox)
        elif index_type == "rtree":
            return self._query_rtree_bounds(index_data, bbox)
        else:
            raise ValueError(f"Unsupported spatial index type: {index_type}")

    def _query_h3_bounds(
        self, index_data: dict[str, Any], bbox: list[float]
    ) -> gpd.GeoDataFrame:
        """Query H3 index by bounds."""
        h3 = _require_h3()

        # Get H3 cells that intersect with bbox
        min_lon, min_lat, max_lon, max_lat = bbox

        # Get H3 cells for bbox polygon (h3 v4 API)
        polygon = Polygon(
            [
                (min_lon, min_lat),
                (min_lon, max_lat),
                (max_lon, max_lat),
                (max_lon, min_lat),
                (min_lon, min_lat),
            ]
        )
        cells = h3.geo_to_cells(polygon, index_data["resolution"])

        # Filter data by H3 cells
        matching_indexes = [
            idx for idx, h3_idx in index_data["indexes"].items() if h3_idx in cells
        ]

        if matching_indexes:
            return index_data["data"].loc[matching_indexes]
        else:
            return gpd.GeoDataFrame()

    def _query_rtree_bounds(
        self, index_data: dict[str, Any], bbox: list[float]
    ) -> gpd.GeoDataFrame:
        """Query R-tree index by bounds.

        Uses the stored R-tree's ``intersection`` to find candidate rows whose
        geometry bounding boxes intersect the query bbox.
        """
        rtree_index = index_data["index"]
        candidate_ids = sorted(rtree_index.intersection(tuple(bbox)))
        if not candidate_ids:
            return gpd.GeoDataFrame()
        return index_data["data"].loc[candidate_ids]

    def latlng_to_cell(self, lat: float, lng: float, resolution: int = 9) -> str:
        """
        Convert latitude/longitude to H3 cell.

        Args:
            lat: Latitude
            lng: Longitude
            resolution: H3 resolution level

        Returns:
            H3 cell index
        """
        return cast(str, _require_h3().latlng_to_cell(lat, lng, resolution))

    def cell_to_latlng(self, cell: str) -> tuple[float, float]:
        """
        Convert H3 cell to latitude/longitude.

        Args:
            cell: H3 cell index

        Returns:
            Tuple of (latitude, longitude)
        """
        return cast(tuple[float, float], _require_h3().cell_to_latlng(cell))


class TemporalIndexer:
    """
    Temporal indexing for efficient time-based queries.

    This class provides temporal indexing capabilities for time-series
    and temporal geospatial data.

    Examples:
        >>> indexer = TemporalIndexer()
        >>>
        >>> # Create temporal index
        >>> index = indexer.create_temporal_index(data, 'timestamp')
        >>>
        >>> # Query by time range
        >>> results = indexer.query_by_time_range(index, start_time, end_time)
    """

    def __init__(self) -> None:
        self.indexes: dict[str, Any] = {}
        logger.info("Initialized TemporalIndexer")

    def create_temporal_index(
        self, data: pd.DataFrame | gpd.GeoDataFrame, time_column: str
    ) -> str:
        """
        Create temporal index for time-based queries.

        Args:
            data: DataFrame or GeoDataFrame with temporal data
            time_column: Name of the datetime column

        Returns:
            Index identifier
        """
        logger.info(f"Creating temporal index for column: {time_column}")

        index_id = f"temporal_{len(self.indexes)}"

        if time_column not in data.columns:
            raise ValueError(f"Time column {time_column} not found in data")

        # Sort by time for efficient range queries
        sorted_data = data.sort_values(time_column).reset_index(drop=True)

        self.indexes[index_id] = {
            "type": "temporal",
            "time_column": time_column,
            "data": sorted_data,
            "min_time": sorted_data[time_column].min(),
            "max_time": sorted_data[time_column].max(),
        }

        logger.info(f"Created temporal index: {index_id}")
        return index_id

    def query_by_time_range(
        self,
        index_id: str,
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Query temporal index by time range.

        Args:
            index_id: Index identifier
            start_time: Start time for query
            end_time: End time for query

        Returns:
            Filtered data
        """
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")

        index_data = self.indexes[index_id]
        time_column = index_data["time_column"]
        data = index_data["data"]

        # Filter by time range
        mask = (data[time_column] >= start_time) & (data[time_column] <= end_time)
        return data[mask]

    def query_by_time_point(
        self, index_id: str, time_point: str | pd.Timestamp
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Query temporal index by time point.

        Args:
            index_id: Index identifier
            time_point: Specific time point

        Returns:
            Data at the specified time point
        """
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")

        index_data = self.indexes[index_id]
        time_column = index_data["time_column"]
        data = index_data["data"]

        # Find exact match or nearest
        mask = data[time_column] == time_point
        return data[mask]
