"""
Tests for SpatialIndexer and TemporalIndexer in geo_infer_data.utils.indexing.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import pytest
import h3
from pyproj import Transformer
from shapely.geometry import Point

from geo_infer_data.utils.indexing import SpatialIndexer, TemporalIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gdf(n: int = 10) -> gpd.GeoDataFrame:
    """Create a test GeoDataFrame with valid Point geometries."""
    lats = np.random.uniform(37.0, 38.0, n)
    lons = np.random.uniform(-123.0, -122.0, n)
    return gpd.GeoDataFrame(
        {"name": [f"pt_{i}" for i in range(n)]},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


def _make_temporal_df(n: int = 20) -> pd.DataFrame:
    """Create a test DataFrame with a timestamp column."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "value": np.random.rand(n),
        }
    )


# ---------------------------------------------------------------------------
# SpatialIndexer
# ---------------------------------------------------------------------------


class TestSpatialIndexer:
    def test_init(self):
        indexer = SpatialIndexer()
        assert indexer.indexes == {}

    def test_unknown_quadtree_strategy_rejected(self):
        indexer = SpatialIndexer()
        gdf = _make_gdf()
        with pytest.raises(ValueError, match="Unknown indexing strategy"):
            indexer.create_spatial_index(gdf, strategy="quadtree")

    def test_query_by_bounds_rtree_filters(self):
        indexer = SpatialIndexer()
        gdf = gpd.GeoDataFrame(
            {"id": list(range(6))},
            geometry=[Point(-122.5 + 0.1 * i, 37.5 + 0.1 * i) for i in range(6)],
            crs="EPSG:4326",
        )
        index_id = indexer.create_spatial_index(gdf, strategy="rtree")
        result = indexer.query_by_bounds(index_id, bbox=[-122.6, 37.4, -122.2, 37.8])
        assert sorted(result["id"]) == [0, 1, 2, 3]

    def test_create_rtree_index_fallback(self):
        indexer = SpatialIndexer()
        gdf = _make_gdf()
        # rtree may or may not be installed; both outcomes are valid
        index_id = indexer.create_spatial_index(gdf, strategy="rtree")
        assert index_id in indexer.indexes
        assert indexer.indexes[index_id]["type"] in ("rtree", "rtree_mock")

    def test_create_h3_index_uses_native_h3(self):
        indexer = SpatialIndexer()
        gdf = _make_gdf()
        index_id = indexer.create_spatial_index(gdf, strategy="h3")
        assert index_id in indexer.indexes
        assert indexer.indexes[index_id]["type"] == "h3"
        assert all(
            h3.is_valid_cell(cell)
            for cell in indexer.indexes[index_id]["indexes"].values()
        )

    def test_create_h3_index_reprojects_projected_crs(self):
        indexer = SpatialIndexer()
        longitude, latitude = -122.4194, 37.7749
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        projected_x, projected_y = transformer.transform(longitude, latitude)
        geographic = gpd.GeoDataFrame(
            {"name": ["point"]},
            geometry=[Point(longitude, latitude)],
            crs="EPSG:4326",
        )
        projected = gpd.GeoDataFrame(
            {"name": ["point"]},
            geometry=[Point(projected_x, projected_y)],
            crs="EPSG:3857",
        )

        geographic_id = indexer.create_spatial_index(geographic, strategy="h3")
        projected_id = indexer.create_spatial_index(projected, strategy="h3")

        assert indexer.indexes[projected_id]["crs"] == "EPSG:4326"
        assert (
            indexer.indexes[projected_id]["indexes"]
            == indexer.indexes[geographic_id]["indexes"]
        )

    def test_create_h3_index_requires_crs(self):
        indexer = SpatialIndexer()
        unreferenced = _make_gdf(1).set_crs(None, allow_override=True)

        with pytest.raises(ValueError, match="declared CRS"):
            indexer.create_spatial_index(unreferenced, strategy="h3")

    def test_unknown_strategy_raises(self):
        indexer = SpatialIndexer()
        gdf = _make_gdf()
        with pytest.raises(ValueError, match="Unknown indexing strategy"):
            indexer.create_spatial_index(gdf, strategy="unknown")

    def test_query_by_bounds_rtree_miss_returns_empty(self):
        indexer = SpatialIndexer()
        gdf = _make_gdf(20)
        index_id = indexer.create_spatial_index(gdf, strategy="rtree")
        result = indexer.query_by_bounds(index_id, bbox=[10.0, 10.0, 11.0, 11.0])
        assert len(result) == 0

    def test_query_by_bounds_missing_index_raises(self):
        indexer = SpatialIndexer()
        with pytest.raises(ValueError, match="not found"):
            indexer.query_by_bounds("nonexistent_index", bbox=[-180, -90, 180, 90])

    def test_latlng_to_cell(self):
        indexer = SpatialIndexer()
        cell = indexer.latlng_to_cell(37.7749, -122.4194, resolution=9)
        assert isinstance(cell, str)
        assert len(cell) > 0

    def test_cell_to_latlng(self):
        indexer = SpatialIndexer()
        # First get a valid cell, then convert back
        cell = indexer.latlng_to_cell(37.7749, -122.4194, resolution=9)
        lat, lng = indexer.cell_to_latlng(cell)
        assert isinstance(lat, float)
        assert isinstance(lng, float)
        assert abs(lat - 37.7749) < 0.01
        assert abs(lng - (-122.4194)) < 0.01


# ---------------------------------------------------------------------------
# TemporalIndexer
# ---------------------------------------------------------------------------


class TestTemporalIndexer:
    def test_init(self):
        indexer = TemporalIndexer()
        assert indexer.indexes == {}

    def test_create_temporal_index(self):
        indexer = TemporalIndexer()
        df = _make_temporal_df()
        index_id = indexer.create_temporal_index(df, "timestamp")
        assert index_id in indexer.indexes
        idx_data = indexer.indexes[index_id]
        assert idx_data["type"] == "temporal"
        assert idx_data["time_column"] == "timestamp"

    def test_create_index_missing_column_raises(self):
        indexer = TemporalIndexer()
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            indexer.create_temporal_index(df, "timestamp")

    def test_query_by_time_range(self):
        indexer = TemporalIndexer()
        df = _make_temporal_df(24)
        index_id = indexer.create_temporal_index(df, "timestamp")
        result = indexer.query_by_time_range(
            index_id,
            start_time=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
            end_time=pd.Timestamp("2024-01-01 15:00", tz="UTC"),
        )
        assert len(result) == 11  # hours 5 through 15

    def test_query_by_time_range_empty(self):
        indexer = TemporalIndexer()
        df = _make_temporal_df(10)
        index_id = indexer.create_temporal_index(df, "timestamp")
        result = indexer.query_by_time_range(
            index_id,
            start_time=pd.Timestamp("2025-01-01", tz="UTC"),
            end_time=pd.Timestamp("2025-12-31", tz="UTC"),
        )
        assert len(result) == 0

    def test_query_by_time_range_missing_index_raises(self):
        indexer = TemporalIndexer()
        with pytest.raises(ValueError, match="not found"):
            indexer.query_by_time_range(
                "missing",
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-12-31", tz="UTC"),
            )

    def test_query_by_time_point(self):
        indexer = TemporalIndexer()
        df = _make_temporal_df(24)
        index_id = indexer.create_temporal_index(df, "timestamp")
        result = indexer.query_by_time_point(
            index_id, pd.Timestamp("2024-01-01 10:00", tz="UTC")
        )
        assert len(result) == 1

    def test_query_by_time_point_no_match(self):
        indexer = TemporalIndexer()
        df = _make_temporal_df(24)
        index_id = indexer.create_temporal_index(df, "timestamp")
        result = indexer.query_by_time_point(
            index_id, pd.Timestamp("2025-06-15", tz="UTC")
        )
        assert len(result) == 0

    def test_index_sorts_data(self):
        """Temporal index should sort data by time column."""
        indexer = TemporalIndexer()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-03-01", "2024-01-01", "2024-02-01"], utc=True
                ),
                "value": [3, 1, 2],
            }
        )
        index_id = indexer.create_temporal_index(df, "timestamp")
        sorted_data = indexer.indexes[index_id]["data"]
        assert sorted_data["value"].tolist() == [1, 2, 3]
