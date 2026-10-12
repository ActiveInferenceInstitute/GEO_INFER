"""
Tests for GeospatialValidator in geo_infer_data.utils.validation.
"""

import asyncio

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from geo_infer_data.models.schemas import QualityStatus
from geo_infer_data.utils.validation import GeospatialValidator


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_gdf(n: int = 5) -> gpd.GeoDataFrame:
    lats = np.linspace(37.0, 38.0, n)
    lons = np.linspace(-122.5, -122.0, n)
    return gpd.GeoDataFrame(
        {"name": [f"pt_{i}" for i in range(n)]},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# validate_data (comprehensive)
# ---------------------------------------------------------------------------


class TestValidateData:
    def test_valid_geodataframe_passes(self):
        validator = GeospatialValidator()
        gdf = _valid_gdf()
        result = _run(validator.validate_data(gdf))
        assert result.score > 0.5
        assert result.status in (QualityStatus.PASS, QualityStatus.WARNING)

    def test_empty_dataframe(self):
        validator = GeospatialValidator()
        df = pd.DataFrame()
        result = _run(validator.validate_data(df))
        # An empty dataframe might fail some checks
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# validate_geometries
# ---------------------------------------------------------------------------


class TestValidateGeometries:
    def test_valid_geometries_pass(self):
        validator = GeospatialValidator()
        gdf = _valid_gdf(10)
        result = validator.validate_geometries(gdf)
        assert result.score >= 0.8
        assert result.status == QualityStatus.PASS

    def test_null_geometry_penalised(self):
        validator = GeospatialValidator()
        gdf = _valid_gdf(4)
        gdf.loc[0, "geometry"] = None
        result = validator.validate_geometries(gdf)
        assert result.score < 1.0
        assert any(i["type"] == "null_geometry" for i in result.issues)

    def test_no_geometry_column_fails(self):
        validator = GeospatialValidator()
        gdf = gpd.GeoDataFrame({"a": [1, 2]})
        gdf = gdf.drop("geometry", axis=1, errors="ignore")
        result = validator.validate_geometries(gdf)
        assert result.score == 0.0
        assert result.status == QualityStatus.FAIL


# ---------------------------------------------------------------------------
# validate_coordinates
# ---------------------------------------------------------------------------


class TestValidateCoordinates:
    def test_valid_gdf_coordinates_pass(self):
        validator = GeospatialValidator()
        gdf = _valid_gdf()
        result = validator.validate_coordinates(gdf)
        assert result.status == QualityStatus.PASS
        assert result.score >= 0.8

    def test_invalid_latitude_in_dataframe(self):
        validator = GeospatialValidator()
        df = pd.DataFrame({"latitude": [37.0, 100.0], "longitude": [-122.0, -122.5]})
        result = validator.validate_coordinates(df)
        assert result.score < 1.0
        assert any(i["type"] == "invalid_latitude" for i in result.issues)

    def test_invalid_longitude_in_dataframe(self):
        validator = GeospatialValidator()
        df = pd.DataFrame({"latitude": [37.0, 38.0], "longitude": [-122.0, 200.0]})
        result = validator.validate_coordinates(df)
        assert result.score < 1.0
        assert any(i["type"] == "invalid_longitude" for i in result.issues)

    def test_valid_lat_lon_passes(self):
        validator = GeospatialValidator()
        df = pd.DataFrame({"latitude": [37.0, 38.0], "longitude": [-122.0, -121.5]})
        result = validator.validate_coordinates(df)
        assert result.status == QualityStatus.PASS


# ---------------------------------------------------------------------------
# validate_temporal_data
# ---------------------------------------------------------------------------


class TestValidateTemporalData:
    def test_chronological_data_passes(self):
        validator = GeospatialValidator()
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2020-01-01", periods=10, freq="D")}
        )
        result = validator.validate_temporal_data(df)
        assert result.status == QualityStatus.PASS

    def test_no_datetime_columns_passes(self):
        validator = GeospatialValidator()
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = validator.validate_temporal_data(df)
        assert result.status == QualityStatus.PASS
        assert result.score == 1.0

    def test_non_chronological_order_penalised(self):
        validator = GeospatialValidator()
        df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"])}
        )
        result = validator.validate_temporal_data(df)
        assert any(i["type"] == "non_chronological" for i in result.issues)

    def test_timezone_aware_data_does_not_raise(self):
        validator = GeospatialValidator()
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC")}
        )
        result = validator.validate_temporal_data(df)
        assert result.status == QualityStatus.PASS
