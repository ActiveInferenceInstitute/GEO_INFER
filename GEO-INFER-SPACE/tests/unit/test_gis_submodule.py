"""Unit tests for the GISManager facade in GEO-INFER-SPACE."""

import pytest

import geopandas as gpd
from shapely.geometry import Point

from geo_infer_space.gis.gis_manager import GISManager


@pytest.fixture
def gis_manager():
    return GISManager()


def test_gis_manager_initialization(gis_manager):
    """Test that the GISManager initializes its core components successfully."""
    assert gis_manager.methods is not None
    assert gis_manager.processor is not None
    assert gis_manager.utils is not None


def test_coordinate_transformation(gis_manager):
    """Test a basic generic spatial method routing through GISManager."""
    # ``always_xy=True`` means WGS84 coordinates use longitude, latitude.
    coords = (-122.4194, 37.7749)
    result = gis_manager.transform_coordinates(
        coords, from_crs="EPSG:4326", to_crs="EPSG:3857"
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_buffer_analysis(gis_manager):
    """Test standard shape buffer analysis routing through GISManager."""
    df = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]}, crs="EPSG:4326")
    buffered = gis_manager.buffer_analysis(df, buffer_distance=1.0)
    assert len(buffered) == 2
    assert buffered.iloc[0].geometry.area > 0


def test_calculate_distance(gis_manager):
    """Test distance calculation via the generic spatial utilities component."""
    # Distance from SF (37.7749, -122.4194) to LA (34.0522, -118.2437)
    dist = gis_manager.calculate_distance((37.7749, -122.4194), (34.0522, -118.2437))
    assert dist > 0
    # True distance is roughly 559km
    assert 500 < dist < 600
