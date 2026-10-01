"""
GEO-INFER Standard conftest.py Template

Copy this file to GEO-INFER-MODULE/tests/conftest.py and customize
the module-specific fixtures for your module's domain.

Usage:
    All fixtures are available to tests via pytest's fixture injection.
    Module-specific fixtures should be added below the standard fixtures.

Fixture Scoping:
    - session: Expensive, read-only objects (coordinate lists, raster metadata).
      Created once per test session.
    - function: Mutable objects (GeoDataFrames, dicts). Fresh copy per test.
      This is the default scope.

Example:
    # In tests/unit/test_example.py
    def test_spatial_query(sample_geodataframe, sample_h3_cells):
        # Both fixtures are injected automatically by pytest
        assert len(sample_geodataframe) == 10
        assert len(sample_h3_cells) == 3
"""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point


# =============================================================================
# Standard Spatial Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_coordinates() -> list[tuple[float, float]]:
    """
    List of (lat, lng) tuples covering diverse geographic locations.

    Returns lat/lng pairs covering: Pacific Northwest, Gulf Coast,
    Great Plains, East Coast, Hawaii, and Alaska.
    Suitable for testing H3 indexing, distance calculations, and spatial queries.

    Returns:
        List of 10 (latitude, longitude) tuples in WGS84 decimal degrees.
    """
    return [
        (47.6062, -122.3321),  # Seattle, WA
        (29.7604, -95.3698),  # Houston, TX
        (39.7392, -104.9903),  # Denver, CO
        (40.7128, -74.0060),  # New York, NY
        (21.3069, -157.8583),  # Honolulu, HI
        (61.2181, -149.9003),  # Anchorage, AK
        (37.7749, -122.4194),  # San Francisco, CA
        (33.4484, -112.0740),  # Phoenix, AZ
        (25.7617, -80.1918),  # Miami, FL
        (44.9778, -93.2650),  # Minneapolis, MN
    ]


@pytest.fixture(scope="session")
def sample_h3_cells() -> list[str]:
    """
    Resolution 8 H3 cells for testing spatial operations.

    Covers cells in diverse geographic regions to test:
    - Cell neighbor computation (grid_disk)
    - Resolution changes (cell_to_parent, cell_to_children)
    - Distance calculations (grid_distance)
    - Boundary polygon generation (cell_to_boundary)

    Uses H3 v4 API (latlng_to_cell, not geo_to_h3).

    Returns:
        List of 3 H3 cell index strings at resolution 8.

    Skips:
        If the h3 package is not installed.
    """
    try:
        import h3
    except ImportError:
        pytest.skip("h3 not installed")

    coords = [
        (47.6062, -122.3321),  # Seattle
        (37.7749, -122.4194),  # San Francisco
        (40.7128, -74.0060),  # New York
    ]
    return [h3.latlng_to_cell(lat, lng, resolution=8) for lat, lng in coords]


@pytest.fixture(scope="function")
def sample_geodataframe() -> gpd.GeoDataFrame:
    """
    GeoDataFrame with geometry column, CRS=EPSG:4326, and standard fields.

    Includes columns: id, name, value, category, geometry (Point), timestamp.
    Contains 10 rows with Point geometries in the Seattle area.
    Suitable for testing spatial joins, overlays, and attribute operations.

    Returns:
        GeoDataFrame with 10 rows, EPSG:4326 CRS, and Point geometries.

    Notes:
        Function-scoped because tests may modify the DataFrame. Each test
        gets a fresh copy.
    """
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    data = {
        "id": list(range(10)),
        "name": [f"location_{i}" for i in range(10)],
        "value": rng.uniform(0, 100, 10).tolist(),
        "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "B"],
        "timestamp": timestamps,
        "geometry": [
            Point(-122.3321 + i * 0.01, 47.6062 + i * 0.01) for i in range(10)
        ],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf


@pytest.fixture(scope="function")
def sample_time_series() -> pd.Series:
    """
    Pandas Series with DatetimeIndex for temporal tests.

    Daily data for 365 days with sinusoidal pattern + Gaussian noise.
    UTC timezone. Values represent a temperature-like signal (baseline 20,
    amplitude 10, noise stddev 1).

    Suitable for trend analysis, seasonality detection, resampling,
    and temporal aggregation tests.

    Returns:
        pd.Series with 365 entries, DatetimeIndex (UTC), name="temperature_celsius".
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=365, freq="D", tz="UTC")
    values = (
        np.sin(np.linspace(0, 4 * np.pi, 365)) * 10  # seasonal signal
        + rng.normal(0, 1, 365)  # noise
        + 20  # baseline
    )
    return pd.Series(values, index=dates, name="temperature_celsius")


@pytest.fixture(scope="session")
def sample_raster() -> dict[str, Any]:
    """
    Numpy array with CRS metadata for raster tests.

    Returns a dict representing a 100x100 pixel elevation raster over the
    Seattle area. Values are uniformly distributed between 0 and 500 meters.

    Keys:
        data: np.ndarray of shape (100, 100), dtype float32
        crs: str, "EPSG:4326"
        transform: dict with west, north, east, south, width, height
        nodata: float, -9999.0
        units: str, "meters"

    Returns:
        Dict with raster data and metadata.
    """
    rng = np.random.default_rng(42)
    return {
        "data": rng.uniform(0, 500, (100, 100)).astype(np.float32),
        "crs": "EPSG:4326",
        "transform": {
            "west": -122.5,
            "north": 47.8,
            "east": -122.0,
            "south": 47.4,
            "width": 100,
            "height": 100,
        },
        "nodata": -9999.0,
        "units": "meters",
    }


@pytest.fixture(scope="function")
def active_inference_state() -> dict[str, Any]:
    """
    Generative model state dict for Active Inference tests.

    Minimal valid state for a 3-state hidden variable model with
    4 observation modalities and 2 actions. All matrices are properly
    normalized as probability distributions.

    Suitable for testing belief updating, free energy computation,
    policy selection, and expected free energy.

    Keys:
        beliefs: np.ndarray (n_states,) - current posterior beliefs
        observations: np.ndarray (n_obs,) - current observation vector
        A: np.ndarray (n_obs, n_states) - likelihood mapping P(o|s)
        B: np.ndarray (n_actions, n_states, n_states) - transitions P(s'|s,a)
        C: np.ndarray (n_obs,) - preferred observations (log-preferences)
        D: np.ndarray (n_states,) - prior beliefs over initial states
        n_states: int
        n_obs: int
        n_actions: int
        time_step: int

    Returns:
        Dict containing the complete generative model state.
    """
    rng = np.random.default_rng(42)
    n_states = 3
    n_obs = 4
    n_actions = 2

    return {
        "beliefs": np.ones(n_states) / n_states,
        "observations": np.zeros(n_obs),
        "A": rng.dirichlet(np.ones(n_states), n_obs),
        "B": np.stack(
            [rng.dirichlet(np.ones(n_states), n_states) for _ in range(n_actions)]
        ),
        "C": np.zeros(n_obs),
        "D": np.ones(n_states) / n_states,
        "n_states": n_states,
        "n_obs": n_obs,
        "n_actions": n_actions,
        "time_step": 0,
    }


@pytest.fixture
def tmp_spatial_dir(tmp_path: Path) -> Path:
    """
    Temporary directory pre-populated with sample spatial files.

    Creates three files in a temporary directory:
    - sample.geojson: FeatureCollection with 5 Point features
    - sample.csv: CSV with lat, lng, value columns (5 rows)
    - sample_polygon.geojson: FeatureCollection with 1 Polygon feature

    All geometries are in the Seattle area (EPSG:4326).

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.

    Returns:
        Path to the temporary directory containing the sample files.
    """
    # Write sample GeoJSON with point features
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.33 + i * 0.01, 47.61 + i * 0.01],
                },
                "properties": {"id": i, "value": float(i * 10)},
            }
            for i in range(5)
        ],
    }
    (tmp_path / "sample.geojson").write_text(json.dumps(geojson))

    # Write sample CSV with coordinate data
    csv_content = "lat,lng,value\n"
    for i in range(5):
        csv_content += f"{47.61 + i * 0.01},{-122.33 + i * 0.01},{i * 10}\n"
    (tmp_path / "sample.csv").write_text(csv_content)

    # Write polygon GeoJSON (bounding box around Seattle)
    polygon_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.5, 47.4],
                            [-121.9, 47.4],
                            [-121.9, 47.8],
                            [-122.5, 47.8],
                            [-122.5, 47.4],
                        ]
                    ],
                },
                "properties": {"name": "test_region", "area_km2": 1200.0},
            }
        ],
    }
    (tmp_path / "sample_polygon.geojson").write_text(json.dumps(polygon_geojson))

    return tmp_path


# =============================================================================
# Module-Specific Fixtures (customize for each module)
# =============================================================================
# Add module-specific fixtures below this line. Remove or replace the examples
# with fixtures appropriate for your module's domain.
#
# Examples by module:
#
#   MARINE:
#     - coastal_polygon: GeoDataFrame with coastline polygon
#     - bathymetry_grid: 2D numpy array of ocean depth values
#     - tidal_time_series: pd.Series with tidal height measurements
#
#   FOREST:
#     - forest_stand_polygon: GeoDataFrame with forest stand boundaries
#     - biomass_data: DataFrame with tree species, DBH, height
#     - canopy_height_raster: dict with CHM raster data and metadata
#
#   TRANSPORT:
#     - road_network_graph: networkx.DiGraph with edge weights
#     - origin_destination_matrix: np.ndarray of travel demands
#     - transit_schedule: DataFrame with stop times and routes
#
#   RISK:
#     - hazard_map: GeoDataFrame with hazard intensity zones
#     - vulnerability_curves: dict mapping building types to damage functions
#     - exposure_layer: GeoDataFrame with asset locations and values
#
#   CLIMATE:
#     - temperature_grid: xarray.DataArray with temperature fields
#     - precipitation_series: pd.Series with daily precipitation
#     - climate_scenario: dict with RCP/SSP scenario parameters
