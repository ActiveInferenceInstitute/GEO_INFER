"""
Pytest fixtures for GEO-INFER-ENERGY tests.

Provides solar irradiance grids, wind speed grids on H3 cells,
energy system configurations, and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
import h3
from shapely.geometry import Point
from pathlib import Path
from typing import Any


@pytest.fixture(scope="session")
def sample_coordinates() -> list[tuple[float, float]]:
    """Standard (lat, lng) coordinate pairs for spatial tests."""
    return [
        (47.6062, -122.3321),
        (37.7749, -122.4194),
        (40.7128, -74.0060),
        (51.5074, -0.1278),
        (35.6762, 139.6503),
    ]


@pytest.fixture(scope="function")
def sample_geodataframe() -> gpd.GeoDataFrame:
    """Standard GeoDataFrame with EPSG:4326 for spatial tests."""
    return gpd.GeoDataFrame(
        {"id": range(5), "value": np.random.uniform(0, 100, 5)},
        geometry=[Point(-122.33 + i * 0.01, 47.61 + i * 0.01) for i in range(5)],
        crs="EPSG:4326",
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def solar_irradiance_grid() -> gpd.GeoDataFrame:
    """GeoDataFrame with H3 cells and solar irradiance values (kWh/m2/day).

    Contains 12 cells representing a transect from cloudy Pacific Northwest
    to sunny desert Southwest, with realistic Global Horizontal Irradiance
    values varying by latitude.
    """
    rng = np.random.default_rng(seed=42)
    n = 12
    # Latitude gradient from 47N (Seattle) to 33N (Phoenix)
    lats = np.linspace(47.0, 33.0, n)
    lngs = np.linspace(-122.0, -112.0, n)
    # GHI increases with decreasing latitude
    ghi = 3.0 + 4.0 * (47.0 - lats) / 14.0 + rng.uniform(-0.3, 0.3, n)

    cells = [h3.latlng_to_cell(lat, lng, 5) for lat, lng in zip(lats, lngs)]

    return gpd.GeoDataFrame(
        {
            "h3_cell": cells,
            "ghi_kwh_m2_day": ghi.round(2),
            "dni_kwh_m2_day": (ghi * 1.3 + rng.uniform(-0.2, 0.2, n)).round(2),
            "diffuse_fraction": rng.uniform(0.2, 0.6, n).round(3),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )


@pytest.fixture
def wind_speed_grid() -> gpd.GeoDataFrame:
    """GeoDataFrame with H3 cells and wind speed values (m/s).

    Contains 10 cells with mean wind speed at 80m hub height,
    Weibull shape and scale parameters for wind resource assessment.
    """
    rng = np.random.default_rng(seed=42)
    n = 10
    lats = 42.0 + rng.uniform(0, 4, n)
    lngs = -100.0 + rng.uniform(0, 5, n)

    cells = [h3.latlng_to_cell(lat, lng, 5) for lat, lng in zip(lats, lngs)]

    return gpd.GeoDataFrame(
        {
            "h3_cell": cells,
            "mean_wind_speed_ms": rng.uniform(5.0, 12.0, n).round(2),
            "weibull_k": rng.uniform(1.8, 2.5, n).round(3),
            "weibull_a": rng.uniform(6.0, 13.0, n).round(3),
            "hub_height_m": np.full(n, 80.0),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )


@pytest.fixture
def energy_config() -> dict[str, Any]:
    """Configuration dict for renewable energy analysis.

    Specifies panel/turbine parameters, grid connection constraints,
    and economic assumptions for energy project feasibility.
    """
    return {
        "solar": {
            "panel_efficiency": 0.20,
            "degradation_rate_annual": 0.005,
            "tilt_angle": 30,
            "azimuth": 180,
        },
        "wind": {
            "hub_height_m": 80,
            "rotor_diameter_m": 90,
            "cut_in_speed_ms": 3.5,
            "rated_speed_ms": 12.0,
            "cut_out_speed_ms": 25.0,
        },
        "economics": {
            "discount_rate": 0.06,
            "project_lifetime_years": 25,
            "electricity_price_kwh": 0.08,
        },
    }
