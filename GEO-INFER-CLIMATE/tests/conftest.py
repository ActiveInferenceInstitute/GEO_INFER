"""
Pytest fixtures for GEO-INFER-CLIMATE tests.

Provides temperature time series, climate grids on H3 cells,
reference period data, and standard spatial fixtures.
"""

import pytest
import numpy as np
import pandas as pd
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
def temperature_time_series() -> pd.Series:
    """365-day daily temperature time series with DatetimeIndex (UTC).

    Simulates a sinusoidal annual temperature cycle for a temperate
    location (mean 12C, amplitude 15C) with Gaussian noise, suitable
    for climate anomaly detection and trend analysis tests.
    """
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D", tz="UTC")
    day_of_year = np.arange(365)
    # Sinusoidal annual cycle: peak in summer (day ~182), trough in winter
    temperature = 12.0 + 15.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature += rng.normal(0, 2.0, 365)
    return pd.Series(temperature, index=dates, name="temperature_c")


@pytest.fixture
def climate_grid() -> gpd.GeoDataFrame:
    """GeoDataFrame with temperature values per H3 cell for grid tests.

    Contains 10 point-geometry cells in the Pacific Northwest with
    mean annual temperature and precipitation values, representing
    a simplified climate grid.
    """
    rng = np.random.default_rng(seed=42)
    n = 10
    lats = 45.0 + np.linspace(0, 3, n)
    lngs = -123.0 + np.linspace(0, 2, n)

    cells = [h3.latlng_to_cell(lat, lng, 5) for lat, lng in zip(lats, lngs)]

    return gpd.GeoDataFrame(
        {
            "h3_cell": cells,
            "mean_temp_c": 8.0 + rng.uniform(-3, 5, n),
            "annual_precip_mm": 800 + rng.uniform(-200, 400, n),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )


@pytest.fixture
def reference_period_data() -> dict[str, Any]:
    """Reference period climate statistics for anomaly computation.

    Provides 1961-1990 baseline period means and standard deviations
    for temperature and precipitation, used to compute climate anomalies.
    """
    return {
        "period": {"start": 1961, "end": 1990},
        "temperature": {
            "mean": 10.5,
            "std": 1.2,
            "monthly_means": [
                2.1,
                3.0,
                5.8,
                9.2,
                13.1,
                16.5,
                18.9,
                18.4,
                15.0,
                10.3,
                5.7,
                2.8,
            ],
        },
        "precipitation": {
            "mean": 950.0,
            "std": 120.0,
            "monthly_means": [
                120,
                100,
                90,
                70,
                55,
                40,
                30,
                35,
                50,
                80,
                110,
                130,
            ],
        },
    }
