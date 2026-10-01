"""
Pytest fixtures for GEO-INFER-ECON tests.

Provides economic indicators, market data time series,
spatial economic GeoDataFrames, and standard spatial fixtures.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
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
def economic_indicators() -> dict[str, Any]:
    """Dictionary of regional economic indicators for analysis tests.

    Contains GDP, unemployment, CPI, and trade balance values for a
    hypothetical region with baseline and current period comparisons.
    """
    return {
        "region": "Pacific Northwest",
        "period": "2024-Q4",
        "gdp_billion_usd": 245.3,
        "gdp_growth_pct": 2.8,
        "unemployment_rate": 4.1,
        "cpi_index": 312.5,
        "inflation_rate_pct": 3.2,
        "trade_balance_million_usd": -12.4,
        "median_household_income": 78500,
        "gini_coefficient": 0.42,
    }


@pytest.fixture
def market_data_series() -> pd.Series:
    """365-day daily market index time series for economic analysis.

    Simulates a market index with geometric Brownian motion starting
    at 1000, with daily returns having mean 0.0003 and volatility 0.012.
    """
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    daily_returns = rng.normal(0.0003, 0.012, 365)
    prices = 1000.0 * np.cumprod(1 + daily_returns)
    return pd.Series(prices, index=dates, name="market_index")


@pytest.fixture
def spatial_econ_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with economic attributes for spatial economic analysis.

    Contains 15 census tract centroids in the Seattle metro area with
    median income, employment rate, and business density attributes.
    """
    rng = np.random.default_rng(seed=42)
    n = 15
    lats = 47.4 + rng.uniform(0, 0.4, n)
    lngs = -122.5 + rng.uniform(0, 0.4, n)

    return gpd.GeoDataFrame(
        {
            "tract_id": [f"53033{i:06d}" for i in range(n)],
            "median_income": rng.integers(35000, 150000, n),
            "employment_rate": rng.uniform(0.88, 0.97, n),
            "business_density_per_km2": rng.uniform(5, 120, n),
            "population": rng.integers(2000, 15000, n),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )
