"""
Pytest fixtures for GEO-INFER-BAYES tests.

Provides GP kernel configurations, MCMC posterior samples,
prior parameter dicts, synthetic spatial data, and standard spatial fixtures.
"""

import pytest
import numpy as np
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
        {"id": range(5), "value": np.random.default_rng(0).uniform(0, 100, 5)},
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
def gp_kernel_config() -> dict[str, Any]:
    """Gaussian Process kernel configuration for spatial modeling.

    Specifies a squared exponential (RBF) kernel with a length scale
    suitable for geographic coordinates (in degrees) and signal variance.
    """
    return {
        "kernel_type": "rbf",
        "length_scale": 0.5,
        "signal_variance": 1.0,
        "noise_variance": 0.1,
        "n_restarts": 5,
        "optimizer": "L-BFGS-B",
        "normalize_y": True,
    }


@pytest.fixture
def mcmc_samples() -> np.ndarray:
    """1000x3 array of MCMC posterior samples for Bayesian model tests.

    Columns represent three parameters: intercept (mean ~2.0),
    slope (mean ~0.5), and noise variance (mean ~0.1). Samples
    drawn from a multivariate normal to simulate converged chains.
    """
    rng = np.random.default_rng(seed=42)
    mean = np.array([2.0, 0.5, 0.1])
    cov = np.array(
        [
            [0.04, 0.002, 0.0],
            [0.002, 0.01, 0.0],
            [0.0, 0.0, 0.001],
        ]
    )
    samples = rng.multivariate_normal(mean, cov, size=1000)
    # Ensure noise variance is positive
    samples[:, 2] = np.abs(samples[:, 2])
    return samples


@pytest.fixture
def prior_params() -> dict[str, Any]:
    """Prior distribution parameters for Bayesian models.

    Specifies normal priors for regression coefficients and
    an inverse-gamma prior for the noise variance.
    """
    return {
        "intercept": {"distribution": "normal", "mean": 0.0, "std": 10.0},
        "slope": {"distribution": "normal", "mean": 0.0, "std": 5.0},
        "noise_variance": {"distribution": "inv_gamma", "alpha": 2.0, "beta": 1.0},
    }


@pytest.fixture
def synthetic_spatial_data() -> gpd.GeoDataFrame:
    """GeoDataFrame with synthetic spatial observations and values.

    Contains 30 point locations in the Pacific Northwest with a
    spatially correlated response variable (temperature-like values)
    suitable for Gaussian Process regression tests.
    """
    rng = np.random.default_rng(seed=42)
    n = 30
    lats = 46.0 + rng.uniform(0, 3, n)
    lngs = -124.0 + rng.uniform(0, 3, n)

    # Spatially correlated response: warmer in south, cooler in north
    values = 15.0 - 2.0 * (lats - 46.0) + rng.normal(0, 0.5, n)

    return gpd.GeoDataFrame(
        {
            "station_id": [f"STN_{i:03d}" for i in range(n)],
            "temperature": values,
            "elevation_m": rng.uniform(0, 2000, n),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )
