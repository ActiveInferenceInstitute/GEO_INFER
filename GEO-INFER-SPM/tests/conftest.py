"""
Pytest fixtures for GEO-INFER-SPM tests.

Provides spatial model parameters, latent variables, SPM configurations,
synthetic SPM data, and standard spatial fixtures.
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
def spatial_model_params() -> dict[str, Any]:
    """Spatial parametric model parameters for SPM fitting tests.

    Specifies a spatial lag model with autoregressive coefficient,
    covariate coefficients, and spatial weight matrix type.
    """
    return {
        "model_type": "spatial_lag",
        "rho": 0.45,
        "beta": [2.5, -0.8, 1.2],
        "sigma2": 1.5,
        "weight_matrix_type": "queen",
        "n_observations": 50,
        "n_covariates": 3,
    }


@pytest.fixture
def latent_variables() -> np.ndarray:
    """50x4 array of latent variables for spatial process modeling.

    Columns represent four latent spatial factors extracted from
    observed data, with values standardized to zero mean and unit
    variance. Suitable for factor analysis and dimensionality
    reduction tests.
    """
    rng = np.random.default_rng(seed=42)
    raw = rng.normal(0, 1, (50, 4))
    # Standardize columns
    return (raw - raw.mean(axis=0)) / raw.std(axis=0)


@pytest.fixture
def spm_config() -> dict[str, Any]:
    """Configuration dict for spatial parametric modeling.

    Specifies estimation method, diagnostic tests, and
    visualization options.
    """
    return {
        "estimation_method": "maximum_likelihood",
        "diagnostics": [
            "moran_i",
            "lagrange_multiplier",
            "likelihood_ratio",
        ],
        "spatial_weights": {
            "type": "knn",
            "k": 5,
            "row_standardize": True,
        },
        "visualization": {
            "residual_map": True,
            "predicted_vs_actual": True,
            "coefficient_plot": True,
        },
    }


@pytest.fixture
def synthetic_spm_data() -> gpd.GeoDataFrame:
    """GeoDataFrame with synthetic data for spatial parametric model fitting.

    Contains 50 observation points with a spatially correlated dependent
    variable and 3 covariates, suitable for spatial regression tests.
    """
    rng = np.random.default_rng(seed=42)
    n = 50
    lats = 47.0 + rng.uniform(0, 1, n)
    lngs = -122.5 + rng.uniform(0, 1, n)

    x1 = rng.normal(5, 2, n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.normal(0, 1, n)

    # True model: y = 2.5 + 0.8*x1 - 0.5*x2 + 1.2*x3 + noise
    y = 2.5 + 0.8 * x1 - 0.5 * x2 + 1.2 * x3 + rng.normal(0, 1.5, n)

    return gpd.GeoDataFrame(
        {
            "obs_id": [f"OBS_{i:03d}" for i in range(n)],
            "y": y,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )
