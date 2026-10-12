"""
Pytest fixtures for GEO-INFER-MATH tests.

Provides spatial weight matrices, coordinate pairs, graph adjacency
structures, and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


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
def spatial_weight_matrix() -> np.ndarray:
    """Symmetric 5x5 spatial weight matrix for spatial statistics tests.

    Row-standardized weights based on inverse distance. Diagonal is zero.
    Matrix is symmetric and rows sum to 1.0, suitable for Moran's I
    and spatial autocorrelation computations.
    """
    W = np.array(
        [
            [0.00, 0.35, 0.10, 0.05, 0.50],
            [0.35, 0.00, 0.25, 0.15, 0.25],
            [0.10, 0.25, 0.00, 0.40, 0.25],
            [0.05, 0.15, 0.40, 0.00, 0.40],
            [0.50, 0.25, 0.25, 0.40, 0.00],
        ]
    )
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    return W / row_sums


@pytest.fixture
def coordinate_pairs() -> list[tuple[float, float]]:
    """List of 10 (lat, lng) tuples for distance and projection tests.

    Points distributed across the continental US for testing geodesic
    distance calculations, projections, and spatial indexing.
    """
    return [
        (47.6062, -122.3321),  # Seattle
        (37.7749, -122.4194),  # San Francisco
        (34.0522, -118.2437),  # Los Angeles
        (40.7128, -74.0060),  # New York
        (41.8781, -87.6298),  # Chicago
        (29.7604, -95.3698),  # Houston
        (33.4484, -112.0740),  # Phoenix
        (39.7392, -104.9903),  # Denver
        (25.7617, -80.1918),  # Miami
        (47.2529, -122.4443),  # Tacoma
    ]


@pytest.fixture
def graph_adjacency() -> dict[str, list[str]]:
    """Graph adjacency dict compatible with networkx for graph algorithm tests.

    Represents a small connected graph with 6 nodes and 8 edges,
    suitable for shortest path, centrality, and clustering tests.
    """
    return {
        "A": ["B", "C"],
        "B": ["A", "C", "D"],
        "C": ["A", "B", "D", "E"],
        "D": ["B", "C", "F"],
        "E": ["C", "F"],
        "F": ["D", "E"],
    }


@pytest.fixture
def symmetric_positive_definite_matrix() -> np.ndarray:
    """6x6 symmetric positive definite matrix for linear algebra tests.

    Constructed via A^T A + I to guarantee positive definiteness,
    suitable for Cholesky decomposition and covariance matrix tests.
    """
    rng = np.random.default_rng(seed=42)
    A = rng.normal(0, 1, (6, 6))
    return A.T @ A + np.eye(6)
