"""
Pytest fixtures for GEO-INFER-ANT tests.

Provides distance matrices, pheromone grids on H3 cells,
ant colony configurations, and standard spatial fixtures.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest


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


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def distance_matrix() -> np.ndarray:
    """Symmetric 5x5 distance matrix for ACO route optimization tests.

    Represents distances between 5 locations with realistic asymmetric
    travel costs. Diagonal is zero. All values are positive.
    """
    return np.array(
        [
            [0.0, 2.0, 9.0, 10.0, 5.0],
            [2.0, 0.0, 7.0, 8.0, 3.0],
            [9.0, 7.0, 0.0, 4.0, 6.0],
            [10.0, 8.0, 4.0, 0.0, 1.0],
            [5.0, 3.0, 6.0, 1.0, 0.0],
        ]
    )


@pytest.fixture
def pheromone_grid() -> list[dict[str, Any]]:
    """List of H3 cells with initial pheromone concentrations.

    Uses H3 v4 API (latlng_to_cell). Each entry has a cell ID string
    and a float concentration value representing pheromone intensity.
    """
    import h3

    cells = [h3.latlng_to_cell(47.6 + i * 0.01, -122.3 + i * 0.01, 9) for i in range(5)]

    return [
        {"cell": cells[0], "concentration": 10.0},
        {"cell": cells[1], "concentration": 7.5},
        {"cell": cells[2], "concentration": 3.2},
        {"cell": cells[3], "concentration": 1.0},
        {"cell": cells[4], "concentration": 0.5},
    ]


@pytest.fixture
def ant_colony_config() -> dict[str, Any]:
    """Configuration dict for an AntColony solver.

    Specifies standard ACO hyperparameters: number of ants, iterations,
    pheromone influence (alpha), distance heuristic influence (beta),
    and evaporation rate.
    """
    return {
        "n_ants": 20,
        "n_iterations": 100,
        "alpha": 1.0,
        "beta": 2.0,
        "evaporation_rate": 0.1,
        "q": 100.0,
        "initial_pheromone": 1.0,
        "elite_weight": 2.0,
    }
