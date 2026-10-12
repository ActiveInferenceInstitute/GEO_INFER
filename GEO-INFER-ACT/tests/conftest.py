"""
Pytest fixtures for GEO-INFER-ACT tests.

Provides Active Inference agents, generative model configurations,
observation sequences, and standard spatial fixtures.
"""

import pytest
import numpy as np
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
def sample_geodataframe() -> Any:
    """Standard GeoDataFrame with EPSG:4326 for spatial tests."""
    import geopandas as gpd
    from shapely.geometry import Point

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
def free_energy_agent() -> dict[str, Any]:
    """Active Inference agent configuration with 3 hidden states.

    Returns a dict containing the generative model matrices (A, B, C, D)
    for a simple 3-state agent. A is the observation model, B is the
    transition model, C is the preference vector, and D is the initial
    state prior.
    """
    n_states = 3
    n_observations = 3
    n_actions = 2

    # Observation model: P(observation | hidden state)
    A = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    # Transition model: P(state_t+1 | state_t, action)
    B = np.zeros((n_actions, n_states, n_states))
    # Action 0: stay
    B[0] = np.eye(n_states)
    # Action 1: cycle forward
    B[1] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # Preference vector (log preferences over observations)
    C = np.array([1.0, 0.0, -1.0])

    # Initial state prior (uniform)
    D = np.ones(n_states) / n_states

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "n_states": n_states,
        "n_observations": n_observations,
        "n_actions": n_actions,
    }


@pytest.fixture
def generative_model_config() -> dict[str, Any]:
    """Configuration dict for an Active Inference generative model.

    Specifies policy length, inference depth, and learning rates
    for a standard Active Inference agent.
    """
    return {
        "policy_length": 3,
        "inference_depth": 5,
        "learning_rate_a": 0.1,
        "learning_rate_b": 0.01,
        "gamma": 16.0,
        "use_states_info_gain": True,
        "use_param_info_gain": False,
        "action_selection": "softmax",
    }


@pytest.fixture
def observation_sequence() -> np.ndarray:
    """Sequence of 20 observations for an Active Inference agent.

    Each observation is an integer index into a 3-observation space,
    representing a plausible time series of sensory inputs.
    """
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 3, size=20)
