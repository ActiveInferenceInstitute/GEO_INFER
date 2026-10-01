"""Regression tests for categorical inference numerical and transition contracts."""

from pathlib import Path
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from geo_infer_act.core.belief_updating import BayesianBeliefUpdate
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.models.base import CategoricalModel
from geo_infer_act.models.multi_agent import MultiAgentModel
from geo_infer_act.utils.math import categorical_posterior, normalize_distribution

REPO_ROOT = Path(__file__).resolve().parents[3]


def _issue_observation_model() -> np.ndarray:
    """Return the observation model used by the reported underflow case."""
    return np.array(
        [
            [0.55, 0.15, 0.05],
            [0.30, 0.35, 0.15],
            [0.10, 0.35, 0.30],
            [0.05, 0.15, 0.50],
        ],
        dtype=float,
    )


def test_generative_model_large_counts_stay_a_distribution() -> None:
    """Issue #2: count-sized observations must not collapse normalization."""
    model = GenerativeModel(
        "categorical",
        {
            "state_dim": 3,
            "obs_dim": 4,
            "A": _issue_observation_model(),
            "message_passing": False,
        },
    )

    for _ in range(3):
        beliefs = model.update_beliefs(
            {"observations": np.array([8.0, 6.0, 4.0, 2.0])}
        )["states"]
        assert np.all(np.isfinite(beliefs))
        assert np.all(beliefs >= 0.0)
        np.testing.assert_allclose(np.sum(beliefs), 1.0)


def test_small_count_update_matches_log_space_bayes_rule() -> None:
    """Stable updates preserve the ordinary categorical Bayes result."""
    observation_model = _issue_observation_model()
    prior = np.ones(3) / 3.0
    observation = np.array([0.8, 0.6, 0.4, 0.2])

    expected_log = observation @ np.log(observation_model) + np.log(prior)
    expected = np.exp(expected_log - np.max(expected_log))
    expected /= np.sum(expected)

    actual = categorical_posterior(prior, observation, observation_model)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_generative_model_transition_matrix_changes_categorical_trajectory() -> None:
    """Issue #3: B must be applied before the categorical observation update."""
    observation_model = np.array(
        [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.2, 0.2, 0.2]],
        dtype=float,
    )
    identity = np.eye(3)
    cycle = np.roll(identity, 1, axis=0)
    params = {
        "state_dim": 3,
        "obs_dim": 3,
        "A": observation_model,
        "D": np.array([1.0, 0.0, 0.0]),
        "message_passing": False,
    }

    frozen = GenerativeModel("categorical", {**params, "B": identity})
    rotating = GenerativeModel("categorical", {**params, "B": cycle})
    observation = {"observations": np.array([1.0, 0.0, 0.0])}

    frozen_beliefs = frozen.update_beliefs(observation)["states"]
    rotating_beliefs = rotating.update_beliefs(observation)["states"]

    np.testing.assert_allclose(frozen_beliefs, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotating_beliefs, [0.0, 1.0, 0.0])


def test_hierarchical_categorical_updates_use_per_level_transitions() -> None:
    """Hierarchical categorical levels share the stable predict-update contract."""
    model = GenerativeModel(
        "categorical",
        {
            "hierarchical": True,
            "state_dims": [3, 2],
            "obs_dims": [3, 2],
            "B": {"level_0": np.eye(3), "level_1": np.roll(np.eye(2), 1, axis=0)},
            "message_passing": False,
        },
    )

    updated = model.update_beliefs(
        {
            "level_0": np.array([8.0, 6.0, 4.0]),
            "level_1": np.array([8.0, 6.0]),
        }
    )
    for level in ("level_0", "level_1"):
        beliefs = updated[level]["states"]
        assert np.all(np.isfinite(beliefs))
        np.testing.assert_allclose(np.sum(beliefs), 1.0)


def test_categorical_inputs_fail_loudly_when_invalid() -> None:
    """Invalid observations and transition matrices must not create NaNs."""
    with pytest.raises(ValueError, match="non-negative"):
        categorical_posterior(
            np.ones(2) / 2.0,
            np.array([1.0, -1.0]),
            np.ones((2, 2)) / 2.0,
        )

    with pytest.raises(ValueError, match="shape"):
        GenerativeModel(
            "categorical",
            {"state_dim": 3, "obs_dim": 2, "B": np.eye(2)},
        )


def test_base_categorical_model_applies_transition_before_update() -> None:
    """``models.base.CategoricalModel`` follows the same predict-update semantics."""
    model = CategoricalModel(state_dim=3, obs_dim=3)
    model.beliefs = np.array([1.0, 0.0, 0.0])
    model.set_transition_matrix(np.roll(np.eye(3), -1, axis=0))
    model.set_likelihood_matrix(
        np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.2, 0.2, 0.2]])
    )

    beliefs = model.update_beliefs(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(beliefs, [0.0, 1.0, 0.0])


def test_shared_belief_updater_handles_large_counts() -> None:
    """The shared categorical helper cannot reintroduce issue #2."""
    beliefs = BayesianBeliefUpdate().update_categorical(
        np.ones(3) / 3.0,
        np.array([8.0, 6.0, 4.0, 2.0]),
        _issue_observation_model(),
    )
    assert np.all(np.isfinite(beliefs))
    np.testing.assert_allclose(np.sum(beliefs), 1.0)


def test_zero_normalization_returns_uniform_distribution() -> None:
    """A zero-mass fallback must still satisfy the distribution contract."""
    np.testing.assert_allclose(normalize_distribution(np.zeros(3)), [1 / 3] * 3)


def test_package_version_matches_project_metadata() -> None:
    """Packaging metadata and the runtime export must not drift apart."""
    import geo_infer_act

    project = tomllib.loads(
        (REPO_ROOT / "GEO-INFER-ACT" / "pyproject.toml").read_text()
    )
    assert geo_infer_act.__version__ == project["project"]["version"]


def test_stigmergic_activity_does_not_decay_from_negative_belief_contrast() -> None:
    """Environmental traces accumulate instead of being erased by one agent."""

    class StubEnvironment:
        def __init__(self) -> None:
            self.environmental_states = {"cell": SimpleNamespace(human_activity=0.0)}

        def observe_environment(self, observations: dict, timestamp: float) -> None:
            for cell, values in observations.items():
                self.environmental_states[cell].human_activity = values[
                    "human_activity"
                ]

    environment = StubEnvironment()
    model = MultiAgentModel(n_agents=0, environmental_engine=environment)
    model._apply_stigmergy(
        {"cell": {"beliefs": [0.1, 0.2, 0.3, 0.4], "agent_index": 0}}
    )
    deposited = environment.environmental_states["cell"].human_activity

    model._apply_stigmergy(
        {"cell": {"beliefs": [0.4, 0.2, 0.3, 0.1], "agent_index": 0}}
    )
    assert environment.environmental_states["cell"].human_activity == deposited
