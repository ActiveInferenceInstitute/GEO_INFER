"""pymdp 1.0.3 backend contracts for H3 active inference."""

from __future__ import annotations

import math

import numpy as np
import h3
import pytest

from geo_infer_act import ActiveInferenceModel, GenerativeModel
from geo_infer_act.core.types import (
    H3GridInferenceResult,
    NestedH3GridInferenceResult,
)
from geo_infer_act.utils import pymdp_adapter
from geo_infer_act.utils.pymdp_adapter import (
    EXPECTED_PYMDP_VERSION,
    real_h3_version_metadata,
    run_model_step,
    run_pymdp_step,
    validate_pymdp_version,
)


def _assert_probability(values: object) -> None:
    array = np.asarray(values, dtype=float).reshape(-1)
    assert array.size > 0
    assert np.all(np.isfinite(array))
    assert np.all(array >= -1e-12)
    assert math.isclose(float(array.sum()), 1.0, abs_tol=1e-6)


def _assert_pymdp_metadata(metadata: dict[str, object]) -> None:
    assert metadata["backend"] == "inferactively-pymdp"
    assert metadata["pymdp_version"] == EXPECTED_PYMDP_VERSION
    assert metadata["h3_version"] == "4.5.0"
    _assert_probability(metadata["action_posterior"])
    neg_efe = np.asarray(metadata["negative_expected_free_energy"], dtype=float)
    assert neg_efe.size == len(metadata["action_posterior"])
    assert np.all(np.isfinite(neg_efe))
    assert math.isfinite(float(metadata["free_energy"]))


def _cells() -> list[str]:
    center = h3.latlng_to_cell(37.7749, -122.4194, 9)
    return [center, *sorted(h3.grid_ring(center, 1))[:2]]


def _observations(cells: list[str]) -> dict[str, np.ndarray]:
    return {cell: np.eye(4, dtype=float)[idx % 4] for idx, cell in enumerate(cells)}


def test_pymdp_version_detection_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_pymdp_version() == EXPECTED_PYMDP_VERSION

    monkeypatch.setattr(pymdp_adapter, "installed_pymdp_version", lambda: "0.0.7.1")
    with pytest.raises(RuntimeError, match="inferactively-pymdp 1.0.3 is required"):
        validate_pymdp_version()


def test_real_h3_version_metadata_is_exact() -> None:
    versions = real_h3_version_metadata()

    assert versions["h3_version"] == "4.5.0"
    assert versions["h3_c_version"] == "4.5.0"


def test_pymdp_adapter_runs_seeded_jax_agent_step() -> None:
    observation_model = np.eye(4, dtype=float)
    transition_model = np.repeat(np.eye(4, dtype=float)[:, :, None], 3, axis=2)
    observation = np.array([0.0, 1.0, 0.0, 0.0])

    first = run_pymdp_step(
        observation=observation,
        observation_model=observation_model,
        transition_model=transition_model,
        preferences=np.zeros(4),
        prior=np.ones(4) / 4,
        action_count=3,
        random_seed=17,
    )
    second = run_pymdp_step(
        observation=observation,
        observation_model=observation_model,
        transition_model=transition_model,
        preferences=np.zeros(4),
        prior=np.ones(4) / 4,
        action_count=3,
        random_seed=17,
    )

    _assert_probability(first.beliefs)
    _assert_probability(first.policy_posterior)
    assert np.all(np.isfinite(first.negative_expected_free_energy))
    assert math.isfinite(first.free_energy)
    assert first.selected_action_index == second.selected_action_index
    assert np.allclose(first.policy_posterior, second.policy_posterior)
    _assert_pymdp_metadata(first.to_metadata())


def test_run_model_step_uses_generativemodel_matrices() -> None:
    model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})

    result = run_model_step(model, np.array([1.0, 0.0, 0.0, 0.0]), random_seed=3)

    _assert_probability(result.beliefs)
    _assert_pymdp_metadata(result.to_metadata())


def test_run_model_step_normalizes_list_valued_action_count() -> None:
    model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})

    result = run_model_step(
        model,
        np.array([1.0, 0.0, 0.0, 0.0]),
        action_count=[2],
        random_seed=3,
    )

    assert result.metadata["action_count"] == 2
    assert result.policy_posterior.size == 2


def test_flat_h3_grid_inference_exposes_real_pymdp_metadata() -> None:
    cells = _cells()
    model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    active = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=11,
    )
    active.set_generative_model(model)

    as_dict = active.infer_over_h3_grid(_observations(cells))
    typed = active.infer_over_h3_grid(_observations(cells), return_result=True)

    assert isinstance(typed, H3GridInferenceResult)
    assert typed.metadata["pymdp_backend"] == "inferactively-pymdp"
    for cell in cells:
        _assert_pymdp_metadata(as_dict[cell]["pymdp"])
        _assert_pymdp_metadata(typed.cell_results[cell].metadata["pymdp"])


def test_nested_h3_grid_inference_exposes_pymdp_parent_child_metadata() -> None:
    cells = _cells()
    model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    model.enable_nested_h3_spatial([7, 8, 9], cells=cells)
    active = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=13,
    )
    active.set_generative_model(model)

    result = active.infer_over_nested_h3_grid(_observations(cells), return_result=True)

    assert isinstance(result, NestedH3GridInferenceResult)
    assert result.metadata["pymdp_backend"] == "inferactively-pymdp"
    assert (
        result.nested_belief_update.metadata["pymdp_backend"] == "inferactively-pymdp"
    )
    assert result.nested_belief_update.parent_beliefs
    assert (
        result.nested_belief_update.spatial_consistency.metadata[
            "cross_level_coherence"
        ]
        >= 0.0
    )
    for metadata in result.nested_belief_update.metadata[
        "pymdp_cell_metadata"
    ].values():
        _assert_pymdp_metadata(metadata)
