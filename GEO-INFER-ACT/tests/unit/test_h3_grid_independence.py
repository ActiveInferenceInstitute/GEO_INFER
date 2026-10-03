"""Spatial scoring preserves independent priors and H3 input permutations."""

from __future__ import annotations

import h3
import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.models.climate import ClimateModel


def test_h3_grid_posteriors_equal_independent_bayes_oracle_in_every_order():
    cells = list(h3.grid_disk(h3.latlng_to_cell(41.75, -124.2, 8), 1))[:3]
    prior = np.array([0.3, 0.7])
    likelihood = np.array([[0.8, 0.1], [0.2, 0.9]])
    model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
    model.observation_model = likelihood
    model.transition_model = np.repeat(np.eye(2)[:, :, None], 2, axis=2)
    model.beliefs = {"states": prior.copy()}
    model.parameters["num_controls"] = 2
    model.preferences = {"observations": np.array([0.0, 1.0])}
    active = ActiveInferenceModel("categorical", random_seed=9)
    active.set_generative_model(model)
    grid = dict(
        zip(cells, [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])])
    )
    forward = active.infer_over_h3_grid(grid, return_result=True)
    reverse = active.infer_over_h3_grid(
        dict(reversed(list(grid.items()))), return_result=True
    )
    for cell, observation in grid.items():
        expected = likelihood[int(np.argmax(observation))] * prior
        expected /= expected.sum()
        np.testing.assert_allclose(
            forward.cell_results[cell].beliefs["states"], expected, atol=1e-12
        )
        np.testing.assert_allclose(
            reverse.cell_results[cell].beliefs["states"], expected, atol=1e-12
        )
        assert forward.cell_results[cell].action == reverse.cell_results[cell].action
    np.testing.assert_array_equal(active.current_beliefs["states"], prior)
    np.testing.assert_array_equal(model.beliefs["states"], prior)
    assert active.history == []


@pytest.mark.parametrize("fail_second_cell", [False, True])
def test_factored_grid_queries_preserve_joint_prior_order_and_next_real_posterior(
    fail_second_cell,
):
    active = ClimateModel(
        config={"policy_selection_mode": "deterministic"}, random_seed=9
    )
    control = ClimateModel(
        config={"policy_selection_mode": "deterministic"}, random_seed=9
    )
    active.step([0, 0])
    control.step([0, 0])
    retained_prior = active._factored_next_prior.copy()
    cells = list(h3.grid_disk(h3.latlng_to_cell(41.75, -124.2, 8), 1))[:2]
    grid = {cells[0]: [2, 2], cells[1]: [1, 1]}
    if fail_second_cell:
        grid[cells[1]] = [np.nan, 1]
        with pytest.raises(ValueError, match="finite"):
            active.infer_over_h3_grid(grid, return_result=True)
    else:
        likelihoods = active.generative_model.observation_model
        forward = active.infer_over_h3_grid(grid, return_result=True)
        reverse = active.infer_over_h3_grid(
            dict(reversed(list(grid.items()))), return_result=True
        )
        for cell, observation in grid.items():
            posterior = retained_prior.copy()
            for matrix, observed_index in zip(likelihoods, observation):
                posterior *= matrix[observed_index].reshape(-1)
            posterior /= posterior.sum()
            for result in (forward, reverse):
                actual = result.cell_results[cell].beliefs
                expected_joint = posterior.reshape(3, 3)
                np.testing.assert_allclose(
                    actual["states"][0], expected_joint.sum(axis=1)
                )
                np.testing.assert_allclose(
                    actual["states"][1], expected_joint.sum(axis=0)
                )
            assert (
                forward.cell_results[cell].action == reverse.cell_results[cell].action
            )
    np.testing.assert_array_equal(active._factored_next_prior, retained_prior)
    assert len(active.history) == len(control.history) == 1
    actual = active.step([1, 2], return_result=True)
    expected = control.step([1, 2], return_result=True)
    for posterior, oracle in zip(actual.beliefs["states"], expected.beliefs["states"]):
        np.testing.assert_array_equal(posterior, oracle)
    assert actual.action == expected.action


@pytest.mark.parametrize("has_measurement", [False, True])
def test_failed_gaussian_grid_query_restores_retained_measurement_free_energy(
    has_measurement,
):
    model = GenerativeModel("gaussian", {"state_dim": 2, "obs_dim": 1})
    active = ActiveInferenceModel("gaussian")
    active.set_generative_model(model)
    if has_measurement:
        active.perceive([2.0])
    original_beliefs = {name: value.copy() for name, value in model.beliefs.items()}
    original_diagnostics = model._last_gaussian_free_energy.copy()
    original_vfe = active.compute_free_energy()
    cell = h3.latlng_to_cell(41.75, -124.2, 8)
    with pytest.raises(ValueError, match="Gaussian action selection requires"):
        active.infer_over_h3_grid({cell: [99.0]}, return_result=True)
    assert model._last_gaussian_free_energy == original_diagnostics
    assert active.compute_free_energy() == original_vfe
    for name, value in model.beliefs.items():
        np.testing.assert_array_equal(value, original_beliefs[name])
    assert active.history == []


@pytest.mark.parametrize("nested", [False, True])
def test_h3_per_cell_scoring_accepts_the_declared_optional_none_seed(nested):
    cell = h3.latlng_to_cell(41.75, -124.2, 8)
    model = GenerativeModel(
        "categorical",
        {
            "state_dim": 2,
            "obs_dim": 2,
            "A": np.eye(2),
            "B": np.eye(2),
            "D": [0.3, 0.7],
            "random_seed": None,
            "spatial_mode": True,
        },
    )
    if nested:
        model.enable_nested_h3_spatial([7, 8], cells=[cell], top_down_weight=0)
        result = model.update_nested_h3_beliefs({cell: [1, 0]}, return_result=True)
        beliefs = result.fine_beliefs
    else:
        model.h3_cells = [cell]
        result = model.update_h3_beliefs({cell: [1, 0]}, return_result=True)
        beliefs = result.h3_beliefs
    np.testing.assert_allclose(beliefs[cell], [1, 0], atol=1e-12)


@pytest.mark.parametrize("nested", [False, True])
def test_observed_h3_aggregate_is_measurement_vfe_from_an_independent_evidence_oracle(
    nested,
):
    cells = list(h3.grid_disk(h3.latlng_to_cell(41.75, -124.2, 8), 1))[:2]
    model = GenerativeModel(
        "categorical",
        {
            "state_dim": 2,
            "obs_dim": 2,
            "A": np.array([[0.8, 0.3], [0.2, 0.7]]),
            "B": np.eye(2),
            "D": [0.6, 0.4],
            "spatial_mode": True,
        },
    )
    observations = {cells[0]: [1, 0], cells[1]: [0, 1]}
    if nested:
        model.enable_nested_h3_spatial([7, 8], cells=cells, top_down_weight=0.3)
        result = model.update_nested_h3_beliefs(observations, return_result=True)
        for summary in result.level_summaries:
            assert "mean_reference_kl_to_uniform" in summary.to_dict()
            assert "mean_free_energy" not in summary.to_dict()
        assert result.metadata["aggregate_free_energy_definition"].endswith(
            "before_spatial_blending"
        )
    else:
        model.h3_cells = cells
        result = model.update_h3_beliefs(observations, return_result=True)
        reference_kl = 0.0
        for belief in result.h3_beliefs.values():
            reference_kl += float(np.sum(belief * np.log(belief / 0.5))) / 2
        assert result.metadata["reference_kl_to_uniform"] == pytest.approx(reference_kl)
    expected = -0.5 * (np.log(0.8 * 0.6 + 0.3 * 0.4) + np.log(0.2 * 0.6 + 0.7 * 0.4))
    assert result.aggregate_free_energy == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize("fail_second_cell", [False, True])
def test_read_only_grid_scoring_preserves_real_analyzer_history(
    tmp_path, fail_second_cell
):
    model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
    model.observation_model = np.array([[0.8, 0.1], [0.2, 0.9]])
    model.transition_model = np.repeat(np.eye(2)[:, :, None], 2, axis=2)
    model.parameters["num_controls"] = 2
    model.preferences = {"observations": np.array([0.0, 1.0])}
    active = ActiveInferenceModel("categorical", random_seed=9, output_dir=tmp_path)
    active.set_generative_model(model)
    active.step(np.array([1.0, 0.0]))
    analyzer = active.analyzer
    original_history = analyzer.export_full_history().read_bytes()
    original_trace_count = len(analyzer.step_history)
    original_agent_count = len(active.history)
    cells = list(h3.grid_disk(h3.latlng_to_cell(41.75, -124.2, 8), 1))[:2]
    grid = {cells[0]: np.array([0.0, 1.0]), cells[1]: np.array([1.0, 0.0])}
    if fail_second_cell:
        grid[cells[1]] = np.array([np.nan, 0.0])
        with pytest.raises(ValueError, match="finite"):
            active.infer_over_h3_grid(grid, return_result=True)
    else:
        active.infer_over_h3_grid(grid, return_result=True)
    assert active.analyzer is analyzer
    assert len(active.history) == original_agent_count
    assert len(analyzer.step_history) == original_trace_count
    assert analyzer.export_full_history().read_bytes() == original_history
    active.step(np.array([0.0, 1.0]))
    assert len(analyzer.step_history) == original_trace_count + 1
