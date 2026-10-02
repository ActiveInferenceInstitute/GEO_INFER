"""Spatial scoring preserves independent priors and H3 input permutations."""

from __future__ import annotations

import h3
import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel


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
