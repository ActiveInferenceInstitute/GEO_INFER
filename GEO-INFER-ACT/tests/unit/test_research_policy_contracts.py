"""Actual research inference obeys finite posterior and policy references."""

from __future__ import annotations

import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.spatial_agent import SpatialActiveInferenceAgent
from geo_infer_act.runners.h3 import (
    generate_realistic_environmental_observations,
    h3_cells_for_config,
    observation_dict_to_vector,
)
from geo_infer_act.utils.pymdp_adapter import PymdpStepResult
from geo_infer_act.utils.spatial_research import (
    apply_h3_research_profile,
    apply_spatial_agent_research_profile,
)
from geo_infer_test.act_research_oracles import (
    assert_research_policy,
    research_posterior_reference,
    research_profile_reference,
)


@pytest.fixture(scope="module")
def research_actor():
    cells = h3_cells_for_config(resolution=8, ring_size=1)
    model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    model.spatial_mode = True
    model.h3_cells = cells
    model.spatial_graph = model._build_h3_neighbor_graph(cells)
    actor = ActiveInferenceModel(
        "categorical", policy_selection_mode="deterministic", random_seed=41
    )
    actor.set_generative_model(model)
    apply_h3_research_profile(model, actor)
    expected_a, expected_b, expected_c, expected_d = research_profile_reference()
    for actual, expected in (
        (model.observation_model, expected_a),
        (model.transition_model, expected_b),
        (model.preferences["observations"], expected_c),
        (actor.current_beliefs["states"], expected_d),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    return cells, actor


@pytest.mark.parametrize("diagnostic", [False, True])
def test_actual_grid_matches_posterior_and_policy_oracles(research_actor, diagnostic):
    cells, actor = research_actor
    observations = (
        {cell: np.eye(4)[index % 4] for index, cell in enumerate(cells)}
        if diagnostic
        else {
            cell: observation_dict_to_vector(value)
            for cell, value in generate_realistic_environmental_observations(
                cells, 0.0, spatial_seed=41
            ).items()
        }
    )
    result = actor.infer_over_h3_grid(observations, return_result=True)
    selected = set()
    for cell, step in result.cell_results.items():
        expected_posterior = research_posterior_reference(observations[cell])
        metadata = step.metadata["pymdp"]
        np.testing.assert_allclose(
            step.beliefs["states"], expected_posterior, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            metadata["policy_beliefs"], expected_posterior, rtol=0, atol=1e-6
        )
        assert_research_policy(
            beliefs=expected_posterior,
            action_posterior=metadata["action_posterior"],
            negative_expected_free_energy=metadata["negative_expected_free_energy"],
            selected_action_index=metadata["selected_action_index"],
            label=cell,
        )
        selected.add(metadata["selected_action_index"])
    assert selected == ({1, 2, 3} if diagnostic else {0})


def test_policy_beliefs_remain_the_scored_posterior_after_spatial_diffusion():
    cells = h3_cells_for_config(resolution=8, ring_size=1)
    agent = SpatialActiveInferenceAgent(
        initial_cells=cells,
        h3_resolution=8,
        state_dim=4,
        obs_dim=4,
        enable_logging=False,
    )
    apply_spatial_agent_research_profile(agent)
    observations = {cell: np.eye(4)[index % 4] for index, cell in enumerate(cells)}
    result = agent.step(observations, return_result=True)
    displayed_differs = False
    for cell, step in result.cell_results.items():
        expected_posterior = research_posterior_reference(observations[cell])
        metadata = step.metadata["pymdp"]
        np.testing.assert_allclose(
            metadata["policy_beliefs"], expected_posterior, rtol=0, atol=1e-6
        )
        assert_research_policy(
            beliefs=expected_posterior,
            action_posterior=metadata["action_posterior"],
            negative_expected_free_energy=metadata["negative_expected_free_energy"],
            selected_action_index=metadata["selected_action_index"],
            label=cell,
        )
        displayed_differs |= not np.allclose(step.beliefs, expected_posterior)
    assert displayed_differs


def test_pymdp_policy_belief_metadata_owns_its_snapshot():
    result = PymdpStepResult(
        beliefs=np.array([0.2, 0.25, 0.3, 0.25]),
        policy_posterior=np.full(4, 0.25),
        negative_expected_free_energy=np.zeros(4),
        selected_action_index=0,
        free_energy=0.0,
    )
    metadata = result.to_metadata()
    metadata["policy_beliefs"][0] = 9.0
    assert result.beliefs[0] == 0.2
