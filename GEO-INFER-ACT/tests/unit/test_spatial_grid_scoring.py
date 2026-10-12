"""
Unit tests for H3 spatial-grid information-gain scoring (active sensing).

Covers the spatial agent and the multi-agent stigmergic grid score, and the
multi-resolution aggregation path (leaf-cell scores averaged to a coarser
H3 parent resolution).
"""

from __future__ import annotations

import numpy as np

from geo_infer_act.core.spatial_agent import SpatialActiveInferenceAgent
from geo_infer_act.models.multi_agent import MultiAgentModel
from geo_infer_act.utils.h3_adapter import get_h3_adapter


def _cells(count: int) -> list[str]:
    """Return real H3 cells at resolution 9."""
    adapter = get_h3_adapter()
    center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
    candidates = [center, *adapter.grid_disk(center, 2)]
    return list(dict.fromkeys(candidates))[:count]


def test_spatial_per_cell_scores_are_normalized_uncertainty() -> None:
    """A uniform cell scores near 1 (max uncertainty); a peaked cell near 0."""
    cells = _cells(2)
    agent = SpatialActiveInferenceAgent(initial_cells=cells, state_dim=4)
    agent.beliefs[0] = np.array([0.25, 0.25, 0.25, 0.25])
    agent.beliefs[1] = np.array([0.9, 0.05, 0.03, 0.02])

    scoring = agent.score_spatial_information_gain()
    scores = scoring["scores"]
    assert set(scores) == set(cells)
    assert scores[cells[0]] > 0.95
    assert scores[cells[1]] < 0.5
    assert scoring["best_cells"][0] == cells[0]
    assert 0.0 <= scoring["mean_score"] <= 1.0


def test_spatial_aggregates_scores_to_coarser_resolution() -> None:
    """Supplying a coarser target resolution produces one score per parent."""
    cells = _cells(8)
    agent = SpatialActiveInferenceAgent(
        initial_cells=cells, state_dim=4, h3_resolution=9
    )
    agent.beliefs[:, :] = 0.25  # uniform → 1.0 everywhere

    coarser = 7
    scoring = agent.score_spatial_information_gain(target_resolution=coarser)
    assert scoring["resolution"] == coarser
    # Several resolution-9 children share each resolution-7 parent.
    assert 0 < len(scoring["scores"]) <= len(cells)
    assert all(np.isclose(v, 1.0, atol=1e-6) for v in scoring["scores"].values())


def test_multi_agent_scoring_ranks_uncertain_cells() -> None:
    """Multi-agent grid scoring ranks a uniform (uncertain) agent highest."""
    boundary = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.42, 37.77],
                [-122.42, 37.80],
                [-122.40, 37.80],
                [-122.40, 37.77],
                [-122.42, 37.77],
            ]
        ],
    }
    model = MultiAgentModel(n_agents=3)
    model.enable_h3_spatial(resolution=7, boundary=boundary)
    assert len(model.agent_models) >= 1
    for agent in model.agent_models:
        agent.beliefs = np.ones(4) / 4.0
    # Peaked beliefs on the first agent drop its score below the others.
    model.agent_models[0].beliefs = np.array([0.95, 0.02, 0.02, 0.01])
    model.agent_models[1].beliefs = np.array([0.25, 0.25, 0.25, 0.25])

    scoring = model.score_spatial_information_gain()
    assert scoring["count_cells"] == len(model.h3_cells)
    assert scoring["best_cells"][0] == str(model.h3_cells[1])
    assert scoring["resolution"] == 7
    assert scoring["mean_score"] > 0.0
