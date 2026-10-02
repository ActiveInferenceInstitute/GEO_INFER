"""Research statistics for spatial H3 active-inference traces."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.runners.h3 import h3_cells_for_config
from geo_infer_act.utils.spatial_research import (
    apply_h3_research_profile,
    build_spatial_research_statistics,
    statistics_summary_rows,
)


@pytest.fixture(scope="module")
def trace_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cells = h3_cells_for_config(resolution=8, ring_size=1)
    model = GenerativeModel(
        "categorical",
        {"state_dim": 4, "obs_dim": 4, "spatial_mode": True},
    )
    model.spatial_mode = True
    model.h3_cells = cells
    model.spatial_graph = model._build_h3_neighbor_graph(cells)
    active = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=19,
    )
    active.set_generative_model(model)
    apply_h3_research_profile(model, active)

    cell_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    previous_beliefs = {}
    for timestep in range(3):
        # Independently scored cells receive different observed states, and
        # each cell advances through those states over time. This analytical
        # input exercises real policy switches without carrying another cell's
        # posterior into the next cell's prior.
        vector_observations = {
            cell: np.eye(4)[(index + timestep) % 4] for index, cell in enumerate(cells)
        }
        grid_result = active.infer_over_h3_grid(
            vector_observations,
            return_result=True,
        )
        trace = active.trace_over_h3_grid(
            vector_observations,
            timestep=timestep,
            previous_beliefs=previous_beliefs,
            grid_result=grid_result,
        )
        previous_beliefs = {item.cell: item.belief for item in trace.cell_diagnostics}
        cell_rows.extend(item.to_dict() for item in trace.cell_diagnostics)
        edge_rows.extend(item.to_dict() for item in trace.edge_diagnostics)
    return cell_rows, edge_rows


def test_spatial_research_statistics_capture_non_degenerate_trace_math(
    trace_rows,
) -> None:
    cell_rows, edge_rows = trace_rows

    statistics = build_spatial_research_statistics(cell_rows, edge_rows, [])

    assert statistics["schema_version"].endswith("spatial-research-statistics/v1")
    assert statistics["metric_summaries"]["entropy"]["std"] > 1e-3
    assert statistics["metric_summaries"]["selected_action_probability"]["std"] > 1e-4
    assert statistics["policy"]["switch_count"] >= 1
    assert statistics["policy"]["dominant_action_share"] < 1.0
    assert math.isfinite(statistics["spatial_graph"]["moran_entropy_proxy"])
    assert statistics["spatial_graph"]["mean_edge_belief_distance"] > 0.0
    assert statistics["spatial_graph"]["mean_neighbor_entropy_contrast"] >= 0.0
    assert statistics["spatial_graph"]["mean_abs_flux_balance"] >= 0.0
    assert statistics["non_degenerate"]["unique_selected_action_count"] >= 2


def test_spatial_research_statistics_nested_residuals_and_summary_rows(
    trace_rows,
) -> None:
    cell_rows, edge_rows = trace_rows
    parent_child_rows = [
        {
            "timestep": 0,
            "parent": "parent_a",
            "child": "child_a",
            "cross_level_consistency": 0.82,
            "cross_level_residual": 0.18,
        },
        {
            "timestep": 1,
            "parent": "parent_a",
            "child": "child_a",
            "cross_level_consistency": 0.88,
            "cross_level_residual": 0.12,
        },
    ]
    parent_rows = [
        {
            **cell_rows[0],
            "cell": "parent_a",
            "aggregate_parent_cell": True,
            "posterior_delta": 0.07,
        }
    ]
    level_rows = [
        {
            "resolution": 7,
            "mean_entropy": 0.8,
            "mean_free_energy": 0.2,
            "mean_policy_entropy": 1.1,
        }
    ]

    statistics = build_spatial_research_statistics(
        [*cell_rows, *parent_rows],
        edge_rows,
        level_rows,
        parent_child_rows,
    )
    rows = statistics_summary_rows(statistics)

    assert np.isclose(statistics["nested"]["mean_parent_child_residual"], 0.15)
    assert np.isclose(statistics["nested"]["max_parent_child_residual"], 0.18)
    assert statistics["nested"]["parent_aggregate_drift"] == 0.07
    assert statistics["nested"]["level_summaries"]["7"]["mean_entropy"] == 0.8
    assert any(row["metric"] == "mean_edge_belief_distance" for row in rows)
