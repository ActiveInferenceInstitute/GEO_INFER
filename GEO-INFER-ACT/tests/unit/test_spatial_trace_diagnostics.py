"""Spatial active-inference trace diagnostics contracts."""

from __future__ import annotations

import json
import math

import numpy as np
import h3

from geo_infer_act import (
    ActiveInferenceModel,
    GenerativeModel,
    SpatialActiveInferenceAgent,
    SpatialInferenceTrace,
)
from geo_infer_act.utils.pymdp_adapter import EXPECTED_PYMDP_VERSION


def _cells(resolution: int = 9) -> list[str]:
    center = h3.latlng_to_cell(37.7749, -122.4194, resolution)
    return [center, *sorted(h3.grid_ring(center, 1))[:2]]


def _observations(cells: list[str]) -> dict[str, np.ndarray]:
    return {cell: np.eye(4, dtype=float)[idx % 4] for idx, cell in enumerate(cells)}


def _assert_trace(trace: SpatialInferenceTrace, expected_cells: int) -> None:
    assert isinstance(trace, SpatialInferenceTrace)
    assert trace.cell_diagnostics
    assert (
        len(
            [
                row
                for row in trace.cell_diagnostics
                if not row.metadata.get("aggregate_parent_cell")
            ]
        )
        == expected_cells
    )
    assert trace.level_diagnostics
    assert trace.backend_metadata["pymdp_backend"] == "inferactively-pymdp"
    for cell in trace.cell_diagnostics:
        assert math.isfinite(cell.entropy)
        assert math.isfinite(cell.free_energy)
        assert math.isfinite(cell.policy_entropy)
        assert math.isfinite(cell.local_coherence)
        assert math.isfinite(cell.posterior_delta)
        assert math.isfinite(cell.belief_flux_divergence)
        assert np.isclose(sum(cell.belief), 1.0, atol=1e-6)
        if not cell.metadata.get("aggregate_parent_cell"):
            assert cell.action_posterior
            assert np.isclose(sum(cell.action_posterior), 1.0, atol=1e-6)
            assert len(cell.action_posterior) == len(cell.negative_expected_free_energy)
            assert cell.metadata["pymdp_version"] == EXPECTED_PYMDP_VERSION
            assert cell.metadata["h3_version"] == "4.5.0"
    for edge in trace.edge_diagnostics:
        assert edge.source != edge.target
        assert math.isfinite(edge.belief_distance)
        assert math.isfinite(edge.coherence)
    json.dumps(trace.to_dict())


def test_active_model_trace_over_h3_grid_exposes_research_diagnostics() -> None:
    cells = _cells()
    gen = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    active = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=31,
    )
    active.set_generative_model(gen)
    grid_result = active.infer_over_h3_grid(_observations(cells), return_result=True)

    trace = active.trace_over_h3_grid(
        _observations(cells),
        timestep=2,
        previous_beliefs={cell: np.ones(4) / 4 for cell in cells},
        grid_result=grid_result,
    )

    _assert_trace(trace, expected_cells=len(cells))
    assert trace.timesteps == [2]
    assert any(row.posterior_delta >= 0.0 for row in trace.cell_diagnostics)


def test_nested_h3_trace_includes_parent_aggregates_and_residuals() -> None:
    cells = _cells()
    gen = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    gen.enable_nested_h3_spatial([7, 8, 9], cells=cells)
    active = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=37,
    )
    active.set_generative_model(gen)
    grid_result = active.infer_over_nested_h3_grid(
        _observations(cells),
        return_result=True,
    )

    trace = active.trace_over_nested_h3_grid(
        _observations(cells),
        timestep=1,
        grid_result=grid_result,
    )

    _assert_trace(trace, expected_cells=len(cells))
    parent_rows = [
        row
        for row in trace.cell_diagnostics
        if row.metadata.get("aggregate_parent_cell")
    ]
    child_rows = [row for row in trace.cell_diagnostics if row.parent_cell]
    assert parent_rows
    assert child_rows
    assert all(row.metadata["cross_level_consistency"] >= 0.0 for row in child_rows)
    assert set(trace.hierarchy_metadata["resolutions"]) == {7, 8, 9}


def test_spatial_agent_trace_step_reuses_typed_grid_result() -> None:
    cells = _cells()
    observations = _observations(cells)
    agent = SpatialActiveInferenceAgent(
        initial_cells=cells,
        h3_resolution=9,
        state_dim=4,
        obs_dim=4,
        enable_logging=False,
    )
    grid_result = agent.step(observations, return_result=True)

    trace = agent.trace_step(
        observations,
        grid_result=grid_result,
        timestep=0,
        previous_beliefs={cell: np.ones(4) / 4 for cell in cells},
    )

    _assert_trace(trace, expected_cells=len(cells))
