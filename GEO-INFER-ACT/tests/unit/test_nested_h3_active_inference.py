"""Nested H3 active inference contracts."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import h3

from geo_infer_act import (
    ActiveInferenceModel,
    GenerativeModel,
    NestedH3BeliefUpdateResult,
    NestedH3GridInferenceResult,
    SpatialActiveInferenceAgent,
)
from geo_infer_act.models.multi_agent import MultiAgentModel
from geo_infer_act.runners import RunConfig, run_scenario
from geo_infer_act.runners.h3 import h3_cells_for_config
from geo_infer_act.utils.pymdp_adapter import EXPECTED_PYMDP_VERSION


def _leaf_cells() -> list[str]:
    return h3_cells_for_config(resolution=9, ring_size=0)


def _observations(cells: list[str]) -> dict[str, np.ndarray]:
    return {cell: np.eye(4, dtype=float)[index % 4] for index, cell in enumerate(cells)}


def _assert_probability(value: object) -> None:
    array = np.asarray(value, dtype=float).reshape(-1)
    assert array.size > 0
    assert np.all(np.isfinite(array))
    assert np.all(array >= -1e-12)
    assert np.isclose(float(array.sum()), 1.0)


def test_generative_model_nested_h3_belief_update_is_typed_and_normalized() -> None:
    cells = _leaf_cells()
    model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    hierarchy = model.enable_nested_h3_spatial([7, 8, 9], cells=cells)

    result = model.update_nested_h3_beliefs(_observations(cells), return_result=True)

    assert isinstance(result, NestedH3BeliefUpdateResult)
    assert hierarchy["validation"]["is_valid"] is True
    assert result.level_summaries
    assert math.isfinite(result.aggregate_free_energy)
    assert result.spatial_consistency.metadata["cross_level_coherence"] >= 0.0
    for belief in result.fine_beliefs.values():
        _assert_probability(belief)
    for belief in result.parent_beliefs.values():
        _assert_probability(belief)

    with pytest.raises(ValueError, match="outside this nested hierarchy"):
        model.update_nested_h3_beliefs(
            {h3.latlng_to_cell(40.7128, -74.0060, 9): np.ones(4) / 4}
        )


def test_active_model_nested_h3_grid_inference_preserves_state() -> None:
    cells = _leaf_cells()
    gen = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    gen.enable_nested_h3_spatial([7, 8, 9], cells=cells)
    active = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=7,
    )
    active.set_generative_model(gen)
    original_beliefs = np.array(gen.beliefs["states"], copy=True)

    result = active.infer_over_nested_h3_grid(_observations(cells), return_result=True)

    assert isinstance(result, NestedH3GridInferenceResult)
    assert len(result.cell_results) == len(cells)
    assert np.allclose(gen.beliefs["states"], original_beliefs)
    assert math.isfinite(result.aggregate_free_energy)


def test_spatial_agent_and_multi_agent_nested_h3_paths() -> None:
    cells = _leaf_cells()
    observations = _observations(cells)

    agent = SpatialActiveInferenceAgent(
        initial_cells=cells,
        h3_resolution=9,
        enable_logging=False,
    )
    agent.enable_nested_h3_spatial([7, 8, 9], cells=cells)
    agent_result = agent.step_nested(observations, return_result=True)
    assert isinstance(agent_result, NestedH3GridInferenceResult)
    assert len(agent_result.cell_results) == len(cells)
    assert math.isfinite(agent_result.aggregate_free_energy)

    multi = MultiAgentModel(n_agents=2)
    multi.enable_nested_h3_spatial([7, 8, 9], cells=cells)
    sim = multi.simulate_nested_h3_lattice(
        1, lambda _cell: np.array([1.0, 0.0, 0.0, 0.0])
    )
    assert len(sim["history"]) == 1
    assert len(sim["nested_history"]) == 1
    assert sim["parent_count"] > 0
    assert sim["nested_history"][0]["level_summaries"]


def test_nested_h3_runner_outputs_are_manifested_and_temp_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]

    def scoped_status() -> str:
        status = subprocess.run(
            [
                "git",
                "status",
                "--short",
                "--",
                "test_output",
                "output",
                "outputs",
                "visualizations_output",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return status.stdout

    status_before = scoped_status()
    result = run_scenario(
        RunConfig(
            scenario="h3",
            output_dir=tmp_path / "h3",
            seed=17,
            deterministic=True,
            timesteps=1,
            h3_resolution=9,
            h3_ring_size=0,
            visualizations=True,
            parameters={"nested_h3": True},
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    generated = {item["path"] for item in manifest["generated_files"]}
    assert manifest["validation"]["status"] == "passed"
    assert "data/h3_hierarchy.csv" in generated
    assert "data/nested_h3_diagnostics.json" in generated
    assert "data/nested_h3_cell_diagnostics.csv" in generated
    assert "data/nested_h3_parent_child_diagnostics.csv" in generated
    assert "data/nested_h3_level_diagnostics.csv" in generated
    assert "data/spatial_inference_trace.json" in generated
    assert "data/spatial_research_statistics.json" in generated
    assert "data/pymdp_h3_diagnostics.json" in generated
    assert "data/pymdp_policy_posteriors.csv" in generated
    assert "visualizations/nested_h3_level_map.html" in generated
    assert "visualizations/nested_h3_hierarchy_map.html" in generated
    assert "visualizations/nested_h3_parent_child_residuals.html" in generated
    assert "visualizations/h3_belief_flux_map.html" in generated
    assert "visualizations/h3_policy_surface.html" in generated
    assert "visualizations/h3_policy_transitions.html" in generated
    assert "visualizations/h3_spatial_autocorrelation.html" in generated
    assert "visualizations/h3_entropy_free_energy_phase.html" in generated
    assert "visualizations/pymdp_policy_free_energy.html" in generated
    assert result.metrics["nested_h3"] is True
    assert result.metrics["nested_orphan_count"] == 0
    assert result.metrics["pymdp_backend"] == "inferactively-pymdp"
    assert result.metrics["pymdp_version"] == EXPECTED_PYMDP_VERSION
    assert result.metrics["h3_version"] == "4.5.0"

    assert scoped_status() == status_before
