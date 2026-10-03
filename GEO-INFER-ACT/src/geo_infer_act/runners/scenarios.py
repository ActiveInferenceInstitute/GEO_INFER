"""Canonical Active Inference scenario runners."""

from __future__ import annotations

import json
import logging
import math
import sys
import warnings
from datetime import datetime, UTC
from pathlib import Path
from typing import (
    Any,
    cast,
)
from collections.abc import Callable, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geo_infer_act.core.active_inference import (
    ActiveInferenceModel,
)
from geo_infer_act.core.types import (
    ActiveInferenceStepResult,
    NestedH3GridInferenceResult,
)
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.spatial_agent import SpatialActiveInferenceAgent
from geo_infer_act.runners.contracts import (
    RunConfig,
    ScenarioRunResult,
    SuiteRunResult,
    normalize_scenario_list,
)
from geo_infer_act.runners.h3 import (
    generate_realistic_environmental_observations,
    h3_cells_for_config,
    observation_dict_to_vector,
)
from geo_infer_act.runners.io import (
    ensure_output_tree,
    save_matplotlib_figure_artifact,
    write_csv,
    write_html_figure_artifact,
    write_json,
    write_run_manifest,
    write_suite_manifest,
)
from geo_infer_act.utils.analysis import ActiveInferenceAnalyzer
from geo_infer_act.utils.h3_adapter import get_h3_adapter, normalize_belief_vector
from geo_infer_act.utils.spatial_research import (
    apply_h3_research_profile,
    apply_spatial_agent_research_profile,
    build_spatial_research_statistics,
    statistics_summary_rows,
)

logger = logging.getLogger(__name__)


SCENARIO_PARAMETERS: dict[str, dict[str, Any]] = {
    "simple": {"phase": 0.0, "amplitude": 0.18, "preference": [0.1, 0.2, 0.3, 0.4]},
    "modern": {"phase": 0.6, "amplitude": 0.24, "preference": [0.2, 0.1, 0.4, 0.3]},
    "spatial": {
        "phase": 1.1,
        "amplitude": 0.22,
        "preference": [0.15, 0.25, 0.35, 0.25],
    },
    "ecological": {
        "phase": 1.7,
        "amplitude": 0.2,
        "preference": [0.05, 0.15, 0.35, 0.45],
    },
    "urban_planning": {
        "phase": 2.2,
        "amplitude": 0.16,
        "preference": [0.2, 0.35, 0.25, 0.2],
    },
    "verification": {
        "phase": 2.8,
        "amplitude": 0.12,
        "preference": [0.25, 0.25, 0.25, 0.25],
    },
    "debug": {"phase": 3.3, "amplitude": 0.1, "preference": [0.4, 0.3, 0.2, 0.1]},
}


def load_run_config(
    path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load a versioned YAML run config and apply explicit overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with Path(path).open() as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Run config must be a mapping: {path}")
        data.update(loaded)
    if overrides:
        data.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    return RunConfig(**data)


def run_scenario(
    config: RunConfig, command: list[str] | None = None
) -> ScenarioRunResult:
    """Run one canonical Active Inference scenario."""
    config = RunConfig(**config.__dict__)
    output_dir = ensure_output_tree(
        config.output_dir or _default_output_dir(config.scenario)
    )
    config.output_dir = output_dir

    if config.scenario == "h3":
        metrics = _run_h3_scenario(config)
    elif config.scenario == "spatial":
        metrics = _run_spatial_scenario(config)
    else:
        metrics = _run_vector_scenario(config)

    manifest = write_run_manifest(output_dir, config, metrics, command=command)
    generated_files = [
        output_dir / item["path"] for item in manifest["generated_files"]
    ]
    return ScenarioRunResult(
        scenario=config.scenario,
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        manifest=manifest,
        metrics=metrics,
        generated_files=generated_files,
    )


def run_all_scenarios(
    output_dir: Path | None = None,
    scenarios: Iterable[str] | None = None,
    seed: int = 42,
    timesteps: int = 8,
    deterministic: bool = True,
    visualizations: bool = True,
    command: list[str] | None = None,
    config: RunConfig | None = None,
) -> SuiteRunResult:
    """Run a suite of scenarios and write a suite manifest."""
    output_dir = ensure_output_tree(output_dir or _default_output_dir("examples"))
    selected = normalize_scenario_list(scenarios)
    results = []
    for scenario in selected:
        values = (
            config.__dict__
            if config is not None
            else dict(
                seed=seed,
                timesteps=timesteps,
                deterministic=deterministic,
                visualizations=visualizations,
            )
        )
        scenario_config = RunConfig(
            **{**values, "scenario": scenario, "output_dir": output_dir / scenario}
        )
        results.append(run_scenario(scenario_config, command=command))

    manifest = write_suite_manifest(output_dir, results, command=command)
    return SuiteRunResult(
        output_dir=output_dir,
        manifest_path=output_dir / "suite_manifest.json",
        manifest=manifest,
        scenario_results=results,
    )


def _default_output_dir(scenario: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / "output" / f"act_{scenario}_{timestamp}"


def _run_vector_scenario(config: RunConfig) -> dict[str, Any]:
    assert config.output_dir is not None
    params = {**SCENARIO_PARAMETERS[config.scenario], **config.parameters}
    rng = np.random.default_rng(config.seed)
    analyzer = ActiveInferenceAnalyzer(str(config.output_dir))
    model = GenerativeModel(
        "categorical",
        {
            "state_dim": 4,
            "obs_dim": 4,
            "prior_precision": 1.0,
        },
    )
    active_model = ActiveInferenceModel(
        "categorical",
        preferences=list(np.asarray(params["preference"], dtype=float)),
        policy_selection_mode="deterministic" if config.deterministic else "sample",
        random_seed=config.seed,
    )
    active_model.set_generative_model(model)

    step_rows: list[dict[str, Any]] = []
    actions = ["observe", "adapt", "coordinate", "conserve"]
    for timestep in range(config.timesteps):
        observation = _scenario_observation(config.scenario, timestep, params, rng)
        result = cast(
            ActiveInferenceStepResult,
            active_model.step(observation, actions, return_result=True),
        )
        beliefs = _belief_vector(result.beliefs)
        entropy = float(-np.sum(beliefs * np.log(beliefs + 1e-12)))
        row = {
            "timestep": timestep,
            "scenario": config.scenario,
            "action": str(result.action),
            "free_energy": float(result.free_energy),
            "expected_free_energy": (
                float(result.expected_free_energy)
                if result.expected_free_energy is not None
                else None
            ),
            "belief_entropy": entropy,
        }
        row.update({f"belief_{idx}": float(value) for idx, value in enumerate(beliefs)})
        step_rows.append(row)
        analyzer.record_step(
            beliefs=beliefs,
            observations=observation,
            actions=result.action,
            policies={
                "selected": result.action,
                "expected_free_energy": result.expected_free_energy,
            },
            free_energy=float(result.free_energy),
            metrics=row,
            timestamp=float(timestep),
        )

    _finalize_analyzer(analyzer)
    write_csv(config.output_dir / "data" / "step_metrics.csv", step_rows)
    summary = _summary_metrics(step_rows)
    write_json(config.output_dir / "analysis" / "run_summary.json", summary)
    if config.visualizations:
        _plot_vector_summary(config, step_rows)
    return summary


def _run_h3_scenario(config: RunConfig) -> dict[str, Any]:
    """Run flat or nested H3 inference and persist diagnostics and summary metrics."""
    assert config.output_dir is not None
    cells = h3_cells_for_config(
        resolution=config.h3_resolution,
        ring_size=config.h3_ring_size,
        cells=config.h3_cells,
    )
    nested_enabled = bool(config.parameters.get("nested_h3"))
    model = GenerativeModel(
        "categorical",
        {"state_dim": 4, "obs_dim": 4, "spatial_mode": True},
    )
    if nested_enabled:
        model.enable_nested_h3_spatial(
            _nested_h3_resolutions(config),
            cells=cells,
            top_down_weight=float(config.parameters.get("top_down_weight", 0.15)),
        )
        cells = list(model.h3_cells)
    else:
        model.spatial_mode = True
        model.h3_cells = cells
        model.spatial_graph = cast(Any, model._build_h3_neighbor_graph(cells))
    active_model = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic" if config.deterministic else "sample",
        random_seed=config.seed,
    )
    active_model.set_generative_model(model)
    if config.parameters.get("research_profile"):
        apply_h3_research_profile(
            model,
            active_model,
            action_count=int(config.parameters.get("research_action_count", 4)),
        )
    analyzer = ActiveInferenceAnalyzer(str(config.output_dir))

    step_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    pymdp_records: list[dict[str, Any]] = []
    trace_fragments: list[Any] = []
    previous_trace_beliefs: dict[str, Any] = {}
    last_cell_results: dict[str, Any] = {}
    for timestep in range(config.timesteps):
        env_obs = generate_realistic_environmental_observations(
            cells,
            timestep=float(timestep),
            spatial_seed=config.seed,
        )
        vector_obs = {
            cell: observation_dict_to_vector(obs) for cell, obs in env_obs.items()
        }
        if nested_enabled:
            grid_result = active_model.infer_over_nested_h3_grid(
                vector_obs,
                return_result=True,
                top_down_weight=float(config.parameters.get("top_down_weight", 0.15)),
            )
            belief_update = grid_result.nested_belief_update
        else:
            belief_update = model.update_h3_beliefs(vector_obs, return_result=True)
            grid_result = active_model.infer_over_h3_grid(
                vector_obs, return_result=True
            )
        if nested_enabled:
            trace = active_model.trace_over_nested_h3_grid(
                vector_obs,
                timestep=timestep,
                previous_beliefs=previous_trace_beliefs,
                grid_result=grid_result,
                top_down_weight=float(config.parameters.get("top_down_weight", 0.15)),
                scenario="h3",
            )
        else:
            trace = active_model.trace_over_h3_grid(
                vector_obs,
                timestep=timestep,
                previous_beliefs=previous_trace_beliefs,
                grid_result=grid_result,
                scenario="h3",
            )
        trace.metadata["observations_by_cell"] = {
            cell: _finite_vector(observation)
            for cell, observation in vector_obs.items()
        }
        trace_fragments.append(trace)
        previous_trace_beliefs = {
            item.cell: item.belief for item in trace.cell_diagnostics
        }
        coherence = belief_update.spatial_consistency.global_coherence
        neighbor_corr = belief_update.spatial_consistency.neighbor_correlations
        row = {
            "timestep": timestep,
            "scenario": "h3",
            "free_energy": float(grid_result.aggregate_free_energy),
            "expected_free_energy": float(
                np.nanmean(
                    [
                        result.expected_free_energy
                        for result in grid_result.cell_results.values()
                        if result.expected_free_energy is not None
                    ]
                    or [0.0]
                )
            ),
            "cell_count": len(cells),
            "edge_count": belief_update.spatial_consistency.edge_count,
            "coherence": float(coherence),
            "neighbor_correlation": float(neighbor_corr),
            "belief_entropy": _average_belief_entropy(
                [result.beliefs for result in grid_result.cell_results.values()]
            ),
        }
        step_rows.append(row)
        diagnostics.append(
            {
                "scenario": "h3",
                "timestep": timestep,
                "spatial_consistency": belief_update.spatial_consistency,
                "aggregate_free_energy": grid_result.aggregate_free_energy,
                "h3_resolution": config.h3_resolution,
                "nested_h3": nested_enabled,
            }
        )
        if nested_enabled:
            diagnostics[-1].update(
                {
                    "level_summaries": belief_update.level_summaries,
                    "resolutions": belief_update.metadata.get("resolutions", []),
                    "cross_level_coherence": belief_update.spatial_consistency.metadata.get(
                        "cross_level_coherence", 0.0
                    ),
                }
            )
        pymdp_records.extend(
            _pymdp_records_from_grid_result(config, timestep, grid_result)
        )
        last_cell_results = grid_result.cell_results
        analyzer.record_step(
            beliefs=(
                np.mean(list(belief_update.fine_beliefs.values()), axis=0)
                if nested_enabled and belief_update.fine_beliefs
                else belief_update.average
            ),
            observations=np.mean(list(vector_obs.values()), axis=0),
            actions="h3_grid_inference",
            policies={
                "selected": "h3_grid_inference",
                "expected_free_energy": row["expected_free_energy"],
            },
            free_energy=row["free_energy"],
            metrics=row,
            timestamp=float(timestep),
        )

    _finalize_analyzer(analyzer)
    write_csv(config.output_dir / "data" / "step_metrics.csv", step_rows)
    cell_metrics = _cell_metrics_from_results(cells, last_cell_results)
    _write_geospatial_cell_outputs(config, cells, cell_metrics)
    write_json(config.output_dir / "data" / "h3_diagnostics.json", diagnostics)
    _write_pymdp_h3_outputs(config, pymdp_records)
    research_statistics = _write_spatial_trace_outputs(config, trace_fragments)
    if nested_enabled and diagnostics:
        _write_nested_h3_outputs(config, belief_update, diagnostics)
    summary = _summary_metrics(step_rows)
    summary.update(
        {
            "cell_count": len(cells),
            "edge_count": step_rows[-1]["edge_count"],
            "coherence": step_rows[-1]["coherence"],
            "neighbor_correlation": step_rows[-1]["neighbor_correlation"],
            "mean_belief_entropy": float(
                np.mean([row["belief_entropy"] for row in step_rows])
            ),
        }
    )
    if nested_enabled:
        validation = getattr(model, "nested_h3_hierarchy", {}).get("validation", {})
        summary.update(
            {
                "nested_h3": True,
                "nested_resolutions": list(model.nested_h3_resolutions),
                "nested_parent_count": validation.get("parent_count", 0),
                "nested_child_count": validation.get("child_count", 0),
                "nested_orphan_count": validation.get("orphan_count", 0),
                "nested_cross_level_coherence": belief_update.spatial_consistency.metadata.get(
                    "cross_level_coherence", 0.0
                ),
                "nested_aggregate_free_energy": belief_update.aggregate_free_energy,
            }
        )
    summary.update(_pymdp_summary_metrics(pymdp_records))
    summary.update(_spatial_research_summary_metrics(research_statistics))
    write_json(config.output_dir / "analysis" / "run_summary.json", summary)
    if config.visualizations:
        _write_geospatial_visualizations(
            config,
            step_rows,
            cell_metrics,
            trace_fragments,
            research_statistics,
        )
    return summary


def _run_spatial_scenario(config: RunConfig) -> dict[str, Any]:
    """Run a spatial agent over H3 cells and persist its inference artifacts."""
    assert config.output_dir is not None
    cells = h3_cells_for_config(
        resolution=config.h3_resolution,
        ring_size=config.h3_ring_size,
        cells=config.h3_cells,
    )
    nested_enabled = bool(config.parameters.get("nested_h3"))
    agent = SpatialActiveInferenceAgent(
        h3_resolution=config.h3_resolution,
        initial_cells=cells,
        state_dim=4,
        obs_dim=4,
        diffusion_rate=float(config.parameters.get("diffusion_rate", 0.15)),
        enable_logging=False,
    )
    if nested_enabled:
        agent.enable_nested_h3_spatial(
            _nested_h3_resolutions(config),
            cells=cells,
            top_down_weight=float(config.parameters.get("top_down_weight", 0.15)),
        )
        cells = list(agent.cells)
    if config.parameters.get("research_profile"):
        apply_spatial_agent_research_profile(
            agent,
            action_count=int(config.parameters.get("research_action_count", 4)),
        )
    analyzer = ActiveInferenceAnalyzer(str(config.output_dir))

    step_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    pymdp_records: list[dict[str, Any]] = []
    trace_fragments: list[Any] = []
    previous_trace_beliefs: dict[str, Any] = {}
    last_cell_results: dict[str, Any] = {}
    for timestep in range(config.timesteps):
        env_obs = generate_realistic_environmental_observations(
            cells,
            timestep=float(timestep),
            spatial_seed=config.seed,
        )
        vector_obs = {
            cell: observation_dict_to_vector(obs) for cell, obs in env_obs.items()
        }
        grid_result: Any
        if nested_enabled:
            grid_result_nested = cast(
                NestedH3GridInferenceResult,
                agent.step_nested(
                    vector_obs,
                    return_result=True,
                    top_down_weight=float(
                        config.parameters.get("top_down_weight", 0.15)
                    ),
                ),
            )
            grid_result = grid_result_nested
            nested_update = grid_result_nested.nested_belief_update
        else:
            grid_result = agent.step(vector_obs, return_result=True)
            nested_update = None
        if nested_enabled:
            trace = agent.trace_nested_step(
                vector_obs,
                grid_result=grid_result_nested,
                timestep=timestep,
                previous_beliefs=previous_trace_beliefs,
                top_down_weight=float(config.parameters.get("top_down_weight", 0.15)),
            )
        else:
            trace = agent.trace_step(
                vector_obs,
                grid_result=grid_result,
                timestep=timestep,
                previous_beliefs=previous_trace_beliefs,
            )
        trace.metadata["observations_by_cell"] = {
            cell: _finite_vector(observation)
            for cell, observation in vector_obs.items()
        }
        trace_fragments.append(trace)
        previous_trace_beliefs = {
            item.cell: item.belief for item in trace.cell_diagnostics
        }
        expected_free_energy = float(
            np.nanmean(
                [
                    result.expected_free_energy
                    for result in grid_result.cell_results.values()
                    if result.expected_free_energy is not None
                ]
                or [0.0]
            )
        )
        row = {
            "timestep": timestep,
            "scenario": "spatial",
            "free_energy": float(grid_result.aggregate_free_energy),
            "expected_free_energy": expected_free_energy,
            "cell_count": len(cells),
            "edge_count": grid_result.spatial_consistency.edge_count,
            "coherence": float(grid_result.spatial_consistency.global_coherence),
            "neighbor_correlation": float(
                grid_result.spatial_consistency.neighbor_correlations
            ),
            "belief_entropy": _average_belief_entropy(
                [result.beliefs for result in grid_result.cell_results.values()]
            ),
        }
        step_rows.append(row)
        diagnostics.append(
            {
                "scenario": "spatial",
                "timestep": timestep,
                "spatial_consistency": grid_result.spatial_consistency,
                "aggregate_free_energy": grid_result.aggregate_free_energy,
                "h3_resolution": config.h3_resolution,
                "nested_h3": nested_enabled,
            }
        )
        if nested_enabled and nested_update is not None:
            diagnostics[-1].update(
                {
                    "level_summaries": nested_update.level_summaries,
                    "resolutions": nested_update.metadata.get("resolutions", []),
                    "cross_level_coherence": nested_update.spatial_consistency.metadata.get(
                        "cross_level_coherence", 0.0
                    ),
                }
            )
        pymdp_records.extend(
            _pymdp_records_from_grid_result(config, timestep, grid_result)
        )
        last_cell_results = grid_result.cell_results
        analyzer.record_step(
            beliefs=np.mean(agent.beliefs, axis=0),
            observations=np.mean(list(vector_obs.values()), axis=0),
            actions=grid_result.metadata.get("selected_action", {}),
            policies={
                "selected": grid_result.metadata.get("selected_action", {}),
                "expected_free_energy": expected_free_energy,
            },
            free_energy=row["free_energy"],
            metrics=row,
            timestamp=float(timestep),
        )

    _finalize_analyzer(analyzer)
    write_csv(config.output_dir / "data" / "step_metrics.csv", step_rows)
    cell_metrics = _cell_metrics_from_results(cells, last_cell_results)
    _write_geospatial_cell_outputs(config, cells, cell_metrics)
    write_json(config.output_dir / "data" / "h3_diagnostics.json", diagnostics)
    _write_pymdp_h3_outputs(config, pymdp_records)
    research_statistics = _write_spatial_trace_outputs(config, trace_fragments)
    if nested_enabled and nested_update is not None:
        _write_nested_h3_outputs(config, nested_update, diagnostics)
    summary = _summary_metrics(step_rows)
    scoring = agent.score_spatial_information_gain()
    summary.update(
        {
            "cell_count": len(cells),
            "edge_count": step_rows[-1]["edge_count"],
            "coherence": step_rows[-1]["coherence"],
            "neighbor_correlation": step_rows[-1]["neighbor_correlation"],
            "mean_belief_entropy": float(
                np.mean([row["belief_entropy"] for row in step_rows])
            ),
            "mean_information_gain": float(scoring["mean_score"]),
            "best_information_gain_cells": list(scoring["best_cells"][:5]),
            "uncertain_cell_fraction": float(scoring["uncertain_cell_fraction"]),
        }
    )
    if nested_enabled and nested_update is not None:
        validation = getattr(agent, "nested_h3_hierarchy", {}).get("validation", {})
        summary.update(
            {
                "nested_h3": True,
                "nested_resolutions": list(agent.nested_h3_resolutions),
                "nested_parent_count": validation.get("parent_count", 0),
                "nested_child_count": validation.get("child_count", 0),
                "nested_orphan_count": validation.get("orphan_count", 0),
                "nested_cross_level_coherence": nested_update.spatial_consistency.metadata.get(
                    "cross_level_coherence", 0.0
                ),
                "nested_aggregate_free_energy": nested_update.aggregate_free_energy,
            }
        )
    summary.update(_pymdp_summary_metrics(pymdp_records))
    summary.update(_spatial_research_summary_metrics(research_statistics))
    write_json(config.output_dir / "analysis" / "run_summary.json", summary)
    if config.visualizations:
        _write_geospatial_visualizations(
            config,
            step_rows,
            cell_metrics,
            trace_fragments,
            research_statistics,
        )
    return summary


def _nested_h3_resolutions(config: RunConfig) -> list[int]:
    """Return ordered nested H3 resolutions for a geospatial run."""
    explicit = config.parameters.get("nested_h3_resolutions")
    if explicit:
        resolutions = [int(value) for value in explicit]
    else:
        finest = int(config.h3_resolution)
        resolutions = [max(0, finest - 2), max(0, finest - 1), finest]
    resolutions = sorted(dict.fromkeys(resolutions))
    if len(resolutions) < 2:
        raise ValueError("nested_h3 requires at least two H3 resolutions")
    if resolutions[-1] != int(config.h3_resolution):
        raise ValueError("The finest nested H3 resolution must equal h3_resolution")
    return resolutions


def _write_spatial_trace_outputs(
    config: RunConfig, traces: list[Any]
) -> dict[str, Any]:
    """Write spatial trace tables and lattice data, returning research statistics."""
    assert config.output_dir is not None
    if not traces:
        raise ValueError("Spatial inference traces are required for geospatial runs")

    cell_rows = _trace_cell_rows(traces)
    edge_rows = [
        _finite_row(edge.to_dict())
        for trace in traces
        for edge in trace.edge_diagnostics
    ]
    level_rows = [
        _finite_row(level.to_dict())
        for trace in traces
        for level in trace.level_diagnostics
    ]
    nested_enabled = any(
        bool(getattr(trace, "hierarchy_metadata", {}).get("nested_h3"))
        for trace in traces
    )
    parent_child_rows = _nested_parent_child_rows(cell_rows) if nested_enabled else []
    research_statistics = build_spatial_research_statistics(
        cell_rows,
        edge_rows,
        level_rows,
        parent_child_rows,
    )
    animation_payload = _build_h3_lattice_animation_payload(
        config,
        traces,
        cell_rows,
        edge_rows,
        level_rows,
        parent_child_rows,
        research_statistics,
    )
    trace_payload = {
        "schema_version": "geo-infer-act-spatial-inference-trace/v1",
        "scenario": config.scenario,
        "timesteps": sorted(
            {int(timestep) for trace in traces for timestep in trace.timesteps}
        ),
        "cell_diagnostics": cell_rows,
        "edge_diagnostics": edge_rows,
        "level_diagnostics": level_rows,
        "hierarchy_metadata": traces[-1].hierarchy_metadata,
        "backend_metadata": traces[-1].backend_metadata,
        "research_statistics": research_statistics,
    }
    write_json(
        config.output_dir / "data" / "spatial_inference_trace.json", trace_payload
    )
    write_csv(config.output_dir / "data" / "h3_cell_diagnostics.csv", cell_rows)
    write_csv(config.output_dir / "data" / "h3_edge_diagnostics.csv", edge_rows)
    write_json(
        config.output_dir / "data" / "h3_lattice_animation.json",
        animation_payload,
    )
    write_json(
        config.output_dir / "data" / "spatial_research_statistics.json",
        research_statistics,
    )

    if nested_enabled:
        write_csv(
            config.output_dir / "data" / "nested_h3_cell_diagnostics.csv",
            cell_rows,
        )
        write_csv(
            config.output_dir / "data" / "nested_h3_parent_child_diagnostics.csv",
            parent_child_rows,
        )
        write_csv(
            config.output_dir / "data" / "nested_h3_level_diagnostics.csv",
            level_rows,
        )
    return research_statistics


def _trace_cell_rows(traces: list[Any]) -> list[dict[str, Any]]:
    """Flatten trace cell diagnostics and add H3 centroid coordinates."""
    adapter = get_h3_adapter()
    rows: list[dict[str, Any]] = []
    for trace in traces:
        observations_by_cell = {
            str(cell): _finite_vector(observation)
            for cell, observation in getattr(trace, "metadata", {})
            .get("observations_by_cell", {})
            .items()
        }
        for item in trace.cell_diagnostics:
            row = item.to_dict()
            cell = row["cell"]
            if cell in observations_by_cell:
                row["observation"] = observations_by_cell[cell]
            if adapter.is_valid_cell(cell):
                lat, lng = adapter.cell_to_latlng(cell)
                row["lat"] = float(lat)
                row["lng"] = float(lng)
            else:
                row["lat"] = 0.0
                row["lng"] = 0.0
            rows.append(_finite_row(row))
    return rows


def _nested_parent_child_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build parent-child residual rows from nested trace cell diagnostics."""
    adapter = get_h3_adapter()
    parent_rows = {(row["cell"], int(row["timestep"])): row for row in cell_rows}
    rows: list[dict[str, Any]] = []
    for row in cell_rows:
        parent = row.get("parent_cell")
        if not parent:
            continue
        parent_row = parent_rows.get((parent, int(row["timestep"])), {})
        consistency = float(row.get("cross_level_consistency", 0.0) or 0.0)
        child_lat, child_lng = adapter.cell_to_latlng(row["cell"])
        parent_lat, parent_lng = adapter.cell_to_latlng(parent)
        rows.append(
            _finite_row(
                {
                    "timestep": int(row["timestep"]),
                    "parent": parent,
                    "child": row["cell"],
                    "parent_resolution": adapter.get_resolution(parent),
                    "child_resolution": adapter.get_resolution(row["cell"]),
                    "parent_lat": float(parent_lat),
                    "parent_lng": float(parent_lng),
                    "child_lat": float(child_lat),
                    "child_lng": float(child_lng),
                    "parent_entropy": float(parent_row.get("entropy", 0.0) or 0.0),
                    "child_entropy": float(row.get("entropy", 0.0) or 0.0),
                    "cross_level_consistency": consistency,
                    "cross_level_residual": float(1.0 - consistency),
                }
            )
        )
    return rows


def _build_h3_lattice_animation_payload(
    config: RunConfig,
    traces: list[Any],
    cell_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    level_rows: list[dict[str, Any]],
    parent_child_rows: list[dict[str, Any]],
    research_statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a JSON-safe animated H3 lattice payload from trace diagnostics."""
    adapter = get_h3_adapter()
    hierarchy_metadata = traces[-1].hierarchy_metadata if traces else {}
    backend_metadata = traces[-1].backend_metadata if traces else {}
    timesteps = sorted({int(row["timestep"]) for row in cell_rows})
    rows_by_timestep: dict[int, list[dict[str, Any]]] = {step: [] for step in timesteps}
    for row in cell_rows:
        rows_by_timestep.setdefault(int(row["timestep"]), []).append(row)

    children_by_parent: dict[str, list[str]] = {}
    parent_by_child: dict[str, str] = {}
    for row in parent_child_rows:
        parent = str(row["parent"])
        child = str(row["child"])
        children_by_parent.setdefault(parent, []).append(child)
        parent_by_child[child] = parent
    children_by_parent = {
        parent: sorted(set(children)) for parent, children in children_by_parent.items()
    }

    cells = []
    for cell in sorted({str(row["cell"]) for row in cell_rows}):
        if not adapter.is_valid_cell(cell):
            raise ValueError(f"Invalid H3 cell in lattice animation payload: {cell}")
        lat, lng = adapter.cell_to_latlng(cell)
        boundary = adapter.cell_to_boundary(cell)
        ring = [
            [float(boundary_lng), float(boundary_lat)]
            for boundary_lat, boundary_lng in boundary
        ]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        cell_rows_for_cell = [row for row in cell_rows if str(row["cell"]) == cell]
        aggregate_parent = any(
            bool(row.get("aggregate_parent_cell", False)) for row in cell_rows_for_cell
        )
        cells.append(
            {
                "cell": cell,
                "resolution": adapter.get_resolution(cell),
                "lat": float(lat),
                "lng": float(lng),
                "centroid": [float(lng), float(lat)],
                "polygon": ring,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "parent_cell": parent_by_child.get(cell),
                "children": children_by_parent.get(cell, []),
                "is_aggregate_parent": aggregate_parent,
            }
        )

    edge_rows_by_timestep: dict[int, list[dict[str, Any]]] = {}
    for row in edge_rows:
        edge_rows_by_timestep.setdefault(int(row["timestep"]), []).append(row)
    parent_child_by_timestep: dict[int, list[dict[str, Any]]] = {}
    for row in parent_child_rows:
        parent_child_by_timestep.setdefault(int(row["timestep"]), []).append(row)
    level_rows_by_timestep: dict[int, list[dict[str, Any]]] = {}
    for row in level_rows:
        level_rows_by_timestep.setdefault(int(row["timestep"]), []).append(row)

    frames = []
    for timestep in timesteps:
        rows = sorted(rows_by_timestep.get(timestep, []), key=lambda item: item["cell"])
        raw_states = [_lattice_cell_state(row) for row in rows]
        states_by_cell = {state["cell"]: state for state in raw_states}
        for state in raw_states:
            if not state["is_aggregate_parent"]:
                continue
            child_states = [
                states_by_cell[child]
                for child in children_by_parent.get(state["cell"], [])
                if child in states_by_cell
            ]
            _fill_parent_lattice_state(state, child_states)
        frame_edges = [
            _lattice_edge_state(row, states_by_cell)
            for row in sorted(
                edge_rows_by_timestep.get(timestep, []),
                key=lambda item: (item["source"], item["target"]),
            )
        ]
        parent_child_links = [
            _lattice_parent_child_state(row)
            for row in sorted(
                parent_child_by_timestep.get(timestep, []),
                key=lambda item: (item["parent"], item["child"]),
            )
        ]
        frames.append(
            {
                "timestep": int(timestep),
                "cells": raw_states,
                # Mapping form is the stable validation/API contract; retain
                # the list above for the existing browser visualization.
                "cell_metrics": states_by_cell,
                "edges": frame_edges,
                "parent_child_links": parent_child_links,
                "level_summaries": [
                    _finite_row(row)
                    for row in sorted(
                        level_rows_by_timestep.get(timestep, []),
                        key=lambda item: int(item["resolution"]),
                    )
                ],
            }
        )

    return {
        "schema_version": "geo-infer-act-h3-lattice-animation/v1",
        "scenario": config.scenario,
        "timesteps": timesteps,
        "cell_count": len(cells),
        "frame_count": len(frames),
        "nested_h3": bool(hierarchy_metadata.get("nested_h3")),
        "cells": cells,
        "frames": frames,
        "hierarchy_metadata": hierarchy_metadata,
        "backend_metadata": backend_metadata,
        "research_statistics": research_statistics,
        "encodings": {
            "hex_fill_default": "belief_argmax_probability",
            "hex_stroke_default": "belief_argmax_index",
            "observation_arrows": "dominant_observation_index and observation_strength",
            "action_arrows": "selected_action_index and selected_action_probability",
            "edge_arrows": "belief_flux_divergence and edge belief_distance",
            "nested_links": "cross_level_residual",
        },
    }


def _lattice_cell_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one finite per-frame cell state for the lattice animation."""
    belief = _normalized_finite_vector(row.get("belief", []))
    observation = _normalized_finite_vector(row.get("observation", []))
    posterior = _normalized_finite_vector(row.get("action_posterior", []))
    negative_efe = _finite_vector(row.get("negative_expected_free_energy", []))
    belief_index, belief_probability = _dominant_index_value(belief)
    observation_index, observation_strength = _dominant_index_value(observation)
    selected_action_index = int(row.get("selected_action_index", 0) or 0)
    if posterior:
        selected_action_index %= len(posterior)
    selected_action_probability = (
        float(posterior[selected_action_index])
        if posterior
        else float(row.get("selected_action_probability", 0.0) or 0.0)
    )
    return {
        "cell": str(row["cell"]),
        "timestep": int(row["timestep"]),
        "resolution": int(row["resolution"]),
        "parent_cell": row.get("parent_cell") or None,
        "is_aggregate_parent": bool(row.get("aggregate_parent_cell", False)),
        "belief": belief,
        "belief_argmax_index": belief_index,
        "belief_argmax_probability": belief_probability,
        "entropy": _finite_float(row.get("entropy", 0.0)),
        "free_energy": _finite_float(row.get("free_energy", 0.0)),
        "expected_free_energy": _finite_float(row.get("expected_free_energy", 0.0)),
        "policy_entropy": _finite_float(row.get("policy_entropy", 0.0)),
        "action_posterior": posterior,
        "negative_expected_free_energy": negative_efe,
        "selected_action": row.get("selected_action"),
        "selected_action_index": selected_action_index,
        "selected_action_probability": _finite_float(selected_action_probability),
        "selected_negative_expected_free_energy": _finite_float(
            row.get("selected_negative_expected_free_energy", 0.0)
        ),
        "observation": observation,
        "dominant_observation_index": observation_index,
        "observation_strength": observation_strength,
        "neighbor_count": int(row.get("neighbor_count", 0) or 0),
        "local_coherence": _finite_float(row.get("local_coherence", 0.0)),
        "posterior_delta": _finite_float(row.get("posterior_delta", 0.0)),
        "belief_flux_in": _finite_float(row.get("belief_flux_in", 0.0)),
        "belief_flux_out": _finite_float(row.get("belief_flux_out", 0.0)),
        "belief_flux_divergence": _finite_float(row.get("belief_flux_divergence", 0.0)),
        "cross_level_consistency": _finite_float(
            row.get("cross_level_consistency", 0.0)
        ),
        "pymdp_version": row.get("pymdp_version", ""),
        "h3_version": row.get("h3_version", ""),
        "backend": row.get("backend", ""),
    }


def _fill_parent_lattice_state(
    parent_state: dict[str, Any], child_states: list[dict[str, Any]]
) -> None:
    """Fill parent observation and action fields from child means."""
    if not child_states:
        return
    if not parent_state["observation"]:
        parent_state["observation"] = _mean_vector(
            [child["observation"] for child in child_states]
        )
        (
            parent_state["dominant_observation_index"],
            parent_state["observation_strength"],
        ) = _dominant_index_value(parent_state["observation"])
    if not parent_state["action_posterior"]:
        parent_state["action_posterior"] = _mean_vector(
            [child["action_posterior"] for child in child_states]
        )
        (
            parent_state["selected_action_index"],
            parent_state["selected_action_probability"],
        ) = _dominant_index_value(parent_state["action_posterior"])


def _lattice_edge_state(
    row: Mapping[str, Any], states_by_cell: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return one directed edge state with finite flux direction and weight."""
    source = str(row["source"])
    target = str(row["target"])
    source_flux = _finite_float(
        states_by_cell.get(source, {}).get("belief_flux_divergence", 0.0)
    )
    target_flux = _finite_float(
        states_by_cell.get(target, {}).get("belief_flux_divergence", 0.0)
    )
    if source_flux >= target_flux:
        flux_source, flux_target = source, target
    else:
        flux_source, flux_target = target, source
    belief_distance = _finite_float(row.get("belief_distance", 0.0))
    flux_delta = abs(source_flux - target_flux)
    return {
        "source": source,
        "target": target,
        "flux_source": flux_source,
        "flux_target": flux_target,
        "timestep": int(row["timestep"]),
        "resolution": int(row["resolution"]),
        "belief_distance": belief_distance,
        "coherence": _finite_float(row.get("coherence", 0.0)),
        "source_entropy": _finite_float(row.get("source_entropy", 0.0)),
        "target_entropy": _finite_float(row.get("target_entropy", 0.0)),
        "flux_delta": _finite_float(flux_delta),
        "weight": _finite_float(max(belief_distance, flux_delta)),
    }


def _lattice_parent_child_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one nested parent-child residual link for the lattice animation."""
    return {
        "timestep": int(row["timestep"]),
        "parent": str(row["parent"]),
        "child": str(row["child"]),
        "parent_resolution": int(row["parent_resolution"]),
        "child_resolution": int(row["child_resolution"]),
        "cross_level_consistency": _finite_float(
            row.get("cross_level_consistency", 0.0)
        ),
        "cross_level_residual": _finite_float(row.get("cross_level_residual", 0.0)),
    }


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return a normalized finite mean vector for same-length vectors."""
    usable = [np.asarray(vector, dtype=float) for vector in vectors if vector]
    if not usable:
        return []
    max_len = max(vector.size for vector in usable)
    padded = []
    for vector in usable:
        if vector.size < max_len:
            vector = np.pad(vector, (0, max_len - vector.size))
        padded.append(vector)
    return _normalized_finite_vector(np.mean(padded, axis=0))


def _dominant_index_value(vector: list[float]) -> tuple[int, float]:
    """Return the dominant index and value for a finite vector."""
    if not vector:
        return 0, 0.0
    array = np.asarray(vector, dtype=float)
    index = int(np.argmax(array))
    return index, _finite_float(array[index])


def _finite_float(value: Any) -> float:
    """Return a finite float with zero as the non-finite fallback."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _finite_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with non-finite numeric values replaced by zero."""
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            cleaned[key] = 0.0
        elif isinstance(value, list):
            cleaned[key] = [
                0.0 if isinstance(item, float) and not math.isfinite(item) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _write_nested_h3_outputs(
    config: RunConfig,
    nested_update: Any,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Persist H3 hierarchy diagnostics and optionally render hierarchy maps."""
    assert config.output_dir is not None
    adapter = get_h3_adapter()
    rows: list[dict[str, Any]] = []
    for parent, children in nested_update.parent_child_map.items():
        parent_resolution = adapter.get_resolution(parent)
        for child in children:
            rows.append(
                {
                    "parent": parent,
                    "child": child,
                    "parent_resolution": parent_resolution,
                    "child_resolution": adapter.get_resolution(child),
                }
            )
    write_csv(config.output_dir / "data" / "h3_hierarchy.csv", rows)
    write_json(
        config.output_dir / "data" / "nested_h3_diagnostics.json",
        diagnostics,
    )
    if config.visualizations:
        _write_nested_h3_level_map(config, nested_update)
        _write_nested_h3_hierarchy_map(config, nested_update)


def _pymdp_records_from_grid_result(
    config: RunConfig, timestep: int, grid_result: Any
) -> list[dict[str, Any]]:
    """Flatten per-cell pymdp metadata from a typed H3 grid result."""
    adapter = get_h3_adapter()
    records: list[dict[str, Any]] = []
    for cell, cell_result in getattr(grid_result, "cell_results", {}).items():
        metadata = getattr(cell_result, "metadata", {}) or {}
        pymdp = metadata.get("pymdp") or {}
        if not pymdp:
            continue
        posterior = [float(value) for value in pymdp.get("action_posterior", [])]
        neg_efe = [
            float(value) for value in pymdp.get("negative_expected_free_energy", [])
        ]
        selected = int(pymdp.get("selected_action_index", 0))
        selected = selected % max(1, len(posterior) or len(neg_efe) or 1)
        lat, lng = adapter.cell_to_latlng(cell)
        beliefs = _belief_vector(cell_result.beliefs)
        record: dict[str, Any] = {
            "scenario": config.scenario,
            "timestep": int(timestep),
            "cell": str(cell),
            "lat": float(lat),
            "lng": float(lng),
            "pymdp_version": pymdp.get("pymdp_version"),
            "h3_version": pymdp.get("h3_version"),
            "h3_c_version": pymdp.get("h3_c_version"),
            "selected_action_index": selected,
            "selected_action_probability": (
                float(posterior[selected]) if posterior else 0.0
            ),
            "selected_negative_expected_free_energy": (
                float(neg_efe[selected]) if neg_efe else 0.0
            ),
            "free_energy": float(pymdp.get("free_energy", cell_result.free_energy)),
            "belief_entropy": float(
                -np.sum(beliefs * np.log(beliefs + 1e-12)) if beliefs.size else 0.0
            ),
        }
        for index, value in enumerate(posterior):
            record[f"policy_posterior_{index}"] = float(value)
        for index, value in enumerate(neg_efe):
            record[f"negative_expected_free_energy_{index}"] = float(value)
        records.append(record)
    return records


def _pymdp_summary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact run-summary metrics for pymdp H3 diagnostics."""
    if not records:
        return {
            "pymdp_backend": "missing",
            "pymdp_cell_update_count": 0,
        }
    return {
        "pymdp_backend": "inferactively-pymdp",
        "pymdp_version": records[0].get("pymdp_version"),
        "h3_version": records[0].get("h3_version"),
        "pymdp_cell_update_count": len(records),
        "mean_pymdp_free_energy": float(
            np.mean([float(row["free_energy"]) for row in records])
        ),
        "mean_selected_action_probability": float(
            np.mean([float(row["selected_action_probability"]) for row in records])
        ),
        "mean_selected_negative_expected_free_energy": float(
            np.mean(
                [
                    float(row["selected_negative_expected_free_energy"])
                    for row in records
                ]
            )
        ),
    }


def _write_pymdp_h3_outputs(config: RunConfig, records: list[dict[str, Any]]) -> None:
    """Write required pymdp H3 diagnostics and optional policy visualizations."""
    assert config.output_dir is not None
    if not records:
        raise ValueError("pymdp H3 diagnostics are required for geospatial runs")
    write_json(config.output_dir / "data" / "pymdp_h3_diagnostics.json", records)
    write_csv(config.output_dir / "data" / "pymdp_policy_posteriors.csv", records)
    if config.visualizations:
        _write_pymdp_policy_free_energy_html(config, records)


def _plotly_or_table(
    fig_builder: Callable[[], Any],
    rows: Iterable[Mapping[str, Any]],
    headers: Sequence[str],
    row_cells: Callable[[Mapping[str, Any]], Sequence[str]],
    title: str,
    *,
    page_title: str | None = None,
    caption: str | None = None,
) -> str:
    """Render a Plotly figure as HTML, degrading to a static table on failure.

    The graceful-degradation contract for scenario reports: any failure while
    importing, building, or serializing the Plotly figure is logged at warning
    level (module name plus exception) and downgraded to a static HTML table so
    report generation continues instead of aborting the run.
    """
    try:
        fig = fig_builder()
        return fig.to_html(include_plotlyjs="cdn", full_html=True)
    except Exception:  # noqa: BLE001 - deliberate degradation path for reports
        logger.warning(
            "Plotly figure %r failed to render; downgrading to a static HTML table",
            title,
            exc_info=True,
        )
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    caption_html = f"<p>{caption}</p>" if caption else ""
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row_cells(row)) + "</tr>"
        for row in rows
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{page_title or title}</title></head><body>"
        f"<h1>{title}</h1>{caption_html}"
        f"<table><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_rows}</tbody></table></body></html>"
    )


def _write_pymdp_policy_free_energy_html(
    config: RunConfig, records: list[dict[str, Any]]
) -> Path:
    """Write an HTML analysis of pymdp policy posterior and free energy."""
    timesteps = sorted({int(row["timestep"]) for row in records})
    aggregate_rows = []
    for timestep in timesteps:
        rows = [row for row in records if int(row["timestep"]) == timestep]
        aggregate_rows.append(
            {
                "timestep": timestep,
                "mean_free_energy": float(
                    np.mean([float(row["free_energy"]) for row in rows])
                ),
                "mean_selected_action_probability": float(
                    np.mean([float(row["selected_action_probability"]) for row in rows])
                ),
                "mean_selected_negative_expected_free_energy": float(
                    np.mean(
                        [
                            float(row["selected_negative_expected_free_energy"])
                            for row in rows
                        ]
                    )
                ),
            }
        )

    def _build_pymdp_policy_figure() -> Any:
        import plotly.graph_objects as go  # noqa: PLC0415

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[row["timestep"] for row in aggregate_rows],
                y=[row["mean_free_energy"] for row in aggregate_rows],
                mode="lines+markers",
                name="Mean VFE",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["timestep"] for row in aggregate_rows],
                y=[
                    row["mean_selected_negative_expected_free_energy"]
                    for row in aggregate_rows
                ],
                mode="lines+markers",
                name="Mean selected negative EFE",
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[row["timestep"] for row in aggregate_rows],
                y=[row["mean_selected_action_probability"] for row in aggregate_rows],
                name="Mean selected action probability",
                yaxis="y3",
                opacity=0.35,
            )
        )
        fig.update_layout(
            title=f"{config.scenario.upper()} pymdp 1.0.3 Policy and Free Energy",
            xaxis_title="Timestep",
            yaxis={"title": "Mean VFE"},
            yaxis2={
                "title": "Negative EFE",
                "overlaying": "y",
                "side": "right",
            },
            yaxis3={
                "title": "Action probability",
                "overlaying": "y",
                "side": "right",
                "anchor": "free",
                "position": 0.95,
                "range": [0.0, 1.0],
            },
            margin={"l": 60, "r": 90, "t": 80, "b": 70},
        )
        return fig

    html = _plotly_or_table(
        _build_pymdp_policy_figure,
        aggregate_rows,
        [
            "Timestep",
            "Mean VFE",
            "Mean selected negative EFE",
            "Mean selected action probability",
        ],
        lambda row: (
            row["timestep"],
            f"{row['mean_free_energy']:.6f}",
            f"{row['mean_selected_negative_expected_free_energy']:.6f}",
            f"{row['mean_selected_action_probability']:.6f}",
        ),
        "pymdp H3 Policy and Free Energy",
    )
    return write_html_figure_artifact(
        config,
        "visualizations/pymdp_policy_free_energy.html",
        html,
        title="pymdp 1.0.3 H3 Policy and Free Energy",
        description="pymdp-derived H3 policy posterior, negative expected free energy, and variational free-energy diagnostics.",
        alt_text="Interactive chart of pymdp policy posterior and free-energy diagnostics by timestep.",
        plotted_metrics=[
            "mean_free_energy",
            "mean_selected_negative_expected_free_energy",
            "mean_selected_action_probability",
        ],
        data_sources=[
            "data/pymdp_h3_diagnostics.json",
            "data/pymdp_policy_posteriors.csv",
        ],
        plotted_data=aggregate_rows,
    )


def _write_nested_h3_level_map(config: RunConfig, nested_update: Any) -> Path:
    """Write an HTML nested H3 level summary visualization."""
    level_rows = [summary.to_dict() for summary in nested_update.level_summaries]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['resolution']}</td>"
        f"<td>{row['cell_count']}</td>"
        f"<td>{row['edge_count']}</td>"
        f"<td>{float(row['mean_reference_kl_to_uniform']):.6f}</td>"
        f"<td>{float(row['mean_entropy']):.6f}</td>"
        f"<td>{float(row['coherence']):.6f}</td>"
        "</tr>"
        for row in level_rows
    )
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Nested H3 Level Diagnostics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Nested H3 Level Diagnostics</h1>
  <table>
    <thead>
      <tr>
        <th>Resolution</th>
        <th>Cells</th>
        <th>Edges</th>
        <th>Mean KL to Uniform</th>
        <th>Mean Entropy</th>
        <th>Coherence</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
    return write_html_figure_artifact(
        config,
        "visualizations/nested_h3_level_map.html",
        html,
        title="Nested H3 Level Diagnostics",
        description="Nested H3 per-resolution cell counts, edge counts, uniform-reference KL, entropy, and coherence.",
        alt_text="Table of nested H3 diagnostics by H3 resolution.",
        plotted_metrics=[
            "cell_count",
            "edge_count",
            "mean_reference_kl_to_uniform",
            "mean_entropy",
            "coherence",
        ],
        data_sources=["data/h3_hierarchy.csv", "data/nested_h3_diagnostics.json"],
        plotted_data=level_rows,
    )


def _write_nested_h3_hierarchy_map(config: RunConfig, nested_update: Any) -> Path:
    """Write an interactive nested H3 parent-child hierarchy map."""
    adapter = get_h3_adapter()
    rows: list[dict[str, Any]] = []
    for parent, children in nested_update.parent_child_map.items():
        parent_belief = nested_update.parent_beliefs.get(parent)
        parent_entropy = (
            float(-np.sum(parent_belief * np.log(parent_belief + 1e-12)))
            if parent_belief is not None
            else 0.0
        )
        parent_lat, parent_lng = adapter.cell_to_latlng(parent)
        for child in children:
            if child not in nested_update.fine_beliefs:
                continue
            child_belief = normalize_belief_vector(nested_update.fine_beliefs[child])
            child_lat, child_lng = adapter.cell_to_latlng(child)
            if parent_belief is not None:
                distance = float(
                    np.linalg.norm(
                        normalize_belief_vector(parent_belief) - child_belief
                    )
                )
                consistency = float(1.0 / (1.0 + distance))
            else:
                consistency = 0.0
            rows.append(
                {
                    "parent": parent,
                    "child": child,
                    "parent_resolution": adapter.get_resolution(parent),
                    "child_resolution": adapter.get_resolution(child),
                    "parent_lat": float(parent_lat),
                    "parent_lng": float(parent_lng),
                    "child_lat": float(child_lat),
                    "child_lng": float(child_lng),
                    "parent_entropy": parent_entropy,
                    "child_entropy": float(
                        -np.sum(child_belief * np.log(child_belief + 1e-12))
                    ),
                    "cross_level_consistency": consistency,
                    "cross_level_residual": float(1.0 - consistency),
                }
            )

    def _build_nested_h3_hierarchy_figure() -> Any:
        import plotly.graph_objects as go  # noqa: PLC0415

        fig = go.Figure()
        parent_cells = sorted({row["parent"] for row in rows})
        child_cells = sorted({row["child"] for row in rows})
        for cell in parent_cells:
            lats, lngs = _h3_boundary_trace(cell)
            fig.add_trace(
                go.Scattergeo(
                    lat=lats,
                    lon=lngs,
                    mode="lines",
                    line={"color": "#2b6cb0", "width": 2},
                    name="Parent cell",
                    hovertext=cell,
                    showlegend=cell == parent_cells[0],
                )
            )
        for cell in child_cells:
            lats, lngs = _h3_boundary_trace(cell)
            fig.add_trace(
                go.Scattergeo(
                    lat=lats,
                    lon=lngs,
                    mode="lines",
                    line={"color": "#38a169", "width": 1},
                    name="Child cell",
                    hovertext=cell,
                    showlegend=cell == child_cells[0],
                )
            )
        fig.add_trace(
            go.Scattergeo(
                lat=[row["child_lat"] for row in rows],
                lon=[row["child_lng"] for row in rows],
                mode="markers",
                marker={
                    "size": 9,
                    "color": [row["cross_level_residual"] for row in rows],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Residual"},
                },
                text=[
                    (
                        f"Child: {row['child']}<br>Parent: {row['parent']}<br>"
                        f"Consistency: {row['cross_level_consistency']:.4f}"
                    )
                    for row in rows
                ],
                hoverinfo="text",
                name="Child residual",
            )
        )
        fig.update_geos(fitbounds="locations", visible=True)
        fig.update_layout(
            title="Nested H3 Parent-Child Diagnostics",
            margin={"r": 0, "t": 70, "l": 0, "b": 35},
        )
        return fig

    html = _plotly_or_table(
        _build_nested_h3_hierarchy_figure,
        rows,
        ["Parent", "Child", "Consistency", "Residual"],
        lambda row: (
            row["parent"],
            row["child"],
            f"{row['cross_level_consistency']:.6f}",
            f"{row['cross_level_residual']:.6f}",
        ),
        "Nested H3 Parent-Child Diagnostics",
        page_title="Nested H3 Hierarchy Map",
    )

    return write_html_figure_artifact(
        config,
        "visualizations/nested_h3_hierarchy_map.html",
        html,
        title="Nested H3 Parent-Child Diagnostics",
        description="Interactive real-H3 parent and child boundary map with cross-level belief consistency residuals.",
        alt_text="Nested H3 hierarchy map showing parent and child cells with child markers colored by cross-level residual.",
        plotted_metrics=[
            "cross_level_consistency",
            "cross_level_residual",
            "parent_entropy",
            "child_entropy",
        ],
        data_sources=[
            "data/h3_hierarchy.csv",
            "data/nested_h3_parent_child_diagnostics.csv",
            "data/nested_h3_diagnostics.json",
        ],
        plotted_data=rows,
    )


def _h3_boundary_trace(cell: str) -> tuple[list[float], list[float]]:
    """Return closed latitude and longitude arrays for one H3 cell boundary."""
    adapter = get_h3_adapter()
    boundary = adapter.cell_to_boundary(cell)
    if boundary and boundary[0] != boundary[-1]:
        boundary = [*boundary, boundary[0]]
    lats = [float(lat) for lat, _lng in boundary]
    lngs = [float(lng) for _lat, lng in boundary]
    return lats, lngs


def _scenario_observation(
    scenario: str,
    timestep: int,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    phase = float(params["phase"])
    amplitude = float(params["amplitude"])
    base = np.asarray(params.get("preference", [0.25, 0.25, 0.25, 0.25]), dtype=float)
    seasonal = np.array(
        [
            math.sin(timestep + phase),
            math.cos((timestep * 0.7) + phase),
            math.sin((timestep * 0.4) + phase + 0.8),
            math.cos((timestep * 0.3) + phase + 1.2),
        ]
    )
    signal = base + amplitude * seasonal
    if scenario in {"verification", "debug"}:
        signal += rng.normal(0.0, 0.005, size=4)
    return cast(np.ndarray, normalize_belief_vector(signal))


def _belief_vector(beliefs: Any) -> np.ndarray:
    if isinstance(beliefs, dict) and "states" in beliefs:
        return cast(np.ndarray, normalize_belief_vector(beliefs["states"]))
    return cast(np.ndarray, normalize_belief_vector(beliefs))


def _finite_vector(value: Any) -> list[float]:
    """Return a one-dimensional finite numeric vector for JSON payloads."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    if array.size == 0:
        return []
    return [float(item) if math.isfinite(float(item)) else 0.0 for item in array]


def _normalized_finite_vector(value: Any) -> list[float]:
    """Return a normalized finite vector when a positive sum is available."""
    vector = np.asarray(_finite_vector(value), dtype=float)
    if vector.size == 0:
        return []
    total = float(np.sum(vector))
    if total > 0.0 and math.isfinite(total):
        vector = vector / total
    return [float(item) for item in vector]


def _finalize_analyzer(analyzer: ActiveInferenceAnalyzer) -> None:
    analyzer.export_full_history()
    analyzer.save_traces_to_csv()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            analyzer.analyze_perception_patterns()
            analyzer.analyze_action_selection_patterns()
            analyzer.analyze_free_energy_patterns()
            analyzer.generate_comprehensive_report()
    except Exception as exc:
        write_json(
            analyzer.output_dir / "analysis" / "analysis_warning.json",
            {"warning": str(exc)},
        )


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    free_energy = [float(row["free_energy"]) for row in rows]
    expected = [
        float(row["expected_free_energy"])
        for row in rows
        if row.get("expected_free_energy") is not None
    ]
    return {
        "timesteps": len(rows),
        "initial_free_energy": free_energy[0],
        "final_free_energy": free_energy[-1],
        "free_energy_change": free_energy[-1] - free_energy[0],
        "mean_free_energy": float(np.mean(free_energy)),
        "mean_expected_free_energy": float(np.mean(expected)) if expected else 0.0,
        "finite_metrics": bool(np.all(np.isfinite(free_energy))),
    }


def _spatial_research_summary_metrics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact run-summary metrics from spatial research statistics."""
    metrics = statistics.get("metric_summaries", {})
    policy = statistics.get("policy", {})
    graph = statistics.get("spatial_graph", {})
    nested = statistics.get("nested", {})
    non_degenerate = statistics.get("non_degenerate", {})
    return {
        "spatial_research_statistics_schema": statistics.get("schema_version"),
        "spatial_entropy_std": metrics.get("entropy", {}).get("std", 0.0),
        "spatial_policy_probability_std": metrics.get(
            "selected_action_probability", {}
        ).get("std", 0.0),
        "spatial_local_coherence_std": metrics.get("local_coherence", {}).get(
            "std", 0.0
        ),
        "spatial_belief_flux_divergence_std": metrics.get(
            "belief_flux_divergence", {}
        ).get("std", 0.0),
        "spatial_policy_switch_count": policy.get("switch_count", 0),
        "spatial_dominant_action_share": policy.get("dominant_action_share", 0.0),
        "spatial_moran_entropy_proxy": graph.get("moran_entropy_proxy", 0.0),
        "spatial_mean_edge_belief_distance": graph.get(
            "mean_edge_belief_distance", 0.0
        ),
        "spatial_mean_abs_flux_balance": graph.get("mean_abs_flux_balance", 0.0),
        "spatial_unique_selected_action_count": non_degenerate.get(
            "unique_selected_action_count", 0
        ),
        "nested_mean_parent_child_residual": nested.get(
            "mean_parent_child_residual", 0.0
        ),
        "nested_max_parent_child_residual": nested.get(
            "max_parent_child_residual", 0.0
        ),
        "nested_parent_aggregate_drift": nested.get("parent_aggregate_drift", 0.0),
    }


def _provenance_caption(config: RunConfig, metric_note: str) -> str:
    """Return compact provenance text for runner visualizations."""
    h3_part = ""
    if config.scenario in {"h3", "spatial"}:
        h3_part = f" | H3 r{config.h3_resolution}, ring {config.h3_ring_size}"
    return (
        f"Scenario: {config.scenario} | seed: {config.seed} | "
        f"timesteps: {config.timesteps}{h3_part} | {metric_note}"
    )


def _add_provenance_caption(
    fig: Any, config: RunConfig, metric_note: str, y: float = 0.005
) -> None:
    """Add visible run provenance to a figure without changing data semantics."""
    fig.text(
        0.01,
        y,
        _provenance_caption(config, metric_note),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4a4a4a",
    )


def _average_belief_entropy(beliefs: list[Any]) -> float:
    """Compute mean entropy for a collection of belief vectors."""
    if not beliefs:
        return 0.0
    entropies = []
    for belief in beliefs:
        vector = _belief_vector(belief)
        entropies.append(float(-np.sum(vector * np.log(vector + 1e-12))))
    return float(np.mean(entropies))


def _cell_metrics_from_results(
    cells: list[str], cell_results: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build per-cell metric rows from typed grid inference results."""
    adapter = get_h3_adapter()
    metrics = []
    for cell in cells:
        lat, lng = adapter.cell_to_latlng(cell)
        result = cell_results.get(cell)
        belief = (
            _belief_vector(result.beliefs) if result is not None else np.ones(4) / 4
        )
        pymdp = getattr(result, "metadata", {}).get("pymdp", {}) if result else {}
        posterior = [float(value) for value in pymdp.get("action_posterior", [])]
        neg_efe = [
            float(value) for value in pymdp.get("negative_expected_free_energy", [])
        ]
        selected = int(pymdp.get("selected_action_index", 0)) if pymdp else 0
        selected = selected % max(1, len(posterior) or len(neg_efe) or 1)
        metrics.append(
            {
                "cell": cell,
                "lat": float(lat),
                "lng": float(lng),
                "resolution": adapter.get_resolution(cell),
                "free_energy": float(result.free_energy) if result is not None else 0.0,
                "expected_free_energy": (
                    float(result.expected_free_energy)
                    if result is not None and result.expected_free_energy is not None
                    else 0.0
                ),
                "belief_entropy": float(-np.sum(belief * np.log(belief + 1e-12))),
                "action": str(result.action) if result is not None else "",
                "pymdp_version": pymdp.get("pymdp_version", ""),
                "h3_version": pymdp.get("h3_version", ""),
                "selected_action_probability": (
                    float(posterior[selected]) if posterior else 0.0
                ),
                "selected_negative_expected_free_energy": (
                    float(neg_efe[selected]) if neg_efe else 0.0
                ),
                "beliefs": belief.tolist(),
            }
        )
    return metrics


def _write_geospatial_cell_outputs(
    config: RunConfig, cells: list[str], cell_metrics: list[dict[str, Any]]
) -> None:
    """Write cell metrics and closed H3 polygon GeoJSON beneath the run directory."""
    assert config.output_dir is not None
    adapter = get_h3_adapter()
    write_csv(config.output_dir / "data" / "h3_cells.csv", cell_metrics)
    features = []
    metrics_by_cell = {row["cell"]: row for row in cell_metrics}
    for cell in cells:
        boundary = adapter.cell_to_boundary(cell)
        ring = [[float(lng), float(lat)] for lat, lng in boundary]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        row = metrics_by_cell[cell]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "h3_cell": cell,
                    "resolution": row["resolution"],
                    "lat": row["lat"],
                    "lng": row["lng"],
                    "free_energy": row["free_energy"],
                    "expected_free_energy": row["expected_free_energy"],
                    "belief_entropy": row["belief_entropy"],
                    "action": row["action"],
                    "pymdp_version": row.get("pymdp_version", ""),
                    "h3_version": row.get("h3_version", ""),
                    "selected_action_probability": row.get(
                        "selected_action_probability", 0.0
                    ),
                    "selected_negative_expected_free_energy": row.get(
                        "selected_negative_expected_free_energy", 0.0
                    ),
                },
            }
        )
    write_json(
        config.output_dir / "data" / "h3_cells.geojson",
        {"type": "FeatureCollection", "features": features},
    )


def _plot_vector_summary(config: RunConfig, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Save a free-energy and belief-entropy figure with provenance sidecars."""
    assert config.output_dir is not None
    timesteps = [int(row["timestep"]) for row in rows]
    free_energy = [float(row["free_energy"]) for row in rows]
    entropy = [float(row["belief_entropy"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].plot(timesteps, free_energy, marker="o")
    axes[0].set_title("Variational Free Energy")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Free-energy value")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(timesteps, entropy, marker="s", color="#2c7fb8")
    axes[1].set_title("Belief Entropy")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"Active Inference Scenario: {config.scenario}")
    _add_provenance_caption(
        fig, config, "Data: data/step_metrics.csv and data/full_history.json"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = save_matplotlib_figure_artifact(
        config,
        fig,
        "visualizations/scenario_summary.png",
        title=f"Active Inference Scenario: {config.scenario}",
        description="Summary plot of variational free energy and belief entropy over timesteps.",
        alt_text=(
            f"Two-panel {config.scenario} Active Inference plot showing "
            "free-energy values and belief entropy by timestep."
        ),
        plotted_metrics=["free_energy", "belief_entropy"],
        data_sources=["data/step_metrics.csv", "data/full_history.json"],
        plotted_data=[
            {
                "timestep": int(row["timestep"]),
                "free_energy": float(row["free_energy"]),
                "belief_entropy": float(row["belief_entropy"]),
            }
            for row in rows
        ],
    )
    plt.close(fig)
    return path


def _write_geospatial_visualizations(
    config: RunConfig,
    rows: Sequence[Mapping[str, Any]],
    cell_metrics: list[dict[str, Any]],
    traces: list[Any],
    research_statistics: Mapping[str, Any],
) -> None:
    """Create the full visualization set required for geospatial scenarios."""
    assert config.output_dir is not None
    _plot_h3_cell_metric_map(config, cell_metrics)
    _plot_free_energy_evolution(config, rows)
    _plot_belief_entropy_coherence(config, rows)
    _write_interactive_h3_map(config, cell_metrics)
    trace_rows = _trace_cell_rows(traces)
    edge_rows = [
        _finite_row(edge.to_dict())
        for trace in traces
        for edge in trace.edge_diagnostics
    ]
    parent_child_rows = _nested_parent_child_rows(trace_rows)
    _write_h3_belief_flux_map(config, trace_rows)
    _write_h3_policy_surface(config, trace_rows)
    _write_h3_policy_transitions(config, trace_rows)
    _write_h3_spatial_autocorrelation(config, trace_rows, edge_rows)
    _write_h3_entropy_free_energy_phase(config, trace_rows)
    animation_payload = json.loads(
        (config.output_dir / "data" / "h3_lattice_animation.json").read_text()
    )
    _write_h3_active_inference_lattice(config, animation_payload)
    if parent_child_rows:
        _write_nested_h3_parent_child_residuals(config, parent_child_rows)
    _write_spatial_inference_research_report(
        config,
        rows,
        cell_metrics,
        trace_rows,
        research_statistics,
        parent_child_rows,
    )


def _plot_h3_cell_metric_map(
    config: RunConfig, cell_metrics: list[dict[str, Any]]
) -> Path:
    """Plot H3 cell centroids colored by final free-energy metric."""
    scenario_label = (
        "H3" if config.scenario == "h3" else f"{config.scenario.title()} H3"
    )
    fig, ax = plt.subplots(figsize=(7, 6.4))
    lngs = [row["lng"] for row in cell_metrics]
    lats = [row["lat"] for row in cell_metrics]
    values = [row["free_energy"] for row in cell_metrics]
    scatter = ax.scatter(lngs, lats, c=values, cmap="viridis", s=150, edgecolor="k")
    ax.set_title(f"{scenario_label} Cell Free Energy")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Final free-energy value")
    _add_provenance_caption(
        fig, config, "Data: data/h3_cells.csv and data/h3_cells.geojson"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = save_matplotlib_figure_artifact(
        config,
        fig,
        "visualizations/h3_cell_metric_map.png",
        title=f"{scenario_label} Cell Free Energy",
        description="Static H3 cell centroid map colored by final variational free energy.",
        alt_text=(
            f"{config.scenario} H3 map with one point per H3 cell, colored by "
            "final free-energy value."
        ),
        plotted_metrics=["free_energy", "lat", "lng"],
        data_sources=["data/h3_cells.csv", "data/h3_cells.geojson"],
        plotted_data=cell_metrics,
    )
    plt.close(fig)
    return path


def _plot_free_energy_evolution(
    config: RunConfig, rows: Sequence[Mapping[str, Any]]
) -> Path:
    """Plot aggregate free-energy and expected-free-energy trajectories."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    timesteps = [int(row["timestep"]) for row in rows]
    ax.plot(timesteps, [float(row["free_energy"]) for row in rows], marker="o")
    ax.plot(
        timesteps,
        [float(row["expected_free_energy"]) for row in rows],
        marker="s",
        linestyle="--",
    )
    ax.set_title(f"{config.scenario.upper()} Free Energy Evolution")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Energy value")
    ax.legend(["Free energy", "Expected free energy"])
    ax.grid(True, alpha=0.25)
    _add_provenance_caption(
        fig, config, "Data: data/step_metrics.csv; FE and EFE are aggregate metrics"
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotted_data = [
        {
            "timestep": int(row["timestep"]),
            "free_energy": float(row["free_energy"]),
            "expected_free_energy": float(row["expected_free_energy"]),
        }
        for row in rows
    ]
    path = save_matplotlib_figure_artifact(
        config,
        fig,
        "visualizations/free_energy_evolution.png",
        title=f"{config.scenario.upper()} Free Energy Evolution",
        description="Line plot comparing aggregate variational free energy and expected free energy across timesteps.",
        alt_text=(
            f"{config.scenario} line chart with free energy and expected free "
            "energy values across timesteps."
        ),
        plotted_metrics=["free_energy", "expected_free_energy"],
        data_sources=["data/step_metrics.csv", "data/full_history.json"],
        plotted_data=plotted_data,
    )
    plt.close(fig)
    return path


def _plot_belief_entropy_coherence(
    config: RunConfig, rows: Sequence[Mapping[str, Any]]
) -> Path:
    """Plot belief entropy and spatial coherence diagnostics over time."""
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    timesteps = [int(row["timestep"]) for row in rows]
    ax1.plot(
        timesteps,
        [float(row["belief_entropy"]) for row in rows],
        marker="o",
        color="#2c7fb8",
    )
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Belief entropy (nats)", color="#2c7fb8")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        timesteps,
        [float(row["coherence"]) for row in rows],
        marker="s",
        color="#d95f0e",
    )
    ax2.set_ylabel("Spatial coherence", color="#d95f0e")
    ax1.set_title(f"{config.scenario.upper()} Entropy and Coherence")
    _add_provenance_caption(
        fig,
        config,
        "Data: data/step_metrics.csv; entropy from normalized beliefs, coherence from H3 neighbors",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plotted_data = [
        {
            "timestep": int(row["timestep"]),
            "belief_entropy": float(row["belief_entropy"]),
            "coherence": float(row["coherence"]),
            "neighbor_correlation": float(row["neighbor_correlation"]),
        }
        for row in rows
    ]
    path = save_matplotlib_figure_artifact(
        config,
        fig,
        "visualizations/belief_entropy_coherence.png",
        title=f"{config.scenario.upper()} Entropy and Spatial Coherence",
        description="Dual-axis plot of mean belief entropy and spatial coherence across timesteps.",
        alt_text=(
            f"{config.scenario} chart with belief entropy on the left axis and "
            "spatial coherence on the right axis across timesteps."
        ),
        plotted_metrics=["belief_entropy", "coherence", "neighbor_correlation"],
        data_sources=["data/step_metrics.csv", "data/h3_diagnostics.json"],
        plotted_data=plotted_data,
    )
    plt.close(fig)
    return path


def _write_interactive_h3_map(
    config: RunConfig, cell_metrics: list[dict[str, Any]]
) -> Path:
    """Write an interactive HTML cell map, using Plotly when available."""
    title = f"{config.scenario.upper()} H3 Active Inference Map"
    description = (
        "Interactive H3 cell map with final free energy, expected free energy, "
        "belief entropy, and selected action hover diagnostics."
    )
    alt_text = (
        f"Interactive {config.scenario} H3 map with each cell plotted by "
        "latitude and longitude and colored by free-energy value."
    )

    def _build_interactive_h3_map_figure() -> Any:
        import plotly.express as px  # noqa: PLC0415

        fig = px.scatter_geo(
            cell_metrics,
            lat="lat",
            lon="lng",
            color="free_energy",
            hover_name="cell",
            hover_data=[
                "expected_free_energy",
                "belief_entropy",
                "action",
                "pymdp_version",
                "selected_action_probability",
                "selected_negative_expected_free_energy",
            ],
            height=520,
            title=title,
        )
        fig.update_geos(fitbounds="locations", visible=True)
        fig.update_layout(
            margin={"r": 0, "t": 70, "l": 0, "b": 35},
            annotations=[
                {
                    "text": _provenance_caption(
                        config,
                        "Data: data/h3_cells.csv and data/h3_cells.geojson",
                    ),
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.08,
                    "xanchor": "left",
                    "font": {"size": 11, "color": "#4a4a4a"},
                }
            ],
        )
        return fig

    html = _plotly_or_table(
        _build_interactive_h3_map_figure,
        cell_metrics,
        ["Cell", "Lat", "Lng", "Free energy", "Belief entropy"],
        lambda row: (
            row["cell"],
            f"{row['lat']:.6f}",
            f"{row['lng']:.6f}",
            f"{row['free_energy']:.6f}",
            f"{row['belief_entropy']:.6f}",
        ),
        title,
        page_title=f"{config.scenario} H3 map",
        caption=_provenance_caption(
            config, "Data: data/h3_cells.csv and data/h3_cells.geojson"
        ),
    )
    return write_html_figure_artifact(
        config,
        "visualizations/interactive_h3_map.html",
        html,
        title=title,
        description=description,
        alt_text=alt_text,
        plotted_metrics=[
            "free_energy",
            "expected_free_energy",
            "belief_entropy",
            "action",
            "selected_action_probability",
            "selected_negative_expected_free_energy",
            "lat",
            "lng",
        ],
        data_sources=["data/h3_cells.csv", "data/h3_cells.geojson"],
        plotted_data=cell_metrics,
    )


def _write_h3_belief_flux_map(
    config: RunConfig, trace_rows: list[dict[str, Any]]
) -> Path:
    """Write an interactive H3 belief-flux and posterior-delta map."""
    rows = _latest_leaf_trace_rows(trace_rows)
    title = f"{config.scenario.upper()} H3 Belief Flux"

    def _build_belief_flux_figure() -> Any:
        import plotly.express as px  # noqa: PLC0415

        fig = px.scatter_geo(
            rows,
            lat="lat",
            lon="lng",
            color="belief_flux_divergence",
            size="posterior_delta",
            hover_name="cell",
            hover_data=[
                "entropy",
                "free_energy",
                "policy_entropy",
                "selected_action_probability",
                "local_coherence",
                "neighbor_count",
            ],
            height=540,
            title=title,
            color_continuous_scale="RdBu",
        )
        fig.update_geos(fitbounds="locations", visible=True)
        fig.update_layout(margin={"r": 0, "t": 70, "l": 0, "b": 35})
        return fig

    html = _plotly_or_table(
        _build_belief_flux_figure,
        rows,
        ["Cell", "Flux divergence", "Posterior delta", "Local coherence"],
        lambda row: (
            row["cell"],
            f"{row['belief_flux_divergence']:.6f}",
            f"{row['posterior_delta']:.6f}",
            f"{row['local_coherence']:.6f}",
        ),
        title,
    )

    return write_html_figure_artifact(
        config,
        "visualizations/h3_belief_flux_map.html",
        html,
        title=title,
        description="Interactive H3 map of belief-flux divergence, posterior delta, local coherence, and policy entropy.",
        alt_text="H3 cell map colored by belief-flux divergence with marker size indicating posterior delta.",
        plotted_metrics=[
            "belief_flux_divergence",
            "posterior_delta",
            "local_coherence",
            "policy_entropy",
            "selected_action_probability",
        ],
        data_sources=[
            "data/spatial_inference_trace.json",
            "data/h3_cell_diagnostics.csv",
            "data/h3_edge_diagnostics.csv",
        ],
        plotted_data=rows,
    )


def _write_h3_policy_surface(
    config: RunConfig, trace_rows: list[dict[str, Any]]
) -> Path:
    """Write a timestep-by-cell policy confidence surface."""
    rows = _leaf_trace_rows(trace_rows)
    timesteps = sorted({int(row["timestep"]) for row in rows})
    cells = sorted({str(row["cell"]) for row in rows})
    value_by_key = {
        (str(row["cell"]), int(row["timestep"])): float(
            row["selected_action_probability"]
        )
        for row in rows
    }
    z = [
        [value_by_key.get((cell, timestep), 0.0) for timestep in timesteps]
        for cell in cells
    ]
    title = f"{config.scenario.upper()} H3 Policy Confidence Surface"

    def _build_policy_surface_figure() -> Any:
        import plotly.graph_objects as go  # noqa: PLC0415

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=timesteps,
                y=cells,
                colorscale="Viridis",
                colorbar={"title": "Selected action probability"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Timestep",
            yaxis_title="H3 cell",
            height=max(420, min(900, 32 * max(1, len(cells)))),
            margin={"r": 20, "t": 70, "l": 160, "b": 55},
        )
        return fig

    html = _plotly_or_table(
        _build_policy_surface_figure,
        rows,
        ["Timestep", "Cell", "Selected action probability", "Policy entropy"],
        lambda row: (
            row["timestep"],
            row["cell"],
            f"{row['selected_action_probability']:.6f}",
            f"{row['policy_entropy']:.6f}",
        ),
        title,
    )

    return write_html_figure_artifact(
        config,
        "visualizations/h3_policy_surface.html",
        html,
        title=title,
        description="H3 policy posterior surface showing selected action confidence across cells and timesteps.",
        alt_text="Heatmap with timesteps on the x-axis, H3 cells on the y-axis, and color showing selected action probability.",
        plotted_metrics=[
            "selected_action_probability",
            "policy_entropy",
            "selected_negative_expected_free_energy",
        ],
        data_sources=[
            "data/spatial_inference_trace.json",
            "data/h3_cell_diagnostics.csv",
            "data/pymdp_policy_posteriors.csv",
        ],
        plotted_data=rows,
    )


def _write_h3_policy_transitions(
    config: RunConfig, trace_rows: list[dict[str, Any]]
) -> Path:
    """Write selected-action transition counts by timestep."""
    rows = _leaf_trace_rows(trace_rows)
    counts: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["timestep"]), str(row["selected_action_index"]))
        counts[key] = counts.get(key, 0) + 1
    plotted = [
        {"timestep": timestep, "selected_action_index": action, "count": count}
        for (timestep, action), count in sorted(counts.items())
    ]
    title = f"{config.scenario.upper()} H3 Policy Transitions"

    def _build_policy_transitions_figure() -> Any:
        import plotly.express as px  # noqa: PLC0415

        fig = px.bar(
            plotted,
            x="timestep",
            y="count",
            color="selected_action_index",
            barmode="stack",
            title=title,
            height=460,
        )
        fig.update_layout(
            xaxis_title="Timestep",
            yaxis_title="Cell count",
            legend_title_text="Selected action",
            margin={"r": 20, "t": 70, "l": 55, "b": 55},
        )
        return fig

    html = _plotly_or_table(
        _build_policy_transitions_figure,
        plotted,
        ["Timestep", "Selected action", "Cell count"],
        lambda row: (
            row["timestep"],
            row["selected_action_index"],
            row["count"],
        ),
        title,
    )

    return write_html_figure_artifact(
        config,
        "visualizations/h3_policy_transitions.html",
        html,
        title=title,
        description="Stacked counts of pymdp-selected actions by H3 timestep.",
        alt_text="Stacked bar chart showing how many H3 cells selected each action at every timestep.",
        plotted_metrics=["selected_action_index", "count"],
        data_sources=[
            "data/spatial_inference_trace.json",
            "data/h3_cell_diagnostics.csv",
            "data/spatial_research_statistics.json",
        ],
        plotted_data=plotted,
    )


def _write_h3_spatial_autocorrelation(
    config: RunConfig,
    trace_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> Path:
    """Write per-timestep graph-aware spatial trace diagnostics."""
    rows = _spatial_autocorrelation_rows(trace_rows, edge_rows)
    title = f"{config.scenario.upper()} H3 Spatial Autocorrelation"

    def _build_spatial_autocorrelation_figure() -> Any:
        import plotly.graph_objects as go  # noqa: PLC0415

        fig = go.Figure()
        for metric in (
            "moran_entropy_proxy",
            "mean_edge_belief_distance",
            "mean_neighbor_entropy_contrast",
            "mean_abs_flux_balance",
        ):
            fig.add_trace(
                go.Scatter(
                    x=[row["timestep"] for row in rows],
                    y=[row[metric] for row in rows],
                    mode="lines+markers",
                    name=metric,
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Timestep",
            yaxis_title="Diagnostic value",
            height=500,
            margin={"r": 20, "t": 70, "l": 65, "b": 55},
        )
        return fig

    html = _plotly_or_table(
        _build_spatial_autocorrelation_figure,
        rows,
        [
            "Timestep",
            "Moran entropy proxy",
            "Mean edge belief distance",
            "Neighbor entropy contrast",
            "Abs flux balance",
        ],
        lambda row: (
            row["timestep"],
            f"{row['moran_entropy_proxy']:.6f}",
            f"{row['mean_edge_belief_distance']:.6f}",
            f"{row['mean_neighbor_entropy_contrast']:.6f}",
            f"{row['mean_abs_flux_balance']:.6f}",
        ),
        title,
    )

    return write_html_figure_artifact(
        config,
        "visualizations/h3_spatial_autocorrelation.html",
        html,
        title=title,
        description="Per-timestep H3 adjacency diagnostics for entropy autocorrelation, edge disagreement, and belief-flux balance.",
        alt_text="Line chart of H3 spatial autocorrelation and edge diagnostics across timesteps.",
        plotted_metrics=[
            "moran_entropy_proxy",
            "mean_edge_belief_distance",
            "mean_neighbor_entropy_contrast",
            "mean_abs_flux_balance",
        ],
        data_sources=[
            "data/spatial_inference_trace.json",
            "data/h3_cell_diagnostics.csv",
            "data/h3_edge_diagnostics.csv",
            "data/spatial_research_statistics.json",
        ],
        plotted_data=rows,
    )


def _write_h3_entropy_free_energy_phase(
    config: RunConfig, trace_rows: list[dict[str, Any]]
) -> Path:
    """Write entropy/free-energy phase-space diagnostics."""
    rows = _leaf_trace_rows(trace_rows)
    title = f"{config.scenario.upper()} H3 Entropy-Free Energy Phase Space"

    def _build_entropy_free_energy_phase_figure() -> Any:
        import plotly.express as px  # noqa: PLC0415

        fig = px.scatter(
            rows,
            x="entropy",
            y="free_energy",
            color="timestep",
            size="selected_action_probability",
            hover_name="cell",
            hover_data=[
                "policy_entropy",
                "posterior_delta",
                "belief_flux_divergence",
                "local_coherence",
            ],
            title=title,
            height=520,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title="Belief entropy",
            yaxis_title="Variational free energy",
            margin={"r": 20, "t": 70, "l": 70, "b": 60},
        )
        return fig

    html = _plotly_or_table(
        _build_entropy_free_energy_phase_figure,
        rows,
        ["Timestep", "Cell", "Entropy", "Free energy", "Selected action probability"],
        lambda row: (
            row["timestep"],
            row["cell"],
            f"{row['entropy']:.6f}",
            f"{row['free_energy']:.6f}",
            f"{row['selected_action_probability']:.6f}",
        ),
        title,
    )

    return write_html_figure_artifact(
        config,
        "visualizations/h3_entropy_free_energy_phase.html",
        html,
        title=title,
        description="Cell-level phase-space view of belief entropy, variational free energy, and selected-action confidence.",
        alt_text="Scatter plot with belief entropy on the x-axis and free energy on the y-axis for each H3 cell and timestep.",
        plotted_metrics=[
            "entropy",
            "free_energy",
            "selected_action_probability",
            "policy_entropy",
            "belief_flux_divergence",
        ],
        data_sources=[
            "data/spatial_inference_trace.json",
            "data/h3_cell_diagnostics.csv",
            "data/spatial_research_statistics.json",
        ],
        plotted_data=rows,
    )


def _write_h3_active_inference_lattice(
    config: RunConfig, payload: Mapping[str, Any]
) -> Path:
    """Write an animated SVG lattice with H3 cells, observations, actions, and flux."""
    title = f"{config.scenario.upper()} Animated H3 Active-Inference Lattice"
    payload_json = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    data_sources = [
        "data/h3_lattice_animation.json",
        "data/spatial_inference_trace.json",
        "data/h3_cell_diagnostics.csv",
        "data/h3_edge_diagnostics.csv",
        "data/h3_cells.geojson",
    ]
    if payload.get("nested_h3"):
        data_sources.extend(
            [
                "data/h3_hierarchy.csv",
                "data/nested_h3_parent_child_diagnostics.csv",
                "data/nested_h3_level_diagnostics.csv",
            ]
        )
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    :root {
      --ink: #1f2328;
      --muted: #57606a;
      --line: #d0d7de;
      --panel: #f6f8fa;
      --obs: #2f7de1;
      --act: #c0392b;
      --flux: #586069;
      --parent: #6f42c1;
    }
    body {
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      max-width: 1220px;
    }
    header, .controls, .fallback {
      max-width: 1220px;
    }
    .controls {
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin: 14px 0;
      padding: 10px 12px;
    }
    label {
      color: var(--muted);
      display: inline-flex;
      font-size: 13px;
      gap: 6px;
      white-space: nowrap;
    }
    button, select, input[type="range"] {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      cursor: pointer;
      padding: 6px 10px;
    }
    input[type="range"] {
      min-width: 220px;
    }
    .stage {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 560px;
      overflow: hidden;
      position: relative;
    }
    svg {
      display: block;
      height: 620px;
      max-height: 72vh;
      width: 100%;
    }
    .detail {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .detail h2 {
      font-size: 16px;
      margin: 0 0 8px;
    }
    .detail dl {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 10px;
      margin: 0;
    }
    .detail dt {
      color: var(--muted);
      font-size: 12px;
    }
    .detail dd {
      margin: 0;
      overflow-wrap: anywhere;
    }
    .legend {
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }
    .cell-path {
      cursor: pointer;
      stroke-width: 1.2;
      transition: fill 180ms ease, stroke 180ms ease, opacity 180ms ease;
    }
    .parent-outline {
      fill: none;
      stroke: var(--parent);
      stroke-dasharray: 6 4;
      stroke-width: 2.1;
      pointer-events: none;
    }
    .flux-edge {
      fill: none;
      marker-end: url(#flux-arrow);
      stroke: var(--flux);
      stroke-linecap: round;
    }
    .parent-link {
      fill: none;
      stroke: var(--parent);
      stroke-dasharray: 3 5;
      stroke-linecap: round;
    }
    .obs-arrow {
      marker-end: url(#obs-arrow);
      stroke: var(--obs);
    }
    .act-arrow {
      marker-end: url(#act-arrow);
      stroke: var(--act);
    }
    .obs-arrow, .act-arrow {
      fill: none;
      stroke-linecap: round;
    }
    .time-chip {
      fill: rgba(255, 255, 255, 0.92);
      stroke: var(--line);
    }
    .fallback {
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 12px;
      padding: 12px;
    }
    .fallback[hidden] {
      display: none;
    }
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
    }
    th {
      background: var(--panel);
    }
    @media (max-width: 760px) {
      body { margin: 12px; }
      main { grid-template-columns: 1fr; }
      svg { height: 520px; max-height: none; }
      input[type="range"] { min-width: 150px; width: 100%; }
    }
    @media (prefers-reduced-motion: reduce) {
      .cell-path { transition: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <p>True H3 polygons are animated over timestep-indexed real pymdp diagnostics. Hex color shows the selected belief-state confidence by default; blue arrows encode observation evidence, red arrows encode selected action, gray arrows encode neighbor belief flux, and dashed purple links encode nested parent-child residuals when present.</p>
  </header>
  <section class="controls" aria-label="Lattice controls">
    <button id="playButton" type="button" aria-label="Play or pause animation">Play</button>
    <label>Timestep <input id="timeSlider" type="range" min="0" value="0" step="1"></label>
    <output id="timeLabel" for="timeSlider">0</output>
    <label>Heatmap
      <select id="metricSelect">
        <option value="belief_argmax_probability">Belief confidence</option>
        <option value="entropy">Belief entropy</option>
        <option value="free_energy">Variational free energy</option>
        <option value="policy_entropy">Policy entropy</option>
        <option value="selected_action_probability">Action confidence</option>
        <option value="local_coherence">Local coherence</option>
        <option value="belief_flux_divergence">Flux divergence</option>
      </select>
    </label>
    <label><input id="layerObservation" type="checkbox" checked> Observations</label>
    <label><input id="layerAction" type="checkbox" checked> Actions</label>
    <label><input id="layerFlux" type="checkbox" checked> Flux edges</label>
    <label><input id="layerNested" type="checkbox" checked> Nested links</label>
    <label><input id="reducedMotion" type="checkbox"> Reduced motion</label>
  </section>
  <main>
    <section class="stage">
      <svg id="h3-lattice-svg" role="img" aria-label="Animated H3 active inference lattice with heatmap cells and observation, action, and flux arrows"></svg>
    </section>
    <aside class="detail" id="detailPanel" aria-live="polite">
      <h2>Cell details</h2>
      <dl id="detailRows"></dl>
      <p class="legend">Hover or focus a cell. The default view emphasizes internal belief confidence while keeping perception, policy, and neighbor flux visible.</p>
    </aside>
  </main>
  <section class="fallback" id="fallback" hidden>
    <h2>Static payload preview</h2>
    <p>The animation requires D3. The payload below still exposes the first timestep's finite H3 diagnostics.</p>
    <table>
      <thead><tr><th>Cell</th><th>Belief</th><th>Entropy</th><th>Action</th><th>Observation</th></tr></thead>
      <tbody id="fallbackRows"></tbody>
    </table>
  </section>
  <script type="application/json" id="h3-lattice-animation-data">__PAYLOAD__</script>
  <script>
(() => {
  const payload = JSON.parse(document.getElementById("h3-lattice-animation-data").textContent);
  const fallback = document.getElementById("fallback");
  const firstFrame = payload.frames[0] || { cells: [] };
  document.getElementById("fallbackRows").innerHTML = firstFrame.cells.slice(0, 24).map((row) => (
    `<tr><td>${row.cell}</td><td>${row.belief_argmax_probability.toFixed(4)}</td><td>${row.entropy.toFixed(4)}</td><td>${row.selected_action_index}</td><td>${row.observation_strength.toFixed(4)}</td></tr>`
  )).join("");
  if (!window.d3 || !payload.cells.length || !payload.frames.length) {
    fallback.hidden = false;
    return;
  }

  const svg = d3.select("#h3-lattice-svg");
  const root = svg.append("g");
  const parentLayer = root.append("g").attr("class", "parent-layer");
  const linkLayer = root.append("g").attr("class", "parent-child-layer");
  const fluxLayer = root.append("g").attr("class", "flux-layer");
  const cellLayer = root.append("g").attr("class", "cell-layer");
  const observationLayer = root.append("g").attr("class", "observation-layer");
  const actionLayer = root.append("g").attr("class", "action-layer");
  const labelLayer = root.append("g").attr("class", "label-layer");
  svg.append("defs").html(`
    <marker id="obs-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--obs)"></path></marker>
    <marker id="act-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="var(--act)"></path></marker>
    <marker id="flux-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="var(--flux)"></path></marker>
  `);

  const cells = new Map(payload.cells.map((cell) => [cell.cell, cell]));
  const frames = new Map(payload.frames.map((frame) => [frame.timestep, frame]));
  const timesteps = payload.timesteps;
  const timeSlider = document.getElementById("timeSlider");
  const timeLabel = document.getElementById("timeLabel");
  const playButton = document.getElementById("playButton");
  const metricSelect = document.getElementById("metricSelect");
  const layerObservation = document.getElementById("layerObservation");
  const layerAction = document.getElementById("layerAction");
  const layerFlux = document.getElementById("layerFlux");
  const layerNested = document.getElementById("layerNested");
  const reducedMotion = document.getElementById("reducedMotion");
  const detailRows = document.getElementById("detailRows");
  const statePalette = ["#0969da", "#1a7f37", "#bf8700", "#cf222e", "#8250df", "#9a6700"];
  timeSlider.max = String(Math.max(0, timesteps.length - 1));

  const featureCollection = {
    type: "FeatureCollection",
    features: payload.cells.map((cell) => ({
      type: "Feature",
      properties: { cell: cell.cell },
      geometry: cell.geometry
    }))
  };
  let projection = d3.geoMercator();
  let path = d3.geoPath(projection);

  function resize() {
    const box = svg.node().getBoundingClientRect();
    const width = Math.max(320, box.width || 900);
    const height = Math.max(420, Math.min(680, box.height || 620));
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    projection = d3.geoMercator().fitExtent([[24, 24], [width - 24, height - 36]], featureCollection);
    path = d3.geoPath(projection);
    update(currentStep(), false);
  }

  function currentStep() {
    return timesteps[Number(timeSlider.value)] ?? timesteps[0];
  }

  function stateMap(frame) {
    return new Map(frame.cells.map((state) => [state.cell, state]));
  }

  function valueExtent(metric) {
    const values = payload.frames.flatMap((frame) => frame.cells.map((state) => Number(state[metric])).filter(Number.isFinite));
    const extent = d3.extent(values.length ? values : [0, 1]);
    if (extent[0] === extent[1]) {
      return [extent[0] - 0.5, extent[1] + 0.5];
    }
    return extent;
  }

  function angle(index, total, offset) {
    return offset + (2 * Math.PI * (index % Math.max(1, total))) / Math.max(1, total);
  }

  function centerOf(cellId) {
    const cell = cells.get(cellId);
    return cell ? projection(cell.centroid) : [0, 0];
  }

  function arrowEndpoints(cellId, index, total, length, inbound) {
    const [cx, cy] = centerOf(cellId);
    const theta = angle(index, total, inbound ? -Math.PI / 2 : Math.PI / 8);
    const dx = Math.cos(theta) * length;
    const dy = Math.sin(theta) * length;
    return inbound
      ? { x1: cx + dx, y1: cy + dy, x2: cx + dx * 0.18, y2: cy + dy * 0.18 }
      : { x1: cx + dx * 0.16, y1: cy + dy * 0.16, x2: cx + dx, y2: cy + dy };
  }

  function transition(selection, animate) {
    return animate && !reducedMotion.checked ? selection.transition().duration(180) : selection;
  }

  function setDetail(state) {
    const rows = [
      ["cell", state.cell],
      ["resolution", state.resolution],
      ["belief state", `${state.belief_argmax_index} (${state.belief_argmax_probability.toFixed(4)})`],
      ["entropy", state.entropy.toFixed(4)],
      ["free energy", state.free_energy.toFixed(4)],
      ["policy entropy", state.policy_entropy.toFixed(4)],
      ["selected action", `${state.selected_action_index} (${state.selected_action_probability.toFixed(4)})`],
      ["observation", `${state.dominant_observation_index} (${state.observation_strength.toFixed(4)})`],
      ["local coherence", state.local_coherence.toFixed(4)],
      ["flux divergence", state.belief_flux_divergence.toFixed(4)],
      ["parent", state.parent_cell || ""]
    ];
    detailRows.innerHTML = rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
  }

  function update(timestep, animate = true) {
    const frame = frames.get(timestep) || payload.frames[0];
    const states = stateMap(frame);
    const metric = metricSelect.value;
    const [minValue, maxValue] = valueExtent(metric);
    const color = d3.scaleSequential(d3.interpolateViridis).domain([minValue, maxValue]);
    timeLabel.value = String(timestep);
    timeLabel.textContent = String(timestep);

    const parentCells = payload.cells.filter((cell) => cell.is_aggregate_parent || cell.children.length);
    parentLayer.selectAll("path")
      .data(parentCells, (cell) => cell.cell)
      .join("path")
      .attr("class", "parent-outline")
      .attr("d", (cell) => path({ type: "Feature", geometry: cell.geometry }))
      .style("display", layerNested.checked && payload.nested_h3 ? null : "none");

    const leafCells = payload.cells.filter((cell) => !cell.is_aggregate_parent || !payload.nested_h3);
    const hexes = cellLayer.selectAll("path")
      .data(leafCells, (cell) => cell.cell)
      .join("path")
      .attr("class", "cell-path")
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("aria-label", (cell) => `H3 cell ${cell.cell}`)
      .attr("d", (cell) => path({ type: "Feature", geometry: cell.geometry }))
      .on("mouseenter focus", (event, cell) => {
        const state = states.get(cell.cell);
        if (state) setDetail(state);
      });
    transition(hexes, animate)
      .attr("fill", (cell) => {
        const state = states.get(cell.cell);
        return state ? color(Number(state[metric])) : "#d8dee4";
      })
      .attr("stroke", (cell) => {
        const state = states.get(cell.cell);
        return state ? statePalette[state.belief_argmax_index % statePalette.length] : "#8c959f";
      })
      .attr("opacity", (cell) => {
        const state = states.get(cell.cell);
        return state ? Math.max(0.48, Math.min(1, 0.56 + state.local_coherence * 0.42)) : 0.4;
      });

    const flux = fluxLayer.selectAll("line")
      .data(layerFlux.checked ? frame.edges : [], (edge) => `${edge.flux_source}-${edge.flux_target}`);
    flux.join("line")
      .attr("class", "flux-edge")
      .attr("x1", (edge) => centerOf(edge.flux_source)[0])
      .attr("y1", (edge) => centerOf(edge.flux_source)[1])
      .attr("x2", (edge) => centerOf(edge.flux_target)[0])
      .attr("y2", (edge) => centerOf(edge.flux_target)[1])
      .attr("stroke-width", (edge) => 0.8 + Math.min(4, edge.weight * 9))
      .attr("opacity", (edge) => 0.18 + Math.min(0.62, edge.coherence * 0.5));
    flux.exit().remove();

    const parentLinks = linkLayer.selectAll("line")
      .data(layerNested.checked && payload.nested_h3 ? frame.parent_child_links : [], (link) => `${link.parent}-${link.child}`);
    parentLinks.join("line")
      .attr("class", "parent-link")
      .attr("x1", (link) => centerOf(link.parent)[0])
      .attr("y1", (link) => centerOf(link.parent)[1])
      .attr("x2", (link) => centerOf(link.child)[0])
      .attr("y2", (link) => centerOf(link.child)[1])
      .attr("stroke-width", (link) => 0.8 + Math.min(4, link.cross_level_residual * 5))
      .attr("opacity", (link) => 0.18 + Math.min(0.7, link.cross_level_residual));
    parentLinks.exit().remove();

    const obs = observationLayer.selectAll("line")
      .data(layerObservation.checked ? frame.cells.filter((state) => !state.is_aggregate_parent && state.observation.length) : [], (state) => state.cell);
    obs.join("line")
      .attr("class", "obs-arrow")
      .each(function(state) {
        const endpoints = arrowEndpoints(state.cell, state.dominant_observation_index, state.observation.length, 22 + state.observation_strength * 34, true);
        d3.select(this).attr("x1", endpoints.x1).attr("y1", endpoints.y1).attr("x2", endpoints.x2).attr("y2", endpoints.y2);
      })
      .attr("stroke-width", (state) => 1 + state.observation_strength * 3)
      .attr("opacity", (state) => 0.32 + state.observation_strength * 0.58);
    obs.exit().remove();

    const acts = actionLayer.selectAll("line")
      .data(layerAction.checked ? frame.cells.filter((state) => !state.is_aggregate_parent && state.action_posterior.length) : [], (state) => state.cell);
    acts.join("line")
      .attr("class", "act-arrow")
      .each(function(state) {
        const endpoints = arrowEndpoints(state.cell, state.selected_action_index, state.action_posterior.length, 24 + state.selected_action_probability * 40, false);
        d3.select(this).attr("x1", endpoints.x1).attr("y1", endpoints.y1).attr("x2", endpoints.x2).attr("y2", endpoints.y2);
      })
      .attr("stroke-width", (state) => 1 + state.selected_action_probability * 3.4)
      .attr("opacity", (state) => 0.32 + state.selected_action_probability * 0.6);
    acts.exit().remove();

    labelLayer.selectAll("g").data([timestep]).join((enter) => {
      const group = enter.append("g").attr("transform", "translate(16, 16)");
      group.append("rect").attr("class", "time-chip").attr("rx", 6).attr("width", 164).attr("height", 32);
      group.append("text").attr("x", 10).attr("y", 21).attr("font-size", 13).attr("fill", "var(--ink)");
      return group;
    }).select("text").text(`timestep ${timestep} | ${metric}`);

    const firstState = frame.cells.find((state) => !state.is_aggregate_parent) || frame.cells[0];
    if (firstState) setDetail(firstState);
  }

  let timer = null;
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    playButton.textContent = "Play";
  }
  playButton.addEventListener("click", () => {
    if (timer) {
      stop();
      return;
    }
    playButton.textContent = "Pause";
    timer = setInterval(() => {
      const next = (Number(timeSlider.value) + 1) % timesteps.length;
      timeSlider.value = String(next);
      update(currentStep());
    }, reducedMotion.checked ? 1800 : 900);
  });
  timeSlider.addEventListener("input", () => {
    stop();
    update(currentStep());
  });
  for (const control of [metricSelect, layerObservation, layerAction, layerFlux, layerNested, reducedMotion]) {
    control.addEventListener("change", () => update(currentStep()));
  }
  window.addEventListener("resize", resize);
  resize();
})();
  </script>
</body>
</html>
"""
    html = html.replace("__TITLE__", title).replace("__PAYLOAD__", payload_json)
    return write_html_figure_artifact(
        config,
        "visualizations/h3_active_inference_lattice.html",
        html,
        title=title,
        description=(
            "Animated SVG H3 lattice showing belief heatmaps, observations, "
            "selected actions, neighbor belief flux, and nested parent-child links."
        ),
        alt_text=(
            "Animated H3 active-inference lattice with colored hex cells and "
            "observation, action, flux, and nested hierarchy arrows."
        ),
        plotted_metrics=[
            "belief_argmax_probability",
            "entropy",
            "free_energy",
            "policy_entropy",
            "selected_action_probability",
            "observation_strength",
            "belief_flux_divergence",
            "cross_level_residual",
        ],
        data_sources=data_sources,
        plotted_data=payload,
    )


def _write_nested_h3_parent_child_residuals(
    config: RunConfig, parent_child_rows: list[dict[str, Any]]
) -> Path:
    """Write nested parent-child consistency residual diagnostics."""
    title = f"{config.scenario.upper()} Nested H3 Parent-Child Residuals"

    def _build_parent_child_residuals_figure() -> Any:
        import plotly.express as px  # noqa: PLC0415

        fig = px.scatter(
            parent_child_rows,
            x="parent_resolution",
            y="cross_level_residual",
            color="timestep",
            size="child_entropy",
            hover_name="child",
            hover_data=[
                "parent",
                "child_resolution",
                "cross_level_consistency",
                "parent_entropy",
            ],
            title=title,
            height=500,
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            xaxis_title="Parent H3 resolution",
            yaxis_title="Cross-level residual",
            margin={"r": 20, "t": 70, "l": 70, "b": 55},
        )
        return fig

    html = _plotly_or_table(
        _build_parent_child_residuals_figure,
        parent_child_rows,
        ["Timestep", "Parent", "Child", "Residual"],
        lambda row: (
            row["timestep"],
            row["parent"],
            row["child"],
            f"{row['cross_level_residual']:.6f}",
        ),
        title,
    )

    return write_html_figure_artifact(
        config,
        "visualizations/nested_h3_parent_child_residuals.html",
        html,
        title=title,
        description="Nested H3 parent-child cross-level consistency residuals from real H3 hierarchy diagnostics.",
        alt_text="Scatter plot of nested H3 parent-child residuals by parent resolution and timestep.",
        plotted_metrics=[
            "cross_level_residual",
            "cross_level_consistency",
            "parent_entropy",
            "child_entropy",
        ],
        data_sources=[
            "data/nested_h3_parent_child_diagnostics.csv",
            "data/nested_h3_level_diagnostics.csv",
            "data/spatial_research_statistics.json",
        ],
        plotted_data=parent_child_rows,
    )


def _write_spatial_inference_research_report(
    config: RunConfig,
    rows: Sequence[Mapping[str, Any]],
    cell_metrics: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    research_statistics: Mapping[str, Any],
    parent_child_rows: list[dict[str, Any]],
) -> Path:
    """Write a compact HTML research report for spatial active inference runs."""
    _leaf_rows = _leaf_trace_rows(trace_rows)
    final_rows = _latest_leaf_trace_rows(trace_rows)
    summary_rows = statistics_summary_rows(research_statistics)
    summary_rows.extend(
        [
            {"group": "run", "metric": "timesteps", "value": float(len(rows))},
            {"group": "run", "metric": "cell_count", "value": float(len(cell_metrics))},
            {
                "group": "run",
                "metric": "final_mean_local_coherence",
                "value": (
                    float(
                        np.mean([float(row["local_coherence"]) for row in final_rows])
                    )
                    if final_rows
                    else 0.0
                ),
            },
        ]
    )
    non_degenerate = research_statistics.get("non_degenerate", {})
    policy = research_statistics.get("policy", {})
    graph = research_statistics.get("spatial_graph", {})
    nested = research_statistics.get("nested", {})
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{row.get('group', '')}</td>"
        f"<td>{row['metric']}</td><td>{float(row['value']):.6f}</td>"
        "</tr>"
        for row in summary_rows[:24]
    )
    diagnostic_cards = "\n".join(
        f"<article><strong>{label}</strong><span>{value}</span></article>"
        for label, value in [
            ("pymdp", "inferactively-pymdp 1.0.3"),
            ("H3", "h3-py 4.5.0"),
            ("policy switches", int(policy.get("switch_count", 0))),
            (
                "unique actions",
                int(non_degenerate.get("unique_selected_action_count", 0)),
            ),
            (
                "policy probability std",
                f"{float(non_degenerate.get('selected_action_probability_std', 0.0)):.6f}",
            ),
            (
                "Moran entropy proxy",
                f"{float(graph.get('moran_entropy_proxy', 0.0)):.6f}",
            ),
            (
                "nested mean residual",
                f"{float(nested.get('mean_parent_child_residual', 0.0)):.6f}",
            ),
        ]
    )
    artifact_links = [
        ("Interactive H3 map", "interactive_h3_map.html"),
        ("pymdp policy/free-energy", "pymdp_policy_free_energy.html"),
        ("Belief flux map", "h3_belief_flux_map.html"),
        ("Policy surface", "h3_policy_surface.html"),
        ("Policy transitions", "h3_policy_transitions.html"),
        ("Spatial autocorrelation", "h3_spatial_autocorrelation.html"),
        ("Entropy/free-energy phase", "h3_entropy_free_energy_phase.html"),
        ("Animated H3 active-inference lattice", "h3_active_inference_lattice.html"),
    ]
    if parent_child_rows:
        artifact_links.extend(
            [
                ("Nested hierarchy map", "nested_h3_hierarchy_map.html"),
                ("Nested level map", "nested_h3_level_map.html"),
                (
                    "Nested parent-child residuals",
                    "nested_h3_parent_child_residuals.html",
                ),
            ]
        )
    artifact_rows = "\n".join(
        f'<li><a href="{href}">{label}</a></li>' for label, href in artifact_links
    )
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{config.scenario.upper()} Spatial Active Inference Research Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; max-width: 1120px; color: #1f2328; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 18px 0; }}
    article {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 10px 12px; background: #f6f8fa; }}
    article strong {{ display: block; font-size: 12px; text-transform: uppercase; color: #57606a; }}
    article span {{ display: block; font-size: 20px; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    a {{ color: #0969da; }}
    .preview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .preview section {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }}
  </style>
</head>
<body>
  <h1>{config.scenario.upper()} Spatial Active Inference Research Report</h1>
  <p>{_provenance_caption(config, "Data: manifest-linked spatial trace, H3 diagnostics, and pymdp policy posterior files")}</p>
  <div class="cards">{diagnostic_cards}</div>
  <h2>Research Statistics</h2>
  <table><thead><tr><th>Group</th><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Visualization Artifacts</h2>
  <div class="preview">
    <section><h3>Policy and Free Energy</h3><ul>
      <li><a href="pymdp_policy_free_energy.html">pymdp_policy_free_energy.html</a></li>
      <li><a href="h3_policy_surface.html">h3_policy_surface.html</a></li>
      <li><a href="h3_policy_transitions.html">h3_policy_transitions.html</a></li>
    </ul></section>
    <section><h3>Spatial Fields</h3><ul>
      <li><a href="interactive_h3_map.html">interactive_h3_map.html</a></li>
      <li><a href="h3_active_inference_lattice.html">h3_active_inference_lattice.html</a></li>
      <li><a href="h3_belief_flux_map.html">h3_belief_flux_map.html</a></li>
      <li><a href="h3_spatial_autocorrelation.html">h3_spatial_autocorrelation.html</a></li>
      <li><a href="h3_entropy_free_energy_phase.html">h3_entropy_free_energy_phase.html</a></li>
    </ul></section>
    <section><h3>All Links</h3><ul>{artifact_rows}</ul></section>
  </div>
  <h2>Trace Data</h2>
  <ul>
    <li><a href="../data/spatial_inference_trace.json">spatial_inference_trace.json</a></li>
    <li><a href="../data/spatial_research_statistics.json">spatial_research_statistics.json</a></li>
    <li><a href="../data/h3_cell_diagnostics.csv">h3_cell_diagnostics.csv</a></li>
    <li><a href="../data/h3_edge_diagnostics.csv">h3_edge_diagnostics.csv</a></li>
    <li><a href="../data/pymdp_h3_diagnostics.json">pymdp_h3_diagnostics.json</a></li>
    <li><a href="../data/pymdp_policy_posteriors.csv">pymdp_policy_posteriors.csv</a></li>
  </ul>
</body>
</html>
"""
    return write_html_figure_artifact(
        config,
        "visualizations/spatial_inference_research_report.html",
        html,
        title=f"{config.scenario.upper()} Spatial Active Inference Research Report",
        description="Research report linking spatial trace diagnostics, graph-aware statistics, belief-flux maps, policy surfaces, and pymdp H3 outputs.",
        alt_text="HTML report summarizing spatial active inference statistics with links to manifest-backed maps and charts.",
        plotted_metrics=[row["metric"] for row in summary_rows],
        data_sources=[
            "data/spatial_inference_trace.json",
            "data/spatial_research_statistics.json",
            "data/h3_cell_diagnostics.csv",
            "data/h3_edge_diagnostics.csv",
            "data/pymdp_h3_diagnostics.json",
            "data/pymdp_policy_posteriors.csv",
        ],
        plotted_data=summary_rows,
    )


def _leaf_trace_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return trace rows for leaf/runtime cells, excluding aggregate parents."""
    return [
        row for row in trace_rows if not bool(row.get("aggregate_parent_cell", False))
    ]


def _spatial_autocorrelation_rows(
    trace_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-timestep H3 adjacency statistics for visualization."""
    rows = _leaf_trace_rows(trace_rows)
    timesteps = sorted({int(row["timestep"]) for row in rows})
    output: list[dict[str, Any]] = []
    for timestep in timesteps:
        timestep_rows = [row for row in rows if int(row["timestep"]) == timestep]
        entropy_by_cell = {
            str(row["cell"]): float(row["entropy"]) for row in timestep_rows
        }
        edge_subset = [
            edge for edge in edge_rows if int(edge.get("timestep", 0)) == timestep
        ]
        neighbor_contrast = []
        products = []
        entropy_values = list(entropy_by_cell.values())
        entropy_mean = float(np.mean(entropy_values)) if entropy_values else 0.0
        entropy_var = (
            float(np.mean([(value - entropy_mean) ** 2 for value in entropy_values]))
            if entropy_values
            else 0.0
        )
        for edge in edge_subset:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source in entropy_by_cell and target in entropy_by_cell:
                source_delta = entropy_by_cell[source] - entropy_mean
                target_delta = entropy_by_cell[target] - entropy_mean
                neighbor_contrast.append(abs(source_delta - target_delta))
                products.append(source_delta * target_delta)
        flux_values = [float(row["belief_flux_divergence"]) for row in timestep_rows]
        output.append(
            {
                "timestep": timestep,
                "moran_entropy_proxy": (
                    float(np.mean(products) / entropy_var)
                    if products and entropy_var > 1e-12
                    else 0.0
                ),
                "mean_edge_belief_distance": (
                    float(
                        np.mean(
                            [
                                float(edge.get("belief_distance", 0.0))
                                for edge in edge_subset
                            ]
                        )
                    )
                    if edge_subset
                    else 0.0
                ),
                "mean_edge_coherence": (
                    float(
                        np.mean(
                            [float(edge.get("coherence", 0.0)) for edge in edge_subset]
                        )
                    )
                    if edge_subset
                    else 0.0
                ),
                "mean_neighbor_entropy_contrast": (
                    float(np.mean(neighbor_contrast)) if neighbor_contrast else 0.0
                ),
                "mean_abs_flux_balance": (
                    abs(float(np.sum(flux_values))) if flux_values else 0.0
                ),
            }
        )
    return output


def _latest_leaf_trace_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return leaf trace rows for the latest timestep."""
    rows = _leaf_trace_rows(trace_rows)
    if not rows:
        return []
    latest = max(int(row["timestep"]) for row in rows)
    return [row for row in rows if int(row["timestep"]) == latest]


def main(argv: list[str] | None = None) -> int:
    """CLI-compatible main for direct module execution."""
    from geo_infer_act.runners.cli import main as cli_main

    return cli_main(argv if argv is not None else sys.argv[1:])
