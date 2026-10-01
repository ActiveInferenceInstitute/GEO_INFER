"""H3 Active Inference runner utilities."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, cast
from collections.abc import Iterable, Mapping

import numpy as np

from geo_infer_act.core.active_inference import (
    ActiveInferenceModel,
    H3GridInferenceResult,
)
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.runners.io import write_csv, write_json
from geo_infer_act.utils.h3_adapter import get_h3_adapter, normalize_belief_vector


def setup_san_francisco_boundary() -> dict[str, Any]:
    """Return a compact San Francisco GeoJSON polygon for H3 examples."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.435, 37.765],
                [-122.405, 37.765],
                [-122.405, 37.795],
                [-122.435, 37.795],
                [-122.435, 37.765],
            ]
        ],
    }


def h3_cells_for_config(
    resolution: int = 8,
    ring_size: int = 1,
    cells: Iterable[str] | None = None,
) -> list[str]:
    """Return validated H3 cells for a compact San Francisco scenario."""
    adapter = get_h3_adapter()
    if cells:
        return sorted(adapter.validate_cells(cells))
    center = adapter.latlng_to_cell(37.7793, -122.4192, resolution)
    return sorted(adapter.validate_cells(adapter.grid_disk(center, ring_size)))


def generate_realistic_environmental_observations(
    h3_cells: Iterable[str],
    timestep: float,
    base_patterns: Mapping[str, Mapping[str, Any]] | None = None,
    spatial_seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Generate deterministic environmental observations for real H3 cells."""
    adapter = get_h3_adapter()
    rng = np.random.default_rng(spatial_seed)
    patterns = dict(
        base_patterns
        or {
            "temperature": {
                "base": 18.0,
                "amplitude": 4.0,
                "spatial_scale": 45.0,
                "temporal_period": 24.0,
            },
            "vegetation_density": {
                "base": 0.52,
                "amplitude": 0.28,
                "spatial_scale": 35.0,
                "temporal_period": 18.0,
                "coastal_gradient": True,
            },
            "air_quality": {
                "base": 0.65,
                "amplitude": 0.18,
                "spatial_scale": 25.0,
                "temporal_period": 10.0,
            },
            "water_stress": {
                "base": 0.35,
                "amplitude": 0.16,
                "spatial_scale": 30.0,
                "temporal_period": 12.0,
            },
            "carbon_flux": {
                "base": 0.0,
                "amplitude": 0.22,
                "spatial_scale": 40.0,
                "temporal_period": 20.0,
            },
        }
    )
    observations: dict[str, dict[str, float]] = {}
    for index, cell in enumerate(h3_cells):
        lat, lng = adapter.cell_to_latlng(str(cell))
        phase = (lat * 3.1) + (lng * 2.7) + (index * 0.17)
        if spatial_seed is not None:
            phase += float(rng.normal(0.0, 0.015))
        cell_observations: dict[str, float] = {}
        for name, pattern in patterns.items():
            base = float(pattern.get("base", 0.5))
            amplitude = float(pattern.get("amplitude", 0.1))
            spatial_scale = float(pattern.get("spatial_scale", 25.0))
            temporal_period = float(pattern.get("temporal_period", 12.0))
            spatial_term = math.sin((lat + lng) * spatial_scale + phase)
            temporal_term = math.cos(
                (float(timestep) / temporal_period) * 2.0 * math.pi
            )
            value = base + amplitude * 0.5 * (spatial_term + temporal_term)
            if pattern.get("coastal_gradient"):
                value += 0.18 * (abs(lng + 122.4) - 0.1)
            if name not in {"temperature", "carbon_flux"}:
                value = float(np.clip(value, 0.0, 1.0))
            cell_observations[name] = float(value)
        observations[str(cell)] = cell_observations
    return observations


def observation_dict_to_vector(observation: Mapping[str, float]) -> np.ndarray:
    """Convert environmental observations into a normalized four-state vector."""
    vector = np.array(
        [
            1.0 - float(observation.get("air_quality", 0.5)),
            float(observation.get("water_stress", 0.5)),
            float(observation.get("vegetation_density", 0.5)),
            1.0 / (1.0 + math.exp(-float(observation.get("carbon_flux", 0.0)))),
        ],
        dtype=float,
    )
    return cast(np.ndarray, normalize_belief_vector(vector))


def create_h3_model(cells: list[str]) -> tuple[GenerativeModel, ActiveInferenceModel]:
    """Create ACT generative and active-inference models for H3 cells."""
    generative_model = GenerativeModel(
        "categorical",
        {"state_dim": 4, "obs_dim": 4, "spatial_mode": True},
    )
    generative_model.spatial_mode = True
    generative_model.h3_cells = list(cells)
    graph = generative_model._build_h3_neighbor_graph(cells)
    if graph is not None:
        generative_model.spatial_graph = graph
    active_model = ActiveInferenceModel(
        "categorical",
        policy_selection_mode="deterministic",
        random_seed=0,
    )
    active_model.set_generative_model(generative_model)
    return generative_model, active_model


def run_h3_active_inference(
    output_dir: Path,
    h3_resolution: int = 8,
    timesteps: int = 10,
    n_agents: int = 3,
    spatial_seed: int | None = 42,
) -> dict[str, Any]:
    """Run a deterministic H3 Active Inference simulation."""
    start_time = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = h3_cells_for_config(h3_resolution, ring_size=1)
    generative_model, active_model = create_h3_model(cells)
    active_model.parameters["random_seed"] = spatial_seed

    history: list[dict[str, Any]] = []
    environmental_history: list[dict[str, dict[str, float]]] = []
    coordination_history: list[dict[str, Any]] = []
    free_energy_evolution: list[float] = []
    step_rows: list[dict[str, Any]] = []

    for timestep in range(timesteps):
        env_obs = generate_realistic_environmental_observations(
            cells,
            timestep=float(timestep),
            spatial_seed=spatial_seed,
        )
        vector_observations = {
            cell: observation_dict_to_vector(obs) for cell, obs in env_obs.items()
        }
        belief_update = generative_model.update_h3_beliefs(
            vector_observations,
            return_result=True,
        )
        grid_result = cast(
            H3GridInferenceResult,
            active_model.infer_over_h3_grid(
                vector_observations,
                return_result=True,
            ),
        )
        cell_records = {}
        for cell, step_result in grid_result.cell_results.items():
            cell_records[cell] = {
                "beliefs": (
                    np.asarray(step_result.beliefs["states"]).tolist()
                    if isinstance(step_result.beliefs, dict)
                    else np.asarray(step_result.beliefs).tolist()
                ),
                "action": str(step_result.action),
                "free_energy": float(step_result.free_energy),
                "expected_free_energy": step_result.expected_free_energy,
            }
        average_fe = float(grid_result.aggregate_free_energy)
        free_energy_evolution.append(average_fe)
        coherence = 0.0
        correlation = 0.0
        sc = getattr(belief_update, "spatial_consistency", None)
        if sc is not None:
            coherence = float(getattr(sc, "global_coherence", 0.0))
            correlation = float(getattr(sc, "neighbor_correlations", 0.0))

        global_metrics = {
            "average_free_energy": average_fe,
            "total_free_energy": float(average_fe * len(cells)),
            "coordination_coherence": coherence,
            "neighbor_correlation": correlation,
            "processing_time": float(time.perf_counter() - start_time),
        }
        history.append(
            {
                "timestep": timestep,
                "cells": cell_records,
                "global_metrics": global_metrics,
            }
        )
        environmental_history.append(env_obs)
        coordination_history.append(
            {
                "timestep": timestep,
                "n_agents": int(n_agents),
                "cell_count": len(cells),
                "coherence": global_metrics["coordination_coherence"],
            }
        )
        step_rows.append(
            {
                "timestep": timestep,
                "scenario": "h3",
                **global_metrics,
                "cell_count": len(cells),
            }
        )

    metrics = {
        "free_energy_evolution": free_energy_evolution,
        "final_free_energy": free_energy_evolution[-1],
        "free_energy_change": free_energy_evolution[-1] - free_energy_evolution[0],
        "total_processing_time": float(time.perf_counter() - start_time),
    }
    results = {
        "simulation_params": {
            "h3_resolution": h3_resolution,
            "n_cells": len(cells),
            "n_agents": n_agents,
            "timesteps": timesteps,
            "spatial_seed": spatial_seed,
        },
        "history": history,
        "metrics": metrics,
        "environmental_observations": environmental_history,
        "agent_coordination_history": coordination_history,
    }
    write_json(output_dir / "data" / "h3_results.json", results)
    write_csv(output_dir / "data" / "h3_step_metrics.csv", step_rows)
    return results
