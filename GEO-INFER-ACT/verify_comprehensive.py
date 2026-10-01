#!/usr/bin/env python3
"""Comprehensive ACT method, output, docs, and visualization audit."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np


ACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ACT_ROOT.parent


def _jsonable(value: Any) -> Any:
    """Convert numpy, dataclass, and path-rich values into JSON-safe values."""
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return path


def _assert_finite(value: float, label: str) -> None:
    if not math.isfinite(float(value)):
        raise AssertionError(f"{label} is not finite: {value!r}")


def _assert_distribution(values: Any, label: str) -> None:
    if isinstance(values, dict):
        if "states" in values:
            values = values["states"]
        elif "mean" in values:
            values = values["mean"]
        else:
            values = next(iter(values.values()))
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise AssertionError(f"{label} is empty")
    if not np.all(np.isfinite(vector)):
        raise AssertionError(f"{label} contains non-finite values")
    if np.any(vector < -1e-9):
        raise AssertionError(f"{label} contains negative probabilities")
    total = float(np.sum(vector))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise AssertionError(f"{label} sums to {total}, expected 1.0")


def _section(
    name: str,
    func: Callable[[Path], dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    start = time.time()
    section_dir = output_dir / "method_audit" / name
    section_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Starting section %s", name)
    try:
        payload = func(section_dir)
        result = {
            "name": name,
            "status": "passed",
            "duration_seconds": round(time.time() - start, 4),
            "payload": payload,
        }
    except Exception as exc:  # pragma: no cover - audit reporting path
        result = {
            "name": name,
            "status": "failed",
            "duration_seconds": round(time.time() - start, 4),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        logging.exception("Section %s failed", name)
    _write_json(section_dir / "result.json", result)
    return result


def audit_active_inference_model(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act import (
        ActiveInferenceModel,
        ActiveInferenceStepResult,
        GenerativeModel,
        H3BeliefUpdateResult,
        H3GridInferenceResult,
    )
    from geo_infer_act.runners.h3 import setup_san_francisco_boundary

    generative_model = GenerativeModel(
        model_type="categorical",
        parameters={"state_dim": 3, "obs_dim": 3},
        model_id="audit_categorical",
    )
    agent = ActiveInferenceModel(
        model_type="categorical",
        policy_selection_mode="deterministic",
        random_seed=7,
    )
    agent.set_generative_model(generative_model)

    observation = np.array([1.0, 0.0, 0.0])
    beliefs = agent.perceive(observation)
    _assert_distribution(beliefs, "ActiveInferenceModel.perceive")
    action = agent.act(["survey", "wait", "sample"])
    agent.update_observations({"observations": observation})
    agent.update_preferences({"states": np.array([0.6, 0.3, 0.1])})
    agent.update_with_outcome(
        {"action": action},
        {"observation": np.array([0.0, 1.0, 0.0])},
    )
    policies = agent.generate_policies(
        [
            {"action": "survey", "predicted_beliefs": [0.7, 0.2, 0.1]},
            {"action": "wait", "predicted_beliefs": [0.2, 0.6, 0.2]},
        ]
    )
    selected_policy = agent.select_policy(policies)
    expected_free_energy = agent.compute_expected_free_energy(policies[0])
    _assert_finite(
        expected_free_energy, "ActiveInferenceModel.compute_expected_free_energy"
    )
    step_result = agent.step(
        np.array([0.0, 0.0, 1.0]),
        available_actions=["survey", "wait", "sample"],
        return_result=True,
    )
    if not isinstance(step_result, ActiveInferenceStepResult):
        raise AssertionError(
            "step(return_result=True) did not return ActiveInferenceStepResult"
        )
    _assert_finite(step_result.free_energy, "ActiveInferenceModel.step.free_energy")
    current_free_energy = agent.compute_free_energy()
    _assert_finite(current_free_energy, "ActiveInferenceModel.compute_free_energy")
    history_len = len(agent.get_history())
    current_state = agent.get_current_state()
    agent.set_preferences({"states": np.array([0.2, 0.5, 0.3])})
    agent.reset()

    h3_model = GenerativeModel(
        model_type="categorical",
        parameters={"state_dim": 4, "obs_dim": 4},
        model_id="audit_h3",
    )
    h3_model.enable_h3_spatial(8, setup_san_francisco_boundary())
    h3_observations = {
        cell: np.array([1.0, 0.0, 0.0, 0.0]) for cell in h3_model.h3_cells[:3]
    }
    h3_agent = ActiveInferenceModel(
        model_type="categorical",
        policy_selection_mode="deterministic",
        random_seed=11,
    )
    h3_agent.set_generative_model(h3_model)
    h3_update = h3_agent.apply_to_h3(h3_observations, return_result=True)
    if not isinstance(h3_update, H3BeliefUpdateResult):
        raise AssertionError(
            "apply_to_h3(return_result=True) did not return H3BeliefUpdateResult"
        )
    h3_grid = h3_agent.infer_over_h3_grid(h3_observations, return_result=True)
    if not isinstance(h3_grid, H3GridInferenceResult):
        raise AssertionError(
            "infer_over_h3_grid(return_result=True) did not return H3GridInferenceResult"
        )
    _assert_finite(h3_update.aggregate_free_energy, "apply_to_h3.aggregate_free_energy")
    _assert_finite(
        h3_grid.aggregate_free_energy, "infer_over_h3_grid.aggregate_free_energy"
    )

    payload = {
        "beliefs": beliefs,
        "action": action,
        "selected_policy": selected_policy,
        "expected_free_energy": expected_free_energy,
        "step_result": step_result,
        "current_free_energy": current_free_energy,
        "history_len": history_len,
        "current_state": current_state,
        "h3_cell_count": len(h3_model.h3_cells),
        "h3_update": h3_update,
        "h3_grid": h3_grid,
    }
    _write_json(output_dir / "active_inference_model_methods.json", payload)
    return payload


def audit_generative_model(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act.core.generative_model import GenerativeModel, MarkovBlanket
    from geo_infer_act.runners.h3 import setup_san_francisco_boundary

    blanket = MarkovBlanket(
        sensory_states=[0],
        active_states=[1],
        internal_states=[2],
        external_states=[3],
    )
    blanket_ok = blanket.check_conditional_independence(
        2, np.array([0.2, 0.3, 0.25, 0.9])
    )

    model = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 3})
    updated = model.update_beliefs({"observations": np.array([1.0, 0.0, 0.0])})
    _assert_distribution(updated["states"], "GenerativeModel.update_beliefs")
    free_energy = model.compute_free_energy()
    _assert_finite(free_energy, "GenerativeModel.compute_free_energy")

    child = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 3})
    model.add_nested_level(child)
    model.update_nested_beliefs({"observations": np.array([0.0, 1.0, 0.0])})
    model.set_preferences({"observations": np.array([0.5, 0.3, 0.2])})
    summary = model.get_model_summary()
    _assert_finite(
        summary["free_energy"], "GenerativeModel.get_model_summary.free_energy"
    )

    nav_model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 1})
    nav_model.enable_spatial_navigation(2)

    h3_model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    h3_model.enable_h3_spatial(8, setup_san_francisco_boundary())
    h3_observations = {
        cell: np.array([1.0, 0.0, 0.0, 0.0]) for cell in h3_model.h3_cells[:4]
    }
    h3_update = h3_model.update_h3_beliefs(h3_observations, return_result=True)
    diffused = h3_model.diffuse_beliefs(h3_update.h3_beliefs, diffusion_rate=0.2)
    aggregated = h3_model.aggregate_beliefs_to_resolution(diffused, target_resolution=7)
    if not aggregated:
        raise AssertionError("aggregate_beliefs_to_resolution returned no parent cells")

    payload = {
        "markov_blanket_conditional_independence": blanket_ok,
        "updated_beliefs": updated,
        "free_energy": free_energy,
        "summary": summary,
        "spatial_navigation_state_dim": nav_model.state_dim,
        "h3_cell_count": len(h3_model.h3_cells),
        "h3_update": h3_update,
        "diffused_cell_count": len(diffused),
        "aggregated_cell_count": len(aggregated),
    }
    _write_json(output_dir / "generative_model_methods.json", payload)
    return payload


def audit_free_energy_and_policy(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act import (
        FreeEnergyBreakdown,
        FreeEnergyCalculator,
        PolicyEvaluation,
        PolicySelector,
    )

    beliefs = np.array([0.6, 0.3, 0.1])
    observations = np.array([1.0, 0.0, 0.0])
    preferences = np.array([0.7, 0.2, 0.1])

    calculator = FreeEnergyCalculator()
    categorical_breakdown = calculator.compute_categorical_free_energy(
        beliefs,
        observations,
        preferences,
        return_breakdown=True,
    )
    if not isinstance(categorical_breakdown, FreeEnergyBreakdown):
        raise AssertionError("categorical VFE did not return FreeEnergyBreakdown")
    gaussian_fe = calculator.compute_gaussian_free_energy(
        mean=np.array([0.1, -0.1]),
        precision=np.eye(2),
        observations=np.array([0.0, 0.0]),
    )
    expected_breakdown = calculator.compute_expected_free_energy(
        beliefs,
        {"action": "survey", "predicted_beliefs": [0.7, 0.2, 0.1]},
        preferences,
        return_breakdown=True,
    )
    compatibility_fe = calculator.compute(
        beliefs=beliefs,
        observations=observations,
        preferences=preferences,
        model_type="categorical",
    )

    selector = PolicySelector(selection_mode="deterministic", random_seed=5)
    policies = [
        {"action": "survey", "predicted_beliefs": [0.7, 0.2, 0.1]},
        {"action": "wait", "predicted_beliefs": [0.2, 0.6, 0.2]},
    ]
    selection = selector.select_policy(beliefs, policies, preferences)
    if not isinstance(selection["evaluation"], PolicyEvaluation):
        raise AssertionError("select_policy did not attach PolicyEvaluation")
    policy_efe = selector.compute_expected_free_energy(
        beliefs,
        policies[0],
        preferences,
    )
    precision = selector.compute_policy_precision(selection["all_free_energies"])
    evaluated = selector.evaluate_policy_set(beliefs, policies, preferences)
    selected_action = selector.select_action(beliefs, ["survey", "wait"])

    for label, value in {
        "categorical_fe": categorical_breakdown.free_energy,
        "gaussian_fe": gaussian_fe,
        "expected_fe": expected_breakdown.free_energy,
        "compatibility_fe": compatibility_fe,
        "policy_efe": policy_efe,
        "policy_precision": precision,
    }.items():
        _assert_finite(value, label)

    payload = {
        "categorical_breakdown": categorical_breakdown,
        "gaussian_free_energy": gaussian_fe,
        "expected_breakdown": expected_breakdown,
        "compatibility_free_energy": compatibility_fe,
        "selection": selection,
        "policy_efe": policy_efe,
        "policy_precision": precision,
        "evaluated_policy_count": len(evaluated["evaluations"]),
        "selected_action": selected_action,
    }
    _write_json(output_dir / "free_energy_policy_methods.json", payload)
    return payload


def audit_inference_math(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act.core.belief_updating import BayesianBeliefUpdate
    from geo_infer_act.core.dynamic_causal_model import DynamicCausalModel
    from geo_infer_act.core.markov_decision_process import MarkovDecisionProcess
    from geo_infer_act.core.variational_inference import VariationalInference

    updater = BayesianBeliefUpdate()
    categorical = updater.update_categorical(
        np.array([0.3, 0.4, 0.3]),
        np.array([1.0, 0.0]),
        np.array([[0.8, 0.2, 0.1], [0.2, 0.8, 0.9]]),
    )
    _assert_distribution(categorical, "BayesianBeliefUpdate.update_categorical")
    gaussian = updater.update_gaussian(
        np.zeros(2),
        np.eye(2),
        np.array([0.2, -0.1]),
        np.eye(2),
        np.eye(2) * 4.0,
    )
    prediction_error = updater.compute_prediction_error(
        np.array([0.1, 0.2]), np.array([0.2, 0.1])
    )
    surprise = updater.compute_surprise(np.array([1.0, 0.0]), np.array([0.8, 0.2]))
    dispatched = updater.update_beliefs(
        np.array([0.3, 0.4, 0.3]),
        np.array([1.0, 0.0]),
        np.array([[0.8, 0.2, 0.1], [0.2, 0.8, 0.9]]),
    )

    vi = VariationalInference(max_iterations=10, random_seed=3)
    mean_field = vi.mean_field_update(
        {"concentration": np.ones(3)},
        {},
        np.array([1.0, 0.0, 0.0]),
    )
    mf_cat = vi.mean_field_update_categorical(
        np.ones(3), np.eye(3), np.array([1.0, 0.0, 0.0])
    )
    mf_gauss = vi.mean_field_update_gaussian(
        np.zeros(2), np.eye(2), np.array([0.1, -0.1])
    )
    structured = vi.structured_update(
        {"variables": {"x": {"dimension": 2}, "y": {"dimension": 3}}},
        {"x": np.array([1.0, 0.0])},
    )
    importance = vi.importance_sampling_update(
        {"mean": np.zeros(2), "covariance": np.eye(2)},
        lambda sample, obs: float(np.exp(-np.sum((sample - obs) ** 2))),
        np.zeros(2),
        n_samples=64,
    )
    elbo = vi.compute_elbo(
        {"mean": np.zeros(2), "precision": np.eye(2)},
        {"mean": np.zeros(2), "precision": np.eye(2)},
        {"precision": np.eye(2)},
        np.zeros(2),
    )

    dcm = DynamicCausalModel(2, 1, 1, random_seed=4)
    derivative = dcm.state_equation(np.zeros(2), 0.0, np.array([1.0]))
    observation = dcm.observation_equation(np.zeros(2))
    time_points = np.linspace(0.0, 0.1, 5)
    inputs = np.ones((len(time_points), 1)) * 0.1
    trajectory = dcm.integrate_dynamics(np.zeros(2), inputs, time_points)
    generated_observations = dcm.generate_observations(trajectory)
    estimates = dcm.estimate_parameters(generated_observations, inputs, time_points)
    dcm.set_parameters(estimates["A"], estimates["B"], estimates["C"])
    dcm.set_noise_parameters(np.eye(2) * 0.02, np.eye(1) * 0.02)

    mdp = MarkovDecisionProcess(3, 2, 2, random_seed=5)
    transition_prob = mdp.get_transition_prob(0, 0)
    observation_prob = mdp.get_observation_prob(0)
    next_state = mdp.transition(0, 0)
    observed = mdp.observe(next_state)
    simulated_states, simulated_observations = mdp.simulate(0, [0, 1], stochastic=False)
    predictive_state = mdp.get_predictive_state(np.array([0.5, 0.3, 0.2]), 0)
    predictive_observation = mdp.get_predictive_observation(predictive_state)
    posterior = mdp.update_belief(np.array([0.5, 0.3, 0.2]), 0)
    mdp.set_transition_matrix(0, 0, np.array([0.7, 0.2, 0.1]))
    mdp.set_observation_matrix(0, np.array([0.8, 0.2]))

    for label, value in {
        "prediction_error": prediction_error,
        "surprise": surprise,
        "elbo": elbo,
    }.items():
        _assert_finite(value, label)

    payload = {
        "categorical_update": categorical,
        "gaussian_update": gaussian,
        "prediction_error": prediction_error,
        "surprise": surprise,
        "dispatched_update": dispatched,
        "mean_field": mean_field,
        "mean_field_categorical": mf_cat,
        "mean_field_gaussian": mf_gauss,
        "structured_update": structured,
        "importance_sampling_mean": importance["mean"],
        "importance_sampling_precision": importance["precision"],
        "elbo": elbo,
        "dcm_derivative": derivative,
        "dcm_observation": observation,
        "dcm_trajectory_shape": trajectory.shape,
        "dcm_generated_observation_shape": generated_observations.shape,
        "dcm_estimate_keys": sorted(estimates.keys()),
        "mdp_transition_prob": transition_prob,
        "mdp_observation_prob": observation_prob,
        "mdp_next_state": next_state,
        "mdp_observed": observed,
        "mdp_simulated_states": simulated_states,
        "mdp_simulated_observations": simulated_observations,
        "mdp_predictive_state": predictive_state,
        "mdp_predictive_observation": predictive_observation,
        "mdp_posterior": posterior,
    }
    _write_json(output_dir / "inference_math_methods.json", payload)
    return payload


def audit_spatial_agent(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act import H3GridInferenceResult, SpatialActiveInferenceAgent
    from geo_infer_act.utils.h3_adapter import get_h3_adapter

    adapter = get_h3_adapter()
    center = adapter.latlng_to_cell(37.7749, -122.4194, 8)
    cells = [center, *list(adapter.grid_ring(center, 1))[:6]]
    agent = SpatialActiveInferenceAgent(
        initial_cells=cells,
        h3_resolution=8,
        state_dim=4,
        obs_dim=4,
        diffusion_rate=0.15,
        enable_logging=True,
    )
    observations = {cell: np.array([1.0, 0.0, 0.0, 0.0]) for cell in cells[:4]}
    preferences = {cell: np.array([0.7, 0.2, 0.05, 0.05]) for cell in cells[:2]}
    agent.set_preferences(preferences)
    agent.set_observation_model(cells[0], np.eye(4))
    transition_model = np.repeat(np.eye(4)[:, :, np.newaxis], agent.n_actions, axis=2)
    agent.set_transition_model(cells[0], transition_model)
    agent.update_precision(cells[0], 2.0)
    updated_beliefs = agent.spatial_perception(observations)
    action = agent.spatial_action()
    result = agent.step(observations, return_result=True)
    if not isinstance(result, H3GridInferenceResult):
        raise AssertionError(
            "SpatialActiveInferenceAgent.step did not return H3GridInferenceResult"
        )
    diagnostics = agent.get_diagnostics()
    export_path = output_dir / "spatial_agent_export.json"
    agent.export_results(str(export_path))
    agent.reset()

    _assert_finite(
        result.aggregate_free_energy,
        "SpatialActiveInferenceAgent.step.aggregate_free_energy",
    )
    payload = {
        "cell_count": len(cells),
        "updated_belief_count": len(updated_beliefs),
        "action": action,
        "result": result,
        "diagnostics": diagnostics,
        "export_path": export_path,
    }
    _write_json(output_dir / "spatial_agent_methods.json", payload)
    return payload


def audit_domain_models(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act.models.base import BaseActiveInferenceModel
    from geo_infer_act.models.base import CategoricalModel, GaussianModel
    from geo_infer_act.models.climate import ClimateModel
    from geo_infer_act.models.ecological import EcologicalModel
    from geo_infer_act.models.multi_agent import MultiAgentModel
    from geo_infer_act.models.resource import ResourceModel
    from geo_infer_act.models.urban import UrbanModel
    from geo_infer_act.runners.h3 import setup_san_francisco_boundary

    base_model = BaseActiveInferenceModel({"purpose": "audit"})
    try:
        base_model.step()
        raise AssertionError("BaseActiveInferenceModel.step() must be abstract")
    except NotImplementedError:
        pass

    categorical = CategoricalModel(state_dim=3, obs_dim=3)
    categorical.set_preferences(np.array([0.6, 0.3, 0.1]))
    categorical.set_transition_matrix(np.eye(3))
    categorical.set_likelihood_matrix(np.eye(3))
    categorical_beliefs = categorical.update_beliefs(np.array([1.0, 0.0, 0.0]))
    categorical_step = categorical.step()
    categorical_fe = categorical.compute_free_energy()
    categorical_reset = categorical.reset()

    gaussian = GaussianModel(state_dim=2, obs_dim=2)
    gaussian.set_preferences(np.zeros(2), np.eye(2))
    gaussian.set_transition_model(np.eye(2), Q=np.eye(2) * 0.01)
    gaussian.set_observation_model(np.eye(2), R=np.eye(2) * 0.01)
    gaussian_beliefs = gaussian.update_beliefs(np.array([0.1, -0.1]))
    gaussian_step = gaussian.step()
    gaussian_reset = gaussian.reset()

    climate = ClimateModel()
    climate_step = climate.step([0, 0])
    ecological = EcologicalModel()
    ecological_step = ecological.step([1, 0])
    resource = ResourceModel(n_resources=2, n_locations=3)
    resource_step = resource.step()
    resource_scores = resource.get_allocation_scores()
    resource_reset = resource.reset()
    urban = UrbanModel(n_agents=2, n_locations=4)
    urban_step = urban.step()
    urban_history = urban.run_simulation(n_steps=2)
    multi = MultiAgentModel(n_agents=2, n_resources=2, n_locations=3)
    multi_step = multi.step()
    multi_coordination = multi.coordinate_agents()
    multi_message = multi.get_agent_messages(0)
    multi.enable_h3_spatial(8, setup_san_francisco_boundary())
    h3_history = multi.simulate_h3_lattice(
        2,
        lambda _cell: np.array([1.0, 0.0, 0.0, 0.0]),
    )
    multi_h3_coordination = multi.coordinate_agents()
    multi_h3_message = multi.get_agent_messages(0)
    for label, value in {
        "categorical_free_energy": categorical_fe,
        "resource_free_energy": resource_step[0]["free_energy"],
    }.items():
        _assert_finite(value, label)

    payload = {
        "categorical_beliefs": categorical_beliefs,
        "categorical_step": categorical_step,
        "categorical_free_energy": categorical_fe,
        "categorical_reset": categorical_reset,
        "gaussian_beliefs": gaussian_beliefs,
        "gaussian_step": gaussian_step,
        "gaussian_reset": gaussian_reset,
        "climate_step": climate_step,
        "ecological_step": ecological_step,
        "resource_step": resource_step,
        "resource_scores": resource_scores,
        "resource_reset": resource_reset,
        "urban_step": urban_step,
        "urban_history": urban_history,
        "multi_step": multi_step,
        "multi_coordination": multi_coordination,
        "multi_message": multi_message,
        "multi_h3_cell_count": len(multi.h3_cells),
        "multi_h3_history_steps": len(h3_history),
        "multi_h3_coordination": multi_h3_coordination,
        "multi_h3_message": multi_h3_message,
    }
    _write_json(output_dir / "domain_model_methods.json", payload)
    return payload


def audit_api_interface(output_dir: Path) -> dict[str, Any]:
    from geo_infer_act.api.interface import ActiveInferenceInterface

    interface = ActiveInferenceInterface()
    interface.create_model(
        "audit_model",
        "categorical",
        {
            "state_dim": 3,
            "obs_dim": 3,
            "policy_selection_mode": "deterministic",
            "random_seed": 17,
        },
    )
    beliefs = interface.update_beliefs(
        "audit_model",
        {"observations": np.array([1.0, 0.0, 0.0])},
    )
    interface.set_preferences("audit_model", {"states": np.array([0.6, 0.3, 0.1])})
    policy = interface.select_policy("audit_model")
    free_energy = interface.get_free_energy("audit_model")
    _assert_finite(free_energy, "ActiveInferenceInterface.get_free_energy")

    payload = {
        "beliefs": beliefs,
        "policy": policy,
        "free_energy": free_energy,
        "model_count": len(interface.models),
    }
    _write_json(output_dir / "api_interface_methods.json", payload)
    return payload


def audit_visualization_methods(output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from geo_infer_act.core.generative_model import MarkovBlanket
    from geo_infer_act.utils.visualization import (
        BeliefVisualizer,
        plot_belief_update,
        plot_free_energy,
        plot_hierarchical_beliefs,
        plot_markov_blanket,
        plot_policies,
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    fig = plot_belief_update(
        {"states": np.array([0.4, 0.4, 0.2])},
        {"states": np.array([0.7, 0.2, 0.1])},
        title="Audit Belief Update",
    )
    path = figures_dir / "belief_update.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path)

    fig = plot_free_energy([2.0, 1.7, 1.4, 1.3], title="Audit Free Energy")
    path = figures_dir / "free_energy.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path)

    fig = plot_policies(
        np.array([0.65, 0.35]),
        policy_labels=["survey", "wait"],
        expected_free_energies=np.array([0.2, 0.6]),
        title="Audit Policy Selection",
    )
    path = figures_dir / "policies.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path)

    fig = plot_hierarchical_beliefs(
        {
            "level_0": {"states": np.array([0.7, 0.3])},
            "level_1": {"states": np.array([0.2, 0.5, 0.3])},
        }
    )
    path = figures_dir / "hierarchical_beliefs.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path)

    fig = plot_markov_blanket(
        MarkovBlanket(
            sensory_states=[0],
            active_states=[1],
            internal_states=[2],
            external_states=[3],
        )
    )
    path = figures_dir / "markov_blanket.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path)

    visualizer = BeliefVisualizer(figures_dir)
    path = figures_dir / "belief_visualizer_evolution.png"
    visualizer.plot_belief_evolution(
        [np.array([0.4, 0.4, 0.2]), np.array([0.7, 0.2, 0.1])],
        output_path=path.name,
    )
    artifacts.append(path)
    path = figures_dir / "belief_visualizer_free_energy.png"
    visualizer.plot_free_energy_trace([2.0, 1.7, 1.4, 1.3], path.name)
    artifacts.append(path)

    for artifact in artifacts:
        if not artifact.exists() or artifact.stat().st_size <= 0:
            raise AssertionError(f"Visualization artifact was not written: {artifact}")

    payload = {"artifacts": artifacts, "artifact_count": len(artifacts)}
    _write_json(output_dir / "visualization_methods.json", payload)
    return payload


def audit_scenario_outputs(output_dir: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    from geo_infer_act.runners import run_all_scenarios

    suite_dir = output_dir / "scenario_suite"
    result = run_all_scenarios(
        output_dir=suite_dir,
        seed=42,
        timesteps=8,
        deterministic=True,
        visualizations=True,
        command=["verify_comprehensive.py", "--output-dir", str(output_dir)],
    )

    errors: list[str] = []
    visualization_count = 0
    generated_file_count = 0
    scenario_summaries: list[dict[str, Any]] = []
    for scenario in result.scenario_results:
        manifest = scenario.manifest
        validation_status = manifest.get("validation", {}).get("status")
        if validation_status != "passed":
            errors.append(
                f"{scenario.scenario}: manifest validation is {validation_status}"
            )
        generated_file_count += len(manifest.get("generated_files", []))
        visualizations = [
            item
            for item in manifest.get("generated_files", [])
            if item.get("artifact_type") == "visualization"
        ]
        if not visualizations:
            errors.append(f"{scenario.scenario}: no visualization artifacts")
        visualization_count += len(visualizations)
        for item in visualizations:
            figure = scenario.output_dir / item["path"]
            metadata = scenario.output_dir / item.get("figure_metadata_path", "")
            data = scenario.output_dir / item.get("figure_data_path", "")
            if not figure.exists() or figure.stat().st_size <= 0:
                errors.append(f"{scenario.scenario}: missing figure {figure}")
            if not metadata.exists() or metadata.stat().st_size <= 0:
                errors.append(
                    f"{scenario.scenario}: missing metadata sidecar for {figure.name}"
                )
            if not data.exists() or data.stat().st_size <= 0:
                errors.append(
                    f"{scenario.scenario}: missing data sidecar for {figure.name}"
                )
            if figure.suffix == ".png":
                image = Image.open(figure)
                if "geo_infer_act_metadata" not in image.info:
                    errors.append(
                        f"{scenario.scenario}: PNG lacks embedded ACT metadata: {figure.name}"
                    )
                if ImageStat.Stat(image.convert("L")).stddev[0] < 1.0:
                    errors.append(
                        f"{scenario.scenario}: PNG appears blank: {figure.name}"
                    )
            if figure.suffix == ".html":
                html = figure.read_text(errors="ignore")
                if "geo-infer-act-figure-metadata" not in html:
                    errors.append(
                        f"{scenario.scenario}: HTML lacks embedded ACT metadata: {figure.name}"
                    )
        scenario_summaries.append(
            {
                "scenario": scenario.scenario,
                "manifest": scenario.manifest_path,
                "generated_files": len(manifest.get("generated_files", [])),
                "visualizations": len(visualizations),
                "metrics": scenario.metrics,
            }
        )

    if errors:
        raise AssertionError("; ".join(errors))

    payload = {
        "suite_manifest": result.manifest_path,
        "suite_validation": result.manifest.get("validation", {}),
        "scenario_count": len(result.scenario_results),
        "generated_file_count": generated_file_count,
        "visualization_count": visualization_count,
        "scenarios": scenario_summaries,
    }
    _write_json(output_dir / "scenario_output_audit.json", payload)
    return payload


def audit_docs_and_mermaid(output_dir: Path) -> dict[str, Any]:
    markdown_files = sorted(
        path for path in ACT_ROOT.rglob("*.md") if ".pytest_cache" not in path.parts
    )
    mermaid_dir = output_dir / "mermaid"
    mermaid_dir.mkdir(parents=True, exist_ok=True)
    mmdc = shutil.which("mmdc")
    mermaid_results = []
    local_link_errors = []

    for markdown_file in markdown_files:
        text = markdown_file.read_text(errors="ignore")
        for index, block in enumerate(
            re.findall(r"```mermaid\n(.*?)\n```", text, flags=re.DOTALL),
            start=1,
        ):
            safe_name = (
                str(markdown_file.relative_to(ACT_ROOT))
                .replace("/", "__")
                .replace(".", "_")
            )
            source = mermaid_dir / f"{safe_name}_{index}.mmd"
            rendered = mermaid_dir / f"{safe_name}_{index}.svg"
            source.write_text(block + "\n")
            entry = {
                "markdown": markdown_file.relative_to(REPO_ROOT),
                "block": index,
                "source": source,
                "renderer": mmdc,
            }
            if mmdc:
                proc = subprocess.run(
                    [mmdc, "-i", str(source), "-o", str(rendered), "-b", "transparent"],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                entry["status"] = "passed" if proc.returncode == 0 else "failed"
                entry["rendered"] = rendered if proc.returncode == 0 else None
                entry["stderr"] = proc.stderr[-1000:] if proc.returncode != 0 else ""
                if proc.returncode != 0:
                    local_link_errors.append(
                        f"Mermaid render failed for {markdown_file}:{index}: {proc.stderr[-300:]}"
                    )
            else:
                entry["status"] = "skipped"
                entry["reason"] = "mmdc not found"
            mermaid_results.append(entry)

        for match in re.finditer(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0].strip()
            if not target or target.startswith("<"):
                continue
            target = target.replace("%20", " ")
            if target.startswith("../"):
                candidate = (markdown_file.parent / target).resolve()
            else:
                candidate = (markdown_file.parent / target).resolve()
            if not candidate.exists():
                local_link_errors.append(
                    f"Missing local link from {markdown_file.relative_to(REPO_ROOT)} to {target}"
                )

    readme_files = [
        path.relative_to(REPO_ROOT)
        for path in markdown_files
        if path.name == "README.md"
    ]
    payload = {
        "markdown_file_count": len(markdown_files),
        "readme_count": len(readme_files),
        "readmes": readme_files,
        "mermaid_block_count": len(mermaid_results),
        "mermaid_results": mermaid_results,
        "local_link_errors": local_link_errors,
    }
    _write_json(output_dir / "docs_mermaid_audit.json", payload)
    if local_link_errors:
        raise AssertionError("; ".join(local_link_errors))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comprehensive GEO-INFER-ACT method, output, docs, and Mermaid verification."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ACT_ROOT / "examples" / "output" / "comprehensive_act_audit",
        help="Logged output directory for data, manifests, visualizations, and audit summaries.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove the output directory before writing new audit artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "comprehensive_audit.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    start = time.time()
    sections = [
        ("active_inference_model", audit_active_inference_model),
        ("generative_model", audit_generative_model),
        ("free_energy_and_policy", audit_free_energy_and_policy),
        ("inference_math", audit_inference_math),
        ("spatial_agent", audit_spatial_agent),
        ("domain_models", audit_domain_models),
        ("api_interface", audit_api_interface),
        ("visualization_methods", audit_visualization_methods),
        ("scenario_outputs", audit_scenario_outputs),
        ("docs_and_mermaid", audit_docs_and_mermaid),
    ]
    results = [_section(name, func, output_dir) for name, func in sections]
    failed = [result for result in results if result["status"] != "passed"]
    summary = {
        "schema_version": "geo-infer-act-comprehensive-audit/v1",
        "status": "failed" if failed else "passed",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - start, 4),
        "output_dir": output_dir,
        "section_count": len(results),
        "failed_sections": [item["name"] for item in failed],
        "sections": results,
    }
    _write_json(output_dir / "comprehensive_audit_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
