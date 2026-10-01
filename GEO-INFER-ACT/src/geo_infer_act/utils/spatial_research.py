"""Research-profile models and statistics for spatial H3 active inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


RESEARCH_STATISTICS_SCHEMA_VERSION = "geo-infer-act-spatial-research-statistics/v1"


def apply_h3_research_profile(
    model: Any,
    active_model: Any | None = None,
    *,
    action_count: int = 4,
) -> None:
    """
    Configure a categorical H3 model for non-degenerate research diagnostics.

    The default categorical model intentionally starts with uniform likelihoods,
    transitions, and preferences. That is useful as a neutral baseline, but it
    makes flat H3 traces visually uninformative. This opt-in profile keeps real
    H3 cells and real pymdp inference while installing a soft identity
    likelihood, asymmetric preferences, and action-conditioned transitions.
    """
    state_dim = int(getattr(model, "parameters", {}).get("state_dim", 4))
    obs_dim = int(getattr(model, "parameters", {}).get("obs_dim", state_dim))
    if state_dim != obs_dim:
        state_dim = obs_dim = min(state_dim, obs_dim)
    action_count = max(2, int(action_count))

    off_diag = 0.04 / max(1, obs_dim - 1)
    observation_model = np.full((obs_dim, state_dim), off_diag, dtype=float)
    np.fill_diagonal(observation_model, 0.96)

    transition_model = np.zeros((state_dim, state_dim, action_count), dtype=float)
    for action_index in range(action_count):
        transition = np.eye(state_dim, dtype=float) * 0.58
        for source_state in range(state_dim):
            target_state = (
                source_state
                if action_index == 0
                else (source_state + action_index) % state_dim
            )
            transition[target_state, source_state] += 0.34
            transition[:, source_state] += 0.08 / state_dim
        transition_model[:, :, action_index] = transition / np.sum(
            transition, axis=0, keepdims=True
        )

    preference_template = np.array([-0.45, -0.15, 0.65, 0.35], dtype=float)
    prior_template = np.array([0.20, 0.25, 0.30, 0.25], dtype=float)
    preferences = np.resize(preference_template, obs_dim)
    prior = _normalize(np.resize(prior_template, state_dim))

    model.observation_model = observation_model
    model.transition_model = transition_model
    model.preferences = {"observations": preferences}
    model.beliefs = {"states": prior.copy()}
    model.parameters["num_controls"] = action_count
    model.parameters["research_profile"] = True

    if active_model is not None:
        active_model.parameters["num_controls"] = action_count
        active_model.current_beliefs = {"states": prior.copy()}


def apply_spatial_agent_research_profile(
    agent: Any,
    *,
    action_count: int = 4,
) -> None:
    """Configure a ``SpatialActiveInferenceAgent`` with research-profile matrices."""
    action_count = max(2, int(action_count))
    state_dim = int(getattr(agent, "state_dim", 4))
    obs_dim = int(getattr(agent, "obs_dim", state_dim))
    n_cells = len(getattr(agent, "cells", []) or [])
    off_diag = 0.04 / max(1, obs_dim - 1)
    observation = np.full((obs_dim, state_dim), off_diag, dtype=float)
    np.fill_diagonal(observation, 0.96)

    transition = np.zeros((state_dim, state_dim, action_count), dtype=float)
    for action_index in range(action_count):
        matrix = np.eye(state_dim, dtype=float) * 0.58
        for source_state in range(state_dim):
            target_state = (
                source_state
                if action_index == 0
                else (source_state + action_index) % state_dim
            )
            matrix[target_state, source_state] += 0.34
            matrix[:, source_state] += 0.08 / state_dim
        transition[:, :, action_index] = matrix / np.sum(matrix, axis=0, keepdims=True)

    preference_template = np.array([-0.45, -0.15, 0.65, 0.35], dtype=float)
    prior_template = np.array([0.20, 0.25, 0.30, 0.25], dtype=float)
    preferences = np.resize(preference_template, obs_dim)
    prior = _normalize(np.resize(prior_template, state_dim))

    agent.n_actions = action_count
    agent.observation_model = np.repeat(observation[None, :, :], n_cells, axis=0)
    agent.transition_model = np.repeat(transition[None, :, :, :], n_cells, axis=0)
    agent.preferences = np.repeat(preferences[None, :], n_cells, axis=0)
    if n_cells:
        agent.beliefs = np.repeat(prior[None, :], n_cells, axis=0)
    agent.research_profile = True


def build_spatial_research_statistics(
    cell_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    level_rows: Sequence[Mapping[str, Any]],
    parent_child_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe run statistics for spatial H3 trace diagnostics."""
    parent_child_rows = list(parent_child_rows or [])
    leaf_rows: Sequence[Mapping[str, Any]] = [
        dict(row)
        for row in cell_rows
        if not _truthy(row.get("aggregate_parent_cell", False))
    ]
    parent_rows: Sequence[Mapping[str, Any]] = [
        dict(row)
        for row in cell_rows
        if _truthy(row.get("aggregate_parent_cell", False))
    ]
    timesteps = sorted({_int(row.get("timestep")) for row in leaf_rows})
    cells = sorted({str(row.get("cell")) for row in leaf_rows if row.get("cell")})

    metric_names = [
        "entropy",
        "free_energy",
        "expected_free_energy",
        "policy_entropy",
        "selected_action_probability",
        "local_coherence",
        "belief_flux_divergence",
        "posterior_delta",
    ]
    metric_summaries = {
        metric: _summary(_numeric_values(leaf_rows, metric)) for metric in metric_names
    }
    temporal_slopes = {
        f"mean_{metric}": _temporal_slope(leaf_rows, metric) for metric in metric_names
    }

    selected_actions = [
        str(row.get("selected_action_index", row.get("selected_action", "")))
        for row in leaf_rows
    ]
    action_counts = dict(Counter(selected_actions))
    dominant_action, dominant_count = _dominant_count(action_counts)
    switch_count = _policy_switch_count(leaf_rows)

    spatial_graph = _graph_statistics(leaf_rows, edge_rows)
    nested = _nested_statistics(parent_child_rows, parent_rows, level_rows)

    return {
        "schema_version": RESEARCH_STATISTICS_SCHEMA_VERSION,
        "cell_count": len(cells),
        "timestep_count": len(timesteps),
        "row_count": len(leaf_rows),
        "metric_summaries": metric_summaries,
        "temporal_slopes": temporal_slopes,
        "policy": {
            "action_counts": action_counts,
            "dominant_action": dominant_action,
            "dominant_action_share": (
                float(dominant_count / len(selected_actions))
                if selected_actions
                else 0.0
            ),
            "switch_count": int(switch_count),
            "switch_rate": (
                float(switch_count / max(1, len(leaf_rows) - len(cells)))
                if leaf_rows
                else 0.0
            ),
        },
        "spatial_graph": spatial_graph,
        "nested": nested,
        "non_degenerate": {
            "entropy_std": metric_summaries["entropy"]["std"],
            "selected_action_probability_std": metric_summaries[
                "selected_action_probability"
            ]["std"],
            "local_coherence_std": metric_summaries["local_coherence"]["std"],
            "belief_flux_divergence_std": metric_summaries["belief_flux_divergence"][
                "std"
            ],
            "unique_selected_action_count": len(action_counts),
        },
    }


def statistics_summary_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten research statistics into compact report table rows."""
    rows: list[dict[str, Any]] = []
    for metric, summary in statistics.get("metric_summaries", {}).items():
        rows.append(
            {
                "group": "metric",
                "metric": metric,
                "mean": _float(summary.get("mean")),
                "std": _float(summary.get("std")),
                "min": _float(summary.get("min")),
                "max": _float(summary.get("max")),
                "value": _float(summary.get("mean")),
            }
        )
    policy = statistics.get("policy", {})
    rows.extend(
        [
            {
                "group": "policy",
                "metric": "switch_count",
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "value": _float(policy.get("switch_count")),
            },
            {
                "group": "policy",
                "metric": "dominant_action_share",
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "value": _float(policy.get("dominant_action_share")),
            },
        ]
    )
    for metric, value in statistics.get("spatial_graph", {}).items():
        rows.append(
            {
                "group": "spatial_graph",
                "metric": metric,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "value": _float(value),
            }
        )
    for metric, value in statistics.get("nested", {}).items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            rows.append(
                {
                    "group": "nested",
                    "metric": metric,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "value": _float(value),
                }
            )
    return rows


def _graph_statistics(
    leaf_rows: Sequence[Mapping[str, Any]], edge_rows: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    entropy_by_cell_time = {
        (str(row.get("cell")), _int(row.get("timestep"))): _float(row.get("entropy"))
        for row in leaf_rows
    }
    timesteps = sorted({time for _cell, time in entropy_by_cell_time})
    flux_balance = []
    for timestep in timesteps:
        values = [
            _float(row.get("belief_flux_divergence"))
            for row in leaf_rows
            if _int(row.get("timestep")) == timestep
        ]
        if values:
            flux_balance.append(abs(float(np.sum(values))))

    neighbor_contrasts = []
    for edge in edge_rows:
        timestep = _int(edge.get("timestep"))
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        key_a = (source, timestep)
        key_b = (target, timestep)
        if key_a in entropy_by_cell_time and key_b in entropy_by_cell_time:
            neighbor_contrasts.append(
                abs(entropy_by_cell_time[key_a] - entropy_by_cell_time[key_b])
            )

    return {
        "mean_neighbor_entropy_contrast": _mean(neighbor_contrasts),
        "mean_edge_belief_distance": _mean(
            _numeric_values(edge_rows, "belief_distance")
        ),
        "mean_edge_coherence": _mean(_numeric_values(edge_rows, "coherence")),
        "moran_entropy_proxy": _moran_proxy(leaf_rows, edge_rows, "entropy"),
        "mean_abs_flux_balance": _mean(flux_balance),
    }


def _nested_statistics(
    parent_child_rows: Sequence[Mapping[str, Any]],
    parent_rows: Sequence[Mapping[str, Any]],
    level_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    residuals = _numeric_values(parent_child_rows, "cross_level_residual")
    consistency = _numeric_values(parent_child_rows, "cross_level_consistency")
    level_summary: dict[str, dict[str, float]] = {}
    for row in level_rows:
        resolution = str(row.get("resolution"))
        bucket = level_summary.setdefault(
            resolution,
            {
                "mean_entropy": 0.0,
                "mean_free_energy": 0.0,
                "mean_policy_entropy": 0.0,
                "count": 0.0,
            },
        )
        bucket["mean_entropy"] += _float(row.get("mean_entropy"))
        bucket["mean_free_energy"] += _float(row.get("mean_free_energy"))
        bucket["mean_policy_entropy"] += _float(row.get("mean_policy_entropy"))
        bucket["count"] += 1.0
    for bucket in level_summary.values():
        count = max(1.0, bucket.pop("count"))
        for key in list(bucket):
            bucket[key] = float(bucket[key] / count)

    return {
        "parent_child_count": len(parent_child_rows),
        "parent_count": len({str(row.get("cell")) for row in parent_rows}),
        "mean_parent_child_residual": _mean(residuals),
        "max_parent_child_residual": float(max(residuals)) if residuals else 0.0,
        "mean_cross_level_consistency": _mean(consistency),
        "cross_level_consistency_slope": _temporal_slope(
            parent_child_rows, "cross_level_consistency"
        ),
        "parent_aggregate_drift": _mean(
            _numeric_values(parent_rows, "posterior_delta")
        ),
        "level_summaries": level_summary,
    }


def _moran_proxy(
    rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> float:
    by_time: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_time[_int(row.get("timestep"))][str(row.get("cell"))] = _float(
            row.get(metric)
        )
    values = []
    for timestep, cell_values in by_time.items():
        if len(cell_values) < 2:
            continue
        mean = float(np.mean(list(cell_values.values())))
        variance = float(
            np.mean([(value - mean) ** 2 for value in cell_values.values()])
        )
        if variance <= 1e-12:
            continue
        products = []
        for edge in edge_rows:
            if _int(edge.get("timestep")) != timestep:
                continue
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source in cell_values and target in cell_values:
                products.append(
                    (cell_values[source] - mean) * (cell_values[target] - mean)
                )
        if products:
            values.append(float(np.mean(products) / variance))
    return _mean(values)


def _policy_switch_count(rows: Sequence[Mapping[str, Any]]) -> int:
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row.get("cell"))].append(row)
    switches = 0
    for cell_rows in by_cell.values():
        ordered = sorted(cell_rows, key=lambda item: _int(item.get("timestep")))
        actions = [
            str(item.get("selected_action_index", item.get("selected_action", "")))
            for item in ordered
        ]
        switches += sum(
            1 for previous, current in zip(actions, actions[1:]) if previous != current
        )
    return switches


def _temporal_slope(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    by_time: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = _maybe_float(row.get(metric))
        if value is not None:
            by_time[_int(row.get("timestep"))].append(value)
    points = sorted(
        (timestep, _mean(values)) for timestep, values in by_time.items() if values
    )
    if len(points) < 2:
        return 0.0
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    if np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _numeric_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _maybe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _dominant_count(counts: Mapping[str, int]) -> tuple[str, int]:
    if not counts:
        return "", 0
    action, count = max(counts.items(), key=lambda item: item[1])
    return str(action), int(count)


def _normalize(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if total <= 1e-12:
        return np.ones_like(array) / array.size
    return array / total


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def _maybe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _float(value: Any) -> float:
    numeric = _maybe_float(value)
    return float(numeric) if numeric is not None else 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
