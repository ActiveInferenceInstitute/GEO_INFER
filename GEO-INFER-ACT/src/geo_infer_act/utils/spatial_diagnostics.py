#!/usr/bin/env python
"""
Spatial Diagnostics Utilities for GEO-INFER-ACT.

This module provides specialized diagnostic and analysis tools for spatial
active inference, including spatial coherence metrics, autocorrelation,
and free energy landscape analysis.

Features:
- H3 neighbor coherence metrics
- Spatial autocorrelation (Moran's I approximation)
- Free energy landscape analysis
- Structured JSON/CSV export
"""

import numpy as np
import logging
import json
import csv
from typing import Any
from collections.abc import Mapping, Iterable
from datetime import datetime
from pathlib import Path

from geo_infer_act.core.types import (
    H3CellDiagnostics,
    H3EdgeDiagnostics,
    H3LevelDiagnostics,
    SpatialInferenceTrace,
)
from geo_infer_act.utils.h3_adapter import (
    edge_count_from_graph,
    get_h3_adapter,
    normalize_belief_vector,
)

logger = logging.getLogger(__name__)


class SpatialDiagnostics:
    """
    Comprehensive spatial diagnostics for active inference agents.

    Provides metrics for analyzing spatial belief coherence,
    information flow, and agent performance in H3 grids.
    """

    def __init__(self, output_dir: str | None = None):
        """
        Initialize spatial diagnostics.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: list[dict] = []
        self.analysis_cache: dict[str, Any] = {}

    def compute_spatial_coherence(
        self, beliefs: np.ndarray, neighbor_matrix: np.ndarray
    ) -> dict[str, Any]:
        """
        Compute spatial coherence metrics.

        Measures how consistent beliefs are across neighboring cells.

        Args:
            beliefs: Array of shape (n_cells, state_dim)
            neighbor_matrix: Binary adjacency matrix (n_cells, n_cells)

        Returns:
            Dict with coherence metrics
        """
        n_cells = beliefs.shape[0]

        if neighbor_matrix.shape != (n_cells, n_cells):
            return {"error": "dimension_mismatch"}

        # Local coherence: similarity to immediate neighbors
        local_coherences = []
        for i in range(n_cells):
            neighbors = np.where(neighbor_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_beliefs = beliefs[neighbors]
                mean_neighbor = np.mean(neighbor_beliefs, axis=0)
                coherence = 1 - np.mean(np.abs(beliefs[i] - mean_neighbor))
                local_coherences.append(coherence)

        # Global coherence: overall belief similarity
        belief_variance = np.var(beliefs)
        global_coherence = 1.0 / (1.0 + belief_variance)

        return {
            "local_mean": float(np.mean(local_coherences)) if local_coherences else 0.0,
            "local_std": float(np.std(local_coherences)) if local_coherences else 0.0,
            "local_min": float(np.min(local_coherences)) if local_coherences else 0.0,
            "local_max": float(np.max(local_coherences)) if local_coherences else 0.0,
            "global": float(global_coherence),
            "n_cells": n_cells,
        }

    @staticmethod
    def build_h3_trace(
        *,
        scenario: str,
        timestep: int,
        cell_results: Mapping[str, Any],
        neighbor_map: Mapping[str, Iterable[str]] | None = None,
        previous_beliefs: Mapping[str, Any] | None = None,
        hierarchy: Mapping[str, Any] | None = None,
        parent_beliefs: Mapping[str, Any] | None = None,
        backend_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SpatialInferenceTrace:
        """
        Build typed H3 trace diagnostics from cell inference results.

        The trace is deterministic and JSON-safe. It includes observed leaf-cell
        diagnostics, optional parent-cell aggregate diagnostics for nested H3,
        same-resolution edge diagnostics, and per-resolution summaries.
        """
        adapter = get_h3_adapter()
        timestep = int(timestep)
        previous_beliefs = previous_beliefs or {}
        parent_beliefs = parent_beliefs or {}
        hierarchy = hierarchy or {}
        child_parent_map = {
            str(child): str(parent)
            for child, parent in hierarchy.get("child_parent_map", {}).items()
        }

        result_beliefs: dict[str, np.ndarray] = {}
        result_payloads: dict[str, Any] = {}
        for cell, result in sorted(cell_results.items()):
            cell = str(cell)
            result_beliefs[cell] = SpatialDiagnostics._belief_from_result(result)
            result_payloads[cell] = result

        aggregate_parent_beliefs = {
            str(cell): _trace_belief_vector(value)
            for cell, value in sorted(parent_beliefs.items())
        }
        all_beliefs: dict[str, np.ndarray] = {
            **aggregate_parent_beliefs,
            **result_beliefs,
        }

        graph = SpatialDiagnostics._same_resolution_graph(
            all_beliefs.keys(),
            neighbor_map=neighbor_map,
            hierarchy=hierarchy,
        )
        local_metrics = SpatialDiagnostics._local_belief_metrics(
            all_beliefs,
            graph,
            previous_beliefs,
        )
        edge_diagnostics = SpatialDiagnostics._edge_diagnostics(
            all_beliefs,
            graph,
            timestep,
        )

        cross_level_scores: dict[str, float] = {}
        for child, parent in child_parent_map.items():
            if child not in all_beliefs or parent not in all_beliefs:
                continue
            distance = float(
                np.linalg.norm(
                    _trace_belief_vector(all_beliefs[child])
                    - _trace_belief_vector(all_beliefs[parent])
                )
            )
            cross_level_scores[child] = _finite_float(1.0 / (1.0 + distance))

        cell_diagnostics: list[H3CellDiagnostics] = []
        for cell in sorted(all_beliefs):
            belief = _trace_belief_vector(all_beliefs[cell])
            result = result_payloads.get(cell)
            pymdp = SpatialDiagnostics._pymdp_metadata_from_result(result)
            posterior = _normalize_optional_distribution(
                pymdp.get("action_posterior", [])
            )
            neg_efe = [
                _finite_float(value)
                for value in pymdp.get("negative_expected_free_energy", [])
            ]
            selected = int(pymdp.get("selected_action_index", 0))
            selected = selected % max(1, len(posterior) or len(neg_efe) or 1)
            selected_probability = float(posterior[selected]) if posterior else 0.0
            selected_negative_efe = float(neg_efe[selected]) if neg_efe else 0.0
            expected_free_energy = SpatialDiagnostics._expected_fe_from_result(
                result,
                selected_negative_efe,
            )
            parent_cell = child_parent_map.get(cell)
            cell_meta = {
                "pymdp_version": pymdp.get("pymdp_version"),
                "h3_version": pymdp.get("h3_version"),
                "h3_c_version": pymdp.get("h3_c_version"),
                "backend": pymdp.get("backend"),
                "cross_level_consistency": cross_level_scores.get(cell),
                "aggregate_parent_cell": cell in aggregate_parent_beliefs
                and cell not in result_payloads,
            }
            try:
                resolution = adapter.get_resolution(cell)
            except Exception:
                resolution = -1
            metric = local_metrics[cell]
            cell_diagnostics.append(
                H3CellDiagnostics(
                    cell=cell,
                    timestep=timestep,
                    resolution=int(resolution),
                    belief=[float(value) for value in belief],
                    entropy=_entropy(belief),
                    free_energy=SpatialDiagnostics._free_energy_from_result(
                        result,
                        belief,
                    ),
                    expected_free_energy=expected_free_energy,
                    selected_action=(
                        getattr(result, "action", None) if result is not None else None
                    ),
                    selected_action_index=selected,
                    selected_action_probability=selected_probability,
                    action_posterior=posterior,
                    negative_expected_free_energy=neg_efe,
                    selected_negative_expected_free_energy=selected_negative_efe,
                    policy_entropy=_entropy(posterior) if posterior else 0.0,
                    neighbor_count=int(metric["neighbor_count"]),
                    local_coherence=metric["local_coherence"],
                    posterior_delta=metric["posterior_delta"],
                    belief_flux_in=metric["belief_flux_in"],
                    belief_flux_out=metric["belief_flux_out"],
                    belief_flux_divergence=metric["belief_flux_divergence"],
                    parent_cell=parent_cell,
                    metadata={
                        key: value
                        for key, value in cell_meta.items()
                        if value is not None
                    },
                )
            )

        level_diagnostics = SpatialDiagnostics._level_diagnostics(
            cell_diagnostics,
            edge_diagnostics,
            timestep,
        )
        trace_backend = {
            **SpatialDiagnostics._first_pymdp_metadata(cell_results),
            **dict(backend_metadata or {}),
        }
        hierarchy_metadata = {
            "nested_h3": bool(hierarchy),
            "resolutions": list(hierarchy.get("resolutions", [])),
            "parent_count": len(hierarchy.get("parent_child_map", {})),
            "child_count": len(hierarchy.get("child_parent_map", {})),
            "orphan_count": hierarchy.get("validation", {}).get("orphan_count", 0),
        }
        return SpatialInferenceTrace(
            scenario=str(scenario),
            timesteps=[timestep],
            cell_diagnostics=cell_diagnostics,
            edge_diagnostics=edge_diagnostics,
            level_diagnostics=level_diagnostics,
            hierarchy_metadata=hierarchy_metadata,
            backend_metadata=trace_backend,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _belief_from_result(result: Any) -> np.ndarray:
        """Extract a normalized belief vector from a typed or dict result."""
        if result is None:
            return np.ones(1, dtype=float)
        if hasattr(result, "beliefs"):
            return _trace_belief_vector(result.beliefs)
        if isinstance(result, Mapping) and "beliefs" in result:
            return _trace_belief_vector(result["beliefs"])
        return _trace_belief_vector(result)

    @staticmethod
    def _pymdp_metadata_from_result(result: Any) -> dict[str, Any]:
        """Extract pymdp metadata from a cell result."""
        if result is None:
            return {}
        metadata = getattr(result, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            pymdp = metadata.get("pymdp") or {}
            return dict(pymdp) if isinstance(pymdp, Mapping) else {}
        if isinstance(result, Mapping):
            pymdp = result.get("pymdp") or {}
            return dict(pymdp) if isinstance(pymdp, Mapping) else {}
        return {}

    @staticmethod
    def _free_energy_from_result(result: Any, belief: np.ndarray) -> float:
        """Return result free energy or a finite KL-to-uniform aggregate."""
        if result is not None and hasattr(result, "free_energy"):
            return _finite_float(result.free_energy)
        uniform = np.ones_like(belief, dtype=float) / max(1, belief.size)
        return _finite_float(np.sum(belief * np.log((belief + 1e-12) / uniform)))

    @staticmethod
    def _expected_fe_from_result(result: Any, selected_negative_efe: float) -> float:
        """Return expected free energy from a result or selected negative EFE."""
        if (
            result is not None
            and getattr(result, "expected_free_energy", None) is not None
        ):
            return _finite_float(result.expected_free_energy)
        return _finite_float(-selected_negative_efe)

    @staticmethod
    def _same_resolution_graph(
        cells: Iterable[str],
        *,
        neighbor_map: Mapping[str, Iterable[str]] | None,
        hierarchy: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        """Build or normalize a same-resolution H3 neighbor graph."""
        adapter = get_h3_adapter()
        cell_set = {str(cell) for cell in cells}
        graph: dict[str, set[str]] = {cell: set() for cell in cell_set}
        if hierarchy.get("same_level_neighbors"):
            for level_neighbors in hierarchy.get("same_level_neighbors", {}).values():
                for cell, neighbors in level_neighbors.items():
                    cell = str(cell)
                    if cell not in cell_set:
                        continue
                    graph[cell].update(
                        str(neighbor)
                        for neighbor in neighbors
                        if str(neighbor) in cell_set
                    )
        elif neighbor_map:
            for cell, neighbors in neighbor_map.items():
                cell = str(cell)
                if cell not in cell_set:
                    continue
                graph[cell].update(
                    str(neighbor) for neighbor in neighbors if str(neighbor) in cell_set
                )
        else:
            for cell in sorted(cell_set):
                try:
                    graph[cell].update(
                        neighbor
                        for neighbor in adapter.grid_ring(cell, 1)
                        if neighbor in cell_set
                    )
                except Exception:
                    continue

        filtered: dict[str, set[str]] = {cell: set() for cell in cell_set}
        for cell, neighbors in graph.items():
            try:
                resolution = adapter.get_resolution(cell)
            except Exception:
                resolution = None
            for neighbor in neighbors:
                if neighbor == cell or neighbor not in cell_set:
                    continue
                try:
                    same_resolution = (
                        resolution is None
                        or adapter.get_resolution(neighbor) == resolution
                    )
                except Exception:
                    same_resolution = True
                if same_resolution:
                    filtered[cell].add(neighbor)
                    filtered.setdefault(neighbor, set()).add(cell)
        return filtered

    @staticmethod
    def _local_belief_metrics(
        beliefs: Mapping[str, np.ndarray],
        graph: Mapping[str, Iterable[str]],
        previous_beliefs: Mapping[str, Any],
    ) -> dict[str, dict[str, float]]:
        """Compute local coherence, posterior delta, and belief flux by cell."""
        normalized = {
            cell: _trace_belief_vector(belief) for cell, belief in beliefs.items()
        }
        previous = {
            str(cell): _trace_belief_vector(value)
            for cell, value in previous_beliefs.items()
            if str(cell) in normalized
        }
        metrics: dict[str, dict[str, float]] = {}
        entropies = {cell: _entropy(belief) for cell, belief in normalized.items()}
        for cell, belief in normalized.items():
            neighbors = [
                neighbor for neighbor in graph.get(cell, []) if neighbor in normalized
            ]
            distances = [
                float(np.linalg.norm(belief - normalized[neighbor]))
                for neighbor in neighbors
            ]
            coherence = (
                _finite_float(1.0 / (1.0 + float(np.mean(distances))))
                if distances
                else 1.0
            )
            previous_belief = previous.get(cell)
            posterior_delta = (
                _finite_float(np.linalg.norm(belief - previous_belief, ord=1))
                if previous_belief is not None and previous_belief.shape == belief.shape
                else 0.0
            )
            flux_in = 0.0
            flux_out = 0.0
            entropy_value = entropies[cell]
            for neighbor in neighbors:
                distance = float(np.linalg.norm(belief - normalized[neighbor]))
                entropy_gap = entropies[neighbor] - entropy_value
                if entropy_gap > 0:
                    flux_in += abs(entropy_gap) * distance
                else:
                    flux_out += abs(entropy_gap) * distance
            metrics[cell] = {
                "neighbor_count": int(len(neighbors)),
                "local_coherence": coherence,
                "posterior_delta": posterior_delta,
                "belief_flux_in": _finite_float(flux_in),
                "belief_flux_out": _finite_float(flux_out),
                "belief_flux_divergence": _finite_float(flux_out - flux_in),
            }
        return metrics

    @staticmethod
    def _edge_diagnostics(
        beliefs: Mapping[str, np.ndarray],
        graph: Mapping[str, Iterable[str]],
        timestep: int,
    ) -> list[H3EdgeDiagnostics]:
        """Return one undirected edge diagnostic per same-resolution edge."""
        adapter = get_h3_adapter()
        normalized = {
            cell: _trace_belief_vector(belief) for cell, belief in beliefs.items()
        }
        rows: list[H3EdgeDiagnostics] = []
        seen: set[tuple[str, str]] = set()
        for source, neighbors in graph.items():
            if source not in normalized:
                continue
            for target in neighbors:
                if target not in normalized:
                    continue
                pair = (str(source), str(target))
                if pair[0] > pair[1]:
                    pair = (pair[1], pair[0])
                if pair in seen:
                    continue
                seen.add(pair)
                distance = _finite_float(
                    np.linalg.norm(normalized[pair[0]] - normalized[pair[1]])
                )
                try:
                    resolution = adapter.get_resolution(pair[0])
                except Exception:
                    resolution = -1
                rows.append(
                    H3EdgeDiagnostics(
                        source=pair[0],
                        target=pair[1],
                        timestep=int(timestep),
                        resolution=int(resolution),
                        belief_distance=distance,
                        coherence=_finite_float(1.0 / (1.0 + distance)),
                        source_entropy=_entropy(normalized[pair[0]]),
                        target_entropy=_entropy(normalized[pair[1]]),
                    )
                )
        return rows

    @staticmethod
    def _level_diagnostics(
        cells: list[H3CellDiagnostics],
        edges: list[H3EdgeDiagnostics],
        timestep: int,
    ) -> list[H3LevelDiagnostics]:
        """Aggregate per-resolution diagnostics from cell and edge rows."""
        by_resolution: dict[int, list[H3CellDiagnostics]] = {}
        for cell in cells:
            by_resolution.setdefault(int(cell.resolution), []).append(cell)
        edge_counts = {
            resolution: edge_count_from_graph(
                {
                    edge.source: [edge.target]
                    for edge in edges
                    if int(edge.resolution) == resolution
                }
            )
            for resolution in by_resolution
        }
        rows: list[H3LevelDiagnostics] = []
        for resolution, level_cells in sorted(by_resolution.items()):
            cross_values = [
                float(value)
                for cell in level_cells
                if (value := cell.metadata.get("cross_level_consistency")) is not None
            ]
            rows.append(
                H3LevelDiagnostics(
                    resolution=resolution,
                    timestep=int(timestep),
                    cell_count=len(level_cells),
                    edge_count=int(edge_counts.get(resolution, 0)),
                    mean_entropy=_mean([cell.entropy for cell in level_cells]),
                    mean_free_energy=_mean([cell.free_energy for cell in level_cells]),
                    mean_expected_free_energy=_mean(
                        [cell.expected_free_energy for cell in level_cells]
                    ),
                    mean_policy_entropy=_mean(
                        [cell.policy_entropy for cell in level_cells]
                    ),
                    mean_local_coherence=_mean(
                        [cell.local_coherence for cell in level_cells]
                    ),
                    mean_belief_flux=_mean(
                        [abs(cell.belief_flux_divergence) for cell in level_cells]
                    ),
                    cross_level_consistency=_mean(cross_values),
                )
            )
        return rows

    @staticmethod
    def _first_pymdp_metadata(cell_results: Mapping[str, Any]) -> dict[str, Any]:
        """Return the first non-empty pymdp metadata payload."""
        for result in cell_results.values():
            pymdp = SpatialDiagnostics._pymdp_metadata_from_result(result)
            if pymdp:
                return pymdp
        return {}

    def compute_morans_i(
        self, values: np.ndarray, weights: np.ndarray
    ) -> dict[str, Any]:
        """
        Compute Moran's I spatial autocorrelation.

        Moran's I measures spatial autocorrelation - how similar
        nearby observations are to each other.

        Args:
            values: Array of values per cell (n_cells,)
            weights: Spatial weights matrix (n_cells, n_cells)

        Returns:
            Dict with Moran's I and related metrics
        """
        n = len(values)
        values = np.asarray(values).flatten()

        if n < 2:
            return {"morans_i": 0.0, "interpretation": "insufficient_data"}

        # Mean and deviation
        mean_val = np.mean(values)
        y = values - mean_val

        # Sum of weights
        W = np.sum(weights)
        if W == 0:
            return {"morans_i": 0.0, "interpretation": "no_neighbors"}

        # Numerator: spatial covariance
        numerator = np.sum(weights * np.outer(y, y))

        # Denominator: total variance
        denominator = np.sum(y**2)

        if denominator == 0:
            return {"morans_i": 0.0, "interpretation": "zero_variance"}

        # Moran's I
        morans_i = (n / W) * (numerator / denominator)

        # Interpretation
        if morans_i > 0.3:
            interpretation = "positive_clustering"
        elif morans_i < -0.3:
            interpretation = "negative_dispersion"
        else:
            interpretation = "random"

        return {
            "morans_i": float(morans_i),
            "interpretation": interpretation,
            "n_cells": n,
            "mean_value": float(mean_val),
        }

    def analyze_free_energy_landscape(
        self, free_energy_history: list[float], belief_history: list[np.ndarray]
    ) -> dict[str, Any]:
        """
        Analyze the free energy landscape over time.

        Args:
            free_energy_history: List of free energy values
            belief_history: List of belief arrays over time

        Returns:
            Dict with landscape analysis
        """
        if len(free_energy_history) < 3:
            return {"status": "insufficient_data"}

        fe = np.array(free_energy_history)

        # Trend analysis
        x = np.arange(len(fe))
        slope, intercept = np.polyfit(x, fe, 1)

        # Volatility
        volatility = np.std(np.diff(fe))

        # Convergence detection
        recent = fe[-min(10, len(fe)) :]
        is_converged = np.std(recent) < 0.01 * np.mean(np.abs(recent) + 1e-8)

        # Find minima
        local_minima = []
        for i in range(1, len(fe) - 1):
            if fe[i] < fe[i - 1] and fe[i] < fe[i + 1]:
                local_minima.append({"step": i, "value": float(fe[i])})

        # Belief entropy evolution
        entropy_history = []
        for beliefs in belief_history:
            entropy = -np.sum(beliefs * np.log(beliefs + 1e-8), axis=-1)
            entropy_history.append(float(np.mean(entropy)))

        return {
            "trend": {
                "slope": float(slope),
                "direction": "decreasing"
                if slope < -0.001
                else ("increasing" if slope > 0.001 else "stable"),
            },
            "volatility": float(volatility),
            "is_converged": is_converged,
            "final_value": float(fe[-1]),
            "min_value": float(np.min(fe)),
            "max_value": float(np.max(fe)),
            "local_minima": local_minima[:5],  # Top 5
            "entropy_trend": {
                "initial": entropy_history[0] if entropy_history else None,
                "final": entropy_history[-1] if entropy_history else None,
            },
        }

    def compute_belief_dynamics(
        self, belief_history: list[np.ndarray]
    ) -> dict[str, Any]:
        """
        Analyze belief dynamics over time.

        Args:
            belief_history: List of belief arrays over time

        Returns:
            Dict with dynamics analysis
        """
        if len(belief_history) < 2:
            return {"status": "insufficient_data"}

        # Belief changes between steps
        changes = []
        for i in range(1, len(belief_history)):
            diff = np.abs(belief_history[i] - belief_history[i - 1])
            changes.append(float(np.mean(diff)))

        # Belief concentration (how peaked are beliefs)
        concentrations = []
        for beliefs in belief_history:
            max_prob = np.max(beliefs, axis=-1)
            concentrations.append(float(np.mean(max_prob)))

        return {
            "mean_change": float(np.mean(changes)),
            "change_trend": "stabilizing" if changes[-1] < changes[0] else "volatile",
            "final_concentration": concentrations[-1],
            "concentration_trend": concentrations[-5:]
            if len(concentrations) >= 5
            else concentrations,
        }

    def record_step(
        self,
        step: int,
        beliefs: np.ndarray,
        free_energy: float,
        action: dict | None = None,
        spatial_metrics: dict | None = None,
    ) -> None:
        """Record a single step for analysis."""
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "free_energy": float(free_energy),
            "belief_entropy": float(-np.sum(beliefs * np.log(beliefs + 1e-8))),
            "belief_max": float(np.max(beliefs)),
            "action": action,
            "spatial_metrics": spatial_metrics,
        }
        self.metrics_history.append(entry)

    def export_to_json(self, filename: str = "spatial_diagnostics.json") -> str:
        """Export diagnostics to JSON."""
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                {
                    "metrics_history": self.metrics_history,
                    "analysis_cache": self.analysis_cache,
                    "export_time": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Exported diagnostics to {filepath}")
        return str(filepath)

    def export_to_csv(self, filename: str = "spatial_diagnostics.csv") -> str:
        """Export step-by-step metrics to CSV."""
        filepath = self.output_dir / filename

        if not self.metrics_history:
            logger.warning("No metrics to export")
            return str(filepath)

        # Flatten metrics for CSV
        fieldnames = [
            "step",
            "timestamp",
            "free_energy",
            "belief_entropy",
            "belief_max",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in self.metrics_history:
                writer.writerow(entry)

        logger.info(f"Exported CSV to {filepath}")
        return str(filepath)


def compute_spatial_kl_divergence(
    beliefs_a: np.ndarray, beliefs_b: np.ndarray
) -> float:
    """
    Compute KL divergence between two spatial belief distributions.

    Args:
        beliefs_a: First belief array (n_cells, state_dim)
        beliefs_b: Second belief array (n_cells, state_dim)

    Returns:
        Average KL divergence across cells
    """
    kl_values = []
    for a, b in zip(beliefs_a, beliefs_b):
        a = np.asarray(a) + 1e-8
        b = np.asarray(b) + 1e-8
        a = a / a.sum()
        b = b / b.sum()
        kl = np.sum(a * np.log(a / b))
        kl_values.append(kl)
    return float(np.mean(kl_values))


def compute_information_flow(
    belief_history: list[np.ndarray], neighbor_matrix: np.ndarray
) -> dict[str, float]:
    """
    Compute information flow between neighboring cells.

    Args:
        belief_history: List of beliefs over time
        neighbor_matrix: Adjacency matrix

    Returns:
        Dict with information flow metrics
    """
    if len(belief_history) < 2:
        return {"flow": 0.0}

    n_cells = belief_history[0].shape[0]
    flow_values = []

    for t in range(1, len(belief_history)):
        prev_beliefs = belief_history[t - 1]
        curr_beliefs = belief_history[t]

        for i in range(n_cells):
            neighbors = np.where(neighbor_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                # Change at this cell
                # Neighbor influence
                neighbor_prev = np.mean(prev_beliefs[neighbors], axis=0)
                influence = np.sum(
                    curr_beliefs[i]
                    * np.log((curr_beliefs[i] + 1e-8) / (neighbor_prev + 1e-8))
                )
                flow_values.append(influence)

    return {
        "mean_flow": float(np.mean(flow_values)) if flow_values else 0.0,
        "max_flow": float(np.max(flow_values)) if flow_values else 0.0,
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or a deterministic fallback."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _normalize_optional_distribution(values: Any) -> list[float]:
    """Normalize optional policy arrays and return an empty list when absent."""
    if values is None:
        return []
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return []
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 1e-12:
        array = np.ones_like(array) / array.size
    else:
        array = array / total
    return [float(value) for value in array]


def _trace_belief_vector(values: Any) -> np.ndarray:
    """Normalize belief values from typed results, dicts, or arrays."""
    if isinstance(values, Mapping) and "states" in values:
        values = values["states"]
    return np.asarray(normalize_belief_vector(values))


def _entropy(values: Any) -> float:
    """Return Shannon entropy for a normalized vector."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return 0.0
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 1e-12:
        return 0.0
    array = array / total
    return _finite_float(-np.sum(array * np.log(array + 1e-12)))


def _mean(values: Iterable[Any]) -> float:
    """Return a finite mean for numeric values."""
    finite = [
        _finite_float(value) for value in values if np.isfinite(_finite_float(value))
    ]
    return float(np.mean(finite)) if finite else 0.0
