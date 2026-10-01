"""
Typed result objects for active inference contracts.

These dataclasses provide stable, inspectable return shapes for callers that
need mathematical diagnostics without breaking the older float/dict APIs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Convert numpy-rich typed results into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass
class FreeEnergyBreakdown:
    """Decomposed free-energy terms for a single inference calculation."""

    free_energy: float
    accuracy: float = 0.0
    complexity: float = 0.0
    entropy: float = 0.0
    pragmatic_value: float = 0.0
    epistemic_value: float = 0.0
    risk: float = 0.0
    ambiguity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe free-energy breakdown."""
        return {
            "free_energy": float(self.free_energy),
            "accuracy": float(self.accuracy),
            "complexity": float(self.complexity),
            "entropy": float(self.entropy),
            "pragmatic_value": float(self.pragmatic_value),
            "epistemic_value": float(self.epistemic_value),
            "risk": float(self.risk),
            "ambiguity": float(self.ambiguity),
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class PolicyEvaluation:
    """Expected-free-energy evaluation for a policy candidate."""

    policy: Any
    expected_free_energy: float
    probability: float
    index: int
    epistemic_value: float = 0.0
    pragmatic_value: float = 0.0
    risk: float = 0.0
    ambiguity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy evaluation."""
        return {
            "policy": _to_jsonable(self.policy),
            "expected_free_energy": float(self.expected_free_energy),
            "probability": float(self.probability),
            "index": int(self.index),
            "epistemic_value": float(self.epistemic_value),
            "pragmatic_value": float(self.pragmatic_value),
            "risk": float(self.risk),
            "ambiguity": float(self.ambiguity),
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class ActiveInferenceStepResult:
    """Typed result for a complete perceive-act step."""

    beliefs: Any
    action: Any
    free_energy: float
    expected_free_energy: float | None = None
    policy_evaluation: PolicyEvaluation | None = None
    observation: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe step result."""
        return {
            "beliefs": _to_jsonable(self.beliefs),
            "action": _to_jsonable(self.action),
            "free_energy": float(self.free_energy),
            "expected_free_energy": (
                float(self.expected_free_energy)
                if self.expected_free_energy is not None
                else None
            ),
            "policy_evaluation": _to_jsonable(self.policy_evaluation),
            "observation": _to_jsonable(self.observation),
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class H3SpatialConsistency:
    """Spatial coherence diagnostics for H3-indexed active inference."""

    global_coherence: float
    neighbor_correlations: float
    cell_count: int = 0
    edge_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized spatial consistency record."""
        return {
            "global_coherence": self.global_coherence,
            "neighbor_correlations": self.neighbor_correlations,
            "cell_count": self.cell_count,
            "edge_count": self.edge_count,
            **self.metadata,
        }


@dataclass
class H3BeliefUpdateResult:
    """Typed result for H3-indexed belief updates."""

    h3_beliefs: dict[str, Any]
    average: Any
    spatial_consistency: H3SpatialConsistency
    aggregate_free_energy: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized belief-update record."""
        return {
            "h3_beliefs": self.h3_beliefs,
            "average": self.average,
            "spatial_consistency": self.spatial_consistency.to_dict(),
            "aggregate_free_energy": self.aggregate_free_energy,
            **self.metadata,
        }


@dataclass
class H3GridInferenceResult:
    """Typed result for active inference over an H3 observation grid."""

    cell_results: dict[str, ActiveInferenceStepResult]
    aggregate_free_energy: float
    spatial_consistency: H3SpatialConsistency
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like dictionary containing typed cell diagnostics."""
        return {
            "cell_results": _to_jsonable(self.cell_results),
            "aggregate_free_energy": self.aggregate_free_energy,
            "spatial_consistency": self.spatial_consistency.to_dict(),
            **_to_jsonable(self.metadata),
        }


@dataclass
class H3CellDiagnostics:
    """Per-cell diagnostics for spatial active inference over an H3 lattice."""

    cell: str
    timestep: int
    resolution: int
    belief: list[float]
    entropy: float
    free_energy: float
    expected_free_energy: float
    selected_action: Any
    selected_action_index: int
    selected_action_probability: float
    action_posterior: list[float]
    negative_expected_free_energy: list[float]
    selected_negative_expected_free_energy: float
    policy_entropy: float
    neighbor_count: int
    local_coherence: float
    posterior_delta: float
    belief_flux_in: float
    belief_flux_out: float
    belief_flux_divergence: float
    parent_cell: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe per-cell diagnostics."""
        return {
            "cell": self.cell,
            "timestep": int(self.timestep),
            "resolution": int(self.resolution),
            "belief": _to_jsonable(self.belief),
            "entropy": float(self.entropy),
            "free_energy": float(self.free_energy),
            "expected_free_energy": float(self.expected_free_energy),
            "selected_action": _to_jsonable(self.selected_action),
            "selected_action_index": int(self.selected_action_index),
            "selected_action_probability": float(self.selected_action_probability),
            "action_posterior": _to_jsonable(self.action_posterior),
            "negative_expected_free_energy": _to_jsonable(
                self.negative_expected_free_energy
            ),
            "selected_negative_expected_free_energy": float(
                self.selected_negative_expected_free_energy
            ),
            "policy_entropy": float(self.policy_entropy),
            "neighbor_count": int(self.neighbor_count),
            "local_coherence": float(self.local_coherence),
            "posterior_delta": float(self.posterior_delta),
            "belief_flux_in": float(self.belief_flux_in),
            "belief_flux_out": float(self.belief_flux_out),
            "belief_flux_divergence": float(self.belief_flux_divergence),
            "parent_cell": self.parent_cell,
            **_to_jsonable(self.metadata),
        }


@dataclass
class H3EdgeDiagnostics:
    """Per-edge diagnostics for neighboring H3 cell belief relationships."""

    source: str
    target: str
    timestep: int
    resolution: int
    belief_distance: float
    coherence: float
    source_entropy: float
    target_entropy: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe per-edge diagnostics."""
        return {
            "source": self.source,
            "target": self.target,
            "timestep": int(self.timestep),
            "resolution": int(self.resolution),
            "belief_distance": float(self.belief_distance),
            "coherence": float(self.coherence),
            "source_entropy": float(self.source_entropy),
            "target_entropy": float(self.target_entropy),
            **_to_jsonable(self.metadata),
        }


@dataclass
class H3LevelDiagnostics:
    """Per-resolution diagnostics for flat or nested H3 inference traces."""

    resolution: int
    timestep: int
    cell_count: int
    edge_count: int
    mean_entropy: float
    mean_free_energy: float
    mean_expected_free_energy: float
    mean_policy_entropy: float
    mean_local_coherence: float
    mean_belief_flux: float
    cross_level_consistency: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe per-level diagnostics."""
        return {
            "resolution": int(self.resolution),
            "timestep": int(self.timestep),
            "cell_count": int(self.cell_count),
            "edge_count": int(self.edge_count),
            "mean_entropy": float(self.mean_entropy),
            "mean_free_energy": float(self.mean_free_energy),
            "mean_expected_free_energy": float(self.mean_expected_free_energy),
            "mean_policy_entropy": float(self.mean_policy_entropy),
            "mean_local_coherence": float(self.mean_local_coherence),
            "mean_belief_flux": float(self.mean_belief_flux),
            "cross_level_consistency": float(self.cross_level_consistency),
            **_to_jsonable(self.metadata),
        }


@dataclass
class SpatialInferenceTrace:
    """Run-level trace for spatial active inference diagnostics."""

    scenario: str
    timesteps: list[int]
    cell_diagnostics: list[H3CellDiagnostics]
    edge_diagnostics: list[H3EdgeDiagnostics]
    level_diagnostics: list[H3LevelDiagnostics]
    hierarchy_metadata: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe spatial trace data."""
        return {
            "scenario": self.scenario,
            "timesteps": [int(value) for value in self.timesteps],
            "cell_diagnostics": [item.to_dict() for item in self.cell_diagnostics],
            "edge_diagnostics": [item.to_dict() for item in self.edge_diagnostics],
            "level_diagnostics": [item.to_dict() for item in self.level_diagnostics],
            "hierarchy_metadata": _to_jsonable(self.hierarchy_metadata),
            "backend_metadata": _to_jsonable(self.backend_metadata),
            **_to_jsonable(self.metadata),
        }


@dataclass
class NestedH3LevelSummary:
    """Per-resolution diagnostics for nested H3 active inference."""

    resolution: int
    cell_count: int
    edge_count: int
    mean_free_energy: float
    mean_entropy: float
    coherence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like level summary."""
        return {
            "resolution": self.resolution,
            "cell_count": self.cell_count,
            "edge_count": self.edge_count,
            "mean_free_energy": self.mean_free_energy,
            "mean_entropy": self.mean_entropy,
            "coherence": self.coherence,
            **self.metadata,
        }


@dataclass
class NestedH3BeliefUpdateResult:
    """Typed result for nested H3 belief updates."""

    fine_beliefs: dict[str, Any]
    parent_beliefs: dict[str, Any]
    level_summaries: list[NestedH3LevelSummary]
    parent_child_map: dict[str, list[str]]
    child_parent_map: dict[str, str]
    spatial_consistency: H3SpatialConsistency
    aggregate_free_energy: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like nested belief update result."""
        return {
            "fine_beliefs": _to_jsonable(self.fine_beliefs),
            "parent_beliefs": _to_jsonable(self.parent_beliefs),
            "level_summaries": [summary.to_dict() for summary in self.level_summaries],
            "parent_child_map": self.parent_child_map,
            "child_parent_map": self.child_parent_map,
            "spatial_consistency": self.spatial_consistency.to_dict(),
            "aggregate_free_energy": self.aggregate_free_energy,
            **_to_jsonable(self.metadata),
        }


@dataclass
class NestedH3GridInferenceResult:
    """Typed result for active inference over a nested H3 hierarchy."""

    cell_results: dict[str, ActiveInferenceStepResult]
    nested_belief_update: NestedH3BeliefUpdateResult
    aggregate_free_energy: float
    spatial_consistency: H3SpatialConsistency
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like nested grid inference result."""
        return {
            "cell_results": _to_jsonable(self.cell_results),
            "nested_belief_update": self.nested_belief_update.to_dict(),
            "aggregate_free_energy": self.aggregate_free_energy,
            "spatial_consistency": self.spatial_consistency.to_dict(),
            **_to_jsonable(self.metadata),
        }
