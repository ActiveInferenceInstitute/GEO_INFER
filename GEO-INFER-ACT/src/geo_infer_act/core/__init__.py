"""
Core components for active inference modeling.

This module contains the essential classes and functions for
implementing active inference models and algorithms.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "ActiveInferenceModel": (
        "geo_infer_act.core.active_inference",
        "ActiveInferenceModel",
    ),
    "ActiveInferenceStepResult": (
        "geo_infer_act.core.types",
        "ActiveInferenceStepResult",
    ),
    "FreeEnergyBreakdown": ("geo_infer_act.core.types", "FreeEnergyBreakdown"),
    "H3BeliefUpdateResult": ("geo_infer_act.core.types", "H3BeliefUpdateResult"),
    "H3CellDiagnostics": ("geo_infer_act.core.types", "H3CellDiagnostics"),
    "H3EdgeDiagnostics": ("geo_infer_act.core.types", "H3EdgeDiagnostics"),
    "H3GridInferenceResult": ("geo_infer_act.core.types", "H3GridInferenceResult"),
    "H3LevelDiagnostics": ("geo_infer_act.core.types", "H3LevelDiagnostics"),
    "H3SpatialConsistency": ("geo_infer_act.core.types", "H3SpatialConsistency"),
    "NestedH3BeliefUpdateResult": (
        "geo_infer_act.core.types",
        "NestedH3BeliefUpdateResult",
    ),
    "NestedH3GridInferenceResult": (
        "geo_infer_act.core.types",
        "NestedH3GridInferenceResult",
    ),
    "NestedH3LevelSummary": ("geo_infer_act.core.types", "NestedH3LevelSummary"),
    "PolicyEvaluation": ("geo_infer_act.core.types", "PolicyEvaluation"),
    "SpatialInferenceTrace": ("geo_infer_act.core.types", "SpatialInferenceTrace"),
    "GenerativeModel": ("geo_infer_act.core.generative_model", "GenerativeModel"),
    "VariationalInference": (
        "geo_infer_act.core.variational_inference",
        "VariationalInference",
    ),
    "FreeEnergyCalculator": ("geo_infer_act.core.free_energy", "FreeEnergyCalculator"),
    "MarkovDecisionProcess": (
        "geo_infer_act.core.markov_decision_process",
        "MarkovDecisionProcess",
    ),
    "BayesianBeliefUpdate": (
        "geo_infer_act.core.belief_updating",
        "BayesianBeliefUpdate",
    ),
    "PolicySelector": ("geo_infer_act.core.policy_selection", "PolicySelector"),
    "CrescentCityIntel": ("geo_infer_act.core.civic_intel", "CrescentCityIntel"),
    "HazardDomain": ("geo_infer_act.core.civic_intel", "HazardDomain"),
    "CivicIntelBounds": ("geo_infer_act.core.civic_intel", "CivicIntelBounds"),
    "parse_crescent_city_intel": (
        "geo_infer_act.core.civic_intel",
        "parse_crescent_city_intel",
    ),
    "hazard_policy_prior": ("geo_infer_act.core.civic_intel", "hazard_policy_prior"),
    "DynamicCausalModel": (
        "geo_infer_act.core.dynamic_causal_model",
        "DynamicCausalModel",
    ),
    "SpatialActiveInferenceAgent": (
        "geo_infer_act.core.spatial_agent",
        "SpatialActiveInferenceAgent",
    ),
}
_SUBMODULES = {
    "active_inference": "geo_infer_act.core.active_inference",
    "types": "geo_infer_act.core.types",
    "generative_model": "geo_infer_act.core.generative_model",
    "variational_inference": "geo_infer_act.core.variational_inference",
    "free_energy": "geo_infer_act.core.free_energy",
    "markov_decision_process": "geo_infer_act.core.markov_decision_process",
    "belief_updating": "geo_infer_act.core.belief_updating",
    "policy_selection": "geo_infer_act.core.policy_selection",
    "civic_intel": "geo_infer_act.core.civic_intel",
    "dynamic_causal_model": "geo_infer_act.core.dynamic_causal_model",
    "spatial_agent": "geo_infer_act.core.spatial_agent",
}

__all__ = [
    "ActiveInferenceModel",
    "ActiveInferenceStepResult",
    "FreeEnergyBreakdown",
    "H3BeliefUpdateResult",
    "H3CellDiagnostics",
    "H3EdgeDiagnostics",
    "H3GridInferenceResult",
    "H3LevelDiagnostics",
    "H3SpatialConsistency",
    "NestedH3BeliefUpdateResult",
    "NestedH3GridInferenceResult",
    "NestedH3LevelSummary",
    "PolicyEvaluation",
    "SpatialInferenceTrace",
    "GenerativeModel",
    "VariationalInference",
    "FreeEnergyCalculator",
    "MarkovDecisionProcess",
    "BayesianBeliefUpdate",
    "PolicySelector",
    "CrescentCityIntel",
    "HazardDomain",
    "CivicIntelBounds",
    "parse_crescent_city_intel",
    "hazard_policy_prior",
    "DynamicCausalModel",
    "SpatialActiveInferenceAgent",
]


def __getattr__(name: str):
    """Load existing public exports only when requested."""
    if name in _EXPORTS:
        module, member = _EXPORTS[name]
        value = getattr(_import_module(module, __name__), member)
    elif name in _SUBMODULES:
        value = _import_module(_SUBMODULES[name])
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
