"""
GEO-INFER-ACT: Active Inference modeling module for GEO-INFER framework.

This module provides mathematical and computational tools for modeling complex
ecological and civic systems using principles from active inference theory.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"
__email__ = "blanket@activeinference.institute"

from importlib import import_module as _import_module

_EXPORTS = {
    "ActiveInferenceModel": (".core.active_inference", "ActiveInferenceModel"),
    "ActiveInferenceStepResult": (".core.types", "ActiveInferenceStepResult"),
    "FreeEnergyBreakdown": (".core.types", "FreeEnergyBreakdown"),
    "H3BeliefUpdateResult": (".core.types", "H3BeliefUpdateResult"),
    "H3CellDiagnostics": (".core.types", "H3CellDiagnostics"),
    "H3EdgeDiagnostics": (".core.types", "H3EdgeDiagnostics"),
    "H3GridInferenceResult": (".core.types", "H3GridInferenceResult"),
    "H3LevelDiagnostics": (".core.types", "H3LevelDiagnostics"),
    "H3SpatialConsistency": (".core.types", "H3SpatialConsistency"),
    "NestedH3BeliefUpdateResult": (".core.types", "NestedH3BeliefUpdateResult"),
    "NestedH3GridInferenceResult": (".core.types", "NestedH3GridInferenceResult"),
    "NestedH3LevelSummary": (".core.types", "NestedH3LevelSummary"),
    "PolicyEvaluation": (".core.types", "PolicyEvaluation"),
    "SpatialInferenceTrace": (".core.types", "SpatialInferenceTrace"),
    "FreeEnergyCalculator": (".core.free_energy", "FreeEnergyCalculator"),
    "GenerativeModel": (".core.generative_model", "GenerativeModel"),
    "BayesianBeliefUpdate": (".core.belief_updating", "BayesianBeliefUpdate"),
    "PolicySelector": (".core.policy_selection", "PolicySelector"),
    "CrescentCityIntel": (".core.civic_intel", "CrescentCityIntel"),
    "HazardDomain": (".core.civic_intel", "HazardDomain"),
    "CivicIntelBounds": (".core.civic_intel", "CivicIntelBounds"),
    "parse_crescent_city_intel": (".core.civic_intel", "parse_crescent_city_intel"),
    "hazard_policy_prior": (".core.civic_intel", "hazard_policy_prior"),
    "VariationalInference": (".core.variational_inference", "VariationalInference"),
    "DynamicCausalModel": (".core.dynamic_causal_model", "DynamicCausalModel"),
    "SpatialActiveInferenceAgent": (
        ".core.spatial_agent",
        "SpatialActiveInferenceAgent",
    ),
    "MarkovDecisionProcess": (".core.markov_decision_process", "MarkovDecisionProcess"),
    "BaseActiveInferenceModel": (".models", "BaseActiveInferenceModel"),
    "CategoricalModel": (".models", "CategoricalModel"),
    "ClimateModel": (".models", "ClimateModel"),
    "ContinuousPOMDPActiveInference": (".models", "ContinuousPOMDPActiveInference"),
    "EcologicalModel": (".models", "EcologicalModel"),
    "GaussianModel": (".models", "GaussianModel"),
    "MultiAgentModel": (".models", "MultiAgentModel"),
    "ResourceModel": (".models", "ResourceModel"),
    "UrbanModel": (".models", "UrbanModel"),
    "IntegrationUtils": (".utils.integration", "IntegrationUtils"),
}
_SUBMODULES = {
    "core": "geo_infer_act.core",
    "models": "geo_infer_act.models",
    "utils": "geo_infer_act.utils",
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
    "FreeEnergyCalculator",
    "GenerativeModel",
    "BayesianBeliefUpdate",
    "PolicySelector",
    "CrescentCityIntel",
    "HazardDomain",
    "CivicIntelBounds",
    "parse_crescent_city_intel",
    "hazard_policy_prior",
    "VariationalInference",
    "DynamicCausalModel",
    "SpatialActiveInferenceAgent",
    "ClimateModel",
    "IntegrationUtils",
    "MarkovDecisionProcess",
    "BaseActiveInferenceModel",
    "CategoricalModel",
    "GaussianModel",
    "EcologicalModel",
    "UrbanModel",
    "ResourceModel",
    "MultiAgentModel",
    "ContinuousPOMDPActiveInference",
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
