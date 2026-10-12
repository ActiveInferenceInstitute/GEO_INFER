"""
GEO-INFER-ACT: Active Inference modeling module for GEO-INFER framework.

This module provides mathematical and computational tools for modeling complex
ecological and civic systems using principles from active inference theory.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"
__email__ = "blanket@activeinference.institute"

from .core.active_inference import ActiveInferenceModel
from .core.types import (
    ActiveInferenceStepResult,
    FreeEnergyBreakdown,
    H3BeliefUpdateResult,
    H3CellDiagnostics,
    H3EdgeDiagnostics,
    H3GridInferenceResult,
    H3LevelDiagnostics,
    H3SpatialConsistency,
    NestedH3BeliefUpdateResult,
    NestedH3GridInferenceResult,
    NestedH3LevelSummary,
    PolicyEvaluation,
    SpatialInferenceTrace,
)
from .core.free_energy import FreeEnergyCalculator
from .core.generative_model import GenerativeModel
from .core.belief_updating import BayesianBeliefUpdate
from .core.policy_selection import PolicySelector
from .core.civic_intel import (
    CrescentCityIntel,
    HazardDomain,
    CivicIntelBounds,
    parse_crescent_city_intel,
    hazard_policy_prior,
)
from .core.variational_inference import VariationalInference
from .core.dynamic_causal_model import DynamicCausalModel
from .core.spatial_agent import SpatialActiveInferenceAgent
from .core.markov_decision_process import MarkovDecisionProcess
from .models import (
    BaseActiveInferenceModel,
    CategoricalModel,
    ClimateModel,
    ContinuousPOMDPActiveInference,
    EcologicalModel,
    GaussianModel,
    MultiAgentModel,
    ResourceModel,
    UrbanModel,
)
from .utils.integration import IntegrationUtils

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
