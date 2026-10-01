"""
Module Integration Layer

This package provides integration layers for connecting GEO-INFER-MATH with
other GEO-INFER modules (AI, ACT, BAYES).
"""

from geo_infer_math.integration.act import (
    BeliefUpdating,
    FreeEnergyCalculator,
    GenerativeModels,
    PolicyOptimization,
    VariationalInferenceHelpers,
)
from geo_infer_math.integration.ai import (
    AIGradientHelpers,
    OptimizationBridges,
    SpatialAttention,
    SpatialLossFunctions,
    SpatialTensorOperations,
)
from geo_infer_math.integration.bayes import (
    BayesianOptimization,
    MCMCHelpers,
    ModelSelection,
    PosteriorHelpers,
    PriorBuilders,
)

__all__ = [
    # AI Integration
    "AIGradientHelpers",
    "SpatialLossFunctions",
    "OptimizationBridges",
    "SpatialTensorOperations",
    "SpatialAttention",
    # ACT Integration
    "FreeEnergyCalculator",
    "VariationalInferenceHelpers",
    "BeliefUpdating",
    "PolicyOptimization",
    "GenerativeModels",
    # BAYES Integration
    "PosteriorHelpers",
    "PriorBuilders",
    "MCMCHelpers",
    "BayesianOptimization",
    "ModelSelection",
]
