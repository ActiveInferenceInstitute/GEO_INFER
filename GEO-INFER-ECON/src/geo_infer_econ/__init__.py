"""
GEO-INFER-ECON: Spatial Economic Modeling, Analysis, and Policy Evaluation

This module provides comprehensive economic modeling capabilities including:
- Microeconomic analysis and modeling
- Macroeconomic modeling and forecasting
- Bioregional market design and ecological economics
- Spatial econometrics and policy analysis
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"

# Core imports
from .core import (
    EconomicModelingEngine,
    SpatialEconometricsEngine,
    PolicyAnalysisEngine,
)

# Microeconomic imports
from .microeconomics import (
    ConsumerChoiceModels,
    ProducerTheoryModels,
    MarketStructureAnalysis,
    GameTheoryModels,
    BehavioralEconomicsEngine,
)

# Macroeconomic imports
from .macroeconomics import (
    AggregateGrowthModels,
    BusinessCycleModels,
    MonetaryPolicyModels,
    FiscalPolicyModels,
    TradeModels,
)

# Bioregional economics imports
from .bioregional import (
    EcologicalEconomicsEngine,
    BioregionalMarketDesign,
    NaturalCapitalAccounting,
    EcosystemServicesValuation,
    CircularEconomyModels,
)

# API imports (optional — requires fastapi)
try:
    from .api import EconomicAnalysisAPI
except ImportError:
    EconomicAnalysisAPI = None  # type: ignore[misc,assignment]

# Utilities
from .utils import DataLoader, ResultsVisualizer, ModelValidator, EconomicIndicators

__all__ = [
    # Core
    "EconomicModelingEngine",
    "SpatialEconometricsEngine",
    "PolicyAnalysisEngine",
    # Microeconomics
    "ConsumerChoiceModels",
    "ProducerTheoryModels",
    "MarketStructureAnalysis",
    "GameTheoryModels",
    "BehavioralEconomicsEngine",
    # Macroeconomics
    "AggregateGrowthModels",
    "BusinessCycleModels",
    "MonetaryPolicyModels",
    "FiscalPolicyModels",
    "TradeModels",
    # Bioregional
    "EcologicalEconomicsEngine",
    "BioregionalMarketDesign",
    "NaturalCapitalAccounting",
    "EcosystemServicesValuation",
    "CircularEconomyModels",
    # API & Utils
    "EconomicAnalysisAPI",
    "DataLoader",
    "ResultsVisualizer",
    "ModelValidator",
    "EconomicIndicators",
]
