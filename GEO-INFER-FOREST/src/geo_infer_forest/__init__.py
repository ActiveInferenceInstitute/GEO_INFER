"""GEO-INFER-FOREST: Forest Management and Analysis Module."""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.forest_inventory import ForestInventory
from .core.carbon_sequestration import CarbonSequestrationModeler
from .core.wildfire_risk import (
    FireDangerRating,
    FireIncident,
    FireWeatherObservation,
    FuelType,
    WildfireRiskAnalyzer,
)
from .core.forest_health import ForestHealthMonitor
from .core.canopy_analysis import CanopyAnalyzer
from .core.deforestation import DeforestationDetector
from .core.fire_risk import FireRiskAssessor

__all__ = [
    "FireDangerRating",
    "FireIncident",
    "FireRiskAssessor",
    "FireWeatherObservation",
    "ForestHealthMonitor",
    "ForestInventory",
    "FuelType",
    "CarbonSequestrationModeler",
    "CanopyAnalyzer",
    "DeforestationDetector",
    "WildfireRiskAnalyzer",
]
