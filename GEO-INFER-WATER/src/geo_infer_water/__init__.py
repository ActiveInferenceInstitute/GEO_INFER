"""GEO-INFER-WATER: Water Resources Management Module."""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.hydrology import HydrologicalModeler
from .core.water_quality import (
    WaterQualityAssessor,
    WaterSample,
    WaterBodyType,
    PollutantType,
)
from .core.water_infrastructure import WaterInfrastructurePlanner
from .core.flood_drought import FloodDroughtAnalyzer
from .core.watershed_delineation import WatershedDelineator
from .core.infiltration import InfiltrationModeler
from .core.water_balance import WaterBalanceModeler

__all__ = [
    "HydrologicalModeler",
    "WaterQualityAssessor",
    "WaterSample",
    "WaterBodyType",
    "PollutantType",
    "WaterInfrastructurePlanner",
    "FloodDroughtAnalyzer",
    "WatershedDelineator",
    "WaterBalanceModeler",
    "InfiltrationModeler",
]
