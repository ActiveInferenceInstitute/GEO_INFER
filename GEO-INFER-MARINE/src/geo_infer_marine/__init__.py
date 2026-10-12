"""GEO-INFER-MARINE: Marine and Oceanographic Analysis Module.

Provides marine and oceanographic analysis capabilities including
coastal management, marine ecosystem monitoring, ocean current modeling,
water quality assessment, and coral reef health analysis.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.oceanographic_data import OceanographicDataProcessor
from .core.coastal_analysis import CoastalAnalyzer
from .core.marine_ecosystems import (
    MarineEcosystemModeler,
    MarineHabitatType,
    SpeciesData,
)
from .core.sea_level import SeaLevelAnalyzer
from .core.marine_spatial_planning import MarineSpatialPlanner
from .core.ocean_currents import OceanCurrentModeler
from .core.water_quality import MarineWaterQuality
from .core.coral_reef import CoralReefAssessor

__all__ = [
    "OceanographicDataProcessor",
    "CoastalAnalyzer",
    "MarineEcosystemModeler",
    "MarineHabitatType",
    "SpeciesData",
    "SeaLevelAnalyzer",
    "MarineSpatialPlanner",
    "OceanCurrentModeler",
    "MarineWaterQuality",
    "CoralReefAssessor",
]
