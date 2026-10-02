"""GEO-INFER-CLIMATE: Climate Modeling and Analysis Module.

Provides climate modeling, weather analysis, and climate change
impact assessment capabilities for geospatial systems.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.climate_data import ClimateDataProcessor
from .core.climate_indices import ClimateIndicesCalculator
from .core.downscaling import DownscalingMethods
from .core.projections import ClimateProjections
from .core.extreme_events import ExtremeEventAnalyzer
from .core.impact_assessment import ClimateImpactAssessor
from .core.classification import ClimateClassifier
from .core.temperature_trends import TemperatureTrendAnalyzer
from .core.precipitation_analysis import PrecipitationAnalyzer

__all__ = [
    "ClimateDataProcessor",
    "ClimateIndicesCalculator",
    "DownscalingMethods",
    "ClimateProjections",
    "ExtremeEventAnalyzer",
    "ClimateImpactAssessor",
    "ClimateClassifier",
    "TemperatureTrendAnalyzer",
    "PrecipitationAnalyzer",
]
