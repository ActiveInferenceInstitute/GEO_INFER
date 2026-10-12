"""
GEO-INFER-AG: Agricultural geospatial inference and analysis module.

This module provides specialized tools and methodologies for agricultural
applications within the GEO-INFER framework, supporting precision farming,
crop management, yield prediction, and sustainable agricultural practices.
"""

__version__ = "0.4.0"

from geo_infer_ag.core import (
    AgriculturalAnalysis,
    FieldBoundaryManager,
    SeasonalAnalysis,
    SustainabilityAssessment,
)

from geo_infer_ag.models import (
    CropYieldModel,
    SoilHealthModel,
    WaterUsageModel,
    CarbonSequestrationModel,
)

# API exports
from geo_infer_ag.api import (
    AgriculturalAPI,
    FieldsResource,
    CropsResource,
    YieldResource,
)

__all__ = [
    # Core components
    "AgriculturalAnalysis",
    "FieldBoundaryManager",
    "SeasonalAnalysis",
    "SustainabilityAssessment",
    # Models
    "CropYieldModel",
    "SoilHealthModel",
    "WaterUsageModel",
    "CarbonSequestrationModel",
    # API
    "AgriculturalAPI",
    "FieldsResource",
    "CropsResource",
    "YieldResource",
]
