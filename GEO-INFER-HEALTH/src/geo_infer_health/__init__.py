# GEO-INFER-HEALTH main package
"""
GEO-INFER-HEALTH: Geospatial Applications for Public Health, Epidemiology, and Healthcare Accessibility

This module provides comprehensive tools for spatial health analytics, including:
- Disease surveillance and outbreak modeling
- Healthcare accessibility analysis
- Environmental health risk assessment
- Spatial epidemiology methods
- Health disparities mapping

The module integrates with the broader GEO-INFER framework and implements
Active Inference principles for intelligent health analytics.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"
__email__ = "geo-infer@activeinference.institute"

# Import core components
from .core import (
    DiseaseHotspotAnalyzer,
    HealthcareAccessibilityAnalyzer,
    EnvironmentalHealthAnalyzer,
)

# Import models
from .models import (
    Location,
    HealthFacility,
    DiseaseReport,
    PopulationData,
    EnvironmentalData,
)

# Import utilities
from .utils import (
    haversine_distance,
    create_bounding_box,
)

# Import API router
from .api import router as api_router

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core analyzers
    "DiseaseHotspotAnalyzer",
    "HealthcareAccessibilityAnalyzer",
    "EnvironmentalHealthAnalyzer",
    # Data models
    "Location",
    "HealthFacility",
    "DiseaseReport",
    "PopulationData",
    "EnvironmentalData",
    # Utilities
    "haversine_distance",
    "create_bounding_box",
    # API
    "api_router",
]
