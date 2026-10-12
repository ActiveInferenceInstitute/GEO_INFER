"""
API interfaces for GEO-INFER-MATH functionality.

This package provides clean, consistent interfaces for accessing the
mathematical operations and models provided by GEO-INFER-MATH.
"""

# Convenience modules are the lightweight, non-web public API and need no
# Flask or Werkzeug.
from geo_infer_math.api.convenience import (
    AIConvenience as AIConvenience,
    ActiveInferenceConvenience as ActiveInferenceConvenience,
    BayesianConvenience as BayesianConvenience,
    InformationTheoryConvenience as InformationTheoryConvenience,
    IntegrationConvenience as IntegrationConvenience,
    SpatialConvenience as SpatialConvenience,
)

_available_apis = []
_missing_optional_dependencies = {}

# The REST API needs the optional ``web`` extra (Flask, Werkzeug).
try:
    from geo_infer_math.api.spatial_analysis import (
        SpatialAnalysisAPI as SpatialAnalysisAPI,
    )

    _available_apis.append("SpatialAnalysisAPI")
except ImportError as exc:
    _missing_optional_dependencies["SpatialAnalysisAPI"] = str(exc)

__all__ = [
    *_available_apis,
    "ActiveInferenceConvenience",
    "BayesianConvenience",
    "AIConvenience",
    "InformationTheoryConvenience",
    "SpatialConvenience",
    "IntegrationConvenience",
]
