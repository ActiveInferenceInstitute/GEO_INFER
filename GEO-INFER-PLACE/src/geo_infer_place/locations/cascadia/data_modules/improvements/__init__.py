"""
Cascadian Agricultural Improvements Module

Analyzes the value and type of improvements (buildings, infrastructure)
on agricultural land parcels.
"""

try:
    from .geo_infer_improvements import GeoInferImprovements
except ImportError:
    GeoInferImprovements = None

__all__ = ["GeoInferImprovements"]
