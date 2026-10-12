"""
Cascadian Agricultural Ownership Module

Analyzes agricultural land ownership patterns, including concentration
metrics and institutional vs. individual ownership classification.
"""

try:
    from .geo_infer_ownership import GeoInferOwnership
except ImportError:
    GeoInferOwnership = None

__all__ = ["GeoInferOwnership"]
