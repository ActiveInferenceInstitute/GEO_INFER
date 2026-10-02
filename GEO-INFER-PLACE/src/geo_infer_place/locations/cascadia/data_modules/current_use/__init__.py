"""
Cascadian Current Agricultural Use Module

Real-time agricultural land use classification and crop production analysis
for agricultural redevelopment planning.
"""

try:
    from .geo_infer_current_use import GeoInferCurrentUse
except ImportError:
    GeoInferCurrentUse = None
from .data_sources import CascadianCurrentUseDataSources

__all__ = ["GeoInferCurrentUse", "CascadianCurrentUseDataSources"]
