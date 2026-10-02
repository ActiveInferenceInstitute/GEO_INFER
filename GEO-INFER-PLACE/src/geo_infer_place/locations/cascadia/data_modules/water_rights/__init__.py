"""
Cascadian Water Rights Module

Analyzes agricultural water rights from state-level sources.
"""

from .data_sources import CascadianWaterRightsDataSources

try:
    from .geo_infer_water_rights import GeoInferWaterRights
except ImportError:
    GeoInferWaterRights = None

__all__ = ["CascadianWaterRightsDataSources", "GeoInferWaterRights"]
