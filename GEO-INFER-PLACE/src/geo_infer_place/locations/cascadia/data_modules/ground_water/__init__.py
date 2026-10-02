"""
Cascadian Groundwater Analysis Module

This module analyzes groundwater availability using real data from the
USGS National Water Information System (NWIS).
"""

try:
    from .geo_infer_ground_water import GeoInferGroundWater
except ImportError:
    GeoInferGroundWater = None
from .data_sources import CascadianGroundWaterDataSources

__all__ = ["GeoInferGroundWater", "CascadianGroundWaterDataSources"]
