"""
Cascadian Power Source & Infrastructure Module

Analyzes proximity to and capacity of electrical power infrastructure,
including transmission lines and substations.
"""

try:
    from .geo_infer_power_source import GeoInferPowerSource
except ImportError:
    GeoInferPowerSource = None

__all__ = ["GeoInferPowerSource"]
