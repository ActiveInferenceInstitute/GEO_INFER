"""
GEO-INFER-TRANSPORT: Transportation Analysis Module

This module provides transportation planning and traffic analysis capabilities
for geospatial systems, including network analysis, routing optimization,
flow modeling, and accessibility analysis.

Key Features:
- Network topology construction and analysis
- Multi-modal routing with optimization
- Traffic flow modeling and simulation
- Accessibility and service area analysis
- Transit network optimization
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.network import TransportNetwork
from .core.routing import RoutingEngine
from .core.traffic import TrafficAnalyzer
from .core.accessibility import AccessibilityAnalyzer
from .core.transit import TransitOptimizer

__all__ = [
    "TransportNetwork",
    "RoutingEngine",
    "TrafficAnalyzer",
    "AccessibilityAnalyzer",
    "TransitOptimizer",
]
