"""
GEO-INFER-EMERGENCY: Emergency Management Module

This module provides emergency management and disaster response capabilities
for geospatial systems, including incident coordination, resource deployment,
evacuation planning, and search and rescue operations.

Key Features:
- Multi-agency incident coordination (ICS)
- Resource deployment optimization
- Evacuation planning and routing
- Situational awareness and common operating picture
- Search and rescue mission planning
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.coordinator import EmergencyCoordinator
from .core.resources import ResourceDeployer
from .core.evacuation import EvacuationPlanner
from .core.awareness import SituationalAwareness
from .core.sar import SearchAndRescue

__all__ = [
    "EmergencyCoordinator",
    "ResourceDeployer",
    "EvacuationPlanner",
    "SituationalAwareness",
    "SearchAndRescue",
]
