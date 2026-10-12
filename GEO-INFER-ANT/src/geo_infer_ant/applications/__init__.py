"""
GEO-INFER-ANT Applications Module

This module contains domain-specific applications of swarm intelligence for
real-world problems, including environmental monitoring, disaster response,
and urban optimization systems.

Applications:
- EnvironmentalMonitoringSwarm: Environmental monitoring and intelligence
- DisasterResponseSwarm: Disaster response and emergency coordination
- UrbanTrafficSwarm: Urban traffic and infrastructure optimization

Integration Points:
- Core swarm components for agent coordination
- Optimization algorithms for multi-objective decision making
- Spatial analytics for geographic optimization
- Communication systems for coordination

Example:
    >>> from geo_infer_ant.applications import EnvironmentalMonitoringSwarm, DisasterResponseSwarm
    >>>
    >>> # Environmental monitoring application
    >>> env_swarm = EnvironmentalMonitoringSwarm(
    ...     swarm_size=200,
    ...     monitoring_objectives=['air_quality', 'biodiversity'],
    ...     adaptive_sampling=True
    ... )
    >>>
    >>> # Disaster response coordination
    >>> disaster_swarm = DisasterResponseSwarm(
    ...     response_types=['search_rescue', 'damage_assessment'],
    ...     coordination_protocol='stigmergic'
    ... )
"""

import logging

from .environmental import (
    EnvironmentalMonitoringSwarm,
    MonitoringObjective,
    SensorReading,
)
from .disaster import DisasterResponseSwarm, DisasterScenario
from .urban import UrbanTrafficSwarm, UrbanSystem

logger = logging.getLogger(__name__)

# Export main classes and functions
__all__ = [
    # Environmental Monitoring
    "EnvironmentalMonitoringSwarm",
    "MonitoringObjective",
    "SensorReading",
    # Disaster Response
    "DisasterResponseSwarm",
    "DisasterScenario",
    # Urban Optimization
    "UrbanTrafficSwarm",
    "UrbanSystem",
]
