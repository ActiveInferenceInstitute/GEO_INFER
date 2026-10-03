"""
GEO-INFER-ANT Core Components

This module contains the core components for swarm intelligence and complex
adaptive systems, including agent base classes, population dynamics, and
communication systems.

Core Components:
- SwarmAgent: Base class for individual swarm agents
- AgentPopulation: Management system for agent populations
- PheromoneSystem: Stigmergic communication infrastructure
- DigitalStigmergy: Digital communication and coordination systems

Integration Points:
- GEO-INFER-ACT: Active Inference for agent decision making
- GEO-INFER-SPACE: Spatial indexing and analytics for agent operations
- GEO-INFER-AGENT: Agent lifecycle management and coordination
- GEO-INFER-TIME: Temporal dynamics for simulation timing

Example:
    >>> from geo_infer_ant.core import SwarmAgent, AgentPopulation
    >>>
    >>> # Create individual swarm agent
    >>> agent = SwarmAgent(
    ...     agent_id="worker_001",
    ...     position=np.array([37.7749, -122.4194]),
    ...     sensory_range=100.0,
    ...     active_inference_enabled=False
    ... )
    >>>
    >>> # Create agent population
    >>> population = AgentPopulation(
    ...     population_size=1000,
    ...     agent_types=['worker', 'scout', 'soldier'],
    ...     spatial_distribution='clustered'
    ... )
"""

from .agent_base import SwarmAgent, SensoryInput, ActionDecision
from .population import AgentPopulation
from .stigmergy import PheromoneSystem
from .digital_stigmergy import DigitalStigmergy

# Export main classes and functions
__all__ = [
    # Agent classes
    "SwarmAgent",
    "SensoryInput",
    "ActionDecision",
    # Population management
    "AgentPopulation",
    # Communication systems
    "PheromoneSystem",
    "DigitalStigmergy",
]
