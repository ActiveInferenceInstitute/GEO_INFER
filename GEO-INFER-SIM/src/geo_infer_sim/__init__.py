"""
GEO-INFER-SIM: Simulation Environments for Geospatial Analysis

This module provides comprehensive simulation capabilities for testing
geospatial hypotheses, evaluating policies, and analyzing complex system
behaviors using agent-based modeling, system dynamics, and other paradigms.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from geo_infer_sim.core.simulation_engine import SimulationEngine, SimulationConfig
from geo_infer_sim.paradigms.abm import AgentBasedModel, Agent
from geo_infer_sim.paradigms.system_dynamics import SystemDynamicsModel
from geo_infer_sim.paradigms.cellular_automata import CellularAutomata
from geo_infer_sim.scenarios.scenario_manager import ScenarioManager
from geo_infer_sim.module_simulations import ModuleSimulations, ModuleSimulationConfig

__all__ = [
    "SimulationEngine",
    "SimulationConfig",
    "AgentBasedModel",
    "Agent",
    "SystemDynamicsModel",
    "CellularAutomata",
    "ScenarioManager",
    "ModuleSimulations",
    "ModuleSimulationConfig",
]
