"""
GEO-INFER-ANT: Complex Adaptive Systems and Swarm Intelligence

This package provides comprehensive implementations for swarm intelligence and complex
adaptive systems modeling using Active Inference principles for emergent collective
behavior in geospatial contexts.

Main Components:
- Core agent framework with Active Inference integration
- Population dynamics and management systems
- Stigmergic communication mechanisms
- Swarm optimization algorithms (ACO, PSO, ABC)
- Domain-specific applications (environmental, disaster response, urban)
- Analysis tools for emergent behavior patterns
- Performance metrics and evaluation frameworks

Integration Points:
- GEO-INFER-ACT: Active Inference for individual agent behaviors
- GEO-INFER-SPACE: Spatial reasoning and H3 indexing for geospatial operations
- GEO-INFER-AGENT: Agent lifecycle management and coordination
- GEO-INFER-MATH: Optimization algorithms and mathematical foundations
- GEO-INFER-TIME: Temporal dynamics and scheduling

Example:
    >>> from geo_infer_ant.core.agent_base import SwarmAgent
    >>> from geo_infer_ant.core.population import AgentPopulation
    >>>
    >>> # Create individual agent with Active Inference
    >>> agent = SwarmAgent(
    ...     agent_id="ant_001",
    ...     position=np.array([37.7749, -122.4194]),
    ...     sensory_range=100.0,
    ...     active_inference_enabled=True
    ... )
    >>>
    >>> # Create agent population
    >>> population = AgentPopulation(
    ...     population_size=1000,
    ...     agent_types=['worker', 'scout', 'soldier'],
    ...     spatial_distribution='clustered'
    ... )
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

# Set up logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "0.3.0"
__author__ = "GEO-INFER Development Team"
__description__ = (
    "Swarm Intelligence and Complex Adaptive Systems for Geospatial Analysis"
)

# Public components are required package dependencies and are imported directly.
from .core.agent_base import SwarmAgent
from .core.population import AgentPopulation
from .core.stigmergy import PheromoneSystem
from .core.digital_stigmergy import DigitalStigmergy
from .algorithms.aco import AntColonyOptimization
from .algorithms.pso import ParticleSwarmOptimization
from .algorithms.abc import ArtificialBeeColony
from .applications.environmental import EnvironmentalMonitoringSwarm
from .applications.disaster import DisasterResponseSwarm
from .applications.urban import UrbanTrafficSwarm
from .analysis.patterns import SwarmPatternAnalyzer
from .analysis.metrics import SwarmPerformanceMetrics
from .utils.config import config_to_dict, load_config, validate_config
from .utils.logging import setup_logging
from .utils.integration import IntegrationManager


# Configuration and setup
def setup_ant_module(config_path: str | None = None) -> dict[str, Any]:
    """
    Set up the GEO-INFER-ANT module with configuration.

    Args:
        config_path: Path to configuration file (YAML/JSON)

    Returns:
        Configuration dictionary with validated parameters, including the
        configured :class:`IntegrationManager` under ``integration_manager``.
    """
    logger.info("Setting up GEO-INFER-ANT module")

    config: dict[str, Any] = {}

    if config_path:
        try:
            config = config_to_dict(load_config(config_path))
            validate_config(config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")

        # Integration settings live outside the AntModuleConfig dataclass
        # contract, so read them from the raw document.
        try:
            raw_path = Path(config_path)
            if raw_path.suffix.lower() in {".yaml", ".yml"}:
                raw_config = yaml.safe_load(raw_path.read_text()) or {}
            else:
                raw_config = json.loads(raw_path.read_text())
            if isinstance(raw_config.get("integrations"), dict):
                config["integrations"] = raw_config["integrations"]
        except Exception as e:
            logger.debug(f"Could not read integration settings from {config_path}: {e}")
    # Set up integrations and keep the manager reachable by callers.
    integration_manager = IntegrationManager()
    integration_manager.setup_integrations(config.get("integrations", {}))
    config["integration_manager"] = integration_manager

    return config


def get_available_components() -> dict[str, list[str]]:
    """
    Get information about available components in the ANT module.

    The mapping is derived from ``__all__`` by inspecting each export's
    defining subpackage, so the export surface and this report cannot drift.

    Returns:
        Dictionary with component availability information
    """
    components: dict[str, list[str]] = {
        "core": [],
        "algorithms": [],
        "applications": [],
        "analysis": [],
        "utils": [],
    }

    for name in __all__:
        if name in {"__version__", "__author__", "__description__"}:
            continue
        module = getattr(globals()[name], "__module__", "")
        for group in ("core", "algorithms", "applications", "analysis"):
            if f".{group}." in module:
                components[group].append(name)
                break
        else:
            components["utils"].append(name)

    return components


# Export main classes and functions
__all__ = [
    # Core classes
    "SwarmAgent",
    "AgentPopulation",
    "PheromoneSystem",
    "DigitalStigmergy",
    # Algorithms
    "AntColonyOptimization",
    "ParticleSwarmOptimization",
    "ArtificialBeeColony",
    # Applications
    "EnvironmentalMonitoringSwarm",
    "DisasterResponseSwarm",
    "UrbanTrafficSwarm",
    # Analysis
    "SwarmPatternAnalyzer",
    "SwarmPerformanceMetrics",
    # Utilities
    "setup_ant_module",
    "get_available_components",
    "load_config",
    "config_to_dict",
    "setup_logging",
    # Module metadata
    "__version__",
    "__author__",
    "__description__",
]
