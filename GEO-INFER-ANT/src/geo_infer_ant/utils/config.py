"""
Configuration Management for GEO-INFER-ANT

This module provides configuration loading, validation, and management utilities
for the GEO-INFER-ANT swarm intelligence module.

Key Features:
- YAML/JSON configuration loading
- Configuration schema validation
- Default configuration generation
- Configuration merging and inheritance
- Environment variable support
"""

import os
import json
import yaml
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from jsonschema import validate, ValidationError

from .spatial import parse_h3_resolution

logger = logging.getLogger(__name__)


# Configuration schema definitions
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "swarm": {
            "type": "object",
            "properties": {
                "population_size": {"type": "integer", "minimum": 1},
                "agent_types": {"type": "array", "items": {"type": "string"}},
                "spatial_distribution": {
                    "type": "string",
                    "enum": ["random", "clustered", "uniform", "custom"],
                },
                "behavioral_heterogeneity": {
                    "type": "string",
                    "enum": ["stochastic", "deterministic", "adaptive"],
                },
            },
            "required": ["population_size", "agent_types"],
        },
        "algorithms": {
            "type": "object",
            "properties": {
                "aco": {
                    "type": "object",
                    "properties": {
                        "number_of_ants": {"type": "integer", "minimum": 1},
                        "pheromone_evaporation_rate": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "alpha": {"type": "number", "minimum": 0},
                        "beta": {"type": "number", "minimum": 0},
                        "max_iterations": {"type": "integer", "minimum": 1},
                    },
                },
                "pso": {
                    "type": "object",
                    "properties": {
                        "swarm_size": {"type": "integer", "minimum": 1},
                        "inertia_weight": {"type": "number"},
                        "cognitive_coefficient": {"type": "number", "minimum": 0},
                        "social_coefficient": {"type": "number", "minimum": 0},
                        "max_iterations": {"type": "integer", "minimum": 1},
                    },
                },
                "abc": {
                    "type": "object",
                    "properties": {
                        "colony_size": {"type": "integer", "minimum": 1},
                        "employed_bees": {"type": "integer", "minimum": 1},
                        "onlooker_bees": {"type": "integer", "minimum": 1},
                        "scout_bees": {"type": "integer", "minimum": 1},
                        "max_iterations": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
        "stigmergy": {
            "type": "object",
            "properties": {
                "pheromone_evaporation_rate": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "pheromone_deposition_amount": {"type": "number", "minimum": 0},
                "diffusion_rate": {"type": "number", "minimum": 0},
                "pheromone_types": {"type": "array", "items": {"type": "string"}},
            },
        },
        "spatial": {
            "type": "object",
            "properties": {
                "bounds": {
                    "type": "object",
                    "properties": {
                        "min_lat": {"type": "number"},
                        "max_lat": {"type": "number"},
                        "min_lng": {"type": "number"},
                        "max_lng": {"type": "number"},
                    },
                },
                "resolution": {"type": "number", "minimum": 0},
                "coordinate_system": {"type": "string"},
            },
        },
        "performance": {
            "type": "object",
            "properties": {
                "evaluation_criteria": {"type": "array", "items": {"type": "string"}},
                "benchmark_datasets": {"type": "array", "items": {"type": "string"}},
                "statistical_analysis": {"type": "array", "items": {"type": "string"}},
            },
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
                "format": {"type": "string"},
                "file": {"type": "string"},
                "console": {"type": "boolean"},
            },
        },
    },
    "required": ["swarm"],
}


@dataclass
class SwarmConfig:
    """Swarm configuration dataclass."""

    population_size: int = 1000
    agent_types: list[str] = field(
        default_factory=lambda: ["worker", "scout", "soldier"]
    )
    spatial_distribution: str = "random"
    behavioral_heterogeneity: str = "stochastic"
    spatial_bounds: dict[str, float] | None = None
    clustering_centers: list[Any] | None = None
    clustering_radius: float = 50.0


@dataclass
class AlgorithmConfig:
    """Algorithm configuration dataclass."""

    aco: dict[str, Any] | None = None
    pso: dict[str, Any] | None = None
    abc: dict[str, Any] | None = None


@dataclass
class StigmergyConfig:
    """Stigmergy configuration dataclass."""

    pheromone_evaporation_rate: float = 0.1
    pheromone_deposition_amount: float = 1.0
    diffusion_rate: float = 0.5
    pheromone_types: list[str] = field(
        default_factory=lambda: ["trail", "alarm", "food", "nest"]
    )


@dataclass
class SpatialConfig:
    """Spatial configuration dataclass."""

    bounds: dict[str, float] | None = None
    resolution: float = 1.0
    coordinate_system: str = "EPSG:4326"


@dataclass
class PerformanceConfig:
    """Performance configuration dataclass."""

    evaluation_criteria: list[str] = field(
        default_factory=lambda: ["efficiency", "robustness", "adaptability"]
    )
    benchmark_datasets: list[str] = field(default_factory=list)
    statistical_analysis: list[str] = field(
        default_factory=lambda: ["hypothesis_testing", "confidence_intervals"]
    )


@dataclass
class LoggingConfig:
    """Logging configuration dataclass."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None
    console: bool = True


@dataclass
class AntModuleConfig:
    """Complete ANT module configuration."""

    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    algorithms: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    stigmergy: StigmergyConfig = field(default_factory=StigmergyConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(
    config_path: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
    validate_schema: bool = True,
) -> AntModuleConfig:
    """
    Load configuration from file or dictionary.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        config_dict: Configuration dictionary (alternative to file)
        validate_schema: Whether to validate against schema

    Returns:
        AntModuleConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If configuration doesn't match schema
        ValueError: If configuration is invalid
    """
    config_data: dict[str, Any] = {}

    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        logger.info(f"Loaded configuration from {config_path}")

    # Load from dictionary if provided
    elif config_dict is not None:
        config_data = config_dict.copy()
        logger.info("Loaded configuration from dictionary")

    # Use environment variables if no config provided
    else:
        config_data = _load_from_environment()
        logger.info("Loaded configuration from environment variables")

    # Normalize the detailed simulation document used by
    # ``config/example_config.yaml`` into the runtime dataclass contract.
    if "swarm" not in config_data and "agents" in config_data:
        config_data = _normalize_simulation_config(config_data)

    # Merge with defaults
    config_data = _merge_with_defaults(config_data)

    # Validate schema if requested
    if validate_schema:
        validate_config(config_data)

    # Convert to dataclass
    return _dict_to_config(config_data)


def validate_config(config: dict[str, Any] | AntModuleConfig) -> bool:
    """
    Validate configuration against schema.

    Args:
        config: Configuration dictionary or AntModuleConfig instance

    Returns:
        True if valid

    Raises:
        ValidationError: If configuration is invalid
    """
    # Convert dataclass to dict if needed
    if isinstance(config, AntModuleConfig):
        config_dict = config_to_dict(config)
    else:
        config_dict = config
    if "swarm" not in config_dict and "agents" in config_dict:
        config_dict = _normalize_simulation_config(config_dict)

    try:
        validate(instance=config_dict, schema=CONFIG_SCHEMA)
        logger.info("Configuration validation passed")
        return True
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e.message}")
        raise ValidationError(
            f"Invalid configuration: {e.message}", e.instance, e.schema_path
        )


def _load_from_environment() -> dict[str, Any]:
    """Load configuration from environment variables."""
    config: dict[str, Any] = {}

    # Swarm configuration
    population_size = os.getenv("ANT_POPULATION_SIZE")
    if population_size:
        config.setdefault("swarm", {})["population_size"] = int(population_size)
    agent_types_env = os.getenv("ANT_AGENT_TYPES")
    if agent_types_env:
        config.setdefault("swarm", {})["agent_types"] = agent_types_env.split(",")

    # Algorithm configuration
    aco_ants = os.getenv("ANT_ACO_ANTS")
    if aco_ants:
        config.setdefault("algorithms", {}).setdefault("aco", {})["number_of_ants"] = (
            int(aco_ants)
        )

    # Stigmergy configuration
    evaporation_rate = os.getenv("ANT_PHEROMONE_EVAPORATION")
    if evaporation_rate:
        config.setdefault("stigmergy", {})["pheromone_evaporation_rate"] = float(
            evaporation_rate
        )

    # Logging configuration
    log_level = os.getenv("ANT_LOG_LEVEL")
    if log_level:
        config.setdefault("logging", {})["level"] = log_level

    return config


def _normalize_simulation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the detailed simulation document to ``AntModuleConfig``."""
    agents = config.get("agents", {})
    agent_types = agents.get("agent_types", ["worker", "scout", "soldier"])
    if agent_types and isinstance(agent_types[0], dict):
        agent_types = [item["name"] for item in agent_types]

    environment = config.get("environment", {})
    spatial = environment.get("spatial", {})
    stigmergy = config.get("stigmergy", {})
    pheromone_types = stigmergy.get(
        "pheromone_types", ["trail", "food", "alarm", "nest"]
    )
    if pheromone_types and isinstance(pheromone_types[0], dict):
        pheromone_types = [item["name"] for item in pheromone_types]

    algorithms = config.get("algorithms", {})
    aco = algorithms.get("ant_colony_optimization", {}).get("parameters", {})
    pso = algorithms.get("particle_swarm_optimization", {}).get("parameters", {})
    abc = algorithms.get("artificial_bee_colony", {}).get("parameters", {})

    resolution = spatial.get("resolution", 8)
    if isinstance(resolution, str):
        resolution = parse_h3_resolution(resolution)
    detailed_performance = config.get("performance", {})
    detailed_logging = config.get("output", {}).get("logging", {})
    normalized = {
        "swarm": {
            "population_size": agents.get("population_size", 1000),
            "agent_types": agent_types,
            "spatial_distribution": "random",
            "behavioral_heterogeneity": "stochastic",
        },
        "algorithms": {
            "aco": {
                **aco,
                "pheromone_evaporation_rate": aco.get(
                    "evaporation_rate", aco.get("pheromone_evaporation_rate", 0.1)
                ),
            },
            "pso": pso,
            "abc": abc,
        },
        "stigmergy": {
            "pheromone_types": pheromone_types,
            "pheromone_evaporation_rate": stigmergy.get("evaporation_rate", 0.1),
            "pheromone_deposition_amount": stigmergy.get("deposition_amount", 1.0),
            "diffusion_rate": stigmergy.get("diffusion_rate", 0.5),
        },
        "spatial": {
            "bounds": spatial.get("bounds"),
            "resolution": resolution,
            "coordinate_system": spatial.get("coordinate_system", "WGS84"),
        },
        "performance": {
            "evaluation_criteria": detailed_performance.get(
                "evaluation_criteria", ["efficiency", "robustness", "adaptability"]
            ),
            "benchmark_datasets": detailed_performance.get("benchmark_datasets", []),
            "statistical_analysis": detailed_performance.get(
                "statistical_analysis", ["hypothesis_testing", "confidence_intervals"]
            ),
        },
        "logging": {
            "level": detailed_logging.get("level", "INFO"),
            "file": detailed_logging.get("file", detailed_logging.get("file_path")),
            "console": detailed_logging.get(
                "console", detailed_logging.get("console_output", True)
            ),
        },
    }
    return normalized


def _merge_with_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Merge configuration with default values."""
    defaults = {
        "swarm": {
            "population_size": 1000,
            "agent_types": ["worker", "scout", "soldier"],
            "spatial_distribution": "random",
            "behavioral_heterogeneity": "stochastic",
        },
        "algorithms": {
            "aco": {
                "number_of_ants": 50,
                "pheromone_evaporation_rate": 0.1,
                "alpha": 1.0,
                "beta": 2.0,
                "max_iterations": 100,
            },
            "pso": {
                "swarm_size": 30,
                "inertia_weight": 0.9,
                "cognitive_coefficient": 2.0,
                "social_coefficient": 2.0,
                "max_iterations": 100,
            },
            "abc": {
                "colony_size": 50,
                "employed_bees": 25,
                "onlooker_bees": 25,
                "scout_bees": 5,
                "max_iterations": 100,
            },
        },
        "stigmergy": {
            "pheromone_evaporation_rate": 0.1,
            "pheromone_deposition_amount": 1.0,
            "diffusion_rate": 0.5,
            "pheromone_types": ["trail", "alarm", "food", "nest"],
        },
        "spatial": {"resolution": 1.0, "coordinate_system": "EPSG:4326"},
        "performance": {
            "evaluation_criteria": ["efficiency", "robustness", "adaptability"],
            "benchmark_datasets": [],
            "statistical_analysis": ["hypothesis_testing", "confidence_intervals"],
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True,
        },
    }

    # Deep merge
    merged: dict[str, Any] = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in defaults.items()
    }
    for key, value in config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged


def _dict_to_config(config_dict: dict[str, Any]) -> AntModuleConfig:
    """Convert dictionary to AntModuleConfig dataclass."""
    return AntModuleConfig(
        swarm=SwarmConfig(**config_dict.get("swarm", {})),
        algorithms=AlgorithmConfig(**config_dict.get("algorithms", {})),
        stigmergy=StigmergyConfig(**config_dict.get("stigmergy", {})),
        spatial=SpatialConfig(**config_dict.get("spatial", {})),
        performance=PerformanceConfig(**config_dict.get("performance", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
    )


def config_to_dict(config: AntModuleConfig) -> dict[str, Any]:
    """Convert AntModuleConfig dataclass to dictionary."""
    return {
        "swarm": asdict(config.swarm),
        "algorithms": asdict(config.algorithms),
        "stigmergy": asdict(config.stigmergy),
        "spatial": asdict(config.spatial),
        "performance": asdict(config.performance),
        "logging": asdict(config.logging),
    }


def save_config(
    config: AntModuleConfig, config_path: str | Path, format: str = "yaml"
) -> None:
    """
    Save configuration to file.

    Args:
        config: AntModuleConfig instance
        config_path: Path to save configuration
        format: File format ('yaml' or 'json')
    """
    config_path = Path(config_path)
    config_dict = config_to_dict(config)

    with open(config_path, "w") as f:
        if format.lower() == "yaml":
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == "json":
            json.dump(config_dict, f, indent=2, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved configuration to {config_path}")


def get_default_config() -> AntModuleConfig:
    """Get default configuration."""
    return AntModuleConfig()


def update_config(config: AntModuleConfig, updates: dict[str, Any]) -> AntModuleConfig:
    """
    Update configuration with new values.

    Args:
        config: Current configuration
        updates: Dictionary of updates

    Returns:
        Updated AntModuleConfig instance
    """
    config_dict = config_to_dict(config)

    # Deep update
    for key, value in updates.items():
        if (
            key in config_dict
            and isinstance(config_dict[key], dict)
            and isinstance(value, dict)
        ):
            config_dict[key].update(value)
        else:
            config_dict[key] = value

    # Validate and convert back
    validate_config(config_dict)
    return _dict_to_config(config_dict)
