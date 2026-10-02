"""
Configuration utilities for GEO-INFER-HEALTH module.

Provides functions for loading, validating, and managing configuration files.
"""

from importlib.metadata import version as distribution_version
import os
import json
import importlib.resources
import re
from pathlib import Path
from typing import Any, cast
from functools import lru_cache

import logging
import yaml
from pydantic import BaseModel, ValidationError, Field

logger = logging.getLogger(__name__)


class HealthConfig(BaseModel):
    """Pydantic model for health configuration validation."""

    # Module metadata
    module: dict[str, Any] = Field(default_factory=dict)

    # API configuration
    api: dict[str, Any] = Field(default_factory=dict)

    # Database configuration
    database: dict[str, Any] = Field(default_factory=dict)

    # Logging configuration
    logging: dict[str, Any] = Field(default_factory=dict)

    # Analysis configuration
    analysis: dict[str, Any] = Field(default_factory=dict)

    # Data configuration
    data: dict[str, Any] = Field(default_factory=dict)

    # Performance configuration
    performance: dict[str, Any] = Field(default_factory=dict)

    # Privacy configuration
    privacy: dict[str, Any] = Field(default_factory=dict)

    # Integration configuration
    integration: dict[str, Any] = Field(default_factory=dict)

    # Monitoring configuration
    monitoring: dict[str, Any] = Field(default_factory=dict)

    # Development configuration
    development: dict[str, Any] = Field(default_factory=dict)

    # Advanced configuration
    advanced: dict[str, Any] = Field(default_factory=dict)

    # Custom configuration
    custom: dict[str, Any] = Field(default_factory=dict)


def load_yaml_config(file_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration data

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"Configuration file is empty: {file_path}")
            return {}

        logger.info(f"Loaded configuration from {file_path}")
        return cast(dict[str, Any], config)

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading configuration file {file_path}: {e}")
        raise


def load_json_config(file_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        file_path: Path to the JSON configuration file

    Returns:
        Dictionary containing configuration data

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            config = json.load(f)

        logger.info(f"Loaded configuration from {file_path}")
        return cast(dict[str, Any], config)

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading configuration file {file_path}: {e}")
        raise


def validate_config(config: dict[str, Any]) -> HealthConfig:
    """
    Validate configuration data against the HealthConfig model.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validated HealthConfig object

    Raises:
        ValidationError: If the configuration is invalid
    """
    module = config.get("module", {})
    version = module.get("version")
    if version is not None and not re.fullmatch(r"\d+\.\d+\.\d+", str(version)):
        raise ValueError(
            "module.version must use semantic version format MAJOR.MINOR.PATCH"
        )

    api = config.get("api", {})
    port = api.get("port")
    if port is not None and not (
        isinstance(port, int) or (isinstance(port, str) and port.isdigit())
    ):
        raise ValueError("api.port must be an integer or integer string")

    try:
        validated_config = HealthConfig(**config)
        logger.info("Configuration validation successful")
        return validated_config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def merge_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def resolve_environment_variables(config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve environment variables in configuration values.

    Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration dictionary with environment variables resolved
    """

    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r"\$\{([^:}]+)(?::([^}]*))?\}"

            def replace_var(match: "re.Match[str]") -> str:
                var_name = match.group(1)
                default_value = match.group(2)

                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
                elif default_value is not None:
                    return default_value
                else:
                    logger.warning(
                        f"Environment variable {var_name} not found and no default provided"
                    )
                    return match.group(0)  # Return original if not found

            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        else:
            return value

    return cast(dict[str, Any], resolve_value(config))


@lru_cache(maxsize=1)
def get_default_config_path() -> Path:
    """
    Get the default configuration file path.

    Returns:
        Path to the default configuration file
    """
    # Try to find config in the following order:
    # 1. Environment variable GEO_INFER_HEALTH_CONFIG (explicit override)
    # 2. Packaged resource shipped inside geo_infer_health
    # 3. Current working directory

    env_config = os.getenv("GEO_INFER_HEALTH_CONFIG")
    if env_config and Path(env_config).exists():
        return Path(env_config)

    packaged = importlib.resources.files("geo_infer_health").joinpath(
        "health_config.yaml"
    )
    if packaged.is_file():
        with importlib.resources.as_file(packaged) as path:
            return path

    # Fall back to the working directory only when the packaged resource is
    # unavailable (e.g. namespace/zip layouts without the resource).
    return Path.cwd() / "config" / "health_config.yaml"


def load_config(config_path: str | Path | None = None) -> HealthConfig:
    """
    Load and validate configuration from file.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        Validated HealthConfig object

    Raises:
        FileNotFoundError: If configuration file is not found
        ValidationError: If configuration is invalid
    """
    if config_path is None:
        config_path = get_default_config_path()

    config_path = Path(config_path)

    # Load configuration based on file extension
    if config_path.suffix.lower() in [".yaml", ".yml"]:
        config_data = load_yaml_config(config_path)
    elif config_path.suffix.lower() == ".json":
        config_data = load_json_config(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    # Resolve environment variables
    config_data = resolve_environment_variables(config_data)

    # Validate configuration
    validated_config = validate_config(config_data)

    return validated_config


def save_config(config: HealthConfig | dict[str, Any], file_path: str | Path) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        file_path: Path to save configuration file
    """
    file_path = Path(file_path)

    # Convert to dict if it's a HealthConfig object
    if isinstance(config, HealthConfig):
        config_data = config.model_dump()
    else:
        config_data = config

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save based on file extension
    if file_path.suffix.lower() in [".yaml", ".yml"]:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
    elif file_path.suffix.lower() == ".json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")

    logger.info(f"Configuration saved to {file_path}")


def get_config_value(config: HealthConfig, key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        config: HealthConfig object
        key_path: Dot-separated path to the configuration value (e.g., "api.host")
        default: Default value if key is not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config.model_dump()

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def create_default_config(output_path: str | Path) -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Path where to save the default configuration
    """
    default_config = {
        "module": {
            "name": "GEO-INFER-HEALTH",
            "version": distribution_version("geo-infer-health"),
            "description": "Geospatial Applications for Public Health, Epidemiology, and Healthcare Accessibility",
            "author": "GEO-INFER Framework Team",
            "contact": "health@geo-infer.org",
        },
        "api": {"host": "127.0.0.1", "port": 8000, "workers": 1, "reload": False},
        "database": {"type": "memory"},
        "logging": {"level": "INFO"},
        "analysis": {
            "disease_surveillance": {
                "default_scan_radius_km": 1.0,
                "hotspot_threshold_cases": 5,
            },
            "healthcare_accessibility": {"default_method": "distance"},
        },
        "development": {"debug_mode": False},
    }

    save_config(default_config, output_path)
    logger.info(f"Default configuration created at {output_path}")


# Global configuration cache
_config_cache: HealthConfig | None = None


def get_global_config(force_reload: bool = False) -> HealthConfig:
    """
    Get the global configuration instance.

    Args:
        force_reload: If True, reload configuration from file

    Returns:
        Global HealthConfig instance
    """
    global _config_cache

    if _config_cache is None or force_reload:
        _config_cache = load_config()

    return _config_cache


def reload_global_config() -> HealthConfig:
    """
    Reload the global configuration from file.

    Returns:
        Reloaded HealthConfig instance
    """
    return get_global_config(force_reload=True)
