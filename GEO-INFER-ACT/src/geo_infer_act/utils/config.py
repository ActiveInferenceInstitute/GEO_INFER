"""
Configuration utilities for GEO-INFER-ACT.
"""

import os
from typing import Any, cast
import yaml


def load_config(path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}
    return cast(dict[str, Any], config)


def save_config(config: dict[str, Any], path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def merge_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Configuration with override values

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_dicts(base[key], value)
            else:
                base[key] = value

    _merge_dicts(merged, override_config)
    return merged


def get_config_value(
    config: dict[str, Any], path: str, default: Any | None = None
) -> Any:
    """
    Get a configuration value using a dot-notated path.

    Args:
        config: Configuration dictionary
        path: Dot-notated path to the value (e.g., 'active_inference.free_energy.algorithm')
        default: Default value to return if path not found

    Returns:
        Configuration value or default
    """
    keys = path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
