"""
Configuration utilities for GEO-INFER-OPS.

This module provides functionality for loading and validating configuration
from YAML files, environment variables, and command-line arguments.
"""

import importlib.resources
import os

import yaml
from typing import Any, cast


def find_config_file(config_path: str | None = None) -> str:
    """
    Find the configuration file to use.

    Order of precedence:
    1. Explicitly provided path (must exist; a missing path raises
       FileNotFoundError rather than being silently ignored)
    2. Environment variable GEO_INFER_OPS_CONFIG (must exist when set)
    3. ./config/local.yaml
    4. ./config/example.yaml


    Args:
        config_path: Optional explicit path to configuration file

    Returns:
        Path to the configuration file to use

    Raises:
        FileNotFoundError: If an explicitly provided path does not exist, or
            if no configuration file could be found
    """
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Explicitly provided configuration file does not exist: {config_path}"
            )
        return config_path

    if env_path := os.environ.get("GEO_INFER_OPS_CONFIG"):
        if os.path.exists(env_path):
            return env_path
        raise FileNotFoundError(
            f"GEO_INFER_OPS_CONFIG is set but does not exist: {env_path}"
        )

    # Try to find config relative to the current working directory
    local_config = os.path.join("config", "local.yaml")
    if os.path.exists(local_config):
        return local_config

    example_config = os.path.join("config", "example.yaml")
    if os.path.exists(example_config):
        return example_config

    # Fall back to the config bundled inside the installed package.
    package_resources = importlib.resources.files("geo_infer_ops")
    for name in ("local.yaml", "example.yaml"):
        candidate = package_resources.joinpath("config", name)
        if candidate.is_file():
            with importlib.resources.as_file(candidate) as path:
                return str(path)

    raise FileNotFoundError("No configuration file found")


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Optional explicit path to configuration file

    Returns:
        Dictionary containing the configuration
    """
    config_file = find_config_file(config_path)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    # Format: GEO_INFER_OPS_SECTION_KEY
    # Example: GEO_INFER_OPS_LOGGING_LEVEL=DEBUG
    for env_var, raw_value in os.environ.items():
        if env_var.startswith("GEO_INFER_OPS_"):
            parts = env_var.lower().split("_")[3:]

            if len(parts) < 1:
                continue

            section = config
            for part in parts[:-1]:
                if part not in section:
                    section[part] = {}
                section = section[part]

            value: Any = raw_value
            # Convert to appropriate type if possible
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                value = float(value)

            section[parts[-1]] = value

    return cast(dict[str, Any], config)
