#!/usr/bin/env python3
"""
LocationConfigLoader: Configuration management for place-based analysis.

This module provides comprehensive configuration loading and validation
for location-specific geospatial analysis parameters, including spatial
bounds, analysis settings, data source configurations, and integration
parameters for different GEO-INFER modules.
"""

import importlib.resources
from copy import deepcopy
import logging
import math
import os
import re
import yaml
from pathlib import Path
from typing import Any
from dataclasses import dataclass

CONFIG_DIR_ENV_VAR = "GEO_INFER_SPACE_CONFIG"
logger = logging.getLogger(__name__)


@dataclass
class LocationBounds:
    """Geographic bounds for a location."""

    north: float
    south: float
    east: float
    west: float

    def __post_init__(self) -> None:
        for name in ("north", "south", "east", "west"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Location bound {name} must be finite and numeric")
        if not -90 <= self.south < self.north <= 90:
            raise ValueError("Expected -90 <= south < north <= 90")
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("Longitude bounds must lie between -180 and 180")
        if self.east == self.west or (self.west == 180 and self.east == -180):
            raise ValueError("Longitude bounds must have positive width")

    def to_bbox(self) -> tuple:
        """Convert to (west, south, east, north) bbox tuple."""
        return (self.west, self.south, self.east, self.north)

    def center(self) -> tuple:
        """Get center point as (lat, lon)."""
        lat = (self.north + self.south) / 2
        east = self.east + 360 if self.east < self.west else self.east
        lon = ((east + self.west) / 2 + 180) % 360 - 180
        return (lat, lon)


class LocationConfigLoader:
    """
    Configuration loader for place-based analysis.

    Handles loading, merging, and validation of configuration from
    packaged base YAML, location YAML, and independent defaults.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        """
        Initialize config loader.

        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = (
            Path(config_dir) if config_dir else self._resolve_default_config_dir()
        )
        self.default_config = self._load_default_config()
        self.loaded_config: dict[str, dict[str, Any]] = {}
        logger.info(f"Config loader initialized with directory: {self.config_dir}")

    @staticmethod
    def _resolve_default_config_dir() -> Path:
        """Resolve the default config directory.

        Honors the explicit GEO_INFER_SPACE_CONFIG env var when set;
        otherwise resolves the packaged config/ directory inside the
        geo_infer_space package via importlib.resources.
        """
        override = os.environ.get(CONFIG_DIR_ENV_VAR)
        if override:
            return Path(override)
        packaged = importlib.resources.files("geo_infer_space").joinpath("config")
        with importlib.resources.as_file(packaged) as config_path:
            return config_path

    def load_location_config(self, location: str) -> dict[str, Any]:
        """
        Load configuration for a specific location.

        Args:
            location: Location identifier (e.g., 'del_norte_county')


        Returns:
            Merged configuration dictionary
        """
        if not isinstance(location, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", location
        ):
            raise ValueError("location must be a simple location identifier")
        config = deepcopy(self.default_config)

        # Load base config
        base_config_path = self.config_dir / "base.yaml"
        if base_config_path.exists():
            with open(base_config_path) as f:
                base_config = yaml.safe_load(f)
            config = self._merge_configs(config, base_config)

        # Load location-specific config
        location_config_path = self.config_dir / f"{location}.yaml"
        if location_config_path.exists():
            with open(location_config_path) as f:
                location_config = yaml.safe_load(f)
            config = self._merge_configs(config, location_config)
        else:
            logger.warning(f"Location config not found: {location_config_path}")

        self._validate_config(config)
        self.loaded_config[location] = deepcopy(config)

        return config

    def get_location_bounds(self, config: dict[str, Any]) -> LocationBounds:
        """Extract location bounds from config."""
        bounds = config.get("location", {}).get("bounds", {})
        return LocationBounds(
            north=bounds.get("north", 90.0),
            south=bounds.get("south", -90.0),
            east=bounds.get("east", 180.0),
            west=bounds.get("west", -180.0),
        )

    def _load_default_config(self) -> dict[str, Any]:
        """Load default configuration values."""
        defaults = {
            "location": {
                "timezone": "UTC",
                "coordinate_systems": {
                    "local_crs": "EPSG:4326",
                    "analysis_crs": "EPSG:3857",
                    "utm_zone": None,
                },
            },
            "spatial": {"h3_resolution": 8, "buffer_distance_meters": 1000},
            "temporal": {"default_frequency": "daily", "lookback_days": 30},
            "data_management": {
                "cache_enabled": True,
                "retention_policy": "1_year",
                "quality_control": {
                    "automated_checks": True,
                    "validation_rules": "standard",
                },
            },
            "reporting": {
                "automated_reports": {
                    "frequency": "monthly",
                    "formats": ["html", "pdf"],
                },
                "dashboard": {
                    "refresh_interval": "hourly",
                    "public_access_level": "summary_only",
                },
            },
        }

        return defaults

    def _merge_configs(self, base: dict, update: dict) -> dict:
        """Merge mappings into an independent result without changing inputs."""
        if not isinstance(base, dict) or not isinstance(update, dict):
            raise ValueError("Configuration files and sections must be mappings")
        result = deepcopy(base)
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration parameters."""
        for section in self.default_config:
            if not isinstance(config.get(section), dict):
                raise ValueError(f"Configuration section {section} must be a mapping")
        # Validate location bounds
        if "location" in config and "bounds" in config["location"]:
            bounds = config["location"]["bounds"]
            if not isinstance(bounds, dict):
                raise ValueError("Location bounds must be a mapping")
            required_bounds = ["north", "south", "east", "west"]

            for bound in required_bounds:
                if bound not in bounds:
                    raise ValueError(f"Missing required location bound: {bound}")

            LocationBounds(**{name: bounds[name] for name in required_bounds})

        # Validate H3 resolution
        if "spatial" in config and "h3_resolution" in config["spatial"]:
            h3_res = config["spatial"]["h3_resolution"]
            if (
                isinstance(h3_res, bool)
                or not isinstance(h3_res, int)
                or not 0 <= h3_res <= 15
            ):
                raise ValueError("H3 resolution must be an integer between 0 and 15")

        logger.debug("Configuration validation passed")
