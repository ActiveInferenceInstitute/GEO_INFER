#!/usr/bin/env python3
"""
Enhanced Configuration Module for Cascadia Framework

This module provides centralized configuration management:
- Analysis parameters
- Visualization settings
- Data processing options
- Module configurations
- Environment-specific settings

Provides a single source of truth for all framework settings.
"""

import yaml
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""

    h3_resolution: int = 8
    target_counties: list[str] = None
    active_modules: list[str] = None
    spatial_analysis_enabled: bool = False
    force_refresh: bool = False
    skip_cache: bool = False
    validate_h3: bool = False
    debug_mode: bool = False
    verbose_logging: bool = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    generate_dashboard: bool = False
    lightweight_viz: bool = True
    datashader_viz: bool = False
    deepscatter_viz: bool = False
    interactive_maps: bool = True
    static_plots: bool = True
    export_data: bool = True
    color_schemes: dict[str, dict[str, str]] = None
    default_center: list[float] = None
    default_zoom: int = 10
    tile_layer: str = "OpenStreetMap"


@dataclass
class DataConfig:
    """Configuration for data processing."""

    output_dir: str = "output"
    export_format: str = "geojson"
    keep_recent_runs: int = 3
    data_quality_threshold: float = 0.8
    cache_enabled: bool = True
    real_data_priority: bool = True


@dataclass
class ModuleConfig:
    """Configuration for individual modules."""

    zoning: dict[str, Any] = None
    current_use: dict[str, Any] = None
    ownership: dict[str, Any] = None
    improvements: dict[str, Any] = None
    water_rights: dict[str, Any] = None
    ground_water: dict[str, Any] = None
    surface_water: dict[str, Any] = None
    power_source: dict[str, Any] = None
    mortgage_debt: dict[str, Any] = None


@dataclass
class CascadiaConfig:
    """Main configuration class for Cascadia framework."""

    analysis: AnalysisConfig
    visualization: VisualizationConfig
    data: DataConfig
    modules: ModuleConfig
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class EnhancedConfigManager:
    """
    Enhanced configuration manager for Cascadia framework.

    Provides:
    - Centralized configuration management
    - Environment-specific settings
    - Validation and defaults
    - Configuration persistence
    """

    def __init__(self, config_dir: Path = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.default_config = self._create_default_config()

        # Load configuration
        self.config = self.load_configuration()

        logger.info("Enhanced Configuration Manager initialized")

    def _create_default_config(self) -> CascadiaConfig:
        """Create default configuration."""
        return CascadiaConfig(
            analysis=AnalysisConfig(
                h3_resolution=8,
                target_counties=["CA:Del Norte"],
                active_modules=["zoning", "current_use", "ownership", "improvements"],
                spatial_analysis_enabled=False,
                force_refresh=False,
                skip_cache=False,
                validate_h3=False,
                debug_mode=False,
                verbose_logging=False,
            ),
            visualization=VisualizationConfig(
                generate_dashboard=False,
                lightweight_viz=True,
                datashader_viz=False,
                deepscatter_viz=False,
                interactive_maps=True,
                static_plots=True,
                export_data=True,
                color_schemes={
                    "zoning": {
                        "Agricultural": "#90EE90",
                        "Residential": "#FFB6C1",
                        "Commercial": "#FFD700",
                        "Industrial": "#A0522D",
                        "Conservation": "#228B22",
                    },
                    "current_use": {
                        "Agriculture": "#90EE90",
                        "Forest": "#228B22",
                        "Residential": "#FFB6C1",
                        "Commercial": "#FFD700",
                        "Industrial": "#A0522D",
                        "Open Space": "#98FB98",
                    },
                    "ownership": {
                        "Private": "#FF6B6B",
                        "Public": "#4ECDC4",
                        "Corporate": "#45B7D1",
                        "Trust": "#96CEB4",
                    },
                    "improvements": {
                        "High": "#FF0000",
                        "Medium": "#FFA500",
                        "Low": "#FFFF00",
                        "None": "#808080",
                    },
                },
                default_center=[41.7558, -124.2016],
                default_zoom=10,
                tile_layer="OpenStreetMap",
            ),
            data=DataConfig(
                output_dir="output",
                export_format="geojson",
                keep_recent_runs=3,
                data_quality_threshold=0.8,
                cache_enabled=True,
                real_data_priority=True,
            ),
            modules=ModuleConfig(
                zoning={"enabled": True, "data_sources": ["county", "state"]},
                current_use={"enabled": True, "data_sources": ["usda", "state"]},
                ownership={"enabled": True, "data_sources": ["assessor", "state"]},
                improvements={"enabled": True, "data_sources": ["permit", "state"]},
                water_rights={"enabled": False, "data_sources": ["state"]},
                ground_water={"enabled": False, "data_sources": ["usgs", "state"]},
                surface_water={"enabled": False, "data_sources": ["usgs", "state"]},
                power_source={"enabled": False, "data_sources": ["utility", "state"]},
                mortgage_debt={"enabled": False, "data_sources": ["federal", "state"]},
            ),
        )

    def load_configuration(self) -> CascadiaConfig:
        """
        Load configuration from files or create default.

        Returns:
            Loaded configuration
        """
        config_file = self.config_dir / "cascadia_config.yaml"

        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)

                # Merge with default configuration
                config = self._merge_configs(self.default_config, config_data)
                logger.info(f"Configuration loaded from {config_file}")
                return config

            except Exception as e:
                logger.warning(f"Failed to load configuration from {config_file}: {e}")
                logger.info("Using default configuration")
                return self.default_config
        else:
            # Create default configuration file
            self.save_configuration(self.default_config)
            logger.info(f"Created default configuration at {config_file}")
            return self.default_config

    def save_configuration(self, config: CascadiaConfig):
        """
        Save configuration to file.

        Args:
            config: Configuration to save
        """
        config_file = self.config_dir / "cascadia_config.yaml"

        try:
            # Convert to dictionary
            config_dict = self._config_to_dict(config)

            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def update_configuration(self, updates: dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        try:
            # Convert current config to dict
            current_dict = self._config_to_dict(self.config)

            # Apply updates
            self._deep_update(current_dict, updates)

            # Convert back to config object
            self.config = self._dict_to_config(current_dict)

            # Save updated configuration
            self.save_configuration(self.config)

            logger.info("Configuration updated successfully")

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")

    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return self.config.analysis

    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self.config.visualization

    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.config.data

    def get_all_module_configs(self) -> ModuleConfig:
        """Get the full module configuration object."""
        return self.config.modules

    def get_active_modules(self) -> list[str]:
        """Get list of active modules."""
        return self.config.analysis.active_modules

    def is_module_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled."""
        if module_name in self.config.analysis.active_modules:
            module_config = getattr(self.config.modules, module_name, {})
            return module_config.get("enabled", True) if module_config else True
        return False

    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get configuration for a specific module."""
        module_config = getattr(self.config.modules, module_name, {})
        return module_config if module_config else {}

    def validate_configuration(self) -> dict[str, Any]:
        """
        Validate configuration and return comprehensive validation results.

        Returns:
            Dictionary with validation results including detailed error analysis
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "validation_details": {},
            "recommendations": [],
        }

        # Validate H3 resolution
        h3_res = self.config.analysis.h3_resolution
        if not isinstance(h3_res, int) or not (0 <= h3_res <= 15):
            validation_results["errors"].append(
                f"H3 resolution must be an integer between 0 and 15, got {h3_res}"
            )
            validation_results["valid"] = False
        else:
            # Provide resolution guidance
            if h3_res < 6:
                validation_results["recommendations"].append(
                    f"Low H3 resolution ({h3_res}) may result in very large hexagons"
                )
            elif h3_res > 12:
                validation_results["recommendations"].append(
                    f"High H3 resolution ({h3_res}) may result in very small hexagons and slow processing"
                )

        validation_results["validation_details"]["h3_resolution"] = {
            "value": h3_res,
            "valid": 0 <= h3_res <= 15,
            "area_km2": self._get_h3_resolution_area(h3_res),
        }

        # Validate target counties format
        target_counties = self.config.analysis.target_counties
        if not target_counties:
            validation_results["warnings"].append(
                "No target counties specified - will use default Del Norte County"
            )
            validation_results["recommendations"].append(
                "Specify target counties in format 'STATE:County' (e.g., 'CA:Del Norte')"
            )
        else:
            # Validate county format
            valid_counties = []
            invalid_counties = []
            for county in target_counties:
                if ":" in county:
                    parts = county.split(":", 1)
                    if len(parts) == 2 and len(parts[0]) == 2 and len(parts[1].strip()) > 0:
                        valid_counties.append(county)
                    else:
                        invalid_counties.append(county)
                else:
                    invalid_counties.append(county)

            if invalid_counties:
                validation_results["errors"].append(
                    f"Invalid county format: {invalid_counties}. Use 'STATE:County' format"
                )
                validation_results["valid"] = False

            validation_results["validation_details"]["target_counties"] = {
                "specified": len(target_counties),
                "valid": len(valid_counties),
                "invalid": invalid_counties,
            }

        # Validate active modules
        active_modules = self.config.analysis.active_modules
        if not active_modules:
            validation_results["errors"].append("No active modules specified")
            validation_results["valid"] = False
        else:
            # Validate module names
            valid_modules = [
                "zoning",
                "current_use",
                "ownership",
                "improvements",
                "water_rights",
                "ground_water",
                "surface_water",
                "power_source",
                "mortgage_debt",
            ]
            invalid_modules = [m for m in active_modules if m not in valid_modules]

            if invalid_modules:
                validation_results["errors"].append(
                    f"Invalid modules: {invalid_modules}. Valid modules: {valid_modules}"
                )
                validation_results["valid"] = False

            validation_results["validation_details"]["active_modules"] = {
                "specified": len(active_modules),
                "valid": len(active_modules) - len(invalid_modules),
                "invalid": invalid_modules,
                "available_modules": valid_modules,
            }

        # Validate output directory
        output_dir = Path(self.config.data.output_dir)
        parent_exists = output_dir.parent.exists()
        can_create = True

        try:
            # Test if we can create the directory
            output_dir.mkdir(parents=True, exist_ok=True)
            # Clean up test directory if created
            if not output_dir.exists():
                output_dir.rmdir()
        except PermissionError:
            validation_results["errors"].append(
                f"Cannot create output directory: {output_dir} (permission denied)"
            )
            validation_results["valid"] = False
            can_create = False
        except Exception as e:
            validation_results["warnings"].append(
                f"Cannot create output directory: {output_dir} ({e})"
            )
            can_create = False

        validation_results["validation_details"]["output_directory"] = {
            "path": str(output_dir),
            "parent_exists": parent_exists,
            "can_create": can_create,
            "writable": (
                can_create and output_dir.parent.is_dir() if output_dir.parent.exists() else False
            ),
        }

        # Validate data quality threshold
        quality_threshold = self.config.data.data_quality_threshold
        if not isinstance(quality_threshold, (int, float)) or not (0.0 <= quality_threshold <= 1.0):
            validation_results["errors"].append(
                f"Data quality threshold must be between 0.0 and 1.0, got {quality_threshold}"
            )
            validation_results["valid"] = False
        else:
            if quality_threshold > 0.9:
                validation_results["recommendations"].append(
                    f"High quality threshold ({quality_threshold}) may reject valid data sources"
                )
            elif quality_threshold < 0.5:
                validation_results["recommendations"].append(
                    f"Low quality threshold ({quality_threshold}) may accept low-quality data"
                )

        validation_results["validation_details"]["data_quality_threshold"] = {
            "value": quality_threshold,
            "valid": 0.0 <= quality_threshold <= 1.0,
        }

        # Validate export format
        export_format = self.config.data.export_format
        valid_formats = ["geojson", "csv", "json", "shp", "gpkg"]
        if export_format not in valid_formats:
            validation_results["errors"].append(
                f"Invalid export format: {export_format}. Valid formats: {valid_formats}"
            )
            validation_results["valid"] = False

        validation_results["validation_details"]["export_format"] = {
            "value": export_format,
            "valid": export_format in valid_formats,
            "available_formats": valid_formats,
        }

        # Validate module configurations
        module_validation = self._validate_module_configurations()
        validation_results["validation_details"]["modules"] = module_validation
        if not module_validation["all_valid"]:
            validation_results["warnings"].extend(module_validation["warnings"])

        # Check for conflicting settings
        if self.config.analysis.force_refresh and self.config.analysis.skip_cache:
            validation_results["warnings"].append(
                "Both force_refresh and skip_cache are enabled - force_refresh takes precedence"
            )

        if self.config.data.cache_enabled and self.config.analysis.skip_cache:
            validation_results["warnings"].append(
                "Cache is enabled but skip_cache is set - cache will be ignored"
            )

        # Generate final recommendations
        if validation_results["valid"]:
            validation_results["recommendations"].append(
                "Configuration validation passed - framework ready to run"
            )
        else:
            validation_results["recommendations"].append(
                "Configuration has errors that must be fixed before running"
            )

        return validation_results

    def _get_h3_resolution_area(self, resolution: int) -> float:
        """Get approximate area of H3 hexagon at given resolution in km²."""
        # Approximate areas for H3 resolutions (km²)
        areas = {
            0: 4357449,
            1: 609298,
            2: 85302,
            3: 11942,
            4: 1672,
            5: 234,
            6: 33,
            7: 5,
            8: 0.7,
            9: 0.1,
            10: 0.015,
            11: 0.002,
            12: 0.0003,
            13: 0.00005,
            14: 0.000007,
            15: 0.000001,
        }
        return areas.get(resolution, 0)

    def _validate_module_configurations(self) -> dict[str, Any]:
        """Validate module-specific configurations."""
        validation = {"all_valid": True, "warnings": [], "details": {}}

        for module_name in [
            "zoning",
            "current_use",
            "ownership",
            "improvements",
            "water_rights",
            "ground_water",
            "surface_water",
            "power_source",
            "mortgage_debt",
        ]:
            module_config = getattr(self.config.modules, module_name, None)
            if module_config is None:
                continue

            module_valid = True
            module_warnings = []

            # Check if module is in active modules but not enabled
            if module_name in self.config.analysis.active_modules:
                if isinstance(module_config, dict) and not module_config.get("enabled", True):
                    module_warnings.append(
                        f"Module {module_name} is active but disabled in configuration"
                    )
                    module_valid = False

            # Validate data sources if specified
            if isinstance(module_config, dict) and "data_sources" in module_config:
                data_sources = module_config["data_sources"]
                if not isinstance(data_sources, list):
                    module_warnings.append(f"Module {module_name} data_sources must be a list")
                    module_valid = False

            validation["details"][module_name] = {
                "valid": module_valid,
                "warnings": module_warnings,
            }

            if not module_valid:
                validation["all_valid"] = False
                validation["warnings"].extend(module_warnings)

        return validation

    def _merge_configs(
        self, default_config: CascadiaConfig, updates: dict[str, Any]
    ) -> CascadiaConfig:
        """Merge default configuration with updates."""
        default_dict = self._config_to_dict(default_config)
        self._deep_update(default_dict, updates)
        return self._dict_to_config(default_dict)

    def _config_to_dict(self, config: CascadiaConfig) -> dict[str, Any]:
        """Convert configuration object to dictionary."""
        config_dict = asdict(config)

        # Convert dataclass objects to dictionaries
        for key, value in config_dict.items():
            if hasattr(value, "__dataclass_fields__"):
                config_dict[key] = asdict(value)

        return config_dict

    def _dict_to_config(self, config_dict: dict[str, Any]) -> CascadiaConfig:
        """Convert dictionary to configuration object."""
        # Reconstruct nested dataclass objects
        if "analysis" in config_dict:
            config_dict["analysis"] = AnalysisConfig(**config_dict["analysis"])

        if "visualization" in config_dict:
            config_dict["visualization"] = VisualizationConfig(**config_dict["visualization"])

        if "data" in config_dict:
            config_dict["data"] = DataConfig(**config_dict["data"])

        if "modules" in config_dict:
            config_dict["modules"] = ModuleConfig(**config_dict["modules"])

        return CascadiaConfig(**config_dict)

    def _deep_update(self, base_dict: dict[str, Any], updates: dict[str, Any]):
        """Recursively update dictionary with new values."""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


def create_enhanced_config_manager(config_dir: Path = None) -> EnhancedConfigManager:
    """
    Create an enhanced configuration manager instance.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        EnhancedConfigManager instance
    """
    return EnhancedConfigManager(config_dir)
