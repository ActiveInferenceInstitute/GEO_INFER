"""
Helper utilities for GEO-INFER-COG

This module provides utility functions for common operations in the
cognitive processing module, including data loading, model management,
and configuration handling.

Utility Functions:
- Cognitive profile loading and saving
- Model configuration management
- Data format conversion utilities
- Performance monitoring helpers
- Logging and debugging utilities

Integration Points:
- GEO-INFER-DATA: Data loading and format utilities
- GEO-INFER-CONFIG: Configuration management patterns
- GEO-INFER-LOG: Logging utilities and patterns
"""

import numpy as np
import logging
import json
import yaml
from typing import Any, cast
from datetime import datetime, timedelta
from pathlib import Path

from ..models.user_profiles import UserCognitiveProfile

logger = logging.getLogger(__name__)


def load_cognitive_profile(
    user_id: str, profile_path: str | Path
) -> UserCognitiveProfile | None:
    """
    Load user cognitive profile from file.

    Args:
        user_id: User identifier
        profile_path: Path to profile file or directory

    Returns:
        Loaded user profile or None if not found
    """
    profile_path = Path(profile_path)

    # Try loading from specific file
    if profile_path.is_file():
        try:
            with open(profile_path) as f:
                profile_data = json.load(f)
                return UserCognitiveProfile.import_profile(profile_data)
        except Exception as e:
            logger.error(f"Error loading profile from {profile_path}: {str(e)}")
            return None

    # Try loading from directory
    if profile_path.is_dir():
        profile_file = profile_path / f"{user_id}_profile.json"
        if profile_file.exists():
            try:
                with open(profile_file) as f:
                    profile_data = json.load(f)
                    return UserCognitiveProfile.import_profile(profile_data)
            except Exception as e:
                logger.error(f"Error loading profile from {profile_file}: {str(e)}")

    logger.warning(f"Profile not found for user {user_id} at {profile_path}")
    return None


def save_cognitive_profile(
    profile: UserCognitiveProfile, profile_path: str | Path
) -> bool:
    """
    Save user cognitive profile to file.

    Args:
        profile: User profile to save
        profile_path: Path to save profile file

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        profile_path = Path(profile_path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        profile_data = profile.export_profile()

        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2, default=str)

        logger.info(f"Profile saved for user {profile.user_id} to {profile_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving profile for user {profile.user_id}: {str(e)}")
        return False


def load_cognitive_model(
    model_path: str | Path, model_type: str = "auto"
) -> dict[str, Any]:
    """
    Load cognitive model from file.

    Args:
        model_path: Path to model file (str or Path)
        model_type: Type of model ('perception', 'reasoning', 'memory', 'auto')

    Returns:
        Loaded model configuration
    """
    try:
        model_path = Path(model_path)
        with model_path.open("r") as f:
            if model_path.suffix.lower() in [".yaml", ".yml"]:
                model_config = yaml.safe_load(f)
            else:
                model_config = json.load(f)

        # Validate model configuration
        if model_type != "auto":
            from .validation import validate_cognitive_model

            validation = validate_cognitive_model(model_config, model_type)

            if not validation["valid"]:
                logger.warning(
                    f"Model configuration has validation issues: {validation['errors']}"
                )

        logger.info(f"Model loaded from {model_path}")
        return cast(dict[str, Any], model_config)

    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return {}


def save_cognitive_model(
    model_config: dict[str, Any], model_path: str | Path, model_type: str = "auto"
) -> bool:
    """
    Save cognitive model configuration to file.

    Args:
        model_config: Model configuration to save
        model_path: Path to save model file
        model_type: Type of model for validation

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Validate before saving
        if model_type != "auto":
            from .validation import validate_cognitive_model

            validation = validate_cognitive_model(model_config, model_type)

            if not validation["valid"]:
                logger.warning(
                    f"Model configuration validation failed: {validation['errors']}"
                )
                # Continue saving despite validation warnings

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "w") as f:
            if model_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(model_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(model_config, f, indent=2, default=str)

        logger.info(f"Model saved to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {str(e)}")
        return False


def create_default_cognitive_config() -> dict[str, Any]:
    """Create default cognitive processing configuration."""
    return {
        "core": {
            "cognitive_framework": "bayesian_attention",
            "spatial_resolution": "adaptive",
            "temporal_modeling": "working_memory",
            "uncertainty_handling": "probabilistic",
        },
        "models": {
            "perception": {
                "attention_capacity": 1.0,
                "focus_radius": 0.5,
                "saliency_threshold": 0.3,
                "grouping_strength": 0.7,
                "scale_sensitivity": 0.8,
            },
            "reasoning": {
                "reasoning_type": "qualitative_spatial",
                "uncertainty_method": "probabilistic",
                "max_reasoning_depth": 5,
                "confidence_threshold": 0.6,
            },
            "memory": {
                "memory_types": ["working", "long_term", "episodic"],
                "consolidation_threshold": 0.7,
                "consolidation_delay": 300,
                "decay_rate": 0.1,
                "working_memory_capacity": 7,
                "long_term_capacity": 10000,
            },
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "rate_limiting": True,
            "cors_enabled": True,
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/geo_infer_cog.log",
            "max_file_size": "10MB",
            "backup_count": 5,
        },
    }


def setup_cognitive_logging(config: dict[str, Any]) -> None:
    """Attach file/stream handlers to the ``geo_infer_cog`` logger.

    CLI-only helper. Command-line entrypoints may call this to route
    ``geo_infer_cog`` records to a log file and stderr. Library modules must
    stay passive: they log through ``logging.getLogger(__name__)`` and never
    call ``basicConfig``, add handlers, or set levels themselves.
    """
    logging_config = config.get("logging", {})

    log_level = getattr(logging, logging_config.get("level", "INFO").upper())
    log_file = logging_config.get("file_path", "logs/geo_infer_cog.log")

    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cog_logger = logging.getLogger("geo_infer_cog")
    cog_logger.setLevel(log_level)

    # Idempotent: repeated CLI invocations must not stack duplicate handlers
    if not cog_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        cog_logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        cog_logger.addHandler(stream_handler)

    logger.info("Cognitive processing logging configured")


def calculate_cognitive_load(
    spatial_data: dict[str, Any], user_profile: UserCognitiveProfile | None = None
) -> float:
    """
    Calculate cognitive load for processing spatial data.

    Args:
        spatial_data: Spatial data to process
        user_profile: User profile for personalized load calculation

    Returns:
        Estimated cognitive load (0-1)
    """
    load_factors = []

    # Base load from data complexity
    if "geometries" in spatial_data:
        geometry_count = len(spatial_data["geometries"])
        load_factors.append(
            min(1.0, geometry_count / 10.0)
        )  # Normalize to 10 geometries

    if "geometry" in spatial_data:
        geometry = spatial_data["geometry"]
        coord_count = len(str(geometry.get("coordinates", [])))
        load_factors.append(
            min(1.0, coord_count / 1000.0)
        )  # Normalize coordinate complexity

    # Properties complexity
    if "properties" in spatial_data:
        prop_count = len(spatial_data["properties"])
        load_factors.append(min(1.0, prop_count / 20.0))  # Normalize to 20 properties

    # Base cognitive load
    base_load = sum(load_factors) / len(load_factors) if load_factors else 0.3

    # Adjust for user profile
    if user_profile:
        # Expertise reduces perceived load
        expertise_factor = 1.0 - (user_profile.spatial_expertise * 0.3)
        base_load *= expertise_factor

        # Cognitive load preference affects perception
        if user_profile.cognitive_load_preference == "low":
            base_load *= 1.2  # Lower tolerance means higher perceived load
        elif user_profile.cognitive_load_preference == "high":
            base_load *= 0.8  # Higher tolerance means lower perceived load

    return min(1.0, max(0.0, base_load))


def format_spatial_data_for_display(
    spatial_data: dict[str, Any], user_profile: UserCognitiveProfile | None = None
) -> dict[str, Any]:
    """
    Format spatial data for user-friendly display.

    Args:
        spatial_data: Raw spatial data
        user_profile: User profile for personalized formatting

    Returns:
        Formatted data suitable for display
    """
    formatted_data: dict[str, Any] = {
        "display_format": "standard",
        "simplified": False,
        "user_optimized": user_profile is not None,
        "elements": [],
    }

    # Extract and format spatial elements
    geometries = spatial_data.get("geometries", [])
    if not geometries and "geometry" in spatial_data:
        geometries = [spatial_data["geometry"]]

    for i, geometry in enumerate(geometries):
        element = {
            "id": f"element_{i}",
            "type": geometry.get("type", "unknown"),
            "simplified_geometry": _simplify_geometry(geometry, user_profile),
            "key_properties": _extract_key_properties(
                spatial_data.get("properties", {}), user_profile
            ),
            "display_priority": _calculate_display_priority(geometry, user_profile),
        }
        formatted_data["elements"].append(element)

    # Determine if data should be simplified
    if user_profile and user_profile.cognitive_load_preference == "low":
        formatted_data["simplified"] = len(formatted_data["elements"]) > 5
        if formatted_data["simplified"]:
            # Keep only highest priority elements
            formatted_data["elements"].sort(
                key=lambda x: x["display_priority"], reverse=True
            )
            formatted_data["elements"] = formatted_data["elements"][:3]

    return formatted_data


def _simplify_geometry(
    geometry: dict[str, Any], user_profile: UserCognitiveProfile | None = None
) -> dict[str, Any]:
    """Simplify geometry for display purposes."""
    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    simplified = {"type": geom_type, "simplified": True}

    # Simplify based on geometry type
    if geom_type == "Point":
        simplified["coordinates"] = coords
        simplified["description"] = "Point location"

    elif geom_type == "LineString":
        if len(coords) > 10:
            # Simplify by keeping first, middle, and last points
            mid_idx = len(coords) // 2
            simplified_coords = [coords[0], coords[mid_idx], coords[-1]]
            simplified["coordinates"] = simplified_coords
            simplified["description"] = f"Line with {len(coords)} points (simplified)"
        else:
            simplified["coordinates"] = coords
            simplified["description"] = f"Line with {len(coords)} points"

    elif geom_type == "Polygon":
        if coords and len(coords[0]) > 20:
            # Simplify polygon by reducing coordinate count
            ring = coords[0]
            step = max(1, len(ring) // 10)  # Keep ~10 points
            simplified_coords = [ring[i] for i in range(0, len(ring), step)]
            simplified["coordinates"] = [simplified_coords]
            simplified["description"] = (
                f"Polygon with {len(ring)} vertices (simplified)"
            )
        else:
            simplified["coordinates"] = coords
            simplified["description"] = (
                f"Polygon with {len(coords[0]) if coords else 0} vertices"
            )

    else:
        simplified["coordinates"] = coords
        simplified["description"] = f"{geom_type} geometry"

    return simplified


def _extract_key_properties(
    properties: dict[str, Any], user_profile: UserCognitiveProfile | None = None
) -> list[str]:
    """Extract key properties for display."""
    key_props = []

    # Priority order for properties
    priority_fields = [
        "name",
        "title",
        "label",
        "id",
        "type",
        "category",
        "class",
        "description",
        "summary",
        "area",
        "length",
        "size",
        "population",
        "elevation",
        "temperature",
    ]

    # Extract priority properties
    for field in priority_fields:
        if field in properties:
            value = properties[field]
            key_props.append(f"{field}: {value}")
            if len(key_props) >= 3:  # Limit to 3 key properties
                break

    # Add user-specific properties if profile available
    if user_profile and user_profile.cognitive_style == "verbalizer":
        # Verbalizers benefit from more descriptive properties
        if "description" in properties:
            key_props.append(f"description: {properties['description']}")

    return key_props


def _calculate_display_priority(
    geometry: dict[str, Any], user_profile: UserCognitiveProfile | None = None
) -> float:
    """Calculate display priority for a geometry element."""
    priority = 0.5  # Base priority

    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    # Type-based priority
    type_priorities = {
        "Point": 0.8,  # Points are usually important landmarks
        "Polygon": 0.7,  # Areas are important for context
        "LineString": 0.6,  # Lines are moderately important
        "MultiPoint": 0.5,
        "MultiLineString": 0.4,
        "MultiPolygon": 0.3,
    }

    priority = type_priorities.get(geom_type, 0.5)

    # Complexity-based adjustment
    if isinstance(coords, list):

        def count_coordinates(c: Any) -> int:
            if isinstance(c, list):
                return sum(count_coordinates(item) for item in c)
            else:
                return 1

        coord_count = count_coordinates(coords)
        complexity_factor = min(0.2, coord_count / 100.0)  # Cap complexity bonus
        priority += complexity_factor

    # User profile adjustments
    if user_profile:
        if user_profile.spatial_expertise > 0.7:
            # Experts can handle more complex displays
            priority += 0.1

        if user_profile.cognitive_load_preference == "low":
            # Reduce priority for complex elements to avoid overload
            if coord_count > 50:
                priority -= 0.2

    return min(1.0, max(0.0, priority))


def create_performance_report(
    processing_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create performance report from processing results.

    Args:
        processing_results: List of processing result dictionaries

    Returns:
        Comprehensive performance report
    """
    if not processing_results:
        return {"error": "No processing results provided"}

    report: dict[str, Any] = {
        "report_generated": datetime.now().isoformat(),
        "total_sessions": len(processing_results),
        "performance_summary": {},
        "cognitive_metrics": {},
        "system_health": {},
        "recommendations": [],
    }

    # Extract performance metrics
    processing_times = [r.get("processing_time", 0) for r in processing_results]
    confidence_scores = []

    for result in processing_results:
        if "decision_result" in result and "decisions" in result["decision_result"]:
            for decision in result["decision_result"]["decisions"]:
                confidence_scores.append(decision.get("confidence_score", 0))

    # Performance summary
    report["performance_summary"] = {
        "average_processing_time": float(np.mean(processing_times)),
        "median_processing_time": float(np.median(processing_times)),
        "min_processing_time": float(np.min(processing_times)),
        "max_processing_time": float(np.max(processing_times)),
        "processing_time_std": float(np.std(processing_times)),
    }

    if confidence_scores:
        report["performance_summary"].update(
            {
                "average_confidence": float(np.mean(confidence_scores)),
                "confidence_distribution": {
                    "high": len([c for c in confidence_scores if c > 0.8]),
                    "medium": len([c for c in confidence_scores if 0.5 <= c <= 0.8]),
                    "low": len([c for c in confidence_scores if c < 0.5]),
                },
            }
        )

    # Cognitive metrics
    cognitive_loads = []
    memory_operations = []

    for result in processing_results:
        if "cognitive_state" in result:
            cognitive_loads.append(result["cognitive_state"].get("cognitive_load", 0))

        if "performance_metrics" in result:
            metrics = result["performance_metrics"]
            memory_operations.append(metrics.get("memory_operations", 0))

    if cognitive_loads:
        report["cognitive_metrics"] = {
            "average_cognitive_load": float(np.mean(cognitive_loads)),
            "cognitive_load_trend": "increasing"
            if len(cognitive_loads) > 1 and cognitive_loads[-1] > cognitive_loads[0]
            else "stable",
        }

    # System health assessment
    if processing_times:
        avg_time = np.mean(processing_times)
        if avg_time < 1.0:
            system_status = "excellent"
        elif avg_time < 5.0:
            system_status = "good"
        elif avg_time < 10.0:
            system_status = "acceptable"
        else:
            system_status = "slow"

        report["system_health"] = {
            "status": system_status,
            "performance_score": min(1.0, 5.0 / avg_time),  # Normalize to 0-1
            "bottlenecks": [],
        }

        if avg_time > 5.0:
            report["system_health"]["bottlenecks"].append(
                "Processing time above optimal threshold"
            )

    # Generate recommendations
    if report["system_health"]["status"] in ["slow", "acceptable"]:
        report["recommendations"].append(
            "Consider optimizing model parameters or reducing input complexity"
        )

    if confidence_scores and np.mean(confidence_scores) < 0.6:
        report["recommendations"].append(
            "Low confidence scores suggest need for model retraining or data quality improvement"
        )

    return report


def export_cognitive_insights(
    insights: dict[str, Any], format: str = "json"
) -> str | dict[str, Any]:
    """
    Export cognitive insights in various formats.

    Args:
        insights: Cognitive insights data
        format: Export format ('json', 'yaml', 'markdown')

    Returns:
        Formatted insights data
    """
    if format == "json":
        return json.dumps(insights, indent=2, default=str)

    elif format == "yaml":
        return cast(str, yaml.dump(insights, default_flow_style=False))

    elif format == "markdown":
        return _format_insights_as_markdown(insights)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def _format_insights_as_markdown(insights: dict[str, Any]) -> str:
    """Format insights as markdown text."""
    markdown = "# Cognitive Insights Report\n\n"

    # Processing summary
    if "processing_summary" in insights:
        summary = insights["processing_summary"]
        markdown += f"**Processing Time:** {summary.get('processing_time', 'N/A')}s\n"
        markdown += (
            f"**Confidence Score:** {summary.get('confidence_score', 'N/A')}\n\n"
        )

    # Attention patterns
    if "attention_patterns" in insights:
        patterns = insights["attention_patterns"]
        markdown += "## Attention Patterns\n\n"
        markdown += (
            f"- **Most Attended Element:** {patterns.get('most_attended', 'N/A')}\n"
        )
        markdown += f"- **Attention Concentration:** {patterns.get('attention_concentration', 'N/A')}\n"
        markdown += f"- **Attention Dispersion:** {patterns.get('attention_dispersion', 'N/A')} elements\n\n"

    # Perceptual groups
    if "perceptual_groups" in insights:
        groups = insights["perceptual_groups"]
        markdown += "## Perceptual Groups\n\n"

        for group_id, group_info in groups.items():
            markdown += f"### Group {group_id}\n"
            markdown += f"- **Elements:** {len(group_info.get('elements', []))}\n"
            markdown += f"- **Size:** {group_info.get('size', 'N/A')}\n\n"

    # Scale distribution
    if "scale_distribution" in insights:
        scales = insights["scale_distribution"]
        markdown += "## Scale Distribution\n\n"
        for scale, count in scales.items():
            markdown += f"- **{scale.title()}:** {count} elements\n"
        markdown += "\n"

    # Uncertainty assessment
    if "uncertainty_assessment" in insights:
        uncertainty = insights["uncertainty_assessment"]
        markdown += "## Uncertainty Assessment\n\n"
        markdown += (
            f"- **Mean Uncertainty:** {uncertainty.get('mean_uncertainty', 'N/A')}\n"
        )
        markdown += (
            f"- **Max Uncertainty:** {uncertainty.get('max_uncertainty', 'N/A')}\n"
        )
        markdown += f"- **Uncertainty Range:** {uncertainty.get('uncertainty_range', 'N/A')}\n\n"

    # User-specific insights
    if "user_specific_insights" in insights:
        user_insights = insights["user_specific_insights"]
        markdown += "## User-Specific Insights\n\n"

        for category, items in user_insights.items():
            markdown += f"### {category.replace('_', ' ').title()}\n"
            for item in items:
                markdown += f"- {item}\n"
            markdown += "\n"

    return markdown


def validate_file_path(file_path: str, required_extension: str | None = None) -> bool:
    """
    Validate file path and check if file exists and is readable.

    Args:
        file_path: Path to validate
        required_extension: Required file extension (optional)

    Returns:
        True if path is valid and accessible
    """
    path = Path(file_path)

    # Check if parent directory exists
    if not path.parent.exists():
        logger.error(f"Directory does not exist: {path.parent}")
        return False

    # Check file extension if required
    if required_extension and path.suffix.lower() != required_extension.lower():
        logger.error(
            f"Invalid file extension. Expected: {required_extension}, got: {path.suffix}"
        )
        return False

    # Check if file exists (for reading operations)
    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return False

    # Check if file is readable
    if not path.is_file():
        logger.error(f"Path is not a file: {path}")
        return False

    try:
        with open(path) as f:
            f.read(1)  # Try to read at least one character
        return True
    except Exception as e:
        logger.error(f"Cannot read file {path}: {str(e)}")
        return False


def create_directory_structure(base_path: str | Path) -> None:
    """Create standard directory structure for cognitive processing."""
    base_path = Path(base_path)

    directories = [
        base_path / "config",
        base_path / "models",
        base_path / "data",
        base_path / "outputs",
        base_path / "logs",
        base_path / "temp",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Directory structure created at {base_path}")


def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.

    Args:
        temp_dir: Directory containing temp files
        max_age_hours: Maximum age in hours before deletion

    Returns:
        Number of files removed
    """
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return 0

    removed_count = 0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

    for file_path in temp_path.iterdir():
        if file_path.is_file():
            # Check file age
            file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

            if file_modified < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove temp file {file_path}: {str(e)}")

    if removed_count > 0:
        logger.info(f"Removed {removed_count} temporary files from {temp_dir}")

    return removed_count
