"""
Validation utilities for GEO-INFER-COG

This module provides validation functions for spatial data, cognitive models,
and user profiles to ensure data integrity and model correctness.

Validation Functions:
- Spatial data validation for geometric consistency
- Cognitive model validation for parameter ranges
- User profile validation for consistency
- Input/output validation for API endpoints
- Configuration validation for model parameters

Mathematical Foundations:
- Geometric validation algorithms
- Statistical validation methods
- Constraint satisfaction for model parameters
- Schema validation for data structures
"""

import logging
from typing import Any
import math

logger = logging.getLogger(__name__)


def validate_spatial_data(spatial_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate spatial data for geometric consistency and completeness.

    Args:
        spatial_data: Spatial data to validate

    Returns:
        Validation results with errors and warnings
    """
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "geometry_checks": {},
        "data_completeness": {},
    }

    # Check for required fields
    required_fields = ["type"]
    for field in required_fields:
        if field not in spatial_data:
            validation_result["errors"].append(f"Missing required field: {field}")
            validation_result["valid"] = False

    # Validate geometry if present
    if "geometry" in spatial_data:
        geometry_validation = validate_geometry(spatial_data["geometry"])
        validation_result["geometry_checks"] = geometry_validation

        if not geometry_validation["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(geometry_validation["errors"])

        validation_result["warnings"].extend(geometry_validation["warnings"])

    # Check data completeness
    completeness = check_data_completeness(spatial_data)
    validation_result["data_completeness"] = completeness

    if completeness["completeness_score"] < 0.5:
        validation_result["warnings"].append(
            "Low data completeness may affect analysis quality"
        )

    return validation_result


def validate_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """
    Validate geometry for topological consistency and coordinate validity.

    Args:
        geometry: GeoJSON geometry object to validate

    Returns:
        Validation results for the geometry
    """
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "geometry_type": None,
        "coordinate_checks": {},
        "topological_checks": {},
    }

    # Check geometry type
    geom_type = geometry.get("type", "")
    validation_result["geometry_type"] = geom_type

    if not geom_type:
        validation_result["errors"].append("Missing geometry type")
        validation_result["valid"] = False
        return validation_result

    # Validate coordinates
    coords = geometry.get("coordinates", [])
    if not coords:
        validation_result["errors"].append("Missing coordinates")
        validation_result["valid"] = False
        return validation_result

    # Type-specific validation
    if geom_type == "Point":
        coord_validation = validate_point_coordinates(coords)
    elif geom_type == "LineString":
        coord_validation = validate_linestring_coordinates(coords)
    elif geom_type == "Polygon":
        coord_validation = validate_polygon_coordinates(coords)
    elif geom_type == "MultiPoint":
        coord_validation = validate_multipoint_coordinates(coords)
    elif geom_type == "MultiLineString":
        coord_validation = validate_multilinestring_coordinates(coords)
    elif geom_type == "MultiPolygon":
        coord_validation = validate_multipolygon_coordinates(coords)
    else:
        validation_result["errors"].append(f"Unsupported geometry type: {geom_type}")
        validation_result["valid"] = False
        return validation_result

    validation_result["coordinate_checks"] = coord_validation

    if not coord_validation["valid"]:
        validation_result["valid"] = False
        validation_result["errors"].extend(coord_validation["errors"])

    validation_result["warnings"].extend(coord_validation["warnings"])

    # Topological checks
    topological_validation = check_topological_validity(geometry)
    validation_result["topological_checks"] = topological_validation

    if not topological_validation["valid"]:
        validation_result["valid"] = False
        validation_result["errors"].extend(topological_validation["errors"])

    validation_result["warnings"].extend(topological_validation["warnings"])

    return validation_result


def validate_point_coordinates(coords: list[float]) -> dict[str, Any]:
    """Validate Point coordinates."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "coordinate_count": len(coords),
        "coordinate_ranges": {},
    }

    if len(coords) < 2:
        validation["errors"].append(
            "Point must have at least 2 coordinates (longitude, latitude)"
        )
        validation["valid"] = False
        return validation

    longitude, latitude = coords[0], coords[1]

    # Validate coordinate ranges
    if not (-180 <= longitude <= 180):
        validation["errors"].append(
            f"Longitude {longitude} out of valid range [-180, 180]"
        )
        validation["valid"] = False

    if not (-90 <= latitude <= 90):
        validation["errors"].append(f"Latitude {latitude} out of valid range [-90, 90]")
        validation["valid"] = False

    validation["coordinate_ranges"] = {
        "longitude": longitude,
        "latitude": latitude,
        "altitude": coords[2] if len(coords) > 2 else None,
    }

    return validation


def validate_linestring_coordinates(coords: list[list[float]]) -> dict[str, Any]:
    """Validate LineString coordinates."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "coordinate_count": len(coords),
        "segment_lengths": [],
    }

    if len(coords) < 2:
        validation["errors"].append("LineString must have at least 2 coordinate pairs")
        validation["valid"] = False
        return validation

    # Validate each coordinate
    for i, coord in enumerate(coords):
        point_validation = validate_point_coordinates(coord)
        if not point_validation["valid"]:
            validation["errors"].extend(
                [
                    f"Invalid coordinate at index {i}: {e}"
                    for e in point_validation["errors"]
                ]
            )
            validation["valid"] = False

    # Check for degenerate segments
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        # Calculate distance
        distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        if distance == 0:
            validation["warnings"].append(f"Degenerate segment at indices {i}-{i + 1}")

        validation["segment_lengths"].append(distance)

    return validation


def validate_polygon_coordinates(coords: list[list[list[float]]]) -> dict[str, Any]:
    """Validate Polygon coordinates."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "ring_count": len(coords),
        "ring_sizes": [],
    }

    if len(coords) == 0:
        validation["errors"].append("Polygon must have at least one ring")
        validation["valid"] = False
        return validation

    # Validate exterior ring (first ring)
    exterior_ring = coords[0]
    exterior_validation = validate_linestring_coordinates(exterior_ring)
    validation["ring_sizes"].append(len(exterior_ring))

    if not exterior_validation["valid"]:
        validation["errors"].extend(
            [f"Invalid exterior ring: {e}" for e in exterior_validation["errors"]]
        )
        validation["valid"] = False

    validation["warnings"].extend(
        [f"Exterior ring warning: {w}" for w in exterior_validation["warnings"]]
    )

    # Validate interior rings (holes)
    for i, interior_ring in enumerate(coords[1:], 1):
        ring_validation = validate_linestring_coordinates(interior_ring)
        validation["ring_sizes"].append(len(interior_ring))

        if not ring_validation["valid"]:
            validation["errors"].extend(
                [f"Invalid interior ring {i}: {e}" for e in ring_validation["errors"]]
            )
            validation["valid"] = False

        validation["warnings"].extend(
            [f"Interior ring {i} warning: {w}" for w in ring_validation["warnings"]]
        )

    # Check polygon closure
    if coords and coords[0]:
        first_point = coords[0][0]
        last_point = coords[0][-1]

        if first_point != last_point:
            validation["warnings"].append(
                "Polygon exterior ring not closed (first != last point)"
            )

    return validation


def validate_multipoint_coordinates(coords: list[list[float]]) -> dict[str, Any]:
    """Validate MultiPoint coordinates."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "point_count": len(coords),
    }

    for i, coord in enumerate(coords):
        point_validation = validate_point_coordinates(coord)
        if not point_validation["valid"]:
            validation["errors"].extend(
                [f"Invalid point at index {i}: {e}" for e in point_validation["errors"]]
            )
            validation["valid"] = False

    return validation


def validate_multilinestring_coordinates(
    coords: list[list[list[float]]],
) -> dict[str, Any]:
    """Validate MultiLineString coordinates."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "linestring_count": len(coords),
    }

    for i, linestring_coords in enumerate(coords):
        ls_validation = validate_linestring_coordinates(linestring_coords)
        if not ls_validation["valid"]:
            validation["errors"].extend(
                [
                    f"Invalid linestring at index {i}: {e}"
                    for e in ls_validation["errors"]
                ]
            )
            validation["valid"] = False

        validation["warnings"].extend(
            [f"Linestring {i} warning: {w}" for w in ls_validation["warnings"]]
        )

    return validation


def validate_multipolygon_coordinates(
    coords: list[list[list[list[float]]]],
) -> dict[str, Any]:
    """Validate MultiPolygon coordinates."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "polygon_count": len(coords),
    }

    for i, polygon_coords in enumerate(coords):
        poly_validation = validate_polygon_coordinates(polygon_coords)
        if not poly_validation["valid"]:
            validation["errors"].extend(
                [
                    f"Invalid polygon at index {i}: {e}"
                    for e in poly_validation["errors"]
                ]
            )
            validation["valid"] = False

        validation["warnings"].extend(
            [f"Polygon {i} warning: {w}" for w in poly_validation["warnings"]]
        )

    return validation


def check_topological_validity(geometry: dict[str, Any]) -> dict[str, Any]:
    """Check topological validity of geometry."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "self_intersections": False,
        "degenerate_features": False,
    }

    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    # Check for self-intersections (simplified check)
    if geom_type == "Polygon" and coords:
        # For polygons, check if exterior ring is simple (no self-intersections)
        exterior_ring = coords[0] if coords else []

        # Simple self-intersection check for rings
        if len(exterior_ring) > 3:
            # Check each edge against all non-adjacent edges
            for i in range(len(exterior_ring) - 1):
                for j in range(i + 2, len(exterior_ring) - 1):
                    if do_edges_intersect(
                        exterior_ring[i],
                        exterior_ring[i + 1],
                        exterior_ring[j],
                        exterior_ring[j + 1],
                    ):
                        validation["self_intersections"] = True
                        validation["warnings"].append(
                            "Polygon may have self-intersections"
                        )

    return validation


def do_edges_intersect(
    p1: list[float], p2: list[float], p3: list[float], p4: list[float]
) -> bool:
    """Check if two line segments intersect."""

    # Simple line segment intersection algorithm
    def ccw(A: list[float], B: list[float], C: list[float]) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B, C, D = p1, p2, p3, p4

    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def check_data_completeness(spatial_data: dict[str, Any]) -> dict[str, Any]:
    """Check completeness of spatial data."""
    completeness: dict[str, Any] = {
        "completeness_score": 0.0,
        "missing_fields": [],
        "optional_fields_present": [],
        "data_quality_indicators": {},
    }

    # Define expected fields for different data types
    expected_fields = {
        "geometry": ["type", "coordinates"],
        "properties": ["id", "name"],
        "metadata": ["created", "source"],
    }

    total_fields = 0
    present_fields = 0

    for category, fields in expected_fields.items():
        if category in spatial_data:
            category_data = spatial_data[category]
            total_fields += len(fields)

            for field in fields:
                if field in category_data:
                    present_fields += 1
                else:
                    completeness["missing_fields"].append(f"{category}.{field}")

    if total_fields > 0:
        completeness["completeness_score"] = present_fields / total_fields

    # Check for optional but valuable fields
    optional_fields = ["crs", "bbox", "description", "tags"]
    for field in optional_fields:
        if field in spatial_data:
            completeness["optional_fields_present"].append(field)

    # Data quality indicators
    if "geometry" in spatial_data and spatial_data["geometry"].get("coordinates"):
        coords = spatial_data["geometry"]["coordinates"]

        # Flatten coordinates for analysis
        all_coords = []
        if isinstance(coords, list):

            def flatten_coords(c: Any) -> None:
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, list) and len(item) >= 2:
                            all_coords.append(item)
                        else:
                            flatten_coords(item)

            flatten_coords(coords)

        if all_coords:
            # Check coordinate precision
            coord_lengths = [
                len(str(c[0]).split(".")[-1]) + len(str(c[1]).split(".")[-1])
                for c in all_coords[:10]
            ]
            avg_precision = (
                sum(coord_lengths) / len(coord_lengths) if coord_lengths else 0
            )

            completeness["data_quality_indicators"] = {
                "coordinate_count": len(all_coords),
                "average_precision_digits": avg_precision,
                "coordinate_range": {
                    "min_lon": min(c[0] for c in all_coords) if all_coords else None,
                    "max_lon": max(c[0] for c in all_coords) if all_coords else None,
                    "min_lat": min(c[1] for c in all_coords) if all_coords else None,
                    "max_lat": max(c[1] for c in all_coords) if all_coords else None,
                },
            }

    return completeness


def validate_cognitive_model(
    model_config: dict[str, Any], model_type: str = "general"
) -> dict[str, Any]:
    """
    Validate cognitive model configuration and parameters.

    Args:
        model_config: Model configuration parameters
        model_type: Type of cognitive model being validated

    Returns:
        Validation results for the model configuration
    """
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "parameter_checks": {},
        "consistency_checks": {},
    }

    # Parameter range validation
    parameter_ranges = {
        "attention_capacity": (0.0, 1.0),
        "focus_radius": (0.0, 1.0),
        "saliency_threshold": (0.0, 1.0),
        "memory_capacity": (1, 100),
        "decay_rate": (0.0, 1.0),
        "learning_rate": (0.0, 1.0),
        "confidence_threshold": (0.0, 1.0),
    }

    for param, (min_val, max_val) in parameter_ranges.items():
        if param in model_config:
            value = model_config[param]

            if not (min_val <= value <= max_val):
                validation_result["errors"].append(
                    f"Parameter {param}={value} out of range [{min_val}, {max_val}]"
                )
                validation_result["valid"] = False

    # Model-specific validation
    if model_type == "perception":
        perception_validation = validate_perception_model(model_config)
        validation_result["parameter_checks"] = perception_validation

    elif model_type == "reasoning":
        reasoning_validation = validate_reasoning_model(model_config)
        validation_result["parameter_checks"] = reasoning_validation

    elif model_type == "memory":
        memory_validation = validate_memory_model(model_config)
        validation_result["parameter_checks"] = memory_validation

    # Propagate parameter_checks validity to top-level result
    param_checks = validation_result.get("parameter_checks", {})
    if param_checks and not param_checks.get("valid", True):
        validation_result["valid"] = False
        validation_result["errors"].extend(param_checks.get("errors", []))

    # Consistency checks
    consistency_checks = check_model_consistency(model_config, model_type)
    validation_result["consistency_checks"] = consistency_checks

    if not consistency_checks["consistent"]:
        validation_result["valid"] = False
        validation_result["errors"].extend(consistency_checks["issues"])

    return validation_result


def validate_perception_model(config: dict[str, Any]) -> dict[str, Any]:
    """Validate perception model configuration."""
    validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

    # Check attention model parameters
    if "attention_capacity" in config and "focus_radius" in config:
        capacity = config["attention_capacity"]
        radius = config["focus_radius"]

        if capacity < radius:
            validation["warnings"].append(
                "Attention capacity is less than focus radius - may limit effectiveness"
            )

    # Check saliency parameters
    if "saliency_threshold" in config:
        threshold = config["saliency_threshold"]

        if threshold > 0.8:
            validation["warnings"].append(
                "High saliency threshold may filter out important elements"
            )

    return validation


def validate_reasoning_model(config: dict[str, Any]) -> dict[str, Any]:
    """Validate reasoning model configuration."""
    validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

    # Check reasoning type validity
    valid_reasoning_types = [
        "qualitative_spatial",
        "analogical",
        "deductive",
        "constraint_based",
    ]
    reasoning_type = config.get("reasoning_type")

    if reasoning_type and reasoning_type not in valid_reasoning_types:
        validation["errors"].append(f"Invalid reasoning type: {reasoning_type}")
        validation["valid"] = False

    # Check uncertainty method validity
    valid_uncertainty_methods = ["probabilistic", "fuzzy", "possibilistic"]
    uncertainty_method = config.get("uncertainty_method")

    if uncertainty_method and uncertainty_method not in valid_uncertainty_methods:
        validation["errors"].append(f"Invalid uncertainty method: {uncertainty_method}")
        validation["valid"] = False

    return validation


def validate_memory_model(config: dict[str, Any]) -> dict[str, Any]:
    """Validate memory model configuration."""
    validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

    # Check memory type validity
    valid_memory_types = ["working", "long_term", "episodic", "semantic", "procedural"]
    memory_types = config.get("memory_types", [])

    for mem_type in memory_types:
        if mem_type not in valid_memory_types:
            validation["errors"].append(f"Invalid memory type: {mem_type}")
            validation["valid"] = False

    # Check capacity limits
    for mem_type in memory_types:
        capacity_key = f"{mem_type}_capacity"
        if capacity_key in config:
            capacity = config[capacity_key]

            if capacity < 1:
                validation["errors"].append(
                    f"Invalid capacity for {mem_type}: {capacity}"
                )
                validation["valid"] = False

    # Check decay rates
    for mem_type in memory_types:
        decay_key = f"{mem_type}_decay_rate"
        if decay_key in config:
            decay_rate = config[decay_key]

            if not (0.0 <= decay_rate <= 1.0):
                validation["errors"].append(
                    f"Invalid decay rate for {mem_type}: {decay_rate}"
                )
                validation["valid"] = False

    return validation


def check_model_consistency(config: dict[str, Any], model_type: str) -> dict[str, Any]:
    """Check consistency of model configuration."""
    consistency: dict[str, Any] = {"consistent": True, "issues": []}

    # Framework-specific consistency checks
    if model_type == "perception":
        # Check that attention and saliency parameters are compatible
        attention_capacity = config.get("attention_capacity", 1.0)
        saliency_threshold = config.get("saliency_threshold", 0.3)

        if attention_capacity < saliency_threshold:
            consistency["issues"].append(
                "Attention capacity should be >= saliency threshold for effective processing"
            )

    elif model_type == "memory":
        # Check that consolidation parameters are reasonable
        consolidation_threshold = config.get("consolidation_threshold", 0.7)
        consolidation_delay = config.get("consolidation_delay", 300)

        if consolidation_delay < 60 and consolidation_threshold > 0.8:
            consistency["issues"].append(
                "Short consolidation delay with high threshold may prevent consolidation"
            )

    return consistency


def validate_user_profile(profile_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate user cognitive profile data.

    Args:
        profile_data: User profile data to validate

    Returns:
        Validation results for the profile
    """
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "profile_completeness": 0.0,
    }

    # Required fields
    required_fields = ["user_id"]
    for field in required_fields:
        if field not in profile_data:
            validation_result["errors"].append(f"Missing required field: {field}")
            validation_result["valid"] = False

    # Validate expertise levels (0-1 range)
    expertise_fields = [
        "spatial_expertise",
        "spatial_memory_capacity",
        "spatial_attention_span",
    ]
    for field in expertise_fields:
        if field in profile_data:
            value = profile_data[field]
            if not (0.0 <= value <= 1.0):
                validation_result["errors"].append(
                    f"Invalid {field}: {value} (must be 0-1)"
                )
                validation_result["valid"] = False

    # Validate categorical fields
    categorical_fields = {
        "cognitive_style": ["visualizer", "verbalizer", "balanced"],
        "learning_preference": ["slow", "moderate", "fast"],
        "cognitive_load_preference": ["low", "moderate", "high"],
        "spatial_reasoning_style": ["qualitative", "quantitative", "balanced"],
        "age_group": ["child", "teen", "adult", "senior"],
        "experience_level": ["beginner", "intermediate", "advanced", "expert"],
    }

    for field, valid_values in categorical_fields.items():
        if field in profile_data:
            value = profile_data[field]
            if value not in valid_values:
                validation_result["errors"].append(f"Invalid {field}: {value}")
                validation_result["valid"] = False

    # Calculate profile completeness
    total_fields = len(set(expertise_fields + list(categorical_fields.keys())))
    present_fields = sum(
        1
        for field in expertise_fields + list(categorical_fields.keys())
        if field in profile_data
    )

    validation_result["profile_completeness"] = (
        present_fields / total_fields if total_fields > 0 else 0.0
    )

    if validation_result["profile_completeness"] < 0.7:
        validation_result["warnings"].append(
            "Low profile completeness may reduce personalization effectiveness"
        )

    return validation_result


def validate_configuration(
    config: dict[str, Any], module_name: str = "cog"
) -> dict[str, Any]:
    """
    Validate module configuration for consistency and completeness.

    Args:
        config: Configuration dictionary to validate
        module_name: Name of the module being configured

    Returns:
        Validation results for the configuration
    """
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config_completeness": 0.0,
        "recommended_settings": {},
    }

    # Module-specific configuration requirements
    if module_name == "cog":
        required_sections = ["core", "models", "api"]
        optional_sections = ["logging", "monitoring", "caching"]

        # Check required sections
        for section in required_sections:
            if section not in config:
                validation_result["warnings"].append(
                    f"Missing recommended configuration section: {section}"
                )

        # Validate core configuration
        if "core" in config:
            core_validation = validate_core_config(config["core"])
            if not core_validation["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(core_validation["errors"])
            validation_result["warnings"].extend(core_validation["warnings"])

    # Calculate configuration completeness
    expected_sections = required_sections + optional_sections
    present_sections = sum(1 for section in expected_sections if section in config)
    validation_result["config_completeness"] = present_sections / len(expected_sections)

    # Generate recommended settings for missing configurations
    validation_result["recommended_settings"] = generate_default_config(module_name)

    return validation_result


def validate_core_config(core_config: dict[str, Any]) -> dict[str, Any]:
    """Validate core module configuration."""
    validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

    # Check cognitive framework
    valid_frameworks = ["bayesian_attention", "act_r", "soar", "gestalt"]
    framework = core_config.get("cognitive_framework")

    if framework and framework not in valid_frameworks:
        validation["errors"].append(f"Invalid cognitive framework: {framework}")
        validation["valid"] = False

    # Check spatial resolution
    valid_resolutions = ["adaptive", "fixed", "hierarchical"]
    resolution = core_config.get("spatial_resolution")

    if resolution and resolution not in valid_resolutions:
        validation["errors"].append(f"Invalid spatial resolution: {resolution}")
        validation["valid"] = False

    return validation


def generate_default_config(module_name: str) -> dict[str, Any]:
    """Generate default configuration for a module."""
    if module_name == "cog":
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
                },
                "reasoning": {
                    "reasoning_type": "qualitative_spatial",
                    "uncertainty_method": "probabilistic",
                },
                "memory": {
                    "memory_types": ["working", "long_term", "episodic"],
                    "consolidation_threshold": 0.7,
                    "consolidation_delay": 300,
                },
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "rate_limiting": True,
            },
        }

    return {}
