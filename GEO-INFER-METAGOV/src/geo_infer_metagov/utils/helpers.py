"""Utility helper functions for METAGOV module."""

import logging
from typing import Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def entity_field(entity: Any, name: str, default: Any) -> Any:
    """Read a field from either a mapping or an object entity.

    Governance entities may be plain dicts (API payloads) or dataclass
    instances (e.g. ``core.multi_level.GovernanceEntity``); this accessor
    works for both.

    Parameters:
    -----------
    entity : Any
        Entity as a dict or object.
    name : str
        Field name to read.
    default : Any
        Value returned when the field is missing or ``None``.

    Returns:
    --------
    Any
        The field value, or ``default``.
    """
    if isinstance(entity, dict):
        value = entity.get(name, default)
    else:
        value = getattr(entity, name, default)
    return value if value is not None else default


def validate_spatial_scope(spatial_scope: Any) -> bool:
    """
    Validate spatial scope dictionary.

    Parameters
    ----------
    spatial_scope : Dict[str, Any]
        Spatial scope to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    required_keys = ["name"]

    if not isinstance(spatial_scope, dict):
        logger.error("Spatial scope must be a dictionary")
        return False

    for key in required_keys:
        if key not in spatial_scope:
            logger.error(f"Missing required key: {key}")
            return False

    return True


def validate_stakeholder_groups(stakeholder_groups: Any) -> bool:
    """
    Validate stakeholder groups list.

    Parameters
    ----------
    stakeholder_groups : List[Dict[str, Any]]
        Stakeholder groups to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if not isinstance(stakeholder_groups, list):
        logger.error("Stakeholder groups must be a list")
        return False

    if len(stakeholder_groups) == 0:
        logger.error("Stakeholder groups cannot be empty")
        return False

    required_keys = ["name"]

    for group in stakeholder_groups:
        if not isinstance(group, dict):
            logger.error("Each stakeholder group must be a dictionary")
            return False

        for key in required_keys:
            if key not in group:
                logger.error(f"Missing required key in stakeholder group: {key}")
                return False

    return True


def validate_decision_domains(decision_domains: Any) -> bool:
    """
    Validate decision domains list.

    Parameters
    ----------
    decision_domains : List[str]
        Decision domains to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if not isinstance(decision_domains, list):
        logger.error("Decision domains must be a list")
        return False

    if len(decision_domains) == 0:
        logger.error("Decision domains cannot be empty")
        return False

    for domain in decision_domains:
        if not isinstance(domain, str):
            logger.error("Each domain must be a string")
            return False

    return True


def calculate_collaboration_potential(stakeholders: list[dict[str, Any]]) -> float:
    """
    Calculate collaboration potential based on stakeholder interests.

    Parameters
    ----------
    stakeholders : List[Dict[str, Any]]
        List of stakeholder dictionaries

    Returns
    -------
    float
        Collaboration potential score (0-1)
    """
    if not stakeholders or len(stakeholders) < 2:
        return 0.0

    # Calculate overlapping interests
    all_interests = []
    for stakeholder in stakeholders:
        if "interests" in stakeholder and isinstance(stakeholder["interests"], list):
            all_interests.extend(stakeholder["interests"])

    if not all_interests:
        return 0.5

    # Calculate overlap ratio
    unique_interests = len(set(all_interests))
    total_interests = len(all_interests)

    overlap_ratio = (
        1.0 - (unique_interests / total_interests) if total_interests > 0 else 0
    )

    return min(0.5 + (overlap_ratio * 0.5), 1.0)


def calculate_power_concentration(
    stakeholders: list[dict[str, Any]],
) -> tuple[float, str]:
    """
    Calculate power concentration among stakeholders.

    Parameters
    ----------
    stakeholders : List[Dict[str, Any]]
        List of stakeholder dictionaries

    Returns
    -------
    Tuple[float, str]
        Power concentration score and balance assessment
    """
    if not stakeholders:
        return 0.0, "balanced"

    # Extract power values
    powers = []
    for stakeholder in stakeholders:
        if "decision_power" in stakeholder:
            powers.append(stakeholder["decision_power"])
        elif "power" in stakeholder:
            powers.append(stakeholder["power"])

    if not powers:
        return 0.0, "balanced"

    max_power = max(powers)
    total_power = sum(powers)
    concentration = max_power / total_power if total_power > 0 else 0

    # Determine balance assessment
    if concentration > 0.6:
        assessment = "unbalanced"
    elif concentration > 0.4:
        assessment = "relatively_balanced"
    else:
        assessment = "balanced"

    return concentration, assessment


def extract_governance_metrics(governance_structure: Any) -> dict[str, Any]:
    """
    Extract key metrics from governance structure.

    Parameters
    ----------
    governance_structure : Any
        Governance structure object

    Returns
    -------
    Dict[str, Any]
        Dictionary of governance metrics
    """
    metrics = {
        "governance_id": getattr(governance_structure, "governance_id", None),
        "num_entities": len(getattr(governance_structure, "entities", [])),
        "num_levels": len(getattr(governance_structure, "governance_levels", [])),
        "num_decision_domains": len(
            getattr(governance_structure, "decision_domains", [])
        ),
        "num_stakeholder_groups": len(
            getattr(governance_structure, "stakeholder_groups", [])
        ),
        "num_coordination_mechanisms": len(
            getattr(governance_structure, "coordination_mechanisms", [])
        ),
    }

    return metrics


def generate_governance_report(
    governance_structure: Any, title: str | None = None
) -> str:
    """
    Generate a governance structure report.

    Parameters
    ----------
    governance_structure : Any
        Governance structure object
    title : Optional[str]
        Optional report title

    Returns
    -------
    str
        Formatted governance report
    """
    metrics = extract_governance_metrics(governance_structure)

    report = []
    report.append("=" * 70)
    report.append(f"GOVERNANCE STRUCTURE REPORT - {title or 'General'}")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    report.append("GOVERNANCE METRICS")
    report.append("-" * 70)

    for key, value in metrics.items():
        report.append(f"{key.replace('_', ' ').title()}: {value}")

    report.append("")
    report.append("GOVERNANCE ENTITIES")
    report.append("-" * 70)

    entities = getattr(governance_structure, "entities", [])
    for i, entity in enumerate(entities, 1):
        entity_name = getattr(entity, "name", f"Entity {i}")
        entity_level = getattr(entity, "governance_level", "Unknown")
        report.append(f"{i}. {entity_name} ({entity_level})")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def format_governance_output(data: Any, format_type: str = "json") -> str:
    """
    Format governance data for output.

    Parameters
    ----------
    data : Any
        Data to format
    format_type : str
        Format type ('json', 'dict', 'string')

    Returns
    -------
    str
        Formatted data
    """
    if format_type == "json":
        try:
            if hasattr(data, "__dict__"):
                return json.dumps(data.__dict__, indent=2, default=str)
            else:
                return json.dumps(data, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error formatting as JSON: {e}")
            return str(data)

    elif format_type == "dict":
        if hasattr(data, "__dict__"):
            return str(data.__dict__)
        else:
            return str(data)

    else:  # string
        return str(data)


def merge_governance_structures(
    structure1: Any, structure2: Any, strategy: str = "union"
) -> dict[str, Any]:
    """
    Merge two governance structures.

    Parameters
    ----------
    structure1 : Any
        First governance structure
    structure2 : Any
        Second governance structure
    strategy : str
        Merge strategy ('union', 'intersection', 'structure1_priority')

    Returns
    -------
    Dict[str, Any]
        Merged governance structure data
    """
    result = {
        "strategy": strategy,
        "structures_merged": 2,
        "merged_entities": len(getattr(structure1, "entities", []))
        + len(getattr(structure2, "entities", [])),
        "merged_domains": list(
            set(
                getattr(structure1, "decision_domains", [])
                + getattr(structure2, "decision_domains", [])
            )
        ),
        "timestamp": datetime.now().isoformat(),
    }

    return result


def validate_ostrom_principles(principles: list[str]) -> tuple[bool, list[str]]:
    """
    Validate Ostrom design principles.

    Parameters
    ----------
    principles : List[str]
        List of principle names to validate

    Returns
    -------
    Tuple[bool, List[str]]
        Validation result and list of valid principles
    """
    valid_principles = [
        "clear_boundaries",
        "congruence",
        "collective_choice_arrangements",
        "monitoring",
        "graduated_sanctions",
        "conflict_resolution",
        "right_to_organize",
        "nested_enterprises",
    ]

    validated = []
    invalid = []

    for principle in principles:
        if principle in valid_principles:
            validated.append(principle)
        else:
            invalid.append(principle)

    is_valid = len(invalid) == 0

    if invalid:
        logger.warning(f"Invalid principles detected: {invalid}")

    return is_valid, validated


def calculate_governance_health_score(metrics: dict[str, float]) -> float:
    """
    Calculate overall governance health score.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of governance metrics with scores 0-1

    Returns
    -------
    float
        Overall health score (0-1)
    """
    if not metrics:
        return 0.5

    # Weight different metrics
    weights = {
        "effectiveness": 0.25,
        "equity": 0.25,
        "sustainability": 0.25,
        "participation": 0.15,
        "transparency": 0.10,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, metric_value in metrics.items():
        weight = weights.get(metric_name, 0.1)
        if 0 <= metric_value <= 1:
            weighted_sum += metric_value * weight
            total_weight += weight

    health_score = weighted_sum / total_weight if total_weight > 0 else 0.5

    return min(health_score, 1.0)
