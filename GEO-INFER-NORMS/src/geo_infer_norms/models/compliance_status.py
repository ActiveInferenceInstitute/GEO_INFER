"""
Compliance status models for representing regulatory compliance information.

This module provides data models for tracking and evaluating compliance with
regulations across entities and jurisdictions.
"""

from dataclasses import dataclass, field
from typing import Any
import datetime
import uuid


@dataclass
class ComplianceStatus:
    """
    A class representing the compliance status of an entity with a regulation.

    This class captures whether an entity complies with a specific regulation,
    along with details about the compliance evaluation.
    """

    id: str
    entity_id: str
    regulation_id: str
    is_compliant: bool
    compliance_level: float  # 0.0 to 1.0
    timestamp: datetime.datetime
    notes: str = ""
    evaluated_by: str | None = None
    evaluation_method: str | None = None
    metric_results: list[dict[str, Any]] | None = None
    evidence: dict[str, Any] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        entity_id: str,
        regulation_id: str,
        is_compliant: bool,
        compliance_level: float,
        notes: str = "",
        evaluated_by: str | None = None,
        evaluation_method: str | None = None,
        metric_results: list[dict[str, Any]] | None = None,
        evidence: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> "ComplianceStatus":
        """
        Create a new ComplianceStatus with a generated UUID.

        Args:
            entity_id: ID of the entity
            regulation_id: ID of the regulation
            is_compliant: Whether the entity is compliant with the regulation
            compliance_level: Level of compliance (0.0 to 1.0)
            notes: Optional notes about the compliance status
            evaluated_by: Optional identifier of who performed the evaluation
            evaluation_method: Optional method used for evaluation
            metric_results: Optional list of metric-specific evaluation results
            evidence: Optional evidence supporting the compliance status
            attributes: Dictionary of additional attributes

        Returns:
            A new ComplianceStatus instance

        Raises:
            ValueError: If compliance_level is outside the range [0.0, 1.0].
        """
        if (
            not isinstance(compliance_level, (int, float))
            or not 0.0 <= compliance_level <= 1.0
        ):
            raise ValueError(
                "compliance_status.create: compliance_level must be within [0.0, 1.0], "
                f"got {compliance_level!r}"
            )
        return cls(
            id=str(uuid.uuid4()),
            entity_id=entity_id,
            regulation_id=regulation_id,
            is_compliant=is_compliant,
            compliance_level=compliance_level,
            timestamp=datetime.datetime.now(),
            notes=notes,
            evaluated_by=evaluated_by,
            evaluation_method=evaluation_method,
            metric_results=metric_results,
            evidence=evidence,
            attributes=attributes or {},
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the compliance status.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def add_evidence(self, key: str, value: Any) -> None:
        """
        Add evidence to support the compliance status.

        Args:
            key: Evidence key
            value: Evidence value
        """
        if self.evidence is None:
            self.evidence = {}

        self.evidence[key] = value

    def add_metric_result(self, metric_result: dict[str, Any]) -> None:
        """
        Add a metric result to the compliance status.

        Args:
            metric_result: Dictionary containing metric evaluation results
        """
        if self.metric_results is None:
            self.metric_results = []

        self.metric_results.append(metric_result)

    def is_recent(self, days: int = 30) -> bool:
        """
        Check if the compliance status is recent.

        Args:
            days: Number of days to consider recent

        Returns:
            True if the status is within the specified number of days, False otherwise
        """
        age = datetime.datetime.now() - self.timestamp
        return age.days <= days


@dataclass
class ComplianceMetric:
    """
    A class representing a metric for evaluating compliance with a regulation.

    Compliance metrics define specific measurements or criteria used to determine
    whether an entity complies with a particular aspect of a regulation.
    """

    id: str
    name: str
    description: str
    regulation_id: str
    evaluation_type: str  # e.g., 'threshold', 'range', 'boolean', 'composite'
    primary_field: str
    required_fields: list[str] = field(default_factory=list)
    threshold_value: float | int | str | None = None
    comparison: str | None = None  # e.g., 'greater_than', 'less_than', 'equal'
    range_min: float | None = None
    range_max: float | None = None
    weight: float = 1.0  # For composite metrics
    sub_metrics: list[str] = field(
        default_factory=list
    )  # IDs of sub-metrics for composite metrics
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        regulation_id: str,
        evaluation_type: str,
        primary_field: str,
        required_fields: list[str] | None = None,
        threshold_value: float | int | str | None = None,
        comparison: str | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
        weight: float = 1.0,
        sub_metrics: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> "ComplianceMetric":
        """
        Create a new ComplianceMetric with a generated UUID.

        Args:
            name: Name of the metric
            description: Description of the metric
            regulation_id: ID of the associated regulation
            evaluation_type: Type of evaluation (e.g., 'threshold', 'range', 'boolean', 'composite')
            primary_field: Primary field used for evaluation
            required_fields: List of fields required for evaluation
            threshold_value: Optional threshold value for threshold evaluations
            comparison: Optional comparison operator for threshold evaluations
            range_min: Optional minimum value for range evaluations
            range_max: Optional maximum value for range evaluations
            weight: Weight of the metric (for composite metrics)
            sub_metrics: List of sub-metric IDs for composite metrics
            attributes: Dictionary of additional attributes

        Returns:
            A new ComplianceMetric instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            regulation_id=regulation_id,
            evaluation_type=evaluation_type,
            primary_field=primary_field,
            required_fields=required_fields or [primary_field],
            threshold_value=threshold_value,
            comparison=comparison,
            range_min=range_min,
            range_max=range_max,
            weight=weight,
            sub_metrics=sub_metrics or [],
            attributes=attributes or {},
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the metric.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_required_field(self, field: str) -> None:
        """
        Add a required field to the metric.

        Args:
            field: Name of the required field
        """
        if field not in self.required_fields:
            self.required_fields.append(field)
            self.updated_at = datetime.datetime.now()

    def add_sub_metric(self, sub_metric_id: str) -> None:
        """
        Add a sub-metric to a composite metric.

        Args:
            sub_metric_id: ID of the sub-metric to add
        """
        if self.evaluation_type != "composite":
            raise ValueError("Can only add sub-metrics to composite metrics")

        if sub_metric_id not in self.sub_metrics:
            self.sub_metrics.append(sub_metric_id)
            self.updated_at = datetime.datetime.now()

    def set_threshold(self, value: float | int | str, comparison: str) -> None:
        """
        Set the threshold value and comparison for the metric.

        Args:
            value: Threshold value
            comparison: Comparison operator (e.g., 'greater_than', 'less_than', 'equal')
        """
        if self.evaluation_type != "threshold":
            raise ValueError("Can only set threshold for threshold metrics")

        self.threshold_value = value
        self.comparison = comparison
        self.updated_at = datetime.datetime.now()

    def set_range(self, min_value: float, max_value: float) -> None:
        """
        Set the range for the metric.

        Args:
            min_value: Minimum value
            max_value: Maximum value
        """
        if self.evaluation_type != "range":
            raise ValueError("Can only set range for range metrics")

        self.range_min = min_value
        self.range_max = max_value
        self.updated_at = datetime.datetime.now()
