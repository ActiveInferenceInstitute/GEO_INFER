"""
Security Data Models for GEO-INFER-SEC

This module defines the core data structures and models used across
Physical, Digital, and Cognitive security domains.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from dataclasses import dataclass, field
import json


class ThreatLevel(Enum):
    """Standardized threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventCategory(Enum):
    """Categories of security events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    NETWORK_ACTIVITY = "network_activity"
    SYSTEM_ACTIVITY = "system_activity"
    PHYSICAL_ACCESS = "physical_access"
    BEHAVIORAL = "behavioral"
    THREAT_DETECTION = "threat_detection"


@dataclass
class SecurityEvent:
    """Base security event model."""

    event_id: str
    event_type: str
    category: SecurityEventCategory = SecurityEventCategory.SYSTEM_ACTIVITY
    timestamp: datetime = field(default_factory=datetime.now)
    source: str | None = None
    target: str | None = None
    severity: ThreatLevel = ThreatLevel.MEDIUM
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target,
            "severity": self.severity.value,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            category=SecurityEventCategory(data.get("category", "system_activity")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source"),
            target=data.get("target"),
            severity=ThreatLevel(data.get("severity", "medium")),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SecurityAlert:
    """Security alert model."""

    alert_id: str
    title: str
    description: str
    severity: ThreatLevel
    category: SecurityEventCategory
    source_events: list[str] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)
    threat_indicators: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolution_notes: str = ""
    confidence_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_status(self, new_status: str, notes: str = "") -> None:
        """Update alert status."""
        self.status = new_status
        self.updated_at = datetime.now()
        if new_status == "resolved":
            self.resolved_at = datetime.now()
            self.resolution_notes = notes


@dataclass
class ThreatIntelligence:
    """Threat intelligence indicator model."""

    indicator_id: str
    indicator_type: str  # ip, domain, hash, url, email, etc.
    indicator_value: str
    threat_type: str
    severity: ThreatLevel
    confidence: float
    source: str
    description: str
    first_seen: datetime
    last_seen: datetime
    tags: list[str] = field(default_factory=list)
    related_indicators: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAsset:
    """Security asset model."""

    asset_id: str
    asset_type: str  # server, workstation, device, application, etc.
    name: str
    description: str
    criticality: ThreatLevel
    owner: str
    location: str | None = None
    ip_addresses: list[str] = field(default_factory=list)
    mac_addresses: list[str] = field(default_factory=list)
    operating_system: str | None = None
    installed_software: list[dict[str, str]] = field(default_factory=list)
    security_controls: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    last_scan: datetime | None = None
    compliance_status: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy model."""

    policy_id: str
    name: str
    description: str
    policy_type: str  # access_control, data_protection, network_security, etc.
    scope: list[str]  # Which assets/systems it applies to
    rules: list[dict[str, Any]]
    enforcement_level: str  # enforcing, permissive, monitoring
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    owner: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    effective_date: datetime | None = None
    expiry_date: datetime | None = None
    compliance_frameworks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityCompliance:
    """Security compliance model."""

    compliance_id: str
    framework: str  # NIST, ISO27001, PCI-DSS, HIPAA, etc.
    version: str
    scope: list[str]
    assessment_date: datetime
    assessor: str
    overall_score: float  # 0-100
    control_results: dict[str, dict[str, Any]]  # control_id -> result
    findings: list[dict[str, Any]]
    recommendations: list[str]
    next_assessment: datetime | None = None
    certification_status: str = "pending"  # pending, certified, non_compliant
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Security metrics model."""

    metric_id: str
    metric_name: str
    metric_type: str  # count, percentage, score, duration
    value: int | float | str
    unit: str
    measurement_time: datetime
    time_period: str  # daily, weekly, monthly, etc.
    source: str
    category: str  # availability, integrity, confidentiality, etc.
    target_value: int | float | None = None
    threshold_critical: int | float | None = None
    threshold_warning: int | float | None = None
    trend: str = "stable"  # improving, declining, stable
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessmentRecord:
    """Static risk-assessment record.

    Distinct from the canonical, behavior-carrying
    :class:`geo_infer_sec.models.risk_assessment.RiskAssessment` used by the
    CLI and acceptance tests; this dataclass is a serialized record shape.
    """

    assessment_id: str
    asset_id: str
    threat_id: str
    vulnerability_id: str
    impact_score: float  # 1-5
    likelihood_score: float  # 1-5
    risk_score: float  # impact * likelihood
    risk_level: ThreatLevel
    current_controls: list[str]
    control_effectiveness: float  # 0-1
    residual_risk_score: float
    risk_owner: str
    mitigation_plan: list[dict[str, Any]]
    assessment_date: datetime
    next_review: datetime | None = None
    status: str = "active"  # active, mitigated, accepted, transferred
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncidentWorkflow:
    """Security incident workflow model."""

    workflow_id: str
    incident_id: str
    workflow_name: str
    current_stage: str
    stages: list[dict[str, Any]]  # stage definitions
    stage_history: list[dict[str, Any]]  # completed stages
    assigned_team: str
    escalation_rules: list[dict[str, Any]]
    automation_rules: list[dict[str, Any]]
    sla_requirements: dict[str, Any]
    communication_plan: list[dict[str, Any]]
    evidence_chain: list[str]
    lessons_learned: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfiguration:
    """Security configuration model."""

    config_id: str
    config_name: str
    config_type: str  # system, application, network, security_tool
    target_asset: str
    configuration_items: dict[str, Any]
    baseline_config: dict[str, Any]
    security_requirements: list[str]
    compliance_mappings: dict[str, list[str]]  # framework -> controls
    change_history: list[dict[str, Any]]
    approved_by: str
    implementation_date: datetime | None = None
    review_date: datetime | None = None
    status: str = "draft"  # draft, approved, implemented, deprecated
    metadata: dict[str, Any] = field(default_factory=dict)


# Utility functions for model operations
class SecurityModelUtils:
    """Utility functions for security models."""

    @staticmethod
    def serialize_event(event: SecurityEvent) -> str:
        """Serialize security event to JSON string."""
        return json.dumps(event.to_dict(), default=str)

    @staticmethod
    def deserialize_event(json_str: str) -> SecurityEvent:
        """Deserialize security event from JSON string."""
        data = json.loads(json_str)
        return SecurityEvent.from_dict(data)

    @staticmethod
    def calculate_risk_score(
        impact: float, likelihood: float, control_effectiveness: float = 0.0
    ) -> float:
        """Calculate risk score with controls."""
        base_risk = impact * likelihood
        residual_risk = base_risk * (1 - control_effectiveness)
        return round(residual_risk, 2)

    @staticmethod
    def get_risk_level(risk_score: float) -> ThreatLevel:
        """Convert risk score to threat level."""
        if risk_score <= 5:
            return ThreatLevel.LOW
        elif risk_score <= 10:
            return ThreatLevel.MEDIUM
        elif risk_score <= 15:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL

    @staticmethod
    def merge_metadata(
        base_metadata: dict[str, Any], additional_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Safely merge metadata dictionaries."""
        merged = base_metadata.copy()
        merged.update(additional_metadata)
        return merged

    @staticmethod
    def filter_events_by_timeframe(
        events: list[SecurityEvent], start_time: datetime, end_time: datetime
    ) -> list[SecurityEvent]:
        """Filter events by time frame."""
        return [event for event in events if start_time <= event.timestamp <= end_time]

    @staticmethod
    def group_events_by_category(
        events: list[SecurityEvent],
    ) -> dict[str, list[SecurityEvent]]:
        """Group events by category."""
        grouped: dict[str, list[SecurityEvent]] = {}
        for event in events:
            category = event.category.value
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(event)
        return grouped

    @staticmethod
    def calculate_confidence_score(
        indicators: list[str], evidence_strength: dict[str, float]
    ) -> float:
        """Calculate confidence score based on indicators and evidence."""
        if not indicators:
            return 0.0

        total_strength = sum(
            evidence_strength.get(indicator, 0.5) for indicator in indicators
        )
        max_possible = len(indicators) * 1.0

        return min(1.0, total_strength / max_possible)

    @staticmethod
    def generate_event_signature(event: SecurityEvent) -> str:
        """Generate a unique signature for an event."""
        import hashlib

        signature_data = f"{event.event_type}:{event.source}:{event.target}"
        return hashlib.md5(signature_data.encode()).hexdigest()[:16]
