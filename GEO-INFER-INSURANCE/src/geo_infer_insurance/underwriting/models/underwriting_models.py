"""
Underwriting Models: Data structures for underwriting case management.

This module provides data models for underwriting operations including:
- Underwriting case management
- Decision structures and criteria
- Guideline and rule management
- Audit and compliance tracking
"""

from typing import Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class DecisionStatus(Enum):
    """Underwriting decision status enumeration."""

    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    REFERRED = "referred"
    CONDITIONAL = "conditional"
    EXPIRED = "expired"


class GuidelineType(Enum):
    """Underwriting guideline type enumeration."""

    MANDATORY = "mandatory"
    ELIGIBILITY = "eligibility"
    PRICING = "pricing"
    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    COMPLIANCE = "compliance"
    RISK_MANAGEMENT = "risk_management"


@dataclass
class Decision:
    """Underwriting decision structure."""

    approved: bool
    reason: str
    confidence: float = 0.0
    risk_score: float = 0.0
    rule_score: float = 0.0
    conditions: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    decision_date: datetime = field(default_factory=datetime.now)
    decision_maker: str = "system"

    def is_final(self) -> bool:
        """Check if decision is final."""
        return self.confidence >= 0.8 and self.decision_date is not None

    def requires_review(self) -> bool:
        """Check if decision requires manual review."""
        return self.confidence < 0.7 or len(self.conditions) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "approved": self.approved,
            "reason": self.reason,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "rule_score": self.rule_score,
            "conditions": self.conditions,
            "requirements": self.requirements,
            "recommendations": self.recommendations,
            "decision_date": self.decision_date.isoformat(),
            "decision_maker": self.decision_maker,
            "is_final": self.is_final(),
            "requires_review": self.requires_review(),
        }


@dataclass
class Guideline:
    """Underwriting guideline structure."""

    guideline_id: str
    guideline_type: GuidelineType
    name: str
    description: str

    # Rule definition
    rule_expression: str
    rule_parameters: dict[str, Any] = field(default_factory=dict)

    # Applicability
    applicable_products: list[str] = field(default_factory=list)
    applicable_regions: list[str] = field(default_factory=list)
    applicable_risk_tiers: list[str] = field(default_factory=list)

    # Metadata
    effective_date: datetime = field(default_factory=datetime.now)
    expiration_date: datetime | None = None
    version: str = "1.0"
    created_by: str = "system"
    approved_by: str | None = None

    # Status
    is_active: bool = True
    priority: int = 1

    def is_applicable(self, product: str, region: str, risk_tier: str) -> bool:
        """Check if guideline is applicable."""
        return (
            self.is_active
            and (not self.applicable_products or product in self.applicable_products)
            and (not self.applicable_regions or region in self.applicable_regions)
            and (
                not self.applicable_risk_tiers
                or risk_tier in self.applicable_risk_tiers
            )
        )

    def is_effective(self) -> bool:
        """Check if guideline is currently effective."""
        now = datetime.now()
        return (
            self.is_active
            and self.effective_date <= now
            and (self.expiration_date is None or self.expiration_date >= now)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert guideline to dictionary."""
        return {
            "guideline_id": self.guideline_id,
            "guideline_type": self.guideline_type.value,
            "name": self.name,
            "description": self.description,
            "rule_expression": self.rule_expression,
            "rule_parameters": self.rule_parameters,
            "applicable_products": self.applicable_products,
            "applicable_regions": self.applicable_regions,
            "applicable_risk_tiers": self.applicable_risk_tiers,
            "effective_date": self.effective_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat()
            if self.expiration_date
            else None,
            "version": self.version,
            "created_by": self.created_by,
            "approved_by": self.approved_by,
            "is_active": self.is_active,
            "priority": self.priority,
            "is_applicable": self.is_applicable("default", "default", "standard"),
            "is_effective": self.is_effective(),
        }


@dataclass
class UnderwritingCase:
    """Underwriting case structure."""

    case_id: str
    application_data: dict[str, Any]
    status: str = "pending"

    # Assessment results
    risk_assessment: dict[str, Any] | None = None
    rule_evaluation: dict[str, Any] | None = None

    # Financial information
    premium: float = 0.0

    # Decision
    decision: Decision | None = None

    # Policy (if approved)
    policy: dict[str, Any] | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: str | None = None
    priority: str = "normal"

    # Error handling
    error_message: str | None = None
    completed_at: datetime | None = None

    def is_completed(self) -> bool:
        """Check if case is completed."""
        return self.completed_at is not None

    def days_open(self) -> int:
        """Calculate days since case was created."""
        now = datetime.now()
        if self.completed_at:
            return (self.completed_at - self.created_at).days
        return (now - self.created_at).days

    def requires_attention(self) -> bool:
        """Check if case requires attention."""
        return (
            self.status in ["pending", "in_review"]
            and self.days_open() > 2  # Cases open > 2 days need attention
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert underwriting case to dictionary."""
        return {
            "case_id": self.case_id,
            "application_data": self.application_data,
            "status": self.status,
            "risk_assessment": self.risk_assessment,
            "rule_evaluation": self.rule_evaluation,
            "premium": self.premium,
            "decision": self.decision.to_dict() if self.decision else None,
            "policy": self.policy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "error_message": self.error_message,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "is_completed": self.is_completed(),
            "days_open": self.days_open(),
            "requires_attention": self.requires_attention(),
        }


@dataclass
class AuditTrail:
    """Audit trail for underwriting operations."""

    audit_id: str
    case_id: str
    action: str
    performed_by: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Action details
    old_values: dict[str, Any] = field(default_factory=dict)
    new_values: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    # Context
    system_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert audit trail to dictionary."""
        return {
            "audit_id": self.audit_id,
            "case_id": self.case_id,
            "action": self.action,
            "performed_by": self.performed_by,
            "timestamp": self.timestamp.isoformat(),
            "old_values": self.old_values,
            "new_values": self.new_values,
            "reason": self.reason,
            "system_context": self.system_context,
        }


@dataclass
class ComplianceCheck:
    """Compliance check result."""

    check_id: str
    check_type: str
    regulation: str
    requirement: str
    status: str  # passed, failed, warning, not_applicable
    details: str = ""
    evidence: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)

    def is_compliant(self) -> bool:
        """Check if compliance check passed."""
        return self.status in ["passed", "not_applicable"]

    def to_dict(self) -> dict[str, Any]:
        """Convert compliance check to dictionary."""
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "regulation": self.regulation,
            "requirement": self.requirement,
            "status": self.status,
            "details": self.details,
            "evidence": self.evidence,
            "checked_at": self.checked_at.isoformat(),
            "is_compliant": self.is_compliant(),
        }


@dataclass
class UnderwritingQueue:
    """Underwriting queue management."""

    queue_id: str
    queue_type: str  # standard, priority, specialist, manual_review
    max_concurrent: int = 10
    priority_levels: list[str] = field(
        default_factory=lambda: ["low", "normal", "high", "urgent"]
    )

    # Queue statistics
    total_pending: int = 0
    average_wait_time: float = 0.0
    longest_wait_time: float = 0.0

    # Completed removals used as the running-average denominator
    _completed_waits: int = field(default=0, repr=False)

    # Entry timestamps per enqueued case
    _case_enqueued_at: dict[str, datetime] = field(default_factory=dict)

    def add_to_queue(self, case_id: str, priority: str = "normal") -> bool:
        """Add case to queue, recording its entry timestamp."""
        if priority not in self.priority_levels:
            return False
        if case_id in self._case_enqueued_at:
            return False

        self._case_enqueued_at[case_id] = datetime.now()
        self.total_pending += 1
        return True

    def remove_from_queue(self, case_id: str) -> bool:
        """Remove case from queue, updating wait-time statistics."""
        enqueued_at = self._case_enqueued_at.pop(case_id, None)
        if enqueued_at is None:
            return False

        wait_seconds = (datetime.now() - enqueued_at).total_seconds()
        self._completed_waits += 1
        self.total_pending = max(0, self.total_pending - 1)
        self.average_wait_time = (
            self.average_wait_time * (self._completed_waits - 1) + wait_seconds
        ) / self._completed_waits
        self.longest_wait_time = max(self.longest_wait_time, wait_seconds)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert queue to dictionary."""
        return {
            "queue_id": self.queue_id,
            "queue_type": self.queue_type,
            "max_concurrent": self.max_concurrent,
            "priority_levels": self.priority_levels,
            "total_pending": self.total_pending,
            "average_wait_time": self.average_wait_time,
            "longest_wait_time": self.longest_wait_time,
        }
