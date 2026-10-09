"""Behavioral tests for underwriting data models (GS-146).

Tests decision finality/review gating, guideline
applicability and effectiveness windows, case state helpers, audit
records, and queue wait-time accounting.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import geo_infer_insurance.underwriting.models.underwriting_models as underwriting_models

from geo_infer_insurance.underwriting.models.underwriting_models import (
    AuditTrail,
    ComplianceCheck,
    Decision,
    DecisionStatus,
    Guideline,
    GuidelineType,
    UnderwritingCase,
    UnderwritingQueue,
)


def _decision(confidence: float = 0.9, conditions: list | None = None) -> Decision:
    return Decision(
        approved=True,
        reason="meets criteria",
        confidence=confidence,
        conditions=conditions or [],
    )


class TestDecision:
    def test_status_values_are_stable_strings(self) -> None:
        assert DecisionStatus.PENDING.value == "pending"
        assert DecisionStatus.APPROVED.value == "approved"
        assert DecisionStatus.DECLINED.value == "declined"
        assert DecisionStatus.REFERRED.value == "referred"

    def test_is_final_requires_high_confidence(self) -> None:
        assert _decision(confidence=0.9).is_final() is True
        assert _decision(confidence=0.8).is_final() is True
        assert _decision(confidence=0.79).is_final() is False

    def test_requires_review_when_confidence_low(self) -> None:
        assert _decision(confidence=0.5).requires_review() is True
        assert _decision(confidence=0.9).requires_review() is False

    def test_requires_review_when_conditions_attached(self) -> None:
        conditional = _decision(confidence=0.95, conditions=["new roof inspection"])

        assert conditional.requires_review() is True

    def test_to_dict_round_trips_core_fields(self) -> None:
        decision = _decision(confidence=0.9)
        decision.risk_score = 0.3

        payload = decision.to_dict()

        assert payload["approved"] is True
        assert payload["reason"] == "meets criteria"
        assert payload["confidence"] == 0.9
        assert payload["risk_score"] == 0.3
        assert payload["is_final"] is True
        assert payload["requires_review"] is False
        assert (
            datetime.fromisoformat(payload["decision_date"]) == decision.decision_date
        )


class TestGuideline:
    def test_guideline_type_values(self) -> None:
        assert GuidelineType.MANDATORY.value == "mandatory"
        assert GuidelineType.RISK_MANAGEMENT.value == "risk_management"

    def test_applicability_filters_by_product_region_and_tier(self) -> None:
        guideline = Guideline(
            guideline_id="g1",
            guideline_type=GuidelineType.ELIGIBILITY,
            name="Coastal rule",
            description="coastal only",
            rule_expression="region in coastal",
            applicable_products=["flood"],
            applicable_regions=["fl"],
            applicable_risk_tiers=["high"],
        )

        assert guideline.is_applicable("flood", "fl", "high") is True
        assert guideline.is_applicable("fire", "fl", "high") is False
        assert guideline.is_applicable("flood", "tx", "high") is False
        assert guideline.is_applicable("flood", "fl", "standard") is False

    def test_empty_applicability_lists_match_anything(self) -> None:
        guideline = Guideline(
            guideline_id="g2",
            guideline_type=GuidelineType.PRICING,
            name="Universal",
            description="no filters",
            rule_expression="always",
        )

        assert guideline.is_applicable("any-product", "any-region", "any-tier") is True

    def test_inactive_guideline_never_applicable_or_effective(self) -> None:
        guideline = Guideline(
            guideline_id="g3",
            guideline_type=GuidelineType.COVERAGE,
            name="Inactive",
            description="disabled",
            rule_expression="x > 1",
            is_active=False,
        )

        assert guideline.is_applicable("flood", "fl", "high") is False
        assert guideline.is_effective() is False

    def test_expired_guideline_not_effective(self) -> None:
        guideline = Guideline(
            guideline_id="g4",
            guideline_type=GuidelineType.EXCLUSION,
            name="Expired",
            description="past window",
            rule_expression="x > 1",
            expiration_date=datetime.now() - timedelta(days=1),
        )

        assert guideline.is_effective() is False

    def test_future_guideline_not_yet_effective(self) -> None:
        guideline = Guideline(
            guideline_id="g5",
            guideline_type=GuidelineType.COMPLIANCE,
            name="Future",
            description="not yet in force",
            rule_expression="x > 1",
            effective_date=datetime.now() + timedelta(days=30),
        )

        assert guideline.is_effective() is False


class TestUnderwritingCase:
    def test_new_case_defaults_and_completion(self) -> None:
        case = UnderwritingCase(case_id="c1", application_data={"value": 100})

        assert case.status == "pending"
        assert case.is_completed() is False
        assert case.decision is None

    def test_completed_case_records_duration(self) -> None:
        created = datetime.now() - timedelta(days=3)
        case = UnderwritingCase(case_id="c2", application_data={})
        case.created_at = created
        case.completed_at = created + timedelta(days=4)

        assert case.is_completed() is True
        assert case.days_open() == 4

    def test_old_pending_case_requires_attention(self) -> None:
        case = UnderwritingCase(case_id="c3", application_data={})
        case.created_at = datetime.now() - timedelta(days=5)

        assert case.requires_attention() is True

    def test_recent_case_does_not_require_attention(self) -> None:
        case = UnderwritingCase(case_id="c4", application_data={})
        case.created_at = datetime.now()

        assert case.requires_attention() is False

    def test_attention_excludes_closed_statuses(self) -> None:
        case = UnderwritingCase(case_id="c5", application_data={})
        case.created_at = datetime.now() - timedelta(days=5)
        case.status = "approved"

        assert case.requires_attention() is False

    def test_to_dict_serializes_nested_decision(self) -> None:
        case = UnderwritingCase(
            case_id="c6",
            application_data={"value": 10},
            decision=_decision(confidence=0.4),
        )

        payload = case.to_dict()

        assert payload["case_id"] == "c6"
        assert payload["decision"]["confidence"] == 0.4
        assert payload["decision"]["reason"] == "meets criteria"
        assert payload["is_completed"] is False


class TestAuditTrailAndComplianceCheck:
    def test_audit_trail_records_action(self) -> None:
        audit = AuditTrail(
            audit_id="a1",
            case_id="c1",
            action="decision_made",
            performed_by="underwriter-7",
            reason="score above threshold",
        )

        payload = audit.to_dict()

        assert payload["action"] == "decision_made"
        assert payload["performed_by"] == "underwriter-7"
        assert payload["case_id"] == "c1"
        assert datetime.fromisoformat(payload["timestamp"]) == audit.timestamp

    def test_compliance_check_compliance_statuses(self) -> None:
        passed = ComplianceCheck(
            check_id="chk1",
            check_type="capital",
            regulation="NAIC RBC",
            requirement="RBC ratio",
            status="passed",
        )
        exempt = ComplianceCheck(
            check_id="chk2",
            check_type="capital",
            regulation="NAIC RBC",
            requirement="RBC ratio",
            status="not_applicable",
        )
        failed = ComplianceCheck(
            check_id="chk3",
            check_type="capital",
            regulation="NAIC RBC",
            requirement="RBC ratio",
            status="failed",
        )
        warned = ComplianceCheck(
            check_id="chk4",
            check_type="capital",
            regulation="NAIC RBC",
            requirement="RBC ratio",
            status="warning",
        )

        assert passed.is_compliant() is True
        assert exempt.is_compliant() is True
        assert failed.is_compliant() is False
        assert warned.is_compliant() is False


class TestUnderwritingQueue:
    def test_add_and_remove_track_pending_count(self) -> None:
        queue = UnderwritingQueue(queue_id="q1", queue_type="standard")

        assert queue.add_to_queue("case-1") is True
        assert queue.add_to_queue("case-2", priority="urgent") is True
        assert queue.total_pending == 2

        assert queue.remove_from_queue("case-1") is True
        assert queue.total_pending == 1

    def test_add_rejects_unknown_priority_and_duplicates(self) -> None:
        queue = UnderwritingQueue(queue_id="q2", queue_type="standard")
        queue.add_to_queue("case-1")

        assert queue.add_to_queue("case-2", priority="asap") is False
        assert queue.add_to_queue("case-1") is False
        assert queue.total_pending == 1

    def test_remove_unknown_case_fails(self) -> None:
        queue = UnderwritingQueue(queue_id="q3", queue_type="standard")

        assert queue.remove_from_queue("ghost") is False

    @pytest.mark.parametrize("wait_seconds", [0.0, 2.0, 6.0])
    def test_wait_time_statistics_update_on_removal(
        self, monkeypatch: pytest.MonkeyPatch, wait_seconds: float
    ) -> None:
        start = datetime(2026, 1, 1)
        timestamps = iter((start, start + timedelta(seconds=wait_seconds)))
        monkeypatch.setattr(
            underwriting_models,
            "datetime",
            SimpleNamespace(now=lambda: next(timestamps)),
        )
        queue = UnderwritingQueue(queue_id="q4", queue_type="priority")

        assert queue.add_to_queue("case-1") is True
        assert queue.remove_from_queue("case-1") is True
        assert queue.average_wait_time == wait_seconds
        assert queue.longest_wait_time == wait_seconds
        assert queue._completed_waits == 1
        assert queue.total_pending == 0

    @pytest.mark.parametrize(
        ("first_wait", "second_wait"),
        [(2.0, 6.0), (6.0, 2.0), (0.0, 0.0), (0.0, 6.0), (6.0, 0.0)],
    )
    def test_average_wait_time_is_running_mean_across_removals(
        self, monkeypatch: pytest.MonkeyPatch, first_wait: float, second_wait: float
    ) -> None:
        start = datetime(2026, 1, 1)
        second_start = start + timedelta(seconds=10)
        timestamps = iter(
            (
                start,
                start + timedelta(seconds=first_wait),
                second_start,
                second_start + timedelta(seconds=second_wait),
            )
        )
        monkeypatch.setattr(
            underwriting_models,
            "datetime",
            SimpleNamespace(now=lambda: next(timestamps)),
        )
        queue = UnderwritingQueue(queue_id="q5", queue_type="standard")

        assert queue.add_to_queue("a") is True
        assert queue.remove_from_queue("a") is True
        assert queue.average_wait_time == first_wait
        assert queue.add_to_queue("b") is True
        assert queue.remove_from_queue("b") is True

        # Unequal known waits distinguish the running mean from the latest wait.
        expected_mean = (first_wait + second_wait) / 2.0
        assert queue._completed_waits == 2
        assert queue.total_pending == 0
        assert queue.average_wait_time == expected_mean
        if expected_mean > 0.0:
            assert queue.average_wait_time > 0.0
        assert queue.longest_wait_time == max(first_wait, second_wait)
