"""Failure-flag tests for analyze_with_active_inference (GS19-14).

Contract: each guarded stage appends its name to ``failed_stages`` and the
result carries ``degraded: True`` when any stage failed, so callers can
distinguish degraded placeholder output from real analysis values.
"""

from datetime import datetime, UTC

import pytest

from geo_infer_health.core import ActiveInferenceDiseaseAnalyzer
from geo_infer_health.models import DiseaseReport, Location

STAGES = {
    "_extract_observations": "extract_observations",
    "_traditional_hotspot_analysis": "traditional_hotspots",
    "_enhanced_hotspot_analysis": "enhanced_hotspots",
    "_generate_predictions": "predictions",
    "_calculate_confidence_intervals": "confidence_intervals",
    "_assess_overall_risk": "risk_assessment",
    "_generate_recommendations": "recommendations",
}


def _report(
    report_id: str, lat: float, lon: float, case_count: int, date: datetime
) -> DiseaseReport:
    return DiseaseReport(
        report_id=report_id,
        disease_code="FLU",
        location=Location(latitude=lat, longitude=lon),
        report_date=date,
        case_count=case_count,
        source="Hospital A",
    )


def _analyzer() -> ActiveInferenceDiseaseAnalyzer:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    reports = [
        _report("c0-a", 34.05, -118.24, 8, base),
        _report("c1-a", 34.05, -118.24, 9, base),
    ]
    return ActiveInferenceDiseaseAnalyzer(reports=reports)


def _raiser():
    def boom(*args, **kwargs):
        raise RuntimeError("stage failed")

    return boom


@pytest.mark.parametrize("helper, stage", sorted(STAGES.items()))
def test_stage_failure_is_flagged(monkeypatch, helper, stage):
    analyzer = _analyzer()
    monkeypatch.setattr(analyzer, helper, _raiser())
    result = analyzer.analyze_with_active_inference()
    assert stage in result["failed_stages"]
    assert result["degraded"] is True


def test_update_beliefs_failure_is_flagged(monkeypatch):
    analyzer = _analyzer()
    monkeypatch.setattr(analyzer, "_update_beliefs", _raiser())
    result = analyzer.analyze_with_active_inference()
    assert "update_beliefs" in result["failed_stages"]
    assert result["degraded"] is True


def test_healthy_run_has_no_failed_stages():
    analyzer = _analyzer()
    result = analyzer.analyze_with_active_inference()
    assert result["failed_stages"] == []
    assert result["degraded"] is False


def test_critical_error_result_is_marked_degraded():
    """The outer critical handler also reports failed_stages + degraded."""
    analyzer = _analyzer()
    analyzer.belief_states = None  # .copy() fails inside the critical handler
    result = analyzer.analyze_with_active_inference()
    assert result["failed_stages"] == ["analysis"]
    assert result["degraded"] is True
