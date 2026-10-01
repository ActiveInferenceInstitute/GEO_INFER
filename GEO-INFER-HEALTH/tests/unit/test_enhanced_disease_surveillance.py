"""Unit tests for the Active Inference disease surveillance analyzer."""

from datetime import datetime, timedelta, UTC

import pytest

from geo_infer_health.core import ActiveInferenceDiseaseAnalyzer
from geo_infer_health.models import DiseaseReport, Location, PopulationData


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


@pytest.fixture
def clustered_reports():
    """Two tight clusters plus reports spread over several days."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    reports = []
    for day in range(10):
        reports.append(
            _report(
                f"c{day}-a",
                34.05 + day * 1e-4,
                -118.24,
                8 + day,
                base + timedelta(days=day),
            )
        )
        reports.append(
            _report(
                f"c{day}-b", 34.06 + day * 1e-4, -118.25, 6, base + timedelta(days=day)
            )
        )
        # A far-away lone case
        reports.append(_report(f"c{day}-c", 40.7, -74.0, 1, base + timedelta(days=day)))
    return reports


@pytest.fixture
def population():
    return [
        PopulationData(area_id="la", population_count=1000),
        PopulationData(area_id="nyc", population_count=1000),
    ]


def test_init_sorts_reports_by_date(clustered_reports):
    analyzer = ActiveInferenceDiseaseAnalyzer(reports=clustered_reports)
    dates = [r.report_date for r in analyzer.reports]
    assert dates == sorted(dates)


def test_init_rejects_non_list_reports():
    with pytest.raises(TypeError):
        ActiveInferenceDiseaseAnalyzer(reports="not-a-list")


def test_init_rejects_non_list_population(clustered_reports):
    with pytest.raises(TypeError):
        ActiveInferenceDiseaseAnalyzer(reports=clustered_reports, population_data=42)


def test_init_initializes_belief_states(clustered_reports):
    analyzer = ActiveInferenceDiseaseAnalyzer(reports=clustered_reports)
    assert analyzer.belief_states
    assert all(isinstance(v, float) for v in analyzer.belief_states.values())


def test_analyze_returns_full_result_structure(clustered_reports, population):
    analyzer = ActiveInferenceDiseaseAnalyzer(
        reports=clustered_reports, population_data=population
    )
    result = analyzer.analyze_with_active_inference()

    for key in (
        "belief_states",
        "belief_precisions",
        "observations",
        "traditional_hotspots",
        "enhanced_hotspots",
        "predictions",
        "confidence_intervals",
        "risk_assessment",
        "recommendations",
    ):
        assert key in result, f"missing key: {key}"

    assert isinstance(result["enhanced_hotspots"], list)
    assert isinstance(result["recommendations"], list) and result["recommendations"]
    assert "error" not in result


def test_analyze_time_window_filters_reports(clustered_reports, population):
    analyzer = ActiveInferenceDiseaseAnalyzer(
        reports=clustered_reports, population_data=population
    )
    result = analyzer.analyze_with_active_inference(time_window_days=3)
    assert "error" not in result
    assert result["observations"]


def test_analyze_rejects_negative_time_window(clustered_reports):
    analyzer = ActiveInferenceDiseaseAnalyzer(reports=clustered_reports)
    with pytest.raises(ValueError):
        analyzer.analyze_with_active_inference(time_window_days=-1)


def test_beliefs_update_after_analysis(clustered_reports):
    analyzer = ActiveInferenceDiseaseAnalyzer(reports=clustered_reports)
    before = dict(analyzer.belief_states)
    analyzer.analyze_with_active_inference()
    assert analyzer.belief_states != before


def test_empty_reports_yields_degenerate_analysis():
    analyzer = ActiveInferenceDiseaseAnalyzer(reports=[])
    result = analyzer.analyze_with_active_inference()
    # No crash, no bogus error: analysis simply runs over zero reports.
    assert "error" not in result
    assert result["traditional_hotspots"] == []
