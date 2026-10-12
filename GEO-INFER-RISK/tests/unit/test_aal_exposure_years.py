"""Regression tests for calculate_aal exposure-years semantics (STATS-04).

AAL must be annualized by the number of exposure years the loss table
spans, not by the number of distinct events (which over-estimates AAL
whenever multiple events occur per year).
"""

import pandas as pd
import pytest

from geo_infer_risk.utils.risk_metrics import calculate_aal


@pytest.fixture
def multi_event_table():
    """4 events across 2 exposure years."""
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "hazard_type": ["flood"] * 4,
            "loss": [100.0, 100.0, 100.0, 100.0],
        }
    )


def test_aal_with_exposure_years_annualizes_correctly(multi_event_table):
    """AAL = total loss / exposure years (400 / 2 = 200)."""
    result = calculate_aal(multi_event_table, exposure_years=2.0)
    assert result["total"] == pytest.approx(200.0)
    assert result["by_hazard"]["flood"] == pytest.approx(200.0)


def test_aal_exposure_years_positive_required(multi_event_table):
    """Zero or negative exposure years is rejected."""
    with pytest.raises(ValueError, match="exposure_years"):
        calculate_aal(multi_event_table, exposure_years=0)
    with pytest.raises(ValueError, match="exposure_years"):
        calculate_aal(multi_event_table, exposure_years=-1)


def test_aal_requires_exposure_years(multi_event_table):
    """The removed event-count fallback: omitting exposure_years is an error."""
    with pytest.raises(TypeError, match="exposure_years"):
        calculate_aal(multi_event_table)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="requires exposure_years"):
        calculate_aal(multi_event_table, exposure_years=None)  # type: ignore[arg-type]


def test_aal_by_hazard_uses_exposure_years_not_event_counts():
    """Per-hazard AAL shares the exposure-years denominator."""
    table = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "hazard_type": ["wind", "wind", "flood"],
            "loss": [50.0, 150.0, 30.0],
        }
    )
    result = calculate_aal(table, exposure_years=10.0)
    assert result["total"] == pytest.approx(23.0)
    assert result["by_hazard"] == {
        "wind": pytest.approx(20.0),
        "flood": pytest.approx(3.0),
    }
