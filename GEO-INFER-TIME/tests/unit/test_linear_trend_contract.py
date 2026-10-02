"""Independent numerical oracles for linear trends on the real UTC model."""

import numpy as np
import pandas as pd
import pytest

from geo_infer_time import TemporalAnalyzer, TimeSeries


def analyze(values):
    series = TimeSeries(
        np.asarray(values, dtype=float),
        timestamps=pd.date_range("2024-01-01", periods=len(values), tz="UTC"),
    )
    return TemporalAnalyzer().detect_trend(series)


@pytest.mark.parametrize("constant", [-1e300, -5.0, 0.0, 5.0, 1e300])
def test_signed_constants_are_stable_with_zero_slope(constant):
    result = analyze([constant] * 8)
    assert result["trend_direction"] == "stable"
    assert result["trend_strength"] == result["slope_per_sample"] == 0
    assert result["r_squared"] == 1
    np.testing.assert_array_equal(result["trend_values"], [constant] * 8)


@pytest.mark.parametrize("slope", [-3.0, 3.0, 1e-30])
def test_exact_lines_have_known_slope_and_perfect_fit(slope):
    values = slope * np.arange(8, dtype=float)
    result = analyze(values)
    assert result["trend_direction"] == ("increasing" if slope > 0 else "decreasing")
    assert result["slope_per_sample"] == pytest.approx(slope, rel=1e-12, abs=0)
    assert result["trend_strength"] == pytest.approx(abs(slope), rel=1e-12, abs=0)
    assert result["r_squared"] == pytest.approx(1)
    np.testing.assert_allclose(
        result["trend_values"], values, rtol=1e-12, atol=abs(slope) * 1e-12
    )


def test_representable_trend_survives_large_constant_offset():
    result = analyze(1e12 + 0.25 * np.arange(8))
    assert result["trend_direction"] == "increasing"
    assert result["slope_per_sample"] == pytest.approx(0.25, abs=1e-4)
    assert result["r_squared"] == pytest.approx(1, abs=1e-6)


def test_zero_slope_is_distinct_from_fit_quality():
    result = analyze([0, 2, 0])
    assert result["trend_direction"] == "stable"
    assert result["trend_strength"] == 0
    assert result["r_squared"] == pytest.approx(0)


@pytest.mark.parametrize("values", [[1], [1, np.nan], [1, np.inf]])
def test_insufficient_or_missing_observations_fail_explicitly(values):
    with pytest.raises(ValueError, match="at least two finite observations"):
        analyze(values)
