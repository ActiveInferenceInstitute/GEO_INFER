"""Deterministic behavioral tests for GEO-INFER-TIME event detection."""

import numpy as np
import pandas as pd
import pytest

from geo_infer_time.core.event_detection import EventDetector
from geo_infer_time.models.timeseries import TimeSeries


def _timeseries(values: list[float]) -> TimeSeries:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return TimeSeries(pd.Series(values, index=dates, name="value"))


@pytest.fixture
def detector() -> EventDetector:
    return EventDetector(threshold_multiplier=3.0, window_size=10)


@pytest.fixture
def series_with_anomalies() -> TimeSeries:
    rng = np.random.default_rng(42)
    values = rng.normal(50, 2, 100)
    values[25] = 150
    values[75] = -50
    return _timeseries(values.tolist())


@pytest.fixture
def series_with_changepoint() -> TimeSeries:
    rng = np.random.default_rng(7)
    values = np.concatenate((rng.normal(10, 0.5, 50), rng.normal(50, 0.5, 50)))
    return _timeseries(values.tolist())


@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_statistical_anomaly_methods_detect_inserted_outliers(
    detector: EventDetector,
    series_with_anomalies: TimeSeries,
    method: str,
) -> None:
    result = detector.detect_anomalies(series_with_anomalies, method=method)

    detected_values = {anomaly["value"] for anomaly in result["anomalies"]}
    assert result["method"] == method
    assert result["count"] == len(result["anomalies"])
    assert {150.0, -50.0}.issubset(detected_values)
    assert all(
        {"timestamp", "value", "type"}.issubset(anomaly)
        for anomaly in result["anomalies"]
    )


def test_isolation_forest_uses_real_timeseries_contract(
    detector: EventDetector, series_with_anomalies: TimeSeries
) -> None:
    result = detector.detect_anomalies(series_with_anomalies, method="isolation_forest")

    assert result["method"] == "isolation_forest"
    assert result["count"] == len(result["anomalies"])
    assert result["count"] > 0
    assert all("score" in anomaly for anomaly in result["anomalies"])


def test_constant_and_missing_series_have_no_anomalies(detector: EventDetector) -> None:
    constant = _timeseries([50.0] * 20)
    missing = _timeseries([np.nan] * 20)

    assert detector.detect_anomalies(constant, method="z_score")["count"] == 0
    assert detector.detect_anomalies(missing, method="iqr")["count"] == 0


def test_invalid_anomaly_method_raises_even_for_empty_series(
    detector: EventDetector,
) -> None:
    empty = TimeSeries(pd.DataFrame(index=pd.DatetimeIndex([])))

    with pytest.raises(ValueError, match="Unknown anomaly detection method"):
        detector.detect_anomalies(empty, method="invalid")


def test_empty_series_returns_empty_event_results(detector: EventDetector) -> None:
    empty = TimeSeries(pd.DataFrame(index=pd.DatetimeIndex([])))

    assert detector.detect_anomalies(empty) == {
        "method": "z_score",
        "anomalies": [],
        "count": 0,
    }
    assert detector.detect_changepoints(empty) == {"changepoints": [], "count": 0}


def test_changepoint_is_detected_near_level_shift(
    detector: EventDetector, series_with_changepoint: TimeSeries
) -> None:
    result = detector.detect_changepoints(series_with_changepoint, sensitivity=0.5)

    assert result["count"] == len(result["changepoints"])
    assert result["count"] > 0
    assert any(40 <= point["index"] <= 60 for point in result["changepoints"])
    assert all(
        {"timestamp", "index", "mean_change", "mean_before", "mean_after"}.issubset(
            point
        )
        for point in result["changepoints"]
    )


def test_constant_series_has_no_changepoints(detector: EventDetector) -> None:
    result = detector.detect_changepoints(_timeseries([50.0] * 100), sensitivity=0)

    assert result == {"changepoints": [], "count": 0}


def test_sensitivity_changes_detection_count() -> None:
    rng = np.random.default_rng(11)
    values = np.concatenate((rng.normal(10, 2, 50), rng.normal(15, 2, 50)))
    series = _timeseries(values.tolist())
    detector = EventDetector(window_size=10)

    low = detector.detect_changepoints(series, sensitivity=0.1)
    high = detector.detect_changepoints(series, sensitivity=5.0)

    assert low["count"] >= high["count"]


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"threshold_multiplier": -1}, ValueError),
        ({"threshold_multiplier": np.inf}, ValueError),
        ({"threshold_multiplier": "3"}, TypeError),
        ({"window_size": 0}, ValueError),
        ({"window_size": 2.5}, TypeError),
    ],
)
def test_detector_rejects_invalid_configuration(kwargs, exception) -> None:
    with pytest.raises(exception):
        EventDetector(**kwargs)


@pytest.mark.parametrize("sensitivity", [-1, np.inf])
def test_changepoint_rejects_invalid_sensitivity(
    detector: EventDetector, sensitivity: float
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        detector.detect_changepoints(_timeseries([1.0, 2.0]), sensitivity=sensitivity)
