#!/usr/bin/env python3
"""
Tests for GEO-INFER-PLACE module bridge.

Validates PlaceDataManager (dataset validation, provenance) and
PlaceTemporalAnalyzer (trend detection, anomalies, forecast).
"""

import pytest

from geo_infer_place.core.module_bridge import (
    PlaceDataManager,
    PlaceTemporalAnalyzer,
)


# -- PlaceDataManager -------------------------------------------------------


class TestPlaceDataManager:
    """Test PlaceDataManager initialization and core methods."""

    def test_can_instantiate(self):
        mgr = PlaceDataManager()
        assert mgr is not None

    def test_validate_dataset_valid(self):
        """A well-formed dict dataset should validate successfully."""
        mgr = PlaceDataManager()
        report = mgr.validate_dataset({"key": "value"}, name="test_ds")
        assert isinstance(report, dict)
        assert "valid" in report

    def test_validate_dataset_empty(self):
        """An empty dataset should be flagged."""
        mgr = PlaceDataManager()
        report = mgr.validate_dataset({}, name="empty_ds")
        assert isinstance(report, dict)

    def test_provenance_logging(self):
        """log_provenance + get_provenance should round-trip."""
        mgr = PlaceDataManager()
        mgr.log_provenance("noaa_tides", metadata={"station": "9419750"})
        prov = mgr.get_provenance()
        assert isinstance(prov, list)
        assert len(prov) >= 1
        assert prov[-1]["source"] == "noaa_tides"


# -- PlaceTemporalAnalyzer --------------------------------------------------


class TestPlaceTemporalAnalyzer:
    """Test PlaceTemporalAnalyzer trend/anomaly/forecast methods."""

    def test_can_instantiate(self):
        analyzer = PlaceTemporalAnalyzer()
        assert analyzer is not None

    @pytest.mark.parametrize("threshold,count", [(0.0, 3), (1.0, 1), (2.0, 0)])
    def test_timestamped_anomaly_threshold_matches_coordinate_free_oracle(
        self, threshold, count
    ):
        """UTC timestamps retain the caller's statistical threshold."""
        values = [0.0, 0.0, 10.0]
        analyzer = PlaceTemporalAnalyzer()
        local = analyzer.detect_anomalies(values, sigma_threshold=threshold)
        actual = analyzer.detect_anomalies(
            values,
            sigma_threshold=threshold,
            timestamps=[f"2026-10-0{day}T00:00:00Z" for day in range(1, 4)],
        )
        assert actual["backend"] == "geo_infer_time"
        assert len(actual["anomalies"]) == len(local["anomalies"]) == count
        if threshold == 1.0:
            assert actual["anomalies"][0]["value"] == 10.0

    @pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1.0])
    def test_invalid_anomaly_threshold_fails_before_backend_selection(self, threshold):
        with pytest.raises(ValueError, match="finite and non-negative"):
            PlaceTemporalAnalyzer().detect_anomalies([], sigma_threshold=threshold)

    def test_detect_trend(self):
        """detect_trend on an increasing series should report positive slope."""
        analyzer = PlaceTemporalAnalyzer()
        values = list(range(100))
        result = analyzer.detect_trend(values, label="linear_up")
        assert isinstance(result, dict)
        assert result["slope"] > 0
        assert result["direction"] in ("increasing", "positive", "up")

    def test_detect_anomalies(self):
        """detect_anomalies should return a dict with an anomalies list."""
        analyzer = PlaceTemporalAnalyzer()
        values = [10.0] * 50 + [999.0] + [10.0] * 49  # one outlier
        result = analyzer.detect_anomalies(values, sigma_threshold=2.0)
        assert isinstance(result, dict)
        assert "anomalies" in result
        assert len(result["anomalies"]) >= 1

    def test_detect_trend_time_backend_preserves_real_axis_and_fit(self):
        """Actual TIME analysis preserves UTC instants and independent fit quality."""
        timestamps = [f"2026-10-0{day}T00:00:00Z" for day in range(1, 6)]
        result = PlaceTemporalAnalyzer().detect_trend(
            [5, 4, 3, 2, 1], timestamps=timestamps
        )
        assert result["backend"] == "geo_infer_time"
        assert result["direction"] == "decreasing"
        assert result["slope"] == pytest.approx(-1)
        assert result["r_squared"] == pytest.approx(1)
        assert result["significant"] is True
        weak = PlaceTemporalAnalyzer().detect_trend(
            [1, 2, 1, 2, 1], timestamps=timestamps
        )
        assert weak["r_squared"] == pytest.approx(0, abs=1e-12)
        assert weak["significant"] is False

    @pytest.mark.parametrize(
        "operation", ["detect_trend", "detect_anomalies", "forecast"]
    )
    def test_temporal_backend_rejects_naive_axis_without_fallback(self, operation):
        """Timestamp contract violations cannot silently change execution backend."""
        analyzer = PlaceTemporalAnalyzer()
        with pytest.raises(ValueError, match="timezone"):
            getattr(analyzer, operation)(
                [1, 2, 3], timestamps=["2026-10-01", "2026-10-02", "2026-10-03"]
            )

    def test_forecast_returns_values(self):
        """forecast should return predicted values list."""
        analyzer = PlaceTemporalAnalyzer()
        values = [float(x) for x in range(50)]
        result = analyzer.forecast(values, horizon=5)
        assert isinstance(result, dict)
        assert "forecast" in result
        assert len(result["forecast"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
