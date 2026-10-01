"""
Tests for unified TemporalAnalyzer.

Test suite covering temporal pattern analysis, anomaly detection,
and trend analysis using backend-agnostic data structures.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any

from geo_infer_space.analytics.temporal import TemporalAnalyzer

# Test data constants
SF_LAT, SF_LNG = 37.7749, -122.4194


@pytest.fixture
def temporal_data() -> list[dict[str, Any]]:
    """Create sample temporal data for testing."""
    data = []
    base_time = datetime(2023, 1, 1, 8, 0, 0)  # 8 AM

    # 24 hours of data, with 3 "sensors" or locations per hour
    for i in range(24):
        for j in range(3):
            # Create temporal patterns (higher activity during business hours)
            hour = i
            if 8 <= hour <= 18:  # Business hours
                activity = 100 + (j * 20)
            elif 19 <= hour <= 22:  # Evening
                activity = 80 + (j * 15)
            else:  # Night/early morning
                activity = 20 + (j * 5)

            record = {
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "trip_count": activity,
                "hour": hour,
                "location_id": f"loc_{j}",
            }
            data.append(record)

    return data


class TestTemporalAnalyzer:
    """Test temporal analysis functionality."""

    def test_analyzer_init(self):
        """Test analyzer initialization."""
        analyzer = TemporalAnalyzer()
        assert analyzer is not None

    def test_analyze_temporal_patterns(self, temporal_data):
        """Test temporal pattern analysis."""
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze_temporal_patterns(
            temporal_data,
            timestamp_column="timestamp",
            value_column="trip_count",
            temporal_resolution="hour",
        )

        assert "temporal_patterns" in result
        assert "aggregated_data" in result
        assert "statistics" in result
        assert "temporal_resolution" in result

        patterns = result["temporal_patterns"]
        assert "peak_periods" in patterns

        # Check peak periods
        peak_periods = patterns["peak_periods"]
        assert isinstance(peak_periods, list)
        if peak_periods:
            peak = peak_periods[0]
            assert "period" in peak
            assert "mean_value" in peak

    def test_temporal_patterns_different_resolutions(self, temporal_data):
        """Test temporal patterns with different resolutions."""
        analyzer = TemporalAnalyzer()

        # Note: Our sample data is only 24 hours, so day/week/month might have single buckets
        for resolution in ["hour", "day", "week", "month"]:
            result = analyzer.analyze_temporal_patterns(
                temporal_data,
                timestamp_column="timestamp",
                value_column="trip_count",
                temporal_resolution=resolution,
            )
            assert result["temporal_resolution"] == resolution
            assert "temporal_patterns" in result

    def test_insufficient_temporal_data(self):
        """Test handling of insufficient temporal data."""
        data = [{"timestamp": datetime.now().isoformat(), "value": 100}]

        analyzer = TemporalAnalyzer()
        # Should work but produce limited stats
        result = analyzer.analyze_temporal_patterns(
            data, timestamp_column="timestamp", value_column="value"
        )

        assert "temporal_patterns" in result
        assert result["data_points"] == 1

    def test_missing_columns(self, temporal_data):
        """Test handling of missing columns."""
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze_temporal_patterns(
            temporal_data, timestamp_column="non_existent", value_column="trip_count"
        )

        assert "error" in result
        assert result["error"] == "No valid temporal data found"


if __name__ == "__main__":
    pytest.main([__file__])
