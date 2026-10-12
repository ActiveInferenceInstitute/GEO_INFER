"""
Tests for extended temporal analysis methods in GEO-INFER-TIME.

Tests the new methods: calculate_rolling_statistics, detect_periodicity,
calculate_granger_causality, and compute_temporal_entropy.
"""

import pytest
import numpy as np
import pandas as pd

from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.models.timeseries import TimeSeries


@pytest.fixture
def analyzer():
    """Create a TemporalAnalyzer instance."""
    return TemporalAnalyzer()


@pytest.fixture
def sample_timeseries():
    """Create a sample time series for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC")
    values = (
        np.sin(np.arange(100) * 2 * np.pi / 12) * 10 + 50 + np.random.randn(100) * 2
    )
    return TimeSeries(data=pd.Series(values, index=dates))


@pytest.fixture
def periodic_timeseries():
    """Create a time series with clear periodicity."""
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D", tz="UTC")
    # Clear 7-day period
    values = 10 * np.sin(np.arange(120) * 2 * np.pi / 7) + 50
    return TimeSeries(data=pd.Series(values, index=dates))


@pytest.fixture
def causal_timeseries_pair():
    """Create a pair of time series where one causes the other."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC")

    # Series 1 - random walk
    np.random.seed(42)
    values1 = np.cumsum(np.random.randn(100))

    # Series 2 - lagged version of series 1 (with noise)
    values2 = np.roll(values1, 3) + np.random.randn(100) * 0.5

    ts1 = TimeSeries(data=pd.Series(values1, index=dates))
    ts2 = TimeSeries(data=pd.Series(values2, index=dates))

    return ts1, ts2


class TestRollingStatistics:
    """Tests for calculate_rolling_statistics method."""

    def test_default_statistics(self, analyzer, sample_timeseries):
        """Test calculating all default statistics."""
        result = analyzer.calculate_rolling_statistics(sample_timeseries, window=10)

        assert "statistics" in result
        assert "window" in result
        assert result["window"] == 10

        # Check all default stats are present
        expected_stats = ["mean", "std", "var", "min", "max", "sum", "median"]
        for stat in expected_stats:
            assert stat in result["statistics"], f"Missing statistic: {stat}"

    def test_specific_statistics(self, analyzer, sample_timeseries):
        """Test calculating specific statistics."""
        result = analyzer.calculate_rolling_statistics(
            sample_timeseries, window=5, statistics=["mean", "std"]
        )

        assert "mean" in result["statistics"]
        assert "std" in result["statistics"]
        assert "min" not in result["statistics"]

    def test_bollinger_bands(self, analyzer, sample_timeseries):
        """Test that Bollinger bands are calculated."""
        result = analyzer.calculate_rolling_statistics(sample_timeseries, window=10)

        assert "bollinger_upper" in result["statistics"]
        assert "bollinger_lower" in result["statistics"]

    def test_latest_values(self, analyzer, sample_timeseries):
        """Test that latest values are included."""
        result = analyzer.calculate_rolling_statistics(sample_timeseries, window=10)

        for stat in ["mean", "std"]:
            assert "latest" in result["statistics"][stat]

    def test_valid_observations_count(self, analyzer, sample_timeseries):
        """Test valid observations count."""
        window = 10
        result = analyzer.calculate_rolling_statistics(sample_timeseries, window=window)

        expected_valid = len(sample_timeseries) - window + 1
        assert result["summary"]["valid_observations"] == expected_valid


class TestPeriodicityDetection:
    """Tests for detect_periodicity method."""

    def test_detect_known_periodicity(self, analyzer, periodic_timeseries):
        """Test detecting a known periodicity."""
        result = analyzer.detect_periodicity(periodic_timeseries, max_period=30)

        assert "dominant_period" in result
        assert "top_periods" in result

        # Should detect period close to 7
        dominant = result["dominant_period"]["period"]
        assert dominant is not None
        assert 5 < dominant < 10  # Allow some tolerance

    def test_periodicity_structure(self, analyzer, sample_timeseries):
        """Test result structure."""
        result = analyzer.detect_periodicity(sample_timeseries, max_period=30)

        assert "series_length" in result
        assert "max_period_searched" in result
        assert "spectral_entropy" in result
        assert "dominant_period" in result
        assert "interpretation" in result["dominant_period"]

    def test_short_series_error(self, analyzer):
        """Test handling of very short series."""
        dates = pd.date_range(start="2024-01-01", periods=3, freq="D", tz="UTC")
        values = [1, 2, 3]
        short_ts = TimeSeries(data=pd.Series(values, index=dates))

        result = analyzer.detect_periodicity(short_ts)

        assert "error" in result

    def test_top_periods_ordered(self, analyzer, sample_timeseries):
        """Test that top periods are ordered by power."""
        result = analyzer.detect_periodicity(sample_timeseries, max_period=30)

        top_periods = result.get("top_periods", [])
        if len(top_periods) > 1:
            powers = [p["power"] for p in top_periods]
            assert powers == sorted(powers, reverse=True)


class TestGrangerCausality:
    """Tests for calculate_granger_causality method."""

    def test_causality_structure(self, analyzer, causal_timeseries_pair):
        """Test result structure."""
        ts1, ts2 = causal_timeseries_pair
        result = analyzer.calculate_granger_causality(ts1, ts2, max_lag=3)

        assert "series_length" in result
        assert "max_lag" in result
        assert "tests" in result
        assert "summary" in result

    def test_both_directions_tested(self, analyzer, causal_timeseries_pair):
        """Test that both causality directions are tested."""
        ts1, ts2 = causal_timeseries_pair
        result = analyzer.calculate_granger_causality(ts1, ts2, max_lag=2)

        tests = result["tests"]

        # Should have tests in both directions
        s1_to_s2 = [k for k in tests.keys() if "series1_causes_series2" in k]
        s2_to_s1 = [k for k in tests.keys() if "series2_causes_series1" in k]

        assert len(s1_to_s2) > 0
        assert len(s2_to_s1) > 0

    def test_summary_interpretation(self, analyzer, causal_timeseries_pair):
        """Test that summary includes interpretation."""
        ts1, ts2 = causal_timeseries_pair
        result = analyzer.calculate_granger_causality(ts1, ts2, max_lag=3)

        assert "interpretation" in result["summary"]
        assert isinstance(result["summary"]["interpretation"], str)

    def test_f_statistics(self, analyzer, causal_timeseries_pair):
        """Test that F-statistics are calculated."""
        ts1, ts2 = causal_timeseries_pair
        result = analyzer.calculate_granger_causality(ts1, ts2, max_lag=2)

        for test_result in result["tests"].values():
            if "error" not in test_result:
                assert "f_statistic" in test_result
                assert "p_value" in test_result
                assert "significant" in test_result


class TestTemporalEntropy:
    """Tests for compute_temporal_entropy method."""

    def test_shannon_entropy(self, analyzer, sample_timeseries):
        """Test Shannon entropy calculation."""
        result = analyzer.compute_temporal_entropy(sample_timeseries, bins=10)

        assert "shannon_entropy" in result
        assert "value" in result["shannon_entropy"]
        assert "normalized" in result["shannon_entropy"]

        # Shannon entropy should be non-negative
        assert result["shannon_entropy"]["value"] >= 0

    def test_normalized_entropy_range(self, analyzer, sample_timeseries):
        """Test normalized entropy is in valid range."""
        result = analyzer.compute_temporal_entropy(sample_timeseries, bins=10)

        normalized = result["shannon_entropy"]["normalized"]
        assert 0 <= normalized <= 1.0

    def test_sample_entropy(self, analyzer, sample_timeseries):
        """Test sample entropy calculation."""
        result = analyzer.compute_temporal_entropy(
            sample_timeseries, bins=10, method="sample"
        )

        assert "sample_entropy" in result

    def test_interpretation(self, analyzer, sample_timeseries):
        """Test entropy interpretation."""
        result = analyzer.compute_temporal_entropy(sample_timeseries, bins=10)

        assert "interpretation" in result
        assert "complexity" in result["interpretation"]
        assert "predictability" in result["interpretation"]
        assert "description" in result["interpretation"]

    def test_low_entropy_series(self, analyzer):
        """Test that a regular series has low entropy."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC")
        # Very regular series
        values = [1, 2] * 50
        regular_ts = TimeSeries(data=pd.Series(values, index=dates))

        result = analyzer.compute_temporal_entropy(regular_ts, bins=10)

        # Regular series should have low normalized entropy
        assert result["shannon_entropy"]["normalized"] < 0.35

    def test_high_entropy_series(self, analyzer):
        """Test that a random series has high entropy."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC")
        np.random.seed(42)
        values = np.random.uniform(0, 100, 100)
        random_ts = TimeSeries(data=pd.Series(values, index=dates))

        result = analyzer.compute_temporal_entropy(random_ts, bins=10)

        # Random series should have high normalized entropy
        assert result["shannon_entropy"]["normalized"] > 0.7


class TestMethodIntegration:
    """Integration tests combining multiple methods."""

    def test_analysis_pipeline(self, analyzer, sample_timeseries):
        """Test running multiple analysis methods in sequence."""
        # Calculate rolling stats
        rolling = analyzer.calculate_rolling_statistics(sample_timeseries, window=10)
        assert "statistics" in rolling

        # Detect periodicity
        periodicity = analyzer.detect_periodicity(sample_timeseries, max_period=30)
        assert "dominant_period" in periodicity

        # Compute entropy
        entropy = analyzer.compute_temporal_entropy(sample_timeseries, bins=10)
        assert "shannon_entropy" in entropy

        # All results should be valid
        assert rolling["window"] == 10
        assert periodicity["series_length"] == len(sample_timeseries)
        assert entropy["series_length"] == len(sample_timeseries)

    def test_paired_analysis(self, analyzer, causal_timeseries_pair):
        """Test analyzing paired time series."""
        ts1, ts2 = causal_timeseries_pair

        # Cross-correlation
        cross_corr = analyzer.calculate_cross_correlation(ts1, ts2, max_lag=10)
        assert "correlations" in cross_corr

        # Granger causality
        granger = analyzer.calculate_granger_causality(ts1, ts2, max_lag=3)
        assert "summary" in granger


class TestDecompose:
    """Regression tests for TemporalAnalyzer.decompose period handling."""

    def test_explicit_period_returns_components(self, analyzer):
        """An explicit period yields trend/seasonal/residual components."""
        dates = pd.date_range(start="2024-01-01", periods=120, freq="D", tz="UTC")
        values = 10 * np.sin(np.arange(120) * 2 * np.pi / 7) + 50
        result = analyzer.decompose(
            TimeSeries(data=pd.Series(values, index=dates)), period=7
        )
        assert sorted(result) == ["model", "period", "residual", "seasonal", "trend"]
        assert result["period"] == 7
        assert result["trend"] and result["seasonal"] and result["residual"]

    def test_daily_frequency_infers_weekly_period(self, analyzer):
        """Daily data infers a 7-sample weekly period."""
        dates = pd.date_range(start="2024-01-01", periods=120, freq="D", tz="UTC")
        ts = TimeSeries(data=pd.Series(np.arange(120.0), index=dates))
        assert ts.frequency == "D"
        assert analyzer.decompose(ts)["period"] == 7

    def test_hourly_frequency_infers_daily_period(self, analyzer):
        """Hourly data infers a 24-sample daily period."""
        dates = pd.date_range(start="2024-01-01", periods=120, freq="h", tz="UTC")
        ts = TimeSeries(data=pd.Series(np.arange(120.0), index=dates))
        assert analyzer.decompose(ts)["period"] == 24

    def test_hourly_multiple_infers_pro_rated_period(self, analyzer):
        """'2h' sampling infers a 12-sample daily period, not 24."""
        dates = pd.date_range(start="2024-01-01", periods=120, freq="2h", tz="UTC")
        ts = TimeSeries(data=pd.Series(np.arange(120.0), index=dates))
        assert analyzer.decompose(ts)["period"] == 12

    def test_sub_daily_sample_frequency_raises(self, analyzer):
        """'15min' data cannot map to a sample count and must raise."""
        dates = pd.date_range(start="2024-01-01", periods=300, freq="15min", tz="UTC")
        ts = TimeSeries(data=pd.Series(np.arange(300.0), index=dates))
        with pytest.raises(ValueError, match="pass period explicitly"):
            analyzer.decompose(ts)

    def test_quarterly_frequency_infers_quarterly_period(self, analyzer):
        """'QE' data maps to a 4-sample period (no misfire on the embedded 'D')."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="QE", tz="UTC")
        ts = TimeSeries(data=pd.Series(np.arange(40.0), index=dates))
        assert analyzer.decompose(ts)["period"] == 4

    def test_missing_frequency_raises(self, analyzer):
        """Irregular data without an inferable frequency must raise."""
        dates = pd.DatetimeIndex(
            [
                "2024-01-01 00:00",
                "2024-01-01 02:17",
                "2024-01-03 05:05",
                "2024-01-09 11:00",
                "2024-01-10 23:30",
                "2024-01-21 08:12",
                "2024-02-02 04:44",
                "2024-02-19 19:19",
            ],
            tz="UTC",
        )
        ts = TimeSeries(data=pd.Series(np.arange(8.0), index=dates))
        assert ts.frequency is None
        with pytest.raises(ValueError, match="regular cadence"):
            analyzer.decompose(ts)

    def test_period_too_large_raises(self, analyzer, sample_timeseries):
        """A period exceeding half the series length must raise."""
        with pytest.raises(ValueError, match="invalid for a series of length"):
            analyzer.decompose(sample_timeseries, period=100)

    def test_period_too_small_raises(self, analyzer, sample_timeseries):
        """A period below 2 is invalid and must raise."""
        with pytest.raises(ValueError, match="period must be an integer >= 2"):
            analyzer.decompose(sample_timeseries, period=1)
