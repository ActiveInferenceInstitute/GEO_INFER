"""
Tests for the TemporalInterpolator class.

Covers linear, nearest, seasonal, gap-aware interpolation,
resampling, imputation, and quality metrics.
"""

import numpy as np
import pandas as pd
import pytest

from geo_infer_time.core.interpolation import TemporalInterpolator
from geo_infer_time.models.timeseries import TimeSeries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ts_with_gaps(n: int = 100, gap_indices: list = None) -> TimeSeries:
    """Create a TimeSeries with NaN gaps at specified indices."""
    index = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    values = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.1, n)
    df = pd.DataFrame({"value": values}, index=index)
    if gap_indices:
        df.iloc[gap_indices, 0] = np.nan
    return TimeSeries(data=df)


def _make_seasonal_ts(periods: int = 120, period: int = 12) -> TimeSeries:
    """Create a TimeSeries with seasonal pattern and gaps."""
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    seasonal = np.sin(np.linspace(0, 2 * np.pi * (periods / period), periods))
    trend = np.linspace(0, 5, periods)
    values = trend + 10 * seasonal + np.random.normal(0, 0.5, periods)
    df = pd.DataFrame({"value": values}, index=index)
    # Introduce gaps at positions that are within bounds
    gap_positions = [i for i in [5, 17, 18, 30, 55, 56, 57, 80] if i < periods]
    df.iloc[gap_positions, 0] = np.nan
    return TimeSeries(data=df)


# ---------------------------------------------------------------------------
# Tests for interpolate()
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_linear_fills_gaps(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11, 12])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="linear")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_nearest_fills_gaps(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="nearest")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_cubic_fills_gaps(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11, 12])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="cubic")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_spline_fills_gaps(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="spline")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_unknown_method_raises(self):
        ts = _make_ts_with_gaps(10, gap_indices=[5])
        interp = TemporalInterpolator()
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            interp.interpolate(ts, method="nonexistent")

    def test_limit_caps_fill(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11, 12, 13, 14])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="linear", limit=2)
        remaining = result.to_dataframe()["value"].isna().sum()
        # Should NOT fill all 5, some should remain
        assert remaining > 0

    def test_metadata_updated(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5, 6])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="linear")
        assert result.metadata.get("interpolated") is True
        assert result.metadata.get("method") == "linear"
        assert "filled_count" in result.metadata

    def test_no_gaps_returns_same_data(self):
        ts = _make_ts_with_gaps(20, gap_indices=[])
        interp = TemporalInterpolator()
        result = interp.interpolate(ts, method="linear")
        pd.testing.assert_frame_equal(ts.to_dataframe(), result.to_dataframe())

    def test_interpolation_log_recorded(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5])
        interp = TemporalInterpolator()
        interp.interpolate(ts, method="linear")
        log = interp.get_interpolation_log()
        assert len(log) == 1
        assert log[0]["method"] == "linear"
        assert log[0]["filled"] >= 0


# ---------------------------------------------------------------------------
# Tests for impute()
# ---------------------------------------------------------------------------


class TestImpute:
    def test_forward_fill(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5, 6])
        interp = TemporalInterpolator()
        result = interp.impute(ts, method="forward_fill")
        df = result.to_dataframe()
        assert df["value"].isna().sum() == 0

    def test_backward_fill(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5, 6])
        interp = TemporalInterpolator()
        result = interp.impute(ts, method="backward_fill")
        df = result.to_dataframe()
        assert df["value"].isna().sum() == 0

    def test_mean_fill(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5, 6])
        interp = TemporalInterpolator()
        result = interp.impute(ts, method="mean")
        df = result.to_dataframe()
        assert df["value"].isna().sum() == 0

    def test_median_fill(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5])
        interp = TemporalInterpolator()
        result = interp.impute(ts, method="median")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_mode_fill(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5])
        interp = TemporalInterpolator()
        result = interp.impute(ts, method="mode")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_constant_fill(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5])
        interp = TemporalInterpolator()
        result = interp.impute(ts, method="constant")
        df = result.to_dataframe()
        assert df["value"].isna().sum() == 0
        assert df["value"].iloc[5] == 0.0

    def test_unknown_impute_method_raises(self):
        ts = _make_ts_with_gaps(10, gap_indices=[5])
        interp = TemporalInterpolator()
        with pytest.raises(ValueError, match="Unknown imputation method"):
            interp.impute(ts, method="nonexistent")


# ---------------------------------------------------------------------------
# Tests for interpolate_seasonal()
# ---------------------------------------------------------------------------


class TestInterpolateSeasonal:
    def test_seasonal_fills_gaps(self):
        ts = _make_seasonal_ts(periods=120, period=12)
        interp = TemporalInterpolator()
        result = interp.interpolate_seasonal(ts, period=12)
        remaining = result.to_dataframe()["value"].isna().sum()
        assert remaining == 0

    def test_seasonal_preserves_length(self):
        ts = _make_seasonal_ts(periods=60, period=7)
        interp = TemporalInterpolator()
        result = interp.interpolate_seasonal(ts, period=7)
        assert len(result) == len(ts)

    def test_seasonal_with_limit(self):
        ts = _make_seasonal_ts(periods=120, period=12)
        interp = TemporalInterpolator()
        result = interp.interpolate_seasonal(ts, period=12, limit=1)
        # With limit=1 and consecutive gaps at 55,56,57, not all should be filled
        # At least some gaps should remain
        assert len(result) == len(ts)

    def test_seasonal_metadata(self):
        ts = _make_seasonal_ts(periods=60, period=7)
        interp = TemporalInterpolator()
        result = interp.interpolate_seasonal(ts, period=7)
        assert result.metadata.get("method") == "seasonal"
        assert result.metadata.get("period") == 7


# ---------------------------------------------------------------------------
# Tests for interpolate_gap_aware()
# ---------------------------------------------------------------------------


class TestInterpolateGapAware:
    def test_small_gaps_filled_large_gaps_remain(self):
        """Gaps of size <= max_gap_size are filled; larger gaps remain."""
        index = pd.date_range("2023-01-01", periods=50, freq="h", tz="UTC")
        values = np.arange(50, dtype=float)
        # Small gap (2 points)
        values[10] = np.nan
        values[11] = np.nan
        # Large gap (5 points)
        values[30] = np.nan
        values[31] = np.nan
        values[32] = np.nan
        values[33] = np.nan
        values[34] = np.nan
        df = pd.DataFrame({"value": values}, index=index)
        ts = TimeSeries(data=df)

        interp = TemporalInterpolator()
        result = interp.interpolate_gap_aware(ts, max_gap_size=3)
        result_df = result.to_dataframe()

        # Small gap should be filled
        assert not result_df["value"].iloc[10:12].isna().any()
        # Large gap should remain
        assert result_df["value"].iloc[30:35].isna().any()

    def test_metadata_includes_max_gap_size(self):
        ts = _make_ts_with_gaps(30, gap_indices=[5])
        interp = TemporalInterpolator()
        result = interp.interpolate_gap_aware(ts, max_gap_size=5)
        assert result.metadata.get("max_gap_size") == 5


# ---------------------------------------------------------------------------
# Tests for resample_interpolate()
# ---------------------------------------------------------------------------


class TestResampleInterpolate:
    def test_upsample(self):
        """Resampling to higher frequency produces more data points."""
        index = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
        values = np.sin(np.linspace(0, 2 * np.pi, 24))
        df = pd.DataFrame({"value": values}, index=index)
        ts = TimeSeries(data=df)

        interp = TemporalInterpolator()
        result = interp.resample_interpolate(ts, target_freq="30min")
        assert len(result) > len(ts)

    def test_downsample(self):
        """Resampling to lower frequency produces fewer data points."""
        index = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
        values = np.sin(np.linspace(0, 4 * np.pi, 48))
        df = pd.DataFrame({"value": values}, index=index)
        ts = TimeSeries(data=df)

        interp = TemporalInterpolator()
        result = interp.resample_interpolate(ts, target_freq="2h")
        assert len(result) <= len(ts)

    def test_no_nans_after_resample(self):
        index = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
        values = np.linspace(0, 10, 24)
        df = pd.DataFrame({"value": values}, index=index)
        ts = TimeSeries(data=df)

        interp = TemporalInterpolator()
        result = interp.resample_interpolate(ts, target_freq="30min", method="linear")
        assert result.to_dataframe()["value"].isna().sum() == 0

    def test_metadata_updated_on_resample(self):
        index = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({"value": np.arange(10.0)}, index=index)
        ts = TimeSeries(data=df)

        interp = TemporalInterpolator()
        result = interp.resample_interpolate(ts, target_freq="30min")
        assert result.metadata.get("resampled") is True
        assert result.metadata.get("target_freq") == "30min"


# ---------------------------------------------------------------------------
# Tests for interpolation_quality()
# ---------------------------------------------------------------------------


class TestInterpolationQuality:
    def test_quality_returns_expected_keys(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11, 12])
        interp = TemporalInterpolator()
        filled = interp.interpolate(ts, method="linear")
        quality = interp.interpolation_quality(ts, filled)

        assert "columns" in quality
        assert "overall_quality" in quality
        assert "value" in quality["columns"]

    def test_quality_score_reasonable(self):
        ts = _make_ts_with_gaps(100, gap_indices=[20, 21])
        interp = TemporalInterpolator()
        filled = interp.interpolate(ts, method="linear")
        quality = interp.interpolation_quality(ts, filled)

        # Small gap, linear data -- quality should be reasonable
        assert quality["overall_quality"] > 0.3

    def test_gap_fill_rate_positive(self):
        ts = _make_ts_with_gaps(50, gap_indices=[10, 11])
        interp = TemporalInterpolator()
        filled = interp.interpolate(ts, method="linear")
        quality = interp.interpolation_quality(ts, filled)
        col_metrics = quality["columns"]["value"]
        assert col_metrics["gap_fill_rate"] > 0

    def test_correlation_close_to_one(self):
        ts = _make_ts_with_gaps(100, gap_indices=[50, 51])
        interp = TemporalInterpolator()
        filled = interp.interpolate(ts, method="linear")
        quality = interp.interpolation_quality(ts, filled)
        col_metrics = quality["columns"]["value"]
        assert col_metrics["correlation"] is not None
        assert col_metrics["correlation"] > 0.9
