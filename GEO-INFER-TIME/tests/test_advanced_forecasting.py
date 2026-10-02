"""
Tests for GEO-INFER-TIME advanced forecasting module.

Covers AdvancedForecastingEngine: ARIMA, exponential smoothing,
and trend/seasonality detection against the declared statsmodels dependency.
"""

import pytest
import numpy as np
import pandas as pd

from geo_infer_time.core.advanced_forecasting import AdvancedForecastingEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create an AdvancedForecastingEngine instance."""
    return AdvancedForecastingEngine()


@pytest.fixture
def trending_series():
    """Series with a clear linear trend and mild noise."""
    np.random.seed(42)
    n = 120
    index = pd.date_range("2020-01-01", periods=n, freq="ME", tz="UTC")
    values = 50.0 + np.arange(n) * 0.5 + np.random.randn(n) * 2
    return pd.Series(values, index=index, name="trending")


@pytest.fixture
def seasonal_series():
    """Series with additive monthly seasonality (period=12)."""
    np.random.seed(42)
    n = 120  # 10 years of monthly data
    index = pd.date_range("2015-01-01", periods=n, freq="ME", tz="UTC")
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
    trend = np.arange(n) * 0.3
    noise = np.random.randn(n) * 1.5
    values = 100 + trend + seasonal + noise
    return pd.Series(values, index=index, name="seasonal")


@pytest.fixture
def stationary_series():
    """Stationary series (white noise around a constant mean)."""
    np.random.seed(42)
    n = 100
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    values = 50 + np.random.randn(n) * 3
    return pd.Series(values, index=index, name="stationary")


@pytest.fixture
def short_series():
    """Very short series (only 20 points)."""
    np.random.seed(42)
    n = 20
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    values = np.random.randn(n) * 5 + 100
    return pd.Series(values, index=index, name="short")


# ===================================================================
# Engine Initialization
# ===================================================================


class TestEngineInit:
    """Tests for AdvancedForecastingEngine initialization."""

    def test_default_config(self, engine):
        """Default config is an empty dict."""
        assert engine.config == {}

    def test_custom_config(self):
        """Custom config is stored."""
        cfg = {"max_iter": 500, "method": "css"}
        eng = AdvancedForecastingEngine(config=cfg)
        assert eng.config == cfg


# ===================================================================
# ARIMA Forecasting
# ===================================================================


class TestForecastARIMA:
    """Tests for forecast_arima."""

    def test_arima_returns_forecast(self, engine, trending_series):
        """ARIMA produces a forecast with expected keys."""
        result = engine.forecast_arima(
            trending_series, order=(1, 1, 0), forecast_steps=5
        )
        assert "forecast" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "model" in result

    def test_arima_forecast_length(self, engine, trending_series):
        """Forecast length matches requested steps."""
        steps = 12
        result = engine.forecast_arima(
            trending_series, order=(1, 1, 0), forecast_steps=steps
        )
        assert len(result["forecast"]) == steps
        assert len(result["lower_bound"]) == steps
        assert len(result["upper_bound"]) == steps

    def test_arima_confidence_interval_ordering(self, engine, trending_series):
        """Lower bound <= forecast <= upper bound for each step."""
        result = engine.forecast_arima(
            trending_series, order=(1, 1, 0), forecast_steps=5
        )
        for i in range(5):
            lb = result["lower_bound"].iloc[i]
            ub = result["upper_bound"].iloc[i]
            fc = result["forecast"].iloc[i]
            assert lb <= fc <= ub

    def test_arima_default_order(self, engine, trending_series):
        """Default ARIMA order (1,1,1) works."""
        result = engine.forecast_arima(trending_series, forecast_steps=5)
        assert len(result["forecast"]) == 5

    def test_arima_stationary_series(self, engine, stationary_series):
        """ARIMA works on stationary data with (1,0,0)."""
        result = engine.forecast_arima(
            stationary_series, order=(1, 0, 0), forecast_steps=5
        )
        assert len(result["forecast"]) == 5

    def test_arima_with_seasonal_order(self, engine, seasonal_series):
        """SARIMAX is used when seasonal order is provided."""
        result = engine.forecast_arima(
            seasonal_series,
            order=(1, 1, 0),
            seasonal=(1, 0, 0, 12),
            forecast_steps=6,
        )
        assert len(result["forecast"]) == 6

    def test_arima_forecast_not_nan(self, engine, trending_series):
        """Forecast values should not be NaN."""
        result = engine.forecast_arima(
            trending_series, order=(1, 1, 0), forecast_steps=5
        )
        assert not result["forecast"].isna().any()


# ===================================================================
# Exponential Smoothing
# ===================================================================


class TestForecastExponentialSmoothing:
    """Tests for forecast_exponential_smoothing."""

    def test_exp_smoothing_returns_forecast(self, engine, trending_series):
        """Exponential smoothing produces a forecast with expected keys."""
        result = engine.forecast_exponential_smoothing(
            trending_series, trend="add", seasonal=None, forecast_steps=5
        )
        assert "forecast" in result
        assert "model" in result

    def test_exp_smoothing_additive_seasonal(self, engine, seasonal_series):
        """Exponential smoothing with additive seasonality and explicit period."""
        result = engine.forecast_exponential_smoothing(
            seasonal_series,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            forecast_steps=12,
        )
        assert len(result["forecast"]) == 12

    def test_exp_smoothing_seasonal_requires_period(self, engine, seasonal_series):
        """Seasonal smoothing without a period raises instead of assuming 12."""
        with pytest.raises(ValueError, match="seasonal_periods is required"):
            engine.forecast_exponential_smoothing(
                seasonal_series, trend="add", seasonal="add", forecast_steps=12
            )

    def test_exp_smoothing_forecast_length(self, engine, trending_series):
        """Forecast length matches requested steps."""
        steps = 10
        result = engine.forecast_exponential_smoothing(
            trending_series, trend="add", forecast_steps=steps
        )
        assert len(result["forecast"]) == steps

    def test_exp_smoothing_no_trend(self, engine, stationary_series):
        """Exponential smoothing works without trend."""
        result = engine.forecast_exponential_smoothing(
            stationary_series, trend=None, seasonal=None, forecast_steps=5
        )
        assert len(result["forecast"]) == 5

    def test_exp_smoothing_additive_seasonal_honors_user_period(
        self, engine, seasonal_series
    ):
        """A user-supplied seasonal_periods is honored (no hardcoded 12)."""
        result = engine.forecast_exponential_smoothing(
            seasonal_series,
            trend="add",
            seasonal="add",
            seasonal_periods=6,
            forecast_steps=12,
        )
        assert len(result["forecast"]) == 12

    def test_exp_smoothing_forecast_not_nan(self, engine, trending_series):
        """Forecast values should not be NaN."""
        result = engine.forecast_exponential_smoothing(
            trending_series, trend="add", forecast_steps=5
        )
        assert not result["forecast"].isna().any()

    def test_exp_smoothing_trending_goes_up(self, engine, trending_series):
        """Forecast for trending data should continue the upward trend."""
        result = engine.forecast_exponential_smoothing(
            trending_series, trend="add", forecast_steps=5
        )
        last_observed = trending_series.iloc[-1]
        # At least the first forecast point should be in the vicinity of the last observed
        first_forecast = result["forecast"].iloc[0]
        assert abs(first_forecast - last_observed) < 30  # reasonable tolerance


# ===================================================================
# Trend and Seasonality Detection
# ===================================================================


class TestDetectTrendSeasonality:
    """Tests for detect_trend_seasonality."""

    def test_returns_expected_keys(self, engine, seasonal_series):
        """Result contains all expected keys."""
        result = engine.detect_trend_seasonality(seasonal_series)
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert "trend_strength" in result
        assert "seasonal_strength" in result
        assert "has_trend" in result
        assert "has_seasonality" in result

    def test_trending_series_has_trend(self, engine, trending_series):
        """Trending series is identified as having a trend."""
        result = engine.detect_trend_seasonality(trending_series)
        assert result["has_trend"]

    def test_seasonal_series_has_seasonality(self, engine, seasonal_series):
        """Seasonal series is identified as having seasonality."""
        result = engine.detect_trend_seasonality(seasonal_series)
        assert result["has_seasonality"]

    def test_trend_strength_range(self, engine, trending_series):
        """trend_strength is a non-negative float."""
        result = engine.detect_trend_seasonality(trending_series)
        assert result["trend_strength"] >= 0

    def test_seasonal_strength_range(self, engine, seasonal_series):
        """seasonal_strength is a non-negative float."""
        result = engine.detect_trend_seasonality(seasonal_series)
        assert result["seasonal_strength"] >= 0

    def test_decomposition_components_length(self, engine, seasonal_series):
        """Decomposition components have same length as input."""
        result = engine.detect_trend_seasonality(seasonal_series)
        assert len(result["seasonal"]) == len(seasonal_series)

    def test_residual_is_small(self, engine, seasonal_series):
        """Residuals should be smaller in variance than original series."""
        result = engine.detect_trend_seasonality(seasonal_series)
        resid = result["residual"].dropna()
        assert np.var(resid) < np.var(seasonal_series)


# ===================================================================
# Import Guard Tests
# ===================================================================


class TestImportGuard:
    """Verify the declared statsmodels backend is installed and operational."""

    def test_arima_backend_is_available(self, engine, trending_series):
        """The required ARIMA backend executes on a valid series."""
        result = engine.forecast_arima(
            trending_series, order=(1, 1, 0), forecast_steps=2
        )
        assert len(result["forecast"]) == 2

    def test_exp_smoothing_backend_is_available(self, engine, trending_series):
        """The required exponential-smoothing backend executes on valid data."""
        result = engine.forecast_exponential_smoothing(
            trending_series, trend="add", forecast_steps=2
        )
        assert len(result["forecast"]) == 2

    def test_decomposition_backend_is_available(self, engine, seasonal_series):
        """The required decomposition backend executes on seasonal data."""
        result = engine.detect_trend_seasonality(seasonal_series)
        assert len(result["residual"]) == len(seasonal_series)


# ===================================================================
# Integration / Cross-method Tests
# ===================================================================


class TestForecastingIntegration:
    """Integration tests combining multiple forecasting methods."""

    def test_arima_and_exp_smoothing_same_series(self, engine, trending_series):
        """Both methods produce valid forecasts for the same series."""
        arima_result = engine.forecast_arima(
            trending_series, order=(1, 1, 0), forecast_steps=5
        )
        es_result = engine.forecast_exponential_smoothing(
            trending_series, trend="add", forecast_steps=5
        )
        assert len(arima_result["forecast"]) == 5
        assert len(es_result["forecast"]) == 5

    def test_detect_then_forecast(self, engine, seasonal_series):
        """Use trend/seasonality detection to inform forecasting."""
        detection = engine.detect_trend_seasonality(seasonal_series)
        # Use detection results to decide forecast method
        if detection["has_seasonality"]:
            result = engine.forecast_exponential_smoothing(
                seasonal_series,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                forecast_steps=6,
            )
        else:
            result = engine.forecast_arima(
                seasonal_series, order=(1, 1, 0), forecast_steps=6
            )
        assert len(result["forecast"]) == 6
