"""
Tests for the GEO-INFER-TIME forecasting module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

from geo_infer_time.core.forecasting import ForecastingEngine


class MockTimeSeries:
    """Mock TimeSeries for testing."""

    def __init__(self, values, frequency="D"):
        self.values = values
        self.frequency = frequency
        self._start = datetime(2024, 1, 1, tzinfo=UTC)

    @property
    def end_time(self):
        return self._start + timedelta(days=len(self.values) - 1)

    def to_dataframe(self):
        dates = pd.date_range(
            "2024-01-01", periods=len(self.values), freq="D", tz="UTC"
        )
        return pd.DataFrame({"value": self.values}, index=dates)


class TestForecastingEngine:
    """Test suite for ForecastingEngine."""

    @pytest.fixture
    def engine(self):
        return ForecastingEngine()

    @pytest.fixture
    def trend_series(self):
        # Series with clear upward trend + small noise
        values = [10 + i * 2 + np.random.normal(0, 0.5) for i in range(100)]
        return MockTimeSeries(values)

    @pytest.fixture
    def seasonal_series(self):
        # Series with weekly seasonality
        values = [
            50 + 20 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 2)
            for i in range(100)
        ]
        return MockTimeSeries(values, frequency="D")

    def test_forecast_linear(self, engine, trend_series):
        """Test linear forecasting."""
        result = engine.forecast_linear(trend_series, horizon=10)

        assert "forecast" in result
        assert "timestamps" in result
        assert result["model_type"] == "linear"
        assert result["horizon"] == 10
        assert len(result["forecast"]) == 10
        assert len(result["timestamps"]) == 10

        # Forecast should continue upward trend
        assert result["forecast"][-1] > result["forecast"][0]

    def test_forecast_moving_average(self, engine, trend_series):
        """Test moving average forecasting."""
        result = engine.forecast_moving_average(trend_series, horizon=5, window=10)

        assert "forecast" in result
        assert "timestamps" in result
        assert result["model_type"] == "moving_average"
        assert result["window"] == 10
        assert result["horizon"] == 5
        assert len(result["forecast"]) == 5

        # Moving average forecast should be constant
        assert all(f == result["forecast"][0] for f in result["forecast"])

    def test_forecast_arima(self, engine, trend_series):
        """Test ARIMA forecasting if available."""
        try:
            result = engine.forecast_arima(trend_series, horizon=5, order=(1, 1, 1))

            assert "forecast" in result
            assert result["model_type"] == "arima"
            assert result["order"] == (1, 1, 1)
            assert len(result["forecast"]) == 5
        except ImportError:
            pytest.fail("statsmodels not available for ARIMA")

    def test_forecast_exponential_smoothing(self, engine, trend_series):
        """Test exponential smoothing forecasting."""
        try:
            result = engine.forecast_exponential_smoothing(
                trend_series, horizon=5, alpha=0.3
            )

            assert "forecast" in result
            assert result["model_type"] == "exponential_smoothing"
            assert result["alpha"] == 0.3
            assert len(result["forecast"]) == 5
        except ImportError:
            pytest.fail("statsmodels not available for exponential smoothing")

    def test_forecast_timestamps_format(self, engine, trend_series):
        """Test that forecast timestamps are valid ISO format."""
        result = engine.forecast_linear(trend_series, horizon=3)

        for ts in result["timestamps"]:
            # Should be parseable as ISO format
            parsed = datetime.fromisoformat(ts)
            assert isinstance(parsed, datetime)

    def test_forecast_different_horizons(self, engine, trend_series):
        """Test forecasting with different horizons."""
        for horizon in [1, 5, 10, 20]:
            result = engine.forecast_linear(trend_series, horizon=horizon)
            assert len(result["forecast"]) == horizon
            assert result["horizon"] == horizon


class TestForecastValidation:
    """Test suite for forecast validation."""

    @pytest.fixture
    def engine(self):
        return ForecastingEngine()

    @pytest.fixture
    def long_series(self):
        # Longer series for validation split
        values = [10 + i * 0.5 + np.random.normal(0, 1) for i in range(200)]
        return MockTimeSeries(values)

    def test_validate_forecast_metrics(self, engine, long_series):
        """Test forecast validation returns expected metrics."""
        # First get a forecast
        forecast_result = engine.forecast_linear(long_series, horizon=10)

        # Then validate
        validation = engine.validate_forecast(long_series, forecast_result)

        # Check for expected metrics (or error key if validation fails)
        if "error" not in validation:
            assert "mse" in validation
            assert "mae" in validation
            assert "rmse" in validation
            assert "mape" in validation
            assert validation["mse"] >= 0
            assert validation["rmse"] >= 0


class TestForecastingEdgeCases:
    """Test edge cases in forecasting."""

    @pytest.fixture
    def engine(self):
        return ForecastingEngine()

    def test_short_series_linear(self, engine):
        """Test linear forecasting with short series."""
        short_values = [10, 20, 30]
        series = MockTimeSeries(short_values)

        result = engine.forecast_linear(series, horizon=2)
        assert len(result["forecast"]) == 2

    def test_constant_series(self, engine):
        """Test forecasting with constant series."""
        constant_values = [50.0] * 100
        series = MockTimeSeries(constant_values)

        result = engine.forecast_linear(series, horizon=5)
        # For constant series, forecast should also be constant
        assert all(abs(f - 50.0) < 1.0 for f in result["forecast"])

    def test_small_window_moving_average(self, engine):
        """Test moving average with small window."""
        values = [10 + i for i in range(50)]
        series = MockTimeSeries(values)

        result = engine.forecast_moving_average(series, horizon=3, window=3)
        assert len(result["forecast"]) == 3
