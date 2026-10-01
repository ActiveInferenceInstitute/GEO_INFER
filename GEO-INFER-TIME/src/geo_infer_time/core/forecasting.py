"""
Forecasting models for GEO-INFER-TIME.

This module provides forecasting capabilities including linear regression,
moving average, ARIMA, and Holt-Winters exponential smoothing for temporal
prediction on TimeSeries objects. The statsmodels fitting itself is shared
with AdvancedForecastingEngine via ``fit_arima_forecast`` and
``fit_exponential_smoothing_forecast`` in ``core.advanced_forecasting``.
"""

import logging
from typing import Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# scikit-learn and statsmodels are declared hard dependencies (pyproject.toml)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import TimeSeries from models
from geo_infer_time.models.timeseries import TimeSeries
from geo_infer_time.core.advanced_forecasting import (
    fit_arima_forecast,
    fit_exponential_smoothing_forecast,
)


class ForecastingEngine:
    """
    Forecasting engine for time series prediction.

    Provides multiple forecasting models including ARIMA, linear regression,
    and simple moving average for temporal prediction.
    """

    def __init__(self) -> None:
        """Initialize the forecasting engine."""

    def forecast_linear(
        self, timeseries: TimeSeries, horizon: int = 10
    ) -> dict[str, Any]:
        """
        Forecast using linear regression.

        Args:
            timeseries: TimeSeries object
            horizon: Number of steps to forecast

        Returns:
            Dictionary with forecast results
        """
        data = timeseries.to_dataframe()
        values = data.iloc[:, 0].dropna().values
        time_points = np.arange(len(values)).reshape(-1, 1)

        # Fit linear model
        model = LinearRegression()
        model.fit(time_points, values)

        # Forecast future points
        future_time_points = np.arange(len(values), len(values) + horizon).reshape(
            -1, 1
        )
        forecast = model.predict(future_time_points)

        # Generate future timestamps
        last_timestamp = timeseries.end_time
        freq = timeseries.frequency or "1D"
        future_timestamps = pd.date_range(
            start=last_timestamp, periods=horizon + 1, freq=freq
        )[1:]

        return {
            "forecast": forecast.tolist(),
            "timestamps": [ts.isoformat() for ts in future_timestamps],
            "model_type": "linear",
            "horizon": horizon,
        }

    def forecast_arima(
        self,
        timeseries: TimeSeries,
        horizon: int = 10,
        order: tuple[int, int, int] = (1, 1, 1),
    ) -> dict[str, Any]:
        """
        Forecast using ARIMA model.

        Args:
            timeseries: TimeSeries object
            horizon: Number of steps to forecast
            order: ARIMA order (p, d, q)

        Returns:
            Dictionary with forecast results
        """
        data = timeseries.to_dataframe()
        values = data.iloc[:, 0].dropna().values

        try:
            result = fit_arima_forecast(values, order=order, forecast_steps=horizon)
            forecast = np.asarray(result["forecast"]).tolist()

            # Generate future timestamps
            last_timestamp = timeseries.end_time
            freq = timeseries.frequency or "1D"
            future_timestamps = pd.date_range(
                start=last_timestamp, periods=horizon + 1, freq=freq
            )[1:]

            return {
                "forecast": forecast,
                "timestamps": [ts.isoformat() for ts in future_timestamps],
                "model_type": "arima",
                "order": order,
                "horizon": horizon,
            }
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            raise

    def forecast_moving_average(
        self, timeseries: TimeSeries, horizon: int = 10, window: int = 5
    ) -> dict[str, Any]:
        """
        Forecast using moving average.

        Args:
            timeseries: TimeSeries object
            horizon: Number of steps to forecast
            window: Moving average window size

        Returns:
            Dictionary with forecast results
        """
        data = timeseries.to_dataframe()
        values = data.iloc[:, 0].dropna().values

        # Use last window values for forecast
        last_values = values[-window:]
        forecast = [np.mean(last_values)] * horizon

        # Generate future timestamps
        last_timestamp = timeseries.end_time
        freq = timeseries.frequency or "1D"
        future_timestamps = pd.date_range(
            start=last_timestamp, periods=horizon + 1, freq=freq
        )[1:]

        return {
            "forecast": forecast,
            "timestamps": [ts.isoformat() for ts in future_timestamps],
            "model_type": "moving_average",
            "window": window,
            "horizon": horizon,
        }

    def forecast_exponential_smoothing(
        self,
        timeseries: Any,
        horizon: int = 10,
        alpha: float = 0.3,
        trend: str | None = None,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
    ) -> dict[str, Any]:
        """
        Forecast using exponential smoothing (Holt-Winters).

        Args:
            timeseries: TimeSeries object
            horizon: Number of steps to forecast
            alpha: Smoothing parameter for level (0-1)
            trend: Trend component ('additive', 'multiplicative', or None)
            seasonal: Seasonal component ('additive', 'multiplicative', or None)
            seasonal_periods: Number of periods in a season

        Returns:
            Dictionary with forecast results
        """
        data = timeseries.to_dataframe()
        values = data.iloc[:, 0].dropna().values

        try:
            result = fit_exponential_smoothing_forecast(
                values,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                alpha=alpha,
                forecast_steps=horizon,
            )
            forecast = np.asarray(result["forecast"]).tolist()

            # Generate future timestamps
            last_timestamp = timeseries.end_time
            freq = timeseries.frequency or "1D"
            future_timestamps = pd.date_range(
                start=last_timestamp, periods=horizon + 1, freq=freq
            )[1:]

            return {
                "forecast": forecast,
                "timestamps": [ts.isoformat() for ts in future_timestamps],
                "model_type": "exponential_smoothing",
                "alpha": alpha,
                "trend": trend,
                "seasonal": seasonal,
                "seasonal_periods": seasonal_periods,
                "horizon": horizon,
            }
        except Exception as e:
            logger.error(f"Exponential smoothing forecasting failed: {e}")
            raise

    def validate_forecast(
        self,
        timeseries: Any,
        forecast_result: dict[str, Any],
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """
        Validate forecast accuracy using time series cross-validation.

        Args:
            timeseries: TimeSeries object
            forecast_result: Forecast result dictionary
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with validation metrics
        """
        data = timeseries.to_dataframe()
        values = data.iloc[:, 0].dropna().values

        # Split data
        split_idx = int(len(values) * (1 - validation_split))
        train_values = values[:split_idx]
        test_values = values[split_idx:]
        horizon = len(test_values)

        if horizon == 0:
            return {"error": "Insufficient data for validation"}

        # Generate forecast for validation period
        # Create temporary time series for training data
        train_dates = data.index[:split_idx]
        train_timeseries = TimeSeries(
            data=train_values,
            timestamps=train_dates,
        )

        # Use the same model type as in forecast_result
        model_type = forecast_result.get("model_type", "linear")

        try:
            if model_type == "arima":
                order = forecast_result.get("order", (1, 1, 1))
                val_forecast = self.forecast_arima(train_timeseries, horizon, order)
            elif model_type == "exponential_smoothing":
                alpha = forecast_result.get("alpha", 0.3)
                trend = forecast_result.get("trend")
                seasonal = forecast_result.get("seasonal")
                seasonal_periods = forecast_result.get("seasonal_periods")
                val_forecast = self.forecast_exponential_smoothing(
                    train_timeseries, horizon, alpha, trend, seasonal, seasonal_periods
                )
            else:
                val_forecast = self.forecast_linear(train_timeseries, horizon)

            val_forecast_values = np.array(val_forecast["forecast"])

            # Calculate validation metrics
            mse = float(mean_squared_error(test_values, val_forecast_values))
            mae = float(mean_absolute_error(test_values, val_forecast_values))
            rmse = float(np.sqrt(mse))
            mape = float(
                np.mean(
                    np.abs((test_values - val_forecast_values) / (test_values + 1e-10))
                )
                * 100
            )

            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "horizon": horizon,
                "model_type": model_type,
            }
        except Exception as e:
            logger.error(f"Forecast validation failed: {e}")
            return {"error": str(e)}
