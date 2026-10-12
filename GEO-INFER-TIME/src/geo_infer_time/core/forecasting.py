"""Forecast complete, regularly sampled univariate TIME data.

Missing observations and irregular cadences require explicit interpolation or
resampling before fitting; they are never dropped or assigned invented dates.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from geo_infer_time.core._validation import (
    positive_integer,
    regular_frequency,
    root_mean_square,
    univariate_series,
)
from geo_infer_time.core.advanced_forecasting import (
    fit_arima_forecast,
    fit_exponential_smoothing_forecast,
)
from geo_infer_time.models.timeseries import TimeSeries


class ForecastingEngine:
    """Forecast one explicitly selected column on its existing regular cadence."""

    @staticmethod
    def _input(timeseries: TimeSeries, horizon: int) -> tuple[pd.Series, list[str]]:
        positive_integer(horizon, "horizon")
        series = univariate_series(timeseries, minimum=2)
        frequency = regular_frequency(series.index)
        future = pd.date_range(series.index[-1], periods=horizon + 1, freq=frequency)[
            1:
        ]
        return series, [stamp.isoformat() for stamp in future]

    def forecast_linear(
        self, timeseries: TimeSeries, horizon: int = 10
    ) -> dict[str, Any]:
        """Fit linear change per sample and extrapolate ``horizon`` model steps."""
        series, timestamps = self._input(timeseries, horizon)
        positions = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(positions, series.to_numpy())
        future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        return {
            "forecast": model.predict(future).tolist(),
            "timestamps": timestamps,
            "model_type": "linear",
            "horizon": horizon,
        }

    def forecast_arima(
        self,
        timeseries: TimeSeries,
        horizon: int = 10,
        order: tuple[int, int, int] = (1, 1, 1),
    ) -> dict[str, Any]:
        """Fit the declared statsmodels ARIMA backend, retaining diagnostics."""
        series, timestamps = self._input(timeseries, horizon)
        result = fit_arima_forecast(series, order=order, forecast_steps=horizon)
        return {
            "forecast": np.asarray(result["forecast"]).tolist(),
            "lower_bound": np.asarray(result["lower_bound"]).tolist(),
            "upper_bound": np.asarray(result["upper_bound"]).tolist(),
            "timestamps": timestamps,
            "model_type": "arima",
            "order": order,
            "horizon": horizon,
            "converged": result["converged"],
            "fit_warnings": result["fit_warnings"],
        }

    def forecast_moving_average(
        self, timeseries: TimeSeries, horizon: int = 10, window: int = 5
    ) -> dict[str, Any]:
        """Repeat the mean of the last ``window`` available model steps."""
        positive_integer(window, "window")
        series, timestamps = self._input(timeseries, horizon)
        return {
            "forecast": [float(series.iloc[-window:].mean())] * horizon,
            "timestamps": timestamps,
            "model_type": "moving_average",
            "window": window,
            "horizon": horizon,
        }

    def forecast_exponential_smoothing(
        self,
        timeseries: TimeSeries,
        horizon: int = 10,
        alpha: float = 0.3,
        trend: str | None = None,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
    ) -> dict[str, Any]:
        """Fit Holt-Winters with explicit level, trend, and seasonal settings."""
        series, timestamps = self._input(timeseries, horizon)
        result = fit_exponential_smoothing_forecast(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            alpha=alpha,
            forecast_steps=horizon,
        )
        return {
            "forecast": np.asarray(result["forecast"]).tolist(),
            "timestamps": timestamps,
            "model_type": "exponential_smoothing",
            "alpha": alpha,
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "horizon": horizon,
        }

    def validate_forecast(
        self,
        timeseries: TimeSeries,
        forecast_result: dict[str, Any],
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """Refit the same model on the prefix and score the untouched holdout.

        This performs one chronological holdout, rather than rolling-origin
        cross-validation. Setup and backend failures propagate to the caller.
        MAPE is undefined for a holdout containing zero and is returned as None.
        """
        if (
            isinstance(validation_split, bool)
            or not isinstance(validation_split, (int, float))
            or not np.isfinite(validation_split)
            or not 0 < validation_split < 1
        ):
            raise ValueError(
                "validation_split must be finite and strictly between 0 and 1"
            )
        series = univariate_series(timeseries, minimum=3)
        regular_frequency(series.index)
        split = int(len(series) * (1 - validation_split))
        if split < 2 or split == len(series):
            raise ValueError(
                "Holdout requires at least two training and one test observation"
            )
        train = TimeSeries(series.iloc[:split])
        actual = series.iloc[split:].to_numpy()
        horizon = len(actual)
        if not isinstance(forecast_result, dict):
            raise TypeError("forecast_result must be a dictionary with model_type")
        model_type = forecast_result.get("model_type")
        if model_type == "linear":
            result = self.forecast_linear(train, horizon)
        elif model_type == "moving_average":
            result = self.forecast_moving_average(
                train, horizon, window=forecast_result.get("window", 5)
            )
        elif model_type == "arima":
            result = self.forecast_arima(
                train, horizon, order=forecast_result.get("order", (1, 1, 1))
            )
        elif model_type == "exponential_smoothing":
            result = self.forecast_exponential_smoothing(
                train,
                horizon,
                alpha=forecast_result.get("alpha", 0.3),
                trend=forecast_result.get("trend"),
                seasonal=forecast_result.get("seasonal"),
                seasonal_periods=forecast_result.get("seasonal_periods"),
            )
        else:
            raise ValueError(f"Unknown forecast model_type: {model_type}")
        predicted = np.asarray(result["forecast"])
        mse = float(mean_squared_error(actual, predicted))
        return {
            "mse": mse,
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": root_mean_square(actual - predicted),
            "mape": float(np.mean(np.abs((actual - predicted) / actual)) * 100)
            if np.all(actual != 0)
            else None,
            "horizon": horizon,
            "model_type": model_type,
        }
