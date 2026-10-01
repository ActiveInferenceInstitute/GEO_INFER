"""
Advanced forecasting methods for time series.

Includes ARIMA/SARIMAX, exponential smoothing, and state space models. This
module owns the single statsmodels fitting implementations shared with
``ForecastingEngine``: ``fit_arima_forecast`` and
``fit_exponential_smoothing_forecast``.

statsmodels is a declared hard dependency (see pyproject.toml) and is imported
unconditionally; there is no optional-fallback path.
"""

import logging
import warnings
from typing import Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import ConvergenceWarning

logger = logging.getLogger(__name__)


def fit_arima_forecast(
    values: Any,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal: tuple[int, int, int, int] | None = None,
    forecast_steps: int = 10,
) -> dict[str, Any]:
    """
    Fit an ARIMA (or SARIMAX when a seasonal order is given) model and
    forecast with confidence intervals.

    Shared implementation used by both :class:`ForecastingEngine` and
    :class:`AdvancedForecastingEngine`.

    Args:
        values: Series of observations (array-like or pandas Series)
        order: ARIMA order (p, d, q)
        seasonal: Optional seasonal order (P, D, Q, s)
        forecast_steps: Number of steps to forecast

    Returns:
        Dict with ``forecast``, ``lower_bound``, ``upper_bound``, and the
        fitted ``model``
    """
    if seasonal is not None:
        model = SARIMAX(values, order=order, seasonal_order=seasonal)
    else:
        model = ARIMA(values, order=order)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Non-stationary starting autoregressive parameters found.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Non-invertible starting MA parameters found.*",
                category=UserWarning,
            )
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_steps)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(np.asarray(forecast))
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        if not isinstance(conf_int, pd.DataFrame):
            bounds = np.asarray(conf_int)
            if bounds.ndim == 1:
                bounds = bounds.reshape(-1, 1)
            conf_int = pd.DataFrame({"lower": bounds[:, 0], "upper": bounds[:, 1]})
        return {
            "forecast": forecast,
            "lower_bound": conf_int.iloc[:, 0],
            "upper_bound": conf_int.iloc[:, 1],
            "model": fitted_model,
        }
    except Exception as e:
        logger.error(f"ARIMA fitting failed: {e}")
        raise


def fit_exponential_smoothing_forecast(
    values: Any,
    trend: str | None = None,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    alpha: float | None = None,
    forecast_steps: int = 10,
) -> dict[str, Any]:
    """
    Fit a Holt-Winters exponential smoothing model and forecast.

    Shared implementation used by both forecasting engines.

    Args:
        values: Series of observations
        trend: Trend component ('add', 'mul', or None)
        seasonal: Seasonal component ('add', 'mul', or None)
        seasonal_periods: Number of observations per seasonal cycle
            (required when ``seasonal`` is set)
        alpha: Optional smoothing level; None lets statsmodels optimize it
        forecast_steps: Number of steps to forecast

    Returns:
        Dict with ``forecast`` and the fitted ``model``

    Raises:
        ValueError: If ``seasonal`` is set without ``seasonal_periods``
    """
    if seasonal is not None and seasonal_periods is None:
        raise ValueError(
            "seasonal_periods is required when seasonal smoothing is enabled"
        )
    model = ExponentialSmoothing(
        values,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    try:
        if alpha is not None:
            fitted_model = model.fit(smoothing_level=alpha)
        else:
            fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_steps)
        return {"forecast": forecast, "model": fitted_model}
    except Exception as e:
        logger.error(f"Exponential smoothing fitting failed: {e}")
        raise


class AdvancedForecastingEngine:
    """
    Advanced forecasting engine with multiple methods.
    """

    def __init__(self, config: dict | None = None):
        """Initialize advanced forecasting engine."""
        self.config = config or {}

    def forecast_arima(
        self,
        time_series: pd.Series,
        order: tuple[int, int, int] = (1, 1, 1),
        forecast_steps: int = 10,
        seasonal: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Forecast using ARIMA (or SARIMAX with a seasonal order).

        Args:
            time_series: Time series data
            order: ARIMA order (p, d, q)
            forecast_steps: Number of steps to forecast
            seasonal: Optional seasonal order (P, D, Q, s)

        Returns:
            Forecast results with confidence intervals
        """
        return fit_arima_forecast(
            time_series,
            order=order,
            seasonal=seasonal,
            forecast_steps=forecast_steps,
        )

    def forecast_exponential_smoothing(
        self,
        time_series: pd.Series,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        forecast_steps: int = 10,
    ) -> dict[str, Any]:
        """
        Forecast using exponential smoothing (Holt-Winters).

        Args:
            time_series: Time series data
            trend: Trend type ('add', 'mul', None)
            seasonal: Seasonal type ('add', 'mul', None)
            seasonal_periods: Number of observations per seasonal cycle;
                required when ``seasonal`` is set (previously hardcoded to 12)
            forecast_steps: Number of steps to forecast

        Returns:
            Forecast results

        Raises:
            ValueError: If ``seasonal`` is set without ``seasonal_periods``
        """
        return fit_exponential_smoothing_forecast(
            time_series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            forecast_steps=forecast_steps,
        )

    def detect_trend_seasonality(
        self,
        time_series: pd.Series,
    ) -> dict[str, Any]:
        """
        Detect trend and seasonality in time series.

        Args:
            time_series: Time series data

        Returns:
            Trend and seasonality analysis
        """
        decomposition = seasonal_decompose(
            time_series,
            model="additive",
            period=12 if len(time_series) > 24 else None,
        )

        # Calculate trend strength
        trend_strength = np.var(decomposition.trend.dropna()) / np.var(time_series)

        # Calculate seasonality strength
        seasonal_strength = np.var(decomposition.seasonal.dropna()) / np.var(
            time_series
        )

        return {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
            "trend_strength": float(trend_strength),
            "seasonal_strength": float(seasonal_strength),
            "has_trend": trend_strength > 0.1,
            "has_seasonality": seasonal_strength > 0.1,
        }
