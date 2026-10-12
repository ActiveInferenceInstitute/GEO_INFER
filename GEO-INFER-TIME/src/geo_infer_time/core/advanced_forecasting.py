"""
Advanced forecasting methods for time series.

Includes ARIMA/SARIMAX, exponential smoothing, and state space models. This
module owns the single statsmodels fitting implementations shared with
``ForecastingEngine``: ``fit_arima_forecast`` and
``fit_exponential_smoothing_forecast``.

statsmodels is a declared hard dependency (see pyproject.toml) and is imported
unconditionally; there is no optional-fallback path.
"""

from copy import deepcopy
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
from geo_infer_time.core._validation import (
    finite_vector,
    positive_integer,
    regular_frequency,
)
from geo_infer_time.core.timestamps import normalize_datetime_index

logger = logging.getLogger(__name__)


def fit_arima_forecast(
    values: Any,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal: tuple[int, int, int, int] | None = None,
    forecast_steps: int = 10,
    fit_kwargs: dict[str, Any] | None = None,
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
    positive_integer(forecast_steps, "forecast_steps")
    checked = finite_vector(values)
    if len(checked) < 2:
        raise ValueError("Forecasting requires at least two finite observations")
    if isinstance(values, pd.Series) and isinstance(values.index, pd.DatetimeIndex):
        values = values.copy()
        values.index = normalize_datetime_index(values.index)
        values = values.asfreq(regular_frequency(values.index))
    else:
        values = checked
    if not isinstance(order, tuple) or len(order) != 3:
        raise ValueError("order must contain three non-negative integers")
    for component in order:
        positive_integer(component, "order component", minimum=0)
    if seasonal is not None:
        if not isinstance(seasonal, tuple) or len(seasonal) != 4:
            raise ValueError("seasonal must contain four non-negative integers")
        for component in seasonal:
            positive_integer(component, "seasonal component", minimum=0)
        if seasonal[3] == 1:
            raise ValueError("seasonal period must be zero or at least two")
        model = SARIMAX(values, order=order, seasonal_order=seasonal)
    else:
        model = ARIMA(values, order=order)
    try:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
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
            fitted_model = model.fit(**(fit_kwargs or {}))
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
            "converged": bool(
                getattr(fitted_model, "mle_retvals", {}).get("converged", True)
            ),
            "fit_warnings": [str(item.message) for item in fit_warnings],
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
    fit_kwargs: dict[str, Any] | None = None,
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
    positive_integer(forecast_steps, "forecast_steps")
    checked = finite_vector(values)
    if len(checked) < 2:
        raise ValueError("Forecasting requires at least two finite observations")
    if isinstance(values, pd.Series) and isinstance(values.index, pd.DatetimeIndex):
        values = values.copy()
        values.index = normalize_datetime_index(values.index)
        values = values.asfreq(regular_frequency(values.index))
    else:
        values = checked
    if alpha is not None and (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be finite and between 0 and 1")
    if seasonal_periods is not None:
        positive_integer(seasonal_periods, "seasonal_periods", minimum=2)
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
            fitted_model = model.fit(smoothing_level=alpha, **(fit_kwargs or {}))
        else:
            fitted_model = model.fit(**(fit_kwargs or {}))
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
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        self.config = deepcopy(config or {})
        unknown = set(self.config) - {
            "arima_fit_kwargs",
            "smoothing_fit_kwargs",
            "seasonal_period",
        }
        if unknown:
            raise ValueError(f"Unknown forecasting configuration: {sorted(unknown)}")
        for name in ("arima_fit_kwargs", "smoothing_fit_kwargs"):
            if name in self.config and not isinstance(self.config[name], dict):
                raise TypeError(f"{name} must be a dictionary")
        if "seasonal_period" in self.config:
            positive_integer(
                self.config["seasonal_period"], "seasonal_period", minimum=2
            )

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
            fit_kwargs=self.config.get("arima_fit_kwargs"),
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
            fit_kwargs=self.config.get("smoothing_fit_kwargs"),
        )

    def detect_trend_seasonality(
        self,
        time_series: pd.Series,
        period: int | None = None,
    ) -> dict[str, Any]:
        """
        Detect trend and seasonality in time series.

        Args:
            time_series: Time series data

        Returns:
            Trend and seasonality analysis
        """
        values = finite_vector(time_series)
        if not len(values):
            raise ValueError("Seasonal decomposition requires observations")
        time_series = time_series.copy()
        time_series.index = normalize_datetime_index(time_series.index)
        regular_frequency(time_series.index)
        if period is None:
            period = self.config.get("seasonal_period")
        if period is None:
            from geo_infer_time.core.analysis import TemporalAnalyzer

            period = TemporalAnalyzer._infer_seasonal_period(
                regular_frequency(time_series.index)
            )
        positive_integer(period, "period", minimum=2)
        if len(values) < 2 * period:
            raise ValueError(
                "Seasonal decomposition requires at least two complete cycles"
            )
        decomposition = seasonal_decompose(
            time_series,
            model="additive",
            period=period,
        )

        # Calculate trend strength
        variance = np.var(values)
        trend_strength = (
            np.var(decomposition.trend.dropna()) / variance if variance > 0 else 0.0
        )

        # Calculate seasonality strength
        seasonal_strength = (
            np.var(decomposition.seasonal.dropna()) / variance if variance > 0 else 0.0
        )

        return {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
            "trend_strength": float(trend_strength),
            "seasonal_strength": float(seasonal_strength),
            "has_trend": trend_strength > 0.1,
            "has_seasonality": seasonal_strength > 0.1,
            "period": period,
        }
