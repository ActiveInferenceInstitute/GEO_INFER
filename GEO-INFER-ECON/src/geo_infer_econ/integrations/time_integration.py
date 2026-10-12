"""
GEO-INFER-TIME Integration Adapter

Provides temporal analysis wrapper for economic time series.
"""

from typing import Any, cast
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import GEO-INFER-TIME modules
try:
    from geo_infer_time.core.analysis import TemporalAnalyzer
    from geo_infer_time.core.forecasting import ForecastingEngine
    from geo_infer_time.models.timeseries import TimeSeries

    TIME_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != "geo_infer_time":
        raise
    TIME_AVAILABLE = False
    TemporalAnalyzer = None
    ForecastingEngine = None
    TimeSeries = None


class TimeIntegration:
    """
    Integration adapter for GEO-INFER-TIME.

    Provides temporal analysis for economic data including:
    - Time series analysis (trends, seasonality, decomposition)
    - Forecasting (economic indicators, growth rates)
    - Temporal alignment and synchronization
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize time integration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        if not TIME_AVAILABLE:
            self.analyzer = None
            self.forecaster = None
        else:
            self.analyzer = TemporalAnalyzer()
            self.forecaster = ForecastingEngine()

    @staticmethod
    def _as_timeseries(time_series: pd.Series) -> "TimeSeries":
        """Convert a pandas series to the current GEO-INFER-TIME model."""
        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")
        return TimeSeries(data=time_series, timestamps=time_series.index)

    def detect_trend(
        self, time_series: pd.Series, method: str = "linear"
    ) -> dict[str, Any] | None:
        """
        Detect trends in economic time series.

        Args:
            time_series: Time series data
            method: Trend detection method ('linear', 'polynomial', etc.)

        Returns:
            Dictionary with trend analysis results or None if unavailable
        """
        if not TIME_AVAILABLE or self.analyzer is None:
            # Fallback to simple linear trend
            try:
                from scipy import stats

                x = np.arange(len(time_series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, time_series.values
                )
                return {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "trend": (
                        "increasing"
                        if slope > 0
                        else "decreasing"
                        if slope < 0
                        else "stable"
                    ),
                }
            except Exception as e:
                logger.error(f"Failed to detect trend: {e}")
                return None

        return cast(
            dict[str, Any],
            self.analyzer.detect_trend(self._as_timeseries(time_series), method=method),
        )

    def analyze_seasonality(
        self, time_series: pd.Series, period: int | None = None
    ) -> dict[str, Any] | None:
        """
        Analyze seasonality in economic time series.

        Args:
            time_series: Time series data
            period: Expected seasonal period (auto-detect if None)

        Returns:
            Dictionary with seasonality analysis results or None if unavailable
        """
        if not TIME_AVAILABLE or self.analyzer is None:
            return None

        timeseries = self._as_timeseries(time_series)
        return cast(
            dict[str, Any],
            self.analyzer.detect_seasonality(timeseries, max_periods=period or 12),
        )

    def decompose_time_series(
        self, time_series: pd.Series, model: str = "additive"
    ) -> dict[str, pd.Series] | None:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            time_series: Time series data
            model: Decomposition model ('additive' or 'multiplicative')

        Returns:
            Dictionary with decomposed components or None if unavailable
        """
        if not TIME_AVAILABLE or self.analyzer is None:
            # Fallback to statsmodels
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose

                decomposition = seasonal_decompose(
                    time_series,
                    model=model,
                    period=(
                        max(2, len(time_series) // 10) if len(time_series) > 10 else 2
                    ),
                )
                return {
                    "trend": decomposition.trend,
                    "seasonal": decomposition.seasonal,
                    "residual": decomposition.resid,
                }
            except Exception as e:
                logger.error(f"Failed to decompose time series: {e}")
                return None

        result = self.analyzer.decompose(self._as_timeseries(time_series), model=model)
        converted: dict[str, pd.Series] = {}
        for name, values in result.items():
            if name in {"trend", "seasonal", "residual"}:
                converted[name] = pd.Series(values, index=time_series.index)
        return converted

    def forecast(
        self, time_series: pd.Series, horizon: int, method: str = "arima", **kwargs: Any
    ) -> dict[str, Any] | None:
        """
        Forecast economic time series.

        Args:
            time_series: Historical time series data
            horizon: Forecast horizon (number of periods ahead)
            method: Forecasting method ('arima', 'exponential_smoothing', etc.)
            **kwargs: Additional parameters for forecasting

        Returns:
            Dictionary with forecast results or None if unavailable
        """
        if not TIME_AVAILABLE or self.forecaster is None:
            return None

        timeseries = self._as_timeseries(time_series)
        methods = {
            "arima": self.forecaster.forecast_arima,
            "linear": self.forecaster.forecast_linear,
            "moving_average": self.forecaster.forecast_moving_average,
            "exponential_smoothing": self.forecaster.forecast_exponential_smoothing,
        }
        if method not in methods:
            raise ValueError(f"Unsupported forecasting method: {method}")
        return cast(
            dict[str, Any], methods[method](timeseries, horizon=horizon, **kwargs)
        )

    def align_time_series(
        self, time_series_list: list[pd.Series], method: str = "interpolate"
    ) -> list[pd.Series] | None:
        """
        Align multiple time series to common time index.

        Args:
            time_series_list: List of time series to align
            method: Alignment method ('interpolate', 'forward_fill', etc.)

        Returns:
            List of aligned time series or None if unavailable
        """
        if not TIME_AVAILABLE or self.analyzer is None:
            # Fallback to pandas alignment
            try:
                # Find common time index
                common_index = time_series_list[0].index
                for ts in time_series_list[1:]:
                    common_index = common_index.union(ts.index)

                aligned = []
                for ts in time_series_list:
                    if method == "interpolate":
                        aligned.append(ts.reindex(common_index).interpolate())
                    elif method == "forward_fill":
                        aligned.append(ts.reindex(common_index).ffill())
                    else:
                        aligned.append(ts.reindex(common_index))

                return aligned
            except Exception as e:
                logger.error(f"Failed to align time series: {e}")
                return None

        common_index = time_series_list[0].index
        for series in time_series_list[1:]:
            common_index = common_index.union(series.index)
        aligned = []
        for series in time_series_list:
            values = series.reindex(common_index)
            if method == "interpolate":
                values = values.interpolate()
            elif method == "forward_fill":
                values = values.ffill()
            elif method != "none":
                raise ValueError(f"Unsupported alignment method: {method}")
            aligned.append(values)
        return aligned

    def is_available(self) -> bool:
        """Check if GEO-INFER-TIME is available."""
        return TIME_AVAILABLE and self.analyzer is not None
