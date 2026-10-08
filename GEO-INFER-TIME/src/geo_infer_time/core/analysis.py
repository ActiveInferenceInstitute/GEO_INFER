"""
Temporal analysis for GEO-INFER-TIME.

This module provides time series analysis including trend detection,
seasonality analysis, decomposition, and statistical analysis.
"""

import logging
import math
from typing import Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from ..models.timeseries import TimeSeries
from geo_infer_time.core._validation import (
    finite_vector,
    paired_series,
    positive_integer,
    regular_frequency,
    root_mean_square,
    univariate_series,
)


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies."""

    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"


@dataclass
class Anomaly:
    """Detected anomaly."""

    index: int
    timestamp: str
    value: float
    expected_value: float
    deviation: float
    anomaly_type: AnomalyType
    severity: str


class TemporalAnalyzer:
    """
    Temporal analyzer for time series data.

    Provides comprehensive temporal analysis including trend detection,
    seasonality analysis, decomposition, and statistical tests.
    """

    def __init__(self) -> None:
        """Initialize the temporal analyzer."""

    def detect_trend(
        self, timeseries: TimeSeries, method: str = "linear"
    ) -> dict[str, Any]:
        """
        Detect trend in time series.

        Args:
            timeseries: TimeSeries object
            method: Trend detection method ('linear', 'polynomial', 'moving_average')

        Returns:
            Dictionary with trend information. Linear ``slope_per_sample``
            and ``trend_strength`` describe change per observation;
            ``r_squared`` measures fit quality. Numerical constants are stable.
        """
        if method not in {"linear", "polynomial", "moving_average"}:
            raise ValueError(f"Unknown trend detection method: {method}")
        series = univariate_series(timeseries, minimum=2)
        values = series.to_numpy()
        time_points = np.arange(len(values))

        if method == "linear":
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(
                    "Linear trend requires at least two finite observations"
                )
            # Centering avoids a large constant offset contaminating the slope.
            centered_time = time_points - time_points.mean()
            scale = max(float(np.abs(values).max()), np.finfo(float).tiny)
            scaled_values = values / scale
            scaled_mean = scaled_values.mean()
            centered_values = scaled_values - scaled_mean
            scaled_slope = float(
                np.dot(centered_time, centered_values)
                / np.dot(centered_time, centered_time)
            )
            if abs(scaled_slope) * (len(values) - 1) <= 8 * np.finfo(float).eps:
                scaled_slope = 0.0
                slope = 0.0
                trend_direction = "stable"
            else:
                slope = scaled_slope * scale
                trend_direction = "increasing" if slope > 0 else "decreasing"
            scaled_trend = scaled_mean + scaled_slope * centered_time
            trend_line = scale * scaled_trend
            total_variation = float(np.dot(centered_values, centered_values))
            residuals = scaled_values - scaled_trend
            r_squared = (
                max(0.0, 1 - float(np.dot(residuals, residuals)) / total_variation)
                if total_variation
                else 1.0
            )
            trend_strength = abs(slope)

        elif method == "polynomial":
            if len(values) < 3:
                raise ValueError(
                    "Polynomial trend requires at least three observations"
                )
            # Polynomial trend
            coeffs = np.polyfit(time_points, values, 2)
            trend_line = np.polyval(coeffs, time_points)
            trend_direction = "non-linear"
            trend_strength = (
                np.std(trend_line) / np.std(values) if np.std(values) else 0.0
            )

        elif method == "moving_average":
            # Moving average trend
            window = min(len(values), max(2, min(30, len(values) // 10)))
            trend_line = (
                pd.Series(values)
                .rolling(window=window, center=True, min_periods=1)
                .mean()
                .values
            )
            trend_direction = "variable"
            trend_strength = (
                np.corrcoef(values, trend_line)[0, 1]
                if np.std(values) and np.std(trend_line)
                else 0.0
            )

        else:
            raise ValueError(f"Unknown trend detection method: {method}")

        result = {
            "method": method,
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "trend_values": trend_line.tolist(),
        }
        if method == "linear":
            result.update(slope_per_sample=slope, r_squared=r_squared)
        return result

    def detect_seasonality(
        self, timeseries: TimeSeries, max_periods: int = 12
    ) -> dict[str, Any]:
        """
        Detect seasonality in time series.

        Args:
            timeseries: TimeSeries object
            max_periods: Maximum period to check for seasonality

        Returns:
            Dictionary with seasonality information
        """
        positive_integer(max_periods, "max_periods", minimum=2)
        series = univariate_series(timeseries, minimum=2)
        regular_frequency(series.index)
        values = series.to_numpy()

        # Use autocorrelation to detect seasonality
        autocorr = []
        for lag in range(2, min(max_periods, len(values) // 2) + 1):
            if len(values) > lag:
                if np.std(values[:-lag]) and np.std(values[lag:]):
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    autocorr.append({"lag": lag, "correlation": float(corr)})

        # Find strongest seasonal pattern
        if autocorr:
            strongest = max(autocorr, key=lambda x: x["correlation"])
            period = strongest["lag"]
            strength = max(0.0, strongest["correlation"])
        else:
            period = None
            strength = 0.0

        return {
            "has_seasonality": strength > 0.5,
            "period": period,
            "strength": strength,
            "autocorrelations": autocorr,
        }

    @staticmethod
    def _infer_seasonal_period(frequency: str | pd.DateOffset | None) -> int:
        """
        Infer a seasonal period in samples from a pandas frequency alias.

        Only unambiguous standard frequencies are mapped: hourly -> 24
        (pro-rated for hourly multiples such as '2h' -> 12), daily -> 7,
        business-daily -> 5, weekly -> 52, monthly -> 12, quarterly -> 4.
        Sub-daily sample frequencies (minutes/seconds), annual frequencies,
        and unknown aliases cannot be mapped to a sample count without extra
        assumptions and are rejected.

        Raises:
            ValueError: When no period can be inferred for the frequency
        """
        if not frequency:
            raise ValueError(
                "seasonal period cannot be inferred without a frequency; "
                "pass period explicitly"
            )
        offset = pd.tseries.frequencies.to_offset(frequency)
        if offset is None:
            raise ValueError(f"unknown frequency {frequency!r}; pass period explicitly")
        base = offset.name
        if base == "h":
            if offset.n > 1 and 24 % offset.n != 0:
                raise ValueError(
                    f"cannot infer a seasonal period from frequency "
                    f"{frequency!r}; pass period explicitly"
                )
            return 24 // offset.n
        if offset.n != 1:
            raise ValueError(
                f"cannot infer a seasonal period from frequency {frequency!r}; "
                "pass period explicitly"
            )
        if base == "D":
            return 7
        if base == "B":
            return 5
        if base.startswith("W"):
            return 52
        if base in {"ME", "MS", "BME", "BMS"}:
            return 12
        if base.startswith(("Q", "BQ")):
            return 4
        raise ValueError(
            f"cannot infer a seasonal period from frequency {frequency!r} "
            f"(canonical alias {base!r}); pass period explicitly"
        )

    def decompose(
        self,
        timeseries: TimeSeries,
        model: str = "additive",
        period: int | None = None,
    ) -> dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            timeseries: TimeSeries object
            model: Decomposition model ('additive', 'multiplicative')
            period: Seasonal period in samples. When omitted, it is inferred
                from the series frequency (hourly -> 24, daily -> 7,
                weekly -> 52, monthly -> 12, quarterly -> 4). Frequencies
                that cannot be mapped unambiguously raise ValueError.

        Returns:
            Dictionary with decomposition components

        Raises:
            ValueError: If no period is given and none can be inferred, or if
                the period is unsuitable for the series length
        """
        series = univariate_series(timeseries, minimum=4)
        frequency = regular_frequency(series.index)

        if period is None:
            period = self._infer_seasonal_period(frequency)

        positive_integer(period, "period", minimum=2)

        if period < 2 or period > len(series) // 2:
            raise ValueError(
                f"seasonal period {period} is invalid for a series of length "
                f"{len(series)}; it must be >= 2 and <= half the length"
            )

        decomposition = seasonal_decompose(
            series, model=model, period=period, extrapolate_trend="freq"
        )

        return {
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.dropna().tolist(),
            "residual": decomposition.resid.dropna().tolist(),
            "model": model,
            "period": period,
        }

    def test_stationarity(self, timeseries: TimeSeries) -> dict[str, Any]:
        """
        Test time series stationarity using Augmented Dickey-Fuller test.

        Args:
            timeseries: TimeSeries object

        Returns:
            Dictionary with stationarity test results
        """
        series = univariate_series(timeseries, minimum=2)
        regular_frequency(series.index)
        values = series.to_numpy()

        try:
            result = adfuller(values)

            return {
                "is_stationary": result[1] < 0.05,  # p-value < 0.05
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "critical_values": {k: float(v) for k, v in result[4].items()},
            }
        except ValueError as e:
            logger.error(f"Stationarity test failed: {e}")
            return {
                "is_stationary": False,
                "error": str(e),
            }

    def detect_anomalies(
        self,
        timeseries: TimeSeries,
        method: str = "zscore",
        threshold: float = 3.0,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Detect anomalies in time series.

        Args:
            timeseries: TimeSeries object
            method: Detection method ('zscore', 'iqr', 'rolling_zscore', 'isolation')
            threshold: Threshold for anomaly detection
            window_size: Window size for rolling methods

        Returns:
            Anomaly detection results
        """
        if method not in {"zscore", "iqr", "rolling_zscore", "isolation"}:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError("threshold must be finite and non-negative")
        if window_size is not None:
            positive_integer(window_size, "window_size", minimum=2)
        series = univariate_series(timeseries)
        values = series.to_numpy()
        timestamps = [stamp.isoformat() for stamp in series.index]

        anomalies = []

        if method == "zscore":
            mean = np.mean(values)
            std = np.std(values)
            z_scores = (values - mean) / std if std > 0 else np.zeros(len(values))

            for i, (z, val) in enumerate(zip(z_scores, values)):
                if abs(z) > threshold:
                    severity = "high" if abs(z) > threshold * 1.5 else "medium"
                    anomalies.append(
                        Anomaly(
                            index=i,
                            timestamp=timestamps[i] if i < len(timestamps) else str(i),
                            value=float(val),
                            expected_value=float(mean),
                            deviation=float(z),
                            anomaly_type=AnomalyType.POINT,
                            severity=severity,
                        )
                    )

        elif method == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr

            for i, val in enumerate(values):
                if val < lower or val > upper:
                    deviation = (val - (q1 + q3) / 2) / (iqr + 1e-10)
                    severity = "high" if abs(deviation) > 3 else "medium"
                    anomalies.append(
                        Anomaly(
                            index=i,
                            timestamp=timestamps[i] if i < len(timestamps) else str(i),
                            value=float(val),
                            expected_value=float((q1 + q3) / 2),
                            deviation=float(deviation),
                            anomaly_type=AnomalyType.POINT,
                            severity=severity,
                        )
                    )

        elif method == "rolling_zscore":
            window = window_size or max(10, len(values) // 20)
            series = pd.Series(values)
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()

            z_scores = (series - rolling_mean) / rolling_std.where(rolling_std > 0)

            for i, (z, val) in enumerate(zip(z_scores.values, values)):
                if not np.isnan(z) and abs(z) > threshold:
                    severity = "high" if abs(z) > threshold * 1.5 else "medium"
                    anomalies.append(
                        Anomaly(
                            index=i,
                            timestamp=timestamps[i] if i < len(timestamps) else str(i),
                            value=float(val),
                            expected_value=float(rolling_mean.iloc[i]),
                            deviation=float(z),
                            anomaly_type=AnomalyType.CONTEXTUAL,
                            severity=severity,
                        )
                    )

        elif method == "isolation":
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(contamination=0.05, random_state=42)
            predictions = model.fit_predict(values.reshape(-1, 1))
            scores = model.decision_function(values.reshape(-1, 1))
            for i in np.flatnonzero(predictions == -1):
                anomalies.append(
                    Anomaly(
                        index=int(i),
                        timestamp=timestamps[i],
                        value=float(values[i]),
                        expected_value=float(np.median(values)),
                        deviation=float(scores[i]),
                        anomaly_type=AnomalyType.POINT,
                        severity="medium",
                    )
                )

        return {
            "method": method,
            "threshold": threshold,
            "total_points": len(values),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(values) * 100,
            "anomalies": [
                {
                    "index": a.index,
                    "timestamp": a.timestamp,
                    "value": a.value,
                    "expected": a.expected_value,
                    "deviation": a.deviation,
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                }
                for a in anomalies
            ],
        }

    def detect_change_points(
        self,
        timeseries: TimeSeries,
        method: str = "cusum",
        min_segment_length: int = 10,
    ) -> dict[str, Any]:
        """
        Detect change points (structural breaks) in time series.

        Args:
            timeseries: TimeSeries object
            method: Detection method ('cusum', 'binary_segmentation')
            min_segment_length: Minimum segment length between change points

        Returns:
            Change point detection results
        """
        positive_integer(min_segment_length, "min_segment_length", minimum=2)
        values = univariate_series(timeseries).to_numpy()
        n = len(values)

        change_points: list[dict[str, Any]] = []

        if method == "cusum":
            # CUSUM-based change point detection
            mean = np.mean(values)
            cumsum = np.cumsum(values - mean)

            # Precompute prefix sums so segment means are O(1) per candidate
            # index instead of O(n) — the scan below stays O(n) overall.
            prefix = np.concatenate(([0.0], np.cumsum(values)))
            total = prefix[n]
            std = np.std(values)

            # Find points where CUSUM changes significantly
            for i in range(min_segment_length, n - min_segment_length + 1):
                left_mean = prefix[i] / i
                right_mean = (total - prefix[i]) / (n - i)
                diff = abs(right_mean - left_mean)

                # Threshold based on overall std
                threshold = 0.5 * std
                if diff > threshold:
                    # Check if this is a local maximum in cumsum
                    window = min_segment_length // 2
                    local_max = max(
                        abs(cumsum[max(0, i - window) : min(n, i + window)])
                    )
                    if abs(cumsum[i]) >= local_max * 0.9:
                        # Avoid duplicates
                        if (
                            not change_points
                            or i - change_points[-1]["index"] >= min_segment_length
                        ):
                            change_points.append(
                                {
                                    "index": i,
                                    "mean_before": float(left_mean),
                                    "mean_after": float(right_mean),
                                    "magnitude": float(diff),
                                }
                            )

        elif method == "binary_segmentation":
            # Simplified binary segmentation
            def find_change_point(start: int, end: int) -> dict[str, Any] | None:
                if end - start < 2 * min_segment_length:
                    return None

                best_cost = float("inf")
                best_idx = None
                segment = values[start:end]

                for i in range(
                    min_segment_length, len(segment) - min_segment_length + 1
                ):
                    left = segment[:i]
                    right = segment[i:]
                    cost = np.var(left) * len(left) + np.var(right) * len(right)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = start + i

                if best_idx:
                    # Check if significant
                    left_mean = np.mean(values[start:best_idx])
                    right_mean = np.mean(values[best_idx:end])
                    if abs(right_mean - left_mean) > 0.3 * np.std(values):
                        return {
                            "index": best_idx,
                            "mean_before": float(left_mean),
                            "mean_after": float(right_mean),
                            "magnitude": float(abs(right_mean - left_mean)),
                        }
                return None

            # Iterative binary segmentation
            segments = [(0, n)]
            while segments:
                start, end = segments.pop(0)
                cp = find_change_point(start, end)
                if cp:
                    change_points.append(cp)
                    segments.append((start, cp["index"]))
                    segments.append((cp["index"], end))
                    if len(change_points) >= 10:  # Limit
                        break
        else:
            raise ValueError(
                f"geo_infer_time.detect_change_points: unknown method {method!r}; "
                "supported methods are 'cusum' and 'binary_segmentation'"
            )

        # Sort by index
        change_points.sort(key=lambda x: x["index"])

        return {
            "method": method,
            "series_length": n,
            "change_points_detected": len(change_points),
            "change_points": change_points,
            "segments": len(change_points) + 1,
        }

    def calculate_cross_correlation(
        self, timeseries1: TimeSeries, timeseries2: TimeSeries, max_lag: int = 20
    ) -> dict[str, Any]:
        """
        Calculate cross-correlation between two time series.

        Args:
            timeseries1: First time series
            timeseries2: Second time series
            max_lag: Maximum lag to compute

        Returns:
            Cross-correlation analysis
        """
        positive_integer(max_lag, "max_lag", minimum=0)
        first, second = paired_series(timeseries1, timeseries2)
        data1, data2 = first.to_numpy(), second.to_numpy()
        min_len = len(data1)
        max_lag = min(max_lag, min_len - 2)
        if np.std(data1) == 0 or np.std(data2) == 0:
            raise ValueError("Cross-correlation requires nonconstant paired series")

        # Calculate cross-correlation
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(data1[-lag:], data2[:lag])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(data1[:-lag], data2[lag:])[0, 1]
            else:
                corr = np.corrcoef(data1, data2)[0, 1]

            if not np.isnan(corr):
                correlations.append({"lag": lag, "correlation": float(corr)})

        # Find peak correlation
        if correlations:
            peak = max(correlations, key=lambda x: abs(x["correlation"]))
        else:
            peak = {"lag": 0, "correlation": 0}

        return {
            "max_lag": max_lag,
            "series_length": min_len,
            "correlations": correlations,
            "peak_correlation": {
                "lag": peak["lag"],
                "correlation": peak["correlation"],
                "interpretation": (
                    f"Series 2 leads by {abs(peak['lag'])} periods"
                    if peak["lag"] < 0
                    else f"Series 1 leads by {peak['lag']} periods"
                    if peak["lag"] > 0
                    else "Series are synchronous"
                ),
            },
            "zero_lag_correlation": float(np.corrcoef(data1, data2)[0, 1]),
        }

    def validate_forecast(
        self,
        actual: list[float],
        predicted: list[float],
        confidence_intervals: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """
        Validate forecast accuracy with multiple metrics.

        Args:
            actual: Actual observed values
            predicted: Predicted values
            confidence_intervals: Optional (lower, upper) confidence bounds

        Returns:
            Forecast validation metrics
        """
        actual_arr = finite_vector(actual, "actual")
        predicted_arr = finite_vector(predicted, "predicted")
        if not len(actual_arr) or len(actual_arr) != len(predicted_arr):
            raise ValueError("actual and predicted must have equal nonzero lengths")
        if confidence_intervals is not None:
            bounds = np.asarray(confidence_intervals, dtype=float)
            if (
                bounds.shape != (len(actual_arr), 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
            ):
                raise ValueError(
                    "confidence_intervals require ordered finite bounds for every observation"
                )

        n = len(actual_arr)
        errors = actual_arr - predicted_arr
        abs_errors = np.abs(errors)

        # Core metrics
        mae = float(np.mean(abs_errors))
        mse = float(np.mean(errors**2))
        rmse = root_mean_square(errors)
        mape = (
            float(np.mean(np.abs(errors / actual_arr)) * 100)
            if np.all(actual_arr != 0)
            else None
        )

        # Symmetric MAPE
        denominator = np.abs(actual_arr) + np.abs(predicted_arr)
        smape = float(
            np.mean(
                np.divide(
                    2 * abs_errors, denominator, out=np.zeros(n), where=denominator != 0
                )
            )
            * 100
        )

        # Directional accuracy
        if len(actual_arr) > 1:
            actual_direction = np.sign(np.diff(actual_arr))
            predicted_direction = np.sign(np.diff(predicted_arr))
            directional_accuracy = float(
                np.mean(actual_direction == predicted_direction) * 100
            )
        else:
            directional_accuracy = None

        # Confidence interval coverage
        if confidence_intervals is not None:
            coverage = (
                sum(
                    1
                    for a, (low, high) in zip(actual_arr, confidence_intervals)
                    if low <= a <= high
                )
                / n
                * 100
            )
        else:
            coverage = None

        # Theil's U statistic
        naive_errors = np.abs(np.diff(actual_arr))
        if len(naive_errors) > 0 and np.mean(naive_errors) > 0:
            theil_u = rmse / root_mean_square(naive_errors)
        else:
            theil_u = None

        return {
            "sample_size": n,
            "metrics": {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "smape": smape,
                "theil_u": theil_u,
                "directional_accuracy": directional_accuracy,
                "confidence_coverage": coverage,
            },
            "interpretation": {
                "rmse_vs_std": rmse / root_mean_square(actual_arr - np.mean(actual_arr))
                if root_mean_square(actual_arr - np.mean(actual_arr)) > 0
                else None,
                "forecast_quality": (
                    "Undefined for zero actual observations"
                    if mape is None
                    else "Excellent"
                    if mape < 10
                    else "Good"
                    if mape < 20
                    else "Acceptable"
                    if mape < 30
                    else "Poor"
                ),
            },
            "residuals": {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
            },
        }

    def calculate_autocorrelation(
        self, timeseries: TimeSeries, max_lag: int = 40
    ) -> dict[str, Any]:
        """
        Calculate autocorrelation function.

        Args:
            timeseries: TimeSeries object
            max_lag: Maximum lag to compute

        Returns:
            Autocorrelation analysis
        """
        positive_integer(max_lag, "max_lag", minimum=0)
        series = univariate_series(timeseries, minimum=2)
        regular_frequency(series.index)
        values = series.to_numpy()
        if np.std(values) == 0:
            raise ValueError("Autocorrelation requires nonconstant observations")

        # Calculate ACF
        nlags = min(max_lag, len(values) // 2)
        acf_values = acf(values, nlags=nlags, fft=True)

        # Find significant lags
        n = len(values)
        conf_bound = 1.96 / np.sqrt(n)

        significant_lags = [
            {"lag": i, "acf": float(a), "significant": abs(a) > conf_bound}
            for i, a in enumerate(acf_values)
            if i > 0
        ]

        # Detect periodicity from ACF peaks
        peaks = []
        for i in range(1, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                if acf_values[i] > conf_bound:
                    peaks.append({"lag": i, "acf": float(acf_values[i])})

        return {
            "max_lag": nlags,
            "confidence_bound": float(conf_bound),
            "acf_values": [float(a) for a in acf_values],
            "significant_lags": [lag for lag in significant_lags if lag["significant"]],
            "detected_periods": peaks[:5] if peaks else [],
            "summary": {
                "first_significant_lag": next(
                    (lag["lag"] for lag in significant_lags if lag["significant"]), None
                ),
                "number_significant": sum(
                    1 for lag in significant_lags if lag["significant"]
                ),
            },
        }

    def calculate_rolling_statistics(
        self,
        timeseries: TimeSeries,
        window: int = 10,
        statistics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate rolling statistics over a time series.

        Args:
            timeseries: TimeSeries object
            window: Rolling window size
            statistics: List of statistics to calculate
                       ('mean', 'std', 'min', 'max', 'var', 'sum', 'median')
                       If None, calculates all.

        Returns:
            Dictionary with rolling statistics for each requested statistic
        """
        positive_integer(window, "window")
        data = timeseries.to_dataframe()
        if data.empty or data.shape[1] != 1:
            raise ValueError("Rolling statistics require a nonempty univariate series")
        values = data.iloc[:, 0]

        available_stats = {
            "mean": lambda s: s.rolling(window=window).mean(),
            "std": lambda s: s.rolling(window=window).std(),
            "var": lambda s: s.rolling(window=window).var(),
            "min": lambda s: s.rolling(window=window).min(),
            "max": lambda s: s.rolling(window=window).max(),
            "sum": lambda s: s.rolling(window=window).sum(),
            "median": lambda s: s.rolling(window=window).median(),
        }

        if statistics is None:
            statistics = list(available_stats.keys())
        if any(name not in available_stats for name in statistics):
            raise ValueError("Unknown rolling statistic")

        results = {}
        for stat_name in statistics:
            if stat_name in available_stats:
                stat_values = available_stats[stat_name](values)
                results[stat_name] = {
                    "values": stat_values.dropna().tolist(),
                    "timestamps": [
                        stamp.isoformat() for stamp in stat_values.dropna().index
                    ],
                    "latest": float(stat_values.iloc[-1])
                    if not pd.isna(stat_values.iloc[-1])
                    else None,
                }
            else:
                logger.warning(f"Unknown statistic: {stat_name}")

        # Calculate Bollinger Bands if we have mean and std
        if "mean" in results and "std" in results:
            mean_vals = pd.Series(values).rolling(window=window).mean()
            std_vals = pd.Series(values).rolling(window=window).std()
            results["bollinger_upper"] = {
                "values": (mean_vals + 2 * std_vals).dropna().tolist(),
                "timestamps": [
                    stamp.isoformat()
                    for stamp in (mean_vals + 2 * std_vals).dropna().index
                ],
            }
            results["bollinger_lower"] = {
                "values": (mean_vals - 2 * std_vals).dropna().tolist(),
                "timestamps": [
                    stamp.isoformat()
                    for stamp in (mean_vals - 2 * std_vals).dropna().index
                ],
            }

        return {
            "window": window,
            "series_length": len(values),
            "statistics": results,
            "summary": {
                "statistics_calculated": list(results.keys()),
                "valid_observations": int(
                    values.rolling(window=window).count().eq(window).sum()
                ),
            },
        }

    def detect_periodicity(
        self, timeseries: TimeSeries, max_period: int = 60
    ) -> dict[str, Any]:
        """
        Detect periodicity in time series using FFT-based spectral analysis.

        Args:
            timeseries: TimeSeries object
            max_period: Maximum period to search for

        Returns:
            Dictionary with periodicity analysis results
        """
        positive_integer(max_period, "max_period", minimum=2)
        series = univariate_series(timeseries, minimum=2)
        regular_frequency(series.index)
        values = series.to_numpy()
        n = len(values)

        if n < 4:
            return {
                "error": "Time series too short for periodicity detection",
                "minimum_length": 4,
                "actual_length": n,
            }

        # Detrend the data
        detrended = values - np.mean(values)

        # FFT
        fft_values = np.fft.rfft(detrended)
        power_spectrum = np.abs(fft_values) ** 2

        # Get frequencies
        freqs = np.fft.rfftfreq(n)

        # Convert to periods (exclude DC component)
        periods = []
        for i in range(1, len(freqs)):
            if freqs[i] > 0:
                period = 1 / freqs[i]
                if period <= max_period:
                    periods.append(
                        {
                            "period": float(period),
                            "power": float(power_spectrum[i]),
                            "frequency": float(freqs[i]),
                        }
                    )

        # Sort by power
        periods.sort(key=lambda x: x["power"], reverse=True)
        top_periods = periods[:5]

        # Determine dominant period
        if top_periods:
            dominant = top_periods[0]
            # Calculate periodicity strength
            total_power = sum(p["power"] for p in periods)
            dominant_strength = (
                dominant["power"] / total_power if total_power > 0 else 0
            )
        else:
            dominant = None
            dominant_strength = 0

        return {
            "series_length": n,
            "max_period_searched": max_period,
            "dominant_period": {
                "period": dominant["period"] if dominant else None,
                "strength": float(dominant_strength),
                "interpretation": (
                    f"Strong periodicity at {dominant['period']:.1f} periods"
                    if dominant_strength > 0.3 and dominant
                    else "No strong periodicity detected"
                ),
            },
            "top_periods": top_periods,
            "spectral_entropy": float(self._spectral_entropy(power_spectrum)),
        }

    def _spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """Calculate spectral entropy from power spectrum."""
        # Normalize to probability distribution
        total = np.sum(power_spectrum)
        if total == 0:
            return 0.0
        probs = power_spectrum / total
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def calculate_granger_causality(
        self, timeseries1: TimeSeries, timeseries2: TimeSeries, max_lag: int = 5
    ) -> dict[str, Any]:
        """
        Test for Granger causality between two time series.

        Tests whether timeseries1 Granger-causes timeseries2 and vice versa.

        Args:
            timeseries1: First time series
            timeseries2: Second time series
            max_lag: Maximum lag to test

        Returns:
            Dictionary with Granger causality test results
        """
        from scipy import stats

        positive_integer(max_lag, "max_lag")
        first, second = paired_series(timeseries1, timeseries2)
        data1, data2 = first.to_numpy(), second.to_numpy()
        min_len = len(data1)

        tests: dict[str, Any] = {}
        results = {"series_length": min_len, "max_lag": max_lag, "tests": tests}

        def test_granger(y: np.ndarray, x: np.ndarray, lag: int) -> dict[str, Any]:
            """Simple F-test for Granger causality."""
            if len(y) - lag <= 2 * lag + 1:
                return {"error": "Insufficient data for lag"}

            # Create lagged variables
            y_lagged = y[lag:]
            y_lags = np.column_stack([y[lag - i - 1 : -i - 1] for i in range(lag)])
            x_lags = np.column_stack([x[lag - i - 1 : -i - 1] for i in range(lag)])

            # Restricted model (only y lags)
            try:
                # Simple OLS
                X_r = np.column_stack([np.ones(len(y_lagged)), y_lags])
                beta_r = np.linalg.lstsq(X_r, y_lagged, rcond=None)[0]
                residuals_r = y_lagged - X_r @ beta_r
                rss_r = np.sum(residuals_r**2)

                # Unrestricted model (y and x lags)
                X_u = np.column_stack([np.ones(len(y_lagged)), y_lags, x_lags])
                if np.linalg.matrix_rank(X_u) < X_u.shape[1]:
                    return {
                        "error": "Singular lagged design; causality is not identifiable"
                    }
                beta_u = np.linalg.lstsq(X_u, y_lagged, rcond=None)[0]
                residuals_u = y_lagged - X_u @ beta_u
                rss_u = np.sum(residuals_u**2)

                # F-test
                n = len(y_lagged)
                k_u = 2 * lag + 1

                if rss_u > 0:
                    f_stat = ((rss_r - rss_u) / lag) / (rss_u / (n - k_u))
                    p_value = stats.f.sf(max(0.0, f_stat), lag, n - k_u)
                else:
                    return {"error": "Zero residual variance; F test is undefined"}

                return {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "lag": lag,
                }
            except np.linalg.LinAlgError as e:
                return {"error": str(e)}

        # Test if series1 Granger-causes series2
        for lag in range(1, max_lag + 1):
            key = f"series1_causes_series2_lag{lag}"
            tests[key] = test_granger(data2, data1, lag)

        # Test if series2 Granger-causes series1
        for lag in range(1, max_lag + 1):
            key = f"series2_causes_series1_lag{lag}"
            tests[key] = test_granger(data1, data2, lag)

        # Summarize
        s1_causes_s2 = any(
            isinstance(v, dict) and v.get("significant", False)
            for k, v in tests.items()
            if "series1_causes_series2" in k
        )
        s2_causes_s1 = any(
            isinstance(v, dict) and v.get("significant", False)
            for k, v in tests.items()
            if "series2_causes_series1" in k
        )

        results["summary"] = {
            "series1_granger_causes_series2": s1_causes_s2,
            "series2_granger_causes_series1": s2_causes_s1,
            "bidirectional_causality": s1_causes_s2 and s2_causes_s1,
            "interpretation": (
                "Bidirectional causality detected"
                if s1_causes_s2 and s2_causes_s1
                else "Series 1 Granger-causes Series 2"
                if s1_causes_s2
                else "Series 2 Granger-causes Series 1"
                if s2_causes_s1
                else "No significant Granger causality detected"
            ),
        }

        return results

    def compute_temporal_entropy(
        self,
        timeseries: TimeSeries,
        bins: int = 10,
        method: str = "shannon",
        *,
        embedding_dim: int = 2,
        tolerance: float | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        """Compute histogram Shannon, sample, or approximate entropy.

        Sample entropy excludes self-matches and compares the same template
        starting positions at dimensions m and m+1. Approximate entropy
        includes self-matches and averages log match frequencies separately.
        Matching uses Chebyshev distance <= tolerance. The default tolerance
        is 0.2 times population standard deviation. Pairwise work is bounded
        by max_points; large callers must explicitly choose their budget.
        """
        if method not in {"shannon", "sample", "approximate"}:
            raise ValueError(f"Unknown entropy method: {method}")
        positive_integer(bins, "bins")
        positive_integer(embedding_dim, "embedding_dim")
        positive_integer(max_points, "max_points")
        series = univariate_series(timeseries)
        values = series.to_numpy()
        n = len(values)
        counts, _ = np.histogram(values, bins=bins)
        probabilities = counts[counts > 0] / n
        shannon = float(-np.dot(probabilities, np.log2(probabilities)))
        normalized = shannon / np.log2(bins) if bins > 1 else 0.0
        results: dict[str, Any] = {
            "series_length": n,
            "method": method,
            "shannon_entropy": {
                "value": shannon,
                "bins": bins,
                "normalized": float(normalized),
            },
        }
        if method != "shannon":
            regular_frequency(series.index)
            if n > max_points:
                raise ValueError(
                    "Entropy calculation exceeds max_points pairwise-work budget"
                )
            if n <= embedding_dim + 1:
                raise ValueError(
                    "Entropy requires more than embedding_dim + 1 observations"
                )
            if tolerance is None:
                tolerance = 0.2 * float(np.std(values))
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance)
                or tolerance < 0
            ):
                raise ValueError("tolerance must be finite and non-negative")

            def templates(length: int) -> np.ndarray:
                return np.lib.stride_tricks.sliding_window_view(values, length)

            def pairs(array: np.ndarray) -> int:
                return sum(
                    int(
                        np.count_nonzero(
                            np.max(np.abs(array[i + 1 :] - row), axis=1) <= tolerance
                        )
                    )
                    for i, row in enumerate(array[:-1])
                )

            # Both counts compare exactly the same N-m starting positions.
            b = pairs(templates(embedding_dim)[:-1])
            a = pairs(templates(embedding_dim + 1))
            results["sample_entropy"] = {
                "value": float(-np.log(a / b)) if a and b else None,
                "embedding_dim": embedding_dim,
                "tolerance": float(tolerance),
                "matching_pairs": b,
                "matching_extensions": a,
            }
            if method == "approximate":

                def phi(length: int) -> float:
                    array = templates(length)
                    frequencies = [
                        np.count_nonzero(
                            np.max(np.abs(array - row), axis=1) <= tolerance
                        )
                        / len(array)
                        for row in array
                    ]
                    return float(np.mean(np.log(frequencies)))

                results["approximate_entropy"] = {
                    "value": phi(embedding_dim) - phi(embedding_dim + 1),
                    "embedding_dim": embedding_dim,
                    "tolerance": float(tolerance),
                }
        results["interpretation"] = {
            "complexity": "High"
            if normalized > 0.8
            else "Medium"
            if normalized > 0.5
            else "Low",
            "predictability": "Low"
            if normalized > 0.8
            else "Medium"
            if normalized > 0.5
            else "High",
            "description": "Histogram diversity describes marginal values, not a forecast accuracy guarantee",
        }
        return results
