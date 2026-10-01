"""
Temporal analysis tools for Statistical Parametric Mapping

This module provides temporal analysis capabilities for SPM, including:
- Time series decomposition and trend detection
- Temporal autocorrelation modeling
- Event-related analysis for discrete temporal phenomena
- Seasonal adjustment and cyclic pattern identification
- Sliding window analysis for dynamic processes

The implementation supports various temporal models including:
- ARIMA/ARMA processes for autocorrelation
- Seasonal decomposition
- Wavelet analysis for multi-scale temporal patterns
- Change point detection for temporal discontinuities

Mathematical Foundation:
Temporal autocorrelation is modeled using ARMA processes:
y_t = φ₁y_{t-1} + ... + φ_py_{t-p} + ε_t - θ₁ε_{t-1} - ... - θ_qε_{t-q}
"""

import numpy as np
from typing import Any
from collections.abc import Callable
from scipy import signal
from scipy.stats import linregress

# Optional statsmodels imports
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
import warnings


class TemporalAnalyzer:
    """
    Temporal analysis tools for SPM data.

    This class provides methods for analyzing temporal structure in time series
    SPM data, including trend detection, seasonal decomposition, and temporal
    autocorrelation modeling.

    Attributes:
        time_points: Temporal coordinates of data points
        time_series: Time series data for analysis
        temporal_model: Fitted temporal model parameters
    """

    def __init__(self, time_points: np.ndarray, time_series: np.ndarray | None = None):
        """
        Initialize temporal analyzer.

        Args:
            time_points: Temporal coordinates (timestamps or time indices)
            time_series: Optional time series data for analysis
        """
        self.time_points = time_points
        self.time_series = time_series
        self.temporal_model = None

        # Sort by time if not already sorted
        if not np.all(np.diff(time_points) >= 0):
            sort_idx = np.argsort(time_points)
            self.time_points = time_points[sort_idx]
            if self.time_series is not None:
                self.time_series = self.time_series[sort_idx]

    def detect_trends(
        self, data: np.ndarray, method: str = "linear", alpha: float = 0.05
    ) -> dict[str, Any]:
        """
        Detect temporal trends in SPM data.

        Args:
            data: Time series data (n_timepoints x n_variables)
            method: Trend detection method ('linear', 'mann_kendall', 'theil_sen')
            alpha: Significance level for trend detection

        Returns:
            Dictionary with trend statistics and significance
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_variables = data.shape[1]
        trends = []

        for var_idx in range(n_variables):
            y = data[:, var_idx]
            x = np.arange(len(y))

            if method == "linear":
                result = self._linear_trend_test(x, y, alpha)
            elif method == "mann_kendall":
                result = self._mann_kendall_test(y, alpha)
            elif method == "theil_sen":
                result = self._theil_sen_trend(y, alpha)
            else:
                raise ValueError(f"Unknown trend method: {method}")

            result["variable_index"] = var_idx
            trends.append(result)

        return {
            "trends": trends,
            "method": method,
            "alpha": alpha,
            "n_variables": n_variables,
        }

    def _linear_trend_test(
        self, x: np.ndarray, y: np.ndarray, alpha: float
    ) -> dict[str, Any]:
        """Test for linear trend using ordinary least squares."""
        if len(x) < 2 or np.unique(x).size < 2:
            return {
                "slope": 0.0,
                "intercept": float(np.mean(y)) if len(y) else 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "std_err": 0.0,
                "trend_magnitude": 0.0,
                "significant": False,
                "direction": "stable",
            }
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Compute trend magnitude over time period
        trend_magnitude = slope * len(x)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "trend_magnitude": trend_magnitude,
            "significant": p_value < alpha,
            "direction": "increasing" if slope > 0 else "decreasing",
        }

    def _mann_kendall_test(self, y: np.ndarray, alpha: float) -> dict[str, Any]:
        """Mann-Kendall test for monotonic trends."""
        n = len(y)
        s = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(y[j] - y[i])

        # Variance of S
        unique_vals, counts = np.unique(y, return_counts=True)
        tp = np.sum(counts * (counts - 1) * (2 * counts + 5)) / 18
        var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18

        if s > 0:
            z = (s - 1) / np.sqrt(var_s) if var_s > 0 else 0
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s) if var_s > 0 else 0
        else:
            z = 0

        # Two-tailed p-value
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(abs(z)))

        # Sen's slope estimator
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                slopes.append((y[j] - y[i]) / (j - i))
        sen_slope = np.median(slopes) if slopes else 0

        return {
            "s_statistic": s,
            "z_score": z,
            "p_value": p_value,
            "sen_slope": sen_slope,
            "significant": p_value < alpha,
            "direction": "increasing" if sen_slope > 0 else "decreasing",
        }

    def _theil_sen_trend(self, y: np.ndarray, alpha: float) -> dict[str, Any]:
        """Theil-Sen estimator for robust trend detection."""
        n = len(y)
        slopes = []

        for i in range(n):
            for j in range(i + 1, n):
                slope = (y[j] - y[i]) / (j - i)
                slopes.append(slope)

        median_slope = np.median(slopes)

        # Deferred: see docs/deferred_statistical_methods.md
        # ("Theil-Sen significance test").
        slope_std = np.std(slopes)
        if slope_std > 0:
            z_score = median_slope / (slope_std / np.sqrt(len(slopes)))
            from scipy.stats import norm

            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1.0

        return {
            "median_slope": median_slope,
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value < alpha,
            "direction": "increasing" if median_slope > 0 else "decreasing",
        }

    def seasonal_decomposition(
        self, data: np.ndarray, period: int | None = None, model: str = "additive"
    ) -> dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            data: Time series data
            period: Seasonal period (auto-detected if None)
            model: Decomposition model ('additive' or 'multiplicative')

        Returns:
            Dictionary with decomposed components
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels package required for seasonal decomposition")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_variables = data.shape[1]
        decompositions = []

        for var_idx in range(n_variables):
            y = data[:, var_idx]

            # Auto-detect period if not provided
            if period is None:
                period = self._detect_seasonal_period(y)

            try:
                # Use statsmodels seasonal decomposition
                decomposition = seasonal_decompose(y, model=model, period=period)

                result = {
                    "trend": decomposition.trend,
                    "seasonal": decomposition.seasonal,
                    "residual": decomposition.resid,
                    "period": period,
                    "model": model,
                    "variable_index": var_idx,
                }

            except Exception as e:
                warnings.warn(
                    f"Seasonal decomposition failed for variable {var_idx}: {e}",
                    stacklevel=2,
                )
                # Fallback to simple decomposition
                result = self._simple_seasonal_decomposition(y, period)

            decompositions.append(result)

        if n_variables == 1:
            single = decompositions[0].copy()
            single.update(
                {
                    "decompositions": decompositions,
                    "n_variables": n_variables,
                    "method": "seasonal_decompose",
                }
            )
            return single

        return {
            "decompositions": decompositions,
            "n_variables": n_variables,
            "method": "seasonal_decompose",
        }

    def _detect_seasonal_period(self, y: np.ndarray) -> int:
        """Auto-detect seasonal period using autocorrelation."""
        # Compute autocorrelation function
        max_lags = min(len(y) // 3, 100)
        autocorr = acf(y, nlags=max_lags, fft=True)

        # Find peaks in autocorrelation (potential periods)
        peaks = signal.find_peaks(autocorr, height=0.1)[0]

        if len(peaks) > 0:
            # Return the period with strongest autocorrelation
            best_peak = peaks[np.argmax(autocorr[peaks])]
            period = best_peak
        else:
            # Default to weekly/monthly patterns
            period = 7 if len(y) > 14 else 12

        return int(max(2, min(int(period), len(y) // 3)))

    def _simple_seasonal_decomposition(
        self, y: np.ndarray, period: int
    ) -> dict[str, Any]:
        """Simple seasonal decomposition as fallback."""
        n = len(y)

        # Estimate seasonal component
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal[indices] = np.mean(y[indices])

        # Remove seasonal component
        deseasonalized = y - seasonal

        # Estimate trend using moving average
        window = min(period, n // 4)
        if window % 2 == 0:
            window += 1
        trend = signal.savgol_filter(deseasonalized, window, 2)

        # Residuals
        residual = deseasonalized - trend

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "period": period,
            "model": "simple",
        }

    def fit_arima_model(
        self,
        data: np.ndarray,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Fit ARIMA model to time series data.

        Args:
            data: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)

        Returns:
            Dictionary with fitted model and diagnostics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels package required for ARIMA modeling")

        if data.ndim > 1:
            # Fit separate models for each variable
            models = []
            for var_idx in range(data.shape[1]):
                y = data[:, var_idx]
                model_result = self._fit_single_arima(y, order, seasonal_order)
                model_result["variable_index"] = var_idx
                models.append(model_result)

            return {
                "models": models,
                "n_variables": data.shape[1],
                "order": order,
                "seasonal_order": seasonal_order,
            }
        else:
            return self._fit_single_arima(data.flatten(), order, seasonal_order)

    def _fit_single_arima(
        self,
        y: np.ndarray,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] | None,
    ) -> dict[str, Any]:
        """Fit ARIMA model to single time series."""
        try:
            if seasonal_order is not None:
                model = ARIMA(y, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(y, order=order)

            fitted_model = model.fit()

            # Get residuals and diagnostics
            residuals = fitted_model.resid
            aic = fitted_model.aic
            bic = fitted_model.bic

            # Forecast next point for validation
            forecast = fitted_model.forecast(steps=1)[0]

            return {
                "model": fitted_model,
                "residuals": residuals,
                "aic": aic,
                "bic": bic,
                "forecast": forecast,
                "order": order,
                "seasonal_order": seasonal_order,
                "success": True,
            }

        except Exception as e:
            warnings.warn(f"ARIMA fitting failed: {e}", stacklevel=2)
            return {
                "error": str(e),
                "success": False,
                "order": order,
                "seasonal_order": seasonal_order,
            }

    def sliding_window_analysis(
        self,
        data: np.ndarray,
        window_size: int,
        step_size: int = 1,
        analysis_func: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform sliding window analysis for dynamic temporal patterns.

        Args:
            data: Time series data
            window_size: Size of sliding window
            step_size: Step size for window movement
            analysis_func: Custom analysis function to apply to each window

        Returns:
            Dictionary with windowed analysis results
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_timepoints, n_variables = data.shape
        n_windows = (n_timepoints - window_size) // step_size + 1

        if analysis_func is None:
            # Default: compute mean and variance for each window
            def analysis_func(x: np.ndarray) -> dict[str, Any]:
                return {"mean": np.mean(x, axis=0), "var": np.var(x, axis=0)}

        window_results = []

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            window_data = data[start_idx:end_idx]

            try:
                result = analysis_func(window_data)
                result["window_start"] = start_idx
                result["window_end"] = end_idx
                result["window_center"] = (start_idx + end_idx) / 2
                window_results.append(result)
            except Exception as e:
                warnings.warn(f"Analysis failed for window {i}: {e}", stacklevel=2)
                continue

        return {
            "window_results": window_results,
            "window_size": window_size,
            "step_size": step_size,
            "n_windows": len(window_results),
            "n_variables": n_variables,
        }

    def change_point_detection(
        self, data: np.ndarray, method: str = "pelt", penalty: float = 10
    ) -> dict[str, Any]:
        """
        Detect change points in time series data.

        Args:
            data: Time series data
            method: Change point detection method ('pelt', 'binary_segmentation')
            penalty: Penalty parameter for change point detection

        Returns:
            Dictionary with detected change points
        """
        try:
            from ruptures import Pelt, Binseg
        except ImportError as exc:
            raise ImportError(
                "ruptures package required for change point detection"
            ) from exc

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_variables = data.shape[1]
        change_points_all = []

        for var_idx in range(n_variables):
            y = data[:, var_idx]

            if method == "pelt":
                algo = Pelt(model="rbf").fit(y)
                change_points = algo.predict(pen=penalty)
            elif method == "binary_segmentation":
                algo = Binseg(model="rbf").fit(y)
                change_points = algo.predict(pen=penalty)
            else:
                raise ValueError(f"Unknown change point method: {method}")

            # Remove the last change point (end of series)
            if change_points and change_points[-1] == len(y):
                change_points = change_points[:-1]

            change_points_all.append(
                {
                    "variable_index": var_idx,
                    "change_points": change_points,
                    "n_changes": len(change_points),
                }
            )

        return {
            "change_points": change_points_all,
            "method": method,
            "penalty": penalty,
            "n_variables": n_variables,
        }

    def temporal_basis_functions(
        self, n_basis: int = 10, basis_type: str = "fourier"
    ) -> np.ndarray:
        """
        Generate temporal basis functions for modeling temporal variation.

        Args:
            n_basis: Number of basis functions
            basis_type: Type of basis functions ('fourier', 'polynomial', 'bspline')

        Returns:
            Basis function matrix (n_timepoints x n_basis)
        """
        n_timepoints = len(self.time_points)
        t_norm = (self.time_points - np.min(self.time_points)) / (
            np.max(self.time_points) - np.min(self.time_points)
        )

        if basis_type == "fourier":
            # Fourier basis functions
            basis = np.zeros((n_timepoints, n_basis))
            basis[:, 0] = 1  # Constant term

            for i in range(1, n_basis, 2):
                freq = (i + 1) // 2
                basis[:, i] = np.sin(2 * np.pi * freq * t_norm)
                if i + 1 < n_basis:
                    basis[:, i + 1] = np.cos(2 * np.pi * freq * t_norm)

        elif basis_type == "polynomial":
            # Polynomial basis functions
            basis_list = []
            for degree in range(n_basis):
                basis_list.append(t_norm**degree)
            basis = np.column_stack(basis_list)

        elif basis_type == "bspline":
            # B-spline basis functions
            try:
                # Create knots for B-splines
                n_knots = max(4, n_basis - 2)
                np.linspace(0, 1, n_knots)

                basis = np.zeros((n_timepoints, n_basis))
                for i in range(n_basis):
                    # Deferred: see docs/deferred_statistical_methods.md
                    # ("Temporal B-spline basis functions").
                    centers = np.linspace(0, 1, n_basis)
                    basis[:, i] = np.exp(-(((t_norm - centers[i]) / 0.1) ** 2))

            except Exception:
                # Fallback to Gaussian basis
                warnings.warn(
                    "B-spline implementation failed, using Gaussian basis", stacklevel=2
                )
                centers = np.linspace(0, 1, n_basis)
                basis = np.zeros((n_timepoints, n_basis))
                for i in range(n_basis):
                    basis[:, i] = np.exp(-(((t_norm - centers[i]) / 0.1) ** 2))

        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

        return basis
