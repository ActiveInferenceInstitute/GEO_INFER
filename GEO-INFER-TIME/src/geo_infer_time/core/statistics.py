"""
Temporal Statistics Module for GEO-INFER-TIME.

Provides statistical methods for time series analysis including stationarity tests,
normality tests, and diagnostic statistics.
"""

import logging
from typing import Any
import numpy as np
from scipy import stats
from geo_infer_time.core._validation import finite_vector, positive_integer
from geo_infer_time.core.timestamps import normalize_datetime_index

logger = logging.getLogger(__name__)


class TemporalStatistics:
    """
    Comprehensive temporal statistics for time series analysis.

    Provides methods for calculating diagnostic statistics, model
    selection criteria, and time series characteristics.
    """

    def __init__(self) -> None:
        """Initialize temporal statistics."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug("TemporalStatistics initialized")

    def calculate_summary(
        self, values: list[float], timestamps: list | None = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive summary statistics for a time series.

        Args:
            values: Numeric values
            timestamps: Optional timestamp array

        Returns:
            Dictionary with comprehensive summary statistics
        """
        values_arr = finite_vector(values)
        n = len(values_arr)
        if timestamps is not None:
            if len(normalize_datetime_index(timestamps)) != n:
                raise ValueError("timestamps must have one instant per value")

        if n == 0:
            return {"error": "No values provided"}

        logger.info(f"Calculating summary statistics for {n} observations")

        mean = np.mean(values_arr)
        median = np.median(values_arr)
        scale = np.max(np.abs(values_arr))
        scaled_values = values_arr / scale if scale else values_arr
        std = scale * np.std(scaled_values, ddof=1) if n > 1 else 0.0
        variance = std**2

        # Standard error
        se = std / np.sqrt(n) if n > 0 else 0.0

        # Coefficient of variation
        cv = (std / abs(mean) * 100) if mean != 0 else None

        # Higher moments
        if n > 2 and std > 0:
            skewness = stats.skew(scaled_values)
            kurtosis = stats.kurtosis(scaled_values)
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Quartiles
        q1, q2, q3 = np.percentile(values_arr, [25, 50, 75])
        iqr = q3 - q1

        # Range
        min_val = np.min(values_arr)
        max_val = np.max(values_arr)
        range_val = max_val - min_val

        # First differences
        if n > 1:
            diffs = np.diff(values_arr)
            mean_abs_diff = float(np.mean(np.abs(diffs)))
            max_abs_diff = float(np.max(np.abs(diffs)))
        else:
            mean_abs_diff = 0.0
            max_abs_diff = 0.0

        # Trend indicator (simple linear)
        if n > 2:
            x = np.arange(n)
            slope, _, r_value, _, _ = stats.linregress(x, values_arr)
            trend_strength = r_value**2
            trend_direction = (
                "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
            )
        else:
            slope = 0.0
            trend_strength = 0.0
            trend_direction = "unknown"

        return {
            "n": n,
            "central_tendency": {
                "mean": float(mean),
                "median": float(median),
                "mode": (
                    float(stats.mode(values_arr, keepdims=True)[0][0])
                    if n > 0
                    else None
                ),
            },
            "dispersion": {
                "std": float(std),
                "variance": float(variance),
                "se": float(se),
                "cv": float(cv) if cv is not None else None,
                "range": float(range_val),
                "iqr": float(iqr),
            },
            "shape": {"skewness": float(skewness), "kurtosis": float(kurtosis)},
            "quantiles": {
                "min": float(min_val),
                "q1": float(q1),
                "median": float(q2),
                "q3": float(q3),
                "max": float(max_val),
            },
            "dynamics": {
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
                "trend_slope": float(slope),
                "trend_strength": float(trend_strength),
                "trend_direction": trend_direction,
            },
        }

    def calculate_differences(
        self, values: list[float], order: int = 1, seasonal_period: int | None = None
    ) -> dict[str, Any]:
        """
        Calculate differenced series for stationarity.

        Args:
            values: Original time series values
            order: Differencing order (1 = first, 2 = second)
            seasonal_period: Optional seasonal differencing period

        Returns:
            Dictionary with differenced series and statistics
        """
        values_arr = finite_vector(values)
        n = len(values_arr)

        positive_integer(order, "order", minimum=0)
        if seasonal_period is not None:
            positive_integer(seasonal_period, "seasonal_period")
        if n <= order:
            return {"error": f"Series too short for order {order} differencing"}

        logger.info(f"Calculating order-{order} differences")

        # Regular differencing
        diff_values = values_arr.copy()
        for _ in range(order):
            diff_values = np.diff(diff_values)

        # Seasonal differencing
        seasonal_diff = None
        if seasonal_period and n > seasonal_period:
            seasonal_diff = values_arr[seasonal_period:] - values_arr[:-seasonal_period]

        result = {
            "original_length": n,
            "order": order,
            "differenced": {
                "values": diff_values.tolist(),
                "length": len(diff_values),
                "mean": float(np.mean(diff_values)),
                "std": float(np.std(diff_values)),
            },
        }

        if seasonal_diff is not None:
            result["seasonal_differenced"] = {
                "period": seasonal_period,
                "values": seasonal_diff.tolist(),
                "length": len(seasonal_diff),
                "mean": float(np.mean(seasonal_diff)),
                "std": float(np.std(seasonal_diff)),
            }

        return result

    def ljung_box_test(self, values: list[float], lags: int = 10) -> dict[str, Any]:
        """
        Perform Ljung-Box test for autocorrelation in residuals.

        Tests whether any of a group of autocorrelations are significantly
        different from zero.

        Args:
            values: Time series values (typically residuals)
            lags: Number of lags to test

        Returns:
            Dictionary with test results
        """
        values_arr = finite_vector(values)
        n = len(values_arr)

        positive_integer(lags, "lags")
        if n <= lags:
            return {"error": "Series too short for specified lags"}

        logger.info(f"Performing Ljung-Box test with {lags} lags")

        # Calculate autocorrelations
        scale = np.max(np.abs(values_arr))
        scaled = values_arr / scale if scale else values_arr
        deviations = scaled - np.mean(scaled)
        var = np.sum(deviations**2) / n

        if var == 0:
            return {"error": "Zero variance in series"}

        acf_values = []
        for k in range(1, lags + 1):
            if n - k > 0:
                acf_k = np.sum(deviations[k:] * deviations[:-k]) / (n * var)
                acf_values.append(acf_k)

        # Ljung-Box statistic
        lb_stat = 0.0
        for k, acf_k in enumerate(acf_values, 1):
            lb_stat += (acf_k**2) / (n - k)
        lb_stat *= n * (n + 2)

        # P-value from chi-square distribution
        p_value = stats.chi2.sf(lb_stat, lags)

        return {
            "lb_statistic": float(lb_stat),
            "p_value": float(p_value),
            "lags": lags,
            "acf_values": [float(a) for a in acf_values],
            "significant": p_value < 0.05,
            "interpretation": (
                "Significant autocorrelation detected (residuals not random)"
                if p_value < 0.05
                else "No significant autocorrelation (residuals appear random)"
            ),
        }

    def jarque_bera_test(self, values: list[float]) -> dict[str, Any]:
        """
        Perform Jarque-Bera test for normality.

        Tests whether the sample data have the skewness and kurtosis
        matching a normal distribution.

        Args:
            values: Time series values

        Returns:
            Dictionary with test results
        """
        values_arr = finite_vector(values)
        n = len(values_arr)

        if n < 3:
            return {"error": "Need at least 3 observations"}

        logger.info("Performing Jarque-Bera normality test")

        # Calculate skewness and kurtosis
        scale = np.max(np.abs(values_arr))
        standardized = values_arr / scale if scale else values_arr
        if np.std(standardized) == 0:
            return {
                "error": "Zero variance; normality test is undefined",
                "is_normal": False,
            }
        skewness = stats.skew(standardized)
        kurtosis = stats.kurtosis(standardized)  # Excess kurtosis

        # Jarque-Bera statistic
        jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)

        # P-value from chi-square distribution with 2 df
        p_value = stats.chi2.sf(jb_stat, 2)

        return {
            "jb_statistic": float(jb_stat),
            "p_value": float(p_value),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "n": n,
            "is_normal": p_value > 0.05,
            "interpretation": (
                "Data appears normally distributed (p > 0.05)"
                if p_value > 0.05
                else "Data significantly deviates from normality (p < 0.05)"
            ),
        }

    def durbin_watson_test(self, residuals: list[float]) -> dict[str, Any]:
        """
        Calculate Durbin-Watson statistic for serial correlation.

        Tests for first-order autocorrelation in residuals.
        DW ≈ 2 indicates no autocorrelation
        DW < 2 indicates positive autocorrelation
        DW > 2 indicates negative autocorrelation

        Args:
            residuals: Model residuals

        Returns:
            Dictionary with test results
        """
        residuals_arr = finite_vector(residuals, "residuals")
        n = len(residuals_arr)

        if n < 2:
            return {"error": "Need at least 2 residuals"}

        logger.info("Calculating Durbin-Watson statistic")

        # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        scale = np.max(np.abs(residuals_arr))
        if scale == 0:
            return {"error": "Zero residual energy; Durbin-Watson is undefined"}
        scaled = residuals_arr / scale
        diff_residuals = np.diff(scaled)
        dw = np.sum(diff_residuals**2) / np.sum(scaled**2)

        # Approximate rho (first-order autocorrelation)
        rho = 1 - dw / 2

        if dw < 1.0:
            interpretation = "Strong positive autocorrelation"
        elif dw < 1.5:
            interpretation = "Moderate positive autocorrelation"
        elif dw < 2.5:
            interpretation = "No significant autocorrelation"
        elif dw < 3.0:
            interpretation = "Moderate negative autocorrelation"
        else:
            interpretation = "Strong negative autocorrelation"

        return {
            "dw_statistic": float(dw),
            "rho": float(rho),
            "n": n,
            "interpretation": interpretation,
            "autocorrelation": (
                "positive" if dw < 1.5 else "negative" if dw > 2.5 else "none"
            ),
        }

    def hurst_exponent(
        self, values: list[float], max_lag: int | None = None
    ) -> dict[str, Any]:
        """
        Calculate Hurst exponent for long-term memory.

        H = 0.5: Random walk (no memory)
        H > 0.5: Positive long-term correlation (trending)
        H < 0.5: Negative long-term correlation (mean-reverting)

        Args:
            values: Time series values
            max_lag: Maximum lag to consider

        Returns:
            Dictionary with Hurst exponent
        """
        values_arr = finite_vector(values)
        n = len(values_arr)

        if n < 10:
            return {"error": "Need at least 10 observations"}

        if max_lag is None:
            max_lag = n // 4

        positive_integer(max_lag, "max_lag", minimum=2)
        logger.info("Calculating Hurst exponent")

        scale = np.max(np.abs(values_arr))
        values_arr = values_arr / scale if scale else values_arr
        # R/S analysis
        lags = []
        rs_values = []

        for lag in range(2, min(max_lag, n // 2) + 1):
            rs_sum = 0
            count = 0

            for start in range(0, n - lag + 1, lag):
                segment = values_arr[start : start + lag]
                if len(segment) < lag:
                    continue

                mean = np.mean(segment)
                cumsum = np.cumsum(segment - mean)
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(segment, ddof=1)

                if s > 0:
                    rs_sum += r / s
                    count += 1

            if count > 0:
                lags.append(np.log(lag))
                rs_values.append(np.log(rs_sum / count))

        if len(lags) < 2:
            return {"error": "Could not calculate R/S for enough lags"}

        # Linear regression to find Hurst exponent
        slope, intercept, r_value, _, _ = stats.linregress(lags, rs_values)
        hurst = slope

        if hurst < 0.4:
            process_type = "mean-reverting (anti-persistent)"
        elif hurst > 0.6:
            process_type = "trending (persistent)"
        else:
            process_type = "random walk"

        return {
            "hurst_exponent": float(hurst),
            "r_squared": float(r_value**2),
            "process_type": process_type,
            "max_lag": max_lag,
            "interpretation": f"H={hurst:.3f} indicates {process_type}",
        }

    def information_criteria(
        self,
        residuals: list[float],
        num_params: int,
        log_likelihood: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate information criteria for model selection.

        Args:
            residuals: Model residuals
            num_params: Number of estimated parameters
            log_likelihood: Log-likelihood (if not provided, estimated from residuals)

        Returns:
            Dictionary with AIC, BIC, and other criteria
        """
        residuals_arr = finite_vector(residuals, "residuals")
        n = len(residuals_arr)
        k = positive_integer(num_params, "num_params", minimum=0)
        if log_likelihood is not None and not np.isfinite(log_likelihood):
            raise ValueError("log_likelihood must be finite")

        if n <= k:
            return {"error": "More parameters than observations"}
        if n < 2:
            return {"error": "At least 2 observations are required"}

        logger.info(f"Calculating information criteria for {k} parameters")

        # Estimate log-likelihood from residuals if not provided
        if log_likelihood is None:
            # Assuming normal distribution
            scale = np.max(np.abs(residuals_arr))
            if scale == 0:
                return {"error": "Zero residual variance; likelihood is undefined"}
            log_sigma2 = 2 * np.log(scale) + np.log(
                np.mean((residuals_arr / scale) ** 2)
            )
            log_likelihood = -n / 2 * (np.log(2 * np.pi) + log_sigma2 + 1)

        # AIC (Akaike Information Criterion)
        aic = -2 * log_likelihood + 2 * k

        # AICc (corrected AIC for small samples)
        if n - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc = float("inf")

        # BIC (Bayesian Information Criterion)
        bic = -2 * log_likelihood + k * np.log(n)

        # HQC (Hannan-Quinn Criterion)
        hqc = -2 * log_likelihood + 2 * k * np.log(np.log(n))

        return {
            "log_likelihood": float(log_likelihood),
            "n": n,
            "k": k,
            "aic": float(aic),
            "aicc": float(aicc),
            "bic": float(bic),
            "hqc": float(hqc),
            "interpretation": (
                "Lower values indicate better model fit. "
                "AIC/AICc penalizes complexity less than BIC."
            ),
        }

    def residual_diagnostics(self, residuals: list[float]) -> dict[str, Any]:
        """
        Comprehensive residual diagnostics.

        Args:
            residuals: Model residuals

        Returns:
            Dictionary with comprehensive diagnostics
        """
        residuals_arr = finite_vector(residuals, "residuals")

        scale = np.max(np.abs(residuals_arr)) if len(residuals_arr) else 0
        scaled = residuals_arr / scale if scale else residuals_arr
        if len(residuals_arr) < 4 or np.std(scaled) == 0:
            return {
                "error": "Residual diagnostics require at least four nonconstant finite residuals"
            }
        logger.info("Running comprehensive residual diagnostics")

        # Summary statistics
        summary = self.calculate_summary(residuals)

        # Normality test
        normality = self.jarque_bera_test(residuals)

        # Serial correlation
        dw = self.durbin_watson_test(residuals)
        lb = self.ljung_box_test(residuals, lags=min(10, len(residuals) - 2))

        # Mean zero test
        from scipy.stats import ttest_1samp

        t_stat, p_value = ttest_1samp(scaled, 0)
        mean_zero = p_value > 0.05

        # Constant variance (simple test)
        n = len(residuals_arr)
        half = n // 2
        if half > 1:
            var1 = np.var(scaled[:half], ddof=1)
            var2 = np.var(scaled[half:], ddof=1)
            var_ratio = (
                max(var1, var2) / min(var1, var2) if min(var1, var2) > 0 else None
            )
            homoscedastic = var_ratio < 2.0 if var_ratio is not None else None
        else:
            var_ratio = 1.0
            homoscedastic = True

        # Overall assessment
        issues = []
        if not normality.get("is_normal", True):
            issues.append("Non-normal distribution")
        if not mean_zero:
            issues.append("Mean significantly different from zero")
        if homoscedastic is None:
            issues.append("Variance comparison undefined for a zero-variance segment")
        elif not homoscedastic:
            issues.append("Possible heteroscedasticity")
        if lb.get("significant", False):
            issues.append("Significant autocorrelation")

        return {
            "summary": summary,
            "normality": normality,
            "serial_correlation": {"durbin_watson": dw, "ljung_box": lb},
            "mean_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "mean_is_zero": mean_zero,
            },
            "variance_test": {
                "variance_ratio": float(var_ratio) if var_ratio is not None else None,
                "homoscedastic": homoscedastic,
            },
            "overall": {
                "issues": issues,
                "residuals_ok": len(issues) == 0,
                "recommendation": (
                    "Residuals appear well-behaved"
                    if len(issues) == 0
                    else f"Issues detected: {', '.join(issues)}"
                ),
            },
        }
