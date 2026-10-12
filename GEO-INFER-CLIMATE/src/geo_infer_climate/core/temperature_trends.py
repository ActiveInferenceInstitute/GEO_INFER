"""Temperature trend analysis with statistical significance testing.

Implements linear regression trends and the Mann-Kendall non-parametric
trend test for detecting monotonic trends in temperature time series.
"""

import logging
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class TemperatureTrendAnalyzer:
    """Analyze temperature trends using parametric and non-parametric methods.

    Implements linear regression and Mann-Kendall trend tests,
    along with Sen's slope estimator for robust trend magnitude.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize temperature trend analyzer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def linear_trend(
        self,
        time_series: np.ndarray,
        years: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calculate linear trend via ordinary least squares regression.

        Args:
            time_series: Temperature observations.
            years: Corresponding year values. If None, uses integer indices.

        Returns:
            Dictionary with slope, intercept, r-squared, p-value.
        """
        n = len(time_series)
        if n < 3:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "slope_per_decade": 0.0,
                "n_observations": n,
            }

        x = (
            np.asarray(years, dtype=float)
            if years is not None
            else np.arange(n, dtype=float)
        )
        y = np.asarray(time_series, dtype=float)

        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]

        if len(x) < 3:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "slope_per_decade": 0.0,
                "n_observations": int(len(x)),
            }

        result = scipy_stats.linregress(x, y)

        return {
            "slope": float(result.slope),
            "intercept": float(result.intercept),
            "r_squared": float(result.rvalue**2),
            "p_value": float(result.pvalue),
            "std_error": float(result.stderr),
            "slope_per_decade": float(result.slope * 10),
            "n_observations": int(len(x)),
        }

    def mann_kendall_test(
        self,
        time_series: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Perform Mann-Kendall trend test.

        Non-parametric test for detecting monotonic trends.
        Does not require normal distribution of data.

        The test statistic S is:
        S = sum_{i<j} sgn(x_j - x_i)

        Variance: Var(S) = n(n-1)(2n+5)/18

        Args:
            time_series: Observations ordered in time.
            alpha: Significance level (default 0.05).

        Returns:
            Dictionary with S statistic, Z value, p-value, trend indicator.
        """
        x = np.asarray(time_series, dtype=float)
        x = x[~np.isnan(x)]
        n = len(x)

        if n < 4:
            return {
                "s_statistic": 0,
                "z_value": 0.0,
                "p_value": 1.0,
                "trend": "no trend",
                "significant": False,
                "n_observations": n,
            }

        # S = sum_{i<j} sgn(x_j - x_i), accumulated in row blocks so the
        # difference matrix never materializes in full
        # (O(n^2) work, O(chunk*n) memory, all in numpy instead of Python).
        s = 0
        cols = np.arange(n)
        chunk = max(1, int(2**23 // max(n, 1)))  # ~64MB of float64 per block
        for start in range(0, n - 1, chunk):
            b = min(chunk, n - 1 - start)
            i_blk = cols[start : start + b]
            d = x[None, :] - x[i_blk, None]  # d[a, j] = x_j - x_i
            if start + b < n:
                s += int(
                    np.sign(d[:, start + b :]).sum()
                )  # all j >= start+b satisfy j > i
            s += int(
                np.triu(np.sign(d[:, start : start + b]), k=1).sum()
            )  # start <= i < j < start+b

        unique, counts = np.unique(x, return_counts=True)
        tied_groups = counts[counts > 1]

        var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
        for t in tied_groups:
            var_s -= (t * (t - 1) * (2 * t + 5)) / 18.0

        if var_s <= 0:
            var_s = 1.0

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        p_value = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))

        significant = p_value < alpha
        if significant:
            trend = "increasing" if s > 0 else "decreasing"
        else:
            trend = "no trend"

        return {
            "s_statistic": int(s),
            "z_value": float(z),
            "p_value": float(p_value),
            "trend": trend,
            "significant": bool(significant),
            "alpha": alpha,
            "n_observations": n,
        }

    def sens_slope(
        self,
        time_series: np.ndarray,
    ) -> dict[str, float]:
        """Calculate Sen's slope estimator.

        Robust, non-parametric estimate of the trend magnitude.
        The median of all pairwise slopes between data points.

        slope_ij = (x_j - x_i) / (j - i) for all j > i

        Args:
            time_series: Observations ordered in time.

        Returns:
            Dictionary with median slope and confidence interval.
        """
        x = np.asarray(time_series, dtype=float)
        x = x[~np.isnan(x)]
        n = len(x)

        if n < 3:
            return {
                "median_slope": 0.0,
                "lower_ci": 0.0,
                "upper_ci": 0.0,
                "slope_per_decade": 0.0,
                "n_slopes": 0,
            }

        # Pairwise slopes slope_ij = (x_j - x_i) / (j - i) for all j > i,
        # built in row blocks to bound peak memory; the median and CI bounds
        # come from a single O(n_slopes) introselect instead of a full sort.
        cols = np.arange(n)
        chunk = max(1, int(2**23 // max(n, 1)))  # ~64MB of float64 per block
        slopes = np.empty(n * (n - 1) // 2)
        off = 0
        for start in range(0, n - 1, chunk):
            b = min(chunk, n - 1 - start)
            i_blk = cols[start : start + b]
            if start + b < n:
                jt = cols[start + b :]
                tail = (x[None, start + b :] - x[i_blk, None]) / (
                    jt[None, :] - i_blk[:, None]
                )
                m = tail.size
                slopes[off : off + m] = tail.ravel()
                off += m
            jc = cols[start : start + b]
            den = jc[None, :] - i_blk[:, None]
            if den.shape[0] == den.shape[1]:
                np.fill_diagonal(den, 1)  # masked by triu below; avoids 0-division
            inb = (x[None, start : start + b] - x[i_blk, None]) / den
            tri = inb[np.triu(np.ones((b, b), dtype=bool), k=1)]
            slopes[off : off + len(tri)] = tri
            off += len(tri)

        n_slopes = len(slopes)
        z_95 = 1.96
        c_alpha = z_95 * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
        m1 = int((n_slopes - c_alpha) / 2)
        m2 = int((n_slopes + c_alpha) / 2)
        k_mid = (n_slopes - 1) // 2  # lower middle rank
        i1, im, i2 = max(0, m1), k_mid, min(n_slopes - 1, m2)
        kths = [i1, im, min(n_slopes - 1, im + 1), i2]
        ordered = np.partition(slopes, kths)[kths]
        if n_slopes % 2 == 1:
            median_slope = float(ordered[1])
        else:
            median_slope = float((ordered[1] + ordered[2]) / 2.0)  # np.median semantics
        lower_ci = float(ordered[0])
        upper_ci = float(ordered[3])
        return {
            "median_slope": median_slope,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "slope_per_decade": float(median_slope * 10),
            "n_slopes": n_slopes,
        }

    def detect_changepoint(
        self,
        time_series: np.ndarray,
    ) -> dict[str, float]:
        """Detect a single changepoint using cumulative sum method.

        Finds the point that maximizes the absolute difference between
        the means of the two segments.

        Args:
            time_series: Observations ordered in time.

        Returns:
            Dictionary with changepoint index, means before/after, magnitude.
        """
        x = np.asarray(time_series, dtype=float)
        x = x[~np.isnan(x)]
        n = len(x)

        if n < 6:
            return {
                "changepoint_index": -1,
                "mean_before": float(np.mean(x)) if n > 0 else 0.0,
                "mean_after": float(np.mean(x)) if n > 0 else 0.0,
                "magnitude": 0.0,
            }

        cumsum = np.cumsum(x - np.mean(x))

        max_idx = int(np.argmax(np.abs(cumsum[2:-2]))) + 2

        mean_before = float(np.mean(x[:max_idx]))
        mean_after = float(np.mean(x[max_idx:]))
        magnitude = mean_after - mean_before

        return {
            "changepoint_index": max_idx,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "magnitude": float(magnitude),
            "n_observations": n,
        }

    def calculate_heat_island_effect(
        self,
        urban_temps: np.ndarray,
        rural_temps: np.ndarray,
    ) -> dict[str, float]:
        """Calculate urban heat island intensity.

        UHI = T_urban - T_rural

        Args:
            urban_temps: Urban temperature observations.
            rural_temps: Rural temperature observations.

        Returns:
            Dictionary with UHI statistics.
        """
        u = np.asarray(urban_temps, dtype=float)
        r = np.asarray(rural_temps, dtype=float)

        min_len = min(len(u), len(r))
        u = u[:min_len]
        r = r[:min_len]

        uhi = u - r

        return {
            "mean_uhi_c": float(np.mean(uhi)),
            "max_uhi_c": float(np.max(uhi)),
            "min_uhi_c": float(np.min(uhi)),
            "std_uhi_c": float(np.std(uhi)),
            "urban_mean_c": float(np.mean(u)),
            "rural_mean_c": float(np.mean(r)),
            "n_observations": min_len,
        }
