"""Precipitation analysis with intensity-duration-frequency curves.

Implements IDF curve fitting, rainfall depth estimation,
and precipitation statistics for hydrological design.
"""

import logging
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class PrecipitationAnalyzer:
    """Analyze precipitation patterns and extreme rainfall.

    Implements Intensity-Duration-Frequency (IDF) curve estimation,
    return period analysis, and rainfall distribution fitting.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize precipitation analyzer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def fit_idf_curve(
        self,
        annual_maxima: dict[float, np.ndarray],
    ) -> dict[float, dict[str, Any]]:
        """Fit Intensity-Duration-Frequency curves from annual maximum series.

        For each duration, fits a Gumbel distribution to annual maxima
        and calculates intensities for standard return periods.

        IDF: i = a / (d + b)^c, where i=intensity, d=duration
        Fitted per return period using Gumbel distribution.

        Args:
            annual_maxima: Dictionary mapping duration (hours) to array
                of annual maximum rainfall depths (mm).

        Returns:
            Dictionary with IDF parameters and intensity tables.
        """
        return_periods = [2, 5, 10, 25, 50, 100]
        results = {}

        for duration_h, maxima in annual_maxima.items():
            data = np.asarray(maxima, dtype=float)
            data = data[~np.isnan(data)]
            n = len(data)

            if n < 3:
                continue

            mean_val = float(np.mean(data))
            std_val = float(np.std(data))

            beta = std_val * np.sqrt(6) / np.pi
            mu = mean_val - 0.5772 * beta

            intensities = {}
            for rp in return_periods:
                yt = -np.log(-np.log(1 - 1.0 / rp))
                depth_mm = mu + beta * yt
                intensity_mm_h = depth_mm / duration_h if duration_h > 0 else 0
                intensities[rp] = {
                    "depth_mm": float(max(0, depth_mm)),
                    "intensity_mm_h": float(max(0, intensity_mm_h)),
                }

            results[duration_h] = {
                "duration_hours": duration_h,
                "n_years": n,
                "gumbel_mu": float(mu),
                "gumbel_beta": float(beta),
                "mean_depth_mm": mean_val,
                "std_depth_mm": std_val,
                "return_period_intensities": intensities,
            }

        return results

    def gumbel_return_period(
        self,
        annual_maxima: np.ndarray,
        design_value: float,
    ) -> dict[str, float | None]:
        """Calculate return period for a given rainfall value using Gumbel distribution.

        Args:
            annual_maxima: Array of annual maximum values.
            design_value: Rainfall value to find return period for.

        Returns:
            Dictionary with return period and exceedance probability.
        """
        data = np.asarray(annual_maxima, dtype=float)
        data = data[~np.isnan(data)]

        if len(data) < 3:
            return {
                "return_period_years": 0.0,
                "exceedance_probability": 1.0,
                "design_value": float(design_value),
            }

        mean_val = float(np.mean(data))
        std_val = float(np.std(data))

        beta = std_val * np.sqrt(6) / np.pi
        mu = mean_val - 0.5772 * beta

        if beta <= 0:
            return {
                "return_period_years": float("inf"),
                "exceedance_probability": 0.0,
                "design_value": float(design_value),
            }

        z = (design_value - mu) / beta
        cdf = np.exp(-np.exp(-z))
        exceedance = 1.0 - cdf

        if exceedance <= 0:
            rp = float("inf")
        else:
            rp = 1.0 / exceedance

        return {
            "return_period_years": float(rp) if rp != float("inf") else None,
            "exceedance_probability": float(exceedance),
            "design_value": float(design_value),
            "gumbel_mu": float(mu),
            "gumbel_beta": float(beta),
        }

    def rainfall_depth_for_return_period(
        self,
        annual_maxima: np.ndarray,
        return_period_years: float,
    ) -> dict[str, float]:
        """Calculate design rainfall depth for a given return period.

        Args:
            annual_maxima: Array of annual maximum rainfall depths (mm).
            return_period_years: Design return period in years.

        Returns:
            Dictionary with design rainfall depth and distribution parameters.
        """
        data = np.asarray(annual_maxima, dtype=float)
        data = data[~np.isnan(data)]

        if len(data) < 3:
            return {
                "design_depth_mm": 0.0,
                "return_period_years": return_period_years,
            }

        mean_val = float(np.mean(data))
        std_val = float(np.std(data))

        beta = std_val * np.sqrt(6) / np.pi
        mu = mean_val - 0.5772 * beta

        yt = -np.log(-np.log(1.0 - 1.0 / return_period_years))
        depth = mu + beta * yt

        return {
            "design_depth_mm": float(max(0, depth)),
            "return_period_years": return_period_years,
            "exceedance_probability": 1.0 / return_period_years,
            "gumbel_mu": float(mu),
            "gumbel_beta": float(beta),
            "frequency_factor": float(yt),
        }

    def calculate_precipitation_statistics(
        self,
        daily_precip: np.ndarray,
    ) -> dict[str, float]:
        """Calculate standard precipitation statistics.

        Args:
            daily_precip: Array of daily precipitation values (mm).

        Returns:
            Dictionary with precipitation statistics.
        """
        p = np.asarray(daily_precip, dtype=float)
        p = p[~np.isnan(p)]

        wet_days = p[p >= 1.0]
        n_total = len(p)
        n_wet = len(wet_days)

        max_consecutive_dry = 0
        max_consecutive_wet = 0
        current_dry = 0
        current_wet = 0

        for val in p:
            if val < 1.0:
                current_dry += 1
                max_consecutive_dry = max(max_consecutive_dry, current_dry)
                current_wet = 0
            else:
                current_wet += 1
                max_consecutive_wet = max(max_consecutive_wet, current_wet)
                current_dry = 0

        return {
            "total_mm": float(np.sum(p)),
            "mean_daily_mm": float(np.mean(p)),
            "max_daily_mm": float(np.max(p)) if len(p) > 0 else 0.0,
            "std_daily_mm": float(np.std(p)),
            "wet_day_count": n_wet,
            "dry_day_count": n_total - n_wet,
            "wet_day_fraction": float(n_wet / n_total) if n_total > 0 else 0.0,
            "mean_wet_day_mm": float(np.mean(wet_days)) if n_wet > 0 else 0.0,
            "max_consecutive_dry_days": max_consecutive_dry,
            "max_consecutive_wet_days": max_consecutive_wet,
            "percentile_95_mm": float(np.percentile(p, 95)) if len(p) > 0 else 0.0,
            "percentile_99_mm": float(np.percentile(p, 99)) if len(p) > 0 else 0.0,
            "n_days": n_total,
        }

    def fit_gamma_distribution(
        self,
        wet_day_precip: np.ndarray,
    ) -> dict[str, float]:
        """Fit gamma distribution to wet-day precipitation.

        Used as a basis for SPI calculation and rainfall modeling.

        Args:
            wet_day_precip: Precipitation on wet days only (mm, all > 0).

        Returns:
            Dictionary with gamma distribution parameters.
        """
        data = np.asarray(wet_day_precip, dtype=float)
        data = data[data > 0]

        if len(data) < 3:
            return {
                "alpha": 0.0,
                "beta": 0.0,
                "mean": 0.0,
                "variance": 0.0,
            }

        try:
            alpha, loc, beta = scipy_stats.gamma.fit(data, floc=0)
        except Exception:
            mean_val = float(np.mean(data))
            var_val = float(np.var(data))
            if var_val > 0:
                alpha = mean_val**2 / var_val
                beta = var_val / mean_val
            else:
                alpha = 1.0
                beta = mean_val
            loc = 0.0

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "loc": float(loc),
            "mean": float(alpha * beta),
            "variance": float(alpha * beta**2),
            "n_observations": len(data),
        }
