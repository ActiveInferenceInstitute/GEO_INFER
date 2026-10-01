"""
Diagnostic visualization tools for SPM analysis

This module provides functions for creating diagnostic plots and
visual assessments of SPM model fit and statistical assumptions.
"""

import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..models.data_models import SPMResult, ContrastResult  # noqa: E402


def plot_model_diagnostics(
    spm_result: SPMResult, figsize: tuple[int, int] = (12, 10)
) -> dict[str, Any]:
    """
    Create comprehensive model diagnostic plots.

    Args:
        spm_result: SPM analysis results
        figsize: Figure size (width, height)

    Returns:
        Dictionary with diagnostic plots and statistics
    """
    if not MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Residual Q-Q plot
    _plot_qq_residuals(spm_result, axes[0, 0])

    # 2. Residuals vs Fitted
    _plot_residuals_vs_fitted(spm_result, axes[0, 1])

    # 3. Scale-Location plot
    _plot_scale_location(spm_result, axes[0, 2])

    # 4. Residual histogram
    _plot_residual_histogram(spm_result, axes[1, 0])

    # 5. Cook's distance
    _plot_cooks_distance(spm_result, axes[1, 1])

    # 6. Leverage plot
    _plot_leverage(spm_result, axes[1, 2])

    plt.tight_layout()

    # Compute diagnostic statistics
    diagnostics = _compute_diagnostic_stats(spm_result)

    return {"matplotlib_figure": fig, "diagnostic_statistics": diagnostics}


def plot_contrast_results(
    contrast_result: ContrastResult, figsize: tuple[int, int] = (10, 6)
) -> dict[str, Any]:
    """
    Create plots for contrast analysis results.

    Args:
        contrast_result: Contrast analysis results
        figsize: Figure size

    Returns:
        Dictionary with contrast plots
    """
    if not MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # T-statistic distribution
    ax1.hist(contrast_result.t_statistic, bins=30, alpha=0.7, edgecolor="black")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_title("T-Statistic Distribution")
    ax1.set_xlabel("T-statistic")
    ax1.set_ylabel("Frequency")

    # P-value distribution
    ax2.hist(contrast_result.p_values, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(x=0.05, color="red", linestyle="--", alpha=0.7, label="α = 0.05")
    ax2.set_title("P-Value Distribution")
    ax2.set_xlabel("P-value")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()

    return {
        "matplotlib_figure": fig,
        "n_significant": contrast_result.n_significant,
        "correction_method": contrast_result.correction_method,
    }


def _plot_qq_residuals(spm_result: SPMResult, ax: Any) -> None:
    """Plot Q-Q plot for residuals."""
    from scipy import stats

    residuals = spm_result.residuals
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot\n(Residuals)")

    # Add R² for normality
    _, (slope, intercept, r_value) = stats.probplot(residuals, dist="norm", plot=None)
    r_squared = r_value**2
    ax.text(
        0.05,
        0.95,
        f"R² = {r_squared:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _plot_residuals_vs_fitted(spm_result: SPMResult, ax: Any) -> None:
    """Plot residuals vs fitted values."""
    residuals = spm_result.residuals
    fitted = spm_result.spm_data.data - residuals

    ax.scatter(fitted, residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # Add smoothed line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(residuals, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="blue", linewidth=2, alpha=0.8)
    except ImportError:
        logger.debug("statsmodels is unavailable; skipping optional diagnostic overlay")


def _plot_scale_location(spm_result: SPMResult, ax: Any) -> None:
    """Plot scale-location plot."""
    residuals = spm_result.residuals
    fitted = spm_result.spm_data.data - residuals

    sqrt_abs_residuals = np.sqrt(np.abs(residuals))

    ax.scatter(fitted, sqrt_abs_residuals, alpha=0.6, s=20)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Residuals|")
    ax.set_title("Scale-Location Plot")

    # Add smoothed line
    try:
        from scipy.stats import lowess

        smoothed = lowess(sqrt_abs_residuals, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="blue", linewidth=2, alpha=0.8)
    except ImportError:
        logger.debug("statsmodels is unavailable; skipping optional diagnostic overlay")


def _plot_residual_histogram(spm_result: SPMResult, ax: Any) -> None:
    """Plot residual histogram with normal distribution overlay."""
    residuals = spm_result.residuals

    # Histogram
    n, bins, patches = ax.hist(
        residuals, bins=30, alpha=0.7, density=True, edgecolor="black", linewidth=0.5
    )

    # Normal distribution overlay
    mean, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(bins[0], bins[-1], 100)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.plot(x, y, "r-", linewidth=2, alpha=0.8, label="Normal fit")

    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Density")
    ax.legend()

    # Add Shapiro-Wilk test
    try:
        from scipy.stats import shapiro

        stat, p_value = shapiro(residuals)
        ax.text(
            0.05,
            0.95,
            f"Shapiro-Wilk\np = {p_value:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    except ImportError:
        logger.debug("statsmodels is unavailable; skipping optional diagnostic overlay")


def _plot_cooks_distance(spm_result: SPMResult, ax: Any) -> None:
    """Plot Cook's distance for influence diagnostics."""
    residuals = spm_result.residuals
    X = spm_result.design_matrix.matrix

    n, p = X.shape
    mse = np.sum(residuals**2) / (n - p)

    # Leverage (diagonal of hat matrix)
    hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
    leverage = np.diag(hat_matrix)

    # Cook's distance
    cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage) ** 2)

    ax.scatter(range(len(cooks_d)), cooks_d, alpha=0.6, s=20)
    ax.axhline(
        y=4 / n, color="red", linestyle="--", alpha=0.7, label=f"4/n = {4 / n:.3f}"
    )
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Influence (Cook's Distance)")
    ax.legend()

    # Mark influential points
    influential = cooks_d > 4 / n
    if np.any(influential):
        ax.scatter(
            np.where(influential)[0],
            cooks_d[influential],
            color="red",
            s=40,
            marker="x",
            linewidth=2,
        )


def _plot_leverage(spm_result: SPMResult, ax: Any) -> None:
    """Plot leverage vs standardized residuals."""
    residuals = spm_result.residuals
    X = spm_result.design_matrix.matrix

    n, p = X.shape

    # Leverage
    hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
    leverage = np.diag(hat_matrix)

    # Standardized residuals
    mse = np.sum(residuals**2) / (n - p)
    std_residuals = residuals / np.sqrt(mse * (1 - leverage))

    ax.scatter(leverage, std_residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7)
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.7, label="±2 SD")
    ax.axhline(y=-2, color="red", linestyle="--", alpha=0.7)

    # Leverage threshold
    leverage_threshold = 2 * p / n
    ax.axvline(
        x=leverage_threshold,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"2p/n = {leverage_threshold:.3f}",
    )

    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized Residuals")
    ax.set_title("Residuals vs Leverage")
    ax.legend()

    # Mark high leverage points
    high_leverage = leverage > leverage_threshold
    if np.any(high_leverage):
        ax.scatter(
            leverage[high_leverage],
            std_residuals[high_leverage],
            color="red",
            s=40,
            marker="x",
            linewidth=2,
        )


def _compute_diagnostic_stats(spm_result: SPMResult) -> dict[str, Any]:
    """Compute comprehensive diagnostic statistics."""
    from scipy import stats

    residuals = spm_result.residuals
    X = spm_result.design_matrix.matrix
    n, p = X.shape

    # Basic statistics
    stats_dict = {
        "n_observations": n,
        "n_parameters": p,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_skewness": float(stats.skew(residuals)),
        "residual_kurtosis": float(stats.kurtosis(residuals)),
    }

    # Normality tests
    try:
        from scipy.stats import shapiro, normaltest

        _, shapiro_p = shapiro(residuals)
        _, normal_p = normaltest(residuals)
        stats_dict["shapiro_normality_p"] = float(shapiro_p)
        stats_dict["dagostino_normality_p"] = float(normal_p)
    except ImportError:
        logger.debug("statsmodels is unavailable; skipping optional diagnostic overlay")

    # Leverage and influence
    hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
    leverage = np.diag(hat_matrix)
    stats_dict["mean_leverage"] = float(np.mean(leverage))
    stats_dict["max_leverage"] = float(np.max(leverage))

    # Cook's distance
    mse = np.sum(residuals**2) / (n - p)
    cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage) ** 2)
    stats_dict["max_cooks_distance"] = float(np.max(cooks_d))
    stats_dict["n_influential_points"] = int(np.sum(cooks_d > 4 / n))

    return stats_dict
