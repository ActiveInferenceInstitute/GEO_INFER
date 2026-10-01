"""
Statistical map visualization for GEO-INFER-SPM

This module provides functions for creating statistical parametric maps
and visualizing SPM analysis results as spatial plots and maps.
"""

import numpy as np
from typing import Any
import warnings
from scipy import stats

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Visualization functions limited.")

from ..models.data_models import SPMResult


def create_statistical_map(
    spm_result: SPMResult,
    contrast_idx: int = 0,
    threshold: float | None = None,
    colormap: str = "RdBu_r",
    title: str | None = None,
) -> dict[str, Any]:
    """
    Create statistical parametric map visualization.

    Args:
        spm_result: SPM analysis results
        contrast_idx: Index of contrast to visualize
        threshold: Statistical threshold for significance
        colormap: Colormap name
        title: Plot title

    Returns:
        Dictionary with visualization data
    """
    if not isinstance(contrast_idx, int) or contrast_idx < 0:
        raise ValueError("contrast_idx must be a non-negative integer")
    coordinates = np.asarray(spm_result.spm_data.coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or coordinates.shape[0] == 0:
        raise ValueError("spm_result coordinates must be a non-empty (n, >=2) array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("spm_result coordinates must contain only finite values")

    if contrast_idx < len(spm_result.contrasts):
        contrast = spm_result.contrasts[contrast_idx]

        # Get statistical values
        if hasattr(contrast, "t_statistic") and contrast.t_statistic.ndim == 1:
            stat_values = contrast.t_statistic
        else:
            stat_values = contrast.t_statistic.flatten()
        stat_values = np.asarray(stat_values, dtype=float).reshape(-1)
        if len(stat_values) != len(coordinates) or not np.all(np.isfinite(stat_values)):
            raise ValueError("contrast statistics must align with finite coordinates")

        # Apply threshold
        if threshold is None:
            threshold = contrast.threshold if hasattr(contrast, "threshold") else 0.05
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")

        significant_mask = None
        if (
            hasattr(contrast, "significance_mask")
            and contrast.significance_mask is not None
        ):
            significant_mask = np.asarray(
                contrast.significance_mask, dtype=bool
            ).reshape(-1)
        else:
            p_values = (
                contrast.p_values
                if hasattr(contrast, "p_values")
                else np.ones_like(stat_values)
            )
            p_values = np.asarray(p_values, dtype=float).reshape(-1)
            if len(p_values) != len(coordinates) or not np.all(np.isfinite(p_values)):
                raise ValueError("contrast p-values must align with finite coordinates")
            significant_mask = p_values < threshold
        if len(significant_mask) != len(coordinates):
            raise ValueError("significance mask must align with coordinates")

        contrast_name = getattr(contrast, "name", f"Contrast {contrast_idx}")
        correction_method = getattr(contrast, "correction_method", "uncorrected")
    elif contrast_idx == 0 and len(spm_result.beta_coefficients) > 0:
        # Fallback: use beta coefficients as stat_values when no contrasts and idx==0
        beta = spm_result.beta_coefficients
        beta = np.asarray(beta, dtype=float)
        if beta.ndim == 1 and len(beta) == len(coordinates):
            stat_values = beta
        elif beta.ndim == 2 and beta.shape[0] == len(coordinates):
            stat_values = beta[:, 0]
        elif beta.ndim == 2 and beta.shape[1] == len(coordinates):
            stat_values = beta[0]
        else:
            stat_values = np.full(len(coordinates), float(np.mean(beta)))
        if not np.all(np.isfinite(stat_values)):
            raise ValueError("beta coefficients must contain finite values")
        if threshold is None:
            threshold = 0.05
        significant_mask = None
        contrast_name = f"Contrast {contrast_idx}"
        correction_method = "uncorrected"
    else:
        raise ValueError(f"Contrast index {contrast_idx} out of range")

    # Create visualization data
    viz_data = {
        "coordinates": coordinates.tolist(),
        "stat_values": stat_values.tolist(),
        "significant_mask": (
            significant_mask.tolist() if significant_mask is not None else None
        ),
        "significance_mask": (
            significant_mask.tolist() if significant_mask is not None else None
        ),
        "threshold": threshold,
        "colormap": colormap,
        "title": title or f"SPM Contrast {contrast_idx}",
        "contrast_name": contrast_name,
        "correction_method": correction_method,
    }

    # Create matplotlib figure if available
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create scatter plot colored by statistic
        sc = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=stat_values,
            cmap=colormap,
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Highlight significant points
        if significant_mask is not None and np.any(significant_mask):
            sig_coords = coordinates[significant_mask]
            ax.scatter(
                sig_coords[:, 0],
                sig_coords[:, 1],
                c="red",
                s=60,
                marker="*",
                label="Significant",
            )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("T-statistic")

        # Set labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(viz_data["title"])
        ax.grid(True, alpha=0.3)

        # Add legend if significant points exist
        if significant_mask is not None and np.any(significant_mask):
            ax.legend()

        plt.tight_layout()

        viz_data["matplotlib_figure"] = fig

    return viz_data


def plot_spm_results(
    spm_result: SPMResult, plot_type: str = "stat_map", **kwargs: Any
) -> dict[str, Any]:
    """
    Create comprehensive SPM results visualization.

    Args:
        spm_result: SPM analysis results
        plot_type: Type of plot ('stat_map', 'beta_map', 'residuals', 'diagnostics')
        **kwargs: Additional plotting parameters

    Returns:
        Dictionary with visualization data
    """
    if not MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available for plotting"}

    if plot_type == "stat_map":
        return create_statistical_map(spm_result, **kwargs)

    elif plot_type == "beta_map":
        # Visualize regression coefficients
        return _plot_beta_coefficients(spm_result, **kwargs)

    elif plot_type == "residuals":
        # Visualize model residuals
        return _plot_residuals(spm_result, **kwargs)

    elif plot_type == "diagnostics":
        # Model diagnostic plots
        return _plot_model_diagnostics(spm_result, **kwargs)

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def _plot_beta_coefficients(spm_result: SPMResult, **kwargs: Any) -> dict[str, Any]:
    """Plot regression coefficient maps."""
    beta = spm_result.beta_coefficients
    coordinates = spm_result.spm_data.coordinates

    if beta.ndim == 1:
        # Single coefficient set
        fig, ax = plt.subplots(figsize=(8, 6))
        # If beta has same length as coordinates, use as color values; else fill with mean
        if len(beta) == len(coordinates):
            c_vals = beta
        else:
            c_vals = np.full(len(coordinates), float(np.mean(beta)))
        sc = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=c_vals,
            cmap="viridis",
            s=50,
            alpha=0.7,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Coefficient Value")
        ax.set_title("Regression Coefficients")
    else:
        # Multiple coefficients - create subplots
        n_coeffs = min(beta.shape[0], 9)  # Limit to 9 subplots
        n_cols = int(np.ceil(np.sqrt(n_coeffs)))
        n_rows = int(np.ceil(n_coeffs / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_coeffs == 1:
            axes = [axes]

        axes = axes.flatten()

        for i in range(n_coeffs):
            ax = axes[i]
            sc = ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=beta[i],
                cmap="viridis",
                s=30,
                alpha=0.7,
            )
            ax.set_title(f"β{i}")
            plt.colorbar(sc, ax=ax)

        # Hide unused subplots
        for i in range(n_coeffs, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()

    return {"plot_type": "beta_coefficients", "matplotlib_figure": fig}


def _plot_residuals(spm_result: SPMResult, **kwargs: Any) -> dict[str, Any]:
    """Plot model residuals."""
    residuals = spm_result.residuals
    coordinates = spm_result.spm_data.coordinates

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Spatial distribution of residuals
    sc = ax1.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=residuals,
        cmap="RdYlBu_r",
        s=50,
        alpha=0.7,
    )
    ax1.set_title("Residual Spatial Distribution")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    cbar1 = plt.colorbar(sc, ax=ax1)
    cbar1.set_label("Residual Value")

    # Residual histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Frequency")

    # Add normality test
    from scipy import stats

    _, p_value = stats.shapiro(residuals)
    ax2.text(
        0.05,
        0.95,
        f"Shapiro-Wilk p = {p_value:.3f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    return {
        "plot_type": "residuals",
        "matplotlib_figure": fig,
        "shapiro_p_value": p_value,
    }


def _plot_model_diagnostics(spm_result: SPMResult, **kwargs: Any) -> dict[str, Any]:
    """Create model diagnostic plots."""
    diagnostics = spm_result.model_diagnostics

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Q-Q plot for residuals
    ax1 = axes[0, 0]
    residuals = spm_result.residuals
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title("Normal Q-Q Plot")

    # 2. Residuals vs Fitted
    ax2 = axes[0, 1]
    fitted = spm_result.spm_data.data - residuals
    ax2.scatter(fitted, residuals, alpha=0.6)
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Fitted")

    # 3. Scale-Location plot
    ax3 = axes[1, 0]
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    ax3.scatter(fitted, sqrt_abs_residuals, alpha=0.6)
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("√|Residuals|")
    ax3.set_title("Scale-Location Plot")

    # 4. Cook's distance — Deferred: see docs/deferred_statistical_methods.md
    # ("Cook's distance (diagnostic plots)").
    ax4 = axes[1, 1]
    # Deferred: see docs/deferred_statistical_methods.md
    # ("Cook's distance (diagnostic plots)").
    n = len(residuals)
    p = spm_result.design_matrix.n_regressors
    mse = np.sum(residuals**2) / max(n - p, 1)
    design = np.asarray(spm_result.design_matrix.matrix, dtype=float)
    leverage = np.diag(design @ np.linalg.pinv(design))
    leverage = np.clip(leverage, 0.0, 1.0 - np.finfo(float).eps)
    cooks_d = (residuals**2 / max(p * mse, np.finfo(float).eps)) * (
        leverage / (1 - leverage) ** 2
    )

    ax4.scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
    ax4.axhline(y=4 / n, color="red", linestyle="--", label="4/n threshold")
    ax4.set_xlabel("Observation Index")
    ax4.set_ylabel("Cook's Distance")
    ax4.set_title("Cook's Distance")
    ax4.legend()

    plt.tight_layout()

    return {
        "plot_type": "diagnostics",
        "matplotlib_figure": fig,
        "diagnostic_stats": {
            "r_squared": diagnostics.get("r_squared"),
            "adjusted_r_squared": diagnostics.get("adjusted_r_squared"),
            "f_statistic": diagnostics.get("f_statistic"),
            "max_cooks_d": np.max(cooks_d),
        },
    }
