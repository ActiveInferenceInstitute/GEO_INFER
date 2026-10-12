"""
Visualization utilities for Bayesian geospatial models.

This module provides visualization functions for posterior
distributions, spatial predictions, and uncertainty quantification.
"""

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _finite_vector(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Return a non-empty finite one-dimensional plotting vector."""
    vector = np.asarray(values, dtype=float)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    vector = vector.reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_spatial_inputs(
    spatial_coords: np.ndarray,
    predictions: np.ndarray,
    observations: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate the aligned arrays used by spatial prediction plots."""
    coords = np.asarray(spatial_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
        raise ValueError("spatial_coords must be a non-empty (n, >=2) array")
    if not np.all(np.isfinite(coords)):
        raise ValueError("spatial_coords must contain only finite values")

    predicted = _finite_vector(predictions, "predictions")
    if len(predicted) != len(coords):
        raise ValueError("predictions and coordinates must have the same length")

    observed = (
        None if observations is None else _finite_vector(observations, "observations")
    )
    if observed is not None and len(observed) != len(coords):
        raise ValueError("observations must have one value per spatial coordinate")

    spread = None if uncertainty is None else _finite_vector(uncertainty, "uncertainty")
    if spread is not None:
        if len(spread) != len(coords):
            raise ValueError("uncertainty must have one value per spatial coordinate")
        if np.any(spread < 0):
            raise ValueError("uncertainty must be non-negative")

    return coords[:, :2], predicted, observed, spread


def _save_figure(fig: plt.Figure, output_path: str | None) -> None:
    """Save a figure when requested, creating its parent directory."""
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")


def plot_posterior(
    samples: dict[str, np.ndarray], parameters: list[str] | None = None
) -> plt.Figure:
    """
    Plot posterior distributions for model parameters.

    Args:
        samples: Dictionary of parameter samples
        parameters: List of parameters to plot (if None, plot all)

    Returns:
        Matplotlib figure object
    """
    if not samples:
        raise ValueError("samples must contain at least one parameter")
    if parameters is None:
        parameters = list(samples.keys())
    if not parameters:
        raise ValueError("parameters must contain at least one parameter")
    missing = [parameter for parameter in parameters if parameter not in samples]
    if missing:
        raise KeyError(f"Unknown posterior parameters: {missing}")

    n_params = len(parameters)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 4 * n_params))

    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, param in enumerate(parameters):
        param_samples = _finite_vector(samples[param], f"samples[{param!r}]")

        # Histogram
        axes[i, 0].hist(param_samples, bins=50, density=True, alpha=0.7)
        axes[i, 0].set_xlabel(param)
        axes[i, 0].set_ylabel("Density")
        axes[i, 0].set_title(f"Posterior Distribution: {param}")

        # Trace plot (simplified)
        axes[i, 1].plot(param_samples)
        axes[i, 1].set_xlabel("Sample Index")
        axes[i, 1].set_ylabel(param)
        axes[i, 1].set_title(f"Trace Plot: {param}")

    fig.tight_layout()
    return fig


def plot_spatial_prediction(
    spatial_coords: np.ndarray,
    predictions: np.ndarray,
    observations: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> plt.Figure:
    """
    Plot spatial predictions with optional uncertainty.

    Args:
        spatial_coords: Spatial coordinates
        predictions: Predicted values
        observations: Observed values (optional)
        uncertainty: Prediction uncertainty (optional)

    Returns:
        Matplotlib figure object
    """
    coords, predicted, observed, spread = _validate_spatial_inputs(
        spatial_coords, predictions, observations, uncertainty
    )
    fig, axes = plt.subplots(1, 2 if spread is not None else 1, figsize=(15, 6))
    axes = np.atleast_1d(axes)

    # Mean prediction
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=predicted, cmap="viridis", s=50)
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    axes[0].set_title("Mean Predictions")
    plt.colorbar(sc, ax=axes[0])

    # Observations
    if observed is not None:
        axes[0].scatter(
            coords[:, 0], coords[:, 1], c="red", marker="x", s=30, label="Observations"
        )
        axes[0].legend()

    # Uncertainty
    if spread is not None:
        sc_unc = axes[1].scatter(
            coords[:, 0], coords[:, 1], c=spread, cmap="plasma", s=50
        )
        axes[1].set_xlabel("X coordinate")
        axes[1].set_ylabel("Y coordinate")
        axes[1].set_title("Prediction Uncertainty")
        plt.colorbar(sc_unc, ax=axes[1])

    fig.tight_layout()
    return fig


def plot_uncertainty(
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    confidence_level: float = 0.95,
) -> plt.Figure:
    """
    Plot prediction uncertainty with confidence intervals.

    Args:
        predictions: Predicted mean values
        uncertainty: Prediction standard deviations
        confidence_level: Confidence level for intervals

    Returns:
        Matplotlib figure object
    """
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and between 0 and 1")
    mean = _finite_vector(predictions, "predictions")
    spread = _finite_vector(uncertainty, "uncertainty")
    if len(mean) != len(spread):
        raise ValueError("predictions and uncertainty must have the same length")
    if np.any(spread < 0):
        raise ValueError("uncertainty must be non-negative")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute confidence intervals
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower_bound = mean - z_score * spread
    upper_bound = mean + z_score * spread

    # Plot mean predictions
    ax.plot(mean, "b-", linewidth=2, label="Predictions")

    # Plot confidence intervals
    ax.fill_between(
        range(len(predictions)),
        lower_bound,
        upper_bound,
        alpha=0.3,
        color="blue",
        label=f"{confidence_level:.0%} Confidence Interval",
    )

    ax.set_xlabel("Location Index")
    ax.set_ylabel("Value")
    ax.set_title("Predictions with Uncertainty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_model_comparison(
    models: list[str], metrics: dict[str, list[float]]
) -> plt.Figure:
    """
    Plot comparison of different models.

    Args:
        models: List of model names
        metrics: Dictionary of metrics for each model

    Returns:
        Matplotlib figure object
    """
    if not models:
        raise ValueError("models must contain at least one model")
    if not metrics:
        raise ValueError("metrics must contain at least one metric")
    validated_metrics: dict[str, np.ndarray] = {}
    for metric_name, values in metrics.items():
        metric_values = _finite_vector(values, f"metrics[{metric_name!r}]")
        if len(metric_values) != len(models):
            raise ValueError(f"Metric {metric_name!r} must have one value per model")
        validated_metrics[metric_name] = metric_values

    fig, axes = plt.subplots(
        1, len(validated_metrics), figsize=(6 * len(validated_metrics), 5)
    )
    ax_list: np.ndarray = np.atleast_1d(axes)

    for i, (metric_name, val_array) in enumerate(validated_metrics.items()):
        ax_list[i].bar(models, val_array)
        ax_list[i].set_title(f"Model Comparison: {metric_name}")
        ax_list[i].set_ylabel(metric_name)
        ax_list[i].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig
