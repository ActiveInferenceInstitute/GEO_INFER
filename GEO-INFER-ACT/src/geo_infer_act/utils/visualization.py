"""
Comprehensive visualization utilities for active inference models.

This module provides sophisticated visualization tools for active inference
components including beliefs, policies, free energy dynamics, and model
interpretability dashboards.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from pathlib import Path
import logging

import h3
from matplotlib.lines import Line2D

# Optional interactive/geometry backends (``backends`` extra)

try:
    from shapely.geometry import Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _belief_vector(payload: Any, name: str) -> np.ndarray:
    """Extract and validate a finite non-negative belief vector."""
    if isinstance(payload, dict):
        for key in ("states", "mean", "beliefs"):
            if key in payload:
                payload = payload[key]
                break
        else:
            if not payload:
                raise ValueError(f"{name} must not be empty")
            payload = next(iter(payload.values()))
    values = np.asarray(payload, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.any(values > 0):
        raise ValueError(f"{name} must contain positive probability mass")
    return values


def _finite_series(values: Any, name: str) -> np.ndarray:
    """Return a non-empty finite one-dimensional numeric series."""
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(series)):
        raise ValueError(f"{name} must contain only finite values")
    return series


def plot_belief_update(
    beliefs_before: dict[str, np.ndarray],
    beliefs_after: dict[str, np.ndarray],
    state_labels: list[str] | None = None,
    title: str = "Belief Update",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot belief updates with enhanced visualization.

    Args:
        beliefs_before: Beliefs before update
        beliefs_after: Beliefs after update
        state_labels: Optional labels for states
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    before = _belief_vector(
        beliefs_before.get("states", next(iter(beliefs_before.values()), None)),
        "beliefs_before",
    )
    after = _belief_vector(
        beliefs_after.get("states", next(iter(beliefs_after.values()), None)),
        "beliefs_after",
    )
    if len(before) != len(after):
        raise ValueError("beliefs_before and beliefs_after must have the same length")

    n_states = len(before)
    if state_labels is None:
        state_labels = [f"State {i}" for i in range(n_states)]
    if len(state_labels) != n_states:
        raise ValueError("state_labels must have one label per state")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Extracted belief arrays are validated above.

    x = np.arange(n_states)
    width = 0.35

    # Before vs After comparison
    bars1 = ax1.bar(
        x - width / 2, before, width, label="Before", alpha=0.8, color="lightblue"
    )
    bars2 = ax1.bar(
        x + width / 2, after, width, label="After", alpha=0.8, color="darkblue"
    )

    ax1.set_xlabel("State")
    ax1.set_ylabel("Belief Probability")
    ax1.set_title("Belief Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Belief change visualization
    change = after - before
    colors = ["red" if c < 0 else "green" for c in change]
    ax2.bar(x, change, color=colors, alpha=0.7)
    ax2.set_xlabel("State")
    ax2.set_ylabel("Belief Change")
    ax2.set_title("Belief Changes")
    ax2.set_xticks(x)
    ax2.set_xticklabels(state_labels, rotation=45)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Entropy comparison
    entropy_before = -np.sum(before * np.log(before + 1e-8))
    entropy_after = -np.sum(after * np.log(after + 1e-8))

    ax3.bar(
        ["Before", "After"],
        [entropy_before, entropy_after],
        color=["lightcoral", "darkred"],
        alpha=0.8,
    )
    ax3.set_ylabel("Entropy")
    ax3.set_title("Belief Entropy")
    ax3.grid(True, alpha=0.3)

    # Add entropy values
    ax3.text(
        0,
        entropy_before + 0.05,
        f"{entropy_before:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    ax3.text(
        1,
        entropy_after + 0.05,
        f"{entropy_after:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_free_energy(
    free_energy_history: list[float],
    iterations: np.ndarray | None = None,
    title: str = "Free Energy Minimization",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot free energy evolution with enhanced analysis.

    Args:
        free_energy_history: History of free energy values
        iterations: Optional iteration numbers
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if len(free_energy_history) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No free energy data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    fe_array = _finite_series(free_energy_history, "free_energy_history")
    if iterations is None:
        iterations = np.arange(len(fe_array))
    else:
        iterations = _finite_series(iterations, "iterations")
        if len(iterations) != len(fe_array):
            raise ValueError("iterations must have one value per free-energy sample")

    # Main free energy plot
    axes[0, 0].plot(
        iterations, fe_array, linewidth=2, color="red", marker="o", markersize=4
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Free Energy")
    axes[0, 0].set_title("Free Energy Evolution")
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line
    if len(fe_array) > 1:
        z = np.polyfit(iterations, fe_array, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(
            iterations,
            p(iterations),
            "--",
            alpha=0.8,
            color="blue",
            label=f"Trend (slope={z[0]:.4f})",
        )
        axes[0, 0].legend()

    # Free energy gradient
    if len(fe_array) > 1:
        gradient = np.gradient(fe_array)
        axes[0, 1].plot(
            iterations[1:],
            gradient[1:],
            linewidth=2,
            color="green",
            marker="s",
            markersize=3,
        )
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Free Energy Gradient")
        axes[0, 1].set_title("Rate of Change")
        axes[0, 1].grid(True, alpha=0.3)

    # Cumulative reduction
    if len(fe_array) > 0:
        cumulative_reduction = fe_array[0] - fe_array
        axes[1, 0].plot(
            iterations,
            cumulative_reduction,
            linewidth=2,
            color="purple",
            marker="^",
            markersize=3,
        )
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Cumulative Reduction")
        axes[1, 0].set_title("Free Energy Reduction")
        axes[1, 0].grid(True, alpha=0.3)

    # Distribution and statistics
    axes[1, 1].hist(
        fe_array, bins=max(1, min(20, len(fe_array) // 2)), alpha=0.7, color="orange"
    )
    axes[1, 1].axvline(
        np.mean(fe_array),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(fe_array):.3f}",
    )
    axes[1, 1].axvline(
        np.median(fe_array),
        color="blue",
        linestyle="--",
        label=f"Median: {np.median(fe_array):.3f}",
    )
    axes[1, 1].set_xlabel("Free Energy")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_policies(
    policy_probabilities: np.ndarray,
    policy_labels: list[str] | None = None,
    expected_free_energies: np.ndarray | None = None,
    title: str = "Policy Evaluation",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot policy analysis with enhanced visualization.

    Args:
        policy_probabilities: Policy probability matrix (time x policies)
        policy_labels: Optional policy labels
        expected_free_energies: Optional expected free energies
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    policy_probabilities = np.asarray(policy_probabilities, dtype=float)
    if policy_probabilities.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No policy data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    if policy_probabilities.ndim not in (1, 2):
        raise ValueError("policy_probabilities must be a one- or two-dimensional array")
    if not np.all(np.isfinite(policy_probabilities)) or np.any(
        policy_probabilities < 0
    ):
        raise ValueError("policy_probabilities must contain finite non-negative values")
    if policy_probabilities.ndim == 1:
        # Single time step
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        n_policies = len(policy_probabilities)

        if policy_labels is None:
            policy_labels = [f"Policy {i}" for i in range(n_policies)]
        if len(policy_labels) != n_policies:
            raise ValueError("policy_labels must have one label per policy")
        if expected_free_energies is not None:
            expected_free_energies = _finite_series(
                expected_free_energies, "expected_free_energies"
            )
            if len(expected_free_energies) != n_policies:
                raise ValueError(
                    "expected_free_energies must have one value per policy"
                )

        # Policy probabilities
        bars = axes[0].bar(
            range(n_policies),
            policy_probabilities,
            alpha=0.8,
            color=plt.get_cmap("viridis")(np.linspace(0, 1, n_policies)),
        )
        axes[0].set_xlabel("Policy")
        axes[0].set_ylabel("Probability")
        axes[0].set_title("Policy Probabilities")
        axes[0].set_xticks(range(n_policies))
        axes[0].set_xticklabels(policy_labels, rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Add probability labels
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Expected free energies if provided
        if expected_free_energies is not None:
            axes[1].bar(
                range(n_policies), expected_free_energies, alpha=0.8, color="red"
            )
            axes[1].set_xlabel("Policy")
            axes[1].set_ylabel("Expected Free Energy")
            axes[1].set_title("Expected Free Energies")
            axes[1].set_xticks(range(n_policies))
            axes[1].set_xticklabels(policy_labels, rotation=45)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No EFE data available",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )

        # Policy entropy
        entropy = -np.sum(policy_probabilities * np.log(policy_probabilities + 1e-8))
        max_entropy = np.log(n_policies)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Create entropy gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        axes[2].plot(x, y, "k-", linewidth=2)
        axes[2].fill_between(x, 0, y, alpha=0.2, color="lightgray")

        # Entropy indicator
        entropy_angle = np.pi * (1 - norm_entropy)
        entropy_x = r * np.cos(entropy_angle)
        entropy_y = r * np.sin(entropy_angle)

        axes[2].arrow(
            0,
            0,
            entropy_x,
            entropy_y,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
            linewidth=3,
        )

        axes[2].set_xlim(-1.2, 1.2)
        axes[2].set_ylim(-0.2, 1.2)
        axes[2].set_aspect("equal")
        axes[2].set_title(
            f"Policy Entropy: {entropy:.3f}\n(Normalized: {norm_entropy:.3f})"
        )
        axes[2].text(0, -0.1, "Deterministic", ha="center", va="top")
        axes[2].text(-1, 0.8, "Random", ha="center", va="center", rotation=45)
        axes[2].text(1, 0.8, "Random", ha="center", va="center", rotation=-45)
        axes[2].axis("off")

    else:
        # Multiple time steps
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        n_timesteps, n_policies = policy_probabilities.shape
        if policy_labels is None:
            policy_labels = [f"Policy {i}" for i in range(n_policies)]
        if n_timesteps == 0 or n_policies == 0:
            raise ValueError(
                "policy_probabilities must have at least one timestep and policy"
            )
        if len(policy_labels) != n_policies:
            raise ValueError("policy_labels must have one label per policy")

        # Policy evolution
        for i in range(n_policies):
            axes[0, 0].plot(
                policy_probabilities[:, i],
                label=policy_labels[i],
                linewidth=2,
                marker="o",
                markersize=3,
            )
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Probability")
        axes[0, 0].set_title("Policy Probability Evolution")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(True, alpha=0.3)

        # Policy heatmap
        im = axes[0, 1].imshow(
            policy_probabilities.T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Policy")
        axes[0, 1].set_title("Policy Selection Heatmap")
        axes[0, 1].set_yticks(range(n_policies))
        axes[0, 1].set_yticklabels(policy_labels)
        plt.colorbar(im, ax=axes[0, 1], label="Probability")

        # Policy entropy over time
        entropies = [
            -np.sum(probs * np.log(probs + 1e-8)) for probs in policy_probabilities
        ]
        axes[1, 0].plot(
            entropies, linewidth=2, color="purple", marker="s", markersize=3
        )
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("Entropy")
        axes[1, 0].set_title("Policy Selection Entropy")
        axes[1, 0].grid(True, alpha=0.3)

        # Average policy distribution
        avg_probs = np.mean(policy_probabilities, axis=0)
        bars = axes[1, 1].bar(
            range(n_policies),
            avg_probs,
            alpha=0.8,
            color=plt.get_cmap("viridis")(np.linspace(0, 1, n_policies)),
        )
        axes[1, 1].set_xlabel("Policy")
        axes[1, 1].set_ylabel("Average Probability")
        axes[1, 1].set_title("Average Policy Distribution")
        axes[1, 1].set_xticks(range(n_policies))
        axes[1, 1].set_xticklabels(policy_labels, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_perception_analysis(
    beliefs_history: list[np.ndarray],
    observations_history: list[np.ndarray],
    output_dir: Path,
    title: str = "Perception Analysis",
) -> None:
    """
    Create comprehensive perception analysis plots.

    Args:
        beliefs_history: History of beliefs
        observations_history: History of observations
        output_dir: Output directory for plots
        title: Overall title for the analysis
    """
    if not beliefs_history or not observations_history:
        logger.warning("Insufficient data for perception analysis")
        return

    # Create comprehensive perception analysis
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    beliefs_array = np.array(beliefs_history)

    # 1. Belief evolution heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(
        beliefs_array.T, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax1.set_title("Belief Evolution Heatmap")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("State")
    plt.colorbar(im1, ax=ax1, label="Belief Probability")

    # 2. Belief entropy over time
    ax2 = fig.add_subplot(gs[0, 2])
    entropies = [-np.sum(b * np.log(b + 1e-8)) for b in beliefs_array]
    ax2.plot(entropies, linewidth=2, color="purple", marker="o", markersize=3)
    ax2.set_title("Belief Entropy")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Entropy")
    ax2.grid(True, alpha=0.3)

    # 3. Dominant state evolution
    ax3 = fig.add_subplot(gs[0, 3])
    dominant_states = np.argmax(beliefs_array, axis=1)
    ax3.plot(dominant_states, linewidth=2, color="red", marker="s", markersize=3)
    ax3.set_title("Dominant State")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("State Index")
    ax3.grid(True, alpha=0.3)

    # 4. Belief changes over time
    ax4 = fig.add_subplot(gs[1, :2])
    if len(beliefs_array) > 1:
        belief_changes = np.diff(beliefs_array, axis=0)
        change_magnitudes = np.linalg.norm(belief_changes, axis=1)
        ax4.plot(
            change_magnitudes, linewidth=2, color="green", marker="^", markersize=3
        )
        ax4.set_title("Belief Change Magnitude")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Change Magnitude")
        ax4.grid(True, alpha=0.3)

    # 5. State probability distributions
    ax5 = fig.add_subplot(gs[1, 2])
    for i in range(min(3, beliefs_array.shape[1])):  # Show first 3 states
        ax5.plot(beliefs_array[:, i], label=f"State {i}", linewidth=2)
    ax5.set_title("State Probabilities")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Probability")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Observation pattern analysis
    ax6 = fig.add_subplot(gs[1, 3])
    obs_array = np.array(observations_history)
    if obs_array.ndim > 1:
        # Multi-dimensional observations
        for i in range(min(3, obs_array.shape[1])):
            ax6.plot(obs_array[:, i], label=f"Obs {i}", linewidth=2)
        ax6.legend()
    else:
        # Single dimensional observations
        ax6.plot(obs_array, linewidth=2, color="brown")
    ax6.set_title("Observation Patterns")
    ax6.set_xlabel("Time Step")
    ax6.set_ylabel("Observation")
    ax6.grid(True, alpha=0.3)

    # 7. Belief-observation correlation
    ax7 = fig.add_subplot(gs[2, :2])
    try:
        # Compute cross-correlation for visualization
        max_states_to_show = min(3, beliefs_array.shape[1])
        correlation_matrix = np.zeros(
            (max_states_to_show, obs_array.shape[1] if obs_array.ndim > 1 else 1)
        )

        for i in range(max_states_to_show):
            belief_series = beliefs_array[:, i]
            if obs_array.ndim > 1:
                for j in range(obs_array.shape[1]):
                    obs_series = obs_array[:, j]
                    if len(belief_series) == len(obs_series) and len(belief_series) > 1:
                        corr = np.corrcoef(belief_series, obs_series)[0, 1]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
            else:
                if len(belief_series) == len(obs_array) and len(belief_series) > 1:
                    corr = np.corrcoef(belief_series, obs_array)[0, 1]
                    correlation_matrix[i, 0] = corr if not np.isnan(corr) else 0

        im7 = ax7.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax7.set_title("Belief-Observation Correlations")
        ax7.set_xlabel("Observation Dimension")
        ax7.set_ylabel("Belief State")
        plt.colorbar(im7, ax=ax7, label="Correlation")

        # Add correlation values as text
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                ax7.text(
                    j,
                    i,
                    f"{correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                )
    except Exception as e:
        ax7.text(
            0.5,
            0.5,
            f"Correlation analysis failed: {str(e)[:50]}...",
            ha="center",
            va="center",
            transform=ax7.transAxes,
        )

    # 8. Perception quality metrics
    ax8 = fig.add_subplot(gs[2, 2:])

    # Calculate quality metrics
    quality_metrics = {
        "Responsiveness": 0.0,
        "Stability": 0.0,
        "Adaptability": 0.0,
        "Consistency": 0.0,
    }

    # Responsiveness: how much beliefs change with observations
    if len(beliefs_array) > 1:
        belief_changes = np.diff(beliefs_array, axis=0)
        responsiveness = np.mean(np.linalg.norm(belief_changes, axis=1))
        quality_metrics["Responsiveness"] = min(
            1.0, responsiveness * 10
        )  # Scale to 0-1

    # Stability: consistency of belief changes
    if len(beliefs_array) > 2:
        belief_change_vars = np.var(np.linalg.norm(belief_changes, axis=1))
        stability = 1.0 / (1.0 + belief_change_vars)
        quality_metrics["Stability"] = stability

    # Adaptability: entropy changes
    if len(entropies) > 1:
        entropy_changes = np.abs(np.diff(entropies))
        adaptability = np.mean(entropy_changes)
        quality_metrics["Adaptability"] = min(1.0, adaptability * 2)  # Scale to 0-1

    # Consistency: state switching rate
    if len(dominant_states) > 1:
        state_switches = np.sum(np.diff(dominant_states) != 0)
        consistency = 1.0 - (state_switches / len(dominant_states))
        quality_metrics["Consistency"] = max(0.0, consistency)

    # Create radar chart for quality metrics
    angles = np.linspace(0, 2 * np.pi, len(quality_metrics), endpoint=False)
    values = list(quality_metrics.values())

    # Close the radar chart
    angles = np.concatenate((angles, [angles[0]]))
    values = values + [values[0]]

    ax8.plot(angles, values, "o-", linewidth=2, color="blue")
    ax8.fill(angles, values, alpha=0.25, color="blue")
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(quality_metrics.keys())
    ax8.set_ylim(0, 1)
    ax8.set_title("Perception Quality Metrics")
    ax8.grid(True)

    # Add metric values as text
    for angle, value in zip(angles[:-1], values[:-1]):
        x = (value + 0.1) * np.cos(angle)
        y = (value + 0.1) * np.sin(angle)
        ax8.text(x, y, f"{value:.2f}", ha="center", va="center", fontweight="bold")

    plt.suptitle(title, fontsize=20, fontweight="bold")
    plt.savefig(
        output_dir / "comprehensive_perception_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(f"Perception analysis saved to {output_dir}")


def plot_action_analysis(
    policy_history: list[dict[str, Any]],
    action_history: list[Any],
    output_dir: Path,
    title: str = "Action Selection Analysis",
) -> None:
    """
    Create comprehensive action selection analysis plots.

    Args:
        policy_history: History of policy selections
        action_history: History of actions taken
        output_dir: Output directory for plots
        title: Overall title for the analysis
    """
    if not policy_history and not action_history:
        logger.warning("No policy or action data available for analysis")
        return

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Extract policy data
    policy_probs = []
    expected_free_energies = []
    selected_policies = []

    for policy_data in policy_history:
        if "all_probabilities" in policy_data:
            policy_probs.append(np.array(policy_data["all_probabilities"]))
        if "all_free_energies" in policy_data:
            expected_free_energies.append(policy_data["all_free_energies"])
        if "policy" in policy_data and "id" in policy_data["policy"]:
            selected_policies.append(policy_data["policy"]["id"])

    # 1. Policy probability evolution
    if policy_probs:
        ax1 = fig.add_subplot(gs[0, :2])
        policy_probs_array = np.array(policy_probs)

        for i in range(policy_probs_array.shape[1]):
            ax1.plot(
                policy_probs_array[:, i],
                label=f"Policy {i}",
                linewidth=2,
                marker="o",
                markersize=2,
            )

        ax1.set_title("Policy Probability Evolution")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Probability")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

    # 2. Policy entropy over time
    if policy_probs:
        ax2 = fig.add_subplot(gs[0, 2])
        # Ensure probs is numpy array for entropy calculation
        policy_probs_for_entropy = [np.array(p) for p in policy_probs]
        entropies = [
            -np.sum(probs * np.log(probs + 1e-8)) for probs in policy_probs_for_entropy
        ]
        ax2.plot(entropies, linewidth=2, color="purple", marker="s", markersize=3)
        ax2.set_title("Policy Selection Entropy")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Entropy")
        ax2.grid(True, alpha=0.3)

        # Add exploration/exploitation phases
        high_entropy = np.array(entropies) > np.median(entropies)
        ax2.fill_between(
            range(len(entropies)),
            0,
            max(entropies),
            where=high_entropy,
            alpha=0.3,
            color="orange",
            label="High Exploration",
        )
        ax2.fill_between(
            range(len(entropies)),
            0,
            max(entropies),
            where=~high_entropy,
            alpha=0.3,
            color="blue",
            label="High Exploitation",
        )
        ax2.legend()

    # 3. Expected free energy evolution
    if expected_free_energies:
        ax3 = fig.add_subplot(gs[0, 3])
        efe_array = np.array(expected_free_energies)

        if efe_array.ndim > 1:
            for i in range(efe_array.shape[1]):
                ax3.plot(efe_array[:, i], label=f"Policy {i}", linewidth=2)
            ax3.legend()
        else:
            ax3.plot(efe_array, linewidth=2, color="red")

        ax3.set_title("Expected Free Energy")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Expected Free Energy")
        ax3.grid(True, alpha=0.3)

    # 4. Policy selection heatmap
    if policy_probs:
        ax4 = fig.add_subplot(gs[1, :2])
        im4 = ax4.imshow(
            policy_probs_array.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        ax4.set_title("Policy Selection Heatmap")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Policy")
        plt.colorbar(im4, ax=ax4, label="Selection Probability")

    # 5. Action consistency analysis
    if action_history:
        ax5 = fig.add_subplot(gs[1, 2])

        # Convert actions to numeric if possible
        try:
            numeric_actions = [
                float(a) if isinstance(a, (int, float)) else hash(str(a)) % 100
                for a in action_history
            ]
            ax5.plot(
                numeric_actions, linewidth=2, color="green", marker="d", markersize=3
            )
            ax5.set_title("Action Sequence")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Action Value")

            # Highlight action changes
            if len(numeric_actions) > 1:
                changes = np.diff(numeric_actions)
                change_points = np.where(np.abs(changes) > 0)[0]
                for cp in change_points:
                    ax5.axvline(x=cp + 1, color="red", linestyle="--", alpha=0.7)

        except (TypeError, ValueError):
            # Categorical actions
            unique_actions = list(set(map(str, action_history)))
            action_mapping = {action: i for i, action in enumerate(unique_actions)}
            mapped_actions = [action_mapping[str(a)] for a in action_history]

            ax5.plot(
                mapped_actions, linewidth=2, color="green", marker="d", markersize=3
            )
            ax5.set_title("Action Sequence (Categorical)")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Action Category")
            ax5.set_yticks(range(len(unique_actions)))
            ax5.set_yticklabels(unique_actions, rotation=45)

        ax5.grid(True, alpha=0.3)

    # 6. Selected policy distribution
    if selected_policies:
        ax6 = fig.add_subplot(gs[1, 3])

        unique_policies, counts = np.unique(selected_policies, return_counts=True)
        colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(unique_policies)))

        bars = ax6.bar(range(len(unique_policies)), counts, color=colors, alpha=0.8)
        ax6.set_title("Policy Selection Frequency")
        ax6.set_xlabel("Policy")
        ax6.set_ylabel("Selection Count")
        ax6.set_xticks(range(len(unique_policies)))
        ax6.set_xticklabels([f"Policy {p}" for p in unique_policies], rotation=45)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.annotate(
                f"{count}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        ax6.grid(True, alpha=0.3)

    # 7. Exploration vs Exploitation timeline
    if policy_probs:
        ax7 = fig.add_subplot(gs[2, :2])

        entropies = [-np.sum(probs * np.log(probs + 1e-8)) for probs in policy_probs]
        max_entropy = np.log(len(policy_probs[0]))
        normalized_entropies = np.array(entropies) / max_entropy

        exploration_scores = normalized_entropies
        exploitation_scores = 1 - normalized_entropies

        ax7.fill_between(
            range(len(exploration_scores)),
            0,
            exploration_scores,
            alpha=0.7,
            color="orange",
            label="Exploration",
        )
        ax7.fill_between(
            range(len(exploitation_scores)),
            exploration_scores,
            exploration_scores + exploitation_scores,
            alpha=0.7,
            color="blue",
            label="Exploitation",
        )

        ax7.set_title("Exploration vs Exploitation Over Time")
        ax7.set_xlabel("Time Step")
        ax7.set_ylabel("Proportion")
        ax7.set_ylim(0, 1)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. Decision quality metrics
    ax8 = fig.add_subplot(gs[2, 2:])

    # Calculate decision quality metrics
    quality_metrics = {
        "Consistency": 0.0,
        "Decisiveness": 0.0,
        "Adaptability": 0.0,
        "Efficiency": 0.0,
    }

    if action_history:
        # Consistency: how often actions repeat
        if len(action_history) > 1:
            action_changes = sum(
                1
                for i in range(1, len(action_history))
                if str(action_history[i]) != str(action_history[i - 1])
            )
            consistency = 1.0 - (action_changes / len(action_history))
            quality_metrics["Consistency"] = max(0.0, consistency)

    if policy_probs:
        # Decisiveness: how concentrated policy probabilities are
        avg_max_prob = np.mean([np.max(probs) for probs in policy_probs])
        quality_metrics["Decisiveness"] = avg_max_prob

        # Adaptability: variance in policy selection
        policy_vars = [np.var(probs) for probs in policy_probs]
        adaptability = np.mean(policy_vars)
        quality_metrics["Adaptability"] = min(1.0, adaptability * 4)  # Scale

    if expected_free_energies:
        # Efficiency: improvement in expected free energy
        efe_array = np.array(expected_free_energies)
        if efe_array.ndim > 1:
            min_efe_per_step = np.min(efe_array, axis=1)
            if len(min_efe_per_step) > 1:
                efe_improvement = min_efe_per_step[0] - min_efe_per_step[-1]
                efficiency = max(
                    0.0,
                    min(
                        1.0,
                        (
                            efe_improvement / abs(min_efe_per_step[0])
                            if min_efe_per_step[0] != 0
                            else 0
                        ),
                    ),
                )
                quality_metrics["Efficiency"] = efficiency

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(quality_metrics), endpoint=False)
    values = list(quality_metrics.values())

    # Close the radar chart
    angles = np.concatenate((angles, [angles[0]]))
    values = values + [values[0]]

    ax8.plot(angles, values, "o-", linewidth=2, color="red")
    ax8.fill(angles, values, alpha=0.25, color="red")
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(quality_metrics.keys())
    ax8.set_ylim(0, 1)
    ax8.set_title("Decision Quality Metrics")
    ax8.grid(True)

    # Add metric values
    for angle, value in zip(angles[:-1], values[:-1]):
        x = (value + 0.1) * np.cos(angle)
        y = (value + 0.1) * np.sin(angle)
        ax8.text(x, y, f"{value:.2f}", ha="center", va="center", fontweight="bold")

    plt.suptitle(title, fontsize=20, fontweight="bold")
    plt.savefig(
        output_dir / "comprehensive_action_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(f"Action analysis saved to {output_dir}")


def create_interpretability_dashboard(analyzer: Any, output_dir: Path) -> None:
    """
    Create a comprehensive interpretability dashboard.

    Args:
        analyzer: ActiveInferenceAnalyzer instance
        output_dir: Output directory for the dashboard
    """
    if not analyzer.traces["beliefs"]:
        logger.warning("No data available for interpretability dashboard")
        return

    # Create main dashboard figure
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)

    # Data preparation
    beliefs_array = np.array(analyzer.traces["beliefs"])
    fe_array = np.array(analyzer.traces["free_energy"])

    # 1. System Overview (top row, left)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.text(
        0.5,
        0.9,
        "Active Inference System Overview",
        ha="center",
        va="top",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    overview_text = "• Total Time Steps: {}\n• State Dimensions: {}\n• Initial Free Energy: {}\n• Final Free Energy: {}\n• Free Energy Reduction: {}".format(
        len(beliefs_array),
        beliefs_array.shape[1] if len(beliefs_array) > 0 else "N/A",
        f"{fe_array[0]:.4f}" if len(fe_array) > 0 else "N/A",
        f"{fe_array[-1]:.4f}" if len(fe_array) > 0 else "N/A",
        f"{fe_array[0] - fe_array[-1]:.4f}" if len(fe_array) > 0 else "N/A",
    )

    ax1.text(
        0.1,
        0.7,
        overview_text,
        ha="left",
        va="top",
        transform=ax1.transAxes,
        fontsize=12,
        fontfamily="monospace",
    )
    ax1.axis("off")

    # 2. Key Metrics Summary (top row, right)
    ax2 = fig.add_subplot(gs[0, 3:])

    # Calculate key metrics
    if len(beliefs_array) > 0:
        final_entropy = -np.sum(beliefs_array[-1] * np.log(beliefs_array[-1] + 1e-8))
        avg_entropy = np.mean([-np.sum(b * np.log(b + 1e-8)) for b in beliefs_array])

        # Create metrics visualization
        metrics = [
            "Free Energy\nReduction",
            "Final\nEntropy",
            "Average\nEntropy",
            "System\nStability",
        ]
        values = [
            (
                (fe_array[0] - fe_array[-1]) / abs(fe_array[0])
                if len(fe_array) > 0 and fe_array[0] != 0
                else 0
            ),
            final_entropy / np.log(beliefs_array.shape[1]),  # Normalized
            avg_entropy / np.log(beliefs_array.shape[1]),
            1.0 / (1.0 + np.std(fe_array)) if len(fe_array) > 0 else 0,
        ]

        colors = [
            "green" if v > 0.5 else "orange" if v > 0.3 else "red" for v in values
        ]
        bars = ax2.bar(metrics, values, color=colors, alpha=0.7)

        ax2.set_ylim(0, 1)
        ax2.set_title("Key Performance Metrics", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # 3. Belief Evolution (second row, left half)
    ax3 = fig.add_subplot(gs[1, :3])
    if len(beliefs_array) > 0:
        im3 = ax3.imshow(
            beliefs_array.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        ax3.set_title("Belief Evolution (Perception)", fontweight="bold")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("State")
        plt.colorbar(im3, ax=ax3, label="Belief Probability")

    # 4. Free Energy Dynamics (second row, right half)
    ax4 = fig.add_subplot(gs[1, 3:])
    if len(fe_array) > 0:
        ax4.plot(fe_array, linewidth=3, color="red", alpha=0.8)
        ax4.fill_between(range(len(fe_array)), fe_array, alpha=0.3, color="red")
        ax4.set_title("Free Energy Minimization", fontweight="bold")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Free Energy")
        ax4.grid(True, alpha=0.3)

        # Add trend line
        if len(fe_array) > 1:
            z = np.polyfit(range(len(fe_array)), fe_array, 1)
            p = np.poly1d(z)
            ax4.plot(
                range(len(fe_array)),
                p(range(len(fe_array))),
                "--",
                color="blue",
                linewidth=2,
                alpha=0.8,
                label=f"Trend: {z[0]:.4f}/step",
            )
            ax4.legend()

    # 5. Policy Analysis (third row, left)
    ax5 = fig.add_subplot(gs[2, :2])
    if analyzer.traces["policies"]:
        policy_probs = []
        for policy_data in analyzer.traces["policies"]:
            if "all_probabilities" in policy_data:
                policy_probs.append(
                    np.array(policy_data["all_probabilities"])
                )  # Ensure probs is numpy array

        if policy_probs:
            entropies = [
                -np.sum(probs * np.log(probs + 1e-8)) for probs in policy_probs
            ]

            ax5.plot(entropies, linewidth=2, color="purple", marker="o", markersize=3)
            ax5.set_title("Policy Selection Entropy (Action)", fontweight="bold")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Entropy")
            ax5.grid(True, alpha=0.3)

            # Add exploration/exploitation annotation
            median_entropy = np.median(entropies)
            high_exploration = np.array(entropies) > median_entropy
            ax5.fill_between(
                range(len(entropies)),
                0,
                max(entropies),
                where=high_exploration,
                alpha=0.2,
                color="orange",
                label="High Exploration",
            )
            ax5.fill_between(
                range(len(entropies)),
                0,
                max(entropies),
                where=~high_exploration,
                alpha=0.2,
                color="blue",
                label="High Exploitation",
            )
            ax5.legend()

    # 6. Perception Quality (third row, center)
    ax6 = fig.add_subplot(gs[2, 2:4])
    if len(beliefs_array) > 1:
        # Calculate perception metrics
        belief_changes = np.diff(beliefs_array, axis=0)
        responsiveness = np.mean(np.linalg.norm(belief_changes, axis=1))
        stability = 1.0 / (1.0 + np.var(np.linalg.norm(belief_changes, axis=1)))

        entropies = [-np.sum(b * np.log(b + 1e-8)) for b in beliefs_array]
        adaptability = (
            np.std(entropies) / np.mean(entropies) if np.mean(entropies) > 0 else 0
        )

        dominant_states = np.argmax(beliefs_array, axis=1)
        consistency = 1.0 - (
            np.sum(np.diff(dominant_states) != 0) / len(dominant_states)
        )

        perception_metrics = ["Responsive", "Stable", "Adaptive", "Consistent"]
        perception_values = [
            min(1.0, responsiveness * 10),
            stability,
            min(1.0, adaptability),
            max(0.0, consistency),
        ]

        # Radar chart for perception quality
        angles = np.linspace(0, 2 * np.pi, len(perception_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        perception_values = perception_values + [perception_values[0]]

        ax6.plot(angles, perception_values, "o-", linewidth=2, color="blue")
        ax6.fill(angles, perception_values, alpha=0.25, color="blue")
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(perception_metrics)
        ax6.set_ylim(0, 1)
        ax6.set_title("Perception Quality", fontweight="bold")
        ax6.grid(True)

    # 7. Action Quality (third row, right)
    ax7 = fig.add_subplot(gs[2, 4:])
    if analyzer.traces["policies"] and analyzer.traces["actions"]:
        # Action quality metrics
        actions = analyzer.traces["actions"]

        # Consistency
        if len(actions) > 1:
            action_changes = sum(
                1
                for i in range(1, len(actions))
                if str(actions[i]) != str(actions[i - 1])
            )
            consistency = 1.0 - (action_changes / len(actions))
        else:
            consistency = 1.0

        # Decisiveness (from policy data)
        if analyzer.traces["policies"]:
            policy_probs = []
            for policy_data in analyzer.traces["policies"]:
                if "all_probabilities" in policy_data:
                    policy_probs.append(np.array(policy_data["all_probabilities"]))

            if policy_probs:
                decisiveness = np.mean([np.max(probs) for probs in policy_probs])
            else:
                decisiveness = 0.5
        else:
            decisiveness = 0.5

        # Calculate Adaptability based on policy probability variance
        if "policy_probs" in locals() and policy_probs:
            policy_vars = [np.var(probs) for probs in policy_probs]
            adaptability = min(
                1.0, np.mean(policy_vars) * 4
            )  # Scaled for 0-1 radar chart
        else:
            adaptability = 0.5

        # Calculate Efficiency based on expected free energy improvements
        efficiency = 0.5
        expected_free_energies = []
        if "policies" in analyzer.traces:
            for policy_data in analyzer.traces["policies"]:
                if "all_free_energies" in policy_data:
                    expected_free_energies.append(policy_data["all_free_energies"])

        if expected_free_energies:
            efe_array = np.array(expected_free_energies)
            if efe_array.ndim > 1:
                min_efe_per_step = np.min(efe_array, axis=1)
                if len(min_efe_per_step) > 1 and min_efe_per_step[0] != 0:
                    efe_improvement = min_efe_per_step[0] - min_efe_per_step[-1]
                    efficiency = max(
                        0.0, min(1.0, efe_improvement / abs(min_efe_per_step[0]))
                    )

        action_metrics = ["Consistent", "Decisive", "Efficient", "Adaptive"]
        action_values = [max(0.0, consistency), decisiveness, efficiency, adaptability]

        # Radar chart for action quality
        angles = np.linspace(0, 2 * np.pi, len(action_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        action_values = action_values + [action_values[0]]

        ax7.plot(angles, action_values, "o-", linewidth=2, color="red")
        ax7.fill(angles, action_values, alpha=0.25, color="red")
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(action_metrics)
        ax7.set_ylim(0, 1)
        ax7.set_title("Action Quality", fontweight="bold")
        ax7.grid(True)

    # 8. System Diagnostics (bottom row)
    ax8 = fig.add_subplot(gs[3, :])

    # Create diagnostic summary
    diagnostics = []

    # Check for flat patterns
    if len(beliefs_array) > 0:
        belief_vars = np.var(beliefs_array, axis=0)
        flat_states = np.sum(belief_vars < 1e-3)
        if flat_states > 0:
            diagnostics.append(
                f"⚠️  {flat_states} states show flat/non-responsive patterns"
            )
        else:
            diagnostics.append("✅ All states show responsive dynamics")

    # Check free energy behavior
    if len(fe_array) > 1:
        fe_reduction = fe_array[0] - fe_array[-1]
        if fe_reduction > 0:
            diagnostics.append("✅ Free energy is successfully minimizing")
        elif abs(fe_reduction) < 1e-6:
            diagnostics.append("⚠️  Free energy shows no significant change")
        else:
            diagnostics.append("❌ Free energy is increasing (concerning)")

    # Check for random patterns
    if len(beliefs_array) > 5:
        # Simple randomness check using autocorrelation
        autocorr_scores = []
        for dim in range(beliefs_array.shape[1]):
            series = beliefs_array[:, dim]
            if len(series) > 1:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorr_scores.append(abs(autocorr))

        if autocorr_scores:
            avg_autocorr = np.mean(autocorr_scores)
            if avg_autocorr < 0.2:
                diagnostics.append("⚠️  Beliefs show high randomness (low structure)")
            elif avg_autocorr > 0.8:
                diagnostics.append(
                    "⚠️  Beliefs show high autocorrelation (may be stuck)"
                )
            else:
                diagnostics.append("✅ Beliefs show appropriate structure")

    # Policy diagnostics
    if analyzer.traces["policies"]:
        policy_probs = []
        for policy_data in analyzer.traces["policies"]:
            if "all_probabilities" in policy_data:
                policy_probs.append(np.array(policy_data["all_probabilities"]))

        if policy_probs:
            policy_probs_array = np.array(policy_probs)
            final_policy_entropy = -np.sum(
                policy_probs_array[-1] * np.log(policy_probs_array[-1] + 1e-8)
            )
            max_entropy = np.log(len(policy_probs_array[-1]))

            if final_policy_entropy / max_entropy > 0.9:
                diagnostics.append("⚠️  Policy selection remains very random")
            elif final_policy_entropy / max_entropy < 0.1:
                diagnostics.append("⚠️  Policy selection may be too deterministic")
            else:
                diagnostics.append(
                    "✅ Policy selection shows good exploration-exploitation balance"
                )

    # Display diagnostics
    ax8.text(
        0.02,
        0.9,
        "System Diagnostics:",
        ha="left",
        va="top",
        transform=ax8.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    for i, diagnostic in enumerate(diagnostics):
        ax8.text(
            0.02,
            0.7 - i * 0.15,
            diagnostic,
            ha="left",
            va="top",
            transform=ax8.transAxes,
            fontsize=12,
        )

    # Performance summary
    performance_text = "Performance Summary:\n• Perception responsiveness: {}\n• Free energy dynamics: {}\n• System stability: {}\n• Overall assessment: {}".format(
        "Good" if len(beliefs_array) > 0 and np.std(beliefs_array) > 0.01 else "Poor",
        (
            "Minimizing"
            if len(fe_array) > 1 and fe_array[-1] < fe_array[0]
            else "Stable/Increasing"
        ),
        (
            "Stable"
            if len(fe_array) > 0 and np.std(fe_array) < np.mean(fe_array)
            else "Unstable"
        ),
        (
            "Healthy"
            if len(diagnostics) > 0
            and sum("OK" in d for d in diagnostics) > len(diagnostics) / 2
            else "Needs Attention"
        ),
    )

    ax8.text(
        0.6,
        0.9,
        performance_text,
        ha="left",
        va="top",
        transform=ax8.transAxes,
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    ax8.axis("off")

    plt.suptitle(
        "Active Inference Interpretability Dashboard", fontsize=24, fontweight="bold"
    )
    plt.savefig(
        output_dir / "interpretability_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(f"Interpretability dashboard saved to {output_dir}")


def _belief_payload_to_vector(belief_payload: Any) -> np.ndarray:
    """Convert supported belief payload shapes into a flat numeric vector."""
    if isinstance(belief_payload, dict):
        for key in ("states", "mean", "beliefs"):
            if key in belief_payload:
                belief_payload = belief_payload[key]
                break
        else:
            belief_payload = next(iter(belief_payload.values()))

    array = np.asarray(belief_payload, dtype=object)
    if array.dtype == object:
        parts = []
        for item in array.reshape(-1):
            item_array = np.asarray(item, dtype=float).reshape(-1)
            parts.append(item_array)
        if parts:
            return np.concatenate(parts)
    return np.asarray(belief_payload, dtype=float).reshape(-1)


def plot_hierarchical_beliefs(beliefs: dict[str, np.ndarray]) -> plt.Figure:
    """Plot beliefs across hierarchical levels."""
    if not beliefs:
        raise ValueError("beliefs must contain at least one hierarchical level")
    fig, axs = plt.subplots(len(beliefs), 1, figsize=(8, 4 * len(beliefs)))
    axs = np.atleast_1d(axs)
    for i, (level, bel) in enumerate(beliefs.items()):
        vector = _belief_payload_to_vector(bel)
        if vector.size == 0 or not np.all(np.isfinite(vector)) or np.any(vector < 0):
            raise ValueError(f"beliefs[{level!r}] must be finite and non-negative")
        axs[i].bar(range(len(vector)), vector)
        axs[i].set_title(f"Level {level}")
    fig.tight_layout()
    return fig


def plot_markov_blanket(blanket: Any) -> plt.Figure:
    """Plot Markov blanket structure.

    Args:
        blanket: Dictionary of state indices organized by Markov blanket
            components (e.g., 'internal', 'active', 'sensory', 'external') or a
            canonical MarkovBlanket-style object with component attributes.
    """
    required_attrs = (
        "internal_states",
        "active_states",
        "sensory_states",
        "external_states",
    )
    if not isinstance(blanket, dict) and all(
        hasattr(blanket, attr) for attr in required_attrs
    ):
        blanket = {
            "internal": list(blanket.internal_states),
            "active": list(blanket.active_states),
            "sensory": list(blanket.sensory_states),
            "external": list(blanket.external_states),
        }

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        import networkx as nx
    except ImportError:
        # Fallback if networkx is not installed
        y = 0.8
        for k, v in blanket.items():
            ax.text(
                0.5,
                y,
                f"{k.capitalize()} States: {len(v)} nodes",
                ha="center",
                va="center",
                size=14,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="gray"),
            )
            y -= 0.2
        ax.axis("off")
    else:
        G = nx.DiGraph()

        # Standard colors and radii for Markov Blanket components
        layers = {"internal": 0, "sensory": 1, "active": 1, "external": 2}
        colors_map = {
            "internal": "#87CEFA",
            "sensory": "#FFA500",
            "active": "#98FB98",
            "external": "#D3D3D3",
        }

        pos = {}
        node_colors = []
        labels = {}

        for group, nodes in blanket.items():
            group_lower = group.lower()
            radius = layers.get(group_lower, 1.5)
            color = colors_map.get(group_lower, "#DDA0DD")

            n = len(nodes)
            if n == 0:
                continue

            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            if group_lower == "active":
                angles += np.pi / max(n, 1)  # offset active from sensory on same radius

            for i, node_id in enumerate(nodes):
                node_name = f"{group[0].upper()}{node_id}"
                G.add_node(node_name)
                labels[node_name] = str(node_id)

                if radius == 0:
                    pos[node_name] = (0.0 + (i * 0.1 if n > 1 else 0), 0.0)
                else:
                    pos[node_name] = (
                        radius * np.cos(angles[i]),
                        radius * np.sin(angles[i]),
                    )
                node_colors.append(color)

        # Add implicit causal edges based on Markov Blanket definition
        valid_edges = {
            ("external", "sensory"),
            ("sensory", "internal"),
            ("internal", "active"),
            ("active", "external"),
        }
        for group1, nodes1 in blanket.items():
            for group2, nodes2 in blanket.items():
                if (group1.lower(), group2.lower()) in valid_edges:
                    for n1 in nodes1:
                        for n2 in nodes2:
                            G.add_edge(
                                f"{group1[0].upper()}{n1}", f"{group2[0].upper()}{n2}"
                            )

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=1500,
            edgecolors="white",
            linewidths=2,
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, arrowsize=15, width=1.5, alpha=0.6, edge_color="gray"
        )
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=12, font_weight="bold")

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=k.capitalize(),
                markerfacecolor=v,
                markersize=10,
            )
            for k, v in colors_map.items()
            if any(g.lower() == k for g in blanket.keys())
        ]
        if legend_elements:
            ax.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1.1)
            )

        ax.axis("off")

    ax.set_title("Markov Blanket Topology", fontsize=16, fontweight="bold", pad=20)
    return fig


def plot_h3_grid_static(
    h3_data: dict[str, Any], metric: str = "fe", title: str = "H3 Grid"
) -> plt.Figure:
    """
    Create static plot of H3 grid data.

    Args:
        h3_data: Dictionary mapping H3 cells to data dictionaries
        metric: Metric to visualize
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    if not SHAPELY_AVAILABLE:
        ax.text(
            0.5,
            0.5,
            "Shapely package not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.set_title(title)
        return fig

    if not h3_data:
        ax.text(
            0.5,
            0.5,
            "No H3 data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.set_title(title)
        return fig

    # Extract values for color normalization
    values_list: list[Any] = []
    for cell_data in h3_data.values():
        if isinstance(cell_data, dict) and metric in cell_data:
            values_list.append(cell_data[metric])
        elif isinstance(cell_data, (int, float)):
            values_list.append(cell_data)
        else:
            values_list.append(0)

    if not values_list:
        ax.text(
            0.5,
            0.5,
            f"No {metric} data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.set_title(title)
        return fig

    values = np.asarray(values_list, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{metric} values must be finite")
    min_val, max_val = float(np.min(values)), float(np.max(values))
    value_range = max_val - min_val if max_val != min_val else 1

    # Plot each H3 cell
    for cell, cell_data in h3_data.items():
        try:
            # Get H3 cell boundary
            boundary = h3.cell_to_boundary(cell)

            # Convert to polygon coordinates (lng, lat) -> (x, y)
            coords = [(lng, lat) for lat, lng in boundary]
            poly = Polygon(coords)
            x, y = poly.exterior.xy

            # Get value for coloring
            if isinstance(cell_data, dict) and metric in cell_data:
                value = cell_data[metric]
            elif isinstance(cell_data, (int, float)):
                value = cell_data
            else:
                value = 0

            # Normalize value for color mapping
            normalized_value = (
                (value - min_val) / value_range if value_range > 0 else 0.5
            )
            color = plt.get_cmap("viridis")(normalized_value)

            # Fill the hexagon
            ax.fill(x, y, color=color, edgecolor="white", linewidth=0.5, alpha=0.8)

        except Exception as e:
            logger.warning(f"Failed to plot H3 cell {cell}: {e}")
            continue

    # Set equal aspect ratio and remove axes
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("viridis"),
        norm=plt.Normalize(
            vmin=min_val, vmax=max_val if max_val > min_val else min_val + 1
        ),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(metric.replace("_", " ").title(), fontsize=12)

    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    return fig


def create_h3_gif(
    history: list[dict[str, Any]], output_path: str, metric: str = "fe"
) -> None:
    """
    Create animated GIF of H3 grid evolution over time.

    Args:
        history: List of H3 data for each timestep
        output_path: Path to save the GIF
        metric: Metric to visualize
    """
    if not IMAGEIO_AVAILABLE:
        logger.error("imageio package required for GIF creation")
        return

    if not history:
        logger.warning("No history data provided for GIF creation")
        return

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        images: list[np.ndarray] = []

        # Get global min/max values for consistent color scaling
        all_values = []
        for timestep_data in history:
            for cell_data in timestep_data.values():
                if isinstance(cell_data, dict) and metric in cell_data:
                    all_values.append(cell_data[metric])
                elif isinstance(cell_data, (int, float)):
                    all_values.append(cell_data)

        if not all_values:
            logger.warning(f"No {metric} data found in history")
            return

        for t, timestep_data in enumerate(history):
            # Create a copy for consistent color scaling
            normalized_data = {}
            for cell, cell_data in timestep_data.items():
                if isinstance(cell_data, dict):
                    normalized_data[cell] = cell_data.copy()
                    if metric in cell_data:
                        # Ensure value is within global range for consistent coloring
                        normalized_data[cell][metric] = cell_data[metric]
                else:
                    normalized_data[cell] = {metric: cell_data}

            # Create figure for this timestep
            fig = plot_h3_grid_static(normalized_data, metric, f"Timestep {t}")

            # Convert figure to image array
            fig.canvas.draw()
            image = np.frombuffer(
                fig.canvas.buffer_rgba(),  # type: ignore[attr-defined]
                dtype=np.uint8,
            )
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            images.append(image)

            plt.close(fig)

        # Save as GIF
        imageio.mimsave(output_path, images, duration=0.5, loop=0)  # type: ignore[arg-type]
        logger.info(f"H3 evolution GIF saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to create H3 GIF: {e}")


def create_interactive_h3_slider(
    history: list[dict[str, Any]], metric: str = "fe"
) -> Any:
    """
    Create interactive slider plot for H3 grid evolution.

    Args:
        history: List of H3 data for each timestep
        metric: Metric to visualize

    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        logger.error("plotly package required for interactive plots")
        return None

    if not history:
        logger.warning("No history data provided for interactive plot")
        return None

    try:
        # Prepare data for plotly
        plot_data = []

        for t, timestep_data in enumerate(history):
            for cell, cell_data in timestep_data.items():
                try:
                    lat, lng = h3.cell_to_latlng(cell)

                    # Extract metric value
                    if isinstance(cell_data, dict) and metric in cell_data:
                        value = cell_data[metric]
                    elif isinstance(cell_data, (int, float)):
                        value = cell_data
                    else:
                        value = 0

                    plot_data.append(
                        {"time": t, "cell": cell, "lat": lat, "lon": lng, metric: value}
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to process cell {cell} at timestep {t}: {e}"
                    )
                    continue

        if not plot_data:
            logger.warning("No valid data for interactive plot")
            return None

        df = pd.DataFrame(plot_data)

        # Create animated scatter plot
        fig = px.scatter_geo(
            df,
            lat="lat",
            lon="lon",
            color=metric,
            animation_frame="time",
            projection="natural earth",
            range_color=[df[metric].min(), df[metric].max()],
            color_continuous_scale="viridis",
            title=f"H3 Grid Evolution: {metric.replace('_', ' ').title()}",
            hover_data=["cell"],
        )

        # Update layout
        fig.update_layout(
            geo=dict(
                showland=True,
                landcolor="lightgray",
                showocean=True,
                oceancolor="lightblue",
            ),
            title_font_size=16,
            coloraxis_colorbar=dict(title=metric.replace("_", " ").title()),
        )

        # Update animation speed
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

        return fig

    except Exception as e:
        logger.error(f"Failed to create interactive H3 plot: {e}")
        return None


class BeliefVisualizer:
    """
    Visualizer for active inference agent beliefs and internal states.
    Wrapper around functional plotting utilities for object-oriented usage.
    """

    def __init__(self, output_dir: str | Path | None = None):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save plots. Defaults to current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_belief_evolution(
        self,
        belief_history: list[np.ndarray],
        observations_history: list[np.ndarray] | None = None,
        output_path: str | None = None,
    ) -> None:
        """
        Plot the evolution of beliefs over time.

        Args:
            belief_history: List of belief arrays (one per timestep)
            observations_history: Optional list of observations
            output_path: Optional filename override (relative to output_dir)
        """
        if not belief_history:
            logger.warning("No belief history provided to plot_belief_evolution")
            return

        # Use the comprehensive perception analysis if observations are present
        if observations_history:
            plot_perception_analysis(
                beliefs_history=belief_history,
                observations_history=observations_history,
                output_dir=self.output_dir,
                title="Belief Evolution Analysis",
            )
        else:
            # Simple belief plot if no observations
            fig, ax = plt.subplots(figsize=(12, 8))
            beliefs_array = np.array(belief_history)

            im = ax.imshow(
                beliefs_array.T, aspect="auto", cmap="viridis", interpolation="nearest"
            )
            ax.set_title("Belief Evolution")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State")
            plt.colorbar(im, ax=ax, label="Belief Probability")

            save_path = self.output_dir / (output_path or "belief_evolution.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Belief evolution plot saved to {save_path}")

    def plot_free_energy_trace(
        self, free_energy_history: list[float], output_path: str = "free_energy.png"
    ) -> None:
        """Plot free energy minimization trace."""
        fig = plot_free_energy(free_energy_history, title="Free Energy Trace")
        save_path = self.output_dir / output_path
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Free energy trace saved to {save_path}")
