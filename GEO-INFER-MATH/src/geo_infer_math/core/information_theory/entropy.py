"""
Entropy Measures for Spatial Data

This module provides various entropy measures for analyzing spatial data,
including Shannon entropy, Renyi entropy, Tsallis entropy, and spatial
entropy measures.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def shannon_entropy(
    probabilities: np.ndarray, base: float = 2.0, normalize: bool = False
) -> float:
    """
    Calculate Shannon entropy for a probability distribution.

    Shannon entropy: H(X) = -Σ p(x) * log(p(x))

    Args:
        probabilities: Probability distribution (must sum to 1)
        base: Logarithm base (default: 2 for bits)
        normalize: Whether to normalize by log(n) for maximum entropy

    Returns:
        Shannon entropy value

    Raises:
        ValueError: If probabilities don't sum to 1 or contain negative values
    """
    probabilities = np.asarray(probabilities)

    # Validate probabilities
    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative")

    # Normalize if needed
    prob_sum = np.sum(probabilities)
    if not np.isclose(prob_sum, 1.0, rtol=1e-10):
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        else:
            raise ValueError("Probabilities must sum to a positive value")

    # Remove zero probabilities (log(0) is undefined)
    non_zero = probabilities > 0
    if not np.any(non_zero):
        return 0.0

    # Calculate entropy
    entropy = -np.sum(
        probabilities[non_zero] * np.log(probabilities[non_zero]) / np.log(base)
    )

    # Normalize if requested
    if normalize and len(probabilities) > 1:
        max_entropy = np.log(len(probabilities)) / np.log(base)
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(entropy)


def renyi_entropy(
    probabilities: np.ndarray, alpha: float = 1.0, base: float = 2.0
) -> float:
    """
    Calculate Renyi entropy for a probability distribution.

    Renyi entropy: H_α(X) = (1/(1-α)) * log(Σ p(x)^α)

    For α=1, this reduces to Shannon entropy.

    Args:
        probabilities: Probability distribution
        alpha: Order parameter (α > 0, α ≠ 1)
        base: Logarithm base

    Returns:
        Renyi entropy value

    Raises:
        ValueError: If alpha <= 0 or alpha == 1
    """
    probabilities = np.asarray(probabilities)

    # Validate probabilities
    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative")

    # Normalize
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        raise ValueError("Probabilities must sum to a positive value")

    # Handle special case: α = 1 (Shannon entropy)
    if np.isclose(alpha, 1.0):
        return shannon_entropy(probabilities, base=base)

    if alpha <= 0:
        raise ValueError("Alpha must be positive")

    # Remove zero probabilities
    non_zero = probabilities > 0
    if not np.any(non_zero):
        return 0.0

    # Calculate Renyi entropy
    prob_power = np.power(probabilities[non_zero], alpha)
    sum_power = np.sum(prob_power)

    if sum_power <= 0:
        return 0.0

    entropy = (1.0 / (1.0 - alpha)) * (np.log(sum_power) / np.log(base))

    return float(entropy)


def tsallis_entropy(
    probabilities: np.ndarray, q: float = 1.0, base: float = 2.0
) -> float:
    """
    Calculate Tsallis entropy for a probability distribution.

    Tsallis entropy: S_q(X) = (1/(q-1)) * (1 - Σ p(x)^q)

    For q=1, this reduces to Shannon entropy.

    Args:
        probabilities: Probability distribution
        q: Entropic index (q > 0, q ≠ 1)
        base: Logarithm base (for normalization)

    Returns:
        Tsallis entropy value

    Raises:
        ValueError: If q <= 0 or q == 1
    """
    probabilities = np.asarray(probabilities)

    # Validate probabilities
    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative")

    # Normalize
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        raise ValueError("Probabilities must sum to a positive value")

    # Handle special case: q = 1 (Shannon entropy)
    if np.isclose(q, 1.0):
        return shannon_entropy(probabilities, base=base)

    if q <= 0:
        raise ValueError("q must be positive")

    # Remove zero probabilities
    non_zero = probabilities > 0
    if not np.any(non_zero):
        return 0.0

    # Calculate Tsallis entropy
    prob_power = np.power(probabilities[non_zero], q)
    sum_power = np.sum(prob_power)

    entropy = (1.0 / (q - 1.0)) * (1.0 - sum_power)

    return float(entropy)


def spatial_entropy(
    coordinates: np.ndarray,
    values: np.ndarray | None = None,
    method: str = "shannon",
    bins: int | tuple[int, int] | None = None,
    bandwidth: float | None = None,
    base: float = 2.0,
) -> float:
    """
    Calculate spatial entropy for point patterns or spatial data.

    Args:
        coordinates: Spatial coordinates (n x 2 or n x 3)
        values: Optional values at each location
        method: Entropy method ('shannon', 'renyi', 'tsallis')
        bins: Number of bins for discretization (int or (int, int) for 2D)
        bandwidth: Bandwidth for kernel density estimation

    Returns:
        Spatial entropy value
    """
    coordinates = np.asarray(coordinates)

    if coordinates.ndim != 2:
        raise ValueError("Coordinates must be 2D array")

    n_points, n_dims = coordinates.shape

    if n_points == 0:
        return 0.0

    # If values provided, use value-based entropy
    if values is not None:
        values = np.asarray(values).flatten()
        if len(values) != n_points:
            raise ValueError("Values must have same length as coordinates")

        # Discretize values
        if bins is None:
            bins = min(20, max(5, int(np.sqrt(n_points))))

        hist, _ = np.histogram(values, bins=bins)
        probabilities = hist / np.sum(hist)

        if method == "shannon":
            return shannon_entropy(probabilities)
        elif method == "renyi":
            return renyi_entropy(probabilities, alpha=2.0)
        elif method == "tsallis":
            return tsallis_entropy(probabilities, q=2.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Spatial pattern entropy using distance-based discretization
    if n_dims == 2:
        # Use 2D grid discretization
        if bins is None:
            bins = (int(np.sqrt(n_points)), int(np.sqrt(n_points)))
        elif isinstance(bins, int):
            bins = (bins, bins)

        x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

        # Create grid
        x_edges = np.linspace(x_min, x_max, bins[0] + 1)
        y_edges = np.linspace(y_min, y_max, bins[1] + 1)

        # Count points in each cell
        hist, _, _ = np.histogram2d(
            coordinates[:, 0], coordinates[:, 1], bins=[x_edges, y_edges]
        )

        probabilities = hist.flatten() / np.sum(hist)
        probabilities = probabilities[probabilities > 0]  # Remove empty cells

        if method == "shannon":
            return shannon_entropy(probabilities)
        elif method == "renyi":
            return renyi_entropy(probabilities, alpha=2.0)
        elif method == "tsallis":
            return tsallis_entropy(probabilities, q=2.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    else:
        raise ValueError("Spatial entropy currently supports 2D coordinates only")


def conditional_entropy(
    probabilities_xy: np.ndarray, probabilities_y: np.ndarray, base: float = 2.0
) -> float:
    """
    Calculate conditional entropy H(X|Y).

    Conditional entropy: H(X|Y) = -Σ p(x,y) * log(p(x|y))

    Args:
        probabilities_xy: Joint probability distribution p(x,y)
        probabilities_y: Marginal probability distribution p(y)
        base: Logarithm base

    Returns:
        Conditional entropy value
    """
    probabilities_xy = np.asarray(probabilities_xy)
    probabilities_y = np.asarray(probabilities_y)

    # Calculate conditional probabilities p(x|y)
    conditional_probs = np.zeros_like(probabilities_xy)

    for y_idx in range(len(probabilities_y)):
        if probabilities_y[y_idx] > 0:
            conditional_probs[:, y_idx] = (
                probabilities_xy[:, y_idx] / probabilities_y[y_idx]
            )

    # Calculate conditional entropy
    entropy = 0.0
    for y_idx in range(len(probabilities_y)):
        if probabilities_y[y_idx] > 0:
            for x_idx in range(len(probabilities_xy)):
                if conditional_probs[x_idx, y_idx] > 0:
                    entropy -= (
                        probabilities_xy[x_idx, y_idx]
                        * np.log(conditional_probs[x_idx, y_idx])
                        / np.log(base)
                    )

    return float(entropy)


def joint_entropy(probabilities_xy: np.ndarray, base: float = 2.0) -> float:
    """
    Calculate joint entropy H(X,Y).

    Joint entropy: H(X,Y) = -Σ p(x,y) * log(p(x,y))

    Args:
        probabilities_xy: Joint probability distribution p(x,y)
        base: Logarithm base

    Returns:
        Joint entropy value
    """
    probabilities_xy = np.asarray(probabilities_xy)

    # Flatten and normalize
    probs_flat = probabilities_xy.flatten()
    probs_flat = probs_flat[probs_flat > 0]  # Remove zeros

    if len(probs_flat) == 0:
        return 0.0

    # Normalize
    prob_sum = np.sum(probs_flat)
    if prob_sum > 0:
        probs_flat = probs_flat / prob_sum

    return shannon_entropy(probs_flat, base=base)


class EntropyCalculator:
    """
    Comprehensive entropy calculator for spatial data.

    Provides methods for calculating various entropy measures
    for spatial patterns and distributions.
    """

    def __init__(self, base: float = 2.0):
        """
        Initialize entropy calculator.

        Args:
            base: Logarithm base for entropy calculations
        """
        self.base = base

    def calculate(
        self, data: np.ndarray, method: str = "shannon", **kwargs: Any
    ) -> float:
        """
        Calculate entropy for given data.

        Args:
            data: Input data (probabilities, coordinates, or values)
            method: Entropy method ('shannon', 'renyi', 'tsallis')
            **kwargs: Additional parameters for specific methods

        Returns:
            Entropy value
        """
        if method == "shannon":
            return shannon_entropy(data, base=self.base, **kwargs)
        elif method == "renyi":
            alpha = kwargs.get("alpha", 2.0)
            return renyi_entropy(data, alpha=alpha, base=self.base)
        elif method == "tsallis":
            q = kwargs.get("q", 2.0)
            return tsallis_entropy(data, q=q, base=self.base)
        else:
            raise ValueError(f"Unknown method: {method}")

    def spatial_entropy(
        self,
        coordinates: np.ndarray,
        values: np.ndarray | None = None,
        method: str = "shannon",
        **kwargs: Any,
    ) -> float:
        """
        Calculate spatial entropy.

        Args:
            coordinates: Spatial coordinates
            values: Optional values at locations
            method: Entropy method
            **kwargs: Additional parameters

        Returns:
            Spatial entropy value
        """
        return spatial_entropy(
            coordinates, values, method=method, base=self.base, **kwargs
        )

    def conditional_entropy(
        self, probabilities_xy: np.ndarray, probabilities_y: np.ndarray
    ) -> float:
        """
        Calculate conditional entropy.

        Args:
            probabilities_xy: Joint probabilities
            probabilities_y: Marginal probabilities

        Returns:
            Conditional entropy value
        """
        return conditional_entropy(probabilities_xy, probabilities_y, base=self.base)

    def joint_entropy(self, probabilities_xy: np.ndarray) -> float:
        """
        Calculate joint entropy.

        Args:
            probabilities_xy: Joint probabilities

        Returns:
            Joint entropy value
        """
        return joint_entropy(probabilities_xy, base=self.base)
