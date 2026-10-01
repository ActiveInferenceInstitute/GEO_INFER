"""
KL Divergence and Related Measures for Spatial Data

This module provides Kullback-Leibler divergence and related
divergence measures for comparing spatial distributions.
"""

import numpy as np
from typing import Any
import logging

from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)


def kl_divergence(
    p: np.ndarray, q: np.ndarray, base: float = 2.0, epsilon: float = 1e-10
) -> float:
    """
    Calculate Kullback-Leibler divergence D_KL(P||Q).

    KL divergence: D_KL(P||Q) = Σ p(x) * log(p(x) / q(x))

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        base: Logarithm base
        epsilon: Small value to avoid division by zero

    Returns:
        KL divergence value (always non-negative)

    Raises:
        ValueError: If distributions have different lengths or invalid probabilities
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    # Normalize
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum <= 0:
        raise ValueError("Distribution P must have positive sum")
    if q_sum <= 0:
        raise ValueError("Distribution Q must have positive sum")

    p = p / p_sum
    q = q / q_sum

    # Add epsilon to avoid division by zero
    q = np.maximum(q, epsilon)

    # Calculate KL divergence
    # Only sum over non-zero p values
    non_zero = p > 0
    if not np.any(non_zero):
        return 0.0

    kl = np.sum(
        p[non_zero]
        * (np.log(p[non_zero] + epsilon) - np.log(q[non_zero]))
        / np.log(base)
    )

    return float(max(0.0, kl))


def js_divergence(
    p: np.ndarray, q: np.ndarray, base: float = 2.0, epsilon: float = 1e-10
) -> float:
    """
    Calculate Jensen-Shannon divergence D_JS(P||Q).

    JS divergence: D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
    where M = 0.5 * (P + Q)

    JS divergence is symmetric and bounded [0, 1] when base=2.

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        base: Logarithm base
        epsilon: Small value to avoid division by zero

    Returns:
        JS divergence value (0 to 1 when base=2)
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    # Normalize
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("Distributions must have positive sums")

    p = p / p_sum
    q = q / q_sum

    # Calculate mixture M
    m = 0.5 * (p + q)

    # Calculate JS divergence
    kl_pm = kl_divergence(p, m, base=base, epsilon=epsilon)
    kl_qm = kl_divergence(q, m, base=base, epsilon=epsilon)

    js = 0.5 * kl_pm + 0.5 * kl_qm

    return float(js)


def spatial_kl_divergence(
    coordinates_p: np.ndarray,
    values_p: np.ndarray,
    coordinates_q: np.ndarray,
    values_q: np.ndarray,
    bins: int | tuple[int, int] | None = None,
    base: float = 2.0,
    method: str = "histogram",
) -> float:
    """
    Calculate KL divergence between two spatial distributions.

    Args:
        coordinates_p: Coordinates for distribution P (n x 2)
        values_p: Values for distribution P (n)
        coordinates_q: Coordinates for distribution Q (m x 2)
        values_q: Values for distribution Q (m)
        bins: Number of bins for discretization
        base: Logarithm base
        method: Method for distribution estimation ('histogram', 'kde')

    Returns:
        Spatial KL divergence value
    """
    coordinates_p = np.asarray(coordinates_p)
    values_p = np.asarray(values_p).flatten()
    coordinates_q = np.asarray(coordinates_q)
    values_q = np.asarray(values_q).flatten()

    if len(values_p) != len(coordinates_p):
        raise ValueError("values_p must have same length as coordinates_p")
    if len(values_q) != len(coordinates_q):
        raise ValueError("values_q must have same length as coordinates_q")

    if method == "histogram":
        # Discretize using histogram
        if bins is None:
            bins = min(20, max(5, int(np.sqrt(min(len(values_p), len(values_q))))))

        # Create histograms with same bins
        all_values = np.concatenate([values_p, values_q])
        hist_p, edges = np.histogram(
            values_p, bins=bins, range=(all_values.min(), all_values.max())
        )
        hist_q, _ = np.histogram(
            values_q, bins=bins, range=(all_values.min(), all_values.max())
        )

        # Normalize to probabilities
        probabilities_p = hist_p / np.sum(hist_p)
        probabilities_q = hist_q / np.sum(hist_q)

        # Calculate KL divergence
        kl = kl_divergence(probabilities_p, probabilities_q, base=base)

        return float(kl)

    elif method == "kde":
        # Kernel density estimation with proper quadrature: both densities
        # are evaluated on a common padded grid, renormalized by their
        # trapezoidal integrals, and the KL divergence is computed as a
        # Riemann sum over the grid (a convergent quadrature of
        # integral p(x) log(p(x)/q(x)) dx).
        from scipy.stats import gaussian_kde

        kde_p = gaussian_kde(values_p)
        kde_q = gaussian_kde(values_q)

        all_values = np.concatenate([values_p, values_q])
        span = all_values.max() - all_values.min()
        low = all_values.min() - 0.5 * span
        high = all_values.max() + 0.5 * span
        grid = np.linspace(low, high, 512)
        dx = float(grid[1] - grid[0])

        p_density = kde_p(grid)
        q_density = kde_q(grid)

        # Renormalize by the numeric integral so each is a proper density
        p_density = p_density / trapezoid(p_density, grid)
        q_density = q_density / trapezoid(q_density, grid)

        integrand = np.where(
            p_density > 0,
            p_density * (np.log(p_density) - np.log(np.maximum(q_density, 1e-300))),
            0.0,
        )
        kl = float(np.sum(integrand) * dx) / np.log(base)
        return float(max(0.0, kl))

    else:
        raise ValueError(f"Unknown method: {method}")


def symmetric_kl_divergence(
    p: np.ndarray, q: np.ndarray, base: float = 2.0, epsilon: float = 1e-10
) -> float:
    """
    Calculate symmetric KL divergence.

    Symmetric KL: D_sym(P||Q) = 0.5 * D_KL(P||Q) + 0.5 * D_KL(Q||P)

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        base: Logarithm base
        epsilon: Small value to avoid division by zero

    Returns:
        Symmetric KL divergence value
    """
    kl_pq = kl_divergence(p, q, base=base, epsilon=epsilon)
    kl_qp = kl_divergence(q, p, base=base, epsilon=epsilon)

    return 0.5 * kl_pq + 0.5 * kl_qp


def renyi_divergence(
    p: np.ndarray,
    q: np.ndarray,
    alpha: float = 1.0,
    base: float = 2.0,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate Renyi divergence D_α(P||Q).

    Renyi divergence: D_α(P||Q) = (1/(α-1)) * log(Σ p(x)^α * q(x)^(1-α))

    For α=1, this reduces to KL divergence.

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        alpha: Order parameter (α > 0, α ≠ 1)
        base: Logarithm base
        epsilon: Small value to avoid division by zero

    Returns:
        Renyi divergence value
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    # Normalize
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("Distributions must have positive sums")

    p = p / p_sum
    q = q / q_sum

    # Handle special case: α = 1 (KL divergence)
    if np.isclose(alpha, 1.0):
        return kl_divergence(p, q, base=base, epsilon=epsilon)

    if alpha <= 0:
        raise ValueError("Alpha must be positive")

    # Add epsilon
    q = np.maximum(q, epsilon)

    # Calculate Renyi divergence
    non_zero = (p > 0) & (q > 0)
    if not np.any(non_zero):
        return 0.0

    sum_term = np.sum(np.power(p[non_zero], alpha) * np.power(q[non_zero], 1.0 - alpha))

    if sum_term <= 0:
        return 0.0

    renyi = (1.0 / (alpha - 1.0)) * (np.log(sum_term) / np.log(base))

    return float(max(0.0, renyi))


class KLDivergenceCalculator:
    """
    Comprehensive KL divergence calculator for spatial data.

    Provides methods for calculating various divergence measures
    for comparing spatial distributions.
    """

    def __init__(self, base: float = 2.0, epsilon: float = 1e-10):
        """
        Initialize KL divergence calculator.

        Args:
            base: Logarithm base
            epsilon: Small value to avoid division by zero
        """
        self.base = base
        self.epsilon = epsilon

    def calculate(self, p: np.ndarray, q: np.ndarray, method: str = "kl") -> float:
        """
        Calculate divergence between distributions.

        Args:
            p: Probability distribution P
            q: Probability distribution Q
            method: Divergence method ('kl', 'js', 'symmetric', 'renyi')

        Returns:
            Divergence value
        """
        if method == "kl":
            return kl_divergence(p, q, base=self.base, epsilon=self.epsilon)
        elif method == "js":
            return js_divergence(p, q, base=self.base, epsilon=self.epsilon)
        elif method == "symmetric":
            return symmetric_kl_divergence(p, q, base=self.base, epsilon=self.epsilon)
        elif method == "renyi":
            alpha = 2.0  # Default alpha
            return renyi_divergence(
                p, q, alpha=alpha, base=self.base, epsilon=self.epsilon
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def spatial_kl_divergence(
        self,
        coordinates_p: np.ndarray,
        values_p: np.ndarray,
        coordinates_q: np.ndarray,
        values_q: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Calculate spatial KL divergence.

        Args:
            coordinates_p: Coordinates for distribution P
            values_p: Values for distribution P
            coordinates_q: Coordinates for distribution Q
            values_q: Values for distribution Q
            **kwargs: Additional parameters

        Returns:
            Spatial KL divergence value
        """
        return spatial_kl_divergence(
            coordinates_p, values_p, coordinates_q, values_q, base=self.base, **kwargs
        )
