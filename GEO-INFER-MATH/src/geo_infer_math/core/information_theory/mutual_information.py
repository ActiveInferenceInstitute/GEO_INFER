"""
Mutual Information for Spatial Data

This module provides mutual information measures for analyzing
dependencies and relationships in spatial data.
"""

import numpy as np
from typing import Any
import logging

from geo_infer_math.core.information_theory.entropy import (
    shannon_entropy,
    conditional_entropy,
    joint_entropy,
)

logger = logging.getLogger(__name__)


def mutual_information(
    probabilities_xy: np.ndarray,
    probabilities_x: np.ndarray,
    probabilities_y: np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Calculate mutual information I(X;Y).

    Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
                          = H(X) - H(X|Y)
                          = H(Y) - H(Y|X)

    Args:
        probabilities_xy: Joint probability distribution p(x,y)
        probabilities_x: Marginal probability distribution p(x)
        probabilities_y: Marginal probability distribution p(y)
        base: Logarithm base

    Returns:
        Mutual information value
    """
    probabilities_xy = np.asarray(probabilities_xy)
    probabilities_x = np.asarray(probabilities_x)
    probabilities_y = np.asarray(probabilities_y)

    # Calculate entropies
    h_x = shannon_entropy(probabilities_x, base=base)
    h_y = shannon_entropy(probabilities_y, base=base)
    h_xy = joint_entropy(probabilities_xy, base=base)

    # Mutual information
    mi = h_x + h_y - h_xy

    return float(max(0.0, mi))  # MI is always non-negative


def conditional_mutual_information(
    probabilities_xyz: np.ndarray,
    probabilities_xz: np.ndarray,
    probabilities_yz: np.ndarray,
    probabilities_z: np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Calculate conditional mutual information I(X;Y|Z).

    Conditional MI: I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)

    Args:
        probabilities_xyz: Joint probability p(x,y,z)
        probabilities_xz: Joint probability p(x,z)
        probabilities_yz: Joint probability p(y,z)
        probabilities_z: Marginal probability p(z)
        base: Logarithm base

    Returns:
        Conditional mutual information value
    """
    # Calculate conditional entropies
    h_xz = conditional_entropy(probabilities_xz, probabilities_z, base=base)
    h_yz = conditional_entropy(probabilities_yz, probabilities_z, base=base)

    # Calculate H(X,Y|Z)
    h_xy_z = 0.0
    for z_idx in range(len(probabilities_z)):
        if probabilities_z[z_idx] > 0:
            # Extract p(x,y|z)
            p_xy_given_z = probabilities_xyz[:, :, z_idx] / probabilities_z[z_idx]
            p_xy_given_z_flat = p_xy_given_z.flatten()
            p_xy_given_z_flat = p_xy_given_z_flat[p_xy_given_z_flat > 0]

            if len(p_xy_given_z_flat) > 0:
                p_xy_given_z_flat = p_xy_given_z_flat / np.sum(p_xy_given_z_flat)
                h_xy_z += probabilities_z[z_idx] * shannon_entropy(
                    p_xy_given_z_flat, base=base
                )

    # Conditional mutual information
    cmi = h_xz + h_yz - h_xy_z

    return float(max(0.0, cmi))


def spatial_mutual_information(
    coordinates_x: np.ndarray,
    values_x: np.ndarray,
    coordinates_y: np.ndarray,
    values_y: np.ndarray,
    bins: int | tuple[int, int] | None = None,
    base: float = 2.0,
    distance_threshold: float | None = None,
) -> float:
    """
    Calculate mutual information between two spatial datasets.

    Args:
        coordinates_x: Coordinates for first dataset (n x 2)
        values_x: Values for first dataset (n)
        coordinates_y: Coordinates for second dataset (m x 2)
        values_y: Values for second dataset (m)
        bins: Number of bins for discretization
        base: Logarithm base
        distance_threshold: Optional distance threshold for spatial matching

    Returns:
        Spatial mutual information value
    """
    coordinates_x = np.asarray(coordinates_x)
    values_x = np.asarray(values_x).flatten()
    coordinates_y = np.asarray(coordinates_y)
    values_y = np.asarray(values_y).flatten()

    if len(values_x) != len(coordinates_x):
        raise ValueError("values_x must have same length as coordinates_x")
    if len(values_y) != len(coordinates_y):
        raise ValueError("values_y must have same length as coordinates_y")

    # If distance threshold provided, match nearby points
    if distance_threshold is not None:
        from scipy.spatial.distance import cdist

        distances = cdist(coordinates_x, coordinates_y)
        matched_pairs = distances < distance_threshold

        # Extract matched values
        matched_x = []
        matched_y = []

        for i in range(len(coordinates_x)):
            matches = np.where(matched_pairs[i])[0]
            if len(matches) > 0:
                # Use closest match
                closest = matches[np.argmin(distances[i, matches])]
                matched_x.append(values_x[i])
                matched_y.append(values_y[closest])

        if len(matched_x) == 0:
            return 0.0

        values_x = np.array(matched_x)
        values_y = np.array(matched_y)

    # Discretize values
    if bins is None:
        bins = min(20, max(5, int(np.sqrt(min(len(values_x), len(values_y))))))

    # Create joint histogram
    hist_xy, x_edges, y_edges = np.histogram2d(values_x, values_y, bins=bins)

    # Normalize to probabilities
    probabilities_xy = hist_xy / np.sum(hist_xy)

    # Calculate marginals
    probabilities_x = np.sum(probabilities_xy, axis=1)
    probabilities_y = np.sum(probabilities_xy, axis=0)

    # Calculate mutual information
    mi = mutual_information(
        probabilities_xy, probabilities_x, probabilities_y, base=base
    )

    return float(mi)


def normalized_mutual_information(
    probabilities_xy: np.ndarray,
    probabilities_x: np.ndarray,
    probabilities_y: np.ndarray,
    base: float = 2.0,
    normalization: str = "min",
) -> float:
    """
    Calculate normalized mutual information.

    Normalization options:
    - 'min': Normalize by min(H(X), H(Y))
    - 'max': Normalize by max(H(X), H(Y))
    - 'geometric': Normalize by sqrt(H(X) * H(Y))
    - 'arithmetic': Normalize by (H(X) + H(Y)) / 2

    Args:
        probabilities_xy: Joint probabilities
        probabilities_x: Marginal probabilities for X
        probabilities_y: Marginal probabilities for Y
        base: Logarithm base
        normalization: Normalization method

    Returns:
        Normalized mutual information (0 to 1)
    """
    mi = mutual_information(
        probabilities_xy, probabilities_x, probabilities_y, base=base
    )

    h_x = shannon_entropy(probabilities_x, base=base)
    h_y = shannon_entropy(probabilities_y, base=base)

    if normalization == "min":
        denominator = min(h_x, h_y)
    elif normalization == "max":
        denominator = max(h_x, h_y)
    elif normalization == "geometric":
        denominator = np.sqrt(h_x * h_y)
    elif normalization == "arithmetic":
        denominator = (h_x + h_y) / 2.0
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if denominator == 0:
        return 0.0

    return float(mi / denominator)


class MutualInformationCalculator:
    """
    Comprehensive mutual information calculator for spatial data.

    Provides methods for calculating mutual information and
    conditional mutual information for spatial patterns.
    """

    def __init__(self, base: float = 2.0):
        """
        Initialize mutual information calculator.

        Args:
            base: Logarithm base for calculations
        """
        self.base = base

    def calculate(
        self,
        probabilities_xy: np.ndarray,
        probabilities_x: np.ndarray,
        probabilities_y: np.ndarray,
        normalized: bool = False,
        normalization: str = "min",
    ) -> float:
        """
        Calculate mutual information.

        Args:
            probabilities_xy: Joint probabilities
            probabilities_x: Marginal probabilities for X
            probabilities_y: Marginal probabilities for Y
            normalized: Whether to return normalized MI
            normalization: Normalization method if normalized=True

        Returns:
            Mutual information value
        """
        if normalized:
            return normalized_mutual_information(
                probabilities_xy,
                probabilities_x,
                probabilities_y,
                base=self.base,
                normalization=normalization,
            )
        else:
            return mutual_information(
                probabilities_xy, probabilities_x, probabilities_y, base=self.base
            )

    def spatial_mutual_information(
        self,
        coordinates_x: np.ndarray,
        values_x: np.ndarray,
        coordinates_y: np.ndarray,
        values_y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Calculate spatial mutual information.

        Args:
            coordinates_x: Coordinates for first dataset
            values_x: Values for first dataset
            coordinates_y: Coordinates for second dataset
            values_y: Values for second dataset
            **kwargs: Additional parameters

        Returns:
            Spatial mutual information value
        """
        return spatial_mutual_information(
            coordinates_x, values_x, coordinates_y, values_y, base=self.base, **kwargs
        )

    def conditional_mutual_information(
        self,
        probabilities_xyz: np.ndarray,
        probabilities_xz: np.ndarray,
        probabilities_yz: np.ndarray,
        probabilities_z: np.ndarray,
    ) -> float:
        """
        Calculate conditional mutual information.

        Args:
            probabilities_xyz: Joint probabilities p(x,y,z)
            probabilities_xz: Joint probabilities p(x,z)
            probabilities_yz: Joint probabilities p(y,z)
            probabilities_z: Marginal probabilities p(z)

        Returns:
            Conditional mutual information value
        """
        return conditional_mutual_information(
            probabilities_xyz,
            probabilities_xz,
            probabilities_yz,
            probabilities_z,
            base=self.base,
        )
