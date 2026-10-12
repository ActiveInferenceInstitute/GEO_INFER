"""
Information Theory Convenience Methods

This module provides convenience methods for information theory operations
on spatial data.
"""

import numpy as np
from typing import Any
import logging

from geo_infer_math.core.information_theory import (
    EntropyCalculator,
    MutualInformationCalculator,
    KLDivergenceCalculator,
)

logger = logging.getLogger(__name__)


def spatial_entropy_helper(
    coordinates: np.ndarray,
    values: np.ndarray | None = None,
    method: str = "shannon",
    **kwargs: Any,
) -> float:
    """
    Helper for calculating spatial entropy.

    Args:
        coordinates: Spatial coordinates
        values: Optional values at locations
        method: Entropy method
        **kwargs: Additional parameters

    Returns:
        Spatial entropy value
    """
    from geo_infer_math.core.information_theory import spatial_entropy

    return spatial_entropy(coordinates, values, method=method, **kwargs)


def mutual_information_helper(
    coordinates_x: np.ndarray,
    values_x: np.ndarray,
    coordinates_y: np.ndarray,
    values_y: np.ndarray,
    **kwargs: Any,
) -> float:
    """
    Helper for calculating spatial mutual information.

    Args:
        coordinates_x: Coordinates for first dataset
        values_x: Values for first dataset
        coordinates_y: Coordinates for second dataset
        values_y: Values for second dataset
        **kwargs: Additional parameters

    Returns:
        Mutual information value
    """
    from geo_infer_math.core.information_theory import spatial_mutual_information

    return spatial_mutual_information(
        coordinates_x, values_x, coordinates_y, values_y, **kwargs
    )


def kl_divergence_helper(
    coordinates_p: np.ndarray,
    values_p: np.ndarray,
    coordinates_q: np.ndarray,
    values_q: np.ndarray,
    **kwargs: Any,
) -> float:
    """
    Helper for calculating spatial KL divergence.

    Args:
        coordinates_p: Coordinates for distribution P
        values_p: Values for distribution P
        coordinates_q: Coordinates for distribution Q
        values_q: Values for distribution Q
        **kwargs: Additional parameters

    Returns:
        KL divergence value
    """
    from geo_infer_math.core.information_theory import spatial_kl_divergence

    return spatial_kl_divergence(
        coordinates_p, values_p, coordinates_q, values_q, **kwargs
    )


class InformationTheoryConvenience:
    """
    Convenience class for information theory operations.

    Provides high-level methods for common information theory tasks.
    """

    def __init__(self) -> None:
        """Initialize information theory convenience class."""
        self.entropy_calc = EntropyCalculator()
        self.mi_calc = MutualInformationCalculator()
        self.kl_calc = KLDivergenceCalculator()

    def calculate_entropy(
        self, data: np.ndarray, method: str = "shannon", **kwargs: Any
    ) -> float:
        """
        Calculate entropy.

        Args:
            data: Input data
            method: Entropy method
            **kwargs: Additional parameters

        Returns:
            Entropy value
        """
        return self.entropy_calc.calculate(data, method=method, **kwargs)

    def calculate_mutual_information(
        self,
        probabilities_xy: np.ndarray,
        probabilities_x: np.ndarray,
        probabilities_y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Calculate mutual information.

        Args:
            probabilities_xy: Joint probabilities
            probabilities_x: Marginal probabilities for X
            probabilities_y: Marginal probabilities for Y
            **kwargs: Additional parameters

        Returns:
            Mutual information value
        """
        return self.mi_calc.calculate(
            probabilities_xy, probabilities_x, probabilities_y, **kwargs
        )

    def calculate_kl_divergence(
        self, p: np.ndarray, q: np.ndarray, **kwargs: Any
    ) -> float:
        """
        Calculate KL divergence.

        Args:
            p: Distribution P
            q: Distribution Q
            **kwargs: Additional parameters

        Returns:
            KL divergence value
        """
        return self.kl_calc.calculate(p, q, **kwargs)
