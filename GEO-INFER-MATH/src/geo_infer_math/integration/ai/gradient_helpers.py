"""
Gradient Computation Helpers for Spatial ML

This module provides gradient computation helpers specifically
designed for spatial machine learning models.
"""

import numpy as np
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


class AIGradientHelpers:
    """
    Gradient computation helpers for spatial ML.

    Provides specialized gradient computation methods for
    spatial neural networks and ML models.
    """

    def __init__(self) -> None:
        """Initialize gradient helpers."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._cache: dict[str, np.ndarray] = {}
        self.logger.debug("AIGradientHelpers initialized")

    def compute_spatial_gradient(
        self,
        function: Callable,
        parameters: np.ndarray,
        spatial_context: np.ndarray | None = None,
        method: str = "finite_difference",
    ) -> np.ndarray:
        """
        Compute gradient with spatial context.

        Args:
            function: Function to differentiate
            parameters: Parameter values
            spatial_context: Optional spatial context
            method: Gradient computation method

        Returns:
            Gradient vector
        """
        from geo_infer_math.api.convenience.ai_convenience import gradient_helper

        return gradient_helper(function, parameters, method=method)

    def compute_hessian(
        self, function: Callable, parameters: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Hessian matrix.

        Args:
            function: Function to differentiate
            parameters: Parameter values
            epsilon: Step size

        Returns:
            Hessian matrix
        """
        parameters = np.asarray(parameters).flatten()
        n_params = len(parameters)
        hessian = np.zeros((n_params, n_params))

        # Compute Hessian using finite differences
        for i in range(n_params):
            for j in range(n_params):
                # Second derivative
                params_ij = parameters.copy()
                params_ij[i] += epsilon
                params_ij[j] += epsilon

                params_i = parameters.copy()
                params_i[i] += epsilon

                params_j = parameters.copy()
                params_j[j] += epsilon

                hessian[i, j] = (
                    function(params_ij)
                    - function(params_i)
                    - function(params_j)
                    + function(parameters)
                ) / (epsilon**2)

        return hessian

    def compute_gradient_with_regularization(
        self,
        function: Callable,
        parameters: np.ndarray,
        regularization: float = 0.01,
        reg_type: str = "l2",
    ) -> np.ndarray:
        """
        Compute gradient with regularization.

        Args:
            function: Function to differentiate
            parameters: Parameter values
            regularization: Regularization strength
            reg_type: Regularization type ('l1', 'l2')

        Returns:
            Gradient with regularization
        """
        from geo_infer_math.api.convenience.ai_convenience import gradient_helper

        # Compute base gradient
        gradient = gradient_helper(function, parameters)

        # Add regularization
        if reg_type == "l2":
            gradient += regularization * parameters
        elif reg_type == "l1":
            gradient += regularization * np.sign(parameters)

        return gradient
