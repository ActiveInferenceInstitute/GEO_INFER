"""
Information Geometry for Spatial Models

This module provides information geometry tools for analyzing
spatial models, including Fisher information, information metrics,
and geodesic distances.
"""

import numpy as np
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


def fisher_information_matrix(
    log_likelihood: Callable,
    parameters: np.ndarray,
    data: np.ndarray | None = None,
    method: str = "observed",
) -> np.ndarray:
    """
    Calculate Fisher information matrix.

    Fisher information: I(θ)_ij = E[∂²log L(θ)/∂θ_i ∂θ_j]

    Args:
        log_likelihood: Log-likelihood function log L(θ; data)
        parameters: Parameter vector θ
        data: Optional data for likelihood evaluation
        method: Calculation method ('observed', 'expected')

    Returns:
        Fisher information matrix
    """
    parameters = np.asarray(parameters).flatten()
    n_params = len(parameters)

    # Numerical differentiation step
    epsilon = 1e-6

    # Calculate Fisher information matrix
    fisher_matrix = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(n_params):
            # Calculate second derivative using finite differences
            params_ij = parameters.copy()
            params_ij[i] += epsilon
            params_ij[j] += epsilon

            params_i = parameters.copy()
            params_i[i] += epsilon

            params_j = parameters.copy()
            params_j[j] += epsilon

            # Evaluate log-likelihood
            if data is not None:
                ll_ij = log_likelihood(params_ij, data)
                ll_i = log_likelihood(params_i, data)
                ll_j = log_likelihood(params_j, data)
                ll_0 = log_likelihood(parameters, data)
            else:
                ll_ij = log_likelihood(params_ij)
                ll_i = log_likelihood(params_i)
                ll_j = log_likelihood(params_j)
                ll_0 = log_likelihood(parameters)

            # Second derivative
            fisher_matrix[i, j] = (ll_ij - ll_i - ll_j + ll_0) / (epsilon**2)

    # Fisher information is negative of Hessian
    fisher_matrix = -fisher_matrix

    return fisher_matrix


def information_metric(fisher_matrix: np.ndarray, delta_theta: np.ndarray) -> float:
    """
    Calculate information metric (Riemannian metric) distance.

    Information metric: ds² = Σ_ij I_ij(θ) dθ_i dθ_j

    Args:
        fisher_matrix: Fisher information matrix
        delta_theta: Parameter difference vector

    Returns:
        Information metric distance
    """
    fisher_matrix = np.asarray(fisher_matrix)
    delta_theta = np.asarray(delta_theta).flatten()

    if fisher_matrix.shape[0] != len(delta_theta):
        raise ValueError(
            "Fisher matrix and delta_theta must have compatible dimensions"
        )

    # Calculate metric distance
    metric = np.dot(delta_theta, np.dot(fisher_matrix, delta_theta))

    return float(np.sqrt(max(0.0, metric)))


def geodesic_distance(
    log_likelihood: Callable,
    theta1: np.ndarray,
    theta2: np.ndarray,
    data: np.ndarray | None = None,
    n_steps: int = 10,
) -> float:
    """
    Calculate geodesic distance between two parameter points.

    Geodesic distance is the shortest path in the information geometry
    space defined by the Fisher information metric.

    Args:
        log_likelihood: Log-likelihood function
        theta1: First parameter vector
        theta2: Second parameter vector
        data: Optional data for likelihood evaluation
        n_steps: Number of steps for numerical integration

    Returns:
        Geodesic distance
    """
    theta1 = np.asarray(theta1).flatten()
    theta2 = np.asarray(theta2).flatten()

    if len(theta1) != len(theta2):
        raise ValueError("Parameter vectors must have same length")

    # Simple approximation: integrate along straight line
    # In practice, this would solve the geodesic equation
    path_length = 0.0

    for i in range(n_steps):
        # Interpolate parameter
        alpha = i / n_steps
        theta = (1 - alpha) * theta1 + alpha * theta2

        # Calculate Fisher information at this point
        fisher_matrix = fisher_information_matrix(log_likelihood, theta, data=data)

        # Calculate step size
        if i < n_steps - 1:
            alpha_next = (i + 1) / n_steps
            theta_next = (1 - alpha_next) * theta1 + alpha_next * theta2
            delta_theta = theta_next - theta

            # Calculate metric distance for this step
            step_length = information_metric(fisher_matrix, delta_theta)
            path_length += step_length

    return float(path_length)


def information_distance(
    fisher_matrix1: np.ndarray,
    fisher_matrix2: np.ndarray,
    method: str = "bhattacharyya",
) -> float:
    """
    Calculate distance between two information geometries.

    Args:
        fisher_matrix1: First Fisher information matrix
        fisher_matrix2: Second Fisher information matrix
        method: Distance method ('bhattacharyya', 'geodesic')

    Returns:
        Information distance
    """
    fisher_matrix1 = np.asarray(fisher_matrix1)
    fisher_matrix2 = np.asarray(fisher_matrix2)

    if fisher_matrix1.shape != fisher_matrix2.shape:
        raise ValueError("Fisher matrices must have same shape")

    if method == "bhattacharyya":
        # Bhattacharyya distance approximation
        try:
            # Average Fisher matrix
            fisher_avg = 0.5 * (fisher_matrix1 + fisher_matrix2)

            # Calculate determinants
            det1 = np.linalg.det(fisher_matrix1)
            det2 = np.linalg.det(fisher_matrix2)
            det_avg = np.linalg.det(fisher_avg)

            if det1 <= 0 or det2 <= 0 or det_avg <= 0:
                return np.inf

            # Bhattacharyya distance
            distance = 0.125 * (
                np.log(det_avg) - 0.5 * np.log(det1) - 0.5 * np.log(det2)
            )

            return float(max(0.0, distance))
        except Exception as e:
            logger.warning(f"Bhattacharyya distance calculation failed: {e}")
            return np.inf

    elif method == "geodesic":
        # Fisher-Rao geodesic distance between two SPD Fisher information
        # matrices: d = sqrt(sum_i log(lambda_i)^2), where lambda_i are the
        # eigenvalues of G1^{-1/2} G2 G1^{-1/2} (equivalently the generalized
        # eigenvalues of the pair (G2, G1)).
        symmetric1 = 0.5 * (fisher_matrix1 + fisher_matrix1.T)
        symmetric2 = 0.5 * (fisher_matrix2 + fisher_matrix2.T)

        eigvals1, eigvecs1 = np.linalg.eigh(symmetric1)
        if np.any(eigvals1 <= 0):
            raise ValueError(
                "Fisher-Rao geodesic distance requires positive-definite "
                "Fisher information matrices"
            )
        sqrt_inv1 = eigvecs1 @ np.diag(1.0 / np.sqrt(eigvals1)) @ eigvecs1.T
        whitened = sqrt_inv1 @ symmetric2 @ sqrt_inv1
        generalized_eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
        if np.any(generalized_eigenvalues <= 0):
            raise ValueError(
                "Fisher-Rao geodesic distance requires positive-definite "
                "Fisher information matrices"
            )
        distance = np.sqrt(np.sum(np.log(generalized_eigenvalues) ** 2))
        return float(distance)

    else:
        raise ValueError(f"Unknown method: {method}")


def spatial_fisher_information(
    coordinates: np.ndarray,
    values: np.ndarray,
    model_function: Callable,
    parameters: np.ndarray,
) -> np.ndarray:
    """
    Calculate Fisher information for a spatial model.

    Args:
        coordinates: Spatial coordinates (n x 2)
        values: Observed values (n)
        model_function: Model function f(θ; x, y)
        parameters: Model parameters θ

    Returns:
        Fisher information matrix
    """
    coordinates = np.asarray(coordinates)
    values = np.asarray(values).flatten()
    parameters = np.asarray(parameters).flatten()

    if len(values) != len(coordinates):
        raise ValueError("Values must have same length as coordinates")

    n_params = len(parameters)
    fisher_matrix = np.zeros((n_params, n_params))

    # Numerical differentiation
    epsilon = 1e-6

    # Calculate gradient at each data point
    for point_idx in range(len(coordinates)):
        coord = coordinates[point_idx]

        # Calculate gradient
        gradient = np.zeros(n_params)

        for param_idx in range(n_params):
            params_plus = parameters.copy()
            params_plus[param_idx] += epsilon

            pred_plus = model_function(params_plus, coord[0], coord[1])
            pred_0 = model_function(parameters, coord[0], coord[1])

            gradient[param_idx] = (pred_plus - pred_0) / epsilon

        # Outer product for Fisher information
        fisher_matrix += np.outer(gradient, gradient)

    return fisher_matrix


class InformationGeometryCalculator:
    """
    Comprehensive information geometry calculator for spatial models.

    Provides methods for calculating Fisher information, information
    metrics, and geodesic distances for spatial models.
    """

    def __init__(self) -> None:
        """Initialize information geometry calculator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._metric_cache: dict[str, np.ndarray] = {}
        self.logger.debug("InformationGeometryCalculator initialized")

    def fisher_information(
        self,
        log_likelihood: Callable,
        parameters: np.ndarray,
        data: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate Fisher information matrix.

        Args:
            log_likelihood: Log-likelihood function
            parameters: Parameter vector
            data: Optional data

        Returns:
            Fisher information matrix
        """
        return fisher_information_matrix(log_likelihood, parameters, data=data)

    def information_distance(
        self,
        fisher_matrix1: np.ndarray,
        fisher_matrix2: np.ndarray,
        method: str = "bhattacharyya",
    ) -> float:
        """
        Calculate distance between information geometries.

        Args:
            fisher_matrix1: First Fisher information matrix
            fisher_matrix2: Second Fisher information matrix
            method: Distance method

        Returns:
            Information distance
        """
        return information_distance(fisher_matrix1, fisher_matrix2, method=method)

    def geodesic_distance(
        self,
        log_likelihood: Callable,
        theta1: np.ndarray,
        theta2: np.ndarray,
        data: np.ndarray | None = None,
    ) -> float:
        """
        Calculate geodesic distance.

        Args:
            log_likelihood: Log-likelihood function
            theta1: First parameter vector
            theta2: Second parameter vector
            data: Optional data

        Returns:
            Geodesic distance
        """
        return geodesic_distance(log_likelihood, theta1, theta2, data=data)

    def spatial_fisher_information(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        model_function: Callable,
        parameters: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Fisher information for spatial model.

        Args:
            coordinates: Spatial coordinates
            values: Observed values
            model_function: Model function
            parameters: Model parameters

        Returns:
            Fisher information matrix
        """
        return spatial_fisher_information(
            coordinates, values, model_function, parameters
        )
