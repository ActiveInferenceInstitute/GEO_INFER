"""
Likelihood functions for Bayesian geospatial models.

This module provides likelihood function classes for Bayesian
inference in geospatial applications.
"""

import logging

import numpy as np
from scipy.special import gammaln
from typing import Any

logger = logging.getLogger(__name__)


class SpatialLikelihood:
    """
    Spatial likelihood functions for Bayesian models.

    This class provides likelihood functions that account for
    spatial structure in the data.
    """

    def __init__(self, likelihood_type: str = "gaussian", **kwargs: Any) -> None:
        """
        Initialize the spatial likelihood.

        Args:
            likelihood_type: Type of likelihood ('gaussian', 'poisson', 'binomial')
            **kwargs: Additional parameters for the likelihood
        """
        self.likelihood_type = likelihood_type.lower()
        self.parameters = kwargs

    def log_likelihood(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        spatial_weights: np.ndarray | None = None,
    ) -> float:
        """
        Compute the log likelihood for spatial data.

        Args:
            predictions: Predicted values
            observations: Observed values
            spatial_weights: Spatial weights for weighted likelihood

        Returns:
            Log likelihood value
        """
        if self.likelihood_type == "gaussian":
            return self._gaussian_likelihood(predictions, observations, spatial_weights)
        elif self.likelihood_type == "poisson":
            return self._poisson_likelihood(predictions, observations)
        elif self.likelihood_type == "binomial":
            return self._binomial_likelihood(predictions, observations)
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood_type}")

    def _gaussian_likelihood(
        self, pred: np.ndarray, obs: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        """Gaussian likelihood for continuous spatial data."""
        if weights is not None:
            # Weighted likelihood
            residuals = obs - pred
            weighted_residuals = residuals * weights
            sigma = self.parameters.get("sigma", 1.0)
            log_likelihood = -0.5 * np.sum(weighted_residuals**2 / sigma**2)
            log_likelihood -= len(obs) * np.log(sigma * np.sqrt(2 * np.pi))
        else:
            # Standard Gaussian likelihood
            sigma = self.parameters.get("sigma", 1.0)
            residuals = obs - pred
            log_likelihood = -0.5 * np.sum(residuals**2 / sigma**2)
            log_likelihood -= len(obs) * np.log(sigma * np.sqrt(2 * np.pi))

        return float(log_likelihood)

    def _poisson_likelihood(self, pred: np.ndarray, obs: np.ndarray) -> float:
        """Poisson likelihood for count spatial data."""
        if obs.size == 0:
            raise ValueError("Poisson likelihood requires at least one observation")
        # Ensure predictions are positive for Poisson
        pred = np.maximum(pred, 1e-10)

        # Poisson log-likelihood: Σ obs_i * log(pred_i) - pred_i - log(obs_i!)
        log_likelihood = np.sum(obs * np.log(pred) - pred)
        log_likelihood -= float(np.sum([gammaln(o + 1) for o in obs]))

        return float(log_likelihood)

    def _binomial_likelihood(self, pred: np.ndarray, obs: np.ndarray) -> float:
        """Binomial likelihood for binary spatial data."""
        n = self.parameters.get("n", 1)  # Number of trials

        # Convert predictions to probabilities
        prob = 1 / (1 + np.exp(-pred))

        # Binomial log-likelihood
        log_likelihood = np.sum(obs * np.log(prob) + (n - obs) * np.log(1 - prob))

        return float(log_likelihood)


class PoissonProcess:
    """
    Poisson process likelihood for spatial point patterns.

    This class provides likelihood functions for spatial point processes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Poisson process likelihood."""
        self.parameters = kwargs

    def log_likelihood(
        self, intensity: np.ndarray, points: np.ndarray, window: dict[str, float]
    ) -> float:
        """
        Compute the log likelihood for a spatial Poisson process.

        This is a homogeneous approximation: the intensity field is reduced
        to its mean over the observation window (via ``_integrate_intensity``),
        so the point term uses ``n_points * log(Λ)`` where ``Λ = ∫ λ`` is the
        integrated intensity, equivalent to ``Σ log λ`` under constant
        intensity. The exact inhomogeneous log-likelihood
        ``Σ log λ(x_i) − ∫ λ`` would require evaluating the intensity at each
        observed point; that refinement is documented but not implemented.

        Args:
            intensity: Intensity function values
            points: Point locations
            window: Observation window bounds

        Returns:
            Log likelihood value
        """
        # Compute integral of intensity over the window
        integral_intensity = self._integrate_intensity(intensity, window)

        # Number of points
        n_points = len(points)

        # Poisson process log-likelihood
        log_likelihood = -integral_intensity + n_points * np.log(integral_intensity)

        return float(log_likelihood)

    def _integrate_intensity(
        self, intensity: np.ndarray, window: dict[str, float]
    ) -> float:
        """Integrate the intensity function over the observation window."""
        # Simple rectangular integration
        area = (window["xmax"] - window["xmin"]) * (window["ymax"] - window["ymin"])
        mean_intensity = np.mean(intensity)

        return float(area * mean_intensity)


class GaussianLikelihood:
    """
    Gaussian likelihood functions for Bayesian models.

    This class provides standard Gaussian likelihood functions.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Gaussian likelihood."""
        self.parameters = kwargs

    def log_likelihood(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """
        Compute the Gaussian log likelihood.

        Args:
            predictions: Predicted values
            observations: Observed values

        Returns:
            Log likelihood value
        """
        sigma = self.parameters.get("sigma", 1.0)
        residuals = observations - predictions

        log_likelihood = -0.5 * np.sum(residuals**2 / sigma**2)
        log_likelihood -= len(observations) * np.log(sigma * np.sqrt(2 * np.pi))

        return float(log_likelihood)
