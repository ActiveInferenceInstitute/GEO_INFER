"""
Spatial clustering models for geospatial applications.

This module provides spatial clustering models for
identifying spatial patterns and clusters in geospatial data.
"""

import numpy as np
from typing import Any
from .base import BayesianModel
from ._model_utils import (
    features_from,
    log_prior_from_parameters,
    observations_from,
    parameter_array,
    posterior_vector,
    predictive_samples,
)


class SpatialClusteringModel(BayesianModel):
    """
    Spatial clustering model for geospatial data.

    This model identifies spatial clusters in geospatial data
    using Bayesian methods.
    """

    def __init__(self, n_clusters: int = 5, **kwargs: Any) -> None:
        """
        Initialize the spatial clustering model.

        Args:
            n_clusters: Number of clusters to identify
            **kwargs: Additional model parameters
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be greater than zero")
        super().__init__(name="SpatialClusteringModel", **kwargs)
        self.n_clusters = n_clusters

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the spatial clustering model."""
        self.parameters = {
            "cluster_means": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "cluster_variances": {
                "prior": "inverse_gamma",
                "hyperparams": {"alpha": 1.0, "beta": 1.0},
            },
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute the log-likelihood for the spatial clustering model."""
        observations = observations_from(data)
        means = parameter_array(
            theta,
            "cluster_means",
            np.linspace(observations.min(), observations.max(), self.n_clusters),
        )
        variances = parameter_array(theta, "cluster_variances", np.ones(means.size))
        if np.any(variances <= 0):
            raise ValueError("cluster_variances must be greater than zero")
        component_ll = np.asarray(
            [
                -0.5
                * (
                    ((observations - mean) ** 2) / variance
                    + np.log(2.0 * np.pi * variance)
                )
                for mean, variance in zip(means, np.resize(variances, means.size))
            ]
        )
        maximum = np.max(component_ll, axis=0)
        return float(
            np.sum(maximum + np.log(np.mean(np.exp(component_ll - maximum), axis=0)))
        )

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the spatial clustering model parameters."""
        return log_prior_from_parameters(self.parameters, theta)

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Make predictions at new locations."""
        signal = np.mean(features_from(X_new), axis=1)
        means = posterior_vector(posterior, "cluster_means", samples)
        variances = posterior_vector(posterior, "cluster_variances", samples)
        if means.size == 0:
            means = np.quantile(signal, np.linspace(0.0, 1.0, self.n_clusters))
        if variances.size == 0:
            spread = float(np.std(signal))
            variances = np.full(means.size, max(spread**2, np.finfo(float).eps))
        nearest = np.argmin(np.abs(signal[:, None] - means[None, :]), axis=1)
        prediction = np.asarray(means[nearest])
        if return_std:
            std_prediction: np.ndarray = np.asarray(
                np.sqrt(np.maximum(variances[nearest], np.finfo(float).eps))
            )
            return prediction, std_prediction
        return prediction

    def posterior_predictive(
        self, posterior: Any, X: np.ndarray | None = None, samples: int = 100
    ) -> np.ndarray:
        """Generate posterior predictive samples."""
        if X is None:
            raise ValueError("X is required to generate posterior predictive samples")
        mean, std = self.predict(X, posterior, samples=samples, return_std=True)
        return predictive_samples(mean, std, samples)
