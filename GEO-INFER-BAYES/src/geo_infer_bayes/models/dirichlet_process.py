"""
Dirichlet Process mixture models for geospatial applications.

This module provides Dirichlet Process mixture models for
spatial clustering and density estimation.
"""

import numpy as np
from typing import Any
from .base import BayesianModel
from ..utils.rng import SeedLike
from ._model_utils import (
    features_from,
    log_prior_from_parameters,
    observations_from,
    posterior_vector,
    predictive_samples,
)


class DirichletProcessMixture(BayesianModel):
    """
    Dirichlet Process mixture model for spatial clustering.

    This model uses a Dirichlet Process prior to automatically
    determine the number of clusters in spatial data.
    """

    def __init__(
        self, alpha: float = 1.0, max_clusters: int = 10, **kwargs: Any
    ) -> None:
        """
        Initialize the Dirichlet Process mixture model.

        Args:
            alpha: Concentration parameter for the Dirichlet Process
            max_clusters: Maximum number of clusters to consider
            **kwargs: Additional model parameters
        """
        super().__init__(name="DirichletProcessMixture", **kwargs)
        self.alpha = alpha
        self.max_clusters = max_clusters

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the Dirichlet Process mixture model."""
        # Define parameter distributions for inference
        self.parameters = {
            "alpha": {"prior": "gamma", "hyperparams": {"shape": 1.0, "scale": 1.0}},
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
        """Compute the log-likelihood for the DP mixture model."""
        observations = observations_from(data)

        n_clusters = max(1, min(self.max_clusters, observations.size))
        means = np.asarray(
            [
                theta.get(
                    f"cluster_means_{cluster}",
                    np.quantile(observations, (cluster + 0.5) / n_clusters),
                )
                for cluster in range(n_clusters)
            ],
            dtype=float,
        )
        variances = np.asarray(
            [
                theta.get(
                    f"cluster_variances_{cluster}",
                    max(np.var(observations), np.finfo(float).eps),
                )
                for cluster in range(n_clusters)
            ],
            dtype=float,
        )
        if np.any(variances <= 0) or not np.all(np.isfinite(means)):
            raise ValueError(
                "Cluster means must be finite and variances must be positive"
            )

        component_ll = np.asarray(
            [
                -0.5
                * (
                    ((observations - mean) ** 2) / variance
                    + np.log(2.0 * np.pi * variance)
                )
                for mean, variance in zip(means, variances)
            ]
        )
        maximum = np.max(component_ll, axis=0)
        return float(
            np.sum(maximum + np.log(np.mean(np.exp(component_ll - maximum), axis=0)))
        )

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the DP mixture model parameters."""
        if "alpha" in theta and float(np.asarray(theta["alpha"])) <= 0:
            return -np.inf
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
        means = posterior_vector(posterior, "cluster_means_0", samples)
        variances = posterior_vector(posterior, "cluster_variances_0", samples)
        if means.size == 0:
            means = np.quantile(
                signal, np.linspace(0.0, 1.0, min(self.max_clusters, signal.size))
            )
        if variances.size == 0:
            variances = np.full(
                means.size, max(float(np.var(signal)), np.finfo(float).eps)
            )
        nearest = np.argmin(np.abs(signal[:, None] - means[None, :]), axis=1)
        prediction = np.asarray(means[nearest])
        if return_std:
            std_prediction: np.ndarray = np.asarray(
                np.sqrt(np.maximum(variances[nearest], np.finfo(float).eps))
            )
            return prediction, std_prediction
        return prediction

    def posterior_predictive(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 100,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """Generate posterior predictive samples.

        ``random_seed`` seeds the predictive draws; see
        :func:`geo_infer_bayes.utils.rng.resolve_rng`.
        """
        if X is None:
            raise ValueError("X is required to generate posterior predictive samples")
        predictions, std = self.predict(X, posterior, samples=samples, return_std=True)
        return predictive_samples(predictions, std, samples, seed=random_seed)
