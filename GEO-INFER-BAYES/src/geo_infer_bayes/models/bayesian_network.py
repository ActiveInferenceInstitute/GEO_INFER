"""
Bayesian network models for geospatial applications.

This module provides Bayesian network models for
modeling causal relationships in geospatial data.
"""

import numpy as np
from typing import Any
from .base import BayesianModel
from ..utils.rng import SeedLike
from ._model_utils import (
    features_from,
    gaussian_log_likelihood,
    log_prior_from_parameters,
    observations_from,
    parameter_array,
    posterior_vector,
    predictive_samples,
)


class BayesianNetwork(BayesianModel):
    """
    Bayesian network model for geospatial causal inference.

    This model uses directed acyclic graphs to model
    causal relationships in geospatial data.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Bayesian network model."""
        super().__init__(name="BayesianNetwork", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the Bayesian network model."""
        self.parameters = {
            "edge_weights": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "node_biases": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute the log-likelihood for the Bayesian network model."""
        if not isinstance(data, dict) or "X" not in data:
            raise ValueError("Bayesian network data must contain X and observations")
        observations = observations_from(data)
        design = features_from(data["X"])
        if design.shape[0] != observations.size:
            raise ValueError("X and observations must have the same length")
        edge_weights = parameter_array(theta, "edge_weights", np.ones(design.shape[1]))
        node_biases = parameter_array(theta, "node_biases", [0.0])
        prediction = np.dot(design, np.resize(edge_weights, design.shape[1])) + float(
            np.mean(node_biases)
        )
        return gaussian_log_likelihood(observations, prediction, 1.0)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the Bayesian network model parameters."""
        return log_prior_from_parameters(self.parameters, theta)

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Make predictions at new locations."""
        design = features_from(X_new)
        weights = posterior_vector(posterior, "edge_weights", samples)
        biases = posterior_vector(posterior, "node_biases", samples)
        if posterior is None:
            weights = np.ones(1)
            biases = np.zeros(1)
        signal = np.mean(design, axis=1)
        predictions = np.asarray(
            [weight * signal + bias for weight, bias in zip(weights, biases)]
        )
        mean_prediction: np.ndarray = np.asarray(predictions.mean(axis=0))
        if return_std:
            std_prediction: np.ndarray = (
                np.asarray(np.std(predictions, axis=0)) + np.finfo(float).eps
            )
            return mean_prediction, std_prediction
        return mean_prediction

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
        mean, std = self.predict(X, posterior, samples=samples, return_std=True)
        return predictive_samples(mean, std, samples, seed=random_seed)
