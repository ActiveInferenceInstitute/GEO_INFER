"""
Spatial causal models for geospatial applications.

This module provides spatial causal models for
causal inference in geospatial contexts.
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
    posterior_vector,
    predictive_samples,
    scalar_parameter,
)


class SpatialCausalModel(BayesianModel):
    """
    Spatial causal model for geospatial causal inference.

    This model extends causal modeling to spatial contexts,
    accounting for spatial confounding and interference.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the spatial causal model."""
        super().__init__(name="SpatialCausalModel", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the spatial causal model."""
        self.parameters = {
            "treatment_effect": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "spatial_confounding": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute the log-likelihood for the spatial causal model."""
        if not isinstance(data, dict) or "X" not in data:
            raise ValueError("Spatial causal data must contain X and observations")
        observations = observations_from(data)
        design = features_from(data["X"])
        if design.shape[0] != observations.size:
            raise ValueError("X and observations must have the same length")
        treatment = design[:, 0]
        confounder = (
            np.mean(design[:, 1:], axis=1)
            if design.shape[1] > 1
            else np.zeros(design.shape[0])
        )
        effect = scalar_parameter(theta, "treatment_effect", 0.0)
        confounding = scalar_parameter(theta, "spatial_confounding", 0.0)
        prediction = effect * treatment + confounding * confounder
        return gaussian_log_likelihood(observations, prediction, 1.0)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the spatial causal model parameters."""
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
        treatment = design[:, 0]
        confounder = (
            np.mean(design[:, 1:], axis=1)
            if design.shape[1] > 1
            else np.zeros(design.shape[0])
        )
        effects = posterior_vector(posterior, "treatment_effect", samples)
        confounders = posterior_vector(posterior, "spatial_confounding", samples)
        if posterior is None:
            effects = np.array([0.0])
            confounders = np.array([0.0])
        predictions = np.asarray(
            [
                effect * treatment + confounding * confounder
                for effect, confounding in zip(effects, confounders)
            ]
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
