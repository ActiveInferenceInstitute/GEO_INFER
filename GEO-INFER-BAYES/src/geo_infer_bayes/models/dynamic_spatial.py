"""
Dynamic spatial models for geospatial applications.

This module provides dynamic spatial models for
time-varying spatial processes.
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


class DynamicSpatialModel(BayesianModel):
    """
    Dynamic spatial model for time-varying spatial processes.

    This model handles spatio-temporal data with dynamic spatial patterns.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the dynamic spatial model."""
        super().__init__(name="DynamicSpatialModel", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the dynamic spatial model."""
        self.parameters = {
            "spatial_trend": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "temporal_trend": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "noise": {"prior": "half_normal", "hyperparams": {"sigma": 1.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute the log-likelihood for the dynamic spatial model."""
        if not isinstance(data, dict) or "X" not in data:
            raise ValueError("Dynamic spatial data must contain X and observations")
        observations = observations_from(data)
        design = features_from(data["X"])
        if design.shape[0] != observations.size:
            raise ValueError("X and observations must have the same length")
        spatial = design[:, 0]
        temporal = (
            design[:, 1]
            if design.shape[1] > 1
            else np.arange(observations.size, dtype=float)
        )
        spatial_trend = scalar_parameter(theta, "spatial_trend", 0.0)
        temporal_trend = scalar_parameter(theta, "temporal_trend", 0.0)
        noise = scalar_parameter(theta, "noise", 1.0)
        prediction = spatial_trend * spatial + temporal_trend * temporal
        return gaussian_log_likelihood(observations, prediction, noise)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the dynamic spatial model parameters."""
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
        spatial = design[:, 0]
        temporal = (
            design[:, 1]
            if design.shape[1] > 1
            else np.arange(design.shape[0], dtype=float)
        )
        spatial_samples = posterior_vector(posterior, "spatial_trend", samples)
        temporal_samples = posterior_vector(posterior, "temporal_trend", samples)
        noise_samples = posterior_vector(posterior, "noise", samples)
        if posterior is not None and (
            spatial_samples.size == 0 or temporal_samples.size == 0
        ):
            raise ValueError(
                "posterior must contain spatial_trend and temporal_trend samples"
            )
        if posterior is None:
            spatial_samples = np.array([0.0])
            temporal_samples = np.array([0.0])
            noise_samples = np.array([1.0])
        predictions = np.asarray(
            [
                spatial_trend * spatial + temporal_trend * temporal
                for spatial_trend, temporal_trend in zip(
                    spatial_samples, temporal_samples
                )
            ]
        )
        mean_prediction: np.ndarray = np.asarray(predictions.mean(axis=0))
        if return_std:
            scale = np.sqrt(np.maximum(noise_samples, np.finfo(float).eps))
            std_prediction: np.ndarray = np.asarray(
                np.sqrt(np.var(predictions, axis=0) + np.mean(scale**2))
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
