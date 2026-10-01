"""
Bayesian time series models for geospatial applications.

This module provides Bayesian time series models for
temporal analysis of geospatial data.
"""

import numpy as np
from typing import Any
from .base import BayesianModel
from ._model_utils import (
    features_from,
    gaussian_log_likelihood,
    log_prior_from_parameters,
    posterior_vector,
    predictive_samples,
    observations_from,
    scalar_parameter,
)


class BayesianTimeSeriesModel(BayesianModel):
    """
    Bayesian time series model for geospatial data.

    This model provides Bayesian analysis of temporal patterns
    in geospatial data.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Bayesian time series model."""
        super().__init__(name="BayesianTimeSeriesModel", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the Bayesian time series model."""
        self.parameters = {
            "trend": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 1.0}},
            "seasonal": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 1.0}},
            "noise": {"prior": "half_normal", "hyperparams": {"sigma": 1.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute the log-likelihood for the time series model."""
        observations = observations_from(data)
        if isinstance(data, dict) and "time" in data:
            time = features_from(data["time"])[:, 0]
        else:
            time = np.arange(observations.size, dtype=float)
        if time.size != observations.size:
            raise ValueError("time and observations must have the same length")
        trend = scalar_parameter(theta, "trend", 0.0)
        seasonal = scalar_parameter(theta, "seasonal", 0.0)
        noise = scalar_parameter(theta, "noise", 1.0)
        period = (
            float(data.get("period", max(observations.size, 2)))
            if isinstance(data, dict)
            else max(observations.size, 2)
        )
        if period <= 0:
            raise ValueError("period must be greater than zero")
        prediction = trend * time + seasonal * np.sin(2.0 * np.pi * time / period)
        return gaussian_log_likelihood(observations, prediction, noise)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the time series model parameters."""
        return log_prior_from_parameters(self.parameters, theta)

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Make predictions at new locations."""
        time = features_from(X_new)[:, 0]
        trend_samples = posterior_vector(posterior, "trend", samples)
        seasonal_samples = posterior_vector(posterior, "seasonal", samples)
        noise_samples = posterior_vector(posterior, "noise", samples)
        if posterior is not None and (
            trend_samples.size == 0 or seasonal_samples.size == 0
        ):
            raise ValueError("posterior must contain trend and seasonal samples")
        if posterior is None:
            trend_samples = np.array([0.0])
            seasonal_samples = np.array([0.0])
            noise_samples = np.array([1.0])
        period = max(time.size, 2)
        predictions = np.asarray(
            [
                trend * time + seasonal * np.sin(2.0 * np.pi * time / period)
                for trend, seasonal in zip(trend_samples, seasonal_samples)
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
        self, posterior: Any, X: np.ndarray | None = None, samples: int = 100
    ) -> np.ndarray:
        """Generate posterior predictive samples."""
        if X is None:
            raise ValueError("X is required to generate posterior predictive samples")
        mean, std = self.predict(X, posterior, samples=samples, return_std=True)
        return predictive_samples(mean, std, samples)
