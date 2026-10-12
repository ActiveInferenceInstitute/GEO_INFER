"""
Spatio-temporal Gaussian Process models for geospatial applications.

This module provides spatio-temporal Gaussian Process models that can handle
both spatial and temporal dependencies in geospatial data.
"""

import numpy as np
from typing import Any, cast
from dataclasses import dataclass
import logging

from .base import BayesianModel
from .spatial_gp import SpatialGP
from ..utils.rng import SeedLike, derive_int_seed, resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class SpatioTemporalConfig:
    """Configuration for spatio-temporal Gaussian Process models."""

    # Spatial parameters
    spatial_length_scale: float = 1.0
    spatial_variance: float = 1.0

    # Temporal parameters
    temporal_length_scale: float = 1.0
    temporal_variance: float = 1.0

    # Noise parameters
    observation_noise: float = 0.1
    process_noise: float = 0.01

    # Computational parameters
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    random_seed: SeedLike = None

    def __post_init__(self) -> None:
        """Validate positive kernel and optimization parameters."""
        positive_fields = (
            "spatial_length_scale",
            "spatial_variance",
            "temporal_length_scale",
            "temporal_variance",
            "observation_noise",
            "process_noise",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if (
            not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if (
            not np.isfinite(self.convergence_tolerance)
            or self.convergence_tolerance < 0
        ):
            raise ValueError("convergence_tolerance must be finite and non-negative")


class SpatioTemporalGP(BayesianModel):
    """
    Spatio-temporal Gaussian Process model for geospatial applications.

    This model combines spatial and temporal dependencies to provide
    comprehensive modeling of spatio-temporal phenomena.
    """

    def __init__(self, config: SpatioTemporalConfig | None = None):
        """
        Initialize the spatio-temporal Gaussian Process model.

        Args:
            config: Configuration parameters for the model
        """
        super().__init__(name="SpatioTemporalGP")
        self.config = config or SpatioTemporalConfig()

        # Initialize spatial and temporal components
        self.spatial_gp = SpatialGP(noise=self.config.observation_noise)
        self.temporal_gp: Any = None  # Will be initialized when needed

        # Model state
        self.is_fitted = False
        self.training_data: Any = None
        self.spatial_coords: np.ndarray | None = None
        self.temporal_coords: np.ndarray | None = None
        self.observations: np.ndarray | None = None

        self.rng: np.random.Generator = resolve_rng(self.config.random_seed)

    def fit(
        self,
        spatial_coords: np.ndarray,
        temporal_coords: np.ndarray,
        observations: np.ndarray,
        **kwargs: Any,
    ) -> "SpatioTemporalGP":
        """
        Fit the spatio-temporal Gaussian Process model to data.

        Args:
            spatial_coords: Array of shape (n_samples, 2) with [lat, lon] coordinates
            temporal_coords: Array of shape (n_samples,) with temporal coordinates
            observations: Array of shape (n_samples,) with observed values
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        logger.info("Fitting spatio-temporal Gaussian Process model...")

        # Validate inputs
        spatial_coords = np.asarray(spatial_coords, dtype=float)
        temporal_coords = np.asarray(temporal_coords, dtype=float).reshape(-1)
        observations = np.asarray(observations, dtype=float).reshape(-1)
        if spatial_coords.ndim != 2 or spatial_coords.shape[1] != 2:
            raise ValueError("spatial_coords must have shape (n_samples, 2)")
        if len(spatial_coords) != len(temporal_coords) or len(spatial_coords) != len(
            observations
        ):
            raise ValueError("All input arrays must have the same length")
        if len(observations) == 0:
            raise ValueError("At least one observation is required")
        if not all(
            np.all(np.isfinite(values))
            for values in (spatial_coords, temporal_coords, observations)
        ):
            raise ValueError("All fit inputs must be finite")

        # Store training data
        self.spatial_coords = spatial_coords.copy()
        self.temporal_coords = temporal_coords.copy()
        self.observations = observations.copy()

        # Fit spatial component
        logger.info("Fitting spatial component...")
        self.spatial_gp.fit(spatial_coords, observations)

        # Fit the temporal trend component.
        logger.info("Fitting temporal component...")
        self._fit_temporal_component()

        self.is_fitted = True
        logger.info("Spatio-temporal GP model fitted successfully")

        return self

    def _fit_temporal_component(self) -> None:
        """Fit the temporal component of the model."""
        assert self.observations is not None
        assert self.spatial_coords is not None
        temporal_residuals = self.observations - self.spatial_gp.predict(
            self.spatial_coords
        )

        # Fit a simple temporal trend
        assert self.temporal_coords is not None
        if len(self.temporal_coords) < 2 or np.allclose(
            self.temporal_coords, self.temporal_coords[0]
        ):
            temporal_trend = np.array([0.0, float(np.mean(temporal_residuals))])
        else:
            temporal_trend = np.polyfit(self.temporal_coords, temporal_residuals, 1)
        self.temporal_trend = temporal_trend

        # Calculate temporal variance
        self.temporal_variance = np.var(temporal_residuals)

    def _predict_temporal(self, temporal_coords: np.ndarray) -> np.ndarray:
        """Make temporal predictions."""
        if hasattr(self, "temporal_trend"):
            # Use fitted temporal trend
            return np.asarray(np.polyval(self.temporal_trend, temporal_coords))
        else:
            # Return zeros if no temporal component fitted
            return np.zeros_like(temporal_coords)

    def sample(
        self,
        spatial_coords: np.ndarray,
        temporal_coords: np.ndarray,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Generate samples from the spatio-temporal model.

        Args:
            spatial_coords: Array of shape (n_samples, 2) with [lat, lon] coordinates
            temporal_coords: Array of shape (n_samples,) with temporal coordinates
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, n_points) with generated samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")

        # Get predictions and uncertainties
        mean_pred, std_pred = self._predict_components(
            np.asarray(spatial_coords, dtype=float),
            np.asarray(temporal_coords, dtype=float),
            return_std=True,
        )

        # Generate samples
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        samples = self.rng.normal(mean_pred, std_pred, size=(n_samples, len(mean_pred)))

        return samples

    def get_model_parameters(self) -> dict[str, Any]:
        """Get the fitted model parameters."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing parameters")

        params = {
            "spatial_parameters": {
                "lengthscale": self.spatial_gp.lengthscale,
                "variance": self.spatial_gp.variance,
                "noise": self.spatial_gp.noise,
            },
            "temporal_variance": self.temporal_variance,
            "temporal_trend": getattr(self, "temporal_trend", None),
            "config": self.config,
        }

        return params

    def cross_validate(
        self,
        spatial_coords: np.ndarray,
        temporal_coords: np.ndarray,
        observations: np.ndarray,
        n_folds: int = 5,
    ) -> dict[str, float]:
        """
        Perform cross-validation on the model.

        Args:
            spatial_coords: Array of shape (n_samples, 2) with [lat, lon] coordinates
            temporal_coords: Array of shape (n_samples,) with temporal coordinates
            observations: Array of shape (n_samples,) with observed values
            n_folds: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation metrics
        """
        from sklearn.model_selection import KFold

        # KFold accepts only an int or a RandomState, so a configured
        # Generator is converted rather than rejected at runtime.
        kf = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=derive_int_seed(self.config.random_seed),
        )

        mse_scores = []
        mae_scores = []

        for train_idx, test_idx in kf.split(spatial_coords):
            # Split data
            train_spatial = spatial_coords[train_idx]
            train_temporal = temporal_coords[train_idx]
            train_obs = observations[train_idx]

            test_spatial = spatial_coords[test_idx]
            test_temporal = temporal_coords[test_idx]
            test_obs = observations[test_idx]

            # Fit model on training data
            model_copy = SpatioTemporalGP(self.config)
            model_copy.fit(train_spatial, train_temporal, train_obs)

            # Predict on test data
            test_X = np.column_stack((test_spatial, test_temporal))
            test_pred = model_copy.predict(test_X)

            # Calculate metrics
            mse = np.mean((test_obs - test_pred) ** 2)
            mae = np.mean(np.abs(test_obs - test_pred))

            mse_scores.append(mse)
            mae_scores.append(mae)

        return {
            "mse_mean": np.mean(mse_scores),
            "mse_std": np.std(mse_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
        }

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the spatio-temporal model structure and parameters."""
        # Define parameter distributions for inference
        self.parameters = {
            "spatial_lengthscale": {
                "prior": "log_normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "spatial_variance": {
                "prior": "log_normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "temporal_lengthscale": {
                "prior": "log_normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "temporal_variance": {
                "prior": "log_normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "noise": {"prior": "log_normal", "hyperparams": {"mu": -2.0, "sigma": 1.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute a Gaussian log-likelihood without mutating model state.

        ``data`` is a mapping containing ``spatial_coords``,
        ``temporal_coords``, and ``observations``.
        """
        if not isinstance(theta, dict) or not isinstance(data, dict):
            raise TypeError("theta must be a mapping and data must be a mapping")
        required = {"spatial_coords", "temporal_coords", "observations"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"data is missing required keys: {sorted(missing)}")
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating likelihood")

        spatial_coords = np.asarray(data["spatial_coords"], dtype=float)
        temporal_coords = np.asarray(data["temporal_coords"], dtype=float)
        observed_values = np.asarray(data["observations"], dtype=float)
        predictions = self._predict_components(
            spatial_coords, temporal_coords, theta=theta, return_std=False
        )
        noise = self._positive_scalar(
            theta.get("noise", self.config.observation_noise), "noise"
        )
        predictions_arr = np.asarray(predictions)
        std_pred = np.full(predictions_arr.shape, np.sqrt(noise), dtype=float)
        return self._gaussian_log_likelihood(observed_values, predictions_arr, std_pred)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """
        Compute the log-prior for the spatio-temporal model parameters.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values

        Returns
        -------
        float
            Log-prior value
        """
        log_prior = 0.0

        # Log-normal prior for spatial lengthscale
        if "spatial_lengthscale" in theta:
            mu = self.parameters["spatial_lengthscale"]["hyperparams"]["mu"]
            sigma = self.parameters["spatial_lengthscale"]["hyperparams"]["sigma"]
            log_prior += (
                -0.5 * ((np.log(theta["spatial_lengthscale"]) - mu) / sigma) ** 2
            )
            log_prior -= np.log(
                theta["spatial_lengthscale"] * sigma * np.sqrt(2 * np.pi)
            )

        # Log-normal prior for spatial variance
        if "spatial_variance" in theta:
            mu = self.parameters["spatial_variance"]["hyperparams"]["mu"]
            sigma = self.parameters["spatial_variance"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((np.log(theta["spatial_variance"]) - mu) / sigma) ** 2
            log_prior -= np.log(theta["spatial_variance"] * sigma * np.sqrt(2 * np.pi))

        # Log-normal prior for temporal lengthscale
        if "temporal_lengthscale" in theta:
            mu = self.parameters["temporal_lengthscale"]["hyperparams"]["mu"]
            sigma = self.parameters["temporal_lengthscale"]["hyperparams"]["sigma"]
            log_prior += (
                -0.5 * ((np.log(theta["temporal_lengthscale"]) - mu) / sigma) ** 2
            )
            log_prior -= np.log(
                theta["temporal_lengthscale"] * sigma * np.sqrt(2 * np.pi)
            )

        # Log-normal prior for temporal variance
        if "temporal_variance" in theta:
            mu = self.parameters["temporal_variance"]["hyperparams"]["mu"]
            sigma = self.parameters["temporal_variance"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((np.log(theta["temporal_variance"]) - mu) / sigma) ** 2
            log_prior -= np.log(theta["temporal_variance"] * sigma * np.sqrt(2 * np.pi))

        # Log-normal prior for noise
        if "noise" in theta:
            mu = self.parameters["noise"]["hyperparams"]["mu"]
            sigma = self.parameters["noise"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((np.log(theta["noise"]) - mu) / sigma) ** 2
            log_prior -= np.log(theta["noise"] * sigma * np.sqrt(2 * np.pi))

        return log_prior

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict from ``[x, y, time]`` rows, optionally over posterior draws.

        Args:
            X_new: Prediction matrix of shape ``(n_points, 3)``: x, y, and time.
            posterior: Posterior draws; ``None`` predicts from the fitted
                hyperparameters.
            samples: Maximum number of posterior draws to average.
            return_std: Also return the predictive standard deviation.
        """
        spatial_coords, temporal_coords = self._split_prediction_input(X_new)

        if posterior is None:
            return self._predict_components(
                spatial_coords, temporal_coords, return_std=return_std
            )

        if not isinstance(samples, (int, np.integer)) or samples < 1:
            raise ValueError("samples must be a positive integer")
        n_draws = min(int(samples), self._posterior_draw_count(posterior))
        if n_draws < 1:
            raise ValueError("posterior must contain at least one draw")
        all_preds = []
        for index in range(n_draws):
            theta = self._posterior_theta(posterior, index)
            prediction = self._predict_components(
                spatial_coords, temporal_coords, theta=theta, return_std=False
            )
            all_preds.append(prediction)
        draws = np.stack(all_preds)
        mean_prediction: np.ndarray = np.asarray(np.mean(draws, axis=0))
        if return_std:
            std_prediction: np.ndarray = np.asarray(np.std(draws, axis=0))
            return mean_prediction, std_prediction
        return mean_prediction

    @staticmethod
    def _split_prediction_input(X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and split the canonical three-column prediction matrix."""
        values = np.asarray(X_new, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("X_new must have shape (n_points, 3): x, y, and time")
        if not np.all(np.isfinite(values)):
            raise ValueError("X_new must contain only finite values")
        return values[:, :2], values[:, 2]

    def _predict_components(
        self,
        spatial_coords: np.ndarray,
        temporal_coords: np.ndarray,
        *,
        theta: dict[str, Any] | None = None,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict for split spatial/temporal coordinates with optional parameters."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        spatial_coords = np.asarray(spatial_coords, dtype=float)
        temporal_coords = np.asarray(temporal_coords, dtype=float).reshape(-1)
        if spatial_coords.ndim != 2 or spatial_coords.shape[1] != 2:
            raise ValueError("spatial_coords must have shape (n_points, 2)")
        if len(spatial_coords) != len(temporal_coords):
            raise ValueError(
                "spatial_coords and temporal_coords must have the same length"
            )
        if not np.all(np.isfinite(spatial_coords)) or not np.all(
            np.isfinite(temporal_coords)
        ):
            raise ValueError("Prediction coordinates must be finite")

        if theta is None:
            spatial_model = self.spatial_gp
            temporal_variance = float(self.temporal_variance)
        else:
            spatial_model = self._fitted_spatial_model(theta)
            temporal_variance = self._positive_scalar(
                theta.get("temporal_variance", self.temporal_variance),
                "temporal_variance",
            )

        if return_std:
            spatial_mean, spatial_std = spatial_model.predict(
                spatial_coords, return_std=True
            )
            mean: np.ndarray = np.asarray(
                spatial_mean + self._predict_temporal(temporal_coords)
            )
            std: np.ndarray = np.asarray(
                np.sqrt(np.maximum(np.square(spatial_std) + temporal_variance, 1e-12))
            )
            return mean, std
        spatial_mean_arr = np.asarray(
            spatial_model.predict(spatial_coords, return_std=False)
        )
        temporal_arr = np.asarray(self._predict_temporal(temporal_coords))
        return cast(np.ndarray, spatial_mean_arr + temporal_arr)

    def _fitted_spatial_model(self, theta: dict[str, Any]) -> SpatialGP:
        """Fit a spatial GP for one valid posterior parameter draw."""
        lengthscale = self._positive_scalar(
            theta.get("spatial_lengthscale", self.spatial_gp.lengthscale),
            "spatial_lengthscale",
        )
        variance = self._positive_scalar(
            theta.get("spatial_variance", self.spatial_gp.variance),
            "spatial_variance",
        )
        noise = self._positive_scalar(
            theta.get("noise", self.config.observation_noise), "noise"
        )
        model = SpatialGP(
            kernel=self.spatial_gp.kernel_type,
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            degree=self.spatial_gp.degree,
            jitter=self.spatial_gp.jitter,
        )
        model.fit(np.asarray(self.spatial_coords), np.asarray(self.observations))
        return model

    @staticmethod
    def _positive_scalar(value: Any, name: str) -> float:
        """Convert a parameter value to a finite positive scalar."""
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError(f"{name} must be scalar")
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return scalar

    @staticmethod
    def _gaussian_log_likelihood(
        observations: np.ndarray, predictions: np.ndarray, std: np.ndarray
    ) -> float:
        """Evaluate a Gaussian log-likelihood after validating aligned arrays."""
        observed = np.asarray(observations, dtype=float).reshape(-1)
        predicted = np.asarray(predictions, dtype=float).reshape(-1)
        scale = np.asarray(std, dtype=float).reshape(-1)
        if not (len(observed) == len(predicted) == len(scale)):
            raise ValueError("observations, predictions, and uncertainty must align")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("observations and predictions must be finite")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("uncertainty must be finite and positive")
        residual = observed - predicted
        return float(
            -0.5 * np.sum(residual**2 / scale**2 + np.log(2 * np.pi * scale**2))
        )

    @staticmethod
    def _posterior_samples(posterior: Any) -> Any:
        """Return a dict-like posterior sample container."""
        samples = getattr(posterior, "samples", posterior)
        if not isinstance(samples, dict) and not hasattr(samples, "__getitem__"):
            raise TypeError("posterior must expose mapping-like samples")
        return samples

    def _posterior_draw_count(self, posterior: Any) -> int:
        """Return the number of flattened posterior draws."""
        samples = self._posterior_samples(posterior)
        try:
            first = (
                next(iter(samples.values()))
                if isinstance(samples, dict)
                else next(iter(samples.data_vars.values()))
            )
        except (AttributeError, StopIteration) as exc:
            raise ValueError("posterior must contain named parameter draws") from exc
        values = np.asarray(getattr(first, "values", first))
        if values.ndim == 0:
            return 1
        return int(values.shape[0])

    def _posterior_theta(self, posterior: Any, index: int) -> dict[str, float]:
        """Extract and validate one scalar posterior parameter draw."""
        samples = self._posterior_samples(posterior)
        theta = {}
        for name in self.parameters:
            try:
                value = samples[name]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"posterior is missing parameter {name!r}") from exc
            values = np.asarray(getattr(value, "values", value)).reshape(-1)
            if index >= values.size:
                raise ValueError(f"posterior parameter {name!r} has too few draws")
            theta[name] = self._positive_scalar(values[index], name)
        return theta

    def posterior_predictive(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 100,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """
        Generate posterior predictive samples.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object
        X : array-like, optional
            Locations to generate predictions for. If None, use observed locations.
        samples : int, default=100
            Number of posterior samples to use
        random_seed : int or numpy.random.Generator, optional
            Seed or generator for the predictive draws. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        ndarray of shape (samples, n_points)
            Posterior predictive samples
        """
        if X is None:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before posterior prediction")
            assert self.spatial_coords is not None
            assert self.temporal_coords is not None
            spatial_coords = self.spatial_coords
            temporal_coords = self.temporal_coords
        else:
            spatial_coords, temporal_coords = self._split_prediction_input(X)

        if not isinstance(samples, (int, np.integer)) or samples < 1:
            raise ValueError("samples must be a positive integer")
        n_draws = min(int(samples), self._posterior_draw_count(posterior))
        if n_draws < 1:
            raise ValueError("posterior must contain at least one draw")
        rng = self.rng if random_seed is None else resolve_rng(random_seed)
        all_samples = []

        for i in range(n_draws):
            param_sample = self._posterior_theta(posterior, i)
            pred = self._predict_components(
                spatial_coords, temporal_coords, theta=param_sample
            )
            noise = self._positive_scalar(param_sample["noise"], "noise")
            noisy_sample = rng.normal(pred, np.sqrt(noise))
            all_samples.append(noisy_sample)

        return np.stack(all_samples)


# Convenience function for creating spatio-temporal GP models
def create_spatiotemporal_gp(
    config: SpatioTemporalConfig | None = None,
) -> SpatioTemporalGP:
    """
    Create a new spatio-temporal Gaussian Process model.

    Args:
        config: Configuration parameters for the model

    Returns:
        Configured SpatioTemporalGP instance
    """
    return SpatioTemporalGP(config)
