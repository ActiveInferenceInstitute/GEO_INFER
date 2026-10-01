"""
Approximate Bayesian Computation implementation for Bayesian inference.

This module provides Approximate Bayesian Computation (ABC) methods for
Bayesian inference when likelihood functions are intractable.
"""

import numpy as np
from typing import Any
from collections.abc import Callable
import logging
from ..utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


class ApproximateBayesianComputation:
    """
    Approximate Bayesian Computation (ABC) for Bayesian inference.

    This class implements ABC methods for models where the likelihood
    function is intractable or computationally expensive.
    """

    def __init__(
        self,
        model: Any,
        distance_metric: str = "euclidean",
        tolerance: float = 0.1,
        n_samples: int = 10000,
        random_seed: SeedLike = None,
    ):
        """
        Initialize the ABC sampler.

        Args:
            model: The model to perform inference on
            distance_metric: Distance metric for comparing simulated and observed data
            tolerance: Tolerance for accepting samples
            n_samples: Number of ABC samples to generate
            random_seed: Seed or generator for every draw this sampler
                makes. ``None`` (default) means a generator seeded from OS
                entropy, so the run is not replayable; pass an int to replay
                it. See :func:`geo_infer_bayes.utils.rng.resolve_rng`.
        """
        valid_metrics = {"euclidean", "manhattan", "mahalanobis"}
        if distance_metric.lower() not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {sorted(valid_metrics)}")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and strictly positive")
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        self.model = model
        self.distance_metric = distance_metric.lower()
        self.tolerance = float(tolerance)
        self.n_samples = int(n_samples)
        self.random_seed = random_seed
        self.rng: np.random.Generator = resolve_rng(random_seed)

    def run(
        self,
        observed_data: Any,
        simulator: Callable | None = None,
        progress_bar: bool = True,
        prior_samples: dict[str, np.ndarray] | Any | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | Any:
        """
        Run ABC sampling for the model.

        Args:
            observed_data: Observed data for inference
            simulator: Function that produces model data for a parameter draw
            progress_bar: Whether to show progress
            **kwargs: Additional arguments

        Returns:
            Posterior samples
        """
        if simulator is None:
            raise ValueError(
                "ABC requires an explicit simulator callable for the configured model"
            )

        samples = []
        accepted_count = 0

        # Generate samples until we have enough accepted samples
        total_attempts = 0
        max_attempts = self.n_samples * 10  # Safety limit

        while accepted_count < self.n_samples and total_attempts < max_attempts:
            # Sample from prior
            theta = self._sample_from_prior(prior_samples)

            # Simulate data
            simulated_data = simulator(theta)

            # Compute distance
            distance = self._compute_distance(simulated_data, observed_data)

            # Accept if distance is below tolerance
            if distance < self.tolerance:
                samples.append(theta)
                accepted_count += 1

            total_attempts += 1

        if accepted_count < self.n_samples:
            logger.warning(
                "Only %d samples accepted out of %d requested",
                accepted_count,
                self.n_samples,
            )

        # Convert to the expected format
        return self._convert_samples_to_dict(samples)

    def _sample_from_prior(
        self, prior_samples: dict[str, np.ndarray] | Any | None = None
    ) -> dict[str, float]:
        """Sample parameter values from the prior distribution."""
        if prior_samples is not None:
            if isinstance(prior_samples, dict):
                if not prior_samples:
                    raise ValueError("prior_samples must not be empty")
                lengths = {len(np.asarray(values)) for values in prior_samples.values()}
                if len(lengths) != 1:
                    raise ValueError("prior_samples parameters must have equal lengths")
                n_previous = next(iter(lengths))
                if n_previous == 0:
                    raise ValueError("prior_samples must contain at least one draw")
                index = int(self.rng.integers(n_previous))
                sampled = {}
                for parameter, values in prior_samples.items():
                    value = np.asarray(values)[index]
                    sampled[parameter] = (
                        value.item() if np.asarray(value).shape == () else value.copy()
                    )
                return sampled
            if hasattr(prior_samples, "data_vars"):
                return self._sample_from_prior(
                    {
                        parameter: values.values
                        for parameter, values in prior_samples.data_vars.items()
                    }
                )
            raise TypeError("prior_samples must be a mapping or xarray Dataset")

        theta = {}

        for param, param_info in self.model.parameters.items():
            if param_info["prior"] == "log_normal":
                mu = param_info["hyperparams"]["mu"]
                sigma = param_info["hyperparams"]["sigma"]
                theta[param] = np.exp(self.rng.normal(mu, sigma))
            elif param_info["prior"] == "normal":
                mu = param_info["hyperparams"]["mu"]
                sigma = param_info["hyperparams"]["sigma"]
                theta[param] = self.rng.normal(mu, sigma)
            elif param_info["prior"] == "uniform":
                low = param_info["hyperparams"]["low"]
                high = param_info["hyperparams"]["high"]
                theta[param] = self.rng.uniform(low, high)
            else:
                theta[param] = self.rng.normal(0, 1)

        return theta

    def _compute_distance(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """Compute a validated distance between simulated and observed data."""
        simulated = np.asarray(simulated, dtype=float).reshape(-1)
        observed = np.asarray(observed, dtype=float).reshape(-1)
        if simulated.shape != observed.shape:
            raise ValueError("simulated and observed data must have the same shape")
        if not np.all(np.isfinite(simulated)) or not np.all(np.isfinite(observed)):
            raise ValueError("simulated and observed data must be finite")
        if self.distance_metric == "euclidean":
            distance = np.linalg.norm(simulated - observed)
        elif self.distance_metric == "manhattan":
            distance = np.sum(np.abs(simulated - observed))
        elif self.distance_metric == "mahalanobis":
            diff = simulated - observed
            # Two draws cannot produce a full-rank covariance in general, so
            # use a regularized pseudoinverse as the explicit ABC approximation.
            covariance = np.atleast_2d(
                np.cov(np.vstack((simulated, observed)), rowvar=False)
            )
            scale = max(float(np.trace(covariance)) / max(covariance.shape[0], 1), 1.0)
            covariance += np.eye(covariance.shape[0]) * (scale * 1e-8)
            distance = np.sqrt(
                max(float(diff @ np.linalg.pinv(covariance) @ diff), 0.0)
            )
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        if not np.isfinite(distance):
            raise ValueError("distance calculation produced a non-finite value")
        return float(distance)

    def _convert_samples_to_dict(
        self, samples: list[dict[str, float]]
    ) -> dict[str, np.ndarray]:
        """Convert list of parameter dictionaries to the expected format."""
        if not samples:
            return {}

        # Get all parameter names
        param_names = samples[0].keys()

        result = {}
        for param in param_names:
            result[param] = np.array([sample[param] for sample in samples])

        return result

    def update(
        self,
        new_data: Any,
        previous_samples: dict[str, np.ndarray] | Any,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | Any:
        """
        Update ABC samples with new data.

        Args:
            new_data: New data for updating
            previous_samples: Previous ABC samples
            **kwargs: Additional arguments

        Returns:
            Updated samples
        """
        if previous_samples is None:
            raise ValueError("previous_samples are required for an ABC update")
        return self.run(new_data, prior_samples=previous_samples, **kwargs)
