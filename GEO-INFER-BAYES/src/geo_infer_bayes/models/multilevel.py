"""
Multi-level Bayesian models for geospatial applications.

This module provides multi-level Bayesian models that can handle
complex hierarchical spatial data structures.
"""

import numpy as np
from typing import Any
from .base import BayesianModel
from ._model_utils import posterior_draw_indices
from ..utils.rng import SeedLike, resolve_rng


class MultilevelModel(BayesianModel):
    """
    Multi-level Bayesian model for complex hierarchical structures.

    This model extends the basic hierarchical model to handle
    more complex multi-level data structures.
    """

    def __init__(self, levels: list[str] | None = None, **kwargs: Any):
        """Initialize the multi-level model.

        Args:
            levels: Level names in the hierarchy, outermost first. The first
                entry is the global level and carries no random effect of its
                own. Defaults to ``["global", "regional", "local"]``.
            **kwargs: Additional model parameters forwarded to the base class.

        Raises:
            ValueError: If fewer than two level names are given, or if any name
                is duplicated.
        """
        resolved = list(levels) if levels else ["global", "regional", "local"]
        if len(resolved) < 2:
            raise ValueError(
                "levels must name a global level and at least one nested level"
            )
        if len(set(resolved)) != len(resolved):
            raise ValueError("level names must be unique")
        # Set before super().__init__, which calls _setup_model and needs these
        # to declare the per-level variance parameters.
        self.levels = resolved
        self.level_structure: dict[str, Any] = {}
        super().__init__(name="MultilevelModel", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the multi-level model structure and parameters."""
        # Define parameter distributions for inference
        self.parameters = {
            "global_mean": {
                "prior": "normal",
                "hyperparams": {"mu": 0.0, "sigma": 10.0},
            },
            "global_variance": {"prior": "half_normal", "hyperparams": {"sigma": 1.0}},
            "noise": {"prior": "half_normal", "hyperparams": {"sigma": 1.0}},
        }

        # Add level-specific parameters
        for level in self.levels[1:]:  # Skip global level
            self.parameters[f"{level}_variance"] = {
                "prior": "half_normal",
                "hyperparams": {"sigma": 1.0},
            }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """Compute the log-likelihood for the multi-level model.

        Uses a Gaussian likelihood with global mean pooling and level-specific
        random effects:
            y_i ~ N(mu_global + sum(level_effects), sigma^2)
        """
        observations = data.get("observations", np.array([]))
        if len(observations) == 0:
            return 0.0

        global_mean = theta.get("global_mean", 0.0)
        noise = theta.get("noise", 1.0)

        # Calculate expected values incorporating partial pool level effects
        predictions = np.full_like(observations, global_mean, dtype=float)

        # Add random effects from each level if data defines group indices
        for level in self.levels[1:]:  # skip global
            level_idx = data.get(f"{level}_indices")
            level_effects = theta.get(f"{level}_effects")
            if level_idx is not None and level_effects is not None:
                predictions += level_effects[level_idx]

        residuals = observations - predictions
        log_likelihood = -0.5 * np.sum(
            residuals**2 / noise**2 + np.log(2 * np.pi * noise**2)
        )
        return float(log_likelihood)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """Compute the log-prior for the multi-level model parameters."""
        log_prior = 0.0

        # Prior for global mean
        if "global_mean" in theta:
            mu = self.parameters["global_mean"]["hyperparams"]["mu"]
            sigma = self.parameters["global_mean"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((theta["global_mean"] - mu) / sigma) ** 2 - np.log(
                sigma * np.sqrt(2 * np.pi)
            )

        # Prior for global variance and noise
        for param in ["global_variance", "noise"]:
            if param in theta:
                sigma = self.parameters[param]["hyperparams"]["sigma"]
                log_prior += -np.log(theta[param]) - sigma**2 / (2 * theta[param] ** 2)

        # Priors for level-specific variances and associated effects
        for level in self.levels[1:]:
            var_key = f"{level}_variance"
            eff_key = f"{level}_effects"

            if var_key in theta:
                sigma = self.parameters[var_key]["hyperparams"]["sigma"]
                log_prior += -np.log(theta[var_key]) - sigma**2 / (
                    2 * theta[var_key] ** 2
                )

            if eff_key in theta and var_key in theta:
                # Hierarchical prior: effects ~ N(0, level_variance)
                level_var = theta[var_key]
                effects = theta[eff_key]
                log_prior += np.sum(
                    -0.5 * (effects / level_var) ** 2
                    - np.log(level_var * np.sqrt(2 * np.pi))
                )

        return float(log_prior)

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
        level_indices: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Make predictions incorporating partial pooled level effects."""
        n_obs = len(X_new)

        if posterior is not None:
            # One index set, reused for every parameter, so the global mean and
            # the level effects always come from the same draw. Draws are spread
            # across the chain rather than taken from its least-converged front.
            draws = posterior_draw_indices(posterior, samples, ["global_mean"])
            global_mean = np.asarray(posterior.samples["global_mean"])[draws]
            predictions = np.repeat(global_mean[:, np.newaxis], n_obs, axis=1)

            # Add level effects if requested
            if level_indices:
                for level, indices in level_indices.items():
                    eff_key = f"{level}_effects"
                    if eff_key in posterior.samples:
                        # eff_samples shape: (n_draws, n_groups)
                        eff_samples = np.asarray(posterior.samples[eff_key])[draws]
                        predictions = predictions + eff_samples[:, indices]

            mean_pred = np.asarray(np.mean(predictions, axis=0), dtype=float)
            if return_std:
                return mean_pred, np.asarray(np.std(predictions, axis=0), dtype=float)
            return mean_pred
        else:
            global_mean = getattr(self, "global_mean", 0.0)
            predictions = np.full(n_obs, global_mean)

            if level_indices:
                for level, indices in level_indices.items():
                    eff = getattr(self, f"{level}_effects", None)
                    if eff is not None:
                        predictions += np.array(eff)[indices]

            if return_std:
                return predictions, np.full_like(predictions, 1.0)
            return predictions

    def posterior_predictive(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 100,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """Generate posterior predictive samples.

        Args:
            posterior: Fitted posterior object.
            X: Input coordinates/features (required).
            samples: Number of posterior predictive samples.
            random_seed: Seed or generator for the observation-noise draws.
                ``None`` (default) means a generator seeded from OS entropy, so
                results are not replayable; pass an int to replay. See
                :func:`geo_infer_bayes.utils.rng.resolve_rng`.
        """
        if X is None:
            raise ValueError(
                "X is required for multilevel posterior predictive sampling"
            )

        # Get predictions
        predictions, std = self.predict(X, posterior, samples=samples, return_std=True)

        # Generate samples with noise
        rng = resolve_rng(random_seed)
        all_samples = []
        for i in posterior_draw_indices(posterior, samples, ["noise"]):
            # Each draw carries its own observation-noise variance, so the
            # predictive spread widens with posterior uncertainty about noise.
            noise_sample = float(np.asarray(posterior.samples["noise"])[i])
            all_samples.append(rng.normal(predictions, np.sqrt(noise_sample)))

        return np.stack(all_samples)
