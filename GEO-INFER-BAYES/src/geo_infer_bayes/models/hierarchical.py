"""
Hierarchical Bayesian models for geospatial applications.

This module provides hierarchical Bayesian models that can handle
multi-level spatial data structures.
"""

import numpy as np
from typing import Any
from .base import BayesianModel
from ._model_utils import posterior_draw_indices
from ..utils.rng import SeedLike, resolve_rng


class HierarchicalBayesianModel(BayesianModel):
    """
    Hierarchical Bayesian model for multi-level spatial data.

    This model implements a hierarchical structure where observations
    are grouped into levels with shared parameters.
    """

    def __init__(self, n_levels: int = 2, **kwargs: Any):
        """Initialize the hierarchical Bayesian model.

        Args:
            n_levels: Number of hierarchical levels. Each level gets its own
                partially pooled intercept.
            **kwargs: Additional model parameters forwarded to the base class.

        Raises:
            ValueError: If ``n_levels`` is not a positive integer.
        """
        if not isinstance(n_levels, (int, np.integer)) or n_levels < 1:
            raise ValueError("n_levels must be a positive integer")
        # Set before super().__init__, which calls _setup_model and needs this
        # to declare the per-level intercept parameters.
        self.n_levels = int(n_levels)
        self.levels: dict[str, Any] = {}
        super().__init__(name="HierarchicalBayesianModel", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the hierarchical model structure and parameters."""
        # Define parameter distributions for inference
        self.parameters = {
            "mu_alpha": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 10.0}},
            "sigma_alpha": {"prior": "half_normal", "hyperparams": {"sigma": 1.0}},
            "noise": {"prior": "half_normal", "hyperparams": {"sigma": 1.0}},
        }

        # Add level-specific parameters
        for level in range(self.n_levels):
            self.parameters[f"alpha_{level}"] = {
                "prior": "normal",
                "hyperparams": {"mu": "mu_alpha", "sigma": "sigma_alpha"},
            }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """
        Compute the log-likelihood for the hierarchical model.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
        data : dict
            Dictionary with data and grouping structure

        Returns
        -------
        float
            Log-likelihood value
        """
        # Extract data components
        observations = data["observations"]
        groups = data["groups"]

        noise = theta["noise"]

        # Extract level parameters
        alphas = [theta[f"alpha_{level}"] for level in range(self.n_levels)]

        # Compute predictions
        predictions = np.array([alphas[group] for group in groups])

        # Compute log-likelihood assuming Gaussian noise
        residuals = observations - predictions
        log_likelihood = -0.5 * np.sum(
            residuals**2 / noise**2 + np.log(2 * np.pi * noise**2)
        )

        return float(log_likelihood)

    def log_prior(self, theta: dict[str, Any]) -> float:
        """
        Compute the log-prior for the hierarchical model parameters.

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

        # Prior for mu_alpha
        mu_alpha = theta["mu_alpha"]
        mu = self.parameters["mu_alpha"]["hyperparams"]["mu"]
        sigma = self.parameters["mu_alpha"]["hyperparams"]["sigma"]
        log_prior += -0.5 * ((mu_alpha - mu) / sigma) ** 2 - np.log(
            sigma * np.sqrt(2 * np.pi)
        )

        # Prior for sigma_alpha
        sigma_alpha = theta["sigma_alpha"]
        sigma = self.parameters["sigma_alpha"]["hyperparams"]["sigma"]
        log_prior += -np.log(sigma_alpha) - sigma**2 / (2 * sigma_alpha**2)

        # Prior for noise
        noise = theta["noise"]
        sigma = self.parameters["noise"]["hyperparams"]["sigma"]
        log_prior += -np.log(noise) - sigma**2 / (2 * noise**2)

        # Priors for level alphas
        for level in range(self.n_levels):
            alpha = theta[f"alpha_{level}"]
            mu = theta["mu_alpha"]  # mu_alpha is used as mean for level alphas
            sigma = theta["sigma_alpha"]
            log_prior += -0.5 * ((alpha - mu) / sigma) ** 2 - np.log(
                sigma * np.sqrt(2 * np.pi)
            )

        return float(log_prior)

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Make predictions at new locations.

        Parameters
        ----------
        X_new : array-like
            New locations to predict at
        posterior : PosteriorAnalysis, optional
            Posterior analysis object. If None, use current parameters.
        samples : int, default=100
            Number of posterior samples to use
        return_std : bool, default=False
            Whether to return standard deviations

        Returns
        -------
        y_pred : ndarray
            Predictions
        y_std : ndarray, optional
            Standard deviations of predictions
        """
        if posterior is not None:
            # Use posterior samples
            all_preds = []

            names = [f"alpha_{level}" for level in range(self.n_levels)]
            for i in posterior_draw_indices(posterior, samples, names):
                # Extract level parameters
                alphas = [
                    posterior.samples[f"alpha_{level}"][i]
                    for level in range(self.n_levels)
                ]

                # Make predictions for new data
                if hasattr(X_new, "__len__") and len(X_new) > 0:
                    if isinstance(X_new[0], (list, tuple, np.ndarray)):
                        # X_new contains group indices
                        predictions = np.array([alphas[group] for group in X_new])
                    else:
                        # Assume all new points belong to level 0
                        predictions = np.full(len(X_new), alphas[0])
                else:
                    predictions = np.array([alphas[0]])

                all_preds.append(predictions)

            # Compute statistics across samples
            stacked = np.stack(all_preds)
            mean_pred = np.asarray(np.mean(stacked, axis=0), dtype=float)

            if return_std:
                return mean_pred, np.asarray(np.std(stacked, axis=0), dtype=float)
            return mean_pred
        else:
            raise RuntimeError(
                "Direct prediction requires a posterior. Pass a posterior or run fit() first."
            )

    def _per_draw_predictions(
        self, X: np.ndarray, posterior: Any, samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-draw predictive means and per-draw observation noise.

        The group alphas and the observation noise are taken from the *same*
        posterior draw, so the predictive refuses to mix group effects across
        draws -- the failure mode of the previous pooled-mean implementation,
        which collapsed the alpha spread down to one pooled mean and so
        understated total uncertainty badly.

        Parameters
        ----------
        X : ndarray
            Prediction locations; either group indices or a length-N scalar row.
        posterior : PosteriorAnalysis
            Posterior analysis object.
        samples : int
            Number of posterior draws to use.

        Returns
        -------
        tuple of ndarray
            ``(per_draw_means, per_draw_noise)`` with shapes ``(draws, n)`` and
            ``(draws,)`` respectively.
        """
        X = np.asarray(X)
        names = [f"alpha_{level}" for level in range(self.n_levels)] + ["noise"]
        indices = posterior_draw_indices(posterior, samples, names)
        per_draw_means: list[np.ndarray] = []
        per_draw_noise: list[float] = []
        for i in indices:
            alphas = [
                float(np.asarray(posterior.samples[f"alpha_{level}"])[i])
                for level in range(self.n_levels)
            ]
            if X.size == 0:
                predictions = np.asarray([alphas[0]], dtype=float)
            elif isinstance(X[0], (list, tuple, np.ndarray)):
                # Each row names a single group id; later rows may wrap it in a
                # length-1 sequence, so the scalar is extracted before indexing.
                predictions = np.asarray(
                    [alphas[int(np.asarray(g).reshape(-1)[0])] for g in X], dtype=float
                )
            else:
                predictions = np.full(len(X), alphas[0], dtype=float)
            per_draw_means.append(np.asarray(predictions, dtype=float))
            per_draw_noise.append(float(np.asarray(posterior.samples["noise"])[i]))
        return np.stack(per_draw_means), np.asarray(per_draw_noise, dtype=float)

    def posterior_predictive(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 100,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """
        Generate posterior predictive samples.

        Each draw yields a prediction from a *single* posterior draw -- both the
        per-group alpha and the observation noise come from the same sample --
        so the predictive spread reflects group-parameter uncertainty as well
        as observation noise, which is what makes the interval calibrated.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object
        X : array-like, optional
            Locations to generate predictions for. If None, use observed locations.
        samples : int, default=100
            Number of posterior samples to use
        random_seed : int or numpy.random.Generator, optional
            Seed or generator for the observation-noise draws. ``None``
            (default) means a generator seeded from OS entropy, so results are
            not replayable; pass an int to replay. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        ndarray of shape (samples, n_points)
            Posterior predictive samples
        """
        if X is None:
            # Fall back to the group structure seen during fitting.
            X = np.asarray(getattr(self, "observed_groups", []))
        means, noise = self._per_draw_predictions(np.asarray(X), posterior, samples)
        rng = resolve_rng(random_seed)
        draws = np.empty((len(noise), means.shape[1]), dtype=float)
        for i in range(len(noise)):
            draws[i] = rng.normal(means[i], np.sqrt(noise[i]))
        return draws

    def predictive_interval(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        level: float = 0.95,
        samples: int = 200,
        random_seed: SeedLike = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a calibrated posterior predictive interval ``(mean, lower, upper)``.

        The interval is taken from *predictive* draws (which carry observation
        noise and per-draw group effects), so its coverage is what a held-out
        observation is expected to satisfy.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object.
        X : array-like, optional
            Locations to predict at. If None, use the observed group structure.
        level : float, default=0.95
            Nominal coverage of the interval, in ``(0, 1)``.
        samples : int, default=200
            Number of posterior predictive draws to base the interval on.
        random_seed : SeedLike, optional
            Seed or generator for the observation-noise draws.

        Returns
        -------
        tuple of ndarray
            ``(mean, lower, upper)``, each with one entry per prediction point.
        """
        interval_level = float(level)
        if not np.isfinite(interval_level) or not 0.0 < interval_level < 1.0:
            raise ValueError(
                "level must be a finite probability strictly between zero and one"
            )
        draws = self.posterior_predictive(
            posterior, X=X, samples=samples, random_seed=random_seed
        )
        tail = (1.0 - interval_level) / 2.0
        mean = np.asarray(np.mean(draws, axis=0), dtype=float)
        lower = np.asarray(np.percentile(draws, 100.0 * tail, axis=0), dtype=float)
        upper = np.asarray(
            np.percentile(draws, 100.0 * (1.0 - tail), axis=0), dtype=float
        )
        return mean, lower, upper

    def uncertainty_decomposition(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 50,
    ) -> dict[str, np.ndarray]:
        """
        Decompose predictive uncertainty into epistemic and aleatoric parts.

        For the hierarchical model the *epistemic* share is the variance of the
        per-draw predictive means (group-parameter uncertainty) and the
        *aleatoric* share is the mean per-draw observation-noise variance.
        ``total = epistemic + aleatoric`` recovers the full predictive variance.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object.
        X : array-like, optional
            Locations to predict at. If None, use the observed group structure.
        samples : int, default=50
            Number of posterior draws to average over.

        Returns
        -------
        dict of str to ndarray
            ``epistemic``, ``aleatoric`` and ``total`` predictive standard
            deviations, plus the pooled ``mean`` prediction, one entry per
            prediction point.
        """
        if X is None:
            X = np.asarray(getattr(self, "observed_groups", []))
        means, noise = self._per_draw_predictions(np.asarray(X), posterior, samples)
        epistemic = np.var(means, axis=0)
        aleatoric = np.full(means.shape[1], float(np.mean(noise)), dtype=float)
        total = epistemic + aleatoric
        return {
            "mean": np.asarray(np.mean(means, axis=0), dtype=float),
            "epistemic": np.asarray(np.sqrt(epistemic), dtype=float),
            "aleatoric": np.asarray(np.sqrt(aleatoric), dtype=float),
            "total": np.asarray(np.sqrt(total), dtype=float),
        }
