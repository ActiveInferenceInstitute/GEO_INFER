"""
Interface to PyMC for Bayesian computation.
"""

import numpy as np
import pymc as pm
import arviz as az
from typing import Any
from ..utils.rng import SeedLike, resolve_rng


class PyMCInterface:
    """
    Interface to PyMC for Bayesian computation.

    This class provides a bridge between GEO-INFER-BAYES models
    and PyMC's Bayesian computation capabilities.

    Parameters
    ----------
    model_config : dict, optional
        Configuration parameters for the PyMC model
    """

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        self.model_config = model_config or {}
        self.pymc_model: pm.Model | None = None
        self.trace: az.InferenceData | None = None
        self._model_type: str | None = None  # 'gp' or 'hierarchical'
        self.gp: pm.gp.Marginal | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def create_spatial_gp_model(
        self, X: np.ndarray, y: np.ndarray, kernel_type: str = "matern", **kwargs: Any
    ) -> pm.Model:
        """
        Create a PyMC Gaussian Process model for spatial data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spatial locations
        y : array-like of shape (n_samples,)
            Observations
        kernel_type : str, default='matern'
            Type of kernel: 'matern', 'rbf', 'exponential'
        **kwargs : dict
            Additional parameters for the GP model

        Returns
        -------
        pm.Model
            PyMC model object
        """
        with pm.Model() as model:
            # Priors for the parameters
            lengthscale = pm.LogNormal(
                "lengthscale",
                mu=kwargs.get("lengthscale_mu", 0.0),
                sigma=kwargs.get("lengthscale_sigma", 1.0),
            )

            variance = pm.LogNormal(
                "variance",
                mu=kwargs.get("variance_mu", 0.0),
                sigma=kwargs.get("variance_sigma", 1.0),
            )

            noise = pm.LogNormal(
                "noise",
                mu=kwargs.get("noise_mu", -2.0),
                sigma=kwargs.get("noise_sigma", 1.0),
            )

            # Define kernel based on type
            if kernel_type == "rbf":
                cov_func = pm.gp.cov.ExpQuad(X.shape[1], ls=lengthscale) * variance
            elif kernel_type == "matern":
                # PyMC's Matern52 kernel provides the supported fixed degree.
                cov_func = pm.gp.cov.Matern52(X.shape[1], ls=lengthscale) * variance
            elif kernel_type == "exponential":
                cov_func = pm.gp.cov.Exponential(X.shape[1], ls=lengthscale) * variance
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

            # Mean function (default to zero)
            mean_func = pm.gp.mean.Zero()

            # Create GP and add white noise
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

            # Add observations
            gp.marginal_likelihood("y_obs", X=X, y=y, sigma=noise)

        self.pymc_model = model
        self.gp = gp
        self.X_train = X
        self.y_train = y
        self._model_type = "gp"
        return model

    def create_hierarchical_model(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs: Any
    ) -> pm.Model:
        """
        Create a PyMC hierarchical Bayesian model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates
        y : array-like of shape (n_samples,)
            Observations
        groups : array-like of shape (n_samples,)
            Group indicators
        **kwargs : dict
            Additional parameters for the hierarchical model

        Returns
        -------
        pm.Model
            PyMC model object
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n_features = X.shape[1]

        with pm.Model() as model:
            # Priors for global parameters
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)

            mu_beta = pm.Normal("mu_beta", mu=0, sigma=10, shape=n_features)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, shape=n_features)

            # Varying intercepts
            alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)

            # Varying slopes
            beta = pm.Normal(
                "beta", mu=mu_beta, sigma=sigma_beta, shape=(n_groups, n_features)
            )

            # Observation noise
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Expected value
            mu = alpha[groups] + pm.math.sum(X * beta[groups], axis=1)

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        self.pymc_model = model
        self._model_type = "hierarchical"
        return model

    def sample(
        self,
        n_samples: int = 1000,
        n_warmup: int = 500,
        chains: int = 4,
        cores: int | None = None,
        sampler: str = "nuts",
        random_seed: SeedLike = 42,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Sample from the PyMC model.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to draw
        n_warmup : int, default=500
            Number of warmup iterations
        chains : int, default=4
            Number of MCMC chains
        cores : int, optional
            Number of cores to use
        sampler : str, default='nuts'
            Sampler to use: 'nuts', 'metropolis'
        random_seed : int or numpy.random.Generator, optional, default=42
            Seed or generator for the sampler, so traces are reproducible. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.
        **kwargs : dict
            Additional parameters for the sampler

        Returns
        -------
        InferenceData
            Inference data with samples
        """
        if self.pymc_model is None:
            raise ValueError("No PyMC model defined. Call create_*_model first.")

        with self.pymc_model:
            if sampler == "nuts":
                self.trace = pm.sample(
                    draws=n_samples,
                    tune=n_warmup,
                    chains=chains,
                    cores=cores,
                    random_seed=random_seed,
                    **kwargs,
                )
            elif sampler == "metropolis":
                self.trace = pm.sample(
                    draws=n_samples,
                    tune=n_warmup,
                    chains=chains,
                    cores=cores,
                    step=pm.Metropolis(),
                    random_seed=random_seed,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

        return self.trace

    def predict(
        self,
        X_new: np.ndarray,
        samples: int = 100,
        return_std: bool = False,
        groups_new: np.ndarray | None = None,
        random_seed: SeedLike = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted PyMC model.

        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features)
            New data to predict on.
        samples : int, default=100
            Number of posterior predictive samples to draw.
        return_std : bool, default=False
            Whether to also return posterior standard deviations.
        groups_new : array-like of shape (n_samples,), optional
            Group indicators for hierarchical model predictions. When None,
            predictions are marginalized over all groups.
        random_seed : int or numpy.random.Generator, optional
            Seed or generator for the subsampling and noise draws. ``None``
            (default) means a generator seeded from OS entropy, so results are
            not replayable; pass an int to replay. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Posterior predictive mean.
        y_std : ndarray of shape (n_samples,), optional
            Posterior predictive standard deviation (only when return_std=True).
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample() first.")
        if self.pymc_model is None:
            raise ValueError("No model defined. Call create_*_model first.")

        if self._model_type == "gp":
            return self._predict_gp(
                X_new, samples=samples, return_std=return_std, random_seed=random_seed
            )
        elif self._model_type == "hierarchical":
            return self._predict_hierarchical(
                X_new,
                samples=samples,
                return_std=return_std,
                groups_new=groups_new,
                random_seed=random_seed,
            )
        else:
            raise ValueError(
                f"Unknown model type '{self._model_type}'. "
                "Call create_spatial_gp_model() or create_hierarchical_model() first."
            )

    def _predict_gp(
        self,
        X_new: np.ndarray,
        samples: int = 100,
        return_std: bool = False,
        random_seed: SeedLike = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """GP posterior predictive via gp.conditional + sample_posterior_predictive."""
        if self.gp is None:
            raise ValueError(
                "GP object not stored. Re-create the model via create_spatial_gp_model()."
            )

        # Add the conditional variable for new locations inside the existing model.
        # Use a unique name to avoid collisions on repeated calls.
        pred_var = "f_pred_new"
        assert self.pymc_model is not None
        assert self.trace is not None
        with self.pymc_model:
            _ = self.gp.conditional(pred_var, Xnew=X_new)
            pred_idata = pm.sample_posterior_predictive(
                self.trace,
                var_names=[pred_var],
                random_seed=self.model_config.get("random_seed", None),
            )

        # Shape: (chains, draws, n_new) → flatten to (total_draws, n_new)
        f_samples = pred_idata.posterior_predictive[pred_var].values
        f_flat = f_samples.reshape(-1, f_samples.shape[-1])

        # Subsample to requested number
        n_available = f_flat.shape[0]
        if samples < n_available:
            rng = resolve_rng(random_seed)
            idx = rng.choice(n_available, size=samples, replace=False)
            f_flat = f_flat[idx]

        mean_pred: np.ndarray = np.asarray(f_flat.mean(axis=0))
        if return_std:
            std_pred: np.ndarray = np.asarray(f_flat.std(axis=0))
            return mean_pred, std_pred
        return mean_pred

    def _predict_hierarchical(
        self,
        X_new: np.ndarray,
        samples: int = 100,
        return_std: bool = False,
        groups_new: np.ndarray | None = None,
        random_seed: SeedLike = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Hierarchical model posterior predictive via linear combination of posterior draws."""
        rng = resolve_rng(random_seed)

        assert self.trace is not None
        post = self.trace.posterior
        alpha_samples = post["alpha"].values  # (chains, draws, n_groups)
        beta_samples = post["beta"].values  # (chains, draws, n_groups, n_features)
        sigma_samples = post["sigma"].values  # (chains, draws)

        n_chains, n_draws, n_groups = alpha_samples.shape

        # Flatten chains and draws
        alpha_flat = alpha_samples.reshape(-1, n_groups)  # (total, n_groups)
        beta_flat = beta_samples.reshape(
            -1, n_groups, X_new.shape[1]
        )  # (total, n_groups, n_feat)
        sigma_flat = sigma_samples.reshape(-1)  # (total,)

        total = alpha_flat.shape[0]
        draw_idx = rng.choice(total, size=min(samples, total), replace=False)

        all_preds = []
        for i in draw_idx:
            alpha_i = alpha_flat[i]  # (n_groups,)
            beta_i = beta_flat[i]  # (n_groups, n_features)
            sigma_i = sigma_flat[i]  # scalar

            if groups_new is not None:
                valid_groups = np.clip(groups_new, 0, n_groups - 1)
                mu = alpha_i[valid_groups] + np.einsum(
                    "ij,ij->i", X_new, beta_i[valid_groups]
                )
            else:
                # Marginalize: use population-level mean coefficients
                mu = np.mean(alpha_i) + X_new @ np.mean(beta_i, axis=0)

            all_preds.append(rng.normal(mu, sigma_i))

        all_preds_arr = np.stack(all_preds)  # (samples, n_new)
        mean_pred: np.ndarray = np.asarray(all_preds_arr.mean(axis=0))
        if return_std:
            std_pred: np.ndarray = np.asarray(all_preds_arr.std(axis=0))
            return mean_pred, std_pred
        return mean_pred

    def convert_to_geo_infer_format(
        self, trace: az.InferenceData | None = None
    ) -> dict[str, np.ndarray]:
        """
        Convert PyMC trace to GEO-INFER-BAYES format.

        Parameters
        ----------
        trace : InferenceData, optional
            PyMC trace. If None, use self.trace

        Returns
        -------
        dict
            Dictionary with parameter samples
        """
        if trace is None:
            trace = self.trace

        if trace is None:
            raise ValueError("No trace available")

        samples = {}
        post = trace.posterior
        for var_name in post.data_vars:
            # Flatten chain and draw dimensions
            samples[var_name] = post[var_name].values.reshape(
                -1, *post[var_name].shape[2:]
            )
            if samples[var_name].ndim > 1:
                # Flatten multi-dimensional parameters
                samples[var_name] = samples[var_name].reshape(
                    samples[var_name].shape[0], -1
                )

        return samples
