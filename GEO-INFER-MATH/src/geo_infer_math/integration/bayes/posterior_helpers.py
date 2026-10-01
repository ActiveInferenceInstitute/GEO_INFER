"""Posterior computation helpers for Bayesian inference.

Implements conjugate posterior calculations for standard Bayesian
families: Normal-Normal, Beta-Binomial, Gamma-Poisson.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class PosteriorHelpers:
    """Conjugate posterior computations for Bayesian inference.

    Provides closed-form posterior updates for conjugate prior-likelihood
    pairs commonly used in spatial statistics.
    """

    def __init__(self, epsilon: float = 1e-16) -> None:
        """Initialize posterior helpers.

        Args:
            epsilon: Numerical stability constant.
        """
        self._epsilon = epsilon
        logger.debug("PosteriorHelpers initialized")

    def calculate_posterior(
        self,
        likelihood_data: np.ndarray,
        prior_params: dict[str, Any],
        family: str = "normal_normal",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate conjugate posterior.

        Args:
            likelihood_data: Observed data array.
            prior_params: Prior hyperparameters. Keys depend on family:
                - normal_normal: 'mu_0', 'sigma_0', 'sigma' (known likelihood variance)
                - beta_binomial: 'alpha', 'beta'
                - gamma_poisson: 'alpha', 'beta'
            family: Conjugate family name.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary with posterior hyperparameters plus 'family' key.
        """
        data = np.asarray(likelihood_data, dtype=np.float64)

        if family == "normal_normal":
            return self._normal_normal(data, prior_params)
        elif family == "beta_binomial":
            return self._beta_binomial(data, prior_params)
        elif family == "gamma_poisson":
            return self._gamma_poisson(data, prior_params)
        else:
            raise ValueError(f"Unknown conjugate family: {family}")

    def _normal_normal(self, data: np.ndarray, prior: dict[str, Any]) -> dict[str, Any]:
        """Normal-Normal conjugate update.

        Prior: μ ~ N(μ₀, σ₀²)
        Likelihood: x_i ~ N(μ, σ²) (known σ²)
        Posterior: μ ~ N(μ_n, σ_n²)
            σ_n² = 1 / (1/σ₀² + n/σ²)
            μ_n  = σ_n² × (μ₀/σ₀² + n·x̄/σ²)
        """
        mu_0 = float(prior.get("mu_0", 0.0))
        sigma_0 = float(prior.get("sigma_0", 1.0))
        sigma = float(prior.get("sigma", 1.0))

        n = len(data)
        x_bar = float(np.mean(data)) if n > 0 else 0.0

        precision_prior = 1.0 / (sigma_0**2 + self._epsilon)
        precision_lik = n / (sigma**2 + self._epsilon)

        sigma_n_sq = 1.0 / (precision_prior + precision_lik)
        mu_n = sigma_n_sq * (mu_0 * precision_prior + x_bar * precision_lik)

        logger.debug(
            "Normal-Normal posterior: mu_n=%.4f, sigma_n=%.4f (n=%d)",
            mu_n,
            np.sqrt(sigma_n_sq),
            n,
        )
        return {
            "family": "normal_normal",
            "mu_n": float(mu_n),
            "sigma_n": float(np.sqrt(sigma_n_sq)),
            "n_obs": n,
        }

    def _beta_binomial(self, data: np.ndarray, prior: dict[str, Any]) -> dict[str, Any]:
        """Beta-Binomial conjugate update.

        Prior: θ ~ Beta(α, β)
        Likelihood: x_i ~ Bernoulli(θ)
        Posterior: θ ~ Beta(α + Σx_i, β + n - Σx_i)
        """
        alpha_0 = float(prior.get("alpha", 1.0))
        beta_0 = float(prior.get("beta", 1.0))

        n = len(data)
        successes = float(np.sum(data > 0.5))  # Treat as binary

        alpha_n = alpha_0 + successes
        beta_n = beta_0 + n - successes

        posterior_mean = alpha_n / (alpha_n + beta_n)

        logger.debug(
            "Beta-Binomial posterior: alpha_n=%.2f, beta_n=%.2f, mean=%.4f (n=%d)",
            alpha_n,
            beta_n,
            posterior_mean,
            n,
        )
        return {
            "family": "beta_binomial",
            "alpha_n": float(alpha_n),
            "beta_n": float(beta_n),
            "posterior_mean": float(posterior_mean),
            "n_obs": n,
        }

    def _gamma_poisson(self, data: np.ndarray, prior: dict[str, Any]) -> dict[str, Any]:
        """Gamma-Poisson conjugate update.

        Prior: λ ~ Gamma(α, β)
        Likelihood: x_i ~ Poisson(λ)
        Posterior: λ ~ Gamma(α + Σx_i, β + n)
        """
        alpha_0 = float(prior.get("alpha", 1.0))
        beta_0 = float(prior.get("beta", 1.0))

        n = len(data)
        total = float(np.sum(data))

        alpha_n = alpha_0 + total
        beta_n = beta_0 + n

        posterior_mean = alpha_n / (beta_n + self._epsilon)

        logger.debug(
            "Gamma-Poisson posterior: alpha_n=%.2f, beta_n=%.2f, mean=%.4f (n=%d)",
            alpha_n,
            beta_n,
            posterior_mean,
            n,
        )
        return {
            "family": "gamma_poisson",
            "alpha_n": float(alpha_n),
            "beta_n": float(beta_n),
            "posterior_mean": float(posterior_mean),
            "n_obs": n,
        }
