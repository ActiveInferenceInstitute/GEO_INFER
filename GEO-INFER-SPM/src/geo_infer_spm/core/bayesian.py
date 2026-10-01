"""
Bayesian extensions for Statistical Parametric Mapping

This module implements Bayesian statistical methods for SPM analysis,
providing hierarchical models, posterior probability mapping, and
Bayesian model selection and comparison.

The implementation follows Bayesian principles for uncertainty quantification
and incorporates Active Inference concepts for optimal model selection.

Key Features:
- Hierarchical Bayesian models for spatial data
- Posterior probability mapping with credible intervals
- Bayesian model selection using Bayes factors
- Markov Chain Monte Carlo (MCMC) sampling
- Variational inference for scalable computation

Mathematical Foundation:
Bayesian inference uses posterior distributions:
P(θ|D) = P(D|θ) * P(θ) / P(D)

where θ are model parameters, D is data, P(D|θ) is likelihood,
P(θ) is prior, and P(D) is marginal likelihood.
"""

import numpy as np
from typing import Any, cast
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

try:  # Modern PyMC (v4+)
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    try:  # Legacy PyMC3 fallback
        import pymc3 as pm  # type: ignore[no-redef]

        PYMC_AVAILABLE = True
    except ImportError:
        PYMC_AVAILABLE = False
        logger.debug("PyMC is unavailable; Bayesian SPM uses empirical Bayes.")

from ..models.data_models import SPMData, SPMResult  # noqa: E402
from ..utils.rng import resolve_rng  # noqa: E402


def gelman_rubin_r_hat(chains: np.ndarray) -> np.ndarray:
    """Compute the split Gelman-Rubin R-hat convergence diagnostic.

    Implements the standard split-R-hat (Gelman et al., *Bayesian Data
    Analysis*, 3rd ed.): each chain is split into two half-chains and the
    potential scale reduction factor is computed from the between-half-chain
    variance ``B`` and within-half-chain variance ``W``::

        var_hat = (n - 1) / n * W + B / n
        R-hat   = sqrt(var_hat / W)

    Args:
        chains: Posterior draws with shape ``(n_chains, n_draws)`` for a
            scalar parameter, or ``(n_chains, n_draws, n_components)`` for a
            vector parameter. Requires at least 2 chains and 4 draws.

    Returns:
        Array of R-hat values, one per component (shape ``(1,)`` for scalar
        parameters). Chains that are constant and identical across chains
        yield 1.0; constant chains that differ between chains yield ``inf``.

    Raises:
        ValueError: If fewer than 2 chains or fewer than 4 draws per chain.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]
    if chains.ndim != 3:
        raise ValueError(
            "chains must have shape (n_chains, n_draws) or "
            f"(n_chains, n_draws, n_components); got {chains.shape}"
        )
    n_chains, n_draws, _ = chains.shape
    if n_chains < 2:
        raise ValueError("Split R-hat requires at least 2 chains")
    half = n_draws // 2
    if half < 2:
        raise ValueError("Split R-hat requires at least 4 draws per chain")

    # Split each chain into two half-chains: (2 * n_chains, half, n_components)
    split = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)

    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    W = variances.mean(axis=0)
    B = half * means.var(axis=0, ddof=1)

    # Constant chains: W == 0. Identical constants are trivially converged;
    # unequal constants never converge.
    var_hat = (half - 1) / half * W + B / half
    safe_W = np.where(W > 0, W, 1.0)
    r_hat = np.where(W > 0, np.sqrt(var_hat / safe_W), np.where(B > 0, np.inf, 1.0))
    return r_hat


class BayesianSPM:
    """
    Bayesian Statistical Parametric Mapping implementation.

    This class provides Bayesian inference methods for SPM analysis,
    including hierarchical models and posterior probability mapping.

    Attributes:
        model_spec: Specification of Bayesian model
        priors: Prior distributions for parameters
        posterior_samples: MCMC posterior samples
        diagnostics: MCMC diagnostics and convergence measures
    """

    def __init__(self, model_type: str = "hierarchical_glm"):
        """
        Initialize Bayesian SPM.

        Args:
            model_type: Type of Bayesian model ('hierarchical_glm', 'spatial_hierarchical')
        """
        self.model_type = model_type
        self.model_spec: dict[str, Any] | None = None
        self.priors: dict[str, Any] = {}
        if not PYMC_AVAILABLE and self.model_type != "empirical_bayes":
            logger.info("Using empirical Bayes approximation for %s", model_type)
        self.posterior_samples: dict[str, Any] | None = None
        self.diagnostics: dict[str, Any] = {}

    def fit_bayesian_glm(
        self,
        data: SPMData,
        design_matrix: np.ndarray,
        priors: dict[str, Any] | None = None,
        n_samples: int = 1000,
        n_tune: int = 1000,
        random_seed: int = 42,
    ) -> SPMResult:
        """
        Fit Bayesian GLM using MCMC sampling.

        Args:
            data: SPMData containing response and covariates
            design_matrix: Design matrix for GLM
            priors: Prior specifications for parameters
            n_samples: Number of MCMC samples
            n_tune: Number of tuning samples
            random_seed: Seed for the MCMC sampler (deterministic traces)

        Returns:
            SPMResult with posterior parameter estimates
        """
        if priors is None:
            priors = self._default_priors(design_matrix.shape[1])

        if PYMC_AVAILABLE and self.model_type != "empirical_bayes":
            return self._fit_pymc_glm(
                data, design_matrix, priors, n_samples, n_tune, random_seed
            )
        return self._fit_empirical_bayes_glm(
            data, design_matrix, priors, random_seed=random_seed
        )

    def _default_priors(self, n_regressors: int) -> dict[str, Any]:
        """Set default prior distributions."""
        priors = {
            "beta": {"type": "normal", "mu": 0, "sigma": 1},
            "sigma": {"type": "half_normal", "sigma": 1},
            "nu": {"type": "exponential", "lam": 1 / 30},  # For robust regression
        }
        # Different priors for intercept vs. other coefficients
        priors["beta_intercept"] = {"type": "normal", "mu": 0, "sigma": 10}

        return priors

    def _fit_pymc_glm(
        self,
        data: SPMData,
        design_matrix: np.ndarray,
        priors: dict[str, Any],
        n_samples: int,
        n_tune: int,
        random_seed: int = 42,
    ) -> SPMResult:
        """Fit GLM using PyMC MCMC sampling."""
        y = data.data.flatten() if data.data.ndim > 1 else data.data
        X = design_matrix

        with pm.Model() as _model:
            # Priors
            beta_intercept = pm.Normal(
                "beta_intercept",
                mu=priors["beta_intercept"]["mu"],
                sigma=priors["beta_intercept"]["sigma"],
            )

            beta_other = pm.Normal(
                "beta_other",
                mu=priors["beta"]["mu"],
                sigma=priors["beta"]["sigma"],
                shape=X.shape[1] - 1,
            )

            beta = pm.math.concatenate([[beta_intercept], beta_other])

            sigma = pm.HalfNormal("sigma", sigma=priors["sigma"]["sigma"])

            # Likelihood
            mu = pm.math.dot(X, beta)
            _likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            # Sample from posterior
            trace = pm.sample(
                n_samples,
                tune=n_tune,
                return_inferencedata=True,
                random_seed=random_seed,
            )

        # Extract posterior samples
        beta_samples = np.column_stack(
            [
                trace.posterior["beta_intercept"].values.flatten(),
                trace.posterior["beta_other"].values.reshape(-1, X.shape[1] - 1),
            ]
        )

        # Compute posterior means and credible intervals
        beta_mean = np.mean(beta_samples, axis=0)
        beta_ci_lower = np.percentile(beta_samples, 2.5, axis=0)
        beta_ci_upper = np.percentile(beta_samples, 97.5, axis=0)

        # Compute residuals
        y_hat = X @ beta_mean
        residuals = y - y_hat

        # Store results
        self.posterior_samples = {
            "beta": beta_samples,
            "sigma": trace.posterior["sigma"].values.flatten(),
        }

        # Create SPMResult
        from ..models.data_models import DesignMatrix

        design = DesignMatrix(matrix=X, names=[f"beta_{i}" for i in range(X.shape[1])])

        result = SPMResult(
            spm_data=data,
            design_matrix=design,
            beta_coefficients=beta_mean,
            residuals=residuals,
            model_diagnostics={
                "method": "Bayesian_GLM_PyMC",
                "n_samples": n_samples,
                "n_tune": n_tune,
                "beta_ci_lower": beta_ci_lower,
                "beta_ci_upper": beta_ci_upper,
                "r_hat": self._compute_r_hat(trace),
                "effective_sample_size": self._compute_ess(trace),
            },
        )

        return result

    def _fit_empirical_bayes_glm(
        self,
        data: SPMData,
        design_matrix: np.ndarray,
        priors: dict[str, Any],
        random_seed: int | None = None,
    ) -> SPMResult:
        """Fit GLM using empirical Bayes approximation.

        Args:
            data: SPMData containing response and covariates.
            design_matrix: Design matrix for GLM.
            priors: Prior specifications for parameters.
            random_seed: Optional seed for reproducible posterior samples. When
                ``None``, a generator seeded from OS entropy is created.
        """
        rng = resolve_rng(random_seed)

        # Use maximum a posteriori (MAP) estimation as approximation
        y = data.data.flatten() if data.data.ndim > 1 else data.data
        X = design_matrix

        def negative_log_posterior(beta: np.ndarray) -> float:
            """Negative log posterior for optimization."""
            mu = X @ beta

            # Likelihood (Gaussian)
            nll_likelihood = 0.5 * np.sum((y - mu) ** 2)

            # Prior (Gaussian)
            beta_prior_mu = np.zeros(len(beta))
            beta_prior_sigma = np.ones(len(beta))
            nll_prior = 0.5 * np.sum((beta - beta_prior_mu) ** 2 / beta_prior_sigma**2)

            return float(nll_likelihood + nll_prior)

        # Optimize MAP estimate
        beta_init = np.linalg.pinv(X) @ y
        result = minimize(negative_log_posterior, beta_init, method="BFGS")

        if not result.success:
            logger.debug("MAP optimization did not converge")
            beta_map = beta_init
        else:
            beta_map = result.x

        # Approximate posterior covariance
        # Hessian of negative log posterior ≈ posterior precision
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Empirical-Bayes MAP posterior covariance").
        cov_beta = np.linalg.pinv(X.T @ X + np.eye(X.shape[1]))

        # Compute residuals
        y_hat = X @ beta_map
        residuals = y - y_hat

        # Generate approximate posterior samples for posterior probability mapping
        try:
            beta_samples = rng.multivariate_normal(beta_map, cov_beta, 500)
        except np.linalg.LinAlgError:
            noise = rng.standard_normal((500, len(beta_map)))
            beta_samples = beta_map + noise * np.sqrt(np.abs(np.diag(cov_beta)))
        self.posterior_samples = {
            "beta": beta_samples,
            "sigma": np.full(500, float(np.std(residuals)) or 1.0),
        }

        # Create SPMResult
        from ..models.data_models import DesignMatrix

        design = DesignMatrix(matrix=X, names=[f"beta_{i}" for i in range(X.shape[1])])

        result = SPMResult(
            spm_data=data,
            design_matrix=design,
            beta_coefficients=beta_map,
            residuals=residuals,
            model_diagnostics={
                "method": "Empirical_Bayes_GLM",
                "beta_covariance": cov_beta,
                "beta_standard_errors": np.sqrt(np.diag(cov_beta)),
            },
        )

        return result

    def posterior_probability_map(
        self, statistical_map: np.ndarray, threshold: float = 0.95
    ) -> np.ndarray:
        """
        Compute posterior probability map.

        Args:
            statistical_map: Statistical parametric map
            threshold: Posterior probability threshold

        Returns:
            Posterior probability map
        """
        if self.posterior_samples is None:
            raise ValueError(
                "Model must be fitted before computing posterior probabilities"
            )

        # For Bayesian GLM, posterior probability that effect > 0
        beta_samples = self.posterior_samples.get("beta", None)
        if beta_samples is None:
            raise ValueError("Beta posterior samples not available")

        n_beta = beta_samples.shape[1] if beta_samples.ndim > 1 else len(beta_samples)

        if len(statistical_map) == n_beta:
            # stat_map aligns with beta dimensions — compute P(beta > 0) per coefficient
            posterior_prob = np.mean(beta_samples > 0, axis=0)
        else:
            # stat_map is over spatial locations — use normal CDF approximation
            posterior_prob = stats.norm.cdf(statistical_map)

        return cast(np.ndarray, posterior_prob)

    def bayesian_model_comparison(
        self, models: list[SPMResult], method: str = "bayes_factor"
    ) -> dict[str, Any]:
        """
        Compare Bayesian models using Bayes factors or information criteria.

        Args:
            models: List of fitted Bayesian models
            method: Comparison method ('bayes_factor', 'dic', 'waic')

        Returns:
            Dictionary with model comparison results
        """
        if method == "bayes_factor":
            return self._compute_bayes_factors(models)
        elif method == "dic":
            return self._compute_dic(models)
        elif method == "waic":
            return self._compute_waic(models)
        else:
            raise ValueError(f"Unknown comparison method: {method}")

    def _compute_bayes_factors(self, models: list[SPMResult]) -> dict[str, Any]:
        """Compute Bayes factors for model comparison."""
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Bayes factors via marginal likelihoods").

        bic_values = []
        for model in models:
            if "bic" in model.model_diagnostics:
                bic_values.append(model.model_diagnostics["bic"])
            else:
                # Approximate BIC
                n = model.spm_data.n_points
                k = model.design_matrix.n_regressors
                rss = np.sum(model.residuals**2)
                bic = n * np.log(rss / n) + k * np.log(n)
                bic_values.append(bic)

        bic_array = np.array(bic_values)
        min_bic = np.min(bic_array)

        # Bayes factor approximation: exp((BIC_min - BIC_i)/2)
        bayes_factors = np.exp((min_bic - bic_array) / 2)

        return {
            "method": "BIC_approximation",
            "bic_values": bic_values,
            "bayes_factors": bayes_factors,
            "best_model_index": np.argmin(bic_values),
        }

    def _compute_dic(self, models: list[SPMResult]) -> dict[str, Any]:
        """Compute Deviance Information Criterion."""
        dic_values = []

        for model in models:
            # DIC = D_bar + p_D, where D_bar is expected deviance, p_D is effective parameters
            # Deferred: see docs/deferred_statistical_methods.md
            # ("DIC effective-parameters").
            deviance = -2 * model.model_diagnostics.get("log_likelihood", 0)
            n_params = model.design_matrix.n_regressors

            # Approximate effective number of parameters
            p_d = n_params

            dic = deviance + 2 * p_d
            dic_values.append(dic)

        return {
            "method": "DIC",
            "dic_values": dic_values,
            "best_model_index": np.argmin(dic_values),
        }

    def _compute_waic(self, models: list[SPMResult]) -> dict[str, Any]:
        """Compute Widely Applicable Information Criterion (WAIC).

        WAIC = -2 * (ELPD - p_WAIC) where:
        - ELPD = sum_i log(mean_s p(y_i|theta_s))   (expected log pointwise predictive density)
        - p_WAIC = sum_i var_s(log p(y_i|theta_s))  (effective number of parameters)

        When posterior samples are not available, falls back to a BIC-approximated
        WAIC using residuals as a proxy for pointwise log-likelihood.
        """
        waic_values = []
        for model in models:
            try:
                y = (
                    model.spm_data.data.flatten()
                    if model.spm_data.data.ndim > 1
                    else model.spm_data.data
                )
                residuals = model.residuals
                n = len(y)
                k = model.design_matrix.n_regressors

                if self.posterior_samples and "beta" in self.posterior_samples:
                    # Real WAIC using posterior samples
                    beta_samples = self.posterior_samples["beta"]  # shape (S, k)
                    X = model.design_matrix.matrix
                    sigma_est = float(np.std(residuals)) or 1.0
                    # Pointwise log-likelihood for each sample: (S, n)
                    mu_s = beta_samples @ X.T  # (S, n)
                    diff = (y[np.newaxis, :] - mu_s) ** 2  # (S, n)
                    lppd_i = np.log(np.mean(np.exp(-0.5 * diff / sigma_est**2), axis=0))
                    var_i = np.var(-0.5 * diff / sigma_est**2, axis=0)
                    elpd = float(np.sum(lppd_i))
                    p_waic = float(np.sum(var_i))
                else:
                    # BIC-based approximation when no samples available
                    rss = float(np.sum(residuals**2))
                    sigma_sq = rss / max(n - k, 1)
                    # Pointwise log-likelihood under Gaussian assumption
                    lppd_approx = -0.5 * n * np.log(2 * np.pi * sigma_sq) - rss / (
                        2 * sigma_sq
                    )
                    elpd = lppd_approx
                    p_waic = k  # rough parameter count

                waic = -2.0 * (elpd - p_waic)
                waic_values.append(waic)
            except Exception as e:
                logger.debug("WAIC computation failed for model: %s", e)
                waic_values.append(float("inf"))

        best_idx = int(np.argmin(waic_values)) if waic_values else 0
        return {
            "method": "WAIC",
            "waic_values": waic_values,
            "best_model_index": best_idx,
        }

    def spatial_hierarchical_model(
        self,
        data: SPMData,
        design_matrix: np.ndarray,
        spatial_structure: dict[str, Any],
        random_seed: int | None = None,
    ) -> SPMResult:
        """
        Fit spatial hierarchical Bayesian model.

        Args:
            data: SPMData with spatial coordinates
            design_matrix: Design matrix for GLM
            spatial_structure: Spatial correlation structure specification
            random_seed: Optional seed for reproducible basis selection and
                posterior samples. When ``None`` the legacy global
                ``np.random`` state is used.

        Returns:
            SPMResult with hierarchical parameter estimates
        """
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Spatial hierarchical model (CAR / Gaussian process)").
        logger.debug(
            "Spatial hierarchical model uses an empirical-Bayes basis augmentation"
        )

        # Add spatial random effects to design matrix
        spatial_basis = self._create_spatial_basis(
            data.coordinates, spatial_structure, random_seed=random_seed
        )

        # Augment design matrix
        X_augmented = np.column_stack([design_matrix, spatial_basis])

        # Fit using empirical Bayes
        result = self._fit_empirical_bayes_glm(
            data,
            X_augmented,
            self._default_priors(X_augmented.shape[1]),
            random_seed=random_seed,
        )

        result.model_diagnostics["spatial_hierarchical"] = True
        result.model_diagnostics["spatial_structure"] = spatial_structure

        return result

    def _create_spatial_basis(
        self,
        coordinates: np.ndarray,
        spatial_structure: dict[str, Any],
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Create spatial basis functions for hierarchical model.

        Args:
            coordinates: Spatial coordinates (n_points, 2).
            spatial_structure: Spatial structure specification.
            random_seed: Optional seed for reproducible center selection. When
                ``None``, a generator seeded from OS entropy is created.
        """
        rng = resolve_rng(random_seed)

        n_points = coordinates.shape[0]
        n_basis = spatial_structure.get("n_basis", min(20, n_points // 10))

        # Simple Gaussian basis functions
        centers = coordinates[rng.choice(n_points, size=n_basis, replace=False)]
        scale = spatial_structure.get("scale", np.std(coordinates) / np.sqrt(n_basis))

        basis = np.zeros((n_points, n_basis))
        for i in range(n_basis):
            distances = np.linalg.norm(coordinates - centers[i], axis=1)
            basis[:, i] = np.exp(-((distances / scale) ** 2))

        return basis

    @staticmethod
    def _posterior_dim_sizes(posterior: Any) -> dict[str, int]:
        """Return posterior dimension sizes (xarray renamed ``.dims`` to ``.sizes``)."""
        sizes = getattr(posterior, "sizes", None)
        return sizes if sizes is not None else posterior.dims

    def _compute_r_hat(self, trace: Any) -> np.ndarray:
        """Compute split Gelman-Rubin R-hat for each posterior parameter.

        For vector parameters the reported value is the maximum across
        components (the conservative summary).
        """
        posterior = getattr(trace, "posterior", None)
        if posterior is not None and hasattr(posterior, "data_vars"):
            r_hat_values = []
            for var in posterior.data_vars:
                samples = np.asarray(posterior[var].values, dtype=float)
                if samples.ndim < 2:
                    continue
                if samples.ndim > 3:
                    samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
                # Conservative summary for vector parameters: worst component.
                r_hat_values.append(float(np.max(gelman_rubin_r_hat(samples))))
            if r_hat_values:
                return np.array(r_hat_values)
        logger.warning(
            "R-hat requires posterior samples with chain/draw dimensions; reporting NaN"
        )
        return np.array([np.nan])

    def _compute_ess(self, trace: Any) -> np.ndarray:
        """Compute effective sample size."""
        # Deferred: see docs/deferred_statistical_methods.md
        # ("MCMC effective sample size").
        try:
            n_draws = len(trace.posterior.draw)
            n_chains = self._posterior_dim_sizes(trace.posterior)["chain"]
            return np.array([n_draws * n_chains])
        except Exception:
            logger.warning(
                "ESS computation failed; reporting NaN (not a real effective "
                "sample size)"
            )
            return np.array([np.nan])

    def variational_inference(
        self, data: SPMData, design_matrix: np.ndarray, n_iterations: int = 100
    ) -> SPMResult:
        """
        Perform variational inference for scalable Bayesian computation.

        Args:
            data: SPMData containing response data
            design_matrix: Design matrix for GLM
            n_iterations: Number of variational inference iterations

        Returns:
            SPMResult with variational parameter estimates
        """
        # Implementation of mean-field variational inference
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Mean-field variational inference").

        y = data.data.flatten() if data.data.ndim > 1 else data.data
        X = design_matrix

        n_regressors = X.shape[1]
        n_points = len(y)

        # Initialize variational parameters (mean-field approximation)
        mu_beta = np.zeros(n_regressors)  # Mean of beta
        _sigma_beta = np.ones(n_regressors)  # Variance of beta
        a_sigma = b_sigma = 1.0  # Gamma parameters for sigma

        # Variational inference loop
        for iteration in range(n_iterations):
            # Update beta posterior given sigma
            sigma_sq = b_sigma / a_sigma  # Expected value of sigma^2
            Lambda_beta = X.T @ X / sigma_sq + np.eye(n_regressors)
            mu_beta = np.linalg.solve(Lambda_beta, X.T @ y / sigma_sq)

            # Update sigma posterior given beta
            residuals = y - X @ mu_beta
            a_sigma = (n_points + n_regressors) / 2
            b_sigma = 0.5 * (np.sum(residuals**2) + np.sum(mu_beta**2))

        # Compute final parameter estimates
        beta_map = mu_beta

        # Approximate covariance
        sigma_sq_final = b_sigma / a_sigma
        cov_beta = sigma_sq_final * np.linalg.inv(X.T @ X + np.eye(n_regressors))

        # Compute residuals
        y_hat = X @ beta_map
        residuals = y - y_hat

        # Create SPMResult
        from ..models.data_models import DesignMatrix

        design = DesignMatrix(matrix=X, names=[f"beta_{i}" for i in range(X.shape[1])])

        result = SPMResult(
            spm_data=data,
            design_matrix=design,
            beta_coefficients=beta_map,
            residuals=residuals,
            model_diagnostics={
                "method": "Variational_Inference",
                "n_iterations": n_iterations,
                "beta_covariance": cov_beta,
                "final_sigma_sq": sigma_sq_final,
            },
        )

        return result
