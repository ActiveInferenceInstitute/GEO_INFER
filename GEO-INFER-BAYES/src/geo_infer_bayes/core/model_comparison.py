"""
Model comparison and selection tools for Bayesian inference.

This module provides tools for comparing and selecting between
different Bayesian models using information criteria (AIC, BIC, DIC,
WAIC) and leave-one-out cross-validation.
"""

import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any
from ..utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Tools for comparing and selecting Bayesian models.

    This class provides methods for model comparison using
    information criteria and cross-validation.
    """

    def __init__(self, models: list[Any] | None = None) -> None:
        """
        Initialize the model comparison tool.

        Args:
            models: List of models to compare. Each model must expose
                    ``log_likelihood(theta, data) -> float`` and
                    ``parameters`` (dict of parameter definitions).
        """
        self.models: list[Any] = models or []
        self.comparison_results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_models(
        self, data: Any, method: str = "loo", random_seed: SeedLike = None
    ) -> dict[str, Any]:
        """
        Compare models using specified method.

        Args:
            data: Data for model comparison. For LOO and WAIC this must be a
                  dict containing 'log_likelihood_matrix' of shape
                  (n_samples, n_obs) **or** a dict that the model's
                  ``log_likelihood`` can accept.
            method: Comparison method ('loo', 'waic', 'dic')
            random_seed: Seed or generator for the prior-draw fallback used
                when ``data`` has no ``log_likelihood_matrix``. ``None``
                (default) means a generator seeded from OS entropy, so the
                estimate is not replayable; pass an int to replay. See
                :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns:
            Dictionary with comparison results keyed by model name, plus a
            'ranking' key containing the ranked model names (best first).
        """
        if not self.models:
            raise ValueError("No models to compare.")

        results: dict[str, Any] = {}

        for i, model in enumerate(self.models):
            model_name = getattr(model, "name", f"Model_{i}")

            if method == "loo":
                results[model_name] = self._loo_comparison(
                    model, data, random_seed=random_seed
                )
            elif method == "waic":
                results[model_name] = self._waic_comparison(
                    model, data, random_seed=random_seed
                )
            elif method == "dic":
                results[model_name] = self._dic_comparison(
                    model, data, random_seed=random_seed
                )
            else:
                raise ValueError(f"Unknown comparison method: {method}")

        # Rank models -- higher elpd_loo is better; lower WAIC / DIC is better
        model_items = [(n, v) for n, v in results.items() if n != "ranking"]
        if method == "loo":
            ranked = sorted(model_items, key=lambda x: x[1]["elpd_loo"], reverse=True)
        elif method == "waic":
            ranked = sorted(model_items, key=lambda x: x[1]["waic"])
        elif method == "dic":
            ranked = sorted(model_items, key=lambda x: x[1]["dic"])
        else:
            ranked = model_items

        results["ranking"] = [name for name, _ in ranked]

        self.comparison_results = results
        return results

    def get_best_model(self, criterion: str = "loo") -> Any | None:
        """
        Get the best model according to the specified criterion.

        Args:
            criterion: Criterion to use for selection ('loo', 'waic', 'dic').

        Returns:
            Best model according to the criterion, or None if not found.
        """
        if not self.comparison_results:
            raise ValueError(
                "No comparison results available. Run compare_models first."
            )

        ranking = self.comparison_results.get("ranking", [])
        if not ranking:
            return None

        best_model_name = ranking[0]

        for model in self.models:
            model_name = getattr(model, "name", None)
            if model_name == best_model_name:
                return model

        return None

    def compute_aic(self, log_likelihood: float, n_params: int) -> float:
        """Compute Akaike Information Criterion.

        AIC = -2 * log_likelihood + 2 * k

        Parameters
        ----------
        log_likelihood : float
            Maximum log-likelihood value.
        n_params : int
            Number of estimated parameters.

        Returns
        -------
        float
        """
        return -2.0 * log_likelihood + 2.0 * n_params

    def compute_bic(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Compute Bayesian Information Criterion.

        BIC = -2 * log_likelihood + k * log(n)

        Parameters
        ----------
        log_likelihood : float
            Maximum log-likelihood value.
        n_params : int
            Number of estimated parameters.
        n_obs : int
            Number of observations.

        Returns
        -------
        float
        """
        return float(-2.0 * log_likelihood + n_params * np.log(n_obs))

    def compute_bayes_factor(
        self, log_evidence_1: float, log_evidence_2: float
    ) -> float:
        """Compute the Bayes factor B_12 = exp(log_evidence_1 - log_evidence_2).

        Parameters
        ----------
        log_evidence_1 : float
            Log marginal likelihood of model 1.
        log_evidence_2 : float
            Log marginal likelihood of model 2.

        Returns
        -------
        float
            Bayes factor favouring model 1 over model 2.
        """
        log_bf = log_evidence_1 - log_evidence_2
        # Guard against overflow
        log_bf = np.clip(log_bf, -500.0, 500.0)
        return float(np.exp(log_bf))

    def plot_comparison(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot model comparison results as a bar chart.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        if not self.comparison_results:
            raise ValueError(
                "No comparison results available. Run compare_models first."
            )

        model_names = [k for k in self.comparison_results if k != "ranking"]
        if not model_names:
            raise ValueError("No model results to plot.")

        # Determine which metric to plot based on available keys
        sample_result = self.comparison_results[model_names[0]]
        if "elpd_loo" in sample_result:
            metric_key = "elpd_loo"
            ylabel = "ELPD LOO"
        elif "waic" in sample_result:
            metric_key = "waic"
            ylabel = "WAIC"
        elif "dic" in sample_result:
            metric_key = "dic"
            ylabel = "DIC"
        else:
            metric_key = list(sample_result.keys())[0]
            ylabel = metric_key

        values = [self.comparison_results[m][metric_key] for m in model_names]
        se_values = [self.comparison_results[m].get("se", 0.0) for m in model_names]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(model_names))
        ax.bar(x, values, yerr=se_values, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Model Comparison")
        fig.tight_layout()

        return fig, ax

    # ------------------------------------------------------------------
    # Internal comparison methods
    # ------------------------------------------------------------------

    @staticmethod
    def _pointwise_log_likelihoods(
        model: Any,
        data: Any,
        n_posterior_samples: int = 200,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """Compute a pointwise log-likelihood matrix from a model.

        If *data* already contains a 'log_likelihood_matrix' key it is
        returned directly; that is the intended path, and the matrix should hold
        *posterior* draws.

        Otherwise the method falls back to drawing parameter sets from the
        model's **prior**. That fallback keeps the comparison runnable without a
        fitted posterior, but prior-draw elpd is not leave-one-out
        cross-validation and its absolute values are not comparable with elpd
        computed from a posterior. Supply a posterior matrix whenever one
        exists.

        Args:
            model: The model to evaluate.
            data: Observations or a dict with 'log_likelihood_matrix'.
            n_posterior_samples: Number of prior parameter draws.
            random_seed: Seed or generator for the prior draws. ``None``
                (default) means a generator seeded from OS entropy, so the
                estimate is not replayable; pass an int to replay. See
                :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        ll_matrix : ndarray of shape (n_posterior_samples, n_obs)
        """
        rng = resolve_rng(random_seed)

        if isinstance(data, dict) and "log_likelihood_matrix" in data:
            return np.asarray(data["log_likelihood_matrix"])

        # Fallback: evaluate model log-likelihood for many parameter draws
        if isinstance(data, dict) and "observations" in data:
            observations = np.asarray(data["observations"])
        elif isinstance(data, np.ndarray):
            observations = data
        else:
            observations = np.asarray(data)

        n_obs = len(observations)
        ll_matrix = np.zeros((n_posterior_samples, n_obs))
        failures = 0
        first_failure: BaseException | None = None

        params = getattr(model, "parameters", {})
        for s in range(n_posterior_samples):
            theta = {}
            for pname, pinfo in params.items():
                hp = pinfo.get("hyperparams", {})
                prior = pinfo.get("prior", "normal")
                if prior == "normal":
                    theta[pname] = rng.normal(hp.get("mu", 0.0), hp.get("sigma", 1.0))
                elif prior == "log_normal":
                    theta[pname] = np.exp(
                        rng.normal(hp.get("mu", 0.0), hp.get("sigma", 1.0))
                    )
                elif prior == "uniform":
                    theta[pname] = rng.uniform(hp.get("low", 0.0), hp.get("high", 1.0))
                elif prior == "half_normal":
                    theta[pname] = abs(rng.normal(0.0, hp.get("sigma", 1.0)))
                else:
                    theta[pname] = rng.normal(0.0, 1.0)

            # Evaluate per-observation log-likelihood
            for j in range(n_obs):
                single_obs = observations[j : j + 1]
                try:
                    ll_matrix[s, j] = model.log_likelihood(theta, single_obs)
                except Exception as exc:
                    # A -inf entry silently drags elpd to -inf, so record why.
                    failures += 1
                    first_failure = first_failure or exc
                    ll_matrix[s, j] = -np.inf

        if failures:
            total = n_posterior_samples * n_obs
            if failures == total:
                raise RuntimeError(
                    f"log_likelihood failed for every one of {total} "
                    f"(draw, observation) pairs; the pointwise matrix carries no "
                    f"information. First error: {first_failure!r}"
                ) from first_failure
            logger.warning(
                "log_likelihood failed for %d of %d (draw, observation) pairs; "
                "those entries are -inf and dominate any elpd computed from "
                "this matrix. First error: %r",
                failures,
                total,
                first_failure,
            )

        return ll_matrix

    def _loo_comparison(
        self, model: Any, data: Any, random_seed: SeedLike = None
    ) -> dict[str, Any]:
        """Pareto-smoothed importance-sampling LOO (PSIS-LOO, Vehtari et al. 2017).

        Uses ``arviz.stats.stats.psislw`` on the pointwise log-likelihood
        matrix and the same elpd/p_loo/se formulas as arviz's ``loo``. Falls
        back to naive log-mean-exp LOO (labelled ``naive-loo``) only when
        arviz is unavailable.

        Args:
            model: The model to compare.
            data: Data for comparison.
            random_seed: Seed or generator forwarded to the prior-draw
                fallback; see :meth:`compare_models`.
        """
        ll_matrix = self._pointwise_log_likelihoods(
            model, data, random_seed=random_seed
        )
        n_samples, n_obs = ll_matrix.shape

        try:
            from arviz.stats.stats import psislw
            from scipy.special import logsumexp

            # arviz loo semantics: psislw expects (n_obs, n_samples) — the
            # LAST dimension is samples — and returns (smoothed_log_weights,
            # pareto_shape). Verified numerically against az.loo.
            ll_t = ll_matrix.T
            log_weights_t, pareto_k = psislw(-ll_t, reff=1.0)
            elpd_loo_i = logsumexp(log_weights_t + ll_t, axis=1)
            # Normalized log predictive density (log of the mean).
            lpd_i = logsumexp(ll_t, axis=1) - np.log(n_samples)
            elpd_loo = float(np.sum(elpd_loo_i))
            p_loo = float(np.sum(lpd_i) - elpd_loo)
            se = float(np.sqrt(n_obs * np.var(elpd_loo_i)))
            return {
                "elpd_loo": elpd_loo,
                "p_loo": p_loo,
                "se": se,
                "method": "psis-loo",
                "pareto_k_max": float(np.max(pareto_k)) if len(pareto_k) else 0.0,
            }
        except ImportError:
            # Naive LOO fallback (no PSIS smoothing).
            elpd_i = np.zeros(n_obs)
            for j in range(n_obs):
                max_ll = np.max(ll_matrix[:, j])
                elpd_i[j] = max_ll + np.log(np.mean(np.exp(ll_matrix[:, j] - max_ll)))

            elpd_loo = float(np.sum(elpd_i))
            se = float(np.sqrt(n_obs * np.var(elpd_i)))
            p_loo = float(np.sum(np.var(ll_matrix, axis=0)))

            return {
                "elpd_loo": elpd_loo,
                "p_loo": p_loo,
                "se": se,
                "method": "naive-loo",
                "pareto_k_max": 0.0,
            }

    def _waic_comparison(
        self, model: Any, data: Any, random_seed: SeedLike = None
    ) -> dict[str, float]:
        """Widely Applicable Information Criterion (Watanabe 2010).

        WAIC = -2 * (lppd - p_waic)

        Args:
            model: The model to compare.
            data: Data for comparison.
            random_seed: Seed or generator forwarded to the prior-draw
                fallback; see :meth:`compare_models`.
        """
        ll_matrix = self._pointwise_log_likelihoods(
            model, data, random_seed=random_seed
        )
        n_samples, n_obs = ll_matrix.shape

        # Log pointwise predictive density (lppd)
        lppd_i = np.zeros(n_obs)
        for j in range(n_obs):
            max_ll = np.max(ll_matrix[:, j])
            lppd_i[j] = max_ll + np.log(np.mean(np.exp(ll_matrix[:, j] - max_ll)))

        lppd = float(np.sum(lppd_i))

        # Effective number of parameters (using variance)
        p_waic = float(np.sum(np.var(ll_matrix, axis=0)))

        waic = -2.0 * (lppd - p_waic)

        # Standard error
        waic_i = -2.0 * (lppd_i - np.var(ll_matrix, axis=0))
        se = float(np.sqrt(n_obs * np.var(waic_i)))

        return {
            "waic": waic,
            "p_waic": p_waic,
            "lppd": lppd,
            "se": se,
        }

    def _dic_comparison(
        self, model: Any, data: Any, random_seed: SeedLike = None
    ) -> dict[str, float]:
        """Deviance Information Criterion.

        DIC = D_bar + p_D
        where D_bar is the posterior mean deviance and p_D is the effective
        number of parameters (D_bar - D_at_mean).

        Args:
            model: The model to compare.
            data: Data for comparison.
            random_seed: Seed or generator forwarded to the prior-draw
                fallback; see :meth:`compare_models`.
        """
        ll_matrix = self._pointwise_log_likelihoods(
            model, data, random_seed=random_seed
        )
        n_samples, n_obs = ll_matrix.shape

        # Total log-likelihood per posterior sample
        total_ll = np.sum(ll_matrix, axis=1)  # shape (n_samples,)

        # Deviance = -2 * log_likelihood
        deviance_samples = -2.0 * total_ll
        d_bar = float(np.mean(deviance_samples))

        # Deviance at posterior mean log-likelihood
        mean_ll_per_obs = np.mean(ll_matrix, axis=0)
        d_at_mean = float(-2.0 * np.sum(mean_ll_per_obs))

        p_d = d_bar - d_at_mean
        dic = d_bar + p_d

        return {
            "dic": dic,
            "p_d": p_d,
            "deviance": d_bar,
        }
