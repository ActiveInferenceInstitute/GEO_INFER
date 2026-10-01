"""Bayesian model selection and comparison.

Implements information criteria (BIC, AIC, WAIC) and evidence-based
model comparison for selecting among competing spatial models.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ModelSelection:
    """Bayesian model selection and comparison.

    Computes information criteria and Bayes factors for comparing
    statistical models of spatial processes.
    """

    def __init__(self) -> None:
        """Initialize model selection tools."""
        logger.debug("ModelSelection initialized")

    def compare_models(
        self,
        models: list[dict[str, Any]],
        method: str = "bic",
    ) -> dict[str, Any]:
        """Compare multiple models using an information criterion.

        Args:
            models: List of model dictionaries. Each must contain:
                - 'name': Model identifier string.
                - 'log_likelihood': Maximum log-likelihood L̂.
                - 'n_params': Number of free parameters k.
                - 'n_obs': Number of observations n.
                Optional for WAIC:
                - 'pointwise_log_likelihoods': (n_obs, n_samples) array.
            method: Criterion — 'bic', 'aic', 'aicc', 'waic'.

        Returns:
            Dictionary with 'best_model', 'rankings', 'scores', 'deltas'.
        """
        if not models:
            raise ValueError("At least one model required.")

        scores = {}
        for model in models:
            name = model["name"]
            if method == "bic":
                scores[name] = self.bic(
                    model["log_likelihood"], model["n_params"], model["n_obs"]
                )
            elif method == "aic":
                scores[name] = self.aic(model["log_likelihood"], model["n_params"])
            elif method == "aicc":
                scores[name] = self.aicc(
                    model["log_likelihood"], model["n_params"], model["n_obs"]
                )
            elif method == "waic":
                scores[name] = self.waic(model["pointwise_log_likelihoods"])
            else:
                raise ValueError(f"Unknown method: {method}")

        # Rank: lower is better for all criteria
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        best = ranked[0][0]
        best_score = ranked[0][1]

        deltas = {name: score - best_score for name, score in scores.items()}

        logger.debug(
            "Model comparison (%s): best=%s, score=%.4f", method, best, best_score
        )

        return {
            "best_model": best,
            "rankings": [name for name, _ in ranked],
            "scores": scores,
            "deltas": deltas,
        }

    def bic(
        self,
        log_likelihood: float,
        n_params: int,
        n_obs: int,
    ) -> float:
        """Bayesian Information Criterion.

        BIC = k · ln(n) - 2 · ln(L̂)

        Penalises model complexity more heavily than AIC for
        large sample sizes.

        Args:
            log_likelihood: Maximum log-likelihood L̂.
            n_params: Number of free parameters k.
            n_obs: Number of observations n.

        Returns:
            BIC score (lower is better).
        """
        return float(n_params * np.log(max(1, n_obs)) - 2.0 * log_likelihood)

    def aic(
        self,
        log_likelihood: float,
        n_params: int,
    ) -> float:
        """Akaike Information Criterion.

        AIC = 2k - 2 · ln(L̂)

        Args:
            log_likelihood: Maximum log-likelihood L̂.
            n_params: Number of free parameters k.

        Returns:
            AIC score (lower is better).
        """
        return 2.0 * n_params - 2.0 * log_likelihood

    def aicc(
        self,
        log_likelihood: float,
        n_params: int,
        n_obs: int,
    ) -> float:
        """Corrected AIC for small sample sizes.

        AICc = AIC + 2k(k+1) / (n-k-1)

        Args:
            log_likelihood: Maximum log-likelihood L̂.
            n_params: Number of free parameters k.
            n_obs: Number of observations n.

        Returns:
            AICc score (lower is better).
        """
        aic_val = self.aic(log_likelihood, n_params)
        denominator = max(1, n_obs - n_params - 1)
        correction = 2.0 * n_params * (n_params + 1) / denominator
        return aic_val + correction

    def waic(
        self,
        pointwise_log_likelihoods: np.ndarray,
    ) -> float:
        """Widely Applicable Information Criterion.

        WAIC = -2 × (lppd - p_WAIC)

        where lppd = Σ_i ln(1/S Σ_s p(y_i|θ_s)) and
        p_WAIC = Σ_i Var_s(ln p(y_i|θ_s))

        Args:
            pointwise_log_likelihoods: (n_obs, n_samples) array where
                entry [i,s] is ln p(y_i | θ_s).

        Returns:
            WAIC score (lower is better).
        """
        ll = np.asarray(pointwise_log_likelihoods, dtype=np.float64)
        n_obs, n_samples = ll.shape

        # Log pointwise predictive density
        # lppd_i = log(mean(exp(ll_is)))
        max_ll = np.max(ll, axis=1, keepdims=True)
        lppd = np.log(np.mean(np.exp(ll - max_ll), axis=1)) + max_ll[:, 0]
        total_lppd = float(np.sum(lppd))

        # Effective number of parameters
        p_waic = float(np.sum(np.var(ll, axis=1)))

        waic_val = -2.0 * (total_lppd - p_waic)

        logger.debug("WAIC=%.4f (lppd=%.4f, p_waic=%.4f)", waic_val, total_lppd, p_waic)
        return waic_val

    def bayes_factor(
        self,
        log_evidence_1: float,
        log_evidence_2: float,
    ) -> dict[str, Any]:
        """Compute Bayes factor B₁₂ = p(D|M₁) / p(D|M₂).

        Args:
            log_evidence_1: Log marginal likelihood of model 1.
            log_evidence_2: Log marginal likelihood of model 2.

        Returns:
            Dictionary with 'log_bf', 'bf', 'interpretation'.
        """
        log_bf = log_evidence_1 - log_evidence_2
        bf = float(np.exp(min(log_bf, 700)))  # Prevent overflow

        if log_bf > np.log(100):
            interpretation = "decisive_for_M1"
        elif log_bf > np.log(10):
            interpretation = "strong_for_M1"
        elif log_bf > np.log(3):
            interpretation = "moderate_for_M1"
        elif log_bf > -np.log(3):
            interpretation = "inconclusive"
        elif log_bf > -np.log(10):
            interpretation = "moderate_for_M2"
        elif log_bf > -np.log(100):
            interpretation = "strong_for_M2"
        else:
            interpretation = "decisive_for_M2"

        logger.debug("Bayes factor: log_bf=%.4f (%s)", log_bf, interpretation)
        return {
            "log_bf": float(log_bf),
            "bf": bf,
            "interpretation": interpretation,
        }
