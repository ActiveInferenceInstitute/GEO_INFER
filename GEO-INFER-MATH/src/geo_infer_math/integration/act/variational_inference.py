"""Variational inference helpers for Active Inference.

Implements coordinate ascent variational inference (CAVI) with
ELBO computation and convergence tracking.

References:
    Blei, D.M., Kucukelbir, A., & McAuliffe, J.D. (2017).
    Variational Inference: A Review for Statisticians. JASA, 112(518).
"""

import numpy as np
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)


class VariationalInferenceHelpers:
    """Variational inference helpers for Active Inference.

    Provides coordinate ascent VI with ELBO tracking, mean-field
    approximation, and convergence diagnostics.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        epsilon: float = 1e-16,
    ) -> None:
        """Initialize VI helpers.

        Args:
            max_iterations: Maximum number of CAVI iterations.
            convergence_threshold: ELBO change threshold for convergence.
            epsilon: Numerical stability constant.
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._epsilon = epsilon
        logger.debug(
            "VariationalInferenceHelpers initialized (max_iter=%d, tol=%.2e)",
            max_iterations,
            convergence_threshold,
        )

    def perform_vi(
        self,
        observations: np.ndarray,
        prior: np.ndarray,
        likelihood: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform mean-field variational inference.

        Minimises the variational free energy F = -ELBO by iteratively
        updating the approximate posterior q(s).

        Args:
            observations: Observation data o, shape (n_obs,).
            prior: Prior distribution p(s), shape (n_states,).
            likelihood: Likelihood matrix p(o|s), shape (n_obs, n_states).
                If None, a Gaussian likelihood is assumed.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary with 'posterior', 'elbo_history', 'converged', 'n_iterations'.
        """
        observations = np.asarray(observations, dtype=np.float64)
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / (prior.sum() + self._epsilon)

        n_states = len(prior)

        # Initialise approximate posterior as the prior
        q = prior.copy()

        elbo_history: list[float] = []

        for iteration in range(self.max_iterations):
            # E-step: compute expected log-likelihood
            if likelihood is not None:
                likelihood_arr = np.asarray(likelihood, dtype=np.float64)
                obs_normalised = observations / (observations.sum() + self._epsilon)
                log_lik = np.log(likelihood_arr.T @ obs_normalised + self._epsilon)
            else:
                # Gaussian likelihood approximation
                log_lik = (
                    -0.5
                    * np.sum((observations[:n_states] - np.arange(n_states)) ** 2)
                    * np.ones(n_states)
                )

            # M-step: update variational posterior
            log_q = np.log(prior + self._epsilon) + log_lik
            q_new = self._softmax(log_q)

            # Compute ELBO = E_q[ln p(o,s)] - E_q[ln q(s)]
            elbo = self._compute_elbo(q_new, prior, log_lik)
            elbo_history.append(elbo)

            # Check convergence
            q = q_new

            if len(elbo_history) > 1:
                elbo_change = abs(elbo_history[-1] - elbo_history[-2])
                if elbo_change < self.convergence_threshold:
                    logger.debug(
                        "VI converged at iteration %d (ELBO change=%.2e)",
                        iteration,
                        elbo_change,
                    )
                    return {
                        "posterior": q,
                        "elbo_history": elbo_history,
                        "converged": True,
                        "n_iterations": iteration + 1,
                    }

        logger.debug(
            "VI did not converge after %d iterations (final ELBO=%.4f)",
            self.max_iterations,
            elbo_history[-1],
        )
        return {
            "posterior": q,
            "elbo_history": elbo_history,
            "converged": False,
            "n_iterations": self.max_iterations,
        }

    def _compute_elbo(
        self,
        q: np.ndarray,
        prior: np.ndarray,
        log_likelihood: np.ndarray,
    ) -> float:
        """Compute Evidence Lower BOund.

        ELBO = E_q[ln p(o|s)] + E_q[ln p(s)] - E_q[ln q(s)]
             = E_q[ln p(o|s)] - D_KL[q(s) || p(s)]
        """
        # Expected log-likelihood
        expected_ll = float(np.sum(q * log_likelihood))

        # Negative KL divergence
        kl = float(np.sum(q * np.log((q + self._epsilon) / (prior + self._epsilon))))

        return expected_ll - kl

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = logits - np.max(logits)
        exp_x = np.exp(x)
        return cast(np.ndarray, exp_x / (exp_x.sum() + self._epsilon))
