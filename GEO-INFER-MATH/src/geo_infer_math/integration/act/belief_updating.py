"""Belief updating for Active Inference.

Implements Bayesian belief updating with precision-weighted prediction
errors for state estimation in Active Inference agents.

References:
    Friston, K. et al. (2017). Active Inference, Curiosity and Insight.
    Neural Computation, 29(10), 2633-2683.
"""

import numpy as np
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)


class BeliefUpdating:
    """Belief updating for Active Inference.

    Provides Bayesian belief update methods including softmax-normalised
    posterior updates and precision-weighted prediction error integration.
    """

    def __init__(self, epsilon: float = 1e-16) -> None:
        """Initialize belief updater.

        Args:
            epsilon: Small constant for numerical stability.
        """
        self._epsilon = epsilon
        logger.debug("BeliefUpdating initialized (epsilon=%.2e)", epsilon)

    def update(
        self,
        current_beliefs: np.ndarray,
        new_observations: np.ndarray,
        likelihood: np.ndarray | None = None,
        precision: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update beliefs given new observations.

        Performs Bayesian belief updating:
            posterior ∝ likelihood(o|s) × prior(s)

        With optional precision weighting:
            posterior ∝ likelihood(o|s)^β × prior(s)

        Args:
            current_beliefs: Prior beliefs q(s), shape (n_states,).
            new_observations: New observations o, shape (n_obs,).
            likelihood: Likelihood matrix p(o|s), shape (n_obs, n_states).
                If None, observations are used as direct log-evidence.
            precision: Precision parameter β weighting sensory evidence.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary with 'posterior', 'prediction_error', 'kl_change'.
        """
        current_beliefs = np.asarray(current_beliefs, dtype=np.float64)
        new_observations = np.asarray(new_observations, dtype=np.float64)

        # Normalise prior
        prior = current_beliefs / (current_beliefs.sum() + self._epsilon)

        if likelihood is not None:
            likelihood = np.asarray(likelihood, dtype=np.float64)
            # Log-evidence for each state: ln p(o|s)
            obs_normalised = new_observations / (new_observations.sum() + self._epsilon)
            log_evidence = np.log(likelihood.T @ obs_normalised + self._epsilon)
        else:
            # Use observations directly as log-evidence
            log_evidence = new_observations[: len(prior)]

        # Precision-weighted log posterior
        log_posterior = np.log(prior + self._epsilon) + precision * log_evidence

        # Softmax normalisation (numerically stable)
        posterior = self._softmax(log_posterior)

        # Prediction error: difference between predicted and actual observations
        if likelihood is not None:
            predicted_obs = likelihood @ prior
            prediction_error = new_observations - predicted_obs[: len(new_observations)]
        else:
            prediction_error = log_evidence - np.log(prior + self._epsilon)

        # KL divergence between posterior and prior
        kl_change = float(
            np.sum(
                posterior
                * np.log((posterior + self._epsilon) / (prior + self._epsilon))
            )
        )

        logger.debug(
            "Belief update: KL change=%.4f, precision=%.2f",
            kl_change,
            precision,
        )

        return {
            "posterior": posterior,
            "prediction_error": prediction_error,
            "kl_change": kl_change,
        }

    def precision_weighted_update(
        self,
        beliefs: np.ndarray,
        prediction_errors: np.ndarray,
        sensory_precision: float = 1.0,
        prior_precision: float = 1.0,
    ) -> np.ndarray:
        """Precision-weighted prediction error belief update.

        Updated beliefs combine prior and sensory information weighted
        by their respective precisions:

            μ_new = (π_prior × μ_prior + π_sensory × μ_data) / (π_prior + π_sensory)

        Args:
            beliefs: Current beliefs (means), shape (n,).
            prediction_errors: Sensory prediction errors, shape (n,).
            sensory_precision: Precision of sensory evidence.
            prior_precision: Precision of prior beliefs.

        Returns:
            Updated beliefs, shape (n,).
        """
        beliefs = np.asarray(beliefs, dtype=np.float64)
        prediction_errors = np.asarray(prediction_errors, dtype=np.float64)

        total_precision = prior_precision + sensory_precision
        learning_rate = sensory_precision / total_precision

        updated = beliefs + learning_rate * prediction_errors

        logger.debug(
            "Precision-weighted update: learning_rate=%.4f, max_error=%.4f",
            learning_rate,
            float(np.max(np.abs(prediction_errors))),
        )
        return cast(np.ndarray, updated)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax normalisation."""
        x = logits - np.max(logits)
        exp_x = np.exp(x)
        return cast(np.ndarray, exp_x / (exp_x.sum() + self._epsilon))
