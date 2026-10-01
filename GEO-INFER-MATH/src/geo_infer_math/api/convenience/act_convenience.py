"""
Active Inference Convenience Methods

This module provides convenience methods for Active Inference operations,
including free energy calculations and variational inference helpers.
"""

import numpy as np
from typing import Any, cast
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


def free_energy_calculation(
    observations: np.ndarray,
    beliefs: np.ndarray,
    generative_model: Callable | None = None,
    precision: float = 1.0,
) -> float:
    """
    Calculate variational free energy for Active Inference.

    Free energy: F = -log p(o|m) + KL[q(s) || p(s|o,m)]

    Args:
        observations: Observed data o
        beliefs: Belief distribution q(s)
        generative_model: Optional generative model p(o|s)
        precision: Precision parameter

    Returns:
        Free energy value
    """
    observations = np.asarray(observations)
    beliefs = np.asarray(beliefs)

    # Normalize beliefs
    beliefs = beliefs / np.sum(beliefs) if np.sum(beliefs) > 0 else beliefs

    # Calculate negative log likelihood (accuracy term)
    if generative_model is not None:
        # Use provided generative model
        log_likelihood = np.log(
            np.maximum(generative_model(observations, beliefs), 1e-10)
        )
        accuracy = -np.sum(beliefs * log_likelihood)
    else:
        # Simplified: assume Gaussian likelihood
        predicted = np.sum(beliefs * np.arange(len(beliefs)))
        error = np.sum((observations - predicted) ** 2)
        accuracy = 0.5 * precision * error

    # Calculate KL divergence (complexity term)
    # Assume uniform prior for simplicity
    prior = np.ones_like(beliefs) / len(beliefs)
    prior = np.maximum(prior, 1e-10)
    beliefs_safe = np.maximum(beliefs, 1e-10)

    kl_divergence = np.sum(beliefs_safe * np.log(beliefs_safe / prior))

    # Free energy = accuracy + complexity
    free_energy = accuracy + kl_divergence

    return float(free_energy)


def variational_inference_helper(
    observations: np.ndarray,
    prior: np.ndarray,
    likelihood: Callable | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Helper for variational inference in Active Inference.

    Performs mean-field variational inference to approximate posterior.

    Args:
        observations: Observed data
        prior: Prior distribution
        likelihood: Optional likelihood function
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (posterior, metadata)
    """
    observations = np.asarray(observations)
    prior = np.asarray(prior)

    # Normalize prior
    prior = prior / np.sum(prior) if np.sum(prior) > 0 else prior

    # Initialize posterior
    posterior = prior.copy()

    # Iterative update
    for iteration in range(max_iterations):
        posterior_old = posterior.copy()

        # Update posterior (simplified mean-field update)
        if likelihood is not None:
            # Use provided likelihood
            likelihood_vals = likelihood(observations, posterior)
            likelihood_vals = np.maximum(likelihood_vals, 1e-10)
            posterior = prior * likelihood_vals
        else:
            # Mean-field Variational Inference update (no likelihood provided).
            # Standard VB update: q*(s_i) ∝ exp(E_q\i[log p(o, s)])
            # Approximated here using a softmax over squared prediction errors:
            #   log q(s_i) ≈ -0.5 * precision * (o_mean - i)^2 / n_states
            n_states = len(prior)
            obs_mean = float(np.mean(observations))  # scalar summary
            log_q = np.array(
                [
                    -0.5
                    * ((obs_mean - i) ** 2)  # negative squared distance to each state
                    for i in range(n_states)
                ]
            )
            # Subtract max for numerical stability before normalisation
            log_q -= np.max(log_q)
            posterior = prior * np.exp(log_q)

        # Normalize
        posterior = (
            posterior / np.sum(posterior) if np.sum(posterior) > 0 else posterior
        )

        # Check convergence
        if np.max(np.abs(posterior - posterior_old)) < tolerance:
            break

    metadata = {
        "iterations": iteration + 1,
        "converged": iteration < max_iterations - 1,
        "final_kl": np.sum(posterior * np.log(posterior / prior + 1e-10)),
    }

    return posterior, metadata


def belief_updating_helper(
    current_beliefs: np.ndarray, new_observations: np.ndarray, precision: float = 1.0
) -> np.ndarray:
    """
    Helper for belief updating in Active Inference.

    Updates beliefs based on new observations using Bayesian updating.

    Args:
        current_beliefs: Current belief distribution
        new_observations: New observations
        precision: Precision parameter

    Returns:
        Updated beliefs
    """
    current_beliefs = np.asarray(current_beliefs)
    new_observations = np.asarray(new_observations)

    # Normalize current beliefs
    current_beliefs = (
        current_beliefs / np.sum(current_beliefs)
        if np.sum(current_beliefs) > 0
        else current_beliefs
    )

    # Calculate likelihood (simplified Gaussian)
    predicted = np.sum(current_beliefs * np.arange(len(current_beliefs)))
    likelihood = np.exp(-0.5 * precision * (new_observations - predicted) ** 2)
    likelihood = (
        likelihood / np.sum(likelihood) if np.sum(likelihood) > 0 else likelihood
    )

    # Bayesian update
    updated_beliefs = current_beliefs * likelihood
    updated_beliefs = (
        updated_beliefs / np.sum(updated_beliefs)
        if np.sum(updated_beliefs) > 0
        else current_beliefs
    )

    return cast(np.ndarray, updated_beliefs)


class ActiveInferenceConvenience:
    """
    Convenience class for Active Inference operations.

    Provides high-level methods for common Active Inference tasks.
    """

    def __init__(self, precision: float = 1.0):
        """
        Initialize Active Inference convenience class.

        Args:
            precision: Default precision parameter
        """
        self.precision = precision

    def calculate_free_energy(
        self, observations: np.ndarray, beliefs: np.ndarray, **kwargs: Any
    ) -> float:
        """
        Calculate free energy.

        Args:
            observations: Observations
            beliefs: Beliefs
            **kwargs: Additional parameters

        Returns:
            Free energy value
        """
        return free_energy_calculation(
            observations, beliefs, precision=self.precision, **kwargs
        )

    def variational_inference(
        self, observations: np.ndarray, prior: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Perform variational inference.

        Args:
            observations: Observations
            prior: Prior distribution
            **kwargs: Additional parameters

        Returns:
            Tuple of (posterior, metadata)
        """
        return variational_inference_helper(observations, prior, **kwargs)

    def update_beliefs(
        self, current_beliefs: np.ndarray, new_observations: np.ndarray
    ) -> np.ndarray:
        """
        Update beliefs with new observations.

        Args:
            current_beliefs: Current beliefs
            new_observations: New observations

        Returns:
            Updated beliefs
        """
        return belief_updating_helper(
            current_beliefs, new_observations, precision=self.precision
        )
