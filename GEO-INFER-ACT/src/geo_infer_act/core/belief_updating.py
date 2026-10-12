"""
Belief updating for Active Inference models.

References:
    - Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference.
      Cambridge, MA: MIT Press.
    - Formal analogue: fep_lean topic fep-017 (posterior joint reconstruction)

The fep_lean topic ids are correspondence-of-constructs references into a
separate Lean formalization catalogue, canonically mapped in
`fep_lean/specs/geo-infer-notation-bridge/data/notation-map.yaml` and
documented in `GEO-INFER-ACT/docs/fep_lean_notation_bridge.md`. They state
no verification relationship between this numerical implementation and the
Lean proofs.
"""

import numpy as np

from geo_infer_act.core.free_energy import validate_spd_precision
from geo_infer_act.utils.math import (
    categorical_posterior,
    compute_surprise as _compute_surprise,
)


class BayesianBeliefUpdate:
    """
    Bayesian belief updating for active inference models.

    This class implements various methods for updating beliefs
    based on new observations and prior knowledge.
    """

    def __init__(self, prior_precision: float = 1.0):
        """
        Initialize the belief updater.

        Args:
            prior_precision: Precision of prior beliefs
        """
        if not np.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior_precision must be finite and strictly positive")
        self.prior_precision = float(prior_precision)

    def update_categorical(
        self,
        prior_beliefs: np.ndarray,
        observation: np.ndarray,
        likelihood_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Update categorical beliefs using Bayes' rule.

        Args:
            prior_beliefs: Prior belief distribution
            observation: Observed data
            likelihood_matrix: Likelihood of observations given states

        Returns:
            Updated posterior beliefs
        """
        return categorical_posterior(
            prior_beliefs,
            observation,
            likelihood_matrix,
        )

    def update_gaussian(
        self,
        prior_mean: np.ndarray,
        prior_precision: np.ndarray,
        observation: np.ndarray,
        observation_matrix: np.ndarray,
        observation_precision: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Update Gaussian beliefs using Kalman filter equations.

        Args:
            prior_mean: Prior mean
            prior_precision: Prior precision matrix
            observation: Observed data
            observation_matrix: Observation matrix (H)
            observation_precision: Observation precision matrix

        Returns:
            Updated mean and precision
        """
        prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
        observation = np.asarray(observation, dtype=float).reshape(-1)
        prior_precision = np.asarray(prior_precision, dtype=float)
        observation_matrix = np.asarray(observation_matrix, dtype=float)
        observation_precision = np.asarray(observation_precision, dtype=float)
        state_dim = prior_mean.size
        if state_dim == 0 or observation.size == 0:
            raise ValueError("Gaussian belief vectors must not be empty")
        if observation_matrix.shape != (observation.size, state_dim):
            raise ValueError(
                f"observation_matrix must have shape ({observation.size}, {state_dim})"
            )
        if not np.all(np.isfinite(observation_matrix)):
            raise ValueError("observation_matrix must be finite")
        if prior_precision.shape != (state_dim, state_dim):
            raise ValueError(
                f"prior_precision must have shape ({state_dim}, {state_dim})"
            )
        if observation_precision.shape != (observation.size, observation.size):
            raise ValueError(
                "observation_precision must be square with one row per observation"
            )
        for name, matrix in (
            ("prior_precision", prior_precision),
            ("observation_precision", observation_precision),
        ):
            validate_spd_precision(name, matrix)
        if not np.all(np.isfinite(prior_mean)) or not np.all(np.isfinite(observation)):
            raise ValueError("Gaussian belief vectors must be finite")

        # Convert precision to covariance without explicitly inverting during
        # the Kalman update.  Explicit inverses amplify conditioning errors.
        identity_state = np.eye(state_dim)
        identity_obs = np.eye(observation.size)
        prior_cov = np.linalg.solve(prior_precision, identity_state)
        obs_cov = np.linalg.solve(observation_precision, identity_obs)

        # Kalman filter update
        H = observation_matrix
        innovation_cov = H @ prior_cov @ H.T + obs_cov
        cross_cov = prior_cov @ H.T
        K = np.linalg.solve(innovation_cov.T, cross_cov.T).T

        # Updated mean
        posterior_mean = prior_mean + K @ (observation - H @ prior_mean)

        # Updated covariance
        # Joseph form keeps the posterior covariance symmetric and positive
        # semidefinite in finite precision arithmetic.
        residual_transform = np.eye(state_dim) - K @ H
        posterior_cov = (
            residual_transform @ prior_cov @ residual_transform.T + K @ obs_cov @ K.T
        )
        posterior_cov = (posterior_cov + posterior_cov.T) / 2.0

        # Convert back to precision
        posterior_precision = np.linalg.solve(posterior_cov, identity_state)
        posterior_precision = (posterior_precision + posterior_precision.T) / 2.0

        return {"mean": posterior_mean, "precision": posterior_precision}

    def compute_prediction_error(
        self, prediction: np.ndarray, observation: np.ndarray, precision: float = 1.0
    ) -> float:
        """
        Compute precision-weighted prediction error.

        Args:
            prediction: Predicted observation
            observation: Actual observation
            precision: Precision weight

        Returns:
            Prediction error
        """
        prediction = np.asarray(prediction, dtype=float)
        observation = np.asarray(observation, dtype=float)
        if prediction.shape != observation.shape:
            raise ValueError("prediction and observation must have the same shape")
        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(observation)):
            raise ValueError("prediction and observation must be finite")
        if not np.isfinite(precision) or precision < 0:
            raise ValueError("precision must be finite and non-negative")
        error = observation - prediction
        return float(precision * np.sum(error**2))

    def compute_surprise(
        self, observation: np.ndarray, predicted_distribution: np.ndarray
    ) -> float:
        """
        Compute surprise (negative log probability) of observation.

        Args:
            observation: Observed data
            predicted_distribution: Predicted probability distribution

        Returns:
            Surprise value
        """
        return _compute_surprise(observation, predicted_distribution)

    def update_beliefs(
        self, prior_beliefs: np.ndarray, observation: np.ndarray, likelihood: np.ndarray
    ) -> np.ndarray:
        """General belief update dispatching to categorical or gaussian.

        Dispatches based on likelihood shape:
        - 2D likelihood matrix → categorical Bayes update
        - 1D likelihood vector → Gaussian Kalman update (using identity matrices for precision)

        Args:
            prior_beliefs: Prior belief distribution (1D array)
            observation: Observed data (1D array)
            likelihood: Likelihood matrix (2D for categorical) or vector (1D for gaussian)

        Returns:
            Updated posterior beliefs
        """
        if prior_beliefs.ndim == 1 and observation.ndim == 1 and likelihood.ndim == 2:
            return self.update_categorical(prior_beliefs, observation, likelihood)
        elif prior_beliefs.ndim == 1 and observation.ndim == 1 and likelihood.ndim == 1:
            # Gaussian update: construct diagonal observation matrix from likelihood vector
            obs_matrix = np.diag(likelihood)
            result = self.update_gaussian(
                prior_beliefs,
                np.eye(len(prior_beliefs)),
                observation,
                obs_matrix,
                np.eye(len(observation)),
            )
            return result["mean"]
        else:
            raise ValueError(
                f"Unsupported input shapes for update_beliefs: "
                f"prior={prior_beliefs.shape}, obs={observation.shape}, likelihood={likelihood.shape}"
            )
