"""
Base models for active inference framework.
"""

from typing import Any
import numpy as np

from geo_infer_act.utils.math import categorical_posterior

# from abc import ABC, abstractmethod


class BaseActiveInferenceModel:
    """
    Base class for active inference models.

    This abstract base class defines the interface for
    all active inference models in the GEO-INFER-ACT module.

    Note: this is distinct from ``geo_infer_act.core.active_inference``
    ``ActiveInferenceModel``, the orchestration agent that this package
    exports at top level.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the model.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else {}

    def step(self, actions: Any | None = None) -> Any:
        """
        Advance the model by one step.

        Domain subclasses (ecological, urban, resource, multi-agent, climate)
        override this with their active-inference cycle; the base
        implementation returns the configuration so a bare model instance
        remains inert but harmless.

        Args:
            actions: Optional actions to apply

        Returns:
            Updated state or relevant information
        """
        return self.config.copy()

    def reset(self) -> Any:
        """
        Reset the model to initial state.

        Returns:
            Initial state
        """
        self.config = self.config if self.config is not None else {}
        return self.config.copy()

    def __str__(self) -> str:
        """Return string representation of model."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Return string representation of model."""
        return self.__str__()


class CategoricalModel(BaseActiveInferenceModel):
    """
    Categorical active inference model.

    This model uses categorical distributions to represent
    beliefs and observations in discrete state spaces.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        state_dim: int = 1,
        obs_dim: int = 1,
    ):
        """
        Initialize categorical model.

        Args:
            config: Configuration dictionary
            state_dim: Dimension of state space
            obs_dim: Dimension of observation space
        """
        super().__init__(config)
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Initialize beliefs and preferences as uniform distributions
        self.beliefs = np.ones(state_dim) / state_dim
        self.preferences = np.ones(obs_dim) / obs_dim

        # Initialize transition and likelihood matrices
        # Rows encode P(next_state | current_state).  The identity default
        # preserves state continuity until a caller supplies dynamics.
        self.transition_matrix = np.eye(state_dim)
        self.likelihood_matrix = np.ones((obs_dim, state_dim)) / obs_dim

        # Spatial attributes (used by multi-agent H3 coordination)
        self.location = 0
        self.cell_id: str | None = None
        self.spatial_index: int | None = None

    def set_preferences(self, preferences: np.ndarray) -> None:
        """
        Set preference distribution.

        Args:
            preferences: Preference distribution
        """
        preferences = np.asarray(preferences, dtype=float)
        if preferences.shape != (self.obs_dim,):
            raise ValueError(f"Preferences must have shape ({self.obs_dim},)")
        if not np.all(np.isfinite(preferences)) or np.any(preferences < 0):
            raise ValueError("Preferences must be finite and non-negative")
        total = float(np.sum(preferences))
        if total <= 0:
            raise ValueError("Preferences must have positive total mass")

        # Normalize
        self.preferences = preferences / total

    def set_transition_matrix(self, transition_matrix: np.ndarray) -> None:
        """
        Set state transition matrix.

        Args:
            transition_matrix: Transition probability matrix
        """
        transition_matrix = np.asarray(transition_matrix, dtype=float)
        expected_shape = (self.state_dim, self.state_dim)
        if transition_matrix.shape != expected_shape:
            raise ValueError(f"Transition matrix must have shape {expected_shape}")
        if not np.all(np.isfinite(transition_matrix)) or np.any(transition_matrix < 0):
            raise ValueError("Transition matrix must be finite and non-negative")
        row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Each transition matrix row needs positive mass")

        # Normalize rows
        self.transition_matrix = transition_matrix / row_sums

    def set_likelihood_matrix(self, likelihood_matrix: np.ndarray) -> None:
        """
        Set observation likelihood matrix.

        Args:
            likelihood_matrix: Likelihood matrix
        """
        likelihood_matrix = np.asarray(likelihood_matrix, dtype=float)
        expected_shape = (self.obs_dim, self.state_dim)
        if likelihood_matrix.shape != expected_shape:
            raise ValueError(f"Likelihood matrix must have shape {expected_shape}")
        if not np.all(np.isfinite(likelihood_matrix)) or np.any(likelihood_matrix < 0):
            raise ValueError("Likelihood matrix must be finite and non-negative")

        # Normalize columns
        col_sums = np.sum(likelihood_matrix, axis=0, keepdims=True)
        if np.any(col_sums <= 0):
            raise ValueError("Each likelihood matrix column needs positive mass")
        self.likelihood_matrix = likelihood_matrix / col_sums

    def _predict_beliefs(self) -> np.ndarray:
        """Apply the row-stochastic transition model to current beliefs."""
        predicted = self.transition_matrix.T @ self.beliefs
        total = float(np.sum(predicted))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Transition model produced an invalid prior")
        return predicted / total

    def update_beliefs(self, observation: np.ndarray) -> np.ndarray:
        """Predict with ``B`` and update with the categorical observation."""
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.obs_dim,):
            raise ValueError(f"Observation must have shape ({self.obs_dim},)")

        # The shared helper evaluates the count likelihood in log space and
        # guarantees a normalized posterior for arbitrarily large counts.
        posterior = categorical_posterior(
            self._predict_beliefs(), observation, self.likelihood_matrix
        )

        # Update beliefs
        self.beliefs = posterior

        return self.beliefs

    def step(self, action: int | None = None) -> np.ndarray:
        """
        Advance the model by one step.

        Args:
            action: Optional action index

        Returns:
            New belief distribution
        """
        # Apply dynamics (prediction step)
        predicted_belief = self._predict_beliefs()

        # Update beliefs
        self.beliefs = predicted_belief

        return self.beliefs

    def reset(self) -> np.ndarray:
        """
        Reset beliefs to uniform distribution.

        Returns:
            Initial belief distribution
        """
        self.beliefs = np.ones(self.state_dim) / self.state_dim
        return self.beliefs

    def compute_free_energy(self) -> float:
        """Compute categorical KL free energy relative to uniform preferences."""
        if not hasattr(self, "beliefs") or self.beliefs is None:
            return float("inf")
        beliefs = np.asarray(self.beliefs, dtype=float).reshape(-1)
        beliefs = np.clip(beliefs, 1e-12, None)
        beliefs = beliefs / np.sum(beliefs)
        preferences = np.ones_like(beliefs) / len(beliefs)
        return float(np.sum(beliefs * (np.log(beliefs) - np.log(preferences))))


class GaussianModel(BaseActiveInferenceModel):
    """
    Gaussian active inference model.

    This model uses multivariate Gaussian distributions to represent
    beliefs and observations in continuous state spaces.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        state_dim: int = 1,
        obs_dim: int = 1,
    ):
        """
        Initialize Gaussian model.

        Args:
            config: Configuration dictionary
            state_dim: Dimension of state space
            obs_dim: Dimension of observation space
        """
        super().__init__(config)
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Initialize beliefs
        self.belief_mean = np.zeros(state_dim)
        self.belief_cov = np.eye(state_dim)

        # Initialize preferences
        self.preference_mean = np.zeros(obs_dim)
        self.preference_cov = np.eye(obs_dim)

        # Initialize transition and observation models
        self.A = np.eye(state_dim)  # State transition matrix
        self.B = np.zeros((state_dim, 1))  # Control input matrix
        self.C = np.eye(min(obs_dim, state_dim))  # Observation matrix

        # Initialize noise covariances
        self.Q = np.eye(state_dim) * 0.01  # Process noise
        self.R = np.eye(obs_dim) * 0.01  # Observation noise

    def set_preferences(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Set preference distribution.

        Args:
            mean: Preference mean
            cov: Preference covariance
        """
        if mean.shape != (self.obs_dim,):
            raise ValueError(f"Preference mean must have shape ({self.obs_dim},)")

        if cov.shape != (self.obs_dim, self.obs_dim):
            raise ValueError(
                f"Preference covariance must have shape ({self.obs_dim}, {self.obs_dim})"
            )

        self.preference_mean = mean
        self.preference_cov = cov

    def set_transition_model(
        self,
        A: np.ndarray,
        B: np.ndarray | None = None,
        Q: np.ndarray | None = None,
    ) -> None:
        """
        Set transition model parameters.

        Args:
            A: State transition matrix
            B: Control input matrix
            Q: Process noise covariance
        """
        expected_A_shape = (self.state_dim, self.state_dim)
        if A.shape != expected_A_shape:
            raise ValueError(f"A matrix must have shape {expected_A_shape}")

        self.A = A

        if B is not None:
            if B.shape[0] != self.state_dim:
                raise ValueError(
                    f"B matrix must have shape ({self.state_dim}, control_dim)"
                )
            self.B = B

        if Q is not None:
            expected_Q_shape = (self.state_dim, self.state_dim)
            if Q.shape != expected_Q_shape:
                raise ValueError(f"Q matrix must have shape {expected_Q_shape}")
            self.Q = Q

    def set_observation_model(self, C: np.ndarray, R: np.ndarray | None = None) -> None:
        """
        Set observation model parameters.

        Args:
            C: Observation matrix
            R: Observation noise covariance
        """
        expected_C_shape = (self.obs_dim, self.state_dim)
        if C.shape != expected_C_shape:
            raise ValueError(f"C matrix must have shape {expected_C_shape}")

        self.C = C

        if R is not None:
            expected_R_shape = (self.obs_dim, self.obs_dim)
            if R.shape != expected_R_shape:
                raise ValueError(f"R matrix must have shape {expected_R_shape}")
            self.R = R

    def update_beliefs(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        """
        Update beliefs given observation (Kalman filter).

        Args:
            observation: Observation vector

        Returns:
            Updated belief distribution (mean and covariance)
        """
        if observation.shape != (self.obs_dim,):
            raise ValueError(f"Observation must have shape ({self.obs_dim},)")

        # Prediction step
        predicted_mean: np.ndarray = self.A @ self.belief_mean
        predicted_cov: np.ndarray = self.A @ self.belief_cov @ self.A.T + self.Q

        # Update step (Kalman filter)
        K: np.ndarray = (
            predicted_cov
            @ self.C.T
            @ np.linalg.inv(self.C @ predicted_cov @ self.C.T + self.R)
        )

        updated_mean: np.ndarray = predicted_mean + K @ (
            observation - self.C @ predicted_mean
        )
        updated_cov: np.ndarray = (np.eye(self.state_dim) - K @ self.C) @ predicted_cov

        # Ensure covariance matrix stays symmetric and positive definite
        updated_cov = (updated_cov + updated_cov.T) / 2
        min_eig = float(np.min(np.real(np.linalg.eigvals(updated_cov))))
        if min_eig < 1e-6:
            updated_cov += (1e-6 - min_eig) * np.eye(self.state_dim)

        # Update beliefs
        self.belief_mean = updated_mean
        self.belief_cov = updated_cov

        return {"mean": self.belief_mean, "cov": self.belief_cov}

    def step(self, control: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Advance the model by one step.

        Args:
            control: Optional control input

        Returns:
            New belief distribution
        """
        # Apply dynamics (prediction step)
        if control is not None:
            predicted_mean = self.A @ self.belief_mean + self.B @ control
        else:
            predicted_mean = self.A @ self.belief_mean

        predicted_cov = self.A @ self.belief_cov @ self.A.T + self.Q

        # Update beliefs
        self.belief_mean = predicted_mean
        self.belief_cov = predicted_cov

        return {"mean": self.belief_mean, "cov": self.belief_cov}

    def reset(self) -> dict[str, np.ndarray]:
        """
        Reset beliefs to initial state.

        Returns:
            Initial belief distribution
        """
        self.belief_mean = np.zeros(self.state_dim)
        self.belief_cov = np.eye(self.state_dim)

        return {"mean": self.belief_mean, "cov": self.belief_cov}
