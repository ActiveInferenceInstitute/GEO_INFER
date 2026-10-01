"""
Dynamic Causal Modeling for Active Inference.
"""

from typing import cast
import numpy as np


class DynamicCausalModel:
    """
    Dynamic Causal Model for continuous-time active inference.

    This class implements dynamic causal modeling using stochastic
    differential equations for continuous-time inference.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        dt: float = 0.01,
        random_seed: int | None = None,
    ):
        """
        Initialize the dynamic causal model.

        Args:
            state_dim: Dimension of hidden states
            input_dim: Dimension of inputs
            output_dim: Dimension of outputs
            dt: Time step for integration
            random_seed: Optional seed for reproducible stochastic dynamics
        """
        for name, value in (
            ("state_dim", state_dim),
            ("input_dim", input_dim),
            ("output_dim", output_dim),
        ):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and strictly positive")
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dt = float(dt)
        self.rng = np.random.default_rng(random_seed)

        # Model parameters
        self.A = np.eye(state_dim) * -0.1  # State dynamics matrix
        self.B = self.rng.normal(0.0, 0.1, (state_dim, input_dim))
        self.C = self.rng.normal(0.0, 0.1, (output_dim, state_dim))

        # Noise parameters
        self.Q = np.eye(state_dim) * 0.01  # State noise
        self.R = np.eye(output_dim) * 0.01  # Observation noise

        # Current state
        self.state = np.zeros(state_dim)

    def state_equation(
        self, state: np.ndarray, t: float, inputs: np.ndarray
    ) -> np.ndarray:
        """
        State evolution equation: dx/dt = f(x, u, t).

        Args:
            state: Current state vector
            t: Current time
            inputs: Input vector

        Returns:
            State derivative
        """
        state = np.asarray(state, dtype=float)
        inputs = np.asarray(inputs, dtype=float)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},)")
        if inputs.shape != (self.input_dim,):
            raise ValueError(f"inputs must have shape ({self.input_dim},)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(inputs)):
            raise ValueError("state and inputs must be finite")

        # Linear dynamics: dx/dt = A*x + B*u
        dxdt = self.A @ state + self.B @ inputs

        return cast(np.ndarray, dxdt)

    def observation_equation(self, state: np.ndarray) -> np.ndarray:
        """
        Observation equation: y = g(x) + noise.

        Args:
            state: Current state vector

        Returns:
            Observation vector
        """
        state = np.asarray(state, dtype=float)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},)")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must be finite")
        # Linear observation: y = C*x
        observation = self.C @ state

        # Add noise
        noise = self.rng.multivariate_normal(np.zeros(self.output_dim), self.R)

        return cast(np.ndarray, observation + noise)

    def integrate_dynamics(
        self, initial_state: np.ndarray, inputs: np.ndarray, time_points: np.ndarray
    ) -> np.ndarray:
        """
        Integrate the system dynamics over time.

        Args:
            initial_state: Initial state vector
            inputs: Input sequence (n_timesteps x input_dim)
            time_points: Time points for integration

        Returns:
            State trajectory (n_timesteps x state_dim)
        """
        initial_state = np.asarray(initial_state, dtype=float)
        inputs = np.asarray(inputs, dtype=float)
        time_points = np.asarray(time_points, dtype=float)
        if initial_state.shape != (self.state_dim,):
            raise ValueError(f"initial_state must have shape ({self.state_dim},)")
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape (n, {self.input_dim})")
        if time_points.ndim != 1 or time_points.size == 0:
            raise ValueError("time_points must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(initial_state)) or not np.all(np.isfinite(inputs)):
            raise ValueError("initial_state and inputs must be finite")
        if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
            raise ValueError("time_points must be finite and strictly increasing")

        n_timesteps = len(time_points)
        state_trajectory = np.zeros((n_timesteps, self.state_dim))

        current_state = initial_state.copy()
        state_trajectory[0] = current_state

        for i in range(1, n_timesteps):
            dt = time_points[i] - time_points[i - 1]
            current_input = (
                inputs[i - 1] if i - 1 < len(inputs) else np.zeros(self.input_dim)
            )

            # Simple Euler integration
            dxdt = self.state_equation(current_state, time_points[i - 1], current_input)
            current_state = current_state + dt * dxdt

            # Add process noise
            noise = self.rng.multivariate_normal(np.zeros(self.state_dim), self.Q * dt)
            current_state += noise

            state_trajectory[i] = current_state

        return state_trajectory

    def generate_observations(self, state_trajectory: np.ndarray) -> np.ndarray:
        """
        Generate observations from state trajectory.

        Args:
            state_trajectory: State trajectory

        Returns:
            Observation trajectory
        """
        n_timesteps = state_trajectory.shape[0]
        observations = np.zeros((n_timesteps, self.output_dim))

        for i in range(n_timesteps):
            observations[i] = self.observation_equation(state_trajectory[i])

        return observations

    def estimate_parameters(
        self,
        observations: np.ndarray,
        inputs: np.ndarray,
        time_points: np.ndarray,
        initial_state: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Estimate model parameters from data.

        Args:
            observations: Observation sequence
            inputs: Input sequence
            time_points: Time points
            initial_state: Initial state estimate

        Returns:
            Estimated parameters
        """
        if initial_state is None:
            initial_state = np.zeros(self.state_dim)

        observations = np.asarray(observations, dtype=float)
        inputs = np.asarray(inputs, dtype=float)
        time_points = np.asarray(time_points, dtype=float)
        initial_state = np.asarray(initial_state, dtype=float)
        if observations.ndim != 2 or observations.shape[1] != self.output_dim:
            raise ValueError(f"observations must have shape (n, {self.output_dim})")
        if observations.shape[0] == 0:
            raise ValueError("observations must contain at least one timestep")
        if time_points.ndim != 1 or len(time_points) != len(observations):
            raise ValueError("time_points must have one value per observation")
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape (n, {self.input_dim})")
        if initial_state.shape != (self.state_dim,):
            raise ValueError(f"initial_state must have shape ({self.state_dim},)")

        # Simplified parameter estimation using least squares
        # In practice, would use more sophisticated methods like EM algorithm

        # Estimate state trajectory using Kalman smoother (simplified)
        estimated_states = self._estimate_states(
            observations, inputs, time_points, initial_state
        )

        # Estimate A and B matrices from state dynamics
        X = estimated_states[:-1]  # Current states
        X_next = estimated_states[1:]  # Next states
        if len(inputs) < len(X):
            raise ValueError("inputs must contain at least n_observations - 1 rows")
        U = inputs[: len(X)]

        # Solve: X_next = X*A.T + U*B.T
        if len(X) > 0:
            XU = np.hstack([X, U])
            AB = np.linalg.lstsq(XU, X_next, rcond=None)[0]

            estimated_A = AB[: self.state_dim].T
            estimated_B = AB[self.state_dim :].T if self.input_dim > 0 else self.B
        else:
            estimated_A = self.A
            estimated_B = self.B

        # Estimate C matrix from observations
        if len(estimated_states) > 0:
            estimated_C = np.linalg.lstsq(estimated_states, observations, rcond=None)[
                0
            ].T
        else:
            estimated_C = self.C

        return {
            "A": estimated_A,
            "B": estimated_B,
            "C": estimated_C,
            "estimated_states": estimated_states,
        }

    def _estimate_states(
        self,
        observations: np.ndarray,
        inputs: np.ndarray,
        time_points: np.ndarray,
        initial_state: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate state trajectory using simplified Kalman filter.

        Args:
            observations: Observation sequence
            inputs: Input sequence
            time_points: Time points
            initial_state: Initial state

        Returns:
            Estimated state trajectory
        """
        n_timesteps = len(observations)
        states = np.zeros((n_timesteps, self.state_dim))

        # Initialize
        current_state = initial_state.copy()
        current_cov = np.eye(self.state_dim)

        states[0] = current_state

        for i in range(1, n_timesteps):
            dt = (
                time_points[i] - time_points[i - 1] if i < len(time_points) else self.dt
            )
            current_input = (
                inputs[i - 1] if i - 1 < len(inputs) else np.zeros(self.input_dim)
            )

            # Prediction step
            pred_state = current_state + dt * self.state_equation(
                current_state, time_points[i - 1], current_input
            )
            if dt <= 0:
                raise ValueError("time_points must be strictly increasing")
            pred_cov = current_cov + self.Q * dt

            # Update step
            innovation = observations[i] - self.C @ pred_state
            innovation_cov = self.C @ pred_cov @ self.C.T + self.R
            cross_cov = pred_cov @ self.C.T
            kalman_gain = np.linalg.solve(innovation_cov.T, cross_cov.T).T

            current_state = pred_state + kalman_gain @ innovation
            residual_transform = np.eye(self.state_dim) - kalman_gain @ self.C
            current_cov = (
                residual_transform @ pred_cov @ residual_transform.T
                + kalman_gain @ self.R @ kalman_gain.T
            )
            current_cov = (current_cov + current_cov.T) / 2.0

            states[i] = current_state

        return states

    def set_parameters(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
        """Set model parameters after validating their matrix contracts."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        expected = {
            "A": ((self.state_dim, self.state_dim), A),
            "B": ((self.state_dim, self.input_dim), B),
            "C": ((self.output_dim, self.state_dim), C),
        }
        for name, (shape, matrix) in expected.items():
            if matrix.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be finite")
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()

    def set_noise_parameters(self, Q: np.ndarray, R: np.ndarray) -> None:
        """Set positive-definite process and observation noise covariances."""
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        expected = {
            "Q": ((self.state_dim, self.state_dim), Q),
            "R": ((self.output_dim, self.output_dim), R),
        }
        for name, (shape, matrix) in expected.items():
            if matrix.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
                raise ValueError(f"{name} must be finite and symmetric")
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name} must be positive definite") from exc
        self.Q = Q.copy()
        self.R = R.copy()
