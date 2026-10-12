"""
Continuous POMDP Active Inference and Gaussian Filter Engine for GEO-INFER-ACT.

Bridges discrete categorical active inference with continuous state-space
generalized predictive coding and Kalman/Laplace filter dynamics.
"""

from __future__ import annotations

import logging
from typing import Any, cast
import numpy as np

from geo_infer_act.core.types import FreeEnergyBreakdown

logger = logging.getLogger(__name__)


class ContinuousPOMDPActiveInference:
    """Linear Gaussian filtering and finite-candidate active control.

    Continuous mode discretizes dx/dt = A x + B u with Euler steps and dt*Q.
    Discrete mode uses x[t+1] = A x[t] + B u[t] and Q per step directly.
    Observation noise R is per measurement in both modes; dt is in seconds.
    """

    def __init__(
        self,
        state_dim: int = 2,
        obs_dim: int = 2,
        action_dim: int = 2,
        dt: float = 0.1,
        process_noise_cov: np.ndarray | None = None,
        obs_noise_cov: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_cov: np.ndarray | None = None,
        target_prior: np.ndarray | None = None,
        random_seed: int | None = None,
        time_domain: str = "continuous",
    ):
        for name, value in (
            ("state_dim", state_dim),
            ("obs_dim", obs_dim),
            ("action_dim", action_dim),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if time_domain not in {"continuous", "discrete"}:
            raise ValueError("time_domain must be continuous or discrete")
        self.time_domain = time_domain
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt
        self.rng = np.random.default_rng(random_seed)

        # Transition dynamics matrices: dx/dt = A x + B u
        self.A = np.zeros((state_dim, state_dim))
        np.fill_diagonal(
            self.A, -0.1 if time_domain == "continuous" else 1.0
        )  # stable decay by default
        self.B = np.eye(state_dim, action_dim)

        # Observation mapping: y = C x
        self.C = np.eye(obs_dim, state_dim)

        # Covariances
        self.Q = (
            process_noise_cov
            if process_noise_cov is not None
            else np.eye(state_dim) * 0.05
        )
        self.R = obs_noise_cov if obs_noise_cov is not None else np.eye(obs_dim) * 0.05

        # Belief states (mean and covariance)
        self.mu = (
            np.zeros(state_dim)
            if prior_mean is None
            else np.asarray(prior_mean, dtype=float).copy()
        )
        self.sigma = (
            np.eye(state_dim)
            if prior_cov is None
            else np.asarray(prior_cov, dtype=float).copy()
        )

        # Desired/preferred continuous observations
        self.target_prior = (
            np.zeros(obs_dim)
            if target_prior is None
            else np.asarray(target_prior, dtype=float).copy()
        )

        self.Q = self._covariance(self.Q, state_dim, "Q", semidefinite=True)
        self.R = self._covariance(self.R, obs_dim, "R")
        self.sigma = self._covariance(self.sigma, state_dim, "prior_cov")
        self.mu = self._array(self.mu, (state_dim,), "prior_mean")
        self.target_prior = self._array(self.target_prior, (obs_dim,), "target_prior")
        self._last_update: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        return array.copy()

    @classmethod
    def _covariance(
        cls, value: Any, size: int, name: str, semidefinite: bool = False
    ) -> np.ndarray:
        array = cls._array(value, (size, size), name)
        if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        array = (array + array.T) / 2
        eigenvalues = np.linalg.eigvalsh(array)
        if np.min(eigenvalues) < 0 or (not semidefinite and np.min(eigenvalues) <= 0):
            raise ValueError(
                f"{name} must be positive {'semidefinite' if semidefinite else 'definite'}"
            )
        return array

    def _discrete_dynamics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.time_domain == "discrete":
            return self.A, self.B, self.Q
        return (
            np.eye(self.state_dim) + self.dt * self.A,
            self.dt * self.B,
            self.dt * self.Q,
        )

    def _entropy(self) -> float:
        return 0.5 * float(
            self.state_dim * np.log(2 * np.pi * np.e) + np.linalg.slogdet(self.sigma)[1]
        )

    def set_system_matrices(
        self,
        A: np.ndarray | None = None,
        B: np.ndarray | None = None,
        C: np.ndarray | None = None,
    ) -> None:
        """Set continuous transition, control, and measurement matrices."""
        new_A = (
            self.A
            if A is None
            else self._array(A, (self.state_dim, self.state_dim), "A")
        )
        new_B = (
            self.B
            if B is None
            else self._array(B, (self.state_dim, self.action_dim), "B")
        )
        new_C = (
            self.C if C is None else self._array(C, (self.obs_dim, self.state_dim), "C")
        )
        self.A, self.B, self.C = new_A, new_B, new_C
        self._last_update = None

    def predict(
        self, action: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate one interval; discrete matrices already include the interval."""
        u = (
            np.zeros(self.action_dim)
            if action is None
            else self._array(action, (self.action_dim,), "action")
        )
        F, control, noise = self._discrete_dynamics()
        return F @ self.mu + control @ u, F @ self.sigma @ F.T + noise

    def update_beliefs(
        self, observation: np.ndarray, action: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Bayesian belief update on receiving continuous measurement.

        Returns:
            (updated_mean, updated_covariance, free_energy)
        """
        y = self._array(observation, (self.obs_dim,), "observation")
        mu_pred, sigma_pred = self.predict(action)

        # Innovation / prediction error
        y_pred = self.C @ mu_pred
        eps = y - y_pred

        # Innovation covariance S = C * sigma_pred * C.T + R
        S = self.C @ sigma_pred @ self.C.T + self.R
        K = np.linalg.solve(S, self.C @ sigma_pred).T

        # Posterior update
        posterior_mean = mu_pred + K @ eps
        I_KC = np.eye(self.state_dim) - K @ self.C
        posterior_cov = I_KC @ sigma_pred @ I_KC.T + K @ self.R @ K.T

        # Variational Free Energy: F = 0.5 * (eps.T @ S^-1 @ eps + log|S| + c)
        sign, logdet = np.linalg.slogdet(S)
        vfe = 0.5 * float(
            eps.T @ np.linalg.solve(S, eps) + logdet + self.obs_dim * np.log(2 * np.pi)
        )

        if sign <= 0 or not np.isfinite(vfe):
            raise ValueError("Invalid innovation covariance or free energy")
        posterior_cov = self._covariance(
            posterior_cov, self.state_dim, "posterior covariance"
        )
        self.mu, self.sigma = posterior_mean, posterior_cov
        self._last_update = (y.copy(), mu_pred.copy(), sigma_pred.copy())
        record = {
            "observation": y.tolist(),
            "belief_mean": self.mu.copy().tolist(),
            "belief_cov_diag": np.diag(self.sigma).tolist(),
            "free_energy": vfe,
        }
        self.history.append(record)
        return self.mu.copy(), self.sigma.copy(), vfe

    # ------------------------------------------------------------------
    # Laplace / Kalman-Bucy filtering diagnostics
    # ------------------------------------------------------------------
    def _innovation_covariance(self, action: np.ndarray | None = None) -> np.ndarray:
        """Return the innovation covariance S = C sigma_pred C^T + R."""
        _, sigma_pred = self.predict(action)
        return cast(np.ndarray, self.C @ sigma_pred @ self.C.T + self.R)

    def _adaptive_precision(self, action: np.ndarray | None = None) -> float:
        """
        Laplace-scaled precision schedule.  Agrees with the inverse
        trace of the innovation covariance so that high innovation
        (surprising observations) downweights the current posterior's
        confidence during action evaluation.
        """
        S = self._innovation_covariance(action)
        trace = float(np.trace(S))
        if not np.isfinite(trace) or trace <= 1e-12:
            return 1.0
        return float(np.clip(1.0 / trace, 1e-3, 1e3))

    def compute_variational_free_energy(
        self,
        observation: np.ndarray | None = None,
        action: np.ndarray | None = None,
    ) -> FreeEnergyBreakdown:
        """Return F = KL(q || predictive prior) - E_q[log p(y | x)].

        With no arguments, evaluate the most recent measurement and its saved
        predictive prior. Before any update, evaluate the current belief against
        its next predictive prior at the current expected measurement. Explicit
        arguments request that latter prospective diagnostic.
        """
        if observation is None and action is None and self._last_update is not None:
            y, mu_pred, sigma_pred = self._last_update
        else:
            y = (
                self.C @ self.mu
                if observation is None
                else self._array(observation, (self.obs_dim,), "observation")
            )
            mu_pred, sigma_pred = self.predict(action)
        sigma_pred = self._covariance(
            sigma_pred, self.state_dim, "predictive covariance"
        )
        residual = y - self.C @ self.mu
        accuracy = -0.5 * float(
            self.obs_dim * np.log(2 * np.pi)
            + np.linalg.slogdet(self.R)[1]
            + residual @ np.linalg.solve(self.R, residual)
            + np.trace(np.linalg.solve(self.R, self.C @ self.sigma @ self.C.T))
        )
        delta = self.mu - mu_pred
        complexity = 0.5 * float(
            np.trace(np.linalg.solve(sigma_pred, self.sigma))
            + delta @ np.linalg.solve(sigma_pred, delta)
            - self.state_dim
            + np.linalg.slogdet(sigma_pred)[1]
            - np.linalg.slogdet(self.sigma)[1]
        )
        return FreeEnergyBreakdown(
            free_energy=complexity - accuracy,
            accuracy=accuracy,
            complexity=complexity,
            entropy=self._entropy(),
            metadata={
                "model_type": "gaussian_laplace",
                "time_domain": self.time_domain,
                "adaptive_precision": self._adaptive_precision(action),
            },
        )

    def compute_expected_free_energy(
        self,
        action: np.ndarray,
        horizon: int = 1,
        return_breakdown: bool = False,
        epistemic_weight: float = 1.0,
        preference_prior: np.ndarray | None = None,
    ) -> float | FreeEnergyBreakdown:
        """
        Expected free energy for a continuous control action under a Laplace
        filter.

        G(pi) = -Epistemic value - Pragmatic value with simultaneous
        decomposition:

        - **Pragmatic cost** is the expected squared error between the
          projected observation trajectory and the preferred target
          (goal-seeking; minimized by approaching ``target_prior``).  It is
          exposed as ``pragmatic_value`` in the breakdown.
        - **Epistemic gain** is the expected information gain measured by
          the sum of conditional Gaussian mutual information
          (uncertainty resolution; strongest when sensing would disambiguate
          hidden states).
        - **Risk** folds in the control effort, favouring parsimonious
          commands.

        Returns a ``FreeEnergyBreakdown`` when requested; the scalar combined
        EFE otherwise.
        """
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        if not np.isfinite(epistemic_weight) or epistemic_weight < 0:
            raise ValueError("epistemic_weight must be finite and nonnegative")
        target = (
            self.target_prior
            if preference_prior is None
            else self._array(preference_prior, (self.obs_dim,), "preference_prior")
        )
        u = self._array(action, (self.action_dim,), "action")
        mu_h = self.mu.copy()
        cov_h = self.sigma.copy()
        cost_cov_h = self.sigma.copy()
        pragmatic = 0.0
        epistemic = 0.0
        F, control_matrix, noise = self._discrete_dynamics()
        for _ in range(horizon):
            mu_h = F @ mu_h + control_matrix @ u
            cov_h = F @ cov_h @ F.T + noise
            cost_cov_h = F @ cost_cov_h @ F.T + noise
            y_proj = self.C @ mu_h
            # Pragmatic cost: summed squared error to the target (lower is
            # better; drives goal-seeking).
            pragmatic += float(
                np.sum((y_proj - target) ** 2)
                + np.trace(self.C @ cost_cov_h @ self.C.T + self.R)
            )
            # Gaussian mutual information I(x;y) = .5 log(|H P H' + R| / |R|).
            # Condition covariance on each anticipated measurement so repeated
            # sensing does not count the same uncertainty again.
            S_h = self.C @ cov_h @ self.C.T + self.R
            sign, ld = np.linalg.slogdet(S_h)
            if sign > 0 and np.isfinite(ld):
                epistemic += 0.5 * float(ld - np.linalg.slogdet(self.R)[1])
            gain = np.linalg.solve(S_h, self.C @ cov_h).T
            residual_map = np.eye(self.state_dim) - gain @ self.C
            cov_h = residual_map @ cov_h @ residual_map.T + gain @ self.R @ gain.T
        control = 0.01 * horizon * float(np.sum(u**2))
        pragmatic_value = float(pragmatic)
        epistemic_value = float(epistemic)
        # Minimise pragmatic cost and control effort while rewarding epistemic
        # (information-gain) value via the subtractive epistemic term.
        expected_free_energy = float(pragmatic - epistemic_weight * epistemic + control)
        if return_breakdown:
            return FreeEnergyBreakdown(
                free_energy=expected_free_energy,
                pragmatic_value=pragmatic_value,
                epistemic_value=epistemic_value,
                risk=control,
                entropy=self._entropy(),
                metadata={
                    "action": u.tolist(),
                    "epistemic_weight": float(epistemic_weight),
                    "adaptive_precision": self._adaptive_precision(),
                },
            )
        return expected_free_energy

    def select_action(self, horizon: int = 5) -> np.ndarray:
        """
        Select continuous control action minimizing Expected Free Energy (EFE).
        """
        # Use the decomposed epistemic + pragmatic EFE to prefer actions that
        # both approach the target and reduce hidden-state uncertainty.
        scoreboard = self.evaluate_actions(horizon=horizon)
        return cast(np.ndarray, scoreboard["best_action"])

    def evaluate_actions(
        self,
        horizon: int = 1,
        candidate_actions: list[np.ndarray] | None = None,
        epistemic_weight: float = 1.0,
    ) -> dict[str, Any]:
        """
        Score a set of continuous control actions by expected free energy,
        returning the per-action decomposition plus the arg-minimum action.

        Returns a dictionary with keys:
            ``candidates``, ``efe_scores``, ``pragmatic_values``,
            ``epistemic_values``, ``best_index``, ``best_action``,
            ``best_efe`` and ``metadata`` (pure Python floats for logging).
        """
        if candidate_actions is None:
            candidates: list[np.ndarray] = [
                np.zeros(self.action_dim),
                np.ones(self.action_dim) * 0.5,
                -np.ones(self.action_dim) * 0.5,
                np.ones(self.action_dim) * 1.0,
                -np.ones(self.action_dim) * 1.0,
            ]
        else:
            candidates = [
                self._array(action, (self.action_dim,), "candidate action")
                for action in candidate_actions
            ]
        if not candidates:
            raise ValueError("candidate_actions must not be empty")
        efe_scores: list[float] = []
        pragmatic_values: list[float] = []
        epistemic_values: list[float] = []
        for cand in candidates:
            breakdown = cast(
                FreeEnergyBreakdown,
                self.compute_expected_free_energy(
                    cand,
                    horizon=horizon,
                    return_breakdown=True,
                    epistemic_weight=epistemic_weight,
                ),
            )
            efe_scores.append(breakdown.free_energy)
            pragmatic_values.append(breakdown.pragmatic_value)
            epistemic_values.append(breakdown.epistemic_value)
        best_index = int(np.argmin(efe_scores))
        return {
            "candidates": [candidate.tolist() for candidate in candidates],
            "efe_scores": efe_scores,
            "pragmatic_values": pragmatic_values,
            "epistemic_values": epistemic_values,
            "best_index": best_index,
            "best_action": candidates[best_index],
            "best_efe": float(efe_scores[best_index]),
            "breakdowns": {
                "pragmatic": pragmatic_values,
                "epistemic": epistemic_values,
                "efe": efe_scores,
            },
        }
