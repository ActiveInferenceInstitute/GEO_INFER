"""Bayesian optimisation for spatial parameter tuning.

Implements Expected Improvement acquisition function with
Gaussian process surrogate model for black-box optimisation.
"""

import numpy as np
from typing import Any, cast
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


class BayesianOptimization:
    """Bayesian optimisation via Expected Improvement.

    Uses a simple RBF-kernel GP surrogate model to guide exploration
    and exploitation of a black-box objective function.
    """

    def __init__(
        self,
        bounds: np.ndarray | None = None,
        n_initial: int = 5,
        max_iterations: int = 25,
        length_scale: float = 1.0,
        noise: float = 1e-6,
    ) -> None:
        """Initialize Bayesian optimiser.

        Args:
            bounds: (d, 2) array of [low, high] bounds per dimension.
            n_initial: Number of initial random evaluations.
            max_iterations: Maximum BO iterations after initialisation.
            length_scale: RBF kernel length scale.
            noise: Observation noise variance.
        """
        self.bounds = bounds
        self.n_initial = n_initial
        self.max_iterations = max_iterations
        self.length_scale = length_scale
        self.noise = noise
        logger.debug(
            "BayesianOptimization initialized (n_init=%d, max_iter=%d)",
            n_initial,
            max_iterations,
        )

    def optimize(
        self,
        objective: Callable,
        bounds: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run Bayesian optimisation loop.

        Args:
            objective: Black-box function f(x) → scalar (to minimise).
            bounds: (d, 2) array of bounds. Overrides constructor bounds.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary with 'best_x', 'best_y', 'X_observed', 'Y_observed',
            'n_evaluations'.
        """
        bounds = bounds if bounds is not None else self.bounds
        if bounds is None:
            raise ValueError("Bounds must be provided.")
        bounds = np.asarray(bounds, dtype=np.float64)

        # Phase 1: random initialisation
        X = self._random_points(bounds, self.n_initial)
        Y = np.array([float(objective(x)) for x in X])

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        logger.debug("Initial best: y=%.6f at x=%s", best_y, best_x)

        # Phase 2: BO loop
        for iteration in range(self.max_iterations):
            # Fit GP surrogate
            mu, sigma = self._gp_predict(X, Y, bounds)

            # Find next point via Expected Improvement
            next_x = self._maximise_ei(mu, sigma, best_y, bounds)

            # Evaluate objective
            next_y = float(objective(next_x))

            # Update dataset
            X = np.vstack([X, next_x])
            Y = np.append(Y, next_y)

            if next_y < best_y:
                best_y = next_y
                best_x = next_x.copy()
                logger.debug(
                    "BO iter %d: new best y=%.6f at x=%s",
                    iteration,
                    best_y,
                    best_x,
                )

        logger.debug(
            "BO complete: best_y=%.6f, n_evals=%d",
            best_y,
            len(Y),
        )

        return {
            "best_x": best_x,
            "best_y": best_y,
            "X_observed": X,
            "Y_observed": Y,
            "n_evaluations": len(Y),
        }

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        sq_dist = np.sum(
            (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2,
            axis=-1,
        )
        return cast(np.ndarray, np.exp(-0.5 * sq_dist / (self.length_scale**2)))

    def _gp_predict(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        bounds: np.ndarray,
        n_test: int = 50,
    ) -> tuple[Callable, Callable]:
        """Fit GP and return prediction functions.

        Returns closures (mu_fn, sigma_fn) that predict at arbitrary points.
        """
        K = self._rbf_kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        K_inv = np.linalg.inv(K + 1e-10 * np.eye(len(K)))
        alpha = K_inv @ Y_train

        def mu_fn(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            k_star = self._rbf_kernel(x, X_train)[0]
            return float(k_star @ alpha)

        def sigma_fn(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            k_star = self._rbf_kernel(x, X_train)[0]
            k_ss = float(self._rbf_kernel(x, x)[0, 0])
            var = k_ss - k_star @ K_inv @ k_star
            return float(np.sqrt(max(0.0, var)))

        return mu_fn, sigma_fn

    def _expected_improvement(
        self,
        x: np.ndarray,
        mu_fn: Callable,
        sigma_fn: Callable,
        best_y: float,
    ) -> float:
        """Compute Expected Improvement at x."""
        mu = mu_fn(x)
        sigma = sigma_fn(x)

        if sigma < 1e-12:
            return 0.0

        z = (best_y - mu) / sigma
        # Standard normal CDF / PDF via numpy
        from scipy.stats import norm

        ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return float(max(0.0, ei))

    def _maximise_ei(
        self,
        mu_fn: Callable,
        sigma_fn: Callable,
        best_y: float,
        bounds: np.ndarray,
        n_candidates: int = 100,
    ) -> np.ndarray:
        """Find point maximising Expected Improvement via random search."""
        candidates = self._random_points(bounds, n_candidates)
        ei_values = np.array(
            [self._expected_improvement(c, mu_fn, sigma_fn, best_y) for c in candidates]
        )
        return cast(np.ndarray, candidates[int(np.argmax(ei_values))])

    def _random_points(self, bounds: np.ndarray, n: int) -> np.ndarray:
        """Generate random points within bounds."""
        d = bounds.shape[0]
        points = np.random.rand(n, d)
        return cast(np.ndarray, bounds[:, 0] + points * (bounds[:, 1] - bounds[:, 0]))
