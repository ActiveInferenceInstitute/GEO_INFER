"""Bridge between MATH optimization and AI training.

Provides bridges between scipy optimisers and ML training loops
with learning rate scheduling, gradient clipping, and spatial
regularisation support.
"""

import numpy as np
from typing import Any
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


class OptimizationBridges:
    """Bridge between MATH optimization and AI training.

    Wraps scipy-style optimizers for use in ML training pipelines,
    adding learning rate scheduling, gradient clipping, and
    convergence monitoring.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        convergence_tol: float = 1e-6,
    ) -> None:
        """Initialize optimization bridge.

        Args:
            learning_rate: Initial learning rate.
            max_iterations: Maximum training iterations.
            convergence_tol: Convergence tolerance.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        logger.debug(
            "OptimizationBridges initialized (lr=%.4f, max_iter=%d)",
            learning_rate,
            max_iterations,
        )

    def bridge_optimize(
        self,
        objective: Callable,
        initial_guess: np.ndarray,
        gradient_fn: Callable | None = None,
        scheduler: str = "constant",
        clip_norm: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimisation with ML-style training loop.

        Args:
            objective: Loss function f(params) → scalar.
            initial_guess: Initial parameter values.
            gradient_fn: Gradient function. If None, uses finite differences.
            scheduler: Learning rate schedule — 'constant', 'step_decay', 'cosine'.
            clip_norm: Maximum gradient norm for clipping. None to disable.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary with 'optimal_params', 'loss_history', 'grad_norm_history',
            'converged', 'n_iterations'.
        """
        params = np.asarray(initial_guess, dtype=np.float64).copy()
        loss_history: list[float] = []
        grad_norm_history: list[float] = []

        for iteration in range(self.max_iterations):
            # Compute loss
            loss = float(objective(params))
            loss_history.append(loss)

            # Compute gradient
            if gradient_fn is not None:
                grad = np.asarray(gradient_fn(params), dtype=np.float64)
            else:
                grad = self._finite_difference_gradient(objective, params)

            # Gradient clipping
            grad_norm = float(np.linalg.norm(grad))
            grad_norm_history.append(grad_norm)

            if clip_norm is not None and grad_norm > clip_norm:
                grad = grad * (clip_norm / grad_norm)
                grad_norm = clip_norm

            # Learning rate schedule
            lr = self._get_learning_rate(iteration, scheduler)

            # Update parameters
            params = params - lr * grad

            # Convergence check
            if len(loss_history) > 1:
                loss_change = abs(loss_history[-1] - loss_history[-2])
                if (
                    loss_change < self.convergence_tol
                    and grad_norm < self.convergence_tol
                ):
                    logger.debug(
                        "Optimization converged at iteration %d (loss=%.6f)",
                        iteration,
                        loss,
                    )
                    return {
                        "optimal_params": params,
                        "loss_history": loss_history,
                        "grad_norm_history": grad_norm_history,
                        "converged": True,
                        "n_iterations": iteration + 1,
                    }

        logger.debug(
            "Optimization did not converge after %d iterations (final loss=%.6f)",
            self.max_iterations,
            loss_history[-1],
        )
        return {
            "optimal_params": params,
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "converged": False,
            "n_iterations": self.max_iterations,
        }

    def _get_learning_rate(self, iteration: int, scheduler: str) -> float:
        """Calculate learning rate for current iteration."""
        if scheduler == "constant":
            return self.learning_rate
        elif scheduler == "step_decay":
            # Halve every 100 iterations
            return self.learning_rate * (0.5 ** (iteration // 100))
        elif scheduler == "cosine":
            # Cosine annealing
            return float(
                self.learning_rate
                * 0.5
                * (1 + np.cos(np.pi * iteration / self.max_iterations))
            )
        else:
            return self.learning_rate

    def _finite_difference_gradient(
        self,
        fn: Callable,
        params: np.ndarray,
        epsilon: float = 1e-7,
    ) -> np.ndarray:
        """Compute gradient via central finite differences."""
        n = len(params)
        grad = np.zeros(n)
        for i in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad[i] = (fn(params_plus) - fn(params_minus)) / (2 * epsilon)
        return grad
