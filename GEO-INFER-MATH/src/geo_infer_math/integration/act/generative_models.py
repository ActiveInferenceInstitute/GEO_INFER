"""Generative model construction for Active Inference.

Provides tools for building POMDP generative models with proper
A (likelihood), B (transition), C (preference), and D (prior) matrices.

References:
    Da Costa, L. et al. (2020). Active inference on discrete state-spaces:
    A synthesis. Journal of Mathematical Psychology, 99, 102447.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class GenerativeModels:
    """Generative model construction tools for Active Inference.

    Builds POMDP generative models parameterised by A/B/C/D matrices
    for use in Active Inference agents.
    """

    def __init__(self, epsilon: float = 1e-16) -> None:
        """Initialize generative model builder.

        Args:
            epsilon: Numerical stability constant.
        """
        self._epsilon = epsilon
        logger.debug("GenerativeModels initialized")

    def create_generative_model(
        self,
        model_type: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Create a POMDP generative model.

        Args:
            model_type: Type of model — 'categorical', 'grid_world', or 'custom'.
            parameters: Model parameters. Expected keys depend on model_type:
                - 'categorical': 'n_states', 'n_obs', 'n_actions'
                - 'grid_world': 'grid_size'
                - 'custom': 'A', 'B', and optionally 'C', 'D'

        Returns:
            Dictionary with 'A', 'B', 'C', 'D' matrices.
        """
        if model_type == "categorical":
            return self._build_categorical(parameters)
        elif model_type == "grid_world":
            return self._build_grid_world(parameters)
        elif model_type == "custom":
            return self._validate_custom(parameters)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _build_categorical(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """Build categorical POMDP with uniform initialisation."""
        n_states = params.get("n_states", 4)
        n_obs = params.get("n_obs", 4)
        n_actions = params.get("n_actions", 2)

        # A: Likelihood p(o|s) — identity-like with noise
        A = np.eye(n_obs, n_states) * 0.8
        if n_obs == n_states:
            A += 0.2 / n_states
        else:
            A += 0.2 / max(n_obs, n_states)
        A = A / A.sum(axis=0, keepdims=True)

        # B: Transition p(s'|s, a) — one matrix per action
        B = np.zeros((n_states, n_states, n_actions))
        for a in range(n_actions):
            # Each action induces a cyclic shift + noise
            shift = (a + 1) % n_states
            for s in range(n_states):
                s_next = (s + shift) % n_states
                B[s_next, s, a] = 0.8
            B[:, :, a] += 0.2 / n_states
            B[:, :, a] = B[:, :, a] / B[:, :, a].sum(axis=0, keepdims=True)

        # C: Preferences ln p(o) — slight preference for first observation
        C = np.zeros(n_obs)
        C[0] = 1.0

        # D: Prior p(s_0) — uniform
        D = np.ones(n_states) / n_states

        logger.debug(
            "Built categorical model: n_states=%d, n_obs=%d, n_actions=%d",
            n_states,
            n_obs,
            n_actions,
        )
        return {"A": A, "B": B, "C": C, "D": D}

    def _build_grid_world(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """Build 2-D grid-world POMDP."""
        grid_size = params.get("grid_size", 3)
        n_states = grid_size * grid_size
        n_obs = n_states  # Full observability
        n_actions = 4  # Up, Down, Left, Right

        # A: Identity (fully observable grid)
        A = np.eye(n_obs, n_states)

        # B: Transition matrices for 4 directional actions
        B = np.zeros((n_states, n_states, n_actions))
        for s in range(n_states):
            row, col = divmod(s, grid_size)
            for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    s_next = nr * grid_size + nc
                else:
                    s_next = s  # Stay in place at boundaries
                B[s_next, s, a] = 1.0

        # C: Prefer goal state (bottom-right corner)
        C = np.zeros(n_obs)
        C[-1] = 2.0

        # D: Start at top-left
        D = np.zeros(n_states)
        D[0] = 1.0

        logger.debug(
            "Built grid_world model: %dx%d (%d states)", grid_size, grid_size, n_states
        )
        return {"A": A, "B": B, "C": C, "D": D}

    def _validate_custom(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """Validate and return custom A/B/C/D matrices."""
        A = np.asarray(params["A"], dtype=np.float64)
        B = np.asarray(params["B"], dtype=np.float64)

        n_obs, n_states = A.shape[:2]

        C = params.get("C", np.zeros(n_obs))
        C = np.asarray(C, dtype=np.float64)

        D = params.get("D", np.ones(n_states) / n_states)
        D = np.asarray(D, dtype=np.float64)
        D = D / (D.sum() + self._epsilon)

        # Validate A columns sum to 1
        col_sums = A.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            logger.warning("A matrix columns do not sum to 1; normalising.")
            A = A / (col_sums + self._epsilon)

        logger.debug("Validated custom model: n_states=%d, n_obs=%d", n_states, n_obs)
        return {"A": A, "B": B, "C": C, "D": D}
