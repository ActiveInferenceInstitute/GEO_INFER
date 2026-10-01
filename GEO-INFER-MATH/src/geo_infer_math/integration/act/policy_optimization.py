"""Policy optimization for Active Inference.

Implements expected free energy minimisation and softmax policy
selection for Active Inference agents.

References:
    Friston, K. et al. (2015). Active inference and epistemic value.
    Cognitive Neuroscience, 6(4), 187-214.
"""

import numpy as np
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)


class PolicyOptimization:
    """Policy optimization for Active Inference via expected free energy.

    Evaluates and selects policies by minimising expected free energy
    G, balancing pragmatic value (exploitation) and epistemic value
    (exploration).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        epsilon: float = 1e-16,
    ) -> None:
        """Initialize policy optimizer.

        Args:
            gamma: Inverse temperature for policy softmax selection.
                Higher values → more exploitative (less stochastic).
            epsilon: Numerical stability constant.
        """
        self.gamma = gamma
        self._epsilon = epsilon
        logger.debug("PolicyOptimization initialized (gamma=%.2f)", gamma)

    def optimize_policy(
        self,
        policies: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray | None = None,
        current_beliefs: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Select a policy by minimising expected free energy.

        Args:
            policies: (n_policies, horizon) array of action indices.
            A: Likelihood matrix p(o|s), shape (n_obs, n_states).
            B: Transition matrices p(s'|s, a), shape (n_states, n_states, n_actions).
            C: Preferred outcomes ln p(o), shape (n_obs,). Defaults to uniform.
            current_beliefs: Current state beliefs q(s), shape (n_states,).
                Defaults to uniform.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary with 'selected_policy', 'policy_probs', 'G_values',
            'pragmatic_values', 'epistemic_values'.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        policies = np.asarray(policies, dtype=int)

        n_obs, n_states = A.shape
        n_policies = len(policies)

        if C is None:
            C = np.zeros(n_obs)
        else:
            C = np.asarray(C, dtype=np.float64)

        if current_beliefs is None:
            current_beliefs = np.ones(n_states) / n_states
        else:
            current_beliefs = np.asarray(current_beliefs, dtype=np.float64)
            current_beliefs = current_beliefs / (current_beliefs.sum() + self._epsilon)

        # Evaluate expected free energy for each policy
        G_values = np.zeros(n_policies)
        pragmatic_values = np.zeros(n_policies)
        epistemic_values = np.zeros(n_policies)

        for pi in range(n_policies):
            G, prag, epist = self._evaluate_policy(
                policies[pi], A, B, C, current_beliefs
            )
            G_values[pi] = G
            pragmatic_values[pi] = prag
            epistemic_values[pi] = epist

        # Softmax policy selection: π(u) = σ(-γ·G)
        policy_probs = self._softmax(-self.gamma * G_values)

        selected = int(np.argmax(policy_probs))

        logger.debug(
            "Policy optimization: selected=%d (prob=%.4f, G=%.4f), n_policies=%d",
            selected,
            policy_probs[selected],
            G_values[selected],
            n_policies,
        )

        return {
            "selected_policy": selected,
            "policy_probs": policy_probs,
            "G_values": G_values,
            "pragmatic_values": pragmatic_values,
            "epistemic_values": epistemic_values,
        }

    def _evaluate_policy(
        self,
        policy: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        beliefs: np.ndarray,
    ) -> tuple:
        """Evaluate expected free energy G for a single policy.

        G = ambiguity + risk
          = E_q[H[p(o|s)]] + D_KL[q(o) || p(o)]
        """
        current = beliefs.copy()

        total_ambiguity = 0.0
        total_risk = 0.0

        for t, action in enumerate(policy):
            # Predict future state: q(s_t+1) = B(a) @ q(s_t)
            if action < B.shape[2]:
                current = B[:, :, action] @ current
            current = current / (current.sum() + self._epsilon)

            # Predicted observations: q(o) = A @ q(s)
            predicted_obs = A @ current
            predicted_obs = predicted_obs / (predicted_obs.sum() + self._epsilon)

            # Ambiguity: expected entropy of likelihood under beliefs
            log_A = np.log(A + self._epsilon)
            ambiguity = -np.sum(current * np.sum(A * log_A, axis=0))
            total_ambiguity += ambiguity

            # Risk: D_KL[q(o) || softmax(C)]
            preferred = self._softmax(C)
            risk = float(
                np.sum(
                    predicted_obs
                    * np.log(
                        (predicted_obs + self._epsilon) / (preferred + self._epsilon)
                    )
                )
            )
            total_risk += risk

        G = total_ambiguity + total_risk
        return G, total_risk, total_ambiguity

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = logits - np.max(logits)
        exp_x = np.exp(x)
        return cast(np.ndarray, exp_x / (exp_x.sum() + self._epsilon))
