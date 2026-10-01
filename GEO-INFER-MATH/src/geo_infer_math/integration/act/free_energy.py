"""Free energy calculations for Active Inference.

Implements variational free energy F = E_q[ln q(s) - ln p(o,s)] and
expected free energy G for policy evaluation. All calculations use
real mathematical operations with numpy.

References:
    Friston, K. (2010). The free-energy principle: a unified brain theory?
    Nature Reviews Neuroscience, 11(2), 127-138.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class FreeEnergyCalculator:
    """Free energy calculations for Active Inference.

    Provides variational free energy, expected free energy, and
    Bethe free energy computations for spatial Active Inference models.
    """

    def __init__(self, epsilon: float = 1e-16) -> None:
        """Initialize calculator.

        Args:
            epsilon: Small constant for numerical stability in log operations.
        """
        self._epsilon = epsilon
        logger.debug("FreeEnergyCalculator initialized (epsilon=%.2e)", epsilon)

    def calculate(
        self,
        observations: np.ndarray,
        beliefs: np.ndarray,
        likelihood: np.ndarray | None = None,
        prior: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate variational free energy.

        F = E_q[ln q(s)] - E_q[ln p(o,s)]
          = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]
          = complexity - accuracy

        Args:
            observations: Observation vector o.
            beliefs: Approximate posterior q(s), must be a valid distribution.
            likelihood: Optional likelihood matrix p(o|s) of shape (n_obs, n_states).
            prior: Optional prior distribution p(s). Defaults to uniform.
            **kwargs: Additional parameters (unused, for API compat).

        Returns:
            Dictionary with 'free_energy', 'complexity', 'accuracy' keys.
        """
        beliefs = np.asarray(beliefs, dtype=np.float64)
        observations = np.asarray(observations, dtype=np.float64)

        # Normalise beliefs
        beliefs = beliefs / (beliefs.sum() + self._epsilon)

        n_states = len(beliefs)

        if prior is None:
            prior = np.ones(n_states) / n_states
        else:
            prior = np.asarray(prior, dtype=np.float64)
            prior = prior / (prior.sum() + self._epsilon)

        # Complexity: D_KL[q(s) || p(s)]
        complexity = self._kl_divergence(beliefs, prior)

        # Accuracy: E_q[ln p(o|s)]
        if likelihood is not None:
            likelihood = np.asarray(likelihood, dtype=np.float64)
            log_likelihood = np.log(likelihood + self._epsilon)
            # Marginalise over observations
            obs_normalised = observations / (observations.sum() + self._epsilon)
            accuracy = np.sum(beliefs * log_likelihood.T @ obs_normalised)
        else:
            # Without an explicit likelihood there is no observation term to
            # evaluate, so accuracy is exactly zero and F reduces to the
            # complexity term, F = complexity - accuracy per the docstring.
            accuracy = 0.0

        free_energy = complexity - accuracy

        logger.debug(
            "Free energy=%.4f (complexity=%.4f, accuracy=%.4f)",
            free_energy,
            complexity,
            accuracy,
        )

        return {
            "free_energy": float(free_energy),
            "complexity": float(complexity),
            "accuracy": float(accuracy),
        }

    def expected_free_energy(
        self,
        beliefs: np.ndarray,
        likelihood: np.ndarray,
        prior_preferences: np.ndarray | None = None,
    ) -> float:
        """Calculate expected free energy G for policy evaluation.

        G = E_q[ln q(s) - ln p(o, s)]
          = ambiguity + risk

        Ambiguity = E_q[H[p(o|s)]]  (expected entropy of likelihood)
        Risk = D_KL[q(o) || p(o)]   (divergence of predicted from preferred outcomes)

        Args:
            beliefs: Predicted state beliefs q(s) under a policy.
            likelihood: Likelihood matrix p(o|s) of shape (n_obs, n_states).
            prior_preferences: Preferred outcome distribution p(o). Defaults to uniform.

        Returns:
            Expected free energy (scalar).
        """
        beliefs = np.asarray(beliefs, dtype=np.float64)
        likelihood = np.asarray(likelihood, dtype=np.float64)
        beliefs = beliefs / (beliefs.sum() + self._epsilon)

        n_obs = likelihood.shape[0]

        # Ambiguity: expected entropy of likelihood under beliefs
        log_lik = np.log(likelihood + self._epsilon)
        ambiguity = -np.sum(beliefs * np.sum(likelihood * log_lik, axis=0))

        # Predicted observations: q(o) = sum_s p(o|s) q(s)
        predicted_obs = likelihood @ beliefs
        predicted_obs = predicted_obs / (predicted_obs.sum() + self._epsilon)

        if prior_preferences is None:
            prior_preferences = np.ones(n_obs) / n_obs
        else:
            prior_preferences = np.asarray(prior_preferences, dtype=np.float64)
            prior_preferences = prior_preferences / (
                prior_preferences.sum() + self._epsilon
            )

        # Risk: D_KL[q(o) || p(o)]
        risk = self._kl_divergence(predicted_obs, prior_preferences)

        G = ambiguity + risk
        logger.debug(
            "Expected free energy G=%.4f (ambiguity=%.4f, risk=%.4f)",
            G,
            ambiguity,
            risk,
        )
        return float(G)

    def bethe_free_energy(
        self,
        node_beliefs: np.ndarray,
        pairwise_beliefs: np.ndarray,
        node_potentials: np.ndarray,
        edge_potentials: np.ndarray,
    ) -> float:
        """Calculate Bethe free energy for loopy belief propagation.

        F_Bethe = sum_i U_i - sum_i (d_i - 1) H_i + sum_{ij} U_{ij} - H_{ij}

        Args:
            node_beliefs: (N,) node belief marginals.
            pairwise_beliefs: (N, N) pairwise belief matrix.
            node_potentials: (N,) node log-potentials.
            edge_potentials: (N, N) edge log-potentials.

        Returns:
            Bethe free energy (scalar).

            The Bethe counting numbers (d_i - 1) are applied to the node
            entropy terms, where d_i is the degree of node i in the graph
            implied by the nonzero pairwise beliefs.
        """
        node_beliefs = np.asarray(node_beliefs, dtype=np.float64)
        node_beliefs = node_beliefs / (node_beliefs.sum() + self._epsilon)

        # Node energy: U_i = -sum_s b_i(s) ln phi_i(s)
        node_energy = -np.sum(node_beliefs * node_potentials)

        # Node entropy: H_i = -sum_s b_i(s) ln b_i(s)
        node_entropy = -np.sum(node_beliefs * np.log(node_beliefs + self._epsilon))
        # Graph structure from the pairwise beliefs: edge (i, j) exists when
        # the pairwise belief entry is nonzero; d_i is the degree of node i.
        pairwise_beliefs = np.asarray(pairwise_beliefs, dtype=np.float64)
        edge_potentials = np.asarray(edge_potentials, dtype=np.float64)
        adjacency = np.abs(pairwise_beliefs) > self._epsilon
        adjacency = adjacency | adjacency.T
        np.fill_diagonal(adjacency, False)
        degrees = adjacency.sum(axis=1)
        n = len(node_beliefs)

        # Edge energy and entropy over unique undirected edges (i < j):
        # U_ij = -b_ij ln phi_ij and H_ij = -b_ij ln b_ij (elementwise).
        edge_energy = 0.0
        edge_entropy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    edge_energy += -pairwise_beliefs[i, j] * edge_potentials[i, j]
                    edge_entropy += -pairwise_beliefs[i, j] * np.log(
                        np.abs(pairwise_beliefs[i, j]) + self._epsilon
                    )

        # Bethe free energy with counting numbers:
        #   F_Bethe = sum_i [U_i - (d_i - 1) H_i]
        #             + sum_{(ij)} [U_ij - H_ij]
        bethe_node_contribution = node_energy - np.sum((degrees - 1) * node_entropy)
        F_bethe = bethe_node_contribution + edge_energy - edge_entropy
        logger.debug("Bethe free energy=%.4f", F_bethe)
        return float(F_bethe)

    def _kl_divergence(self, q: np.ndarray, p: np.ndarray) -> float:
        """KL divergence D_KL[q || p] = sum q * ln(q/p)."""
        q_safe = q + self._epsilon
        p_safe = p + self._epsilon
        return float(np.sum(q_safe * np.log(q_safe / p_safe)))
