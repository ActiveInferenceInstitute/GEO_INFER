"""Unit tests for the ACT free-energy calculator fixes."""

import numpy as np
import pytest


from geo_infer_math.integration.act.free_energy import FreeEnergyCalculator


class TestFreeEnergyFallback:
    """Free-energy fallback without a likelihood must be complexity - 0."""

    def setup_method(self):
        self.calc = FreeEnergyCalculator()

    def test_fallback_accuracy_is_zero(self):
        obs = np.array([0.8, 0.1, 0.1])
        beliefs = np.array([0.5, 0.3, 0.2])

        result = self.calc.calculate(obs, beliefs)

        assert result["accuracy"] == 0.0
        assert result["free_energy"] == pytest.approx(result["complexity"])

    def test_fallback_equals_kl_against_uniform_prior(self):
        obs = np.ones(4) / 4
        beliefs = np.array([0.7, 0.1, 0.1, 0.1])
        prior = np.ones(4) / 4

        result = self.calc.calculate(obs, beliefs, prior=prior)

        expected_kl = np.sum(beliefs * np.log(beliefs * 4))
        assert result["complexity"] == pytest.approx(expected_kl)
        assert result["free_energy"] == pytest.approx(expected_kl)


class TestBetheCountingNumbers:
    """Bethe free energy applies the (d_i - 1) node-entropy counting numbers."""

    def setup_method(self):
        self.calc = FreeEnergyCalculator()

    def test_two_node_chain_counting_number_zero(self):
        # Each node has degree 1 -> (d_i - 1) = 0, so the node entropy
        # drops out entirely.
        beliefs = np.array([0.5, 0.5])
        pairwise = np.array([[0.0, 0.25], [0.0, 0.0]])
        node_potentials = np.array([0.3, -0.2])
        edge_potentials = np.array([[0.0, 0.4], [0.0, 0.0]])

        u_nodes = -(0.5 * 0.3 + 0.5 * (-0.2))
        u_edge = -0.25 * 0.4
        h_edge = -0.25 * np.log(0.25)
        expected = u_nodes + u_edge - h_edge

        result = self.calc.bethe_free_energy(
            beliefs, pairwise, node_potentials, edge_potentials
        )
        assert result == pytest.approx(expected)

    def test_triangle_graph_counting_numbers(self):
        # Fully connected 3-node graph: every d_i = 2, so the counting
        # number is 1 and each node entropy enters once with a minus sign.
        beliefs = np.ones(3) / 3
        pairwise = 0.1 * np.ones((3, 3))
        np.fill_diagonal(pairwise, 0.0)
        node_potentials = np.array([0.1, 0.2, 0.3])
        edge_potentials = 0.05 * np.ones((3, 3))

        u_nodes = float(-np.sum(beliefs * node_potentials))
        h_node = float(-np.sum(beliefs * np.log(beliefs)))
        # counting number (d_i - 1) = 1 per node
        expected_nodes = u_nodes - 3.0 * 1.0 * h_node

        edge_energy = 0.0
        edge_entropy = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                edge_energy += -pairwise[i, j] * edge_potentials[i, j]
                edge_entropy += -pairwise[i, j] * np.log(pairwise[i, j])
        expected = expected_nodes + edge_energy - edge_entropy

        result = self.calc.bethe_free_energy(
            beliefs, pairwise, node_potentials, edge_potentials
        )
        assert result == pytest.approx(expected)
