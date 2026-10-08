"""Analytical witnesses for finite Markov currents and graph Hodge splits."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from geo_infer_math.core.circulation import (
    circulation_part,
    current_divergence,
    graph_hodge_decomposition,
    probability_current,
    satisfies_detailed_balance,
    skew_part,
)

TRIANGLE = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]])


def test_skew_identities_and_input_ownership():
    matrix = np.array([[3.0, 7, -2], [1, 4, 9], [6, -3, 5]])
    original = matrix.copy()
    skew = skew_part(matrix)
    symmetric = matrix * 0.5 + matrix.T * 0.5
    assert_array_equal(skew, -skew.T)
    assert np.trace(skew) == 0
    x = np.array([2.0, -1, 3])
    assert x @ skew @ x == 0
    assert_allclose(symmetric + skew, matrix)
    assert_array_equal(matrix, original)
    extreme = np.array([[0.0, 1e308], [-1e308, 0.0]])
    assert_array_equal(skew_part(extreme), extreme)


def test_nonuniform_detailed_balance_and_nonstationary_divergence():
    kernel = np.array([[0.75, 0.25], [0.5, 0.5]])
    pi = np.array([2 / 3, 1 / 3])
    assert satisfies_detailed_balance(kernel, pi)
    assert_allclose(probability_current(kernel, pi), 0, atol=1e-15)
    changed_pi = np.array([0.5, 0.5])
    current = probability_current(kernel, changed_pi)
    assert not satisfies_detailed_balance(kernel, changed_pi)
    assert_allclose(current_divergence(current), changed_pi @ kernel - changed_pi)
    assert_allclose(circulation_part(kernel, changed_pi) * 2, current)


def test_stationary_cycle_is_divergence_free_but_not_reversible():
    kernel = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
    pi = np.full(3, 1 / 3)
    current = probability_current(kernel, pi)
    assert_allclose(current_divergence(current), 0, atol=1e-15)
    assert not satisfies_detailed_balance(kernel, pi)
    assert_allclose(current, (kernel - kernel.T) / 3)
    # Dropping circulation loses a nonzero current despite zero divergence.
    assert np.linalg.norm(current) > 0.8
    permutation = [2, 0, 1]
    assert_allclose(
        probability_current(kernel[np.ix_(permutation, permutation)], pi[permutation]),
        current[np.ix_(permutation, permutation)],
    )


def test_rate_kernel_current_and_stationarity():
    rate = np.array([[-2.0, 2], [1, -1]])
    pi = np.array([1 / 3, 2 / 3])
    assert satisfies_detailed_balance(rate, pi, kind="rate")
    changed_pi = np.array([0.5, 0.5])
    current = probability_current(rate, changed_pi, kind="rate")
    assert_allclose(current_divergence(current), changed_pi @ rate)


def test_hodge_triangle_with_and_without_face():
    flow = np.array([2.0, -1, 5])
    original = flow.copy()
    closed = graph_hodge_decomposition(TRIANGLE, flow, np.ones((3, 1)))
    assert_allclose(closed.solenoidal, [2, 2, 2])
    assert_allclose(closed.gradient, [0, -3, 3], atol=1e-14)
    assert_allclose(closed.harmonic, 0, atol=1e-14)
    assert_allclose(closed.gradient + closed.solenoidal + closed.harmonic, flow)
    assert_allclose(TRIANGLE @ closed.solenoidal, 0, atol=1e-14)
    assert abs(closed.gradient @ closed.solenoidal) < 1e-13
    open_complex = graph_hodge_decomposition(TRIANGLE, flow)
    assert_allclose(open_complex.solenoidal, 0)
    assert_allclose(open_complex.harmonic, [2, 2, 2])
    assert_allclose(TRIANGLE @ open_complex.harmonic, 0, atol=1e-14)
    assert_array_equal(flow, original)


def test_disconnected_graph_gauge_and_edgeless_graph():
    incidence = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
    result = graph_hodge_decomposition(incidence, [2, -3])
    assert_allclose(result.gradient, [2, -3])
    assert_allclose(result.harmonic, 0, atol=1e-14)
    assert_allclose(result.node_potential, [-1, 1, 1.5, -1.5, 0])
    empty = graph_hodge_decomposition(np.empty((2, 0)), np.empty(0))
    assert empty.gradient.size == 0
    assert_array_equal(empty.node_potential, [0, 0])


@pytest.mark.parametrize(
    "matrix", [[], [[1, 2]], [[1, np.nan], [0, 1]], [[1j]], [[True]], [["1"]]]
)
def test_skew_rejects_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        skew_part(matrix)


@pytest.mark.parametrize(
    "kernel,pi,kind",
    [
        ([[1, 0], [0, 1]], [1], "transition"),
        ([[1, 0], [0, 1]], [0.5, -0.5], "transition"),
        ([[1, 0], [0, 1]], [1, 1], "transition"),
        ([[1, 0], [0, 1]], [np.nan, 0], "transition"),
        ([[1, -1], [0, 1]], [0.5, 0.5], "transition"),
        ([[0.5, 0], [0, 1]], [0.5, 0.5], "transition"),
        ([[1, 0], [0, 1]], [0.5, 0.5], "unknown"),
        ([[-1, -1], [1, -1]], [0.5, 0.5], "rate"),
        ([[-1, 2], [1, -1]], [0.5, 0.5], "rate"),
    ],
)
def test_kernel_contracts(kernel, pi, kind):
    with pytest.raises(ValueError):
        probability_current(kernel, pi, kind=kind)


@pytest.mark.parametrize("atol", [-1, np.inf, np.nan, True, "0", [0]])
def test_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        satisfies_detailed_balance(np.eye(2), [0.5, 0.5], atol=atol)


def test_invalid_current_and_graph_inputs():
    with pytest.raises(ValueError, match="skew-symmetric"):
        current_divergence(np.eye(2))
    for incidence, flow, faces in [
        ([[1], [0]], [1], None),
        (TRIANGLE, [1, 2], None),
        (TRIANGLE, [1, 2, np.inf], None),
        (TRIANGLE, [1, 2, 3], [[1], [0], [0]]),
        (TRIANGLE, [1, 2, 3], [[1], [1]]),
    ]:
        with pytest.raises(ValueError):
            graph_hodge_decomposition(incidence, flow, faces)
