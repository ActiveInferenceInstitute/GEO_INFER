from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import expit

from geo_infer_place.locations.california.contracts import CaliforniaError
from geo_infer_place.locations.california.inference import (
    COSTS,
    allocate_budget,
    infer_field,
)


def test_uncoupled_field_matches_independent_bayes() -> None:
    """Match independently calculated binary Bayes when coupling is zero."""
    prior = np.array([0.13, 0.8, 0.45])
    likelihood = np.log(np.array([[0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]))
    result = infer_field(prior, likelihood, ((0, 1), (1, 2)), 0.0)
    unnormalized = np.column_stack((1 - prior, prior)) * np.exp(likelihood)
    expected = unnormalized[:, 1] / unnormalized.sum(axis=1)
    np.testing.assert_allclose(result.posterior, expected, atol=2e-10, rtol=0)
    assert result.residual <= 1e-10
    assert not result.posterior.flags.writeable


def test_coupled_field_against_enumerated_two_site_distribution() -> None:
    """Check mean-field approximation against an enumerated two-site distribution."""
    prior = np.array([0.2, 0.7])
    beta = 0.7
    result = infer_field(prior, np.zeros((2, 2)), ((0, 1),), beta)
    states = np.asarray(list(itertools.product((0, 1), repeat=2)))
    probabilities = np.prod(np.where(states, prior, 1 - prior), axis=1) * np.exp(
        beta * (states[:, 0] == states[:, 1]) / 6
    )
    exact = probabilities @ states / probabilities.sum()
    assert np.max(np.abs(result.posterior - exact)) < 0.002
    fixed = expit(
        np.log(prior / (1 - prior)) + beta * (2 * result.posterior[::-1] - 1) / 6
    )
    np.testing.assert_allclose(result.posterior, fixed, atol=1e-10, rtol=0)
    assert not np.allclose(result.posterior, prior)


@pytest.mark.parametrize(
    "condition",
    [
        "scalar",
        "empty",
        "shape",
        "zero",
        "nan",
        "edge",
        "duplicate",
        "degree",
        "coupling",
        "exhausted",
    ],
)
def test_field_rejects_invalid_or_exhausted_inputs(condition: str) -> None:
    """Reject malformed graph fields and fail when convergence work is exhausted."""
    prior, likelihood, edges, beta, iterations = (
        np.full(8, 0.2),
        np.zeros((8, 2)),
        (),
        0.7,
        128,
    )
    if condition == "scalar":
        prior = np.array(0.2)
    elif condition == "empty":
        prior = np.array([])
    elif condition == "shape":
        likelihood = np.zeros((8, 3))
    elif condition == "zero":
        prior[0] = 0
    elif condition == "nan":
        likelihood[0, 0] = np.nan
    elif condition == "edge":
        edges = ((1, 0),)
    elif condition == "duplicate":
        edges = ((0, 1), (0, 1))
    elif condition == "degree":
        edges = tuple((0, index) for index in range(1, 8))
    elif condition == "coupling":
        beta = 1.1
    else:
        likelihood[:, 1], iterations = 1.0, 1
    with pytest.raises(CaliforniaError):
        infer_field(prior, likelihood, edges, beta, max_iterations=iterations)


@pytest.mark.parametrize("budget", [0, 1, 3, 5, 12])
def test_knapsack_matches_exhaustive_optimum(budget: int) -> None:
    """Compare the resource optimum with exhaustive feasible action enumeration."""
    costs = np.random.default_rng(9).normal(size=(4, 3))
    result = allocate_budget(costs, budget)
    feasible = [
        modes
        for modes in itertools.product(range(3), repeat=4)
        if sum(COSTS[mode] for mode in modes) <= budget
    ]
    optimum = max(
        sum(costs[index, 0] - costs[index, mode] for index, mode in enumerate(modes))
        for modes in feasible
    )
    assert result.objective_gain == pytest.approx(optimum, abs=1e-12)
    assert sum(COSTS[mode] for mode in result.modes) == result.spent <= budget


@pytest.mark.parametrize(
    "costs,budget",
    [
        (np.zeros((2, 2)), 1),
        (np.array([[0, np.nan, 1]]), 1),
        (np.zeros((0, 3)), 1),
        (np.zeros((2, 3)), True),
        (np.zeros((2, 3)), 513),
    ],
)
def test_budget_rejects_invalid_inputs(costs: np.ndarray, budget: int) -> None:
    """Reject invalid policy costs and noninteger or out-of-range budgets."""
    with pytest.raises(CaliforniaError):
        allocate_budget(costs, budget)


def test_zero_and_negative_information_gain_spends_nothing() -> None:
    """Retain passive actions when expenditure cannot improve the objective."""
    result = allocate_budget(np.array([[0.0, 0.1, 0.2], [0, 0, 0]]), 6)
    assert result.modes == (0, 0) and result.spent == 0 and result.objective_gain == 0
