"""H3 mean-field inference and exact budgeted allocation of real GNN policy costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, xlogy

from geo_infer_act.core.resource_allocation import allocate_resources
from geo_infer_place.locations.california.contracts import CaliforniaError

FloatVector = NDArray[np.float64]
COSTS: Final = (0, 1, 3)


@dataclass(frozen=True, slots=True)
class FieldResult:
    posterior: FloatVector
    iterations: int
    residual: float
    variational_free_energy: float


def infer_field(
    prior: FloatVector,
    log_likelihood: NDArray[np.float64],
    edges: tuple[tuple[int, int], ...],
    coupling: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 128,
) -> FieldResult:
    """Solve a bounded binary mean-field fixed point, retaining its residual and objective."""
    if prior.ndim != 1:
        raise CaliforniaError("field arrays have incompatible dimensions")
    count = len(prior)
    if not count or log_likelihood.shape != (count, 2):
        raise CaliforniaError("field arrays have incompatible dimensions")
    if (
        not np.all(np.isfinite(prior))
        or np.any(prior <= 0)
        or np.any(prior >= 1)
        or not np.all(np.isfinite(log_likelihood))
    ):
        raise CaliforniaError("field priors need full support and finite likelihoods")
    if (
        type(coupling) not in (float, int)
        or not 0 <= coupling <= 1
        or not 0 < tolerance < 1
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise CaliforniaError("invalid fixed-point coupling or solver budget")
    if len(set(edges)) != len(edges) or any(not 0 <= i < j < count for i, j in edges):
        raise CaliforniaError("edges must be unique ordered carrier indices")
    degree = np.zeros(count, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    if np.any(degree > 6):
        raise CaliforniaError("field graph exceeds the six-neighbor H3 contract")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            unary = (
                np.log(prior)
                - np.log1p(-prior)
                + log_likelihood[:, 1]
                - log_likelihood[:, 0]
            )
        except FloatingPointError as exc:
            raise CaliforniaError(
                "field log odds exceed finite numerical range"
            ) from exc

    def fixed_point(q: FloatVector) -> FloatVector:
        interaction = np.zeros(count)
        for i, j in edges:
            interaction[i] += (2 * q[j] - 1) / 6
            interaction[j] += (2 * q[i] - 1) / 6
        return np.asarray(expit(unary + coupling * interaction), dtype=np.float64)

    q = prior.copy()
    residual = float("inf")
    for _iteration in range(1, max_iterations + 1):
        following = fixed_point(q)
        q = 0.5 * q + 0.5 * following
        residual = float(np.max(np.abs(fixed_point(q) - q)))
        if residual <= tolerance:
            break
    else:
        raise CaliforniaError("mean-field solver exhausted its iteration budget")
    if np.any(q <= 0) or np.any(q >= 1):
        raise CaliforniaError("field posterior lost floating-point full support")
    kl = xlogy(q, q) - xlogy(q, prior) + xlogy(1 - q, 1 - q) - xlogy(1 - q, 1 - prior)
    energy = float(
        np.sum(kl - ((1 - q) * log_likelihood[:, 0] + q * log_likelihood[:, 1]))
    )
    energy -= coupling * sum(
        (q[i] * q[j] + (1 - q[i]) * (1 - q[j])) / 6 for i, j in edges
    )
    if not np.isfinite(energy) or not np.all(np.isfinite(q)):
        raise CaliforniaError("mean-field objective exceeds finite numerical range")
    q.flags.writeable = False
    return FieldResult(q, _iteration, residual, energy)


@dataclass(frozen=True, slots=True)
class Allocation:
    modes: tuple[int, ...]
    spent: int
    objective_gain: float


def allocate_budget(costs: NDArray[np.float64], budget: int) -> Allocation:
    """Optimize a multiple-choice knapsack over passive/coarse/detailed GNN policies."""
    if (
        costs.ndim != 2
        or costs.shape[1] != 3
        or not 1 <= len(costs) <= 30000
        or not np.all(np.isfinite(costs))
    ):
        raise CaliforniaError("policy costs must be a finite N by three array")
    if type(budget) is not int or not 0 <= budget <= 512:
        raise CaliforniaError("budget must be an integer in 0..512")
    try:
        result = allocate_resources(costs, COSTS, budget)
    except ValueError as exc:
        raise CaliforniaError(str(exc)) from exc
    return Allocation(result.actions, result.spent, result.objective_gain)
