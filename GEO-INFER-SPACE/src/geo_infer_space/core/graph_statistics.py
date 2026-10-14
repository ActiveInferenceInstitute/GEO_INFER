"""Sparse graph randomization statistics and area-conserving group summaries."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MoranResult:
    statistic: float
    randomization_expectation: float
    two_sided_permutation_p: float
    permutations: int
    seed: int
    observations: int
    undirected_edges: int
    isolated_nodes: int


def graph_moran(
    values: NDArray[np.float64],
    edges: tuple[tuple[int, int], ...],
    *,
    permutations: int = 199,
    seed: int = 0,
) -> MoranResult:
    if (
        values.ndim != 1
        or not 3 <= len(values) <= 30000
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Moran values must be a finite vector with 3..30000 observations"
        )
    count = len(values)
    if (
        not edges
        or len(set(edges)) != len(edges)
        or any(
            type(i) is not int or type(j) is not int or not 0 <= i < j < count
            for i, j in edges
        )
    ):
        raise ValueError("Moran graph needs distinct ordered undirected edges")
    if (
        type(permutations) is not int
        or not 19 <= permutations <= 9999
        or permutations * count > 2_000_000
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("Moran randomization exceeds its seed or work budget")
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            "Moran statistic is undefined for constant or overflowing values"
        )
    left, right = np.asarray(edges, dtype=int).T
    scale = count / len(edges) / denominator

    def statistic(vector: NDArray[np.float64]) -> float:
        value = float(scale * np.sum(vector[left] * vector[right]))
        if not np.isfinite(value):
            raise ValueError("Moran statistic exceeds finite numeric range")
        return value

    observed, expectation = statistic(centered), -1 / (count - 1)
    distance = abs(observed - expectation)
    generator = np.random.default_rng(seed)
    exceedances = sum(
        abs(statistic(generator.permutation(centered)) - expectation) >= distance
        for _ in range(permutations)
    )
    return MoranResult(
        statistic=observed,
        randomization_expectation=expectation,
        two_sided_permutation_p=(1 + exceedances) / (1 + permutations),
        permutations=permutations,
        seed=seed,
        observations=count,
        undirected_edges=len(edges),
        isolated_nodes=count - len(set(left.tolist() + right.tolist())),
    )


@dataclass(frozen=True, slots=True)
class WeightedGroup:
    identifier: str
    allocated_area: float
    weighted_mean: float
    memberships: int


def weighted_group_means(
    values: NDArray[np.float64], groups: Sequence[str], areas: NDArray[np.float64]
) -> tuple[WeightedGroup, ...]:
    if (
        values.ndim != 1
        or values.shape != areas.shape
        or len(groups) != len(values)
        or not 1 <= len(values) <= 1_000_000
    ):
        raise ValueError("Group arrays must share a nonempty one-dimensional carrier")
    if (
        any(not isinstance(group, str) or not group for group in groups)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(areas))
        or np.any(areas <= 0)
    ):
        raise ValueError("Group values and positive allocation areas must be finite")
    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        buckets.setdefault(group, []).append(index)
    summaries = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for group, indices in sorted(buckets.items()):
                weights = areas[indices]
                total = float(weights.sum())
                mean = float(np.sum(values[indices] * (weights / total)))
                if not np.isfinite(total) or not np.isfinite(mean):
                    raise ValueError("Group summary exceeds finite numeric range")
                summaries.append(WeightedGroup(group, total, mean, len(indices)))
        except FloatingPointError as exc:
            raise ValueError("Group summary exceeds finite numeric range") from exc
    return tuple(summaries)
