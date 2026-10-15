"""Exact multiple-choice resource allocation with optional group expenditure floors."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    actions: tuple[int, ...]
    spent: int
    objective_gain: float
    group_spend: tuple[tuple[str, int], ...]


def _frontier(
    gains: NDArray[np.float64], units: tuple[int, ...], budget: int
) -> tuple[NDArray[np.float64], NDArray[np.int16]]:
    values = np.full(budget + 1, -np.inf)
    values[0] = 0.0
    choices = np.zeros((len(gains), budget + 1), dtype=np.int16)
    for index, row in enumerate(gains):
        following = values.copy()
        for action, resource in enumerate(units[1:], start=1):
            if resource > budget:
                continue
            candidate = values[: budget + 1 - resource] + row[action]
            target = following[resource:]
            improves = candidate > target
            target[improves] = candidate[improves]
            choices[index, resource:][improves] = action
        values = following
    return values, choices


def allocate_resources(
    policy_costs: NDArray[np.float64],
    units: tuple[int, ...],
    budget: int,
    *,
    groups: Sequence[str] | None = None,
    minimum_spend: Mapping[str, int] | None = None,
) -> ResourceAllocation:
    if (
        not isinstance(policy_costs, np.ndarray)
        or policy_costs.ndim != 2
        or policy_costs.dtype.kind not in "fiu"
        or not 1 <= len(policy_costs) <= 30000
        or not 1 <= policy_costs.shape[1] <= 32
        or not np.all(np.isfinite(policy_costs))
    ):
        raise ValueError(
            "Resource policy costs must be a bounded finite N by action array"
        )
    if (
        type(budget) is not int
        or not 0 <= budget <= 512
        or len(units) != policy_costs.shape[1]
        or not units
        or units[0] != 0
        or any(type(unit) is not int or not 0 <= unit <= 512 for unit in units)
    ):
        raise ValueError(
            "Resource units must match actions with action zero costing zero"
        )
    labels = tuple(groups) if groups is not None else ("all",) * len(policy_costs)
    if (
        isinstance(groups, str)
        or len(labels) != len(policy_costs)
        or any(not isinstance(group, str) or not group for group in labels)
    ):
        raise ValueError(
            "Resource groups must be nonempty labels on the policy carrier"
        )
    identifiers = tuple(sorted(set(labels)))
    floors = dict(minimum_spend or {})
    if (
        len(identifiers) > 32
        or not set(floors) <= set(identifiers)
        or any(
            type(value) is not int or not 0 <= value <= budget
            for value in floors.values()
        )
        or sum(floors.values()) > budget
    ):
        raise ValueError(
            "Resource group floors are invalid or exceed the shared budget"
        )
    if (
        len(policy_costs) * (budget + 1) * len(units)
        + len(identifiers) * (budget + 1) ** 2
        > 50_000_000
    ):
        raise ValueError("Resource allocation exceeds its bounded work contract")
    with np.errstate(over="raise", invalid="raise"):
        try:
            gains = policy_costs[:, :1] - policy_costs
        except FloatingPointError as exc:
            raise ValueError("Resource gains exceed finite numeric range") from exc
    frontiers, choices, indices = [], [], []
    for group in identifiers:
        members = [index for index, label in enumerate(labels) if label == group]
        with np.errstate(over="raise", invalid="raise"):
            try:
                values, decisions = _frontier(gains[members], units, budget)
            except FloatingPointError as exc:
                raise ValueError(
                    "Resource frontier exceeds finite numeric range"
                ) from exc
        values[: floors.get(group, 0)] = -np.inf
        frontiers.append(values)
        choices.append(decisions)
        indices.append(members)
    combined = np.full(budget + 1, -np.inf)
    combined[0] = 0.0
    group_choices = np.full((len(identifiers), budget + 1), -1, dtype=np.int16)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for group_index, frontier in enumerate(frontiers):
                following = np.full(budget + 1, -np.inf)
                for group_spend in np.flatnonzero(np.isfinite(frontier)):
                    candidate = (
                        combined[: budget + 1 - group_spend] + frontier[group_spend]
                    )
                    target = following[group_spend:]
                    improves = candidate > target
                    target[improves] = candidate[improves]
                    group_choices[group_index, group_spend:][improves] = group_spend
                combined = following
        except FloatingPointError as exc:
            raise ValueError(
                "Resource allocation exceeds finite numeric range"
            ) from exc
    if not np.any(np.isfinite(combined)):
        raise ValueError("Resource floors have no feasible action assignment")
    spent = int(np.argmax(combined))
    remainder = spent
    allocations = [0] * len(identifiers)
    for index in range(len(identifiers) - 1, -1, -1):
        allocations[index] = int(group_choices[index, remainder])
        remainder -= allocations[index]
    actions = [0] * len(policy_costs)
    for members, decisions, assigned in zip(indices, choices, allocations, strict=True):
        remaining = assigned
        for member_index in range(len(members) - 1, -1, -1):
            action = int(decisions[member_index, remaining])
            actions[members[member_index]] = action
            remaining -= units[action]
        if remaining:
            raise ValueError("Resource action backtracking failed conservation")
    if remainder or sum(units[action] for action in actions) != spent:
        raise ValueError("Resource group backtracking failed conservation")
    return ResourceAllocation(
        tuple(actions),
        spent,
        float(combined[spent]),
        tuple(zip(identifiers, allocations, strict=True)),
    )
