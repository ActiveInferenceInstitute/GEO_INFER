from __future__ import annotations

import itertools

import numpy as np
import pytest

from geo_infer_act.core.resource_allocation import allocate_resources


@pytest.mark.parametrize("budget,floor", [(0, 0), (2, 0), (4, 1), (6, 2), (9, 3)])
def test_group_floor_solution_matches_brute_force(budget: int, floor: int) -> None:
    """Compare constrained allocation with every feasible small action vector."""
    costs = np.random.default_rng(31).normal(size=(4, 3))
    groups, units = ("a", "a", "b", "b"), (0, 1, 3)
    result = allocate_resources(
        costs, units, budget, groups=groups, minimum_spend={"a": floor, "b": floor}
    )
    feasible = [
        actions
        for actions in itertools.product(range(3), repeat=4)
        if sum(units[action] for action in actions) <= budget
        and all(
            sum(
                units[action]
                for action, group in zip(actions, groups, strict=True)
                if group == identifier
            )
            >= floor
            for identifier in ("a", "b")
        )
    ]
    expected = max(
        sum(
            costs[index, 0] - costs[index, action]
            for index, action in enumerate(actions)
        )
        for actions in feasible
    )
    assert result.objective_gain == pytest.approx(expected, abs=1e-12)
    assert sum(units[action] for action in result.actions) == result.spent <= budget
    assert dict(result.group_spend) == {
        identifier: sum(
            units[action]
            for action, group in zip(result.actions, groups, strict=True)
            if group == identifier
        )
        for identifier in ("a", "b")
    }
    assert all(spend >= floor for _, spend in result.group_spend)


def test_floor_does_not_invent_feasibility_and_free_actions_are_real() -> None:
    """Reject infeasible floors and retain beneficial actions with zero unit cost."""
    with pytest.raises(ValueError, match="feasible"):
        allocate_resources(np.array([[0.0, -1.0]]), (0, 3), 2, minimum_spend={"all": 1})
    result = allocate_resources(np.array([[0.0, -1.0], [0.0, -2.0]]), (0, 0), 0)
    assert result.actions == (1, 1) and result.spent == 0 and result.objective_gain == 3
    passive = allocate_resources(np.array([[0.0, 1.0]]), (0, 1), 3)
    assert passive.actions == (0,) and passive.spent == 0


@pytest.mark.parametrize(
    "condition",
    [
        "shape",
        "nan",
        "empty",
        "budget",
        "units",
        "passive",
        "groups",
        "floor_group",
        "floor_bool",
        "floor_total",
        "group_count",
        "work",
        "overflow",
    ],
)
def test_allocator_rejects_invalid_configuration(condition: str) -> None:
    """Reject malformed costs, units, groups and excessive allocation work."""
    costs, units, budget, groups, floors = (
        np.zeros((2, 3)),
        (0, 1, 3),
        4,
        ("a", "b"),
        {},
    )
    if condition == "shape":
        costs = np.zeros((2, 2))
    elif condition == "nan":
        costs[0, 0] = np.nan
    elif condition == "empty":
        costs = np.zeros((0, 3))
    elif condition == "budget":
        budget = True
    elif condition == "units":
        units = (0, True, 3)
    elif condition == "passive":
        units = (1, 2, 3)
    elif condition == "groups":
        groups = ("a",)
    elif condition == "floor_group":
        floors = {"unknown": 1}
    elif condition == "floor_bool":
        floors = {"a": True}
    elif condition == "floor_total":
        floors = {"a": 3, "b": 3}
    elif condition == "group_count":
        costs, groups = np.zeros((33, 3)), tuple(map(str, range(33)))
    elif condition == "work":
        costs, budget, groups = np.zeros((30000, 32)), 512, ("a",) * 30000
        units = tuple(range(32))
    else:
        costs[0, :2] = [1e308, -1e308]
    with pytest.raises(ValueError):
        allocate_resources(costs, units, budget, groups=groups, minimum_spend=floors)
