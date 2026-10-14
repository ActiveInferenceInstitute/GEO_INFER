from __future__ import annotations

import numpy as np
import pytest

from geo_infer_space.core.graph_statistics import graph_moran, weighted_group_means


def test_moran_matches_hand_computation_and_replays_randomization() -> None:
    """Reproduce hand-computed Moran I and seeded permutation probabilities."""
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    edges = ((0, 1), (1, 2), (2, 3))
    result = graph_moran(values, edges, permutations=199, seed=7)
    assert result.statistic == pytest.approx(0.4)
    assert result.randomization_expectation == pytest.approx(-1 / 3)
    assert 1 / 200 <= result.two_sided_permutation_p <= 1
    assert result.isolated_nodes == 0
    assert result == graph_moran(values, edges, permutations=199, seed=7)
    shifted_scaled = graph_moran(3 * values + 11, edges, permutations=199, seed=7)
    assert shifted_scaled.statistic == pytest.approx(result.statistic)
    assert shifted_scaled.two_sided_permutation_p == result.two_sided_permutation_p
    isolated = graph_moran(values, ((0, 1),), permutations=19, seed=1)
    assert isolated.isolated_nodes == 2


@pytest.mark.parametrize(
    "condition",
    [
        "constant",
        "nan",
        "shape",
        "empty_edges",
        "reversed",
        "duplicate",
        "permutations",
        "seed",
        "budget",
    ],
)
def test_moran_rejects_undefined_or_unbounded_inputs(condition: str) -> None:
    """Reject undefined statistics, invalid graph edges and unbounded requests."""
    values, edges, permutations, seed = (
        np.array([0.0, 1.0, 2.0]),
        ((0, 1), (1, 2)),
        199,
        0,
    )
    if condition == "constant":
        values[:] = 1
    elif condition == "nan":
        values[0] = np.nan
    elif condition == "shape":
        values = np.zeros((3, 2))
    elif condition == "empty_edges":
        edges = ()
    elif condition == "reversed":
        edges = ((1, 0),)
    elif condition == "duplicate":
        edges = ((0, 1), (0, 1))
    elif condition == "permutations":
        permutations = True
    elif condition == "seed":
        seed = -1
    else:
        values, permutations = np.arange(30000, dtype=float), 199
    with pytest.raises(ValueError):
        graph_moran(values, edges, permutations=permutations, seed=seed)


def test_area_weighted_groups_conserve_mass_and_weighted_value() -> None:
    """Conserve allocated area and weighted measurements across group summaries."""
    values = np.array([0.1, 0.8, 0.4, 0.4])
    areas = np.array([3.0, 1.0, 2.0, 4.0])
    result = weighted_group_means(values, ("a", "a", "a", "b"), areas)
    assert tuple(item.identifier for item in result) == ("a", "b")
    assert result[0].weighted_mean == pytest.approx((0.3 + 0.8 + 0.8) / 6)
    assert result[0].memberships == 3 and result[1].memberships == 1
    assert sum(item.allocated_area for item in result) == pytest.approx(areas.sum())
    assert sum(
        item.allocated_area * item.weighted_mean for item in result
    ) == pytest.approx(values @ areas)


@pytest.mark.parametrize(
    "condition",
    ["shape", "empty", "label", "length", "nan", "zero", "negative", "overflow"],
)
def test_group_summaries_refuse_invalid_measurements(condition: str) -> None:
    """Reject nonfinite values, invalid areas and incompatible group carriers."""
    values, groups, areas = np.array([0.2, 0.4]), ("a", "b"), np.array([1.0, 2.0])
    if condition == "shape":
        values = np.zeros((2, 2))
    elif condition == "empty":
        values = areas = np.array([])
        groups = ()
    elif condition == "label":
        groups = ("", "b")
    elif condition == "length":
        groups = ("a",)
    elif condition == "nan":
        values[0] = np.nan
    elif condition == "zero":
        areas[0] = 0
    elif condition == "negative":
        areas[0] = -1
    else:
        groups, areas = ("a", "a"), np.array([1e308, 1e308])
    with pytest.raises(ValueError):
        weighted_group_means(values, groups, areas)
