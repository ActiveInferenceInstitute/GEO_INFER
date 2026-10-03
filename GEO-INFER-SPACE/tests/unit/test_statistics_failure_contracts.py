"""Tiny analytical and error oracles for retained SPACE statistics contracts."""

import math
from types import SimpleNamespace

import h3
import pytest

from geo_infer_space.core.statistics import SpatialStatistics

CELL = h3.latlng_to_cell(37.7, -122.4, 8)


def test_getis_star_matches_independent_binary_weight_formula_and_scale():
    cells = list(h3.grid_disk(CELL, 1))
    values = [0, 1, 3, 6, 10, 15, 21]
    mean = sum(values) / 7
    variance = sum((value - mean) ** 2 for value in values) / 7
    expected = {}
    for i, cell in enumerate(cells):
        indices = [
            j
            for j, other in enumerate(cells)
            if i == j or h3.are_neighbor_cells(cell, other)
        ]
        w = len(indices)
        expected[cell] = (
            None
            if w == 7
            else (sum(values[j] for j in indices) - mean * w)
            / math.sqrt(variance * w * (7 - w) / 6)
        )
    stats = SpatialStatistics("h3")
    for scale in (1, 1e-100, 1e100):
        result = stats.getis_ord_g(cells, [value * scale for value in values])
        assert result["g_stars"][CELL] is None
        assert result["undefined"][CELL] == "neighborhood_covers_entire_domain"
        for cell, score in expected.items():
            if score is not None:
                assert result["g_stars"][cell] == pytest.approx(score)


@pytest.mark.parametrize(
    "values,reason",
    [
        ([3], "insufficient_observations"),
        ([0, 0], "zero_global_variance"),
        ([2, 2], "zero_global_variance"),
    ],
)
def test_getis_undefined_is_never_a_neutral_score(values, reason):
    cells = list(h3.grid_disk(CELL, 1))[: len(values)]
    result = SpatialStatistics("h3").getis_ord_g(cells, values)
    assert result["g_stars"] == dict.fromkeys(cells)
    assert result["undefined"] == dict.fromkeys(cells, reason)
    assert result["hotspots"] == result["coldspots"] == []


@pytest.mark.parametrize(
    "values,reason",
    [
        ([], "insufficient_observations"),
        ([3], "insufficient_observations"),
        ([0, 0], "zero_mean"),
    ],
)
def test_undefined_vmr_has_no_significance_or_pattern(values, reason):
    result = SpatialStatistics().variance_mean_ratio(values)
    assert (
        result["vmr"]
        is result["p_value"]
        is result["chi_square"]
        is result["pattern"]
        is None
    )
    assert result["undefined"] == reason and not result["reference_tested"]


def test_vmr_independent_analytical_count_reference_and_intensity_boundary():
    # Sample variance of [0, 2, 4] is 4, mean is 2, VMR=2 and chi-square=4.
    # Chi-square(2) survival is exp(-x/2); two-sided p=2*exp(-2).
    stats = SpatialStatistics()
    result = stats.variance_mean_ratio([0, 2, 4])
    assert result["variance"] == 4 and result["mean"] == 2 and result["vmr"] == 2
    assert result["chi_square"] == 4 and result["df"] == 2
    assert result["p_value"] == pytest.approx(2 * math.exp(-2))
    assert result["reference_tested"]
    descriptive = stats.variance_mean_ratio([0, 0.5, 1])
    assert descriptive["vmr"] == 0.5 and not descriptive["reference_tested"]
    assert descriptive["p_value"] is descriptive["chi_square"] is None
    equal_moments = stats.variance_mean_ratio([0.5, 0.5, 0.5, 2.5])
    assert equal_moments["vmr"] == 1
    assert equal_moments["pattern"] == "variance equals mean"


def test_quadrat_preserves_exact_parents_and_total_with_permutations():
    children = h3.cell_to_children(h3.cell_to_parent(CELL, 7), 8)
    other_parent = next(
        c
        for c in h3.grid_disk(h3.cell_to_parent(CELL, 7), 1)
        if c != h3.cell_to_parent(CELL, 7)
    )
    other_child = h3.cell_to_children(other_parent, 8)[0]
    cells = [children[2], other_child, children[0]]
    values = [0, 3, 2]
    expected = {h3.cell_to_parent(CELL, 7): 2, other_parent: 3}
    stats = SpatialStatistics("h3")
    for order in ([0, 1, 2], [2, 0, 1]):
        result = stats.quadrat_count(
            [cells[i] for i in order], [values[i] for i in order], quadrat_size=1
        )
        assert result["quadrat_counts"] == expected and result["total_count"] == 5
        assert result["scope"] == "observed_parent_quadrats"
    retained = stats.quadrat_count(cells, values, quadrat_size=0)
    assert retained["quadrat_counts"] == dict(zip(cells, values))
    one = stats.quadrat_count([CELL], [0], quadrat_size=0)
    assert one["total_count"] == 0 and one["vmr"] is None and one["pattern"] is None


@pytest.mark.parametrize(
    "cells,values,steps",
    [
        ([CELL, "bad"], [3], 1),
        ([CELL], [], 1),
        ([CELL, CELL], [1, 2], 1),
        ([CELL, h3.cell_to_parent(CELL, 7)], [1, 2], 1),
        ([CELL], [math.nan], 1),
        ([CELL], [math.inf], 1),
        ([CELL], [-1], 1),
        ([CELL], [True], 1),
        ([CELL], [1], True),
        ([CELL], [1], -0.5),
        ([CELL], [1], 1.5),
        ([CELL], [1], 9),
        ([], [], 0),
    ],
)
def test_quadrat_rejects_complete_invalid_input_before_backend(cells, values, steps):
    stats = SpatialStatistics()

    def forbidden(*args):
        raise AssertionError("Invalid input reached backend resolution")

    stats._dispatcher = SimpleNamespace(_resolve_backend_name=forbidden)
    with pytest.raises(ValueError):
        stats.quadrat_count(cells, values, steps)


@pytest.mark.parametrize("method", ["getis", "quadrat"])
def test_topology_failure_propagates_without_partial_statistics(method):
    stats = SpatialStatistics("h3")
    calls = []

    class Broken:
        def get_cell_neighbors(self, cell, k=1):
            raise RuntimeError("native neighborhood failure")

        def get_cell_parent(self, cell, resolution):
            calls.append(cell)
            if len(calls) == 2:
                raise RuntimeError("second native parent failure")
            return h3.cell_to_parent(cell, resolution)

    stats._dispatcher = SimpleNamespace(
        _resolve_backend_name=lambda *args: "h3", get_backend=lambda name: Broken()
    )
    cells = list(h3.grid_disk(CELL, 1))[:3]
    with pytest.raises(RuntimeError, match="native.*failure"):
        if method == "getis":
            stats.getis_ord_g(cells, [0, 1, 3])
        else:
            stats.quadrat_count(cells, [0, 1, 3], 1)


def test_quadrat_rejects_backend_fabricated_parent_identity():
    stats = SpatialStatistics("h3")
    stats._dispatcher = SimpleNamespace(
        _resolve_backend_name=lambda *args: "h3",
        get_backend=lambda name: SimpleNamespace(
            get_cell_parent=lambda *args: "invented"
        ),
    )
    with pytest.raises(ValueError, match="invalid H3 parent"):
        stats.quadrat_count([CELL], [1], 1)
