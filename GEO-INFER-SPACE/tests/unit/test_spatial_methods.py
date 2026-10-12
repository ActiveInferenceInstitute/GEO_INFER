"""Real H3 composition oracles, including pentagons and allocation boundaries."""

import math

import h3
import numpy as np
import pytest

from geo_infer_space.core.spatial_methods import SpatialMethods


@pytest.fixture
def spatial_methods():
    """Use native H3, never fabricated host topology."""
    return SpatialMethods()


@pytest.fixture(params=[h3.latlng_to_cell(37.7, -122.4, 8), h3.get_pentagons(8)[0]])
def center(request):
    return request.param


def test_buffer_union_has_disjoint_minimum_rings(spatial_methods, center):
    sources = list(h3.grid_disk(center, 1))[:2]
    result = spatial_methods.buffer_analysis(sources, 2)
    expected = set().union(*(set(h3.grid_disk(cell, 2)) for cell in sources))
    assert set(result["all_cells"]) == expected
    assert not set(result["rings"][1]) & set(result["rings"][2])
    assert not set(sources) & set(result["rings"][1])
    without = spatial_methods.buffer_analysis(sources, 2, include_center=False)
    assert set(without["all_cells"]) == expected - set(sources)
    with pytest.raises(ValueError, match="allocation"):
        spatial_methods.buffer_analysis(sources, 2, max_cells=1)


def test_overlay_exact_sets(spatial_methods, center):
    cells = list(h3.grid_disk(center, 1))[:3]
    a, b = cells[:2], cells[1:]
    for operation, expected in [
        ("intersection", set(a) & set(b)),
        ("union", set(a) | set(b)),
        ("difference", set(a) - set(b)),
        ("symmetric_difference", set(a) ^ set(b)),
    ]:
        result = spatial_methods.overlay_cells(a + a, b, operation)
        assert set(result["result_cells"]) == expected
        assert result["overlap_ratio"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        spatial_methods.overlay_cells([], [], "invalid")


def test_filters_retain_input_identity_and_order(spatial_methods, center):
    cells = list(h3.grid_disk(center, 1))[:3]
    original = cells.copy()
    for kwargs, expected in [
        ({"threshold": 15}, cells[1:]),
        ({"filter_type": "percentile", "percentile": 50}, cells[1:]),
        ({"filter_type": "top_n", "top_n": 1}, cells[1:2]),
    ]:
        result = spatial_methods.spatial_filter(cells, [10, 30, 20], **kwargs)
        assert result["filtered_cells"] == expected
    assert cells == original
    assert (
        spatial_methods.spatial_filter(cells, [1, 1, 1], filter_type="outliers")[
            "filtered_cells"
        ]
        == []
    )


def test_complete_children_transfer_mass_and_intensity(spatial_methods, center):
    children = h3.cell_to_children(center, 9)
    assert len(children) == (6 if h3.is_pentagon(center) else 7)
    result = spatial_methods.disaggregate_to_cells([center], [42.0], 9)
    assert set(result["disaggregated"]) == set(children)
    assert math.fsum(result["disaggregated"].values()) == pytest.approx(42)
    back = spatial_methods.aggregate_to_region(
        list(result["disaggregated"]), list(result["disaggregated"].values()), 8, "sum"
    )
    assert back["aggregated"][center]["value"] == pytest.approx(42)
    intensive = spatial_methods.disaggregate_to_cells(
        [center], [-2.0], 9, "proportional"
    )
    assert set(intensive["disaggregated"].values()) == {-2.0}
    for reducer, expected in [
        ("mean", 3.0),
        ("sum", 3 * len(children)),
        ("min", 3),
        ("max", 3),
        ("count", len(children)),
    ]:
        aggregated = spatial_methods.aggregate_to_region(
            children, [3.0] * len(children), 8, reducer
        )
        assert aggregated["aggregated"][center]["value"] == expected
    with pytest.raises(ValueError, match="allocation"):
        spatial_methods.disaggregate_to_cells([center], [1], 15, max_cells=1)
    with pytest.raises(ValueError, match="Aggregation target"):
        spatial_methods.aggregate_to_region([center], [1], 9)
    with pytest.raises(ValueError, match="Unknown aggregation"):
        spatial_methods.aggregate_to_region([center], [1], 8, "typo")


def test_coverage_deduplicates_and_measures_real_area(spatial_methods, center):
    cells = list(h3.grid_disk(center, 1))
    result = spatial_methods.calculate_coverage(cells[:2] * 2, cells * 2)
    assert result["num_cells"] == 2
    assert result["total_area_km2"] == pytest.approx(
        sum(h3.cell_area(c, "km^2") for c in cells[:2])
    )
    assert result["coverage_ratio"] == pytest.approx(2 / len(cells))


def test_weights_and_accessibility_have_analytical_oracles(spatial_methods, center):
    cells = list(h3.grid_disk(center, 1))
    for mode in ["queen", "rook", "distance"]:
        if mode == "distance" and h3.is_pentagon(center):
            with pytest.raises(h3.H3FailedError):
                spatial_methods.calculate_spatial_weights(cells, mode, 2)
            continue
        result = spatial_methods.calculate_spatial_weights(cells, mode, 2)
        for cell, row in result["weights"].items():
            neighbors = set(h3.grid_disk(cell, 2)) & set(cells) - {cell}
            assert set(row) == neighbors
            assert sum(row.values()) == pytest.approx(1)
            if mode == "distance":
                expected = {n: 1 / (h3.grid_distance(cell, n) + 1) for n in neighbors}
                total = sum(expected.values())
                assert row == pytest.approx({n: w / total for n, w in expected.items()})
    access = spatial_methods.compute_accessibility([center], cells, 1)["accessibility"][
        center
    ]
    assert access["reachable_destinations"] == len(cells)
    assert access["min_distance"] == 0
    assert access["avg_distance"] == pytest.approx((len(cells) - 1) / len(cells))
    assert access["accessibility_score"] == 1
    quadrants = spatial_methods.find_spatial_outliers(cells, [2] * len(cells))
    assert quadrants["significance_tested"] is False
    assert quadrants["outliers"]["not_significant"] == len(cells)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "2"])
def test_observation_contract_rejects_nonfinite_and_coercion(
    spatial_methods, center, value
):
    with pytest.raises(ValueError):
        spatial_methods.spatial_filter([center], [value], threshold=0)


@pytest.mark.parametrize("bad", [True, 1.5, -1])
def test_integer_configuration_rejects_coercion(spatial_methods, center, bad):
    with pytest.raises(ValueError):
        spatial_methods.buffer_analysis([center], bad)
    with pytest.raises(ValueError):
        spatial_methods.calculate_spatial_weights([center], k=bad)


def test_domain_contracts_and_array_inputs(spatial_methods, center):
    cells = list(h3.grid_disk(center, 1))[:2]
    assert spatial_methods.calculate_coverage(np.array(cells))["num_cells"] == 2
    for invalid in [["fake"], [center, center], [center, h3.cell_to_parent(center, 7)]]:
        with pytest.raises(ValueError):
            spatial_methods.spatial_filter(invalid, [1] * len(invalid), threshold=0)
