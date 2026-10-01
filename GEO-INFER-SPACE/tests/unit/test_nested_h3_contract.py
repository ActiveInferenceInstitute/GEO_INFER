"""Focused nested H3 hierarchy contract tests."""

from __future__ import annotations

import h3
import numpy as np
import pytest

from geo_infer_space.nested import NestedH3Grid


def _sf_boundary() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.435, 37.765],
                [-122.405, 37.765],
                [-122.405, 37.795],
                [-122.435, 37.795],
                [-122.435, 37.765],
            ]
        ],
    }


def test_nested_h3_hierarchy_from_boundary_is_closed() -> None:
    grid = NestedH3Grid("contract")
    hierarchy = grid.build_h3_hierarchy_from_boundary(_sf_boundary(), [7, 8, 9])

    validation = hierarchy["validation"]
    assert validation["is_valid"] is True
    assert validation["orphan_count"] == 0
    assert validation["multi_child_parent"] is True
    assert hierarchy["resolutions"] == [7, 8, 9]

    for parent, children in hierarchy["parent_child_map"].items():
        parent_resolution = h3.get_resolution(parent)
        for child in children:
            assert h3.cell_to_parent(child, parent_resolution) == parent

    for resolution, neighbor_map in hierarchy["same_level_neighbors"].items():
        resolution = int(resolution)
        for cell, neighbors in neighbor_map.items():
            assert h3.get_resolution(cell) == resolution
            assert all(
                h3.get_resolution(neighbor) == resolution for neighbor in neighbors
            )


def test_nested_h3_aggregation_is_finite_and_normalized() -> None:
    grid = NestedH3Grid("aggregation")
    hierarchy = grid.build_h3_hierarchy_from_boundary(_sf_boundary(), [7, 8, 9])
    values = {
        cell: [1.0, float(index + 1), 0.5]
        for index, cell in enumerate(hierarchy["leaf_cells"][:5])
    }

    aggregated = grid.aggregate_child_values_to_parents(values, 8)

    assert aggregated
    for belief in aggregated.values():
        array = np.asarray(belief, dtype=float)
        assert np.all(np.isfinite(array))
        assert np.isclose(array.sum(), 1.0)


def test_nested_h3_rejects_invalid_cells_and_unordered_resolutions() -> None:
    grid = NestedH3Grid("negative")

    with pytest.raises(ValueError, match="Invalid H3 cell"):
        grid.build_h3_hierarchy_from_cells(["not-a-cell"], [7, 8, 9])

    cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
    with pytest.raises(ValueError, match="ascending"):
        grid.build_h3_hierarchy_from_cells([cell], [8, 7, 9])


def test_nested_h3_validation_detects_orphans_and_mismatched_parent() -> None:
    grid = NestedH3Grid("validation")
    hierarchy = grid.build_h3_hierarchy_from_boundary(_sf_boundary(), [7, 8, 9])
    child, expected_parent = next(iter(hierarchy["child_parent_map"].items()))
    wrong_parent = next(
        parent for parent in hierarchy["parent_child_map"] if parent != expected_parent
    )

    broken = dict(hierarchy)
    broken["child_parent_map"] = dict(hierarchy["child_parent_map"])
    broken["child_parent_map"][child] = wrong_parent

    report = grid.validate_h3_hierarchy(broken)
    assert report["is_valid"] is False
    assert any("expected parent" in issue for issue in report["issues"])

    orphaned = dict(hierarchy)
    orphaned["child_parent_map"] = dict(hierarchy["child_parent_map"])
    del orphaned["child_parent_map"][child]
    report = grid.validate_h3_hierarchy(orphaned)
    assert report["is_valid"] is False
    assert report["orphan_count"] >= 1
