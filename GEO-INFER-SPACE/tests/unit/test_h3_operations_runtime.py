"""Regression tests for the public H3 operation helpers."""

import importlib.util
from pathlib import Path

import pytest
import h3

from geo_infer_space.backends.h3.h3_backend import H3Backend
from geo_infer_space.backends.h3.operations import (
    are_neighbor_cells,
    cell_area,
    cell_to_boundary,
    cell_to_children,
    cell_to_coordinates,
    cell_to_parent,
    cell_resolution,
    cells_difference,
    cells_intersection,
    cells_to_geojson,
    cells_to_polygon,
    cells_union,
    compact_cells,
    coordinate_to_cell,
    create_h3_grid_for_bounds,
    find_optimal_resolution,
    get_resolution_info,
    grid_disk,
    grid_distance,
    grid_path,
    grid_ring,
    grid_statistics,
    is_valid_cell,
    neighbor_cells,
    polygon_to_cells,
    uncompact_cells,
)
from geo_infer_space.core.interfaces import H3UnavailableError
from geo_infer_space.utils import h3_utils
from geo_infer_space.utils.h3_utils import (
    cell_to_latlngjson,
    polygon_to_cells as geojson_polygon_to_cells,
)


def test_cell_to_coordinates_round_trip() -> None:
    cell = coordinate_to_cell(37.7749, -122.4194, 9)
    latitude, longitude = cell_to_coordinates(cell)

    assert abs(latitude - 37.7749) < 0.01
    assert abs(longitude + 122.4194) < 0.01


def test_cell_to_coordinates_rejects_invalid_cell() -> None:
    with pytest.raises(ValueError):
        cell_to_coordinates("not-an-h3-cell")


def test_runtime_h3_version_is_current_v4_contract() -> None:
    major, minor, patch = (int(part) for part in h3.__version__.split(".")[:3])
    assert (major, minor, patch) >= (4, 5, 0)
    assert major < 5


@pytest.mark.parametrize("resolution", [0, 7, 15])
def test_resolution_info_uses_native_h3_v4_metrics(resolution: int) -> None:
    info = get_resolution_info(resolution)

    assert info["avg_area_km2"] == pytest.approx(
        h3.average_hexagon_area(resolution, unit="km^2")
    )
    assert info["avg_edge_length_km"] == pytest.approx(
        h3.average_hexagon_edge_length(resolution, unit="km")
    )
    assert info["avg_area_m2"] == pytest.approx(info["avg_area_km2"] * 1_000_000)


@pytest.mark.parametrize("resolution", [-1, 16])
def test_resolution_info_rejects_out_of_range_values(resolution: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 15"):
        get_resolution_info(resolution)


@pytest.mark.parametrize("resolution", [True, 7.5, "7"])
def test_resolution_info_requires_an_integer(resolution: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        get_resolution_info(resolution)  # type: ignore[arg-type]


def test_optimal_resolution_validates_inputs_and_targets_native_metrics() -> None:
    recommendation = find_optimal_resolution(100.0, target_cells=100)

    assert 0 <= recommendation["recommended_resolution"] <= 15
    assert recommendation["target_cells"] == 100
    assert recommendation["all_options"][0]["efficiency_score"] >= 0

    with pytest.raises(ValueError, match="greater than zero"):
        find_optimal_resolution(0)
    with pytest.raises(ValueError, match="Target cells"):
        find_optimal_resolution(100, target_cells=0)


def test_cell_boundary_supports_native_and_geojson_coordinate_orders() -> None:
    cell = coordinate_to_cell(37.7749, -122.4194, 9)

    native = cell_to_boundary(cell)
    geojson = cell_to_boundary(cell, geo_json=True)

    assert len(native) == len(geojson) == 6
    assert geojson[0] == (native[0][1], native[0][0])


def test_h3_geojson_exports_emit_closed_lng_lat_rings() -> None:
    cell = coordinate_to_cell(37.7749, -122.4194, 9)

    feature_collection = cells_to_geojson([cell])
    ring = feature_collection["features"][0]["geometry"]["coordinates"][0]

    assert ring[0] == ring[-1]
    assert all(-180 <= point[0] <= 180 and -90 <= point[1] <= 90 for point in ring)


def test_polygon_and_reverse_polygon_wrappers_use_h3_v4_shapes() -> None:
    coords = [
        (37.70, -122.50),
        (37.70, -122.30),
        (37.80, -122.30),
        (37.80, -122.50),
    ]
    cells = polygon_to_cells(coords, 8)

    assert cells
    boundary = cells_to_polygon({cells[0]})
    assert boundary[0] == boundary[-1]


def test_geojson_polygon_preserves_holes_and_multipolygons() -> None:
    polygon_with_hole = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.50, 37.70],
                [-122.30, 37.70],
                [-122.30, 37.80],
                [-122.50, 37.80],
                [-122.50, 37.70],
            ],
            [
                [-122.46, 37.74],
                [-122.34, 37.74],
                [-122.34, 37.76],
                [-122.46, 37.76],
                [-122.46, 37.74],
            ],
        ],
    }
    multipolygon = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-122.50, 37.70],
                    [-122.45, 37.70],
                    [-122.45, 37.75],
                    [-122.50, 37.75],
                    [-122.50, 37.70],
                ]
            ],
            [
                [
                    [-122.35, 37.75],
                    [-122.30, 37.75],
                    [-122.30, 37.80],
                    [-122.35, 37.80],
                    [-122.35, 37.75],
                ]
            ],
        ],
    }

    assert set(geojson_polygon_to_cells(polygon_with_hole, 8)) == set(
        h3.geo_to_cells(polygon_with_hole, 8)
    )
    assert set(geojson_polygon_to_cells(multipolygon, 8)) == set(
        h3.geo_to_cells(multipolygon, 8)
    )


def test_geojson_antimeridian_polygon_stays_near_dateline() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [179.8, 10.0],
                [-179.8, 10.0],
                [-179.8, 10.2],
                [179.8, 10.2],
                [179.8, 10.0],
            ]
        ],
    }

    cells = geojson_polygon_to_cells(geometry, 7)
    longitudes = [h3.cell_to_latlng(cell)[1] for cell in cells]

    assert cells
    assert all(longitude > 150 or longitude < -150 for longitude in longitudes)


def test_pentagon_neighbors_are_the_exact_native_disk_ring() -> None:
    backend = H3Backend()
    pentagon = h3.get_pentagons(0)[0]

    assert set(backend.get_cell_neighbors(pentagon)) == set(
        h3.grid_disk(pentagon, 1)
    ) - {pentagon}


def test_compaction_round_trip_matches_native_h3() -> None:
    parent = h3.latlng_to_cell(37.7749, -122.4194, 7)
    children = set(h3.cell_to_children(parent, 9))
    compacted = set(compact_cells(children))

    assert compacted == set(h3.compact_cells(children))
    assert set(uncompact_cells(compacted, 9)) == children


def test_grid_and_hierarchy_helpers_match_native_h3_v4() -> None:
    center = coordinate_to_cell(37.7749, -122.4194, 9)
    disk = grid_disk(center, 2)
    ring = grid_ring(center, 2)

    assert disk == sorted(h3.grid_disk(center, 2))
    assert ring == sorted(h3.grid_ring(center, 2))
    assert neighbor_cells(center) == sorted(set(h3.grid_disk(center, 1)) - {center})
    assert grid_distance(center, ring[0]) == 2
    assert grid_path(center, ring[0])[0] == center
    assert grid_path(center, ring[0])[-1] == ring[0]
    assert are_neighbor_cells(center, neighbor_cells(center)[0])

    parent = cell_to_parent(center, 8)
    assert cell_resolution(parent) == 8
    assert center in cell_to_children(parent, 9)
    assert is_valid_cell(center)
    assert not is_valid_cell("not-an-h3-cell")


def test_collection_helpers_are_deterministic_and_report_statistics() -> None:
    center = coordinate_to_cell(37.7749, -122.4194, 8)
    disk = set(h3.grid_disk(center, 1))
    neighbors = disk - {center}

    assert cells_intersection(disk, neighbors) == sorted(neighbors)
    assert cells_union({center}, neighbors) == sorted(disk)
    assert cells_difference(disk, neighbors) == [center]

    stats = grid_statistics(disk)
    assert stats["total_cells"] == len(disk)
    assert stats["unique_resolutions"] == [8]
    assert stats["resolution_counts"] == {8: len(disk)}
    assert stats["total_area_km2"] == pytest.approx(
        sum(cell_area(cell) for cell in disk)
    )


def test_bounding_box_grid_is_deterministic_and_uses_v4_shapes() -> None:
    first = create_h3_grid_for_bounds(37.70, 37.72, -122.50, -122.48, 9)
    second = create_h3_grid_for_bounds(37.70, 37.72, -122.50, -122.48, 9)

    assert first
    assert first == sorted(first)
    assert first == second


def test_geojson_utility_does_not_mutate_inputs_or_properties() -> None:
    flat_polygon = {
        "type": "Polygon",
        "coordinates": [
            [-122.50, 37.70],
            [-122.30, 37.70],
            [-122.30, 37.80],
            [-122.50, 37.80],
            [-122.50, 37.70],
        ],
    }
    original_coordinates = list(flat_polygon["coordinates"])
    properties = {"cell": {"label": "sample"}}
    cell = coordinate_to_cell(37.7749, -122.4194, 9)

    assert geojson_polygon_to_cells(flat_polygon, 8)
    assert flat_polygon["coordinates"] == original_coordinates

    cell_to_latlngjson([cell], properties)
    assert properties == {"cell": {"label": "sample"}}


def test_public_space_h3_exports_cover_the_native_helpers() -> None:
    from geo_infer_space.utils import (
        cell_to_latlngjson as exported_cell_to_latlngjson,
        geojson_to_h3 as exported_geojson_to_h3,
        grid_ring as exported_grid_ring,
    )

    assert exported_cell_to_latlngjson is cell_to_latlngjson
    assert callable(exported_geojson_to_h3)
    cell = coordinate_to_cell(37.7749, -122.4194, 9)
    assert set(exported_grid_ring(cell, 1)) == set(h3.grid_disk(cell, 1)) - {cell}


@pytest.mark.parametrize("version", ["4.4.9", "5.0.0", "invalid", None])
def test_unsupported_h3_version_retains_backend_and_utility_refusals(
    monkeypatch, version
) -> None:
    """One version policy retains unavailable backends and fail-fast utilities."""
    monkeypatch.setattr(h3, "__version__", version)
    backend = H3Backend()
    assert backend.is_available() is False
    assert backend.get_capabilities()["available"] is False
    with pytest.raises(H3UnavailableError):
        backend.latlng_to_cell(37.7749, -122.4194, 9)

    specification = importlib.util.spec_from_file_location(
        "geo_infer_space.utils._h3_version_refusal_probe", Path(h3_utils.__file__)
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with pytest.raises(RuntimeError, match="requires h3-py >=4.5.0,<5"):
        specification.loader.exec_module(module)
