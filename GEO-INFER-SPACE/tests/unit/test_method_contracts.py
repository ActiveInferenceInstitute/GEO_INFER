"""Analytical regressions for public spatial, temporal and configuration seams."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import math
from dataclasses import FrozenInstanceError
import importlib.util
import builtins
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
from pyproj import Geod, Transformer
import pytest
from shapely.geometry import Point
import yaml

from geo_infer_space.analytics.temporal import TemporalAnalyzer
from geo_infer_space.analytics.vector import topology_operations
from geo_infer_space.backends.h3.analytics import H3TemporalAnalyzer
from geo_infer_space.backends.h3.core import H3Analytics, H3Cell, H3Grid
from geo_infer_space.backends.h3.h3_backend import H3Backend
from geo_infer_space.backends.h3.ml_integration import H3MLFeatureEngine
from geo_infer_space.core.algorithm_registry import (
    ProcessingContext,
    build_reference_registry,
)
from geo_infer_space.core.geometric_operations import GeometricOperationsInterface
from geo_infer_space.core.h3_policy import (
    check_cell_budget,
    estimate_cell_count,
    suggest_h3_resolution,
)
from geo_infer_space.core.statistics import SpatialStatistics
from geo_infer_space.nested.messaging.message_broker import Message, MessageType
from geo_infer_space.utils.config_loader import LocationBounds, LocationConfigLoader

CELL = h3.latlng_to_cell(37.7, -122.4, 8)


@pytest.mark.parametrize(
    "operation", ["buffer", "simplify", "convex_hull", "envelope", "dissolve"]
)
@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857"])
def test_topology_keeps_coordinates_bound_to_crs(operation, crs):
    source = gpd.GeoDataFrame(
        {"value": [1]}, geometry=[Point(-122.4, 37.7)], crs="EPSG:4326"
    )
    if crs == "EPSG:3857":
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            -122.4, 37.7
        )
        source = gpd.GeoDataFrame({"value": [1]}, geometry=[Point(x, y)], crs=crs)
    before = source.copy(deep=True)
    result = topology_operations(source, operation, tolerance=100)
    assert result.crs == source.crs
    centroid = result.geometry.iloc[0].centroid
    lon, lat = Transformer.from_crs(result.crs, "EPSG:4326", always_xy=True).transform(
        centroid.x, centroid.y
    )
    assert lon == pytest.approx(-122.4, abs=1e-5)
    assert lat == pytest.approx(37.7, abs=1e-5)
    assert source.equals(before)


def test_transform_dispatches_geometric_surface():
    from geo_infer_space.core.dispatcher import SpatialBackendDispatcher

    dispatcher = SpatialBackendDispatcher()

    class GeometryBackend:
        def get_capabilities(self):
            return {"geometric": {"transform_geometry": True}}

        def is_available(self):
            return True

        def transform_geometry(self, geometry, from_crs, to_crs):
            assert from_crs == "EPSG:4326" and to_crs == "EPSG:3857"
            return {"type": "Point", "coordinates": [1, 2]}

    dispatcher.register_backend("seam", GeometryBackend())
    interface = GeometricOperationsInterface(backend="seam")
    interface.dispatcher = dispatcher
    assert interface.transform_geometry(
        {"type": "Point", "coordinates": [0, 0]}, "EPSG:4326", "EPSG:3857"
    )["coordinates"] == [1, 2]


def test_config_merge_owns_nested_results_and_cache(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"spatial": {"h3_resolution": 0}})
    )
    (tmp_path / "one.yaml").write_text(
        yaml.safe_dump({"reporting": {"automated_reports": {"formats": ["csv"]}}})
    )
    loader = LocationConfigLoader(tmp_path)
    defaults = deepcopy(loader.default_config)
    first = loader.load_location_config("one")
    first["reporting"]["automated_reports"]["formats"].append("mutated")
    second = loader.load_location_config("two")
    assert loader.default_config == defaults
    assert second["reporting"]["automated_reports"]["formats"] == ["html", "pdf"]
    assert loader.loaded_config["one"]["reporting"]["automated_reports"]["formats"] == [
        "csv"
    ]
    assert first["spatial"]["h3_resolution"] == 0
    with pytest.raises(ValueError):
        loader.load_location_config("../one")
    (tmp_path / "bad.yaml").write_text("spatial: null\n")
    with pytest.raises(ValueError):
        loader.load_location_config("bad")
    assert "bad" not in loader.loaded_config


def test_location_antimeridian_center_and_invalid_bounds():
    bounds = LocationBounds(north=10, south=-10, west=170, east=-170)
    assert bounds.center() == (0, -180)
    for kwargs in [{"north": math.nan}, {"west": True}, {"east": 170}, {"south": 10}]:
        values = dict(north=10, south=-10, west=170, east=-170)
        values.update(kwargs)
        with pytest.raises(ValueError):
            LocationBounds(**values)


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, -1])
def test_h3_policy_budget_is_fail_closed(bad):
    with pytest.raises(ValueError):
        estimate_cell_count(bad, 8)
    with pytest.raises(ValueError):
        check_cell_budget(bad)
    with pytest.raises(ValueError):
        suggest_h3_resolution(bad)


@pytest.mark.parametrize("bad", [True, 1.5, -1, 16])
def test_h3_resolution_rejects_nonintegers_and_out_of_range(bad):
    with pytest.raises(ValueError):
        estimate_cell_count(1, bad)
    assert not H3Backend().validate_resolution(bad)["valid"]
    with pytest.raises(ValueError):
        H3Cell.from_coordinates(0, 0, bad)


def test_cell_and_grid_validate_and_bound_topology():
    for kwargs in [
        {"index": "fake", "resolution": 8},
        {"index": CELL, "resolution": 7},
        {"index": CELL, "resolution": 8, "created_at": datetime(2020, 1, 1)},
    ]:
        with pytest.raises(ValueError):
            H3Cell(**kwargs)
    cell = H3Cell(index=CELL, resolution=8)
    assert cell.created_at.tzinfo is UTC
    assert len(cell.neighbors()) == 6
    assert cell.parent().resolution == 7
    assert len(cell.children()) == 7
    assert cell.distance_to(cell) == 0
    assert not cell.is_neighbor(cell)
    with pytest.raises(ValueError):
        cell.distance_to(cell.parent())
    with pytest.raises(ValueError, match="allocation"):
        cell.children(15, max_cells=1)
    with pytest.raises(ValueError):
        cell.neighbors(True)
    with pytest.raises(ValueError):
        cell.parent(8)
    with pytest.raises(ValueError):
        cell.children(8)
    base = H3Cell(index=h3.cell_to_parent(CELL, 0), resolution=0)
    assert base.parent() is None
    source = [cell]
    grid = H3Grid(source)
    source.clear()
    assert len(grid) == 1
    with pytest.raises(ValueError):
        H3Grid([cell, cell])
    with pytest.raises(ValueError):
        H3Grid.from_center(math.nan, 0, 8)
    with pytest.raises(ValueError, match="allocation"):
        H3Grid.from_center(0, 0, 8, k=100, max_cells=1)
    with pytest.raises(ValueError):
        H3Grid.from_polygon([], 8)
    children = H3Grid(cell.children())
    assert {c.index for c in children.compact()} == {CELL}
    assert {c.index for c in H3Grid([cell]).uncompact(9)} == {c.index for c in children}
    with pytest.raises(ValueError, match="overlap"):
        H3Grid([cell, *cell.children()]).uncompact(9)


def test_topology_failure_never_returns_empty_grid(monkeypatch):
    cell = H3Cell(index=CELL, resolution=8)
    grid = H3Grid([cell])

    def unavailable(*args):
        raise h3.H3FailedError("unavailable")

    monkeypatch.setattr(h3, "grid_disk", unavailable)
    for operation in [
        cell.neighbors,
        lambda: H3Grid.from_center(0, 0, 8),
        lambda: H3Analytics(grid).connectivity_analysis(),
        lambda: H3Backend().calculate_density([CELL], [0]),
    ]:
        with pytest.raises(h3.H3FailedError):
            operation()


def test_interpolation_and_density_preserve_zero_and_reject_missing_sources():
    backend = H3Backend()
    neighbors = list(h3.grid_disk(CELL, 1))
    targets = [c for c in neighbors if c != CELL][:2]
    values = [0.0, 4.0]
    density = backend.calculate_density([CELL, targets[0]], values)["densities"]
    assert density[CELL] == pytest.approx(4 / 3)
    result = backend.interpolate_values([CELL, targets[0]], values, [CELL, targets[1]])
    assert result["interpolated"][CELL] == 0
    distances = [h3.grid_distance(targets[1], c) for c in [CELL, targets[0]]]
    weights = [1 / d**2 for d in distances]
    assert result["interpolated"][targets[1]] == pytest.approx(
        sum(w * v for w, v in zip(weights, values)) / sum(weights)
    )
    with pytest.raises(ValueError):
        backend.interpolate_values([], [], targets)
    with pytest.raises(ValueError):
        backend.calculate_density([CELL, CELL], [1, 2])


def test_temporal_patterns_weight_real_observations_and_utc_hours():
    rows = [
        {"timestamp": "2020-01-01T00:00:00Z", "value": 0},
        {"timestamp": "2019-12-31T16:00:00-08:00", "value": 2},
        {"timestamp": "2020-01-01T01:00:00Z", "value": 10},
        {"timestamp": "2020-01-01T02:00:00Z", "value": None},
    ]
    generic = TemporalAnalyzer().analyze_temporal_patterns(rows, "timestamp", "value")
    indexes = list(h3.grid_disk(CELL, 1))[:4]
    grid = H3Grid(
        [H3Cell(index=c, resolution=8, properties=row) for c, row in zip(indexes, rows)]
    )
    temporal = H3TemporalAnalyzer(grid).analyze_temporal_patterns("timestamp", "value")
    for result in [generic, temporal]:
        assert result["data_points"] == 3
        assert result["aggregated_data"][0]["count"] == 2
        assert result["statistics"]["overall_mean"] == 4
        assert result["statistics"]["overall_std"] == pytest.approx(np.std([0, 2, 10]))
        assert result["statistics"]["overall_min"] == 0
        assert result["statistics"]["overall_max"] == 10
    with pytest.raises(ValueError):
        TemporalAnalyzer().analyze_temporal_patterns(
            [{"timestamp": "2020-01-01", "value": 1}], "timestamp", "value"
        )
    with pytest.raises(ValueError):
        TemporalAnalyzer().analyze_temporal_patterns(
            [{"timestamp": "2020-01-01T00:00:00Z", "value": math.nan}],
            "timestamp",
            "value",
        )
    features = H3MLFeatureEngine(grid)._extract_temporal_features(grid.cells[1])
    assert features["hour"] == 0


def test_grid_temporal_axes_and_missing_observations():
    grids = [
        H3Grid([H3Cell(index=CELL, resolution=8, properties={"v": v})])
        for v in [-10, None, -5]
    ]
    for i, grid in enumerate(grids):
        grid.created_at = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=i)
    analyzer = H3TemporalAnalyzer(grids)
    trend = analyzer.analyze_temporal_trends("v")["cell_trends"][CELL]
    assert trend == {
        "trend": "increasing",
        "change": 5.0,
        "percent_change": 50.0,
        "data_points": 2,
    }
    assert analyzer.detect_anomalies("v")["mean_value"] == -7.5
    with pytest.raises(ValueError):
        H3TemporalAnalyzer(grids[::-1])
    with pytest.raises(ValueError):
        H3TemporalAnalyzer([grids[0], grids[0]])


def test_registry_bounds_all_geometry_depths_and_nulls():
    geometries = [
        None,
        {"type": "Point", "coordinates": [9, 10]},
        {"type": "LineString", "coordinates": [[1, 2], [3, 4]]},
        {"type": "MultiPolygon", "coordinates": [[[[0, 0], [2, 0], [2, 2], [0, 0]]]]},
        {
            "type": "GeometryCollection",
            "geometries": [{"type": "MultiPoint", "coordinates": [[-5, -6], [5, 6]]}],
        },
    ]
    context = ProcessingContext(
        layers=[
            {
                "id": "x",
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": g, "properties": {}}
                        for g in geometries
                    ],
                },
            }
        ],
        parameters={"layer": "x"},
    )
    assert build_reference_registry().run("calculate-bounds", context) == [
        -5,
        -6,
        9,
        10,
    ]


def test_nearest_neighbor_uses_metric_area_and_actual_geodesic_distance():
    other = next(c for c in h3.grid_disk(CELL, 1) if c != CELL)
    stats = SpatialStatistics()
    descriptive = stats.nearest_neighbor_index([CELL, other])
    assert descriptive["reference_tested"] is False and "p_value" not in descriptive
    lat1, lon1 = h3.cell_to_latlng(CELL)
    lat2, lon2 = h3.cell_to_latlng(other)
    distance = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)[2] / 1000
    assert descriptive["observed_mean_distance"] == pytest.approx(distance)
    reference = stats.nearest_neighbor_index([CELL, other], study_area_km2=8)
    assert reference["expected_mean_distance"] == 1
    assert reference["nni"] == pytest.approx(distance)
    assert reference["edge_corrected"] is False
    assert stats.calculate_summary_statistics([3, 3, 3])["skewness"] == 0


def test_message_metadata_aware_time_and_zero_ttl():
    created = datetime(2020, 1, 1, tzinfo=UTC)
    message = Message(
        "x",
        "sender",
        "recipient",
        MessageType.STATUS,
        created_at=created,
        ttl=timedelta(0),
    )
    assert message.expires_at == created and message.is_expired()
    with pytest.raises(ValueError):
        Message("x", "s", "r", MessageType.STATUS, created_at=datetime(2020, 1, 1))


def test_dispatcher_does_not_hide_broken_backend_import(monkeypatch):
    from geo_infer_space.core.dispatcher import SpatialBackendDispatcher

    def broken(self):
        raise ImportError("broken internal H3 module")

    monkeypatch.setattr(SpatialBackendDispatcher, "_load_h3_backend", broken)
    with pytest.raises(ImportError, match="broken internal H3 module"):
        SpatialBackendDispatcher()


@pytest.mark.parametrize("failure", ["absent_extra", "broken_dependency", "broken_api"])
def test_srai_absence_is_distinct_from_broken_installation(monkeypatch, failure):
    from geo_infer_space.backends.srai import srai_backend
    from geo_infer_space.core.interfaces import SRAIUnavailableError

    path = Path(srai_backend.__file__)
    spec = importlib.util.spec_from_file_location(srai_backend.__name__, path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def import_boundary(name, *args, **kwargs):
        if name == "srai":
            if failure == "broken_api":
                raise ImportError("Broken installed SRAI API")
            missing = "srai" if failure == "absent_extra" else "srai_dependency"
            raise ModuleNotFoundError(f"No module named {missing}", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_boundary)
    if failure == "absent_extra":
        spec.loader.exec_module(module)
        backend = module.SraiBackend()
        assert not backend.is_available()
        with pytest.raises(SRAIUnavailableError):
            backend.latlng_to_cell(37.7, -122.4, 8)
    elif failure == "broken_dependency":
        with pytest.raises(ModuleNotFoundError, match="srai_dependency"):
            spec.loader.exec_module(module)
    else:
        with pytest.raises(ImportError, match="Broken installed SRAI API"):
            spec.loader.exec_module(module)


def test_grid_statistics_refresh_and_reserved_identity():
    cell = H3Cell(
        index=CELL,
        resolution=8,
        properties={"h3_index": "fake", "resolution": 99, "v": 0},
    )
    grid = H3Grid([cell])
    analyzer = H3Analytics(grid)
    assert analyzer.basic_statistics()["cell_count"] == 1
    neighbor = cell.neighbors()[0]
    grid.add_cell(neighbor)
    assert analyzer.basic_statistics()["cell_count"] == 2
    assert cell.to_geojson()["properties"]["h3_index"] == CELL
    assert grid.to_dataframe().iloc[0]["h3_index"] == CELL
    assert grid.to_dataframe().iloc[0]["resolution"] == 8


def test_cell_identity_and_grid_membership_preserve_lookup_and_owned_properties():
    properties = {"nested": {"observations": [0]}}
    cell = H3Cell(index=CELL, resolution=8, properties=properties)
    grid = H3Grid([cell])
    neighbor = cell.neighbors()[0]
    for name, value in [("index", neighbor.index), ("resolution", 7), ("latitude", 0)]:
        with pytest.raises(FrozenInstanceError):
            setattr(cell, name, value)
    with pytest.raises(AttributeError):
        grid.cells.append(neighbor)
    with pytest.raises(AttributeError):
        grid.cells = (neighbor,)
    properties["nested"]["observations"].append(7)
    assert cell.properties["nested"]["observations"] == [0]
    exported = cell.to_geojson()
    exported["properties"]["nested"]["observations"].append(8)
    frame = grid.to_dataframe()
    frame.iloc[0]["nested"]["observations"].append(9)
    assert cell.properties["nested"]["observations"] == [0]
    cell.properties["nested"]["observations"].append(1)
    assert grid.get_cell(CELL).properties["nested"]["observations"] == [0, 1]
    previous_membership = grid.cells
    grid.add_cell(neighbor)
    assert grid.get_cell(neighbor.index) is neighbor and len(grid) == 2
    assert previous_membership == (cell,)
    assert grid.cells == (cell, neighbor)
    assert grid.remove_cell(CELL)
    assert not grid.has_cell(CELL) and [c.index for c in grid] == [neighbor.index]


def test_native_allocations_fail_before_topology_and_keep_pentagon_identity(
    monkeypatch,
):
    backend = H3Backend()
    cell = H3Cell(index=CELL, resolution=8)
    pentagon = h3.get_pentagons(8)[0]
    children = backend.get_cell_children(pentagon, 9, max_cells=6)
    assert len(children) == 6
    assert set(backend.uncompact_cells([pentagon], 9, max_cells=6)) == set(children)
    assert all(h3.cell_to_parent(index, 8) == pentagon for index in children)

    def allocation_must_not_run(*args, **kwargs):
        raise AssertionError("Native allocation occurred before budget rejection")

    for function in ["grid_disk", "grid_ring", "cell_to_children", "uncompact_cells"]:
        monkeypatch.setattr(h3, function, allocation_must_not_run)
    for operation in [
        lambda: cell.neighbors(1000, max_cells=10),
        lambda: backend.get_cell_neighbors(CELL, 1000, max_cells=10),
        lambda: backend.get_cells_within_radius(CELL, 1000, max_cells=10),
        lambda: backend.get_cell_ring(CELL, 1000, max_cells=10),
        lambda: backend.get_cell_children(CELL, 15, max_cells=10),
        lambda: backend.uncompact_cells([CELL], 15, max_cells=10),
    ]:
        with pytest.raises(ValueError, match="allocation"):
            operation()
    with pytest.raises(ValueError, match="overlap"):
        backend.uncompact_cells([pentagon, children[0]], 9)


def test_native_demo_import_is_quiet_and_checks_real_composition(capsys):
    path = Path(__file__).parents[2] / "examples" / "demo_all_methods.py"
    spec = importlib.util.spec_from_file_location("space_contract_demo", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    assert capsys.readouterr().out == ""
    result = example.run_demo(resolution=8)
    assert result["child_count"] == 7 and result["total"] == pytest.approx(42)
    assert result["observed_count"] == 2 and result["missing_count"] == 12
    assert result["backend"] == "native_h3_cpu"
    assert len(result["checks"]) == 6
    with pytest.raises(ValueError, match="allocation"):
        example.run_demo(max_cells=1)


def test_state_space_rejects_fabricated_topology_and_boolean_coordinates(monkeypatch):
    from geo_infer_space.core.state_space import H3StateSpace

    state = H3StateSpace([CELL])
    with pytest.raises(ValueError):
        state.locate(True, 0)
    monkeypatch.setattr(h3, "grid_disk", lambda *args: [CELL])
    with pytest.raises(ValueError, match="neighbor domain"):
        state.transitions()


def test_metric_neighbor_allocation_is_bounded_before_topology():
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

    indexer = SpatialIndexingInterface()
    for bad in [math.nan, math.inf, True, -1]:
        with pytest.raises(ValueError):
            indexer.get_neighbors((37.7, -122.4), bad)
    with pytest.raises(ValueError, match="allocation"):
        indexer.get_neighbors((37.7, -122.4), 1000, max_cells=1)


def test_base_orchestration_configuration_and_explicit_output(tmp_path):
    from geo_infer_space.core.base_module import BaseAnalysisModule

    class DomainModule(BaseAnalysisModule):
        def acquire_raw_data(self):
            return tmp_path / "data.geojson"

        def run_final_analysis(self, data):
            return data

    config = tmp_path / "domain.yaml"
    config.write_text("spatial:\n  h3_resolution: 0\n")
    module = DomainModule("domain", config_path=config, output_dir=tmp_path / "cache")
    assert module.h3_resolution == 0
    assert module.h3_cache_path == tmp_path / "cache" / "domain_h3_res0.json"
    assert (
        DomainModule(
            "domain", config_path=config, h3_resolution=3, output_dir=tmp_path / "other"
        ).h3_resolution
        == 3
    )
    config.write_text("[]")
    with pytest.raises(ValueError):
        DomainModule("domain", config_path=config, output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(FileNotFoundError):
        DomainModule("domain", config_path=tmp_path / "absent")
    with pytest.raises(ValueError):
        DomainModule("../escape", output_dir=tmp_path / "invalid")


def test_ml_features_keep_missing_neighbors_distinct_from_observed_zero():
    grid = H3Grid([H3Cell(index=CELL, resolution=8, properties={"v": 0})])
    result = H3MLFeatureEngine(grid).create_spatial_features("v", neighbor_rings=1)
    feature = result["features"][0]
    assert feature["target_value"] == 0 and feature["ring_1_count"] == 0
    assert math.isnan(feature["ring_1_mean"]) and math.isnan(feature["neighbor_avg"])
    assert feature["neighbor_total"] == 0
    grid.cells[0].properties["v"] = None
    assert H3MLFeatureEngine(grid).create_spatial_features("v")["features"] == []
    grid.cells[0].properties["v"] = math.nan
    with pytest.raises(ValueError):
        H3MLFeatureEngine(grid).create_spatial_features("v")


def test_grid_bounds_enclose_actual_cell_boundaries_including_single_cell():
    cells = [CELL, h3.latlng_to_cell(0, 179.99, 4)]
    for indexes in ([cells[0]], cells):
        grid = H3Grid(
            [
                H3Cell(index=index, resolution=h3.get_resolution(index))
                for index in indexes
            ]
        )
        boundary = np.array(
            [vertex for index in indexes for vertex in h3.cell_to_boundary(index)]
        )
        bounds = grid.bounds()
        expected = (*boundary.min(axis=0), *boundary.max(axis=0))
        np.testing.assert_array_equal(bounds, expected)
        assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
        assert H3Analytics(grid).spatial_distribution()["bounding_box_area_deg2"] > 0
    assert H3Grid([]).bounds() == (0, 0, 0, 0)
