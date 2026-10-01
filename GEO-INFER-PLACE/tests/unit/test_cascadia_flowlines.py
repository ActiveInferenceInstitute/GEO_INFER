#!/usr/bin/env python3
"""
Unit tests for Cascadia high-resolution hydrography, flowline networks, and topology validation.

Covers:
- CascadiaFlowlineNetwork creation and graph construction
- FlowlineTopologyValidator (DAG cycle check, Strahler monotonicity, headwaters, outlets)
- Downstream and upstream topology traversals (trace_downstream, trace_upstream)
- Flow distance and upstream network length calculations
- High-order Pacific Northwest tributary filtering (stream order >= 4, 5, 7, 8)
- H3 hexagonal indexing of flowlines (intersect_flowlines_to_h3)
- GeoInferSurfaceWater module integration with topology traversal
"""

from __future__ import annotations


import geopandas as gpd
import pytest
from shapely.geometry import LineString

from geo_infer_place.hydrography import (
    CascadiaFlowlineNetwork,
    FlowlineTopologyValidator,
    CascadianSurfaceWaterDataSources,
    GeoInferSurfaceWater,
    sample_flowlines,
)


@pytest.fixture
def sample_flowlines_gdf() -> gpd.GeoDataFrame:
    """Create a deterministic synthetic GeoDataFrame representing a tributary network."""
    records = [
        {
            "comid": 101,
            "gnis_name": "Columbia River Reach 1",
            "reachcode": "17080006000101",
            "stream_order": 8,
            "from_node": "N1",
            "to_node": "N0",
            "length_km": 50.0,
            "slope": 0.0001,
            "drainage_area_sqkm": 600000.0,
            "mainstem": "Columbia River",
            "basin": "Columbia River Basin",
            "geometry": LineString([(-123.5, 46.2), (-124.0, 46.25)]),
        },
        {
            "comid": 102,
            "gnis_name": "Columbia River Reach 2",
            "reachcode": "17080006000102",
            "stream_order": 8,
            "from_node": "N2",
            "to_node": "N1",
            "length_km": 60.0,
            "slope": 0.0002,
            "drainage_area_sqkm": 550000.0,
            "mainstem": "Columbia River",
            "basin": "Columbia River Basin",
            "geometry": LineString([(-122.8, 45.9), (-123.5, 46.2)]),
        },
        {
            "comid": 103,
            "gnis_name": "Willamette River Mainstem",
            "reachcode": "17090012000103",
            "stream_order": 7,
            "from_node": "W1",
            "to_node": "N2",
            "length_km": 40.0,
            "slope": 0.0004,
            "drainage_area_sqkm": 29000.0,
            "mainstem": "Willamette River",
            "basin": "Columbia River Basin",
            "geometry": LineString([(-122.65, 45.4), (-122.8, 45.9)]),
        },
        {
            "comid": 104,
            "gnis_name": "McKenzie River",
            "reachcode": "17090004000104",
            "stream_order": 5,
            "from_node": "M1",
            "to_node": "W1",
            "length_km": 80.0,
            "slope": 0.002,
            "drainage_area_sqkm": 3400.0,
            "mainstem": "McKenzie River",
            "basin": "Columbia River Basin",
            "geometry": LineString([(-122.0, 44.2), (-122.65, 45.4)]),
        },
    ]
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


@pytest.fixture
def flowline_network(sample_flowlines_gdf) -> CascadiaFlowlineNetwork:
    """Instantiate a CascadiaFlowlineNetwork from sample GeoDataFrame."""
    return CascadiaFlowlineNetwork(sample_flowlines_gdf)


class TestCascadiaFlowlineNetworkConstruction:
    """Test CascadiaFlowlineNetwork graph generation and property lookup."""

    def test_network_nodes_and_edges(self, flowline_network):
        """Graph contains correct number of nodes and directed edges."""
        graph = flowline_network.graph
        assert graph.number_of_nodes() == 5  # N0, N1, N2, W1, M1
        assert graph.number_of_edges() == 4

    def test_get_flowline_by_comid(self, flowline_network):
        """Retrieve attributes for reach by its COMID."""
        fl = flowline_network.get_flowline_by_comid(101)
        assert fl is not None
        assert fl["gnis_name"] == "Columbia River Reach 1"
        assert fl["stream_order"] == 8
        assert fl["length_km"] == 50.0

    def test_get_nonexistent_comid_returns_none(self, flowline_network):
        """Querying invalid COMID returns None."""
        assert flowline_network.get_flowline_by_comid(999999) is None

    def test_from_geojson_file(self, tmp_path):
        """Round-trip the traceable real Smith River excerpt through GeoJSON."""
        geojson_path = tmp_path / "smith.geojson"
        sample_flowlines().to_file(geojson_path, driver="GeoJSON")
        network = CascadiaFlowlineNetwork.from_geojson(geojson_path)
        assert network.graph.number_of_edges() == 34
        assert len(network._comid_to_edge) == 34


class TestFlowlineTopologyValidation:
    """Test validation methods for graph acyclicity, Strahler order, and connectivity."""

    def test_valid_network_passes_all_checks(self, flowline_network):
        """Dendritic river network should be a valid DAG with monotonic stream orders."""
        report = flowline_network.validate()
        assert report["valid"] is True
        assert report["dag_validation"]["is_dag"] is True
        assert report["dag_validation"]["has_cycles"] is False
        assert report["strahler_monotonicity"]["monotonic"] is True

    def test_find_headwaters_and_outlets(self, flowline_network):
        """Accurately identify headwater source reaches and terminal basin outlets."""
        validator = FlowlineTopologyValidator(flowline_network.graph)
        headwaters = validator.find_headwaters()
        outlets = validator.find_outlets()

        assert "M1" in headwaters
        assert "N0" in outlets
        assert len(outlets) == 1

    def test_cycle_detection(self):
        """Cycle in flowline topology must be detected and reported."""
        records = [
            {
                "comid": 1,
                "from_node": "A",
                "to_node": "B",
                "stream_order": 4,
                "length_km": 10.0,
                "geometry": LineString([(0, 0), (1, 1)]),
            },
            {
                "comid": 2,
                "from_node": "B",
                "to_node": "C",
                "stream_order": 4,
                "length_km": 10.0,
                "geometry": LineString([(1, 1), (2, 2)]),
            },
            {
                "comid": 3,
                "from_node": "C",
                "to_node": "A",
                "stream_order": 4,
                "length_km": 10.0,
                "geometry": LineString([(2, 2), (0, 0)]),
            },
        ]
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        net = CascadiaFlowlineNetwork(gdf)
        report = net.validate()
        assert report["valid"] is False
        assert report["dag_validation"]["is_dag"] is False
        assert report["dag_validation"]["has_cycles"] is True
        assert report["dag_validation"]["cycle_count"] >= 1

    def test_strahler_monotonicity_violation(self):
        """Downstream decrease in stream order must be flagged as violation."""
        records = [
            {
                "comid": 1,
                "from_node": "A",
                "to_node": "B",
                "stream_order": 6,
                "length_km": 10.0,
                "geometry": LineString([(0, 0), (1, 1)]),
            },
            {
                "comid": 2,
                "from_node": "B",
                "to_node": "C",
                "stream_order": 4,  # Invalid decrease downstream
                "length_km": 10.0,
                "geometry": LineString([(1, 1), (2, 2)]),
            },
        ]
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        net = CascadiaFlowlineNetwork(gdf)
        report = net.validate()
        assert report["valid"] is False
        assert report["strahler_monotonicity"]["monotonic"] is False
        assert report["strahler_monotonicity"]["violation_count"] == 1


class TestFlowlineTraversalsAndMetrics:
    """Test upstream tributary tracing, downstream routing, and flow distances."""

    def test_trace_downstream(self, flowline_network):
        """Downstream trace follows flow from headwater tributary to estuary outlet."""
        path = flowline_network.trace_downstream(104)  # McKenzie River
        comids = [e["comid"] for e in path]
        assert comids == [104, 103, 102, 101]

    def test_trace_upstream(self, flowline_network):
        """Upstream trace gathers all contributing tributaries."""
        upstream = flowline_network.trace_upstream(102)  # Columbia River Reach 2
        comids = {e["comid"] for e in upstream}
        assert 102 in comids
        assert 103 in comids
        assert 104 in comids
        assert 101 not in comids  # 101 is downstream

    def test_calculate_downstream_distance(self, flowline_network):
        """Calculate cumulative downstream flow distance in km."""
        dist = flowline_network.calculate_downstream_distance(104)
        # 80 + 40 + 60 + 50 = 230 km
        assert dist == pytest.approx(230.0, 0.1)

    def test_calculate_upstream_network_length(self, flowline_network):
        """Calculate cumulative tributary stream length upstream in km."""
        length = flowline_network.calculate_upstream_network_length(102)
        # Reach 102 (60) + Reach 103 (40) + Reach 104 (80) = 180 km
        assert length == pytest.approx(180.0, 0.1)

    def test_get_pnw_high_order_flowlines(self, flowline_network):
        """Filter flowlines by Strahler stream order threshold."""
        order_7_plus = flowline_network.get_pnw_high_order_flowlines(min_stream_order=7)
        assert len(order_7_plus) == 3  # COMID 101 (8), 102 (8), 103 (7)
        order_8_only = flowline_network.get_pnw_high_order_flowlines(min_stream_order=8)
        assert len(order_8_only) == 2


class TestH3SpatialIntegrationAndSurfaceWaterModule:
    """Test H3 indexing and GeoInferSurfaceWater integration."""

    def test_index_to_h3(self, flowline_network):
        """Index flowline network onto H3 cells and verify metrics."""
        h3_metrics = flowline_network.index_to_h3(resolution=7)
        assert len(h3_metrics) > 0
        for data in h3_metrics.values():
            assert "flowline_length_km" in data
            assert data["flowline_length_km"] > 0
            assert "max_stream_order" in data
            assert data["max_stream_order"] >= 5
            assert "river_names" in data
            assert isinstance(data["river_names"], list)

    def test_data_sources_load_pnw_flowlines(self):
        """CascadianSurfaceWaterDataSources correctly loads local NHDPlus HR flowlines."""
        ds = CascadianSurfaceWaterDataSources(flowlines=sample_flowlines())
        gdf = ds.load_pnw_high_order_flowlines(min_stream_order=4)
        assert not gdf.empty
        assert set(gdf["comid"]) == set(
            sample_flowlines().query("stream_order >= 4")["comid"]
        )
        assert "comid" in gdf.columns
        assert "stream_order" in gdf.columns

    def test_data_sources_get_flowline_network(self):
        """CascadianSurfaceWaterDataSources constructs validated network."""
        ds = CascadianSurfaceWaterDataSources(flowlines=sample_flowlines())
        net = ds.get_flowline_network(min_stream_order=5)
        report = net.validate()
        assert report["dag_validation"]["is_dag"] is True
        assert report["edge_count"] == 34
        assert len(net.selected_comids) < len(net.flowlines_gdf)

    def test_geoinfer_surface_water_methods(self, sample_flowlines_gdf):
        """GeoInferSurfaceWater provides network topology and validation APIs."""

        # Minimal mock-free backend dummy
        class MinimalBackend:
            h3_resolution = 8
            target_hexagons: list[str] = []

        sw = GeoInferSurfaceWater(
            MinimalBackend(),
            data_source=CascadianSurfaceWaterDataSources(
                flowlines=sample_flowlines_gdf
            ),
        )  # type: ignore[arg-type]
        assert sw.network is not None
        validation = sw.validate_network_topology()
        assert validation["valid"] is True

        downstream = sw.trace_downstream_flowpath(104)  # Constructed tributary fixture
        assert len(downstream) >= 2
        assert any("Columbia River" in e.get("gnis_name", "") for e in downstream)

        upstream = sw.trace_upstream_tributaries(101)  # Constructed outlet fixture
        assert len(upstream) >= 4


def test_parallel_reaches_preserve_ids_and_traversal(sample_flowlines_gdf):
    """Two channels connecting the same junctions must not overwrite one another."""
    duplicate_channel = sample_flowlines_gdf.iloc[[1]].copy()
    duplicate_channel["comid"] = 105
    import pandas as pd

    net = CascadiaFlowlineNetwork(
        gpd.GeoDataFrame(
            pd.concat([sample_flowlines_gdf, duplicate_channel]), crs="EPSG:4326"
        )
    )
    assert net.graph.number_of_edges() == 5
    assert net.get_flowline_by_comid(102)["comid"] == 102
    assert net.get_flowline_by_comid(105)["comid"] == 105
    assert {r["comid"] for r in net.trace_upstream(101)} == {101, 102, 103, 104, 105}


def test_threshold_does_not_remove_tributary_connectivity(sample_flowlines_gdf):
    source = CascadianSurfaceWaterDataSources(flowlines=sample_flowlines_gdf)
    net = source.get_flowline_network(min_stream_order=8)
    assert net.selected_comids == {101, 102}
    assert {r["comid"] for r in net.trace_upstream(101)} == {101, 102, 103, 104}
    sample_flowlines_gdf.loc[sample_flowlines_gdf["comid"] == 103, "stream_order"] = 4
    source = CascadianSurfaceWaterDataSources(flowlines=sample_flowlines_gdf)
    selected = source.load_pnw_high_order_flowlines(8)
    assert not selected.attrs["full_network_validation"]["valid"]


def test_missing_dataset_is_not_an_empty_network():
    with pytest.raises(FileNotFoundError, match="Supply"):
        CascadianSurfaceWaterDataSources().get_flowline_network()


@pytest.mark.parametrize("column", ["from_node", "comid"])
def test_missing_topology_is_not_invented(sample_flowlines_gdf, column):
    with pytest.raises(ValueError, match="missing"):
        CascadiaFlowlineNetwork(sample_flowlines_gdf.drop(columns=[column]))


def test_duplicate_identifiers_are_rejected(sample_flowlines_gdf):
    sample_flowlines_gdf["comid"] = 101
    with pytest.raises(ValueError, match="duplicate"):
        CascadiaFlowlineNetwork(sample_flowlines_gdf)


def test_multiline_h3_index_retains_all_parts_and_labels_estimates(
    sample_flowlines_gdf,
):
    from shapely.geometry import MultiLineString

    frame = sample_flowlines_gdf.iloc[[0]].copy()
    frame["geometry"] = [
        MultiLineString(
            [[(-124, 41), (-123.99, 41.01)], [(-122, 43), (-121.99, 43.01)]]
        )
    ]
    net = CascadiaFlowlineNetwork(frame)
    indexed = net.index_to_h3(7)
    from geo_infer_space.utils.h3_utils import latlng_to_cell

    assert latlng_to_cell(41, -124, 7) in indexed
    assert latlng_to_cell(43, -122, 7) in indexed
    assert all(
        value["coverage_method"] == "vertex_midpoint_sampling"
        for value in indexed.values()
    )
    assert all(
        value["length_method"] == "equal_share_source_length"
        for value in indexed.values()
    )
    assert sum(
        value["flowline_length_km"] for value in indexed.values()
    ) == pytest.approx(50, abs=0.001)


def test_projected_geojson_never_gets_relabelled_as_wgs84(
    tmp_path, sample_flowlines_gdf
):
    from geo_infer_place.hydrography import load_flowlines

    path = tmp_path / "projected.geojson"
    sample_flowlines_gdf.to_crs("EPSG:3857").to_file(path, driver="GeoJSON")
    with pytest.raises(ValueError, match="WGS84"):
        load_flowlines(path)


def test_upstream_environment_path_selects_local_dataset(tmp_path, monkeypatch):
    path = tmp_path / "smith.geojson"
    sample_flowlines().to_file(path, driver="GeoJSON")
    monkeypatch.setenv("GEO_INFER_CASCADIA_FLOWLINES_PATH", str(path))
    source = CascadianSurfaceWaterDataSources()
    assert source.dataset_path == path
    assert source.get_flowline_network().graph.number_of_edges() == 34


def test_explicit_sources_take_precedence_over_environment(
    tmp_path, monkeypatch, sample_flowlines_gdf
):
    monkeypatch.setenv(
        "GEO_INFER_CASCADIA_FLOWLINES_PATH", str(tmp_path / "absent.geojson")
    )
    explicit = tmp_path / "constructed.geojson"
    sample_flowlines_gdf.to_file(explicit, driver="GeoJSON")
    assert (
        CascadianSurfaceWaterDataSources(explicit)
        .get_flowline_network()
        .graph.number_of_edges()
        == 4
    )
    assert (
        CascadianSurfaceWaterDataSources(flowlines=sample_flowlines_gdf)
        .get_flowline_network()
        .graph.number_of_edges()
        == 4
    )


def test_offline_environment_never_replaces_absence_with_empty_success(monkeypatch):
    monkeypatch.delenv("GEO_INFER_CASCADIA_FLOWLINES_PATH", raising=False)
    monkeypatch.setenv("GEO_INFER_SURFACE_WATER_OFFLINE", "1")
    source = CascadianSurfaceWaterDataSources()
    assert source.offline
    with pytest.raises(FileNotFoundError, match="Offline"):
        source.fetch_surface_water_features((-124.22, 41.90, -124.18, 41.94))


def test_offline_selection_uses_complete_cache_without_network(tmp_path, monkeypatch):
    import hashlib
    import json
    from geo_infer_place.hydrography import HydrographySelection
    from geo_infer_place.hydrography.ingestion import IncompleteHydrographyError

    bbox = (-124.22, 41.90, -124.18, 41.94)
    key = hashlib.sha256(
        json.dumps(HydrographySelection(bbox=bbox).query(), sort_keys=True).encode()
    ).hexdigest()[:20]
    cached = tmp_path / key
    cached.mkdir()
    path = cached / "flowlines.geojson"
    sample_flowlines().to_file(path, driver="GeoJSON")
    manifest = {
        "dataset": "flowlines.geojson",
        "status": "complete",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (cached / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.delenv("GEO_INFER_CASCADIA_FLOWLINES_PATH", raising=False)
    source = CascadianSurfaceWaterDataSources(cache_dir=tmp_path, offline=True)
    assert len(source.fetch_surface_water_features(bbox)["flowlines"]) == 34
    manifest["status"] = "incomplete"
    (cached / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IncompleteHydrographyError, match="not complete"):
        source.fetch_surface_water_features(bbox)
