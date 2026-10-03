"""Real file, geometry and failure oracles for the owning SPACE orchestrator."""

from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import h3
import pytest
from pyproj import Transformer
from shapely.geometry import Point, Polygon, LineString, box

from geo_infer_space.core.base_module import BaseAnalysisModule


class ConcreteModule(BaseAnalysisModule):
    def __init__(self, raw_path: Path, output_dir: Path):
        super().__init__("observations", h3_resolution=8, output_dir=output_dir)
        self.raw_path = raw_path
        self.acquisitions = 0

    def acquire_raw_data(self) -> Path:
        self.acquisitions += 1
        return self.raw_path

    def run_final_analysis(self, h3_data: dict) -> dict:
        return deepcopy(h3_data)


@pytest.fixture
def point_module(tmp_path):
    path = tmp_path / "point.geojson"
    gpd.GeoDataFrame(
        {"value": [0]}, geometry=[Point(-122.4, 37.7)], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")
    return ConcreteModule(path, tmp_path / "cache")


def test_run_analysis_real_point_cache_and_explicit_refresh(point_module):
    module = point_module
    expected = h3.latlng_to_cell(37.7, -122.4, 8)
    result = module.run_analysis()
    assert set(result) == {expected}
    feature = result[expected]["observations"][0]
    assert feature["properties"] == {"value": 0}
    assert feature["geometry"]["type"] == "Point"
    assert feature["geometry"]["coordinates"] == (-122.4, 37.7)
    assert module.acquisitions == 1 and module._validate_cache_file(
        module.h3_cache_path
    )
    result[expected]["observations"][0]["properties"]["value"] = 99
    cached = module.run_analysis()
    assert cached[expected]["observations"][0]["properties"]["value"] == 0
    assert module.acquisitions == 1
    gpd.GeoDataFrame(
        {"value": [4]}, geometry=[Point(-121, 38)], crs="EPSG:4326"
    ).to_file(module.raw_path, driver="GeoJSON")
    refreshed = module.run_analysis(use_cache=False)
    assert set(refreshed) == {h3.latlng_to_cell(38, -121, 8)}
    assert module.acquisitions == 2
    assert not list(module.output_dir.glob("*.tmp"))
    with pytest.raises(TypeError):
        module.run_analysis(use_cache="yes")


def test_projected_vector_point_maps_to_actual_wgs84_cell(tmp_path):
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(
        -122.4, 37.7
    )
    path = tmp_path / "projected.gpkg"
    gpd.GeoDataFrame({"value": [0]}, geometry=[Point(x, y)], crs="EPSG:3857").to_file(
        path, driver="GPKG"
    )
    module = ConcreteModule(path, tmp_path / "cache")
    result = module.process_to_h3(path)
    assert set(result) == {h3.latlng_to_cell(37.7, -122.4, 8)}
    coordinates = next(iter(result.values()))["observations"][0]["geometry"][
        "coordinates"
    ]
    assert coordinates == pytest.approx((-122.4, 37.7), abs=1e-8)


def test_polygon_holes_and_multipolygon_use_native_center_containment(tmp_path):
    shell = box(-122.46, 37.65, -122.34, 37.76)
    hole = box(-122.42, 37.69, -122.38, 37.73)
    polygon = Polygon(shell.exterior.coords, [hole.exterior.coords])
    second = box(-122.30, 37.65, -122.27, 37.68)
    from shapely.geometry import MultiPolygon

    geometry = MultiPolygon([polygon, second])
    path = tmp_path / "polygons.geojson"
    gpd.GeoDataFrame({"value": [0]}, geometry=[geometry], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    module = ConcreteModule(path, tmp_path / "cache")
    result = module.process_to_h3(path)
    assert set(result) == set(h3.geo_to_cells(geometry.__geo_interface__, 8))
    assert set(h3.geo_to_cells(hole.__geo_interface__, 8)).isdisjoint(result)
    assert (
        len(
            next(iter(result.values()))["observations"][0]["geometry"]["coordinates"][0]
        )
        == 2
    )


def test_orchestration_failures_propagate_and_preserve_invalid_cache(
    point_module, monkeypatch
):
    module = point_module
    cache_bytes = b'{"fabricated_cell": {}}'
    module.h3_cache_path.write_bytes(cache_bytes)
    assert not module._validate_cache_file(module.h3_cache_path)
    module.raw_path = module.raw_path.with_name("missing.geojson")
    with pytest.raises(FileNotFoundError):
        module.run_analysis()
    assert module.h3_cache_path.read_bytes() == cache_bytes

    def failed_processing(path):
        raise h3.H3FailedError("Unsupported native topology")

    monkeypatch.setattr(module, "_direct_h3_processing", failed_processing)
    with pytest.raises(h3.H3FailedError, match="Unsupported native topology"):
        module.process_to_h3(module.raw_path)


def test_invalid_crs_and_unsupported_geometry_are_explicit(point_module, monkeypatch):
    module = point_module
    source = gpd.GeoDataFrame({"value": [0]}, geometry=[Point(-122.4, 37.7)])
    monkeypatch.setattr(gpd, "read_file", lambda path: source)
    with pytest.raises(ValueError, match="reference system"):
        module.process_to_h3(module.raw_path)
    line = gpd.GeoDataFrame(
        {"value": [0]}, geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326"
    )
    monkeypatch.setattr(gpd, "read_file", lambda path: line)
    with pytest.raises(ValueError, match="Unsupported vector geometry"):
        module.process_to_h3(module.raw_path)


def test_failed_cache_write_never_replaces_existing_bytes(point_module, monkeypatch):
    module = point_module
    cache_bytes = b"invalid existing cache retained for diagnosis"
    module.h3_cache_path.write_bytes(cache_bytes)

    def failed_sync(descriptor):
        raise OSError("Disk write failed")

    monkeypatch.setattr("geo_infer_space.core.base_module.os.fsync", failed_sync)
    with pytest.raises(OSError, match="Disk write failed"):
        module.run_analysis()
    assert module.h3_cache_path.read_bytes() == cache_bytes
    assert not list(module.output_dir.glob(".*.tmp"))
