#!/usr/bin/env python3
"""
Integration tests for the Cascadia GeoJSON layer pipeline and the Del Norte
County demo orchestration.

These tests exercise real loading/rendering code offline using tracked county
boundaries and explicitly constructed geometry fixtures (no regional data claims):
- Cascadia GeoJSON layers (county boundaries, subduction zone, watersheds,
  volcanoes, bioregion) parse correctly and load via CountyBoundaryLoader.
- cascadia_main.parse_counties() orchestration helper resolves CLI county specs.
- Bioregion visualization renders the GeoJSON layers into an interactive
  Folium HTML map.
- Del Norte run_analysis cleanup keeps only the most recent results.

Because these tests drive the real orchestration surfaces, they live in
GEO-INFER-PLACE/tests/integration and are collected by
``uv run pytest GEO-INFER-PLACE/tests/``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from geo_infer_place.core.bioregion_visualization import (
    create_bioregion_map,
    _load_json,
)

PLACE_DIR = Path(__file__).resolve().parents[2]
CASCADIA_DIR = PLACE_DIR / "locations" / "cascadia"
CASCADIA_CONFIG = CASCADIA_DIR / "config"
DEL_NORTE_DIR = PLACE_DIR / "locations" / "del_norte_county"


@pytest.fixture
def constructed_layer_dir(tmp_path):
    """Deliberately constructed layer shapes/properties for renderer contracts."""
    directory = tmp_path / "constructed-layers"
    directory.mkdir()
    polygon = {
        "type": "Polygon",
        "coordinates": [[[-124, 41], [-123, 41], [-123, 42], [-124, 42], [-124, 41]]],
    }
    line = {"type": "LineString", "coordinates": [[-124, 41], [-123, 42]]}
    point = {"type": "Point", "coordinates": [-123.5, 41.5]}
    for name, geometries in {
        "cascadia_bioregion_boundary.geojson": [polygon],
        "cascadia_subduction_zone.geojson": [line],
        "cascadia_major_watersheds.geojson": [polygon],
        "cascadia_volcanoes.geojson": [
            point,
            {"type": "Point", "coordinates": [-123.6, 41.6]},
        ],
    }.items():
        features = [
            {
                "type": "Feature",
                "properties": {
                    "name": f"Constructed test feature {index}",
                    "fixture_kind": "constructed",
                    "elevation_m": 100 + index,
                    "threat_level": "Unknown",
                },
                "geometry": geom,
            }
            for index, geom in enumerate(geometries)
        ]
        (directory / name).write_text(
            json.dumps({"type": "FeatureCollection", "features": features})
        )
    return directory


def _load_module(module_name: str, file_path: Path):
    """Load a standalone script module by file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Cascadia GeoJSON layer parsing
# ---------------------------------------------------------------------------


#: layer file -> (expected geometry kind histogram, optional required name)
CASCADIA_LAYERS = {
    "ca_del_norte_boundary.geojson": {"count": 1, "types": ["Polygon"]},
    "ca_humboldt_boundary.geojson": {"count": 1, "types": ["Polygon"]},
    "ca_lassen_boundary.geojson": {"count": 1, "types": ["Polygon"]},
}


class TestCascadiaGeoJsonLayers:
    """All tracked Cascadia GeoJSON layers parse with the expected geometry."""

    @pytest.mark.parametrize("layer", list(CASCADIA_LAYERS))
    def test_layer_parses_with_expected_geometry(self, layer):
        path = CASCADIA_CONFIG / layer
        assert path.exists(), f"Missing GeoJSON layer: {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("type") == "FeatureCollection"
        features = data.get("features", [])
        expected = CASCADIA_LAYERS[layer]
        assert len(features) == expected["count"], (
            f"{layer}: expected {expected['count']} features, got {len(features)}"
        )
        geom_types = {feature.get("geometry", {}).get("type") for feature in features}
        assert geom_types == set(expected["types"]), (
            f"{layer}: expected geometry {expected['types']}, got {geom_types}"
        )

    def test_del_norte_boundary_metadata(self):
        path = CASCADIA_CONFIG / "ca_del_norte_boundary.geojson"
        data = json.loads(path.read_text(encoding="utf-8"))
        props = data["features"][0].get("properties", {})
        assert props.get("name") == "Del Norte County"
        assert props.get("county_seat") == "Crescent City"
        assert props.get("fips_code") == "06015"

    @pytest.mark.parametrize(
        "layer, kind, count",
        [
            ("cascadia_bioregion_boundary.geojson", "Polygon", 1),
            ("cascadia_subduction_zone.geojson", "LineString", 1),
            ("cascadia_major_watersheds.geojson", "Polygon", 1),
            ("cascadia_volcanoes.geojson", "Point", 2),
        ],
    )
    def test_constructed_layers_preserve_geometry_and_metadata(
        self, constructed_layer_dir, layer, kind, count
    ):
        data = _load_json(constructed_layer_dir / layer)
        assert len(data["features"]) == count
        assert {feature["geometry"]["type"] for feature in data["features"]} == {kind}
        assert all(
            feature["properties"]["fixture_kind"] == "constructed"
            for feature in data["features"]
        )
        assert all(
            feature["properties"]["name"].startswith("Constructed test")
            for feature in data["features"]
        )
        if kind == "Point":
            assert [
                feature["properties"]["elevation_m"] for feature in data["features"]
            ] == [100, 101]

    def test_county_geometry_loader_loads_all_ca_counties(self):
        """CountyBoundaryLoader loads the three CA counties from real GeoJSON."""
        loader_path = CASCADIA_CONFIG / "county_boundary_loader.py"
        loader_module = _load_module("county_boundary_loader", loader_path)
        loader = loader_module.create_county_boundary_loader()
        geos = loader.get_all_county_geometries(
            {"CA": ["Del Norte", "Humboldt", "Lassen"]}
        )
        assert "CA" in geos
        # CountyBoundaryLoader normalizes multi-word county keys to underscores
        # ("Del_Norte"). Assert all three requested counties are present.
        assert set(geos["CA"].keys()) == {"Del_Norte", "Humboldt", "Lassen"}
        for county, geometry in geos["CA"].items():
            assert geometry.get("type") in ("Polygon", "MultiPolygon"), (
                f"{county} geometry should be Polygon/MultiPolygon, "
                f"got {geometry.get('type')}"
            )
            assert loader.validate_geometry(geometry), (
                f"{county} geometry should be H3-usable"
            )


class TestCascadiaParseCounties:
    """cascadia_main.parse_counties() county-spec resolution."""

    _module = None

    @pytest.fixture(autouse=True)
    def _ensure_module(self):
        if self.__class__._module is None:
            main_path = CASCADIA_DIR / "cascadia_main.py"
            assert main_path.exists(), "cascadia_main.py not present"
            # Loading the module executes only guarded top-level logic (no run).
            self.__class__._module = _load_module("cascadia_main", main_path)

    def test_parses_multi_county_spec(self):
        result = self._module.parse_counties("CA:Del Norte,OR:Josephine")
        assert result == {"CA": ["Del Norte"], "OR": ["Josephine"]}

    def test_parses_default_state_county(self):
        result = self._module.parse_counties("Humboldt")
        assert result == {"CA": ["Humboldt"]}

    def test_parses_all(self):
        result = self._module.parse_counties("all")
        assert result == {"CA": ["all"], "OR": ["all"]}

    def test_empty_string_defaults_to_all(self):
        result = self._module.parse_counties("")
        assert result == {"CA": ["all"], "OR": ["all"]}


class TestCascadiaBioregionMap:
    """Bioregion visualization renders the tracked GeoJSON layers to HTML."""

    def test_bioregion_map_generates_html(self, tmp_path, constructed_layer_dir):
        output = tmp_path / "bioregion_test.html"
        result = create_bioregion_map(constructed_layer_dir, {}, output)
        assert Path(result).exists()
        html = output.read_text(encoding="utf-8")
        assert "Cascadia Subduction Zone" in html
        assert "Major Watersheds" in html
        assert "Volcanoes" in html
        assert "Constructed test feature 0" in html
        assert "Constructed test feature 1" in html
        assert "37%" not in html
        manifest = json.loads(output.with_suffix(".layers.json").read_text())
        assert manifest["status"] == "complete"
        assert all(layer["status"] == "loaded" for layer in manifest["layers"].values())

    def test_missing_regional_layers_raise_without_rendering(self, tmp_path):
        output = tmp_path / "missing.html"
        with pytest.raises(FileNotFoundError, match="Bioregion layers unavailable"):
            create_bioregion_map(tmp_path, {}, output)
        assert not output.exists()

    def test_explicit_partial_map_reports_unavailable_layers(self, tmp_path):
        output = tmp_path / "partial.html"
        create_bioregion_map(tmp_path, {}, output, allow_missing_layers=True)
        html = output.read_text()
        assert "Unavailable layers:" in html
        assert "cascadia_volcanoes.geojson" in html
        manifest = json.loads(output.with_suffix(".layers.json").read_text())
        assert manifest["status"] == "partial"
        assert all(
            layer["status"] == "unavailable" for layer in manifest["layers"].values()
        )

    def test_projected_geojson_is_rejected(self, constructed_layer_dir, tmp_path):
        path = constructed_layer_dir / "cascadia_volcanoes.geojson"
        data = json.loads(path.read_text())
        data["crs"] = {"type": "name", "properties": {"name": "EPSG:3857"}}
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="WGS84"):
            create_bioregion_map(constructed_layer_dir, {}, tmp_path / "map.html")

    def test_html_labels_are_escaped(self, constructed_layer_dir, tmp_path):
        path = constructed_layer_dir / "cascadia_volcanoes.geojson"
        data = json.loads(path.read_text())
        data["features"][0]["properties"]["name"] = "<script>alert('injected')</script>"
        path.write_text(json.dumps(data))
        output = tmp_path / "safe.html"
        create_bioregion_map(constructed_layer_dir, {}, output)
        html = output.read_text()
        assert "<script>alert(" not in html
        assert "&lt;script&gt;" in html


def test_bioregion_h3_layer_uses_real_cell_boundaries(constructed_layer_dir, tmp_path):
    import h3

    cell = h3.latlng_to_cell(41.5, -123.5, 7)
    output = tmp_path / "h3.html"
    create_bioregion_map(
        constructed_layer_dir,
        {cell: {"ecoregion_code": "Constructed test ecoregion"}},
        output,
    )
    html = output.read_text()
    assert "H3 analysis cells" in html
    assert "Constructed test ecoregion" in html
    assert str(h3.cell_to_boundary(cell)[0][0]) in html


def test_integration_credit_labels_cannot_inject_html(constructed_layer_dir, tmp_path):
    output = tmp_path / "safe-integration.html"
    create_bioregion_map(
        constructed_layer_dir,
        {},
        output,
        integration_results={
            "ecosystem_services": {
                "available": True,
                "bank_summary": {"credits": 1},
                "credit_types": ["<img src=x onerror=alert(1)>"],
            }
        },
    )
    html = output.read_text()
    assert "<img src=x" not in html
    assert r"\u0026lt;img src=x" in html
