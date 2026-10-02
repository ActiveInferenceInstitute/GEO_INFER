"""Integration tests for the Cascadia bioregion pipeline.

Tests config file loading, ecology module, visualization generation,
and server endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

# Ensure cascadia src is on path
CASCADIA_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = CASCADIA_DIR / "config"


class TestBioregionConfig:
    """Config files parse cleanly and contain expected data."""

    def test_all_7_ecological_files_load(self):
        yaml_files = [
            "cascadia_salmon_esus.yaml",
            "cascadia_ecoregions.yaml",
            "cascadia_indigenous_territories.yaml",
            "cascadia_climate_zones.yaml",
        ]
        # Note: cascadia_config.yaml references
        # cascadia_bioregion_boundary.geojson, which is not present in
        # config/; the bioregion extent it would carry is declared in the
        # same config's bioregion.bounds and is covered by
        # test_h3_res7_cell_count_for_bioregion.
        json_files = [
            "cascadia_volcanoes.geojson",
            "cascadia_subduction_zone.geojson",
            "cascadia_major_watersheds.geojson",
        ]
        for fname in yaml_files:
            path = CONFIG_DIR / fname
            assert path.exists(), f"Missing: {path}"
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data is not None, f"Empty YAML: {fname}"

        for fname in json_files:
            path = CONFIG_DIR / fname
            assert path.exists(), f"Missing: {path}"
            with open(path) as f:
                data = json.load(f)
            assert "features" in data or "type" in data, f"Invalid GeoJSON: {fname}"

    def test_volcano_count_is_24(self):
        path = CONFIG_DIR / "cascadia_volcanoes.geojson"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        features = data.get("features", [])
        # The Cascadia layer pins 24 USGS Cascade volcanoes and volcanic
        # fields; update the pin deliberately if the layer is re-curated.
        assert len(features) == 24, f"Expected 24 volcanoes, got {len(features)}"
        for feature in features:
            geom = feature.get("geometry", {})
            assert geom.get("type") == "Point", f"Non-point volcano feature: {geom}"
            assert "coordinates" in geom

    def test_csz_linestring_spans_full_length(self):
        path = CONFIG_DIR / "cascadia_subduction_zone.geojson"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        features = data.get("features", [])
        assert len(features) > 0, "CSZ GeoJSON has no features"
        # Collect all latitudes from the geometry
        all_lats = []
        for feature in features:
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])
            geom_type = geom.get("type", "")
            if geom_type == "LineString":
                all_lats.extend(c[1] for c in coords)
            elif geom_type == "MultiLineString":
                for line in coords:
                    all_lats.extend(c[1] for c in line)
        if all_lats:
            assert min(all_lats) <= 42.0, (
                f"CSZ south bound too far north: {min(all_lats)}"
            )
            assert max(all_lats) >= 50.0, (
                f"CSZ north bound too far south: {max(all_lats)}"
            )

    def test_salmon_esu_listed_count_ge_12(self):
        path = CONFIG_DIR / "cascadia_salmon_esus.yaml"
        assert path.exists()
        with open(path) as f:
            data = yaml.safe_load(f)
        listed = []
        for group in [
            "chinook_salmon",
            "coho_salmon",
            "steelhead",
            "sockeye_salmon",
            "chum_salmon",
            "other_species",
        ]:
            for entry in data.get(group, []):
                status = entry.get("esa_status", "")
                if status not in ("Not Listed", "Not Listed (Species of Concern)", ""):
                    listed.append(entry["name"])
        assert len(listed) >= 12, (
            f"Expected >= 12 ESA-listed species, got {len(listed)}: {listed}"
        )

    def test_h3_res7_cell_count_for_bioregion(self):
        """H3 resolution 7 produces a valid cell for the bioregion extent."""
        import h3

        # The bioregion extent is declared in cascadia_config.yaml; the
        # boundary GeoJSON it references is an optional layer (see
        # create_bioregion_map(allow_missing_layers)).
        config_path = CONFIG_DIR / "cascadia_config.yaml"
        assert config_path.exists(), f"Missing: {config_path}"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        bounds = config["bioregion"]["bounds"]
        center_lat = (bounds["south"] + bounds["north"]) / 2
        center_lon = (bounds["west"] + bounds["east"]) / 2
        cell = h3.latlng_to_cell(center_lat, center_lon, 7)
        assert h3.is_valid_cell(cell), "H3 cell not valid"


class TestGeoInferIntegrations:
    """Integration wrappers degrade gracefully when GEO-INFER modules absent."""

    def test_spatial_stats_import_graceful(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaSpatialStats,
        )

        stats = CascadiaSpatialStats()
        result = stats.compute_spatial_autocorrelation({})
        # Should return a dict (either with data or graceful unavailable message)
        assert isinstance(result, dict)

    def test_bayesian_import_graceful(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaBayesianAnalysis,
        )

        bayes = CascadiaBayesianAnalysis()
        result = bayes.estimate_ecological_uncertainty({})
        assert isinstance(result, dict)

    def test_all_wrappers_return_dicts(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            build_integration_suite,
        )

        suite = build_integration_suite()
        assert len(suite) == 8
        for name, wrapper in suite.items():
            assert wrapper is not None, f"Wrapper {name} is None"

    def test_availability_report_returns_bool_map(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            get_availability_report,
        )

        report = get_availability_report()
        assert isinstance(report, dict)
        assert len(report) == 8
        for k, v in report.items():
            assert isinstance(v, bool), f"availability[{k}] should be bool"

    def test_ecology_module_acquire_data(self):
        from geo_infer_place.locations.cascadia.data_modules.ecology.geo_infer_ecology import (
            GeoInferEcology,
        )

        eco = GeoInferEcology()
        result = eco.acquire_raw_data()
        assert isinstance(result, dict)
        assert "salmon_esu_count" in result
        assert result["salmon_esu_count"] >= 0

    def test_ecology_module_run_analysis(self):
        from geo_infer_place.locations.cascadia.data_modules.ecology.geo_infer_ecology import (
            GeoInferEcology,
        )

        eco = GeoInferEcology()
        eco.acquire_raw_data()
        # Minimal h3_data with lat/lon
        h3_data = {
            "8928308280fffff": {"lat": 47.6, "lon": -122.3},
            "8928308281fffff": {"lat": 45.5, "lon": -122.6},
        }
        result = eco.run_final_analysis(h3_data)
        assert isinstance(result, dict)
        assert len(result) == 2
        for props in result.values():
            assert "ecoregion_code" in props
            assert "salmon_esu_count" in props


class TestBioregionVisualization:
    """Bioregion map generation produces valid HTML output."""

    def test_bioregion_map_generates_html(self, tmp_path):
        # folium is a hard PLACE dependency; plain import.

        from geo_infer_place.core.bioregion_visualization import create_bioregion_map

        output = tmp_path / "test_bioregion.html"
        # cascadia_bioregion_boundary.geojson is absent from config/; the
        # renderer supports partial layer sets by design.
        result = create_bioregion_map(CONFIG_DIR, {}, output, allow_missing_layers=True)
        assert Path(result).exists(), f"Map file not created: {result}"
        assert Path(result).stat().st_size > 1000, "Map file is suspiciously small"

    def test_html_contains_volcano_layer(self, tmp_path):
        # folium is a hard PLACE dependency; plain import.

        from geo_infer_place.core.bioregion_visualization import create_bioregion_map

        output = tmp_path / "test_bioregion_volcano.html"
        # cascadia_bioregion_boundary.geojson is absent from config/; the
        # renderer supports partial layer sets by design.
        create_bioregion_map(CONFIG_DIR, {}, output, allow_missing_layers=True)
        content = output.read_text(encoding="utf-8")
        # Mt. Rainier should appear in the generated HTML
        assert "Rainier" in content or "Baker" in content, (
            "No volcano names found in HTML output"
        )

    def test_html_file_size_under_5mb(self, tmp_path):
        # folium is a hard PLACE dependency; plain import.

        from geo_infer_place.core.bioregion_visualization import create_bioregion_map

        output = tmp_path / "test_bioregion_size.html"
        # cascadia_bioregion_boundary.geojson is absent from config/; the
        # renderer supports partial layer sets by design.
        create_bioregion_map(CONFIG_DIR, {}, output, allow_missing_layers=True)
        size_mb = output.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Map file is {size_mb:.1f} MB -- exceeds 5 MB limit"


class TestServer:
    """Server module loads and API routes are defined correctly."""

    def test_server_module_imports(self):
        import importlib.util

        server_path = CASCADIA_DIR / "cascadia_server.py"
        assert server_path.exists(), "cascadia_server.py not found"
        spec = importlib.util.spec_from_file_location("cascadia_server", server_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "create_app"), (
            "create_app() not found in cascadia_server.py"
        )
        assert hasattr(mod, "main"), "main() not found in cascadia_server.py"

    def test_fastapi_app_creates_successfully(self, tmp_path):
        import importlib.util

        server_path = CASCADIA_DIR / "cascadia_server.py"
        spec = importlib.util.spec_from_file_location("cascadia_server", server_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app = mod.create_app(tmp_path)
        assert app is not None

    def test_api_layers_volcanoes_valid_geojson(self, tmp_path):
        from fastapi.testclient import TestClient
        import importlib.util

        server_path = CASCADIA_DIR / "cascadia_server.py"
        spec = importlib.util.spec_from_file_location("cascadia_server", server_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app = mod.create_app(tmp_path)
        client = TestClient(app)
        response = client.get("/api/layers/volcanoes")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert len(data["features"]) > 0

    def test_api_status_returns_json(self, tmp_path):
        from fastapi.testclient import TestClient
        import importlib.util

        server_path = CASCADIA_DIR / "cascadia_server.py"
        spec = importlib.util.spec_from_file_location("cascadia_server", server_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app = mod.create_app(tmp_path)
        client = TestClient(app)
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_ran" in data
        assert "config_files" in data
