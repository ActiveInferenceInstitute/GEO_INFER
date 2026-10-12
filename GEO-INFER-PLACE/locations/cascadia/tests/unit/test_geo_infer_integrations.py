"""Unit tests for GEO-INFER integration wrappers.

Tests graceful degradation when GEO-INFER modules are not installed,
and verifies the integration bridge API surface.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import sys


CASCADIA_DIR = Path(__file__).resolve().parents[2]


class TestCascadiaSpatialStats:
    def test_unavailable_returns_dict_with_available_false(self, monkeypatch):
        """A missing optional extra reports absence without resetting native imports."""
        import geo_infer_place.locations.cascadia.core.geo_infer_integrations as mod

        try:
            with monkeypatch.context() as isolated:
                for name in (
                    "geo_infer_math",
                    "geo_infer_math.core",
                    "geo_infer_math.core.spatial_statistics",
                    "geo_infer_math.core.interpolation",
                ):
                    isolated.setitem(sys.modules, name, None)
                importlib.reload(mod)
                result = mod.CascadiaSpatialStats().compute_spatial_autocorrelation(
                    {"cell1": {"score": 0.5}}
                )
                assert result["available"] is False
        finally:
            importlib.reload(mod)

    def test_available_module_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaSpatialStats,
        )

        stats = CascadiaSpatialStats()
        result = stats.compute_spatial_autocorrelation({})
        assert isinstance(result, dict)

    def test_interpolate_sparse_data_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaSpatialStats,
        )

        stats = CascadiaSpatialStats()
        result = stats.interpolate_sparse_data({}, resolution=7)
        assert isinstance(result, dict)


class TestCascadiaBayesianAnalysis:
    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaBayesianAnalysis,
        )

        bayes = CascadiaBayesianAnalysis()
        result = bayes.estimate_ecological_uncertainty({})
        assert isinstance(result, dict)
        assert "available" in result

    def test_with_sample_data(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaBayesianAnalysis,
        )

        bayes = CascadiaBayesianAnalysis()
        h3_data = {f"cell{i}": {"score": i * 0.1} for i in range(5)}
        result = bayes.estimate_ecological_uncertainty(h3_data)
        assert isinstance(result, dict)


class TestCascadiaSeismicRisk:
    def test_missing_geojson_returns_error_dict(self, tmp_path):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaSeismicRisk,
        )

        risk = CascadiaSeismicRisk()
        missing = tmp_path / "nonexistent.geojson"
        result = risk.compute_csz_hazard(["cell1"], missing)
        assert isinstance(result, dict)
        # Should return unavailable or error, not raise
        assert "available" in result or "error" in result

    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaSeismicRisk,
        )

        risk = CascadiaSeismicRisk()
        result = risk.compute_csz_hazard([], Path("/nonexistent/path.geojson"))
        assert isinstance(result, dict)


class TestCascadiaForestHealth:
    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaForestHealth,
        )

        fh = CascadiaForestHealth()
        result = fh.assess_forest_health({}, {})
        assert isinstance(result, dict)
        assert "available" in result


class TestCascadiaCoastalAnalysis:
    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaCoastalAnalysis,
        )

        ca = CascadiaCoastalAnalysis()
        result = ca.assess_coastal_resilience({})
        assert isinstance(result, dict)
        assert "available" in result


class TestCascadiaEcosystemServices:
    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaEcosystemServices,
        )

        es = CascadiaEcosystemServices()
        result = es.value_ecosystem_services({}, {})
        assert isinstance(result, dict)
        assert "available" in result


class TestCascadiaDataQuality:
    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaDataQuality,
        )

        dq = CascadiaDataQuality()
        result = dq.validate_module_outputs({"module_a": {"score": 0.8}})
        assert isinstance(result, dict)
        assert "available" in result

    def test_validates_h3_hexagon_outputs_with_geo_infer_data(self):
        import h3
        import pytest

        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaDataQuality,
        )

        center = h3.latlng_to_cell(41.75, -124.2, 7)
        hexagons = {cell: {"score": 0.5} for cell in sorted(h3.grid_disk(center, 1))}
        result = CascadiaDataQuality().validate_module_outputs(
            {
                "zoning": {"hexagons": hexagons},
                "raw_cells": hexagons,
                "summary": {"score": 0.8},
            }
        )
        assert result["available"] is True
        scores = result["quality_scores"]
        for name in ("zoning", "raw_cells"):
            assert scores[name]["status"] == "pass"
            assert scores[name]["score"] == pytest.approx(1.0)
            assert scores[name]["n_features"] == len(hexagons)
        assert scores["summary"]["status"] == "skipped"

    def test_reports_invalid_geometry(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaDataQuality,
        )

        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        frame = gpd.GeoDataFrame({"score": [0.1]}, geometry=[bowtie], crs="EPSG:4326")
        result = CascadiaDataQuality().validate_module_outputs({"bad": frame})
        check = result["quality_scores"]["bad"]
        assert check["score"] < 1.0
        assert any(issue["type"] == "invalid_geometry" for issue in check["issues"])


class TestCascadiaClimateAnalysis:
    def test_missing_yaml_returns_error_dict(self, tmp_path):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaClimateAnalysis,
        )

        ca = CascadiaClimateAnalysis()
        result = ca.assign_climate_zones({}, tmp_path / "nonexistent.yaml")
        assert isinstance(result, dict)
        assert "available" in result

    def test_returns_dict(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            CascadiaClimateAnalysis,
        )

        ca = CascadiaClimateAnalysis()
        result = ca.assign_climate_zones({}, Path("/nonexistent/climate.yaml"))
        assert isinstance(result, dict)


class TestIntegrationSuite:
    def test_build_integration_suite_returns_8_wrappers(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            build_integration_suite,
        )

        suite = build_integration_suite()
        expected_keys = {
            "spatial_stats",
            "bayesian",
            "seismic_risk",
            "forest_health",
            "coastal",
            "ecosystem_services",
            "data_quality",
            "climate",
        }
        assert set(suite.keys()) == expected_keys

    def test_get_availability_report_all_bool(self):
        from geo_infer_place.locations.cascadia.core.geo_infer_integrations import (
            get_availability_report,
        )

        report = get_availability_report()
        assert len(report) == 8
        for k, v in report.items():
            assert isinstance(v, bool), f"{k}: expected bool, got {type(v)}"
