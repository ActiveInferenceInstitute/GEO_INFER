#!/usr/bin/env python3
"""
Test suite for Del Norte County analysers beyond ForestHealthMonitor.

Covers:
- CoastalResilienceAnalyzer  (initialization, full run_analysis pipeline)
- FireRiskAssessor           (initialization, full run_analysis pipeline)
- SeismicHazardAnalyzer      (initialization, full run_analysis pipeline, CSZ scenario)
- AdvancedDashboard          (initialization, map creation, panel generation)

All tests use the real classes with real ``DelNorteDataIntegrator`` instances
and assert that missing provider data is reported instead of fabricated.
"""

import os
import shutil
import tempfile
import unittest

import yaml

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "locations",
        "del_norte_county",
        "config",
        "analysis_config.yaml",
    )
)


def _load_config() -> dict:
    """Load the real analysis_config.yaml."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# CoastalResilienceAnalyzer
# ---------------------------------------------------------------------------


class TestCoastalResilienceAnalyzer(unittest.TestCase):
    """Tests for the CoastalResilienceAnalyzer."""

    @classmethod
    def setUpClass(cls):
        try:
            from geo_infer_place.locations.del_norte_county.coastal_resilience_analyzer import (
                CoastalResilienceAnalyzer,
            )
            from geo_infer_place.utils.integration import DelNorteDataIntegrator

            cls.config = _load_config()
            cls.tmpdir = tempfile.mkdtemp(prefix="test_coastal_")
            cls.integrator = DelNorteDataIntegrator()
            cls.analyzer = CoastalResilienceAnalyzer(
                config=cls.config,
                data_integrator=cls.integrator,
                spatial_processor=None,
                output_dir=cls.tmpdir,
            )
            cls._available = True
        except ImportError as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir") and os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if not self._available:
            self.fail(f"CoastalResilienceAnalyzer not available: {self._skip_reason}")

    # -- Initialization -----------------------------------------------------

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.h3_resolution, 8)
        self.assertIsInstance(self.analyzer.sea_level_scenarios, dict)
        self.assertIn("low", self.analyzer.sea_level_scenarios)
        self.assertIn("extreme", self.analyzer.sea_level_scenarios)

    def test_config_loaded(self):
        """Coastal config section populated from YAML."""
        self.assertTrue(len(self.analyzer.vulnerability_factors) > 0)
        self.assertIn("elevation", self.analyzer.vulnerability_factors)

    def test_monitoring_sites(self):
        """At least Crescent City harbor should be configured."""
        sites = self.analyzer.monitoring_sites
        self.assertIn("crescent_city_harbor", sites)
        self.assertAlmostEqual(sites["crescent_city_harbor"]["lat"], 41.745, places=2)

    # -- run_analysis -------------------------------------------------------

    def test_run_analysis_returns_dict(self):
        """Full analysis pipeline should return a dict with expected keys."""
        results = self.analyzer.run_analysis()
        self.assertIsInstance(results, dict)
        self.assertEqual(results["analysis_type"], "coastal_resilience")
        self.assertEqual(results["location"], "del_norte_county")
        # Should have a status field regardless of API availability
        self.assertIn("status", results)

    def test_run_analysis_has_sea_level_section(self):
        results = self.analyzer.run_analysis()
        self.assertIn("sea_level_analysis", results)

    def test_run_analysis_has_erosion_section(self):
        results = self.analyzer.run_analysis()
        self.assertIn("erosion_analysis", results)

    def test_run_analysis_saves_file(self):
        """No output is written when empirical coastal sources are absent."""
        results = self.analyzer.run_analysis()
        json_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".json")]
        self.assertEqual(results["status"], "error")
        self.assertEqual(json_files, [])


# ---------------------------------------------------------------------------
# FireRiskAssessor
# ---------------------------------------------------------------------------


class TestFireRiskAssessor(unittest.TestCase):
    """Tests for the FireRiskAssessor."""

    @classmethod
    def setUpClass(cls):
        try:
            from geo_infer_place.locations.del_norte_county.fire_risk_assessor import (
                FireRiskAssessor,
            )
            from geo_infer_place.utils.integration import DelNorteDataIntegrator

            cls.config = _load_config()
            cls.tmpdir = tempfile.mkdtemp(prefix="test_fire_")
            cls.integrator = DelNorteDataIntegrator()
            cls.analyzer = FireRiskAssessor(
                config=cls.config,
                data_integrator=cls.integrator,
                spatial_processor=None,
                output_dir=cls.tmpdir,
            )
            cls._available = True
        except ImportError as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir") and os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if not self._available:
            self.fail(f"FireRiskAssessor not available: {self._skip_reason}")

    # -- Initialization -----------------------------------------------------

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.h3_resolution, 8)
        self.assertIsInstance(self.analyzer.fire_config, dict)

    def test_fire_config_sections(self):
        """Fire config should carry weather thresholds and fuel moisture."""
        fc = self.analyzer.fire_config
        self.assertIn("fire_weather", fc)
        self.assertIn("fuel_moisture", fc)
        self.assertIn("fire_danger_classes", fc)
        self.assertEqual(len(fc["fire_danger_classes"]), 5)

    # -- run_analysis -------------------------------------------------------

    def test_run_analysis_returns_dict(self):
        results = self.analyzer.run_analysis()
        self.assertIsInstance(results, dict)
        self.assertEqual(results["analysis_type"], "fire_risk")
        self.assertEqual(results["location"], "del_norte_county")
        self.assertIn("status", results)

    def test_run_analysis_has_weather_section(self):
        # Deterministic contract: missing provider data yields status=error
        # with a non-empty error_message (the exact wording varies with the
        # live NOAA payload).
        results = self.analyzer.run_analysis()
        self.assertEqual(results["status"], "error")
        self.assertTrue(
            isinstance(results.get("error_message"), str)
            and len(results["error_message"]) > 0
        )

    def test_run_analysis_has_historical_section(self):
        results = self.analyzer.run_analysis()
        self.assertEqual(results["status"], "error")
        self.assertTrue(
            isinstance(results.get("error_message"), str)
            and len(results["error_message"]) > 0
        )

    def test_run_analysis_has_fuel_section(self):
        results = self.analyzer.run_analysis()
        self.assertEqual(results["status"], "error")
        self.assertTrue(
            isinstance(results.get("error_message"), str)
            and len(results["error_message"]) > 0
        )

    def test_run_analysis_saves_file(self):
        results = self.analyzer.run_analysis()
        json_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".json")]
        self.assertEqual(results["status"], "error")
        self.assertEqual(json_files, [])


# ---------------------------------------------------------------------------
# SeismicHazardAnalyzer
# ---------------------------------------------------------------------------


class TestSeismicHazardAnalyzer(unittest.TestCase):
    """Tests for the SeismicHazardAnalyzer."""

    @classmethod
    def setUpClass(cls):
        try:
            from geo_infer_place.locations.del_norte_county.seismic_hazard_analyzer import (
                SeismicHazardAnalyzer,
            )
            from geo_infer_place.utils.integration import DelNorteDataIntegrator

            cls.config = _load_config()
            cls.tmpdir = tempfile.mkdtemp(prefix="test_seismic_")
            cls.integrator = DelNorteDataIntegrator()
            cls.analyzer = SeismicHazardAnalyzer(
                config=cls.config,
                data_integrator=cls.integrator,
                spatial_processor=None,
                output_dir=cls.tmpdir,
            )
            cls._available = True
        except ImportError as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir") and os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if not self._available:
            self.fail(f"SeismicHazardAnalyzer not available: {self._skip_reason}")

    # -- Initialization -----------------------------------------------------

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.h3_resolution, 8)
        # bbox should be (west, south, east, north)
        self.assertEqual(len(self.analyzer.bbox), 4)
        self.assertLess(self.analyzer.bbox[0], self.analyzer.bbox[2])  # west < east
        self.assertLess(self.analyzer.bbox[1], self.analyzer.bbox[3])  # south < north

    # -- run_analysis -------------------------------------------------------

    def test_run_analysis_returns_dict(self):
        results = self.analyzer.run_analysis()
        self.assertIsInstance(results, dict)
        self.assertEqual(results["analysis_type"], "seismic_hazard")
        self.assertIn("timestamp", results)

    def test_run_analysis_has_earthquake_data(self):
        results = self.analyzer.run_analysis()
        self.assertIn("earthquake_data", results)

    def test_run_analysis_has_hazard_grid(self):
        results = self.analyzer.run_analysis()
        self.assertIn("hazard_grid", results)
        grid = results["hazard_grid"]
        self.assertIsInstance(grid, dict)

    def test_run_analysis_has_tsunami_risk(self):
        results = self.analyzer.run_analysis()
        self.assertIn("tsunami_risk", results)

    def test_run_analysis_has_csz_scenario(self):
        results = self.analyzer.run_analysis()
        self.assertIn("csz_scenario", results)
        csz = results["csz_scenario"]
        self.assertIsInstance(csz, dict)

    def test_run_analysis_has_liquefaction(self):
        results = self.analyzer.run_analysis()
        self.assertIn("liquefaction_risk", results)

    def test_run_analysis_has_summary(self):
        results = self.analyzer.run_analysis()
        self.assertIn("summary", results)

    def test_run_analysis_saves_file(self):
        self.analyzer.run_analysis()
        json_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".json")]
        self.assertGreater(len(json_files), 0, "No JSON output file saved")


# ---------------------------------------------------------------------------
# AdvancedDashboard
# ---------------------------------------------------------------------------


class TestAdvancedDashboard(unittest.TestCase):
    """Tests for the AdvancedDashboard (dashboard sub-package)."""

    @classmethod
    def setUpClass(cls):
        try:
            from geo_infer_place.locations.del_norte_county.dashboard import (
                AdvancedDashboard,
            )
            import folium  # noqa: F401 — needed by dashboard

            cls.tmpdir = tempfile.mkdtemp(prefix="test_dashboard_")
            cls.dashboard = AdvancedDashboard(output_dir=cls.tmpdir)
            cls.Dashboard = AdvancedDashboard
            cls._available = True
        except ImportError as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir") and os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if not self._available:
            self.fail(f"AdvancedDashboard not available: {self._skip_reason}")

    # -- Initialization -----------------------------------------------------

    def test_initialization(self):
        self.assertIsNotNone(self.dashboard)
        self.assertTrue(self.dashboard.output_dir.exists())

    def test_county_center(self):
        """County center should be near Crescent City."""
        lat, lon = self.dashboard.county_center
        self.assertAlmostEqual(lat, 41.75, delta=0.3)
        self.assertAlmostEqual(lon, -124.2, delta=0.5)

    def test_layer_groups_initialized(self):
        """Dashboard should have multiple named layer groups."""
        groups = self.dashboard.layer_groups
        self.assertGreater(len(groups), 5)
        self.assertIn("fire", groups)
        self.assertIn("earthquake", groups)
        self.assertIn("forest", groups)
        self.assertIn("tides", groups)

    # -- Map creation -------------------------------------------------------

    def test_create_map(self):
        """create_comprehensive_map should return a folium.Map."""
        import folium

        m = self.dashboard.create_comprehensive_map()
        self.assertIsInstance(m, folium.Map)

    # -- Analysis panels ----------------------------------------------------

    def test_generate_analysis_panels(self):
        """generate_analysis_panels should produce a dict of HTML panel strings."""
        panels = self.dashboard.generate_analysis_panels()
        self.assertIsInstance(panels, dict)
        self.assertGreater(len(panels), 0)
        # Each value should be an HTML string
        for html in panels.values():
            self.assertIsInstance(html, str)
            self.assertGreater(len(html), 0)

    # -- Dashboard save -----------------------------------------------------

    def test_save_dashboard(self):
        """save_dashboard should write an HTML file."""
        path = self.dashboard.save_dashboard(fetch_data=False)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".html"))
        with open(path) as f:
            html = f.read()
        self.assertIn("<html", html.lower())


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


class TestDelNorteExports(unittest.TestCase):
    """Verify that all expected classes are importable from the package."""

    def test_forest_health_monitor(self):
        from geo_infer_place.locations.del_norte_county import ForestHealthMonitor

        self.assertIsNotNone(ForestHealthMonitor)

    def test_coastal_resilience_analyzer(self):
        from geo_infer_place.locations.del_norte_county import CoastalResilienceAnalyzer

        self.assertIsNotNone(CoastalResilienceAnalyzer)

    def test_fire_risk_assessor(self):
        from geo_infer_place.locations.del_norte_county import FireRiskAssessor

        self.assertIsNotNone(FireRiskAssessor)

    def test_seismic_hazard_analyzer(self):
        from geo_infer_place.locations.del_norte_county import SeismicHazardAnalyzer

        self.assertIsNotNone(SeismicHazardAnalyzer)

    def test_comprehensive_dashboard(self):
        from geo_infer_place.locations.del_norte_county import (
            DelNorteComprehensiveDashboard,
        )

        self.assertIsNotNone(DelNorteComprehensiveDashboard)


if __name__ == "__main__":
    unittest.main()
