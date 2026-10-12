#!/usr/bin/env python3
"""
Test suite for GEO-INFER-PLACE PlaceAnalyzer

This module tests the core place-based analysis functionality including
location management, analysis workflows, and integration capabilities.
"""

import os
import sys
import unittest

import yaml


class TestPlaceAnalyzer(unittest.TestCase):
    """Tests for the PlaceInterface API (replaces phantom PlaceAnalyzer tests)."""

    def test_place_interface_is_primary_entry_point(self):
        """PlaceInterface is the primary API for geo_infer_place."""
        from geo_infer_place import PlaceInterface

        pi = PlaceInterface("del_norte")
        self.assertIsNotNone(pi)

    def test_get_supported_locations_exists(self):
        """get_supported_locations() returns a non-empty list."""
        from geo_infer_place import get_supported_locations

        locs = get_supported_locations()
        self.assertIsInstance(locs, list)
        self.assertGreater(len(locs), 0)

    def test_create_analyzer_returns_place_interface(self):
        """create_analyzer() returns a PlaceInterface instance."""
        from geo_infer_place import create_analyzer, PlaceInterface

        pi = create_analyzer("del_norte")
        self.assertIsInstance(pi, PlaceInterface)

    def test_module_version_exists(self):
        """geo_infer_place exposes a __version__ string."""
        import geo_infer_place

        self.assertTrue(hasattr(geo_infer_place, "__version__"))
        self.assertIsInstance(geo_infer_place.__version__, str)

    def test_configuration_loading(self):
        """Del Norte County analysis_config.yaml exists and is valid."""
        del_norte_config = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "locations",
            "del_norte_county",
            "config",
            "analysis_config.yaml",
        )
        if not os.path.exists(del_norte_config):
            self.fail("Del Norte County config file not found")
        import yaml

        with open(del_norte_config) as f:
            config = yaml.safe_load(f)
        self.assertIn("location", config)
        self.assertIn("analyses", config)


class TestLocationConfigurations(unittest.TestCase):
    """Test suite for location-specific configurations."""

    def test_del_norte_county_config(self):
        """Test Del Norte County configuration structure."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "locations",
            "del_norte_county",
            "config",
            "analysis_config.yaml",
        )

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Test required configuration sections
            self.assertIn("location", config)
            self.assertIn("analyses", config)

            # Test location metadata
            location = config["location"]
            self.assertIn("name", location)
            self.assertIn("bounds", location)
            self.assertIn("coordinate_systems", location)

            # Test analysis configurations
            analyses = config["analyses"]
            expected_analyses = [
                "forest_health",
                "coastal_resilience",
                "fire_risk",
                "community_development",
            ]

            for analysis in expected_analyses:
                self.assertIn(
                    analysis,
                    analyses,
                    f"Analysis '{analysis}' should be configured",
                )

        else:
            self.fail("Del Norte County config file not found")

    def test_requirements_files_exist(self):
        """Test that location-specific requirements files exist."""
        locations = ["del_norte_county", "australia", "siberia"]

        # Locations live at project root: GEO-INFER-PLACE/locations/
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")

        for location in locations:
            req_path = os.path.join(
                project_root, "locations", location, "requirements.txt"
            )

            self.assertTrue(
                os.path.exists(req_path),
                f"Requirements file for {location} should exist",
            )


class TestForestHealthMonitor(unittest.TestCase):
    """Test suite for ForestHealthMonitor functionality."""

    def setUp(self):
        """Set up test fixtures for forest health monitor."""
        try:
            from geo_infer_place.locations.del_norte_county.forest_health_monitor import (
                ForestHealthMonitor,
            )
            from geo_infer_place.utils.integration import DelNorteDataIntegrator

            self.config = {
                "location": {
                    "bounds": {
                        "north": 42.006,
                        "south": 41.458,
                        "east": -123.536,
                        "west": -124.408,
                    }
                },
                "spatial": {"h3_resolution": 8},
                "analyses": {
                    "forest_health": {
                        "vegetation_indices": {
                            "ndvi": {
                                "threshold_healthy": 0.7,
                                "threshold_stressed": 0.4,
                                "threshold_critical": 0.2,
                            }
                        },
                        "forest_types": ["Redwood", "Douglas Fir", "Mixed Conifer"],
                        "change_detection": {
                            "baseline_years": [2010, 2015, 2020],
                            "minimum_change_threshold": 0.1,
                            "time_series_length": 10,
                        },
                    }
                },
            }

            # Mock data integrator
            self.data_integrator = DelNorteDataIntegrator()
            self.spatial_processor = None  # Mock for now

            self.monitor = ForestHealthMonitor(
                config=self.config,
                data_integrator=self.data_integrator,
                spatial_processor=self.spatial_processor,
                output_dir="/tmp/test_output",
            )

        except ImportError as e:
            self.fail(f"ForestHealthMonitor not available: {e}")

    def test_monitor_initialization(self):
        """Test that ForestHealthMonitor initializes correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.h3_resolution, 8)
        self.assertEqual(len(self.monitor.forest_types), 3)
        self.assertIn("ndvi", self.monitor.vegetation_indices)

    def test_vegetation_index_analysis(self):
        """Test vegetation index analysis functionality."""
        # Create mock vegetation data
        mock_data = {
            "data_sources": {
                "vegetation_indices": {
                    "ndvi_measurements": [
                        {
                            "date": "2024-01-01",
                            "lat": 41.7,
                            "lon": -124.0,
                            "ndvi": 0.8,
                            "evi": 0.6,
                            "moisture_stress": 0.2,
                            "h3_cell": "88281ca123fffff",
                        },
                        {
                            "date": "2024-01-02",
                            "lat": 41.7,
                            "lon": -124.0,
                            "ndvi": 0.3,
                            "evi": 0.2,
                            "moisture_stress": 0.8,
                            "h3_cell": "88281ca123fffff",
                        },
                        {
                            "date": "2024-01-03",
                            "lat": 41.7,
                            "lon": -124.0,
                            "ndvi": 0.6,
                            "evi": 0.4,
                            "moisture_stress": 0.4,
                            "h3_cell": "88281ca123fffff",
                        },
                    ]
                }
            }
        }

        analysis = self.monitor._analyze_vegetation_indices(mock_data)

        # Check that analysis contains expected keys
        self.assertIn("ndvi_analysis", analysis)
        self.assertIn("total_measurements", analysis)
        self.assertIn("temporal_coverage", analysis)
        self.assertIn("spatial_coverage", analysis)

        # Check NDVI analysis results
        ndvi_analysis = analysis["ndvi_analysis"]
        self.assertIn("mean", ndvi_analysis)
        self.assertIn("healthy_percent", ndvi_analysis)
        self.assertIn("stressed_percent", ndvi_analysis)
        self.assertIn("critical_percent", ndvi_analysis)

        # Verify calculations
        self.assertEqual(analysis["total_measurements"], 3)
        self.assertAlmostEqual(
            ndvi_analysis["healthy_percent"], 33.333333333333336, places=5
        )  # 1 out of 3 >= 0.7

    def test_forest_type_health_assessment(self):
        """Test forest type health assessment."""
        mock_data = {
            "data_sources": {
                "forest_inventory": {
                    "forest_plots": [
                        {
                            "plot_id": "DN_001",
                            "lat": 41.7,
                            "lon": -124.0,
                            "forest_type": "Redwood",
                            "basal_area_m2_ha": 120,
                            "tree_density_per_ha": 400,
                            "average_height_m": 75,
                            "canopy_cover_percent": 85,
                            "understory_diversity": 2.5,
                            "health_rating": "Good",
                            "age_class": "Mature",
                            "h3_cell": "88281ca123fffff",
                        },
                        {
                            "plot_id": "DN_002",
                            "lat": 41.8,
                            "lon": -123.9,
                            "forest_type": "Douglas Fir",
                            "basal_area_m2_ha": 80,
                            "tree_density_per_ha": 600,
                            "average_height_m": 45,
                            "canopy_cover_percent": 70,
                            "understory_diversity": 1.8,
                            "health_rating": "Fair",
                            "age_class": "Young",
                            "h3_cell": "88281ca456fffff",
                        },
                    ]
                }
            }
        }

        analysis = self.monitor._assess_forest_type_health(mock_data)

        # Check that analysis contains expected forest types
        self.assertIn("Redwood", analysis)
        self.assertIn("Douglas Fir", analysis)

        # Check Redwood analysis
        redwood_analysis = analysis["Redwood"]
        self.assertEqual(redwood_analysis["plot_count"], 1)
        self.assertIn("structure_metrics", redwood_analysis)
        self.assertIn("health_distribution", redwood_analysis)

        # Check structure metrics
        structure = redwood_analysis["structure_metrics"]
        self.assertEqual(structure["mean_basal_area"], 120)
        self.assertEqual(structure["mean_tree_density"], 400)
        self.assertEqual(structure["mean_height"], 75)

    def test_change_detection_analysis(self):
        """Test change detection analysis functionality."""
        mock_data = {
            "data_sources": {
                "vegetation_indices": {
                    "ndvi_measurements": [
                        {
                            "date": "2024-01-01",
                            "h3_cell": "88281ca123fffff",
                            "ndvi": 0.7,
                            "lat": 41.7,
                            "lon": -124.0,
                        },
                        {
                            "date": "2024-01-15",
                            "h3_cell": "88281ca123fffff",
                            "ndvi": 0.8,
                            "lat": 41.7,
                            "lon": -124.0,
                        },
                        {
                            "date": "2024-02-01",
                            "h3_cell": "88281ca123fffff",
                            "ndvi": 0.6,
                            "lat": 41.7,
                            "lon": -124.0,
                        },
                        {
                            "date": "2024-01-01",
                            "h3_cell": "88281ca456fffff",
                            "ndvi": 0.5,
                            "lat": 41.8,
                            "lon": -124.1,
                        },
                        {
                            "date": "2024-02-01",
                            "h3_cell": "88281ca456fffff",
                            "ndvi": 0.2,
                            "lat": 41.8,
                            "lon": -124.1,
                        },  # Significant decline
                    ]
                }
            }
        }

        analysis = self.monitor._perform_change_detection(mock_data)

        # Check analysis structure
        self.assertIn("h3_cell_changes", analysis)
        self.assertIn("significant_changes_count", analysis)
        self.assertEqual(analysis["minimum_change_threshold"], 0.1)

        # Check that significant changes are detected
        changes = analysis["h3_cell_changes"]
        significant_changes = [c for c in changes if c["change_significant"]]
        self.assertGreater(len(significant_changes), 0)

        # Check second cell has significant change
        second_cell_change = next(c for c in changes if "88281ca456" in c["h3_cell"])
        self.assertTrue(second_cell_change["change_significant"])
        self.assertAlmostEqual(
            second_cell_change["recent_change"], -0.3, places=5
        )  # 0.5 - 0.2

    def test_risk_assessment_generation(self):
        """Test risk assessment generation."""
        mock_results = {
            "vegetation_analysis": {
                "ndvi_analysis": {
                    "critical_percent": 15.0,  # 15% critical
                    "stressed_percent": 25.0,  # 25% stressed
                }
            },
            "change_analysis": {
                "significant_changes_count": 8,
                "h3_cell_changes": [{"h3_cell": f"cell_{i}"} for i in range(20)],
            },
            "mortality_analysis": {"mortality_rate_percent": 6.0},  # 6% mortality rate
            "climate_vulnerability": {"vulnerability_score": 0.6},
        }

        risk_assessment = self.monitor._generate_risk_assessment(mock_results)

        # Check risk assessment structure
        self.assertIn("overall_risk_score", risk_assessment)
        self.assertIn("risk_factors", risk_assessment)
        self.assertIn("recommendations", risk_assessment)

        # Check risk factors
        risk_factors = risk_assessment["risk_factors"]
        self.assertIn("vegetation_stress", risk_factors)
        self.assertIn("change_detection", risk_factors)
        self.assertIn("tree_mortality", risk_factors)
        self.assertIn("climate_vulnerability", risk_factors)

        # Check overall risk calculation (weighted average)
        expected_risk = (
            risk_factors["vegetation_stress"] * 0.3
            + risk_factors["change_detection"] * 0.3
            + risk_factors["tree_mortality"] * 0.25
            + risk_factors["climate_vulnerability"] * 0.15
        )
        self.assertAlmostEqual(
            risk_assessment["overall_risk_score"], expected_risk, places=5
        )

        # Check recommendations generation
        recommendations = risk_assessment["recommendations"]
        self.assertIsInstance(recommendations, list)

        # Should generate recommendations for high vegetation stress
        vegetation_risk = risk_factors["vegetation_stress"]
        if vegetation_risk > 0.5:
            self.assertTrue(any("vegetation" in rec.lower() for rec in recommendations))


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPlaceAnalyzer,
        TestLocationConfigurations,
        TestForestHealthMonitor,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)
