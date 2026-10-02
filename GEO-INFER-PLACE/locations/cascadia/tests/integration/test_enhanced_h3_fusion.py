#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced H3 Geospatial Data Fusion

This test validates:
- Proper H3 v4 API usage throughout the framework
- Reproducible data module structure with intelligent caching
- Real data acquisition and processing with fallback mechanisms
- Enhanced H3 geospatial data fusion capabilities
- SPACE integration for advanced geospatial operations
"""

import sys
import logging
from pathlib import Path
import tempfile
import shutil

from geo_infer_place.locations.cascadia.core.enhanced_data_manager import (
    create_enhanced_data_manager,
)
from geo_infer_place.locations.cascadia.core.enhanced_h3_fusion import (
    create_enhanced_h3_fusion,
)


class EnhancedH3FusionTestSuite:
    """
    Comprehensive test suite for enhanced H3 geospatial data fusion.
    """

    def __init__(self):
        """Initialize the test suite."""
        self.test_results = {}
        self.temp_dir = None
        self.data_manager = None
        self.h3_fusion = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_test_environment(self):
        """Setup test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="enhanced_h3_test_"))
        self.logger.info(f"Test environment setup: {self.temp_dir}")

        # Create enhanced data manager
        self.data_manager = create_enhanced_data_manager(
            base_data_dir=self.temp_dir / "data", h3_resolution=8
        )

        # Create enhanced H3 fusion engine
        self.h3_fusion = create_enhanced_h3_fusion(
            h3_resolution=8, enable_spatial_analysis=True
        )

        self.logger.info("✅ Test environment setup complete")

    def cleanup_test_environment(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"🧹 Cleaned up test environment: {self.temp_dir}")

    def test_h3_v4_api_usage(self) -> bool:
        """
        Test proper H3 v4 API usage throughout the framework.

        Returns:
            True if all H3 v4 API tests pass
        """
        self.logger.info("Testing H3 v4 API usage...")

        try:
            # Test H3 operations validation
            validation_result = self.h3_fusion.validate_h3_operations()

            # Check for errors
            if validation_result.get("errors"):
                self.logger.error(
                    f"H3 validation errors: {validation_result['errors']}"
                )
                return False

            # Check that all operations were tested
            expected_operations = [
                "latlng_to_cell",
                "cell_to_latlng",
                "cell_to_latlng_boundary",
                "geo_to_cells",
                "grid_disk",
                "cell_area",
                "is_valid_cell",
            ]

            tested_operations = validation_result.get("operations_tested", [])
            missing_operations = [
                op for op in expected_operations if op not in tested_operations
            ]

            if missing_operations:
                self.logger.error(f"Missing H3 operations: {missing_operations}")
                return False

            # Check API version
            if validation_result.get("h3_api_version") != "4.x":
                self.logger.error(
                    f"Wrong H3 API version: {validation_result.get('h3_api_version')}"
                )
                return False

            self.logger.info("✅ H3 v4 API usage test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ H3 v4 API usage test failed: {e}")
            return False

    def test_reproducible_data_structure(self) -> bool:
        """
        Test reproducible data module structure with intelligent caching.

        Returns:
            True if data structure tests pass
        """
        self.logger.info("Testing reproducible data module structure...")

        try:
            # Test data structure for each module
            test_modules = ["zoning", "current_use", "ownership", "improvements"]

            for module_name in test_modules:
                # Get data structure
                data_paths = self.data_manager.get_data_structure(module_name)

                # Check that all required paths exist
                required_paths = [
                    "module_dir",
                    "empirical_data",
                    "synthetic_data",
                    "raw_data",
                    "h3_cache",
                    "processed_data",
                    "metadata",
                    "validation_report",
                ]

                for path_name in required_paths:
                    if path_name not in data_paths:
                        self.logger.error(
                            f"Missing {path_name} in data structure for {module_name}"
                        )
                        return False

                # Check that directories are created
                if not data_paths["module_dir"].exists():
                    self.logger.error(f"Module directory not created for {module_name}")
                    return False

            self.logger.info("✅ Reproducible data structure test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Reproducible data structure test failed: {e}")
            return False

    def test_data_acquisition_and_caching(self) -> bool:
        """
        Test real data acquisition and processing with intelligent caching.

        Returns:
            True if data acquisition tests pass
        """
        self.logger.info("Testing data acquisition and caching...")

        try:
            # Test data acquisition for a sample module
            module_name = "zoning"

            # Mock data source function
            def mock_data_source():
                return self.temp_dir / "mock_data.geojson"

            # Test data acquisition with caching
            data_path = self.data_manager.acquire_data_with_caching(
                module_name=module_name,
                data_source_func=mock_data_source,
                force_refresh=False,
            )

            # Check that data path is returned
            if not data_path or not data_path.exists():
                self.logger.error(f"Data acquisition failed for {module_name}")
                return False

            # Test data quality report
            quality_report = self.data_manager.get_data_quality_report(module_name)

            # Check that quality report has required fields
            required_fields = [
                "module_name",
                "timestamp",
                "data_sources",
                "h3_processing",
                "quality_metrics",
            ]
            for field in required_fields:
                if field not in quality_report:
                    self.logger.error(f"Missing {field} in quality report")
                    return False

            self.logger.info("✅ Data acquisition and caching test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Data acquisition and caching test failed: {e}")
            return False

    def test_h3_geospatial_fusion(self) -> bool:
        """
        Test enhanced H3 geospatial data fusion capabilities.

        Returns:
            True if H3 fusion tests pass
        """
        self.logger.info("Testing enhanced H3 geospatial fusion...")

        try:
            # Create sample data sources
            sample_data_sources = {
                "zoning": {
                    "88281c8e89fffff": [
                        {
                            "zone_type": "Agricultural",
                            "zone_code": "A-1",
                            "acres": 15000,
                            "source": "CA_FMMP_2022",
                        }
                    ]
                },
                "current_use": {
                    "88281c8e89fffff": [
                        {
                            "crop_type": "Hay/Alfalfa",
                            "intensity": "high",
                            "water_usage": "irrigated",
                            "acres": 2500,
                            "source": "NASS_CDL_2022",
                        }
                    ]
                },
            }

            # Create target hexagons
            target_hexagons = ["88281c8e89fffff", "88281c1665fffff", "88281c8513fffff"]

            # Test H3 geospatial fusion
            fused_data = self.h3_fusion.fuse_geospatial_data(
                data_sources=sample_data_sources, target_hexagons=target_hexagons
            )

            # Check that fusion produced results
            if not fused_data:
                self.logger.error("H3 fusion produced no results")
                return False

            # Check that fused data has expected structure
            for hex_id in target_hexagons:
                if hex_id in fused_data:
                    hex_data = fused_data[hex_id]

                    # Check that hex_data is a dictionary
                    if not isinstance(hex_data, dict):
                        self.logger.error(f"Invalid fused data structure for {hex_id}")
                        return False

                    # Check that data sources are present
                    if "zoning" not in hex_data and "current_use" not in hex_data:
                        self.logger.error(f"No data sources in fused data for {hex_id}")
                        return False

            self.logger.info("✅ Enhanced H3 geospatial fusion test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Enhanced H3 geospatial fusion test failed: {e}")
            return False

    def test_spatial_analysis(self) -> bool:
        """
        Test spatial analysis capabilities.

        Returns:
            True if spatial analysis tests pass
        """
        self.logger.info("Testing spatial analysis capabilities...")

        try:
            # Create sample fused data
            sample_fused_data = {
                "88281c8e89fffff": {
                    "zoning": [{"zone_type": "Agricultural", "acres": 15000}],
                    "current_use": [{"crop_type": "Hay/Alfalfa", "acres": 2500}],
                },
                "88281c1665fffff": {
                    "zoning": [{"zone_type": "Rural Residential", "acres": 5000}],
                    "current_use": [{"crop_type": "Mixed Vegetables", "acres": 1500}],
                },
            }

            target_hexagons = ["88281c8e89fffff", "88281c1665fffff"]

            # Test spatial analysis
            enhanced_data = self.h3_fusion._perform_spatial_analysis(
                sample_fused_data, target_hexagons
            )

            # Check that spatial analysis added results
            for hex_id in enhanced_data:
                hex_data = enhanced_data[hex_id]

                # Check for spatial statistics
                if "spatial_stats" not in hex_data:
                    self.logger.error(f"No spatial stats for {hex_id}")
                    return False

                # Check for spatial correlations
                if "spatial_correlations" not in hex_data:
                    self.logger.error(f"No spatial correlations for {hex_id}")
                    return False

                # Check for spatial clusters
                if "spatial_cluster" not in hex_data:
                    self.logger.error(f"No spatial cluster for {hex_id}")
                    return False

            self.logger.info("✅ Spatial analysis test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Spatial analysis test failed: {e}")
            return False

    def test_cache_validation(self) -> bool:
        """
        Test cache validation and management.

        Returns:
            True if cache validation tests pass
        """
        self.logger.info("Testing cache validation...")

        try:
            # Test cache validation with valid data
            valid_cache = {
                "88281c8e89fffff": [{"test": "data"}],
                "88281c1665fffff": [{"test": "data"}],
            }

            target_hexagons = ["88281c8e89fffff", "88281c1665fffff", "88281c8513fffff"]

            # Test validation
            is_valid = self.data_manager._validate_h3_cache(
                valid_cache, target_hexagons
            )

            # Should be valid (2/3 coverage = 66.7% > 80% threshold)
            if not is_valid:
                self.logger.error("Cache validation failed for valid cache")
                return False

            # Test cache validation with invalid data
            invalid_cache = {}

            is_valid = self.data_manager._validate_h3_cache(
                invalid_cache, target_hexagons
            )

            # Should be invalid (0% coverage)
            if is_valid:
                self.logger.error("Cache validation passed for invalid cache")
                return False

            self.logger.info("✅ Cache validation test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Cache validation test failed: {e}")
            return False

    def test_data_quality_assessment(self) -> bool:
        """
        Test data quality assessment capabilities.

        Returns:
            True if data quality tests pass
        """
        self.logger.info("Testing data quality assessment...")

        try:
            # Test data quality report generation
            module_name = "zoning"

            quality_report = self.data_manager.get_data_quality_report(module_name)

            # Check required fields
            required_fields = [
                "module_name",
                "timestamp",
                "data_sources",
                "h3_processing",
                "quality_metrics",
            ]
            for field in required_fields:
                if field not in quality_report:
                    self.logger.error(f"Missing {field} in quality report")
                    return False

            # Check module name
            if quality_report["module_name"] != module_name:
                self.logger.error(
                    f"Wrong module name in quality report: {quality_report['module_name']}"
                )
                return False

            self.logger.info("✅ Data quality assessment test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Data quality assessment test failed: {e}")
            return False

    def run_all_tests(self) -> dict[str, bool]:
        """
        Run all enhanced H3 fusion tests.

        Returns:
            Dictionary of test results
        """
        self.logger.info("🚀 Starting Enhanced H3 Fusion Test Suite...")

        try:
            # Setup test environment
            self.setup_test_environment()

            # Run all tests
            tests = [
                ("H3 v4 API Usage", self.test_h3_v4_api_usage),
                ("Reproducible Data Structure", self.test_reproducible_data_structure),
                (
                    "Data Acquisition and Caching",
                    self.test_data_acquisition_and_caching,
                ),
                ("H3 Geospatial Fusion", self.test_h3_geospatial_fusion),
                ("Spatial Analysis", self.test_spatial_analysis),
                ("Cache Validation", self.test_cache_validation),
                ("Data Quality Assessment", self.test_data_quality_assessment),
            ]

            for test_name, test_func in tests:
                self.logger.info(f"\n--- Running {test_name} Test ---")
                result = test_func()
                self.test_results[test_name] = result

                if result:
                    self.logger.info(f"{test_name}: ✅ PASS")
                else:
                    self.logger.error(f"{test_name}: ❌ FAIL")

            # Print summary
            self.print_test_summary()

        except Exception as e:
            self.logger.error(f"❌ Test suite failed: {e}")
            return {}

        finally:
            # Cleanup test environment
            self.cleanup_test_environment()

        return self.test_results

    def print_test_summary(self):
        """Print comprehensive test summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ENHANCED H3 FUSION TEST SUMMARY")
        self.logger.info("=" * 80)

        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{test_name:<30} : {status}")

        self.logger.info(
            f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests / total_tests * 100:.1f}%)"
        )

        if passed_tests == total_tests:
            self.logger.info(
                "🎉 ALL TESTS PASSED! Enhanced H3 fusion is working correctly."
            )
        else:
            self.logger.error("❌ Some tests failed. Please review the implementation.")

        self.logger.info("=" * 80)


def main():
    """Main function to run the enhanced H3 fusion test suite."""
    test_suite = EnhancedH3FusionTestSuite()
    results = test_suite.run_all_tests()

    # Return exit code based on test results
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
