"""
Cross-Module Integration Tests for GEO-INFER Ecosystem

Tests real data flow using the actual validators and core components
from the geo_infer_test package — no mocks.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# Real imports from the testing library
from geo_infer_test import (
    BayesianValidator,
    DataQualityValidator,
    IoTValidator,
    PerformanceValidator,
    QualityController,
    SpatialValidator,
)
from geo_infer_test.core.log_integration import (
    LogAnalyzer,
    LogIntegration,
    LoggingTestReporter,
)


# ---------------------------------------------------------------------------
# Fixtures – real data
# ---------------------------------------------------------------------------


@pytest.fixture
def sensor_dataframe():
    """Real IoT sensor dataframe with timestamps, IDs, and radiation readings."""
    now = datetime.now(timezone.utc)
    records = []
    for i in range(20):
        records.append(
            {
                "sensor_id": f"sensor_{i}",
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "radiation_level": float(np.random.uniform(0.5, 15.0)),
                "latitude": float(np.random.uniform(37.0, 38.0)),
                "longitude": float(np.random.uniform(-123.0, -122.0)),
                "value": float(np.random.uniform(10, 100)),
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def spatial_dataframe():
    """Real spatial dataframe with lat/lon and h3 indices."""
    import h3

    coords = [
        (37.7749, -122.4194),
        (34.0522, -118.2437),
        (40.7128, -74.0060),
        (41.8781, -87.6298),
        (29.7604, -95.3698),
    ]
    records = []
    for lat, lon in coords:
        records.append(
            {
                "latitude": lat,
                "longitude": lon,
                "h3_index": h3.latlng_to_cell(lat, lon, 7),
                "value": float(np.random.uniform(1, 100)),
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def bayesian_results():
    """Realistic Bayesian inference results dict."""
    n = 50
    predictions = np.random.normal(10.0, 2.0, n).tolist()
    uncertainty = np.abs(np.random.normal(0.5, 0.1, n)).tolist()
    return {
        "converged": True,
        "predictions": predictions,
        "uncertainty": uncertainty,
        "prior_mean": 10.0,
        "length_scale": 1.5,
        "processing_time": 0.42,
    }


# ---------------------------------------------------------------------------
# Integration tests — real cross-module data flow
# ---------------------------------------------------------------------------


class TestCrossModuleDataFlow:
    """Validate data flowing across actual validators."""

    def test_sensor_to_spatial_validation(self, sensor_dataframe):
        """IoT data → DataQualityValidator → SpatialValidator pipeline."""
        dq = DataQualityValidator(config={}, logger=None)
        dq_results = dq.validate(sensor_dataframe)

        assert dq_results["total_records"] == 20
        assert dq_results["quality_score"] > 0.0

        sv = SpatialValidator(config={}, logger=None)
        sv_results = sv.validate(sensor_dataframe)

        assert sv_results["total_records"] == 20
        assert "coordinate_validity" in sv_results["spatial_validation"]
        valid_coords = sv_results["spatial_validation"]["coordinate_validity"][
            "valid_coordinates"
        ]
        assert valid_coords == 20

    def test_bayesian_validation(self, bayesian_results):
        """Validate real Bayesian inference outputs."""
        bv = BayesianValidator(config={}, logger=None)
        results = bv.validate(bayesian_results)

        assert results["inference_validation"]["convergence"] is True
        assert results["overall_quality"] in ("excellent", "good")
        assert (
            results["inference_validation"]["prediction_quality"]["total_predictions"]
            == 50
        )

    def test_performance_validation(self):
        """Validate performance metrics through PerformanceValidator."""
        pv = PerformanceValidator(config={})
        metrics = {
            "inference_time": 2.5,
            "accuracy": 0.92,
            "memory_usage": 500 * 1024 * 1024,  # 500 MB
        }
        results = pv.validate_performance(metrics)

        perf = results["performance_validation"]
        assert perf["overall_performance"] == "acceptable"
        assert perf["throughput_checks"]["accuracy_acceptable"] is True

    def test_quality_controller_comprehensive(self, sensor_dataframe, bayesian_results):
        """Full QualityController pipeline across IoT + Bayesian + performance."""
        qc = QualityController(config={})
        results = qc.run_comprehensive_validation(
            sensor_data=sensor_dataframe,
            inference_results=bayesian_results,
            performance_metrics={
                "inference_time": 1.5,
                "accuracy": 0.95,
                "memory_usage": 200 * 1024 * 1024,
            },
        )

        assert "overall_results" in results
        assert results["overall_results"]["system_quality"] in (
            "excellent",
            "good",
            "acceptable",
        )
        assert results["overall_results"]["components_tested"] >= 2
        assert len(results["components_validated"]) >= 2


class TestCrossModuleLogging:
    """Validate logging integration across modules."""

    def test_log_integration_with_validators(self, sensor_dataframe):
        """Test that LogIntegration records real validator runs."""
        log = LogIntegration({})

        with log.test_context("xmod_001", "IOT", "test_sensor_quality"):
            dq = DataQualityValidator(config={}, logger=None)
            result = dq.validate(sensor_dataframe)
            assert result["quality_score"] > 0.0

        assert len(log.test_entries) == 1
        assert log.test_entries[0].status == "PASS"

    def test_reporter_after_validation(self, sensor_dataframe):
        """LoggingTestReporter generates report from real validator runs."""
        log = LogIntegration({})

        with log.test_context("xmod_002", "DATA", "data_quality"):
            dq = DataQualityValidator(config={}, logger=None)
            dq.validate(sensor_dataframe)

        reporter = LoggingTestReporter(log)
        report = reporter.generate_test_report()

        assert report["summary"]["total_tests"] == 1
        assert report["summary"]["passed"] == 1

    def test_analyzer_after_multiple_tests(self, sensor_dataframe, bayesian_results):
        """LogAnalyzer works on real multi-module test data."""
        log = LogIntegration({})

        with log.test_context("xmod_003", "IOT", "sensor_check"):
            IoTValidator(config={}, logger=None).validate(sensor_dataframe)

        with log.test_context("xmod_004", "BAYES", "inference_check"):
            BayesianValidator(config={}, logger=None).validate(bayesian_results)

        analyzer = LogAnalyzer(log)
        patterns = analyzer.analyze_test_patterns()

        assert patterns["total_tests_analyzed"] == 2
        assert "IOT" in patterns["module_reliability"]
        assert "BAYES" in patterns["module_reliability"]


class TestCrossModuleErrorHandling:
    """Validate error handling across real modules."""

    def test_spatial_validator_with_invalid_data(self):
        """SpatialValidator handles invalid coordinates gracefully."""
        sv = SpatialValidator(config={}, logger=None)
        bad_data = pd.DataFrame(
            {
                "latitude": [200.0, -100.0, 37.0],
                "longitude": [-122.0, 500.0, -73.0],
            }
        )
        results = sv.validate(bad_data)
        invalid = results["spatial_validation"]["coordinate_validity"][
            "invalid_coordinates"
        ]
        assert invalid >= 2

    def test_bayesian_validator_unconverged(self):
        """BayesianValidator correctly flags non-converged results."""
        bv = BayesianValidator(config={}, logger=None)
        results = bv.validate(
            {
                "converged": False,
                "predictions": [1.0, 2.0, float("nan")],
                "uncertainty": [0.1, -0.5, 0.3],
            }
        )
        assert results["overall_quality"] == "poor"
        assert results["inference_validation"]["convergence"] is False
