"""
Unit tests for IoT data ingestion functionality.

Tests the IoTDataIngestion class and related components for correct
behavior with various input data and configurations.
"""

import unittest
import asyncio
from datetime import datetime, timedelta, UTC
import numpy as np
import h3

# Import the module to test


from geo_infer_iot.core.ingestion import (
    SensorMeasurement,
    IoTDataIngestion,
    RadiationMonitoringSystem,
    SpatialInferenceConfig,
)
from geo_infer_iot.core.registry import SensorRegistry
from geo_infer_iot.models.measurement import Measurement
from geo_infer_iot.models.sensor import Location, Sensor, SensorCapabilities


def build_radiation_measurements(count: int) -> list[dict]:
    """Build deterministic measurements representing an external sensor feed."""
    timestamp = datetime.now(UTC).isoformat()
    return [
        {
            "sensor_id": f"sensor-{index:03d}",
            "timestamp": timestamp,
            "variable": "gamma_radiation",
            "value": 0.1 + index * 0.001,
            "unit": "μSv/h",
            "latitude": 40.7 + index * 0.001,
            "longitude": -74.0 - index * 0.001,
        }
        for index in range(count)
    ]


class TestSensorMeasurement(unittest.TestCase):
    """Test the SensorMeasurement data class."""

    def test_measurement_creation_with_coordinates(self):
        """Test creating a measurement with latitude/longitude."""
        measurement = SensorMeasurement(
            sensor_id="test_sensor_001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
            h3_resolution=8,
        )

        self.assertEqual(measurement.sensor_id, "test_sensor_001")
        self.assertEqual(measurement.variable, "temperature")
        self.assertEqual(measurement.value, 25.5)
        self.assertIsNotNone(measurement.h3_index)
        self.assertTrue(h3.is_valid_cell(measurement.h3_index))

    def test_measurement_creation_with_zero_coordinates(self):
        """Test creating a measurement with zero coordinates (null island)."""
        measurement = SensorMeasurement(
            sensor_id="test_sensor_002",
            timestamp=datetime.now(UTC),
            variable="humidity",
            value=65.0,
            unit="percent",
            latitude=0.0,
            longitude=0.0,
        )

        self.assertEqual(measurement.sensor_id, "test_sensor_002")
        # latitude=0.0 is falsy, so h3_index won't be computed by __post_init__
        self.assertEqual(measurement.latitude, 0.0)
        self.assertEqual(measurement.longitude, 0.0)

    def test_measurement_validation(self):
        """Test measurement validation."""
        # Valid measurement
        valid_measurement = SensorMeasurement(
            sensor_id="valid_sensor",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
        )

        # Test coordinate validation
        self.assertTrue(-90 <= valid_measurement.latitude <= 90)
        self.assertTrue(-180 <= valid_measurement.longitude <= 180)

        # Test value validation
        self.assertIsInstance(valid_measurement.value, (int, float))
        self.assertFalse(np.isnan(valid_measurement.value))


class TestPydanticH3Models(unittest.TestCase):
    """Ensure public Pydantic models use the H3 v4 validation API."""

    def test_measurement_generates_and_validates_h3_cell(self):
        measurement = Measurement(
            measurement_id="measurement-1",
            sensor_id="sensor-1",
            variable="temperature",
            value=21.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
        )
        self.assertTrue(h3.is_valid_cell(measurement.h3_index))

        with self.assertRaises(ValueError):
            Measurement(
                measurement_id="measurement-2",
                sensor_id="sensor-1",
                variable="temperature",
                value=21.5,
                unit="celsius",
                h3_index="not-a-cell",
            )

    def test_sensor_location_validates_h3_cell(self):
        location = Location(latitude=40.7128, longitude=-74.0060)
        self.assertTrue(h3.is_valid_cell(location.h3_index))

        with self.assertRaises(ValueError):
            Sensor(
                sensor_id="sensor-1",
                network_id="network-1",
                sensor_type="temperature",
                location=Location(
                    latitude=40.7128,
                    longitude=-74.0060,
                    h3_index="not-a-cell",
                ),
                capabilities=SensorCapabilities(measured_variables=["temperature"]),
            )


class TestIoTDataIngestion(unittest.TestCase):
    """Test the IoTDataIngestion class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "spatial": {"h3_resolution": 8},
            "inference": {
                "mean_function": "constant",
                "covariance": "matern_52",
                "length_scale": 1000,
                "noise_variance": 0.01,
            },
        }

        # Use real SensorRegistry instead of mock
        self.registry = SensorRegistry()
        self.ingestion = IoTDataIngestion(self.registry, self.config)

    def test_ingestion_initialization(self):
        """Test IoTDataIngestion initialization."""
        self.assertIsNotNone(self.ingestion.registry)
        self.assertIsNotNone(self.ingestion.config)
        self.assertIsInstance(self.ingestion.measurements, list)
        self.assertIsInstance(self.ingestion.spatial_index, dict)

    def test_dict_to_measurement_conversion(self):
        """Test conversion from dictionary to SensorMeasurement."""
        measurement_dict = {
            "sensor_id": "test_sensor",
            "timestamp": datetime.now(UTC).isoformat(),
            "variable": "temperature",
            "value": 25.5,
            "unit": "celsius",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "h3_resolution": 8,
            "quality_flags": ["validated"],
            "metadata": {"test": "data"},
        }

        measurement = self.ingestion._dict_to_measurement(measurement_dict)

        self.assertIsInstance(measurement, SensorMeasurement)
        self.assertEqual(measurement.sensor_id, "test_sensor")
        self.assertEqual(measurement.value, 25.5)
        self.assertEqual(measurement.latitude, 40.7128)
        self.assertEqual(measurement.longitude, -74.0060)
        self.assertIn("validated", measurement.quality_flags)
        self.assertEqual(measurement.metadata["test"], "data")

    def test_measurement_validation(self):
        """Test measurement validation."""
        # Valid measurement
        valid_measurement = SensorMeasurement(
            sensor_id="valid_sensor",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
        )

        self.assertTrue(self.ingestion._validate_measurement(valid_measurement))

        # Invalid coordinate
        invalid_measurement = SensorMeasurement(
            sensor_id="invalid_sensor",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=100.0,  # Invalid latitude
            longitude=-74.0060,
        )

        self.assertFalse(self.ingestion._validate_measurement(invalid_measurement))

    def test_measurement_ingestion(self):
        """Test measurement ingestion."""
        measurement = SensorMeasurement(
            sensor_id="test_sensor",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
        )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.ingestion.ingest_measurement(measurement)
            )
        finally:
            loop.close()

        self.assertTrue(result)
        self.assertEqual(len(self.ingestion.measurements), 1)
        self.assertIn(measurement.h3_index, self.ingestion.spatial_index)

    def test_spatial_indexing(self):
        """Test H3 spatial indexing."""
        measurement = SensorMeasurement(
            sensor_id="test_sensor",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
            h3_resolution=8,
        )

        self.ingestion._add_spatial_index(measurement)

        self.assertIsNotNone(measurement.h3_index)
        self.assertTrue(h3.is_valid_cell(measurement.h3_index))
        # h3_neighbors and h3_stats are only populated when GEO-INFER-SPACE is available

    def test_measurement_statistics(self):
        """Test measurement statistics calculation."""
        # Add some test measurements
        for i in range(5):
            measurement = SensorMeasurement(
                sensor_id=f"sensor_{i}",
                timestamp=datetime.now(UTC),
                variable="temperature",
                value=20.0 + i,
                unit="celsius",
                latitude=40.7128 + i * 0.01,
                longitude=-74.0060 + i * 0.01,
            )
            self.ingestion.measurements.append(measurement)

        stats = self.ingestion.get_measurement_statistics()

        self.assertEqual(stats["total_measurements"], 5)
        self.assertEqual(stats["unique_sensors"], 5)
        self.assertEqual(stats["unique_variables"], 1)
        self.assertIn("time_range", stats)
        # spatial_inference_enabled depends on GEO-INFER-BAYES availability
        self.assertIn("spatial_inference_enabled", stats)

    def test_recent_measurements_filtering(self):
        """Test filtering of recent measurements."""
        # Add measurements with different timestamps using timedelta offsets
        base_time = datetime.now(UTC)

        # One recent (10 min ago), one older (3 hours ago), one very old (10 hours ago)
        offsets_minutes = [10, 180, 600]
        for i, offset in enumerate(offsets_minutes):
            measurement = SensorMeasurement(
                sensor_id="test_sensor",
                timestamp=base_time - timedelta(minutes=offset),
                variable="temperature",
                value=20.0 + i,
                unit="celsius",
                latitude=40.7128,
                longitude=-74.0060,
            )
            self.ingestion.measurements.append(measurement)

        # Test getting recent measurements (last 2 hours)
        recent = self.ingestion._get_recent_measurements("temperature", hours=2)

        # Should get the one measurement from 10 min ago (within 2 hours)
        self.assertGreater(len(recent), 0)
        for measurement in recent:
            time_diff = (
                base_time - measurement.timestamp.replace(tzinfo=UTC)
            ).total_seconds()
            self.assertLess(time_diff, 7200)  # 2 hours in seconds


class TestRadiationMonitoringSystem(unittest.TestCase):
    """Test the RadiationMonitoringSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "radiation_baseline": {"background_radiation": 0.1, "noise_level": 0.02},
            "quality_control": {
                "sensor_validation": {"min_radiation": 0.0, "max_radiation": 100.0}
            },
        }

        self.monitoring_system = RadiationMonitoringSystem(self.config)

    def test_system_initialization(self):
        """Test RadiationMonitoringSystem initialization."""
        self.assertIsNotNone(self.monitoring_system.config)
        self.assertIsNotNone(self.monitoring_system.registry)
        self.assertIsNotNone(self.monitoring_system.ingestion)
        self.assertEqual(self.monitoring_system.metrics["measurements_processed"], 0)

    def test_empirical_measurement_processing(self):
        """Test processing of supplied radiation measurements."""
        measurements = build_radiation_measurements(10)

        self.assertEqual(len(measurements), 10)

        for measurement in measurements:
            self.assertIn("sensor_id", measurement)
            self.assertIn("latitude", measurement)
            self.assertIn("longitude", measurement)
            self.assertIn("value", measurement)
            self.assertIn("variable", measurement)
            self.assertEqual(measurement["variable"], "gamma_radiation")
            self.assertTrue(0 <= measurement["value"] <= 100)  # Reasonable range

    def test_measurement_processing(self):
        """Test processing of measurements."""
        measurements = build_radiation_measurements(5)

        # Setup spatial inference
        self.monitoring_system.setup_spatial_inference()

        # Process measurements
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                self.monitoring_system.process_measurements(measurements)
            )
        finally:
            loop.close()

        self.assertIn("processed", results)
        self.assertIn("failed", results)
        self.assertIn("anomalies", results)
        self.assertIn("spatial_cells", results)
        self.assertGreater(results["processed"], 0)

    def test_quality_control(self):
        """Test quality control functionality."""
        measurement = {
            "sensor_id": "test_sensor",
            "timestamp": datetime.now(UTC).isoformat(),
            "variable": "gamma_radiation",
            "value": 0.15,  # Normal value
            "unit": "μSv/h",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

        # Convert to SensorMeasurement
        sensor_measurement = self.monitoring_system.ingestion._dict_to_measurement(
            measurement
        )

        # Test quality control
        qc_result = self.monitoring_system._quality_control(sensor_measurement)

        self.assertIn("passed", qc_result)
        self.assertIn("issues", qc_result)
        self.assertIn("quality_score", qc_result)
        self.assertIsInstance(qc_result["quality_score"], float)

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Normal measurement
        normal_measurement = {
            "sensor_id": "test_sensor",
            "timestamp": datetime.now(UTC).isoformat(),
            "variable": "gamma_radiation",
            "value": 0.1,  # Normal background level
            "unit": "μSv/h",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

        sensor_measurement = self.monitoring_system.ingestion._dict_to_measurement(
            normal_measurement
        )

        # Should not be anomaly
        self.assertFalse(self.monitoring_system._is_anomaly(sensor_measurement))

        # High radiation measurement (anomaly)
        anomaly_measurement = {
            "sensor_id": "test_sensor",
            "timestamp": datetime.now(UTC).isoformat(),
            "variable": "gamma_radiation",
            "value": 5.0,  # Very high level
            "unit": "μSv/h",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

        anomaly_sensor_measurement = (
            self.monitoring_system.ingestion._dict_to_measurement(anomaly_measurement)
        )

        # Should be anomaly
        self.assertTrue(self.monitoring_system._is_anomaly(anomaly_sensor_measurement))

    def test_system_health_validation(self):
        """Test system health validation."""

        # Process some measurements
        self.monitoring_system.metrics["measurements_processed"] = 10
        self.monitoring_system.metrics["spatial_inferences"] = 1
        self.monitoring_system.metrics["errors_encountered"] = 0

        health = self.monitoring_system.validate_system_health()

        self.assertIn("overall_healthy", health)
        self.assertIn("checks", health)
        self.assertIn("metrics", health)
        self.assertIn("timestamp", health)

        # Verify the health check structure has expected keys
        self.assertIn("measurements_processing", health["checks"])
        self.assertIn("error_rate_acceptable", health["checks"])
        self.assertIn("performance_acceptable", health["checks"])


class TestSpatialInferenceConfig(unittest.TestCase):
    """Test the SpatialInferenceConfig data class."""

    def test_config_creation(self):
        """Test creating spatial inference configuration."""
        config = SpatialInferenceConfig(
            variable="temperature",
            h3_resolution=8,
            temporal_window_hours=1.0,
            spatial_range_km=10.0,
            covariance_function="matern_52",
            length_scale=1000.0,
            noise_variance=0.01,
            confidence_levels=[0.68, 0.95],
        )

        self.assertEqual(config.variable, "temperature")
        self.assertEqual(config.h3_resolution, 8)
        self.assertEqual(config.temporal_window_hours, 1.0)
        self.assertEqual(config.confidence_levels, [0.68, 0.95])


if __name__ == "__main__":
    unittest.main()
