"""
Integration tests for GEO-INFER-IOT module.

Tests the integration between different components of the IoT module
and its interaction with other GEO-INFER modules.
"""

import unittest
import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, Mock, patch

# Import the module to test


from geo_infer_iot.core.ingestion import IoTDataIngestion, SensorMeasurement
from geo_infer_iot.core.registry import SensorRegistry
from geo_infer_iot.core.spatial_fusion import SpatialDataFusion
from geo_infer_iot.core.quality_control import QualityController


class TestIoTModuleIntegration(unittest.TestCase):
    """Test integration between IoT module components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "spatial": {"h3_resolution": 8},
            "quality_control": {"enabled": True},
            "spatial_fusion": {"method": "inverse_distance_weighted"},
        }

    def test_registry_ingestion_integration(self):
        """Test integration between SensorRegistry and IoTDataIngestion."""
        # Create registry and ingestion
        registry = SensorRegistry(self.config)
        ingestion = IoTDataIngestion(registry, self.config)

        # Register a sensor
        sensor = registry.register_sensor(
            {
                "sensor_id": "test_sensor_001",
                "network_id": "test_network",
                "sensor_type": "temperature",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "h3_resolution": 8,
            }
        )

        self.assertEqual(sensor.sensor_id, "test_sensor_001")
        self.assertIn("test_sensor_001", registry.sensors)

        # Ingest a measurement
        measurement = SensorMeasurement(
            sensor_id="test_sensor_001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=25.5,
            unit="celsius",
            latitude=40.7128,
            longitude=-74.0060,
        )

        # Mock spatial inference to avoid dependency issues
        ingestion._update_spatial_inference = AsyncMock(return_value=None)

        # Test ingestion
        result = asyncio.run(ingestion.ingest_measurement(measurement))

        self.assertTrue(result)
        self.assertEqual(len(ingestion.measurements), 1)

        # Verify sensor was updated in registry
        updated_sensor = registry.sensors["test_sensor_001"]
        self.assertEqual(updated_sensor.last_seen, measurement.timestamp)

    def test_spatial_fusion_integration(self):
        """Test integration with SpatialDataFusion."""
        fusion = SpatialDataFusion(self.config)

        # Create test measurements
        measurements = []
        for i in range(5):
            measurement = {
                "sensor_id": f"sensor_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
                "variable": "temperature",
                "value": 20.0 + i,
                "unit": "celsius",
                "latitude": 40.7128 + i * 0.01,
                "longitude": -74.0060 + i * 0.01,
            }
            measurements.append(measurement)

        # Test spatial fusion
        fusion_result = fusion.fuse_sensor_data(measurements, "temperature")

        self.assertIn("overall_statistics", fusion_result)
        self.assertIn("cell_aggregates", fusion_result)
        self.assertEqual(fusion_result["measurement_count"], 5)

    def test_quality_control_integration(self):
        """Test integration with QualityController."""
        qc = QualityController(self.config)

        # Create test measurements
        measurements = []
        for i in range(5):
            measurement = {
                "sensor_id": f"sensor_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
                "variable": "temperature",
                "value": 20.0 + i,
                "unit": "celsius",
                "latitude": 40.7128,
                "longitude": -74.0060,
            }
            measurements.append(measurement)

        # Test batch validation
        validation_result = qc.validate_batch(measurements)

        self.assertIn("total_measurements", validation_result)
        self.assertIn("passed_measurements", validation_result)
        self.assertIn("results", validation_result)
        self.assertEqual(validation_result["total_measurements"], 5)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Setup complete system
        registry = SensorRegistry(self.config)
        ingestion = IoTDataIngestion(registry, self.config)
        fusion = SpatialDataFusion(self.config)
        qc = QualityController(self.config)

        # Register sensors
        sensors = []
        for i in range(3):
            sensor = registry.register_sensor(
                {
                    "sensor_id": f"test_sensor_{i}",
                    "network_id": "test_network",
                    "sensor_type": "temperature",
                    "latitude": 40.7128 + i * 0.01,
                    "longitude": -74.0060 + i * 0.01,
                    "h3_resolution": 8,
                }
            )
            sensors.append(sensor)

        # Generate measurements
        measurements = []
        for i in range(3):
            measurement = {
                "sensor_id": f"test_sensor_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
                "variable": "temperature",
                "value": 20.0 + i,
                "unit": "celsius",
                "latitude": 40.7128 + i * 0.01,
                "longitude": -74.0060 + i * 0.01,
            }
            measurements.append(measurement)

        # Quality control
        qc_result = qc.validate_batch(measurements)
        self.assertGreater(qc_result["pass_rate"], 0.8)

        # Ingestion
        for measurement in measurements:
            sensor_measurement = ingestion._dict_to_measurement(measurement)
            ingestion._update_spatial_inference = AsyncMock(return_value=None)
            asyncio.run(ingestion.ingest_measurement(sensor_measurement))

        # Spatial fusion
        fusion_result = fusion.fuse_sensor_data(measurements, "temperature")
        self.assertIn("overall_statistics", fusion_result)

        # Verify complete workflow
        self.assertEqual(len(ingestion.measurements), 3)
        self.assertEqual(len(registry.sensors), 3)

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        registry = SensorRegistry(self.config)
        ingestion = IoTDataIngestion(registry, self.config)

        # Test with invalid measurement data
        invalid_measurement = {
            "sensor_id": "invalid_sensor",
            "timestamp": datetime.now(UTC).isoformat(),
            "variable": "temperature",
            "value": float("inf"),  # Invalid value
            "unit": "celsius",
            "latitude": 100.0,  # Invalid latitude
            "longitude": -74.0060,
        }

        # Should handle gracefully
        sensor_measurement = ingestion._dict_to_measurement(invalid_measurement)
        ingestion._update_spatial_inference = AsyncMock(return_value=None)

        result = asyncio.run(ingestion.ingest_measurement(sensor_measurement))

        # Should fail gracefully (return False)
        self.assertFalse(result)


class TestCrossModuleIntegration(unittest.TestCase):
    """Test integration with other GEO-INFER modules."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "spatial": {"h3_resolution": 8},
            "bayesian_inference": {"enabled": True},
        }

    def test_bayes_integration(self):
        """Test integration with GEO-INFER-BAYES."""
        registry = SensorRegistry(self.config)
        ingestion = IoTDataIngestion(registry, self.config)

        # Mock BAYES components
        mock_gp = Mock()
        mock_gp.fit = Mock(return_value=mock_gp)
        mock_gp.predict = Mock(return_value=([0.1, 0.2], [0.01, 0.02]))

        with patch(
            "geo_infer_iot.core.ingestion.GaussianProcess", return_value=mock_gp
        ):
            with patch("geo_infer_iot.core.ingestion.SpatialCovariance"):
                # Setup spatial inference
                config = type(
                    "Config",
                    (),
                    {
                        "variable": "temperature",
                        "h3_resolution": 8,
                        "temporal_window_hours": 1.0,
                        "covariance_function": "matern_52",
                        "length_scale": 1000,
                        "noise_variance": 0.01,
                    },
                )()

                ingestion.setup_spatial_inference(config)

                # Test spatial inference update
                asyncio.run(ingestion._update_spatial_inference("temperature"))

                # Verify model was created and used
                self.assertIsNotNone(ingestion.spatial_models.get("temperature"))

    def test_space_integration(self):
        """Test integration with GEO-INFER-SPACE."""
        registry = SensorRegistry(self.config)

        # Mock SPACE components
        mock_spatial_ops = Mock()
        mock_spatial_ops.latlon_to_meters = Mock(return_value=(1000, 2000))

        with patch(
            "geo_infer_iot.core.ingestion.SpatialOperations",
            return_value=mock_spatial_ops,
        ):
            with patch(
                "geo_infer_iot.core.ingestion.get_h3_neighbors",
                return_value=["neighbor1", "neighbor2"],
            ):
                with patch(
                    "geo_infer_iot.core.ingestion.h3_resolution_stats",
                    return_value={"area": 1000},
                ):
                    ingestion = IoTDataIngestion(registry, self.config)

                    # Test spatial indexing
                    measurement = SensorMeasurement(
                        sensor_id="test_sensor",
                        timestamp=datetime.now(UTC),
                        variable="temperature",
                        value=25.5,
                        unit="celsius",
                        latitude=40.7128,
                        longitude=-74.0060,
                    )

                    ingestion._add_spatial_index(measurement)

                    # Verify enhanced spatial metadata was added
                    self.assertIn("h3_neighbors", measurement.metadata)
                    self.assertIn("h3_stats", measurement.metadata)


if __name__ == "__main__":
    unittest.main()
