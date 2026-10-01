"""Tests for IoT sensor data models and measurements."""

from datetime import datetime, UTC

from geo_infer_iot.core.ingestion import SensorMeasurement, SpatialInferenceConfig


class TestSensorMeasurement:
    def test_create_measurement(self):
        m = SensorMeasurement(
            sensor_id="s-001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=22.5,
            unit="celsius",
            latitude=37.7749,
            longitude=-122.4194,
        )
        assert m.sensor_id == "s-001"
        assert m.value == 22.5
        assert m.h3_index is not None

    def test_h3_index_auto_generated(self):
        m = SensorMeasurement(
            sensor_id="s-002",
            timestamp=datetime.now(UTC),
            variable="humidity",
            value=65.0,
            unit="percent",
            latitude=40.7128,
            longitude=-74.006,
        )
        assert len(m.h3_index) > 0

    def test_custom_h3_resolution(self):
        m = SensorMeasurement(
            sensor_id="s-003",
            timestamp=datetime.now(UTC),
            variable="pressure",
            value=1013.25,
            unit="hPa",
            latitude=51.5074,
            longitude=-0.1278,
            h3_resolution=6,
        )
        assert m.h3_resolution == 6

    def test_quality_flags(self):
        m = SensorMeasurement(
            sensor_id="s-004",
            timestamp=datetime.now(UTC),
            variable="radiation",
            value=0.12,
            unit="uSv/h",
            latitude=35.6762,
            longitude=139.6503,
            quality_flags=["ok", "calibrated"],
        )
        assert "ok" in m.quality_flags
        assert len(m.quality_flags) == 2

    def test_metadata(self):
        m = SensorMeasurement(
            sensor_id="s-005",
            timestamp=datetime.now(UTC),
            variable="wind_speed",
            value=15.3,
            unit="m/s",
            latitude=48.8566,
            longitude=2.3522,
            metadata={"station": "paris-01", "height_m": 10},
        )
        assert m.metadata["station"] == "paris-01"


class TestSpatialInferenceConfig:
    def test_default_config(self):
        config = SpatialInferenceConfig(variable="temperature")
        assert config.variable == "temperature"
        assert config.h3_resolution == 8
        assert config.covariance_function == "matern_52"

    def test_custom_config(self):
        config = SpatialInferenceConfig(
            variable="radiation",
            h3_resolution=5,
            temporal_window_hours=2.0,
            spatial_range_km=50.0,
            covariance_function="rbf",
            length_scale=5000.0,
            noise_variance=0.05,
        )
        assert config.h3_resolution == 5
        assert config.covariance_function == "rbf"
        assert config.length_scale == 5000.0
