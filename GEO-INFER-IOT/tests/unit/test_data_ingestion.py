"""Tests for IoT data ingestion engine."""

import pytest
from datetime import datetime, UTC

from geo_infer_iot.core.ingestion import IoTDataIngestion, SensorMeasurement
from geo_infer_iot.core.registry import SensorRegistry


@pytest.fixture
def registry():
    return SensorRegistry()


@pytest.fixture
def ingestion(registry):
    return IoTDataIngestion(registry=registry)


class TestIoTDataIngestion:
    def test_init(self, ingestion):
        assert ingestion.measurements == []
        assert ingestion.is_processing is False

    def test_dict_to_measurement(self, ingestion):
        data = {
            "sensor_id": "s-001",
            "timestamp": datetime.now(UTC).isoformat(),
            "variable": "temperature",
            "value": 22.5,
            "unit": "celsius",
            "latitude": 37.7749,
            "longitude": -122.4194,
        }
        m = ingestion._dict_to_measurement(data)
        assert isinstance(m, SensorMeasurement)
        assert m.sensor_id == "s-001"

    def test_validate_measurement_valid(self, ingestion):
        m = SensorMeasurement(
            sensor_id="s-001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=22.5,
            unit="celsius",
            latitude=37.7749,
            longitude=-122.4194,
        )
        assert ingestion._validate_measurement(m) is True

    def test_validate_measurement_invalid_latitude(self, ingestion):
        m = SensorMeasurement(
            sensor_id="s-001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=22.5,
            unit="celsius",
            latitude=91.0,  # Invalid
            longitude=-122.4194,
        )
        assert ingestion._validate_measurement(m) is False

    def test_validate_measurement_invalid_longitude(self, ingestion):
        m = SensorMeasurement(
            sensor_id="s-001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=22.5,
            unit="celsius",
            latitude=37.7749,
            longitude=-181.0,  # Invalid
        )
        assert ingestion._validate_measurement(m) is False

    def test_validate_measurement_empty_sensor_id(self, ingestion):
        m = SensorMeasurement(
            sensor_id="",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=22.5,
            unit="celsius",
            latitude=37.0,
            longitude=-122.0,
        )
        assert ingestion._validate_measurement(m) is False

    @pytest.mark.asyncio
    async def test_ingest_measurement(self, ingestion):
        m = SensorMeasurement(
            sensor_id="s-001",
            timestamp=datetime.now(UTC),
            variable="temperature",
            value=22.5,
            unit="celsius",
            latitude=37.7749,
            longitude=-122.4194,
        )
        result = await ingestion.ingest_measurement(m)
        assert result is True
        assert len(ingestion.measurements) == 1

    @pytest.mark.asyncio
    async def test_ingest_measurement_from_dict(self, ingestion):
        data = {
            "sensor_id": "s-001",
            "variable": "temperature",
            "value": 22.5,
            "unit": "celsius",
            "latitude": 37.7749,
            "longitude": -122.4194,
        }
        result = await ingestion.ingest_measurement(data)
        assert result is True

    def test_get_measurement_statistics_empty(self, ingestion):
        stats = ingestion.get_measurement_statistics()
        assert stats == {}

    def test_get_spatial_distribution_not_available(self, ingestion):
        result = ingestion.get_spatial_distribution("temperature")
        assert result is None
