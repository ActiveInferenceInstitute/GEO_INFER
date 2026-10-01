"""Tests for IoT data ingestion engine."""

import pytest
from datetime import datetime, timedelta, timezone, UTC

from geo_infer_iot.core.ingestion import IoTDataIngestion, SensorMeasurement
from geo_infer_iot.core.registry import SensorRegistry
from geo_infer_iot.models.measurement import Measurement


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


def _reading(timestamp):
    return {
        "sensor_id": "s-001",
        "timestamp": timestamp,
        "variable": "temperature",
        "value": 22.5,
        "unit": "celsius",
        "latitude": 37.7749,
        "longitude": -122.4194,
    }


class TestTimestampContract:
    """IOT measurement timestamps are aware UTC; naive input is rejected."""

    def test_dict_to_measurement_rejects_naive_iso_string(self, ingestion):
        with pytest.raises(ValueError, match="timezone-aware"):
            ingestion._dict_to_measurement(_reading("2026-01-01T12:00:00"))

    def test_dict_to_measurement_normalizes_offset_to_utc(self, ingestion):
        m = ingestion._dict_to_measurement(_reading("2026-01-01T14:00:00+02:00"))
        assert m.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert m.timestamp.tzinfo is UTC

    def test_dict_to_measurement_defaults_missing_timestamp_to_utc_now(self, ingestion):
        data = _reading(None)
        del data["timestamp"]
        before = datetime.now(UTC)
        m = ingestion._dict_to_measurement(data)
        assert before <= m.timestamp <= datetime.now(UTC)

    def test_sensor_measurement_rejects_naive_datetime(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            SensorMeasurement(
                sensor_id="s-001",
                timestamp=datetime(2026, 1, 1, 12, 0),
                variable="temperature",
                value=22.5,
                unit="celsius",
                latitude=37.7749,
                longitude=-122.4194,
            )

    @pytest.mark.asyncio
    async def test_ingest_reports_naive_timestamp_as_failed(self, ingestion):
        result = await ingestion.ingest_measurement(_reading("2026-01-01T12:00:00"))
        assert result is False
        assert "timezone-aware" in ingestion.last_ingest_error

    def test_measurement_model_normalizes_and_rejects(self):
        aware = Measurement(
            measurement_id="m-1",
            sensor_id="s-001",
            variable="temperature",
            value=22.5,
            unit="celsius",
            timestamp=datetime(2026, 1, 1, 7, 0, tzinfo=timezone(timedelta(hours=-5))),
        )
        assert aware.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        default = Measurement(
            measurement_id="m-2",
            sensor_id="s-001",
            variable="temperature",
            value=22.5,
            unit="celsius",
        )
        assert default.timestamp.tzinfo is UTC
        with pytest.raises(ValueError, match="timezone-aware"):
            Measurement(
                measurement_id="m-3",
                sensor_id="s-001",
                variable="temperature",
                value=22.5,
                unit="celsius",
                timestamp="2026-01-01T12:00:00",
            )
