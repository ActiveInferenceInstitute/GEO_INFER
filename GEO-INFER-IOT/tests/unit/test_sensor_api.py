"""Unit tests for SensorAPI booted with a stub registry.

Covers the healthy-install branch (core modules wired, not the silent
degradation fallback) and the REST flow register sensor -> submit
measurement -> spatially filtered query response shape.
"""

from datetime import datetime
from typing import Any

import h3
from fastapi.testclient import TestClient

from geo_infer_iot.api import sensor_api as sensor_api_module
from geo_infer_iot.api.sensor_api import SensorAPI

LATITUDE = 40.7
LONGITUDE = -74.0
H3_RESOLUTION = 8

SENSOR_DATA: dict[str, Any] = {
    "sensor_id": "s-1",
    "network_id": "net-1",
    "sensor_type": "temperature",
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
}

MEASUREMENT_DATA: dict[str, Any] = {
    "sensor_id": "s-1",
    "timestamp": "2026-09-22T00:00:00+00:00",
    "variable": "temperature",
    "value": 21.5,
    "unit": "C",
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
}


class StubSensorRecord:
    """Minimal stand-in for registry.SensorMetadata."""

    def __init__(
        self,
        sensor_id: str,
        network_id: str,
        sensor_type: str,
        latitude: float,
        longitude: float,
    ) -> None:
        self.sensor_id = sensor_id
        self.network_id = network_id
        self.sensor_type = sensor_type
        self.latitude = latitude
        self.longitude = longitude
        self.h3_index = h3.latlng_to_cell(latitude, longitude, 8)
        self.status = "active"
        self.registered_at = datetime.now()
        self.last_seen = None
        self.metadata: dict[str, Any] = {}


class StubRegistry:
    """Registry surface consumed by the SensorAPI routes."""

    def __init__(self) -> None:
        self.sensors: dict[str, StubSensorRecord] = {}
        self.networks: dict[str, Any] = {}

    def register_sensor(self, sensor_info: dict[str, Any]) -> StubSensorRecord:
        record = StubSensorRecord(
            sensor_info["sensor_id"],
            sensor_info["network_id"],
            sensor_info["sensor_type"],
            float(sensor_info["latitude"]),
            float(sensor_info["longitude"]),
        )
        self.sensors[record.sensor_id] = record
        return record

    def get_sensors_in_h3_cell(self, h3_index: str) -> list[StubSensorRecord]:
        return [s for s in self.sensors.values() if s.h3_index == h3_index]


class StubMeasurement:
    """Minimal stand-in for ingestion.SensorMeasurement."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.sensor_id = data["sensor_id"]
        self.timestamp = datetime.fromisoformat(data["timestamp"])
        self.variable = data["variable"]
        self.value = float(data["value"])
        self.unit = data.get("unit", "")
        self.latitude = float(data["latitude"])
        self.longitude = float(data["longitude"])
        self.h3_index = h3.latlng_to_cell(self.latitude, self.longitude, 8)
        self.quality_flags: list[str] = list(data.get("quality_flags", []))
        self.metadata: dict[str, Any] = dict(data.get("metadata", {}))


class StubIngestion:
    """Ingestion stand-in exposing the attribute surface the API reads."""

    def __init__(self) -> None:
        self.measurements: list[StubMeasurement] = []

    async def ingest_measurement(self, data: dict[str, Any]) -> bool:
        self.measurements.append(StubMeasurement(data))
        return True

    def get_measurement_statistics(self) -> dict[str, int]:
        return {"total_measurements": len(self.measurements)}


def test_healthy_install_wires_core_components() -> None:
    assert sensor_api_module.HAS_CORE_MODULES is True
    api = SensorAPI({})
    # Healthy install: real registry/ingestion wired, not the 503 stubs.
    assert api.registry is not None
    assert api.ingestion is not None


def test_register_sensor_via_post() -> None:
    api = SensorAPI({})
    api.registry = StubRegistry()
    api.ingestion = StubIngestion()
    client = TestClient(api.get_app())

    response = client.post("/sensors", json=SENSOR_DATA)
    assert response.status_code == 200
    body = response.json()
    assert body["sensor_id"] == "s-1"
    assert body["status"] == "registered"

    listing = client.get("/sensors")
    assert listing.status_code == 200
    sensors = listing.json()["sensors"]
    assert len(sensors) == 1
    assert sensors[0]["sensor_id"] == "s-1"
    assert sensors[0]["h3_index"] == h3.latlng_to_cell(LATITUDE, LONGITUDE, 8)


def test_measurement_spatial_filter_response_shape() -> None:
    api = SensorAPI({})
    api.registry = StubRegistry()
    api.ingestion = StubIngestion()
    client = TestClient(api.get_app())
    assert client.post("/sensors", json=SENSOR_DATA).status_code == 200

    submitted = client.post("/measurements", json=[MEASUREMENT_DATA])
    assert submitted.status_code == 200
    assert submitted.json()["processed_count"] == 1
    assert submitted.json()["failed_count"] == 0

    expected_cell = h3.latlng_to_cell(LATITUDE, LONGITUDE, H3_RESOLUTION)
    hit = client.get("/measurements", params={"h3_index": expected_cell})
    assert hit.status_code == 200
    body = hit.json()
    assert set(body) == {
        "measurements",
        "total_count",
        "returned_count",
        "query_parameters",
    }
    assert body["total_count"] == 1
    assert body["returned_count"] == 1

    measurement = body["measurements"][0]
    assert measurement["sensor_id"] == "s-1"
    assert measurement["variable"] == "temperature"
    assert measurement["value"] == 21.5
    assert measurement["unit"] == "C"
    assert measurement["timestamp"] == "2026-09-22T00:00:00+00:00"
    assert measurement["location"]["latitude"] == LATITUDE
    assert measurement["location"]["longitude"] == LONGITUDE
    assert measurement["location"]["h3_index"] == expected_cell
    assert measurement["quality_flags"] == []
    assert measurement["metadata"] == {}

    assert body["query_parameters"]["h3_index"] == expected_cell
    assert body["query_parameters"]["h3_resolution"] == H3_RESOLUTION

    # A different valid cell filters the measurement out.
    other_cell = h3.latlng_to_cell(35.0, 139.7, H3_RESOLUTION)
    miss = client.get("/measurements", params={"h3_index": other_cell})
    assert miss.status_code == 200
    assert miss.json()["total_count"] == 0
    assert miss.json()["measurements"] == []

    # Invalid H3 index is rejected before filtering.
    assert (
        client.get("/measurements", params={"h3_index": "not-a-cell"}).status_code
        == 400
    )
