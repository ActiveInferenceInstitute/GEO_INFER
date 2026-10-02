"""Independent UTC and coordinate membership regressions for IoT boundaries."""

from datetime import UTC, datetime, timedelta, timezone

import h3
import pytest

from geo_infer_iot.models.sensor import Location, Sensor
from geo_infer_iot.models.measurement import Measurement, MeasurementBatch
from geo_infer_iot.core.registry import SensorRegistry, SensorMetadata


def sensor(**kwargs):
    return Sensor(
        sensor_id="sensor",
        network_id="network",
        sensor_type="temperature",
        location=Location(latitude=41.75, longitude=-124.2),
        **kwargs,
    )


def measurement(**kwargs):
    return Measurement(
        measurement_id="measurement",
        sensor_id="sensor",
        variable="temperature",
        value=1.0,
        unit="C",
        latitude=41.75,
        longitude=-124.2,
        **kwargs,
    )


def test_sensor_health_uses_aware_utc_and_offset_inputs():
    local = datetime.now(timezone(timedelta(hours=-7)))
    record = sensor(
        last_communication=local,
        calibration={"last_calibration": local - timedelta(days=200)},
    )
    assert record.last_communication.tzinfo is UTC
    assert record.calibration.last_calibration.tzinfo is UTC
    assert record.created_at.tzinfo is UTC
    assert record.updated_at.tzinfo is UTC
    assert record.get_health_score() == pytest.approx(0.85)


@pytest.mark.parametrize(
    "field", ["created_at", "updated_at", "last_communication", "operational_since"]
)
def test_sensor_naive_inputs_and_assignments_fail(field):
    naive = datetime(2026, 10, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        sensor(**{field: naive})
    record = sensor()
    original = getattr(record, field)
    with pytest.raises(ValueError, match="timezone-aware"):
        setattr(record, field, naive)
    assert getattr(record, field) == original


@pytest.mark.parametrize("value", [1672531200, 1672531200.0, True])
@pytest.mark.parametrize(
    "factory,field", [(sensor, "created_at"), (measurement, "timestamp")]
)
def test_numeric_instants_require_explicit_unit_conversion(factory, field, value):
    with pytest.raises(ValueError, match="datetime or an ISO-8601 string"):
        factory(**{field: value})
    record = factory()
    original = getattr(record, field)
    with pytest.raises(ValueError, match="datetime or an ISO-8601 string"):
        setattr(record, field, value)
    assert getattr(record, field) == original


@pytest.mark.parametrize("value", [datetime(2026, 10, 1), 1672531200])
def test_batch_timestamp_mapping_rejects_implicit_instants(value):
    valid = {"start": datetime.now(UTC), "end": datetime.now(UTC)}
    batch = MeasurementBatch(
        batch_id="batch",
        allow_empty=True,
        batch_size=0,
        measurements=[],
        time_range=valid,
    )
    original = batch.model_dump()
    with pytest.raises(ValueError, match="timestamp must"):
        MeasurementBatch(
            batch_id="batch",
            allow_empty=True,
            batch_size=0,
            measurements=[],
            time_range={"start": value, "end": datetime.now(UTC)},
        )
    with pytest.raises(ValueError, match="timestamp must"):
        batch.time_range = {"start": value, "end": datetime.now(UTC)}
    assert batch.model_dump() == original


@pytest.mark.parametrize("factory", [sensor, measurement])
def test_resolution_zero_and_invalid_update_are_atomic(factory):
    record = factory()
    before = record.model_dump()
    with pytest.raises(ValueError):
        record.update_location(91.0, 0.0, h3_resolution=0)
    assert record.model_dump() == before
    record.update_location(0.0, 0.0, h3_resolution=0)
    location = record.location if isinstance(record, Sensor) else record
    assert location.h3_resolution == 0
    assert location.h3_index == h3.latlng_to_cell(0.0, 0.0, 0)


def test_measurement_constructs_resolution_zero_cell():
    record = measurement(h3_resolution=0)
    assert record.h3_index == h3.latlng_to_cell(41.75, -124.2, 0)


def test_calibration_update_rejects_naive_time_atomically():
    record = sensor()
    before = record.model_dump()
    with pytest.raises(ValueError, match="timezone-aware"):
        record.update_calibration(
            {
                "last_calibration": datetime(2026, 1, 1),
                "calibration_parameters": {"slope": 2},
            }
        )
    assert record.model_dump() == before


@pytest.mark.parametrize("resolution", [0, 7, 8, 15])
@pytest.mark.parametrize(
    "bounds",
    [
        {"lat_min": 41.74, "lat_max": 41.76, "lon_min": -124.21, "lon_max": -124.19},
        {
            "lat_min": 41.749999,
            "lat_max": 41.750001,
            "lon_min": -124.200001,
            "lon_max": -124.199999,
        },
        {"lat_min": 41.75, "lat_max": 41.75, "lon_min": -124.2, "lon_max": -124.2},
        {"lat_min": -90, "lat_max": 90, "lon_min": 170, "lon_max": -170},
    ],
)
def test_bbox_membership_matches_direct_oracle(bounds, resolution):
    registry = SensorRegistry()
    records = [
        ("center", 41.75, -124.2, 8),
        ("west", 0.0, -179.0, 0),
        ("east", 0.0, 179.0, 15),
        ("outside", 50.0, 0.0, 7),
    ]
    for name, latitude, longitude, sensor_resolution in records:
        registry.register_sensor(
            {
                "sensor_id": name,
                "network_id": "network",
                "sensor_type": "temperature",
                "latitude": latitude,
                "longitude": longitude,
                "h3_resolution": sensor_resolution,
            }
        )
    expected = set()
    for name, lat, lon, _ in records:
        in_longitude = (
            bounds["lon_min"] <= lon <= bounds["lon_max"]
            if bounds["lon_min"] <= bounds["lon_max"]
            else lon >= bounds["lon_min"] or lon <= bounds["lon_max"]
        )
        if bounds["lat_min"] <= lat <= bounds["lat_max"] and in_longitude:
            expected.add(name)
    assert {
        item.sensor_id for item in registry.get_sensors_in_area(bounds, resolution)
    } == expected


def test_registry_rejects_naive_timestamp_and_duplicate_ids():
    with pytest.raises(ValueError, match="timezone-aware"):
        SensorMetadata(
            sensor_id="sensor",
            network_id="network",
            sensor_type="temperature",
            latitude=0,
            longitude=0,
            registered_at=datetime(2026, 1, 1),
        )
    registry = SensorRegistry()
    info = {
        "sensor_id": "sensor",
        "network_id": "network",
        "sensor_type": "temperature",
        "latitude": 0,
        "longitude": 0,
    }
    registry.register_sensor(info)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_sensor(info)
    assert len(registry.sensors) == 1
