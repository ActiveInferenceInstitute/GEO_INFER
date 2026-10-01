"""Tests for the /ws/sensor-stream measurement forward loop.

The endpoint previously only sent heartbeats: measurements submitted through
the shared IoTDataIngestion instance never reached subscribed sockets.
"""

import json
from datetime import datetime, UTC

import h3
from fastapi.testclient import TestClient

from geo_infer_iot.api.streaming_api import StreamingAPI
from geo_infer_iot.core.ingestion import SensorMeasurement


def _measurement(sensor_id: str = "s1", value: float = 21.5) -> SensorMeasurement:
    return SensorMeasurement(
        sensor_id=sensor_id,
        timestamp=datetime.now(UTC),
        variable="temperature",
        value=value,
        unit="C",
        latitude=40.0,
        longitude=-74.0,
    )


def _drain_until(ws, wanted_type: str, max_messages: int = 6):
    """Read up to max_messages, skipping heartbeats, until wanted_type arrives."""
    for _ in range(max_messages):
        message = ws.receive_json()
        if message.get("type") == wanted_type:
            return message
    return None


def test_subscribed_websocket_receives_pushed_measurement():
    api = StreamingAPI({})
    with TestClient(api.get_app()) as client:
        with client.websocket_connect("/ws/sensor-stream") as ws:
            ws.send_text(json.dumps({"sensor_ids": ["s1"], "h3_indices": []}))
            confirmed = ws.receive_json()
            assert confirmed["type"] == "subscription_confirmed"

            api.ingestion.measurements.append(_measurement("s1", 21.5))

            message = _drain_until(ws, "sensor_measurement")
            assert message is not None
            assert message["data"]["sensor_id"] == "s1"
            assert message["data"]["value"] == 21.5
            assert message["data"]["variable"] == "temperature"
            assert message["data"]["h3_index"] == "882a13ba47fffff"


def test_spatial_subscription_receives_spatial_measurement():
    api = StreamingAPI({})
    h3_index = h3.latlng_to_cell(40.0, -74.0, 8)
    with TestClient(api.get_app()) as client:
        with client.websocket_connect("/ws/sensor-stream") as ws:
            ws.send_text(json.dumps({"sensor_ids": [], "h3_indices": [h3_index]}))
            assert ws.receive_json()["type"] == "subscription_confirmed"

            api.ingestion.measurements.append(_measurement("other-sensor"))

            message = _drain_until(ws, "spatial_measurement")
            assert message is not None
            assert message["h3_index"] == h3_index
            assert message["data"]["sensor_id"] == "other-sensor"


def test_unsubscribed_sensor_is_not_forwarded():
    api = StreamingAPI({})
    with TestClient(api.get_app()) as client:
        with client.websocket_connect("/ws/sensor-stream") as ws:
            ws.send_text(json.dumps({"sensor_ids": ["s2"], "h3_indices": []}))
            assert ws.receive_json()["type"] == "subscription_confirmed"

            api.ingestion.measurements.append(_measurement("s1"))

            for _ in range(3):
                message = ws.receive_json()
                assert message["type"] == "heartbeat"
