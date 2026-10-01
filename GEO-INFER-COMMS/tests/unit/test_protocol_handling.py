"""Tests for COMMS protocol handling and message serialization."""

import json
import pytest
from datetime import datetime, UTC

from geo_infer_comms.models.message import (
    MessagePriority,
    MessageType,
    MessageStatus,
    EventType,
)


class TestMessageSerialization:
    """Test message serialization and deserialization for protocol handling."""

    def test_serialize_text_message(self):
        message = {
            "type": MessageType.TEXT.value,
            "content": "Hello world",
            "priority": MessagePriority.NORMAL.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        serialized = json.dumps(message)
        deserialized = json.loads(serialized)
        assert deserialized["type"] == "text"
        assert deserialized["content"] == "Hello world"

    def test_serialize_alert_message(self):
        message = {
            "type": MessageType.ALERT.value,
            "content": "Sensor threshold exceeded",
            "priority": MessagePriority.URGENT.value,
            "metadata": {"sensor_id": "s-001", "value": 95.5},
        }
        serialized = json.dumps(message)
        deserialized = json.loads(serialized)
        assert deserialized["priority"] == "urgent"
        assert deserialized["metadata"]["sensor_id"] == "s-001"

    def test_serialize_location_message(self):
        message = {
            "type": MessageType.LOCATION.value,
            "coordinates": {"latitude": 37.7749, "longitude": -122.4194},
            "priority": MessagePriority.LOW.value,
        }
        serialized = json.dumps(message)
        deserialized = json.loads(serialized)
        assert deserialized["coordinates"]["latitude"] == 37.7749


class TestEventTypes:
    def test_all_event_types_have_values(self):
        for event_type in EventType:
            assert event_type.value is not None
            assert len(event_type.value) > 0

    def test_event_type_from_string(self):
        assert EventType("data_update") == EventType.DATA_UPDATE
        assert EventType("system_alert") == EventType.SYSTEM_ALERT

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValueError):
            EventType("nonexistent")


class TestMessageStatusTransitions:
    def test_valid_status_values(self):
        valid = {
            MessageStatus.SENT,
            MessageStatus.DELIVERED,
            MessageStatus.READ,
            MessageStatus.FAILED,
            MessageStatus.QUEUED,
            MessageStatus.PROCESSING,
        }
        assert len(valid) == 6

    def test_status_from_string(self):
        assert MessageStatus("sent") == MessageStatus.SENT
        assert MessageStatus("delivered") == MessageStatus.DELIVERED
        assert MessageStatus("failed") == MessageStatus.FAILED
