"""Tests for COMMS validation utilities."""

import pytest
from datetime import datetime, timedelta, UTC

from geo_infer_comms.utils.validation import (
    sanitize_message_content,
    validate_and_sanitize_inputs,
    validate_channel_id,
    validate_collaboration_session,
    validate_configuration,
    validate_coordinates,
    validate_crs,
    validate_delivery_methods,
    validate_email,
    validate_event_type,
    validate_file_size,
    validate_geojson_feature,
    validate_geojson_geometry,
    validate_message_content,
    validate_message_priority,
    validate_message_recipients,
    validate_message_type,
    validate_notification_type,
    validate_phone,
    validate_spatial_bounds,
    validate_spatial_filter,
    validate_stream_config,
    validate_timestamp,
    validate_url,
    validate_user_id,
)


class TestCoordinateValidation:
    def test_valid_coordinates(self):
        assert validate_coordinates(-122.4, 37.7) is True
        assert validate_coordinates(0.0, 0.0) is True
        assert validate_coordinates(180.0, 90.0) is True
        assert validate_coordinates(-180.0, -90.0) is True

    def test_invalid_longitude(self):
        assert validate_coordinates(181.0, 0.0) is False
        assert validate_coordinates(-181.0, 0.0) is False

    def test_invalid_latitude(self):
        assert validate_coordinates(0.0, 91.0) is False
        assert validate_coordinates(0.0, -91.0) is False


class TestCRSValidation:
    def test_valid_crs(self):
        assert validate_crs("EPSG:4326") is True

    def test_invalid_crs_empty(self):
        assert validate_crs("") is False


class TestContactValidation:
    def test_email(self):
        assert validate_email("person@example.com") is True
        assert validate_email("first.last+tag@sub.example.org") is True
        assert validate_email("no-at-sign") is False
        assert validate_email("missing-tld@example") is False
        assert validate_email(123) is False

    def test_phone(self):
        assert validate_phone("1 (415) 555-0100") is True
        assert validate_phone("4155550100") is True
        assert validate_phone("12345") is False  # too short
        assert validate_phone("abcdefghij") is False
        assert validate_phone(4155550100) is False

    def test_url(self):
        assert validate_url("https://example.com/path?q=1") is True
        assert validate_url("http://sub.example.org") is True
        assert validate_url("ftp://example.com") is False
        assert validate_url(None) is False


class TestMessageValidation:
    def test_message_content(self):
        assert validate_message_content("hello world") is True
        assert validate_message_content("   ") is False
        assert validate_message_content("") is False
        assert validate_message_content(None) is False
        assert validate_message_content("x" * 11, max_length=10) is False

    def test_message_priority(self):
        for priority in ("low", "NORMAL", "High", "urgent"):
            assert validate_message_priority(priority) is True
        assert validate_message_priority("critical") is False
        assert validate_message_priority("") is False

    def test_message_type(self):
        assert validate_message_type("text") is True
        assert validate_message_type("SENSOR_DATA") is True
        assert validate_message_type("gossip") is False

    def test_user_and_channel_ids(self):
        assert validate_user_id("user-1") is True
        assert validate_user_id("A_b.c") is True
        assert validate_user_id("   ") is False
        assert validate_user_id("bad id!") is False
        assert validate_user_id("x" * 101) is False
        assert validate_user_id(None) is False
        assert validate_channel_id("channel-1") is True
        assert validate_channel_id("bad channel") is False

    def test_message_recipients(self):
        assert validate_message_recipients(["user-1", "user-2"]) is True
        assert validate_message_recipients([]) is False
        assert validate_message_recipients("user-1") is False
        assert validate_message_recipients(["bad recipient"]) is False
        assert validate_message_recipients(["ok"] * 1001) is False


class TestSpatialValidation:
    def test_spatial_bounds(self):
        bounds = {
            "min_longitude": -122.6,
            "min_latitude": 37.6,
            "max_longitude": -122.2,
            "max_latitude": 37.9,
        }
        assert validate_spatial_bounds(bounds) is True
        assert validate_spatial_bounds("not-a-dict") is False
        assert validate_spatial_bounds({"min_longitude": 1}) is False
        inverted = dict(bounds, min_longitude=0.0, max_longitude=-1.0)
        assert validate_spatial_bounds(inverted) is False
        degenerate = dict(bounds, min_latitude=37.9, max_latitude=37.9)
        assert validate_spatial_bounds(degenerate) is False
        non_numeric = dict(bounds, min_longitude="west")
        assert validate_spatial_bounds(non_numeric) is False

    def test_geojson_geometry(self):
        point = {"type": "Point", "coordinates": [-122.4, 37.8]}
        assert validate_geojson_geometry(point) is True
        assert (
            validate_geojson_geometry({"type": "Point", "coordinates": ["x", 1]})
            is False
        )
        assert (
            validate_geojson_geometry(
                {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
            )
            is True
        )
        assert (
            validate_geojson_geometry({"type": "LineString", "coordinates": [[0, 0]]})
            is False
        )
        assert (
            validate_geojson_geometry(
                {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
            )
            is True
        )
        assert (
            validate_geojson_geometry(
                {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}
            )
            is False
        )
        assert (
            validate_geojson_geometry(
                {"type": "MultiPoint", "coordinates": [[0, 0], [1, 1]]}
            )
            is True
        )
        assert validate_geojson_geometry({"type": "Point"}) is False
        assert validate_geojson_geometry({"type": "Vortex", "coordinates": []}) is False
        assert (
            validate_geojson_geometry({"type": "Point", "coordinates": "0,0"}) is False
        )
        assert validate_geojson_geometry("Point") is False

    def test_geojson_feature(self):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-122.4, 37.8]},
            "properties": {"name": "office"},
        }
        assert validate_geojson_feature(feature) is True
        assert validate_geojson_feature(feature | {"properties": "flat"}) is False
        assert validate_geojson_feature(feature | {"geometry": None}) is False
        assert validate_geojson_feature({"type": "Feature"}) is False
        assert validate_geojson_feature({"geometry": {}}) is False
        assert validate_geojson_feature([feature]) is False

    def test_spatial_filter(self):
        bounds_filter = {
            "filter_type": "bounds",
            "parameters": {
                "bounds": {
                    "min_longitude": -122.6,
                    "min_latitude": 37.6,
                    "max_longitude": -122.2,
                    "max_latitude": 37.9,
                }
            },
        }
        assert validate_spatial_filter(bounds_filter) is True
        assert validate_spatial_filter(
            {
                "filter_type": "radius",
                "parameters": {
                    "center": {"longitude": -122.4, "latitude": 37.8},
                    "radius_meters": 500,
                },
            }
        )
        assert (
            validate_spatial_filter(
                {
                    "filter_type": "radius",
                    "parameters": {"center": {}, "radius_meters": -1},
                }
            )
            is False
        )
        assert (
            validate_spatial_filter(
                {
                    "filter_type": "polygon",
                    "parameters": {"polygon": [[0, 0], [1, 0], [1, 1]]},
                }
            )
            is True
        )
        assert (
            validate_spatial_filter(
                {
                    "filter_type": "proximity",
                    "parameters": {
                        "target_location": {"longitude": -122.4, "latitude": 37.8},
                        "max_distance_meters": 100,
                    },
                }
            )
            is True
        )
        assert (
            validate_spatial_filter(
                {
                    "filter_type": "proximity",
                    "parameters": {"target_location": {}},
                }
            )
            is False
        )
        assert validate_spatial_filter({"filter_type": "bounds"}) is False
        assert (
            validate_spatial_filter({"filter_type": "vortex", "parameters": {}})
            is False
        )
        assert (
            validate_spatial_filter({"filter_type": "bounds", "parameters": "flat"})
            is False
        )
        assert validate_spatial_filter("bounds") is False


class TestNotificationAndEventValidation:
    def test_notification_type(self):
        for kind in ("info", "WARNING", "Error", "success", "reminder"):
            assert validate_notification_type(kind) is True
        assert validate_notification_type("panic") is False

    def test_delivery_methods(self):
        assert validate_delivery_methods(["in_app", "email"]) is True
        assert validate_delivery_methods([]) is True
        assert validate_delivery_methods(["carrier_pigeon"]) is False
        assert validate_delivery_methods("email") is False

    def test_event_type(self):
        assert validate_event_type("data_update") is True
        assert validate_event_type("SYSTEM_ALERT") is True
        assert validate_event_type("mystery") is False

    def test_timestamp(self):
        assert validate_timestamp("2026-01-01T00:00:00+00:00") is True
        assert validate_timestamp("2026-01-01T00:00:00Z") is True
        assert validate_timestamp(datetime.now(UTC)) is True
        assert validate_timestamp(datetime.now(UTC) + timedelta(days=400)) is False
        assert validate_timestamp("not-a-timestamp") is False
        assert validate_timestamp(12345) is False

    def test_file_size(self):
        assert validate_file_size(1024) is True
        assert validate_file_size(11 * 1024 * 1024) is False
        assert validate_file_size(2 * 1024 * 1024, max_size_mb=1.0) is False


class TestSessionAndStreamValidation:
    def test_collaboration_session(self):
        session = {
            "name": "planning session",
            "session_type": "planning",
            "participants": ["user-1", "user-2"],
            "duration": 60,
            "features": ["screen_share"],
        }
        assert validate_collaboration_session(session) is True
        assert (
            validate_collaboration_session(session | {"session_type": "party"}) is False
        )
        assert validate_collaboration_session(session | {"duration": 0}) is False
        assert validate_collaboration_session(session | {"duration": 481}) is False
        assert (
            validate_collaboration_session(session | {"features": ["hologram"]})
            is False
        )
        assert validate_collaboration_session(session | {"participants": []}) is False
        assert validate_collaboration_session(session | {"name": "   "}) is False
        assert validate_collaboration_session(session | {"name": "x" * 201}) is False
        assert (
            validate_collaboration_session(session | {"participants": ["bad!"]})
            is False
        )
        assert validate_collaboration_session("session") is False

    def test_stream_config(self):
        stream = {"name": "sensor feed", "stream_type": "sensor"}
        assert validate_stream_config(stream) is True
        assert (
            validate_stream_config(
                stream
                | {
                    "geospatial_filter": {
                        "filter_type": "radius",
                        "parameters": {
                            "center": {"longitude": -122.4, "latitude": 37.8},
                            "radius_meters": 100,
                        },
                    }
                }
            )
            is True
        )
        assert (
            validate_stream_config(
                stream
                | {"geospatial_filter": {"filter_type": "bogus", "parameters": {}}}
            )
            is False
        )
        assert validate_stream_config({"name": "x"}) is False
        assert validate_stream_config({"name": "", "stream_type": "data"}) is False
        assert validate_stream_config({"name": "x", "stream_type": "hologram"}) is False
        assert validate_stream_config({"name": "x", "stream_type": "data"}) is True
        assert validate_stream_config("stream") is False


class TestSanitization:
    def test_sanitize_message_content(self):
        assert (
            sanitize_message_content("  hello <script>alert(1)</script>  ") == "hello"
        )
        assert sanitize_message_content("click javascript:alert(1)") == "click alert(1)"
        assert sanitize_message_content('<a onclick="x()">y</a>') == '<a "x()">y</a>'
        assert sanitize_message_content("clean text") == "clean text"
        assert sanitize_message_content(None) == ""
        assert sanitize_message_content(123) == ""

    def test_validate_and_sanitize_inputs(self):
        results = validate_and_sanitize_inputs(
            contact_email="Person@Example.com",
            message="  hi <script>x()</script> ",
            user_id="user-1",
            priority="urgent",
        )
        assert results["contact_email"] == "person@example.com"
        assert results["message"] == "hi"
        assert results["user_id"] == "user-1"
        assert results["priority"] == "urgent"  # unmanaged keys pass through

    def test_validate_and_sanitize_inputs_rejects_bad_email(self):
        with pytest.raises(ValueError, match="Invalid email"):
            validate_and_sanitize_inputs(contact_email="no-at-sign")
        with pytest.raises(ValueError, match="Invalid user/channel ID"):
            validate_and_sanitize_inputs(user_id="bad id!")

    def test_validate_configuration(self):
        assert validate_configuration({"a": 1, "b": 2}, ["a"]) is True
        assert validate_configuration({"a": 1}, ["a", "b"]) is False
        assert validate_configuration(None, []) is False
