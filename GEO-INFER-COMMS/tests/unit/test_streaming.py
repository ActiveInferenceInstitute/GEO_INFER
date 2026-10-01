"""Unit tests for the streaming system: buffers, managers, protocols, analytics."""

import asyncio
import time
from datetime import datetime, timedelta, UTC

import pytest

from geo_infer_comms.core.streaming import (
    DataStream,
    GeospatialDataStream,
    MQTTStreamingProtocol,
    ServerSentEventsProtocol,
    StreamManager,
    StreamMetrics,
    StreamingAnalytics,
    StreamingOrchestrator,
    StreamingProtocol,
    StreamingProtocolManager,
    WebSocketStreamingProtocol,
)
from geo_infer_comms.models.message import StreamRequest
from geo_infer_comms.models.spatial import GeospatialMetadata, GeospatialPoint

BAY_AREA_BOUNDS = {
    "min_longitude": -122.6,
    "min_latitude": 37.6,
    "max_longitude": -122.2,
    "max_latitude": 37.9,
}


def _bounds_request(name: str = "bay-stream") -> StreamRequest:
    return StreamRequest(
        name=name,
        stream_type="data",
        geospatial_filter={
            "filter_type": "bounds",
            "parameters": {"bounds": BAY_AREA_BOUNDS},
        },
    )


def _sf_metadata(lon: float = -122.4, lat: float = 37.8) -> GeospatialMetadata:
    return GeospatialMetadata(location=GeospatialPoint(longitude=lon, latitude=lat))


class TestDataStream:
    def test_add_data_point_requires_active_stream(self) -> None:
        stream = DataStream("s1", _bounds_request())

        assert stream.add_data_point({"temp": 20}) is False
        assert stream.data_points_sent == 0

    def test_active_stream_buffers_data_points(self) -> None:
        stream = DataStream("s1", _bounds_request())
        stream.is_active = True

        assert stream.add_data_point({"temp": 20}) is True
        assert stream.data_points_sent == 1
        assert stream.bytes_transferred > 0

        received = stream.get_data_points(count=5)
        assert len(received) == 1
        assert received[0]["data"] == {"temp": 20}
        assert received[0]["stream_id"] == "s1"
        assert received[0]["geospatial_context"] is None

    def test_geospatial_filter_rejects_out_of_bounds_points(self) -> None:
        stream = DataStream("s1", _bounds_request())
        stream.is_active = True

        inside = stream.add_data_point("ok", _sf_metadata(-122.4, 37.8))
        outside = stream.add_data_point("no", _sf_metadata(-74.0, 40.7))

        assert inside is True
        assert outside is False
        assert stream.data_points_sent == 1

    def test_buffer_full_returns_false(self) -> None:
        stream = DataStream("s1", _bounds_request(), buffer_size=1)
        stream.is_active = True

        assert stream.add_data_point("first") is True
        assert stream.add_data_point("second") is False

    def test_get_data_points_returns_empty_on_timeout(self) -> None:
        stream = DataStream("s1", _bounds_request())

        assert stream.get_data_points(count=3, timeout=0.01) == []

    def test_get_stats_reports_state(self) -> None:
        stream = DataStream("s1", _bounds_request())
        stream.is_active = True

        stats = stream.get_stats()
        assert stats["stream_id"] == "s1"
        assert stats["is_active"] is True
        assert stats["stream_type"] == "data"
        assert stats["geospatial_filter"] is not None
        assert stats["data_points_sent"] == 0

    def test_get_stats_without_geospatial_filter(self) -> None:
        stream = DataStream("s1", StreamRequest(name="plain"))
        assert stream.get_stats()["geospatial_filter"] is None


class TestStreamManager:
    def test_create_stream_registers_and_indexes(self) -> None:
        manager = StreamManager(enable_persistence=False)
        response = manager.create_stream(_bounds_request(), creator_id="creator-1")

        stored = manager.get_stream(response.stream_id)
        assert stored is not None
        assert stored.config.name == "bay-stream"
        assert manager.metrics.streams_created == 1
        # Bounds-filtered streams are indexed by their center location.
        assert len(manager.spatial_streams) == 1

    def test_create_stream_rejects_invalid_geospatial_filter(self) -> None:
        manager = StreamManager(enable_persistence=False)
        request = StreamRequest(
            name="bad-filter",
            geospatial_filter={"filter_type": "bogus", "parameters": {}},
        )

        with pytest.raises(ValueError, match="Invalid stream configuration"):
            manager.create_stream(request, creator_id="creator-1")

    def test_create_stream_enforces_limit(self) -> None:
        manager = StreamManager(max_streams=1, enable_persistence=False)
        manager.create_stream(StreamRequest(name="one"), creator_id="creator-1")

        with pytest.raises(ValueError, match="Maximum number of streams"):
            manager.create_stream(StreamRequest(name="two"), creator_id="creator-1")

    def test_get_streams_filters_by_type_and_limit(self) -> None:
        manager = StreamManager(enable_persistence=False)
        for index in range(3):
            stream_type = "data" if index % 2 == 0 else "video"
            manager.create_stream(
                StreamRequest(name=f"s{index}", stream_type=stream_type),
                creator_id="creator-1",
            )

        data_streams = manager.get_streams(stream_type="data")
        assert [s.config.name for s in data_streams] == ["s0", "s2"]

        limited = manager.get_streams(limit=2)
        assert len(limited) == 2

    def test_subscribe_unsubscribe_lifecycle(self) -> None:
        manager = StreamManager(enable_persistence=False)
        response = manager.create_stream(
            StreamRequest(name="s"), creator_id="creator-1"
        )
        stream_id = response.stream_id

        assert manager.subscribe_to_stream(stream_id, "user-1") is True
        assert manager.subscribe_to_stream(stream_id, "user-1") is True
        stored = manager.get_stream(stream_id)
        assert stored is not None
        assert stored.subscribers == {"user-1"}
        assert manager.stream_subscribers[stream_id] == {"user-1"}

        assert manager.unsubscribe_from_stream(stream_id, "user-1") is True
        assert stored.subscribers == set()
        assert manager.unsubscribe_from_stream("missing", "user-1") is True

    def test_subscribe_to_missing_stream(self) -> None:
        manager = StreamManager(enable_persistence=False)
        assert manager.subscribe_to_stream("missing", "user-1") is False

    def test_publish_requires_active_stream(self) -> None:
        manager = StreamManager(enable_persistence=False)
        response = manager.create_stream(
            StreamRequest(name="s"), creator_id="creator-1"
        )

        assert manager.publish_to_stream("missing", "x") is False

        stored = manager.get_stream(response.stream_id)
        assert stored is not None
        assert stored.is_active is False
        assert manager.publish_to_stream(response.stream_id, "x") is False

        stored.is_active = True
        assert manager.publish_to_stream(response.stream_id, {"v": 1}) is True
        assert (
            manager.publish_to_stream(response.stream_id, "geo", _sf_metadata()) is True
        )

    def test_get_streams_by_location(self) -> None:
        manager = StreamManager(enable_persistence=False)
        manager.create_stream(_bounds_request(), creator_id="creator-1")

        nearby = manager.get_streams_by_location(
            GeospatialPoint(longitude=-122.42, latitude=37.77), radius_km=10
        )
        assert len(nearby) == 1

        far = manager.get_streams_by_location(
            GeospatialPoint(longitude=-74.0, latitude=40.7), radius_km=10
        )
        assert far == []

    def test_radius_filter_is_not_spatially_indexed(self) -> None:
        manager = StreamManager(enable_persistence=False)
        manager.create_stream(
            StreamRequest(
                name="radius-stream",
                geospatial_filter={
                    "filter_type": "radius",
                    "parameters": {
                        "center": {"longitude": -122.4, "latitude": 37.8},
                        "radius_meters": 1000,
                    },
                },
            ),
            creator_id="creator-1",
        )

        assert manager.spatial_streams == {}
        assert (
            manager.get_streams_by_location(
                GeospatialPoint(longitude=-122.4, latitude=37.8)
            )
            == []
        )

    def test_get_stream_statistics(self) -> None:
        manager = StreamManager(enable_persistence=False)
        response = manager.create_stream(
            StreamRequest(name="s"), creator_id="creator-1"
        )
        manager.subscribe_to_stream(response.stream_id, "user-1")

        stats = manager.get_stream_statistics()
        assert stats["total_streams"] == 1
        assert stats["active_streams"] == 0
        assert stats["total_subscribers"] == 1
        assert "metrics" in stats

    def test_start_stop_processes_buffered_data(self) -> None:
        manager = StreamManager(enable_persistence=False)
        manager.start()
        try:
            manager.start()  # idempotent

            response = manager.create_stream(
                StreamRequest(name="live"), creator_id="creator-1"
            )
            stream = manager.get_stream(response.stream_id)
            assert stream is not None
            stream.is_active = True
            assert manager.publish_to_stream(response.stream_id, {"v": 1}) is True

            deadline = time.monotonic() + 5.0
            while (
                manager.metrics.data_points_delivered == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)

            assert manager.metrics.data_points_delivered >= 1
        finally:
            manager.stop()
            assert manager._running is False


class TestStreamMetrics:
    def test_to_dict_and_reset(self) -> None:
        metrics = StreamMetrics(streams_created=3, data_points_delivered=2)
        report = metrics.to_dict()
        assert report["streams_created"] == 3
        assert report["delivery_success_rate"] == 200.0
        assert report["uptime_seconds"] >= 0

        metrics.reset()
        assert metrics.to_dict()["streams_created"] == 0
        assert metrics.data_points_delivered == 0


class TestGeospatialDataStream:
    def _stream(self) -> GeospatialDataStream:
        return GeospatialDataStream("geo-1", {"format": "sensor"})

    def test_add_and_query_geospatial_data(self) -> None:
        stream = self._stream()
        point = GeospatialPoint(longitude=-122.4, latitude=37.8)

        stream.add_geospatial_data(point, 10.0)
        stream.add_geospatial_data(point, 12.0)

        nearby = stream.get_data_at_location(point, radius_km=1.0)
        assert len(nearby) == 1
        assert nearby[0]["data"] == 12.0

        series = stream.get_temporal_series(point)
        assert len(series) == 2

        agg = stream.spatial_aggregations[stream._generate_location_key(point)]
        assert agg["count"] == 2
        assert agg["min"] == 10.0
        assert agg["max"] == 12.0
        assert agg["avg"] == pytest.approx(11.0)

    def test_non_numeric_data_skips_aggregation(self) -> None:
        stream = self._stream()
        point = GeospatialPoint(longitude=-122.4, latitude=37.8)

        stream.add_geospatial_data(point, "label")

        assert stream.temporal_data[stream._generate_location_key(point)]
        assert stream._generate_location_key(point) not in stream.spatial_aggregations

    def test_temporal_window_filters_old_points(self) -> None:
        stream = self._stream()
        point = GeospatialPoint(longitude=-122.4, latitude=37.8)
        old_timestamp = datetime.now(UTC) - timedelta(hours=1)

        stream.add_geospatial_data(point, "old", timestamp=old_timestamp)
        stream.add_geospatial_data(point, "new")

        recent = stream.get_temporal_series(point, time_window=timedelta(minutes=5))
        assert [p["data"] for p in recent] == ["new"]

    def test_location_key_rounds_to_resolution(self) -> None:
        stream = self._stream()

        key = stream._generate_location_key(
            GeospatialPoint(longitude=-122.4000001, latitude=37.7999999)
        )
        assert key == "-122.400000,37.800000"

    def test_hotspot_detection_flags_dense_locations(self) -> None:
        stream = self._stream()
        dense = GeospatialPoint(longitude=-122.4, latitude=37.8)
        sparse = GeospatialPoint(longitude=-74.0, latitude=40.7)

        for _ in range(10):
            stream.add_geospatial_data(dense, 1.0)
        stream.add_geospatial_data(sparse, 1.0)

        hotspot_keys = {h["location_key"] for h in stream.hotspots}
        assert stream._generate_location_key(dense) in hotspot_keys
        assert stream._generate_location_key(sparse) not in hotspot_keys

    def test_anomaly_detection_flags_large_deviations(self) -> None:
        stream = self._stream()
        point = GeospatialPoint(longitude=-122.4, latitude=37.8)

        for _ in range(10):
            stream.add_geospatial_data(point, 10.0)
        stream.add_geospatial_data(point, 100.0)

        assert stream.anomalies, "expected the outlier value to be flagged"
        assert stream.anomalies[0]["location_key"] == stream._generate_location_key(
            point
        )


class TestStreamingProtocols:
    def test_protocol_manager_lists_and_registers(self) -> None:
        manager = StreamManager(enable_persistence=False)
        protocols = StreamingProtocolManager(manager)

        assert protocols.list_available_protocols() == ["websocket", "mqtt", "sse"]
        assert isinstance(
            protocols.get_protocol("websocket"), WebSocketStreamingProtocol
        )
        assert protocols.get_protocol("missing") is None

        custom = StreamingProtocol(manager)
        protocols.register_protocol("custom", custom)
        assert protocols.get_protocol("custom") is custom

    def test_base_protocol_requires_implementation(self) -> None:
        manager = StreamManager(enable_persistence=False)
        protocol = StreamingProtocol(manager)

        with pytest.raises(RuntimeError, match="start_streaming"):
            asyncio.run(protocol.start_streaming("s1"))
        with pytest.raises(RuntimeError, match="stop_streaming"):
            asyncio.run(protocol.stop_streaming("s1"))

    def test_websocket_streaming_lifecycle(self) -> None:
        manager = StreamManager(enable_persistence=False)
        protocol = WebSocketStreamingProtocol(manager)

        assert asyncio.run(protocol.start_streaming("s1")) is True
        assert "s1" in protocol.active_connections
        assert asyncio.run(protocol.stop_streaming("s1")) is True
        assert "s1" not in protocol.active_connections

    def test_mqtt_streaming_lifecycle(self) -> None:
        manager = StreamManager(enable_persistence=False)
        protocol = MQTTStreamingProtocol(manager)

        assert asyncio.run(protocol.start_streaming("s1")) is True
        assert protocol.mqtt_topics["s1"] == "geoinfer/streams/s1"
        assert asyncio.run(protocol.stop_streaming("s1")) is True
        assert "s1" not in protocol.mqtt_topics

    def test_sse_streaming_lifecycle(self) -> None:
        manager = StreamManager(enable_persistence=False)
        protocol = ServerSentEventsProtocol(manager)

        assert asyncio.run(protocol.start_streaming("s1")) is True
        assert protocol.sse_clients["s1"] == []
        assert asyncio.run(protocol.stop_streaming("s1")) is True
        assert "s1" not in protocol.sse_clients

    def test_protocol_stats_report_class_name(self) -> None:
        manager = StreamManager(enable_persistence=False)
        protocol = MQTTStreamingProtocol(manager)

        assert protocol.get_protocol_stats() == {"protocol": "MQTTStreamingProtocol"}


class TestStreamingAnalytics:
    def test_stream_analytics_requires_events(self) -> None:
        analytics = StreamingAnalytics(StreamManager(enable_persistence=False))

        assert "message" in analytics.get_streaming_analytics("s1")
        assert "message" in analytics.get_system_streaming_analytics()

    def test_record_and_analyze_events(self) -> None:
        manager = StreamManager(enable_persistence=False)
        analytics = StreamingAnalytics(manager)

        analytics.record_streaming_event(
            "s1", "data_streamed", {"protocol": "websocket"}
        )
        analytics.record_streaming_event("s1", "stream_started")
        analytics.record_streaming_event("s2", "stream_started")

        report = analytics.get_streaming_analytics("s1")
        assert report["total_events"] == 2
        assert report["event_types"] == {"data_streamed": 1, "stream_started": 1}
        assert report["time_range"]["start"] <= report["time_range"]["end"]

        system_report = analytics.get_system_streaming_analytics()
        assert system_report["total_streaming_events"] == 3
        assert system_report["active_streams"] == 2
        assert system_report["most_active_streams"][0][0] == "s1"

    def test_history_is_capped(self) -> None:
        analytics = StreamingAnalytics(StreamManager(enable_persistence=False))
        for index in range(5001):
            analytics.record_streaming_event("s1", "tick", {"index": index})

        assert len(analytics.streaming_history) == 5000


class TestStreamingOrchestrator:
    def test_create_geospatial_stream_registers(self) -> None:
        orchestrator = StreamingOrchestrator(StreamManager(enable_persistence=False))

        stream = orchestrator.create_geospatial_stream("geo-1", {"format": "sensor"})
        assert orchestrator.geospatial_streams["geo-1"] is stream

    def test_stream_geospatial_data_records_event(self) -> None:
        manager = StreamManager(enable_persistence=False)
        orchestrator = StreamingOrchestrator(manager)
        orchestrator.create_geospatial_stream("geo-1", {})
        location = GeospatialPoint(longitude=-122.4, latitude=37.8)

        assert orchestrator.stream_geospatial_data("geo-1", location, 42.0) is True
        assert (
            orchestrator.analytics.streaming_history[0]["event_type"] == "data_streamed"
        )

        events = orchestrator.analytics.get_streaming_analytics("geo-1")
        assert events["total_events"] == 1

    def test_stream_geospatial_data_unknown_protocol_still_succeeds(self) -> None:
        orchestrator = StreamingOrchestrator(StreamManager(enable_persistence=False))
        location = GeospatialPoint(longitude=-122.4, latitude=37.8)

        assert (
            orchestrator.stream_geospatial_data(
                "unknown", location, 1.0, protocol="bogus"
            )
            is True
        )
        assert orchestrator.analytics.streaming_history == []

    def test_get_streaming_insights_shape(self) -> None:
        manager = StreamManager(enable_persistence=False)
        orchestrator = StreamingOrchestrator(manager)
        location = GeospatialPoint(longitude=-122.4, latitude=37.8)
        orchestrator.stream_geospatial_data("geo-1", location, 1.0)

        insights = orchestrator.get_streaming_insights()
        assert insights["geospatial_stream_count"] == 0
        assert set(insights["protocol_stats"]) == {"websocket", "mqtt", "sse"}
        assert "stream_manager_stats" in insights
        assert "analytics" in insights
