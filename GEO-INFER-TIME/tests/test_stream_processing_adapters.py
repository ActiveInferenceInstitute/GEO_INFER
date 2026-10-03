"""
Tests for TIME-01: WebSocket/Kafka stream ingest adapters, bounded
watermarking, session window duration, and automated sliding-window
anomaly alert handlers for GEO-INFER-TIME's StreamProcessor.
"""

import pytest
from datetime import datetime, timedelta, UTC

from geo_infer_time.core.stream_processing import (
    StreamProcessor,
    StreamIngestAdapter,
    ReplayIngestAdapter,
    WebSocketIngestAdapter,
    KafkaIngestAdapter,
)


def _ts(base, seconds_offset):
    """Quick helper to create a timestamp offset from a base."""
    return base + timedelta(seconds=seconds_offset)


BASE_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


# ===================================================================
# StreamIngestAdapter / parse_record
# ===================================================================


class TestStreamIngestAdapter:
    def test_config_must_be_dict(self):
        with pytest.raises(TypeError):
            ReplayIngestAdapter([], config="bad")

    def test_config_defaults_to_empty_dict(self):
        adapter = ReplayIngestAdapter([])
        assert adapter.config == {}
        assert adapter.is_connected is False

    def test_parse_iso_datetime_record(self):
        adapter = ReplayIngestAdapter([])
        record = {"timestamp": "2024-01-01T00:00:00Z", "value": 42.5, "sensor": "a"}
        ts, value, meta = adapter.parse_record(record)
        assert ts == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert value == 42.5
        assert meta == {"sensor": "a"}

    def test_parse_json_string_record(self):
        adapter = ReplayIngestAdapter([])
        ts, value, meta = adapter.parse_record(
            '{"timestamp":"2024-01-01T00:00:00Z","value":7}'
        )
        assert value == 7.0
        assert meta == {}

    def test_parse_bytes_record(self):
        adapter = ReplayIngestAdapter([])
        ts, value, meta = adapter.parse_record(
            b'{"timestamp":"2024-01-01T00:00:00Z","value":3.0}'
        )
        assert value == 3.0

    def test_parse_nested_data_value(self):
        adapter = ReplayIngestAdapter([])
        record = {"timestamp": "2024-01-01T00:00:00Z", "data": {"measurement": 11.0}}
        ts, value, meta = adapter.parse_record(record)
        assert value == 11.0

    def test_parse_invalid_json_raises(self):
        adapter = ReplayIngestAdapter([])
        with pytest.raises(ValueError):
            adapter.parse_record("{not-json")

    def test_parse_missing_value_closes_base(self):
        adapter = ReplayIngestAdapter([])
        with pytest.raises(ValueError):
            adapter.parse_record({"timestamp": "2024-01-01T00:00:00Z"})

    def test_parse_invalid_type_raises(self):
        adapter = ReplayIngestAdapter([])
        with pytest.raises(TypeError):
            adapter.parse_record(123)

    def test_base_transport_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            StreamIngestAdapter()


def asyncio_run(coro):
    """Run a coroutine to completion synchronously."""
    import asyncio

    return asyncio.run(coro)


# ===================================================================
# WebSocketIngestAdapter
# ===================================================================


class TestNetworkConfiguration:
    def test_websocket_defaults(self):
        adapter = WebSocketIngestAdapter()
        assert adapter.url == "ws://localhost:8765"
        assert not adapter.is_connected

    def test_kafka_defaults(self):
        adapter = KafkaIngestAdapter()
        assert adapter.bootstrap_servers == ["localhost:9092"]
        assert adapter.group_id == "geo_infer_time_group"
        assert adapter.topic == "geo_infer_temporal_events"

    def test_bootstrap_servers_normalization(self):
        assert KafkaIngestAdapter(
            {"bootstrap_servers": "kafka:9092"}
        ).bootstrap_servers == ["kafka:9092"]

    def test_implicit_simulation_rejected(self):
        with pytest.raises(TypeError, match="ReplayIngestAdapter"):
            WebSocketIngestAdapter({"simulated_records": []})


class TestReplayIngest:
    def test_replay_respects_limit(self):
        records = [{"timestamp": index, "value": index} for index in range(3)]

        async def collect():
            return [
                record
                async for record in ReplayIngestAdapter(records).stream_data(
                    max_messages=2
                )
            ]

        assert asyncio_run(collect()) == records[:2]

    def test_ingest_adapter_stream_counts_points(self):
        processor = StreamProcessor(window_size=timedelta(minutes=1))
        records = [{"timestamp": index, "value": index} for index in range(3)]
        assert (
            asyncio_run(
                processor.ingest_adapter_stream(
                    ReplayIngestAdapter(records, {"timestamp_unit": "s"})
                )
            )
            == 3
        )
        assert processor.get_stats()["total_points"] == 3

    def test_ingest_adapter_stream_auto_process_windows(self):
        processor = StreamProcessor(window_size=timedelta(minutes=1))
        records = [{"timestamp": index, "value": index} for index in range(2)]
        asyncio_run(
            processor.ingest_adapter_stream(
                ReplayIngestAdapter(records, {"timestamp_unit": "s"}),
                auto_process_windows=True,
            )
        )
        assert processor.get_stats()["total_windows"] == 2

    def test_ingest_adapter_stream_rejects_non_adapter(self):
        processor = StreamProcessor(window_size=timedelta(minutes=1))
        with pytest.raises(TypeError):
            asyncio_run(processor.ingest_adapter_stream("not-an-adapter"))


# ===================================================================
# Bounded watermarking
# ===================================================================


class TestBoundedWatermarking:
    def test_watermark_advances_with_delay(self):
        processor = StreamProcessor(
            window_size=timedelta(minutes=5), watermark_delay=timedelta(seconds=10)
        )
        processor.add_data_point(_ts(BASE_TIME, 0), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 30), 2.0)
        # max_timestamp = t+30, watermark = t+30 - 10s = t+20
        assert processor.get_watermark() == _ts(BASE_TIME, 20)

    def test_late_data_beyond_watermark_delay(self):
        processor = StreamProcessor(
            window_size=timedelta(minutes=5), watermark_delay=timedelta(seconds=10)
        )
        processor.add_data_point(_ts(BASE_TIME, 100), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 5), 2.0)
        # After t+100, watermark = t+90. t+5 < t+90 => late.
        assert processor.get_watermark() == _ts(BASE_TIME, 90)
        assert [p["value"] for p in processor.get_late_data()] == [2.0]

    def test_within_delay_is_not_late(self):
        processor = StreamProcessor(
            window_size=timedelta(minutes=5), watermark_delay=timedelta(seconds=10)
        )
        processor.add_data_point(_ts(BASE_TIME, 100), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 95), 2.0)
        # watermark = t+90, so t+95 >= t+90 is not late.
        assert processor.get_late_data() == []
        assert processor.get_stats()["late_arrivals"] == 0

    def test_zero_delay_bounds_watermark_to_max(self):
        processor = StreamProcessor(
            window_size=timedelta(minutes=5), watermark_delay=timedelta(0)
        )
        processor.add_data_point(_ts(BASE_TIME, 100), 1.0)
        assert processor.get_watermark() == _ts(BASE_TIME, 100)

    def test_watermark_delay_validation(self):
        with pytest.raises(TypeError):
            StreamProcessor(window_size=timedelta(minutes=5), watermark_delay="10")
        with pytest.raises(ValueError):
            StreamProcessor(
                window_size=timedelta(minutes=5), watermark_delay=timedelta(seconds=-1)
            )

    def test_buffer_retention_uses_max_timestamp(self):
        processor = StreamProcessor(
            window_size=timedelta(seconds=10), watermark_delay=timedelta(seconds=3)
        )
        processor.add_data_point(_ts(BASE_TIME, 0), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 100), 2.0)
        # Old point at t=0 evicted (max timestamp t=100 - 10s = t+90).
        assert len(processor.buffer) == 1


# ===================================================================
# Session windowing
# ===================================================================


class TestSessionWindowing:
    def test_session_duration_seconds_present(self):
        processor = StreamProcessor(window_size=timedelta(minutes=5))
        processor.add_data_point(_ts(BASE_TIME, 0), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 5), 2.0)
        processor.add_data_point(_ts(BASE_TIME, 100), 3.0)  # new session

        sessions = processor.process_session_windows(timedelta(seconds=15))
        assert [s["count"] for s in sessions] == [2, 1]
        assert sessions[0]["session_duration_seconds"] == 5.0
        assert sessions[1]["session_duration_seconds"] == 0.0

    def test_single_session_duration(self):
        processor = StreamProcessor(window_size=timedelta(minutes=1))
        processor.add_data_point(_ts(BASE_TIME, 0), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 5), 2.0)
        processor.add_data_point(_ts(BASE_TIME, 10), 3.0)

        sessions = processor.process_session_windows(timedelta(minutes=1))
        assert len(sessions) == 1
        assert sessions[0]["session_duration_seconds"] == 10.0


# ===================================================================
# Automated sliding-window anomaly alert handlers
# ===================================================================


class TestSlidingWindowAnomalyAlerts:
    def test_alert_emitted_for_spike(self):
        processor = StreamProcessor(window_size=timedelta(seconds=30))
        for i in range(40):
            processor.add_data_point(_ts(BASE_TIME, i), 1.0 if i != 25 else 100.0)

        handled = []
        processor.register_anomaly_alert_handler(handled.append)

        alerts = processor.process_sliding_window_anomaly_alerts(z_threshold=3.0)
        assert len(alerts) >= 1
        assert alerts[0]["type"] == "anomaly_alert"
        assert "window_start" in alerts[0]
        assert "window_end" in alerts[0]
        assert alerts[0]["method"] == "sliding_window_zscore"
        # The alert handler was invoked with the same alert payload.
        assert handled == alerts

    def test_alert_handler_stats_incremented(self):
        processor = StreamProcessor(window_size=timedelta(seconds=30))
        for i in range(40):
            processor.add_data_point(_ts(BASE_TIME, i), 1.0 if i != 25 else 100.0)

        processor.process_sliding_window_anomaly_alerts(z_threshold=3.0)
        stats = processor.get_stats()
        assert stats["anomaly_alerts"] >= 1
        assert stats["events_detected"] >= stats["anomaly_alerts"]

    def test_no_alerts_for_flat_series(self):
        processor = StreamProcessor(window_size=timedelta(seconds=30))
        for i in range(40):
            processor.add_data_point(_ts(BASE_TIME, i), 1.0)

        alerts = processor.process_sliding_window_anomaly_alerts(z_threshold=3.0)
        assert alerts == []

    def test_insufficient_points_returns_empty(self):
        processor = StreamProcessor(window_size=timedelta(seconds=30))
        processor.add_data_point(_ts(BASE_TIME, 0), 1.0)
        processor.add_data_point(_ts(BASE_TIME, 1), 2.0)

        alerts = processor.process_sliding_window_anomaly_alerts(min_window_points=3)
        assert alerts == []

    def test_alert_validation_of_threshold(self):
        processor = StreamProcessor(window_size=timedelta(seconds=30))
        processor.add_data_point(_ts(BASE_TIME, 0), 1.0)
        with pytest.raises(ValueError):
            processor.process_sliding_window_anomaly_alerts(z_threshold=0)
        with pytest.raises(TypeError):
            processor.process_sliding_window_anomaly_alerts(z_threshold="high")
        with pytest.raises(ValueError):
            processor.process_sliding_window_anomaly_alerts(min_window_points=1)
        with pytest.raises(TypeError):
            processor.process_sliding_window_anomaly_alerts(min_window_points="three")

    def test_multiple_handlers_all_invoked(self):
        processor = StreamProcessor(window_size=timedelta(seconds=30))
        for i in range(40):
            processor.add_data_point(_ts(BASE_TIME, i), 1.0 if i != 25 else 100.0)

        first = []
        second = []
        processor.register_anomaly_alert_handler(first.append)
        processor.register_anomaly_alert_handler(second.append)

        processor.process_sliding_window_anomaly_alerts(z_threshold=3.0)
        assert len(first) == len(second)
        assert len(first) >= 1


def test_normalize_record_preserves_metadata_and_canonicalizes_aliases():
    record = {"time": "2024-01-01T02:00:00+02:00", "value": "3", "sensor": "a"}
    normalized = ReplayIngestAdapter([]).normalize_record(record)
    assert normalized == {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "value": 3.0,
        "sensor": "a",
    }
    assert "timestamp" not in record


def test_real_websocket_normalizes_records_and_injected_adapter_owns_url():
    import json
    from websockets.asyncio.server import serve

    async def run():
        async def handler(connection):
            for index in range(3):
                await connection.send(
                    json.dumps(
                        {"time": index, "value": str(index), "sensor": f"ws-{index}"}
                    )
                )

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter(
                {"url": f"ws://127.0.0.1:{port}", "timestamp_unit": "s"}
            )
            records = [record async for record in adapter.stream_data()]
            assert [record["value"] for record in records] == [0.0, 1.0, 2.0]
            assert [record["sensor"] for record in records] == ["ws-0", "ws-1", "ws-2"]
            assert records[0]["timestamp"] == "1970-01-01T00:00:00+00:00"
            assert not adapter.is_connected
            processor = StreamProcessor(timedelta(seconds=10))
            assert (
                await processor.ingest_websocket_stream(url="ignored", adapter=adapter)
                == 3
            )
            assert not adapter.is_connected

    asyncio_run(run())


@pytest.mark.parametrize(
    "adapter_type,dependency",
    [(WebSocketIngestAdapter, "websockets"), (KafkaIngestAdapter, "aiokafka")],
)
@pytest.mark.parametrize("entrypoint", ["connect", "stream_data"])
def test_missing_optional_transport_dependency_is_actionable(
    monkeypatch, adapter_type, dependency, entrypoint
):
    import builtins

    original = builtins.__import__

    def without_dependency(name, *args, **kwargs):
        if name == dependency or name.startswith(dependency + "."):
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_dependency)
    adapter = adapter_type()

    async def run():
        with pytest.raises(RuntimeError, match=dependency + " not installed"):
            if entrypoint == "connect":
                await adapter.connect()
            else:
                await anext(adapter.stream_data())
        assert not adapter.is_connected

    asyncio_run(run())


@pytest.mark.parametrize("adapter_type", [WebSocketIngestAdapter, KafkaIngestAdapter])
def test_network_adapters_reject_simulation_arguments(adapter_type):
    with pytest.raises(TypeError):
        adapter_type(allow_simulated=True)
    with pytest.raises(TypeError):
        adapter_type().stream_data(simulated_records=[{"timestamp": 0, "value": 1}])
