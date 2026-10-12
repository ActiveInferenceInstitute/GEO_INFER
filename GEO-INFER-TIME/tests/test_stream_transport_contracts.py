"""Regressions for actual transports, explicit replay, and event-time ordering."""

import asyncio
import json
from datetime import datetime, timedelta, UTC

import pytest

from geo_infer_time import ReplayIngestAdapter, StreamIngestAdapter, StreamProcessor


def test_epoch_zero_is_preserved():
    timestamp, _, _ = ReplayIngestAdapter([], {"timestamp_unit": "s"}).parse_record(
        {"timestamp": 0, "value": 1}
    )
    assert timestamp == datetime(1970, 1, 1, tzinfo=UTC)


def test_offsets_normalize_to_utc():
    adapter = ReplayIngestAdapter([])
    first, _, _ = adapter.parse_record(
        {"timestamp": "2024-01-01T02:00:00+02:00", "value": 1}
    )
    second, _, _ = adapter.parse_record(
        {"timestamp": "2024-01-01T00:00:00Z", "value": 1}
    )
    assert first == second == datetime(2024, 1, 1, tzinfo=UTC)


def test_naive_timestamps_are_rejected():
    adapter = ReplayIngestAdapter([])
    with pytest.raises(ValueError, match="timezone-aware"):
        adapter.parse_record({"timestamp": "2024-01-01T00:00:00", "value": 1})
    with pytest.raises(ValueError, match="timezone-aware"):
        StreamProcessor(timedelta(seconds=10)).add_data_point(datetime(2024, 1, 1), 1)


def test_missing_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        ReplayIngestAdapter([]).parse_record({"value": 1})


def test_out_of_order_retention_and_bounds():
    processor = StreamProcessor(
        timedelta(seconds=10), watermark_delay=timedelta(seconds=10)
    )
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for seconds in [10, 5, 18]:
        processor.add_data_point(base + timedelta(seconds=seconds), seconds)
    assert [p["value"] for p in processor.buffer] == [10, 18]
    assert (
        processor.process_window()["window_start"]
        == (base + timedelta(seconds=10)).isoformat()
    )


def test_explicit_replay_and_zero_limit():
    from geo_infer_time import ReplayIngestAdapter

    async def run():
        processor = StreamProcessor(timedelta(seconds=10))
        adapter = ReplayIngestAdapter(
            [{"timestamp": 0, "value": 2}], {"timestamp_unit": "s"}
        )
        assert await processor.ingest_adapter_stream(adapter, max_messages=0) == 0
        assert await processor.ingest_adapter_stream(adapter) == 1
        assert not adapter.is_connected

    asyncio.run(run())


def test_actual_local_websocket_and_cleanup():
    from websockets.asyncio.server import serve
    from geo_infer_time import WebSocketIngestAdapter

    async def run():
        async def handler(connection):
            await connection.send(
                json.dumps({"timestamp": "1970-01-01T00:00:00Z", "value": 7})
            )
            await connection.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter(
                {"url": f"ws://127.0.0.1:{port}", "max_retries": 0}
            )
            processor = StreamProcessor(timedelta(seconds=10))
            assert await processor.ingest_adapter_stream(adapter, max_messages=1) == 1
            assert processor.buffer[0]["value"] == 7
            assert not adapter.is_connected

    asyncio.run(run())


@pytest.mark.parametrize("timestamp", [True, float("nan"), float("inf"), "not-a-date"])
def test_invalid_timestamps_rejected(timestamp):
    with pytest.raises((TypeError, ValueError)):
        ReplayIngestAdapter([]).parse_record({"timestamp": timestamp, "value": 1})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "bad"])
def test_nonfinite_values_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        ReplayIngestAdapter([]).parse_record(
            {"timestamp": "1970-01-01T00:00:00Z", "value": value}
        )


def test_buffer_capacity_does_not_partially_accept_record():
    processor = StreamProcessor(timedelta(seconds=10), max_buffer_points=2)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    processor.add_data_point(base, 1)
    processor.add_data_point(base, 2)
    with pytest.raises(BufferError):
        processor.add_data_point(base, 3)
    assert processor.get_stats()["total_points"] == 2
    assert [point["value"] for point in processor.buffer] == [1, 2]
    processor.add_data_point(base + timedelta(seconds=11), 4)
    assert [point["value"] for point in processor.buffer] == [4]


def test_late_buffer_is_bounded_and_watermark_never_regresses():
    processor = StreamProcessor(timedelta(seconds=10), max_buffer_points=1)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    processor.add_data_point(base + timedelta(seconds=5), 1)
    processor.add_data_point(base, 2)
    with pytest.raises(BufferError):
        processor.add_data_point(base, 3)
    assert processor.get_watermark() == base + timedelta(seconds=5)
    assert len(processor.flush_late_data()) == 1
    processor.add_data_point(base, 3)
    assert processor.get_stats()["late_arrivals"] == 2


def test_history_capacity_and_session_gap_equality():
    processor = StreamProcessor(timedelta(seconds=60), max_history_windows=2)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for seconds in (0, 5, 10, 16):
        processor.add_data_point(base + timedelta(seconds=seconds), 1)
        processor.process_window()
    assert len(processor.get_recent_windows()) == 2
    assert processor.get_stats()["total_windows"] == 4
    assert [
        window["count"]
        for window in processor.process_session_windows(timedelta(seconds=5))
    ] == [3, 1]


def test_websocket_failed_handshake_never_claims_connection():
    from websockets.exceptions import InvalidHandshake

    from geo_infer_time import WebSocketIngestAdapter

    async def run():
        async def reject(reader, writer):
            writer.close()
            await writer.wait_closed()

        async with await asyncio.start_server(reject, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter(
                {"url": f"ws://127.0.0.1:{port}", "max_retries": 0}
            )
            with pytest.raises(InvalidHandshake):
                await adapter.connect()
            assert not adapter.is_connected

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload", ["{bad", '{"value": 2}', '{"timestamp":0,"value":"NaN"}']
)
def test_websocket_malformed_record_closes_connection(payload):
    from websockets.asyncio.server import serve
    from geo_infer_time import WebSocketIngestAdapter

    async def run():
        async def handler(connection):
            await connection.send(payload)
            await connection.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter({"url": f"ws://127.0.0.1:{port}"})
            with pytest.raises(ValueError):
                await StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(
                    adapter
                )
            assert not adapter.is_connected

    asyncio.run(run())


def test_websocket_cancellation_releases_socket():
    from websockets.asyncio.server import serve
    from geo_infer_time import WebSocketIngestAdapter

    async def run():
        ready = asyncio.Event()
        closed = asyncio.Event()

        async def handler(connection):
            ready.set()
            await connection.wait_closed()
            closed.set()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter({"url": f"ws://127.0.0.1:{port}"})
            task = asyncio.create_task(
                StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(adapter)
            )
            await asyncio.wait_for(ready.wait(), 2)
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), 2)
            assert not adapter.is_connected

    asyncio.run(run())


def test_websocket_reconnect_and_finite_retry_budget():
    from websockets.asyncio.server import serve
    from geo_infer_time import WebSocketIngestAdapter

    async def run():
        connections = 0

        async def handler(connection):
            nonlocal connections
            connections += 1
            if connections == 1:
                await connection.close(code=1011)
            else:
                await connection.send('{"timestamp":"1970-01-01T00:00:00Z","value":9}')
                await connection.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter(
                {
                    "url": f"ws://127.0.0.1:{port}",
                    "max_retries": 1,
                    "reconnect_interval": 0,
                }
            )
            processor = StreamProcessor(timedelta(seconds=10))
            assert await processor.ingest_adapter_stream(adapter, max_messages=1) == 1
            assert connections == 2
            assert processor.buffer[0]["value"] == 9

    asyncio.run(run())


def test_websocket_idle_timeout_is_bounded():
    from websockets.asyncio.server import serve
    from geo_infer_time import WebSocketIngestAdapter

    async def run():
        connections = 0

        async def handler(connection):
            nonlocal connections
            connections += 1
            await connection.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = WebSocketIngestAdapter(
                {
                    "url": f"ws://127.0.0.1:{port}",
                    "max_retries": 1,
                    "reconnect_interval": 0,
                    "receive_timeout": 0.03,
                }
            )
            with pytest.raises(ConnectionError, match="budget"):
                await asyncio.wait_for(
                    StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(
                        adapter
                    ),
                    3,
                )
            assert connections == 2
            assert not adapter.is_connected

    asyncio.run(run())


def test_processor_limit_and_cleanup_apply_to_custom_adapters():
    class InfiniteAdapter(StreamIngestAdapter):
        closed = False

        async def connect(self):
            self.is_connected = True
            return True

        async def stream_data(self, max_messages=None):
            try:
                while True:
                    yield {"timestamp": "1970-01-01T00:00:00Z", "value": 1}
            finally:
                self.closed = True

    adapter = InfiniteAdapter()
    processor = StreamProcessor(timedelta(seconds=10))
    assert asyncio.run(processor.ingest_adapter_stream(adapter, max_messages=2)) == 2
    assert adapter.closed
    assert len(processor.buffer) == 2


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_processor_rejects_invalid_message_limits(limit):
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(
            StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(
                ReplayIngestAdapter([]), max_messages=limit
            )
        )
