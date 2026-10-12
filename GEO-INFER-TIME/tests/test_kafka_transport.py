"""Kafka lifecycle and delivery invariants using controlled client failures.

The separate kafka_service_check.py executes the actual broker round trip.
"""

import asyncio
import json
from collections import deque
from datetime import timedelta, UTC
from types import SimpleNamespace

import pytest

from geo_infer_time import KafkaIngestAdapter, StreamProcessor


@pytest.fixture
def kafka_client(monkeypatch):
    import aiokafka

    state = SimpleNamespace(
        records=deque(), calls=[], start_error=None, commit_error=None, waiting=None
    )

    class Consumer:
        def __init__(self, *topics, **options):
            state.calls.append(("create", topics, options))

        async def start(self):
            state.calls.append(("start",))
            if state.start_error:
                raise state.start_error

        async def stop(self):
            state.calls.append(("stop",))

        async def getone(self):
            state.calls.append(("receive",))
            if state.waiting is not None:
                state.waiting.set()
                await asyncio.Event().wait()
            item = state.records.popleft()
            if isinstance(item, Exception):
                raise item
            return item

        async def commit(self, offsets):
            state.calls.append(("commit", offsets))
            if state.commit_error:
                raise state.commit_error

    monkeypatch.setattr(aiokafka, "AIOKafkaConsumer", Consumer)
    return state


def message(value=7, offset=4, partition=2):
    return SimpleNamespace(
        topic="events",
        partition=partition,
        offset=offset,
        value=json.dumps(
            {"timestamp": "1970-01-01T00:00:00Z", "value": value}
        ).encode(),
    )


def test_kafka_commit_follows_processing_and_targets_one_partition(kafka_client):
    from aiokafka import TopicPartition

    kafka_client.records.extend([message(), message(8, 5)])

    def aggregate(values):
        kafka_client.calls.append(("process", list(values)))
        return sum(values)

    processor = StreamProcessor(timedelta(seconds=10), aggregation_func=aggregate)
    adapter = KafkaIngestAdapter({"topic": "events"})
    assert (
        asyncio.run(
            processor.ingest_adapter_stream(
                adapter, max_messages=2, auto_process_windows=True
            )
        )
        == 2
    )
    assert [call[0] for call in kafka_client.calls] == [
        "create",
        "start",
        "receive",
        "process",
        "commit",
        "receive",
        "process",
        "commit",
        "stop",
    ]
    assert kafka_client.calls[4][1] == {TopicPartition("events", 2): 5}
    assert kafka_client.calls[7][1] == {TopicPartition("events", 2): 6}
    assert kafka_client.calls[0][2]["enable_auto_commit"] is False
    assert processor.buffer[0]["metadata"]["_kafka"] == {
        "topic": "events",
        "partition": 2,
        "offset": 4,
    }
    assert not adapter.is_connected


def test_kafka_parse_failure_does_not_commit(kafka_client):
    kafka_client.records.append(message("NaN"))
    adapter = KafkaIngestAdapter({"topic": "events"})
    with pytest.raises(ValueError, match="finite"):
        asyncio.run(
            StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(adapter)
        )
    assert not any(call[0] == "commit" for call in kafka_client.calls)
    assert kafka_client.calls[-1] == ("stop",)


def test_kafka_processing_failure_does_not_commit(kafka_client):
    kafka_client.records.append(message())

    def fail(values):
        raise RuntimeError("aggregation failed")

    with pytest.raises(RuntimeError, match="aggregation failed"):
        asyncio.run(
            StreamProcessor(
                timedelta(seconds=10), aggregation_func=fail
            ).ingest_adapter_stream(KafkaIngestAdapter(), auto_process_windows=True)
        )
    assert not any(call[0] == "commit" for call in kafka_client.calls)
    assert kafka_client.calls[-1] == ("stop",)


def test_kafka_commit_failure_propagates_for_replay(kafka_client):
    from aiokafka.errors import CommitFailedError

    kafka_client.records.append(message())
    kafka_client.commit_error = CommitFailedError("rebalance")
    adapter = KafkaIngestAdapter()
    with pytest.raises(CommitFailedError):
        asyncio.run(
            StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(adapter)
        )
    assert kafka_client.calls[-1] == ("stop",)
    assert not adapter.is_connected


def test_kafka_requires_ack_before_next_delivery(kafka_client):
    kafka_client.records.extend([message(), message(8)])

    async def run():
        adapter = KafkaIngestAdapter()
        records = adapter.stream_data()
        record = await anext(records)
        with pytest.raises(ValueError, match="outstanding"):
            await adapter.acknowledge(dict(record))
        with pytest.raises(RuntimeError, match="Acknowledge"):
            await anext(records)
        assert not adapter.is_connected

    asyncio.run(run())
    assert not any(call[0] == "commit" for call in kafka_client.calls)


def test_kafka_start_failure_has_finite_retries_and_cleanup(kafka_client):
    from aiokafka.errors import KafkaConnectionError

    kafka_client.start_error = KafkaConnectionError("unreachable")
    adapter = KafkaIngestAdapter({"max_retries": 1, "reconnect_interval": 0})
    with pytest.raises(ConnectionError, match="budget"):
        asyncio.run(
            StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(adapter)
        )
    assert sum(call[0] == "start" for call in kafka_client.calls) == 2
    assert sum(call[0] == "stop" for call in kafka_client.calls) == 2
    assert not adapter.is_connected


def test_kafka_cancellation_stops_consumer(kafka_client):
    async def run():
        kafka_client.waiting = asyncio.Event()
        adapter = KafkaIngestAdapter()
        task = asyncio.create_task(
            StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(adapter)
        )
        await asyncio.wait_for(kafka_client.waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not adapter.is_connected

    asyncio.run(run())
    assert kafka_client.calls[-1] == ("stop",)
    assert not any(call[0] == "commit" for call in kafka_client.calls)


@pytest.mark.parametrize(
    "options",
    [
        {"enable_auto_commit": True},
        {"value_deserializer": json.loads},
        {"max_poll_records": 100},
    ],
)
def test_delivery_controls_cannot_be_overridden(options):
    with pytest.raises(ValueError, match="controls"):
        KafkaIngestAdapter({"consumer_options": options})


@pytest.mark.parametrize(
    "broker_time,seconds", [(0, 0), (1000, 1), (1704067200000, 1704067200)]
)
def test_broker_timestamp_is_explicit_milliseconds_and_normalized(
    kafka_client, broker_time, seconds
):
    from datetime import datetime

    msg = message()
    msg.value = b'{"value":"7"}'
    msg.timestamp = broker_time
    kafka_client.records.append(msg)
    adapter = KafkaIngestAdapter({"topic": "events"})

    async def run():
        records = adapter.stream_data(max_messages=1)
        try:
            record = await anext(records)
            assert (
                record["timestamp"] == datetime.fromtimestamp(seconds, UTC).isoformat()
            )
            assert record["value"] == 7.0
            assert record["topic"] == "events"
            assert not any(call[0] == "commit" for call in kafka_client.calls)
            await adapter.acknowledge(record)
        finally:
            await records.aclose()

    asyncio.run(run())
    assert kafka_client.calls[-1] == ("stop",)


@pytest.mark.parametrize("alias", ["timestamp", "time", "datetime"])
def test_payload_timestamp_precedes_broker_timestamp(kafka_client, alias):
    msg = message()
    msg.value = json.dumps({alias: "1970-01-01T00:00:00Z", "value": 7}).encode()
    msg.timestamp = 1704067200000
    kafka_client.records.append(msg)
    processor = StreamProcessor(timedelta(seconds=10))
    adapter = KafkaIngestAdapter({"topic": "events"})
    assert (
        asyncio.run(
            processor.ingest_kafka_stream(
                topic="ignored",
                bootstrap_servers=["ignored"],
                adapter=adapter,
                max_messages=1,
            )
        )
        == 1
    )
    assert processor.buffer[0]["timestamp"].isoformat() == "1970-01-01T00:00:00+00:00"
    assert kafka_client.calls[0][1] == ("events",)
    assert not adapter.is_connected


@pytest.mark.parametrize("broker_time", [None, -1, True, float("nan")])
def test_missing_payload_and_unavailable_broker_timestamp_fail_without_commit(
    kafka_client, broker_time
):
    msg = message()
    msg.value = b'{"value":7}'
    msg.timestamp = broker_time
    kafka_client.records.append(msg)
    adapter = KafkaIngestAdapter()
    with pytest.raises(ValueError, match="timestamp"):
        asyncio.run(
            StreamProcessor(timedelta(seconds=10)).ingest_adapter_stream(adapter)
        )
    assert not any(call[0] == "commit" for call in kafka_client.calls)
    assert not adapter.is_connected
