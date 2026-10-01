"""Async transports and explicit replay for validated temporal records.

Network dependencies load only on connection. Kafka delivery requires an explicit
acknowledgement after processing; network errors never produce synthetic records.

Note for other GEO-INFER modules: these transport adapters (replay, WebSocket,
Kafka) are self-contained and substantial. If your module needs validated
temporal record ingestion, import and reuse them from here
(``geo_infer_time.core.stream_ingest``) rather than reimplementing the
delivery, watermarking, and replay contracts.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import json
import math
from datetime import datetime, UTC
from typing import Any
from collections.abc import AsyncIterator, Iterable
from urllib.parse import urlsplit


def normalize_timestamp(value: datetime) -> datetime:
    """Normalize a datetime to UTC, interpreting legacy naive values as UTC."""
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value != value:  # Reject pandas NaT, a datetime subclass.
        raise ValueError("timestamp must not be NaT")
    if value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _duration(value: Any, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        )
    return float(value)


def _decode(record: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, (str, bytes)):
        try:
            record = json.loads(record)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Failed to decode JSON stream record") from exc
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary or a JSON object")
    return dict(record)


class StreamIngestAdapter(ABC):
    """Transport interface with record parsing and explicit acknowledgement."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        self.config = dict(config or {})
        if "simulated_records" in self.config:
            raise TypeError("Use ReplayIngestAdapter(records) for offline input")
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Open a source; subclasses must establish their actual connection."""
        raise NotImplementedError("abstract transport contract")

    async def disconnect(self) -> None:
        """Release source resources."""
        self.is_connected = False

    @abstractmethod
    async def stream_data(
        self, max_messages: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield records; concrete adapters implement their transport."""
        raise NotImplementedError("abstract transport contract")
        yield  # pragma: no cover

    async def acknowledge(self, record: dict[str, Any]) -> None:
        """Confirm successful processing (no-op for sources without offsets)."""

    def normalize_record(self, record: str | bytes | dict[str, Any]) -> dict[str, Any]:
        """Validate a record and return canonical UTC ISO time and numeric value."""
        timestamp, value, metadata = self.parse_record(record)
        return {**metadata, "timestamp": timestamp.isoformat(), "value": value}

    def parse_record(
        self, record: str | bytes | dict[str, Any]
    ) -> tuple[datetime, float, dict[str, Any]]:
        """Parse UTC event time and a finite value, retaining other metadata.

        Naive timestamps are interpreted as UTC. Numeric timestamps are seconds,
        or milliseconds when their absolute value exceeds 1e11. Event time must
        be supplied explicitly; wall-clock time is never substituted.
        """
        data = _decode(record)
        raw_ts = next(
            (
                data[key]
                for key in ("timestamp", "time", "datetime")
                if data.get(key) is not None
            ),
            None,
        )
        if raw_ts is None:
            raise ValueError("Stream record missing timestamp")
        if isinstance(raw_ts, datetime):
            timestamp = raw_ts
        elif isinstance(raw_ts, bool):
            raise TypeError("timestamp must not be a boolean")
        elif isinstance(raw_ts, (int, float)):
            if not math.isfinite(raw_ts):
                raise ValueError("timestamp must be finite")
            timestamp = datetime.fromtimestamp(
                raw_ts / 1000 if abs(raw_ts) > 1e11 else raw_ts, tz=UTC
            )
        elif isinstance(raw_ts, str):
            timestamp = datetime.fromisoformat(raw_ts)
        else:
            raise TypeError("Unsupported timestamp type")
        timestamp = normalize_timestamp(timestamp)
        value = data.get("value")
        if value is None and isinstance(data.get("data"), dict):
            value = next(
                (
                    data["data"][key]
                    for key in ("value", "measurement", "temperature", "reading", "val")
                    if key in data["data"]
                ),
                None,
            )
        if value is None:
            raise ValueError("Stream record missing 'value' field")
        if isinstance(value, bool):
            raise TypeError("value must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("value must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in ("value", "timestamp", "time", "datetime")
        }
        return timestamp, value, metadata


class ReplayIngestAdapter(StreamIngestAdapter):
    """Explicit finite replay source; iterables are consumed lazily, without copying.

    A list can be replayed on each call. A one-shot iterator remains one-shot.
    The caller owns the iterable's storage and lifetime.
    """

    def __init__(
        self, records: Iterable[dict[str, Any]], config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(config)
        self.records = records

    async def connect(self) -> bool:
        self.is_connected = True
        return True

    async def stream_data(
        self, max_messages: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        if max_messages is not None:
            _integer(max_messages, "max_messages")
        try:
            if max_messages == 0:
                return
            await self.connect()
            for count, record in enumerate(self.records, 1):
                yield _decode(record)
                if max_messages is not None and count >= max_messages:
                    return
                await asyncio.sleep(0)
        finally:
            await self.disconnect()


class _NetworkAdapter(StreamIngestAdapter):
    """Common finite timeout/retry budget for a single streaming invocation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.connect_timeout = _duration(
            self.config.get("connect_timeout", 10), "connect_timeout"
        )
        self.receive_timeout = _duration(
            self.config.get("receive_timeout", 30), "receive_timeout"
        )
        self.close_timeout = _duration(
            self.config.get("close_timeout", 5), "close_timeout"
        )
        self.reconnect_interval = _duration(
            self.config.get("reconnect_interval", 1), "reconnect_interval", True
        )
        self.max_retries = _integer(self.config.get("max_retries", 3), "max_retries")
        self.max_message_size = _integer(
            self.config.get("max_message_size", 1048576), "max_message_size", 1
        )

    async def _retry(self, attempt: int) -> None:
        await self.disconnect()
        if attempt >= self.max_retries:
            raise ConnectionError("Stream transport retry budget exhausted")
        await asyncio.sleep(self.reconnect_interval)


class WebSocketIngestAdapter(_NetworkAdapter):
    """Real WebSocket transport with bounded frame queue and reconnect budget."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.url = self.config.get("url", "ws://localhost:8765")
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("ws", "wss")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "url must be a ws/wss endpoint without embedded credentials"
            )
        self.max_queue = _integer(self.config.get("max_queue", 16), "max_queue", 1)
        self._connection: Any = None

    async def connect(self) -> bool:
        if self.is_connected:
            return True
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "websockets not installed or incompatible; add geo-infer-time[streaming]"
            ) from exc

        self._connection = await connect(
            self.url,
            open_timeout=self.connect_timeout,
            close_timeout=self.close_timeout,
            max_queue=self.max_queue,
            max_size=self.max_message_size,
        )
        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        connection, self._connection = self._connection, None
        self.is_connected = False
        if connection is not None:
            await asyncio.wait_for(connection.close(), self.close_timeout)

    async def stream_data(
        self, max_messages: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
        except ImportError as exc:
            raise RuntimeError(
                "websockets not installed or incompatible; add geo-infer-time[streaming]"
            ) from exc

        if max_messages is not None:
            _integer(max_messages, "max_messages")
        attempts = count = 0
        try:
            while max_messages is None or count < max_messages:
                try:
                    await self.connect()
                    raw = await asyncio.wait_for(
                        self._connection.recv(), self.receive_timeout
                    )
                except ConnectionClosedOK:
                    return
                except (OSError, TimeoutError, ConnectionClosedError):
                    await self._retry(attempts)
                    attempts += 1
                    continue
                yield self.normalize_record(raw)
                count += 1
        finally:
            await self.disconnect()


class KafkaIngestAdapter(_NetworkAdapter):
    """Kafka consumer with one outstanding record and manual partition commits.

    Delivery is at least once relative to successful acknowledgement. A failed
    commit or crash can redeliver processed records; downstream durable effects
    must be idempotent. Automatic commits cannot be enabled through options.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        servers = self.config.get("bootstrap_servers", ["localhost:9092"])
        self.bootstrap_servers = (
            [servers] if isinstance(servers, str) else list(servers)
        )
        self.group_id = self.config.get("group_id", "geo_infer_time_group")
        self.topic = self.config.get("topic", "geo_infer_temporal_events")
        if not self.bootstrap_servers or not all(
            isinstance(s, str) and s for s in self.bootstrap_servers
        ):
            raise ValueError("bootstrap_servers must contain server addresses")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(self.topic, str) or not self.topic.strip():
            raise ValueError("topic must be a non-empty string")
        self.consumer_options = dict(self.config.get("consumer_options", {}))
        reserved = {
            "enable_auto_commit",
            "group_id",
            "bootstrap_servers",
            "value_deserializer",
            "max_poll_records",
            "fetch_max_bytes",
            "max_partition_fetch_bytes",
        }
        if reserved.intersection(self.consumer_options):
            raise ValueError(
                "consumer_options cannot override delivery or buffering controls"
            )
        self._consumer: Any = None
        self._pending: tuple[dict[str, Any], Any] | None = None

    async def connect(self) -> bool:
        if self.is_connected:
            return True
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError as exc:
            raise RuntimeError(
                "aiokafka not installed; add geo-infer-time[streaming]"
            ) from exc

        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            max_poll_records=1,
            fetch_max_bytes=self.max_message_size,
            max_partition_fetch_bytes=self.max_message_size,
            **self.consumer_options,
        )
        try:
            await asyncio.wait_for(self._consumer.start(), self.connect_timeout)
        except BaseException:
            await self.disconnect()
            raise
        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        consumer, self._consumer = self._consumer, None
        self.is_connected = False
        self._pending = None
        if consumer is not None:
            await asyncio.wait_for(consumer.stop(), self.close_timeout)

    async def acknowledge(self, record: dict[str, Any]) -> None:
        from aiokafka import TopicPartition

        if (
            self._pending is None
            or self._pending[0] is not record
            or not self.is_connected
        ):
            raise ValueError("Can only acknowledge the outstanding Kafka record")
        message = self._pending[1]
        await asyncio.wait_for(
            self._consumer.commit(
                {TopicPartition(message.topic, message.partition): message.offset + 1}
            ),
            self.receive_timeout,
        )
        self._pending = None

    async def stream_data(
        self, max_messages: int | None = None, topic: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            from aiokafka.errors import KafkaError
        except ImportError as exc:
            raise RuntimeError(
                "aiokafka not installed; add geo-infer-time[streaming]"
            ) from exc

        if topic is not None and topic != self.topic:
            raise ValueError("topic must match the adapter's configured topic")
        if max_messages is not None:
            _integer(max_messages, "max_messages")
        attempts = count = 0
        try:
            while max_messages is None or count < max_messages:
                if self._pending is not None:
                    raise RuntimeError(
                        "Acknowledge the outstanding Kafka record "
                        "before requesting another"
                    )
                try:
                    await self.connect()
                    message = await asyncio.wait_for(
                        self._consumer.getone(), self.receive_timeout
                    )
                except (OSError, TimeoutError, KafkaError):
                    await self._retry(attempts)
                    attempts += 1
                    continue
                if message.value is None or len(message.value) > self.max_message_size:
                    raise ValueError(
                        "Kafka record is empty or exceeds max_message_size"
                    )
                record = _decode(message.value)
                if not any(
                    record.get(key) is not None
                    for key in ("timestamp", "time", "datetime")
                ):
                    broker_time = getattr(message, "timestamp", None)
                    if (
                        isinstance(broker_time, (int, float))
                        and not isinstance(broker_time, bool)
                        and math.isfinite(broker_time)
                        and broker_time >= 0
                    ):
                        # Kafka timestamps are always milliseconds, including epoch zero.
                        record["timestamp"] = datetime.fromtimestamp(
                            broker_time / 1000, tz=UTC
                        )
                record.setdefault("topic", message.topic)
                record = self.normalize_record(record)
                record["_kafka"] = {
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                }
                self._pending = (record, message)
                yield record
                count += 1
        finally:
            await self.disconnect()
