"""
Stream connectors for GEO-INFER-DATA.

This module provides streaming data connectivity for real-time geospatial
data sources: MQTT (via ``aiomqtt``), WebSocket (via ``aiohttp``), and an
explicit Kafka connector whose client library is not a declared
dependency of this package.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional


logger = logging.getLogger(__name__)


class StreamConnector:
    """
    Base class for streaming data connectors.

    This abstract base class defines the interface for connecting to streaming
    data sources including MQTT, Kafka, WebSocket, and other real-time data
    streams.
    """

    async def connect(self) -> bool:
        """
        Establish connection to streaming data source.

        Returns:
            True if connection successful
        """
        raise RuntimeError("Stream connector subclasses must implement connect()")

    async def stream_data(self, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream data from source.

        Args:
            **kwargs: Streaming parameters

        Yields:
            Data records as they become available
        """
        raise RuntimeError("Stream connector subclasses must implement stream_data()")

    async def disconnect(self) -> None:
        """Close streaming connection."""
        logger.debug(
            "%s has no persistent streaming connection to close", type(self).__name__
        )


def _decode_payload(payload: Any) -> Any:
    """Decode an MQTT payload into text or parsed JSON where possible."""
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text
    return payload


class MQTTConnector(StreamConnector):
    """
    MQTT streaming data connector backed by ``aiomqtt``.

    The connector opens a real TCP connection to the broker on ``connect()``
    and subscribes to the requested topic on ``stream_data()``. Every yielded
    record comes from the broker; nothing is fabricated locally.

    Config keys:
        host: Broker hostname (default ``localhost``).
        port: Broker port (default ``1883``).
        client_id: MQTT client identifier (default ``geo_infer_data``).
        username / password: Optional broker credentials.
        qos: Subscription QoS level (default ``0``).
    """

    def __init__(self, config: Dict[str, Any]):
        import aiomqtt

        self._aiomqtt = aiomqtt
        self.config = config
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 1883)
        self.client_id = config.get("client_id", "geo_infer_data")
        self.username = config.get("username")
        self.password = config.get("password")
        self.qos = config.get("qos", 0)
        self._client: Optional[Any] = None

        logger.info(f"Initialized MQTTConnector for {self.host}:{self.port}")

    def _require_client(self) -> Any:
        """Return the connected aiomqtt client or raise."""
        if self._client is None:
            raise RuntimeError("MQTTConnector is not connected; call connect() first")
        return self._client

    async def connect(self) -> bool:
        """
        Connect to the MQTT broker.

        Returns:
            True if connection successful

        Raises:
            aiomqtt.MqttError: If the broker cannot be reached or rejects
                the credentials.
        """
        client = self._aiomqtt.Client(
            hostname=self.host,
            port=self.port,
            identifier=self.client_id,
            username=self.username,
            password=self.password,
        )
        await client.connect()
        self._client = client
        logger.info(f"Connected to MQTT broker at {self.host}:{self.port}")
        return True

    async def stream_data(  # type: ignore[override]
        self, topic: str, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream messages from an MQTT topic.

        Args:
            topic: MQTT topic to subscribe to
            **kwargs: Additional subscription parameters

        Yields:
            One record per broker message with keys ``topic``, ``payload``,
            ``qos``, ``retain``, and ``properties`` (when present).
        """
        client = self._require_client()
        await client.subscribe(topic, qos=self.qos)
        logger.info(f"Subscribed to MQTT topic: {topic}")

        async for message in client.messages:
            record: Dict[str, Any] = {
                "topic": message.topic.value,
                "payload": _decode_payload(message.payload),
                "qos": message.qos,
                "retain": message.retain,
            }
            if getattr(message, "properties", None) is not None:
                record["properties"] = message.properties
            yield record

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
        logger.info("MQTT connection closed")


class KafkaConnector(StreamConnector):
    """
    Apache Kafka streaming connector for the not-installed-client case.

    No Kafka client library (e.g. ``aiokafka``) is a declared dependency of
    this package, so every operation raises ``RuntimeError`` naming the
    missing library rather than returning fabricated records.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.group_id = config.get("group_id", "geo_infer_data")

        logger.info(f"Initialized KafkaConnector for {self.bootstrap_servers}")

    def _unavailable(self, operation: str) -> RuntimeError:
        return RuntimeError(
            f"KafkaConnector does not implement {operation}: no Kafka client "
            "library is a declared dependency of geo-infer-data. Declare "
            "'aiokafka' (or equivalent) and implement the connector to enable "
            "Kafka streaming."
        )

    async def connect(self) -> bool:
        raise self._unavailable("connect()")

    async def stream_data(  # type: ignore[override]
        self, topic: str, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        raise self._unavailable("stream_data()")
        yield  # pragma: no cover - makes this an async generator

    async def disconnect(self) -> None:
        logger.info("Kafka connection closed")


class WebSocketConnector(StreamConnector):
    """
    WebSocket streaming data connector backed by ``aiohttp``.

    The connector opens a real WebSocket connection on ``connect()`` and
    yields every message received from the server. Nothing is fabricated
    locally.

    Config keys:
        url: WebSocket URL (e.g. ``ws://localhost:8765/geo``).
        reconnect_interval: Seconds between reconnect attempts (reserved for
            callers that drive their own reconnection loop).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config.get("url", "ws://localhost:8765")
        self.reconnect_interval = config.get("reconnect_interval", 5)
        self._session: Optional[Any] = None
        self._connection: Optional[Any] = None

        logger.info(f"Initialized WebSocketConnector for {self.url}")

    def _require_connection(self) -> Any:
        """Return the live WebSocket connection or raise."""
        if self._connection is None or self._connection.closed:
            raise RuntimeError(
                "WebSocketConnector is not connected; call connect() first"
            )
        return self._connection

    async def connect(self) -> bool:
        """
        Connect to the WebSocket server.

        Returns:
            True if connection successful

        Raises:
            aiohttp.ClientError: If the server cannot be reached or rejects
                the upgrade request.
        """
        import aiohttp

        self._session = aiohttp.ClientSession()
        try:
            self._connection = await self._session.ws_connect(self.url)
        except BaseException:
            # Covers asyncio.CancelledError (a BaseException, not an
            # Exception) so a cancelled connect() cannot leak the session.
            await self._session.close()
            self._session = None
            raise
        logger.info(f"Connected to WebSocket: {self.url}")
        return True

    async def stream_data(  # type: ignore[override, misc]
        self, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream messages from the WebSocket connection.

        Args:
            **kwargs: Additional streaming parameters

        Yields:
            One record per server message with keys ``type`` (``text`` or
            ``binary``) and ``data``; text payloads that parse as JSON are
            yielded as parsed objects.
        """
        import aiohttp

        connection = self._require_connection()
        async for message in connection:
            if message.type == aiohttp.WSMsgType.TEXT:
                try:
                    data: Any = json.loads(message.data)
                except (ValueError, TypeError):
                    data = message.data
                yield {"type": "text", "data": data}
            elif message.type == aiohttp.WSMsgType.BINARY:
                yield {"type": "binary", "data": _decode_payload(message.data)}
            elif message.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"WebSocket stream error: {connection.exception()}")
            elif message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                break

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server and release the session."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        logger.info("WebSocket connection closed")
