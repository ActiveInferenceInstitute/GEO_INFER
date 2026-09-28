"""
Tests for stream connectors in geo_infer_data.connectors.stream.
"""

import asyncio
import socket

import aiohttp
import pytest
from aiohttp import web

from geo_infer_data.connectors.stream import (
    StreamConnector,
    MQTTConnector,
    KafkaConnector,
    WebSocketConnector,
)


def _collect(gen, n=None):
    """Drain an async generator with asyncio.run."""

    async def _run_all():
        out = []
        async for record in gen:
            out.append(record)
            if n is not None and len(out) >= n:
                break
        return out

    return asyncio.run(_run_all())


# ---------------------------------------------------------------------------
# StreamConnector base class
# ---------------------------------------------------------------------------


class TestStreamConnectorBase:
    def test_connect_raises(self):
        connector = StreamConnector()
        with pytest.raises(RuntimeError):
            asyncio.run(connector.connect())

    def test_stream_data_raises(self):
        connector = StreamConnector()
        with pytest.raises(RuntimeError):
            asyncio.run(connector.stream_data())

    def test_disconnect_does_not_raise(self):
        connector = StreamConnector()
        asyncio.run(connector.disconnect())


# ---------------------------------------------------------------------------
# MQTTConnector (aiomqtt)
# ---------------------------------------------------------------------------


class TestMQTTConnector:
    def test_init_defaults(self):
        connector = MQTTConnector({})
        assert connector.host == "localhost"
        assert connector.port == 1883
        assert connector.client_id == "geo_infer_data"

    def test_init_custom(self):
        connector = MQTTConnector(
            {"host": "broker.example.com", "port": 8883, "client_id": "custom"}
        )
        assert connector.host == "broker.example.com"
        assert connector.port == 8883
        assert connector.client_id == "custom"

    def test_message_payload_decoding(self):
        from geo_infer_data.connectors.stream import _decode_payload

        assert _decode_payload(b'{"temp": 21}') == {"temp": 21}
        assert _decode_payload(b"plain text") == "plain text"
        assert _decode_payload(42) == 42


# ---------------------------------------------------------------------------
# KafkaConnector: explicit placeholder
# ---------------------------------------------------------------------------


class TestKafkaConnector:
    def test_init(self):
        connector = KafkaConnector(
            {"bootstrap_servers": ["kafka:9092"], "group_id": "test-group"}
        )
        assert connector.bootstrap_servers == ["kafka:9092"]
        assert connector.group_id == "test-group"

    def test_operations_raise_clear_error(self):
        connector = KafkaConnector({})
        with pytest.raises(RuntimeError, match="Kafka client library"):
            asyncio.run(connector.connect())
        gen = connector.stream_data(topic="geo-events")
        with pytest.raises(RuntimeError, match="Kafka client library"):
            _collect(gen)
        asyncio.run(connector.disconnect())


# ---------------------------------------------------------------------------
# WebSocketConnector (aiohttp)
# ---------------------------------------------------------------------------


async def _start_echo_server(n_messages):
    """Start an aiohttp server that pushes n_messages then closes the socket."""

    async def push(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        for i in range(n_messages):
            await ws.send_str(f'{{"seq": {i}}}')
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/ws", push)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


class TestWebSocketConnector:
    def test_init(self):
        connector = WebSocketConnector(
            {"url": "ws://example.com:8080", "reconnect_interval": 10}
        )
        assert connector.url == "ws://example.com:8080"
        assert connector.reconnect_interval == 10

    def test_connect_refused_raises(self):
        # Use a loopback port the kernel just handed out and released: nothing
        # is bound there, so the connect is refused with RST, unlike a fixed
        # low port that a DROP firewall rule could blackhole into a hang.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        connector = WebSocketConnector({"url": f"ws://127.0.0.1:{port}/ws"})
        with pytest.raises(aiohttp.ClientError):
            # Explicit client-side bound: a hang fails loudly with
            # TimeoutError instead of stalling the suite.
            asyncio.run(asyncio.wait_for(connector.connect(), timeout=10))

    def test_stream_data_requires_connection(self):
        connector = WebSocketConnector({})
        with pytest.raises(RuntimeError, match="connect"):
            _collect(connector.stream_data())

    def test_receives_real_server_messages(self):
        async def main():
            runner, port = await _start_echo_server(n_messages=3)
            try:
                connector = WebSocketConnector({"url": f"ws://127.0.0.1:{port}/ws"})
                assert await connector.connect() is True

                messages = []
                async for record in connector.stream_data():
                    messages.append(record)
                    if len(messages) >= 3:
                        break

                assert all(r["type"] == "text" for r in messages)
                assert [r["data"]["seq"] for r in messages] == [0, 1, 2]

                await connector.disconnect()
            finally:
                await runner.cleanup()

        asyncio.run(main())
