"""Integration tests for WebSocket real-time broadcasting (GS-226).

Starts a real GeospatialCommunicationSystem + WebSocketAPIManager, connects a
websockets client, and asserts that messages published through the broker,
events published through the event manager, and notifications delivered via
the websocket delivery handler all reach the connected client. Publishing
happens from plain worker threads, exercising the loop-safe broadcast
handoff (no ``asyncio.create_task``-from-sync-thread RuntimeError).
"""

import asyncio
import json
import threading
import time

import pytest
import websockets

from geo_infer_comms import GeospatialCommunicationSystem
from geo_infer_comms.api.websocket_api import WebSocketAPIManager
from geo_infer_comms.models.message import (
    EventPublishRequest,
    MessageRequest,
    NotificationRequest,
)


def _get_port(manager: WebSocketAPIManager) -> int:
    sockets = manager.websocket_server.server.sockets
    return sockets[0].getsockname()[1]


async def _wait_for_message(websocket, timeout: float = 5.0) -> dict:
    """Read frames until the first non-welcome message, or fail."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                "no broadcast received within timeout; welcome/other frames: " + "none"
            )
        frame = json.loads(await asyncio.wait_for(websocket.recv(), remaining))
        if frame.get("type") != "connection_established":
            return frame


class _Publisher:
    """Runs a publish callable on a plain thread, like a broker worker."""

    def __init__(self, fn):
        self.error = None
        thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        thread.start()
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise AssertionError("publisher thread hung")
        if self.error is not None:
            raise self.error

    def _run(self, fn):
        try:
            fn()
        except BaseException as exc:  # surfaced to the test thread
            self.error = exc


@pytest.fixture()
def running_system():
    system = GeospatialCommunicationSystem(
        {
            "enable_persistence": False,
            "message_queue_size": 1000,
            "max_events": 1000,
            "max_notifications": 1000,
        }
    )
    system.start()
    try:
        yield system
    finally:
        system.stop()


async def _scenario_publish_kind(running_system, kind: str) -> dict:
    manager = WebSocketAPIManager(running_system)
    await manager.start()
    try:
        port = _get_port(manager)
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            if kind == "message":
                _Publisher(
                    lambda: running_system.message_broker.send_message(
                        MessageRequest(
                            content="hello over websocket",
                            recipients=["user_1"],
                        ),
                        sender_id="user_0",
                    )
                )
            elif kind == "event":
                _Publisher(
                    lambda: running_system.event_manager.publish_event(
                        EventPublishRequest(
                            event_type="data_update",
                            payload={"value": 1},
                            source="integration-test",
                        )
                    )
                )
            else:
                notification = running_system.notification_manager.create_notification(
                    NotificationRequest(
                        title="Real-time notice",
                        content="delivered over websocket",
                        recipients=["user_1"],
                        delivery_method=["websocket"],
                    )
                )
                _Publisher(
                    lambda: running_system.notification_manager.send_notification(
                        notification.notification_id
                    )
                )
            return await _wait_for_message(client)
    finally:
        await manager.stop()


@pytest.mark.parametrize("kind", ["message", "event", "notification"])
def test_broker_publish_reaches_websocket_client(running_system, kind):
    frame = asyncio.run(_scenario_publish_kind(running_system, kind))
    assert frame["type"] == kind


def test_sync_broadcast_from_plain_thread_does_not_raise(running_system):
    """Loop-safe handoff: a bare thread with no event loop can broadcast."""
    manager = WebSocketAPIManager(running_system)

    _Publisher(
        lambda: manager.websocket_server.websocket_manager.broadcast_message(
            {"type": "system_message", "content": "thread broadcast"}
        )
    )


def test_get_stats_reports_real_broadcast_count(running_system):
    async def scenario():
        manager = WebSocketAPIManager(running_system)
        await manager.start()
        try:
            port = _get_port(manager)
            async with websockets.connect(f"ws://127.0.0.1:{port}"):
                assert (
                    manager.get_stats()["message_broadcaster"]["broadcasts_sent"] == 0
                )
                _Publisher(
                    lambda: running_system.event_manager.publish_event(
                        EventPublishRequest(
                            event_type="system_alert",
                            payload={"alert": True},
                            source="integration-test",
                        )
                    )
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    sent = manager.get_stats()["message_broadcaster"]["broadcasts_sent"]
                    if sent >= 1:
                        break
                    await asyncio.sleep(0.05)
                assert (
                    manager.get_stats()["message_broadcaster"]["broadcasts_sent"] >= 1
                ), "broadcasts_sent never advanced"
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_broadcasters_registered_with_system_emission_points(running_system):
    manager = WebSocketAPIManager(running_system)

    broker_subs = running_system.message_broker.subscription_callbacks
    assert any(
        owner == "websocket_api_broadcaster" for owner, _ in broker_subs.values()
    )
    assert (
        "websocket_api_broadcaster" in running_system.event_manager.subscriber_callbacks
    )
    assert (
        running_system.notification_manager.delivery_handlers.get("websocket")
        is not None
    )
    assert manager.websocket_server.websocket_manager.broadcasts_sent == 0
