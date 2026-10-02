#!/usr/bin/env python3

"""
Tests for the messaging module: pub/sub, queue operations, message routing.
"""

import asyncio
import unittest

import pytest
import pytest_asyncio
from datetime import datetime, timedelta

from geo_infer_agent.api.messaging import Message, MessagingService


class TestMessage(unittest.TestCase):
    """Tests for the Message class."""

    def test_message_creation(self) -> None:
        """A message is created with all fields populated."""
        msg = Message(
            from_agent_id="sender",
            to_agent_id="receiver",
            content={"data": "payload"},
            message_type="standard",
            priority=5,
        )
        self.assertEqual(msg.from_agent_id, "sender")
        self.assertEqual(msg.to_agent_id, "receiver")
        self.assertEqual(msg.content["data"], "payload")
        self.assertEqual(msg.priority, 5)
        self.assertFalse(msg.delivered)
        self.assertFalse(msg.read)
        self.assertIsNotNone(msg.message_id)
        self.assertIsInstance(msg.created_at, datetime)

    def test_priority_clamped(self) -> None:
        """Priority is clamped between 1 and 10."""
        low = Message("a", "b", {}, priority=0)
        high = Message("a", "b", {}, priority=15)
        self.assertEqual(low.priority, 1)
        self.assertEqual(high.priority, 10)

    def test_message_not_expired_by_default(self) -> None:
        """A message without expires_at is never expired."""
        msg = Message("a", "b", {})
        self.assertFalse(msg.is_expired())

    def test_message_expired(self) -> None:
        """A message with a past expires_at is expired."""
        past = datetime.now() - timedelta(hours=1)
        msg = Message("a", "b", {}, expires_at=past)
        self.assertTrue(msg.is_expired())

    def test_message_not_yet_expired(self) -> None:
        """A message with a future expires_at is not expired."""
        future = datetime.now() + timedelta(hours=1)
        msg = Message("a", "b", {}, expires_at=future)
        self.assertFalse(msg.is_expired())

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """Message serialization and deserialization roundtrip works."""
        msg = Message("alpha", "beta", {"key": "val"}, priority=7)
        d = msg.to_dict()
        restored = Message.from_dict(d)
        self.assertEqual(restored.message_id, msg.message_id)
        self.assertEqual(restored.from_agent_id, "alpha")
        self.assertEqual(restored.to_agent_id, "beta")
        self.assertEqual(restored.priority, 7)
        self.assertEqual(restored.content["key"], "val")


class TestMessagingService:
    """Tests for MessagingService queue and pub/sub operations."""

    def _set_up(self) -> None:
        """Reset singleton for clean tests."""
        MessagingService._instance = None
        self.service = MessagingService()

    def _tear_down(self) -> None:
        MessagingService._instance = None

    def test_register_agent_creates_queue(self) -> None:
        """register_agent creates a message queue for the agent."""
        self.service.register_agent("agent-1")
        unittest.TestCase().assertIn("agent-1", self.service.message_queues)
        unittest.TestCase().assertEqual(len(self.service.message_queues["agent-1"]), 0)

    def test_unregister_agent_removes_queue_and_subscriptions(self) -> None:
        """unregister_agent removes queue and channel subscriptions."""
        self.service.register_agent("agent-1")
        self.service.subscribe("agent-1", "alerts")
        self.service.unregister_agent("agent-1")
        unittest.TestCase().assertNotIn("agent-1", self.service.message_queues)
        unittest.TestCase().assertNotIn(
            "agent-1", self.service.channels.get("alerts", set())
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_message_queues_for_recipient(self) -> None:
        """Sending a message places it in the recipient's queue."""
        self.service.register_agent("receiver")
        msg = Message("sender", "receiver", {"text": "hi"})
        success = await self.service.send_message(msg)
        unittest.TestCase().assertTrue(success)
        unittest.TestCase().assertEqual(len(self.service.message_queues["receiver"]), 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_expired_message_rejected(self) -> None:
        """An already-expired message is rejected."""
        self.service.register_agent("target")
        past = datetime.now() - timedelta(hours=1)
        msg = Message("sender", "target", {}, expires_at=past)
        success = await self.service.send_message(msg)
        unittest.TestCase().assertFalse(success)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_messages_sorted_by_priority(self) -> None:
        """Messages in the queue are sorted by priority (highest first)."""
        self.service.register_agent("r")
        m_low = Message("s", "r", {"p": "low"}, priority=1)
        m_high = Message("s", "r", {"p": "high"}, priority=9)
        m_mid = Message("s", "r", {"p": "mid"}, priority=5)
        await self.service.send_message(m_low)
        await self.service.send_message(m_high)
        await self.service.send_message(m_mid)

        queue = self.service.message_queues["r"]
        priorities = [m.priority for m in queue]
        unittest.TestCase().assertEqual(priorities, [9, 5, 1])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_messages_marks_as_delivered(self) -> None:
        """get_messages marks retrieved messages as delivered and read."""
        self.service.register_agent("r")
        msg = Message("s", "r", {"data": 1})
        await self.service.send_message(msg)

        messages = await self.service.get_messages("r")
        unittest.TestCase().assertEqual(len(messages), 1)
        unittest.TestCase().assertTrue(messages[0].delivered)
        unittest.TestCase().assertTrue(messages[0].read)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_messages_filters_expired(self) -> None:
        """get_messages filters out expired messages."""
        self.service.register_agent("r")
        past = datetime.now() - timedelta(seconds=1)
        expired_msg = Message("s", "r", {}, expires_at=past)
        # Force into queue (bypass send_message expiry check)
        self.service.message_queues["r"].append(expired_msg)
        valid_msg = Message("s", "r", {"valid": True})
        await self.service.send_message(valid_msg)

        messages = await self.service.get_messages("r")
        unittest.TestCase().assertEqual(len(messages), 1)
        unittest.TestCase().assertTrue(messages[0].content.get("valid"))

    def test_subscribe_and_unsubscribe(self) -> None:
        """Agents can subscribe and unsubscribe from channels."""
        self.service.subscribe("a1", "weather")
        unittest.TestCase().assertIn("a1", self.service.channels["weather"])
        self.service.unsubscribe("a1", "weather")
        unittest.TestCase().assertNotIn(
            "a1", self.service.channels.get("weather", set())
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_broadcast_sends_to_all_subscribers(self) -> None:
        """broadcast_message sends to all channel subscribers."""
        self.service.register_agent("sub1")
        self.service.register_agent("sub2")
        self.service.subscribe("sub1", "alerts")
        self.service.subscribe("sub2", "alerts")

        count = await self.service.broadcast_message(
            "broadcaster", {"alert": "fire"}, "alerts"
        )
        unittest.TestCase().assertEqual(count, 2)
        unittest.TestCase().assertEqual(len(self.service.message_queues["sub1"]), 1)
        unittest.TestCase().assertEqual(len(self.service.message_queues["sub2"]), 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_broadcast_to_nonexistent_channel(self) -> None:
        """Broadcasting to a nonexistent channel sends to zero agents."""
        count = await self.service.broadcast_message("sender", {}, "ghost_channel")
        unittest.TestCase().assertEqual(count, 0)

    def test_register_message_callback(self) -> None:
        """A callback can be registered for an agent."""
        received = []
        self.service.register_agent("cb-agent")
        self.service.register_message_callback(
            "cb-agent", lambda msg: received.append(msg)
        )
        unittest.TestCase().assertIn("cb-agent", self.service.message_callbacks)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_successful_callback_consumes_message_once(
        self, managed_task
    ) -> None:
        """A successful callback receives a queued message exactly once."""
        received = []
        self.service.register_agent("cb-agent")
        delivered = asyncio.Event()

        def receive(message):
            received.append(message.message_id)
            delivered.set()

        self.service.register_message_callback("cb-agent", receive)
        message = Message("sender", "cb-agent", {"value": 1})
        await self.service.send_message(message)

        self.service.running = True
        try:
            async with managed_task(self.service._process_messages()):
                await asyncio.wait_for(delivered.wait(), timeout=2)
        finally:
            self.service.running = False
        unittest.TestCase().assertEqual(received, [message.message_id])
        unittest.TestCase().assertEqual(self.service.message_queues["cb-agent"], [])

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        self._set_up()
        try:
            yield
        finally:
            self._tear_down()
