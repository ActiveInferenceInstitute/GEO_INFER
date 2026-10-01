#!/usr/bin/env python3

"""
Tests for the messaging module: pub/sub, queue operations, message routing.
"""

import asyncio
import unittest
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


class TestMessagingService(unittest.TestCase):
    """Tests for MessagingService queue and pub/sub operations."""

    def setUp(self) -> None:
        """Reset singleton for clean tests."""
        MessagingService._instance = None
        self.service = MessagingService()

    def tearDown(self) -> None:
        MessagingService._instance = None

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_register_agent_creates_queue(self) -> None:
        """register_agent creates a message queue for the agent."""
        self.service.register_agent("agent-1")
        self.assertIn("agent-1", self.service.message_queues)
        self.assertEqual(len(self.service.message_queues["agent-1"]), 0)

    def test_unregister_agent_removes_queue_and_subscriptions(self) -> None:
        """unregister_agent removes queue and channel subscriptions."""
        self.service.register_agent("agent-1")
        self.service.subscribe("agent-1", "alerts")
        self.service.unregister_agent("agent-1")
        self.assertNotIn("agent-1", self.service.message_queues)
        self.assertNotIn("agent-1", self.service.channels.get("alerts", set()))

    def test_send_message_queues_for_recipient(self) -> None:
        """Sending a message places it in the recipient's queue."""
        self.service.register_agent("receiver")
        msg = Message("sender", "receiver", {"text": "hi"})
        success = self._run(self.service.send_message(msg))
        self.assertTrue(success)
        self.assertEqual(len(self.service.message_queues["receiver"]), 1)

    def test_send_expired_message_rejected(self) -> None:
        """An already-expired message is rejected."""
        self.service.register_agent("target")
        past = datetime.now() - timedelta(hours=1)
        msg = Message("sender", "target", {}, expires_at=past)
        success = self._run(self.service.send_message(msg))
        self.assertFalse(success)

    def test_messages_sorted_by_priority(self) -> None:
        """Messages in the queue are sorted by priority (highest first)."""
        self.service.register_agent("r")
        m_low = Message("s", "r", {"p": "low"}, priority=1)
        m_high = Message("s", "r", {"p": "high"}, priority=9)
        m_mid = Message("s", "r", {"p": "mid"}, priority=5)
        self._run(self.service.send_message(m_low))
        self._run(self.service.send_message(m_high))
        self._run(self.service.send_message(m_mid))

        queue = self.service.message_queues["r"]
        priorities = [m.priority for m in queue]
        self.assertEqual(priorities, [9, 5, 1])

    def test_get_messages_marks_as_delivered(self) -> None:
        """get_messages marks retrieved messages as delivered and read."""
        self.service.register_agent("r")
        msg = Message("s", "r", {"data": 1})
        self._run(self.service.send_message(msg))

        messages = self._run(self.service.get_messages("r"))
        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0].delivered)
        self.assertTrue(messages[0].read)

    def test_get_messages_filters_expired(self) -> None:
        """get_messages filters out expired messages."""
        self.service.register_agent("r")
        past = datetime.now() - timedelta(seconds=1)
        expired_msg = Message("s", "r", {}, expires_at=past)
        # Force into queue (bypass send_message expiry check)
        self.service.message_queues["r"].append(expired_msg)
        valid_msg = Message("s", "r", {"valid": True})
        self._run(self.service.send_message(valid_msg))

        messages = self._run(self.service.get_messages("r"))
        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0].content.get("valid"))

    def test_subscribe_and_unsubscribe(self) -> None:
        """Agents can subscribe and unsubscribe from channels."""
        self.service.subscribe("a1", "weather")
        self.assertIn("a1", self.service.channels["weather"])
        self.service.unsubscribe("a1", "weather")
        self.assertNotIn("a1", self.service.channels.get("weather", set()))

    def test_broadcast_sends_to_all_subscribers(self) -> None:
        """broadcast_message sends to all channel subscribers."""
        self.service.register_agent("sub1")
        self.service.register_agent("sub2")
        self.service.subscribe("sub1", "alerts")
        self.service.subscribe("sub2", "alerts")

        count = self._run(
            self.service.broadcast_message("broadcaster", {"alert": "fire"}, "alerts")
        )
        self.assertEqual(count, 2)
        self.assertEqual(len(self.service.message_queues["sub1"]), 1)
        self.assertEqual(len(self.service.message_queues["sub2"]), 1)

    def test_broadcast_to_nonexistent_channel(self) -> None:
        """Broadcasting to a nonexistent channel sends to zero agents."""
        count = self._run(self.service.broadcast_message("sender", {}, "ghost_channel"))
        self.assertEqual(count, 0)

    def test_register_message_callback(self) -> None:
        """A callback can be registered for an agent."""
        received = []
        self.service.register_agent("cb-agent")
        self.service.register_message_callback(
            "cb-agent", lambda msg: received.append(msg)
        )
        self.assertIn("cb-agent", self.service.message_callbacks)

    def test_successful_callback_consumes_message_once(self) -> None:
        """A successful callback receives a queued message exactly once."""
        received = []
        self.service.register_agent("cb-agent")
        self.service.register_message_callback(
            "cb-agent", lambda msg: received.append(msg.message_id)
        )
        message = Message("sender", "cb-agent", {"value": 1})
        self._run(self.service.send_message(message))

        async def process_once() -> None:
            self.service.running = True
            task = asyncio.create_task(self.service._process_messages())
            await asyncio.sleep(0.15)
            self.service.running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._run(process_once())
        self.assertEqual(received, [message.message_id])
        self.assertEqual(self.service.message_queues["cb-agent"], [])


if __name__ == "__main__":
    unittest.main()
