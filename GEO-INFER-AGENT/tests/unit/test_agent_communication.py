#!/usr/bin/env python3

"""
Tests for agent-to-agent communication via BaseAgent message queue.
"""

import unittest

import pytest
import uuid
from datetime import datetime

from geo_infer_agent.core.agent_base import ExampleAgent


class TestAgentMessagePassing:
    """Tests for message passing between BaseAgent instances."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_receive_message_enqueues(self) -> None:
        """receive_message puts a message on the agent's internal queue."""
        agent = ExampleAgent(agent_id="receiver")
        msg = {
            "from": "sender-1",
            "to": "receiver",
            "content": {"text": "hello"},
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),
        }
        await agent.receive_message(msg)
        unittest.TestCase().assertFalse(agent.message_queue.empty())

    @pytest.mark.asyncio(loop_scope="function")
    async def test_process_messages_drains_queue(self) -> None:
        """process_messages consumes all queued messages."""
        agent = ExampleAgent(agent_id="processor")
        for i in range(3):
            msg = {
                "from": f"sender-{i}",
                "to": "processor",
                "content": {"index": i},
                "timestamp": datetime.now().isoformat(),
                "message_id": str(uuid.uuid4()),
            }
            await agent.receive_message(msg)

        unittest.TestCase().assertEqual(agent.message_queue.qsize(), 3)
        await agent.process_messages()
        unittest.TestCase().assertTrue(agent.message_queue.empty())

    @pytest.mark.asyncio(loop_scope="function")
    async def test_message_handling_records_in_memory(self) -> None:
        """_handle_message records the message in agent state memory."""
        agent = ExampleAgent(agent_id="mem-agent")
        msg = {
            "from": "alpha",
            "to": "mem-agent",
            "content": {"data": 42},
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),
        }
        await agent.receive_message(msg)
        await agent.process_messages()

        received_entries = [
            m for m in agent.state.memory if m.get("type") == "message_received"
        ]
        unittest.TestCase().assertEqual(len(received_entries), 1)
        unittest.TestCase().assertEqual(received_entries[0]["message"]["from"], "alpha")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_message_delivers_to_registered_agent(self) -> None:
        """send_message delivers to a registered recipient's queue and returns True."""
        from geo_infer_agent.core.agent_registry import AgentRegistry

        AgentRegistry._instance = None
        registry = AgentRegistry()
        try:
            await registry.create_agent(
                agent_type="default", config={}, agent_id="receiver-x"
            )
            sender = ExampleAgent(agent_id="sender-x")
            result = await sender.send_message("receiver-x", {"command": "start"})
            unittest.TestCase().assertTrue(result)
            receiver = registry.get_agent("receiver-x")
            unittest.TestCase().assertFalse(receiver.message_queue.empty())
            delivered = receiver.message_queue.get_nowait()
            unittest.TestCase().assertEqual(delivered["content"], {"command": "start"})
            unittest.TestCase().assertEqual(delivered["from"], "sender-x")
        finally:
            AgentRegistry._instance = None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_message_to_unknown_agent_returns_false(self) -> None:
        """send_message returns False (not a fake success) for an unregistered recipient."""
        from geo_infer_agent.core.agent_registry import AgentRegistry

        AgentRegistry._instance = None
        AgentRegistry()  # ensure a clean, empty registry
        try:
            sender = ExampleAgent(agent_id="sender-y")
            result = await sender.send_message(
                "nobody-registered", {"command": "start"}
            )
            unittest.TestCase().assertFalse(result)
        finally:
            AgentRegistry._instance = None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_agents_communicate(self) -> None:
        """Two agents can exchange messages through their queues."""
        agent_b = ExampleAgent(agent_id="agent-b")

        # Agent A sends to Agent B by placing on B's queue
        msg = {
            "from": "agent-a",
            "to": "agent-b",
            "content": {"task": "analyze"},
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),
        }
        await agent_b.receive_message(msg)

        # Agent B processes messages
        await agent_b.process_messages()

        # Verify B recorded the message
        received = [
            m for m in agent_b.state.memory if m.get("type") == "message_received"
        ]
        unittest.TestCase().assertEqual(len(received), 1)
        unittest.TestCase().assertEqual(
            received[0]["message"]["content"]["task"], "analyze"
        )
