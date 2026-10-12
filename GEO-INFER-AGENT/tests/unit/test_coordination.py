#!/usr/bin/env python3

"""
Tests for multi-agent coordination through the AgentRegistry.
"""

import unittest

import pytest
import pytest_asyncio

from geo_infer_agent.core.agent_registry import AgentRegistry


class TestAgentRegistry:
    """Tests for the AgentRegistry managing multiple agents."""

    def _set_up(self) -> None:
        """Reset the singleton for clean tests."""
        AgentRegistry._instance = None
        self.registry = AgentRegistry()

    def _tear_down(self) -> None:
        AgentRegistry._instance = None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_create_agent_returns_id(self) -> None:
        """create_agent returns the agent ID."""
        agent_id = await self.registry.create_agent(
            agent_type="default", config={}, agent_id="test-1"
        )
        unittest.TestCase().assertEqual(agent_id, "test-1")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_create_agent_with_auto_id(self) -> None:
        """create_agent auto-generates ID when none provided."""
        agent_id = await self.registry.create_agent(agent_type="default", config={})
        unittest.TestCase().assertIsNotNone(agent_id)
        unittest.TestCase().assertGreater(len(agent_id), 10)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_duplicate_agent_id_raises(self) -> None:
        """Creating an agent with a duplicate ID raises ValueError."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="dup"
        )
        with unittest.TestCase().assertRaises(ValueError):
            await self.registry.create_agent(
                agent_type="default", config={}, agent_id="dup"
            )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unknown_agent_type_raises(self) -> None:
        """Creating an agent with unknown type raises ValueError."""
        with unittest.TestCase().assertRaises((ValueError, ImportError)):
            await self.registry.create_agent(agent_type="nonexistent_type", config={})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_agent(self) -> None:
        """get_agent retrieves the agent instance by ID."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="get-test"
        )
        agent = self.registry.get_agent("get-test")
        unittest.TestCase().assertIsNotNone(agent)
        unittest.TestCase().assertEqual(agent.agent_id, "get-test")

    def test_get_agent_not_found_raises(self) -> None:
        """get_agent raises KeyError for unknown agent ID."""
        with unittest.TestCase().assertRaises(KeyError):
            self.registry.get_agent("nonexistent")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_remove_agent(self) -> None:
        """remove_agent deletes the agent from registry."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="rm-test"
        )
        self.registry.remove_agent("rm-test")
        with unittest.TestCase().assertRaises(KeyError):
            self.registry.get_agent("rm-test")

    def test_remove_nonexistent_raises(self) -> None:
        """remove_agent raises KeyError for unknown agent."""
        with unittest.TestCase().assertRaises(KeyError):
            self.registry.remove_agent("ghost")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_list_agents(self) -> None:
        """list_agents returns JSON-safe info for all registered agents."""
        await self.registry.create_agent(agent_type="default", config={}, agent_id="a1")
        await self.registry.create_agent(agent_type="default", config={}, agent_id="a2")
        listed = self.registry.list_agents()
        unittest.TestCase().assertEqual(
            {item["agent_id"] for item in listed}, {"a1", "a2"}
        )
        unittest.TestCase().assertTrue(all(item["created_at"] for item in listed))

    def test_list_agent_types(self) -> None:
        """list_agent_types returns the available type mappings."""
        types = self.registry.list_agent_types()
        unittest.TestCase().assertIn("default", types)
        unittest.TestCase().assertIsInstance(types, dict)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_is_agent_running_initially_false(self) -> None:
        """A newly created agent is not running."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="run-check"
        )
        unittest.TestCase().assertFalse(self.registry.is_agent_running("run-check"))

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        self._set_up()
        try:
            yield
        finally:
            self._tear_down()


class TestMultiAgentCoordination:
    """Tests for coordinating multiple agents through the registry."""

    def _set_up(self) -> None:
        AgentRegistry._instance = None
        self.registry = AgentRegistry()

    def _tear_down(self) -> None:
        AgentRegistry._instance = None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_message_between_agents(self) -> None:
        """Messages can be sent between two registered agents."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="sender"
        )
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="receiver"
        )
        success = await self.registry.send_message(
            "sender", "receiver", {"cmd": "ping"}
        )
        unittest.TestCase().assertTrue(success)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_message_unknown_sender_raises(self) -> None:
        """Sending from unknown agent raises KeyError."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="target"
        )
        with unittest.TestCase().assertRaises(KeyError):
            await self.registry.send_message("unknown", "target", {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_message_unknown_receiver_raises(self) -> None:
        """Sending to unknown agent raises KeyError."""
        await self.registry.create_agent(
            agent_type="default", config={}, agent_id="origin"
        )
        with unittest.TestCase().assertRaises(KeyError):
            await self.registry.send_message("origin", "unknown", {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_region_passed_to_agent_config(self) -> None:
        """Region parameter is included in agent config."""
        await self.registry.create_agent(
            agent_type="default",
            config={"key": "val"},
            agent_id="geo-agent",
            region="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        )
        agent = self.registry.get_agent("geo-agent")
        unittest.TestCase().assertEqual(
            agent.config.get("region"), "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
        )

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        self._set_up()
        try:
            yield
        finally:
            self._tear_down()
