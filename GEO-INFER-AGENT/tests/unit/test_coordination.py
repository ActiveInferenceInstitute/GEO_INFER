#!/usr/bin/env python3

"""
Tests for multi-agent coordination through the AgentRegistry.
"""

import asyncio
import unittest

from geo_infer_agent.core.agent_registry import AgentRegistry


class TestAgentRegistry(unittest.TestCase):
    """Tests for the AgentRegistry managing multiple agents."""

    def setUp(self) -> None:
        """Reset the singleton for clean tests."""
        AgentRegistry._instance = None
        self.registry = AgentRegistry()

    def tearDown(self) -> None:
        AgentRegistry._instance = None

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_create_agent_returns_id(self) -> None:
        """create_agent returns the agent ID."""
        agent_id = self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="test-1"
            )
        )
        self.assertEqual(agent_id, "test-1")

    def test_create_agent_with_auto_id(self) -> None:
        """create_agent auto-generates ID when none provided."""
        agent_id = self._run(
            self.registry.create_agent(agent_type="default", config={})
        )
        self.assertIsNotNone(agent_id)
        self.assertGreater(len(agent_id), 10)

    def test_duplicate_agent_id_raises(self) -> None:
        """Creating an agent with a duplicate ID raises ValueError."""
        self._run(
            self.registry.create_agent(agent_type="default", config={}, agent_id="dup")
        )
        with self.assertRaises(ValueError):
            self._run(
                self.registry.create_agent(
                    agent_type="default", config={}, agent_id="dup"
                )
            )

    def test_unknown_agent_type_raises(self) -> None:
        """Creating an agent with unknown type raises ValueError."""
        with self.assertRaises((ValueError, ImportError)):
            self._run(
                self.registry.create_agent(agent_type="nonexistent_type", config={})
            )

    def test_get_agent(self) -> None:
        """get_agent retrieves the agent instance by ID."""
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="get-test"
            )
        )
        agent = self.registry.get_agent("get-test")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_id, "get-test")

    def test_get_agent_not_found_raises(self) -> None:
        """get_agent raises KeyError for unknown agent ID."""
        with self.assertRaises(KeyError):
            self.registry.get_agent("nonexistent")

    def test_remove_agent(self) -> None:
        """remove_agent deletes the agent from registry."""
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="rm-test"
            )
        )
        self.registry.remove_agent("rm-test")
        with self.assertRaises(KeyError):
            self.registry.get_agent("rm-test")

    def test_remove_nonexistent_raises(self) -> None:
        """remove_agent raises KeyError for unknown agent."""
        with self.assertRaises(KeyError):
            self.registry.remove_agent("ghost")

    def test_list_agents(self) -> None:
        """list_agents returns JSON-safe info for all registered agents."""
        self._run(
            self.registry.create_agent(agent_type="default", config={}, agent_id="a1")
        )
        self._run(
            self.registry.create_agent(agent_type="default", config={}, agent_id="a2")
        )
        listed = self.registry.list_agents()
        self.assertEqual({item["agent_id"] for item in listed}, {"a1", "a2"})
        self.assertTrue(all(item["created_at"] for item in listed))

    def test_list_agent_types(self) -> None:
        """list_agent_types returns the available type mappings."""
        types = self.registry.list_agent_types()
        self.assertIn("default", types)
        self.assertIsInstance(types, dict)

    def test_is_agent_running_initially_false(self) -> None:
        """A newly created agent is not running."""
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="run-check"
            )
        )
        self.assertFalse(self.registry.is_agent_running("run-check"))


class TestMultiAgentCoordination(unittest.TestCase):
    """Tests for coordinating multiple agents through the registry."""

    def setUp(self) -> None:
        AgentRegistry._instance = None
        self.registry = AgentRegistry()

    def tearDown(self) -> None:
        AgentRegistry._instance = None

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_send_message_between_agents(self) -> None:
        """Messages can be sent between two registered agents."""
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="sender"
            )
        )
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="receiver"
            )
        )
        success = self._run(
            self.registry.send_message("sender", "receiver", {"cmd": "ping"})
        )
        self.assertTrue(success)

    def test_send_message_unknown_sender_raises(self) -> None:
        """Sending from unknown agent raises KeyError."""
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="target"
            )
        )
        with self.assertRaises(KeyError):
            self._run(self.registry.send_message("unknown", "target", {}))

    def test_send_message_unknown_receiver_raises(self) -> None:
        """Sending to unknown agent raises KeyError."""
        self._run(
            self.registry.create_agent(
                agent_type="default", config={}, agent_id="origin"
            )
        )
        with self.assertRaises(KeyError):
            self._run(self.registry.send_message("origin", "unknown", {}))

    def test_region_passed_to_agent_config(self) -> None:
        """Region parameter is included in agent config."""
        self._run(
            self.registry.create_agent(
                agent_type="default",
                config={"key": "val"},
                agent_id="geo-agent",
                region="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            )
        )
        agent = self.registry.get_agent("geo-agent")
        self.assertEqual(
            agent.config.get("region"), "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
        )


if __name__ == "__main__":
    unittest.main()
