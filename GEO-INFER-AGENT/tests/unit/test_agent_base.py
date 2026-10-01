#!/usr/bin/env python3

"""
Tests for agent_base module: AgentState and BaseAgent lifecycle.
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime

import numpy as np

from geo_infer_agent.core.agent_base import AgentState, ExampleAgent


class TestAgentState(unittest.TestCase):
    """Tests for the AgentState class."""

    def test_initial_state_is_empty(self) -> None:
        """AgentState starts with empty beliefs, desires, intentions, and memory."""
        state = AgentState(capacity=50)
        self.assertEqual(state.beliefs, {})
        self.assertEqual(state.desires, [])
        self.assertEqual(state.intentions, [])
        self.assertEqual(state.memory, [])
        self.assertEqual(state.memory_capacity, 50)
        self.assertIsInstance(state.creation_time, datetime)
        self.assertIsInstance(state.last_update, datetime)

    def test_update_belief_records_change(self) -> None:
        """Updating a belief stores the value and logs to memory."""
        state = AgentState()
        state.update_belief("temperature", 22.5)
        self.assertEqual(state.beliefs["temperature"], 22.5)
        # Memory should contain a belief_update entry
        updates = [m for m in state.memory if m["type"] == "belief_update"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["key"], "temperature")
        self.assertIsNone(updates[0]["old_value"])
        self.assertEqual(updates[0]["new_value"], 22.5)

    def test_update_belief_same_value_no_memory(self) -> None:
        """Re-setting a belief to the same value should not add to memory."""
        state = AgentState()
        state.update_belief("key", "value")
        mem_count = len(state.memory)
        state.update_belief("key", "value")
        # No new memory entry because value did not change
        self.assertEqual(len(state.memory), mem_count)

    def test_update_array_belief_uses_elementwise_equality(self) -> None:
        """Array-valued beliefs do not trigger ambiguous truth-value errors."""
        state = AgentState()
        state.update_belief("vector", np.array([1.0, 2.0]))
        state.update_belief("vector", np.array([1.0, 2.0]))

        updates = [item for item in state.memory if item["key"] == "vector"]
        self.assertEqual(len(updates), 1)

    def test_add_desire_requires_keys(self) -> None:
        """add_desire raises ValueError if priority or description is missing."""
        state = AgentState()
        with self.assertRaises(ValueError):
            state.add_desire({"priority": 1})
        with self.assertRaises(ValueError):
            state.add_desire({"description": "test"})

    def test_desires_sorted_by_priority(self) -> None:
        """Desires are maintained in descending priority order."""
        state = AgentState()
        state.add_desire({"priority": 1, "description": "low"})
        state.add_desire({"priority": 10, "description": "high"})
        state.add_desire({"priority": 5, "description": "mid"})
        priorities = [d["priority"] for d in state.desires]
        self.assertEqual(priorities, [10, 5, 1])

    def test_get_top_desire(self) -> None:
        """get_top_desire returns highest priority desire or None."""
        state = AgentState()
        self.assertIsNone(state.get_top_desire())
        state.add_desire({"priority": 3, "description": "a"})
        state.add_desire({"priority": 7, "description": "b"})
        top = state.get_top_desire()
        self.assertIsNotNone(top)
        self.assertEqual(top["description"], "b")

    def test_set_intention_requires_actions(self) -> None:
        """set_intention raises ValueError without actions key."""
        state = AgentState()
        with self.assertRaises(ValueError):
            state.set_intention({"name": "plan_without_actions"})

    def test_memory_capacity_enforcement(self) -> None:
        """Memory should not exceed the configured capacity."""
        state = AgentState(capacity=5)
        for i in range(10):
            state.add_to_memory({"index": i})
        self.assertEqual(len(state.memory), 5)
        # Oldest items should have been evicted
        self.assertEqual(state.memory[0]["index"], 5)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """State serialization and deserialization produce equivalent state."""
        state = AgentState()
        state.update_belief("sensor", 42)
        state.add_desire({"priority": 5, "description": "explore"})
        state.set_intention({"actions": ["step1", "step2"]})

        d = state.to_dict()
        restored = AgentState.from_dict(d)

        self.assertEqual(restored.beliefs, state.beliefs)
        self.assertEqual(len(restored.desires), len(state.desires))
        self.assertEqual(len(restored.intentions), len(state.intentions))


class TestBaseAgentLifecycle(unittest.TestCase):
    """Tests for BaseAgent initialization and lifecycle methods."""

    def test_agent_gets_unique_id(self) -> None:
        """Agent auto-generates a UUID when no ID is provided."""
        agent = ExampleAgent()
        self.assertIsNotNone(agent.agent_id)
        self.assertGreater(len(agent.agent_id), 10)

    def test_agent_uses_given_id(self) -> None:
        """Agent uses the explicitly provided agent_id."""
        agent = ExampleAgent(agent_id="custom-id")
        self.assertEqual(agent.agent_id, "custom-id")

    def test_agent_initial_state_not_running(self) -> None:
        """Newly created agent is not running."""
        agent = ExampleAgent()
        self.assertFalse(agent.running)
        self.assertIsNone(agent.start_time)
        self.assertIsNone(agent.stop_time)

    def test_stop_sets_running_false(self) -> None:
        """Calling stop() sets running to False."""
        agent = ExampleAgent()
        agent.running = True
        agent.stop()
        self.assertFalse(agent.running)

    def test_save_and_load_state(self) -> None:
        """Agent state can be saved to and loaded from a JSON file."""
        agent = ExampleAgent(agent_id="save-test")
        agent.state.update_belief("key", "value")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name

        try:
            saved_path = agent.save_state(filepath)
            self.assertEqual(saved_path, filepath)

            # Verify JSON is valid
            with open(filepath) as f:
                data = json.load(f)
            self.assertEqual(data["agent_id"], "save-test")
            self.assertEqual(data["beliefs"]["key"], "value")
        finally:
            os.unlink(filepath)


class TestExampleAgentAsync(unittest.TestCase):
    """Tests for ExampleAgent async methods."""

    def test_initialize_sets_beliefs_and_desires(self) -> None:
        """ExampleAgent.initialize() populates initial beliefs and desires."""
        agent = ExampleAgent()
        asyncio.get_event_loop().run_until_complete(agent.initialize())
        self.assertIn("environment_known", agent.state.beliefs)
        self.assertFalse(agent.state.beliefs["environment_known"])
        self.assertGreater(len(agent.state.desires), 0)

    def test_perceive_returns_dict(self) -> None:
        """ExampleAgent.perceive() returns a dictionary with expected keys."""
        agent = ExampleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.perceive())
        self.assertIsInstance(result, dict)
        self.assertIn("current_time", result)
        self.assertIn("random_observation", result)

    def test_decide_returns_explore_when_unknown(self) -> None:
        """Agent decides to explore when environment_known is False."""
        agent = ExampleAgent()
        asyncio.get_event_loop().run_until_complete(agent.initialize())
        action = asyncio.get_event_loop().run_until_complete(agent.decide())
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "explore")

    def test_decide_returns_none_when_known(self) -> None:
        """Agent returns None action after environment is known."""
        agent = ExampleAgent()
        asyncio.get_event_loop().run_until_complete(agent.initialize())
        agent.state.update_belief("environment_known", True)
        action = asyncio.get_event_loop().run_until_complete(agent.decide())
        self.assertIsNone(action)

    def test_act_explore_returns_success(self) -> None:
        """Acting on an explore action returns success."""
        agent = ExampleAgent(config={"decision_frequency": 0.01})
        action = {"type": "explore", "target": "environment", "params": {}}
        result = asyncio.get_event_loop().run_until_complete(agent.act(action))
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
