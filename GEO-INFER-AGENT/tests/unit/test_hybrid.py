#!/usr/bin/env python3

"""
Tests for the hybrid agent architecture: SubAgentWrapper, HybridState, decision policy.

Note: hybrid.py references ``agent.id`` but ExampleAgent stores the identifier
as ``agent_id``.  The helper below patches the attribute for compatibility.
"""

import unittest

from geo_infer_agent.core.agent_base import ExampleAgent
from geo_infer_agent.models.hybrid import HybridState, SubAgentWrapper


def _patch_agent(agent: ExampleAgent) -> ExampleAgent:
    """Add ``id`` alias so hybrid.py can find it via ``agent.id``."""
    agent.id = agent.agent_id
    return agent


class TestSubAgentWrapper(unittest.TestCase):
    """Tests for SubAgentWrapper activation and stats tracking."""

    def _make_wrapper(
        self,
        agent_type: str = "default",
        priority: int = 5,
        activation_conditions: dict = None,
    ) -> SubAgentWrapper:
        agent = _patch_agent(ExampleAgent(agent_id=f"sub-{agent_type}"))
        return SubAgentWrapper(
            agent_type=agent_type,
            agent=agent,
            priority=priority,
            activation_conditions=activation_conditions or {},
            description=f"Test {agent_type} agent",
        )

    def test_wrapper_initialization(self) -> None:
        """SubAgentWrapper stores agent metadata correctly."""
        wrapper = self._make_wrapper("rule_based", priority=8)
        self.assertEqual(wrapper.agent_type, "rule_based")
        self.assertEqual(wrapper.priority, 8)
        self.assertTrue(wrapper.is_active)
        self.assertEqual(wrapper.decision_count, 0)
        self.assertAlmostEqual(wrapper.total_reward, 0.0)

    def test_activation_no_conditions(self) -> None:
        """Wrapper with no conditions is always activated."""
        wrapper = self._make_wrapper()
        self.assertTrue(wrapper.check_activation({"any": "context"}))
        self.assertTrue(wrapper.check_activation({}))

    def test_activation_conditions_met(self) -> None:
        """Wrapper activates when all conditions match context."""
        wrapper = self._make_wrapper(
            activation_conditions={"mode": "analysis", "ready": True}
        )
        context = {"mode": "analysis", "ready": True, "extra": "data"}
        self.assertTrue(wrapper.check_activation(context))

    def test_activation_conditions_not_met(self) -> None:
        """Wrapper does not activate when conditions are unmet."""
        wrapper = self._make_wrapper(activation_conditions={"mode": "analysis"})
        self.assertFalse(wrapper.check_activation({"mode": "monitoring"}))
        self.assertFalse(wrapper.check_activation({}))

    def test_record_decision_updates_stats(self) -> None:
        """record_decision increments counters and accumulates reward."""
        wrapper = self._make_wrapper()
        wrapper.record_decision(successful=True, reward=1.5)
        wrapper.record_decision(successful=False, reward=-0.5)
        wrapper.record_decision(successful=True, reward=2.0)

        self.assertEqual(wrapper.decision_count, 3)
        self.assertEqual(wrapper.successful_decision_count, 2)
        self.assertAlmostEqual(wrapper.total_reward, 3.0)
        self.assertAlmostEqual(wrapper.last_reward, 2.0)
        self.assertIsNotNone(wrapper.last_activated)

    def test_to_dict(self) -> None:
        """Wrapper serialization includes all fields."""
        wrapper = self._make_wrapper("rl", priority=3)
        wrapper.record_decision(True, 1.0)
        d = wrapper.to_dict()
        self.assertEqual(d["agent_type"], "rl")
        self.assertEqual(d["priority"], 3)
        self.assertEqual(d["stats"]["decision_count"], 1)


class TestHybridState(unittest.TestCase):
    """Tests for HybridState shared context and sub-agent management."""

    def _make_wrapper(self, agent_id: str, priority: int = 5) -> SubAgentWrapper:
        agent = _patch_agent(ExampleAgent(agent_id=agent_id))
        return SubAgentWrapper(agent_type="default", agent=agent, priority=priority)

    def test_add_and_remove_sub_agent(self) -> None:
        """Sub-agents can be added to and removed from HybridState."""
        state = HybridState()
        wrapper = self._make_wrapper("agent-a")
        state.add_sub_agent(wrapper)
        self.assertIn("agent-a", state.sub_agents)

        removed = state.remove_sub_agent("agent-a")
        self.assertTrue(removed)
        self.assertNotIn("agent-a", state.sub_agents)

    def test_remove_nonexistent_returns_false(self) -> None:
        """Removing a sub-agent that doesn't exist returns False."""
        state = HybridState()
        self.assertFalse(state.remove_sub_agent("ghost"))

    def test_get_active_agents_sorted_by_priority(self) -> None:
        """get_active_agents returns agents sorted by priority descending."""
        state = HybridState()
        state.add_sub_agent(self._make_wrapper("low", priority=1))
        state.add_sub_agent(self._make_wrapper("high", priority=10))
        state.add_sub_agent(self._make_wrapper("mid", priority=5))

        active = state.get_active_agents()
        priorities = [w.priority for w in active]
        self.assertEqual(priorities, [10, 5, 1])

    def test_inactive_agents_excluded(self) -> None:
        """Agents with is_active=False are not in the active list."""
        state = HybridState()
        w1 = self._make_wrapper("active", priority=5)
        w2 = self._make_wrapper("inactive", priority=10)
        w2.is_active = False
        state.add_sub_agent(w1)
        state.add_sub_agent(w2)

        active = state.get_active_agents()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].agent.agent_id, "active")

    def test_context_update_and_retrieval(self) -> None:
        """Shared context can be updated and retrieved."""
        state = HybridState()
        state.update_context("weather", "sunny")
        state.update_context("temperature", 25)

        self.assertEqual(state.get_context_value("weather"), "sunny")
        self.assertEqual(state.get_context_value("temperature"), 25)
        self.assertIsNone(state.get_context_value("missing"))
        self.assertEqual(state.get_context_value("missing", "default"), "default")

    def test_record_decision_and_result(self) -> None:
        """Decisions and results are properly tracked in state."""
        state = HybridState()
        w = self._make_wrapper("agent-x")
        state.add_sub_agent(w)

        state.record_decision("agent-x", {"type": "explore"})
        self.assertEqual(state.total_decisions, 1)
        self.assertEqual(state.last_decision["agent_id"], "agent-x")

        state.record_result({"status": "ok"}, success=True, reward=1.5)
        self.assertEqual(state.success_count, 1)
        self.assertEqual(state.failure_count, 0)
        self.assertAlmostEqual(state.total_reward, 1.5)

    def test_decision_history_trimmed(self) -> None:
        """Decision history does not exceed max_history_size."""
        state = HybridState()
        state.max_history_size = 3

        for i in range(5):
            state.record_decision(f"agent-{i}", {"step": i})

        self.assertEqual(len(state.decision_history), 3)
        self.assertEqual(state.total_decisions, 5)

    def test_activation_with_context_conditions(self) -> None:
        """Sub-agents with context conditions activate based on shared context."""
        state = HybridState()
        agent = _patch_agent(ExampleAgent(agent_id="cond-agent"))
        wrapper = SubAgentWrapper(
            agent_type="default",
            agent=agent,
            priority=5,
            activation_conditions={"mode": "analysis"},
        )
        state.add_sub_agent(wrapper)

        # Without the right context, agent should not be active
        active = state.get_active_agents()
        self.assertEqual(len(active), 0)

        # Set the context to match
        state.update_context("mode", "analysis")
        active = state.get_active_agents()
        self.assertEqual(len(active), 1)

    def test_to_dict(self) -> None:
        """HybridState serialization includes all tracked data."""
        state = HybridState()
        state.update_context("key", "val")
        state.record_decision("a1", {"action": "x"})
        state.record_result({"ok": True}, True, 0.5)

        d = state.to_dict()
        self.assertIn("context", d)
        self.assertIn("decision_history", d)
        self.assertEqual(d["total_decisions"], 1)
        self.assertAlmostEqual(d["total_reward"], 0.5)


if __name__ == "__main__":
    unittest.main()
