#!/usr/bin/env python3

"""
Tests for plan generation, selection, and execution in BDI agents.
"""

import asyncio
import unittest

from geo_infer_agent.models import BDIAgent


class TestPlanLibrary(unittest.TestCase):
    """Tests for the BDI agent plan library and plan selection."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_agent(self, plans=None, desires=None, beliefs=None):
        """Create and initialize a BDIAgent with given config."""
        config = {
            "plans": plans or [],
            "initial_desires": desires or [],
            "initial_beliefs": beliefs or {},
        }
        agent = BDIAgent(agent_id="plan-agent", config=config)
        self._run(agent.initialize())
        return agent

    def test_plans_loaded_from_config(self) -> None:
        """Plans defined in config are loaded into the plan library."""
        plans = [
            {
                "name": "collect_plan",
                "desire_name": "collect_data",
                "actions": [{"type": "log", "message": "collecting", "level": "info"}],
            }
        ]
        agent = self._make_agent(plans=plans)
        self.assertIn("collect_plan", agent.plan_library)

    def test_find_plan_for_desire_returns_matching_plan(self) -> None:
        """_find_plan_for_desire returns a Plan that addresses the desire."""
        plans = [
            {
                "name": "monitor_plan",
                "desire_name": "monitor",
                "actions": [{"type": "log", "message": "monitoring", "level": "info"}],
            }
        ]
        desires = [
            {"name": "monitor", "description": "Monitor sources", "priority": 0.8}
        ]
        agent = self._make_agent(plans=plans, desires=desires)

        plan = agent._find_plan_for_desire("monitor")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.desire_name, "monitor")

    def test_find_plan_returns_none_for_unknown_desire(self) -> None:
        """_find_plan_for_desire returns None if no plan matches."""
        agent = self._make_agent()
        plan = agent._find_plan_for_desire("nonexistent")
        self.assertIsNone(plan)

    def test_context_conditions_checked_before_plan_adoption(self) -> None:
        """Plan with unmet context_conditions is not selected."""
        plans = [
            {
                "name": "conditional_plan",
                "desire_name": "process",
                "actions": [{"type": "log", "message": "processing", "level": "info"}],
                "context_conditions": {"data_ready": True},
            }
        ]
        desires = [{"name": "process", "description": "Process data", "priority": 0.7}]

        # No belief for data_ready, so condition is unmet
        agent = self._make_agent(plans=plans, desires=desires)
        plan = agent._find_plan_for_desire("process")
        self.assertIsNone(plan)

        # Now set the belief so condition is met
        agent.state.update_belief("data_ready", True)
        plan = agent._find_plan_for_desire("process")
        self.assertIsNotNone(plan)


class TestPlanExecution(unittest.TestCase):
    """Tests for plan execution through the decide/act cycle."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_decide_selects_plan_for_highest_priority_desire(self) -> None:
        """decide() picks a plan for the highest-priority unachieved desire."""
        plans = [
            {
                "name": "low_plan",
                "desire_name": "low_goal",
                "actions": [{"type": "log", "message": "low", "level": "info"}],
            },
            {
                "name": "high_plan",
                "desire_name": "high_goal",
                "actions": [{"type": "log", "message": "high", "level": "info"}],
            },
        ]
        desires = [
            {"name": "low_goal", "description": "Low", "priority": 0.3},
            {"name": "high_goal", "description": "High", "priority": 0.9},
        ]
        config = {"plans": plans, "initial_desires": desires}
        agent = BDIAgent(agent_id="priority-agent", config=config)
        self._run(agent.initialize())

        action = self._run(agent.decide())
        self.assertIsNotNone(action)
        # The current intention should be for the high-priority desire
        current = agent.state.get_current_intention()
        self.assertIsNotNone(current)
        self.assertEqual(current.desire_name, "high_goal")

    def test_decide_skips_achieved_desires(self) -> None:
        """decide() skips desires that are already achieved."""
        plans = [
            {
                "name": "plan_a",
                "desire_name": "goal_a",
                "actions": [{"type": "log", "message": "a", "level": "info"}],
            },
            {
                "name": "plan_b",
                "desire_name": "goal_b",
                "actions": [{"type": "log", "message": "b", "level": "info"}],
            },
        ]
        desires = [
            {"name": "goal_a", "description": "A", "priority": 0.9},
            {"name": "goal_b", "description": "B", "priority": 0.5},
        ]
        config = {"plans": plans, "initial_desires": desires}
        agent = BDIAgent(agent_id="skip-agent", config=config)
        self._run(agent.initialize())

        # Mark goal_a as achieved
        desire_a = agent.state.get_desire("goal_a")
        desire_a.set_achieved(True)

        action = self._run(agent.decide())
        self.assertIsNotNone(action)
        current = agent.state.get_current_intention()
        self.assertEqual(current.desire_name, "goal_b")

    def test_decide_returns_none_when_no_desires(self) -> None:
        """decide() returns None when there are no desires."""
        agent = BDIAgent(agent_id="empty-agent", config={})
        self._run(agent.initialize())
        action = self._run(agent.decide())
        self.assertIsNone(action)

    def test_act_executes_log_action(self) -> None:
        """act() executes a log action through the registered handler."""
        agent = BDIAgent(agent_id="act-agent", config={})
        self._run(agent.initialize())
        action = {"type": "log", "message": "test message", "level": "info"}
        result = self._run(agent.act(action))
        self.assertTrue(result.get("success", False))

    def test_act_returns_error_for_unknown_type(self) -> None:
        """act() returns error for an unrecognized action type."""
        agent = BDIAgent(agent_id="err-agent", config={})
        self._run(agent.initialize())
        action = {"type": "unknown_action"}
        result = self._run(agent.act(action))
        self.assertFalse(result.get("success", True))


if __name__ == "__main__":
    unittest.main()
