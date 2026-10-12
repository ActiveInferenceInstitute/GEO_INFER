#!/usr/bin/env python3

"""
Tests for plan generation, selection, and execution in BDI agents.
"""

import unittest

import pytest

from geo_infer_agent.models import BDIAgent


class TestPlanLibrary:
    """Tests for the BDI agent plan library and plan selection."""

    async def _make_agent(self, plans=None, desires=None, beliefs=None):
        """Create and initialize a BDIAgent with given config."""
        config = {
            "plans": plans or [],
            "initial_desires": desires or [],
            "initial_beliefs": beliefs or {},
        }
        agent = BDIAgent(agent_id="plan-agent", config=config)
        await agent.initialize()
        return agent

    @pytest.mark.asyncio(loop_scope="function")
    async def test_plans_loaded_from_config(self) -> None:
        """Plans defined in config are loaded into the plan library."""
        plans = [
            {
                "name": "collect_plan",
                "desire_name": "collect_data",
                "actions": [{"type": "log", "message": "collecting", "level": "info"}],
            }
        ]
        agent = await self._make_agent(plans=plans)
        unittest.TestCase().assertIn("collect_plan", agent.plan_library)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_find_plan_for_desire_returns_matching_plan(self) -> None:
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
        agent = await self._make_agent(plans=plans, desires=desires)

        plan = agent._find_plan_for_desire("monitor")
        unittest.TestCase().assertIsNotNone(plan)
        unittest.TestCase().assertEqual(plan.desire_name, "monitor")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_find_plan_returns_none_for_unknown_desire(self) -> None:
        """_find_plan_for_desire returns None if no plan matches."""
        agent = await self._make_agent()
        plan = agent._find_plan_for_desire("nonexistent")
        unittest.TestCase().assertIsNone(plan)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_context_conditions_checked_before_plan_adoption(self) -> None:
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
        agent = await self._make_agent(plans=plans, desires=desires)
        plan = agent._find_plan_for_desire("process")
        unittest.TestCase().assertIsNone(plan)

        # Now set the belief so condition is met
        agent.state.update_belief("data_ready", True)
        plan = agent._find_plan_for_desire("process")
        unittest.TestCase().assertIsNotNone(plan)


class TestPlanExecution:
    """Tests for plan execution through the decide/act cycle."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_decide_selects_plan_for_highest_priority_desire(self) -> None:
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
        await agent.initialize()

        action = await agent.decide()
        unittest.TestCase().assertIsNotNone(action)
        # The current intention should be for the high-priority desire
        current = agent.state.get_current_intention()
        unittest.TestCase().assertIsNotNone(current)
        unittest.TestCase().assertEqual(current.desire_name, "high_goal")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_decide_skips_achieved_desires(self) -> None:
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
        await agent.initialize()

        # Mark goal_a as achieved
        desire_a = agent.state.get_desire("goal_a")
        desire_a.set_achieved(True)

        action = await agent.decide()
        unittest.TestCase().assertIsNotNone(action)
        current = agent.state.get_current_intention()
        unittest.TestCase().assertEqual(current.desire_name, "goal_b")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_decide_returns_none_when_no_desires(self) -> None:
        """decide() returns None when there are no desires."""
        agent = BDIAgent(agent_id="empty-agent", config={})
        await agent.initialize()
        action = await agent.decide()
        unittest.TestCase().assertIsNone(action)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_executes_log_action(self) -> None:
        """act() executes a log action through the registered handler."""
        agent = BDIAgent(agent_id="act-agent", config={})
        await agent.initialize()
        action = {"type": "log", "message": "test message", "level": "info"}
        result = await agent.act(action)
        unittest.TestCase().assertTrue(result.get("success", False))

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_returns_error_for_unknown_type(self) -> None:
        """act() returns error for an unrecognized action type."""
        agent = BDIAgent(agent_id="err-agent", config={})
        await agent.initialize()
        action = {"type": "unknown_action"}
        result = await agent.act(action)
        unittest.TestCase().assertFalse(result.get("success", True))
