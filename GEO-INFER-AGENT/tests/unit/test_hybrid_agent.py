#!/usr/bin/env python3

"""
Unit tests for the hybrid agent architecture: HybridAgent lifecycle,
decision policies, default action handlers, and state persistence.
"""

import json
import os
import tempfile
import unittest

import pytest

from geo_infer_agent.models.hybrid import HybridAgent, SubAgentWrapper


def _rule_sub_config(rule_id, action_type, action_id, facts=None):
    """Config for a rule-based sub-agent that always matches one rule."""
    return {
        "rules": [
            {
                "id": rule_id,
                "condition": facts or {},
                "action": {"action_type": action_type, "action_id": action_id},
                "priority": 1,
            }
        ],
        "initial_facts": facts or {},
    }


class TestHybridAgentConstruction(unittest.TestCase):
    """Tests for HybridAgent construction and handler registration."""

    def test_default_handlers_registered(self) -> None:
        agent = HybridAgent(agent_id="hy-1")
        for action_type in (
            "wait",
            "update_context",
            "query_agents",
            "enable_agent",
            "disable_agent",
        ):
            self.assertIn(action_type, agent.action_handlers)
        self.assertIn("sensor_data", agent.perception_handlers)

    def test_max_history_size_from_config(self) -> None:
        agent = HybridAgent(agent_id="hy-2", config={"max_history_size": 3})
        self.assertEqual(agent.state.max_history_size, 3)


class TestHybridAgentLifecycle:
    """Tests for initialize/perceive/shutdown behavior."""

    def _agent(self, config):
        return HybridAgent(agent_id="hy-life", config=config)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialize_creates_sub_agents(self) -> None:
        agent = self._agent(
            {
                "sub_agents": [
                    {
                        "type": "rule_based",
                        "id": "rb1",
                        "priority": 5,
                        "activation_conditions": {"go": True},
                        "config": _rule_sub_config("r1", "scan", "a1", {"go": True}),
                    },
                    {
                        "type": "bdi",
                        "id": "bdi1",
                        "priority": 1,
                        "config": {},
                    },
                    {"type": "bogus_type", "id": "bad"},
                    {"id": "missing-type-field"},
                ]
            }
        )
        await agent.initialize()

        unittest.TestCase().assertIn("rb1", agent.state.sub_agents)
        unittest.TestCase().assertIn("bdi1", agent.state.sub_agents)
        unittest.TestCase().assertNotIn("bad", agent.state.sub_agents)
        unittest.TestCase().assertNotIn("missing-type-field", agent.state.sub_agents)
        wrapper = agent.state.sub_agents["rb1"]
        unittest.TestCase().assertEqual(wrapper.agent_type, "rule_based")
        unittest.TestCase().assertEqual(wrapper.priority, 5)
        unittest.TestCase().assertEqual(wrapper.activation_conditions, {"go": True})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialize_loads_initial_context(self) -> None:
        agent = self._agent({"initial_context": {"region": "north", "scale": 2}})
        await agent.initialize()
        unittest.TestCase().assertEqual(
            agent.state.get_context_value("region"), "north"
        )
        unittest.TestCase().assertEqual(agent.state.get_context_value("scale"), 2)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialize_loads_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hybrid_state.json")
            with open(path, "w") as handle:
                json.dump(
                    {
                        "context": {"region": "saved"},
                        "decision_history": [{"agent_id": "old"}],
                        "total_decisions": 7,
                        "success_count": 4,
                        "failure_count": 3,
                        "total_reward": 1.5,
                    },
                    handle,
                )
            agent = self._agent({"state_path": path})
            await agent.initialize()
            unittest.TestCase().assertEqual(
                agent.state.get_context_value("region"), "saved"
            )
            unittest.TestCase().assertEqual(agent.state.total_decisions, 7)
            unittest.TestCase().assertEqual(agent.state.success_count, 4)
            unittest.TestCase().assertEqual(agent.state.failure_count, 3)
            unittest.TestCase().assertAlmostEqual(agent.state.total_reward, 1.5)

    def test_update_beliefs_updates_context(self) -> None:
        agent = self._agent({})
        agent.update_beliefs({"temperature": 21})
        unittest.TestCase().assertEqual(
            agent.state.get_context_value("temperature"), 21
        )
        unittest.TestCase().assertIn("_last_perception_time", agent.state.context)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_perceive_updates_context_and_forwards_to_active_agents(self) -> None:
        agent = self._agent(
            {
                "sensor_readings": {"temperature": 21},
                "sub_agents": [
                    {
                        "type": "rule_based",
                        "id": "rb1",
                        "activation_conditions": {"temperature": 21},
                        "config": _rule_sub_config("r1", "scan", "a1"),
                    },
                    {
                        "type": "rule_based",
                        "id": "rb2",
                        "activation_conditions": {"temperature": 99},
                        "config": _rule_sub_config("r2", "scan", "a1"),
                    },
                ],
            }
        )
        await agent.initialize()

        perceptions = await agent.perceive()
        unittest.TestCase().assertEqual(perceptions["temperature"], 21)
        unittest.TestCase().assertEqual(
            agent.state.last_perception, {"temperature": 21}
        )
        # Only the sub-agent whose activation conditions match the updated
        # context receives the forwarded perception; rb2's conditions do not
        # match, so it never sees the reading.
        unittest.TestCase().assertEqual(
            agent.state.sub_agents["rb1"].agent.last_perception,
            {"temperature": 21},
        )
        unittest.TestCase().assertEqual(
            agent.state.sub_agents["rb2"].agent.last_perception, {}
        )
        unittest.TestCase().assertEqual(
            agent.state.get_context_value("temperature"), 21
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_shutdown_saves_state_and_shuts_sub_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "final_state.json")
            agent = self._agent(
                {
                    "state_save_path": path,
                    "sub_agents": [
                        {"type": "bdi", "id": "bdi1", "config": {}},
                    ],
                }
            )
            await agent.initialize()
            await agent.shutdown()
            with open(path) as handle:
                saved = json.load(handle)
            unittest.TestCase().assertIn("bdi1", saved["sub_agents"])


class TestHybridAgentDecide:
    """Tests for decision collection and selection policies."""

    async def _make(self, extra_config=None, sub_agents=None):
        config = {
            "sub_agents": sub_agents
            if sub_agents is not None
            else [
                {
                    "type": "rule_based",
                    "id": "high",
                    "priority": 5,
                    "config": _rule_sub_config("r1", "scan", "a1", {"go": True}),
                },
                {
                    "type": "rule_based",
                    "id": "low",
                    "priority": 1,
                    "config": _rule_sub_config("r2", "move", "b2", {"go": True}),
                },
            ],
            "initial_facts": {"go": True},
        }
        config.update(extra_config or {})
        agent = HybridAgent(agent_id="hy-decide", config=config)
        await agent.initialize()
        return agent

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_active_agents_returns_default_action(self) -> None:
        default = {"action_type": "wait", "parameters": {"duration": 0}}
        agent = await self._make({"default_action": default, "sub_agents": []})
        action = await agent.decide()
        unittest.TestCase().assertEqual(action, default)
        unittest.TestCase().assertIsNot(
            action, default
        )  # a copy, not the config object

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_active_agents_without_default_returns_none(self) -> None:
        agent = await self._make({"sub_agents": []})
        unittest.TestCase().assertIsNone(await agent.decide())

    @pytest.mark.asyncio(loop_scope="function")
    async def test_priority_policy_selects_highest_priority(self) -> None:
        agent = await self._make()
        action = await agent.decide()
        unittest.TestCase().assertEqual(action["action_id"], "a1")
        unittest.TestCase().assertEqual(action["_hybrid_source"]["agent_id"], "high")
        unittest.TestCase().assertEqual(
            action["_hybrid_source"]["agent_type"], "rule_based"
        )
        unittest.TestCase().assertEqual(agent.state.total_decisions, 1)
        unittest.TestCase().assertEqual(len(agent.state.decision_history), 1)
        unittest.TestCase().assertEqual(
            agent.state.decision_history[0]["agent_id"], "high"
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_voting_policy_counts_signatures(self) -> None:
        # Two agents vote for scan:a1, one votes for move:b2.  The majority
        # wins and, among the winners, the higher priority agent's decision.
        agent = await self._make(
            {"decision_policy": "voting"},
            sub_agents=[
                {
                    "type": "rule_based",
                    "id": "v-low",
                    "priority": 1,
                    "config": _rule_sub_config("r1", "scan", "a1", {"go": True}),
                },
                {
                    "type": "rule_based",
                    "id": "v-high",
                    "priority": 9,
                    "config": _rule_sub_config("r2", "scan", "a1", {"go": True}),
                },
                {
                    "type": "rule_based",
                    "id": "v-other",
                    "priority": 5,
                    "config": _rule_sub_config("r3", "move", "b2", {"go": True}),
                },
            ],
        )
        action = await agent.decide()
        unittest.TestCase().assertEqual(action["_hybrid_source"]["agent_id"], "v-high")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_negotiation_policy_falls_back_to_priority(self) -> None:
        agent = await self._make({"decision_policy": "negotiation"})
        action = await agent.decide()
        unittest.TestCase().assertEqual(action["_hybrid_source"]["agent_id"], "high")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unknown_policy_returns_none(self) -> None:
        agent = await self._make({"decision_policy": "quantum"})
        unittest.TestCase().assertIsNone(await agent.decide())

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_records_result_and_updates_source_agent(self) -> None:
        agent = await self._make()
        decision = await agent.decide()
        result = await agent.act(decision)

        # "scan" has no handler on the hybrid agent itself → dispatch error.
        unittest.TestCase().assertEqual(result["status"], "error")
        unittest.TestCase().assertEqual(agent.state.failure_count, 1)
        unittest.TestCase().assertFalse(
            agent.state.get_context_value("last_action_success")
        )

        # A successful action records success on the shared state and the
        # source sub-agent's wrapper stats.
        agent.state.record_decision("high", {"action_type": "wait"})
        result = await agent.act({"action_type": "wait", "parameters": {"duration": 0}})
        unittest.TestCase().assertEqual(result["status"], "success")
        unittest.TestCase().assertEqual(agent.state.success_count, 1)
        unittest.TestCase().assertTrue(
            agent.state.get_context_value("last_action_success")
        )

        result = await agent.act(
            {
                "action_type": "wait",
                "parameters": {"duration": 0},
                "_hybrid_source": {"agent_id": "high", "agent_type": "rule_based"},
            }
        )
        unittest.TestCase().assertEqual(result["status"], "success")
        # Every recorded result attributed to "high" (via the hybrid state's
        # last_decision) bumps the wrapper's own stats.
        wrapper = agent.state.sub_agents["high"]
        unittest.TestCase().assertEqual(wrapper.decision_count, 3)
        unittest.TestCase().assertEqual(wrapper.successful_decision_count, 2)


class TestHybridAgentActionHandlers:
    """Tests for the hybrid agent's built-in action handlers."""

    async def _make(self, sub_agents=None):
        config = {
            "sub_agents": sub_agents
            if sub_agents is not None
            else [
                {
                    "type": "bdi",
                    "id": "sub1",
                    "priority": 1,
                    "config": {},
                }
            ]
        }
        agent = HybridAgent(agent_id="hy-actions", config=config)
        await agent.initialize()
        return agent

    @pytest.mark.asyncio(loop_scope="function")
    async def test_update_context_handler(self) -> None:
        agent = await self._make()
        result = await agent.act(
            {
                "action_type": "update_context",
                "action_id": "u1",
                "parameters": {"key": "zone", "value": "east"},
            }
        )
        unittest.TestCase().assertEqual(result["status"], "success")
        unittest.TestCase().assertEqual(agent.state.get_context_value("zone"), "east")

        result = await agent.act(
            {"action_type": "update_context", "parameters": {"key": "zone"}}
        )
        unittest.TestCase().assertEqual(result["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_agents_active(self) -> None:
        agent = await self._make()
        result = await agent.act({"action_type": "query_agents", "parameters": {}})
        unittest.TestCase().assertEqual(result["status"], "success")
        unittest.TestCase().assertEqual(
            result["active_agents"],
            [{"id": "sub1", "type": "bdi", "priority": 1}],
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_agents_all_and_unknown(self) -> None:
        agent = await self._make()
        result = await agent.act(
            {"action_type": "query_agents", "parameters": {"query_type": "all"}}
        )
        unittest.TestCase().assertEqual(result["agents"][0]["id"], "sub1")
        unittest.TestCase().assertTrue(result["agents"][0]["is_active"])

        result = await agent.act(
            {"action_type": "query_agents", "parameters": {"query_type": "bogus"}}
        )
        unittest.TestCase().assertEqual(result["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_agents_performance(self) -> None:
        agent = await self._make()
        agent.state.record_decision("sub1", {"action_type": "wait"})
        wrapper = agent.state.sub_agents["sub1"]
        wrapper.record_decision(True, 0.4)
        result = await agent.act(
            {
                "action_type": "query_agents",
                "parameters": {"query_type": "performance"},
            }
        )
        unittest.TestCase().assertEqual(
            result["performance"]["overall"]["total_decisions"], 1
        )
        unittest.TestCase().assertEqual(
            result["performance"]["agents"]["sub1"]["decision_count"], 1
        )
        unittest.TestCase().assertAlmostEqual(
            result["performance"]["agents"]["sub1"]["total_reward"], 0.4
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_enable_disable_agent(self) -> None:
        agent = await self._make()
        result = await agent.act(
            {"action_type": "disable_agent", "parameters": {"agent_id": "sub1"}}
        )
        unittest.TestCase().assertEqual(result["status"], "success")
        unittest.TestCase().assertFalse(agent.state.sub_agents["sub1"].is_active)
        unittest.TestCase().assertEqual(agent.state.get_active_agents(), [])

        result = await agent.act(
            {"action_type": "enable_agent", "parameters": {"agent_id": "sub1"}}
        )
        unittest.TestCase().assertEqual(result["status"], "success")
        unittest.TestCase().assertEqual(len(agent.state.get_active_agents()), 1)

        for handler in ("enable_agent", "disable_agent"):
            result = await agent.act({"action_type": handler, "parameters": {}})
            unittest.TestCase().assertEqual(result["status"], "error")
            result = await agent.act(
                {"action_type": handler, "parameters": {"agent_id": "ghost"}}
            )
            unittest.TestCase().assertEqual(result["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_perceptions_handler(self) -> None:
        agent = await self._make()
        agent._handle_sensor_perceptions(agent, {"readings": {"temp": 5}})
        unittest.TestCase().assertEqual(agent.state.get_context_value("sensor_temp"), 5)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_state_roundtrip_through_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cycle.json")
            agent = await self._make()
            agent.state.update_context("region", "west")
            await_state = agent._save_state(path)
            await await_state
            fresh = await self._make()
            await fresh._load_state(path)
            unittest.TestCase().assertEqual(
                fresh.state.get_context_value("region"), "west"
            )


class TestSubAgentWrapperExtras(unittest.TestCase):
    """Additional wrapper behavior tests: nested conditions and to_dict."""

    def _wrapper(self, conditions=None, priority=0):
        from geo_infer_agent.core.agent_base import ExampleAgent

        agent = ExampleAgent(agent_id="wrap-sub")
        agent.id = agent.agent_id
        return SubAgentWrapper(
            agent_type="example",
            agent=agent,
            priority=priority,
            activation_conditions=conditions,
            description="",
        )

    def test_nested_condition_matching(self) -> None:
        wrapper = self._wrapper(
            conditions={"location": {"zone": "east", "level": {"depth": 2}}}
        )
        self.assertTrue(
            wrapper.check_activation(
                {"location": {"zone": "east", "level": {"depth": 2}}}
            )
        )
        self.assertFalse(
            wrapper.check_activation(
                {"location": {"zone": "east", "level": {"depth": 3}}}
            )
        )
        self.assertFalse(wrapper.check_activation({"location": {"zone": "east"}}))
        self.assertFalse(wrapper.check_activation({}))

    def test_to_dict_round_fields(self) -> None:
        wrapper = self._wrapper(priority=3)
        wrapper.record_decision(True, 0.25)
        d = wrapper.to_dict()
        self.assertEqual(d["agent_type"], "example")
        self.assertEqual(d["priority"], 3)
        self.assertEqual(d["stats"]["decision_count"], 1)
        self.assertEqual(d["stats"]["successful_decision_count"], 1)
        self.assertIsNotNone(d["stats"]["last_activated"])
