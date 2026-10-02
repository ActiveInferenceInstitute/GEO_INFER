#!/usr/bin/env python3

"""
Unit tests for the rule-based agent architecture: Rule, RuleSet,
RuleBasedState, RuleBasedAgent lifecycle, action handlers, persistence.
"""

import json
import os
import tempfile
import unittest

import pytest

from geo_infer_agent.models.rule_based import (
    Rule,
    RuleBasedAgent,
    RuleBasedState,
    RuleSet,
)


class TestRule(unittest.TestCase):
    """Tests for Rule condition evaluation and serialization."""

    def test_dict_condition(self) -> None:
        rule = Rule(
            rule_id="r1", condition={"zone": "east"}, action={"action_type": "scan"}
        )
        self.assertTrue(rule.matches({"zone": "east"}))
        self.assertFalse(rule.matches({"zone": "west"}))
        self.assertFalse(rule.matches({}))
        self.assertEqual(rule.match_count, 1)
        self.assertIsNotNone(rule.last_matched)

    def test_nested_dict_condition(self) -> None:
        rule = Rule(
            rule_id="r2",
            condition={"location": {"zone": "east", "detail": {"depth": 2}}},
            action={"action_type": "scan"},
        )
        self.assertTrue(
            rule.matches({"location": {"zone": "east", "detail": {"depth": 2}}})
        )
        self.assertFalse(
            rule.matches({"location": {"zone": "east", "detail": {"depth": 3}}})
        )
        self.assertFalse(rule.matches({"location": {"zone": "east"}}))

    def test_callable_condition(self) -> None:
        rule = Rule(rule_id="r3", condition=lambda state: state.get("hot"), action={})
        self.assertTrue(rule.matches({"hot": True}))
        self.assertFalse(rule.matches({"hot": False}))

        # A raising condition is treated as non-matching, not propagated.
        def boom(_state):
            raise RuntimeError("boom")

        failing = Rule(rule_id="r4", condition=boom, action={})
        self.assertFalse(failing.matches({}))

    def test_pattern_condition(self) -> None:
        rule = Rule(rule_id="r5", condition=r"temp=\d+", action={})
        self.assertTrue(rule.matches({"state_string": "temp=25 ok"}))
        self.assertFalse(rule.matches({"state_string": "no digits"}))
        self.assertFalse(rule.matches({"state_string": 42}))
        self.assertFalse(rule.matches({}))

    def test_invalid_pattern_never_matches(self) -> None:
        rule = Rule(rule_id="r6", condition="([unclosed", action={})
        self.assertFalse(rule.matches({"state_string": "anything"}))

    def test_disabled_rule_never_matches(self) -> None:
        rule = Rule(rule_id="r7", condition={"a": 1}, action={}, enabled=False)
        self.assertFalse(rule.matches({"a": 1}))

    def test_roundtrip_dict_condition(self) -> None:
        rule = Rule(
            rule_id="r8", condition={"a": 1}, action={"action_type": "x"}, priority=3
        )
        rule.matches({"a": 1})
        restored = Rule.from_dict(rule.to_dict())
        self.assertEqual(restored.id, "r8")
        self.assertEqual(restored.condition, {"a": 1})
        self.assertEqual(restored.priority, 3)
        self.assertEqual(restored.match_count, 1)

    def test_roundtrip_pattern_and_function_conditions(self) -> None:
        pattern_rule = Rule(rule_id="r9", condition="abc", action={})
        restored_pattern = Rule.from_dict(pattern_rule.to_dict())
        self.assertEqual(restored_pattern.condition, "abc")

        fn_rule = Rule(rule_id="r10", condition=lambda s: True, action={})
        restored_fn = Rule.from_dict(fn_rule.to_dict())
        # Function conditions cannot be serialized; the restored rule is a
        # documented always-false placeholder.
        self.assertFalse(restored_fn.matches({}))


class TestRuleSet(unittest.TestCase):
    """Tests for RuleSet management."""

    def test_add_get_remove_enable_disable(self) -> None:
        rule_set = RuleSet()
        rule = Rule(rule_id="a", condition={"x": 1}, action={})
        rule_set.add_rule(rule)

        self.assertIs(rule_set.get_rule("a"), rule)
        self.assertIsNone(rule_set.get_rule("missing"))
        self.assertTrue(rule_set.disable_rule("a"))
        self.assertFalse(rule.matches({"x": 1}))
        self.assertTrue(rule_set.enable_rule("a"))
        self.assertTrue(rule.matches({"x": 1}))

        self.assertFalse(rule_set.disable_rule("missing"))
        self.assertFalse(rule_set.enable_rule("missing"))
        self.assertTrue(rule_set.remove_rule("a"))
        self.assertFalse(rule_set.remove_rule("a"))

    def test_find_matching_rules_sorted_by_priority(self) -> None:
        rule_set = RuleSet()
        low = Rule(rule_id="low", condition={"go": True}, action={}, priority=1)
        high = Rule(rule_id="high", condition={"go": True}, action={}, priority=9)
        off = Rule(
            rule_id="off", condition={"go": True}, action={}, priority=5, enabled=False
        )
        other = Rule(rule_id="other", condition={"stop": True}, action={}, priority=99)
        for rule in (low, high, off, other):
            rule_set.add_rule(rule)
        matching = rule_set.find_matching_rules({"go": True})
        self.assertEqual([r.id for r in matching], ["high", "low"])

    def test_roundtrip(self) -> None:
        rule_set = RuleSet()
        rule_set.add_rule(
            Rule(rule_id="a", condition={"x": 1}, action={"action_type": "y"})
        )
        restored = RuleSet.from_dict(rule_set.to_dict())
        self.assertIn("a", restored.rules)
        self.assertEqual(restored.rules["a"].condition, {"x": 1})


class TestRuleBasedState(unittest.TestCase):
    """Tests for RuleBasedState facts, history, and serialization."""

    def test_fact_management(self) -> None:
        state = RuleBasedState()
        state.update_fact("temp", 20)
        self.assertEqual(state.get_fact("temp"), 20)
        self.assertEqual(state.get_fact("missing", "dflt"), "dflt")
        self.assertTrue(state.remove_fact("temp"))
        self.assertFalse(state.remove_fact("temp"))

    def test_execution_history_trimmed(self) -> None:
        state = RuleBasedState()
        state.max_history_size = 3
        for i in range(5):
            state.record_execution("r", {"n": i}, {"ok": True})
        self.assertEqual(len(state.execution_history), 3)
        self.assertEqual(state.execution_history[0]["action"]["n"], 2)

    def test_find_matching_rules_uses_facts(self) -> None:
        state = RuleBasedState()
        state.add_rule(Rule(rule_id="r", condition={"go": True}, action={}))
        self.assertEqual(state.find_matching_rules(), [])
        state.update_fact("go", True)
        self.assertEqual([r.id for r in state.find_matching_rules()], ["r"])

    def test_roundtrip(self) -> None:
        state = RuleBasedState()
        state.update_fact("a", 1)
        state.record_execution("r1", {"x": 1}, {"ok": False})
        restored = RuleBasedState.from_dict(state.to_dict())
        self.assertEqual(restored.get_fact("a"), 1)
        self.assertEqual(restored.execution_history[0]["rule_id"], "r1")


class TestRuleBasedAgentLifecycle:
    """Tests for RuleBasedAgent construction, init, and perception."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_handlers_registered_and_config_applied(self) -> None:
        agent = RuleBasedAgent(
            agent_id="rb-1",
            config={"max_history_size": 7, "initial_facts": {"boot": True}},
        )
        for handler in (
            "wait",
            "update_fact",
            "remove_fact",
            "add_rule",
            "remove_rule",
            "enable_rule",
            "disable_rule",
            "query_facts",
        ):
            unittest.TestCase().assertIn(handler, agent.action_handlers)
        unittest.TestCase().assertIn("sensor_data", agent.perception_handlers)
        unittest.TestCase().assertEqual(agent.state.max_history_size, 7)

        await agent.initialize()
        unittest.TestCase().assertTrue(agent.state.get_fact("boot"))

    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialize_loads_rules_from_config(self) -> None:
        agent = RuleBasedAgent(
            agent_id="rb-2",
            config={
                "rules": [
                    {
                        "id": "r1",
                        "condition": {"a": 1},
                        "action": {"action_type": "scan"},
                    },
                    {"action": {"action_type": "x"}},  # missing id/condition
                    {"id": "r2", "action": {"action_type": "x"}},  # missing condition
                ]
            },
        )
        await agent.initialize()
        unittest.TestCase().assertIn("r1", agent.state.rule_set.rules)
        unittest.TestCase().assertNotIn("r2", agent.state.rule_set.rules)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialize_loads_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rb_state.json")
            saved_agent = RuleBasedAgent(agent_id="rb-save")
            saved_agent.state.update_fact("carry", "me")
            saved_agent._save_state(path)

            agent = RuleBasedAgent(agent_id="rb-load", config={"state_path": path})
            await agent.initialize()
            unittest.TestCase().assertEqual(agent.state.get_fact("carry"), "me")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_update_beliefs_and_perceive(self) -> None:
        agent = RuleBasedAgent(agent_id="rb-3")
        await agent.initialize()
        agent.update_beliefs({"temp": 21, "_internal": "skip"})
        unittest.TestCase().assertEqual(agent.state.get_fact("temp"), 21)
        unittest.TestCase().assertNotIn("_internal", agent.state.facts)
        unittest.TestCase().assertIn("_last_perception_time", agent.state.facts)

        agent.config["sensor_readings"] = {"temp": 22}
        perceptions = await agent.perceive()
        unittest.TestCase().assertEqual(perceptions["temp"], 22)
        unittest.TestCase().assertEqual(agent.state.get_fact("temp"), 22)


class TestRuleBasedAgentDecideAct:
    """Tests for the decide/act cycle."""

    async def _make(self, config):
        agent = RuleBasedAgent(agent_id="rb-cycle", config=config)
        await agent.initialize()
        return agent

    @pytest.mark.asyncio(loop_scope="function")
    async def test_matching_rule_action_carries_rule_id(self) -> None:
        agent = await self._make(
            {
                "rules": [
                    {
                        "id": "hot-rule",
                        "condition": {"temp": 30},
                        "action": {"action_type": "scan", "action_id": "s1"},
                        "priority": 5,
                    }
                ],
                "initial_facts": {"temp": 30},
            }
        )
        action = await agent.decide()
        unittest.TestCase().assertEqual(action["action_id"], "s1")
        unittest.TestCase().assertEqual(action["_rule_id"], "hot-rule")

        result = await agent.act(action)
        # No handler for "scan" → dispatch error, but still recorded.
        unittest.TestCase().assertEqual(result["status"], "error")
        unittest.TestCase().assertEqual(len(agent.state.execution_history), 1)
        unittest.TestCase().assertEqual(
            agent.state.execution_history[0]["rule_id"], "hot-rule"
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_match_uses_default_action_and_copies_it(self) -> None:
        default = {"action_type": "wait", "parameters": {"duration": 0}}
        agent = await self._make(
            {"default_action": default, "initial_facts": {"cold": True}}
        )
        action = await agent.decide()
        unittest.TestCase().assertEqual(action, default)
        unittest.TestCase().assertIsNot(action, default)
        unittest.TestCase().assertNotIn("_rule_id", action)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_match_without_default_returns_none(self) -> None:
        agent = await self._make({})
        unittest.TestCase().assertIsNone(await agent.decide())

    @pytest.mark.asyncio(loop_scope="function")
    async def test_action_result_facts_update_state(self) -> None:
        agent = await self._make({})
        result = await agent.act(
            {
                "action_type": "update_fact",
                "parameters": {"key": "zone", "value": "east"},
            }
        )
        unittest.TestCase().assertEqual(result["status"], "success")
        unittest.TestCase().assertEqual(agent.state.get_fact("zone"), "east")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_shutdown_saves_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "final.json")
            agent = await self._make(
                {"state_save_path": path, "initial_facts": {"k": "v"}}
            )
            await agent.shutdown()
            with open(path) as handle:
                saved = json.load(handle)
            unittest.TestCase().assertEqual(saved["facts"]["k"], "v")


class TestRuleBasedAgentActionHandlers:
    """Tests for the rule agent's built-in action handlers."""

    async def _make(self):
        agent = RuleBasedAgent(
            agent_id="rb-handlers",
            config={
                "rules": [
                    {
                        "id": "existing",
                        "condition": {"a": 1},
                        "action": {"action_type": "x"},
                    }
                ],
                "initial_facts": {"temp": 20, "zone": "west"},
            },
        )
        await agent.initialize()
        return agent

    @pytest.mark.asyncio(loop_scope="function")
    async def test_update_fact_handler_validation(self) -> None:
        agent = await self._make()
        ok = await agent.act(
            {
                "action_type": "update_fact",
                "parameters": {"key": "temp", "value": 25},
            }
        )
        unittest.TestCase().assertEqual(ok["status"], "success")
        unittest.TestCase().assertEqual(agent.state.get_fact("temp"), 25)

        bad = await agent.act(
            {"action_type": "update_fact", "parameters": {"key": "t"}}
        )
        unittest.TestCase().assertEqual(bad["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_remove_fact_handler(self) -> None:
        agent = await self._make()
        ok = await agent.act(
            {"action_type": "remove_fact", "parameters": {"key": "temp"}}
        )
        unittest.TestCase().assertEqual(ok["status"], "success")
        unittest.TestCase().assertFalse(agent.state.remove_fact("temp"))

        missing = await agent.act(
            {"action_type": "remove_fact", "parameters": {"key": "ghost"}}
        )
        unittest.TestCase().assertEqual(missing["status"], "warning")

        bad = await agent.act({"action_type": "remove_fact", "parameters": {}})
        unittest.TestCase().assertEqual(bad["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_add_rule_handler(self) -> None:
        agent = await self._make()
        ok = await agent.act(
            {
                "action_type": "add_rule",
                "parameters": {
                    "id": "new-rule",
                    "condition": {"b": 2},
                    "action": {"action_type": "scan"},
                    "priority": 4,
                },
            }
        )
        unittest.TestCase().assertEqual(ok["status"], "success")
        unittest.TestCase().assertEqual(
            agent.state.rule_set.rules["new-rule"].priority, 4
        )

        bad = await agent.act(
            {"action_type": "add_rule", "parameters": {"id": "incomplete"}}
        )
        unittest.TestCase().assertEqual(bad["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_remove_enable_disable_rule_handlers(self) -> None:
        agent = await self._make()
        ok = await agent.act(
            {"action_type": "disable_rule", "parameters": {"id": "existing"}}
        )
        unittest.TestCase().assertEqual(ok["status"], "success")
        unittest.TestCase().assertFalse(agent.state.rule_set.rules["existing"].enabled)

        ok = await agent.act(
            {"action_type": "enable_rule", "parameters": {"id": "existing"}}
        )
        unittest.TestCase().assertEqual(ok["status"], "success")
        unittest.TestCase().assertTrue(agent.state.rule_set.rules["existing"].enabled)

        ok = await agent.act(
            {"action_type": "remove_rule", "parameters": {"id": "existing"}}
        )
        unittest.TestCase().assertEqual(ok["status"], "success")
        unittest.TestCase().assertNotIn("existing", agent.state.rule_set.rules)

        for handler in ("remove_rule", "enable_rule", "disable_rule"):
            missing = await agent.act(
                {"action_type": handler, "parameters": {"id": "ghost"}}
            )
            unittest.TestCase().assertEqual(missing["status"], "warning")
            bad = await agent.act({"action_type": handler, "parameters": {}})
            unittest.TestCase().assertEqual(bad["status"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_facts_handler(self) -> None:
        agent = await self._make()
        some = await agent.act(
            {
                "action_type": "query_facts",
                "parameters": {"keys": ["temp", "ghost"]},
            }
        )
        unittest.TestCase().assertEqual(some["facts"], {"temp": 20})

        everything = await agent.act({"action_type": "query_facts", "parameters": {}})
        unittest.TestCase().assertEqual(everything["facts"]["zone"], "west")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_perceptions_handler(self) -> None:
        agent = await self._make()
        agent._handle_sensor_perceptions(agent, {"readings": {"temp": 30}})
        unittest.TestCase().assertEqual(agent.state.get_fact("sensor_temp"), 30)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_load_state_corrupt_file_keeps_state(self) -> None:
        agent = await self._make()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write("not json")
            path = handle.name
        try:
            agent._load_state(path)
        finally:
            os.unlink(path)
        unittest.TestCase().assertEqual(agent.state.get_fact("temp"), 20)
