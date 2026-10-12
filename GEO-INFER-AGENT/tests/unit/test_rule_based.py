#!/usr/bin/env python3

"""
Tests for the rule-based agent: Rule matching, RuleSet, RuleBasedState.
"""

import unittest

from geo_infer_agent.models.rule_based import Rule, RuleBasedState, RuleSet


class TestRuleMatching(unittest.TestCase):
    """Tests for the Rule condition matching logic."""

    def test_dict_condition_matches(self) -> None:
        """A rule with dict condition matches when all keys match."""
        rule = Rule(
            rule_id="r1",
            condition={"temperature": "high", "humidity": "low"},
            action={"type": "cool_down"},
        )
        state = {"temperature": "high", "humidity": "low", "wind": "calm"}
        self.assertTrue(rule.matches(state))

    def test_dict_condition_does_not_match(self) -> None:
        """A rule does not match when state values differ."""
        rule = Rule(
            rule_id="r2",
            condition={"temperature": "high"},
            action={"type": "cool_down"},
        )
        state = {"temperature": "low"}
        self.assertFalse(rule.matches(state))

    def test_dict_condition_missing_key(self) -> None:
        """A rule does not match when state is missing required keys."""
        rule = Rule(
            rule_id="r3",
            condition={"pressure": "high"},
            action={"type": "alert"},
        )
        state = {"temperature": "normal"}
        self.assertFalse(rule.matches(state))

    def test_callable_condition(self) -> None:
        """A rule with a callable condition invokes the function."""
        rule = Rule(
            rule_id="r4",
            condition=lambda s: s.get("value", 0) > 50,
            action={"type": "escalate"},
        )
        self.assertTrue(rule.matches({"value": 75}))
        self.assertFalse(rule.matches({"value": 25}))

    def test_regex_condition(self) -> None:
        """A rule with a string condition uses regex matching against state_string."""
        rule = Rule(
            rule_id="r5",
            condition=r"ERROR:\d+",
            action={"type": "handle_error"},
        )
        self.assertTrue(rule.matches({"state_string": "ERROR:404 not found"}))
        self.assertFalse(rule.matches({"state_string": "OK:200 success"}))

    def test_disabled_rule_does_not_match(self) -> None:
        """A disabled rule never matches regardless of state."""
        rule = Rule(
            rule_id="r6",
            condition={"always": True},
            action={"type": "noop"},
            enabled=False,
        )
        self.assertFalse(rule.matches({"always": True}))

    def test_match_count_incremented(self) -> None:
        """Successful matches increment the match counter."""
        rule = Rule(rule_id="r7", condition={"x": 1}, action={"type": "y"})
        rule.matches({"x": 1})
        rule.matches({"x": 1})
        rule.matches({"x": 2})
        self.assertEqual(rule.match_count, 2)
        self.assertIsNotNone(rule.last_matched)

    def test_nested_dict_condition(self) -> None:
        """Nested dict conditions are matched recursively."""
        rule = Rule(
            rule_id="r8",
            condition={"sensor": {"type": "temp", "status": "active"}},
            action={"type": "read_sensor"},
        )
        state = {"sensor": {"type": "temp", "status": "active", "id": "s1"}}
        self.assertTrue(rule.matches(state))

        state_bad = {"sensor": {"type": "temp", "status": "offline"}}
        self.assertFalse(rule.matches(state_bad))


class TestRuleSet(unittest.TestCase):
    """Tests for the RuleSet collection management."""

    def _make_rule(self, rule_id: str, condition: dict, priority: int = 0) -> Rule:
        return Rule(
            rule_id=rule_id,
            condition=condition,
            action={"type": f"action_{rule_id}"},
            priority=priority,
        )

    def test_add_and_get_rule(self) -> None:
        """Rules can be added to and retrieved from a RuleSet."""
        rs = RuleSet()
        rule = self._make_rule("r1", {"a": 1})
        rs.add_rule(rule)
        self.assertIs(rs.get_rule("r1"), rule)

    def test_remove_rule(self) -> None:
        """Rules can be removed from the set."""
        rs = RuleSet()
        rs.add_rule(self._make_rule("r1", {"a": 1}))
        self.assertTrue(rs.remove_rule("r1"))
        self.assertIsNone(rs.get_rule("r1"))
        self.assertFalse(rs.remove_rule("r1"))

    def test_enable_disable_rule(self) -> None:
        """Rules can be enabled and disabled."""
        rs = RuleSet()
        rule = self._make_rule("r1", {"a": 1})
        rs.add_rule(rule)

        rs.disable_rule("r1")
        self.assertFalse(rule.enabled)

        rs.enable_rule("r1")
        self.assertTrue(rule.enabled)

    def test_find_matching_rules_sorted_by_priority(self) -> None:
        """find_matching_rules returns matches sorted by priority descending."""
        rs = RuleSet()
        rs.add_rule(self._make_rule("low", {"x": True}, priority=1))
        rs.add_rule(self._make_rule("high", {"x": True}, priority=10))
        rs.add_rule(self._make_rule("mid", {"x": True}, priority=5))
        rs.add_rule(self._make_rule("no_match", {"y": True}, priority=100))

        matching = rs.find_matching_rules({"x": True})
        self.assertEqual(len(matching), 3)
        priorities = [r.priority for r in matching]
        self.assertEqual(priorities, [10, 5, 1])

    def test_to_dict_and_from_dict(self) -> None:
        """RuleSet serialization roundtrip preserves rules."""
        rs = RuleSet()
        rs.add_rule(self._make_rule("r1", {"k": "v"}, priority=3))
        rs.add_rule(self._make_rule("r2", {"k2": "v2"}, priority=7))

        d = rs.to_dict()
        restored = RuleSet.from_dict(d)
        self.assertIsNotNone(restored.get_rule("r1"))
        self.assertIsNotNone(restored.get_rule("r2"))
        self.assertEqual(restored.get_rule("r2").priority, 7)


class TestRuleBasedState(unittest.TestCase):
    """Tests for RuleBasedState fact management and rule evaluation."""

    def test_update_and_get_fact(self) -> None:
        """Facts can be set and retrieved."""
        state = RuleBasedState()
        state.update_fact("sensor_temp", 25.5)
        self.assertAlmostEqual(state.get_fact("sensor_temp"), 25.5)

    def test_get_fact_default(self) -> None:
        """get_fact returns default for missing keys."""
        state = RuleBasedState()
        self.assertIsNone(state.get_fact("missing"))
        self.assertEqual(state.get_fact("missing", 0), 0)

    def test_remove_fact(self) -> None:
        """Facts can be removed."""
        state = RuleBasedState()
        state.update_fact("key", "val")
        self.assertTrue(state.remove_fact("key"))
        self.assertFalse(state.remove_fact("key"))

    def test_find_matching_rules_against_facts(self) -> None:
        """Rules are matched against the current facts."""
        state = RuleBasedState()
        state.add_rule(
            Rule("r1", {"alert_level": "critical"}, {"type": "evacuate"}, priority=10)
        )
        state.add_rule(
            Rule("r2", {"alert_level": "warning"}, {"type": "notify"}, priority=5)
        )

        state.update_fact("alert_level", "critical")
        matching = state.find_matching_rules()
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].id, "r1")

    def test_record_execution_and_history_limit(self) -> None:
        """Execution history is recorded and trimmed to max size."""
        state = RuleBasedState()
        state.max_history_size = 3

        for i in range(5):
            state.record_execution(f"r{i}", {"type": "test"}, {"ok": True})

        self.assertEqual(len(state.execution_history), 3)
        # Oldest entries should be evicted
        self.assertEqual(state.execution_history[0]["rule_id"], "r2")


if __name__ == "__main__":
    unittest.main()
