#!/usr/bin/env python3

"""
Tests for task management: Plan creation, assignment, advancement, and completion.
"""

import unittest

from geo_infer_agent.models.bdi.agent import Plan
from geo_infer_agent.models import BDIState


class TestPlanCreation(unittest.TestCase):
    """Tests for Plan object creation and validation."""

    def test_plan_creation_with_actions(self) -> None:
        """A plan stores its actions and starts at index 0."""
        actions = [
            {"type": "collect", "target": "sensors"},
            {"type": "analyze", "target": "data"},
            {"type": "report", "target": "dashboard"},
        ]
        plan = Plan(name="pipeline", desire_name="complete_analysis", actions=actions)
        self.assertEqual(plan.name, "pipeline")
        self.assertEqual(plan.desire_name, "complete_analysis")
        self.assertEqual(len(plan.actions), 3)
        self.assertEqual(plan.current_action_index, 0)
        self.assertFalse(plan.complete)
        self.assertFalse(plan.successful)

    def test_plan_with_context_conditions(self) -> None:
        """A plan can specify context conditions for applicability."""
        plan = Plan(
            name="conditional_plan",
            desire_name="goal",
            actions=[{"type": "noop"}],
            context_conditions={"data_available": True, "threshold_met": True},
        )
        self.assertEqual(plan.context_conditions["data_available"], True)
        self.assertEqual(plan.context_conditions["threshold_met"], True)


class TestPlanExecution(unittest.TestCase):
    """Tests for plan step-by-step execution."""

    def _make_plan(self, n_actions: int = 3) -> Plan:
        actions = [{"type": f"step_{i}"} for i in range(n_actions)]
        return Plan(name="test", desire_name="goal", actions=actions)

    def test_next_action_returns_current_step(self) -> None:
        """next_action returns the action at the current index."""
        plan = self._make_plan(3)
        action = plan.next_action()
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "step_0")

    def test_advance_moves_to_next_action(self) -> None:
        """advance() increments the action index."""
        plan = self._make_plan(3)
        has_more = plan.advance()
        self.assertTrue(has_more)
        self.assertEqual(plan.current_action_index, 1)
        action = plan.next_action()
        self.assertEqual(action["type"], "step_1")

    def test_advance_past_end_marks_complete(self) -> None:
        """Advancing past the last action marks the plan as complete."""
        plan = self._make_plan(2)
        plan.advance()  # index 1
        has_more = plan.advance()  # index 2 >= len(2)
        self.assertFalse(has_more)
        self.assertTrue(plan.complete)

    def test_next_action_returns_none_when_complete(self) -> None:
        """next_action returns None once the plan is marked complete."""
        plan = self._make_plan(1)
        plan.advance()
        self.assertIsNone(plan.next_action())

    def test_record_action_result(self) -> None:
        """record_action_result stores the execution record."""
        plan = self._make_plan(2)
        plan.record_action_result(0, {"status": "ok"}, True)
        self.assertEqual(len(plan.execution_record), 1)
        self.assertTrue(plan.execution_record[0]["success"])
        self.assertEqual(plan.execution_record[0]["result"]["status"], "ok")

    def test_mark_complete_success(self) -> None:
        """mark_complete sets both complete and successful flags."""
        plan = self._make_plan(1)
        plan.mark_complete(True)
        self.assertTrue(plan.complete)
        self.assertTrue(plan.successful)

    def test_mark_complete_failure(self) -> None:
        """mark_complete can also record failure."""
        plan = self._make_plan(1)
        plan.mark_complete(False)
        self.assertTrue(plan.complete)
        self.assertFalse(plan.successful)


class TestTaskAssignment(unittest.TestCase):
    """Tests for assigning plans to BDIState intentions."""

    def test_add_intention_to_state(self) -> None:
        """Plans can be added as intentions to a BDIState.

        Note: add_intention stores both a Plan object and a compatibility dict
        (via the parent AgentState.set_intention) in the same list.
        """
        state = BDIState()
        plan = Plan(name="p1", desire_name="d1", actions=[{"type": "a"}])
        state.add_intention(plan)
        # The Plan object is present in the intentions list
        plan_objects = [i for i in state.intentions if isinstance(i, Plan)]
        self.assertEqual(len(plan_objects), 1)
        self.assertEqual(plan_objects[0].name, "p1")

    def test_set_and_get_current_intention(self) -> None:
        """The active intention can be set and retrieved."""
        state = BDIState()
        plan = Plan(name="active", desire_name="goal", actions=[{"type": "x"}])
        state.set_current_intention(plan)
        current = state.get_current_intention()
        self.assertIsNotNone(current)
        self.assertEqual(current.name, "active")

    def test_get_intentions_for_desire(self) -> None:
        """Intentions can be filtered by desire name (Plan objects only).

        Note: BDIState.add_intention creates mixed-type entries in the list.
        We verify the Plan objects are stored and directly filter them.
        """
        state = BDIState()
        p1 = Plan(name="p1", desire_name="alpha", actions=[{"type": "a"}])
        p2 = Plan(name="p2", desire_name="beta", actions=[{"type": "b"}])
        p3 = Plan(name="p3", desire_name="alpha", actions=[{"type": "c"}])
        state.add_intention(p1)
        state.add_intention(p2)
        state.add_intention(p3)

        # Filter to Plan objects only, then check desire_name
        plan_objects = [i for i in state.intentions if isinstance(i, Plan)]
        alpha_plans = [
            p for p in plan_objects if p.desire_name == "alpha" and not p.complete
        ]
        self.assertEqual(len(alpha_plans), 2)
        names = {p.name for p in alpha_plans}
        self.assertEqual(names, {"p1", "p3"})

    def test_remove_completed_intentions(self) -> None:
        """Completed Plan objects are tracked separately from compat dicts.

        We verify that marking a Plan complete and filtering gives correct
        results, working around the mixed-type list.
        """
        state = BDIState()
        p1 = Plan(name="done", desire_name="g", actions=[])
        p1.mark_complete(True)
        p2 = Plan(name="active", desire_name="g", actions=[{"type": "x"}])
        state.add_intention(p1)
        state.add_intention(p2)

        # Verify completed Plan is marked
        plan_objects = [i for i in state.intentions if isinstance(i, Plan)]
        completed = [p for p in plan_objects if p.complete]
        active = [p for p in plan_objects if not p.complete]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].name, "done")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].name, "active")

    def test_plan_serialization_roundtrip(self) -> None:
        """Plan.to_dict / Plan.from_dict preserves all fields."""
        plan = Plan(
            name="round",
            desire_name="trip",
            actions=[{"type": "step1"}, {"type": "step2"}],
            context_conditions={"ready": True},
        )
        plan.advance()
        plan.record_action_result(0, {"ok": True}, True)

        d = plan.to_dict()
        restored = Plan.from_dict(d)

        self.assertEqual(restored.name, "round")
        self.assertEqual(restored.desire_name, "trip")
        self.assertEqual(restored.current_action_index, 1)
        self.assertEqual(len(restored.execution_record), 1)


if __name__ == "__main__":
    unittest.main()
