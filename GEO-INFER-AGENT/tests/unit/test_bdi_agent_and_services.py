#!/usr/bin/env python3

"""
Unit tests for BDI agent internals (belief merging, intention lifecycle,
serialization, decision helpers) and the messaging/telemetry service
lifecycle paths not exercised elsewhere.
"""

import asyncio
import tempfile
import os
import unittest
from datetime import datetime, timedelta
from unittest import mock

from geo_infer_agent.api.messaging import Message, messaging_service
from geo_infer_agent.api.telemetry import (
    telemetry_service,
)
from geo_infer_agent.models.bdi import BDIAgent, BDIState
from geo_infer_agent.models.bdi.belief import Belief
from geo_infer_agent.models.bdi.plan import Plan


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestBDIStateInternals(unittest.TestCase):
    """Tests for BDIState belief merging and intention bookkeeping."""

    def test_add_belief_merges_existing(self) -> None:
        state = BDIState()
        state.add_belief(Belief(name="temp", value=20, confidence=0.9))
        state.add_belief(Belief(name="temp", value=25, confidence=0.8))
        belief = state.get_belief("temp")
        # The merge updates the existing Belief in place.
        self.assertEqual(belief.value, 25)
        self.assertAlmostEqual(belief.confidence, 0.8)
        # The merge is recorded in working memory.
        kinds = [m["type"] for m in state.memory]
        self.assertIn("belief_added", kinds)
        self.assertIn("belief_updated", kinds)

    def test_intention_helpers(self) -> None:
        state = BDIState()
        plan = Plan(
            name="p1", desire_name="d1", actions=[{"type": "log", "message": "x"}]
        )
        state.set_current_intention(plan)
        self.assertIs(state.get_current_intention(), plan)
        state.set_current_intention(None)
        self.assertIsNone(state.get_current_intention())

        done = Plan(
            name="p2", desire_name="d2", actions=[{"type": "wait", "duration": 0}]
        )
        state.intentions.append(plan)
        state.intentions.append(done)
        done.complete = True
        self.assertEqual(
            [p.name for p in state.get_intentions_for_desire("d1")], ["p1"]
        )
        self.assertEqual(state.get_intentions_for_desire("d2"), [])
        self.assertEqual(state.remove_completed_intentions(), 1)
        self.assertEqual(len(state.intentions), 1)

    def test_serialization_roundtrip(self) -> None:
        state = BDIState()
        state.add_belief(Belief(name="plain", value=5))
        state.add_belief(
            Belief(name="rich", value="x", confidence=0.5, metadata={"src": "test"})
        )
        plan = Plan(
            name="p1", desire_name="d1", actions=[{"type": "wait", "duration": 0}]
        )
        state.set_current_intention(plan)

        restored = BDIState.from_dict(state.to_dict())
        self.assertEqual(restored.get_belief("plain").value, 5)
        self.assertEqual(restored.get_belief("rich").confidence, 0.5)
        self.assertIsNotNone(restored.get_current_intention())
        self.assertEqual(restored.get_current_intention().name, "p1")

    def test_deserialization_with_raw_values(self) -> None:
        # Plain (non-dict) belief payloads fall back to a bare Belief.
        state = BDIState.from_dict(
            {
                "beliefs": {"raw": 42},
                "desires": {},
                "intentions": [],
                "current_intention": None,
            }
        )
        self.assertEqual(state.get_belief("raw").value, 42)
        self.assertIsNone(state.get_current_intention())


class TestBDIAgentAct(unittest.TestCase):
    """Tests for the BDI act/decide contract."""

    def _agent(self, config=None):
        agent = BDIAgent(agent_id="bdi-act", config=config or {})
        _run(agent.initialize())
        return agent

    def test_act_rejects_invalid_action(self) -> None:
        agent = self._agent()
        result = _run(agent.act({}))
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Invalid action")

        result = _run(agent.act({"message": "no type"}))
        self.assertFalse(result["success"])

    def test_act_unknown_action_type(self) -> None:
        agent = self._agent()
        result = _run(agent.act({"type": "teleport"}))
        self.assertFalse(result["success"])
        self.assertIn("No handler", result["error"])

    def test_act_handler_exception_returned_as_error(self) -> None:
        agent = self._agent()

        async def boom(agent, action):
            raise RuntimeError("kaboom")

        agent.action_handlers["explode"] = boom
        result = _run(agent.act({"type": "boom", "action_type": "boom"}))
        self.assertFalse(result["success"])
        self.assertIn("boom", result["error"])

    def test_act_completes_desire_when_conditions_met(self) -> None:
        agent = self._agent(
            {
                "initial_desires": [
                    {
                        "name": "inform",
                        "description": "Know the temperature",
                        "conditions": {"sensor.temp": 25},
                    }
                ],
                "plans": [
                    {
                        "name": "read",
                        "desire_name": "inform",
                        "actions": [{"type": "log", "message": "reading"}],
                    }
                ],
            }
        )
        # The desire's completion condition must hold for the desire to be
        # marked achieved once its plan finishes.
        agent.state.update_belief("sensor.temp", 25)
        action = _run(agent.decide())
        self.assertIsNotNone(action)
        result = _run(agent.act(action))
        self.assertTrue(result["success"])
        self.assertTrue(agent.state.get_desire("inform").achieved)

    def test_act_advances_intention_without_completing(self) -> None:
        agent = self._agent(
            {
                "initial_desires": [{"name": "d", "description": "multi-step"}],
                "plans": [
                    {
                        "name": "p",
                        "desire_name": "d",
                        "actions": [
                            {"type": "log", "message": "one"},
                            {"type": "log", "message": "two"},
                        ],
                    }
                ],
            }
        )
        result = _run(agent.act(_run(agent.decide())))
        self.assertTrue(result["success"])
        # Intention still mid-plan: desire not yet achieved.
        self.assertFalse(agent.state.get_desire("d").achieved)
        # Current intention is retained and next decide returns step two.
        current = agent.state.get_current_intention()
        self.assertIsNotNone(current)
        self.assertEqual(current.current_action_index, 1)

    def test_act_handler_exception_is_caught(self) -> None:
        agent = self._agent()
        agent.action_handlers["explode"] = mock.Mock(side_effect=RuntimeError("x"))
        result = _run(agent.act({"type": "explode"}))
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "x")

    def test_action_handlers_belief_and_log(self) -> None:
        agent = self._agent()

        result = _run(
            agent.act(
                {"type": "update_belief", "belief_name": "zone", "belief_value": "E"}
            )
        )
        self.assertTrue(result["success"])
        self.assertEqual(agent.state.get_belief("zone").value, "E")

        missing = _run(agent.act({"type": "update_belief"}))
        self.assertFalse(missing["success"])

        found = _run(agent.act({"type": "query_belief", "belief_name": "zone"}))
        self.assertTrue(found["success"])
        self.assertEqual(found["belief_value"], "E")

        not_found = _run(agent.act({"type": "query_belief", "belief_name": "ghost"}))
        self.assertFalse(not_found["success"])

        unnamed = _run(agent.act({"type": "query_belief"}))
        self.assertFalse(unnamed["success"])

        logged = _run(agent.act({"type": "log", "message": "hi", "level": "error"}))
        self.assertTrue(logged["success"])
        self.assertEqual(logged["level"], "error")

    def test_invalid_desire_and_deadline_skipped(self) -> None:
        agent = BDIAgent(
            agent_id="bdi-desire",
            config={
                "initial_desires": [
                    {"description": "no name"},
                    {
                        "name": "bad-deadline",
                        "description": "x",
                        "deadline": "not-a-date",
                    },
                    {
                        "name": "dated",
                        "description": "x",
                        "deadline": "2030-01-01T00:00:00",
                    },
                ]
            },
        )
        _run(agent.initialize())
        self.assertEqual(
            {d.name for d in agent.state.get_desires_by_priority()},
            {"bad-deadline", "dated"},
        )
        self.assertIsNone(agent.state.get_desire("bad-deadline").deadline)
        self.assertIsNotNone(agent.state.get_desire("dated").deadline)

    def test_invalid_plan_template_skipped(self) -> None:
        agent = BDIAgent(
            agent_id="bdi-plan",
            config={
                "plans": [
                    {"desire_name": "d", "actions": []},  # missing name
                    {"name": "p1", "actions": []},  # missing desire_name
                    {"name": "p2", "desire_name": "d", "actions": [{"type": "log"}]},
                ]
            },
        )
        _run(agent.initialize())
        self.assertEqual(list(agent.plan_library), ["p2"])

    def test_belief_initialization_dict_and_scalar(self) -> None:
        agent = BDIAgent(
            agent_id="bdi-beliefs",
            config={
                "initial_beliefs": {
                    "scalar": 7,
                    "rich": {"value": "v", "confidence": 0.3, "metadata": {"m": 1}},
                }
            },
        )
        _run(agent.initialize())
        self.assertEqual(agent.state.get_belief("scalar").value, 7)
        rich = agent.state.get_belief("rich")
        self.assertEqual(rich.value, "v")
        self.assertAlmostEqual(rich.confidence, 0.3)
        self.assertEqual(rich.metadata, {"m": 1})

    def test_resolve_placeholders_unknown_config_key(self) -> None:
        agent = BDIAgent(agent_id="bdi-ph", config={"interval": 5})
        with self.assertRaises(ValueError):
            agent._resolve_placeholders("$CONFIG:missing_key")
        # Nested structures resolve recursively and keep config values typed.
        resolved = agent._resolve_placeholders(
            [{"type": "wait", "duration": "$CONFIG:interval"}, {"x": 1}]
        )
        self.assertEqual(resolved[0]["duration"], 5)
        self.assertEqual(resolved[1], {"x": 1})

    def test_desire_satisfaction_requires_all_conditions(self) -> None:
        agent = self._agent()
        self.assertFalse(agent._is_desire_satisfied("ghost"))
        agent.config["initial_desires"] = [
            {
                "name": "d",
                "description": "needs a and b",
                "conditions": {"a": 1, "b": 2},
            }
        ]
        _run(agent.initialize())
        agent.state.update_belief("a", 1)
        self.assertFalse(agent._is_desire_satisfied("d"))
        agent.state.update_belief("b", 2)
        self.assertTrue(agent._is_desire_satisfied("d"))

    def test_perceive_includes_region_and_sensors(self) -> None:
        agent = BDIAgent(
            agent_id="bdi-perceive",
            config={
                "region": "POLYGON((0 0, 1 1))",
                "sensor_readings": {"temp": 21},
            },
        )
        _run(agent.initialize())
        perceptions = _run(agent.perceive())
        self.assertEqual(perceptions["region"], "POLYGON((0 0, 1 1))")
        self.assertEqual(perceptions["sensors"], {"temp": 21})
        self.assertEqual(perceptions["agent_id"], "bdi-perceive")


class TestMessagingServiceLifecycle(unittest.TestCase):
    """Tests for messaging service start/stop and callback processing."""

    def tearDown(self) -> None:
        messaging_service.message_queues.clear()
        messaging_service.channels.clear()
        messaging_service.message_callbacks.clear()

    def test_start_is_idempotent_and_stop_clears_state(self) -> None:
        _run(messaging_service.start())
        task = messaging_service.processing_task
        _run(messaging_service.start())  # second start is a no-op
        self.assertIs(messaging_service.processing_task, task)

        _run(messaging_service.stop())
        self.assertFalse(messaging_service.running)
        _run(messaging_service.stop())  # already stopped → no-op

    def test_unregistered_agent_removed_everywhere(self) -> None:
        messaging_service.register_agent("x1")
        messaging_service.subscribe("x1", "chan")
        messaging_service.register_message_callback("x1", lambda m: None)

        messaging_service.unregister_agent("x1")
        self.assertNotIn("x1", messaging_service.message_queues)
        self.assertNotIn("x1", messaging_service.channels["chan"])
        self.assertNotIn("x1", messaging_service.message_callbacks)
        # Second call is a no-op.
        messaging_service.unregister_agent("x1")

    def test_get_messages_unknown_agent_and_expiry(self) -> None:
        self.assertEqual(_run(messaging_service.get_messages("ghost")), [])

        expired = Message(
            from_agent_id="a",
            to_agent_id="b",
            content={},
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        fresh = Message(from_agent_id="a", to_agent_id="b", content={"k": 1})
        messaging_service.message_queues["b"] = [expired, fresh]

        got = _run(messaging_service.get_messages("b"))
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].delivered)
        self.assertTrue(got[0].read)
        # Only the expired message is purged; the fresh one is retained.
        self.assertEqual(messaging_service.message_queues["b"], [fresh])

        # mark_as_read=False keeps the unread flag (on a message that was
        # not already marked read by the call above).
        unread = Message(from_agent_id="a", to_agent_id="b", content={"k": 2})
        messaging_service.message_queues["b"] = [unread]
        got = _run(messaging_service.get_messages("b", mark_as_read=False))
        self.assertTrue(got[0].delivered)
        self.assertFalse(got[0].read)

    def test_failed_callback_retains_message(self) -> None:
        received = []

        def bad_callback(message):
            raise RuntimeError("handler down")

        messaging_service.register_agent("cb1")
        messaging_service.register_message_callback("cb1", bad_callback)
        messaging_service.message_queues["cb1"].append(
            Message(from_agent_id="a", to_agent_id="cb1", content={"n": 1})
        )
        messaging_service.running = True
        try:
            try:
                _run(
                    asyncio.wait_for(messaging_service._process_messages(), timeout=0.2)
                )
            except TimeoutError:
                pass  # loop idles after processing; timeout cancels it
        finally:
            messaging_service.running = False
        # Callback failed → message retained for retry, marked undelivered.
        self.assertEqual(len(messaging_service.message_queues["cb1"]), 1)

        messaging_service.message_callbacks["cb1"] = lambda m: received.append(m)
        messaging_service.running = True
        try:
            try:
                _run(
                    asyncio.wait_for(messaging_service._process_messages(), timeout=0.2)
                )
            except TimeoutError:
                pass
        finally:
            messaging_service.running = False
        self.assertEqual(len(received), 1)
        self.assertEqual(messaging_service.message_queues["cb1"], [])


class TestTelemetryServiceLifecycle(unittest.TestCase):
    """Tests for telemetry service lifecycle and registration helpers."""

    def tearDown(self) -> None:
        telemetry_service.metrics.clear()
        telemetry_service.agent_health.clear()
        telemetry_service.metric_callbacks.clear()

    def test_start_stop_cycle_and_idempotence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GEO_INFER_TELEMETRY_DIR"] = tmp
            try:
                _run(telemetry_service.start(reporting_interval=60))
                first_task = telemetry_service.reporting_task
                _run(telemetry_service.start(reporting_interval=1))  # no-op
                self.assertIs(telemetry_service.reporting_task, first_task)

                telemetry_service.register_counter(
                    "c1", "counter", agent_id="a1"
                ).increment(3)
                # Let the reporting task take one pass.
                _run(asyncio.sleep(0.15))
                snapshots = [f for f in os.listdir(tmp) if f.startswith("telemetry_")]
                self.assertTrue(snapshots)

                _run(telemetry_service.stop())
                self.assertFalse(telemetry_service.running)
                _run(telemetry_service.stop())  # already stopped → no-op
            finally:
                os.environ.pop("GEO_INFER_TELEMETRY_DIR", None)

    def test_register_existing_metric_returns_same(self) -> None:
        counter = telemetry_service.register_counter("dup", "d", agent_id="a2")
        again = telemetry_service.register_counter("dup", "d", agent_id="a2")
        self.assertIs(counter, again)

        gauge = telemetry_service.register_gauge("dup", "d", agent_id="a2")
        self.assertIs(gauge, again)
        histogram = telemetry_service.register_histogram("dup", "d", agent_id="a2")
        self.assertIs(histogram, again)
        timer = telemetry_service.register_timer("dup", "d", agent_id="a2")
        self.assertIs(timer, again)

    def test_metric_id_includes_agent_and_tags(self) -> None:
        metric_id = telemetry_service._get_metric_id("m", "a9", {"z": 1, "a": 2})
        self.assertEqual(metric_id, "a9:m;a=2;z=1")
        self.assertEqual(telemetry_service._get_metric_id("m", None, None), "m")

    def test_health_and_metric_filtering(self) -> None:
        telemetry_service.update_health("h1", "degraded", {"cpu": 5})
        self.assertEqual(
            telemetry_service.get_health_status("h1")["h1"]["status"], "degraded"
        )
        self.assertEqual(
            telemetry_service.get_health_status("ghost"),
            {"ghost": {"status": "unknown"}},
        )
        self.assertIn("h1", telemetry_service.get_health_status())

        counter = telemetry_service.register_counter("cm", "d", agent_id="h1")
        counter.increment()
        metrics = telemetry_service.get_metrics("h1")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(list(metrics.values())[0]["value"], 1)
        self.assertEqual(len(telemetry_service.get_metrics()), 1)

    def test_metric_callback_registration_stored(self) -> None:
        def observer(metric_id, metric):
            pass

        telemetry_service.register_metric_callback("cb", observer)
        self.assertEqual(telemetry_service.metric_callbacks["cb"], [observer])
        # Registering twice accumulates handlers for the same metric name.
        telemetry_service.register_metric_callback("cb", observer)
        self.assertEqual(len(telemetry_service.metric_callbacks["cb"]), 2)

    def test_monitor_resources_without_psutil_returns(self) -> None:
        # With psutil uninstalled the monitor exits quietly.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            _run(telemetry_service._monitor_resources())
        self.assertNotIn("system.cpu.usage", telemetry_service.metrics)


if __name__ == "__main__":
    unittest.main()
