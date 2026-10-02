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

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest import mock

from geo_infer_agent.api.messaging import Message, messaging_service
from geo_infer_agent.api.telemetry import (
    telemetry_service,
)
from geo_infer_agent.models.bdi import BDIAgent, BDIState
from geo_infer_agent.models.bdi.belief import Belief
from geo_infer_agent.models.bdi.plan import Plan


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


class TestBDIAgentAct:
    """Tests for the BDI act/decide contract."""

    async def _agent(self, config=None):
        agent = BDIAgent(agent_id="bdi-act", config=config or {})
        await agent.initialize()
        return agent

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_rejects_invalid_action(self) -> None:
        agent = await self._agent()
        result = await agent.act({})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertEqual(result["error"], "Invalid action")

        result = await agent.act({"message": "no type"})
        unittest.TestCase().assertFalse(result["success"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_unknown_action_type(self) -> None:
        agent = await self._agent()
        result = await agent.act({"type": "teleport"})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertIn("No handler", result["error"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_handler_exception_returned_as_error(self) -> None:
        agent = await self._agent()

        async def boom(agent, action):
            raise RuntimeError("kaboom")

        agent.action_handlers["explode"] = boom
        result = await agent.act({"type": "boom", "action_type": "boom"})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertIn("boom", result["error"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_completes_desire_when_conditions_met(self) -> None:
        agent = await self._agent(
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
        action = await agent.decide()
        unittest.TestCase().assertIsNotNone(action)
        result = await agent.act(action)
        unittest.TestCase().assertTrue(result["success"])
        unittest.TestCase().assertTrue(agent.state.get_desire("inform").achieved)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_advances_intention_without_completing(self) -> None:
        agent = await self._agent(
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
        result = await agent.act(await agent.decide())
        unittest.TestCase().assertTrue(result["success"])
        # Intention still mid-plan: desire not yet achieved.
        unittest.TestCase().assertFalse(agent.state.get_desire("d").achieved)
        # Current intention is retained and next decide returns step two.
        current = agent.state.get_current_intention()
        unittest.TestCase().assertIsNotNone(current)
        unittest.TestCase().assertEqual(current.current_action_index, 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_act_handler_exception_is_caught(self) -> None:
        agent = await self._agent()
        agent.action_handlers["explode"] = mock.Mock(side_effect=RuntimeError("x"))
        result = await agent.act({"type": "explode"})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertEqual(result["error"], "x")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_action_handlers_belief_and_log(self) -> None:
        agent = await self._agent()

        result = await agent.act(
            {"type": "update_belief", "belief_name": "zone", "belief_value": "E"}
        )
        unittest.TestCase().assertTrue(result["success"])
        unittest.TestCase().assertEqual(agent.state.get_belief("zone").value, "E")

        missing = await agent.act({"type": "update_belief"})
        unittest.TestCase().assertFalse(missing["success"])

        found = await agent.act({"type": "query_belief", "belief_name": "zone"})
        unittest.TestCase().assertTrue(found["success"])
        unittest.TestCase().assertEqual(found["belief_value"], "E")

        not_found = await agent.act({"type": "query_belief", "belief_name": "ghost"})
        unittest.TestCase().assertFalse(not_found["success"])

        unnamed = await agent.act({"type": "query_belief"})
        unittest.TestCase().assertFalse(unnamed["success"])

        logged = await agent.act({"type": "log", "message": "hi", "level": "error"})
        unittest.TestCase().assertTrue(logged["success"])
        unittest.TestCase().assertEqual(logged["level"], "error")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_invalid_desire_and_deadline_skipped(self) -> None:
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
        await agent.initialize()
        unittest.TestCase().assertEqual(
            {d.name for d in agent.state.get_desires_by_priority()},
            {"bad-deadline", "dated"},
        )
        unittest.TestCase().assertIsNone(
            agent.state.get_desire("bad-deadline").deadline
        )
        unittest.TestCase().assertIsNotNone(agent.state.get_desire("dated").deadline)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_invalid_plan_template_skipped(self) -> None:
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
        await agent.initialize()
        unittest.TestCase().assertEqual(list(agent.plan_library), ["p2"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_belief_initialization_dict_and_scalar(self) -> None:
        agent = BDIAgent(
            agent_id="bdi-beliefs",
            config={
                "initial_beliefs": {
                    "scalar": 7,
                    "rich": {"value": "v", "confidence": 0.3, "metadata": {"m": 1}},
                }
            },
        )
        await agent.initialize()
        unittest.TestCase().assertEqual(agent.state.get_belief("scalar").value, 7)
        rich = agent.state.get_belief("rich")
        unittest.TestCase().assertEqual(rich.value, "v")
        unittest.TestCase().assertAlmostEqual(rich.confidence, 0.3)
        unittest.TestCase().assertEqual(rich.metadata, {"m": 1})

    def test_resolve_placeholders_unknown_config_key(self) -> None:
        agent = BDIAgent(agent_id="bdi-ph", config={"interval": 5})
        with unittest.TestCase().assertRaises(ValueError):
            agent._resolve_placeholders("$CONFIG:missing_key")
        # Nested structures resolve recursively and keep config values typed.
        resolved = agent._resolve_placeholders(
            [{"type": "wait", "duration": "$CONFIG:interval"}, {"x": 1}]
        )
        unittest.TestCase().assertEqual(resolved[0]["duration"], 5)
        unittest.TestCase().assertEqual(resolved[1], {"x": 1})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_desire_satisfaction_requires_all_conditions(self) -> None:
        agent = await self._agent()
        unittest.TestCase().assertFalse(agent._is_desire_satisfied("ghost"))
        agent.config["initial_desires"] = [
            {
                "name": "d",
                "description": "needs a and b",
                "conditions": {"a": 1, "b": 2},
            }
        ]
        await agent.initialize()
        agent.state.update_belief("a", 1)
        unittest.TestCase().assertFalse(agent._is_desire_satisfied("d"))
        agent.state.update_belief("b", 2)
        unittest.TestCase().assertTrue(agent._is_desire_satisfied("d"))

    @pytest.mark.asyncio(loop_scope="function")
    async def test_perceive_includes_region_and_sensors(self) -> None:
        agent = BDIAgent(
            agent_id="bdi-perceive",
            config={
                "region": "POLYGON((0 0, 1 1))",
                "sensor_readings": {"temp": 21},
            },
        )
        await agent.initialize()
        perceptions = await agent.perceive()
        unittest.TestCase().assertEqual(perceptions["region"], "POLYGON((0 0, 1 1))")
        unittest.TestCase().assertEqual(perceptions["sensors"], {"temp": 21})
        unittest.TestCase().assertEqual(perceptions["agent_id"], "bdi-perceive")


class TestMessagingServiceLifecycle:
    """Tests for messaging service start/stop and callback processing."""

    def _tear_down(self) -> None:
        messaging_service.message_queues.clear()
        messaging_service.channels.clear()
        messaging_service.message_callbacks.clear()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_start_is_idempotent_and_stop_clears_state(self) -> None:
        await messaging_service.start()
        task = messaging_service.processing_task
        await messaging_service.start()  # second start is a no-op
        unittest.TestCase().assertIs(messaging_service.processing_task, task)

        await messaging_service.stop()
        unittest.TestCase().assertFalse(messaging_service.running)
        await messaging_service.stop()  # already stopped → no-op

    def test_unregistered_agent_removed_everywhere(self) -> None:
        messaging_service.register_agent("x1")
        messaging_service.subscribe("x1", "chan")
        messaging_service.register_message_callback("x1", lambda m: None)

        messaging_service.unregister_agent("x1")
        unittest.TestCase().assertNotIn("x1", messaging_service.message_queues)
        unittest.TestCase().assertNotIn("x1", messaging_service.channels["chan"])
        unittest.TestCase().assertNotIn("x1", messaging_service.message_callbacks)
        # Second call is a no-op.
        messaging_service.unregister_agent("x1")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_messages_unknown_agent_and_expiry(self) -> None:
        unittest.TestCase().assertEqual(
            await messaging_service.get_messages("ghost"), []
        )

        expired = Message(
            from_agent_id="a",
            to_agent_id="b",
            content={},
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        fresh = Message(from_agent_id="a", to_agent_id="b", content={"k": 1})
        messaging_service.message_queues["b"] = [expired, fresh]

        got = await messaging_service.get_messages("b")
        unittest.TestCase().assertEqual(len(got), 1)
        unittest.TestCase().assertTrue(got[0].delivered)
        unittest.TestCase().assertTrue(got[0].read)
        # Only the expired message is purged; the fresh one is retained.
        unittest.TestCase().assertEqual(messaging_service.message_queues["b"], [fresh])

        # mark_as_read=False keeps the unread flag (on a message that was
        # not already marked read by the call above).
        unread = Message(from_agent_id="a", to_agent_id="b", content={"k": 2})
        messaging_service.message_queues["b"] = [unread]
        got = await messaging_service.get_messages("b", mark_as_read=False)
        unittest.TestCase().assertTrue(got[0].delivered)
        unittest.TestCase().assertFalse(got[0].read)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_failed_callback_retains_message(self, managed_task) -> None:
        received = []
        failed, delivered = asyncio.Event(), asyncio.Event()

        def bad_callback(message):
            failed.set()
            raise RuntimeError("handler down")

        messaging_service.register_agent("cb1")
        messaging_service.register_message_callback("cb1", bad_callback)
        messaging_service.message_queues["cb1"].append(
            Message(from_agent_id="a", to_agent_id="cb1", content={"n": 1})
        )
        messaging_service.running = True
        try:
            async with managed_task(messaging_service._process_messages()):
                await asyncio.wait_for(failed.wait(), timeout=2)
                # Failure leaves the same undelivered message available for retry.
                assert len(messaging_service.message_queues["cb1"]) == 1
                assert not messaging_service.message_queues["cb1"][0].delivered

                def receive(message):
                    received.append(message)
                    delivered.set()

                messaging_service.message_callbacks["cb1"] = receive
                await asyncio.wait_for(delivered.wait(), timeout=2)
        finally:
            messaging_service.running = False
        unittest.TestCase().assertEqual(len(received), 1)
        unittest.TestCase().assertEqual(messaging_service.message_queues["cb1"], [])

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        try:
            yield
        finally:
            self._tear_down()


class TestTelemetryServiceLifecycle:
    """Tests for telemetry service lifecycle and registration helpers."""

    def _tear_down(self) -> None:
        telemetry_service.metrics.clear()
        telemetry_service.agent_health.clear()
        telemetry_service.metric_callbacks.clear()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_start_stop_cycle_and_idempotence(self) -> None:
        import json

        snapshot_written = asyncio.Event()
        real_dump = json.dump

        def write_snapshot(*args, **kwargs):
            real_dump(*args, **kwargs)
            snapshot_written.set()

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GEO_INFER_TELEMETRY_DIR"] = tmp
            try:
                telemetry_service.register_counter(
                    "c1", "counter", agent_id="a1"
                ).increment(3)
                with mock.patch(
                    "geo_infer_agent.api.telemetry.json.dump",
                    side_effect=write_snapshot,
                ):
                    await telemetry_service.start(reporting_interval=60)
                    first_task = telemetry_service.reporting_task
                    await telemetry_service.start(reporting_interval=1)  # no-op
                    unittest.TestCase().assertIs(
                        telemetry_service.reporting_task, first_task
                    )
                    await asyncio.wait_for(snapshot_written.wait(), timeout=2)
                snapshots = [f for f in os.listdir(tmp) if f.startswith("telemetry_")]
                unittest.TestCase().assertTrue(snapshots)
                with open(os.path.join(tmp, snapshots[0])) as handle:
                    assert json.load(handle)["a1:c1"]["value"] == 3

                await telemetry_service.stop()
                unittest.TestCase().assertFalse(telemetry_service.running)
                await telemetry_service.stop()  # already stopped → no-op
            finally:
                os.environ.pop("GEO_INFER_TELEMETRY_DIR", None)

    def test_register_existing_metric_returns_same(self) -> None:
        counter = telemetry_service.register_counter("dup", "d", agent_id="a2")
        again = telemetry_service.register_counter("dup", "d", agent_id="a2")
        unittest.TestCase().assertIs(counter, again)

        gauge = telemetry_service.register_gauge("dup", "d", agent_id="a2")
        unittest.TestCase().assertIs(gauge, again)
        histogram = telemetry_service.register_histogram("dup", "d", agent_id="a2")
        unittest.TestCase().assertIs(histogram, again)
        timer = telemetry_service.register_timer("dup", "d", agent_id="a2")
        unittest.TestCase().assertIs(timer, again)

    def test_metric_id_includes_agent_and_tags(self) -> None:
        metric_id = telemetry_service._get_metric_id("m", "a9", {"z": 1, "a": 2})
        unittest.TestCase().assertEqual(metric_id, "a9:m;a=2;z=1")
        unittest.TestCase().assertEqual(
            telemetry_service._get_metric_id("m", None, None), "m"
        )

    def test_health_and_metric_filtering(self) -> None:
        telemetry_service.update_health("h1", "degraded", {"cpu": 5})
        unittest.TestCase().assertEqual(
            telemetry_service.get_health_status("h1")["h1"]["status"], "degraded"
        )
        unittest.TestCase().assertEqual(
            telemetry_service.get_health_status("ghost"),
            {"ghost": {"status": "unknown"}},
        )
        unittest.TestCase().assertIn("h1", telemetry_service.get_health_status())

        counter = telemetry_service.register_counter("cm", "d", agent_id="h1")
        counter.increment()
        metrics = telemetry_service.get_metrics("h1")
        unittest.TestCase().assertEqual(len(metrics), 1)
        unittest.TestCase().assertEqual(list(metrics.values())[0]["value"], 1)
        unittest.TestCase().assertEqual(len(telemetry_service.get_metrics()), 1)

    def test_metric_callback_registration_stored(self) -> None:
        def observer(metric_id, metric):
            pass

        telemetry_service.register_metric_callback("cb", observer)
        unittest.TestCase().assertEqual(
            telemetry_service.metric_callbacks["cb"], [observer]
        )
        # Registering twice accumulates handlers for the same metric name.
        telemetry_service.register_metric_callback("cb", observer)
        unittest.TestCase().assertEqual(
            len(telemetry_service.metric_callbacks["cb"]), 2
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_monitor_resources_without_psutil_returns(self) -> None:
        # With psutil uninstalled the monitor exits quietly.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            await telemetry_service._monitor_resources()
        unittest.TestCase().assertNotIn("system.cpu.usage", telemetry_service.metrics)

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        try:
            yield
        finally:
            self._tear_down()
