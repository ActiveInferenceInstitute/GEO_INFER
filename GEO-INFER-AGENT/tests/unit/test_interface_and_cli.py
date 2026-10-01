#!/usr/bin/env python3

"""
Unit tests for the unified AgentInterface (api/interface.py), the CLI module,
and the JSON-schema models package surface (models/schemas).
"""

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

import yaml

from geo_infer_agent.api.interface import AgentInterface
from geo_infer_agent.api.messaging import Message, messaging_service
from geo_infer_agent.api.telemetry import telemetry_service
from geo_infer_agent.cli import (
    AGENT_MODULES,
    CONFIG_TEMPLATES,
    create_config_command,
    list_agents_command,
    load_agent_class,
    load_config,
)
from geo_infer_agent.core.agent_registry import agent_registry


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAgentInterface(unittest.TestCase):
    """Tests for the AgentInterface facade over the registry and services."""

    def setUp(self) -> None:
        self.interface = AgentInterface()

    def tearDown(self) -> None:
        # The registry is a singleton; drop agents created by each test.
        for agent_id in list(agent_registry.agents):
            try:
                agent_registry.remove_agent(agent_id)
            except RuntimeError:
                continue

    def test_create_list_info_state(self) -> None:
        agent_id = _run(self.interface.create_agent("default", {}, agent_id="iface-1"))
        self.assertEqual(agent_id, "iface-1")

        listed = self.interface.list_agents()
        self.assertEqual([a["agent_id"] for a in listed], ["iface-1"])

        info = self.interface.get_agent_info("iface-1")
        self.assertEqual(info["agent_type"], "ExampleAgent")
        self.assertFalse(info["is_running"])

        state = _run(self.interface.get_agent_state("iface-1"))
        self.assertEqual(state["agent_id"], "iface-1")
        self.assertFalse(state["is_running"])
        self.assertIn("state", state)

    def test_create_duplicate_agent_raises_value_error(self) -> None:
        _run(self.interface.create_agent("default", {}, agent_id="dup"))
        with self.assertRaises(ValueError):
            _run(self.interface.create_agent("default", {}, agent_id="dup"))

    def test_create_unknown_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            _run(self.interface.create_agent("warp_drive", {}))

    def test_region_added_to_config(self) -> None:
        _run(
            self.interface.create_agent(
                "default",
                {},
                agent_id="geo-1",
                region="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            )
        )
        agent = agent_registry.get_agent("geo-1")
        self.assertEqual(agent.config["region"], "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")

    def test_stop_and_delete_agent(self) -> None:
        _run(self.interface.create_agent("default", {}, agent_id="lifecycle"))
        # Not running → stop logs a warning but succeeds.
        self.assertTrue(_run(self.interface.stop_agent("lifecycle")))
        self.assertTrue(_run(self.interface.delete_agent("lifecycle")))
        self.assertEqual(agent_registry.agents, {})

    def test_stop_missing_agent_returns_false(self) -> None:
        self.assertFalse(_run(self.interface.stop_agent("ghost")))

    def test_delete_missing_agent_returns_false(self) -> None:
        self.assertFalse(_run(self.interface.delete_agent("ghost")))

    def test_perform_action_missing_agent_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            _run(self.interface.perform_action("ghost", "noop", {}))

    def test_perform_action_unsupported_raises_value_error(self) -> None:
        _run(self.interface.create_agent("default", {}, agent_id="actor"))
        with self.assertRaises(ValueError):
            _run(self.interface.perform_action("actor", "teleport", {"x": 1}))

    def test_perform_action_supported(self) -> None:

        _run(
            self.interface.create_agent(
                "data_collector",
                {"data_sources": [{"id": "a", "type": "file"}]},
                agent_id="collector",
            )
        )
        result = _run(
            self.interface.perform_action(
                "collector",
                "configure_source",
                {"source_id": "a", "config": {"path": "x"}},
            )
        )
        self.assertTrue(result["success"])

    def test_send_message_between_agents(self) -> None:
        for suffix in ("sender", "receiver"):
            _run(self.interface.create_agent("default", {}, agent_id=suffix))
        success = _run(self.interface.send_message("sender", "receiver", {"ping": 1}))
        self.assertTrue(success)

    def test_broadcast_message_and_channels(self) -> None:
        for suffix in ("b1", "b2"):
            _run(self.interface.create_agent("default", {}, agent_id=suffix))
        self.interface.subscribe_to_channel("b1", "alerts")
        self.interface.subscribe_to_channel("b2", "alerts")

        messaging_service.message_queues.clear()
        messaging_service.channels.clear()
        messaging_service.subscribe("b1", "alerts2")
        messaging_service.subscribe("b2", "alerts2")
        sent = _run(self.interface.broadcast_message("b1", {"alarm": True}, "alerts2"))
        self.assertEqual(sent, 2)

    def test_send_message_queues_even_for_unregistered_recipient(self) -> None:
        # The messaging service queues per-recipient without checking any
        # registry: delivery succeeds and the queue is created lazily.
        _run(self.interface.create_agent("default", {}, agent_id="lonely"))
        messaging_service.message_queues.clear()
        self.assertTrue(_run(self.interface.send_message("lonely", "nobody", {"x": 1})))
        self.assertIn("nobody", messaging_service.message_queues)

    def test_get_agent_metrics_and_health(self) -> None:
        _run(self.interface.create_agent("default", {}, agent_id="watched"))
        telemetry_service.update_health("watched", "healthy", {"cpu": 1.0})
        self.assertEqual(
            self.interface.get_agent_health("watched")["status"], "healthy"
        )
        self.assertEqual(
            self.interface.get_agent_health("ghost"), {"status": "unknown"}
        )
        # No metrics registered yet → empty mapping for the agent.
        self.assertEqual(self.interface.get_agent_metrics("watched"), {})

    def test_initialize_and_shutdown_services(self) -> None:
        _run(self.interface.initialize_services(reporting_interval=1))
        self.assertTrue(messaging_service.running)
        self.assertTrue(telemetry_service.running)
        _run(self.interface.shutdown_services())
        self.assertFalse(messaging_service.running)
        self.assertFalse(telemetry_service.running)

    def test_expired_message_not_sent(self) -> None:
        expired = Message(
            from_agent_id="a",
            to_agent_id="b",
            content={},
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        self.assertFalse(_run(messaging_service.send_message(expired)))


class TestCli(unittest.TestCase):
    """Tests for the geo-infer-agent CLI helpers."""

    def test_load_agent_class_all_types(self) -> None:
        for agent_type, expected_name in {
            "default": "ExampleAgent",
            "bdi": "BDIAgent",
            "hybrid": "HybridAgent",
            "rule_based": "RuleBasedAgent",
            "reinforcement_learning": "RLAgent",
            "active_inference": "ActiveInferenceAgent",
            "data_collector": "DataCollectorAgent",
        }.items():
            self.assertEqual(load_agent_class(agent_type).__name__, expected_name)

    def test_load_agent_class_unknown_exits(self) -> None:
        with self.assertRaises(SystemExit):
            load_agent_class("warp_drive")

    def test_list_agents_command_prints_all_types(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            list_agents_command(mock.Mock())
        out = buffer.getvalue()
        for agent_type in AGENT_MODULES:
            self.assertIn(agent_type, out)
        self.assertIn("Use with: geo-infer-agent run", out)

    def test_create_config_command_yaml_roundtrip(self) -> None:
        for agent_type in CONFIG_TEMPLATES:
            with tempfile.TemporaryDirectory() as tmp:
                out_path = os.path.join(tmp, "config.yaml")
                args = mock.Mock(type=agent_type, output=out_path, force=False)
                create_config_command(args)
                with open(out_path) as handle:
                    data = yaml.safe_load(handle)
                self.assertEqual(data, CONFIG_TEMPLATES[agent_type])

    def test_create_config_command_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "config.yaml")
            open(out_path, "w").close()
            args = mock.Mock(type="default", output=out_path, force=False)
            with self.assertRaises(SystemExit):
                create_config_command(args)

            args.force = True
            create_config_command(args)  # --force overwrites

    def test_create_config_command_unknown_type_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(
                type="warp_drive", output=os.path.join(tmp, "c.yaml"), force=False
            )
            with self.assertRaises(SystemExit):
                create_config_command(args)

    def test_load_config_json_yaml_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "c.json")
            with open(json_path, "w") as handle:
                import json as json_mod

                json_mod.dump({"agent_type": "default"}, handle)
            self.assertEqual(load_config(json_path)["agent_type"], "default")

            yaml_path = os.path.join(tmp, "c.yaml")
            with open(yaml_path, "w") as handle:
                yaml.dump({"agent_type": "bdi"}, handle)
            self.assertEqual(load_config(yaml_path)["agent_type"], "bdi")

            with self.assertRaises(SystemExit):
                load_config(os.path.join(tmp, "missing.json"))

            bad_path = os.path.join(tmp, "c.toml")
            open(bad_path, "w").close()
            with self.assertRaises(SystemExit):
                load_config(bad_path)

            broken = os.path.join(tmp, "broken.json")
            with open(broken, "w") as handle:
                handle.write("{not json")
            with self.assertRaises(SystemExit):
                load_config(broken)


class TestSchemasPackage(unittest.TestCase):
    """Tests for the models.schemas package surface."""

    def test_schemas_package_exports(self) -> None:
        from geo_infer_agent.models import schemas

        for name in (
            "GENERATIVE_MODEL_SCHEMA",
            "ACTIVE_INFERENCE_STATE_SCHEMA",
            "ACTIVE_INFERENCE_AGENT_SCHEMA",
            "ACTIVE_INFERENCE_SCHEMAS",
        ):
            self.assertTrue(hasattr(schemas, name), name)
        self.assertEqual(
            set(schemas.ACTIVE_INFERENCE_SCHEMAS),
            {"active_inference_agent", "active_inference_state", "generative_model"},
        )

    def test_active_inference_agent_schema_shape(self) -> None:
        from geo_infer_agent.models.schemas import (
            ACTIVE_INFERENCE_AGENT_SCHEMA,
            GENERATIVE_MODEL_SCHEMA,
        )

        self.assertEqual(GENERATIVE_MODEL_SCHEMA["type"], "object")
        self.assertIn("state_dimensions", GENERATIVE_MODEL_SCHEMA["required"])
        self.assertIn("properties", ACTIVE_INFERENCE_AGENT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
