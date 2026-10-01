#!/usr/bin/env python3

"""
Unit tests for the DataCollectorAgent: collection handlers, source
availability checks, dataset processing, and source configuration.

Network access is faked by patching ``requests.get``/``requests.head`` in the
module under test; file sources use real temporary files.
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import requests

import geo_infer_agent.agents.data_collector as dc_module
from geo_infer_agent.agents.data_collector import DataCollectorAgent


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, payload=None, ok=True, raise_status=False):
        self._payload = payload
        self.ok = ok
        self._raise_status = raise_status

    def raise_for_status(self):
        if self._raise_status:
            raise requests.HTTPError("boom")

    def json(self):
        return self._payload


class TestDataCollectorConstruction(unittest.TestCase):
    """Tests for construction-time defaults and belief seeding."""

    def test_default_config_merged(self) -> None:
        agent = DataCollectorAgent(agent_id="dc-1")
        self.assertEqual(agent.config["collection_interval"], 300)
        self.assertEqual(agent.config["max_retries"], 3)
        self.assertEqual(agent.config["timeout"], 30)
        self.assertEqual(agent.config["storage_path"], "data")
        self.assertEqual(len(agent.config["initial_desires"]), 3)
        self.assertEqual(len(agent.config["plans"]), 3)
        self.assertEqual(agent.datasets, [])
        self.assertEqual(agent.unprocessed_data, [])

    def test_user_config_wins_over_defaults(self) -> None:
        agent = DataCollectorAgent(
            agent_id="dc-2", config={"timeout": 5, "max_retries": 1}
        )
        self.assertEqual(agent.config["timeout"], 5)
        self.assertEqual(agent.config["max_retries"], 1)
        # untouched defaults remain
        self.assertEqual(agent.config["collection_interval"], 300)

    def test_storage_path_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "custom_store")
            agent = DataCollectorAgent(agent_id="dc-3", config={"storage_path": path})
            self.assertEqual(agent.storage_path, path)
            self.assertTrue(os.path.isdir(path))

    def test_initialize_seeds_source_beliefs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = DataCollectorAgent(
                agent_id="dc-4",
                config={
                    "storage_path": tmp,
                    "data_sources": [
                        {
                            "id": "weather",
                            "name": "Weather",
                            "type": "api",
                            "url": "http://x",
                        },
                        {
                            "type": "file",
                            "path": "nowhere.json",
                        },  # no id → index fallback
                    ],
                },
            )
            _run(agent.initialize())

            self.assertEqual(
                agent.state.get_belief("data_source.weather.name").value, "Weather"
            )
            self.assertEqual(
                agent.state.get_belief("data_source.weather.type").value, "api"
            )
            self.assertFalse(
                agent.state.get_belief("data_source.weather.available").value
            )
            self.assertIsNone(
                agent.state.get_belief("data_source.weather.last_collection").value
            )
            # Fallback id uses the source's index.
            self.assertEqual(
                agent.state.get_belief("data_source.source_1.type").value, "file"
            )
            self.assertFalse(agent.state.get_belief("has_unprocessed_data").value)
            self.assertEqual(
                agent.state.get_belief("total_collected_datasets").value, 0
            )


class TestCollectDataHandler(unittest.TestCase):
    """Tests for the collect_data_from_sources handler."""

    def _agent(self, sources):
        return DataCollectorAgent(
            agent_id="dc-collect", config={"data_sources": sources}
        )

    def _belief(self, agent, name):
        return agent.state.get_belief(name).value

    def test_collect_from_file_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "store")
            os.makedirs(
                store, exist_ok=True
            )  # the agent only makedirs its top storage_path
            payload = {"source_id": "f1", "records": [1, 2, 3]}
            src_file = os.path.join(tmp, "src.json")
            with open(src_file, "w") as handle:
                json.dump(payload, handle)

            agent = self._agent([{"id": "f1", "type": "file", "path": src_file}])
            agent.storage_path = store
            agent.state.update_belief("data_source.f1.available", True)

            result = _run(agent._handle_collect_data_action(agent, {}))
            self.assertTrue(result["success"])
            self.assertEqual(result["collected_count"], 1)
            self.assertEqual(result["error_count"], 0)
            self.assertEqual(len(agent.datasets), 1)
            self.assertEqual(len(agent.unprocessed_data), 1)
            self.assertTrue(self._belief(agent, "has_unprocessed_data"))
            self.assertEqual(self._belief(agent, "total_collected_datasets"), 1)
            self.assertIsNotNone(self._belief(agent, "data_source.f1.last_collection"))
            with open(agent.datasets[0]["filename"]) as handle:
                self.assertEqual(json.load(handle), payload)

    def test_unavailable_source_skipped(self) -> None:
        agent = self._agent([{"id": "off", "type": "file", "path": "missing.json"}])
        agent.state.update_belief("data_source.off.available", False)
        result = _run(agent._handle_collect_data_action(agent, {}))
        self.assertEqual(result["collected_count"], 0)
        self.assertEqual(agent.datasets, [])

    def test_unsupported_type_counts_error_when_belief_absent(self) -> None:
        agent = self._agent([{"id": "bad", "type": "carrier_pigeon"}])
        # A missing 'available' belief falls through to collection.
        agent.state.beliefs_dict.pop("data_source.bad.available", None)
        result = _run(agent._handle_collect_data_action(agent, {}))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_count"], 1)
        self.assertIn("Unsupported data source type", result["results"][0]["error"])

    def test_missing_available_belief_defaults_to_collecting(self) -> None:
        # Only an explicit False belief skips a source; absence collects.
        with tempfile.TemporaryDirectory() as tmp:
            src_file = os.path.join(tmp, "src.json")
            with open(src_file, "w") as handle:
                json.dump({"k": "v"}, handle)
            agent = self._agent([{"id": "s", "type": "file", "path": src_file}])
            agent.storage_path = tmp
            agent.state.beliefs_dict.pop("data_source.s.available", None)
            result = _run(agent._handle_collect_data_action(agent, {}))
            self.assertEqual(result["collected_count"], 1)

    def test_total_collected_datasets_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"k": "v"}
            src_file = os.path.join(tmp, "src.json")
            with open(src_file, "w") as handle:
                json.dump(payload, handle)
            agent = self._agent([{"id": "s", "type": "file", "path": src_file}])
            agent.storage_path = tmp
            _run(agent.initialize())
            agent.state.update_belief("data_source.s.available", True)
            _run(agent._handle_collect_data_action(agent, {}))
            _run(agent._handle_collect_data_action(agent, {}))
            self.assertEqual(self._belief(agent, "total_collected_datasets"), 2)


class TestCollectFromSource(unittest.TestCase):
    """Tests for per-source-type collection primitives."""

    def setUp(self) -> None:
        self.agent = DataCollectorAgent(agent_id="dc-src", config={"timeout": 1})

    def test_api_collect_list_payload(self) -> None:
        response = _FakeResponse(payload=[{"id": 1}])
        with mock.patch.object(dc_module.requests, "get", return_value=response) as get:
            data = _run(self.agent._collect_from_api({"id": "api1", "url": "http://x"}))
        get.assert_called_once()
        self.assertEqual(data, {"source_id": "api1", "records": [{"id": 1}]})

    def test_api_collect_dict_payload(self) -> None:
        response = _FakeResponse(payload={"k": "v"})
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            data = _run(self.agent._collect_from_api({"id": "api2", "url": "http://x"}))
        self.assertEqual(data, {"k": "v"})

    def test_api_collect_scalar_payload_raises(self) -> None:
        response = _FakeResponse(payload=42)
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            with self.assertRaises(ValueError):
                _run(self.agent._collect_from_api({"id": "api3", "url": "http://x"}))

    def test_api_collect_http_error_propagates(self) -> None:
        response = _FakeResponse(raise_status=True)
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            with self.assertRaises(requests.HTTPError):
                _run(self.agent._collect_from_api({"id": "api4", "url": "http://x"}))

    def test_api_collect_without_url_returns_none(self) -> None:
        self.assertIsNone(_run(self.agent._collect_from_api({"id": "api5", "url": ""})))

    def test_file_collect_json_and_geojson(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".json", ".geojson"):
                path = os.path.join(tmp, f"src{suffix}")
                with open(path, "w") as handle:
                    json.dump({"records": [1]}, handle)
                data = _run(self.agent._collect_from_file({"path": path}))
                self.assertEqual(data, {"records": [1]})

    def test_file_collect_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "src.csv")
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
            data = _run(self.agent._collect_from_file({"id": "csv1", "path": path}))
            self.assertEqual(data["source_id"], "csv1")
            self.assertEqual(data["records"], [{"a": 1, "b": 3}, {"a": 2, "b": 4}])

    def test_file_collect_rejects_array_json_and_unknown_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            array_path = os.path.join(tmp, "arr.json")
            with open(array_path, "w") as handle:
                json.dump([1, 2], handle)
            with self.assertRaises(ValueError):
                _run(self.agent._collect_from_file({"path": array_path}))

            unknown_path = os.path.join(tmp, "x.parquet")
            open(unknown_path, "w").close()
            with self.assertRaises(ValueError):
                _run(self.agent._collect_from_file({"path": unknown_path}))

    def test_file_collect_missing_and_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _run(
                    self.agent._collect_from_file(
                        {"path": os.path.join(tmp, "nope.json")}
                    )
                )
        self.assertIsNone(_run(self.agent._collect_from_file({"path": ""})))

    def test_sensor_collect(self) -> None:
        response = _FakeResponse(payload={"sensor_id": "s1", "reading": 9})
        with mock.patch.object(dc_module.requests, "get", return_value=response) as get:
            data = _run(
                self.agent._collect_from_sensor(
                    {"sensor_id": "s1", "url": "http://x", "params": {"unit": "c"}}
                )
            )
        self.assertEqual(
            get.call_args.kwargs["params"], {"sensor_id": "s1", "unit": "c"}
        )
        self.assertEqual(data["reading"], 9)

    def test_sensor_collect_uses_endpoint_fallback(self) -> None:
        response = _FakeResponse(payload={"r": 1})
        with mock.patch.object(dc_module.requests, "get", return_value=response) as get:
            data = _run(
                self.agent._collect_from_sensor(
                    {"sensor_id": "s2", "endpoint": "http://e"}
                )
            )
        self.assertEqual(get.call_args.args[0], "http://e")
        self.assertEqual(data, {"r": 1})

    def test_sensor_collect_requires_sensor_id(self) -> None:
        self.assertIsNone(_run(self.agent._collect_from_sensor({"sensor_id": ""})))
        with self.assertRaises(ValueError):
            _run(self.agent._collect_from_sensor({"sensor_id": "s3"}))

    def test_sensor_collect_rejects_non_dict_payload(self) -> None:
        response = _FakeResponse(payload=[1])
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            with self.assertRaises(ValueError):
                _run(
                    self.agent._collect_from_sensor(
                        {"sensor_id": "s4", "url": "http://x"}
                    )
                )

    def test_dispatch_unknown_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            _run(self.agent._collect_from_source({"type": "smoke_signal"}))


class TestCheckSourcesHandler(unittest.TestCase):
    """Tests for the check_sources_availability handler."""

    def _agent(self, sources):
        return DataCollectorAgent(agent_id="dc-check", config={"data_sources": sources})

    def test_only_url_bearing_sources_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.json")
            open(good, "w").close()
            agent = self._agent(
                [
                    {"id": "good", "type": "file", "path": good, "url": "http://f"},
                    {"id": "nourl", "type": "file", "path": ""},  # no url → skipped
                ]
            )
            result = _run(agent._handle_check_sources_action(agent, {}))
            self.assertTrue(result["success"])
            self.assertEqual(result["available_count"], 1)
            self.assertEqual(result["unavailable_count"], 0)
            self.assertEqual(len(result["results"]), 1)
            self.assertTrue(agent.state.get_belief("data_source.good.available").value)
            self.assertIsNotNone(agent.state.get_belief("last_monitoring_time").value)
            self.assertIsNotNone(
                agent.state.get_belief("data_source.good.last_check").value
            )

    def test_missing_file_source_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(
                [
                    {
                        "id": "bad",
                        "type": "file",
                        "path": os.path.join(tmp, "nope.json"),
                        "url": "http://f",
                    }
                ]
            )
            result = _run(agent._handle_check_sources_action(agent, {}))
            self.assertEqual(result["unavailable_count"], 1)
            self.assertFalse(agent.state.get_belief("data_source.bad.available").value)

    def test_api_sources_use_head(self) -> None:
        agent = self._agent(
            [
                {"id": "up", "type": "api", "url": "http://up"},
                {"id": "down", "type": "api", "url": "http://down"},
                {"id": "nourl", "type": "api", "url": ""},
            ]
        )
        ok = _FakeResponse(ok=True)
        bad = _FakeResponse(ok=False)
        with mock.patch.object(
            dc_module.requests, "head", side_effect=[ok, bad]
        ) as head:
            result = _run(agent._handle_check_sources_action(agent, {}))
        self.assertEqual(head.call_count, 2)
        self.assertEqual(result["available_count"], 1)
        self.assertEqual(result["unavailable_count"], 1)

    def test_api_request_exception_marks_unavailable(self) -> None:
        agent = self._agent([{"id": "dead", "type": "api", "url": "http://dead"}])
        with mock.patch.object(
            dc_module.requests,
            "head",
            side_effect=requests.ConnectionError("refused"),
        ):
            result = _run(agent._handle_check_sources_action(agent, {}))
        self.assertEqual(result["unavailable_count"], 1)
        self.assertFalse(agent.state.get_belief("data_source.dead.available").value)

    def test_sensor_sources(self) -> None:
        agent = self._agent(
            [
                {"id": "s-ok", "type": "sensor", "sensor_id": "s1", "url": "http://x"},
                {"id": "s-bad", "type": "sensor", "sensor_id": "s2", "url": "http://y"},
            ]
        )
        with mock.patch.object(
            dc_module.requests,
            "get",
            side_effect=[_FakeResponse(ok=True), requests.Timeout("slow")],
        ):
            result = _run(agent._handle_check_sources_action(agent, {}))
        self.assertEqual(result["available_count"], 1)
        self.assertEqual(result["unavailable_count"], 1)

    def test_url_gate_and_type_guards_interplay(self) -> None:
        # Sources with no url key are skipped by the handler's url gate
        # before any type dispatch: no results, no transport calls.
        gated = self._agent([{"id": "u", "type": "smoke"}])
        with mock.patch.object(dc_module.requests, "get") as get:
            result = _run(gated._handle_check_sources_action(gated, {}))
        self.assertEqual(get.call_count, 0)
        self.assertEqual(result["results"], [])

        # A sensor with a url but no sensor_id is checked, but the type
        # guard reports it unavailable without touching the transport.
        sensor = self._agent([{"id": "s-noid", "type": "sensor", "url": "http://z"}])
        with mock.patch.object(dc_module.requests, "get") as get:
            result = _run(sensor._handle_check_sources_action(sensor, {}))
        self.assertEqual(get.call_count, 0)
        self.assertEqual(result["unavailable_count"], 1)
        self.assertEqual(
            result["results"], [{"source_id": "s-noid", "available": False}]
        )


class TestProcessDataHandler(unittest.TestCase):
    """Tests for the process_collected_data handler."""

    def _agent(self):
        return DataCollectorAgent(agent_id="dc-proc")

    def test_no_unprocessed_data(self) -> None:
        agent = self._agent()
        result = _run(agent._handle_process_data_action(agent, {}))
        self.assertTrue(result["success"])
        self.assertEqual(result["processed_count"], 0)
        self.assertIn("No unprocessed data", result["message"])

    def test_processes_dataset_and_clears_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent()
            agent.storage_path = tmp
            src = os.path.join(tmp, "ds.json")
            with open(src, "w") as handle:
                json.dump({"records": [1, 2]}, handle)
            dataset = {"source_id": "s1", "filename": src, "processed": False}
            agent.unprocessed_data.append(dataset)

            result = _run(agent._handle_process_data_action(agent, {}))
            self.assertTrue(result["success"])
            self.assertEqual(result["processed_count"], 1)
            self.assertEqual(len(agent.unprocessed_data), 0)
            self.assertFalse(agent.state.get_belief("has_unprocessed_data").value)
            self.assertIsNotNone(agent.state.get_belief("last_processing_time").value)

            # The handler mutates the dataset entry in place and writes the
            # processed copy next to the original.
            processed_path = src.replace(".json", "_processed.json")
            self.assertTrue(os.path.exists(processed_path))
            self.assertTrue(dataset["processed"])
            self.assertEqual(dataset["processed_filename"], processed_path)
            with open(processed_path) as handle:
                self.assertEqual(json.load(handle)["records"], [1, 2])

            result = _run(agent._handle_process_data_action(agent, {}))
            self.assertIn("No unprocessed data", result["message"])

    def test_processing_error_counted(self) -> None:
        agent = self._agent()
        agent.unprocessed_data.append(
            {"source_id": "ghost", "filename": "/nonexistent/path.json"}
        )
        result = _run(agent._handle_process_data_action(agent, {}))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_count"], 1)


class TestProcessDataset(unittest.TestCase):
    """Tests for _process_dataset statistics computation."""

    def setUp(self) -> None:
        self.agent = DataCollectorAgent(agent_id="dc-stats")

    def test_feature_stats_computed(self) -> None:
        data = {
            "features": [
                {"properties": {"value": 2.0}},
                {"properties": {"value": 4.0}},
                {"properties": {"other": 1}},
            ]
        }
        processed = _run(self.agent._process_dataset(data, {}))
        stats = processed["stats"]
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["min"], 2.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["stddev"], 1.0)
        self.assertIn("processed_timestamp", processed)

    def test_no_features_returns_copy_with_timestamp(self) -> None:
        processed = _run(self.agent._process_dataset({"plain": True}, {}))
        self.assertEqual(processed["plain"], True)
        self.assertIn("processed_timestamp", processed)
        # The original data object is not mutated.
        self.assertNotIn("processed_timestamp", self.agent.datasets)

    def test_error_returns_none(self) -> None:
        # A non-dict "features" value forces an exception inside processing.
        processed = _run(self.agent._process_dataset({"features": 7}, {}))
        self.assertIsNone(processed)


class TestSourceConfiguration(unittest.TestCase):
    """Tests for action_configure_source and action_get_collected_data."""

    def test_configure_existing_and_new_source(self) -> None:
        agent = DataCollectorAgent(
            agent_id="dc-cfg",
            config={"data_sources": [{"id": "s1", "type": "file"}]},
        )
        result = _run(
            agent.action_configure_source("s1", {"type": "api", "url": "http://x"})
        )
        self.assertTrue(result["success"])
        self.assertEqual(agent.config["data_sources"][0]["url"], "http://x")
        self.assertEqual(agent.state.get_belief("data_source.s1.url").value, "http://x")

        result = _run(
            agent.action_configure_source("new", {"type": "file", "path": "a.json"})
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(agent.config["data_sources"]), 2)
        self.assertEqual(agent.config["data_sources"][1]["id"], "new")
        self.assertEqual(agent.state.get_belief("data_source.new.path").value, "a.json")

    def test_configure_source_error_reported(self) -> None:
        agent = DataCollectorAgent(agent_id="dc-cfg2")
        with mock.patch.dict(agent.config, {"data_sources": None}):
            result = _run(agent.action_configure_source("x", {"type": "file"}))
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_get_collected_data_filters(self) -> None:
        agent = DataCollectorAgent(agent_id="dc-get")
        agent.datasets.extend(
            [
                {"source_id": "a", "processed": False, "timestamp": "2026-01-02"},
                {"source_id": "a", "processed": True, "timestamp": "2026-01-01"},
                {"source_id": "b", "processed": True, "timestamp": "2026-01-03"},
            ]
        )
        result = _run(agent.action_get_collected_data({"source_id": "a"}))
        self.assertEqual(result["count"], 2)

        result = _run(agent.action_get_collected_data({"processed": True}))
        self.assertEqual(result["count"], 2)
        self.assertEqual([d["source_id"] for d in result["datasets"]], ["a", "b"])

        result = _run(
            agent.action_get_collected_data(
                {"after": "2026-01-01", "before": "2026-01-03"}
            )
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["datasets"][0]["source_id"], "a")


if __name__ == "__main__":
    unittest.main()
