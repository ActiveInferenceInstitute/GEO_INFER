#!/usr/bin/env python3

"""
Unit tests for the DataCollectorAgent: collection handlers, source
availability checks, dataset processing, and source configuration.

Network access is faked by patching ``requests.get``/``requests.head`` in the
module under test; file sources use real temporary files.
"""

import json
import os
import tempfile
import unittest

import pytest
import pytest_asyncio
from unittest import mock

import pandas as pd
import requests

import geo_infer_agent.agents.data_collector as dc_module
from geo_infer_agent.agents.data_collector import DataCollectorAgent


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


class TestDataCollectorConstruction:
    """Tests for construction-time defaults and belief seeding."""

    def test_default_config_merged(self) -> None:
        agent = DataCollectorAgent(agent_id="dc-1")
        unittest.TestCase().assertEqual(agent.config["collection_interval"], 300)
        unittest.TestCase().assertEqual(agent.config["max_retries"], 3)
        unittest.TestCase().assertEqual(agent.config["timeout"], 30)
        unittest.TestCase().assertEqual(agent.config["storage_path"], "data")
        unittest.TestCase().assertEqual(len(agent.config["initial_desires"]), 3)
        unittest.TestCase().assertEqual(len(agent.config["plans"]), 3)
        unittest.TestCase().assertEqual(agent.datasets, [])
        unittest.TestCase().assertEqual(agent.unprocessed_data, [])

    def test_user_config_wins_over_defaults(self) -> None:
        agent = DataCollectorAgent(
            agent_id="dc-2", config={"timeout": 5, "max_retries": 1}
        )
        unittest.TestCase().assertEqual(agent.config["timeout"], 5)
        unittest.TestCase().assertEqual(agent.config["max_retries"], 1)
        # untouched defaults remain
        unittest.TestCase().assertEqual(agent.config["collection_interval"], 300)

    def test_storage_path_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "custom_store")
            agent = DataCollectorAgent(agent_id="dc-3", config={"storage_path": path})
            unittest.TestCase().assertEqual(agent.storage_path, path)
            unittest.TestCase().assertTrue(os.path.isdir(path))

    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialize_seeds_source_beliefs(self) -> None:
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
            await agent.initialize()

            unittest.TestCase().assertEqual(
                agent.state.get_belief("data_source.weather.name").value, "Weather"
            )
            unittest.TestCase().assertEqual(
                agent.state.get_belief("data_source.weather.type").value, "api"
            )
            unittest.TestCase().assertFalse(
                agent.state.get_belief("data_source.weather.available").value
            )
            unittest.TestCase().assertIsNone(
                agent.state.get_belief("data_source.weather.last_collection").value
            )
            # Fallback id uses the source's index.
            unittest.TestCase().assertEqual(
                agent.state.get_belief("data_source.source_1.type").value, "file"
            )
            unittest.TestCase().assertFalse(
                agent.state.get_belief("has_unprocessed_data").value
            )
            unittest.TestCase().assertEqual(
                agent.state.get_belief("total_collected_datasets").value, 0
            )


class TestCollectDataHandler:
    """Tests for the collect_data_from_sources handler."""

    def _agent(self, sources):
        return DataCollectorAgent(
            agent_id="dc-collect", config={"data_sources": sources}
        )

    def _belief(self, agent, name):
        return agent.state.get_belief(name).value

    @pytest.mark.asyncio(loop_scope="function")
    async def test_collect_from_file_source(self) -> None:
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

            result = await agent._handle_collect_data_action(agent, {})
            unittest.TestCase().assertTrue(result["success"])
            unittest.TestCase().assertEqual(result["collected_count"], 1)
            unittest.TestCase().assertEqual(result["error_count"], 0)
            unittest.TestCase().assertEqual(len(agent.datasets), 1)
            unittest.TestCase().assertEqual(len(agent.unprocessed_data), 1)
            unittest.TestCase().assertTrue(self._belief(agent, "has_unprocessed_data"))
            unittest.TestCase().assertEqual(
                self._belief(agent, "total_collected_datasets"), 1
            )
            unittest.TestCase().assertIsNotNone(
                self._belief(agent, "data_source.f1.last_collection")
            )
            with open(agent.datasets[0]["filename"]) as handle:
                unittest.TestCase().assertEqual(json.load(handle), payload)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unavailable_source_skipped(self) -> None:
        agent = self._agent([{"id": "off", "type": "file", "path": "missing.json"}])
        agent.state.update_belief("data_source.off.available", False)
        result = await agent._handle_collect_data_action(agent, {})
        unittest.TestCase().assertEqual(result["collected_count"], 0)
        unittest.TestCase().assertEqual(agent.datasets, [])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unsupported_type_counts_error_when_belief_absent(self) -> None:
        agent = self._agent([{"id": "bad", "type": "carrier_pigeon"}])
        # A missing 'available' belief falls through to collection.
        agent.state.beliefs_dict.pop("data_source.bad.available", None)
        result = await agent._handle_collect_data_action(agent, {})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertEqual(result["error_count"], 1)
        unittest.TestCase().assertIn(
            "Unsupported data source type", result["results"][0]["error"]
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_available_belief_defaults_to_collecting(self) -> None:
        # Only an explicit False belief skips a source; absence collects.
        with tempfile.TemporaryDirectory() as tmp:
            src_file = os.path.join(tmp, "src.json")
            with open(src_file, "w") as handle:
                json.dump({"k": "v"}, handle)
            agent = self._agent([{"id": "s", "type": "file", "path": src_file}])
            agent.storage_path = tmp
            agent.state.beliefs_dict.pop("data_source.s.available", None)
            result = await agent._handle_collect_data_action(agent, {})
            unittest.TestCase().assertEqual(result["collected_count"], 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_total_collected_datasets_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"k": "v"}
            src_file = os.path.join(tmp, "src.json")
            with open(src_file, "w") as handle:
                json.dump(payload, handle)
            agent = self._agent([{"id": "s", "type": "file", "path": src_file}])
            agent.storage_path = tmp
            await agent.initialize()
            agent.state.update_belief("data_source.s.available", True)
            await agent._handle_collect_data_action(agent, {})
            await agent._handle_collect_data_action(agent, {})
            unittest.TestCase().assertEqual(
                self._belief(agent, "total_collected_datasets"), 2
            )


class TestCollectFromSource:
    """Tests for per-source-type collection primitives."""

    def _set_up(self) -> None:
        self.agent = DataCollectorAgent(agent_id="dc-src", config={"timeout": 1})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_collect_list_payload(self) -> None:
        response = _FakeResponse(payload=[{"id": 1}])
        with mock.patch.object(dc_module.requests, "get", return_value=response) as get:
            data = await self.agent._collect_from_api({"id": "api1", "url": "http://x"})
        get.assert_called_once()
        unittest.TestCase().assertEqual(
            data, {"source_id": "api1", "records": [{"id": 1}]}
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_collect_dict_payload(self) -> None:
        response = _FakeResponse(payload={"k": "v"})
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            data = await self.agent._collect_from_api({"id": "api2", "url": "http://x"})
        unittest.TestCase().assertEqual(data, {"k": "v"})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_collect_scalar_payload_raises(self) -> None:
        response = _FakeResponse(payload=42)
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            with unittest.TestCase().assertRaises(ValueError):
                await self.agent._collect_from_api({"id": "api3", "url": "http://x"})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_collect_http_error_propagates(self) -> None:
        response = _FakeResponse(raise_status=True)
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            with unittest.TestCase().assertRaises(requests.HTTPError):
                await self.agent._collect_from_api({"id": "api4", "url": "http://x"})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_collect_without_url_returns_none(self) -> None:
        unittest.TestCase().assertIsNone(
            await self.agent._collect_from_api({"id": "api5", "url": ""})
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_collect_json_and_geojson(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".json", ".geojson"):
                path = os.path.join(tmp, f"src{suffix}")
                with open(path, "w") as handle:
                    json.dump({"records": [1]}, handle)
                data = await self.agent._collect_from_file({"path": path})
                unittest.TestCase().assertEqual(data, {"records": [1]})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_collect_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "src.csv")
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
            data = await self.agent._collect_from_file({"id": "csv1", "path": path})
            unittest.TestCase().assertEqual(data["source_id"], "csv1")
            unittest.TestCase().assertEqual(
                data["records"], [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
            )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_collect_rejects_array_json_and_unknown_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            array_path = os.path.join(tmp, "arr.json")
            with open(array_path, "w") as handle:
                json.dump([1, 2], handle)
            with unittest.TestCase().assertRaises(ValueError):
                await self.agent._collect_from_file({"path": array_path})

            unknown_path = os.path.join(tmp, "x.parquet")
            open(unknown_path, "w").close()
            with unittest.TestCase().assertRaises(ValueError):
                await self.agent._collect_from_file({"path": unknown_path})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_collect_missing_and_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.TestCase().assertRaises(FileNotFoundError):
                await self.agent._collect_from_file(
                    {"path": os.path.join(tmp, "nope.json")}
                )
        unittest.TestCase().assertIsNone(
            await self.agent._collect_from_file({"path": ""})
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_collect(self) -> None:
        response = _FakeResponse(payload={"sensor_id": "s1", "reading": 9})
        with mock.patch.object(dc_module.requests, "get", return_value=response) as get:
            data = await self.agent._collect_from_sensor(
                {"sensor_id": "s1", "url": "http://x", "params": {"unit": "c"}}
            )
        unittest.TestCase().assertEqual(
            get.call_args.kwargs["params"], {"sensor_id": "s1", "unit": "c"}
        )
        unittest.TestCase().assertEqual(data["reading"], 9)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_collect_uses_endpoint_fallback(self) -> None:
        response = _FakeResponse(payload={"r": 1})
        with mock.patch.object(dc_module.requests, "get", return_value=response) as get:
            data = await self.agent._collect_from_sensor(
                {"sensor_id": "s2", "endpoint": "http://e"}
            )
        unittest.TestCase().assertEqual(get.call_args.args[0], "http://e")
        unittest.TestCase().assertEqual(data, {"r": 1})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_collect_requires_sensor_id(self) -> None:
        unittest.TestCase().assertIsNone(
            await self.agent._collect_from_sensor({"sensor_id": ""})
        )
        with unittest.TestCase().assertRaises(ValueError):
            await self.agent._collect_from_sensor({"sensor_id": "s3"})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_collect_rejects_non_dict_payload(self) -> None:
        response = _FakeResponse(payload=[1])
        with mock.patch.object(dc_module.requests, "get", return_value=response):
            with unittest.TestCase().assertRaises(ValueError):
                await self.agent._collect_from_sensor(
                    {"sensor_id": "s4", "url": "http://x"}
                )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_dispatch_unknown_type_raises(self) -> None:
        with unittest.TestCase().assertRaises(ValueError):
            await self.agent._collect_from_source({"type": "smoke_signal"})

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        self._set_up()
        try:
            yield
        finally:
            pass


class TestCheckSourcesHandler:
    """Tests for the check_sources_availability handler."""

    def _agent(self, sources):
        return DataCollectorAgent(agent_id="dc-check", config={"data_sources": sources})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_only_url_bearing_sources_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.json")
            open(good, "w").close()
            agent = self._agent(
                [
                    {"id": "good", "type": "file", "path": good, "url": "http://f"},
                    {"id": "nourl", "type": "file", "path": ""},  # no url → skipped
                ]
            )
            result = await agent._handle_check_sources_action(agent, {})
            unittest.TestCase().assertTrue(result["success"])
            unittest.TestCase().assertEqual(result["available_count"], 1)
            unittest.TestCase().assertEqual(result["unavailable_count"], 0)
            unittest.TestCase().assertEqual(len(result["results"]), 1)
            unittest.TestCase().assertTrue(
                agent.state.get_belief("data_source.good.available").value
            )
            unittest.TestCase().assertIsNotNone(
                agent.state.get_belief("last_monitoring_time").value
            )
            unittest.TestCase().assertIsNotNone(
                agent.state.get_belief("data_source.good.last_check").value
            )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_file_source_is_unavailable(self) -> None:
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
            result = await agent._handle_check_sources_action(agent, {})
            unittest.TestCase().assertEqual(result["unavailable_count"], 1)
            unittest.TestCase().assertFalse(
                agent.state.get_belief("data_source.bad.available").value
            )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_sources_use_head(self) -> None:
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
            result = await agent._handle_check_sources_action(agent, {})
        unittest.TestCase().assertEqual(head.call_count, 2)
        unittest.TestCase().assertEqual(result["available_count"], 1)
        unittest.TestCase().assertEqual(result["unavailable_count"], 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_api_request_exception_marks_unavailable(self) -> None:
        agent = self._agent([{"id": "dead", "type": "api", "url": "http://dead"}])
        with mock.patch.object(
            dc_module.requests,
            "head",
            side_effect=requests.ConnectionError("refused"),
        ):
            result = await agent._handle_check_sources_action(agent, {})
        unittest.TestCase().assertEqual(result["unavailable_count"], 1)
        unittest.TestCase().assertFalse(
            agent.state.get_belief("data_source.dead.available").value
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sensor_sources(self) -> None:
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
            result = await agent._handle_check_sources_action(agent, {})
        unittest.TestCase().assertEqual(result["available_count"], 1)
        unittest.TestCase().assertEqual(result["unavailable_count"], 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_url_gate_and_type_guards_interplay(self) -> None:
        # Sources with no url key are skipped by the handler's url gate
        # before any type dispatch: no results, no transport calls.
        gated = self._agent([{"id": "u", "type": "smoke"}])
        with mock.patch.object(dc_module.requests, "get") as get:
            result = await gated._handle_check_sources_action(gated, {})
        unittest.TestCase().assertEqual(get.call_count, 0)
        unittest.TestCase().assertEqual(result["results"], [])

        # A sensor with a url but no sensor_id is checked, but the type
        # guard reports it unavailable without touching the transport.
        sensor = self._agent([{"id": "s-noid", "type": "sensor", "url": "http://z"}])
        with mock.patch.object(dc_module.requests, "get") as get:
            result = await sensor._handle_check_sources_action(sensor, {})
        unittest.TestCase().assertEqual(get.call_count, 0)
        unittest.TestCase().assertEqual(result["unavailable_count"], 1)
        unittest.TestCase().assertEqual(
            result["results"], [{"source_id": "s-noid", "available": False}]
        )


class TestProcessDataHandler:
    """Tests for the process_collected_data handler."""

    def _agent(self):
        return DataCollectorAgent(agent_id="dc-proc")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_unprocessed_data(self) -> None:
        agent = self._agent()
        result = await agent._handle_process_data_action(agent, {})
        unittest.TestCase().assertTrue(result["success"])
        unittest.TestCase().assertEqual(result["processed_count"], 0)
        unittest.TestCase().assertIn("No unprocessed data", result["message"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_processes_dataset_and_clears_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent()
            agent.storage_path = tmp
            src = os.path.join(tmp, "ds.json")
            with open(src, "w") as handle:
                json.dump({"records": [1, 2]}, handle)
            dataset = {"source_id": "s1", "filename": src, "processed": False}
            agent.unprocessed_data.append(dataset)

            result = await agent._handle_process_data_action(agent, {})
            unittest.TestCase().assertTrue(result["success"])
            unittest.TestCase().assertEqual(result["processed_count"], 1)
            unittest.TestCase().assertEqual(len(agent.unprocessed_data), 0)
            unittest.TestCase().assertFalse(
                agent.state.get_belief("has_unprocessed_data").value
            )
            unittest.TestCase().assertIsNotNone(
                agent.state.get_belief("last_processing_time").value
            )

            # The handler mutates the dataset entry in place and writes the
            # processed copy next to the original.
            processed_path = src.replace(".json", "_processed.json")
            unittest.TestCase().assertTrue(os.path.exists(processed_path))
            unittest.TestCase().assertTrue(dataset["processed"])
            unittest.TestCase().assertEqual(
                dataset["processed_filename"], processed_path
            )
            with open(processed_path) as handle:
                unittest.TestCase().assertEqual(json.load(handle)["records"], [1, 2])

            result = await agent._handle_process_data_action(agent, {})
            unittest.TestCase().assertIn("No unprocessed data", result["message"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_processing_error_counted(self) -> None:
        agent = self._agent()
        agent.unprocessed_data.append(
            {"source_id": "ghost", "filename": "/nonexistent/path.json"}
        )
        result = await agent._handle_process_data_action(agent, {})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertEqual(result["error_count"], 1)


class TestProcessDataset:
    """Tests for _process_dataset statistics computation."""

    def _set_up(self) -> None:
        self.agent = DataCollectorAgent(agent_id="dc-stats")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_feature_stats_computed(self) -> None:
        data = {
            "features": [
                {"properties": {"value": 2.0}},
                {"properties": {"value": 4.0}},
                {"properties": {"other": 1}},
            ]
        }
        processed = await self.agent._process_dataset(data, {})
        stats = processed["stats"]
        unittest.TestCase().assertEqual(stats["count"], 2)
        unittest.TestCase().assertEqual(stats["min"], 2.0)
        unittest.TestCase().assertEqual(stats["max"], 4.0)
        unittest.TestCase().assertEqual(stats["mean"], 3.0)
        unittest.TestCase().assertEqual(stats["stddev"], 1.0)
        unittest.TestCase().assertIn("processed_timestamp", processed)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_features_returns_copy_with_timestamp(self) -> None:
        processed = await self.agent._process_dataset({"plain": True}, {})
        unittest.TestCase().assertEqual(processed["plain"], True)
        unittest.TestCase().assertIn("processed_timestamp", processed)
        # The original data object is not mutated.
        unittest.TestCase().assertNotIn("processed_timestamp", self.agent.datasets)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_error_returns_none(self) -> None:
        # A non-dict "features" value forces an exception inside processing.
        processed = await self.agent._process_dataset({"features": 7}, {})
        unittest.TestCase().assertIsNone(processed)

    @pytest_asyncio.fixture(autouse=True)
    async def _case_state(self):
        self._set_up()
        try:
            yield
        finally:
            pass


class TestSourceConfiguration:
    """Tests for action_configure_source and action_get_collected_data."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_configure_existing_and_new_source(self) -> None:
        agent = DataCollectorAgent(
            agent_id="dc-cfg",
            config={"data_sources": [{"id": "s1", "type": "file"}]},
        )
        result = await agent.action_configure_source(
            "s1", {"type": "api", "url": "http://x"}
        )
        unittest.TestCase().assertTrue(result["success"])
        unittest.TestCase().assertEqual(
            agent.config["data_sources"][0]["url"], "http://x"
        )
        unittest.TestCase().assertEqual(
            agent.state.get_belief("data_source.s1.url").value, "http://x"
        )

        result = await agent.action_configure_source(
            "new", {"type": "file", "path": "a.json"}
        )
        unittest.TestCase().assertTrue(result["success"])
        unittest.TestCase().assertEqual(len(agent.config["data_sources"]), 2)
        unittest.TestCase().assertEqual(agent.config["data_sources"][1]["id"], "new")
        unittest.TestCase().assertEqual(
            agent.state.get_belief("data_source.new.path").value, "a.json"
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_configure_source_error_reported(self) -> None:
        agent = DataCollectorAgent(agent_id="dc-cfg2")
        with mock.patch.dict(agent.config, {"data_sources": None}):
            result = await agent.action_configure_source("x", {"type": "file"})
        unittest.TestCase().assertFalse(result["success"])
        unittest.TestCase().assertIn("error", result)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_collected_data_filters(self) -> None:
        agent = DataCollectorAgent(agent_id="dc-get")
        agent.datasets.extend(
            [
                {"source_id": "a", "processed": False, "timestamp": "2026-01-02"},
                {"source_id": "a", "processed": True, "timestamp": "2026-01-01"},
                {"source_id": "b", "processed": True, "timestamp": "2026-01-03"},
            ]
        )
        result = await agent.action_get_collected_data({"source_id": "a"})
        unittest.TestCase().assertEqual(result["count"], 2)

        result = await agent.action_get_collected_data({"processed": True})
        unittest.TestCase().assertEqual(result["count"], 2)
        unittest.TestCase().assertEqual(
            [d["source_id"] for d in result["datasets"]], ["a", "b"]
        )

        result = await agent.action_get_collected_data(
            {"after": "2026-01-01", "before": "2026-01-03"}
        )
        unittest.TestCase().assertEqual(result["count"], 1)
        unittest.TestCase().assertEqual(result["datasets"][0]["source_id"], "a")
