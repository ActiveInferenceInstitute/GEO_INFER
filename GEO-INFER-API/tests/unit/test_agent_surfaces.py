"""Actual script/CLI/MCP contracts and scientific object composition."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from geo_infer_api.agent_catalog import MODULES, catalog, definitions
from geo_infer_api.agent_runtime import AgentSession, resolve
from geo_infer_api.agent_schema import request_schema
from geo_infer_api.agent_skills import bundle_bytes, read_guidance, skill_inventory


ROOT = Path(__file__).resolve().parents[3]


def test_all_workspace_modules_are_discoverable_without_importing_backends(tmp_path):
    """Discovery works outside the checkout and never imports discovered modules."""
    program = """
import json,sys
from geo_infer_api.agent_catalog import MODULES,definitions,module_inventory
before=set(sys.modules)
counts={name:len(definitions(name)) for name in MODULES}
added=set(sys.modules)-before
assert not any(name.startswith(('geo_infer_space','numpy','pandas','pymc')) for name in added), added
assert all(row['installed'] for row in module_inventory())
print(json.dumps(counts))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    counts = json.loads(result.stdout)
    assert set(counts) == {
        path.name.removeprefix("GEO-INFER-")
        for path in ROOT.glob("GEO-INFER-*")
        if (path / "pyproject.toml").exists()
    }
    assert len(counts) == 45
    assert all(count > 0 for count in counts.values())


@pytest.mark.parametrize("module", MODULES)
def test_each_module_has_paginated_stable_public_source_inventory(module):
    rows = definitions(module)
    page = catalog(module, limit=1)
    assert page["total"] == len(rows) > 0
    assert len(page["items"]) == 1
    assert page["next_offset"] == (1 if len(rows) > 1 else None)
    assert len({row["target"] for row in rows}) == len(rows)
    assert all(row["target"].startswith(f"geo_infer_{module.lower()}") for row in rows)


def test_function_static_method_enum_and_bound_object_calls():
    session = AgentSession()
    cell = session.execute(
        {
            "op": "call",
            "target": "geo_infer_space:latlng_to_cell",
            "args": [37.7749, -122.4194, 9],
            "save_as": "cell",
        }
    )
    assert cell["ok"]
    center = session.execute(
        {
            "op": "call",
            "target": "geo_infer_space:cell_to_latlng",
            "args": [{"$ref": "cell"}],
        }
    )
    assert center["ok"]
    assert center["result"] == pytest.approx([37.7735, -122.4183], abs=0.003)
    description = session.execute(
        {"op": "describe", "target": "geo_infer_energy:EnergyDemandForecaster"}
    )
    assert description["result"]["callable"]
    assert "forecast_demand" in description["result"]["members"]
    enum = session.execute({"op": "get", "target": "geo_infer_api:GeoJSONType.POLYGON"})
    assert enum == {"ok": True, "result": "Polygon"}


def test_xarray_energy_forecast_composes_through_native_handles():
    session = AgentSession()
    result = session.execute(
        {
            "op": "batch",
            "steps": [
                {
                    "op": "call",
                    "target": "geo_infer_energy:EnergyDemandForecaster",
                    "save_as": "forecaster",
                },
                {
                    "op": "call",
                    "handle": "forecaster",
                    "member": "forecast_demand",
                    "kwargs": {
                        "historical_demand": {
                            "$type": "dataarray",
                            "data": [100, 110, 120],
                            "dims": ["time"],
                            "coords": {"time": [2020, 2021, 2022]},
                        },
                        "forecast_years": 2,
                    },
                    "save_as": "forecast",
                },
                {"op": "call", "handle": "forecast", "member": "to_dict"},
                {"op": "release", "handle": "forecast"},
            ],
        }
    )
    assert result["ok"], result
    values = result["result"]["steps"][2]["result"]["data_vars"]["demand_forecast"][
        "data"
    ]
    assert values == pytest.approx([130, 140])
    assert "forecast" not in session.objects


def test_typed_numpy_and_result_path_references():
    session = AgentSession()
    result = session.execute(
        {
            "op": "call",
            "target": "geo_infer_math.core.circulation:skew_part",
            "args": [
                {"$type": "ndarray", "data": [[0, 4], [2, 0]], "dtype": "float64"}
            ],
            "save_as": "skew",
        }
    )
    assert result["ok"]
    assert session.decode({"$ref": "skew", "path": [0, 1]}) == 1
    assert session.execute({"op": "get", "handle": "skew", "member": "shape"})[
        "result"
    ] == [2, 2]
    assert session.execute({"op": "get", "handle": "skew"})["result"] == [
        [0, 1],
        [-1, 0],
    ]


@pytest.mark.parametrize(
    "tag,data",
    [
        ("datetime", "2026-10-16T12:00:00+00:00"),
        ("date", "2026-10-16"),
        ("path", "relative/output.json"),
        ("tuple", [1, 2]),
        ("set", [1, 2]),
        ("float", "nan"),
    ],
)
def test_explicit_type_codecs(tag, data):
    session = AgentSession()
    decoded = session.decode({"$type": tag, "data": data})
    assert decoded is not None
    json.dumps(session.encode(decoded), allow_nan=False)


def test_dataframe_series_geometry_and_literal_codecs():
    session = AgentSession()
    frame = session.decode(
        {"$type": "dataframe", "data": [[1, 2]], "columns": ["a", "b"]}
    )
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "b"] == 2
    series = session.decode({"$type": "series", "data": [1, 2], "name": "x"})
    assert series.sum() == 3
    geometry = session.decode(
        {"$type": "geometry", "data": {"type": "Point", "coordinates": [1, 2]}}
    )
    assert geometry.x == 1
    geo = session.decode(
        {
            "$type": "geodataframe",
            "data": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {"type": "Point", "coordinates": [1, 2]},
                    }
                ],
            },
        }
    )
    assert geo.crs.to_epsg() == 4326
    empty = session.decode(
        {"$type": "geodataframe", "data": {"type": "FeatureCollection", "features": []}}
    )
    assert empty.empty and empty.crs.to_epsg() == 4326
    assert session.decode(
        {"$literal": {"$type": "business_data", "$ref": "literal_key"}}
    ) == {"$type": "business_data", "$ref": "literal_key"}


def test_batch_reports_failure_without_running_later_steps():
    session = AgentSession()
    result = session.execute(
        {
            "op": "batch",
            "steps": [
                {
                    "op": "call",
                    "target": "geo_infer_energy:EnergyDemandForecaster",
                    "save_as": "engine",
                },
                {"op": "call", "handle": "engine", "member": "missing"},
                {"op": "release", "handle": "engine"},
            ],
        }
    )
    assert not result["ok"]
    assert result["result"]["completed"] == 2
    assert "engine" in session.objects


def test_scientific_codecs_preserve_units_dtype_and_native_index_references():
    session = AgentSession()
    array = session.decode(
        {
            "$type": "dataarray",
            "data": [100, 110],
            "dims": ["time"],
            "attrs": {"units": "kWh"},
        }
    )
    assert array.attrs == {"units": "kWh"}
    frame = session.decode(
        {
            "$type": "dataframe",
            "data": [[1, 2]],
            "columns": ["a", "b"],
            "dtype": "float32",
        }
    )
    assert all(dtype == np.dtype("float32") for dtype in frame.dtypes)
    session.objects["matrix"] = np.arange(4).reshape(2, 2)
    assert (
        session.decode({"$ref": "matrix", "path": [{"$type": "tuple", "data": [1, 0]}]})
        == 2
    )


@pytest.mark.parametrize(
    "argument",
    [
        {"$type": "dataarray", "data": [1], "atrs": {"units": "kWh"}},
        {"$type": "geodataframe", "data": {}, "crss": "EPSG:3857"},
        {"$type": "ndarray"},
        {"$type": ""},
        {"$callable": "geo_infer_api:Point", "unexpected": True},
        {"$literal": {}, "unexpected": True},
        {"$ref": "missing", "pah": [0]},
        {"$ref": "missing", "path": "key"},
    ],
)
def test_reserved_argument_operators_reject_silent_typos(argument):
    with pytest.raises(ValueError):
        AgentSession().decode(argument)


def test_entire_batch_structure_validated_before_side_effects():
    session = AgentSession()
    result = session.execute(
        {
            "op": "batch",
            "steps": [
                {
                    "op": "call",
                    "target": "geo_infer_energy:EnergyDemandForecaster",
                    "save_as": "engine",
                },
                {"op": "release", "handle": "engine", "unexpected": True},
            ],
        }
    )
    assert not result["ok"]
    assert not session.objects


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "call", "target": "os:system", "args": ["echo unsafe"]},
        {"op": "get", "target": "geo_infer_api:__dict__"},
        {
            "op": "call",
            "target": "geo_infer_energy:EnergyDemandForecaster",
            "handle": "x",
        },
        {
            "op": "call",
            "target": "geo_infer_energy:EnergyDemandForecaster",
            "kwargs": [],
        },
        {"op": "catalog", "module": "NOT_REAL"},
        {"op": "catalog", "module": "API", "limit": True},
        {"op": "batch", "steps": [{"op": "batch", "steps": []}]},
        {"op": "modules", "surprise": True},
    ],
)
def test_invalid_requests_fail_with_machine_readable_errors(operation):
    result = AgentSession().execute(operation)
    assert not result["ok"]
    assert result["error"]["type"]
    assert result["error"]["message"]


def test_result_budget_and_cycles_keep_native_objects():
    session = AgentSession(max_items=2, max_handles=2)
    handle = session.encode(np.arange(10))
    assert np.array_equal(session.decode(handle), np.arange(10))
    cycle = []
    cycle.append(cycle)
    reference = session.encode(cycle)
    assert session.decode(reference) is cycle
    with pytest.raises(ValueError, match="Handle limit"):
        session.encode(object())
    assert session.execute({"op": "release", "handle": handle["$ref"]})["ok"]


def test_async_application_and_callable_arguments():
    async def scenario():
        session = AgentSession()
        with pytest.raises(RuntimeError, match="await"):
            session.execute({"op": "modules"})
        result = await session.aexecute({"op": "modules"})
        assert result["ok"]
        native = await session.aexecute(
            {
                "op": "call",
                "target": "geo_infer_data:validate_data_integrity",
                "args": [[]],
            }
        )
        assert native["ok"], native
        assert native["result"]["datasets_validated"] == 0
        assert session.decode(
            {"$callable": "geo_infer_energy:EnergyDemandForecaster"}
        ) is resolve("geo_infer_energy:EnergyDemandForecaster")

    asyncio.run(scenario())


def test_real_context_manager_lifecycle():
    session = AgentSession()
    result = session.execute(
        {
            "op": "batch",
            "steps": [
                {
                    "op": "call",
                    "target": "geo_infer_test.core.log_integration:LogIntegration",
                    "save_as": "logger",
                },
                {
                    "op": "call",
                    "handle": "logger",
                    "member": "test_context",
                    "args": ["agent-test", "API", "context"],
                    "save_as": "context",
                },
                {"op": "enter", "handle": "context", "save_as": "entry"},
            ],
        }
    )
    assert result["ok"], result
    assert not session.execute({"op": "release", "handle": "context"})["ok"]
    assert session.execute({"op": "exit", "handle": "context"})["ok"]
    assert (
        session.execute({"op": "get", "handle": "entry", "member": "status"})["result"]
        == "PASS"
    )
    assert session.execute({"op": "release", "handle": "context"})["ok"]


def test_cli_entrypoint_outside_checkout_json_and_exit_codes(tmp_path):
    entrypoint = Path(sys.executable).parent / (
        "geo-infer.exe" if os.name == "nt" else "geo-infer"
    )
    process = subprocess.run(
        [
            str(entrypoint),
            "call",
            "geo_infer_math.core.circulation:skew_part",
            "--args",
            "[[[0,4],[2,0]]]",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["result"] == [[0, 1], [-1, 0]]
    invalid = subprocess.run(
        [str(entrypoint), "run", "-"],
        input='{"op":"call","target":"os:system"}',
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert invalid.returncode == 1
    assert json.loads(invalid.stdout)["ok"] is False
    malformed = subprocess.run(
        [str(entrypoint), "run", "{bad json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert malformed.returncode == 1
    assert json.loads(malformed.stdout)["error"]["type"] == "JSONDecodeError"


def test_schema_exposes_discriminated_requests():
    assert "oneOf" in request_schema()
    assert AgentSession().execute({"op": "schema"})["ok"]


def test_packaged_skill_guidance_matches_all_actual_source_skills():
    asset = ROOT / "GEO-INFER-API/src/geo_infer_api/agent_skills.json"
    assert asset.read_bytes() == bundle_bytes(ROOT)
    assert len(skill_inventory()) == 46
    for module in ("ROOT", *MODULES):
        entry = read_guidance(module)
        assert entry["text"] == (ROOT / entry["path"]).read_text(encoding="utf-8")
    session = AgentSession()
    assert len(session.execute({"op": "skills"})["result"]) == 46
    assert (
        session.execute({"op": "skill", "module": "SPACE"})["result"]["text"]
        == read_guidance("SPACE")["text"]
    )
    assert session.execute({"op": "guide"})["result"]["name"] == "GUIDE"


def test_real_mcp_sdk_stdio_discovery_call_error_and_resource(tmp_path):
    """Spawn a real MCP process and exercise negotiation and native state."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "geo_infer_api.agent_mcp"],
            cwd=str(tmp_path),
        )
        async with (
            stdio_client(params) as (reader, writer),
            ClientSession(reader, writer) as client,
        ):
            await client.initialize()
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "geo_modules",
                "geo_catalog",
                "geo_describe",
                "geo_call",
                "geo_get",
                "geo_batch",
                "geo_handles",
                "geo_release",
                "geo_context",
            }
            modules = await client.call_tool("geo_modules", {})
            assert not modules.isError
            assert len(modules.structuredContent["result"]) == 45
            created = await client.call_tool(
                "geo_call",
                {
                    "target": "geo_infer_energy:EnergyDemandForecaster",
                    "save_as": "engine",
                },
            )
            assert not created.isError
            inspected = await client.call_tool(
                "geo_describe", {"handle": "engine", "member": "forecast_demand"}
            )
            assert (
                "historical_demand"
                in inspected.structuredContent["result"]["signature"]
            )
            failed = await client.call_tool("geo_call", {"target": "os:system"})
            assert failed.isError
            released = await client.call_tool("geo_release", {"handle": "engine"})
            assert not released.isError
            resources = await client.list_resources()
            assert {str(resource.uri) for resource in resources.resources} == {
                "geo-infer://modules",
                "geo-infer://schema",
                "geo-infer://skills",
                "geo-infer://guide",
            }
            resource = await client.read_resource("geo-infer://modules")
            assert len(json.loads(resource.contents[0].text)["result"]) == 45
            skill = await client.read_resource("geo-infer://skills/SPACE")
            assert (
                json.loads(skill.contents[0].text)["result"]["text"]
                == read_guidance("SPACE")["text"]
            )
            templates = await client.list_resource_templates()
            assert (
                str(templates.resourceTemplates[0].uriTemplate)
                == "geo-infer://skills/{module}"
            )
            prompts = await client.list_prompts()
            assert prompts.prompts[0].name == "compose_analysis"

    asyncio.run(scenario())
