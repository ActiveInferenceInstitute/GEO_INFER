"""Real native/child stdout isolation on minimal API installations."""

import asyncio
from datetime import timedelta
import json
import os
import shlex
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def _child_request() -> dict:
    arguments = [
        sys.executable,
        "-c",
        "import sys; print('CHILD_STDOUT_PROBE',repr(sys.stdin.read()))",
    ]
    command = (
        subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
    )
    return {
        "op": "call",
        "target": "geo_infer_api.agent_stdio:os.system",
        "args": [command],
    }


def _native_request() -> dict:
    return {
        "op": "batch",
        "steps": [
            {
                "op": "call",
                "target": "geo_infer_api.agent_stdio:ctypes.CDLL",
                "args": ["ucrtbase" if os.name == "nt" else None],
                "save_as": "library",
            },
            {
                "op": "call",
                "target": "geo_infer_api.agent_stdio:os.fsencode",
                "args": ["C_STDOUT_PROBE\n"],
                "save_as": "buffer",
            },
            {
                "op": "call",
                "handle": "library",
                "member": "puts",
                "args": [{"$ref": "buffer"}],
            },
            {"op": "call", "handle": "library", "member": "fflush", "args": [None]},
        ],
    }


@pytest.mark.parametrize("kind", ["child", "native"])
def test_cli_preserves_single_json_with_child_and_c_output(tmp_path, kind):
    request = _child_request() if kind == "child" else _native_request()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "geo_infer_api.agent_cli",
            "run",
            json.dumps(request),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"]
    marker = "CHILD_STDOUT_PROBE" if kind == "child" else "C_STDOUT_PROBE"
    assert marker in result.stderr
    assert marker not in result.stdout


def test_native_descriptor_restoration_on_exception(tmp_path):
    program = """
import json,os
from geo_infer_api.agent_stdio import protocol_stdout
try:
    with protocol_stdout():
        os.write(1,b'DIRECT_FD_PROBE\\n')
        raise ValueError('expected')
except ValueError:
    pass
print(json.dumps({'restored':True}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"restored": True}
    assert "DIRECT_FD_PROBE" in result.stderr


def test_mcp_survives_native_and_inherited_child_stdout(tmp_path):
    """Actual SDK negotiation stays usable after child and buffered C writes."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-I", "-m", "geo_infer_api.agent_mcp"],
            cwd=str(tmp_path),
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(
                reader, writer, read_timeout_seconds=timedelta(seconds=30)
            ) as client,
        ):
            await client.initialize()
            child = _child_request()
            child.pop("op")
            response = await client.call_tool("geo_call", child)
            assert not response.isError and response.structuredContent["result"] == 0
            native = await client.call_tool(
                "geo_batch", {"steps": _native_request()["steps"]}
            )
            assert not native.isError
            point = await client.call_tool(
                "geo_call",
                {"target": "geo_infer_api:Point", "kwargs": {"coordinates": [1, 2]}},
            )
            assert not point.isError and point.structuredContent["result"][
                "coordinates"
            ] == [1, 2]
            skill = await client.read_resource("geo-infer://skills/SPACE")
            assert json.loads(skill.contents[0].text)["result"]["name"] == "SPACE"

    asyncio.run(scenario())


@pytest.mark.parametrize("explicit_close", [False, True])
def test_mcp_releases_entered_file_on_close_and_server_shutdown(
    tmp_path, explicit_close
):
    """Actual server lifecycle flushes entered resources without manual exit."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    output = tmp_path / "owned-context.txt"

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-I", "-m", "geo_infer_api.agent_mcp"],
            cwd=str(tmp_path),
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(
                reader, writer, read_timeout_seconds=timedelta(seconds=30)
            ) as client,
        ):
            await client.initialize()
            steps = [
                dict(
                    op="call",
                    target="geo_infer_api.agent_runtime:Path",
                    args=[str(output)],
                    save_as="path",
                ),
                dict(
                    op="call", handle="path", member="open", args=["w"], save_as="file"
                ),
                dict(op="enter", handle="file", save_as="stream"),
                dict(
                    op="call",
                    handle="stream",
                    member="write",
                    args=["flushed by native context exit"],
                ),
            ]
            result = await client.call_tool("geo_batch", {"steps": steps})
            assert not result.isError, result
            diagnostics = await client.call_tool("geo_diagnostics", {})
            assert (
                diagnostics.structuredContent["result"]["session"]["active_contexts"]
                == 1
            )
            if explicit_close:
                closed = await client.call_tool("geo_close", {})
                assert closed.structuredContent["result"]["exited_contexts"] == 1
                assert output.read_text() == "flushed by native context exit"
                after = await client.call_tool("geo_diagnostics", {})
                assert after.structuredContent["result"]["session"]["closed"]
                failure = await client.call_tool(
                    "geo_call",
                    {
                        "target": "geo_infer_api:Point",
                        "kwargs": {"coordinates": [1, 2]},
                    },
                )
                assert failure.isError

    asyncio.run(scenario())
    assert output.read_text() == "flushed by native context exit"


def test_cli_native_selection_and_capacity_safe_persistent_reads(tmp_path):
    selection = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "geo_infer_api.agent_cli",
            "get",
            "geo_infer_api.agent_catalog:MODULES",
            "--path",
            '[{"$type":"slice","data":[0,3,null]}]',
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert selection.returncode == 0, selection.stderr
    expected = json.loads(selection.stdout)["result"]
    assert len(expected) == 3 and expected[0] == "ACT"
    requests = [
        dict(
            op="call",
            target="geo_infer_api.agent_runtime:AgentSession",
            save_as="native",
        )
    ]
    requests += [dict(op="get", handle="native") for _ in range(5)]
    requests += [
        dict(op="get", handle="native", member="objects", path=[]),
        dict(op="close"),
    ]
    persistent = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "geo_infer_api.agent_cli",
            "session",
            "--max-handles",
            "1",
        ],
        input="".join(json.dumps(request) + "\n" for request in requests),
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert persistent.returncode == 0, persistent.stderr
    results = [json.loads(line) for line in persistent.stdout.splitlines()]
    assert len(results) == len(requests) and all(row["ok"] for row in results)
    assert all(row["result"]["$ref"] == "native" for row in results[:6])
    assert results[-2]["result"] == {}
    assert results[-1]["result"]["released_handles"] == 1


def test_mcp_native_selection_and_handle_identity(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-I", "-m", "geo_infer_api.agent_mcp"],
            cwd=str(tmp_path),
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(
                reader, writer, read_timeout_seconds=timedelta(seconds=30)
            ) as client,
        ):
            await client.initialize()
            created = await client.call_tool(
                "geo_call",
                dict(
                    target="geo_infer_api.agent_runtime:AgentSession",
                    save_as="native",
                ),
            )
            assert not created.isError
            for _ in range(3):
                read = await client.call_tool("geo_get", dict(handle="native"))
                assert read.structuredContent["result"]["$ref"] == "native"
            selected = await client.call_tool(
                "geo_get",
                dict(
                    target="geo_infer_api.agent_catalog:MODULES",
                    path=[{"$type": "slice", "data": [0, 3, None]}],
                ),
            )
            assert not selected.isError
            assert len(selected.structuredContent["result"]) == 3
            malformed = await client.call_tool(
                "geo_get",
                dict(
                    target="geo_infer_api.agent_catalog:MODULES",
                    path=[{"$type": "slice", "data": [None, None, 0]}],
                ),
            )
            assert malformed.isError
            handles = await client.call_tool("geo_handles", {})
            assert len(handles.structuredContent["result"]) == 1

    asyncio.run(scenario())
