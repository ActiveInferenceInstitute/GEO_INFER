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
