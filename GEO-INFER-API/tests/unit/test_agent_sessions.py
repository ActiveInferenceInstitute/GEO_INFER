"""Persistent transports and real native context lifecycle without scientific extras."""

import asyncio
import json
import subprocess
import sys

import pytest

from geo_infer_api.agent_runtime import AgentSession
from geo_infer_api.agent_schema import request_schema

pytestmark = pytest.mark.unit


def test_schema_results_and_catalog_pages_do_not_share_mutable_results():
    schema = request_schema()
    schema.clear()
    assert "oneOf" in request_schema()
    session = AgentSession()
    request = dict(op="catalog", module="API", limit=1)
    first = session.execute(request)
    first["result"]["items"][0]["summary"] = "caller-owned mutation"
    assert (
        session.execute(request)["result"]["items"][0]["summary"]
        != "caller-owned mutation"
    )


def test_session_context_closes_real_file_and_releases_handles(tmp_path):
    output = tmp_path / "native.txt"
    with AgentSession() as session:
        file = output.open("w")
        session.objects["file"] = file
        assert session.execute(dict(op="enter", handle="file", save_as="stream"))["ok"]
        assert session.execute(
            dict(op="call", handle="stream", member="write", args=["native context"])
        )["ok"]
    assert file.closed and output.read_text() == "native context"
    assert session.closed and not session.objects and not session.contexts
    assert session.close() == dict(closed=True, released_handles=0, exited_contexts=0)
    assert not session.execute(
        dict(op="call", target="geo_infer_api.agent_catalog:package_name", args=["API"])
    )["ok"]
    assert session.execute(dict(op="diagnostics"))["result"]["session"]["closed"]


def test_session_exception_reaches_all_native_exits_in_reverse_order():
    events = []

    class Manager:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, kind, value, trace):
            events.append((self.name, kind))

    with pytest.raises(ValueError, match="body failure"):
        with AgentSession() as session:
            for name in ["outer", "inner"]:
                session.objects[name] = Manager(name)
                assert session.execute(
                    dict(op="enter", handle=name, save_as=name + "-value")
                )["ok"]
            raise ValueError("body failure")
    assert events == [("inner", ValueError), ("outer", ValueError)]
    assert session.closed and not session.objects


def test_cleanup_failure_still_attempts_other_contexts_and_clears_handles():
    events = []

    class Manager:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *error):
            events.append(self.name)
            if self.name == "inner":
                raise OSError("cleanup failure")

    session = AgentSession()
    for name in ["outer", "inner"]:
        session.objects[name] = Manager(name)
        assert session.execute(dict(op="enter", handle=name, save_as=name + "-value"))[
            "ok"
        ]
    with pytest.raises(OSError, match="cleanup failure"):
        session.close()
    assert events == ["inner", "outer"] and session.closed
    assert not session.objects and not session.contexts


def test_async_session_context_and_sync_misuse_are_explicit():
    events = []

    class Manager:
        async def __aenter__(self):
            events.append("enter")
            return self

        async def __aexit__(self, *error):
            events.append("exit")

    async def scenario():
        async with AgentSession() as session:
            session.objects["async"] = Manager()
            assert (
                await session.aexecute(
                    dict(op="enter", handle="async", save_as="value")
                )
            )["ok"]
            with pytest.raises(RuntimeError, match="aclose"):
                session.close()
        assert session.closed

    asyncio.run(scenario())
    assert events == ["enter", "exit"]


def _cli(tmp_path, *arguments, input=None):
    return subprocess.run(
        [sys.executable, "-I", "-m", "geo_infer_api.agent_cli", *arguments],
        cwd=tmp_path,
        input=input,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_jsonl_retains_handles_rejects_duplicates_and_exits_explicitly(tmp_path):
    lines = [
        json.dumps(
            dict(
                op="call",
                target="geo_infer_api.agent_catalog:package_name",
                args=["API"],
                save_as="name",
            )
        ),
        json.dumps(dict(op="call", handle="name", member="upper")),
        '{"op":"handles","op":"close"}',
        json.dumps(dict(op="get", handle="name")),
        json.dumps(dict(op="close")),
    ]
    result = _cli(tmp_path, "session", input="\n".join(lines) + "\n")
    assert result.returncode == 1, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 5 and responses[1]["result"] == "GEO_INFER_API"
    assert (
        not responses[2]["ok"]
        and "Duplicate JSON field" in responses[2]["error"]["message"]
    )
    assert responses[3]["result"] == "geo_infer_api"
    assert responses[4]["result"] == dict(
        closed=True, released_handles=1, exited_contexts=0
    )


def test_jsonl_eof_exits_native_file_context_and_flushes(tmp_path):
    output = tmp_path / "persistent.txt"
    requests = [
        dict(
            op="call",
            target="geo_infer_api.agent_cli:Path",
            args=[str(output)],
            save_as="path",
        ),
        dict(op="call", handle="path", member="open", args=["w"], save_as="file"),
        dict(op="enter", handle="file", save_as="stream"),
        dict(op="call", handle="stream", member="write", args=["flushed at EOF"]),
    ]
    result = _cli(
        tmp_path, "session", input="".join(json.dumps(r) + "\n" for r in requests)
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 4 and all(
        json.loads(line)["ok"] for line in result.stdout.splitlines()
    )
    assert output.read_text() == "flushed at EOF"


def test_cli_doctor_version_and_double_stdin_error(tmp_path):
    result = _cli(tmp_path, "doctor")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)["result"]
    assert data["python"]["version"].startswith(
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )
    assert (
        data["profiling"]["sampling_permissions"] == "not_probed"
        and len(data["modules"]) == 45
    )
    assert _cli(tmp_path, "--version").stdout.startswith("geo-infer ")
    rejected = _cli(
        tmp_path,
        "call",
        "geo_infer_api.agent_catalog:package_name",
        "--args",
        "-",
        "--kwargs",
        "-",
        input="[]",
    )
    assert (
        rejected.returncode == 1
        and "only one" in json.loads(rejected.stdout)["error"]["message"]
    )


def test_benchmark_exercises_all_native_operations_and_validates_bounds():
    from geo_infer_api.agent_benchmarks import benchmark_agent

    result = asyncio.run(benchmark_agent(iterations=2, repeats=2, warmup=0))
    assert len(result["module_definitions"]) == 45
    assert result["module_definitions"]["API"] > 0
    assert result["definitions"] == sum(result["module_definitions"].values())
    assert set(result["measurements"]) == {"schema", "catalog", "call", "batch"}
    assert result["measurements"]["batch"]["operations_per_sample"] == 50
    for row in result["measurements"].values():
        assert len(row["samples_seconds"]) == 2 and row["median_seconds"] > 0
    with pytest.raises(ValueError):
        asyncio.run(benchmark_agent(iterations=True))


def test_catalog_statement_walk_preserves_conditional_definitions_and_excludes_locals():
    import ast
    from geo_infer_api.agent_catalog import _definitions

    tree = ast.parse("""
if True:
    def conditional():
        def local(): pass
else:
    class Alternate:
        def member(self): pass
try:
    def attempted(): pass
except Exception:
    def recovered(): pass
finally:
    def finalized(): pass
match 1:
    case 1:
        async def matched(): pass
""")
    targets = {
        row["target"] for row in _definitions(tree.body, "geo_infer_api.example")
    }
    assert targets == {
        f"geo_infer_api.example:{name}"
        for name in [
            "conditional",
            "Alternate",
            "Alternate.member",
            "attempted",
            "recovered",
            "finalized",
            "matched",
        ]
    }


@pytest.mark.parametrize("number", ["1e999", "-1e999"])
def test_cli_rejects_overflowed_json_numbers_and_session_can_continue(tmp_path, number):
    request = (
        '{"op":"call","target":"geo_infer_api.agent_runtime:math.isfinite","args":['
        + number
        + "]}"
    )
    single = _cli(tmp_path, "run", request)
    assert single.returncode == 1
    assert "tagged float" in json.loads(single.stdout)["error"]["message"]
    valid = dict(
        op="call", target="geo_infer_api.agent_runtime:math.isfinite", args=[1e300]
    )
    lines = (
        request + "\n" + json.dumps(valid) + "\n" + json.dumps(dict(op="close")) + "\n"
    )
    persistent = _cli(tmp_path, "session", input=lines)
    results = [json.loads(line) for line in persistent.stdout.splitlines()]
    assert persistent.returncode == 1 and len(results) == 3
    assert not results[0]["ok"] and results[1] == {"ok": True, "result": True}
    assert results[2]["result"]["closed"]
