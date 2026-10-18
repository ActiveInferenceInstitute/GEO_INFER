"""Persistent transports and real native context lifecycle without scientific extras."""

import asyncio
import contextlib
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("suppress", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_failed_enter_unwinds_only_new_context_at_handle_capacity(
    tmp_path, asynchronous, suppress, cleanup_failure
):
    events = []
    observed = []
    files = []

    class Resource:
        def enter(self):
            events.append("enter")
            stream = (tmp_path / "native.txt").open("w")
            files.append(stream)
            return stream

        def exit(self, kind, error, trace):
            events.append("exit")
            observed.append((kind, error, trace))
            files[-1].close()
            if cleanup_failure:
                raise OSError("native cleanup failed")
            return suppress

    class SyncManager(Resource):
        def __enter__(self):
            return self.enter()

        def __exit__(self, *error):
            return self.exit(*error)

    class AsyncManager(Resource):
        async def __aenter__(self):
            await asyncio.sleep(0)
            return self.enter()

        async def __aexit__(self, *error):
            await asyncio.sleep(0)
            return self.exit(*error)

    @contextlib.contextmanager
    def outer():
        yield
        events.append("outer exit")

    async def scenario():
        async with AgentSession(max_handles=3) as session:
            manager = AsyncManager() if asynchronous else SyncManager()
            session.objects.update(manager=manager, alias=manager, outer=outer())
            assert (await session.aexecute(dict(op="enter", handle="outer")))["ok"]
            result = await session.aexecute(dict(op="enter", handle="manager"))
            assert result["ok"] is False
            assert result["error"]["type"] == (
                "OSError" if cleanup_failure else "ValueError"
            )
            assert files[0].closed
            assert events == ["enter", "exit"]
            assert list(session.contexts) == ["outer"]
            assert session.objects["manager"] is session.objects["alias"] is manager
            assert set(session.objects) == {"manager", "alias", "outer"}
        assert events == ["enter", "exit", "outer exit"]

    asyncio.run(scenario())
    assert len(observed) == 1
    kind, error, trace = observed[0]
    assert kind is ValueError and "Handle limit reached" in str(error)
    assert trace is not None


def test_enter_at_capacity_reuses_existing_native_value(tmp_path):
    stream = (tmp_path / "native.txt").open("w")
    with AgentSession(max_handles=1) as session:
        session.objects["stream"] = stream
        result = session.execute(dict(op="enter", handle="stream"))
        assert result["ok"] and result["result"]["$ref"] == "stream"
        assert list(session.contexts) == ["stream"]
    assert stream.closed


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


def test_session_preserves_native_sync_exception_suppression():
    observed = []

    @contextlib.contextmanager
    def outer():
        try:
            yield
        except BaseException as error:
            observed.append(type(error))
            raise
        else:
            observed.append(None)

    with AgentSession() as session:
        session.objects.update(outer=outer(), inner=contextlib.suppress(ValueError))
        assert session.execute(dict(op="enter", handle="outer"))["ok"]
        assert session.execute(dict(op="enter", handle="inner"))["ok"]
        raise ValueError("suppressed by the native inner context")
    assert observed == [None]
    assert session.closed and not session.contexts and not session.objects


def test_session_preserves_native_async_exception_suppression():
    observed = []

    @contextlib.asynccontextmanager
    async def manager():
        try:
            yield
        except ValueError as error:
            observed.append(str(error))

    async def scenario():
        async with AgentSession() as session:
            session.objects["manager"] = manager()
            assert (await session.aexecute(dict(op="enter", handle="manager")))["ok"]
            raise ValueError("suppressed asynchronously")
        assert session.closed and not session.contexts and not session.objects

    asyncio.run(scenario())
    assert observed == ["suppressed asynchronously"]


@pytest.mark.parametrize(
    "selection",
    [
        dict(target=""),
        dict(handle=""),
        dict(target="geo_infer_api:__version__", handle=""),
        dict(target="", handle="existing"),
        dict(handle="existing", member=""),
    ],
)
def test_empty_selections_fail_before_any_batch_side_effects(selection):
    with AgentSession() as session:
        request = dict(
            op="batch",
            steps=[
                dict(
                    op="call",
                    target="geo_infer_api.agent_catalog:package_name",
                    args=["API"],
                    save_as="would-be-created",
                ),
                dict(op="get", **selection),
            ],
        )
        result = session.execute(request)
        assert not result["ok"] and result["error"]["type"] == "ValidationError"
        assert not session.objects


@pytest.mark.parametrize("operator", ["$ref", "$type", "$callable", "$literal"])
def test_reserved_result_keys_retain_native_data_and_support_index_selection(operator):
    from datetime import UTC, datetime

    payload = {
        operator: "business-data",
        "timestamp": datetime(2026, 10, 17, tzinfo=UTC),
        "measurement": float("inf"),
    }
    with AgentSession(max_handles=1) as session:
        encoded = session.encode(payload)
        assert session.decode(encoded) is payload
        assert (
            session.execute(dict(op="get", handle=encoded["$ref"]))["result"] == encoded
        )
        assert session.execute(
            dict(op="get", handle=encoded["$ref"], path=[operator])
        ) == dict(ok=True, result="business-data")
        assert (
            session.decode({**encoded, "path": ["timestamp"]}) is payload["timestamp"]
        )
        assert session.decode({"$literal": {operator: "business-data"}}) == {
            operator: "business-data"
        }
        assert len(session.objects) == 1


def test_nested_reserved_result_keys_retain_the_complete_native_result():
    payload = {"results": [{"$ref": "customer-record", "amount": 5}]}
    with AgentSession() as session:
        encoded = session.encode(payload)
        assert session.decode(encoded) is payload
        assert (
            session.decode({**encoded, "path": ["results", 0, "$ref"]})
            == "customer-record"
        )
        assert len(session.objects) == 1


def test_jsonl_composes_reserved_result_maps_and_explicit_literals(tmp_path):
    payload = {"$ref": "customer-record", "amount": 5}
    requests = [
        dict(
            op="call",
            target="geo_infer_api.agent_schema:json.loads",
            args=[json.dumps(payload)],
            save_as="payload",
        ),
        dict(op="get", handle="payload"),
        dict(op="get", handle="payload", path=["$ref"]),
        dict(
            op="call",
            target="geo_infer_api.agent_schema:json.dumps",
            args=[{"$ref": "payload"}],
        ),
        dict(
            op="call",
            target="geo_infer_api.agent_schema:json.dumps",
            args=[{"$literal": payload}],
        ),
    ]
    result = _cli(
        tmp_path, "session", input="".join(json.dumps(r) + "\n" for r in requests)
    )
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 5 and all(row["ok"] for row in responses)
    assert responses[1]["result"]["$ref"] == "payload"
    assert responses[2]["result"] == "customer-record"
    assert json.loads(responses[3]["result"]) == payload
    assert json.loads(responses[4]["result"]) == payload


def test_real_mcp_rejects_raw_coercion_typos_and_conflicting_selections(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    output = tmp_path / "must-not-be-written.txt"

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "geo_infer_api.agent_mcp"],
            cwd=str(tmp_path),
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as client,
        ):
            await client.initialize()
            invalid = [
                ("geo_catalog", dict(module="API", limit=True)),
                ("geo_catalog", dict(module="API", limit="1")),
                ("geo_catalog", dict(module="API", offset="0")),
                ("geo_catalog", dict(module="API", query=None)),
                ("geo_catalog", dict(module="API", limt=1)),
                ("geo_catalog", dict(module="API", limt=None)),
                ("geo_modules", dict(unexpected=True)),
                ("geo_modules", dict(op="close")),
                (
                    "geo_call",
                    dict(
                        target="geo_infer_api.agent_catalog:module_inventory", args="[]"
                    ),
                ),
                (
                    "geo_call",
                    dict(
                        target="geo_infer_api.agent_catalog:module_inventory",
                        kwargs="{}",
                    ),
                ),
                (
                    "geo_get",
                    dict(target="geo_infer_api.agent_catalog:MODULES", handle=""),
                ),
                (
                    "geo_describe",
                    dict(target="geo_infer_api.agent_catalog:MODULES", handle="path"),
                ),
            ]
            for tool, arguments in invalid:
                result = await client.call_tool(tool, arguments)
                assert result.isError, (tool, arguments, result)
                assert result.structuredContent["ok"] is False
                assert result.structuredContent["error"]["message"]

            created = await client.call_tool(
                "geo_call",
                dict(
                    target="geo_infer_api.agent_cli:Path",
                    args=[str(output)],
                    save_as="path",
                ),
            )
            assert not created.isError
            rejected = await client.call_tool(
                "geo_call",
                dict(
                    handle="path",
                    member="write_text",
                    args=["unexpected"],
                    saveAs="typo",
                ),
            )
            assert rejected.isError and not output.exists()

            # Explicit nulls on advertised optional fields keep their existing
            # meaning, while an unknown null-valued field above is an error.
            inventory = await client.call_tool(
                "geo_call",
                dict(
                    target="geo_infer_api.agent_catalog:module_inventory",
                    handle=None,
                    member=None,
                    args=None,
                    kwargs=None,
                    save_as=None,
                ),
            )
            assert not inventory.isError
            assert len(inventory.structuredContent["result"]) == 45
            selected = await client.call_tool(
                "geo_get",
                dict(
                    target="geo_infer_api.agent_catalog:MODULES",
                    handle=None,
                    member=None,
                    path=None,
                    save_as=None,
                ),
            )
            assert (
                not selected.isError and len(selected.structuredContent["result"]) == 45
            )
            valid = await client.call_tool(
                "geo_call",
                dict(handle="path", member="write_text", args=["authorized"]),
            )
            assert not valid.isError
            assert output.read_text() == "authorized"

    asyncio.run(scenario())


def _cli(tmp_path, *arguments, input=None):
    return subprocess.run(
        [sys.executable, "-I", "-m", "geo_infer_api.agent_cli", *arguments],
        cwd=tmp_path,
        input=input,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("transport", ["cli", "mcp"])
def test_real_transport_failed_enter_closes_yielded_file_at_capacity(
    tmp_path, transport
):
    capacity = 2 if transport == "cli" else 256
    requests = [
        dict(
            op="call",
            target="geo_infer_api.agent_runtime:Path",
            args=[str(tmp_path / "native.txt")],
            save_as="path",
        ),
        dict(op="call", handle="path", member="open", args=["w"], save_as="file"),
        dict(op="release", handle="path"),
        dict(
            op="call",
            target="geo_infer_api.agent_runtime:contextlib.closing",
            args=[{"$ref": "file"}],
            save_as="manager",
        ),
        dict(op="release", handle="file"),
    ]
    aliases = [
        dict(op="get", handle="manager", save_as=f"alias-{index}")
        for index in range(capacity - 1)
    ]
    requests.extend(
        dict(op="batch", steps=aliases[start : start + 100])
        for start in range(0, len(aliases), 100)
    )
    failed_index = len(requests)
    requests.extend(
        [
            dict(op="enter", handle="manager"),
            dict(op="diagnostics"),
            dict(op="release", handle="alias-0"),
            dict(op="get", handle="manager", member="thing", save_as="file"),
            dict(op="get", handle="file", member="closed"),
            dict(op="close"),
        ]
    )
    if transport == "cli":
        run = _cli(
            tmp_path,
            "session",
            "--max-handles",
            str(capacity),
            input="\n".join(json.dumps(request) for request in requests) + "\n",
        )
        assert run.returncode == 1, run.stderr
        responses = [json.loads(line) for line in run.stdout.splitlines()]
    else:
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
                ClientSession(reader, writer) as client,
            ):
                await client.initialize()
                responses = []
                for request in requests:
                    operation = request["op"]
                    arguments = {k: v for k, v in request.items() if k != "op"}
                    if operation == "enter":
                        operation = "context"
                        arguments["action"] = "enter"
                    result = await client.call_tool("geo_" + operation, arguments)
                    responses.append(result.structuredContent)
                    assert result.isError is (not result.structuredContent["ok"])
                return responses

        responses = asyncio.run(scenario())
    assert len(responses) == len(requests)
    assert [i for i, row in enumerate(responses) if not row["ok"]] == [failed_index]
    assert responses[failed_index]["error"]["type"] == "ValueError"
    facts = responses[failed_index + 1]["result"]["session"]
    assert facts["handles"] == capacity and facts["active_contexts"] == 0
    assert responses[failed_index + 4] == {"ok": True, "result": True}
    assert responses[-1]["result"]["exited_contexts"] == 0


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


def test_native_sync_protocol_bypasses_instance_attribute_hooks():
    events = []

    class Manager:
        def __getattribute__(self, name):
            if name in {"__enter__", "__exit__", "__aenter__", "__aexit__"}:
                raise RuntimeError("special methods must be resolved on the type")
            return object.__getattribute__(self, name)

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *error):
            events.append("exit")

    with Manager():
        pass
    assert events == ["enter", "exit"]
    events.clear()
    with AgentSession() as session:
        session.objects["manager"] = Manager()
        assert session.execute(dict(op="enter", handle="manager", save_as="value"))[
            "ok"
        ]
    assert events == ["enter", "exit"]
    assert session.closed and not session.contexts and not session.objects


@pytest.mark.parametrize("explicit_exit", [False, True])
def test_native_exit_is_bound_before_entry_and_survives_method_replacement(
    explicit_exit,
):
    events = []

    class Manager:
        def __enter__(self):
            Manager.__exit__ = lambda *args: events.append("replacement")
            return self

        def __exit__(self, *error):
            events.append("original")

    with AgentSession() as session:
        session.objects["manager"] = Manager()
        assert session.execute(dict(op="enter", handle="manager", save_as="value"))[
            "ok"
        ]
        if explicit_exit:
            assert session.execute(dict(op="exit", handle="manager"))["ok"]
    assert events == ["original"]


def test_incomplete_native_protocol_is_rejected_before_enter_side_effects():
    events = []

    class Incomplete:
        def __enter__(self):
            events.append("entered")
            return self

    with AgentSession() as session:
        session.objects["manager"] = Incomplete()
        result = session.execute(dict(op="enter", handle="manager", save_as="value"))
        assert not result["ok"] and result["error"]["type"] == "TypeError"
        assert not events and not session.contexts


def test_native_async_protocol_bypasses_instance_attribute_hooks():
    events = []

    class Manager:
        def __getattribute__(self, name):
            if name in {"__aenter__", "__aexit__"}:
                raise RuntimeError("special methods must be resolved on the type")
            return object.__getattribute__(self, name)

        async def __aenter__(self):
            events.append("enter")
            return self

        async def __aexit__(self, *error):
            events.append("exit")

    async def scenario():
        async with AgentSession() as session:
            session.objects["manager"] = Manager()
            assert (
                await session.aexecute(
                    dict(op="enter", handle="manager", save_as="value")
                )
            )["ok"]
        assert session.closed and not session.contexts

    asyncio.run(scenario())
    assert events == ["enter", "exit"]
