"""Machine-readable CLI for the shared GEO-INFER agent session."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any, TextIO

from geo_infer_api.agent_runtime import AgentSession
from geo_infer_api.agent_stdio import protocol_stdout, protocol_stdio
from geo_infer_api import __version__

MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _load(source: str) -> Any:
    if source == "-":
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    elif source.startswith("@"):
        with Path(source[1:]).open("rb") as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
    else:
        raw = source.encode("utf-8")
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Request exceeds 8 MiB")
    return _parse(raw)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse(raw: bytes) -> Any:
    return json.loads(
        raw,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_fields,
    )


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Use a tagged float for a nonfinite JSON number")
    return number


async def _session(
    reader: TextIO, writer: TextIO, *, max_handles: int, max_items: int
) -> int:
    """Execute sequential JSON Lines in one event loop with persistent handles."""
    session = AgentSession(max_handles=max_handles, max_items=max_items)
    status = 0
    try:
        while raw := reader.buffer.readline(MAX_REQUEST_BYTES + 1):
            fatal = len(raw) > MAX_REQUEST_BYTES
            if not raw.strip() and not fatal:
                continue
            try:
                if fatal:
                    raise ValueError("Request exceeds 8 MiB; session input is closed")
                result = await session.aexecute(_parse(raw))
            except (Exception, SystemExit) as error:
                result = session._error(error)
            writer.write(json.dumps(result, allow_nan=False, ensure_ascii=True) + "\n")
            writer.flush()
            if not result["ok"]:
                status = 1
            if fatal or session.closed:
                break
    finally:
        try:
            await session.aclose()
        except (Exception, SystemExit) as error:
            sys.stderr.write(json.dumps(session._error(error)) + "\n")
            status = 1
    return status


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Use a tagged float instead of non-JSON constant {value}")


def main(argv: list[str] | None = None) -> int:
    """Execute a JSON request or a discoverable short CLI command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"geo-infer {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("modules", help="List all 45 modules and installation status")
    commands.add_parser("schema", help="Print the complete JSON request schema")
    commands.add_parser(
        "doctor", help="Inspect actual runtime, optional dependencies and session facts"
    )
    persistent = commands.add_parser(
        "session", help="Serve JSON Lines with persistent native handles"
    )
    persistent.add_argument("--max-handles", type=int, default=256)
    persistent.add_argument("--max-items", type=int, default=10000)
    commands.add_parser(
        "skills", help="List 46 packaged, hash-verified skill references"
    )
    skill = commands.add_parser("skill", help="Read a packaged module skill or ROOT")
    skill.add_argument("module")
    commands.add_parser("guide", help="Read packaged agent interface instructions")
    discovery = commands.add_parser(
        "catalog", help="Search import-free public definitions"
    )
    discovery.add_argument("module")
    discovery.add_argument("--query", default="")
    discovery.add_argument("--offset", type=int, default=0)
    discovery.add_argument("--limit", type=int, default=50)
    descriptor = commands.add_parser("describe", help="Inspect a runtime API signature")
    descriptor.add_argument("target")
    getter = commands.add_parser("get", help="Read a public value or indexed window")
    getter.add_argument("target")
    getter.add_argument(
        "--path", default="[]", help="JSON index array, @file, or - for stdin"
    )
    call = commands.add_parser("call", help="Call a function or constructor")
    call.add_argument("target")
    call.add_argument("--args", default="[]", help="JSON array, @file, or - for stdin")
    call.add_argument(
        "--kwargs", default="{}", help="JSON object, @file, or - for stdin"
    )
    request = commands.add_parser(
        "run", help="Execute a request or batch in one session"
    )
    request.add_argument("request", help="JSON object, @file, or - for stdin")
    commands.add_parser("mcp", help="Run the optional stdio MCP server")
    options = parser.parse_args(argv)
    if options.command == "mcp":
        from geo_infer_api.agent_mcp import main as run_mcp

        return run_mcp()
    if options.command == "session":
        if options.max_handles < 1 or options.max_items < 1:
            parser.error("Session bounds must be positive")
        with protocol_stdio() as (reader, writer):
            return asyncio.run(
                _session(
                    reader,
                    writer,
                    max_handles=options.max_handles,
                    max_items=options.max_items,
                )
            )
    try:
        if options.command == "doctor":
            operation = {"op": "diagnostics"}
        elif options.command == "run":
            operation = _load(options.request)
        elif options.command == "get":
            operation = {
                "op": "get",
                "target": options.target,
                "path": _load(options.path),
            }
        elif options.command == "call":
            if options.args == options.kwargs == "-":
                raise ValueError("stdin may supply only one of --args and --kwargs")
            operation = {
                "op": "call",
                "target": options.target,
                "args": _load(options.args),
                "kwargs": _load(options.kwargs),
            }
        else:
            operation = {
                "op": options.command,
                **{
                    key: value
                    for key, value in vars(options).items()
                    if key != "command"
                },
            }
        with protocol_stdout(), AgentSession() as session:
            result = session.execute(operation)
    except (Exception, SystemExit) as error:
        result = {
            "ok": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    sys.stdout.write(json.dumps(result, allow_nan=False, ensure_ascii=True) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
