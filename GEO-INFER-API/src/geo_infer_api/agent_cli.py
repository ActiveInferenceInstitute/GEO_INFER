"""Machine-readable CLI for the shared GEO-INFER agent session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from geo_infer_api.agent_runtime import AgentSession
from geo_infer_api.agent_stdio import protocol_stdout

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
    return json.loads(raw, parse_constant=lambda value: _invalid_constant(value))


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Use a tagged float instead of non-JSON constant {value}")


def main(argv: list[str] | None = None) -> int:
    """Execute a JSON request or a discoverable short CLI command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("modules", help="List all 45 modules and installation status")
    commands.add_parser("schema", help="Print the complete JSON request schema")
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
    try:
        if options.command == "run":
            operation = _load(options.request)
        elif options.command == "call":
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
        with protocol_stdout():
            result = AgentSession().execute(operation)
    except (ValueError, OSError) as error:
        result = {
            "ok": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    sys.stdout.write(json.dumps(result, allow_nan=False, ensure_ascii=True) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
