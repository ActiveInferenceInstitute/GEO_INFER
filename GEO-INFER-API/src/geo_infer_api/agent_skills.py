"""Reproducible, wheel-packaged skill guidance for script, CLI and MCP clients."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from functools import lru_cache
from pathlib import Path
from typing import Any

from geo_infer_api.agent_catalog import MODULES


def build_bundle(repository: Path) -> dict[str, Any]:
    """Capture the actual root/module skills and shared agent guide as data."""
    sources = {"ROOT": "SKILL.md"}
    sources.update({module: f"GEO-INFER-{module}/SKILL.md" for module in MODULES})
    sources["GUIDE"] = "GEO-INFER-INTRA/docs/integration/agent_interfaces.md"
    entries = {}
    for name, relative in sources.items():
        raw = (repository / relative).read_text(encoding="utf-8").encode("utf-8")
        entries[name] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
    return {"schema": "geo-infer-agent-skills/1", "entries": entries}


def bundle_bytes(repository: Path) -> bytes:
    """Encode an exact, deterministic package asset without timestamps."""
    return (
        json.dumps(
            build_bundle(repository), sort_keys=True, ensure_ascii=True, indent=2
        )
        + "\n"
    ).encode("utf-8")


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    bundle = json.loads(
        files("geo_infer_api").joinpath("agent_skills.json").read_text(encoding="utf-8")
    )
    if bundle["schema"] != "geo-infer-agent-skills/1":
        raise ValueError("Unsupported agent skill bundle")
    return bundle


def read_guidance(name: str) -> dict[str, Any]:
    """Read hash-verified LF-normalized guidance from package resources."""
    bundle = _bundle()
    normalized = name.upper().removeprefix("GEO-INFER-").removeprefix("GEO_INFER_")
    entry = bundle["entries"][normalized]
    if hashlib.sha256(entry["text"].encode("utf-8")).hexdigest() != entry["sha256"]:
        raise ValueError("Agent guidance hash mismatch")
    return {"name": normalized, **entry}


def skill_inventory() -> list[dict[str, Any]]:
    """Discover packaged skill references, including uninstalled domain modules."""
    return [
        {key: value for key, value in read_guidance(name).items() if key != "text"}
        for name in ("ROOT", *MODULES)
    ]


def main(argv: list[str] | None = None) -> int:
    """Export/check the package asset after skills or guide source changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    content = bundle_bytes(args.repo)
    if args.check:
        return 0 if args.output.read_bytes() == content else 1
    args.output.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
