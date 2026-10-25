"""Exact released Torch CUDA12 variant custody; no imports or native execution.

This single reviewed variant maps to its public release's advisory identity.
The mapping does not certify the wheel, ABI, CUDA operation or GPU hardware.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tomllib

CONTRACT_SHA256 = "f95c4cfd6bced5fee0bd40dfef64c1ec48f11f1bcd5c68cb3e042afe40295b29"
CONSTRAINT = (
    "torch==2.13.0+cu126; sys_platform == 'linux' and platform_machine == 'x86_64'"
)


def official_variant_contract() -> dict:
    """Read the immutable, hash-bound release record instead of accepting a URL."""
    path = Path(__file__).with_name("torch_cu126_variant.json")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError("Reviewed Torch variant contract changed")
    return json.loads(raw)


def _binding(root: Path, contract: dict) -> None:
    """Require the reviewed Linux-x86-only source and explicit index contract."""
    project = tomllib.loads((root / "pyproject.toml").read_text())
    settings = tomllib.loads((root / "uv.toml").read_text())
    uv = project.get("tool", {}).get("uv", {})
    expected_source = {
        "index": contract["index_name"],
        "marker": contract["marker"],
    }
    expected_index = {
        "name": contract["index_name"],
        "url": contract["source"]["registry"],
        "explicit": True,
    }
    if uv.get("sources", {}).get("torch") != expected_source:
        raise ValueError("Torch source is not the reviewed Linux-x86 binding")
    if any(
        CONSTRAINT not in value.get("constraint-dependencies", [])
        for value in (uv, settings)
    ):
        raise ValueError("Exact platform-scoped Torch variant constraint missing")
    for indices in (uv.get("index", []), settings.get("index", [])):
        named = [row for row in indices if row.get("name") == contract["index_name"]]
        if named != [expected_index]:
            raise ValueError("Explicit Torch index settings differ")
    for path in root.glob("GEO-INFER-*/pyproject.toml"):
        member = tomllib.loads(path.read_text())
        if "torch" in member.get("tool", {}).get("uv", {}).get("sources", {}):
            raise ValueError("A workspace member overrides the root Torch source")


def official_variant_record(root: Path, package: dict) -> dict:
    """Validate one full locked variant and retain its explicit advisory mapping."""
    contract = official_variant_contract()
    if any(package.get(key) != contract[key] for key in ("name", "version", "source")):
        raise ValueError("Unreviewed non-PyPI dependency source or Torch variant")
    _binding(root, contract)
    wheels = package.get("wheels")
    if package.get("sdist") or not isinstance(wheels, list) or not wheels:
        raise ValueError("Reviewed Torch variant requires locked wheels only")
    admitted, seen = [], set()
    for wheel in wheels:
        url, digest = wheel.get("url"), wheel.get("hash")
        matches = [
            (filename, row)
            for filename, row in contract["wheels"].items()
            if row["url"] == url and "sha256:" + row["sha256"] == digest
        ]
        if len(matches) != 1 or url in seen:
            raise ValueError("Torch wheel is absent, duplicated or not release-bound")
        seen.add(url)
        filename, reviewed = matches[0]
        admitted.append({"filename": filename, **wheel, "release_record": reviewed})
    for minor in ("311", "312"):
        filename = f"torch-2.13.0+cu126-cp{minor}-cp{minor}-manylinux_2_28_x86_64.whl"
        if not any(row["filename"] == filename for row in admitted):
            raise ValueError("A required Linux Python Torch wheel is missing")
    return {
        "locked_identity": {key: package[key] for key in ("name", "version", "source")},
        "resolution_markers": package.get("resolution-markers", []),
        "advisory_identity": contract["advisory_identity"],
        "variant_provenance": contract["release_provenance"],
        "contract_sha256": CONTRACT_SHA256,
        "locked_wheels": admitted,
        "claim": "Public-release advisory mapping only; native and artifact acceptance remain separate.",
    }


def provenance_hashes() -> dict[str, str]:
    """Bind the actual owner implementation and release-record bytes in receipts."""
    paths = [Path(__file__), Path(__file__).with_name("torch_cu126_variant.json")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def locked_torch_constraint(root: Path, target: dict) -> dict | None:
    """Select one current-tag locked URL without installing an optional package."""
    if (target["sys_platform"], target["machine"]) != ("linux", "x86_64"):
        return None
    version = target["version"]
    if (
        target["implementation"] != "cpython"
        or version not in ([3, 11], [3, 12])
        or not re.fullmatch(
            rf"cpython-{version[0]}{version[1]}-x86_64-linux-gnu", target["soabi"] or ""
        )
    ):
        raise ValueError("Unreviewed Linux Torch installation ABI")
    raw = (root / "uv.lock").read_bytes()
    packages = tomllib.loads(raw.decode())["package"]
    contract = official_variant_contract()
    variants = [
        row
        for row in packages
        if row.get("name") == "torch" and row.get("source") == contract["source"]
    ]
    if len(variants) != 1:
        raise ValueError("Exactly one reviewed locked Torch variant is required")
    record = official_variant_record(root, variants[0])
    minor = f"{version[0]}{version[1]}"
    filename = f"torch-2.13.0+cu126-cp{minor}-cp{minor}-manylinux_2_28_x86_64.whl"
    wheel = next(row for row in record["locked_wheels"] if row["filename"] == filename)
    url, digest = wheel["url"], wheel["hash"].removeprefix("sha256:")
    return {
        "requirement": f"torch @ {url}#sha256={digest}",
        "lock_sha256": hashlib.sha256(raw).hexdigest(),
        "target": target,
        "selected_wheel": wheel,
        **record,
    }
