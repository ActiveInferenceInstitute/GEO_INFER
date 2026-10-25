"""Offline identity checks for GEO-INFER and its published companion sources.

This checks Git identities, metadata and declared pins. Numerical interchange,
native Lean proofs and anonymous public-download verification are separate gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tomllib
from typing import Any

MANIFEST_PATH = ".github/companion-releases.json"
MAX_INPUT_BYTES = 128 * 1024
REPOSITORIES = {
    "geo": "ActiveInferenceInstitute/GEO_INFER",
    "gnn": "ActiveInferenceInstitute/Generalized_Notation_Notation",
    "fep": "ActiveInferenceInstitute/fep_formal",
}
INTERCHANGE_SCHEMAS = (
    "gnn-geo-infer/1",
    "gnn-geo-infer/2",
    "gnn-geo-infer/factored/1",
)
EVIDENCE_BOUNDARIES = {
    "gnn_release_pins_this_geo_release": False,
    "fep_proves_geo_implementation": False,
    "doi_metadata_verified_by_this_gate": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git(root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _require(result.returncode == 0, "required local Git identity is unavailable")
    _require(len(result.stdout) <= 2 * 1024 * 1024, "Git response exceeds limit")
    return result.stdout


def _oid(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _keys(value: object, expected: set[str], label: str) -> None:
    _require(
        isinstance(value, dict) and set(value) == expected, f"invalid {label} keys"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(data: bytes) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_unique_object)
    _require(isinstance(value, dict), "JSON input must be an object")
    return value


def _tracked_bytes(root: Path, relative: str) -> bytes:
    path = root / relative
    _require(stat.S_ISREG(path.lstat().st_mode), f"nonregular input: {relative}")
    _require(path.stat().st_size <= MAX_INPUT_BYTES, f"oversized input: {relative}")
    data = path.read_bytes()
    _require(
        data == _git(root, "show", f"HEAD:{relative}"), f"uncommitted input: {relative}"
    )
    return data


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _keys(
        manifest,
        {"schema", "geo", "companions", "interchange_schemas", "evidence_boundaries"},
        "manifest",
    )
    _require(
        manifest["schema"] == "geo-infer-companion-releases/v1",
        "unknown manifest schema",
    )
    _keys(manifest["geo"], {"repository", "target_version", "baseline_revision"}, "GEO")
    geo = manifest["geo"]
    _require(geo["repository"] == REPOSITORIES["geo"], "unexpected GEO repository")
    _require(_oid(geo["baseline_revision"]), "invalid GEO baseline")
    _require(
        isinstance(geo["target_version"], str)
        and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", geo["target_version"]) is not None,
        "invalid GEO target version",
    )
    _keys(manifest["companions"], {"gnn", "fep"}, "companions")
    for name, companion in manifest["companions"].items():
        _keys(
            companion,
            {
                "repository",
                "version",
                "tag",
                "revision",
                "tree",
                "tag_object",
                "source_doi",
            },
            name,
        )
        _require(
            companion["repository"] == REPOSITORIES[name],
            f"unexpected {name} repository",
        )
        _require(
            isinstance(companion["version"], str)
            and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", companion["version"])
            is not None,
            f"invalid {name} version",
        )
        _require(companion["tag"] == "v" + companion["version"], f"invalid {name} tag")
        for key in ("revision", "tree", "tag_object"):
            _require(_oid(companion[key]), f"invalid {name} {key}")
        _require(
            isinstance(companion["source_doi"], str)
            and re.fullmatch(r"10\.5281/zenodo\.[1-9][0-9]*", companion["source_doi"])
            is not None,
            f"invalid {name} source DOI",
        )
    _require(
        manifest["interchange_schemas"] == list(INTERCHANGE_SCHEMAS),
        "unsupported interchange schema roster",
    )
    boundaries = manifest["evidence_boundaries"]
    _keys(boundaries, set(EVIDENCE_BOUNDARIES), "evidence boundaries")
    _require(
        all(boundaries[key] is False for key in EVIDENCE_BOUNDARIES),
        "unsupported evidence claim",
    )


def _snapshot(root: Path, tag: str | None = None) -> dict[str, Any]:
    _require(root.is_dir(), "checkout directory is absent")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    _require(top == root, "input must be a Git checkout root")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    _require(_oid(revision) and _oid(tree), "invalid checkout identity")
    _require(
        not _git(root, "status", "--porcelain=v1", "--untracked-files=normal"),
        "checkout is not clean",
    )
    result: dict[str, Any] = {"revision": revision, "tree": tree, "clean": True}
    if tag is not None:
        ref = f"refs/tags/{tag}"
        _require(
            _git(root, "cat-file", "-t", ref).strip() == b"tag",
            "companion tag must be annotated",
        )
        result["tag_object"] = _git(root, "rev-parse", ref).decode().strip()
        result["tagged_revision"] = (
            _git(root, "rev-parse", f"{ref}^{{commit}}").decode().strip()
        )
        tag_data = _git(root, "cat-file", "-p", ref)
        headers = tag_data.split(b"\n\n", 1)[0].splitlines()
        _require(
            headers[:3]
            == [f"object {revision}".encode(), b"type commit", f"tag {tag}".encode()],
            "annotated tag must directly name the declared commit and tag",
        )
    return result


def validate_companion_releases(
    geo_root: Path,
    gnn_root: Path,
    fep_root: Path,
    *,
    expected_geo_revision: str | None = None,
) -> dict[str, Any]:
    """Check three clean local sources without executing their scientific code.

    The GEO manifest is read from its tracked HEAD. The two companions must be
    at their exact released revisions with exact annotated tag objects. The
    result establishes local identity and pin compatibility, not remote origin,
    DOI metadata, numerical agreement, native proof or publication acceptance.
    """
    roots = {
        "geo": geo_root.resolve(strict=True),
        "gnn": gnn_root.resolve(strict=True),
        "fep": fep_root.resolve(strict=True),
    }
    _require(len(set(roots.values())) == 3, "three distinct checkouts are required")
    geo_before = _snapshot(roots["geo"])
    manifest_data = _tracked_bytes(roots["geo"], MANIFEST_PATH)
    manifest = _json(manifest_data)
    _validate_manifest(manifest)
    companions = manifest["companions"]
    before = {"geo": geo_before}
    for name, declaration in companions.items():
        before[name] = _snapshot(roots[name], declaration["tag"])
        for key in ("revision", "tree", "tag_object"):
            _require(
                before[name][key] == declaration[key],
                f"{name} {key} differs from release manifest",
            )
        _require(
            before[name]["tagged_revision"] == declaration["revision"],
            f"{name} tag commit differs",
        )
    if expected_geo_revision is not None:
        _require(_oid(expected_geo_revision), "invalid expected GEO revision")
        _require(
            geo_before["revision"] == expected_geo_revision,
            "unexpected GEO event revision",
        )
    baseline = manifest["geo"]["baseline_revision"]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, geo_before["revision"]],
        cwd=roots["geo"],
        env=dict(os.environ, GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0"),
        capture_output=True,
        timeout=30,
        check=False,
    )
    _require(
        ancestry.returncode == 0,
        "GEO source does not retain the published GNN baseline ancestry",
    )
    inputs: dict[str, str] = {
        f"geo:{MANIFEST_PATH}": hashlib.sha256(manifest_data).hexdigest()
    }
    raw_inputs: list[tuple[str, str, bytes]] = [("geo", MANIFEST_PATH, manifest_data)]

    def read(name: str, relative: str) -> bytes:
        data = _tracked_bytes(roots[name], relative)
        raw_inputs.append((name, relative, data))
        inputs[f"{name}:{relative}"] = hashlib.sha256(data).hexdigest()
        return data

    for name in roots:
        metadata = tomllib.loads(read(name, "pyproject.toml").decode("utf-8"))
        expected = (
            manifest["geo"]["target_version"]
            if name == "geo"
            else companions[name]["version"]
        )
        _require(
            metadata.get("project", {}).get("version") == expected,
            f"{name} project version differs",
        )
        citation = read(name, "CITATION.cff").decode("utf-8")
        versions = re.findall(
            r"^version:\s*[\"']?([0-9]+\.[0-9]+\.[0-9]+)[\"']?\s*$",
            citation,
            re.MULTILINE,
        )
        _require(versions == [expected], f"{name} citation version differs")
    pin_expectations = (
        (
            "geo",
            ".github/gnn-pair.json",
            REPOSITORIES["gnn"],
            companions["gnn"]["revision"],
        ),
        (
            "gnn",
            ".github/fep-lean-pair.json",
            REPOSITORIES["fep"],
            companions["fep"]["revision"],
        ),
        ("gnn", ".github/gnn-pair.json", REPOSITORIES["geo"], baseline),
    )
    for name, relative, repository, revision in pin_expectations:
        pin = _json(read(name, relative))
        _keys(pin, {"repository", "revision"}, f"{name} companion pin")
        _require(
            pin == {"repository": repository, "revision": revision},
            f"{name} companion pin differs",
        )
    for name, relative, original in raw_inputs:
        _require(
            _tracked_bytes(roots[name], relative) == original,
            "input changed during identity review",
        )
    after = {"geo": _snapshot(roots["geo"])}
    after.update(
        {
            name: _snapshot(roots[name], declaration["tag"])
            for name, declaration in companions.items()
        }
    )
    _require(before == after, "source identity changed during review")
    return {
        "schema": "geo-infer-companion-release-identity-review/v1",
        "status": "passed_local_identity_and_pin_review",
        "source": before,
        "manifest_sha256": inputs[f"geo:{MANIFEST_PATH}"],
        "input_sha256": inputs,
        "interchange_schemas": list(INTERCHANGE_SCHEMAS),
        "source_dois": {
            name: declaration["source_doi"] for name, declaration in companions.items()
        },
        "gnn_published_geo_baseline": baseline,
        "geo_retains_baseline_ancestry": True,
        "source_and_input_bookends_equal": True,
        "claims": {
            "local_identity_and_declared_pins": True,
            "numerical_interchange": False,
            "native_Lean_proof": False,
            "public_bytes_or_DOI_metadata": False,
        },
    }
