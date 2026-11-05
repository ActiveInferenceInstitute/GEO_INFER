"""Bounded lineage and freshness review for the reviewed GNN interchange pin.

The published companion-identity gate checks that GEO's interchange pin equals
the reviewed declaration. This separate check answers the next question with
local Git reads only: which release the pin descends from, which release tags
point at it, and how far the GNN checkout has moved past it. It never fetches,
executes scientific code or treats tag names as authenticity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

PIN_PATH = ".github/gnn-pair.json"
MANIFEST_PATH = ".github/companion-releases.json"
MAX_INPUT_BYTES = 128 * 1024
MAX_RELEASE_TAGS = 200
MAX_GIT_BYTES = 2 * 1024 * 1024
GNN_REPOSITORY = "ActiveInferenceInstitute/Generalized_Notation_Notation"
RELEASE_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    _require(path.is_file() and not path.is_symlink(), f"missing input: {relative}")
    _require(path.stat().st_size <= MAX_INPUT_BYTES, f"oversized input: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"input must be an object: {relative}")
    return value


def _git(root: Path, *arguments: str) -> str:
    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *arguments],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _require(result.returncode == 0, "required local Git identity is unavailable")
    _require(len(result.stdout) <= MAX_GIT_BYTES, "Git response exceeds limit")
    return result.stdout.decode("utf-8").strip()


def _git_ok(root: Path, *arguments: str) -> bool:
    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *arguments],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _require(len(result.stdout) <= MAX_GIT_BYTES, "Git response exceeds limit")
    return result.returncode == 0


def _commit(root: Path, revision: str) -> str | None:
    """Resolve one revision to its commit, or report it as unavailable."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "rev-parse",
            "--verify",
            "-q",
            f"{revision}^{{commit}}",
        ],
        cwd=root,
        env=dict(os.environ, GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8").strip()


def _release_tags(root: Path) -> list[str]:
    tags = [tag for tag in _git(root, "tag", "--list").splitlines() if tag]
    _require(len(tags) <= MAX_RELEASE_TAGS, "release tag roster exceeds bound")
    releases = [tag for tag in tags if RELEASE_TAG.fullmatch(tag)]
    releases.sort(
        key=lambda tag: tuple(int(part) for part in RELEASE_TAG.fullmatch(tag).groups())
    )
    return releases


def validate_gnn_pin_freshness(
    geo_root: Path,
    gnn_root: Path,
    *,
    gnn_ref: str = "HEAD",
    require_current: bool = False,
) -> dict[str, Any]:
    """Report the reviewed pin's release lineage; never fetch or execute models.

    ``.github/gnn-pair.json`` is the canonical home of the current pin. The pin
    must exist in the GNN checkout and retain the published companion revision
    in the release manifest as an ancestor. Release currency is reported, and
    only ``require_current`` turns a pin behind the newest release tag into a
    refusal.
    """
    geo = geo_root.resolve(strict=True)
    gnn = gnn_root.resolve(strict=True)
    _require(gnn.is_dir(), "GNN checkout is absent")
    pin_data = _read_json(geo, PIN_PATH)
    _require(
        set(pin_data)
        in ({"repository", "revision"}, {"repository", "revision", "tag"}),
        "invalid pin keys",
    )
    _require(pin_data["repository"] == GNN_REPOSITORY, "unexpected pin repository")
    _require(
        isinstance(pin_data["revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", pin_data["revision"]) is not None,
        "invalid pin revision",
    )
    if "tag" in pin_data:
        _require(
            isinstance(pin_data["tag"], str)
            and re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", pin_data["tag"]) is not None,
            "invalid pin tag",
        )
    manifest = _read_json(geo, MANIFEST_PATH)
    _require(
        manifest.get("schema") == "geo-infer-companion-releases/v1",
        "unknown companion manifest schema",
    )
    released = manifest.get("companions", {}).get("gnn", {}).get("revision")
    _require(
        isinstance(released, str)
        and re.fullmatch(r"[0-9a-f]{40}", released) is not None,
        "invalid published companion revision",
    )

    pin_commit = _commit(gnn, pin_data["revision"])
    _require(
        pin_commit == pin_data["revision"],
        "pinned revision is absent from the checkout",
    )
    released_commit = _commit(gnn, released)
    _require(released_commit == released, "published companion revision is absent")
    _require(
        _git_ok(gnn, "merge-base", "--is-ancestor", released_commit, pin_commit),
        "pin does not retain the published companion revision",
    )
    ref_commit = _commit(gnn, gnn_ref)
    _require(ref_commit is not None, "GNN reference revision is absent")

    tags_at_pin = sorted(
        tag for tag in _git(gnn, "tag", "--points-at", pin_commit).splitlines() if tag
    )
    releases = _release_tags(gnn)
    newest_tag = releases[-1] if releases else None
    newest_commit = _commit(gnn, newest_tag) if newest_tag else None
    equals_newest = newest_commit is not None and newest_commit == pin_commit
    newest_contains_pin = (
        newest_commit is not None
        and not equals_newest
        and _git_ok(gnn, "merge-base", "--is-ancestor", pin_commit, newest_commit)
    )
    pin_contains_newest = (
        newest_commit is not None
        and not equals_newest
        and _git_ok(gnn, "merge-base", "--is-ancestor", newest_commit, pin_commit)
    )
    current = equals_newest or pin_contains_newest
    commits_ahead_of_pin = int(
        _git(gnn, "rev-list", "--count", f"{pin_commit}..{ref_commit}")
    )
    commits_behind_newest = (
        int(_git(gnn, "rev-list", "--count", f"{pin_commit}..{newest_commit}"))
        if newest_commit is not None and newest_contains_pin
        else None
    )
    if require_current:
        _require(current, "pin is behind the newest GNN release tag")

    return {
        "schema": "geo-infer-gnn-pin-freshness/v1",
        "status": "passed_pin_freshness_review",
        "pin": {
            "repository": pin_data["repository"],
            "revision": pin_commit,
            "tag": pin_data.get("tag"),
            "tags_at_pin": tags_at_pin,
            "release_tags_at_pin": [
                tag for tag in tags_at_pin if RELEASE_TAG.fullmatch(tag)
            ],
        },
        "published_companion": {
            "revision": released_commit,
            "is_ancestor_of_pin": True,
        },
        "checkout": {
            "ref": gnn_ref,
            "commit": ref_commit,
            "commits_ahead_of_pin": commits_ahead_of_pin,
        },
        "newest_release": {
            "tag": newest_tag,
            "commit": newest_commit,
            "equals_pin": equals_newest,
            "pin_is_ancestor": newest_contains_pin,
            "is_ancestor_of_pin": pin_contains_newest,
            "commits_behind": commits_behind_newest,
            "pin_is_current": current,
        },
        "claims": {
            "local_git_lineage": True,
            "release_currency": current,
            "numerical_interchange": False,
            "native_Lean_proof": False,
        },
    }
