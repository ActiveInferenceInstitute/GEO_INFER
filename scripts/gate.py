#!/usr/bin/env python3
"""GEO-INFER local gate: one entry point, one command string per CI step.

An arm is a named list of the exact commands a GitHub Actions job runs. The
commands are copied from the workflow sources, never re-derived, and
``GEO-INFER-TEST/tests/unit/test_gate_contracts.py`` fails when an arm and its
workflow disagree. GitHub Actions is unavailable to this estate, so these arms
are the gate: run every arm locally before landing a change.

Arms run against the single locked workspace ``.venv`` (``uv sync
--all-packages --all-extras --all-groups``); an arm never creates a second
environment. ``.venv/bin`` is prepended to ``PATH`` so a bare ``python`` means
the workspace interpreter, matching the interpreter CI selects with
``setup-python``.

Usage:
    uv run --frozen python scripts/gate.py --list
    uv run --frozen python scripts/gate.py lint
    uv run --frozen python scripts/gate.py all

Arms: ``lint``, ``docs``, ``fast``, ``gates``, ``tests``, ``gnn``,
``manuscript``, and ``all``. There is no ``types`` arm: the repository declares
``[tool.mypy]`` and a ``mypy`` dependency but runs no type gate, so a type arm
would invent a contract rather than mirror one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / ".geo-infer-test-results"
PIN_FILE = REPO_ROOT / ".github" / "gnn-pair.json"
PIN_SCHEMA = "gnn-importer-pin/v1"
GITLEAKS_VERSION = "8.30.1"

# The exact commands of ci.yml "validate" -> "Run source runtime hygiene".
LINT_COMMANDS = (
    "uv run --with 'ruff>=0.15.6,<0.16' ruff check .",
    "uv run --with 'ruff>=0.15.6,<0.16' ruff format --check .",
)

# ci.yml "validate" -> "Validate repository contracts" documentation steps.
DOCS_COMMANDS = (
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_documentation.py --strict",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/rewrite_readme_agents.py --check",
)

# ci.yml "fast-contracts" -> "Run execution and SPACE TIME composition contracts".
FAST_COMMAND = (
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --timeout 600 --paths"
    " GEO-INFER-TEST/tests/unit/test_execution_contracts.py"
    " GEO-INFER-TEST/tests/unit/test_run_unified_tests.py"
    " GEO-INFER-TEST/tests/unit/test_ci_workflow_contracts.py"
    " GEO-INFER-TEST/tests/unit/test_check_coverage_floor_gate.py"
    " GEO-INFER-TEST/tests/unit/test_documentation_census.py"
    " GEO-INFER-TEST/tests/unit/test_process_ownership_contracts.py"
    " GEO-INFER-TEST/tests/unit/test_measure_module_coverage.py"
    " GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py"
)

# ci.yml "validate" gates that are not documentation or lint.
GATE_COMMANDS = (
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_packaging.py --strict",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_logging_hygiene.py",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_doc_examples.py",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_active_inference_contract.py",
    "python -m compileall GEO-INFER-*/src GEO-INFER-*/examples",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_test_contracts.py --strict",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py --timeout 1200 -- gitleaks detect --source . --config .gitleaks.toml --redact --verbose",
)

# ci.yml "test" -> one leg per interpreter x category (the 3.12 legs here).
TEST_COMMANDS = (
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category unit --timeout 600",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category integration --timeout 600",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category performance --timeout 600",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category system --timeout 600",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --h3-migration",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category slow --timeout 600",
)

# ci.yml "manuscript": producer, convergence, render, then the ROOT profiles.
MANUSCRIPT_COMMANDS = (
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py --timeout 1800 -- python manuscript/generate_research_artifacts.py --verify --rerun-verification",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py --timeout 600 -- python manuscript/generate_research_artifacts.py --check",
    "uv run python scripts/render_manuscript_pdf.py",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ROOT --category manuscript-render --timeout 600 --show-failures",
    "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ROOT --category manuscript --timeout 600 --show-failures",
)


@dataclass(frozen=True)
class Arm:
    name: str
    source: str
    commands: tuple[str, ...]
    needs: tuple[str, ...] = ()


ARMS: dict[str, Arm] = {
    "lint": Arm("lint", "ci.yml: validate > Run source runtime hygiene", LINT_COMMANDS),
    "docs": Arm(
        "docs", "ci.yml: validate > Validate repository contracts", DOCS_COMMANDS
    ),
    "fast": Arm("fast", "ci.yml: fast-contracts", (FAST_COMMAND,)),
    "gates": Arm("gates", "ci.yml: validate (non-lint, non-docs)", GATE_COMMANDS),
    "tests": Arm("tests", "ci.yml: test", TEST_COMMANDS),
    "gnn": Arm(
        "gnn",
        "gnn-interchange.yml: interchange",
        (),
        needs=(
            "geometry: a clean worktree, a GNN checkout at the pin, and that checkout's interpreter",
        ),
    ),
    "manuscript": Arm("manuscript", "ci.yml: manuscript", MANUSCRIPT_COMMANDS),
}

ALL_ORDER = ("lint", "docs", "fast", "gates", "tests", "gnn", "manuscript")


def _shell_env() -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = str(REPO_ROOT / ".venv" / "bin")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env.pop("VIRTUAL_ENV", None)
    return env


def _read_pin() -> dict[str, object]:
    pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    if pin.get("schema") != PIN_SCHEMA:
        raise SystemExit(
            f"gate: {PIN_FILE} schema is {pin.get('schema')!r}, expected {PIN_SCHEMA!r}"
        )
    revision = str(pin.get("revision", ""))
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise SystemExit(
            f"gate: {PIN_FILE} revision {revision!r} is not a full lowercase object id"
        )
    return pin


def _gnn_interchange(args: argparse.Namespace) -> int:
    pin = _read_pin()
    override = args.gnn_repo or os.environ.get("GEO_INFER_GNN_REPO")
    gnn_repo = (
        Path(override).resolve()
        if override
        else (REPO_ROOT.parent / "Generalized_Notation_Notation").resolve()
    )
    if not (gnn_repo / ".git").exists():
        print(
            f"gate gnn: no GNN checkout at {gnn_repo}; pass --gnn-repo or set GEO_INFER_GNN_REPO",
            file=sys.stderr,
        )
        return 2
    # `.venv/bin/python` is a symlink to the managed interpreter; resolving it
    # would run that interpreter outside the venv and lose the checkout's
    # site-packages.
    gnn_python = Path(
        args.gnn_python or str(gnn_repo / ".venv" / "bin" / "python")
    ).absolute()
    if not gnn_python.exists():
        print(
            f"gate gnn: no GNN interpreter at {gnn_python}; sync that checkout with "
            "`uv sync --locked --python 3.11 --extra dev --extra geo-infer` or pass --gnn-python",
            file=sys.stderr,
        )
        return 2
    head = subprocess.run(
        ["git", "-C", str(gnn_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != pin["revision"]:
        print(
            f"gate gnn: checkout {gnn_repo} is at {head}, pin records {pin['revision']} ({pin['tag']})",
            file=sys.stderr,
        )
        return 2
    receipt = RESULTS / "gnn" / f"interchange-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    command = (
        "python GEO-INFER-TEST/validate_gnn_interchange.py"
        f" --gnn-repo '{gnn_repo}' --gnn-python '{gnn_python}'"
        f" --output '{receipt}'"
    )
    started = time.monotonic()
    result = subprocess.run(
        command, shell=True, cwd=REPO_ROOT, env=_shell_env(), executable="/bin/bash"
    )
    elapsed = time.monotonic() - started
    print(
        f"[gate] gnn command=1/{1} seconds={elapsed:.2f} status={'ok' if result.returncode == 0 else 'failed'}"
    )
    if result.returncode != 0:
        return result.returncode
    document = json.loads(receipt.read_text(encoding="utf-8"))
    problems: list[str] = []
    if document.get("success") is not True:
        problems.append(
            f"receipt success={document.get('success')!r} {document.get('error')}"
        )
    if document["gnn"]["revision"] != pin["revision"]:
        problems.append(
            f"gnn revision {document['gnn']['revision']} != pin {pin['revision']}"
        )
    if document["geo_execution"].get("custody_complete") is not True:
        problems.append("geo_execution.custody_complete is not true")
    artifacts = document.get("artifacts") or {}
    if not artifacts:
        problems.append("receipt retains no artifact digests")
    retained = receipt.parent / f"{receipt.stem}-artifacts"
    for relative, digest in artifacts.items():
        artifact = (retained / relative).resolve()
        if not str(artifact).startswith(str(retained.resolve())):
            problems.append(f"artifact {relative} escapes the retained directory")
        elif (
            not artifact.is_file()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
        ):
            problems.append(f"artifact {relative} does not match its retained digest")
    dirty = [name for name in ("geo", "gnn") if document[name]["dirty"]]
    if dirty:
        message = f"dirty worktrees recorded: {', '.join(dirty)}"
        if args.allow_dirty:
            print(f"[gate] gnn warning: {message} (non-publication run)")
        else:
            problems.append(message + "; commit or pass --allow-dirty")
    if problems:
        for problem in problems:
            print(f"gate gnn: {problem}", file=sys.stderr)
        return 1
    print(
        f"[gate] gnn receipt {receipt.relative_to(REPO_ROOT)} verified against {PIN_FILE.name}"
    )
    return 0


def _run_arm(name: str, args: argparse.Namespace) -> int:
    arm = ARMS[name]
    if name == "gnn":
        return _gnn_interchange(args)
    environment = _shell_env()
    wall = 0.0
    for index, command in enumerate(arm.commands, start=1):
        started = time.monotonic()
        result = subprocess.run(
            command, shell=True, cwd=REPO_ROOT, env=environment, executable="/bin/bash"
        )
        elapsed = time.monotonic() - started
        wall += elapsed
        status = "ok" if result.returncode == 0 else f"failed={result.returncode}"
        print(
            f"[gate] {name} command={index}/{len(arm.commands)} seconds={elapsed:.2f} {status}"
        )
        if result.returncode != 0:
            print(f"[gate] {name} failed: {command}", file=sys.stderr)
            return result.returncode
    print(f"[gate] {name} ok wall={wall:.2f}s commands={len(arm.commands)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "arm", nargs="?", choices=(*ALL_ORDER, "all"), help="arm to run"
    )
    parser.add_argument(
        "--list", action="store_true", help="print every arm and its commands"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="accept dirty worktrees in the GNN receipt",
    )
    parser.add_argument(
        "--gnn-repo",
        help="GNN checkout to conform against (default: sibling checkout or GEO_INFER_GNN_REPO)",
    )
    parser.add_argument(
        "--gnn-python",
        help="interpreter inside that checkout (default: <checkout>/.venv/bin/python)",
    )
    args = parser.parse_args(argv)
    if args.list or not args.arm:
        for name in ALL_ORDER:
            arm = ARMS[name]
            print(f"{name}: {arm.source}")
            for need in arm.needs:
                print(f"  needs {need}")
            for command in arm.commands:
                print(f"  $ {command}")
        return 0
    names = ALL_ORDER if args.arm == "all" else (args.arm,)
    started = time.monotonic()
    for name in names:
        code = _run_arm(name, args)
        if code != 0:
            print(f"[gate] arm {name} failed", file=sys.stderr)
            return code
    print(f"[gate] {args.arm} green wall={time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
