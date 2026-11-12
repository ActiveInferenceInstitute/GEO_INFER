#!/usr/bin/env python3
"""GEO-INFER local gate: the single definition of every gate command.

Each arm is a named list of commands. The workflows do not repeat those
commands: ``ci.yml`` calls ``scripts/gate.py <arm>`` with the same arms a
developer runs locally, so a command exists once in the repository. Structural
contract tests in ``GEO-INFER-TEST/tests/unit/test_gate_contracts.py`` fail when
a workflow step stops calling an arm, when an inline copy of a gate command
reappears, or when an arm loses a command a workflow contract pins.

Arms run against the single locked workspace ``.venv`` (``uv sync
--all-packages --all-extras --all-groups``); an arm never creates a second
environment. ``.venv/bin`` is prepended to ``PATH`` so a bare ``python`` in a
command means the workspace interpreter, matching the interpreter CI selects
with ``actions/setup-python``.

Usage:
    uv run --frozen python scripts/gate.py --list
    uv run --frozen python scripts/gate.py lint
    uv run --frozen python scripts/gate.py docs gates
    uv run --frozen python scripts/gate.py all

Leaf arms: ``lint``, ``docs``, ``fast``, ``gates``, ``test-<category>`` for each
matrix category, ``gnn``, ``manuscript-artifacts``, ``manuscript-convergence``,
``manuscript-render``. ``tests``, ``manuscript`` and ``all`` expand to leaves.
There is no ``types`` arm: the repository declares ``[tool.mypy]`` and a
``mypy`` dependency but runs no type gate, so a type arm would invent a
contract rather than define one.
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

# `ci.yml` "validate" -> "Run source runtime hygiene"; also the documented
# whole-tree formatting contract.
LINT_COMMANDS = (
    "uv run --with 'ruff>=0.15.6,<0.16' ruff check .",
    "uv run --with 'ruff>=0.15.6,<0.16' ruff format --check .",
)

# `ci.yml` "validate" -> "Run documentation gates".
DOCS_COMMANDS = (
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/validate_documentation.py --strict",
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py -- python GEO-INFER-TEST/rewrite_readme_agents.py --check",
)

# `ci.yml` "fast-contracts" -> "Run execution and SPACE TIME composition contracts".
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

# `ci.yml` "validate" -> "Run repository, model and secret gates".
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

# `ci.yml` "test" -> one arm per matrix category. The h3 leg runs the dedicated
# migration validator instead of `--category h3`, exactly as the branched step
# did before the arm owned the branch.
TEST_CATEGORY_COMMANDS: dict[str, str] = {
    "unit": "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category unit --timeout 600",
    "integration": "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category integration --timeout 600",
    "performance": "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category performance --timeout 600",
    "system": "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category system --timeout 600",
    "slow": "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category slow --timeout 600",
    "h3": "uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --h3-migration",
}

# `ci.yml` "manuscript": the producer, its convergence check, and the render
# followed by the two registered ROOT profiles.
MANUSCRIPT_ARTIFACT_COMMAND = (
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py --timeout 1800 -- "
    "python manuscript/generate_research_artifacts.py --verify --rerun-verification"
)
MANUSCRIPT_CONVERGENCE_COMMAND = (
    "uv run --no-sync python GEO-INFER-TEST/record_validation.py --timeout 600 -- "
    "python manuscript/generate_research_artifacts.py --check"
)
MANUSCRIPT_RENDER_COMMANDS = (
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


TEST_ARMS = tuple(f"test-{category}" for category in TEST_CATEGORY_COMMANDS)
MANUSCRIPT_ARMS = (
    "manuscript-artifacts",
    "manuscript-convergence",
    "manuscript-render",
)

ARMS: dict[str, Arm] = {
    "lint": Arm("lint", "ci.yml: validate > Run source runtime hygiene", LINT_COMMANDS),
    "docs": Arm("docs", "ci.yml: validate > Run documentation gates", DOCS_COMMANDS),
    "fast": Arm("fast", "ci.yml: fast-contracts", (FAST_COMMAND,)),
    "gates": Arm(
        "gates",
        "ci.yml: validate > Run repository, model and secret gates",
        GATE_COMMANDS,
    ),
    **{
        f"test-{category}": Arm(
            f"test-{category}",
            f"ci.yml: test (matrix category {category})",
            (command,),
        )
        for category, command in TEST_CATEGORY_COMMANDS.items()
    },
    "gnn": Arm(
        "gnn",
        "gnn-interchange.yml: interchange",
        (),
        needs=(
            "geometry: a clean worktree, a GNN checkout at the pin, and that checkout's interpreter",
        ),
    ),
    "manuscript-artifacts": Arm(
        "manuscript-artifacts",
        "ci.yml: manuscript > Generate research artifacts",
        (MANUSCRIPT_ARTIFACT_COMMAND,),
    ),
    "manuscript-convergence": Arm(
        "manuscript-convergence",
        "ci.yml: manuscript > Check published artifacts converge",
        (MANUSCRIPT_CONVERGENCE_COMMAND,),
    ),
    "manuscript-render": Arm(
        "manuscript-render",
        "ci.yml: manuscript > Render the manuscript and run root profiles",
        MANUSCRIPT_RENDER_COMMANDS,
    ),
}

GROUPS: dict[str, tuple[str, ...]] = {
    "tests": TEST_ARMS,
    "manuscript": MANUSCRIPT_ARMS,
    "all": (
        "lint",
        "docs",
        "fast",
        "gates",
        *TEST_ARMS,
        "gnn",
        *MANUSCRIPT_ARMS,
    ),
}

LEAVES = GROUPS["all"]


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
    receipt = (
        Path(args.output).absolute()
        if args.output
        else RESULTS / "gnn" / f"interchange-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    if receipt.exists():
        print(
            f"gate gnn: receipt {receipt} already exists; receipts are immutable",
            file=sys.stderr,
        )
        return 2
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
        f"[gate] gnn command=1/1 seconds={elapsed:.2f} status={'ok' if result.returncode == 0 else 'failed'}"
    )
    if result.returncode != 0:
        return result.returncode
    document = json.loads(receipt.read_text(encoding="utf-8"))
    problems: list[str] = []
    if document.get("schema_version") != "1.0":
        problems.append(f"receipt schema_version={document.get('schema_version')!r}")
    if document.get("success") is not True:
        problems.append(
            f"receipt success={document.get('success')!r} {document.get('error')}"
        )
    if document["gnn"]["revision"] != pin["revision"]:
        problems.append(
            f"gnn revision {document['gnn']['revision']} != pin {pin['revision']}"
        )
    expected_geo = (
        args.expect_geo_revision
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if document["geo"]["revision"] != expected_geo:
        problems.append(
            f"geo revision {document['geo']['revision']} != candidate {expected_geo}"
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
    try:
        shown = receipt.relative_to(REPO_ROOT)
    except ValueError:  # a caller-supplied receipt may live outside the checkout
        shown = receipt
    print(f"[gate] gnn receipt {shown} verified against {PIN_FILE.name}")
    return 0


def _check_gitleaks() -> str | None:
    """Return a failure reason when the pinned secret scanner is unavailable."""
    try:
        probe = subprocess.run(
            ["gitleaks", "version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return (
            f"gitleaks is not on PATH; CI installs {GITLEAKS_VERSION} "
            "from a pinned release tarball"
        )
    version = (probe.stdout or probe.stderr).strip()
    if version != GITLEAKS_VERSION:
        return (
            f"gitleaks {version or 'reports no version'} != required {GITLEAKS_VERSION}"
        )
    return None


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


def _expand(names: list[str]) -> list[str]:
    leaves: list[str] = []
    for name in names:
        for leaf in GROUPS.get(name, (name,)):
            if leaf not in leaves:
                leaves.append(leaf)
    return leaves


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arms", nargs="*", help="arms or groups to run; see --list")
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
    parser.add_argument(
        "--output",
        help=(
            "immutable receipt path for the GNN arm "
            "(default: .geo-infer-test-results/gnn/interchange-<utc>.json)"
        ),
    )
    parser.add_argument(
        "--expect-geo-revision",
        help=(
            "exact GEO candidate commit the receipt must record "
            "(default: this checkout's HEAD)"
        ),
    )
    args = parser.parse_args(argv)
    if args.list or not args.arms:
        for name in LEAVES:
            arm = ARMS[name]
            print(f"{name}: {arm.source}")
            for need in arm.needs:
                print(f"  needs {need}")
            for command in arm.commands:
                print(f"  $ {command}")
        print("groups: " + ", ".join(sorted(GROUPS)))
        return 0
    unknown = [name for name in args.arms if name not in ARMS and name not in GROUPS]
    if unknown:
        parser.error(
            f"unknown arm(s) {unknown}; known arms are {sorted(ARMS)} and groups {sorted(GROUPS)}"
        )
    names = _expand(args.arms)
    if "gates" in names or "all" in args.arms:
        reason = _check_gitleaks()
        if reason:
            print(f"gate: {reason}", file=sys.stderr)
            return 2
    started = time.monotonic()
    for name in names:
        code = _run_arm(name, args)
        if code != 0:
            print(f"[gate] arm {name} failed", file=sys.stderr)
            return code
    print(f"[gate] {', '.join(args.arms)} green wall={time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
