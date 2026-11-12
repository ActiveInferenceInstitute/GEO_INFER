"""Contracts for the shared gate arms and the GNN pointer they document.

``scripts/gate.py`` is the only place a gate command is written down. These
contracts fail when a workflow copies a gate command back into itself, when a
workflow stops calling an arm, when two arms repeat a command, when an arm
loses a command a workflow contract pins, or when the committed GNN pointer
stops describing the pin file and the conformance job it names.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from functools import cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE = REPO_ROOT / "scripts" / "gate.py"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Gate commands are CI-only when the measurement needs the event (a base SHA)
# or a disposable service; those steps own their command and are not arms.
CI_ONLY_MARKERS = (
    "check_coverage_floor.py",
    "validate_research_evidence.py",
    "provision_duckdb_spatial.py",
)

GATE_MARKERS = (
    "validate_repo_contracts.py --strict-source-language --strict-import-smoke",
    "validate_packaging.py --strict",
    "validate_logging_hygiene.py",
    "validate_doc_examples.py",
    "validate_skills.py --check-xrefs --warnings-fatal",
    "validate_active_inference_contract.py",
    "validate_test_contracts.py --strict",
    "validate_model_contracts.py --strict --seed 42",
    "run_model_audit.py --seed 42 --reproducible",
    "gitleaks detect --source . --config .gitleaks.toml --redact --verbose",
    "validate_documentation.py --strict",
    "rewrite_readme_agents.py --check",
    "ruff check .",
    "ruff format --check .",
    "test_space_time_composition_contract.py",
    "generate_research_artifacts.py --verify --rerun-verification",
    "generate_research_artifacts.py --check",
    "render_manuscript_pdf.py",
    "--module ROOT --category manuscript-render",
    "--module ROOT --category manuscript",
)


def _normalize(text: str) -> str:
    # Join shell line continuations first, then collapse YAML block whitespace.
    return re.sub(r"\s+", " ", text.replace("\\\n", " ")).strip()


@cache
def _gate():
    spec = importlib.util.spec_from_file_location("geo_infer_gate", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[module.__name__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _all_commands() -> list[tuple[str, str]]:
    gate = _gate()
    return [
        (arm.name, command) for arm in gate.ARMS.values() for command in arm.commands
    ]


def _workflow_text(name: str) -> str:
    return _normalize((WORKFLOWS / name).read_text(encoding="utf-8"))


def _trigger_paths(workflow: dict) -> list[str]:
    # PyYAML resolves a bare `on:` key to the boolean True (YAML 1.1).
    trigger = workflow.get("on", workflow.get(True, {}))
    return list(trigger["pull_request"]["paths"])


def test_workflows_call_arms_instead_of_copying_commands():
    """Every gate command is written once, in the gate module."""
    ci = _workflow_text("ci.yml")
    for arm, command in _all_commands():
        assert _normalize(command) not in ci, (arm, command)
    for marker in GATE_MARKERS:
        assert marker not in ci, f"gate command copied into ci.yml: {marker}"


def test_ci_calls_every_arm_its_jobs_own():
    """The jobs run arms by name; a renamed arm breaks here, not in CI."""
    gate = _gate()
    ci = _workflow_text("ci.yml")
    for arm in (
        "docs",
        "gates",
        "lint",
        "fast",
        "manuscript-artifacts",
        "manuscript-convergence",
        "manuscript-render",
    ):
        assert arm in gate.ARMS, arm
        assert "scripts/gate.py" in ci and arm in ci, f"ci.yml does not call {arm}"
    for category in gate.TEST_CATEGORY_COMMANDS:
        assert f"test-{category}" in gate.ARMS
    assert "scripts/gate.py test-${{ matrix.category }}" in ci
    # CI-only steps keep their own command (event base, disposable services).
    for marker in CI_ONLY_MARKERS:
        assert marker in ci, f"CI-only command vanished: {marker}"


def test_no_command_appears_in_two_arms():
    """An arm is the only owner of its command."""
    owner: dict[str, str] = {}
    for arm, command in _all_commands():
        assert command not in owner, (command, arm, owner.get(command))
        owner[command] = arm


def test_arms_keep_every_command_a_workflow_contract_pins():
    """Moving a command into the gate must not weaken or drop it."""
    joined = "\n".join(command for _, command in _all_commands())
    for marker in GATE_MARKERS:
        assert marker in joined, marker
    # The root pyproject [tool.ruff.lint] table is the only lint contract; a
    # --select override would silently narrow it.
    assert "--select" not in joined
    assert joined.count("ruff check .") == 1
    assert "xargs -0" not in joined


def test_tests_arm_covers_every_matrix_category():
    """The local tests group runs exactly the categories the matrix defines."""
    gate = _gate()
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    assert set(gate.TEST_ARMS) == {
        f"test-{category}" for category in matrix["category"]
    }
    commands = {arm: gate.ARMS[arm].commands for arm in gate.TEST_ARMS}
    assert any(
        "--h3-migration" in command for cmds in commands.values() for command in cmds
    )


def test_gnn_arm_owns_the_conformance_invocation_and_receipt_checks():
    """The GNN job runs the arm for its exact candidate; the arm asserts it."""
    source = _workflow_text("gnn-interchange.yml")
    assert "geo/scripts/gate.py gnn" in source
    assert '--gnn-repo "$PWD/gnn"' in source
    assert '--gnn-python "$PWD/gnn/.venv/bin/python"' in source
    assert "--output receipts/interchange.json" in source
    assert '--expect-geo-revision "$GITHUB_SHA"' in source
    # No second copy of the validator invocation in any step; the validator
    # path may still appear in the trigger's path filter, which is not a
    # command and must keep retriggering the job.
    workflow = yaml.safe_load(
        (WORKFLOWS / "gnn-interchange.yml").read_text(encoding="utf-8")
    )
    runs = "\n".join(
        str(step.get("run", "")) for step in workflow["jobs"]["interchange"]["steps"]
    )
    assert "validate_gnn_interchange.py" not in runs
    assert "GEO-INFER-TEST/validate_gnn_interchange.py" in _trigger_paths(workflow)
    gate_source = _normalize(GATE.read_text(encoding="utf-8"))
    for marker in (
        "validate_gnn_interchange.py",
        'document.get("schema_version") != "1.0"',
        'document["gnn"]["revision"] != pin["commit"]',
        "expect_geo_revision",
        "custody_complete",
        "hashlib.sha256",
    ):
        assert marker in gate_source, marker
    assert gate_source.count("validate_gnn_interchange.py") == 1
    # The pin, the pointer and the specifications are the shared reader's: the
    # arm calls `python -m gnn.importers` and keeps no pin shape of its own.
    for marker in (
        "gnn.importers",
        "--require-pointer",
        ".github/workflows/gnn-interchange.yml",
        "interchange",
    ):
        assert marker in gate_source, marker
    for absent in ("PIN_SCHEMA", "gnn-pair.json", "def _read_pin("):
        assert absent not in gate_source, absent


def test_gate_has_no_type_arm_the_repository_cannot_run():
    """mypy is declared but ungated; a type arm would invent a contract."""
    assert "types" not in _gate().ARMS
    assert "mypy" in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_groups_expand_to_leaf_arms_once():
    """`all`, `tests` and `manuscript` are aliases, not second definitions."""
    gate = _gate()
    assert gate._expand(["all"]) == list(gate.LEAVES)
    assert len(gate.LEAVES) == len(set(gate.LEAVES))
    assert gate._expand(["tests"]) == list(gate.TEST_ARMS)
    assert gate._expand(["manuscript"]) == list(gate.MANUSCRIPT_ARMS)
    assert gate._expand(["lint", "lint"]) == ["lint"]
    assert "all" not in gate.ARMS and "tests" not in gate.ARMS


def test_gnn_conformance_job_key_and_display_name_are_stable():
    """The required fleet context is the 3.12 leg of the `interchange` job."""
    workflow = yaml.safe_load(
        (WORKFLOWS / "gnn-interchange.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["interchange"]
    assert sorted(job["strategy"]["matrix"]["python"]) == ["3.11", "3.12"]
    assert "gnn conformance (pinned)" in job["name"]
    called = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    assert (
        called["jobs"]["paired-interchange"]["uses"]
        == "./.github/workflows/gnn-interchange.yml"
    )


def test_workflows_do_not_use_the_shared_setup_python_uv_action():
    """This repository pins astral-sh/setup-uv directly; docs must say so."""
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        assert "setup-python-uv" not in workflow.read_text(encoding="utf-8"), (
            workflow.name
        )


def test_sidecar_tasks_use_the_canonical_gate_arms():
    sidecar = yaml.safe_load((REPO_ROOT / ".aii/config.yaml").read_text())
    for task, arm in (("test", "test-unit"), ("lint", "lint"), ("validate", "gates")):
        assert sidecar["tasks"][task]["cmd"] == (
            f"uv run --frozen python scripts/gate.py {arm}"
        )
