"""Contracts for the shared gate arms and the GNN pointer they document.

``scripts/gate.py`` is the only place a gate command is written down. These
contracts fail when a workflow copies a gate command back into itself, when a
workflow stops calling an arm, when two arms repeat a command, when an arm
loses a command a workflow contract pins, or when the committed GNN pointer
stops describing the pin file and the conformance job it names.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from functools import cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE = REPO_ROOT / "scripts" / "gate.py"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PIN = REPO_ROOT / ".github" / "gnn-pair.json"
POINTER = REPO_ROOT / ".github" / "GNN.md"

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


def test_gnn_arm_runs_the_conformance_validator_its_workflow_runs():
    """The GNN arm calls the same validator with the same receipt contract."""
    source = _workflow_text("gnn-interchange.yml")
    for marker in (
        "validate_gnn_interchange.py",
        "--gnn-repo",
        "--gnn-python",
        "--output",
        "receipt['gnn']['revision'] == pair['revision']",
        "custody_complete",
    ):
        assert marker in source, marker
    # The arm builds the validator invocation itself (it takes a local
    # checkout and interpreter), so assert the source still names it once.
    assert (
        _normalize(GATE.read_text(encoding="utf-8")).count(
            "validate_gnn_interchange.py"
        )
        == 1
    )


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


def test_gnn_pointer_records_the_pin_and_the_conformance_job():
    """`.github/GNN.md` must keep naming the real pin, job and command."""
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    pointer = POINTER.read_text(encoding="utf-8")
    assert pin["schema"] == "gnn-importer-pin/v1"
    assert re.fullmatch("[0-9a-f]{40}", pin["revision"])
    assert re.fullmatch(r"v\d+\.\d+\.\d+", pin["tag"])
    for marker in (
        ".github/gnn-pair.json",
        "gnn-importer-pin/v1",
        pin["tag"],
        pin["revision"],
        "gnn conformance (pinned)",
        "gnn-interchange.yml",
        "GEO-INFER-TEST/validate_gnn_interchange.py",
    ):
        assert marker in pointer, marker


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
