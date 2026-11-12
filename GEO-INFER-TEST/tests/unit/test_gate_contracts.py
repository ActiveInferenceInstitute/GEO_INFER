"""Contracts for the local gate entry point and the GNN pointer it documents.

``scripts/gate.py`` is the one place a maintainer runs the gate, and every arm
command is copied from a workflow step. These contracts fail when the two
disagree, when a gate is duplicated across jobs, or when the committed GNN
pointer stops describing the pin file and the conformance job it names.
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


def _workflow_text(name: str) -> str:
    return _normalize((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_every_arm_command_is_a_workflow_command():
    """Each arm command must appear verbatim in the workflow it mirrors."""
    gate = _gate()
    for arm in gate.ARMS.values():
        if arm.name == "gnn":
            continue
        source = _workflow_text(arm.source.split(":")[0])
        for command in arm.commands:
            normalized = _normalize(command)
            if arm.name == "tests":
                # The matrix supplies the category; keep the arm's literal
                # category in lock step with the workflow matrix below.
                normalized = re.sub(
                    r"--category [a-z0-9]+",
                    "--category ${{ matrix.category }}",
                    normalized,
                )
            assert normalized in source, (arm.name, command)


def test_tests_arm_covers_every_matrix_category():
    """The local tests arm runs exactly the categories the matrix defines."""
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    commands = _gate().ARMS["tests"].commands
    categories = {
        match
        for command in commands
        for match in re.findall(r"--category ([a-z0-9]+)", command)
    }
    # The h3 leg runs the dedicated migration validator instead of
    # `--category h3`, exactly as the workflow branches.
    assert categories | {"h3"} == set(matrix["category"])
    assert any("--h3-migration" in command for command in commands)


def test_gnn_arm_runs_the_conformance_validator_its_workflow_runs():
    """The GNN arm calls the same validator with the same receipt contract."""
    source = _workflow_text(_gate().ARMS["gnn"].source.split(":")[0])
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


def test_arms_do_not_repeat_a_gate_across_jobs():
    """One gate, one job: the duplicated validate/lint steps stay removed."""
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    jobs = ci["jobs"]
    validate = _normalize(str(jobs["validate"]))
    fast = _normalize(str(jobs["fast-contracts"]))
    assert validate.count("validate_test_contracts.py") == 1
    assert "validate_test_contracts.py" not in fast
    assert validate.count("xargs -0 uv run") == 0, (
        "changed-file lint duplicated the whole-tree lint"
    )
    assert validate.count("ruff check .") == 1
    assert validate.count("ruff format --check .") == 1
    assert "ruff check" not in fast


def test_gate_has_no_type_arm_the_repository_cannot_run():
    """mypy is declared but ungated; a type arm would invent a contract."""
    gate = _gate()
    assert "types" not in gate.ARMS
    assert "mypy" in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


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
