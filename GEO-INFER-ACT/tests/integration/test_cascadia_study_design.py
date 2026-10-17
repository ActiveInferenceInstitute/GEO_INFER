"""Portable study-design parsing, bounded inputs, and real CLI rejection controls."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from geo_infer_act.models.river_homing_study import (
    Condition,
    StudyDesign,
    load_study_design,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cascadia"


def test_portable_reference_design_matches_defaults():
    assert load_study_design(EXAMPLE / "study_design.json") == StudyDesign()


def test_portable_custom_design_preserves_order(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(
        '{"seeds": [9, 3], "conditions": ["uniform_habit", "reference"], "steps": 5, "agent_seed": 7}'
    )
    design = load_study_design(path)
    assert design == StudyDesign(
        seeds=(9, 3),
        conditions=(Condition.UNIFORM_HABIT, Condition.REFERENCE),
        steps=5,
        agent_seed=7,
    )


@pytest.mark.parametrize(
    "content",
    [
        '{"steps": 2, "steps": 3}',
        '{"steps": true}',
        '{"steps": "4"}',
        '{"seeds": [1, 1]}',
        '{"seeds": [true]}',
        '{"conditions": ["missing"]}',
        '{"conditions": []}',
        '{"steps": 513}',
        '{"agent_seed": -1}',
        '{"unknown": 2}',
        "[]",
        "{",
    ],
)
def test_invalid_design_fails_before_outputs(tmp_path, content):
    path = tmp_path / "study.json"
    path.write_text(content)
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "cascadia_study.py"),
            "--design",
            str(path),
            "--output",
            str(output),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "override",
    [
        ["--steps", "48"],
        ["--seeds", "0"],
        ["--conditions", "reference"],
        ["--agent-seed", "0"],
    ],
)
def test_explicit_override_conflicts_even_when_equal_to_default(tmp_path, override):
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "cascadia_study.py"),
            "--design",
            str(EXAMPLE / "study_design.json"),
            "--output",
            str(output),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
            *override,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "--design cannot be combined" in completed.stderr
    assert not output.exists()


def test_default_and_explicit_cli_designs_are_preserved(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cascadia_study import parse_study_arguments; import sys; "
            "print(parse_study_arguments(sys.argv[1:])[1].model_dump_json())",
            "--output",
            str(tmp_path / "output"),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
        ],
        cwd=EXAMPLE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert StudyDesign.model_validate_json(completed.stdout) == StudyDesign()


def test_design_parser_does_not_load_scientific_backends(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; from cascadia_study import parse_study_arguments; "
            "paths, design = parse_study_arguments(sys.argv[1:]); "
            "print(json.dumps(dict(design=design.model_dump(mode='json'), "
            "heavy=sorted(set(sys.modules) & "
            "{'numpy','scipy','matplotlib','geopandas','jax','pymc','torch'}))))",
            "--output",
            str(tmp_path / "output"),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
        ],
        cwd=EXAMPLE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["heavy"] == []
    assert (
        StudyDesign.model_validate_json(json.dumps(payload["design"])) == StudyDesign()
    )
    assert not (tmp_path / "output").exists()
