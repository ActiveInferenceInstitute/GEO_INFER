"""Known categorical scores, clock alignment and retained-archive diagnostics."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from geo_infer_act.runners.study_diagnostics import DiagnosticInputError, diagnose_frame
from geo_infer_act.runners.study_visualization import project_study

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cascadia"


@pytest.fixture
def study(tmp_path: Path) -> Path:
    with ZipFile(EXAMPLE / "data/cascadia_seed_study_20261012_svg_v2.zip") as archive:
        archive.extractall(tmp_path)
    return tmp_path / "cascadia_seed_study_20261012_svg_v2"


def test_two_state_scores_match_known_values() -> None:
    result = diagnose_frame((0.75, 0.25), 0, (0.5, 0.5), elapsed_seconds=60.0)
    assert result.brier_score == 0.125
    assert result.posterior_entropy_nats == pytest.approx(
        -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    )
    assert result.policy_entropy_nats == pytest.approx(math.log(2))
    assert result.true_state_surprisal_nats == pytest.approx(-math.log(0.75))


def test_zero_true_mass_retains_maximal_score_and_explicit_infinite_surprisal() -> None:
    result = diagnose_frame((0.0, 1.0), 0, (1.0,), elapsed_seconds=0.0)
    assert result.brier_score == 2 and result.posterior_entropy_nats == 0
    assert result.true_state_surprisal_nats is None
    assert json.loads(result.model_dump_json())["true_state_surprisal_nats"] is None


@pytest.mark.parametrize(
    "vector", [(), (0.5, 0.6), (-0.1, 1.1), (math.nan,), (math.inf,), (True, False)]
)
def test_malformed_distribution_is_rejected(vector: tuple[float, ...]) -> None:
    with pytest.raises(DiagnosticInputError):
        diagnose_frame(vector, 0, (1.0,), elapsed_seconds=0.0)


def test_invalid_policy_vector_and_state_are_rejected() -> None:
    with pytest.raises(DiagnosticInputError):
        diagnose_frame((1.0,), 0, (0.9,), elapsed_seconds=0.0)
    with pytest.raises(DiagnosticInputError):
        diagnose_frame((1.0,), 1, (1.0,), elapsed_seconds=0.0)


def test_archive_scores_use_observation_time_state_and_physical_clock(
    study: Path,
) -> None:
    view = project_study(study)
    run = next(
        r
        for r in view.runs
        if r.outcome.condition.value == "reference" and r.outcome.process_seed == 7
    )
    first = run.diagnostics[0]
    assert first.true_state_probability == pytest.approx(18 / 19)
    assert first.brier_score == pytest.approx(2 / 19**2)
    assert run.diagnostics[1].elapsed_seconds == 21600
    assert len(run.diagnostics) == len(run.trajectory.true_states) == 48


@pytest.mark.parametrize("mutation", ["naive", "clock", "observation"])
def test_display_rejects_misaligned_time_or_cues(study: Path, mutation: str) -> None:
    path = study / "reference/seed_0/closed_loop.json"
    data = json.loads(path.read_bytes())
    if mutation == "naive":
        data["trace"]["steps"][0]["timestamp"] = "2026-10-01T00:00:00"
    elif mutation == "clock":
        data["trace"]["steps"][1]["timestamp"] = data["trace"]["steps"][0]["timestamp"]
    else:
        data["observations"][0] = (data["observations"][0] + 1) % 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="display"):
        project_study(study)


def test_report_cli_is_finite_create_only_and_qualified(
    study: Path, tmp_path: Path
) -> None:
    output = tmp_path / "diagnostics.json"
    command = [
        sys.executable,
        str(EXAMPLE / "study_diagnostics.py"),
        str(study),
        str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert completed.returncode == 0, completed.stderr
    original = output.read_bytes()
    report = json.loads(original)
    assert len(report["runs"]) == 24 and report["native_claim_ready"] is False
    assert b"NaN" not in original and b"Infinity" not in original
    refused = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert refused.returncode != 0 and output.read_bytes() == original
