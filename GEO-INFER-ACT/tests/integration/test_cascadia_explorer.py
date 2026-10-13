"""Real archived trajectories, invalid vectors and create-only visual output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from geo_infer_act.runners.study_visualization import project_study

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cascadia"


@pytest.fixture
def study(tmp_path: Path) -> Path:
    with ZipFile(EXAMPLE / "data/cascadia_seed_study_20261012_svg_v2.zip") as archive:
        archive.extractall(tmp_path)
    return tmp_path / "cascadia_seed_study_20261012_svg_v2"


def test_real_projection_retains_all_seeds_and_raw_probabilities(study: Path) -> None:
    view = project_study(study)
    assert len(view.runs) == 24 and not view.native_claim_ready
    run = next(r for r in view.runs if r.outcome.process_seed == 7)
    assert run.trajectory.trace.steps[0].posterior[0] == pytest.approx(18 / 19)
    assert len(run.cells) == 26 and len(run.trajectory.trace.steps) == 48
    assert run.cells[0].role == "sea"
    assert all(
        -90 <= c.latitude <= 90 and -180 <= c.longitude <= 180 for c in run.cells
    )
    assert sum(r.outcome.measures.reached_home_step is None for r in view.runs) == 16


@pytest.mark.parametrize(
    "mutation", ["probability", "order", "action", "state", "horizon"]
)
def test_projection_rejects_broken_display_data(study: Path, mutation: str) -> None:
    path = study / "reference/seed_0/closed_loop.json"
    data = json.loads(path.read_bytes())
    if mutation == "probability":
        data["trace"]["steps"][0]["posterior"][0] = 1.1
    elif mutation == "order":
        data["trace"]["state_ids"].reverse()
    elif mutation == "action":
        data["trace"]["steps"][0]["action"] = 4
    elif mutation == "state":
        data["true_states"][0] = 26
    else:
        data["trace"]["steps"].pop()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="display"):
        project_study(study)


def test_explorer_qualifies_before_writing_and_preserves_existing_html(
    study: Path, tmp_path: Path
) -> None:
    output = tmp_path / "explorer.html"
    command = [
        sys.executable,
        str(EXAMPLE / "study_explorer.py"),
        str(study),
        str(output),
    ]
    first = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert first.returncode == 0, first.stderr
    content = output.read_bytes()
    assert b"__STUDY_DATA__" not in content and b"application/json" in content
    again = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert again.returncode != 0 and output.read_bytes() == content
    (study / "summary.json").write_text("[]", encoding="utf-8")
    refused = tmp_path / "refused.html"
    failed = subprocess.run(
        [*command[:-1], str(refused)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert failed.returncode != 0 and not refused.exists()
