"""Real paired inference, malformed sealed scores and deterministic HTML output."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, assert_never

import pytest

from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.runners.model_pair import PairDesign, create_pair_edition
from geo_infer_act.runners.model_pair_explorer import (
    load_pair_comparison,
    render_pair_comparison,
)
from geo_infer_act.runners.model_pair_plot import plot_model_pairs


@pytest.fixture
def pair_inputs() -> tuple[GNNArtifact, GNNArtifact]:
    agent = GNNArtifact.from_dict(
        dict(
            schema_version="gnn-geo-infer/1",
            model_type="categorical",
            model_name="two-state comparison",
            dimensions=dict(states=2, observations=2, actions=1),
            matrices=dict(
                A=[[0.75, 0.25], [0.25, 0.75]],
                B=[[[1], [0]], [[0], [1]]],
                C=[0, 0],
                D=[0.5, 0.5],
                E=[1],
            ),
            space=dict(kind="categorical", state_ids=["west", "east"]),
            time=dict(step_seconds=60),
            provenance=dict(producer="analytical test fixture", source_sha256="0" * 64),
        )
    )
    process = agent.to_dict()
    process["matrices"].update(A=[[1, 0], [0, 1]], B=[[[0], [1]], [[1], [0]]])
    return agent, GNNArtifact.from_dict(process)


@pytest.fixture
def edition(pair_inputs: tuple[GNNArtifact, GNNArtifact], tmp_path: Path) -> Path:
    output = tmp_path / "recorded_pair"
    create_pair_edition(
        *pair_inputs, PairDesign(initial_state=0, steps=3, seeds=(0, 7)), output
    )
    return output


def test_display_preserves_analytical_scores_clock_seed_order_and_identity(
    edition: Path,
) -> None:
    view = load_pair_comparison((edition,))
    assert view.seeds == (0, 7) and view.hours == (0, 1 / 60, 2 / 60)
    assert not view.native_claim_ready
    first = view.editions[0].runs[0].frames[0]
    assert first.brier_score == pytest.approx(0.125)
    assert first.posterior_entropy_nats == pytest.approx(
        -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    )
    assert (
        view.editions[0].manifest_sha256
        == hashlib.sha256((edition / "pair_manifest.json").read_bytes()).hexdigest()
    )


def test_html_is_deterministic_freshness_checked_and_create_only(
    edition: Path, tmp_path: Path
) -> None:
    first, second = tmp_path / "first.html", tmp_path / "second.html"
    render_pair_comparison((edition,), first)
    render_pair_comparison((edition,), second)
    assert first.read_bytes() == second.read_bytes()
    match = re.search(
        r'<script id="pair-data" type="application/json">(.*?)</script>',
        first.read_text(),
        re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["editions"][0]["runs"][0]["frames"][0][
        "brier_score"
    ] == pytest.approx(0.125)
    assert payload["native_claim_ready"] is False
    render_pair_comparison((edition,), first, check=True)
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        render_pair_comparison((edition,), first)
    assert first.read_bytes() == original
    first.write_bytes(b"stale preserved HTML")
    with pytest.raises(ValueError, match="HTML bytes"):
        render_pair_comparison((edition,), first, check=True)
    assert first.read_bytes() == b"stale preserved HTML"


@pytest.mark.parametrize("mutation", ["clock", "horizon", "nonfinite", "duplicate"])
@pytest.mark.parametrize("renderer", [render_pair_comparison, plot_model_pairs])
def test_resealed_invalid_scores_cannot_enter_comparison(
    edition: Path,
    tmp_path: Path,
    mutation: Literal["clock", "horizon", "nonfinite", "duplicate"],
    renderer: Callable[[Sequence[Path], Path], Path],
) -> None:
    path = edition / "seed_0/scores.json"
    scores = json.loads(path.read_bytes())
    match mutation:
        case "clock":
            scores[1]["elapsed_seconds"] = 99
            content = json.dumps(scores)
        case "horizon":
            scores.pop()
            content = json.dumps(scores)
        case "nonfinite":
            scores[0]["posterior_entropy_nats"] = float("nan")
            content = json.dumps(scores)
        case "duplicate":
            content = json.dumps(scores).replace(
                '"brier_score":', '"brier_score": 0, "brier_score":', 1
            )
        case unreachable:
            assert_never(unreachable)
    path.write_text(content, encoding="utf-8")
    manifest = edition / "pair_manifest.json"
    seal = json.loads(manifest.read_bytes())
    seal["files"]["seed_0/scores.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(seal), encoding="utf-8")
    output = tmp_path / "refused.html"
    with pytest.raises(ValueError):
        renderer((edition,), output)
    assert not output.exists()


def test_common_grid_required_and_one_step_clock_is_supported(
    edition: Path, pair_inputs: tuple[GNNArtifact, GNNArtifact], tmp_path: Path
) -> None:
    other = tmp_path / "one_step"
    create_pair_edition(
        *pair_inputs, PairDesign(initial_state=0, steps=1, seeds=(0, 7)), other
    )
    assert load_pair_comparison((other,)).hours == (0,)
    for selected in ((), (edition, edition), (edition, other)):
        with pytest.raises(ValueError):
            load_pair_comparison(selected)
    same = tmp_path / "same_grid"
    create_pair_edition(
        *pair_inputs, PairDesign(initial_state=0, steps=3, seeds=(0, 7)), same
    )
    assert len(load_pair_comparison((edition, same)).editions) == 2


def test_cli_html_roundtrip_and_unsupported_check_refusal(
    edition: Path, tmp_path: Path
) -> None:
    script = (
        Path(__file__).resolve().parents[2] / "examples/cascadia/plot_model_pairs.py"
    )
    output = tmp_path / "cli.html"
    command = [
        sys.executable,
        str(script),
        "--editions",
        str(edition),
        "--output",
        str(output),
        "--html",
    ]
    created = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert created.returncode == 0, created.stderr
    checked = subprocess.run(
        [*command, "--check"], capture_output=True, text=True, timeout=60, check=False
    )
    assert checked.returncode == 0, checked.stderr
    rejected = subprocess.run(
        [*command[:-1], "--check"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert rejected.returncode != 0
