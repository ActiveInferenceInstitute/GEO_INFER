"""Real pymdp inference against a separately specified categorical process."""

from __future__ import annotations

import pytest
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, assert_never

from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference
from geo_infer_act.runners.closed_loop import run_closed_loop
from geo_infer_act.runners.model_pair import (
    PairDesign,
    check_pair_edition,
    create_pair_edition,
)
from geo_infer_act.runners.model_pair_plot import plot_model_pairs


@pytest.fixture
def agent() -> GNNArtifact:
    return GNNArtifact.from_dict(
        dict(
            schema_version="gnn-geo-infer/1",
            model_type="categorical",
            model_name="persistent agent",
            dimensions=dict(states=2, observations=2, actions=2),
            matrices=dict(
                A=[[0.75, 0.25], [0.25, 0.75]],
                B=[[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
                C=[0, 0],
                D=[0.5, 0.5],
                E=[1, 0],
            ),
            space=dict(kind="categorical", state_ids=["west", "east"]),
            time=dict(step_seconds=60),
            provenance=dict(producer="exact numerical fixture", source_sha256="0" * 64),
        )
    )


@pytest.fixture
def process(agent: GNNArtifact) -> GNNArtifact:
    data = agent.to_dict()
    data["model_name"] = "alternating observed process"
    data["matrices"].update(
        A=[[1, 0], [0, 1]],
        B=[[[0, 0], [1, 1]], [[1, 1], [0, 0]]],
        C=[99, -99],
        D=[0, 1],
        E=[0, 1],
    )
    return GNNArtifact.from_dict(data)


def test_process_generates_cues_but_agent_owns_inference(
    agent: GNNArtifact,
    process: GNNArtifact,
) -> None:
    # Given a persistent agent and an alternating, perfectly observed process.
    # When the actual agent closes its loop against the separate process.
    result = run_closed_loop(
        agent,
        process_artifact=process,
        initial_state=0,
        steps=3,
        process_seed=7,
    )
    # Then process preferences/initial prior/habit cannot leak into inference.
    assert result.true_states == (0, 1, 0)
    assert result.observations == (0, 1, 0)
    first, second, third = result.trace["steps"]
    assert [row["action"] for row in result.trace["steps"]] == [0, 0, 0]
    assert first["posterior"] == pytest.approx([0.75, 0.25])
    assert second["prior"] == pytest.approx([0.75, 0.25])
    assert second["posterior"] == pytest.approx([0.5, 0.5])
    assert third["posterior"] == pytest.approx([0.75, 0.25])
    replay = run_gnn_inference(
        agent,
        [
            {"timestamp": r["timestamp"], "observation": o}
            for r, o in zip(result.trace["steps"], result.observations)
        ],
    )
    assert result.trace == replay
    assert result.to_dict()["process_artifact_sha256"] == process.digest
    assert result.trace["artifact_sha256"] == agent.digest


def test_explicit_matched_process_preserves_legacy_trace(agent: GNNArtifact) -> None:
    # Given the same process, seeds and physical clock in both forms.
    legacy = run_closed_loop(agent, initial_state=0, steps=3, process_seed=7)
    # When the process artifact is supplied explicitly.
    explicit = run_closed_loop(
        agent,
        process_artifact=agent,
        initial_state=0,
        steps=3,
        process_seed=7,
    )
    # Then the numerical run is identical and the legacy serialization is stable.
    assert explicit.trace == legacy.trace
    assert explicit.true_states == legacy.true_states
    assert explicit.observations == legacy.observations
    assert "process_artifact_sha256" not in legacy.to_dict()


@pytest.mark.parametrize("field", ["space", "time", "dimensions"])
def test_incompatible_process_is_rejected(
    agent: GNNArtifact,
    field: Literal["space", "time", "dimensions"],
) -> None:
    # Given an individually valid process that cannot share this agent's indices.
    data = agent.to_dict()
    match field:
        case "space":
            data["space"]["state_ids"].reverse()
        case "time":
            data["time"]["step_seconds"] = 120
        case "dimensions":
            data["dimensions"]["actions"] = 1
            data["matrices"]["B"] = [[[1], [0]], [[0], [1]]]
            data["matrices"]["E"] = [1]
        case unreachable:
            assert_never(unreachable)
    other = GNNArtifact.from_dict(data)
    # When pairing incompatible carriers/clocks, fail before inference.
    with pytest.raises(ValueError, match="process.*compatible"):
        run_closed_loop(
            agent, process_artifact=other, initial_state=0, steps=1, process_seed=7
        )


def test_impossible_process_cue_is_not_smoothed(
    agent: GNNArtifact,
    process: GNNArtifact,
) -> None:
    # Given an agent assigning zero mass to a cue that the process will emit.
    data = agent.to_dict()
    data["matrices"]["A"] = [[1, 1], [0, 0]]
    impossible = GNNArtifact.from_dict(data)
    # When the east process state emits cue 1, retain the genuine model failure.
    with pytest.raises(ValueError, match="zero probability"):
        run_closed_loop(
            impossible,
            process_artifact=process,
            initial_state=1,
            steps=1,
            process_seed=7,
        )


@pytest.mark.parametrize("state", [True, -1, 2])
def test_initial_state_must_be_an_index(agent: GNNArtifact, state: int) -> None:
    # Given an invalid index, including Python's boolean/integer alias.
    # When starting a closed loop, reject the input before sampling.
    with pytest.raises(ValueError, match="initial_state"):
        run_closed_loop(agent, initial_state=state, steps=1, process_seed=7)


@pytest.fixture
def edition(agent, process, tmp_path):
    output = tmp_path / "edition"
    create_pair_edition(agent, process, PairDesign(initial_state=0, steps=3), output)
    return output


def test_real_edition_replays_and_reports_known_brier(edition) -> None:
    result = check_pair_edition(edition, replay=True)
    summary = json.loads((edition / "summary.json").read_bytes())
    assert result.complete and not result.native_claim_ready
    assert summary[0]["mean_brier"] == pytest.approx(0.25)
    assert summary[0]["max_pymdp_identity_gap"] < 1e-5


def test_replay_rejects_resealed_wrong_scores(edition) -> None:
    path = edition / "seed_7/scores.json"
    scores = json.loads(path.read_bytes())
    scores[0]["brier_score"] = 1.75
    path.write_text(json.dumps(scores), encoding="utf-8")
    manifest = edition / "pair_manifest.json"
    data = json.loads(manifest.read_bytes())
    data["files"]["seed_7/scores.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    check_pair_edition(edition)
    with pytest.raises(ValueError, match="scores differs"):
        check_pair_edition(edition, replay=True)


@pytest.mark.parametrize(
    "mutation", ["digest", "extra", "symlink", "flag", "duplicate"]
)
def test_integrity_refuses_corrupt_editions(edition, mutation: str) -> None:
    manifest = edition / "pair_manifest.json"
    if mutation == "digest":
        (edition / "summary.json").write_text("[]", encoding="utf-8")
    elif mutation == "extra":
        (edition / "seed_7/pair_manifest.json").write_text("{}", encoding="utf-8")
    elif mutation == "symlink":
        (edition / "link.json").symlink_to(edition / "summary.json")
    elif mutation == "flag":
        data = json.loads(manifest.read_bytes())
        data["native_claim_ready"] = True
        manifest.write_text(json.dumps(data), encoding="utf-8")
    else:
        content = manifest.read_text(encoding="utf-8")
        manifest.write_text(
            content.replace('"complete": true', '"complete": true, "complete": true'),
            encoding="utf-8",
        )
    with pytest.raises(ValueError):
        check_pair_edition(edition)


def test_existing_pair_output_and_failed_attempt_are_preserved(
    agent, process, tmp_path
) -> None:
    output = tmp_path / "kept"
    output.mkdir()
    (output / "pair_manifest.json").write_bytes(b"existing edition")
    with pytest.raises(FileExistsError):
        create_pair_edition(
            agent, process, PairDesign(initial_state=0, steps=1), output
        )
    assert (output / "pair_manifest.json").read_bytes() == b"existing edition"
    data = agent.to_dict()
    data["matrices"]["A"] = [[1, 1], [0, 0]]
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="zero probability"):
        create_pair_edition(
            GNNArtifact.from_dict(data),
            process,
            PairDesign(initial_state=1, steps=1),
            failed,
        )
    assert (failed / "attempt.json").is_file()
    assert not (failed / "pair_manifest.json").exists()
    failure = json.loads((failed / "failure.json").read_bytes())
    assert failure["true_states"] == [1] and failure["observations"] == [1]


@pytest.mark.parametrize(
    "values",
    [
        {"seeds": ()},
        {"seeds": (7, 7)},
        {"seeds": (True,)},
        {"steps": 0},
        {"initial_state": True},
    ],
)
def test_invalid_design_fails_before_execution(values) -> None:
    with pytest.raises(ValueError):
        PairDesign(**{"initial_state": 0, **values})


def test_pair_svg_is_valid_and_existing_bytes_are_preserved(edition, tmp_path) -> None:
    from xml.etree import ElementTree

    output = tmp_path / "comparison.svg"
    plot_model_pairs((edition,), output)
    original = output.read_bytes()
    root = ElementTree.fromstring(original)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert all(line == line.rstrip() for line in original.decode().splitlines())
    assert len(root.findall(".//{http://www.w3.org/2000/svg}path")) > 10
    with pytest.raises(FileExistsError):
        plot_model_pairs((edition,), output)
    assert output.read_bytes() == original


def test_plot_rejects_corrupt_inputs_and_empty_selection(edition, tmp_path) -> None:
    output = tmp_path / "rejected.svg"
    with pytest.raises(ValueError):
        plot_model_pairs((), output)
    (edition / "summary.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        plot_model_pairs((edition,), output)
    assert not output.exists()


def test_readonly_cli_rejects_ignored_execution_override(edition) -> None:
    script = Path(__file__).resolve().parents[2] / "examples/cascadia/run_model_pair.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check", str(edition), "--steps", "2"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0, "an execution override must not be silently ignored"
