"""Real matrix execution, portable design admission and retained failure controls."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, assert_never

import pytest
from pydantic import ValidationError

from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.runners.closed_loop import ClosedLoopInferenceError
from geo_infer_act.runners.model_matrix import check_model_matrix, create_model_matrix
from geo_infer_act.runners.model_matrix_design import (
    MatrixCase,
    MatrixDesign,
    MatrixError,
    MatrixModel,
    load_matrix_design,
)
from geo_infer_act.runners.model_pair import PairDesign


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, MatrixDesign]:
    agent = GNNArtifact.from_dict(
        dict(
            schema_version="gnn-geo-infer/1",
            model_type="categorical",
            model_name="persistent",
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
            provenance=dict(producer="analytical fixture", source_sha256="0" * 64),
        )
    )
    weak = agent.to_dict()
    weak["matrices"]["A"] = [[0.5, 0.5], [0.5, 0.5]]
    root = tmp_path / "inputs"
    root.mkdir()
    agent.write(root / "reference.json")
    GNNArtifact.from_dict(weak).write(root / "weak.json")
    design = MatrixDesign(
        schema_version="geo-infer-act/model-matrix-design/1",
        models=(
            MatrixModel(name="reference", artifact="reference.json"),
            MatrixModel(name="weak", artifact="weak.json"),
        ),
        cases=(
            MatrixCase(
                name="matched_reference", agent="reference", process="reference"
            ),
            MatrixCase(name="matched_weak", agent="weak", process="weak"),
            MatrixCase(name="weak_agent", agent="weak", process="reference"),
            MatrixCase(name="weak_process", agent="reference", process="weak"),
        ),
        run=PairDesign(initial_state=0, steps=3, seeds=(0, 7)),
    )
    return root, design


@pytest.fixture
def edition(inputs: tuple[Path, MatrixDesign], tmp_path: Path) -> Path:
    root, design = inputs
    output = tmp_path / "matrix"
    create_model_matrix(design, root, output)
    return output


def test_real_matrix_preserves_order_clock_scores_and_both_model_identities(
    edition: Path, inputs: tuple[Path, MatrixDesign]
) -> None:
    root, design = inputs
    result = check_model_matrix(edition, replay=True)
    assert result.design == design and result.complete and not result.native_claim_ready
    assert tuple(case.name for case in result.cases) == tuple(
        case.name for case in design.cases
    )
    assert (
        result.models[0].input_sha256
        == hashlib.sha256((root / "reference.json").read_bytes()).hexdigest()
    )
    assert (
        result.models[0].artifact_sha256
        == GNNArtifact.load(root / "reference.json").digest
    )
    score = json.loads(
        (edition / "pairs/matched_reference/seed_0/scores.json").read_bytes()
    )[0]
    assert score["brier_score"] == pytest.approx(0.125)
    weak = json.loads((edition / "pairs/weak_agent/seed_0/scores.json").read_bytes())[0]
    assert weak["brier_score"] == pytest.approx(0.5)
    assert (edition / "comparison.html").is_file() and (
        edition / "comparison.svg"
    ).is_file()


Invalid = Literal[
    "duplicate_model",
    "unknown_ref",
    "unused_model",
    "duplicate_case",
    "duplicate_pair",
    "bool_seed",
    "budget",
    "absolute",
    "parent",
    "backslash",
    "alias",
    "unknown_field",
]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_model",
        "unknown_ref",
        "unused_model",
        "duplicate_case",
        "duplicate_pair",
        "bool_seed",
        "budget",
        "absolute",
        "parent",
        "backslash",
        "alias",
        "unknown_field",
    ],
)
def test_invalid_design_is_refused(
    inputs: tuple[Path, MatrixDesign], tmp_path: Path, mutation: Invalid
) -> None:
    _, design = inputs
    raw = design.model_dump(mode="json")
    match mutation:
        case "duplicate_model":
            raw["models"].append(raw["models"][0])
        case "unknown_ref":
            raw["cases"][0]["agent"] = "missing"
        case "unused_model":
            raw["models"].append({"name": "unused", "artifact": "unused.json"})
        case "duplicate_case":
            raw["cases"][1]["name"] = raw["cases"][0]["name"]
        case "duplicate_pair":
            raw["cases"][1].update(agent="reference", process="reference")
        case "bool_seed":
            raw["run"]["seeds"] = [True]
        case "budget":
            raw["run"].update(steps=512, seeds=list(range(5)))
        case "absolute":
            raw["models"][0]["artifact"] = "/external.json"
        case "parent":
            raw["models"][0]["artifact"] = "../external.json"
        case "backslash":
            raw["models"][0]["artifact"] = "directory\\external.json"
        case "alias":
            raw["models"][0]["artifact"] = "./reference.json"
        case "unknown_field":
            raw["unexpected"] = 0
        case unreachable:
            assert_never(unreachable)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((MatrixError, ValidationError)):
        load_matrix_design(path)


def test_design_duplicate_keys_and_byte_budget_fail(tmp_path: Path) -> None:
    path = tmp_path / "design.json"
    path.write_text('{"run": {}, "run": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_matrix_design(path)
    path.write_bytes(b" " * 65537)
    with pytest.raises(MatrixError, match="64 KiB"):
        load_matrix_design(path)


@pytest.mark.parametrize(
    "incompatibility", ["clock", "carrier", "start", "symlink", "missing"]
)
def test_model_preflight_fails_before_output_creation(
    inputs: tuple[Path, MatrixDesign], tmp_path: Path, incompatibility: str
) -> None:
    root, design = inputs
    path = root / "weak.json"
    data = json.loads(path.read_bytes())
    if incompatibility == "clock":
        data["time"]["step_seconds"] = 120
    elif incompatibility == "carrier":
        data["space"]["state_ids"].reverse()
    elif incompatibility == "start":
        design = design.model_copy(update={"run": PairDesign(initial_state=2, steps=1)})
    elif incompatibility == "symlink":
        path.rename(root / "preserved.json")
        path.symlink_to(root / "preserved.json")
    else:
        path.rename(root / "preserved.json")
    if incompatibility in {"clock", "carrier"}:
        path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "refused"
    with pytest.raises((MatrixError, FileNotFoundError)):
        create_model_matrix(design, root, output)
    assert not output.exists()


def test_failed_case_keeps_finished_predecessor_and_no_success_manifest(
    inputs: tuple[Path, MatrixDesign], tmp_path: Path
) -> None:
    root, _ = inputs
    safe = json.loads((root / "reference.json").read_bytes())
    safe["matrices"]["A"] = [[1, 1], [0, 0]]
    (root / "reference.json").write_text(json.dumps(safe), encoding="utf-8")
    bad = json.loads((root / "weak.json").read_bytes())
    bad["matrices"]["A"] = [[0, 0], [1, 1]]
    (root / "weak.json").write_text(json.dumps(bad), encoding="utf-8")
    design = MatrixDesign(
        schema_version="geo-infer-act/model-matrix-design/1",
        models=(
            MatrixModel(name="a", artifact="reference.json"),
            MatrixModel(name="b", artifact="weak.json"),
        ),
        cases=(
            MatrixCase(name="safe", agent="a", process="a"),
            MatrixCase(name="impossible", agent="a", process="b"),
        ),
        run=PairDesign(initial_state=0, steps=1),
    )
    output = tmp_path / "failed"
    with pytest.raises(ClosedLoopInferenceError):
        create_model_matrix(design, root, output)
    assert (output / "pairs/safe/pair_manifest.json").is_file()
    assert json.loads((output / "failure.json").read_bytes())["case"] == "impossible"
    assert not (output / "matrix_manifest.json").exists()


def test_integrity_and_replay_reject_resealed_owner_drift(edition: Path) -> None:
    manifest = edition / "matrix_manifest.json"
    data = json.loads(manifest.read_bytes())
    data["owners"]["runners/model_matrix.py"] = "f" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    check_model_matrix(edition)
    with pytest.raises(MatrixError, match="owners have changed"):
        check_model_matrix(edition, replay=True)
    (edition / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MatrixError, match="roster"):
        check_model_matrix(edition)


def test_create_only_and_cli_recheck(
    inputs: tuple[Path, MatrixDesign], tmp_path: Path
) -> None:
    root, design = inputs
    path = tmp_path / "design.json"
    path.write_text(design.model_dump_json(), encoding="utf-8")
    output = tmp_path / "cli"
    script = (
        Path(__file__).resolve().parents[2] / "examples/cascadia/run_model_matrix.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--design",
            str(path),
            "--models-root",
            str(root),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    result = subprocess.run(
        [sys.executable, str(script), "--check", str(output), "--replay"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    assert json.loads(result.stdout)["complete"] is True
    before = (output / "matrix_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_model_matrix(design, root, output)
    assert (output / "matrix_manifest.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [("complete", 1), ("native_claim_ready", 0)])
def test_receipt_rejects_numeric_boolean_aliases(
    edition: Path, field: str, value: int
) -> None:
    manifest = edition / "matrix_manifest.json"
    data = json.loads(manifest.read_bytes())
    data[field] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        check_model_matrix(edition)


def test_resealed_semantic_corruption_is_refused(edition: Path, tmp_path: Path) -> None:
    for name in (
        "flags",
        "design",
        "models",
        "cases",
        "model_digest",
        "pair_digest",
        "binding",
        "html",
    ):
        folder = tmp_path / name
        shutil.copytree(edition, folder)
        path = folder / "matrix_manifest.json"
        raw = json.loads(path.read_bytes())
        match name:
            case "flags":
                raw["native_claim_ready"] = True
            case "design":
                raw["design"]["run"]["steps"] = 2
            case "models":
                raw["models"].reverse()
            case "cases":
                raw["cases"].reverse()
            case "model_digest":
                raw["models"][0]["artifact_sha256"] = "f" * 64
            case "pair_digest":
                raw["cases"][0]["manifest_sha256"] = "f" * 64
            case "binding":
                raw["design"]["run"]["steps"] = 2
                (folder / "matrix_design.json").write_text(
                    json.dumps(raw["design"]), encoding="utf-8"
                )
            case "html":
                (folder / "comparison.html").write_text(
                    "stale comparison", encoding="utf-8"
                )
        raw["files"] = {
            item.relative_to(folder).as_posix(): hashlib.sha256(
                item.read_bytes()
            ).hexdigest()
            for item in folder.rglob("*")
            if item.is_file() and item != path
        }
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError):
            check_model_matrix(folder)


def test_manifest_budget_duplicate_fields_and_symlinks(
    edition: Path, tmp_path: Path
) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(edition, target_is_directory=True)
    with pytest.raises(MatrixError, match="symlink"):
        check_model_matrix(alias)
    link = edition / "extra.json"
    link.symlink_to(edition / "matrix_design.json")
    with pytest.raises(MatrixError, match="symlink"):
        check_model_matrix(edition)
    link.rename(tmp_path / "preserved-link")
    manifest = edition / "matrix_manifest.json"
    manifest.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(MatrixError, match="eight MiB"):
        check_model_matrix(edition)
    manifest.write_text('{"complete": true, "complete": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        check_model_matrix(edition)


def test_oversized_artifact_refused_before_output(
    inputs: tuple[Path, MatrixDesign], tmp_path: Path
) -> None:
    from geo_infer_act.core.gnn_contract import MAX_ARTIFACT_BYTES

    root, design = inputs
    (root / "weak.json").write_bytes(b" " * (MAX_ARTIFACT_BYTES + 1))
    output = tmp_path / "refused"
    with pytest.raises(MatrixError, match="byte budget"):
        create_model_matrix(design, root, output)
    assert not output.exists()
