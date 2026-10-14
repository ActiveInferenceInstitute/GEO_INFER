"""Create and recheck real model-pair matrices without duplicating inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from geo_infer_act.core.gnn_contract import (
    GNNArtifact,
    MAX_ARTIFACT_BYTES,
    _unique_object,
)
from geo_infer_act.runners.closed_loop import ClosedLoopInferenceError
from geo_infer_act.runners.model_matrix_design import (
    Digest,
    MatrixDesign,
    MatrixError,
    MatrixModel,
    Name,
)
from geo_infer_act.runners.model_pair import (
    check_pair_edition,
    create_pair_edition,
    runner_fingerprint,
)
from geo_infer_act.runners.model_pair_explorer import render_pair_comparison
from geo_infer_act.runners.model_pair_plot import plot_model_pairs


class ModelIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    name: Name
    input_sha256: Digest
    artifact_sha256: Digest


class CaseIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    name: Name
    manifest_sha256: Digest


class MatrixManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal["geo-infer-act/model-matrix/1"]
    design: MatrixDesign
    models: tuple[ModelIdentity, ...]
    cases: tuple[CaseIdentity, ...]
    files: dict[str, Digest] = Field(min_length=1)
    owners: dict[str, Digest] = Field(min_length=1)
    complete: bool = True
    native_claim_ready: bool = False

    @model_validator(mode="after")
    def evidence_flags(self) -> Self:
        if not self.complete or self.native_claim_ready:
            raise MatrixError("invalid matrix evidence flags")
        return self


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    identity: ModelIdentity
    artifact: GNNArtifact


def _snapshot(root: Path, model: MatrixModel) -> ModelSnapshot:
    path = root
    for part in model.artifact.split("/"):
        path = path / part
        if path.is_symlink():
            raise MatrixError("model path contains a symlink")
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise MatrixError("model exceeds artifact byte budget")
    artifact = GNNArtifact(raw.decode("utf-8"))
    return ModelSnapshot(
        ModelIdentity(
            name=model.name,
            input_sha256=hashlib.sha256(raw).hexdigest(),
            artifact_sha256=artifact.digest,
        ),
        artifact,
    )


def _owners() -> dict[str, str]:
    package = Path(__file__).parents[1]
    names = (
        "runners/model_matrix.py",
        "runners/model_matrix_design.py",
        "runners/model_pair_explorer.py",
        "runners/model_pair_explorer.html",
        "runners/model_pair_plot.py",
    )
    return runner_fingerprint() | {name: _digest(package / name) for name in names}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(folder: Path) -> dict[str, str]:
    if folder.is_symlink():
        raise MatrixError("edition is a symlink")
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise MatrixError("edition contains a symlink")
        if path.is_file() and path != folder / "matrix_manifest.json":
            result[path.relative_to(folder).as_posix()] = _digest(path)
    return result


def _compatible(design: MatrixDesign, snapshots: tuple[ModelSnapshot, ...]) -> None:
    first = snapshots[0].artifact.to_dict()
    for snapshot in snapshots:
        model = snapshot.artifact.to_dict()
        if any(model[key] != first[key] for key in ("dimensions", "space", "time")):
            raise MatrixError("all selected carriers, dimensions and clocks must agree")
    if design.run.initial_state >= first["dimensions"]["states"]:
        raise MatrixError("initial state outside the selected carrier")


def create_model_matrix(design: MatrixDesign, models_root: Path, output: Path) -> Path:
    """Preflight all models, freeze snapshots, then execute every declared pair."""
    owners = _owners()
    snapshots = {
        model.name: _snapshot(models_root.resolve(strict=True), model)
        for model in design.models
    }
    _compatible(design, tuple(snapshots.values()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "pairs").mkdir()
    with (output / "matrix_design.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(design.model_dump_json(indent=2) + "\n")
    for name, snapshot in snapshots.items():
        snapshot.artifact.write(output / f"models/{name}.geo-infer.json")
    cases = []
    for case in design.cases:
        try:
            manifest = create_pair_edition(
                snapshots[case.agent].artifact,
                snapshots[case.process].artifact,
                design.run,
                output / "pairs" / case.name,
            )
        except (ClosedLoopInferenceError, ValueError, RuntimeError, OSError) as error:
            with (output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "case": case.name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "native_claim_ready": False,
                    },
                    stream,
                    allow_nan=False,
                    indent=2,
                )
                stream.write("\n")
            raise
        cases.append(CaseIdentity(name=case.name, manifest_sha256=_digest(manifest)))
    editions = tuple(output / "pairs" / case.name for case in design.cases)
    render_pair_comparison(editions, output / "comparison.html")
    plot_model_pairs(editions, output / "comparison.svg")
    if _owners() != owners:
        raise MatrixError("matrix execution or display owners changed during execution")
    record = MatrixManifest(
        schema_version="geo-infer-act/model-matrix/1",
        design=design,
        models=tuple(s.identity for s in snapshots.values()),
        cases=tuple(cases),
        files=_files(output),
        owners=owners,
    )
    manifest = output / "matrix_manifest.json"
    with manifest.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(record.model_dump_json(indent=2) + "\n")
    return manifest


def check_model_matrix(output: Path, *, replay: bool = False) -> MatrixManifest:
    """Qualify a closed matrix; replay additionally re-executes every recorded pair."""
    path = output / "matrix_manifest.json"
    if output.is_symlink() or path.is_symlink():
        raise MatrixError("manifest is a symlink")
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise MatrixError("manifest exceeds eight MiB")
    json.loads(raw, object_pairs_hook=_unique_object)
    record = MatrixManifest.model_validate_json(raw)
    owners = _owners()
    if replay and record.owners != owners:
        raise MatrixError("matrix execution or display owners have changed")
    if _files(output) != record.files:
        raise MatrixError("closed file roster or digest differs")
    design_raw = (output / "matrix_design.json").read_bytes()
    json.loads(design_raw, object_pairs_hook=_unique_object)
    if MatrixDesign.model_validate_json(design_raw) != record.design:
        raise MatrixError("design snapshot differs")
    if tuple(model.name for model in record.models) != tuple(
        model.name for model in record.design.models
    ):
        raise MatrixError("model identity roster differs")
    if tuple(case.name for case in record.cases) != tuple(
        case.name for case in record.design.cases
    ):
        raise MatrixError("case identity roster differs")
    identities = {model.name: model.artifact_sha256 for model in record.models}
    snapshots = tuple(
        ModelSnapshot(
            model, GNNArtifact.load(output / f"models/{model.name}.geo-infer.json")
        )
        for model in record.models
    )
    _compatible(record.design, snapshots)
    for snapshot in snapshots:
        if snapshot.artifact.digest != snapshot.identity.artifact_sha256:
            raise MatrixError("model snapshot identity differs")
    for case, identity in zip(record.design.cases, record.cases, strict=True):
        folder = output / "pairs" / case.name
        if _digest(folder / "pair_manifest.json") != identity.manifest_sha256:
            raise MatrixError("pair manifest identity differs")
        pair = check_pair_edition(folder, replay=replay)
        if (
            pair.design != record.design.run
            or pair.agent_artifact_sha256 != identities[case.agent]
            or pair.process_artifact_sha256 != identities[case.process]
        ):
            raise MatrixError("case binding or common run design differs")
    render_pair_comparison(
        tuple(output / "pairs" / case.name for case in record.design.cases),
        output / "comparison.html",
        check=True,
    )
    if replay and _owners() != owners:
        raise MatrixError("matrix owners changed during replay")
    return record
