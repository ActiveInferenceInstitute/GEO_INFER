"""Create-only execution editions for separate GNN agent/process artifacts."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from geo_infer_act.core.gnn_contract import GNNArtifact, _unique_object
from geo_infer_act.runners.closed_loop import (
    ClosedLoopResult,
    ClosedLoopInferenceError,
    run_closed_loop,
    trace_diagnostics,
)
from geo_infer_act.runners.study_diagnostics import diagnose_frame


class PairDesign(BaseModel):
    """Strict ordered seed grid, bounded before any output is created."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    initial_state: int = Field(ge=0)
    steps: int = Field(default=48, ge=1, le=512)
    seeds: tuple[int, ...] = (7,)
    agent_seed: int = Field(default=0, ge=0, le=2**32 - 513)

    @model_validator(mode="after")
    def bounded_grid(self) -> PairDesign:
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds
            )
            or len(self.seeds) * self.steps > 8192
        ):
            raise ValueError(
                "seeds must be unique uint32 indices within the 8192-step budget"
            )
        return self


class PairManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal["geo-infer-act/model-pair/1"]
    agent_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design: PairDesign
    native_claim_ready: bool
    backend: dict[str, str]
    complete: bool
    files: dict[str, str]
    runner_sha256: dict[str, str]


def create_pair_edition(
    agent: GNNArtifact,
    process: GNNArtifact,
    design: PairDesign,
    output: Path,
) -> Path:
    """Retain model identities and every real run; issue a manifest only on success."""
    agent_data, process_data = agent.to_dict(), process.to_dict()
    if any(
        agent_data[key] != process_data[key] for key in ("dimensions", "space", "time")
    ):
        raise ValueError("process carrier, dimensions and clock must be compatible")
    if design.initial_state >= agent_data["dimensions"]["states"]:
        raise ValueError("initial_state must be an integer in range")
    output.mkdir(parents=True, exist_ok=False)
    owners = runner_fingerprint()
    agent.write(output / "agent.geo-infer.json")
    process.write(output / "process.geo-infer.json")
    identity: dict[str, JsonValue] = {
        "schema_version": "geo-infer-act/model-pair/1",
        "agent_artifact_sha256": agent.digest,
        "process_artifact_sha256": process.digest,
        "design": design.model_dump(mode="json"),
        "native_claim_ready": False,
        "backend": {"inferactively-pymdp": version("inferactively-pymdp")},
    }
    _write(output / "attempt.json", identity)
    summaries = []
    for seed in design.seeds:
        try:
            run = run_closed_loop(
                agent,
                process_artifact=process,
                initial_state=design.initial_state,
                steps=design.steps,
                process_seed=seed,
                agent_seed=design.agent_seed,
            )
        except ClosedLoopInferenceError as error:
            _write(
                output / "failure.json",
                {
                    "process_seed": seed,
                    "error": str(error),
                    "true_states": list(error.true_states),
                    "observations": list(error.observations),
                    "native_claim_ready": False,
                },
            )
            raise
        folder = output / f"seed_{seed}"
        folder.mkdir()
        _write(folder / "closed_loop.json", run.to_dict())
        measures = _measure(agent, run)
        _write(folder / "policy_terms.json", measures["policy_terms"])
        _write(folder / "scores.json", measures["scores"])
        summaries.append(measures["summary"])
    _write(output / "summary.json", summaries)
    files = {
        path.relative_to(output).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    if runner_fingerprint() != owners:
        raise RuntimeError("pair execution owners changed during execution")
    manifest = output / "pair_manifest.json"
    record: dict[str, JsonValue] = dict(identity)
    record["complete"] = True
    record["files"] = {key: value for key, value in files.items()}
    record["runner_sha256"] = {key: value for key, value in owners.items()}
    _write(manifest, record)
    return manifest


def _measure(agent: GNNArtifact, run: ClosedLoopResult) -> dict[str, JsonValue]:
    data = agent.to_dict()
    terms = trace_diagnostics(agent, run)
    scores = tuple(
        diagnose_frame(
            tuple(row["posterior"]),
            state,
            tuple(row["policy_posterior"]),
            elapsed_seconds=index * data["time"]["step_seconds"],
        )
        for index, (row, state) in enumerate(zip(run.trace["steps"], run.true_states))
    )
    gap = max(row["pymdp_identity_gap"] for row in terms)
    if gap >= 1e-5:
        raise RuntimeError("pair policy scores disagree with pymdp by at least 1e-5")
    return {
        "policy_terms": [{key: value for key, value in row.items()} for row in terms],
        "scores": [row.model_dump(mode="json") for row in scores],
        "summary": {
            "seed": run.process_seed,
            "observed_steps": len(scores),
            "mean_brier": sum(row.brier_score for row in scores) / len(scores),
            "mean_free_energy": sum(row["free_energy"] for row in run.trace["steps"])
            / len(scores),
            "max_pymdp_identity_gap": gap,
        },
    }


def runner_fingerprint() -> dict[str, str]:
    package = Path(__file__).parents[1]
    names = (
        "runners/model_pair.py",
        "runners/closed_loop.py",
        "runners/study_diagnostics.py",
        "core/gnn_contract.py",
        "core/efe_decomposition.py",
        "utils/pymdp_adapter.py",
    )
    return {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in names
    }


def check_pair_edition(output: Path, *, replay: bool = False) -> PairManifest:
    """Check closed file integrity; optional replay performs fresh process and agent execution."""
    if output.is_symlink() or (output / "pair_manifest.json").is_symlink():
        raise ValueError("pair edition contains a symlink")
    raw = (output / "pair_manifest.json").read_bytes()
    json.loads(raw, object_pairs_hook=_unique_object)
    record = PairManifest.model_validate_json(raw)
    if not record.complete or record.native_claim_ready:
        raise ValueError("pair manifest has invalid readiness flags")
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("pair edition contains a symlink")
        if path.is_file() and path != output / "pair_manifest.json":
            actual.add(path.relative_to(output).as_posix())
    if actual != set(record.files):
        raise ValueError("pair file roster differs")
    for name, digest in record.files.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"pair file digest differs: {name}")
    agent = GNNArtifact.load(output / "agent.geo-infer.json")
    process = GNNArtifact.load(output / "process.geo-infer.json")
    if (
        agent.digest != record.agent_artifact_sha256
        or process.digest != record.process_artifact_sha256
    ):
        raise ValueError("pair artifact identity differs")
    if replay:
        if record.runner_sha256 != runner_fingerprint() or record.backend != {
            "inferactively-pymdp": version("inferactively-pymdp")
        }:
            raise ValueError("pair execution owners or backend have changed")
        summaries = []
        for seed in record.design.seeds:
            rerun = run_closed_loop(
                agent,
                process_artifact=process,
                initial_state=record.design.initial_state,
                steps=record.design.steps,
                process_seed=seed,
                agent_seed=record.design.agent_seed,
            )
            retained = json.loads(
                (output / f"seed_{seed}/closed_loop.json").read_bytes(),
                object_pairs_hook=_unique_object,
            )
            if rerun.to_dict() != retained:
                raise ValueError(f"pair replay differs: seed {seed}")
            measures = _measure(agent, rerun)
            for key in ("policy_terms", "scores"):
                saved = json.loads(
                    (output / f"seed_{seed}/{key}.json").read_bytes(),
                    object_pairs_hook=_unique_object,
                )
                if saved != measures[key]:
                    raise ValueError(f"pair replay {key} differs: seed {seed}")
            summaries.append(measures["summary"])
        if (
            json.loads(
                (output / "summary.json").read_bytes(), object_pairs_hook=_unique_object
            )
            != summaries
        ):
            raise ValueError("pair replay summary differs")
    return record


def _write(path: Path, value: JsonValue) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
