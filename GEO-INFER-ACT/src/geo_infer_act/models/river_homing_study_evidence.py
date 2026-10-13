"""Static cross-file study checks; these do not execute inference or replay."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.models.river_homing_study import (
    RunMeasures,
    SeedOutcome,
    StudyDesign,
)


class _Step(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    observation: int


class _Trace(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    schema_version: Literal["gnn-geo-infer/1"]
    artifact_sha256: str
    state_ids: list[str]
    random_seed: int
    steps: list[_Step]


class _Run(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    process_seed: int
    agent_seed: int
    true_states: list[int]
    observations: list[int]
    trace: _Trace


def retained_run_problems(
    root: Path, item: SeedOutcome, design: StudyDesign
) -> tuple[str, ...]:
    """Compare the retained run against its declared design and condition artifact."""
    directory = root / item.condition.value / f"seed_{item.process_seed}"
    try:
        measures = RunMeasures.model_validate_json(
            (directory / "summary.json").read_bytes()
        )
        outcome = SeedOutcome.model_validate_json(
            (directory / "outcome.json").read_bytes()
        )
        run = _Run.model_validate_json((directory / "closed_loop.json").read_bytes())
        artifact = GNNArtifact.load(
            directory.parent / "smith_river_homing.geo-infer.json"
        )
        state_ids = TypeAdapter(list[str]).validate_json(
            (directory.parent / "smith_river_homing.state_ids.json").read_bytes(),
            strict=True,
        )
    except (OSError, ValueError, ValidationError) as exc:
        return (f"invalid retained run {item.run_directory}: {exc}",)
    problems = []
    if measures != item.measures:
        problems.append("run summary differs from aggregate measures")
    if outcome != item:
        problems.append("run outcome differs from aggregate outcome")
    if run.process_seed != item.process_seed:
        problems.append("process seed differs from outcome")
    if (
        run.agent_seed != design.agent_seed
        or run.trace.random_seed != design.agent_seed
    ):
        problems.append("agent seed differs from design")
    if any(
        len(values) != design.steps
        for values in (run.true_states, run.observations, run.trace.steps)
    ):
        problems.append("closed-loop horizon differs from design")
    data = artifact.to_dict()
    if run.trace.artifact_sha256 != artifact.digest:
        problems.append("trace artifact digest differs from condition artifact")
    if (
        run.trace.state_ids != data["space"]["state_ids"]
        or state_ids != run.trace.state_ids
    ):
        problems.append(
            "trace state order differs from condition artifact or state sidecar"
        )
    if any(not 0 <= state < data["dimensions"]["states"] for state in run.true_states):
        problems.append("true state outside condition artifact")
    if any(
        not 0 <= value < data["dimensions"]["observations"]
        for value in run.observations
    ):
        problems.append("observation outside condition artifact")
    if run.observations != [step.observation for step in run.trace.steps]:
        problems.append("trace observations differ from closed-loop observations")
    return tuple(f"{item.run_directory}: {problem}" for problem in problems)
