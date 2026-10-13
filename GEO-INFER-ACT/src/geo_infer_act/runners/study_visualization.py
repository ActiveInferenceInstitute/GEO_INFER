"""Typed, finite projections of retained Cascadia trajectories for offline visuals."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from geo_infer_act.models.river_homing_study import SeedOutcome
from geo_infer_act.runners.study_diagnostics import StepDiagnostics, diagnose_frame


class Frame(BaseModel):
    """Recorded probabilities; no interpolation or probability amplification."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    action: int = Field(ge=0)
    observation: int = Field(ge=0)
    posterior: tuple[float, ...]
    policy_posterior: tuple[float, ...]
    free_energy: float = Field(allow_inf_nan=False)
    timestamp: datetime


class Trace(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    state_ids: tuple[str, ...]
    steps: tuple[Frame, ...]


class Trajectory(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    true_states: tuple[int, ...]
    observations: tuple[int, ...]
    trace: Trace


class Cell(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    cell: str
    label: str
    role: str
    successor: str | None
    upstream: str | None
    side: str | None


class Network(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    cells: tuple[Cell, ...]


class Domain(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    network: Network
    action_labels: tuple[str, ...]
    observation_labels: tuple[str, ...]


class DisplayCell(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str
    role: str
    latitude: float
    longitude: float


class DisplayRun(BaseModel):
    model_config = ConfigDict(frozen=True)
    outcome: SeedOutcome
    trajectory: Trajectory
    cells: tuple[DisplayCell, ...]
    edges: tuple[tuple[int, int], ...]
    action_labels: tuple[str, ...]
    observation_labels: tuple[str, ...]
    diagnostics: tuple[StepDiagnostics, ...]


class Clock(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    step_seconds: float = Field(gt=0, allow_inf_nan=False)


class ClockArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    time: Clock


class StudyView(BaseModel):
    model_config = ConfigDict(frozen=True)
    manifest_sha256: str
    runs: tuple[DisplayRun, ...]
    native_claim_ready: bool = False


def project_study(root: Path) -> StudyView:
    """Read a statically qualified edition and check dimensions before plotting."""
    import h3

    outcomes = TypeAdapter(tuple[SeedOutcome, ...]).validate_json(
        (root / "outcomes.json").read_bytes()
    )
    runs = []
    for outcome in outcomes:
        folder = root / outcome.condition.value
        domain = Domain.model_validate_json(
            (folder / "smith_river_homing.domain.json").read_bytes()
        )
        trajectory = Trajectory.model_validate_json(
            (root / outcome.run_directory / "closed_loop.json").read_bytes()
        )
        clock = ClockArtifact.model_validate_json(
            (folder / "smith_river_homing.geo-infer.json").read_bytes()
        ).time
        rows = domain.network.cells
        identifiers = tuple(row.cell for row in rows)
        if trajectory.trace.state_ids != identifiers:
            message = "display state order differs from the domain"
            raise ValueError(message)
        if (
            len(trajectory.true_states) != outcome.measures.steps
            or len(trajectory.trace.steps) != outcome.measures.steps
            or trajectory.observations
            != tuple(frame.observation for frame in trajectory.trace.steps)
        ):
            message = "display horizon differs from retained outcome"
            raise ValueError(message)
        diagnostics = []
        start = trajectory.trace.steps[0].timestamp
        if start.utcoffset() is None:
            message = "display timestamps must be timezone-aware"
            raise ValueError(message)
        for step, (state, frame) in enumerate(
            zip(trajectory.true_states, trajectory.trace.steps, strict=True)
        ):
            if frame.timestamp.utcoffset() is None:
                message = "display timestamps must be timezone-aware"
                raise ValueError(message)
            elapsed = (frame.timestamp - start).total_seconds()
            if not math.isclose(
                elapsed, step * clock.step_seconds, rel_tol=0, abs_tol=1e-6
            ):
                message = "display timestamps differ from the artifact clock"
                raise ValueError(message)
            if not 0 <= state < len(rows) or not 0 <= frame.action < len(
                domain.action_labels
            ):
                message = "display state or action is outside its carrier"
                raise ValueError(message)
            if not 0 <= frame.observation < len(domain.observation_labels):
                message = "display observation is outside its carrier"
                raise ValueError(message)
            for probabilities, size in (
                (frame.posterior, len(rows)),
                (frame.policy_posterior, len(domain.action_labels)),
            ):
                if (
                    len(probabilities) != size
                    or any(
                        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
                    )
                    or not math.isclose(math.fsum(probabilities), 1, abs_tol=1e-8)
                ):
                    message = "display probability vector is malformed"
                    raise ValueError(message)
            diagnostics.append(
                diagnose_frame(
                    frame.posterior,
                    state,
                    frame.policy_posterior,
                    elapsed_seconds=elapsed,
                )
            )
        index = {cell: k for k, cell in enumerate(identifiers)}
        edges: set[tuple[int, int]] = set()
        for k, row in enumerate(rows):
            for neighbor in (row.successor, row.upstream, row.side):
                if neighbor is not None:
                    if neighbor not in index or not h3.are_neighbor_cells(
                        row.cell, neighbor
                    ):
                        message = (
                            "display relation is outside the recorded H3 neighborhood"
                        )
                        raise ValueError(message)
                    edges.add((min(k, index[neighbor]), max(k, index[neighbor])))
        cells = tuple(
            DisplayCell(
                label=row.label,
                role=row.role,
                latitude=h3.cell_to_latlng(row.cell)[0],
                longitude=h3.cell_to_latlng(row.cell)[1],
            )
            for row in rows
        )
        runs.append(
            DisplayRun(
                outcome=outcome,
                trajectory=trajectory,
                cells=cells,
                edges=tuple(sorted(edges)),
                action_labels=domain.action_labels,
                observation_labels=domain.observation_labels,
                diagnostics=tuple(diagnostics),
            )
        )
    if not runs:
        message = "display requires at least one retained trajectory"
        raise ValueError(message)
    return StudyView(
        manifest_sha256=hashlib.sha256(
            (root / "study_manifest.json").read_bytes()
        ).hexdigest(),
        runs=tuple(runs),
    )
