"""Deterministic offline comparison of integrity-checked model-pair records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from geo_infer_act.core.gnn_contract import GNNArtifact, _unique_object
from geo_infer_act.runners.model_pair import check_pair_edition
from geo_infer_act.runners.study_diagnostics import StepDiagnostics

__all__ = ["PairComparison", "load_pair_comparison", "render_pair_comparison"]
TEMPLATE: Final = Path(__file__).with_name("model_pair_explorer.html")


class PairDisplayError(ValueError):
    """A selected edition cannot supply a finite, aligned display."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"invalid pair comparison: {field}")


class PairRunView(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    seed: int
    frames: tuple[StepDiagnostics, ...]


class PairEditionView(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    name: str
    manifest_sha256: str
    agent_sha256: str
    process_sha256: str
    runs: tuple[PairRunView, ...]


class PairComparison(BaseModel):
    """The display's ordered records; no inferred intermediate values."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: str = "geo-infer-act/pair-comparison/1"
    seeds: tuple[int, ...]
    steps: int
    hours: tuple[float, ...]
    editions: tuple[PairEditionView, ...]
    native_claim_ready: bool = False


def load_pair_comparison(editions: Sequence[Path]) -> PairComparison:
    """Parse saved scores only after closed-roster integrity qualification."""
    if not editions or len({path.resolve() for path in editions}) != len(editions):
        raise PairDisplayError("empty or duplicate edition selection")
    views = []
    adapter = TypeAdapter(tuple[StepDiagnostics, ...])
    common_seeds: tuple[int, ...] | None = None
    common_seconds: tuple[float, ...] | None = None
    for path in editions:
        record = check_pair_edition(path)
        clock = GNNArtifact.load(path / "agent.geo-infer.json").to_dict()["time"][
            "step_seconds"
        ]
        seconds = tuple(index * float(clock) for index in range(record.design.steps))
        if common_seeds is not None and (
            record.design.seeds != common_seeds or seconds != common_seconds
        ):
            raise PairDisplayError("ordered seed grid, horizon or clock differs")
        common_seeds, common_seconds = record.design.seeds, seconds
        runs = []
        for seed in record.design.seeds:
            raw = (path / f"seed_{seed}/scores.json").read_bytes()
            json.loads(raw, object_pairs_hook=_unique_object)
            frames = adapter.validate_json(raw)
            if tuple(frame.elapsed_seconds for frame in frames) != seconds:
                raise PairDisplayError("score horizon or physical clock differs")
            runs.append(PairRunView(seed=seed, frames=frames))
        views.append(
            PairEditionView(
                name=path.name,
                manifest_sha256=hashlib.sha256(
                    (path / "pair_manifest.json").read_bytes()
                ).hexdigest(),
                agent_sha256=record.agent_artifact_sha256,
                process_sha256=record.process_artifact_sha256,
                runs=tuple(runs),
            )
        )
    assert common_seeds is not None and common_seconds is not None
    return PairComparison(
        seeds=common_seeds,
        steps=len(common_seconds),
        hours=tuple(seconds / 3600 for seconds in common_seconds),
        editions=tuple(views),
    )


def render_pair_comparison(
    editions: Sequence[Path], output: Path, *, check: bool = False
) -> Path:
    """Create an offline HTML edition, or compare existing bytes without repair."""
    view = load_pair_comparison(editions)
    payload = view.model_dump_json().replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__PAIR_DATA__") != 1:
        raise PairDisplayError("template data slot")
    rendered = template.replace("__PAIR_DATA__", payload)
    if check:
        if output.is_symlink() or output.read_bytes() != rendered.encode("utf-8"):
            raise PairDisplayError("HTML bytes differ")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    return output
