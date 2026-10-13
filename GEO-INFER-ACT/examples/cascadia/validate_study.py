# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run python GEO-INFER-ACT/examples/cascadia/validate_study.py OUTPUT
"""Read-only custody and summary checks; this does not replay or compile Lean."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from study_contract import ConditionSummary, SeedOutcome, StudyDesign, summarize
from study_custody import SourceCustody, source_problems
from geo_infer_act.models.river_homing_study_evidence import retained_run_problems


class StudyManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    schema_version: Literal["geo-infer-act/cascadia-seed-study/1"]
    complete: bool
    qualified: bool
    native_claim_ready: Literal[False]
    design: StudyDesign
    versions: dict[str, str]
    source_custody: SourceCustody
    source_drift: tuple[str, ...]
    files: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]


def validate_study(output: Path, geo_infer_root: Path | None = None) -> tuple[str, ...]:
    """Re-hash every retained file and recompute grid, gates and summary counts."""
    root = output.resolve()
    manifest = StudyManifest.model_validate_json(
        (root / "study_manifest.json").read_bytes()
    )
    problems = []
    if not manifest.complete or not manifest.qualified:
        problems.append("manifest is incomplete or unqualified")
    if manifest.source_drift:
        problems.append("source drift was recorded during execution")
    for name, digest in manifest.source_custody.source_owner_sha256.items():
        if manifest.files.get(f"source_owners/{name}") != digest:
            problems.append(f"source owner snapshot is unbound: {name}")
    if geo_infer_root is not None:
        problems.extend(
            source_problems(geo_infer_root.resolve(), manifest.source_custody)
        )
    for name, digest in manifest.files.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            problems.append(f"unsafe manifest path: {name}")
            continue
        if not path.resolve().is_relative_to(root) or not path.is_file():
            problems.append(f"missing or external file: {name}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"digest mismatch: {name}")
    for required in (
        "outcomes.json",
        "summary.json",
        "study_design.json",
        "reference_config.yaml",
        "seed_outcomes.png",
    ):
        if required not in manifest.files:
            problems.append(f"required file is unbound: {required}")
    if problems:
        return tuple(problems)
    outcomes = TypeAdapter(tuple[SeedOutcome, ...]).validate_json(
        (root / "outcomes.json").read_bytes()
    )
    actual = [(item.condition, item.process_seed) for item in outcomes]
    expected = [
        (condition, seed)
        for condition in manifest.design.conditions
        for seed in manifest.design.seeds
    ]
    if actual != expected:
        problems.append("outcome grid differs from the declared ordered design")
    for item in outcomes:
        expected_directory = f"{item.condition.value}/seed_{item.process_seed}"
        if item.run_directory != expected_directory:
            problems.append(
                "run directory differs from the declared condition and seed"
            )
        for filename in (
            "closed_loop.json",
            "diagnostics.json",
            "summary.json",
            "outcome.json",
        ):
            if f"{expected_directory}/{filename}" not in manifest.files:
                problems.append(f"run file is unbound: {expected_directory}/{filename}")
        if item.measures.steps != manifest.design.steps:
            problems.append("run horizon differs from the design")
        computed = item.measures.qualification_problems()
        if computed or item.problems != computed or item.qualified != (not computed):
            problems.append(f"invalid qualification: {expected_directory}")
        problems.extend(retained_run_problems(root, item, manifest.design))
    for condition in manifest.design.conditions:
        for suffix in (
            "gnn.md",
            "state_ids.json",
            "domain.json",
            "geo-infer.json",
            "export.json",
        ):
            name = f"{condition.value}/smith_river_homing.{suffix}"
            if name not in manifest.files:
                problems.append(f"condition file is unbound: {name}")
    stored = TypeAdapter(tuple[ConditionSummary, ...]).validate_json(
        (root / "summary.json").read_bytes()
    )
    if stored != summarize(outcomes):
        problems.append("condition summary differs from retained seed outcomes")
    declared = StudyDesign.model_validate_json(
        (root / "study_design.json").read_bytes()
    )
    if declared != manifest.design:
        problems.append("design sidecar differs from manifest")
    return tuple(problems)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--geo-infer-root", type=Path)
    args = parser.parse_args(argv)
    problems = validate_study(args.output, args.geo_infer_root)
    print(
        json.dumps(
            {
                "custody_and_summary_ready": not problems,
                "native_claim_ready": False,
                "problems": problems,
            },
            indent=2,
        )
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
