"""Closed-file report integrity, distinct from live source or execution currency."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Digest,
    ObservatoryConfig,
    Strategy,
    StrictModel,
    digest,
    strict_bytes,
)

FILE_DIGESTS = TypeAdapter(dict[str, Digest])


class ReportIdentity(BaseModel):
    """Parse report identity while preserving its extensible scientific payload."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    schema_version: Literal["geo-infer-place/california-report/1"]
    config: ObservatoryConfig
    landscape_sha256: Digest
    source_receipt_sha256: Digest
    experiment_sha256: Digest
    cells: int = Field(ge=1)
    counties: Literal[58]


class ReportIntegrity(StrictModel):
    """A supplied-manifest byte check, carrying no authenticity or replay claim."""

    schema_version: Literal["geo-infer-place/california-report-integrity/1"]
    report_sha256: Digest
    files: dict[str, Digest]


def report_roster(config: ObservatoryConfig) -> set[str]:
    """Return the complete report surface required by the selected design."""
    return {
        "report.json", "historical_indicators.png", "empirical_priors.png",
        "score_comparison.png", "trajectory.csv", "comparison.csv", "COMPARISON.md",
    } | {
        f"{strategy.value}_allocation.png" for strategy in Strategy
    } | {
        f"{strategy.value}_{round_index:02d}.png"
        for strategy in Strategy
        for round_index in range(config.rounds + 1)
    }


def check_report(folder: Path) -> ReportIntegrity:
    """Reject missing, extra, linked or changed files without contacting a service."""
    if folder.is_symlink():
        raise CaliforniaError("report folder is a symlink")
    manifest = FILE_DIGESTS.validate_json(strict_bytes(folder / "files.json"))
    if (
        any(Path(name).name != name or name in {".", "..", "files.json"} for name in manifest)
        or {path.name for path in folder.iterdir()} != set(manifest) | {"files.json"}
    ):
        raise CaliforniaError("report output roster differs from its manifest")
    for name, checksum in manifest.items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise CaliforniaError("report output differs from its sealed bytes")
    identity = ReportIdentity.model_validate_json(
        strict_bytes(folder / "report.json", limit=256 * 1024 * 1024)
    )
    if set(manifest) != report_roster(identity.config):
        raise CaliforniaError("report output roster differs from its design")
    return ReportIntegrity(
        schema_version="geo-infer-place/california-report-integrity/1",
        report_sha256=manifest["report.json"],
        files=manifest,
    )
