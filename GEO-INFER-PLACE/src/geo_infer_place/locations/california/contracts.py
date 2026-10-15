"""Strict boundaries for statewide data custody and monitoring configuration."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from importlib.metadata import version
from importlib.resources import files
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

RAW_BYTES: Final = 20 * 1024 * 1024
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CaliforniaError(ValueError):
    """An explicit data, model, numerical or custody boundary failed."""

    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        super().__init__(f"California observatory: {boundary}")


class Hazard(StrEnum):
    FIRE = "fire"
    SEISMIC = "seismic"
    DROUGHT = "drought"


class Strategy(StrEnum):
    ACTIVE = "active"
    QUOTA_ACTIVE = "quota_active"
    ROTATING = "rotating"


class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class ExecutionBinding(StrictModel):
    owners: dict[str, Digest]
    executable: str
    prefix: str
    base_prefix: str
    python: str
    packages: dict[str, str]


def execution_binding() -> ExecutionBinding:
    """Bind the installed owners without assuming a monorepo layout."""
    local = files("geo_infer_place.locations.california")
    owners = {
        f"GEO-INFER-PLACE/src/geo_infer_place/locations/california/{item.name}": hashlib.sha256(
            item.read_bytes()
        ).hexdigest()
        for item in sorted(local.iterdir(), key=lambda item: item.name)
        if item.is_file() and item.name.endswith(".py")
    }
    for package, module in (
        ("geo_infer_act", "core/gnn_factored_batch.py"),
        ("geo_infer_act", "core/gnn_factored_contract.py"),
        ("geo_infer_act", "core/resource_allocation.py"),
        ("geo_infer_space", "core/graph_statistics.py"),
        ("geo_infer_place", "core/regional_layers.py"),
        ("geo_infer_place", "core/_regional_download_worker.py"),
    ):
        module_owner = package.removeprefix("geo_infer_").upper()
        key = f"GEO-INFER-{module_owner}/src/{package}/{module}"
        owners[key] = hashlib.sha256(
            files(package).joinpath(module).read_bytes()
        ).hexdigest()
    return ExecutionBinding(
        owners=owners,
        executable=str(Path(sys.executable).resolve()),
        prefix=str(Path(sys.prefix).resolve()),
        base_prefix=str(Path(sys.base_prefix).resolve()),
        python=sys.version,
        packages={
            name: version(name)
            for name in ("numpy", "scipy", "h3", "pydantic", "pyproj", "shapely")
        },
    )


class ObservatoryConfig(StrictModel):
    schema_version: Literal["geo-infer-place/california-design/1"] = (
        "geo-infer-place/california-design/1"
    )
    resolution: int = Field(default=5, ge=4, le=5)
    parent_resolution: int = Field(default=3, ge=2, le=3)
    first_year: int = Field(default=2016, ge=2004, le=2024)
    target_year: int = Field(default=2024, ge=2005, le=2025)
    fire_fraction: float = Field(default=0.01, gt=0, le=1, allow_inf_nan=False)
    drought_fraction: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    drought_category: int = Field(default=1, ge=0, le=4)
    earthquake_magnitude: float = Field(default=3.0, ge=2, le=6, allow_inf_nan=False)
    geometry_offset_degrees: float = Field(
        default=0.001, ge=0, le=0.01, allow_inf_nan=False
    )
    geometry_precision: int = Field(default=6, ge=4, le=8)
    projection_segment_degrees: float = Field(
        default=0.02, ge=0.005, le=0.1, allow_inf_nan=False
    )
    rounds: int = Field(default=8, ge=1, le=24)
    budget: int = Field(default=96, ge=1, le=512)
    hazard_budget_floor: int = Field(default=12, ge=0, le=170)
    seeds: tuple[int, ...] = (0, 7)
    coupling: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    prior_strength: float = Field(default=2.0, gt=0, le=50, allow_inf_nan=False)
    sensor_cost: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)
    coarse_accuracy: float = Field(default=0.8, gt=0.5, lt=1, allow_inf_nan=False)
    detailed_accuracy: float = Field(default=0.97, gt=0.5, lt=1, allow_inf_nan=False)
    coarse_coverage: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)
    detailed_coverage: float = Field(default=0.95, gt=0, lt=1, allow_inf_nan=False)
    repair_geometry: bool = True
    max_cells: int = Field(default=6000, ge=1, le=10000)

    @model_validator(mode="after")
    def bounded_design(self) -> Self:
        if not 2 <= self.target_year - self.first_year <= 10:
            raise CaliforniaError("require 2..10 strictly earlier training years")
        if not 1 <= len(self.seeds) <= 4 or len(set(self.seeds)) != len(self.seeds):
            raise CaliforniaError("require 1..4 distinct seeds")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise CaliforniaError("seeds must be uint32 integers")
        if self.detailed_accuracy <= self.coarse_accuracy:
            raise CaliforniaError(
                "detailed sensor must be more accurate than coarse sensor"
            )
        if self.hazard_budget_floor * len(Hazard) > self.budget:
            raise CaliforniaError("hazard expenditure floors exceed the shared budget")
        if self.detailed_coverage <= self.coarse_coverage:
            raise CaliforniaError(
                "detailed sensor must have greater reporting coverage"
            )
        if len(self.seeds) * self.rounds * self.max_cells > 192000:
            raise CaliforniaError("experiment exceeds 192000 cell-round budget")
        return self


class CapturedFile(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9_\-.]+$")
    url: str
    sha256: Digest
    bytes: int = Field(gt=0, le=RAW_BYTES)


class SourceReceipt(StrictModel):
    schema_version: Literal["geo-infer-place/california-sources/1"]
    config: ObservatoryConfig
    files: tuple[CapturedFile, ...]
    county_ids: tuple[int, ...]
    fire_ids: tuple[int, ...]
    complete: bool

    @model_validator(mode="after")
    def closed_capture(self) -> Self:
        names = [item.name for item in self.files]
        if not self.complete or len(set(names)) != len(names) or not names:
            raise CaliforniaError("incomplete or duplicate source capture")
        if len(self.county_ids) != 58 or len(set(self.county_ids)) != 58:
            raise CaliforniaError("source must contain the 58 distinct county records")
        if not self.fire_ids or len(set(self.fire_ids)) != len(self.fire_ids):
            raise CaliforniaError("missing or duplicate selected fire IDs")
        return self


def last_tuesday(year: int) -> str:
    day = date(year, 12, 31)
    return (day - timedelta(days=(day.weekday() - 1) % 7)).strftime("%Y%m%d")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise CaliforniaError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise CaliforniaError("nonfinite JSON number")
    return number


def _invalid(value: str) -> None:
    raise CaliforniaError(f"invalid JSON constant {value}")


def strict_bytes(path: Path, *, limit: int = RAW_BYTES) -> bytes:
    if path.is_symlink():
        raise CaliforniaError("input is a symlink")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise CaliforniaError("input exceeds byte budget")
    json.loads(
        raw,
        object_pairs_hook=_unique_fields,
        parse_float=_finite,
        parse_constant=_invalid,
    )
    return raw


def write_record(path: Path, record: BaseModel) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(record.model_dump_json(indent=2) + "\n")


def load_config(path: Path) -> ObservatoryConfig:
    return ObservatoryConfig.model_validate_json(strict_bytes(path, limit=65536))
