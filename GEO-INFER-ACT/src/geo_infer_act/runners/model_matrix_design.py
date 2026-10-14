"""Portable, bounded experiment matrices over named categorical GNN models."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_infer_act.core.gnn_contract import _unique_object
from geo_infer_act.runners.model_pair import PairDesign

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
DESIGN_BYTES: Final = 65536


class MatrixError(ValueError):
    """An experiment matrix failed a named admission or integrity boundary."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"invalid model matrix: {field}")


class MatrixModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    name: Name
    artifact: str

    @field_validator("artifact")
    @classmethod
    def portable_path(cls, value: str) -> str:
        if (
            PurePosixPath(value).is_absolute()
            or "\\" in value
            or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or not value.endswith(".json")
        ):
            raise MatrixError("artifact path must be a portable relative JSON path")
        return value


class MatrixCase(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    name: Name
    agent: Name
    process: Name


class MatrixDesign(BaseModel):
    """Ordered cases share one seed, horizon, start-state and agent-seed contract."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal["geo-infer-act/model-matrix-design/1"]
    models: tuple[MatrixModel, ...] = Field(min_length=1, max_length=8)
    cases: tuple[MatrixCase, ...] = Field(min_length=1, max_length=64)
    run: PairDesign

    @model_validator(mode="after")
    def closed_selection(self) -> MatrixDesign:
        names = tuple(model.name for model in self.models)
        selected = {name for case in self.cases for name in (case.agent, case.process)}
        pairs = tuple((case.agent, case.process) for case in self.cases)
        if len(set(names)) != len(names) or selected != set(names):
            raise MatrixError("model names must be unique, declared and all selected")
        if len({case.name for case in self.cases}) != len(self.cases):
            raise MatrixError("case names must be unique")
        if len(set(pairs)) != len(pairs):
            raise MatrixError("duplicate agent/process pair")
        if (
            len(self.run.seeds) > 64
            or len(self.cases) * len(self.run.seeds) * self.run.steps > 8192
        ):
            raise MatrixError("matrix exceeds 64 seeds or the total 8192-step budget")
        return self


def load_matrix_design(path: Path) -> MatrixDesign:
    """Parse duplicate-free, strict JSON before creating an execution directory."""
    with path.open("rb") as stream:
        raw = stream.read(DESIGN_BYTES + 1)
    if len(raw) > DESIGN_BYTES:
        raise MatrixError("design exceeds 64 KiB")
    json.loads(raw, object_pairs_hook=_unique_object)
    return MatrixDesign.model_validate_json(raw)
