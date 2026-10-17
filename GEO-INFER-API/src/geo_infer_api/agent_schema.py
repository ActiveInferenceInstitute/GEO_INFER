"""Transport-independent request validation and machine-readable JSON schemas."""

from typing import Annotated, Any, Literal
from functools import lru_cache
import json

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Modules(_Request):
    op: Literal[
        "modules", "handles", "schema", "skills", "guide", "diagnostics", "close"
    ]


class _Skill(_Request):
    op: Literal["skill"]
    module: str


class _Catalog(_Request):
    op: Literal["catalog"]
    module: str
    query: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)


class _Selection(_Request):
    target: str | None = None
    handle: str | None = None
    member: str | None = None

    @model_validator(mode="after")
    def selection(self) -> "_Selection":
        if bool(self.target) == bool(self.handle):
            raise ValueError("Supply exactly one of target or handle")
        if self.member is not None and not self.handle:
            raise ValueError("member requires a handle")
        return self


class _Describe(_Selection):
    op: Literal["describe"]


class _Get(_Selection):
    op: Literal["get"]
    save_as: str | None = Field(default=None, min_length=1, max_length=128)


class _Call(_Selection):
    op: Literal["call"]
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    save_as: str | None = Field(default=None, min_length=1, max_length=128)


class _Release(_Request):
    op: Literal["release"]
    handle: str


class _Enter(_Request):
    op: Literal["enter"]
    handle: str
    save_as: str | None = Field(default=None, min_length=1, max_length=128)


class _Exit(_Request):
    op: Literal["exit"]
    handle: str


class _Batch(_Request):
    op: Literal["batch"]
    steps: list[dict[str, Any]] = Field(min_length=1, max_length=100)


REQUEST = TypeAdapter(
    Annotated[
        _Modules
        | _Skill
        | _Catalog
        | _Describe
        | _Get
        | _Call
        | _Release
        | _Enter
        | _Exit
        | _Batch,
        Field(discriminator="op"),
    ]
)


def validate_request(request: Any) -> dict[str, Any]:
    """Validate transport keys without coercing scientific argument values."""
    model = REQUEST.validate_python(request)
    validated = {
        name: getattr(model, name)
        for name in type(model).model_fields
        if getattr(model, name) is not None
    }
    if validated["op"] == "batch":
        steps = []
        for step in validated["steps"]:
            if step.get("op") == "batch":
                raise ValueError("Nested batches are not supported")
            steps.append(validate_request(step))
        validated["steps"] = steps
    return validated


def request_schema() -> dict[str, Any]:
    """Return the exact JSON Schema used by the script and CLI request surface."""
    # Decode a canonical cached encoding so callers cannot mutate the cache.
    return json.loads(_schema_json())


@lru_cache(maxsize=1)
def _schema_json() -> str:
    return json.dumps(REQUEST.json_schema(), allow_nan=False)
