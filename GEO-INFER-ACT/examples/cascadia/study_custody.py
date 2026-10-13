"""Freeze the experiment and ACT/SPACE source owners before a seed grid runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceCustody(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    geo_infer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    workspace_dirty: bool
    source_owner_sha256: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = (
        Field(min_length=1)
    )

    @field_validator("source_owner_sha256")
    @classmethod
    def safe_owner_paths(cls, owners: dict[str, str]) -> dict[str, str]:
        for name in owners:
            path = Path(name)
            supported = (
                path.suffix in {".py", ".toml", ".lock"}
                or path.name == ".python-version"
            )
            if path.is_absolute() or ".." in path.parts or not supported:
                message = f"unsafe source owner path: {name}"
                raise ValueError(message)
        return owners


def freeze_sources(root: Path, output: Path) -> SourceCustody:
    """Retain exact owner bytes, including uncommitted edits, without promoting HEAD."""
    owners = set((root / "GEO-INFER-ACT/examples/cascadia").glob("*.py"))
    owners.update(
        path
        for path in (
            root / "pyproject.toml",
            root / "uv.lock",
            root / ".python-version",
        )
        if path.is_file()
    )
    for module in ("GEO-INFER-ACT", "GEO-INFER-SPACE"):
        owners.update((root / module / "src").rglob("*.py"))
        manifest = root / module / "pyproject.toml"
        if manifest.is_file():
            owners.add(manifest)
    digests = {}
    for path in sorted(owners):
        name = str(path.relative_to(root))
        target = output / "source_owners" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        target.write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=60
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "status", "--porcelain"],
        cwd=root,
        text=True,
        timeout=60,
    ).strip()
    return SourceCustody(
        geo_infer_revision=revision,
        workspace_dirty=bool(dirty),
        source_owner_sha256=digests,
    )


def source_problems(root: Path, custody: SourceCustody) -> tuple[str, ...]:
    """Detect source drift during execution or against a named live checkout."""
    problems = []
    for name, digest in custody.source_owner_sha256.items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            problems.append(f"source owner differs: {name}")
    return tuple(problems)
