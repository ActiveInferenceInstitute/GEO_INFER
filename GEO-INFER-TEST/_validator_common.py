"""Shared plumbing for the GEO-INFER-TEST validators.

``validate_packaging.py``, ``validate_repo_contracts.py`` and
``rewrite_readme_agents.py`` previously re-implemented module discovery,
pyproject parsing, requirement-name normalization and the error/warning
report independently. This module is the single definition of that plumbing.

Repository-root discipline: helpers take ``repo_root`` (or operate on a
passed module directory) as an explicit parameter and never capture a
repository root at import time. Each validator keeps its own module-global
``REPO_ROOT`` and passes it in, so test suites can monkeypatch one
validator's root without silently retargeting the shared helpers.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MODULE_PREFIX = "GEO-INFER-"


@dataclass
class ContractReport:
    """Accumulated validator findings.

    ``errors`` fail the run, ``warnings`` fail under ``--strict``-style
    promotion, and ``diagnostics`` are informational only.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def diagnostic(self, message: str) -> None:
        self.diagnostics.append(message)


def discover_module_dirs(repo_root: Path, prefix: str = MODULE_PREFIX) -> list[Path]:
    """Return tracked module directories (``GEO-INFER-*``) in stable order."""
    return sorted(
        path
        for path in repo_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )


def read_toml(path: Path) -> dict:
    """Parse a TOML file, raising ``FileNotFoundError``/``TOMLDecodeError``."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def read_pyproject(module_dir: Path) -> dict:
    """Parse a module's ``pyproject.toml`` ({} when absent or invalid).

    Lenient by design: callers that must surface missing/invalid metadata
    report it themselves (see ``validate_repo_contracts.parse_pyproject``).
    """
    pyproject = module_dir / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    try:
        return read_toml(pyproject)
    except tomllib.TOMLDecodeError:
        return {}


def distribution_name(pyproject: dict) -> str | None:
    """Return the declared PyPI distribution name, or None when absent."""
    name = pyproject.get("project", {}).get("name")
    return name if isinstance(name, str) and name else None


def package_name_from_distribution(distribution: str) -> str:
    """Map a distribution name to its importable package name (``-`` -> ``_``)."""
    return distribution.replace("-", "_")


def expected_package_name(pyproject: dict) -> str | None:
    """Return the package directory name implied by ``[project].name``."""
    distribution = distribution_name(pyproject)
    if distribution is None:
        return None
    return package_name_from_distribution(distribution)


# Requirement-name normalization: one definition shared by the validators so
# pyproject dependency strings compare by normalized distribution name only.
_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_dependency_name(raw: str) -> str:
    """Normalize a distribution name (lowercase, underscores -> dashes)."""
    match = _REQUIREMENT_NAME_RE.match(raw.strip())
    if not match:
        return ""
    return match.group(1).lower().replace("_", "-")


def pyproject_dependency_names(pyproject: dict) -> set:
    """Return normalized runtime dependency names from [project.dependencies]."""
    deps = pyproject.get("project", {}).get("dependencies") or []
    return {name for name in (normalize_dependency_name(str(d)) for d in deps) if name}


def pyproject_optional_names(pyproject: dict) -> set:
    """Return normalized names across all [project.optional-dependencies] groups."""
    groups = pyproject.get("project", {}).get("optional-dependencies") or {}
    names: set = set()
    for group in groups.values():
        for dep in group:
            name = normalize_dependency_name(str(dep))
            if name:
                names.add(name)
    return names


def pyproject_group_names(pyproject: dict, group: str | None = None) -> set:
    """Return normalized names from PEP 735 ``[dependency-groups]``.

    With ``group`` the named group is resolved, following
    ``{include-group = "..."}`` entries transitively; without it every group
    is resolved and the union returned. An include naming an absent group, or
    an include cycle, contributes nothing (uv rejects both at lock time).
    """
    groups = pyproject.get("dependency-groups") or {}
    if not isinstance(groups, dict):
        return set()

    def resolve(name: str, seen: frozenset[str]) -> set:
        entries = groups.get(name)
        if name in seen or not isinstance(entries, list):
            return set()
        names: set = set()
        for entry in entries:
            if isinstance(entry, str):
                normalized = normalize_dependency_name(entry)
                if normalized:
                    names.add(normalized)
            elif isinstance(entry, dict) and isinstance(
                included := entry.get("include-group"), str
            ):
                names |= resolve(included, seen | {name})
        return names

    selected = [group] if group is not None else list(groups)
    resolved: set = set()
    for name in selected:
        resolved |= resolve(name, frozenset())
    return resolved


# Standard-library modules that must never be listed as PyPI dependencies.
STDLIB_REQUIREMENT_NAMES = {
    "argparse",
    "asyncio",
    "ast",
    "bz2",
    "colorsys",
    "concurrent",
    "configparser",
    "contextlib",
    "csv",
    "email",
    "functools",
    "gc",
    "gzip",
    "hashlib",
    "heapq",
    "inspect",
    "io",
    "itertools",
    "json",
    "math",
    "pickle",
    "queue",
    "random",
    "re",
    "secrets",
    "shutil",
    "sqlite3",
    "statistics",
    "subprocess",
    "tempfile",
    "threading",
    "unittest",
    "urllib",
    "uuid",
    "weakref",
}


def internal_requirement_names(repo_root: Path) -> frozenset[str]:
    """Internal distribution names, derived from the workspace tree.

    A dependency pinned ``>=0.0.0`` names an internal workspace member
    masquerading as a PyPI dependency; such entries must be resolved through
    ``[tool.uv.sources]`` instead. Each member's own ``geo-infer-*``
    distribution name comes from its ``pyproject.toml``, so the set never
    needs a hand-maintained allowlist.
    """
    names: set[str] = set()
    for module_dir in discover_module_dirs(repo_root):
        distribution = distribution_name(read_pyproject(module_dir))
        if distribution:
            names.add(distribution)
    return frozenset(names)
