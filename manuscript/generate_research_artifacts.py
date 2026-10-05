#!/usr/bin/env python3
"""Generate repository-derived manuscript variables, figures, and evidence.

The tracked manuscript is intentionally authored with ``{{UPPERCASE_TOKENS}}``.
This module is the single producer for those values.  It scans the checkout,
creates figures from the scan, registers their captions and provenance, and
writes resolved manuscript copies to ``output/manuscript``.  Generated output
is disposable; the source markdown remains the reviewable manuscript surface.

The optional ``--verify`` and ``--full-validation`` modes record the commands
that were actually executed.  A failed or unrun command is never represented
as a passing research result.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
EXCLUDED_MANUSCRIPT_DOCS = frozenset({"README.md", "AGENTS.md", "SYNTAX.md"})
FIGURE_SCHEMA = "geo-infer-manuscript-figures/v1"
RESEARCH_SCHEMA = "geo-infer-manuscript-evidence/v1"
FOCUS_MODULES = ("GEO-INFER-ACT", "GEO-INFER-BAYES", "GEO-INFER-RISK")

# Editorial grouping of the module set, mirroring the "Module Themes" table in
# README.md. This is a classification, not a measurement: the counts beside
# each module are read from the checkout, but which theme a module belongs to
# is a judgement and has to be declared somewhere. ``_theme_assignments``, the
# single validation authority behind both module tables (``_module_table`` and
# ``_module_purpose_table``), refuses to render unless every measured module
# appears in exactly one theme, so adding a module without theming it fails
# the build instead of dropping it silently.
MODULE_THEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Spatial and place-based",
        (
            "GEO-INFER-SPACE",
            "GEO-INFER-PLACE",
            "GEO-INFER-TIME",
            "GEO-INFER-MARINE",
            "GEO-INFER-WATER",
            "GEO-INFER-FOREST",
            "GEO-INFER-CLIMATE",
            "GEO-INFER-ENERGY",
            "GEO-INFER-TRANSPORT",
            "GEO-INFER-EMERGENCY",
        ),
    ),
    (
        "Bayesian and active inference",
        (
            "GEO-INFER-BAYES",
            "GEO-INFER-SIM",
            "GEO-INFER-SPM",
            "GEO-INFER-COG",
            "GEO-INFER-ACT",
            "GEO-INFER-MATH",
        ),
    ),
    (
        "Agents and AI orchestration",
        (
            "GEO-INFER-AGENT",
            "GEO-INFER-AG",
            "GEO-INFER-AI",
            "GEO-INFER-ANT",
            "GEO-INFER-OPS",
            "GEO-INFER-COMMS",
        ),
    ),
    (
        "Governance, risk and domain",
        (
            "GEO-INFER-INSURANCE",
            "GEO-INFER-RISK",
            "GEO-INFER-METAGOV",
            "GEO-INFER-NORMS",
            "GEO-INFER-ECON",
            "GEO-INFER-PEP",
            "GEO-INFER-REQ",
            "GEO-INFER-SEC",
            "GEO-INFER-CIV",
            "GEO-INFER-HEALTH",
            "GEO-INFER-ORG",
        ),
    ),
    (
        "Data, API and applications",
        (
            "GEO-INFER-API",
            "GEO-INFER-APP",
            "GEO-INFER-DATA",
            "GEO-INFER-IOT",
            "GEO-INFER-ART",
            "GEO-INFER-EDU",
        ),
    ),
    (
        "Infrastructure and validation",
        (
            "GEO-INFER-INTRA",
            "GEO-INFER-TEST",
            "GEO-INFER-LOG",
            "GEO-INFER-GIT",
            "GEO-INFER-EXAMPLES",
            "GEO-INFER-BIO",
        ),
    ),
)

# Printable geometry of the template's LaTeX text block, in inches.  Pinned
# to the margin=1.5cm geometry of ``scripts/render_manuscript_pdf.py``
# (textwidth 528.93pt / 72.27 = 7.32in, textheight 709.60pt / 72.27 =
# 9.82in); the authoritative re-check is the same values printed in
# output/pdf/_combined_manuscript.log at the next render.  Every figure is
# typeset inside this box, so a figure drawn larger than it is scaled DOWN
# and its type shrinks with it: a 13in canvas lettered at 8.5pt printed at
# 3.1pt, roughly half the ~6pt floor for legible print.  Drawing at the
# printed size keeps the scale near 1.0 and the type at its authored point
# size.
TEXT_BLOCK_WIDTH_IN = 7.32
TEXT_BLOCK_HEIGHT_IN = 9.82
# Must stay in lock-step with ``rendering.figure_height_fraction`` in
# manuscript/config.yaml, which is what the renderer writes into the
# ``height=<fraction>\textheight`` bound on every \includegraphics.
FIGURE_HEIGHT_FRACTION = 0.9
MAX_FIGURE_HEIGHT_IN = TEXT_BLOCK_HEIGHT_IN * FIGURE_HEIGHT_FRACTION
# PNG raster density. Figures are drawn at the repo render's printed size
# (528.93pt text block), so 340 DPI is that lane's printed raster density at
# print scale 1.0; the template lane (margin=2cm, 500.484pt text block)
# downscales every figure by 500.484/528.93 = 0.946x, which lands its
# effective density at ~360 DPI — both lanes clear the ~264 DPI floor for
# close inspection.
FIGURE_DPI = 340
# Inches of vertical space per module row in the inventory figure.  A row now
# carries both bars for one module, so at 8pt type this is about 12pt of
# leading per label.
INVENTORY_ROW_HEIGHT_IN = 0.17
# Inches the inventory figure spends on everything that is not a module row:
# the title, the two axis labels, the legend, and the surrounding padding.
INVENTORY_FIGURE_CHROME_IN = 1.0
# A float is granted a page of its own once it fills this fraction of the text
# block, and such a page prints a caption and a folio and nothing else.  The
# value is the ``\floatpagefraction`` set in manuscript/preamble.md; the
# inventory figure is laid out to stay under it once its caption is added.
FLOAT_PAGE_FRACTION = 0.85
# Inches a three-line figure caption occupies under the float at 10pt/12pt.
FIGURE_CAPTION_HEIGHT_IN = 0.55

# Shared figure palette. The source-versus-test encoding reuses the same two
# hues in every figure, test categories and repository surfaces get one hue
# each, and the graphical abstract reuses the same accents, so a colour
# learned in one figure carries to every other one. Status colours are
# reserved for recorded verification outcomes.
SOURCE_COLOR = "#2f6f9f"
TEST_COLOR = "#d17a2f"
CATEGORY_COLOR = "#5b8e7d"
SURFACE_COLOR = "#6f5b9e"
SPINE_COLOR = "#1b3a5b"
STATUS_COLORS = {"passed": "#2e7d32", "failed": "#b3261e", "not-run": "#8a8a8a"}


@dataclass(frozen=True)
class ModuleMetrics:
    """Measured implementation and test surfaces for one module.

    ``purpose`` is the module's one-line self-description, taken from
    README.md (else AGENTS.md) by :func:`_module_purpose`; ``unavailable``
    marks a module that ships neither file with an opening prose paragraph.
    """

    name: str
    package: str
    source_files: int
    source_lines: int
    test_files: int
    tests_by_category: dict[str, int]
    purpose: str


@dataclass(frozen=True)
class FigureSpec:
    """Publication figure metadata generated with the corresponding image.

    ``sha256`` is the digest of the PNG bytes actually written, so the registry
    carries per-figure content provenance rather than only the repository-wide
    build hash.  It is empty until the image exists and is filled in by
    :func:`generate_figures`; :func:`write_figure_registry` refuses a spec that
    still carries an empty digest.
    """

    label: str
    filename: str
    caption: str
    generated_by: str
    alt_text: str
    sha256: str = ""


@dataclass(frozen=True)
class RepositoryInventory:
    """Deterministic repository measurements used by the manuscript."""

    project_version: str
    project_license: str
    commit: str
    branch: str
    commit_date: str
    manuscript_source_date: str
    dirty_file_count: int
    source_hash: str
    modules: tuple[ModuleMetrics, ...]
    module_directory_count: int
    test_files_by_category: dict[str, int]
    h3_test_files: int
    documentation_pages: int
    validator_files: int
    test_tooling_files: int
    source_files: int
    source_lines: int
    test_files: int
    python_version: str

    @property
    def module_count(self) -> int:
        return len(self.modules)

    @property
    def modules_with_tests(self) -> int:
        return sum(module.test_files > 0 for module in self.modules)

    @property
    def focused_modules(self) -> tuple[ModuleMetrics, ...]:
        by_name = {module.name: module for module in self.modules}
        return tuple(by_name[name] for name in FOCUS_MODULES if name in by_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESEARCH_SCHEMA,
            "project_version": self.project_version,
            "project_license": self.project_license,
            "commit": self.commit,
            "branch": self.branch,
            "commit_date": self.commit_date,
            "manuscript_source_date": self.manuscript_source_date,
            "dirty_file_count": self.dirty_file_count,
            "source_hash": self.source_hash,
            "modules": [asdict(module) for module in self.modules],
            "module_directory_count": self.module_directory_count,
            "test_files_by_category": dict(self.test_files_by_category),
            "h3_test_files": self.h3_test_files,
            "documentation_pages": self.documentation_pages,
            "validator_files": self.validator_files,
            "test_tooling_files": self.test_tooling_files,
            "source_files": self.source_files,
            "source_lines": self.source_lines,
            "test_files": self.test_files,
            "python_version": self.python_version,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one explicitly executed research verification command."""

    name: str
    command: str
    status: str
    return_code: int | None
    duration_seconds: float | None
    output_tail: str
    receipt: str | None = None


@dataclass(frozen=True)
class VerificationRecord:
    """The executed-command evidence a build publishes, and the tree it names.

    The results alone are not publishable: a record is only interpretable
    against the checkout whose commands produced it.  Carrying the commit and
    source fingerprint alongside the results is what lets a build that ran no
    command republish a measured record without claiming to have measured this
    tree — and what lets the manuscript say which tree the numbers came from.
    """

    results: tuple[VerificationResult, ...]
    source_commit: str
    source_hash: str
    full_validation_requested: bool
    measured_elsewhere: bool = False

    def __bool__(self) -> bool:
        """A record is truthy when it holds at least one executed command."""
        return bool(self.results)

    @property
    def defined_groups(self) -> tuple[str, ...]:
        """The command groups this record's own tier defines.

        The denominator every published verification count is taken against.
        It is read from the record rather than from the build's request
        because the two diverge routinely: a build that carries a
        full-validation record forward asked for the default tier, and
        counting eleven measured outcomes against a seven-group definition is
        how the manuscript came to publish "9 of 7".  The tier travels with
        the results, so there is no argument by which a caller can pair them
        with a definition they were not measured against.
        """
        return defined_command_groups(full_validation=self.full_validation_requested)

    @classmethod
    def unmeasured(cls, *, full_validation: bool = False) -> VerificationRecord:
        """A record holding no executed command and naming no tree.

        The stamps are empty rather than borrowed from a build: an unmeasured
        record has no tree to name, and ``build_variables`` falls back to the
        inventory's own commit and fingerprint for the fields that must still
        print something.
        """
        return cls(
            results=(),
            source_commit="",
            source_hash="",
            full_validation_requested=full_validation,
        )


VERIFICATION_TIMEOUT_SECONDS = 600


FULL_VALIDATION_TIMEOUT_SECONDS = 3600


VERIFICATION_COMMANDS: tuple[tuple[str, str], ...] = (
    (
        "compile",
        "python -m compileall -q GEO-INFER-*/src GEO-INFER-*/examples manuscript",
    ),
    (
        "repository-contracts",
        "uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language",
    ),
    (
        "documentation",
        "uv run python GEO-INFER-TEST/validate_documentation.py --strict",
    ),
    (
        "skills",
        "uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs",
    ),
    (
        "test-contracts",
        "uv run python GEO-INFER-TEST/validate_test_contracts.py --strict",
    ),
    (
        "model-contracts",
        "uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42",
    ),
    (
        "reproducibility",
        "uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible",
    ),
)

FULL_VALIDATION_COMMANDS: tuple[tuple[str, str], ...] = (
    (
        "unit-tests",
        "uv run python GEO-INFER-TEST/run_unified_tests.py "
        "--category unit --timeout 600",
    ),
    (
        "integration-tests",
        "uv run python GEO-INFER-TEST/run_unified_tests.py "
        "--category integration --timeout 600",
    ),
    (
        "performance-tests",
        "uv run python GEO-INFER-TEST/run_unified_tests.py "
        "--category performance --timeout 600",
    ),
    (
        "h3-contracts",
        "uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration",
    ),
)


FULL_VALIDATION_GROUPS = frozenset(name for name, _ in FULL_VALIDATION_COMMANDS)
ALL_COMMAND_GROUPS = frozenset(
    name for name, _ in (*VERIFICATION_COMMANDS, *FULL_VALIDATION_COMMANDS)
)


def _iter_files(directory: Path, suffix: str | None = None) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    paths = directory.rglob("*")
    return (
        path
        for path in paths
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and (suffix is None or path.suffix == suffix)
    )


def _python_files(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(_iter_files(directory, ".py")))


def _nonempty_lines(path: Path) -> int:
    return sum(
        bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
    )


def _run_git(root: Path, *args: str, default: str = "unavailable") -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return default
    return result.stdout.strip() or default


def _dirty_file_count(root: Path) -> int:
    """Count working-tree entries that differ from the tracked commit.

    ``git status --porcelain`` prints one line per added, modified, deleted,
    renamed, unmerged or untracked path.  A clean checkout prints nothing, so
    the honest answer for a clean tree is ``0``.

    Returns:
        The number of differing entries, or ``-1`` when git cannot answer.
        ``-1`` is deliberately not ``0``: an unavailable answer must never be
        published as a clean tree.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return -1
    return sum(bool(line.strip()) for line in result.stdout.splitlines())


def _dirty_marker(count: int) -> str:
    """Return the commit-stamp suffix that describes ``count``."""
    if count > 0:
        return "-dirty"
    if count < 0:
        return "-unverified"
    return ""


def _project_metadata(root: Path) -> dict[str, str]:
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {"version": "unavailable", "license": "unavailable"}
    license_value = project.get("license", {})
    if isinstance(license_value, dict):
        license_text = license_value.get("text") or license_value.get("file")
    else:
        license_text = license_value
    return {
        "version": str(project.get("version", "unavailable")),
        "license": str(license_text or "unavailable"),
    }


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths: set[Path] = set(_iter_files(root / "manuscript"))
    for module in _module_paths(root):
        paths.update(_python_files(module / "src"))
        paths.update(
            path
            for path in _python_files(module / "tests")
            if path.name.startswith("test_")
        )
    paths.update(
        path for path in (root / "GEO-INFER-TEST").glob("*.py") if path.is_file()
    )
    paths.update(
        path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file()
    )
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _module_package(module_path: Path) -> str:
    """Return the importable package directory shipped under ``src/``."""
    packages = sorted(
        path.name
        for path in (module_path / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    return packages[0] if packages else "unavailable"


def _first_paragraph(path: Path) -> str | None:
    """Return a file's first non-heading prose paragraph, or ``None``.

    The rule is deliberately mechanical, so the value is a deterministic
    function of the file's bytes: skip blank and ATX heading lines, then join
    the run of consecutive non-blank, non-heading lines that follows.  In
    every module README that run is the one-sentence description under the
    title; in AGENTS.md it is the opening summary.
    """
    if not path.is_file():
        return None
    paragraph: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph) if paragraph else None


def _module_purpose(module_path: Path) -> str:
    """Return the module's one-line purpose, deterministically.

    README.md's first paragraph is the purpose; AGENTS.md is the fallback;
    ``unavailable`` marks a module that ships neither with opening prose,
    mirroring :func:`_module_package`.  Pipes are escaped so the sentence
    survives the Markdown pipe tables it is rendered into.  Same bytes in,
    same sentence out — no wall-clock input anywhere.
    """
    for name in ("README.md", "AGENTS.md"):
        paragraph = _first_paragraph(module_path / name)
        if paragraph is not None:
            return paragraph.replace("|", "\\|")
    return "unavailable"


def _module_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.glob("GEO-INFER-*")
            if path.is_dir() and (path / "src").is_dir()
        )
    )


def _test_files(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for module in _module_paths(root):
        paths.extend(
            path
            for path in _python_files(module / "tests")
            if path.name.startswith("test_")
        )
    paths.extend(
        path
        for path in _python_files(root / "GEO-INFER-TEST" / "tests")
        if path.name.startswith("test_")
    )
    return tuple(sorted(set(paths)))


TEST_CATEGORIES: tuple[str, ...] = ("unit", "integration", "performance", "other")


def _test_category(path: Path) -> str:
    """Classify a test file on the mutually-exclusive directory axis.

    Every test lands in exactly one of :data:`TEST_CATEGORIES`, so the four
    published category counts always sum to the published total.

    Subject-matter tags such as H3 are deliberately *not* on this axis.  They
    were, and the bucket was unreachable: an H3 test lives under ``tests/unit``
    or ``tests/integration`` like any other, matched an earlier branch, and the
    ``h3`` count published as zero while H3-named files existed.  H3 is now
    counted orthogonally by :func:`_h3_test_files`.
    """
    parts = {part.lower() for part in path.parts}
    for category in ("unit", "integration", "performance"):
        if category in parts:
            return category
    return "other"


def _h3_test_files(paths: Iterable[Path]) -> int:
    """Count H3-named test files, independently of their directory category."""
    return sum("h3" in path.name.lower() for path in paths)


def collect_inventory(
    root: Path, *, dirty_file_count: int | None = None
) -> RepositoryInventory:
    """Measure the current checkout without importing application modules.

    Args:
        root: Repository root to measure.
        dirty_file_count: Pre-measured working-tree dirtiness.  ``generate``
            passes the count taken *before* it writes anything, so the
            generator's own outputs are never mistaken for un-stamped source.
            ``None`` measures it here.
    """
    module_directories = tuple(
        path for path in root.glob("GEO-INFER-*") if path.is_dir()
    )
    # The directory census is a different population from ``_module_paths``:
    # it counts every GEO-INFER-* directory at the root, whether or not it
    # ships a src/ package.  Today the two coincide; the distinction is kept
    # explicit so a source-less module can appear without falsifying the
    # src/-bearing counts.
    modules: list[ModuleMetrics] = []
    all_tests = _test_files(root)
    category_counts: dict[str, int] = {}
    for path in all_tests:
        category = _test_category(path)
        category_counts[category] = category_counts.get(category, 0) + 1

    for module_path in _module_paths(root):
        source = _python_files(module_path / "src")
        tests = tuple(path for path in all_tests if module_path in path.parents)
        by_category: dict[str, int] = {}
        for path in tests:
            category = _test_category(path)
            by_category[category] = by_category.get(category, 0) + 1
        modules.append(
            ModuleMetrics(
                name=module_path.name,
                package=_module_package(module_path),
                source_files=len(source),
                source_lines=sum(_nonempty_lines(path) for path in source),
                test_files=len(tests),
                tests_by_category=dict(sorted(by_category.items())),
                purpose=_module_purpose(module_path),
            )
        )

    source_files = tuple(
        path for module in _module_paths(root) for path in _python_files(module / "src")
    )
    documentation_pages = len(
        tuple(_iter_files(root / "GEO-INFER-INTRA" / "docs", ".md"))
    )
    # A validator is a ``validate_*.py`` entry point, not every top-level file
    # in GEO-INFER-TEST: that directory also holds packaging (setup.py), build
    # (build_package_wheels.py), documentation (rewrite_readme_agents.py) and
    # runner (run_unified_tests.py) utilities, none of which validate anything.
    validator_files = len(
        tuple(
            path
            for path in (root / "GEO-INFER-TEST").glob("validate_*.py")
            if path.is_file()
        )
    )
    test_tooling_files = len(
        tuple(path for path in (root / "GEO-INFER-TEST").glob("*.py") if path.is_file())
    )
    project_metadata = _project_metadata(root)
    dirty_files = (
        _dirty_file_count(root) if dirty_file_count is None else dirty_file_count
    )
    commit = _run_git(root, "rev-parse", "--short", "HEAD")
    return RepositoryInventory(
        project_version=project_metadata["version"],
        project_license=project_metadata["license"],
        commit=commit + _dirty_marker(dirty_files),
        branch=_run_git(root, "branch", "--show-current"),
        commit_date=_run_git(root, "show", "-s", "--format=%cI", default="unavailable"),
        manuscript_source_date=_manuscript_source_date(root),
        dirty_file_count=dirty_files,
        source_hash=_source_hash(root),
        modules=tuple(modules),
        module_directory_count=len(module_directories),
        test_files_by_category=dict(sorted(category_counts.items())),
        h3_test_files=_h3_test_files(all_tests),
        documentation_pages=documentation_pages,
        validator_files=validator_files,
        test_tooling_files=test_tooling_files,
        source_files=len(source_files),
        source_lines=sum(_nonempty_lines(path) for path in source_files),
        test_files=len(all_tests),
        python_version=platform.python_version(),
    )


def _theme_assignments(measured_names: Iterable[str]) -> dict[str, str]:
    """Map every measured module to its declared theme, fail-closed.

    The single validation authority behind both module tables.  Raises the
    same three incoherences ``_module_table`` has always refused: a module
    declared in more than one theme, a measured module with no declared
    theme, and a theme naming a module absent from the checkout.  Either way
    the table would silently misrepresent the module set.
    """
    measured = set(measured_names)
    declared = [name for _theme, names in MODULE_THEMES for name in names]
    duplicates = sorted({name for name in declared if declared.count(name) > 1})
    if duplicates:
        raise ValueError(f"modules declared in more than one theme: {duplicates}")
    unthemed = sorted(measured - set(declared))
    if unthemed:
        raise ValueError(f"measured modules with no declared theme: {unthemed}")
    missing = sorted(set(declared) - measured)
    if missing:
        raise ValueError(f"themed modules absent from the checkout: {missing}")
    return {name: theme for theme, names in MODULE_THEMES for name in names}


def _module_table(inventory: RepositoryInventory) -> str:
    """Render every measured module as a themed Markdown table.

    Raises:
        ValueError: when the declared themes and the measured module set
            disagree, through :func:`_theme_assignments`.
    """
    _theme_assignments(module.name for module in inventory.modules)
    measured = {module.name: module for module in inventory.modules}
    # Pandoc derives each column's relative width from the dash count in the
    # separator row.  Equal dashes gave the Module column less width than
    # ``GEO-INFER-INSURANCE`` needs, and the template's breakable-monospace
    # macro then split module names mid-word ("GEO-INFER-IN / SURANCE") or ran
    # them into the Package column.  The proportions below are sized from the
    # longest value each column actually holds.
    rows = [
        "| Theme | Module | Package | Source files | Test files |",
        "| "
        + " | ".join(("-" * 16, "-" * 21, "-" * 21, "-" * 7 + ":", "-" * 7 + ":"))
        + " |",
    ]
    for theme, names in MODULE_THEMES:
        ordered = sorted(
            (measured[name] for name in names),
            key=lambda item: (-item.source_files, item.name),
        )
        for position, module in enumerate(ordered):
            label = theme if position == 0 else ""
            rows.append(
                f"| {label} | `{module.name}` | `{module.package}` | "
                f"{module.source_files} | {module.test_files} |"
            )
    return "\n".join(rows)


def _module_purpose_table(inventory: RepositoryInventory) -> str:
    """Render every measured module with its theme, purpose, and census.

    One self-contained row per module — theme and purpose repeat on every
    row, so any row can be quoted on its own (the per-module manuscript
    sections slice this table).  The theme comes from the declared
    :data:`MODULE_THEMES` mapping — reusing it keeps one taxonomy in the
    build, and :func:`_theme_assignments` fails the build when the declared
    set and the measured module set disagree.  The purpose is the module's
    README.md (else AGENTS.md) opening paragraph, extracted by the
    deterministic rule in :func:`_first_paragraph`.  LOC is the measured
    non-empty ``src/`` line census and Tests the module's test-file census,
    the same measurements the inventory figure bars.  Rows are alphabetical
    by module name and nothing reads the clock, so two runs on one tree are
    byte-identical.

    Raises:
        ValueError: when the declared themes and the measured module set
            disagree, through :func:`_theme_assignments`.
    """
    theme_by_module = _theme_assignments(module.name for module in inventory.modules)
    # Dash counts set the columns' relative pandoc width, sized from the
    # longest value each column holds: a backticked GEO-INFER-INSURANCE, the
    # longest declared theme label, and the widest share for prose.
    rows = [
        "| Module | Theme | Purpose | LOC | Tests |",
        "| "
        + " | ".join(("-" * 21, "-" * 27, "-" * 44, "-" * 7 + ":", "-" * 7 + ":"))
        + " |",
    ]
    for module in inventory.modules:
        rows.append(
            f"| `{module.name}` | {theme_by_module[module.name]} | "
            f"{module.purpose} | {module.source_lines} | {module.test_files} |"
        )
    return "\n".join(rows)


def _format_count(value: int) -> str:
    return f"{value:,}"


def _spine_names(inventory: RepositoryInventory) -> str:
    """Comma-joined short names of the research-focus modules."""
    return ", ".join(
        module.name.removeprefix("GEO-INFER-") for module in inventory.focused_modules
    )


def _caption_module_inventory(inventory: RepositoryInventory) -> str:
    return (
        f"Repository-derived inventory of the {inventory.module_count} src/-bearing "
        f"GEO-INFER modules at commit {inventory.commit}. Each row is one module: the "
        "blue bar counts its tracked Python source files and the orange bar its tracked "
        "Python test files, both read against the shared Files axis at the foot of each "
        "panel. Modules are ordered by source-file count, largest first, and the rows "
        "continue from the left panel into the right one, so bar length is comparable "
        "across both panels; row labels drawn in the dark accent colour mark the "
        f"{_spine_names(inventory)} research-spine modules. Counts are measured from "
        f"the tracked checkout at source hash {inventory.source_hash}, not entered by "
        "hand. They describe implementation and evidence surfaces, not correctness: "
        "what the spread shows is where the repository concentrates its code, and that "
        "several modules carry more test files than source files, so the verification "
        "surface does not scale with implementation size."
    )


def _caption_research_spine(inventory: RepositoryInventory) -> str:
    focus = _spine_names(inventory)
    return (
        f"Implementation and verification surfaces for the {focus} research spine at "
        f"commit {inventory.commit}. Each group is one module: the blue bar counts its "
        "tracked Python source files and the orange bar its tracked Python test files, "
        "both read against the shared Files axis, and the number above each bar states "
        "the count directly, so no axis tracing is needed. Measured from the tracked "
        f"checkout at source hash {inventory.source_hash} by the same inventory pass "
        "that produces every other quantity in this manuscript. The spine is "
        "deliberately small, and each module's test surface is on the order of its "
        "source surface, which is what makes the free-energy, sampler, and risk-bound "
        "obligations quoted in the Methods section reviewable in a single sitting."
    )


def _caption_validation_surface(inventory: RepositoryInventory) -> str:
    categories = (
        ", ".join(inventory.test_files_by_category) or "the discovered test suite"
    )
    return (
        f"Repository validation surface at commit {inventory.commit}. The left panel "
        f"counts the {inventory.test_files:,} tracked Python test files by the directory "
        f"category they live in ({categories}); those categories are mutually exclusive, "
        "and the build refuses to publish if they do not sum to the total. The right "
        "panel reports three measured evidence surfaces: the "
        f"{inventory.module_count} src/-bearing modules, the "
        f"{inventory.documentation_pages} cross-module documentation pages under "
        f"GEO-INFER-INTRA/docs/, and the {inventory.validator_files} validator entry "
        "points in GEO-INFER-TEST. Numbers above each bar state the counts directly. "
        f"Measured from the tracked checkout at source hash {inventory.source_hash}. "
        "These bars describe the evidence surfaces that exist, not claims that any of "
        "them passed; executed-command outcomes are published separately in the "
        "per-group verification record."
    )


def _caption_graphical_abstract(
    inventory: RepositoryInventory, verification: VerificationRecord | None
) -> str:
    """The cover figure's caption, with the record's summary when one is held."""
    if verification is not None and verification.results:
        summary = _verification_summary(
            verification.results,
            full_validation=verification.full_validation_requested,
        )[0]
        tier = (
            "full-validation" if verification.full_validation_requested else "default"
        )
        outcome = f"the recorded verification summary is {summary} at the {tier} tier"
    else:
        outcome = "this view records no verification summary"
    return (
        f"Graphical abstract of the GEO-INFER composition contract at commit "
        f"{inventory.commit}. Reading panel one to panel four: the "
        f"{inventory.module_count} src/-bearing modules are grouped into "
        f"{len(MODULE_THEMES)} declared themes; those themes compose into the spatial "
        "substrate, the probabilistic machinery, and the domain engines named in the "
        "second panel; the third panel shows the evidence loop, which measures the "
        "checkout, draws the publication figures, and runs the seven default-tier "
        "verification command groups, each drawn with its recorded outcome; the fourth "
        f"panel states what the loop establishes. Counts are measured from the tracked "
        f"checkout at source hash {inventory.source_hash}, and {outcome}. The figure is "
        "a map of the evidence pipeline, not a proof of model quality: every arrow "
        "marks a declared data path, and each claim in the last panel is only as good "
        "as the recorded command outcome behind it."
    )


def _alt_module_inventory(inventory: RepositoryInventory) -> str:
    return (
        f"Horizontal grouped bar chart with one row per module for "
        f"{inventory.module_count} modules, ordered by source-file count with the "
        "largest at the top of the left panel and continuing into the right panel. "
        "Each row carries a blue source-file bar and an orange test-file bar, both "
        "panels share one count axis, and the research-spine module labels are drawn "
        "in the accent colour."
    )


def _alt_research_spine(inventory: RepositoryInventory) -> str:
    focus = _spine_names(inventory)
    return (
        f"Grouped bar chart with one group per module for {focus}. Each group pairs a "
        "blue source-file bar with an orange test-file bar on a shared count axis, and "
        "each bar carries its count printed above it, so implementation and "
        "verification size can be compared per module without tracing to the axis."
    )


def _alt_validation_surface(inventory: RepositoryInventory) -> str:
    categories = (
        ", ".join(inventory.test_files_by_category) or "the discovered test suite"
    )
    return (
        "Two-panel bar chart, every bar carrying its count printed above it. The left "
        f"panel counts test files per discovered category ({categories}); the right "
        "panel reports the measured module, documentation-page, and validator-file "
        "surface counts."
    )


def _alt_graphical_abstract(
    inventory: RepositoryInventory, verification: VerificationRecord | None
) -> str:
    return (
        "Four-panel flow, read left to right and top to bottom: modules grouped into "
        "declared themes with their module short names; the geospatial and "
        "probabilistic models those themes compose into; the evidence and "
        "verification loop with the recorded outcome of each command group; and the "
        "verified claims the loop establishes. A footer records the commit, source "
        "hash, and version the figure was generated from."
    )


def _import_matplotlib() -> tuple[Any, Any]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _png_size_inches(path: Path, dpi: int) -> tuple[float, float]:
    """Return the (width, height) of a PNG in inches at ``dpi``.

    Read from the IHDR chunk rather than an image library: the pixel geometry
    is the only thing needed and it must be readable wherever the generator
    runs.
    """
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width / dpi, height / dpi


def _save_figure(fig: Any, path: Path, caption: str, source_hash: str) -> str:
    """Write one figure and return the SHA-256 of the bytes that were written.

    Raises:
        ValueError: when the written figure is larger than the printable text
            block, which would force the typesetter to scale it down and take
            its type below the legibility floor with it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        metadata={
            "Title": path.stem.replace("_", " ").title(),
            "Description": caption,
            "Source": f"GEO-INFER repository source hash {source_hash}",
        },
    )
    width_in, height_in = _png_size_inches(path, FIGURE_DPI)
    tolerance = 1.02
    if (
        width_in > TEXT_BLOCK_WIDTH_IN * tolerance
        or height_in > MAX_FIGURE_HEIGHT_IN * tolerance
    ):
        raise ValueError(
            f"{path.name} is {width_in:.2f}in x {height_in:.2f}in, larger than the "
            f"printable box {TEXT_BLOCK_WIDTH_IN}in x {MAX_FIGURE_HEIGHT_IN:.2f}in; "
            "it would be scaled down at typeset time and its type with it"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_leaves_room_for_text(height_in: float, filename: str) -> None:
    """Refuse a float tall enough to claim a page of its own.

    ``\floatpagefraction`` decides this at typeset time, and it measures the
    whole float — image plus caption.  A figure that clears it is moved to a
    float page carrying the caption and the folio and nothing else, which is
    the defect this bound exists to prevent.  Checking it here means the
    generator fails rather than the PDF quietly growing a near-empty page.

    Raises:
        ValueError: when the float would fill more of the text block than
            ``FLOAT_PAGE_FRACTION`` allows.
    """
    float_height = height_in + FIGURE_CAPTION_HEIGHT_IN
    limit = TEXT_BLOCK_HEIGHT_IN * FLOAT_PAGE_FRACTION
    if float_height > limit:
        raise ValueError(
            f"{filename} would be typeset as a {float_height:.2f}in float "
            f"against a {limit:.2f}in float-page threshold, so LaTeX would "
            "give it a page of its own; shorten the figure"
        )


def _record_statuses(
    verification: VerificationRecord | None,
) -> dict[str, str]:
    """Map each default-tier command group to its recorded outcome.

    A group the record does not mention reads as ``not-run`` rather than as
    passed, so an abstract drawn from a stale record cannot show a pass the
    record does not hold.
    """
    recorded = (
        {result.name: result.status for result in verification.results}
        if verification is not None
        else {}
    )
    return {
        name: recorded.get(name, "not-run") for name, _command in VERIFICATION_COMMANDS
    }


def _wrap(text: str, width_chars: int) -> str:
    """Greedy word wrap for figure panel text."""
    lines: list[str] = []
    for word in text.split():
        if not lines or len(lines[-1]) + 1 + len(word) > width_chars:
            lines.append(word)
        else:
            lines[-1] = f"{lines[-1]} {word}"
    return "\n".join(lines)


def _draw_graphical_abstract(
    plt: Any, inventory: RepositoryInventory, verification: VerificationRecord | None
) -> Any:
    """Draw the four-panel graphical abstract at the printed text-block width.

    Panel content is declared, not inferred: themes come from
    ``MODULE_THEMES`` with the spotlight modules named by
    ``GRAPHICAL_ABSTRACT_SPOTLIGHT``, the model families restate the Methods
    section, and the outcome dots come from the verification record when one
    is supplied (tests draw the figure without a record and get neutral
    dots).  The smallest type on the canvas is 7pt, headers are 8.4pt or
    larger, and every colour is one of the shared palette constants.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    dark = "#1a1a1a"
    soft = "#444444"
    width = TEXT_BLOCK_WIDTH_IN
    margin, gutter = 0.12, 0.22
    panel_w = (width - 2 * margin - gutter) / 2
    row_height, row_gap = 1.70, 0.24
    height = 0.30 + 0.10 + row_height + row_gap + row_height + 0.14 + 0.28
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("#fbfcfd")
    # The axes is the content extent, not the whole canvas: _save_figure
    # typesets the tight-cropped PNG, and a full-canvas axes would drag the
    # crop out to figsize + the 0.1in savefig pad on every side, past the
    # printable box.  Data inches map 1:1 onto figure inches inside this
    # viewport, so the panel geometry below is authored in printed inches.
    content_x0, content_y0 = 0.12, 0.17
    content_x1, content_y1 = width - 0.12, height - 0.05
    ax = fig.add_axes(
        (
            content_x0 / width,
            content_y0 / height,
            (content_x1 - content_x0) / width,
            (content_y1 - content_y0) / height,
        )
    )
    ax.set_xlim(content_x0, content_x1)
    ax.set_ylim(content_y0, content_y1)
    ax.axis("off")

    def panel(x0: float, y0: float, number: str, title: str, accent: str) -> tuple:
        """Draw one rounded panel; return the origin of its content area."""
        box_h = 0.34
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                panel_w,
                row_height,
                boxstyle="round,pad=0.015,rounding_size=0.05",
                facecolor="white",
                edgecolor=accent,
                linewidth=1.3,
                zorder=0,
            )
        )
        ax.text(
            x0 + 0.09,
            y0 + row_height - 0.16,
            f"{number} · {title}",
            fontsize=8.4,
            fontweight="bold",
            color=accent,
            va="top",
            ha="left",
            zorder=3,
        )
        ax.plot(
            [x0 + 0.07, x0 + panel_w - 0.07],
            [y0 + row_height - box_h, y0 + row_height - box_h],
            color=accent,
            linewidth=0.7,
            alpha=0.45,
            zorder=1,
        )
        return x0 + 0.09, y0 + row_height - box_h - 0.06

    def arrow(pos_a: tuple, pos_b: tuple) -> None:
        ax.add_patch(
            FancyArrowPatch(
                pos_a,
                pos_b,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#555555",
                zorder=4,
            )
        )

    row1_y = 0.42 + row_height + row_gap
    row2_y = 0.42
    left_x, right_x = margin, margin + panel_w + gutter
    mid1, mid2 = row1_y + row_height / 2, row2_y + row_height / 2

    ax.text(
        width / 2,
        height - 0.30,
        f"GEO-INFER: from {inventory.module_count} modules to verified claims",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
        ha="center",
        va="baseline",
    )

    # Panel 1: the measured module set, grouped by declared theme.
    px, py = panel(
        left_x,
        row1_y,
        "1",
        f"{inventory.module_count} modules in {len(MODULE_THEMES)} themes",
        SOURCE_COLOR,
    )
    subcol_w = (panel_w - 0.27) / 2
    columns = (px, px + subcol_w + 0.09)
    cursors = {columns[0]: py, columns[1]: py}
    for index, (theme, members) in enumerate(MODULE_THEMES):
        suffixes = {name.removeprefix("GEO-INFER-") for name in members}
        spotlight = GRAPHICAL_ABSTRACT_SPOTLIGHT[theme]
        missing = [name for name in spotlight if name not in suffixes]
        if missing:
            raise ValueError(
                f"graphical abstract spotlights modules absent from the {theme} "
                f"theme: {', '.join(missing)}"
            )
        rest = len(members) - len(spotlight)
        names = ", ".join(spotlight) + (f" +{rest} more" if rest else "")
        column = columns[index // 3]
        cursor = cursors[column]
        ax.text(
            column,
            cursor,
            GRAPHICAL_ABSTRACT_THEME_SHORT[theme],
            fontsize=7.4,
            fontweight="bold",
            color=dark,
            va="top",
            ha="left",
            zorder=3,
        )
        wrapped = _wrap(names, 22)
        ax.text(
            column,
            cursor - 0.145,
            wrapped,
            fontsize=7.0,
            color="#555555",
            va="top",
            ha="left",
            linespacing=1.15,
            zorder=3,
        )
        cursors[column] = cursor - 0.145 - (wrapped.count("\n") + 1) * 0.098 - 0.055

    # Panel 2: the model families those themes compose into.
    px, py = panel(
        right_x, row1_y, "2", "Geospatial & inference models", CATEGORY_COLOR
    )
    for title, body in (
        (
            "Spatial substrate",
            "H3 indexing and nested hexagons (SPACE), place-scoped analysis "
            "(PLACE), validated UTC schedules (TIME)",
        ),
        (
            "Probabilistic machinery",
            "Free energy (ACT), samplers with diagnostics (BAYES), mapping "
            "(SPM), numerical primitives (MATH)",
        ),
        (
            "Domain engines",
            "Extreme-value risk bounds (RISK), insurance underwriting (INSURANCE)",
        ),
    ):
        ax.text(
            px,
            py,
            title,
            fontsize=7.6,
            fontweight="bold",
            color=dark,
            va="top",
            zorder=3,
        )
        body_wrapped = _wrap(body, 46)
        ax.text(
            px,
            py - 0.135,
            body_wrapped,
            fontsize=7.0,
            color=soft,
            va="top",
            linespacing=1.18,
            zorder=3,
        )
        py -= 0.135 + (body_wrapped.count("\n") + 1) * 0.098 + 0.05

    # Panel 3: the evidence and verification loop, with recorded outcomes.
    px, py = panel(left_x, row2_y, "3", "Evidence & verification loop", TEST_COLOR)
    ax.text(
        px,
        py,
        "each build: measure → draw → run → publish",
        fontsize=7.4,
        style="italic",
        color=soft,
        va="top",
        zorder=3,
    )
    statuses = _record_statuses(verification)
    first_row_y = py - 0.155
    cursor = first_row_y
    for name, status in statuses.items():
        ax.add_patch(
            plt.Circle(
                (px + 0.055, cursor - 0.045),
                0.026,
                color=STATUS_COLORS.get(status, STATUS_COLORS["not-run"]),
                zorder=3,
            )
        )
        label = name if status == "passed" else f"{name} ({status.replace('-', ' ')})"
        ax.text(
            px + 0.10,
            cursor - 0.045,
            label,
            fontsize=7.0,
            color=dark,
            va="center",
            ha="left",
            zorder=3,
        )
        cursor -= 0.112
    ax.add_patch(
        FancyArrowPatch(
            (px + 2.05, cursor + 0.067),
            (px + 2.05, first_row_y - 0.045),
            connectionstyle="arc3,rad=-0.42",
            arrowstyle="-|>",
            mutation_scale=8,
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            color="#777777",
            zorder=2,
        )
    )
    ax.text(
        px,
        cursor + 0.02,
        "record reused while it names this tree",
        fontsize=6.9,
        color="#777777",
        va="top",
        zorder=3,
    )

    # Panel 4: what the loop establishes.
    px, py = panel(right_x, row2_y, "4", "Verified claims", SURFACE_COLOR)
    failed = any(status not in ("passed", "not-run") for status in statuses.values())
    complete = statuses and all(status == "passed" for status in statuses.values())
    if complete:
        stat, stat_color = (
            f"{sum(status == 'passed' for status in statuses.values())} / "
            f"{len(statuses)} command groups passed",
            STATUS_COLORS["passed"],
        )
    elif failed:
        stat, stat_color = (
            "a recorded group failed",
            STATUS_COLORS["failed"],
        )
    else:
        stat, stat_color = "outcomes published per group", "#555555"
    ax.text(
        px,
        py,
        stat,
        fontsize=11,
        fontweight="bold",
        color=stat_color,
        va="top",
        zorder=3,
    )
    cursor = py - 0.26
    for bullet in (
        "Counts are cross-checked to partition their totals before publication",
        "Figures and captions ship with SHA-256 provenance in the registry",
        "A publication build refuses an empty or failing record",
    ):
        wrapped = _wrap(bullet, 44)
        ax.text(
            px,
            cursor,
            wrapped,
            fontsize=7.2,
            color=soft,
            va="top",
            linespacing=1.2,
            zorder=3,
        )
        cursor -= (wrapped.count("\n") + 1) * 0.098 + 0.045
    ax.text(
        px,
        cursor - 0.02,
        _wrap(
            "Correctness, reproducibility, and documentation claims become executable.",
            44,
        ),
        fontsize=7.6,
        style="italic",
        color=dark,
        va="top",
        linespacing=1.2,
        zorder=3,
    )

    # Flow arrows: 1 to 2, an elbow 2 to 3, and 3 to 4.
    arrow((left_x + panel_w + 0.005, mid1), (right_x - 0.005, mid1))
    elbow_y = row2_y + row_height + row_gap / 2
    ax.plot(
        [right_x + panel_w / 2, right_x + panel_w / 2],
        [row1_y, elbow_y],
        color="#555555",
        linewidth=1.2,
        zorder=4,
        solid_capstyle="round",
    )
    ax.plot(
        [right_x + panel_w / 2, left_x + panel_w / 2],
        [elbow_y, elbow_y],
        color="#555555",
        linewidth=1.2,
        zorder=4,
    )
    arrow(
        (left_x + panel_w / 2, elbow_y),
        (left_x + panel_w / 2, row2_y + row_height + 0.01),
    )
    arrow((left_x + panel_w + 0.005, mid2), (right_x - 0.005, mid2))

    ax.text(
        width / 2,
        0.24,
        f"Measured from the checkout at commit {inventory.commit} · "
        f"source hash {inventory.source_hash[:12]} · v{inventory.project_version}",
        fontsize=7.2,
        color="#666666",
        ha="center",
    )
    return fig


GRAPHICAL_ABSTRACT_SPOTLIGHT: dict[str, tuple[str, ...]] = {
    "Spatial and place-based": ("SPACE", "TIME", "MARINE"),
    "Bayesian and active inference": ("BAYES", "ACT", "MATH"),
    "Agents and AI orchestration": ("AGENT", "OPS", "COMMS"),
    "Governance, risk and domain": ("INSURANCE", "RISK", "SEC"),
    "Data, API and applications": ("API", "DATA", "IOT"),
    "Infrastructure and validation": ("TEST", "INTRA", "EXAMPLES"),
}

GRAPHICAL_ABSTRACT_THEME_SHORT: dict[str, str] = {
    "Spatial and place-based": "Spatial & place",
    "Bayesian and active inference": "Bayes & active",
    "Agents and AI orchestration": "Agents & AI",
    "Governance, risk and domain": "Governance & risk",
    "Data, API and applications": "Data & apps",
    "Infrastructure and validation": "Infra & validation",
}


def generate_figures(
    inventory: RepositoryInventory,
    output_dir: Path,
    verification: VerificationRecord | None = None,
) -> tuple[FigureSpec, ...]:
    """Generate publication figures from the measured inventory.

    ``verification`` is the record this build publishes, when it holds one;
    the graphical abstract draws each command group's recorded outcome from
    it.  Callers without a record (the figure regression suite) get neutral
    outcome dots.
    """
    _matplotlib, plt = _import_matplotlib()
    specs = (
        FigureSpec(
            "fig:graphical_abstract",
            "graphical_abstract.png",
            _caption_graphical_abstract(inventory, verification),
            "manuscript.generate_research_artifacts.generate_figures",
            _alt_graphical_abstract(inventory, verification),
        ),
        FigureSpec(
            "fig:module_inventory",
            "module_inventory.png",
            _caption_module_inventory(inventory),
            "manuscript.generate_research_artifacts.generate_figures",
            _alt_module_inventory(inventory),
        ),
        FigureSpec(
            "fig:research_spine",
            "research_spine.png",
            _caption_research_spine(inventory),
            "manuscript.generate_research_artifacts.generate_figures",
            _alt_research_spine(inventory),
        ),
        FigureSpec(
            "fig:validation_surface",
            "validation_surface.png",
            _caption_validation_surface(inventory),
            "manuscript.generate_research_artifacts.generate_figures",
            _alt_validation_surface(inventory),
        ),
    )
    digests: dict[str, str] = {}
    module_rows = sorted(
        inventory.modules, key=lambda item: (-item.source_files, item.name)
    )
    labels = [item.name.removeprefix("GEO-INFER-") for item in module_rows]
    source_counts = [item.source_files for item in module_rows]
    test_counts = [item.test_files for item in module_rows]

    spine_labels = {
        module.name.removeprefix("GEO-INFER-") for module in inventory.focused_modules
    }

    fig = _draw_graphical_abstract(plt, inventory, verification)
    digests[specs[0].filename] = _save_figure(
        fig, output_dir / specs[0].filename, specs[0].caption, inventory.source_hash
    )
    plt.close(fig)
    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
            "font.size": 8,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#666666",
        }
    ):
        # One column of 45 module rows cannot be both legible and short: at a
        # readable row height it fills nine tenths of the text block, LaTeX
        # grants it a float page, and that page carries a caption and a folio
        # and nothing else.  Splitting the same rows across two panels halves
        # the height at unchanged row spacing, so the figure sits at the top of
        # an ordinary text page with the section's prose beneath it.
        split = ceil(len(labels) / 2)
        panels = ((0, split), (split, len(labels)))
        inventory_height = min(
            MAX_FIGURE_HEIGHT_IN,
            max(3.0, INVENTORY_FIGURE_CHROME_IN + split * INVENTORY_ROW_HEIGHT_IN),
        )
        _assert_leaves_room_for_text(inventory_height, "module_inventory.png")
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_BLOCK_WIDTH_IN, inventory_height))
        # Both panels share one count axis, so a bar in the right panel is
        # directly comparable with a bar in the left one.
        count_limit = max((*source_counts, *test_counts, 1)) * 1.08
        bar_height = 0.38
        for axis, (start, stop) in zip(axes, panels):
            panel_labels = labels[start:stop]
            y = list(range(len(panel_labels)))
            axis.barh(
                [position - bar_height / 2 for position in y],
                source_counts[start:stop],
                bar_height,
                color=SOURCE_COLOR,
                alpha=0.92,
                label="Source files",
            )
            axis.barh(
                [position + bar_height / 2 for position in y],
                test_counts[start:stop],
                bar_height,
                color=TEST_COLOR,
                alpha=0.92,
                label="Test files",
            )
            axis.set_yticks(y, panel_labels)
            for tick in axis.get_yticklabels():
                if tick.get_text() in spine_labels:
                    tick.set_fontweight("bold")
                    tick.set_color(SPINE_COLOR)
            axis.set_ylim(len(panel_labels) - 0.5, -0.5)
            axis.set_xlim(0, count_limit)
            axis.tick_params(axis="y", labelsize=8)
            axis.tick_params(axis="x", labelsize=7.5)
            axis.set_title(f"Modules {start + 1}–{stop} of {len(labels)}", fontsize=8.5)
            axis.set_xlabel("Tracked Python files")
            axis.set_axisbelow(True)
        axes[0].legend(frameon=False, loc="lower right", fontsize=7.5)
        fig.suptitle(
            "GEO-INFER module evidence inventory", fontsize=11, fontweight="bold"
        )
        fig.tight_layout()
        digests[specs[1].filename] = _save_figure(
            fig, output_dir / specs[1].filename, specs[1].caption, inventory.source_hash
        )
        plt.close(fig)

        focus = inventory.focused_modules
        focus_labels = [item.name.removeprefix("GEO-INFER-") for item in focus]
        focus_source = [item.source_files for item in focus]
        focus_tests = [item.test_files for item in focus]
        fig, ax = plt.subplots(figsize=(TEXT_BLOCK_WIDTH_IN, 3.3))
        positions = list(range(len(focus_labels)))
        bar_width = 0.36
        top = max((*focus_source, *focus_tests, 1))
        for offset, counts, colour, name in (
            (-bar_width / 2, focus_source, SOURCE_COLOR, "Source files"),
            (bar_width / 2, focus_tests, TEST_COLOR, "Test files"),
        ):
            bars = ax.bar(
                [position + offset for position in positions],
                counts,
                bar_width,
                label=name,
                color=colour,
                alpha=0.92,
            )
            ax.bar_label(bars, fmt="%d", fontsize=8.5, padding=2, color="#222222")
        ax.set_title("Research-spine evidence surfaces", fontsize=10.5)
        ax.set_ylabel("Tracked Python files", fontsize=8.5)
        ax.set_xticks(positions, focus_labels, fontsize=9)
        ax.set_ylim(0, top * 1.25)
        ax.legend(frameon=False, loc="upper center", ncols=2, fontsize=8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        digests[specs[2].filename] = _save_figure(
            fig, output_dir / specs[2].filename, specs[2].caption, inventory.source_hash
        )
        plt.close(fig)

        categories = tuple(inventory.test_files_by_category)
        category_counts = [
            inventory.test_files_by_category[category] for category in categories
        ]
        evidence_labels = ("Modules", "Documentation\npages", "Validator\nfiles")
        evidence_counts = (
            inventory.module_count,
            inventory.documentation_pages,
            inventory.validator_files,
        )
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_BLOCK_WIDTH_IN, 3.3))
        for axis, names, counts, colour, label in (
            (axes[0], categories, category_counts, CATEGORY_COLOR, "Files"),
            (axes[1], evidence_labels, evidence_counts, SURFACE_COLOR, "Count"),
        ):
            bars = axis.bar(names, counts, color=colour, alpha=0.92, width=0.62)
            axis.bar_label(bars, fmt="%d", fontsize=8, padding=2, color="#222222")
            axis.set_ylim(0, max((*counts, 1)) * 1.22)
            axis.set_ylabel(label, fontsize=8.5)
            axis.tick_params(axis="x", labelsize=7.5)
            axis.tick_params(axis="y", labelsize=7.5)
            axis.set_axisbelow(True)
        axes[0].set_title("Test files by directory category", fontsize=9)
        axes[1].set_title("Repository evidence surfaces", fontsize=9)
        fig.suptitle(
            "Validation and documentation evidence", fontsize=10.5, fontweight="bold"
        )
        fig.tight_layout()
        digests[specs[3].filename] = _save_figure(
            fig, output_dir / specs[3].filename, specs[3].caption, inventory.source_hash
        )
        plt.close(fig)
    return tuple(replace(spec, sha256=digests[spec.filename]) for spec in specs)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_figure_registry(
    path: Path, specs: Sequence[FigureSpec], inventory: RepositoryInventory
) -> None:
    """Write a fail-closed registry for the generated figure set."""
    if not specs:
        raise ValueError("figure registry requires at least one generated figure")
    filenames = [spec.filename for spec in specs]
    labels = [spec.label for spec in specs]
    if len(set(filenames)) != len(filenames) or len(set(labels)) != len(labels):
        raise ValueError("figure registry labels and filenames must be unique")
    for spec in specs:
        if not spec.label.startswith("fig:") or not spec.caption.strip():
            raise ValueError(f"invalid figure specification: {spec!r}")
        if not spec.alt_text.strip():
            raise ValueError(f"invalid figure specification: {spec!r}")
        if len(spec.sha256) != 64:
            raise ValueError(
                f"figure {spec.filename} carries no content digest: {spec.sha256!r}"
            )
        if not (path.parent / spec.filename).is_file():
            raise FileNotFoundError(path.parent / spec.filename)
    _write_json(
        path,
        {
            "schema_version": FIGURE_SCHEMA,
            "source_commit": inventory.commit,
            "source_hash": inventory.source_hash,
            "figures": [
                asdict(spec) for spec in sorted(specs, key=lambda item: item.label)
            ],
        },
    )


def _verification_payload(record: VerificationRecord) -> dict[str, Any]:
    """Serialise a verification record, stamped with the tree it describes.

    The commit and source fingerprint are part of the record because the
    commands are expensive: a later build has to be able to tell whether an
    existing record still describes the checkout in front of it, and an
    unstamped record cannot answer that.  They are the record's own, never the
    writing build's: a carried-forward record that got restamped with the
    commit of the build that merely republished it would look measured here
    and would never be recognisable as carried forward again.
    """
    return {
        "schema_version": RESEARCH_SCHEMA,
        "full_validation_requested": record.full_validation_requested,
        "results": [
            {
                key: value
                for key, value in asdict(result).items()
                if key != "receipt" or value is not None
            }
            for result in record.results
        ],
        "source_commit": record.source_commit,
        "source_hash": record.source_hash,
    }


UNSTAMPED = "unstamped"


def _stamp(value: object) -> str:
    """Return a record's provenance stamp, or ``UNSTAMPED`` when it has none.

    Records written before the stamps existed hold results and no provenance.
    Rejecting them would put them back in the class of evidence a
    non-verifying build may delete, which is the defect.  ``UNSTAMPED`` is not
    a commit and never equals one, so such a record can never be mistaken for
    a match — it can only be carried forward and published as unattributed.
    """
    return value if isinstance(value, str) and value else UNSTAMPED


def _verification_record_path(root: Path) -> Path:
    """Return the one path the verification record is read from and written to."""
    return root / "output" / "data" / "research_verification.json"


def _load_stored_verification(root: Path) -> VerificationRecord | None:
    """Return the stored verification record, whatever tree it describes.

    Provenance is reported, not filtered: the caller decides whether a record
    naming another tree is this build's evidence.  ``None`` means there is no
    evidence on disk to lose — the file is absent, unreadable, malformed, or
    holds no executed command — and only then may a build write an empty
    record over it.
    """
    path = _verification_record_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("results")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        results = tuple(VerificationResult(**entry) for entry in raw)
    except TypeError:
        return None
    return VerificationRecord(
        results=results,
        source_commit=_stamp(payload.get("source_commit")),
        source_hash=_stamp(payload.get("source_hash")),
        full_validation_requested=bool(payload.get("full_validation_requested")),
    )


def load_matching_verification(
    root: Path, inventory: RepositoryInventory, *, full_validation: bool
) -> tuple[VerificationResult, ...] | None:
    """Return a stored verification record that still describes this tree.

    Verification takes minutes, and the render hydrates on a bounded
    subprocess timeout, so re-running it on every render is not viable.
    Reuse is only safe when the record names the same source fingerprint and
    was produced at the same tier; anything else returns ``None`` and the
    caller runs the commands.
    """
    stored = _load_stored_verification(root)
    if stored is None:
        return None
    if stored.source_hash != inventory.source_hash:
        return None
    if stored.source_commit != inventory.commit:
        return None
    if stored.full_validation_requested != full_validation:
        return None
    return stored.results


def _prune_undefined_groups(
    record: VerificationRecord | None,
) -> VerificationRecord | None:
    """Drop results for groups no tier defines any more, and say which.

    A result whose group name is outside both command tables cannot be
    counted against any denominator, so publishing it is the ``9 of 7``
    arithmetic by another route.  Dropping it is not the deletion
    :func:`resolve_verification` exists to prevent: that rule protects
    evidence a build declined to re-measure, and a group the generator no
    longer defines cannot be re-measured at all.  The names are printed so
    the removal is never silent.

    Returns:
        The record with only defined groups kept, or ``None`` when nothing
        interpretable survives — the same answer a missing record gives, so
        the caller treats both alike.
    """
    if record is None:
        return None
    kept = tuple(
        result for result in record.results if result.name in ALL_COMMAND_GROUPS
    )
    dropped = sorted(
        result.name
        for result in record.results
        if result.name not in ALL_COMMAND_GROUPS
    )
    if dropped:
        print(
            "dropping stored verification results for command groups no tier "
            f"defines: {', '.join(dropped)}",
            file=sys.stderr,
        )
    if not kept:
        return None
    return replace(record, results=kept)


def _effective_tier(
    stored: VerificationRecord | None, *, full_validation: bool
) -> bool:
    """Return the tier a build must work at so it narrows no stored record.

    Section 6.2's rule is that a build never deletes an executed-command
    result it did not re-measure.  A verifying build replaces the record
    wholesale, so running the default tier over a stored full-validation
    record deleted the unit and integration results — the two failures — and
    republished "7 passed, 0 failed".  Widening the run is what makes the
    replacement lossless: every group the stored record covers is measured
    again, so nothing is dropped and every published result names this
    build's own tree.

    A build that is not verifying is unaffected in substance — it replaces
    nothing — but it publishes at the same tier, so the denominator matches
    the record it carries forward.
    """
    if stored is None:
        return full_validation
    if full_validation:
        return True
    if stored.full_validation_requested:
        return True
    return any(result.name in FULL_VALIDATION_GROUPS for result in stored.results)


def resolve_verification(
    root: Path,
    inventory: RepositoryInventory,
    *,
    verify: bool,
    full_validation: bool,
    reuse_verification: bool,
) -> VerificationRecord:
    """Return the verification record this build publishes.

    The reuse lookup runs at every tier, not only when this build was asked to
    verify.  A stored record names the tree it describes, so while it still
    describes this one it is this build's evidence too.  Gating the lookup on
    ``verify`` made a default render — which is what the supported render path
    performs — overwrite a measured record with an empty one and republish
    "not run" in its place.

    Matching on identity alone was not enough.  ``inventory.commit`` carries a
    ``-dirty`` suffix the moment ``git status --porcelain`` prints a line, so a
    single untracked file, or any commit made after the evidence was measured,
    made the lookup miss and the empty record win again.  That is not a rare
    state: it is every render taken during ordinary development.  So a record
    that does not match is still not discarded — it is carried forward with
    its own commit and source hash intact, and the manuscript publishes those
    alongside this build's, so a reader can see the evidence was measured on a
    different tree.  A build that ran no command can now only ever add
    provenance to the record; it can never empty it.  The carry-forward is
    not gated on ``reuse_verification``: that flag chooses whether a matching
    record may stand in for a run this build was already asked to make, and
    declining that shortcut is not a licence to delete the measurement.  A
    caller that passes ``reuse_verification=False`` with ``verify=False`` has
    asked for no measurement at all, and gets the stored record carried
    forward.

    Returns:
        The reused record when a stored one still describes this tree, the
        freshly run commands when this build was asked to verify, the stored
        record marked ``measured_elsewhere`` when one exists but describes
        another tree, and an empty record only when there is no stored
        evidence at all.
    """
    stored = _prune_undefined_groups(_load_stored_verification(root))
    tier = _effective_tier(stored, full_validation=full_validation)
    if (
        reuse_verification
        and stored is not None
        and (
            stored.source_hash == inventory.source_hash
            and stored.source_commit == inventory.commit
            and stored.full_validation_requested == tier
        )
    ):
        print(
            "reusing the stored verification record for source hash "
            f"{inventory.source_hash}",
            file=sys.stderr,
        )
        return stored
    if verify:
        if tier != full_validation:
            widened = sorted(
                set(defined_command_groups(full_validation=tier))
                - set(defined_command_groups(full_validation=full_validation))
            )
            print(
                "widening this run to the full-validation tier: the stored "
                "record covers "
                f"{', '.join(widened)}, which this tier does not define. "
                "Re-measuring them is the only way to replace the record "
                "without dropping results this build did not re-run.",
                file=sys.stderr,
            )
        return VerificationRecord(
            results=run_verification(root, full_validation=tier),
            source_commit=inventory.commit,
            source_hash=inventory.source_hash,
            full_validation_requested=tier,
        )
    if stored is not None:
        print(
            "carrying the stored verification record forward: it was measured "
            f"at commit {stored.source_commit} (source hash "
            f"{stored.source_hash}), not at this build's {inventory.commit} "
            f"(source hash {inventory.source_hash})",
            file=sys.stderr,
        )
        return replace(stored, measured_elsewhere=True)
    return VerificationRecord(
        results=(),
        source_commit=inventory.commit,
        source_hash=inventory.source_hash,
        full_validation_requested=tier,
    )


def _verification_argv(command: str, root: Path) -> list[str]:
    """Parse a trusted command declaration and expand compileall path operands.

    Only compileall paths use globbing. Quoted scripts and other arguments keep
    their literal content; no shell interprets them or expands substitutions.
    Missing paths and unmatched patterns are rejected: compileall can print
    "Can't list" while still exiting successfully. At least one explicit path
    is required, rather than implicitly compiling the interpreter's sys.path.
    """
    argv = shlex.split(command)
    if not argv:
        raise ValueError("Verification commands must not be empty")
    if argv[1:3] != ["-m", "compileall"]:
        return argv
    expanded = argv[:3]
    value_options = {
        "-r",
        "-d",
        "-s",
        "-p",
        "-x",
        "-i",
        "-j",
        "--workers",
        "--invalidation-mode",
        "-o",
        "-e",
    }
    option_value = False
    only_paths = False
    path_count = 0
    for argument in argv[3:]:
        if option_value:
            expanded.append(argument)
            option_value = False
            continue
        if not only_paths and argument.startswith("-"):
            expanded.append(argument)
            option_value = argument in value_options
            only_paths = argument == "--"
            continue
        if glob.has_magic(argument):
            paths = sorted(glob.glob(argument, root_dir=root))
            if not paths:
                raise ValueError(
                    f"Verification compileall path pattern matched no paths: {argument}"
                )
        else:
            paths = [argument]
        for path in paths:
            if not ((root / path).is_file() or (root / path).is_dir()):
                raise ValueError(f"Verification compileall path does not exist: {path}")
        expanded.extend(paths)
        path_count += len(paths)
    if not path_count:
        raise ValueError(
            "Verification compileall must select at least one explicit path"
        )
    return expanded


def run_verification(
    root: Path, *, full_validation: bool = False
) -> tuple[VerificationResult, ...]:
    """Run and record the research verification commands.

    Shared execution provides monotonic deadlines, process-tree cleanup and
    immutable attempts, including retained output and artifact hashes. Each
    result links to that receipt relative to ``root``; its bounded output tail
    is a summary. Legacy records without receipt references remain readable.
    A timeout or ordinary failure is recorded without retrying or aborting the
    remaining groups. Importing the generator does not initialize execution.
    """
    from geo_infer_test.execution import run_command

    commands = (
        *VERIFICATION_COMMANDS,
        *(FULL_VALIDATION_COMMANDS if full_validation else ()),
    )
    if not commands:
        raise ValueError("Verification must select at least one command group")
    names = [name for name, _command in commands]
    if any(not name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("Verification command group names must be unique and nonempty")
    planned_commands = [
        (name, command, _verification_argv(command, root)) for name, command in commands
    ]
    results: list[VerificationResult] = []
    for name, command, argv in planned_commands:
        # The bare validators get the same 600s envelope as the unified test
        # runners' own per-command timeout.  The full-validation runner
        # groups aggregate a whole category of per-module commands, so their
        # outer envelope has to exceed the runner's own aggregate budget or
        # it would cut off healthy runs; the CI job's own wall budget is the
        # backstop above it.
        timeout = (
            FULL_VALIDATION_TIMEOUT_SECONDS
            if name in FULL_VALIDATION_GROUPS
            else VERIFICATION_TIMEOUT_SECONDS
        )
        completed = run_command(argv, name, timeout, cwd=root)
        combined = f"{completed.stdout}\n{completed.stderr}".strip()
        results.append(
            VerificationResult(
                name=name,
                command=command,
                status=(
                    "timeout"
                    if completed.status == "TIMEOUT"
                    else "passed"
                    if completed.success and completed.status == "PASS"
                    else "failed"
                ),
                return_code=completed.returncode,
                duration_seconds=round(completed.duration, 3),
                output_tail=combined[-2000:],
                receipt=(
                    Path(os.path.relpath(completed.receipt, root)).as_posix()
                    if completed.receipt
                    else None
                ),
            )
        )
    return tuple(results)


def defined_command_groups(*, full_validation: bool) -> tuple[str, ...]:
    """Return the names of every verification command group this build defines."""
    commands = (
        *VERIFICATION_COMMANDS,
        *(FULL_VALIDATION_COMMANDS if full_validation else ()),
    )
    return tuple(name for name, _command in commands)


def _verification_table(
    results: Sequence[VerificationResult],
    *,
    full_validation: bool,
) -> str:
    """Render one row per defined command group, run or not.

    The manuscript previously described the verification record in prose and
    left the per-command outcomes inside a JSON file no reader of the PDF
    opens.  Every group this build defines gets a row here, so a skipped group
    is a visible "not run" line rather than an absence, and a failed group is
    published with its return code instead of being summarised away.
    """
    commands = (
        *VERIFICATION_COMMANDS,
        *(FULL_VALIDATION_COMMANDS if full_validation else ()),
    )
    recorded = {result.name: result for result in results}
    # Column widths are proportional to the dash counts pandoc reads from the
    # separator row.  The Group column has to hold the longest group name
    # without breaking mid-word, and single-word headers keep the header from
    # wrapping and stealing rows from the body.
    rows = [
        "| Group | Command | Status | Exit | Seconds |",
        "| "
        + " | ".join(("-" * 21, "-" * 33, "-" * 8, "-" * 4 + ":", "-" * 7 + ":"))
        + " |",
    ]
    for name, command in commands:
        result = recorded.get(name)
        if result is None:
            rows.append(f"| `{name}` | `{command}` | not run | -- | -- |")
            continue
        rows.append(
            f"| `{name}` | `{result.command}` | {result.status} | "
            f"{result.return_code} | {result.duration_seconds} |"
        )
    # A result for a group this tier does not define is an incoherent
    # record: ``build_variables`` refuses it, because counting it as a pass
    # against this tier's denominator is exactly the "9 of 7" arithmetic.
    # These rows exist so a caller inspecting the table directly can see what
    # the refusal is about rather than seeing the group vanish.
    unrecorded = sorted(set(recorded) - {name for name, _ in commands})
    for name in unrecorded:
        result = recorded[name]
        rows.append(
            f"| `{name}` | `{result.command}` | {result.status} | "
            f"{result.return_code} | {result.duration_seconds} |"
        )
    return "\n".join(rows)


def _verification_summary(
    results: Sequence[VerificationResult],
    *,
    full_validation: bool,
) -> tuple[str, int, int, int]:
    """Summarise a verification record against the command groups it defines.

    ``unrun`` is measured, never assumed: it counts the defined command groups
    that this record leaves without an outcome — no entry at all, or an entry
    whose status is ``not-run``.  Both cases are counted once, over the defined
    groups, so a record cannot inflate the count by carrying a ``not-run``
    entry for a group this tier does not define.  ``run_verification`` only
    ever writes ``passed`` or ``failed``; ``not-run`` reaches this function
    from a stored record, which is deserialised straight from JSON.  An empty
    record therefore reports every defined group as skipped rather than a
    constant, and a full pass reports zero.
    """
    defined = defined_command_groups(full_validation=full_validation)
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status in ("failed", "timeout") for result in results)
    recorded = {result.name: result for result in results}
    unrun = sum(
        1
        for name in defined
        if name not in recorded or recorded[name].status == "not-run"
    )
    if not results:
        return "not run", passed, failed, unrun
    if failed:
        return f"{passed} passed, {failed} failed", passed, failed, unrun
    return f"{passed} passed", passed, failed, unrun


def _verification_provenance(
    results: Sequence[VerificationResult],
    *,
    inventory: RepositoryInventory,
    measured_commit: str,
    measured_hash: str,
) -> str:
    """State which tree the published verification results were measured on.

    A record can outlive the tree it describes: the commands take minutes, the
    render is bounded, and the commit stamp gains a ``-dirty`` suffix as soon
    as one file in the checkout differs.  Discarding the record in that state
    deleted real evidence; republishing it silently would attribute measured
    outcomes to a tree they never ran against.  This sentence is the third
    option — publish the evidence and name its tree — and it is generated, so
    the manuscript cannot drift from what the record actually says.
    """
    if not results:
        return (
            "holding no executed command: none ran for this build and no "
            "stored record was available, so every defined group is published "
            "as not run"
        )
    same_tree = (
        measured_commit == inventory.commit and measured_hash == inventory.source_hash
    )
    if same_tree:
        return (
            f"measured on this build's own tree, commit `{measured_commit}` "
            f"(source hash `{measured_hash}`)"
        )
    return (
        f"carried forward from commit `{measured_commit}` (source hash "
        f"`{measured_hash}`); the commands were not re-run against this "
        f"build's tree, whose commit is `{inventory.commit}` (source hash "
        f"`{inventory.source_hash}`)"
    )


def build_variables(
    inventory: RepositoryInventory,
    specs: Sequence[FigureSpec],
    verification: VerificationRecord,
) -> dict[str, str]:
    """Return every manuscript replacement from measured inputs.

    The verification evidence arrives as one record rather than as results
    plus a separately-supplied tier and a separately-supplied pair of stamps.
    That is the point: the tier the counts are taken against, the results
    counted, and the tree they were measured on are three views of one
    measurement, and every published incoherence in this area came from a
    caller pairing them differently.  There is now no argument by which a
    caller can hand this function eleven results and a seven-group
    definition.

    Raises:
        ValueError: when the published per-category test counts do not sum to
            the published total, so a distribution that silently loses a bucket
            fails the build instead of shipping; and when the published
            verification counts do not sum to the published defined-group
            count, so an incoherent evidence summary fails the build instead
            of being printed in the abstract.
    """
    full_validation = verification.full_validation_requested
    results = verification.results
    defined = verification.defined_groups
    verification_summary, passed, failed, unrun = _verification_summary(
        results, full_validation=full_validation
    )
    if passed + failed + unrun != len(defined):
        raise ValueError(
            "verification outcomes must partition the defined command groups: "
            f"{passed} passed + {failed} failed + {unrun} unrun = "
            f"{passed + failed + unrun} against {len(defined)} defined at the "
            f"{'full-validation' if full_validation else 'default'} tier "
            f"({', '.join(defined)}); the record holds "
            f"{', '.join(sorted(result.name for result in results)) or 'no group'}"
        )
    measured_commit = verification.source_commit or inventory.commit
    measured_hash = verification.source_hash or inventory.source_hash
    verification_provenance = _verification_provenance(
        results,
        inventory=inventory,
        measured_commit=measured_commit,
        measured_hash=measured_hash,
    )
    distribution = inventory.test_files_by_category
    unnamed = set(distribution) - set(TEST_CATEGORIES)
    if unnamed:
        raise ValueError(
            "test categories are published individually and must all be named; "
            f"unnamed: {', '.join(sorted(unnamed))}"
        )
    categorised = sum(distribution.values())
    if categorised != inventory.test_files:
        raise ValueError(
            f"test-file distribution sums to {categorised} against a total of "
            f"{inventory.test_files}"
        )
    if inventory.module_directory_count != len(inventory.modules):
        raise ValueError(
            "GEO_MODULE_COUNT counts every GEO-INFER-* directory at the "
            f"repository root ({inventory.module_directory_count}) but "
            "GEO_MODULE_TABLE renders only the "
            f"{len(inventory.modules)} src/-bearing modules the census "
            "measures, so the published count and its published table would "
            "disagree. Give the extra directory a src/ package (or remove "
            "it) so the two populations coincide"
        )
    variables: dict[str, str] = {
        "PROJECT_VERSION": inventory.project_version,
        "PROJECT_LICENSE": inventory.project_license,
        "MODULE_COUNT": str(inventory.module_count),
        "MODULE_NAMES": ", ".join(module.name for module in inventory.modules),
        "MODULE_TABLE": _module_table(inventory),
        "MODULE_THEME_COUNT": str(len(MODULE_THEMES)),
        "MODULES_WITH_TESTS_COUNT": str(inventory.modules_with_tests),
        # The GEO_* tokens are the per-module-section surface: the count of
        # GEO-INFER-* directories at the repository root (whether or not they
        # ship a src/ package), one self-contained table row per measured
        # module, and the repository-wide test-file census.  The count and
        # the table describe one population — the check above fails the
        # build when a source-less directory would split them.  Each is a
        # deterministic pure function of the tree at HEAD — alphabetical
        # order, declared-theme lookup, file-derived purpose, no wall clock —
        # so two runs on the same tree are byte-identical.
        "GEO_MODULE_COUNT": str(inventory.module_directory_count),
        "GEO_MODULE_TABLE": _module_purpose_table(inventory),
        "GEO_TEST_COUNT": str(inventory.test_files),
        "SOURCE_FILE_COUNT": _format_count(inventory.source_files),
        "SOURCE_LINE_COUNT": _format_count(inventory.source_lines),
        "TEST_FILE_COUNT": _format_count(inventory.test_files),
        "UNIT_TEST_FILE_COUNT": str(inventory.test_files_by_category.get("unit", 0)),
        "INTEGRATION_TEST_FILE_COUNT": str(
            inventory.test_files_by_category.get("integration", 0)
        ),
        "PERFORMANCE_TEST_FILE_COUNT": str(
            inventory.test_files_by_category.get("performance", 0)
        ),
        "OTHER_TEST_FILE_COUNT": str(inventory.test_files_by_category.get("other", 0)),
        "H3_TEST_FILE_COUNT": str(inventory.h3_test_files),
        "DOCUMENTATION_PAGE_COUNT": str(inventory.documentation_pages),
        "VALIDATOR_FILE_COUNT": str(inventory.validator_files),
        "TEST_TOOLING_FILE_COUNT": str(inventory.test_tooling_files),
        "RESEARCH_COMMIT": inventory.commit,
        "RESEARCH_BRANCH": inventory.branch,
        "RESEARCH_TREE_DIRTY_FILE_COUNT": (
            "unavailable"
            if inventory.dirty_file_count < 0
            else str(inventory.dirty_file_count)
        ),
        "RESEARCH_COMMIT_DATE": inventory.commit_date,
        "RESEARCH_YEAR": (
            inventory.commit_date[:4]
            if inventory.commit_date[:4].isdigit()
            else "unavailable"
        ),
        "MANUSCRIPT_SOURCE_DATE": inventory.manuscript_source_date,
        "MANUSCRIPT_SOURCE_YEAR": _research_year(inventory.manuscript_source_date),
        "RESEARCH_SOURCE_HASH": inventory.source_hash,
        "PYTHON_VERSION": inventory.python_version,
        "FIGURE_COUNT": str(len(specs)),
        "FIGURE_LABELS": ", ".join(spec.label for spec in specs),
        "VERIFICATION_STATUS": verification_summary,
        "VERIFICATION_PASS_COUNT": str(passed),
        "VERIFICATION_FAIL_COUNT": str(failed),
        "VERIFICATION_UNRUN_COUNT": str(unrun),
        "VERIFICATION_DEFINED_COUNT": str(len(defined)),
        "VERIFICATION_RECORD_TIER": (
            "full-validation" if full_validation else "default"
        ),
        "VERIFICATION_TABLE": _verification_table(
            results, full_validation=full_validation
        ),
        "VERIFICATION_RECORD_COMMIT": measured_commit,
        "VERIFICATION_RECORD_SOURCE_HASH": measured_hash,
        "VERIFICATION_RECORD_PROVENANCE": verification_provenance,
    }
    for module in inventory.focused_modules:
        key = module.name.removeprefix("GEO-INFER-")
        variables[f"{key}_SOURCE_FILE_COUNT"] = str(module.source_files)
        variables[f"{key}_TEST_FILE_COUNT"] = str(module.test_files)
        variables[f"{key}_SOURCE_LINE_COUNT"] = _format_count(module.source_lines)
    for spec in specs:
        token = spec.label.removeprefix("fig:").upper()
        variables[f"{token}_CAPTION"] = spec.caption
    return variables


def substitute_manuscript_text(
    text: str, variables: Mapping[str, str]
) -> tuple[str, tuple[str, ...]]:
    """Resolve uppercase manuscript tokens and return unresolved names."""
    unresolved: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            unresolved.add(key)
            return match.group(0)
        return str(variables[key])

    return TOKEN_RE.sub(replace, text), tuple(sorted(unresolved))


_CONFIG_OWNED_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("  version: ", "PROJECT_VERSION", "paper.version"),
    ("  date: ", "MANUSCRIPT_SOURCE_DATE", "paper.date"),
    ("  year: ", "MANUSCRIPT_SOURCE_YEAR", "publication.year"),
    ("  license: ", "PROJECT_LICENSE", "metadata.license"),
)


def _manuscript_source_date(root: Path) -> str:
    """Commit date of the last change to anything but the generator-owned config.

    ``manuscript/config.yaml`` is tracked *and* generator-owned: it carries the
    title-page date, derived from git.  Deriving it from ``HEAD`` makes the
    value un-settleable — recording the refreshed file creates a newer commit
    whose date the file no longer holds — so a clean-tree publication build
    could never start.  Excluding the file from its own input reaches a fixed
    point: committing the refreshed config does not move the date it carries.

    ``RESEARCH_COMMIT_DATE`` is unaffected and still reports ``HEAD``, which is
    what section 6.3 quotes beside ``RESEARCH_COMMIT``.
    """
    return _run_git(
        root,
        "log",
        "-1",
        "--format=%cI",
        "--",
        ":/",
        ":(top,exclude)manuscript/config.yaml",
        default="unavailable",
    )


def _research_year(commit_date: str) -> str:
    return commit_date[:4] if commit_date[:4].isdigit() else "unavailable"


def config_metadata_values(root: Path) -> dict[str, str]:
    """Derive the generator-owned ``config.yaml`` values without a full scan.

    ``config.yaml`` is inside the source-hash input set, so it has to be
    written *before* the digest is taken.  These four values are cheap and come
    from the same producers ``build_variables`` uses (``pyproject.toml`` and
    ``git show``), which is what lets ``generate`` cross-check the two.
    """
    metadata = _project_metadata(root)
    source_date = _manuscript_source_date(root)
    return {
        "PROJECT_VERSION": metadata["version"],
        "PROJECT_LICENSE": metadata["license"],
        "MANUSCRIPT_SOURCE_DATE": source_date,
        "MANUSCRIPT_SOURCE_YEAR": _research_year(source_date),
    }


def refresh_config_metadata(
    root: Path, variables: Mapping[str, str], *, dry_run: bool = False
) -> tuple[str, ...]:
    """Write measured metadata into the authored ``manuscript/config.yaml``.

    The render template copies ``config.yaml`` verbatim, so a ``{{TOKEN}}``
    placed there is never substituted and reaches the title page as literal
    text (or, once LaTeX sees the underscores, as mangled math). The template's
    own exemplar therefore keeps literal metadata refreshed by a script. This
    function is that script for GEO-INFER: the values stay measured rather than
    hand-entered, and the file stays verbatim-copyable.

    The scan is section-aware: each field's dotted name names its top-level
    section, and a two-space-prefixed key only counts while the scanner is
    inside that section, so a same-prefixed key under another section can
    never absorb the write.

    Args:
        root: Repository root holding ``manuscript/config.yaml``.
        variables: Resolved values keyed by the token each field is owned by.
        dry_run: Report the fields that are stale without writing them.

    Returns:
        The dotted names of the fields that changed (or would change).
    """
    config = root / "manuscript" / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    lines = config.read_text(encoding="utf-8").splitlines(keepends=True)
    updated: list[str] = []
    for prefix, key, field in _CONFIG_OWNED_FIELDS:
        if key not in variables:
            raise KeyError(f"config metadata variable is not produced: {key}")
        value = variables[key]
        section = field.split(".")[0]
        in_section = False
        for index, line in enumerate(lines):
            if line and not line[0].isspace():
                in_section = line.startswith(f"{section}:")
                continue
            if not in_section or not line.startswith(prefix):
                continue
            replacement = f'{prefix}"{value}"  # generator-owned ({key})\n'
            if lines[index] != replacement:
                lines[index] = replacement
                updated.append(field)
            break
        else:
            raise ValueError(
                f"config.yaml has no {field} key in its {section} section "
                f"(expected a line starting with {prefix!r})"
            )
    if updated and not dry_run:
        config.write_text("".join(lines), encoding="utf-8")
    return tuple(updated)


def write_resolved_manuscript(
    root: Path, variables: Mapping[str, str]
) -> tuple[Path, ...]:
    """Write only publication manuscript files with all tokens resolved."""
    source_dir = root / "manuscript"
    output_dir = root / "output" / "manuscript"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.md"):
        stale.unlink()
    for stale in output_dir.glob("*.bib"):
        stale.unlink()
    for filename in ("config.yaml", "preamble.md"):
        stale = output_dir / filename
        if stale.exists():
            stale.unlink()
    written: list[Path] = []
    unresolved: dict[str, tuple[str, ...]] = {}
    for source in sorted(source_dir.glob("*.md")):
        if source.name in EXCLUDED_MANUSCRIPT_DOCS:
            continue
        resolved, missing = substitute_manuscript_text(
            source.read_text(encoding="utf-8"), variables
        )
        if missing:
            unresolved[source.name] = missing
        # Resolved manuscripts live one directory shallower than the sources
        # (output/manuscript/ vs manuscript/), so authored figure references
        # written against the source layout are rewritten for the output
        # layout. The literal rewrite assumes every figure reference spells
        # the path exactly "../output/figures/" — CI pins that assumption in
        # tests/test_manuscript_paths.py::TestFigurePathLiterals, which fails
        # the build if any manuscript file introduces another spelling.
        resolved = resolved.replace("../output/figures/", "../figures/")
        destination = output_dir / source.name
        destination.write_text(resolved, encoding="utf-8")
        written.append(destination)
    for filename in ("config.yaml", "preamble.md"):
        source = source_dir / filename
        if source.is_file():
            resolved, missing = substitute_manuscript_text(
                source.read_text(encoding="utf-8"), variables
            )
            if missing:
                unresolved[source.name] = missing
            (output_dir / filename).write_text(resolved, encoding="utf-8")
    for source in sorted(source_dir.glob("*.bib")):
        shutil.copy2(source, output_dir / source.name)
    if unresolved:
        details = "; ".join(
            f"{name}: {', '.join(keys)}" for name, keys in unresolved.items()
        )
        raise ValueError(f"unresolved manuscript variables: {details}")
    return tuple(written)


BIB_ENTRY_RE = re.compile(r"^@[A-Za-z]+\{\s*([^,\s]+)\s*,", re.MULTILINE)
# A pandoc citation key is introduced by ``@`` at the start of a token, so the
# ``@`` must follow whitespace or a bracket. Without that guard an email
# address in the author block reads as a citation.
CITATION_RE = re.compile(r"(?:^|[\s\[;(])@([A-Za-z][A-Za-z0-9_.:+-]*)", re.MULTILINE)
# pandoc-crossref references share the citation syntax but resolve against
# labels in the document, not against the bibliography.
CROSSREF_PREFIXES = ("fig:", "tbl:", "sec:", "eq:", "lst:")

# ``bibliography.fail_on_missing`` / ``fail_on_unused`` are documented config
# keys that no renderer code reads, so a project could set them and get no
# gate. The generator honours them here, using the same line-oriented read as
# _CONFIG_OWNED_FIELDS so config.yaml stays parseable without a YAML dependency.
_BIBLIOGRAPHY_POLICY: tuple[tuple[str, str, bool], ...] = (
    ("  fail_on_missing: ", "fail_on_missing", True),
    ("  fail_on_unused: ", "fail_on_unused", False),
)


def bibliography_policy(root: Path) -> dict[str, bool]:
    """Read the bibliography gate settings from ``manuscript/config.yaml``.

    The read is section-aware: a key only counts while the scanner is inside
    the top-level ``bibliography:`` block, so a same-named key under another
    section (or the block itself reindented so the two-space prefix no longer
    matches) can never silently steer the gates.  A prefix that matches
    nothing raises instead of falling back to defaults — ``fail_on_unused``
    defaults to ``False``, so a miss would quietly disable the
    orphaned-citation gate exactly when the config was edited in a way the
    author believed had enabled it.
    """
    config = root / "manuscript" / "config.yaml"
    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    lines = text.splitlines()
    policy: dict[str, bool] = {}
    matched: dict[str, bool] = {}
    in_bibliography = False
    for line in lines:
        stripped = line.strip()
        if line and not line[0].isspace():
            in_bibliography = stripped == "bibliography:"
            continue
        if not in_bibliography:
            continue
        for prefix, name, _default in _BIBLIOGRAPHY_POLICY:
            if line.startswith(prefix):
                policy[name] = line[len(prefix) :].strip().lower() == "true"
                matched[name] = True
                break
    for prefix, name, default in _BIBLIOGRAPHY_POLICY:
        if name not in matched:
            raise ValueError(
                f"config.yaml has no {name} key under the bibliography block "
                f"(expected a line starting with {prefix!r}); refusing to "
                "fall back to the default silently"
            )
        policy.setdefault(name, default)
    return policy


def audit_bibliography(
    root: Path, manuscript_files: Sequence[Path]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compare the reference database against the citations that use it.

    Returns:
        ``(uncited, undefined)`` — bibliography keys that no manuscript file
        cites, and citation keys with no bibliography entry.
    """
    bib = root / "manuscript" / "references.bib"
    if not bib.is_file():
        return (), ()
    entries = set(BIB_ENTRY_RE.findall(bib.read_text(encoding="utf-8")))
    cited: set[str] = set()
    for path in manuscript_files:
        cited.update(CITATION_RE.findall(path.read_text(encoding="utf-8")))
    undefined = {
        key for key in cited - entries if not key.startswith(CROSSREF_PREFIXES)
    }
    return tuple(sorted(entries - cited)), tuple(sorted(undefined))


def audit_module_census(root: Path, inventory: RepositoryInventory) -> None:
    """Apply the same lightweight literal-count contract as the early docs gate."""
    from geo_infer_test.documentation_contracts import module_catalog_census_errors

    drift = module_catalog_census_errors(
        root, {module.name: module.test_files for module in inventory.modules}
    )
    if drift:
        raise ValueError(
            "module catalog census drift (stated literals vs measured tree): "
            + "; ".join(drift)
        )


def generate(
    root: Path,
    *,
    verify: bool = False,
    full_validation: bool = False,
    allow_dirty: bool = False,
    publication: bool = False,
    reuse_verification: bool = True,
) -> dict[str, Any]:
    """Generate the complete evidence bundle and resolved manuscript.

    The working-tree dirtiness is measured once, before anything is written,
    so the commit stamp describes the state that was actually measured and the
    generator's own outputs are never counted against it.

    A failing verification command is recorded and published, not swallowed
    and not fatal: ``VERIFICATION_STATUS`` reports ``N passed, M failed`` and
    ``{{VERIFICATION_TABLE}}`` prints the per-group return codes.  Aborting the
    build on the first failure made that summary branch unreachable, which is
    the same defect class as a hardcoded count — the manuscript could only ever
    say "not run" or "all passed".  ``publication=True`` still refuses.

    ``reuse_verification`` defaults to ``True`` and is honoured at every tier,
    including a build that was not asked to verify.  The stored record names
    the source hash, commit, and tier it describes, so when it still describes
    this tree it is this build's evidence.  When it describes a different tree
    it is still not discarded: a build with ``verify=False`` cannot write an
    empty record over a populated one at all.  The stored record is carried
    forward with its own commit and source hash, and
    ``{{VERIFICATION_RECORD_PROVENANCE}}`` publishes that it was measured
    elsewhere.  ``reuse_verification=False`` (``--rerun-verification``) makes a
    verifying build run the commands again instead of accepting a matching
    record; it requests no measurement of its own, and it does not license
    deleting a record this build did not replace.

    ``full_validation`` is a floor, not a ceiling.  A verifying build is
    widened to cover every command group the stored record already holds,
    because replacing an eleven-group record with a seven-group one deletes
    four measured outcomes — which is how ``--verify`` turned a record
    holding two failures into a published "7 passed, 0 failed".  The tier the
    build actually measured at travels with the record and is the
    denominator every published verification count is taken against.

    Raises:
        RuntimeError: when the checkout is dirty (or git cannot say) and
            ``allow_dirty`` is not set.  A publication build must not attribute
            uncommitted work to a commit that does not contain it.  Also when
            ``publication`` is set and the evidence record is empty or
            contains a failed group.  A publication build cannot carry a
            record forward: it is refused unless it verifies, and a build that
            verifies publishes what it measured.
    """
    dirty_files = _dirty_file_count(root)
    if dirty_files != 0 and not allow_dirty:
        detail = (
            "git could not report working-tree state"
            if dirty_files < 0
            else f"{dirty_files} uncommitted working-tree entries"
        )
        raise RuntimeError(
            f"refusing to generate from an unclean checkout: {detail}. "
            "Commit the tree, or pass --allow-dirty to stamp the build "
            "'<sha>-dirty' and record the count in "
            "RESEARCH_TREE_DIRTY_FILE_COUNT."
        )
    if publication and not verify:
        raise RuntimeError(
            "a publication build must execute its verification commands; "
            "re-run with --verify or --full-validation"
        )
    # config.yaml is a hashed input, so refresh it before the digest is taken.
    # Writing it afterwards would publish a fingerprint of a tree that no
    # longer exists by the time the run ends.
    refresh_config_metadata(root, config_metadata_values(root))
    inventory = collect_inventory(root, dirty_file_count=dirty_files)
    # The catalog census audit needs only the inventory, so it runs before
    # any figure or resolved file is written: a drifted section fails the
    # build without paying for output it cannot correct.
    audit_module_census(root, inventory)
    output = root / "output"
    data_dir = output / "data"
    figures_dir = output / "figures"
    _write_json(data_dir / "research_inventory.json", inventory.to_dict())
    # The record is resolved before the figures are drawn so the graphical
    # abstract can carry each command group's recorded outcome; nothing in the
    # record depends on figure bytes, so the order swap is safe.
    record = resolve_verification(
        root,
        inventory,
        verify=verify,
        full_validation=full_validation,
        reuse_verification=reuse_verification,
    )
    verification = record.results
    _write_json(_verification_record_path(root), _verification_payload(record))
    specs = generate_figures(inventory, figures_dir, verification=record)
    write_figure_registry(figures_dir / "figure_registry.json", specs, inventory)
    variables = build_variables(inventory, specs, record)
    _write_json(data_dir / "manuscript_variables.json", variables)
    stale_config = refresh_config_metadata(root, variables, dry_run=True)
    if stale_config:
        raise RuntimeError(
            "config.yaml disagrees with the measured variables after the "
            f"pre-scan refresh: {', '.join(stale_config)}"
        )
    written = write_resolved_manuscript(root, variables)
    uncited, undefined = audit_bibliography(root, written)
    policy = bibliography_policy(root)
    if undefined and policy["fail_on_missing"]:
        raise ValueError(
            "citations with no bibliography entry: " + ", ".join(undefined)
        )
    if uncited:
        message = "bibliography entries are never cited: " + ", ".join(uncited)
        if policy["fail_on_unused"]:
            raise ValueError(message)
        print(f"warning: {message}", file=sys.stderr)
    failed_groups = [
        result.name for result in verification if result.status != "passed"
    ]
    manifest = {
        "schema_version": RESEARCH_SCHEMA,
        "source_commit": inventory.commit,
        "dirty_file_count": inventory.dirty_file_count,
        "source_hash": inventory.source_hash,
        "verification_failures": failed_groups,
        "verification_source_commit": record.source_commit,
        "verification_source_hash": record.source_hash,
        "verification_measured_elsewhere": record.measured_elsewhere,
        "resolved_manuscript_files": [
            path.relative_to(root).as_posix() for path in written
        ],
        "figure_registry": "output/figures/figure_registry.json",
        "variables": "output/data/manuscript_variables.json",
        "verification": "output/data/research_verification.json",
    }
    _write_json(data_dir / "research_manifest.json", manifest)
    if publication and not verification:
        raise RuntimeError(
            "refusing to publish an empty evidence record: "
            f"{len(record.defined_groups)} "
            "verification command groups are defined and none ran"
        )
    if publication and failed_groups:
        raise RuntimeError(
            "refusing to publish a build whose verification record contains "
            "failures: " + ", ".join(failed_groups)
        )
    return manifest


def _published_verification_problems(
    root: Path, published: Mapping[str, Any]
) -> list[str]:
    """Return every way the published counts disagree with the stored record.

    The published summary and the record it summarises are two files, and
    nothing compared them.  That is how a bundle could ship an abstract
    reading "of the 7 command groups this build defines, 9 passed" beside a
    record holding eleven results: both files were internally well-formed and
    ``--check`` only ever looked at the source hash and four config-owned
    fields.

    The counts are recomputed here from the record on disk, at the record's
    own tier, and compared to what the bundle publishes.  A bundle whose
    denominator was taken at a different tier from the record it republishes
    is stale in exactly the sense this function exists to detect.
    """
    problems: list[str] = []
    numeric: dict[str, int] = {}
    for key in (
        "VERIFICATION_DEFINED_COUNT",
        "VERIFICATION_PASS_COUNT",
        "VERIFICATION_FAIL_COUNT",
        "VERIFICATION_UNRUN_COUNT",
    ):
        raw = published.get(key)
        if not isinstance(raw, str) or not raw.lstrip("-").isdigit():
            problems.append(f"published variables omit a usable {key}: {raw!r}")
            continue
        numeric[key] = int(raw)
    if len(numeric) == 4:
        total = (
            numeric["VERIFICATION_PASS_COUNT"]
            + numeric["VERIFICATION_FAIL_COUNT"]
            + numeric["VERIFICATION_UNRUN_COUNT"]
        )
        if total != numeric["VERIFICATION_DEFINED_COUNT"]:
            problems.append(
                "published verification counts do not partition the defined "
                f"groups: {numeric['VERIFICATION_PASS_COUNT']} passed + "
                f"{numeric['VERIFICATION_FAIL_COUNT']} failed + "
                f"{numeric['VERIFICATION_UNRUN_COUNT']} unrun = {total} "
                f"against {numeric['VERIFICATION_DEFINED_COUNT']} defined"
            )
    stored = _load_stored_verification(root)
    if stored is None:
        if numeric.get("VERIFICATION_PASS_COUNT") or numeric.get(
            "VERIFICATION_FAIL_COUNT"
        ):
            problems.append(
                "published variables report executed command groups but "
                f"{_verification_record_path(root).relative_to(root).as_posix()} "
                "holds no record"
            )
        return problems
    tier = stored.full_validation_requested
    summary, passed, failed, unrun = _verification_summary(
        stored.results, full_validation=tier
    )
    expected: tuple[tuple[str, str], ...] = (
        ("VERIFICATION_DEFINED_COUNT", str(len(stored.defined_groups))),
        ("VERIFICATION_PASS_COUNT", str(passed)),
        ("VERIFICATION_FAIL_COUNT", str(failed)),
        ("VERIFICATION_UNRUN_COUNT", str(unrun)),
        ("VERIFICATION_STATUS", summary),
        ("VERIFICATION_RECORD_TIER", "full-validation" if tier else "default"),
        ("VERIFICATION_RECORD_COMMIT", stored.source_commit),
        ("VERIFICATION_RECORD_SOURCE_HASH", stored.source_hash),
    )
    for key, want in expected:
        got = published.get(key)
        if got != want:
            problems.append(
                f"{key} disagrees with the stored verification record: "
                f"published {got!r}, record says {want!r}"
            )
    return problems


def check_published_artifacts(root: Path) -> tuple[str, ...]:
    """Return every reason the published artifacts no longer describe the tree.

    Three comparisons, all of which were missing.

    ``RESEARCH_SOURCE_HASH`` is computed on every run and published in four
    places, but nothing ever compared it to anything: a render that skipped
    regeneration would republish stale counts and exit 0.

    The published verification counts are compared to the record they
    summarise, at that record's own tier.  Without it a bundle could
    republish an eleven-result record under a seven-group denominator and
    exit 0.

    ``manuscript/config.yaml`` supplies the title page and is generator-owned.
    It is checked against the *published variables*, not against a fresh read
    of ``HEAD``: the file is tracked, so the commit that records it is always
    newer than the commit date it holds, and comparing to ``HEAD`` could never
    pass.  What must hold — and what the published PDF depends on — is that the
    title page and the evidence bundle shipped beside it describe the same
    build.

    Returns:
        Every reason the artifacts are stale.  An empty tuple is the only
        passing answer.
    """
    problems: list[str] = []
    variables_path = root / "output" / "data" / "manuscript_variables.json"
    if not variables_path.is_file():
        return (
            f"no published variables at {variables_path.relative_to(root).as_posix()}",
        )
    try:
        published = json.loads(variables_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"published variables are unreadable: {exc}",)
    measured = _source_hash(root)
    if published.get("RESEARCH_SOURCE_HASH") != measured:
        problems.append(
            "RESEARCH_SOURCE_HASH is stale: published "
            f"{published.get('RESEARCH_SOURCE_HASH')!r}, measured {measured!r}"
        )
    missing = [
        key for _prefix, key, _field in _CONFIG_OWNED_FIELDS if key not in published
    ]
    if missing:
        problems.append(
            "published variables omit config metadata: " + ", ".join(missing)
        )
    else:
        problems.extend(
            f"manuscript/config.yaml disagrees with the published build: {field}"
            for field in refresh_config_metadata(root, published, dry_run=True)
        )
    problems.extend(_published_verification_problems(root, published))
    return tuple(problems)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run strict research verification commands",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="also run the full unit, integration, performance, and H3 suites",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help=(
            "refuse to produce a build whose evidence record is empty; implies "
            "that verification must have been requested and produced results"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "write nothing; exit non-zero when the published source hash, the "
            "generator-owned config.yaml metadata, or the published "
            "verification counts no longer match the tree and the stored "
            "evidence record"
        ),
    )
    parser.add_argument(
        "--rerun-verification",
        action="store_true",
        help=(
            "run the verification commands even when the stored record already "
            "names this source hash, commit, and tier; reuse is the default so "
            "that a build which measures nothing cannot overwrite a record that "
            "still describes the tree"
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "generate from an unclean checkout; the build is stamped "
            "'<sha>-dirty' and the uncommitted entry count is published"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    if args.check:
        problems = check_published_artifacts(root)
        for problem in problems:
            print(f"stale manuscript artifact: {problem}", file=sys.stderr)
        if problems:
            return 1
        print("manuscript artifacts match the current tree")
        return 0
    try:
        manifest = generate(
            root,
            verify=args.verify or args.full_validation,
            full_validation=args.full_validation,
            allow_dirty=args.allow_dirty,
            publication=args.publication,
            reuse_verification=not args.rerun_verification,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"research artifact generation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    failures = manifest.get("verification_failures") or ()
    if failures:
        # The artifacts are written and publish the failure honestly; the
        # non-zero exit is the CI signal, not a refusal to build. Only
        # ``--publication`` refuses.
        print(
            "verification command groups did not pass: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
