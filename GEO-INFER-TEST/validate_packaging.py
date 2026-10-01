#!/usr/bin/env python3
"""
Packaging-configuration validation for the GEO-INFER monorepo platform.

This validator enforces the unified multi-package wheel-release contract
across all ``GEO-INFER-*`` modules:

- PyPI distribution namespace: every ``[project].name`` must use the
  ``geo-infer-*`` distribution prefix and normalize to a lowercase package
  directory under ``src/``.
- Version uniformity: member pyprojects must agree on ``[project].version``
  (warn on outliers, error under ``--strict``). Known deviations are listed
  in ``KNOWN_VERSION_DEVIATIONS`` and surfaced as diagnostics only — the
  promotion decision belongs to the release process (ledger row REL-01).
- Citation version: ``CITATION.cff`` must cite the fleet-majority member
  ``[project].version`` (warn on mismatch, error under ``--strict``), so a
  release cannot publish a citation for a stale version.
- Classifier consistency: modules declaring a ``Development Status``
  classifier must agree on the fleet mode; the root framework distribution's
  status is expected to differ and is surfaced as a diagnostic.
- Package-data inclusion: every wheel must ship its YAML/JSON/MD/TXT
  configuration resources so runtime config discovery works from an
  installed wheel without relying on repository-local ``config/`` roots.
  Declared globs are expanded against the package tree on disk, so a data
  file matching no glob (a resource that would silently miss the wheel) is
  an error, and per-module deviations from the canonical pattern set are
  surfaced.
- Out-of-package source traversal is reported as a diagnostic so authors can
  migrate ``Path(__file__).parent...`` config lookups to an installed-wheel
  safe discovery mechanism when publishing wheels. Modules migrated to
  importlib.resources (``MIGRATED_SOURCE_TRAVERSAL_MODULES``) fail the run
  outright if a parent climb reappears.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
from collections import Counter
from pathlib import Path
import sys

from _validator_common import (
    ContractReport,
    STDLIB_REQUIREMENT_NAMES,
    discover_module_dirs,
    distribution_name,
    internal_requirement_names,
    package_name_from_distribution,
    pyproject_dependency_names,
    pyproject_group_names,
    pyproject_optional_names,
    read_pyproject as parse_pyproject,
    read_toml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Distribution namespace used for every published PyPI package.
DISTRIBUTION_PREFIX = "geo-infer-"

# Package-data resource globs expected in [tool.setuptools.package-data].
PACKAGE_DATA_RESOURCES = ("*.yaml", "*.yml", "*.json", "*.md", "*.txt")

# CI-03: the citation file pinned to the fleet-majority member version.
CITATION_CFF_PATH = REPO_ROOT / "CITATION.cff"

PROJECT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

DEVELOPMENT_STATUS_PATTERN = re.compile(r"^Development Status :: (\d) - ")

# Member versions that intentionally deviate from the fleet majority. Keyed
# by module directory name. The release gate (REL-01) owns promoting these;
# the uniformity check reports them as diagnostics so --strict stays
# meaningful for NEW outliers.
KNOWN_VERSION_DEVIATIONS: dict[str, str] = {}

# Modules whose out-of-package ``Path(__file__)`` resource climbs have been
# migrated to importlib.resources discovery under the package tree
# (SCOPE-2026-09-11 GS-023). For these modules a parent climb is an ERROR,
# not a diagnostic: the wheel contract is enforced, not merely reported.
MIGRATED_SOURCE_TRAVERSAL_MODULES = frozenset(
    {
        "GEO-INFER-ACT",
        "GEO-INFER-EXAMPLES",
        "GEO-INFER-GIT",
        "GEO-INFER-HEALTH",
        "GEO-INFER-INTRA",
        "GEO-INFER-OPS",
        "GEO-INFER-PLACE",
        "GEO-INFER-RISK",
        "GEO-INFER-SEC",
        "GEO-INFER-SPACE",
    }
)

_PACKAGE_DATA_EXCLUDED_DIRS = ("__pycache__",)


def module_dirs() -> list[Path]:
    """Return ``GEO-INFER-*`` module directories in stable order."""
    return discover_module_dirs(REPO_ROOT)


def wheel_metadata_name(distribution: str) -> str:
    """The normalized package-name prefix used in built wheel filenames."""
    return package_name_from_distribution(distribution)


def valid_distribution_namespace(name: str) -> bool:
    """True when a distribution name conforms to the shared PyPI namespace."""
    if not name.startswith(DISTRIBUTION_PREFIX):
        return False
    suffix = name[len(DISTRIBUTION_PREFIX) :]
    return bool(PROJECT_NAME_PATTERN.fullmatch(suffix))


def wheel_filename_is_valid(built_name: str, expected_distribution: str) -> bool:
    """True when a built wheel filename belongs to the expected distribution."""
    normalized = wheel_metadata_name(expected_distribution)
    return built_name.startswith(f"{normalized}-")


def _package_data_patterns(pyproject: dict) -> list[str] | None:
    """Return the union of [tool.setuptools.package-data] globs, or None."""
    table = pyproject.get("tool", {}).get("setuptools", {}).get("package-data")
    if not isinstance(table, dict) or not table:
        return None
    patterns: list[str] = []
    for patterns_by_key in table.values():
        if not isinstance(patterns_by_key, list):
            continue
        for pattern in patterns_by_key:
            if isinstance(pattern, str) and pattern not in patterns:
                patterns.append(pattern)
    return patterns


def _package_data_files(package_dir: Path):
    """Yield non-Python data files under a package directory."""
    for path in sorted(package_dir.rglob("*")):
        if path.is_dir():
            continue
        parts = path.relative_to(package_dir).parts
        if any(
            part.startswith(".")
            or part in _PACKAGE_DATA_EXCLUDED_DIRS
            or part.endswith(".egg-info")
            for part in parts[:-1]
        ):
            continue
        name = path.name
        if name.endswith((".py", ".pyc")) or name.startswith("."):
            continue
        yield path.relative_to(package_dir).as_posix()


def validate_package_data(
    module_dir: Path, pyproject: dict, report: ContractReport
) -> None:
    """Validate [tool.setuptools.package-data] against the package tree.

    Missing canonical patterns are warnings (the wheel will not ship those
    resource classes); extra patterns beyond the canonical set are
    diagnostics (intentional additions, e.g. PLACE's ``*.geojson``); a data
    file on disk matching no declared glob is an error, because it would
    silently miss the built wheel.
    """
    label = module_dir.name
    patterns = _package_data_patterns(pyproject)
    if patterns is None:
        report.warning(
            f"{label}: missing [tool.setuptools.package-data]; wheel will not "
            "ship YAML/JSON configuration resources"
        )
        return

    declared = set(patterns)
    for resource in PACKAGE_DATA_RESOURCES:
        if resource not in declared:
            report.warning(f"{label}: package-data does not include {resource}")
    for pattern in sorted(declared - set(PACKAGE_DATA_RESOURCES)):
        report.diagnostic(
            f"{label}: package-data includes {pattern} beyond the canonical "
            f"resource set {list(PACKAGE_DATA_RESOURCES)}"
        )

    distribution = distribution_name(pyproject) or ""
    package_dir = module_dir / "src" / package_name_from_distribution(distribution)
    if not package_dir.is_dir():
        return
    for relative in _package_data_files(package_dir):
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in declared):
            report.error(
                f"{label}: package data file {relative} matches no "
                "[tool.setuptools.package-data] glob; it would be omitted "
                "from the built wheel"
            )


def _development_status(pyproject: dict) -> str | None:
    """Return the ``Development Status :: X - Y`` classifier, or None."""
    classifiers = pyproject.get("project", {}).get("classifiers") or []
    for classifier in classifiers:
        if isinstance(classifier, str) and DEVELOPMENT_STATUS_PATTERN.match(classifier):
            return classifier.split("::")[-1].strip()
    return None


def validate_version_uniformity(
    inventories: list[tuple[str, dict]], report: ContractReport
) -> None:
    """Member pyprojects must agree on ``[project].version``."""
    versions = {
        module_name: version
        for module_name, pyproject in inventories
        if isinstance(version := pyproject.get("project", {}).get("version"), str)
        and version
    }
    counts = Counter(versions.values())
    if len(counts) < 2:
        return
    majority_version, _ = counts.most_common(1)[0]
    for module_name, version in sorted(versions.items()):
        if version == majority_version:
            continue
        if KNOWN_VERSION_DEVIATIONS.get(module_name) == version:
            report.diagnostic(
                f"{module_name}: version {version!r} deviates from the fleet "
                f"majority {majority_version!r} (known deviation; the release "
                "gate owns promotion)"
            )
        else:
            report.warning(
                f"{module_name}: version {version!r} deviates from the fleet "
                f"majority {majority_version!r}"
            )


_CITATION_VERSION_RE = re.compile(
    r"""^version:[ \t]*["']?([^"'\s]+)["']?[ \t]*$""", re.MULTILINE
)


def citation_cff_version(citation_path: Path) -> str | None:
    """Return the top-level ``version`` field of a CITATION.cff, or None."""
    if not citation_path.is_file():
        return None
    match = _CITATION_VERSION_RE.search(
        citation_path.read_text(encoding="utf-8", errors="ignore")
    )
    if match is None:
        return None
    return match.group(1)


def validate_citation_version(
    inventories: list[tuple[str, dict]],
    report: ContractReport,
    citation_path: Path | None = None,
) -> None:
    """CITATION.cff must cite the fleet-majority member version (CI-03).

    Nothing else pins the citable version to the fleet: a stale CITATION
    version could ship next to wheels carrying another. Semantics match
    member version uniformity — a mismatch is a warning here and an error
    under ``--strict`` promotion; a missing file or version field warns so
    the release gate cannot silently skip the citation contract.
    """
    versions = {
        module_name: version
        for module_name, pyproject in inventories
        if isinstance(version := pyproject.get("project", {}).get("version"), str)
        and version
    }
    if not versions:
        report.diagnostic(
            "CITATION.cff: no member [project].version declared to pin against"
        )
        return
    majority_version, _ = Counter(versions.values()).most_common(1)[0]
    path = citation_path if citation_path is not None else CITATION_CFF_PATH
    cited = citation_cff_version(path)
    if cited is None:
        report.warning(
            f"{path.name}: missing or has no top-level version field to cite the fleet"
        )
        return
    if cited != majority_version:
        report.warning(
            f"{path.name}: version {cited!r} deviates from the fleet majority "
            f"{majority_version!r}"
        )


def validate_classifier_consistency(
    inventories: list[tuple[str, dict]], report: ContractReport
) -> None:
    """Warn on module Development Status outliers; root asymmetry is a note."""
    statuses = {
        module_name: status
        for module_name, pyproject in inventories
        if (status := _development_status(pyproject)) is not None
    }
    undeclared = sorted(
        module_name for module_name, _ in inventories if module_name not in statuses
    )
    for module_name in undeclared:
        report.diagnostic(f"{module_name}: no Development Status classifier declared")
    counts = Counter(statuses.values())
    if not counts:
        return
    majority_status, _ = counts.most_common(1)[0]
    for module_name, status in sorted(statuses.items()):
        if status != majority_status:
            report.warning(
                f"{module_name}: Development Status {status!r} deviates from "
                f"the fleet majority {majority_status!r}"
            )
    root_pyproject = REPO_ROOT / "pyproject.toml"
    root_status = (
        _development_status(read_toml(root_pyproject))
        if root_pyproject.is_file()
        else None
    )
    if root_status is not None and root_status != majority_status:
        report.diagnostic(
            f"root pyproject: Development Status {root_status!r} differs from "
            f"the member fleet {majority_status!r} (framework meta-package "
            "asymmetry; the release gate owns alignment)"
        )


trove_top_level_categories = {
    "Development Status",
    "Environment",
    "Framework",
    "Intended Audience",
    "Natural Language",
    "Operating System",
    "Programming Language",
    "Topic",
    "Typing",
}


def validate_classifier_validity(
    inventories: list[tuple[str, dict]], report: ContractReport
) -> None:
    """Reject trove classifiers PyPI would refuse on upload.

    The fleet declares licenses via PEP 639 SPDX expressions in
    [project].license, so any ``License ::`` classifier is deprecated
    alongside a license expression and PyPI rejects the combination;
    every License classifier is an error regardless of children.
    """
    for module_name, pyproject in inventories:
        classifiers = pyproject.get("project", {}).get("classifiers") or []
        for classifier in classifiers:
            top_level = classifier.split(" :: ")[0]
            if top_level == "License":
                report.error(
                    f"{module_name}: deprecated trove classifier {classifier!r} "
                    "(License :: classifiers are rejected by PyPI alongside a "
                    "PEP 639 license expression; the license field carries the "
                    "license)"
                )
            elif top_level not in trove_top_level_categories:
                report.error(
                    f"{module_name}: invalid trove classifier {classifier!r} "
                    f"(unknown top-level category {top_level!r})"
                )


def validate_module(module_dir: Path, pyproject: dict, report: ContractReport) -> None:
    """Validate namespace and package-data metadata for one module."""
    label = module_dir.name
    distribution = distribution_name(pyproject)
    if not distribution:
        report.errors.append(f"{label}: missing [project].name")
        return

    if not valid_distribution_namespace(distribution):
        report.errors.append(
            f"{label}: distribution name {distribution!r} must use the "
            f"{DISTRIBUTION_PREFIX}* namespace"
        )

    package = package_name_from_distribution(distribution)
    if package != distribution.lower().replace("-", "_"):
        report.errors.append(
            f"{label}: project name must normalize to lowercase Python package"
        )

    src_dir = module_dir / "src"
    package_dir = src_dir / package
    if src_dir.is_dir() and not package_dir.is_dir():
        report.errors.append(f"{label}: expected package directory src/{package}")

    validate_package_data(module_dir, pyproject, report)


def validate_source_traversal(module_dir: Path, report: ContractReport) -> None:
    """Flag source files that reach outside the module package for resources."""
    src_dir = module_dir / "src"
    if not src_dir.is_dir():
        return
    for path in src_dir.rglob("*.py"):
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "__file__" not in text:
            continue
        # Only a parent chain attached to a ``__file__`` expression is a
        # climb: one ``.parent`` stays inside the package (wheel-safe), and
        # ``.parent`` on an unrelated Path (e.g. a log dir) is not a climb.
        climbs = 0
        for line in text.splitlines():
            if "__file__" not in line:
                continue
            chains = re.findall(r"\.parents\[\s*-?\d+\s*\]", line)
            chains += re.findall(r"parents\[\s*\d+\s*\]", line)
            chains += re.findall(r"\.parent(?:\.parent)*", line)
            for chain in chains:
                if chain.endswith("]") or "]" in chain:
                    climbs = max(climbs, int(re.search(r"-?\d+", chain).group()))
                else:
                    climbs = max(climbs, len(chain.split(".")) - 1)
        if climbs >= 2:
            rel = path.relative_to(src_dir)
            finding = f"{module_dir.name}/{rel}: climbs parent dirs from __file__"
            if module_dir.name in MIGRATED_SOURCE_TRAVERSAL_MODULES:
                report.errors.append(finding)
            else:
                report.diagnostics.append(finding)


# Third-party import roots that map to a different distribution name.
IMPORT_ROOT_ALIASES = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "Bio": "biopython",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "git": "gitpython",
    "jwt": "pyjwt",
    "jose": "python-jose",
    "strawberry": "strawberry-graphql",
    "_pytest": "pytest",
    "pymdp": "inferactively-pymdp",
}

# Complete standard-library root set for the running interpreter; the
# hand-maintained STDLIB_REQUIREMENT_NAMES stays as a belt-and-braces floor.
_STDLIB_ROOTS = frozenset(getattr(sys, "stdlib_module_names", ())) | frozenset(
    STDLIB_REQUIREMENT_NAMES
)


def _handles_import_error(handler: ast.ExceptHandler) -> bool:
    """True when the handler catches ImportError (bare, tuple, or catch-all)."""
    if handler.type is None:
        return True
    names: list[str] = []
    if isinstance(handler.type, ast.Name):
        names = [handler.type.id]
    elif isinstance(handler.type, ast.Tuple):
        names = [elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)]
    return any(
        name in ("ImportError", "ModuleNotFoundError", "Exception") for name in names
    )


def _is_type_checking(test: ast.expr) -> bool:
    """True for the ``if TYPE_CHECKING:`` guard pattern."""
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _top_level_import_roots(tree: ast.Module) -> set:
    """Import roots that execute at module import time.

    Walks module-level statements directly, so a root imported BOTH at the
    top level and inside a guard is still reported. Optional-import sites
    are excluded: imports inside functions only run when the code path is
    exercised, imports under ``if TYPE_CHECKING:`` never run at runtime, and
    imports inside a try block whose handlers catch ImportError (or bare/
    Exception) degrade gracefully — none of these can break ``import
    <package>`` on a clean install. A try block WITHOUT such a handler is
    descended into, because its body still executes at import time.
    """
    roots: set = set()

    def record(node) -> None:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])

    def walk_body(body) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                record(node)
            elif isinstance(node, ast.Try):
                if not any(_handles_import_error(handler) for handler in node.handlers):
                    walk_body(node.body)
            elif isinstance(node, ast.If):
                if _is_type_checking(node.test):
                    continue
                walk_body(node.body)
                walk_body(node.orelse)

    walk_body(tree.body)
    return roots


def module_import_roots(module_dir: Path) -> dict:
    """Top-level (import-time) import roots per module, via ast walk.

    Returns ``{module_dir_name: {import_root, ...}}`` for modules with a
    ``src/<package>/`` layout; guarded and function-local imports are
    excluded because they cannot break a clean install.
    """
    src_dir = module_dir / "src"
    result: dict = {}
    if not src_dir.is_dir():
        return result
    packages = sorted(p for p in src_dir.iterdir() if (p / "__init__.py").is_file())
    for package in packages:
        roots: set = set()
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            roots.update(_top_level_import_roots(tree))
        result[module_dir.name] = roots
    return result


def validate_import_parity(
    module_dir: Path, pyproject: dict, report: ContractReport
) -> None:
    """Compare top-level third-party import roots against declared dependencies.

    An unguarded module-level ``import x`` must be backed by a declared
    dependency (runtime or optional extra), otherwise a clean wheel install
    raises ModuleNotFoundError on ``import <package>``. Standard-library
    roots and this repo's own ``geo_infer_*`` packages are exempt; the
    rest map through IMPORT_ROOT_ALIASES to distribution names.
    """
    declared = {
        name.rstrip(".")
        for name in (
            pyproject_dependency_names(pyproject) | pyproject_optional_names(pyproject)
        )
    }
    declared |= internal_requirement_names(REPO_ROOT)
    for roots in module_import_roots(module_dir).values():
        for root in sorted(roots):
            if root in _STDLIB_ROOTS:
                continue
            if root == "src":
                continue
            dist = IMPORT_ROOT_ALIASES.get(root, root.lower().replace("_", "-"))
            if dist in declared:
                continue
            rel = None
            src_dir = module_dir / "src"
            for path in sorted((src_dir).rglob("*.py")):
                text = path.read_text(encoding="utf-8", errors="ignore")
                if re.search(
                    rf"^import {re.escape(root)}\b", text, re.MULTILINE
                ) or re.search(rf"^from {re.escape(root)}\b", text, re.MULTILINE):
                    rel = path.relative_to(module_dir)
                    break
            report.errors.append(
                f"{module_dir.name}: top-level import '{root}' is not declared in "
                f"pyproject dependencies{f' (first: {rel})' if rel else ''}"
            )


# Call targets that import the module named by their first argument; a
# string literal there is a dynamic test import that counts toward parity.
_DYNAMIC_IMPORT_CALLS = frozenset(
    {"import_module", "importorskip", "__import__", "find_spec"}
)

_DOTTED_MODULE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _dynamic_import_call(node: ast.Call) -> bool:
    """True for ``import_module(...)``-style calls, by bare or attribute name."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name in _DYNAMIC_IMPORT_CALLS and bool(node.args)


def _literal_import_root(node: ast.Call) -> str | None:
    """Root of a dynamic import call whose first argument is a string literal."""
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    if not _DOTTED_MODULE_PATH.fullmatch(first.value):
        return None
    return first.value.split(".")[0]


def all_import_roots(
    tree: ast.Module, workspace_packages: frozenset[str] = frozenset()
) -> set[str]:
    """Every absolute import root in a module, wherever it appears.

    Unlike ``_top_level_import_roots`` this includes function-local imports,
    imports under try/except ImportError guards, ``if`` and ``if
    TYPE_CHECKING:`` branches, and string-literal dynamic imports
    (``importlib.import_module``, ``pytest.importorskip``, ``__import__``,
    ``importlib.util.find_spec``): a test suite must declare everything it
    can import. When the module also performs a dynamic import with a
    computed argument (``import_module(name)`` over a parametrized list),
    every dotted-path string literal rooted at a ``workspace_packages`` name
    counts as imported too, because the call's argument cannot be resolved
    statically and workspace package names are unambiguous.
    """
    roots: set[str] = set()
    computed_dynamic_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and _dynamic_import_call(node):
            root = _literal_import_root(node)
            if root is None:
                computed_dynamic_import = True
            else:
                roots.add(root)
    if computed_dynamic_import and workspace_packages:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _DOTTED_MODULE_PATH.fullmatch(node.value)
                and (root := node.value.split(".")[0]) in workspace_packages
            ):
                roots.add(root)
    return roots


def _local_import_roots(module_dir: Path, tests_dir: Path) -> set[str]:
    """Import roots that resolve to files inside the module itself.

    Covers the module's own ``src/`` packages, the ``src``/``tests`` roots,
    module-root packages and scripts (e.g. TEST's ``demo`` package and
    ``run_unified_tests.py``), and test helper modules or packages under
    ``tests/`` that pytest's rootdir-relative ``sys.path`` insertion makes
    importable by bare name. Plain directories without ``__init__.py`` are
    not import roots and are not exempted.
    """
    local: set[str] = {"src", "tests"}
    src_dir = module_dir / "src"
    if src_dir.is_dir():
        local.update(path.name for path in src_dir.iterdir() if path.is_dir())
    candidates = [*module_dir.iterdir(), *tests_dir.rglob("*")]
    for path in candidates:
        if "__pycache__" in path.parts:
            continue
        if path.is_dir() and (path / "__init__.py").is_file():
            local.add(path.name)
        elif path.is_file() and path.suffix == ".py":
            local.add(path.stem)
    return local


def workspace_package_names(repo_root: Path) -> frozenset[str]:
    """Importable package names of every workspace member (``geo_infer_*``)."""
    return frozenset(
        package_name_from_distribution(name)
        for name in internal_requirement_names(repo_root)
    )


def collect_test_import_roots(
    module_dir: Path, workspace_packages: frozenset[str] = frozenset()
) -> dict[str, Path]:
    """Map each absolute import root used under ``tests/`` to its first file."""
    tests_dir = module_dir / "tests"
    roots: dict[str, Path] = {}
    if not tests_dir.is_dir():
        return roots
    for path in sorted(tests_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for root in sorted(all_import_roots(tree, workspace_packages)):
            roots.setdefault(root, path.relative_to(module_dir))
    return roots


def validate_test_import_parity(
    module_dir: Path,
    pyproject: dict,
    report: ContractReport,
    workspace_packages: frozenset[str] | None = None,
) -> None:
    """Every third-party or sibling import under ``tests/`` must be declared.

    A test import is satisfied by a runtime dependency, any optional extra,
    or any PEP 735 ``[dependency-groups]`` group (convention: ``test``).
    Sibling ``geo_infer_*`` packages are NOT exempt here, unlike the runtime
    import check: a test suite importing a sibling must declare it (with a
    ``[tool.uv.sources]`` workspace source) or it only resolves through a
    workspace-wide sync. Standard-library roots and roots resolving to the
    module's own files are exempt; guarded and function-local imports count.
    """
    tests_dir = module_dir / "tests"
    if not tests_dir.is_dir():
        return
    if workspace_packages is None:
        workspace_packages = workspace_package_names(REPO_ROOT)
    declared = {
        name.rstrip(".")
        for name in (
            pyproject_dependency_names(pyproject)
            | pyproject_optional_names(pyproject)
            | pyproject_group_names(pyproject)
        )
    }
    local = _local_import_roots(module_dir, tests_dir)
    roots = collect_test_import_roots(module_dir, workspace_packages)
    for root, first in sorted(roots.items()):
        if root not in IMPORT_ROOT_ALIASES and (root in _STDLIB_ROOTS or root in local):
            continue
        dist = IMPORT_ROOT_ALIASES.get(root, root.lower().replace("_", "-"))
        if dist in declared:
            continue
        report.error(
            f"{module_dir.name}: test import '{root}' (distribution {dist!r}) is "
            "not declared in dependencies, an extra, or a dependency group "
            f"(first: {first})"
        )


def validate_all(target_dirs: list[Path] | None = None) -> ContractReport:
    report = ContractReport()
    if target_dirs is None:
        target_dirs = module_dirs()
    inventories: list[tuple[str, dict]] = []
    workspace_packages = workspace_package_names(REPO_ROOT)
    for module_dir in target_dirs:
        pyproject = parse_pyproject(module_dir)
        if not pyproject:
            report.errors.append(f"{module_dir.name}: invalid/missing pyproject")
            continue
        inventories.append((module_dir.name, pyproject))
        validate_module(module_dir, pyproject, report)
        validate_import_parity(module_dir, pyproject, report)
        validate_test_import_parity(module_dir, pyproject, report, workspace_packages)
        validate_source_traversal(module_dir, report)
    validate_version_uniformity(inventories, report)
    validate_citation_version(inventories, report)
    validate_classifier_consistency(inventories, report)
    validate_classifier_validity(inventories, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate GEO-INFER packaging configuration"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote packaging warnings to errors.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show out-of-package __file__ traversal and metadata diagnostics.",
    )
    args = parser.parse_args()

    report = validate_all()
    if args.strict:
        report.errors.extend(report.warnings)
        report.warnings = []
    print(f"Modules checked: {len(module_dirs())}")
    print(f"Errors: {len(report.errors)}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"Warnings: {len(report.warnings)}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if args.diagnostics:
        for diag in report.diagnostics:
            print(f"DIAG: {diag}")

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
