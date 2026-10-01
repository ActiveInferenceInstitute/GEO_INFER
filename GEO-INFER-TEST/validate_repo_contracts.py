#!/usr/bin/env python3
"""
Validate repo-wide GEO-INFER structural contracts.

The default mode fails on structural drift that should never be tolerated:
module inventory, local signposting, package casing, pyproject-only packaging
(no setup.py / setup.cfg / requirements.txt mirrors), and pyproject
package-name sanity. Source-language debt is reported by default and
can be made fatal with ``--strict-source-language``. Import-smoke failures are
advisory warnings by default and can be made fatal with
``--strict-import-smoke``.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import importlib
import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from import_probe import run_import_probe
from _validator_common import (
    ContractReport,
    STDLIB_REQUIREMENT_NAMES,
    discover_module_dirs,
    expected_package_name,
    internal_requirement_names,
    normalize_dependency_name,
    read_toml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MODULE_COUNT = 45
SIGNPOST_FILES = ("README.md", "AGENTS.md", "SKILL.md")
MIN_TEST_FILES_PER_MODULE = 4
CANONICAL_UV_SYNC_COMMAND = "uv sync --all-packages --all-extras"
CANONICAL_UV_DOC_FILES = (
    "README.md",
    "AGENTS.md",
    "SKILL.md",
    "GEO-INFER-TEST/README.md",
    "GEO-INFER-TEST/AGENTS.md",
)
SOURCE_LANGUAGE_PATTERN = re.compile(
    r"\b(mock|stub|fake|placeholder)\b|NotImplementedError",
    re.IGNORECASE,
)
ROOT_GENERATED_ARTIFACT_PATHS = (
    "logs",
    ".geo-infer-test-results",
    "test_output",
    "output",
    "outputs",
    "visualizations_output",
)
TASK_MARKER_PATTERN = re.compile(
    # Matches bare task-marker tokens, with two deliberate allowances so the
    # module-root script scan below can include the validator scripts
    # themselves without self-flagging: a token adjacent to a "|" belongs to
    # this or another validator's own token-alternation definition, and a
    # token immediately followed by ".md" names the root TODO.md ledger
    # file rather than planning work.
    r"(?<![|\w])(TODO|FIXME|XXX|HACK|TBD)\b(?!\||\.md)"
)
TASK_MARKER_SCAN_GLOBS = (
    "GEO-INFER-*/src/**/*.py",
    "GEO-INFER-*/tests/**/*.py",
    # GS19-85: module-root scripts (validate_*.py, run_unified_tests.py,
    # measure_module_coverage.py, ...) are module surface too; markers there
    # were invisible to this gate because the scan covered only src/ and
    # tests/.
    "GEO-INFER-*/*.py",
)
# User-facing example code keeps a softer contract: markers are surfaced as
# warnings so example authors still see them, without failing CI on demos.
TASK_MARKER_EXAMPLE_SCAN_GLOBS = ("GEO-INFER-*/examples/**/*.py",)
PYTHON_VERSION_PIN_PATTERN = re.compile(r"^3\.(11|12)(?:\.\d+)?$")
LOGGING_BASIC_CONFIG_ATTR = "basicConfig"
SOURCE_LANGUAGE_ALLOWLIST = (
    "placeholder=",
    "Subclasses must implement",
    "Must be implemented by subclasses",
    "If not implemented in subclass",
    "If requested optimization is not supported",
    "abstract",
    "not supported",
    "not available",
    "for testing",
    "fake data",
)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
LEGACY_PYTHON_METADATA_PATTERN = re.compile(
    r"Programming Language :: Python :: 3\.(8|9|10)"
    r"|python_requires\s*=\s*[\"']>=3\.(8|9|10)[\"']"
    r"|requires-python\s*=\s*[\"']>=3\.(8|9|10)[\"']"
)
LEGACY_H3_PATTERN = re.compile(r"\bh3\s*>=\s*(?:3\.|4\.0\.0)", re.IGNORECASE)
LEGACY_PYMDP_RUNTIME_IMPORTS = ("pymdp.control", "pymdp.inference")
RUFF_TARGET_VERSION_MINIMUM = 11
# pyproject.toml + uv.lock are the only packaging sources of truth; these
# module-root files are retired mirrors and must not reappear.
RETIRED_PACKAGING_FILES = ("setup.py", "setup.cfg", "requirements.txt")
RETIRED_TOOL_SECTIONS = ("black", "isort", "flake8", "pydocstyle")

# Internal distributions are derived from the workspace tree (each member's
# geo-infer-* distribution name) instead of a hand-maintained allowlist; see
# _validator_common.internal_requirement_names for the derivation rule and
# its documented limits.
INTERNAL_REQUIREMENT_NAMES = internal_requirement_names(REPO_ROOT)


def find_module_dirs() -> list[Path]:
    return discover_module_dirs(REPO_ROOT)


def parse_pyproject(module_dir: Path, report: ContractReport) -> dict:
    pyproject = module_dir / "pyproject.toml"
    if not pyproject.exists():
        report.error(f"{module_dir.name}: missing pyproject.toml")
        return {}
    try:
        return read_toml(pyproject)
    except tomllib.TOMLDecodeError as exc:
        report.error(f"{module_dir.name}: invalid pyproject.toml: {exc}")
        return {}


def validate_inventory(module_dirs: list[Path], report: ContractReport) -> None:
    if len(module_dirs) != EXPECTED_MODULE_COUNT:
        report.error(
            f"Expected {EXPECTED_MODULE_COUNT} GEO-INFER-* modules, found {len(module_dirs)}"
        )


def validate_signposting(module_dirs: list[Path], report: ContractReport) -> None:
    for module_dir in module_dirs:
        for filename in SIGNPOST_FILES:
            if not (module_dir / filename).exists():
                report.error(f"{module_dir.name}: missing {filename}")


def validate_package_casing(module_dirs: list[Path], report: ContractReport) -> None:
    uppercase_package_dirs: list[str] = []
    for module_dir in module_dirs:
        for path in (module_dir / "src").glob("geo_infer_*"):
            if path.is_dir() and any(char.isupper() for char in path.name):
                uppercase_package_dirs.append(str(path.relative_to(REPO_ROOT)))
    if uppercase_package_dirs:
        for path in uppercase_package_dirs:
            report.error(f"Uppercase Python package directory: {path}")


def validate_pyproject_packages(
    module_dirs: list[Path], report: ContractReport
) -> None:
    for module_dir in module_dirs:
        pyproject = parse_pyproject(module_dir, report)
        package_name = expected_package_name(pyproject)
        if package_name is None:
            report.error(f"{module_dir.name}: missing [project].name")
            continue

        if package_name.lower() != package_name:
            report.error(f"{module_dir.name}: project name must normalize to lowercase")

        src_dir = module_dir / "src"
        package_dir = src_dir / package_name
        if src_dir.exists() and not package_dir.exists():
            report.error(
                f"{module_dir.name}: expected package directory {package_dir.relative_to(REPO_ROOT)}"
            )

        dependencies = pyproject.get("project", {}).get("dependencies", [])
        if dependencies is not None and not isinstance(dependencies, list):
            report.error(f"{module_dir.name}: project.dependencies must be a list")


def validate_uv_environment(report: ContractReport) -> None:
    """Validate the root uv workspace and Python version pin."""
    root_pyproject = REPO_ROOT / "pyproject.toml"
    if not root_pyproject.exists():
        report.error("Root pyproject.toml is required for uv workspace resolution")
        return

    try:
        root_config = tomllib.loads(root_pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        report.error(f"Root pyproject.toml is invalid: {exc}")
        return

    requires_python = root_config.get("project", {}).get("requires-python")
    if requires_python != ">=3.11":
        report.error(
            "Root pyproject.toml must declare project.requires-python = '>=3.11'"
        )

    workspace_members = (
        root_config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members")
    )
    if workspace_members != ["GEO-INFER-*"]:
        report.error("Root [tool.uv.workspace].members must be ['GEO-INFER-*']")

    if not (REPO_ROOT / "uv.lock").exists():
        report.error("Root uv.lock is required; do not rely on module-local locks")

    python_version_file = REPO_ROOT / ".python-version"
    if not python_version_file.exists():
        report.error("Root .python-version is required for reproducible uv runs")
        return
    python_version = python_version_file.read_text(encoding="utf-8").strip()
    if not PYTHON_VERSION_PIN_PATTERN.match(python_version):
        report.error(
            f".python-version must pin Python 3.11 or 3.12, got {python_version!r}"
        )


def validate_uv_setup_documentation(report: ContractReport) -> None:
    """Ensure signposts advertise the workspace-wide uv sync command."""
    for relative_path in CANONICAL_UV_DOC_FILES:
        doc_path = REPO_ROOT / relative_path
        if not doc_path.exists():
            report.error(f"{relative_path}: missing uv setup signpost")
            continue
        doc_text = doc_path.read_text(encoding="utf-8", errors="ignore")
        if CANONICAL_UV_SYNC_COMMAND not in doc_text:
            report.error(
                f"{relative_path}: must document `{CANONICAL_UV_SYNC_COMMAND}`"
            )


def validate_test_inventory(module_dirs: list[Path], report: ContractReport) -> None:
    """Ensure every module keeps a minimally useful local pytest surface."""
    for module_dir in module_dirs:
        tests_dir = module_dir / "tests"
        if not tests_dir.exists():
            report.error(f"{module_dir.name}: missing tests/")
            continue
        test_files = {
            *tests_dir.rglob("test_*.py"),
            *tests_dir.rglob("*_test.py"),
        }
        if len(test_files) < MIN_TEST_FILES_PER_MODULE:
            report.error(
                f"{module_dir.name}: expected at least "
                f"{MIN_TEST_FILES_PER_MODULE} test files, found {len(test_files)}"
            )


def scan_task_marker_files(scan_globs: tuple[str, ...]) -> list[str]:
    """Return ``file:line: text`` hits for task markers below the glob roots."""
    hits: list[str] = []
    scan_files = {
        path
        for pattern in scan_globs
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    }
    for source_file in sorted(scan_files):
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if TASK_MARKER_PATTERN.search(line):
                hits.append(
                    f"{source_file.relative_to(REPO_ROOT).as_posix()}:{lineno}: {line.strip()}"
                )
    return hits


def validate_module_task_markers(report: ContractReport) -> None:
    """Keep actionable planning markers out of source, tests, and module
    root scripts."""
    hits = scan_task_marker_files(TASK_MARKER_SCAN_GLOBS)
    if hits:
        report.error(
            "Module-local task markers found; track planned work in root TODO.md "
            "or issues. First hits: " + "; ".join(hits[:8])
        )

    example_hits = scan_task_marker_files(TASK_MARKER_EXAMPLE_SCAN_GLOBS)
    if example_hits:
        report.warning(
            "Task markers found in user-facing examples/ directories; clean these "
            "before publishing the examples. First hits: " + "; ".join(example_hits[:8])
        )


def validate_logging_configuration(report: ContractReport) -> None:
    """Library imports should not configure process-wide logging."""
    hits: list[str] = []
    for source_file in sorted(REPO_ROOT.glob("GEO-INFER-*/src/**/*.py")):
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text, filename=str(source_file))
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and is_main_guard(node.test):
                continue
            for child in ast.walk(node):
                if is_logging_basic_config_call(child):
                    line = text.splitlines()[child.lineno - 1].strip()
                    hits.append(
                        f"{source_file.relative_to(REPO_ROOT)}:{child.lineno}: {line}"
                    )

    if hits:
        report.error(
            "Importable library code must not call logging.basicConfig() at import "
            "time. Configure handlers only in explicit setup functions or CLI "
            "entrypoints. First hits: " + "; ".join(hits[:8])
        )


def is_logging_basic_config_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == LOGGING_BASIC_CONFIG_ATTR
        and isinstance(func.value, ast.Name)
        and func.value.id == "logging"
    )


def is_main_guard(node: ast.AST) -> bool:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
    ):
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left = node.left
    right = node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def validate_pyproject_only_packaging(
    module_dirs: list[Path], report: ContractReport
) -> None:
    """Reject retired packaging mirrors at module roots.

    Every module builds through ``setuptools.build_meta`` from its
    ``pyproject.toml``; dependency resolution is pinned by the root
    ``uv.lock``. A ``setup.py`` shim or ``requirements.txt`` copy only
    reintroduces a second, drifting dependency declaration.
    """
    for module_dir in module_dirs:
        for name in RETIRED_PACKAGING_FILES:
            candidate = module_dir / name
            if candidate.exists():
                report.error(
                    f"{candidate.relative_to(REPO_ROOT)}: retired packaging file; "
                    "declare metadata and dependencies in pyproject.toml only"
                )


def validate_python_source_syntax(report: ContractReport) -> None:
    """Fail fast on any syntax errors in module source and examples."""
    source_files = [
        *REPO_ROOT.glob("GEO-INFER-*/src/**/*.py"),
        *REPO_ROOT.glob("GEO-INFER-*/examples/**/*.py"),
    ]

    for source_file in sorted(source_files):
        if not source_file.is_file():
            continue
        try:
            compile(
                source_file.read_text(encoding="utf-8"),
                filename=str(source_file),
                mode="exec",
            )
        except SyntaxError as exc:
            line = exc.text.strip() if exc.text else "n/a"
            report.error(
                f"{source_file.relative_to(REPO_ROOT)}:{exc.lineno}:{exc.offset}: "
                f"{exc.msg} ({line})"
            )
            return
        except UnicodeDecodeError as exc:
            report.error(
                f"{source_file.relative_to(REPO_ROOT)}: decoding error: {exc.reason}"
            )
            return


def parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Build a parent lookup map for an AST tree."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def ancestor_of_type(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    node_types: tuple[type[ast.AST], ...],
) -> ast.AST | None:
    """Return the nearest ancestor matching one of the given AST node types."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, node_types):
            return current
        current = parents.get(current)
    return None


def is_abstract_function(node: ast.AST) -> bool:
    """Return true when a function is decorated with abstractmethod."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
            return True
    return False


def is_pass_allowed(
    node: ast.Pass,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Allow pass only for abstract methods and exception-handling blocks."""
    if ancestor_of_type(node, parents, (ast.ExceptHandler,)) is not None:
        return True
    function = ancestor_of_type(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
    return bool(function and is_abstract_function(function))


def validate_no_concrete_pass_bodies(report: ContractReport) -> None:
    """Reject no-op pass bodies in concrete source code paths."""
    hits: list[str] = []
    for source_file in sorted(REPO_ROOT.glob("GEO-INFER-*/src/**/*.py")):
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text, filename=str(source_file))
        except SyntaxError:
            continue
        parents = parent_map(tree)
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Pass):
                continue
            if is_pass_allowed(node, parents):
                continue
            line = lines[node.lineno - 1].strip()
            hits.append(f"{source_file.relative_to(REPO_ROOT)}:{node.lineno}: {line}")

    if hits:
        report.error(
            "Concrete pass bodies found in source; use real behavior, an explicit "
            "return, or an abstract method. First hits: " + "; ".join(hits[:8])
        )


def validate_runtime_metadata(module_dirs: list[Path], report: ContractReport) -> None:
    metadata_files = [
        REPO_ROOT / "pyproject.toml",
        *REPO_ROOT.glob("GEO-INFER-*/**/pyproject.toml"),
    ]

    for metadata_file in sorted(set(metadata_files)):
        text = metadata_file.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if LEGACY_PYTHON_METADATA_PATTERN.search(line):
                report.error(
                    f"{metadata_file.relative_to(REPO_ROOT)}:{lineno}: "
                    "metadata advertises Python < 3.11"
                )


def validate_python_tool_targets(report: ContractReport) -> None:
    """Ensure Python tooling does not target interpreter versions below 3.11."""
    for pyproject_file in sorted(
        [REPO_ROOT / "pyproject.toml", *REPO_ROOT.glob("GEO-INFER-*/pyproject.toml")]
    ):
        if not pyproject_file.exists():
            continue
        try:
            pyproject = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            report.error(
                f"{pyproject_file.relative_to(REPO_ROOT)}: invalid TOML: {exc}"
            )
            continue

        relative = pyproject_file.relative_to(REPO_ROOT)
        tool = pyproject.get("tool", {})
        for section in RETIRED_TOOL_SECTIONS:
            if section in tool:
                report.error(
                    f"{relative}: [tool.{section}] is retired; Ruff owns lint "
                    "and format configuration in the root pyproject.toml"
                )
        target = tool.get("ruff", {}).get("target-version")
        if target is None:
            continue
        match = re.fullmatch(r"py3(\d+)", str(target))
        if not match or int(match.group(1)) < RUFF_TARGET_VERSION_MINIMUM:
            report.error(
                f"{relative}: Ruff target-version {target!r} is below Python 3.11"
            )


def validate_h3_dependency_metadata(report: ContractReport) -> None:
    metadata_files = [
        *REPO_ROOT.glob("GEO-INFER-*/pyproject.toml"),
        *REPO_ROOT.glob("GEO-INFER-*/locations/*/requirements*.txt"),
        REPO_ROOT / "pyproject.toml",
    ]
    for metadata_file in sorted(path for path in metadata_files if path.exists()):
        text = metadata_file.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if LEGACY_H3_PATTERN.search(line):
                report.error(
                    f"{metadata_file.relative_to(REPO_ROOT)}:{lineno}: "
                    "H3 dependency must require real h3-py 4.5.x "
                    "(h3>=4.5.0,<5)"
                )


def validate_pymdp_runtime_imports(report: ContractReport) -> None:
    """Reject legacy pymdp runtime paths in production ACT source."""
    act_src = REPO_ROOT / "GEO-INFER-ACT" / "src"
    if not act_src.exists():
        return

    for source_file in sorted(act_src.glob("**/*.py")):
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        for legacy_import in LEGACY_PYMDP_RUNTIME_IMPORTS:
            if legacy_import not in text:
                continue
            report.error(
                f"{source_file.relative_to(REPO_ROOT)}: "
                f"legacy pymdp runtime import {legacy_import!r} is forbidden; "
                "use geo_infer_act.utils.pymdp_adapter"
            )


def _smoke_problem(report: ContractReport, strict: bool, message: str) -> None:
    """Route an import-smoke failure to warnings or fatal errors."""
    if strict:
        report.error(message)
    else:
        report.warning(message)


def validate_import_smoke(
    module_dirs: list[Path],
    report: ContractReport,
    timeout: float = 30,
    *,
    strict: bool = False,
) -> None:
    """Probe source packages in bounded processes and verify their import origins.

    Failures are advisory warnings unless ``strict`` is set, which promotes
    every probe failure to a contract error so CI can fail on a package
    that raises on import or resolves from outside its src tree.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Import timeout must be finite and positive")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(p / "src") for p in module_dirs)
    for module_dir in module_dirs:
        package_name = expected_package_name(parse_pyproject(module_dir, report))
        if not package_name or not (module_dir / "src").exists():
            continue
        print(f"Import probe: {package_name}", flush=True)
        command = [
            sys.executable,
            "-P",
            "-c",
            "import importlib,json,pathlib,sys\n"
            "package = importlib.import_module(sys.argv[1])\n"
            "expected = pathlib.Path(sys.argv[2]).resolve()\n"
            "origin = getattr(package, '__file__', None)\n"
            "if origin is None or not pathlib.Path(origin).resolve().is_relative_to(expected):\n"
            "    raise ImportError(f'Imported {sys.argv[1]} from {origin}, outside expected source {expected}')\n"
            "print(json.dumps({'package':sys.argv[1],'probe_token':sys.argv[-1],'status':'ok'}))\n",
            package_name,
            str(module_dir / "src" / package_name),
        ]
        try:
            run_import_probe(
                command,
                package=package_name,
                cwd=REPO_ROOT,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _smoke_problem(
                report,
                strict,
                f"{module_dir.name}: import {package_name} timed out after {timeout}s",
            )
            continue
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()[-1500:]
            _smoke_problem(
                report,
                strict,
                f"{module_dir.name}: import {package_name} failed ({exc.returncode}): {detail}",
            )
        except ValueError as exc:
            _smoke_problem(
                report,
                strict,
                f"{module_dir.name}: import {package_name} failed: {exc}",
            )


def validate_source_language(report: ContractReport, strict: bool) -> None:
    hits: list[str] = []
    for source_file in sorted(REPO_ROOT.glob("GEO-INFER-*/src/**/*.py")):
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not SOURCE_LANGUAGE_PATTERN.search(line):
                continue
            if any(
                allowed.lower() in line.lower() for allowed in SOURCE_LANGUAGE_ALLOWLIST
            ):
                continue
            hits.append(
                f"{source_file.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
            )

    if not hits:
        return

    message = f"Source-language debt hits: {len(hits)}. First hits: " + "; ".join(
        hits[:8]
    )
    if strict:
        report.error(message)
    else:
        report.warning(message)


def requirement_name(requirement: str) -> str:
    """Normalize a requirement string to its distribution name.

    Thin delegate to the shared normalizer; strips inline comments, environment
    markers, extras and version specifiers.
    """
    cleaned = requirement.split("#", 1)[0].split(";", 1)[0].strip()
    return normalize_dependency_name(cleaned)


def _declared_requirements(pyproject: dict) -> list[str]:
    """Return runtime and optional requirement strings from a pyproject."""
    project = pyproject.get("project", {})
    requirements = [str(dep) for dep in project.get("dependencies") or []]
    for group in (project.get("optional-dependencies") or {}).values():
        requirements.extend(str(dep) for dep in group)
    return requirements


def validate_declared_requirements(report: ContractReport) -> None:
    """Reject stdlib modules and unpinned internal modules as dependencies."""
    for pyproject_file in sorted(REPO_ROOT.glob("GEO-INFER-*/pyproject.toml")):
        relative = pyproject_file.relative_to(REPO_ROOT)
        try:
            pyproject = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            continue  # validate_python_tool_targets reports invalid TOML.
        for requirement in _declared_requirements(pyproject):
            name = requirement_name(requirement)
            if not name:
                continue
            if name.replace("-", "_") in STDLIB_REQUIREMENT_NAMES:
                report.error(
                    f"{relative}: stdlib module listed as dependency: {requirement}"
                )
            if requirement.strip().endswith(">=0.0.0") and (
                name in INTERNAL_REQUIREMENT_NAMES
            ):
                report.error(
                    f"{relative}: internal module listed with a placeholder "
                    f"floor: {requirement}"
                )


def validate_markdown_local_links(report: ContractReport) -> None:
    excluded = {
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        ".geo-infer-test-results",
        "build",
        "dist",
    }
    markdown_files = []
    for directory, children, files in os.walk(REPO_ROOT):
        children[:] = sorted(name for name in children if name not in excluded)
        markdown_files.extend(
            Path(directory) / name
            for name in ("README.md", "AGENTS.md")
            if name in files
        )
    for markdown_file in sorted(markdown_files):
        text = markdown_file.read_text(encoding="utf-8", errors="ignore")
        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            target = match.group(1).strip()
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
                or target.startswith("app://")
                or target.startswith("/")
            ):
                continue
            target_path = target.split("#", 1)[0].strip("<>")
            if not target_path:
                continue
            candidate = (markdown_file.parent / target_path).resolve()
            try:
                candidate.relative_to(REPO_ROOT)
            except ValueError:
                report.error(
                    f"{markdown_file.relative_to(REPO_ROOT)}: link escapes repo: {target}"
                )
                continue
            if not candidate.exists():
                report.error(
                    f"{markdown_file.relative_to(REPO_ROOT)}: broken local link: {target}"
                )


def validate_runner_documentation(report: ContractReport) -> None:
    runner = REPO_ROOT / "GEO-INFER-TEST" / "run_unified_tests.py"
    readme = REPO_ROOT / "README.md"
    runner_text = runner.read_text(encoding="utf-8", errors="ignore")
    readme_text = readme.read_text(encoding="utf-8", errors="ignore")
    for flag in ("--module", "--category", "--h3-migration"):
        if flag in readme_text and flag not in runner_text:
            report.error(f"README documents {flag}, but run_unified_tests.py lacks it")


def load_doc_rewriter():
    """Load the generated documentation renderer without mutating files."""
    rewriter_path = REPO_ROOT / "GEO-INFER-TEST" / "rewrite_readme_agents.py"
    spec = importlib.util.spec_from_file_location(
        "geo_infer_rewrite_readme_agents", rewriter_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {rewriter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.REPO_ROOT = REPO_ROOT
    return module


def validate_generated_doc_freshness(
    report: ContractReport,
    rewriter=None,
) -> None:
    """Ensure tracked README.md and AGENTS.md files match generated facts."""
    if rewriter is None:
        try:
            rewriter = load_doc_rewriter()
        except Exception as exc:  # noqa: BLE001 - validator reports load failures
            report.error(f"Unable to load documentation rewriter: {exc}")
            return

    stale: list[str] = []
    for doc_path, expected in rewriter.expected_doc_files():
        current = doc_path.read_text(encoding="utf-8")
        if current != expected:
            stale.append(str(doc_path.relative_to(REPO_ROOT)))

    if stale:
        report.error(
            "Generated README.md/AGENTS.md files are stale. Run "
            "`uv run python GEO-INFER-TEST/rewrite_readme_agents.py`. "
            "First mismatches: " + "; ".join(stale[:8])
        )


def validate_generated_artifacts(report: ContractReport) -> None:
    for relative_path in ROOT_GENERATED_ARTIFACT_PATHS:
        generated_path = REPO_ROOT / relative_path
        if relative_path == "logs" and generated_path.exists():
            report.error(
                "Root logs/ exists; imports and validators must not create log files"
            )

    try:
        status = subprocess.run(
            ["git", "status", "--short", "--", *ROOT_GENERATED_ARTIFACT_PATHS],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return

    generated_lines = [line for line in status.stdout.splitlines() if line.strip()]
    if generated_lines:
        report.error(
            "Generated artifact churn detected: " + "; ".join(generated_lines[:8])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GEO-INFER repo contracts")
    parser.add_argument(
        "--strict-source-language",
        action="store_true",
        help="Fail when mock/stub/fake/placeholder language remains in source.",
    )
    parser.add_argument(
        "--skip-import-smoke",
        action="store_true",
        help="Skip best-effort per-package import smoke warnings.",
    )
    parser.add_argument(
        "--strict-import-smoke",
        action="store_true",
        help="Fail when any per-package import probe fails or resolves "
        "outside its expected src tree.",
    )
    parser.add_argument(
        "--import-timeout",
        type=float,
        default=30,
        help="Seconds allowed per isolated import probe",
    )
    args = parser.parse_args()

    report = ContractReport()
    module_dirs = find_module_dirs()

    validate_inventory(module_dirs, report)
    validate_signposting(module_dirs, report)
    validate_package_casing(module_dirs, report)
    validate_pyproject_packages(module_dirs, report)
    validate_uv_environment(report)
    validate_uv_setup_documentation(report)
    validate_test_inventory(module_dirs, report)
    validate_pyproject_only_packaging(module_dirs, report)
    validate_python_source_syntax(report)
    validate_no_concrete_pass_bodies(report)
    validate_runtime_metadata(module_dirs, report)
    validate_python_tool_targets(report)
    validate_h3_dependency_metadata(report)
    validate_pymdp_runtime_imports(report)
    validate_declared_requirements(report)
    validate_markdown_local_links(report)
    validate_runner_documentation(report)
    validate_generated_doc_freshness(report)
    validate_generated_artifacts(report)
    validate_module_task_markers(report)
    validate_logging_configuration(report)
    if not args.skip_import_smoke:
        validate_import_smoke(
            module_dirs,
            report,
            timeout=args.import_timeout,
            strict=args.strict_import_smoke,
        )
    validate_source_language(report, strict=args.strict_source_language)

    print(f"Modules checked: {len(module_dirs)}")
    print(f"Errors: {len(report.errors)}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"Warnings: {len(report.warnings)}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
