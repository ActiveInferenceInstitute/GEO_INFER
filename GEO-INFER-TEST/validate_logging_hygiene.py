#!/usr/bin/env python3
"""
Logging-hygiene validation for the GEO-INFER monorepo platform.

Enforces the passive library-logging contract across all ``GEO-INFER-*``
module sources (``GEO-INFER-*/src/**/*.py``):

- ``logging.basicConfig(...)`` must not appear in library code: it mutates
  the process-wide root logger at import or call time.
- ``.addHandler(...)`` / ``.setLevel(...)`` must only be called on a
  module-local logger, i.e. a name (or ``self``-attribute) assigned from
  ``logging.getLogger(__name__)`` in the same module. Calls on the root
  logger, on third-party loggers such as ``logging.getLogger('urllib3')``,
  on locally built handler objects, or on loggers derived from any other
  name are violations.

Exemptions, all documented here and kept auditable:

- Entry points: CLI modules (``cli.py``/``main.py``), example scripts
  (paths containing ``examples``), and ``if __name__ == "__main__":``
  guard bodies may configure logging — that is the app layer.
- Named-logger facilities (``WHITELIST_FILES``): modules whose documented
  purpose is building a *dedicated, non-root* logger with handlers. They
  never mutate the root logger or call ``basicConfig``; each is invoked
  from a CLI entrypoint or a documented module workflow.

Violations are reported as a ``file:line`` list and the validator exits
non-zero. Library fixes belong to the module owners; CLI entrypoints
should route through the shared ``configure_logging`` entry instead of
configuring logging locally.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PREFIX = "GEO-INFER-"
SOURCE_GLOB = f"{MODULE_PREFIX}*/src/**/*.py"

# Documented named-logger facilities: classes/functions whose purpose is
# building a dedicated (non-root) logger with handlers. These never touch
# the root logger or basicConfig at import time; each is referenced from a
# CLI entrypoint or a documented module workflow.
WHITELIST_FILES: frozenset = frozenset(
    {
        Path("GEO-INFER-OPS/src/geo_infer_ops/utils/shared_logging.py"),
        Path("GEO-INFER-GIT/src/geo_infer_git/utils/logging_utils.py"),
        Path("GEO-INFER-LOG/src/geo_infer_log/__init__.py"),
        Path("GEO-INFER-RISK/src/geo_infer_risk/core/risk_engine.py"),
        Path("GEO-INFER-COG/src/geo_infer_cog/utils/helpers.py"),
        Path("GEO-INFER-ANT/src/geo_infer_ant/utils/logging.py"),
        Path("GEO-INFER-TEST/src/geo_infer_test/core/log_integration.py"),
    }
)
WHITELIST_FUNCTION = "configure_logging"

# Entry points are allowed to configure logging: CLI modules, __main__ demo
# guards, and example scripts.
ENTRYPOINT_FILENAMES = frozenset({"cli.py", "main.py"})
ENTRYPOINT_DIR_MARKERS = frozenset({"examples"})

MAIN_GUARD_TEST = "__name__ == '__main__'"

MUTATING_METHODS = ("addHandler", "setLevel")
CONFIG_METHODS = MUTATING_METHODS + ("basicConfig",)

GETLOGGER_ARG_LOCAL = "__name__"


@dataclass
class HygieneReport:
    errors: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)


@dataclass
class _LocalLoggers:
    """Names bound to ``logging.getLogger(__name__)`` within one module."""

    names: set = field(default_factory=set)

    def add_binding(self, target: ast.expr, call: ast.Call) -> None:
        if not is_local_getlogger_call(call):
            return
        if isinstance(target, ast.Name):
            self.names.add(target.id)
        elif isinstance(target, ast.Attribute):
            self.names.add(ast.unparse(target))


def is_logging_call(node: ast.AST, attr: str) -> bool:
    """True for ``logging.<attr>(...)`` or ``from logging import <attr>`` calls."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == attr
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == attr
    return False


def is_local_getlogger_call(call: ast.Call) -> bool:
    """True for ``logging.getLogger(__name__)`` (module-local acquisition)."""
    if not is_logging_call(call, "getLogger"):
        return False
    if len(call.args) != 1 or call.keywords:
        return False
    return isinstance(call.args[0], ast.Name) and call.args[0].id == GETLOGGER_ARG_LOCAL


def receiver_is_module_local(receiver: ast.expr, local: _LocalLoggers) -> bool:
    """True when the receiver is a module-local ``getLogger(__name__)`` logger."""
    if isinstance(receiver, ast.Call):
        return is_local_getlogger_call(receiver)
    if isinstance(receiver, (ast.Name, ast.Attribute)):
        return ast.unparse(receiver) in local.names
    return False


def inside_whitelist(source_file: Path, tree: ast.AST, node: ast.AST) -> bool:
    """True inside a documented facility file or the whitelisted function."""
    if source_file in WHITELIST_FILES:
        return True
    for candidate in ast.walk(tree):
        if (
            isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            and candidate.name == WHITELIST_FUNCTION
            and candidate.lineno <= node.lineno <= candidate.end_lineno
        ):
            return True
    return False


def inside_main_guard(tree: ast.AST, node: ast.AST) -> bool:
    """True when the node sits under an ``if __name__ == "__main__":`` guard."""
    for candidate in ast.walk(tree):
        if not (
            isinstance(candidate, ast.If)
            and ast.unparse(candidate.test).strip() == MAIN_GUARD_TEST
        ):
            continue
        end = candidate.end_lineno or candidate.lineno
        if candidate.lineno <= node.lineno <= end:
            return True
    return False


def is_entrypoint_path(relative: Path) -> bool:
    """True for CLI modules, ``main.py`` modules, and example scripts."""
    if relative.name in ENTRYPOINT_FILENAMES:
        return True
    return any(part in ENTRYPOINT_DIR_MARKERS for part in relative.parts)


def collect_local_loggers(tree: ast.AST) -> _LocalLoggers:
    """Register every assignment of ``<name> = logging.getLogger(__name__)``."""
    local = _LocalLoggers()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                local.add_binding(target, node.value)
    return local


def scan_source_file(source_file: Path, report: HygieneReport) -> None:
    """Record logging-hygiene violations found in one source file."""
    text = source_file.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(source_file))
    except SyntaxError:
        return

    relative = source_file.relative_to(REPO_ROOT)
    local = collect_local_loggers(tree)
    lines = text.splitlines()
    violations_in_file = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr not in CONFIG_METHODS:
            continue
        suppressed = (
            inside_whitelist(relative, tree, node)
            or is_entrypoint_path(relative)
            or inside_main_guard(tree, node)
        )
        violation: str | None = None
        if attr == "basicConfig":
            violation = (
                f"{relative}:{node.lineno}: logging.basicConfig() mutates the "
                "root logger; use logging.getLogger(__name__) and leave "
                "configuration to the app entry (shared_logging.configure_logging)"
            )
        elif not receiver_is_module_local(node.func.value, local):
            snippet = (
                lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            )
            violation = (
                f"{relative}:{node.lineno}: {attr}() on a non-module-local "
                f"logger: {snippet}"
            )
        if violation is None:
            continue
        violations_in_file += 1
        if not suppressed:
            report.error(violation)

    if relative in WHITELIST_FILES and violations_in_file == 0:
        report.error(
            f"{relative}: file is whitelisted but no violation found; remove it "
            "from WHITELIST_FILES"
        )


def validate_logging_hygiene(report: HygieneReport | None = None) -> HygieneReport:
    """Scan module sources and return the hygiene report."""
    if report is None:
        report = HygieneReport()
    for source_file in sorted(REPO_ROOT.glob(SOURCE_GLOB)):
        if source_file.is_file():
            scan_source_file(source_file, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate passive library-logging hygiene in GEO-INFER modules"
    )
    parser.parse_args()

    report = validate_logging_hygiene()
    print(f"Errors: {len(report.errors)}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if not report.errors:
        print(
            "Logging hygiene OK: passive library logging (getLogger(__name__)) "
            f"with {len(WHITELIST_FILES)} documented named-logger facility "
            "exemptions and entry-point allowances."
        )
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
