#!/usr/bin/env python3
"""Run GEO-INFER test and validation suites.

The runner intentionally mirrors the commands documented in the root README:

* ``--module NAME`` runs one module's tests.
* ``--category unit|slow|integration|performance|coverage`` runs a focused
  suite.
* ``--h3-migration`` runs the H3/Active Inference and ACT script-orchestration
  contract validators.
* ``--show-failures`` prints the failing test names of every failed suite in
  the final summary verdict; summary.json records the names either way.

With no arguments, the runner executes the same broad module sweep that older
versions performed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_PREFIX = "GEO-INFER-"
TEST_DIR_NAME = "tests"
RESULTS_DIR = PROJECT_ROOT / ".geo-infer-test-results"
PYTEST_NO_TESTS_EXIT_CODE = 5
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

# Nested test trees that top-level module discovery cannot reach. The
# cascadia unit tree lives under GEO-INFER-PLACE/locations/cascadia and joins
# the PLACE lanes explicitly; its integration tree stays deferred per the
# PLACE-V14 licensed-data deferral rather than being wired red.
EXTRA_TEST_PATHS: dict[str, dict[str, tuple[Path, ...]]] = {
    "PLACE": {
        "unit": (
            PROJECT_ROOT
            / "GEO-INFER-PLACE"
            / "locations"
            / "cascadia"
            / "tests"
            / "unit",
        ),
    },
}


@dataclass
class CommandResult:
    name: str
    success: bool
    duration: float
    command: list[str]
    stdout: str = ""
    stderr: str = ""
    timeout: int = 0


@dataclass
class Module:
    name: str
    path: Path
    test_path: Path
    has_tests: bool


@dataclass
class SuiteReport:
    results: list[CommandResult] = field(default_factory=list)

    def add(self, result: CommandResult) -> None:
        self.results.append(result)

    @property
    def success(self) -> bool:
        return all(result.success for result in self.results)


def discover_geo_infer_modules() -> list[Module]:
    """Discover all top-level GEO-INFER modules in stable order."""
    modules: list[Module] = []
    for item in sorted(PROJECT_ROOT.iterdir()):
        if not item.is_dir() or not item.name.startswith(MODULE_PREFIX):
            continue
        test_path = item / TEST_DIR_NAME
        modules.append(
            Module(
                name=item.name.removeprefix(MODULE_PREFIX),
                path=item,
                test_path=test_path,
                has_tests=has_test_files(test_path),
            )
        )
    return modules


def ensure_results_dir(clean: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    if not clean:
        return
    for path in RESULTS_DIR.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def workspace_src_paths() -> list[Path]:
    """Return all workspace source directories in deterministic order."""
    return [
        module.path / "src"
        for module in discover_geo_infer_modules()
        if (module.path / "src").exists()
    ]


def build_subprocess_env() -> dict[str, str]:
    """Build the child-process environment used for pytest subprocesses."""
    env = os.environ.copy()
    pythonpath_parts = [str(path) for path in workspace_src_paths()]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.extend(
            part for part in existing_pythonpath.split(os.pathsep) if part
        )
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_parts))
    return env


def junit_path(command: list[str]) -> Path | None:
    """Return the JUnit path embedded in a pytest command, if present."""
    prefix = "--junitxml="
    for argument in command:
        if argument.startswith(prefix):
            return Path(argument.removeprefix(prefix))
    return None


def junit_contract_errors(path: Path | None) -> list[str]:
    """Reject skipped, xfailed, and xpassed entries in a JUnit report."""
    if path is None or not path.exists():
        return []
    root = ET.parse(path).getroot()
    errors: list[str] = []
    for testcase in root.iter("testcase"):
        skipped = testcase.find("skipped")
        if skipped is not None:
            name = (
                testcase.attrib.get("classname", "")
                + "::"
                + testcase.attrib.get("name", "")
            )
            reason = skipped.attrib.get("message", "") or (skipped.text or "")
            errors.append(f"forbidden skipped/xfail testcase {name}: {reason}")
    return errors


def junit_failure_names(path: Path | None) -> list[str]:
    """Return failing/erroring testcase names from a JUnit report.

    Missing or malformed reports yield an empty list: failure-name
    extraction is best-effort enrichment for the summary verdict and must
    never raise where the run itself already recorded the failure. The
    identifier shape mirrors ``junit_contract_errors``' testcase names.
    """
    if path is None or not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    names: list[str] = []
    for testcase in root.iter("testcase"):
        if testcase.find("failure") is None and testcase.find("error") is None:
            continue
        name = "::".join(
            part
            for part in (
                testcase.attrib.get("classname", ""),
                testcase.attrib.get("name", ""),
            )
            if part
        )
        names.append(name)
    return names


def result_failure_names(result: CommandResult) -> list[str]:
    """Return failing testcase names recorded by a command's JUnit report."""
    return junit_failure_names(junit_path(result.command))


def print_failure_details(report: SuiteReport) -> None:
    """Print failing test names per failed suite for ``--show-failures`` verdicts.

    JUnit-backed pytest commands contribute exact testcase names; commands
    without a JUnit report (validators, timeouts, pre-pytest failures) fall
    back to a bounded tail of their captured output so the verdict stays
    self-sufficient without opening the per-suite logs.
    """
    print("\n== Failing tests")
    for result in report.results:
        if result.success:
            continue
        print(f"-- {result.name}")
        names = result_failure_names(result)
        if names:
            for name in names:
                print(f"  FAILED {name}")
            continue
        tail = _text_tail(result.stderr or result.stdout, limit=1200).strip()
        if tail:
            for line in tail.splitlines()[-12:]:
                print(f"  {line}")


def run_command(
    command: list[str],
    name: str,
    timeout: int,
    cwd: Path = PROJECT_ROOT,
    env_overrides: dict[str, str] | None = None,
    allow_empty: bool = False,
    _is_retry: bool = False,
) -> CommandResult:
    """Run a subprocess and capture a compact result.

    ``allow_empty`` treats pytest's "collected no tests" exit (5) as a
    pass-with-note — expected for lanes whose marker filter selects nothing
    in most modules (e.g. the slow category). Crash-class failures (the
    interpreter killed by a signal, or a missing/empty JUnit report despite a
    failed run) get one bounded retry, mirroring the coverage-floor gate's
    GS19-01 semantics: a deterministic failure still fails on the retry.
    """
    print(f"\n== {name}")
    print("$ " + " ".join(command))
    started = time.time()
    subprocess_env = build_subprocess_env()
    if env_overrides:
        subprocess_env.update(env_overrides)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=subprocess_env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        print(f"TIMEOUT after {duration:.2f}s")
        return CommandResult(
            name=name,
            success=False,
            duration=duration,
            command=command,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"Timed out after {timeout}s",
            timeout=timeout,
        )

    duration = time.time() - started
    junit_errors = junit_contract_errors(junit_path(command))
    if completed.returncode == PYTEST_NO_TESTS_EXIT_CODE:
        if allow_empty:
            print(
                f"PASS in {duration:.2f}s (no tests collected — allowed for this lane)"
            )
            return CommandResult(
                name=name,
                success=True,
                duration=duration,
                command=command,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timeout=timeout,
            )
        junit_errors.append("pytest collected no tests (exit code 5)")
    crash_class = completed.returncode < 0 or (
        completed.returncode != 0
        and not junit_errors
        and not junit_path(command).exists()
    )
    if crash_class and not _is_retry:
        print(
            f"CRASH-COMPLETION rc={completed.returncode} — one bounded retry for "
            "crash-class failure"
        )
        # A killed interpreter's faulthandler banner (the crash reason and
        # the faulting import) sits at the HEAD of stderr; the default
        # failure print below only shows the tail, so surface the head.
        banner = (completed.stderr or "")[:1500]
        if banner:
            print("--- crash stderr head ---\n" + banner)
        return run_command(
            command,
            name,
            timeout=timeout,
            cwd=cwd,
            env_overrides=env_overrides,
            allow_empty=allow_empty,
            _is_retry=True,
        )
    if junit_errors:
        completed.stderr = "\n".join((*filter(None, [completed.stderr]), *junit_errors))
    success = completed.returncode == 0 and not junit_errors
    outcome = "PASS" if success else "FAIL"
    print(f"{outcome} in {duration:.2f}s")
    if not success:
        failure_output = "\n".join(
            part[-4000:] for part in (completed.stdout, completed.stderr) if part
        )
        if failure_output:
            print(failure_output)

    return CommandResult(
        name=name,
        success=success,
        duration=duration,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timeout=timeout,
    )


def module_by_name(name: str) -> Module:
    wanted = name.upper().removeprefix(MODULE_PREFIX)
    for module in discover_geo_infer_modules():
        if module.name.upper() == wanted:
            return module
    known = ", ".join(module.name for module in discover_geo_infer_modules())
    raise SystemExit(f"Unknown module {name!r}. Known modules: {known}")


def pytest_base_args() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(PROJECT_ROOT / "pyproject.toml"),
        "-v",
        "--tb=short",
        "--durations=10",
        "-W",
        "error",
    ]


def test_file_paths(path: Path, *, recursive: bool = True) -> list[Path]:
    """Return deterministic pytest file paths below *path*.

    Keeping file discovery in one helper prevents the module, category, and
    performance runners from silently diverging.
    """
    if not path.is_dir():
        return []
    glob = path.rglob if recursive else path.glob
    return sorted(
        {
            candidate
            for pattern in TEST_FILE_PATTERNS
            for candidate in glob(pattern)
            if candidate.is_file()
        }
    )


def has_test_files(path: Path) -> bool:
    """Return true when a directory contains pytest-discoverable test files."""
    return bool(test_file_paths(path))


def category_test_paths(module: Module, category: str) -> list[Path]:
    """Resolve a module's test files for one canonical category.

    The unit category covers ``tests/unit/``, test files directly under
    ``tests/`` and any nested test trees registered in ``EXTRA_TEST_PATHS``,
    so it cannot silently omit behavior tests. Integration, system, and performance remain bounded
    by their named directories. The ``slow`` category shares the unit path
    set; the marker filter applied by :func:`run_module_category_tests`
    selects the slow-marked complement.
    """
    path_category = "unit" if category == "slow" else category
    category_path = module.test_path / path_category
    paths = test_file_paths(category_path)
    if path_category != "unit":
        return paths
    paths.extend(test_file_paths(module.test_path, recursive=False))
    for extra_path in EXTRA_TEST_PATHS.get(module.name, {}).get(path_category, ()):
        paths.extend(test_file_paths(extra_path))
    return sorted(set(paths))


def module_test_files(module: Module) -> list[Path]:
    """Return all pytest files for a module, including explicit extras."""
    test_files = list(test_file_paths(module.test_path))
    for extra_paths in EXTRA_TEST_PATHS.get(module.name, {}).values():
        for extra_path in extra_paths:
            test_files.extend(test_file_paths(extra_path))
    return sorted(set(test_files))


def run_module_tests(module: Module, timeout: int) -> CommandResult:
    test_files = module_test_files(module)
    if not test_files:
        return CommandResult(
            name=f"{module.name} tests",
            success=False,
            duration=0.0,
            command=[],
            stderr=f"No tests found in {module.test_path}",
        )
    ensure_results_dir()
    command = [
        *pytest_base_args(),
        *map(str, test_files),
        f"--junitxml={RESULTS_DIR / f'{module.name}_results.xml'}",
    ]
    return run_command(command, f"{module.name} tests", timeout=timeout)


def run_module_category_tests(
    category: str, timeout: int, fail_fast: bool = False
) -> SuiteReport:
    """Run one test category per module to avoid cross-module pytest state leaks."""
    report = SuiteReport()
    ensure_results_dir(clean=True)
    discovered = False
    for module in discover_geo_infer_modules():
        paths = category_test_paths(module, category)
        if not paths:
            continue
        discovered = True
        marker_filter = {
            "unit": ["-m", "not slow"],
            "slow": ["-m", "slow"],
        }.get(category, [])
        command = [
            *pytest_base_args(),
            *marker_filter,
            *map(str, paths),
            f"--junitxml={RESULTS_DIR / f'{module.name}_{category}_results.xml'}",
        ]
        result = run_command(
            command,
            f"{module.name} {category} tests",
            timeout=timeout,
            allow_empty=category == "slow",
        )
        report.add(result)
        if fail_fast and not result.success:
            break

    if not discovered:
        report.add(
            CommandResult(
                name=f"{category} tests",
                success=False,
                duration=0.0,
                command=[],
                stderr=f"No {category} tests discovered.",
            )
        )
    return report


def run_unit_tests(timeout: int, fail_fast: bool = False) -> SuiteReport:
    return run_module_category_tests("unit", timeout=timeout, fail_fast=fail_fast)


def run_slow_tests(timeout: int, fail_fast: bool = False) -> SuiteReport:
    """Run each module's ``slow``-marked tests (the unit lane's complement)."""
    return run_module_category_tests("slow", timeout=timeout, fail_fast=fail_fast)


def run_integration_tests(timeout: int, fail_fast: bool = False) -> SuiteReport:
    return run_module_category_tests(
        "integration", timeout=timeout, fail_fast=fail_fast
    )


def run_system_tests(timeout: int, fail_fast: bool = False) -> SuiteReport:
    return run_module_category_tests("system", timeout=timeout, fail_fast=fail_fast)


def run_performance_tests(timeout: int, fail_fast: bool = False) -> SuiteReport:
    report = SuiteReport()
    ensure_results_dir(clean=True)
    discovered = False
    for module in discover_geo_infer_modules():
        performance_dir = module.test_path / "performance"
        # The directory is the canonical category boundary.  A unit test can
        # legitimately contain "performance" in its filename while still
        # exercising a small utility in the unit suite; selecting by filename
        # made this command silently execute unit tests twice.
        performance_files = test_file_paths(performance_dir)
        if not performance_files:
            continue
        discovered = True
        command = [
            *pytest_base_args(),
            *map(str, performance_files),
            f"--junitxml={RESULTS_DIR / f'{module.name}_performance_results.xml'}",
        ]
        result = run_command(
            command, f"{module.name} performance tests", timeout=timeout
        )
        report.add(result)
        if fail_fast and not result.success:
            break

    if not discovered:
        report.add(
            CommandResult(
                name="performance tests",
                success=False,
                duration=0.0,
                command=[],
                stderr="No performance tests discovered.",
            )
        )
        return report
    return report


def run_coverage_analysis(timeout: int, fail_fast: bool = False) -> SuiteReport:
    """Collect fleet coverage in isolated module subprocesses, then combine it."""
    report = SuiteReport()
    ensure_results_dir()
    coverage_data = RESULTS_DIR / ".coverage"
    coverage_json = RESULTS_DIR / "coverage.json"
    coverage_data.unlink(missing_ok=True)
    coverage_json.unlink(missing_ok=True)
    for stale_report in RESULTS_DIR.glob("*_coverage_results.xml"):
        stale_report.unlink()
    coverage_env = {"COVERAGE_FILE": str(coverage_data)}

    for module in discover_geo_infer_modules():
        source_dir = module.path / "src"
        test_files = test_file_paths(module.test_path)
        if not source_dir.exists() or not test_files:
            continue
        command = [
            *pytest_base_args(),
            *map(str, test_files),
            f"--cov={source_dir}",
            "--cov-append",
            "--cov-report=",
            f"--junitxml={RESULTS_DIR / f'{module.name}_coverage_results.xml'}",
        ]
        result = run_command(
            command,
            f"{module.name} coverage tests",
            timeout=timeout,
            env_overrides=coverage_env,
        )
        report.add(result)
        if fail_fast and not result.success:
            return report

    if not report.results:
        report.add(
            CommandResult(
                name="coverage tests",
                success=False,
                duration=0.0,
                command=[],
                stderr="No modules with source and tests were discovered.",
            )
        )
        return report

    report.add(
        run_command(
            [sys.executable, "-m", "coverage", "json", "-o", str(coverage_json)],
            "coverage JSON report",
            timeout=timeout,
            env_overrides=coverage_env,
        )
    )
    report.add(
        run_command(
            [sys.executable, "-m", "coverage", "report", "--show-missing"],
            "coverage terminal report",
            timeout=timeout,
            env_overrides=coverage_env,
        )
    )
    return report


def run_h3_contracts(timeout: int) -> SuiteReport:
    report = SuiteReport()
    validators = [
        "validate_act_geospatial_contract.py",
        "validate_act_script_orchestration.py",
        "validate_h3_active_inference_contract.py",
    ]
    for validator in validators:
        command = [sys.executable, str(PROJECT_ROOT / "GEO-INFER-TEST" / validator)]
        report.add(run_command(command, validator, timeout=timeout))
    return report


def run_all_modules(timeout: int, fail_fast: bool = False) -> SuiteReport:
    report = SuiteReport()
    ensure_results_dir(clean=True)
    for module in discover_geo_infer_modules():
        result = run_module_tests(module, timeout=timeout)
        report.add(result)
        if fail_fast and not result.success:
            break
    return report


def _text_tail(value: object, limit: int = 2000) -> str:
    """Return a JSON-safe tail for subprocess output.

    ``subprocess.TimeoutExpired`` can expose captured output as ``bytes`` even
    when ``text=True`` was requested.  Normalizing at the report boundary keeps
    a timed-out command from causing the overall test run to fail while its
    diagnostic summary is being written.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif not isinstance(value, str):
        value = str(value)
    return value[-limit:]


def category_budget_lines(report: SuiteReport) -> list[str]:
    """Render one line per category with its aggregate timeout budget.

    Each module command runs under the per-command ``--timeout``; the budget
    below is the worst case for the whole category, which is what a CI
    ``timeout-minutes`` job ceiling must exceed. Commands whose names carry a
    canonical category suffix (``<module> unit tests``) aggregate under that
    category; everything else (validators, coverage plumbing) aggregates
    under ``validators``.
    """
    category_re = re.compile(r" (unit|integration|system|performance|coverage) tests$")
    budgets: dict[str, int] = {}
    for result in report.results:
        if result.timeout <= 0:
            continue
        match = category_re.search(result.name)
        category = match.group(1) if match else "validators"
        budgets[category] = budgets.get(category, 0) + result.timeout
    lines = [
        f"{name}: {seconds}s budget ({seconds // 60}m {seconds % 60}s)"
        for name, seconds in sorted(budgets.items())
    ]
    if budgets:
        lines.append(f"total: {sum(budgets.values())}s")
    return lines


def write_summary(report: SuiteReport, show_failures: bool = False) -> None:
    ensure_results_dir()
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "success": report.success,
        "results": [
            {
                "name": result.name,
                "success": result.success,
                "duration": round(result.duration, 3),
                "command": result.command,
                "failures": result_failure_names(result),
                "stdout_tail": _text_tail(result.stdout),
                "stderr_tail": _text_tail(result.stderr),
            }
            for result in report.results
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    total = len(report.results)
    passed = sum(1 for result in report.results if result.success)
    print("\n== Summary")
    print(f"Passed: {passed}/{total}")
    if show_failures and not report.success:
        print_failure_details(report)
    budget_lines = category_budget_lines(report)
    if budget_lines:
        print("\n== Per-category timeout budget")
        for line in budget_lines:
            print(line)
    print(f"Summary: {RESULTS_DIR / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", help="Run tests for one GEO-INFER module.")
    parser.add_argument(
        "--category",
        choices=[
            "unit",
            "slow",
            "integration",
            "system",
            "performance",
            "coverage",
            "all",
        ],
        help="Run a focused test category.",
    )
    parser.add_argument(
        "--h3-migration",
        action="store_true",
        help="Run H3/Active Inference migration contract validators.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first module failure and exit non-zero immediately.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help=(
            "Print the failing test names of each failed suite in the final "
            "summary; summary.json records the names either way."
        ),
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Print discovered modules and exit.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-command timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_modules:
        for module in discover_geo_infer_modules():
            print(module.name)
        return 0

    report = SuiteReport()

    if args.module:
        report.add(run_module_tests(module_by_name(args.module), timeout=args.timeout))
    elif args.h3_migration:
        report = run_h3_contracts(timeout=args.timeout)
    elif args.category == "unit":
        report = run_unit_tests(timeout=args.timeout, fail_fast=args.fail_fast)
    elif args.category == "slow":
        report = run_slow_tests(timeout=args.timeout, fail_fast=args.fail_fast)
    elif args.category == "integration":
        report = run_integration_tests(timeout=args.timeout, fail_fast=args.fail_fast)
    elif args.category == "system":
        report = run_system_tests(timeout=args.timeout, fail_fast=args.fail_fast)
    elif args.category == "performance":
        report = run_performance_tests(timeout=args.timeout, fail_fast=args.fail_fast)
    elif args.category == "coverage":
        report = run_coverage_analysis(timeout=args.timeout, fail_fast=args.fail_fast)
    else:
        report = run_all_modules(timeout=args.timeout, fail_fast=args.fail_fast)

    write_summary(report, show_failures=args.show_failures)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
