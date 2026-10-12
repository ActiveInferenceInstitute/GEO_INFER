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
import hashlib
import math
import uuid
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

from .process import (
    run_process,
    terminate_running_processes,
    reset_process_cancellation,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PREFIX = "GEO-INFER-"
TEST_DIR_NAME = "tests"
RESULTS_DIR = PROJECT_ROOT / ".geo-infer-test-results"
RUN_ID = uuid.uuid4().hex
DEFAULT_WORKERS = 2
PYTEST_NO_TESTS_EXIT_CODE = 5
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

# Nested test trees that top-level module discovery cannot reach. The
# Cascadia unit and integration trees join their corresponding PLACE lanes.
# Tests use local fixtures; external licensed-data acquisition remains separate.
EXTRA_TEST_PATHS: dict[str, dict[str, tuple[Path, ...]]] = {
    "PLACE": {
        "unit": (Path("locations/cascadia/tests/unit"),),
        "integration": (Path("locations/cascadia/tests/integration"),),
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
    timeout: float = 0
    returncode: int | None = None
    status: str = "PASS"
    executed: int = 0
    receipt: str = ""

    def __post_init__(self) -> None:
        if not self.success and self.status == "PASS":
            self.status = "FAIL"


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
        return (
            bool(self.results)
            and any(result.status != "EMPTY" for result in self.results)
            and all(result.success for result in self.results)
        )


def discover_geo_infer_modules(root: Path | None = None) -> list[Module]:
    """Discover all top-level GEO-INFER modules in stable order."""
    modules: list[Module] = []
    for item in sorted((root or PROJECT_ROOT).iterdir()):
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


def discover_workspace_test_targets(root: Path | None = None) -> list[Module]:
    """Discover 45 owning packages plus the separately profiled root suite."""
    root = root or PROJECT_ROOT
    targets = discover_geo_infer_modules(root)
    tests = root / "tests"
    if has_test_files(tests):
        targets.append(Module("ROOT", root, tests, True))
    return targets


ROOT_RENDER_FILE = "test_manuscript_pdf_layout.py"
ROOT_RENDER_NODE = "tests/test_manuscript_paths.py::TestFigurePathLiterals::test_the_combined_document_keeps_the_rewrite_inside_image_targets"


def profile_selection_args(module: Module, category: str) -> list[str]:
    """Select declared runtime profiles without silently dropping tests."""
    if module.name != "ROOT":
        return []
    if category == "manuscript-render":
        return [
            "-k",
            "test_manuscript_pdf_layout or test_the_combined_document_keeps_the_rewrite_inside_image_targets",
        ]
    return ["--deselect", ROOT_RENDER_NODE]


def ensure_results_dir(clean: bool = False) -> None:
    """Create the run directory; previous attempts are never deleted."""
    del clean
    run_results_dir().mkdir(parents=True, exist_ok=True)


def run_results_dir() -> Path:
    return RESULTS_DIR / "runs" / RUN_ID


def runtime_receipt(*, timeout: float = 10) -> dict:
    """Bind results to revision, dirty source bytes, interpreter, and lock."""
    deadline = time.monotonic() + timeout
    receipt = {"run_id": RUN_ID, "python": sys.version, "executable": sys.executable}
    commands = (
        ("revision", ["git", "rev-parse", "HEAD"]),
        ("dirty", ["git", "status", "--porcelain"]),
        (
            "modified",
            ["git", "ls-files", "--modified", "--others", "--exclude-standard", "-z"],
        ),
        (
            "staged",
            ["git", "diff", "--cached", "--name-only", "--no-renames", "-z", "HEAD"],
        ),
    )
    for name, args in commands:
        try:
            # Custody reads must not start Git's optional persistent daemon.
            args = [args[0], "-c", "core.fsmonitor=false", *args[1:]]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            completed = run_process(
                args, cwd=PROJECT_ROOT, env=os.environ.copy(), timeout=remaining
            )
            receipt[name] = (
                (
                    completed.stdout
                    if name in {"modified", "staged"}
                    else completed.stdout.rstrip("\r\n")
                )
                if completed.returncode == 0
                else None
            )
        except (OSError, subprocess.SubprocessError):
            receipt[name] = None
    modified = receipt.pop("modified")
    staged = receipt.pop("staged")
    hashes = {}
    if modified is not None and staged is not None:
        for name in sorted(set((modified + staged).split("\0")) - {""}):
            path = PROJECT_ROOT / name
            if not path.is_file():
                hashes[name] = None
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(
                            ["receipt source hashing"], timeout
                        )
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
    receipt["dirty_sha256"] = hashes
    receipt["custody_complete"] = (
        bool(receipt.get("revision"))
        and receipt.get("dirty") is not None
        and modified is not None
        and staged is not None
    )
    lock = PROJECT_ROOT / "uv.lock"
    receipt["lock_sha256"] = (
        hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
    )
    return receipt


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


def _is_pytest_command(command: list[str]) -> bool:
    """Recognize public pytest entry points without inspecting script arguments."""
    if not command:
        return False
    executable = command[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if executable in {"pytest", "pytest.exe", "py.test", "py.test.exe"}:
        return True
    if not re.fullmatch(
        r"(?:pythonw?(?:\d+(?:\.\d+)*)?d?|pypy(?:\d+(?:\.\d+)*)?)(?:\.exe)?",
        executable,
    ):
        return False
    index = 1
    while index < len(command):
        argument = command[index]
        if argument == "--check-hash-based-pycs":
            index += 2
            continue
        if argument.startswith("--check-hash-based-pycs="):
            index += 1
            continue
        if argument in {"-", "--"} or not argument.startswith("-"):
            return False
        options = argument[1:]
        for position, option in enumerate(options):
            if option in {"c", "h", "V"}:
                return False
            if option == "m":
                module = options[position + 1 :]
                if not module and index + 1 < len(command):
                    module = command[index + 1]
                return module == "pytest"
            if option in {"W", "X"}:
                if position + 1 == len(options):
                    index += 1
                break
            if option not in "bBdEiIOPqRsSuvx":
                return False
        index += 1
    return False


def _junit_arguments(
    command: list[str],
) -> tuple[list[tuple[int, int, str]], list[str]]:
    """Parse report operands before ``--`` in pytest's last-option-wins order."""
    reports: list[tuple[int, int, str]] = []
    errors: list[str] = []
    index = 0
    while index < len(command):
        argument = command[index]
        if argument == "--":
            break
        option, separator, value = argument.partition("=")
        if option not in {"--junitxml", "--junit-xml"}:
            index += 1
            continue
        stop = index + 1
        if not separator:
            if stop == len(command):
                errors.append(f"JUnit report option {option} omitted its path")
                index = stop
                continue
            value = command[stop]
            # argparse treats a lone dash and negative numbers as operands.
            if (
                value.startswith("-")
                and value != "-"
                and not re.fullmatch(r"-\d+|-\d*\.\d+", value)
            ):
                errors.append(f"JUnit report option {option} omitted its path")
                index = stop
                continue
            stop += 1
        if value:
            reports.append((index, stop, value))
        else:
            errors.append(f"JUnit report option {option} requires a nonempty path")
        index = stop
    return reports, errors


def junit_path(command: list[str]) -> Path | None:
    """Return the final valid report operand, respecting aliases and ``--``."""
    reports, _ = _junit_arguments(command)
    return Path(reports[-1][2]) if reports else None


def _bind_junit_report(
    command: list[str], owned_path: Path, cwd: Path
) -> tuple[list[str], list[str]]:
    """Bind valid promises to one fresh report while retaining invalid arguments."""
    reports, errors = _junit_arguments(command)
    operands = {}
    for start, stop, value in reports:
        path = Path(value)
        if not path.is_absolute():
            path = cwd / path
        try:
            if path.is_dir():
                errors.append("JUnit report path must be a file, not a directory")
                continue
        except (OSError, ValueError):
            errors.append("JUnit report path could not be inspected")
            continue
        operands[start] = stop
    bound: list[str] = []
    index = 0
    while index < len(command):
        stop = operands.get(index)
        if stop is not None:
            bound.append(f"--junitxml={owned_path}")
            index = stop
        else:
            bound.append(command[index])
            index += 1
    return bound, errors


def junit_contract_errors(path: Path | None, *, allow_empty: bool = False) -> list[str]:
    """Require complete pytest evidence and reject forbidden test outcomes."""
    if path is None:
        return []
    if not path.is_file():
        return [f"pytest produced no JUnit report: {path}"]
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return [f"invalid JUnit report: {exc}"]
    if root.tag not in {"testsuites", "testsuite"}:
        return [f"invalid JUnit root: {root.tag}"]
    cases = list(root.iter("testcase"))
    errors: list[str] = []
    if not cases and not allow_empty:
        errors.append("pytest JUnit report contains no testcases")
    suites = list(root.iter("testsuite"))
    try:
        declared = sum(int(suite.attrib["tests"]) for suite in suites)
        if declared != len(cases):
            errors.append(
                f"JUnit testcase count mismatch: declared {declared}, recorded {len(cases)}"
            )
    except (KeyError, ValueError):
        errors.append("JUnit report omitted a valid test count")
    for testcase in cases:
        name = (
            testcase.attrib.get("classname", "")
            + "::"
            + testcase.attrib.get("name", "")
        )
        for tag in ("skipped", "failure", "error"):
            entry = testcase.find(tag)
            if entry is not None:
                reason = entry.attrib.get("message", "") or entry.text or ""
                errors.append(
                    f"forbidden skipped/xfail testcase {name}: {reason}"
                    if tag == "skipped"
                    else f"{tag} testcase {name}: {reason}"
                )
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
    except (ET.ParseError, OSError):
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


def junit_selection_errors(path: Path | None, selection: dict) -> list[str]:
    """Bind JUnit testcase identities to the independently recorded execution."""
    if path is None or not path.is_file():
        return []
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return []
    executed = selection["executed"]
    errors = []
    if len(cases) != len(executed):
        errors.append(
            f"JUnit/selection executed count mismatch: report {len(cases)}, selection {len(executed)}"
        )
    files = {
        node.partition("::")[0].removesuffix(".py").replace("/", "."): node.partition(
            "::"
        )[0]
        for node in executed
    }
    recorded = []
    for case in cases:
        classname, name = case.attrib.get("classname", ""), case.attrib.get("name", "")
        namespaces = [
            namespace
            for namespace in files
            if classname == namespace or classname.startswith(namespace + ".")
        ]
        if not namespaces:
            errors.append(
                f"JUnit testcase identity is absent from selection: {classname}::{name}"
            )
            continue
        namespace = max(namespaces, key=len)
        classes = (
            classname[len(namespace) :].lstrip(".").split(".")
            if classname != namespace
            else []
        )
        recorded.append("::".join([files[namespace], *classes, name]))
    if Counter(recorded) != Counter(executed):
        errors.append("JUnit testcase identities do not match recorded execution")
    return errors


def result_failure_names(result: CommandResult) -> list[str]:
    """Return failing testcase names recorded by a command's JUnit report."""
    return junit_failure_names(junit_path(result.command))


def _console_print(value: str, *, flush: bool = False) -> None:
    """Display diagnostics on restricted consoles without changing UTF-8 artifacts."""
    try:
        print(value, flush=flush)
    except UnicodeEncodeError as exc:
        escaped = value.encode(exc.encoding, errors="backslashreplace").decode(
            exc.encoding
        )
        print(escaped, flush=flush)


def print_failure_details(report: SuiteReport) -> None:
    """Print failing test names per failed suite for ``--show-failures`` verdicts.

    JUnit-backed pytest commands contribute exact testcase names; commands
    without a JUnit report (validators, timeouts, pre-pytest failures) fall
    back to a bounded tail of their captured output so the verdict stays
    self-sufficient without opening the per-suite logs.
    """
    _console_print("\n== Failing tests")
    for result in report.results:
        if result.success:
            continue
        _console_print(f"-- {result.name}")
        names = result_failure_names(result)
        if names:
            for name in names:
                _console_print(f"  FAILED {name}")
            continue
        tail = _text_tail(result.stderr or result.stdout, limit=1200).strip()
        if tail:
            for line in tail.splitlines()[-12:]:
                _console_print(f"  {line}")


def run_command(
    command: list[str],
    name: str,
    timeout: float,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    allow_empty: bool = False,
    completion_token: str | None = None,
) -> CommandResult:
    """Execute one immutable attempt; a retry is a distinct explicit command."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be finite and positive")
    cwd = PROJECT_ROOT if cwd is None else cwd
    started = time.monotonic()
    metadata = {}
    metadata_error = None
    try:
        metadata = runtime_receipt(timeout=min(timeout, 10))
        if metadata and not metadata.get("custody_complete", True):
            raise OSError("checkout revision or dirty-source inventory unavailable")
    except Exception as exc:
        metadata_error = exc
    ensure_results_dir()
    attempt_dir = run_results_dir() / "attempts" / uuid.uuid4().hex
    attempt_dir.mkdir(parents=True)
    is_pytest = _is_pytest_command(command)
    promised_junit = junit_path(command)
    has_coverage_report = any(arg.startswith("--cov-report=json:") for arg in command)
    owned_junit = attempt_dir / "junit.xml"
    command, report_argument_errors = _bind_junit_report(command, owned_junit, cwd)
    report_path = owned_junit if f"--junitxml={owned_junit}" in command else None
    command = [
        f"--cov-report=json:{attempt_dir / 'coverage.json'}"
        if arg.startswith("--cov-report=json:")
        else arg
        for arg in command
    ]
    _console_print(f"\n== {name}")
    _console_print("$ " + " ".join(command), flush=True)
    env = build_subprocess_env()
    if env_overrides:
        env.update(env_overrides)
    if has_coverage_report:
        env["COVERAGE_FILE"] = str(attempt_dir / ".coverage")
    selection_path = attempt_dir / "selection.json"
    env["GEO_INFER_TEST_SELECTION"] = str(selection_path)
    rc = None
    stdout = stderr = ""
    status = "FAIL"
    errors: list[str] = report_argument_errors.copy()
    launched = False
    process_evidence = {}
    process_started = time.monotonic()
    setup_seconds = process_started - started
    try:
        if metadata_error is not None:
            status = "METADATA_ERROR"
            raise metadata_error
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        launched = True
        completed = run_process(command, cwd=cwd, env=env, timeout=remaining)
        process_evidence = getattr(completed, "process_evidence", {})
        rc, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        process_evidence = getattr(exc, "process_evidence", {})
        stdout, stderr = (
            _text(exc.stdout),
            _text(exc.stderr),
        )
        errors.append(
            f"Timed out after {timeout}s; "
            + (
                "process-tree cleanup attempted; see retained diagnostics"
                if launched
                else "setup deadline exhausted before command launch"
            )
        )
        status = "TIMEOUT"
    except Exception as exc:
        process_evidence = getattr(exc, "process_evidence", {})
        stdout, stderr = (
            _text(getattr(exc, "output", None)),
            _text(getattr(exc, "stderr", None)),
        )
        errors.append(f"Command could not complete: {type(exc).__name__}: {exc}")
    except KeyboardInterrupt as exc:
        process_evidence = getattr(exc, "process_evidence", {})
        stdout, stderr = (
            _text(getattr(exc, "output", None)),
            _text(getattr(exc, "stderr", None)),
        )
        errors.append(
            "Command interrupted; "
            + (
                "process-tree cleanup attempted; see retained diagnostics"
                if launched
                else "command was not launched"
            )
        )
        status = "INTERRUPTED"
    process_finished = time.monotonic()
    completion = None
    if completion_token is not None and rc == 0:
        try:
            completion = json.loads(stdout.strip().splitlines()[-1])
            if (
                not isinstance(completion, dict)
                or completion.get("completion_token") != completion_token
                or completion.get("status") != "ok"
            ):
                raise ValueError("terminal token/status mismatch")
        except (ValueError, IndexError) as exc:
            errors.append(f"missing or invalid terminal completion receipt: {exc}")
    empty = is_pytest and allow_empty and rc == PYTEST_NO_TESTS_EXIT_CODE
    errors.extend(junit_contract_errors(report_path, allow_empty=empty))
    if is_pytest and promised_junit is None:
        errors.append("pytest command omitted its required JUnit report")
    selected: dict = {}
    if "geo_infer_test.selection" in command:
        try:
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            errors.extend(selected.get("errors", []))
            if selected["unaccounted"]:
                errors.append(
                    f"unaccounted test deselection: {selected['unaccounted']}"
                )
            if not empty and not selected["executed"]:
                errors.append("pytest executed no tests")
            if rc == 0 and set(selected["selected"]) != set(selected["executed"]):
                errors.append("pytest did not execute every selected test")
            errors.extend(junit_selection_errors(report_path, selected))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"missing or invalid test selection receipt: {exc}")
    if is_pytest and rc == PYTEST_NO_TESTS_EXIT_CODE and not allow_empty:
        errors.append("pytest collected no tests (exit code 5)")
    success = (rc == 0 or empty) and not errors
    if success:
        status = "EMPTY" if empty else "PASS"
    stderr = "\n".join(part for part in [stderr, *errors] if part)
    duration = time.monotonic() - started
    path = report_path
    executed = (
        len(list(ET.parse(path).getroot().iter("testcase")))
        if path and path.is_file() and not any("invalid JUnit" in e for e in errors)
        else 0
    )
    result = CommandResult(
        name,
        success,
        duration,
        command,
        stdout,
        stderr,
        timeout,
        rc,
        status,
        executed,
        str(attempt_dir / "receipt.json"),
    )
    artifact_started = time.monotonic()
    (attempt_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (attempt_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    receipt = {
        **metadata,
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "status": status,
        "success": success,
        "returncode": rc,
        "duration": duration,
        "timeout": timeout,
        "executed": executed,
        "selection": selected,
        "completion": completion,
        "process_evidence": process_evidence,
        "timing": {
            "setup_seconds": setup_seconds,
            "process_seconds": process_finished - process_started,
            "validation_seconds": artifact_started - process_finished,
        },
        "artifacts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in attempt_dir.iterdir()
            if p.is_file()
        },
    }
    # This boundary includes log writing and artifact hashing. Receipt
    # serialization itself is excluded to avoid a self-referential rewrite.
    receipt["timing"]["artifact_seconds"] = time.monotonic() - artifact_started
    with (attempt_dir / "receipt.json").open("x", encoding="utf-8") as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write("\n")
    _console_print(f"{status} in {duration:.2f}s ({executed} testcases)", flush=True)
    if not success:
        _console_print("\n".join(part[-4000:] for part in (stdout, stderr) if part))
    return result


def module_by_name(name: str) -> Module:
    wanted = name.upper().removeprefix(MODULE_PREFIX)
    for module in discover_workspace_test_targets():
        if module.name.upper() == wanted:
            return module
    known = ", ".join(module.name for module in discover_workspace_test_targets())
    raise SystemExit(f"Unknown module {name!r}. Known modules: {known}")


def pytest_base_args() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "geo_infer_test.selection",
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
    if module.name == "ROOT":
        if category == "manuscript":
            return [
                path
                for path in test_file_paths(module.test_path)
                if path.name != ROOT_RENDER_FILE
            ]
        if category == "manuscript-render":
            return [
                module.test_path / ROOT_RENDER_FILE,
                module.test_path / "test_manuscript_paths.py",
            ]
        return []
    path_category = "unit" if category == "slow" else category
    category_path = module.test_path / path_category
    paths = test_file_paths(category_path)
    if path_category == "unit":
        paths.extend(test_file_paths(module.test_path, recursive=False))
    for extra_path in EXTRA_TEST_PATHS.get(module.name, {}).get(path_category, ()):
        paths.extend(test_file_paths(module.path / extra_path))
    return sorted(set(paths))


def module_test_files(module: Module) -> list[Path]:
    """Return all pytest files for a module, including explicit extras."""
    test_files = list(test_file_paths(module.test_path))
    for extra_paths in EXTRA_TEST_PATHS.get(module.name, {}).values():
        for extra_path in extra_paths:
            test_files.extend(test_file_paths(module.path / extra_path))
    return sorted(set(test_files))


def run_module_tests(module: Module, timeout: int) -> CommandResult:
    if module.name == "ROOT":
        paths = category_test_paths(module, "manuscript")
        return run_command(
            [
                *pytest_base_args(),
                *profile_selection_args(module, "manuscript"),
                *map(str, paths),
                f"--junitxml={run_results_dir() / 'ROOT_manuscript_results.xml'}",
            ],
            "ROOT manuscript tests",
            timeout,
        )
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
        f"--junitxml={run_results_dir() / f'{module.name}_results.xml'}",
    ]
    return run_command(command, f"{module.name} tests", timeout=timeout)


def execute_module_tasks(
    tasks: list[Callable[[], CommandResult]], *, workers: int, fail_fast: bool
) -> SuiteReport:
    """Bound module concurrency and retain completed attempts on interruption."""
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 16
    ):
        raise ValueError("workers must be an integer between 1 and 16")
    reset_process_cancellation()
    report = SuiteReport()

    def cancel_process_trees() -> None:
        """Retain cleanup failure without losing the fleet interruption report."""
        try:
            terminate_running_processes()
        except Exception as exc:
            report.add(
                CommandResult(
                    "fleet cleanup failed",
                    False,
                    0,
                    [],
                    stderr=f"{type(exc).__name__}: {exc}",
                )
            )

    if workers == 1:
        try:
            for task in tasks:
                try:
                    result = task()
                except Exception as exc:
                    result = CommandResult(
                        "module execution error", False, 0, [], stderr=str(exc)
                    )
                report.add(result)
                if result.status == "INTERRUPTED" or (fail_fast and not result.success):
                    break
        except KeyboardInterrupt:
            cancel_process_trees()
            report.add(
                CommandResult(
                    "fleet interrupted",
                    False,
                    0,
                    [],
                    status="INTERRUPTED",
                    stderr="Fleet cancellation requested; cleanup failures retained separately",
                )
            )
        finally:
            reset_process_cancellation()
        return report
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        for task in tasks:
            futures.append(executor.submit(task))
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                result = future.result()
            except Exception as exc:
                result = CommandResult(
                    "module execution error", False, 0, [], stderr=str(exc)
                )
            report.add(result)
            if result.status == "INTERRUPTED" or (fail_fast and not result.success):
                for pending in futures:
                    pending.cancel()
                cancel_process_trees()
                break
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        cancel_process_trees()
        report.add(
            CommandResult(
                "fleet interrupted",
                False,
                0,
                [],
                status="INTERRUPTED",
                stderr="Fleet cancellation requested; cleanup failures retained separately",
            )
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        reset_process_cancellation()
        recorded = {result.receipt for result in report.results if result.receipt}
        for future in futures:
            if future.cancelled() or not future.done():
                continue
            try:
                result = future.result()
            except Exception:
                continue
            if result.receipt and result.receipt not in recorded:
                report.add(result)
                recorded.add(result.receipt)
    return report


def run_module_category_tests(
    category: str,
    timeout: int,
    fail_fast: bool = False,
    modules: list[Module] | None = None,
    workers: int | None = None,
) -> SuiteReport:
    """Run one test category per module to avoid cross-module pytest state leaks."""
    report = SuiteReport()
    ensure_results_dir(clean=True)
    tasks = []
    discovered = False
    for module in modules if modules is not None else discover_workspace_test_targets():
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
            *profile_selection_args(module, category),
            *map(str, paths),
            f"--junitxml={run_results_dir() / f'{module.name}_{category}_results.xml'}",
        ]
        tasks.append(
            lambda command=command, name=f"{module.name} {category} tests": run_command(
                command, name, timeout=timeout, allow_empty=category == "slow"
            )
        )
    if tasks:
        report = execute_module_tasks(
            tasks,
            workers=DEFAULT_WORKERS if workers is None else workers,
            fail_fast=fail_fast,
        )

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
    return run_module_category_tests("performance", timeout, fail_fast)


def run_coverage_analysis(timeout: int, fail_fast: bool = False) -> SuiteReport:
    """Collect fleet coverage in isolated module subprocesses, then combine it."""
    report = SuiteReport()
    ensure_results_dir()
    coverage_data = run_results_dir() / ".coverage"
    coverage_json = run_results_dir() / "coverage.json"
    coverage_env = {"COVERAGE_FILE": str(coverage_data)}
    coverage_inputs: list[Path] = []

    for module in discover_workspace_test_targets():
        source_dirs = (
            [module.path / "manuscript", module.path / "scripts"]
            if module.name == "ROOT"
            else [module.path / "src"]
        )
        test_files = (
            category_test_paths(module, "manuscript")
            if module.name == "ROOT"
            else module_test_files(module)
        )
        if not all(path.exists() for path in source_dirs) or not test_files:
            continue
        command = [
            *pytest_base_args(),
            *profile_selection_args(module, "manuscript"),
            *map(str, test_files),
            *(f"--cov={source_dir}" for source_dir in source_dirs),
            f"--cov-report=json:{coverage_json}",
            f"--junitxml={run_results_dir() / f'{module.name}_coverage_results.xml'}",
        ]
        result = run_command(
            command,
            f"{module.name} coverage tests",
            timeout=timeout,
            env_overrides=coverage_env,
        )
        report.add(result)
        if result.receipt:
            attempt_data = Path(result.receipt).parent / ".coverage"
            if attempt_data.is_file():
                coverage_inputs.append(attempt_data)
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

    if not coverage_inputs:
        report.add(
            CommandResult(
                "coverage combine",
                False,
                0,
                [],
                stderr="No attempt-bound coverage data was produced",
            )
        )
        return report
    combined = run_command(
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            "--keep",
            *map(str, coverage_inputs),
        ],
        "coverage combine",
        timeout=timeout,
        env_overrides=coverage_env,
    )
    report.add(combined)
    if not combined.success:
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
    modules = discover_workspace_test_targets()
    tasks = [
        lambda module=module: run_module_tests(module, timeout) for module in modules
    ]
    return execute_module_tasks(tasks, workers=DEFAULT_WORKERS, fail_fast=fail_fast)


def _text(value: object) -> str:
    """Decode subprocess output without discarding retained diagnostics.

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
    return value


def _text_tail(value: object, limit: int = 2000) -> str:
    """Bound displayed summaries while immutable attempt logs retain all output."""
    return _text(value)[-limit:]


def category_budget_lines(report: SuiteReport) -> list[str]:
    """Render one line per category with its aggregate timeout budget.

    Each module command runs under the per-command ``--timeout``; the budget
    below is the worst case for the whole category, which is what a CI
    ``timeout-minutes`` job ceiling must exceed. Commands whose names carry a
    canonical category suffix (``<module> unit tests``) aggregate under that
    category; everything else (validators, coverage plumbing) aggregates
    under ``validators``.
    """
    category_re = re.compile(
        r" (unit|slow|integration|system|performance|coverage) tests$"
    )
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
        **runtime_receipt(),
        "generated_at": datetime.now(UTC).isoformat(),
        "workspace_inventory": {
            module.name: {
                "test_files": [
                    str(path.relative_to(PROJECT_ROOT))
                    for path in module_test_files(module)
                ],
                "profiles": {
                    category: {
                        "files": [
                            str(path.relative_to(PROJECT_ROOT))
                            for path in category_test_paths(module, category)
                        ],
                        "selection_args": profile_selection_args(module, category),
                    }
                    for category in (
                        ("manuscript", "manuscript-render")
                        if module.name == "ROOT"
                        else ("unit", "slow", "integration", "performance", "system")
                    )
                },
            }
            for module in discover_workspace_test_targets()
        },
        "artifacts": {
            str(path.relative_to(run_results_dir())): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in run_results_dir().rglob("*")
            if path.is_file() and path.name != "summary.json"
        },
        "success": report.success,
        "results": [
            {
                "name": result.name,
                "success": result.success,
                "duration": round(result.duration, 3),
                "command": result.command,
                "status": result.status,
                "returncode": result.returncode,
                "executed": result.executed,
                "receipt": result.receipt,
                "failures": result_failure_names(result),
                "stdout_tail": _text_tail(result.stdout),
                "stderr_tail": _text_tail(result.stderr),
            }
            for result in report.results
        ],
    }
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (run_results_dir() / "summary.json").write_text(serialized, encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    total = len(report.results)
    passed = sum(
        1 for result in report.results if result.success and result.status != "EMPTY"
    )
    empty = sum(result.status == "EMPTY" for result in report.results)
    _console_print("\n== Summary")
    _console_print(f"Passed: {passed}/{total}; empty selections: {empty}")
    if show_failures and not report.success:
        print_failure_details(report)
    budget_lines = category_budget_lines(report)
    if budget_lines:
        _console_print("\n== Per-category timeout budget")
        for line in budget_lines:
            _console_print(line)
    _console_print(f"Summary: {RESULTS_DIR / 'summary.json'}")


def record_validation_main(argv: list[str] | None = None) -> int:
    """Record a standalone validator with the same immutable attempt engine."""
    parser = argparse.ArgumentParser(description=record_validation_main.__doc__)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a validator command is required after --")
    if command[0] == "python":
        command[0] = sys.executable
    result = run_command(
        command, "validator: " + " ".join(command[1:]), timeout=args.timeout
    )
    write_summary(SuiteReport([result]), show_failures=True)
    return 0 if result.success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", help="Run tests for one GEO-INFER module.")
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        help="Run explicit test files as a named contract selection.",
    )
    parser.add_argument(
        "--category",
        choices=[
            "unit",
            "slow",
            "integration",
            "system",
            "performance",
            "coverage",
            "manuscript",
            "manuscript-render",
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
        type=float,
        default=300,
        help="Per-command timeout in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent module processes, between 1 and 16 (default: 2).",
    )
    parser.add_argument(
        "--results-dir", type=Path, help="Parent directory for immutable run receipts."
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if sum(bool(x) for x in (args.module, args.h3_migration, args.paths)) > 1:
        parser.error("--module, --paths, and --h3-migration cannot be combined")
    return args


def main() -> int:
    global RESULTS_DIR, RUN_ID, DEFAULT_WORKERS
    RUN_ID = uuid.uuid4().hex
    args = parse_args()
    DEFAULT_WORKERS = args.workers
    if args.results_dir:
        RESULTS_DIR = args.results_dir.resolve()

    if args.list_modules:
        for module in discover_geo_infer_modules():
            _console_print(module.name)
        return 0

    report = SuiteReport()

    if args.paths:
        paths = [path.resolve() for path in args.paths]
        if any(
            not path.is_relative_to(PROJECT_ROOT) or not path.is_file()
            for path in paths
        ):
            raise SystemExit(
                "--paths must name existing test files inside the checkout"
            )
        report.add(
            run_command(
                [
                    *pytest_base_args(),
                    *map(str, paths),
                    f"--junitxml={run_results_dir() / 'contracts.xml'}",
                ],
                "contract selection",
                timeout=args.timeout,
            )
        )
    elif args.module:
        module = module_by_name(args.module)
        if args.category in {
            "unit",
            "slow",
            "integration",
            "system",
            "performance",
            "manuscript",
            "manuscript-render",
        }:
            report = run_module_category_tests(
                args.category, args.timeout, args.fail_fast, [module]
            )
        elif args.category == "coverage":
            raise SystemExit(
                "--module with --category coverage is unsupported; use measure_module_coverage.py --modules"
            )
        else:
            report.add(run_module_tests(module, timeout=args.timeout))
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
    elif args.category in {"manuscript", "manuscript-render"}:
        report = run_module_category_tests(args.category, args.timeout, args.fail_fast)
    else:
        report = run_all_modules(timeout=args.timeout, fail_fast=args.fail_fast)

    write_summary(report, show_failures=args.show_failures)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
