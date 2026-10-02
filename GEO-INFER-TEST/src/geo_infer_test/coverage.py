#!/usr/bin/env python3
"""Per-module coverage measurement for the TEST-03 baseline.

Runs each module's own test selection (unit + integration categories;
performance and system suites are excluded from the baseline and noted)
under pytest-cov, and records the measured line coverage of the module's
``src`` package.

Usage::

    uv run --no-sync --with pytest-cov python \
        GEO-INFER-TEST/measure_module_coverage.py \
        [--modules GEO-INFER-ACT,GEO-INFER-MATH] [--json PATH]

pytest-cov is used deliberately: plain ``coverage run -m pytest`` cannot
see xdist's execnet workers (the controller imports none of the source),
while pytest-cov distributes coverage to workers and combines it.  The
in-module pytest parallelism is two workers; the module process budget
defaults to at most half the available CPUs, capped at four. ``--workers``
sets that module budget explicitly. ROOT measures manuscript/scripts with
the registered manuscript profile, after package measurements finish.

Output: one JSON object per module on stdout
(``{"module": ..., "coverage_percent": ..., "seconds": ..., "status": ...}``).
A module whose suite crashes is reported with ``status: "error"`` and no
coverage number; it is never silently skipped.
When pytest exits non-zero, the result also carries ``failing_tests``: the
``classname::name`` values recorded in the run's JUnit report, so gate
verdicts can name the failing tests instead of hiding them behind a single
``FAILED-SUITE`` line.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .execution import (
    Module,
    category_test_paths,
    run_results_dir,
    run_command,
    pytest_base_args,
    discover_workspace_test_targets,
    profile_selection_args,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_MODULE_WORKERS = max(1, min(4, (os.cpu_count() or 1) // 2))
MODULE_PYTEST_WORKERS = 2
MODULE_TIMEOUT_SECONDS = 1800


def _module_package(module: str) -> str | None:
    src = REPO_ROOT / module / "src"
    if not src.is_dir():
        return None
    for child in sorted(src.iterdir()):
        if child.is_dir() and child.name.startswith("geo_infer_"):
            return child.name
    return None


def measure_module(module: str) -> dict:
    """Measure one module's line coverage over its unit+integration tests."""
    package = _module_package(module) if module != "ROOT" else None
    if package is None and module != "ROOT":
        return {"module": module, "status": "error", "reason": "no src package"}
    descriptor = (
        Module("ROOT", REPO_ROOT, REPO_ROOT / "tests", True)
        if module == "ROOT"
        else Module(
            module.removeprefix("GEO-INFER-"),
            REPO_ROOT / module,
            REPO_ROOT / module / "tests",
            True,
        )
    )
    paths = sorted(
        set(
            category_test_paths(descriptor, "manuscript")
            if module == "ROOT"
            else category_test_paths(descriptor, "unit")
            + category_test_paths(descriptor, "integration")
        )
    )
    if not paths:
        return {
            "module": module,
            "status": "error",
            "reason": "no unit/integration test files",
        }
    import uuid

    data_path = run_results_dir() / "coverage" / uuid.uuid4().hex
    data_path.mkdir(parents=True)
    report_path = data_path / "report.json"
    started = time.monotonic()
    result = run_command(
        [
            *pytest_base_args(),
            *profile_selection_args(descriptor, "manuscript"),
            *map(str, paths),
            "-n",
            str(MODULE_PYTEST_WORKERS),
            "-p",
            "no:benchmark",
            "-p",
            "no:cacheprovider",
            *(
                (f"--cov={REPO_ROOT / 'manuscript'}", f"--cov={REPO_ROOT / 'scripts'}")
                if module == "ROOT"
                else (f"--cov={package}",)
            ),
            f"--cov-report=json:{report_path}",
            f"--junitxml={data_path / 'junit.xml'}",
        ],
        f"{module} coverage tests",
        timeout=MODULE_TIMEOUT_SECONDS,
        cwd=REPO_ROOT,
        env_overrides={"COVERAGE_FILE": str(data_path / ".coverage")},
    )
    junit_path = next(
        Path(arg.removeprefix("--junitxml="))
        for arg in result.command
        if arg.startswith("--junitxml=")
    )
    receipt = {
        "module": module,
        "pytest_rc": result.returncode,
        "seconds": round(time.monotonic() - started, 1),
        "receipt": result.receipt,
        "pytest_tail": result.stdout[-2000:],
        "pytest_err_tail": result.stderr[-2000:],
        **_failing_tests_field(junit_path),
    }
    if not result.success:
        return {
            **receipt,
            "status": "error",
            "reason": result.stderr[-2000:] or f"pytest rc={result.returncode}",
        }
    report_path = next(
        Path(arg.removeprefix("--cov-report=json:"))
        for arg in result.command
        if arg.startswith("--cov-report=json:")
    )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        measured = float(payload["totals"]["percent_covered"])
        if not math.isfinite(measured) or not 0 <= measured <= 100:
            raise ValueError("coverage percentage must be finite and between 0 and 100")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            **receipt,
            "status": "error",
            "reason": f"missing/invalid coverage report: {exc}",
        }
    return {**receipt, "status": "measured", "coverage_percent": round(measured, 1)}


def junit_failure_names(path: Path) -> list[str]:
    """Return ``classname::name`` for failing/erroring testcases in a JUnit report.

    Missing, empty, or malformed reports yield an empty list: failure-name
    extraction is best-effort enrichment and must never break the measurement
    contract. This mirrors the skip-rejection parsing in run_unified_tests.py
    (``junit_contract_errors``) and reports the same identifier shape.
    """
    if not path.is_file():
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


def _failing_tests_field(junit_path: Path) -> dict[str, list | list[dict[str, str]]]:
    """Return failing-test names + assertion texts, omitted when empty."""
    names = junit_failure_names(junit_path)
    if not names:
        return {}
    return {
        "failing_tests": names,
        "failing_details": junit_failure_details(junit_path),
    }


def junit_failure_details(path: Path) -> list[dict[str, str]]:
    """Return name + message + text for each failing/erroring testcase.

    Best-effort like :func:`junit_failure_names`: missing, empty, or
    malformed reports yield an empty list and never break the measurement
    contract. Texts are truncated so a few failures cannot flood the gate
    log.
    """
    if not path.is_file():
        return []
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    details: list[dict[str, str]] = []
    for testcase in root.iter("testcase"):
        node = testcase.find("failure")
        if node is None:
            node = testcase.find("error")
        if node is None:
            continue
        name = "::".join(
            part
            for part in (
                testcase.attrib.get("classname", ""),
                testcase.attrib.get("name", ""),
            )
            if part
        )
        text = ((node.text or "") + " " + (node.attrib.get("message", "")))[:500]
        details.append({"name": name, "text": text})
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", default="", help="comma-separated subset")
    parser.add_argument("--json", default="", help="write the results array here")
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_MODULE_WORKERS,
        help="module process budget (1..4); two pytest workers per module",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")

    if args.modules:
        modules = [
            name for name in (part.strip() for part in args.modules.split(",")) if name
        ]
    else:
        modules = [
            module.path.name if module.name != "ROOT" else "ROOT"
            for module in discover_workspace_test_targets(REPO_ROOT)
        ]
    if not modules:
        parser.error("coverage selection cannot be empty")
    known = {
        module.path.name if module.name != "ROOT" else "ROOT"
        for module in discover_workspace_test_targets(REPO_ROOT)
    }
    if unknown := sorted(set(modules) - known):
        parser.error(f"unknown coverage modules: {unknown}")
    results: list[dict] = []
    include_root = "ROOT" in modules
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(measure_module, module): module
            for module in modules
            if module != "ROOT"
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "module": futures[future],
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(json.dumps(result), flush=True)
    if include_root:
        try:
            result = measure_module("ROOT")
        except Exception as exc:
            result = {
                "module": "ROOT",
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(json.dumps(result), flush=True)
    results.sort(key=lambda entry: entry["module"])
    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=1, sort_keys=True), encoding="utf-8"
        )
    failures = [
        entry
        for entry in results
        if entry["status"] != "measured" or entry.get("pytest_rc") != 0
    ]
    if failures:
        print(f"coverage sweep: {len(failures)} module(s) errored", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
