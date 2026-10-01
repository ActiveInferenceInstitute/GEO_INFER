#!/usr/bin/env python3
"""Diff-scoped coverage-floor gate for the TEST-03 baseline.

Re-measures line coverage for every module whose ``src`` or ``tests``
changed between two revisions and fails when a touched module's measured
coverage falls below its recorded floor in
``GEO-INFER-TEST/coverage_baseline.json``.

Usage::

    uv run --with pytest-cov python GEO-INFER-TEST/check_coverage_floor.py \
        --base <sha> --head <sha> [--modules M1,M2]

``--modules`` forces a module set regardless of the diff (used for local
verification).  With no changed modules the gate passes immediately.
Failures are reported per module and the exit code is 1.  A FAILED-SUITE
verdict names the failing tests recorded in the measurement's JUnit report
so the CI log no longer hides the per-test detail behind a one-line summary.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from measure_module_coverage import measure_module

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = REPO_ROOT / "GEO-INFER-TEST" / "coverage_baseline.json"
DIFF_FILTER = "ACMRTUXB"
# GS19-01: release commits bump ``__version__`` in all 45 modules'
# ``__init__.py``; those literal-only changes are not code changes and must
# not trigger a full-fleet re-measurement.
VERSION_LINE = re.compile(r'^[+-]\s*__version__\s*=\s*["\'][^"\']*["\']')


def _fail(message: str) -> None:
    """Print a gate failure and exit non-zero."""
    print(f"coverage floor gate failure: {message}", file=sys.stderr)
    raise SystemExit(1)


def _version_only_file(base: str, head: str, path: str) -> bool:
    """Return True when every changed line in ``path`` is a version literal.

    Fail-open: any git error or non-literal change makes the file count as a
    content change so the module stays in the re-measurement set.
    """
    completed = subprocess.run(
        ["git", "diff", "-U0", base, head, "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return False
    changed = [
        line
        for line in completed.stdout.splitlines()
        if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
    ]
    return bool(changed) and all(VERSION_LINE.match(line) for line in changed)


def _changed_modules(base: str, head: str) -> set[str]:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            f"--diff-filter={DIFF_FILTER}",
            base,
            head,
            "--",
            "GEO-INFER-*/src/*.py",
            "GEO-INFER-*/tests/*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _fail(f"git diff failed: {completed.stderr[-400:]}")
    module_files: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        top = line.split("/", 1)[0]
        if top.startswith("GEO-INFER-"):
            module_files.setdefault(top, []).append(line)
    # GS19-01: a module whose only src change is its ``__version__`` literal
    # (release-commit version bumps) is not re-measured.
    return {
        module
        for module, files in module_files.items()
        if any(not _version_only_file(base, head, path) for path in files)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--modules", default="", help="force a module set")
    args = parser.parse_args(argv)

    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("modules", {})

    if args.modules:
        modules = {name for name in args.modules.split(",") if name}
    else:
        modules = _changed_modules(args.base, args.head)
    if not modules:
        print("no module changes; coverage floor gate passes")
        return 0

    all_modules = {
        path.name
        for path in REPO_ROOT.iterdir()
        if path.is_dir()
        and path.name.startswith("GEO-INFER-")
        and (path / "tests").is_dir()
    }
    missing = sorted(all_modules - set(entries))
    if missing:
        _fail(f"coverage_baseline.json lacks floors for: {missing}")

    known = set(entries)
    unknown = sorted(modules - known)
    if unknown:
        _fail(
            "coverage_baseline.json has no entry for diff-derived module(s) "
            f"{unknown}; the module was deleted or renamed without updating "
            "the baseline — add or drop its floor entry"
        )

    violations: list[str] = []
    for module in sorted(modules & known):
        floor = entries[module]["floor_percent"]
        result = measure_module(module)
        if result["status"] != "measured":
            violations.append(
                f"{module}: measurement failed ({result.get('reason', 'unknown')})"
            )
            continue
        pytest_rc = result.get("pytest_rc")
        if pytest_rc not in (None, 0):
            # GS19-01: one bounded retry — a full-fleet re-measurement under
            # filterwarnings=['error'] is a flake lottery; a single retry
            # absorbs a transient failure before the gate fails the release.
            print(
                f"{module}: measurement ran with pytest rc={pytest_rc}; "
                "retrying once (GS19-01 bounded retry)"
            )
            result = measure_module(module)
            if result["status"] != "measured":
                violations.append(
                    f"{module}: measurement failed ({result.get('reason', 'unknown')})"
                )
                continue
            pytest_rc = result.get("pytest_rc")
        measured = float(result["coverage_percent"])
        if pytest_rc not in (None, 0):
            # GS-004: pytest rc=1 with a coverage report means the suite
            # failed mid-measurement; the number is coverage of whatever ran
            # before the failure and must never read as a passing floor.
            print(f"{module}: measured {measured}% vs floor {floor}% -> FAILED-SUITE")
            tail = result.get("pytest_tail", "")
            if tail:
                # Surface the measurement's captured stdout tail: failing-test
                # diagnostics (e.g. parity-diff spies) live there, not in the
                # JUnit names alone.
                print(tail)
            failing = result.get("failing_tests", [])
            for name in failing[:20]:
                print(f"  FAILED {name}")
            if len(failing) > 20:
                print(f"  ... and {len(failing) - 20} more failing tests")
            for detail in result.get("failing_details", [])[:20]:
                print(f"  DETAIL {detail['name']}: {detail['text']}")
            violations.append(
                f"{module}: measurement ran with pytest rc={pytest_rc}"
                + (f" ({len(failing)} failing tests)" if failing else "")
                + "; failing tests make the measured floor unreliable"
            )
            continue
        verdict = "ok" if measured >= floor else "VIOLATION"
        print(f"{module}: measured {measured}% vs floor {floor}% -> {verdict}")
        if measured < floor:
            violations.append(f"{module}: {measured}% < floor {floor}%")
    if violations:
        for line in violations:
            print(f"coverage floor violation: {line}", file=sys.stderr)
        _fail(f"{len(violations)} module(s) below their coverage floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
