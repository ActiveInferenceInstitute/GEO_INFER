#!/usr/bin/env python3
"""ROOT-01 render-lane coverage metric for the root manuscript suite.

The ROOT-01 acceptance line requires every root test to have a recorded CI
run: the tracked workflows must jointly execute the full ``tests/`` battery
instead of silently excluding the render-dependent tests.  This script
measures that acceptance line against the current tree, deterministically
and offline:

1.  Collect the root test battery from ``tests/`` (the 109-test baseline).
2.  Parse every pytest selection found in the tracked GitHub Actions
    workflows (``.github/workflows/*.yml``), replay each selection through
    real pytest collection, and report the union of collected test ids as
    the number of root tests CI actually runs.
3.  Run the repository's own manuscript render path when it exists
    (``scripts/render_manuscript_pdf.py``) and execute the render-dependent
    tests against its artifacts, reporting how many pass.  The tests are
    _fail-closed by design: without a render they _fail, and the honest
    measurement is a low count, never a skip.

Metrics are printed one per line as ``METRIC name=value``; diagnostics as
``ASI key=value``.  The harness exits non-zero only when it cannot measure
(broken workflow syntax, failed collection, failed render), never because
the measured state falls short of the goal: the metric value carries that
signal.

Environment contract: the shared uv workspace is already synced
(``uv sync --all-packages --all-extras``); every pytest run uses
``uv run --no-sync`` so the workload never touches the network.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_manuscript_pdf.py"
ROOT_TEST_DIR = "tests/"
EXPECTED_ROOT_TESTS = 109
RENDER_DEPENDENT_TARGETS: tuple[str, ...] = (
    "tests/test_manuscript_pdf_layout.py",
    "tests/test_manuscript_paths.py::TestFigurePathLiterals"
    "::test_the_combined_document_keeps_the_rewrite_inside_image_targets",
)

# _Selection semantics this parser models.  A workflow that selects tests
# through anything else must extend this script deliberately rather than
# silently change what the coverage metric means.
FORBIDDEN_FLAGS = frozenset(
    {
        "-k",
        "-m",
        "-g",
        "--co",
        "--collect-only",
        "--lf",
        "--ff",
        "--nf",
        "--continue-on-collection-errors",
    }
)
# Flags that take a separate value but do not affect which tests run.
VALUE_FLAGS = frozenset({"-p", "--tb", "--maxfail"})
# Single-token flags that do not affect which tests run.
NOISE_FLAGS = frozenset({"-q", "-v", "--no-header", "-rA", "-ra"})

_COLLECT_PLUGIN = """\
def pytest_collection_finish(session):
    import os

    with open(os.environ["GEO_NODEID_FILE"], "w", encoding="utf-8") as handle:
        handle.write("\\n".join(item.nodeid for item in session.items))
"""

_OUTCOME_PLUGIN = """\
import json
import os

_OUT = os.environ["GEO_OUTCOME_FILE"]
_STATES = {}


def pytest_runtest_logreport(report):
    entry = _STATES.setdefault(
        report.nodeid, {"setup": "passed", "call": None, "teardown": "passed"}
    )
    entry[report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    passing = sorted(
        nodeid
        for nodeid, entry in _STATES.items()
        if entry["call"] == "passed"
        and entry["setup"] == "passed"
        and entry["teardown"] == "passed"
    )
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump({"passing": passing, "states": _STATES}, handle)
"""


@dataclass(frozen=True)
class _Selection:
    """One pytest selection found in a tracked workflow.

    ``targets`` are the positional pytest arguments (paths or node ids);
    an empty tuple means the invocation collected the default root test
    directory.  ``ignores`` and ``deselects`` are the ``--ignore`` and
    ``--deselect`` arguments exactly as the workflow writes them.
    """

    targets: tuple[str, ...]
    ignores: tuple[str, ...]
    deselects: tuple[str, ...]
    source: str

    def args(self) -> tuple[str, ...]:
        """The pytest arguments that replay this selection."""
        args: list[str] = list(self.targets) or [ROOT_TEST_DIR]
        for ignore in self.ignores:
            args.extend(["--ignore", ignore])
        for deselect in self.deselects:
            args.extend(["--deselect", deselect])
        return tuple(args)


def _fail(message: str) -> None:
    """Print a harness failure and exit non-zero."""
    print(f"harness failure: {message}", file=sys.stderr)
    raise SystemExit(1)


def _join_continuations(text: str) -> list[str]:
    """Join shell line continuations so each logical command is one line."""
    joined: list[str] = []
    buffer = ""
    for line in text.splitlines():
        if buffer:
            buffer = f"{buffer} {line.strip()}"
        else:
            buffer = line
        if buffer.rstrip().endswith("\\"):
            buffer = buffer.rstrip()[:-1]
            continue
        joined.append(buffer)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def _parse_selections(text: str, source: str) -> list[_Selection]:
    """Extract every root-suite pytest selection from one workflow file.

    An invocation qualifies when one of its positional targets or
    selection values addresses the root test directory (a path starting
    with ``tests/``).  Invocations aimed elsewhere (for example the
    standalone import probes under ``GEO-INFER-TEST/tests``) are skipped,
    so their unrelated flags never have to be modelled here.
    """
    selections: list[_Selection] = []
    for command in _join_continuations(text):
        stripped = command.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            _fail(f"{source}: cannot tokenize command: {exc}: {stripped!r}")
        start: int | None = None
        for index, token in enumerate(tokens):
            if token == "pytest" and (index == 0 or tokens[index - 1] == "-m"):
                start = index + 1
                break
        if start is None:
            continue
        targets: list[str] = []
        ignores: list[str] = []
        deselects: list[str] = []
        unknown: list[str] = []
        index = start
        while index < len(tokens):
            token = tokens[index]
            head = token.split("=", 1)[0]
            if head in ("--ignore", "--deselect"):
                if "=" in token:
                    value = token.split("=", 1)[1]
                    index += 1
                else:
                    if index + 1 >= len(tokens):
                        _fail(f"{source}: {head} is missing its value")
                    value = tokens[index + 1]
                    index += 2
                (ignores if head == "--ignore" else deselects).append(value)
            elif head in FORBIDDEN_FLAGS:
                unknown.append(token)
                index += 1
            elif head in VALUE_FLAGS:
                index += 2 if "=" not in token else 1
            elif head in NOISE_FLAGS:
                index += 1
            elif token.startswith("-"):
                unknown.append(token)
                index += 1
            else:
                targets.append(token)
                index += 1
        root_targeted = any(
            value.startswith(ROOT_TEST_DIR)
            for value in (*targets, *ignores, *deselects)
        )
        if not root_targeted:
            continue
        if unknown:
            _fail(
                f"{source}: pytest flag(s) {unknown} select or run tests in "
                "a way render_lane_metric.py does not model; extend the "
                "parser deliberately"
            )
        selections.append(
            _Selection(
                targets=tuple(targets),
                ignores=tuple(ignores),
                deselects=tuple(deselects),
                source=source,
            )
        )
    return selections


def _load_workflow_selections() -> list[_Selection]:
    """Parse every tracked workflow file for root-suite pytest selections."""
    if not WORKFLOW_DIR.is_dir():
        _fail(f"missing workflows directory: {WORKFLOW_DIR}")
    selections: list[_Selection] = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(
        WORKFLOW_DIR.glob("*.yaml")
    ):
        selections.extend(
            _parse_selections(path.read_text(encoding="utf-8"), path.name)
        )
    if not selections:
        _fail("no root-suite pytest selection found in any tracked workflow")
    return selections


def _run_pytest(
    args: Sequence[str], plugin: str, env_key: str, *, allow_test_failures: bool
) -> str:
    """Run pytest in a subprocess; return the plugin's captured output.

    ``allow_test_failures`` accepts exit code 1: for the render-dependent
    test run, failing tests are the measured signal, not a harness
    failure.  Every other non-zero exit is a harness failure.
    """
    with tempfile.TemporaryDirectory(prefix="render-lane-metric-") as tmp:
        plugin_path = Path(tmp) / "geo_nodeid_plugin.py"
        plugin_path.write_text(plugin, encoding="utf-8")
        out_path = Path(tmp) / "out.txt"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp) + os.pathsep + env.get("PYTHONPATH", "")
        env[env_key] = str(out_path)
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "pytest",
            "-p",
            "geo_nodeid_plugin",
            "-p",
            "no:cacheprovider",
            *args,
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        accepted = completed.returncode == 0 or (
            allow_test_failures and completed.returncode == 1
        )
        if not accepted or not out_path.is_file():
            _fail(
                "pytest failed for "
                f"{list(args)}: rc={completed.returncode} "
                f"stderr={completed.stderr[-2000:]}"
            )
        return out_path.read_text(encoding="utf-8")


def _collect_node_ids(args: Sequence[str]) -> frozenset[str]:
    """Collect test node ids for one pytest selection."""
    content = _run_pytest(
        ["--collect-only", "-q", *args],
        _COLLECT_PLUGIN,
        "GEO_NODEID_FILE",
        allow_test_failures=False,
    )
    ids = frozenset(line for line in content.splitlines() if line)
    if not ids:
        _fail(f"selection {list(args)} collected no tests")
    return ids


def _run_render_dependent_tests() -> tuple[int, int]:
    """Run the 7 render-dependent tests; return (passing, collected)."""
    payload = json.loads(
        _run_pytest(
            list(RENDER_DEPENDENT_TARGETS),
            _OUTCOME_PLUGIN,
            "GEO_OUTCOME_FILE",
            allow_test_failures=True,
        )
    )
    return len(payload["passing"]), len(payload["states"])


def _toolchain_presence() -> dict[str, int]:
    """Report which external render tools are on PATH (1 present, 0 absent)."""
    return {
        "xelatex": int(shutil.which("xelatex") is not None),
        "pdftotext": int(shutil.which("pdftotext") is not None),
        "pandoc": int(shutil.which("pandoc") is not None),
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    selections = _load_workflow_selections()
    distinct: dict[tuple[str, ...], _Selection] = {}
    for selection in selections:
        distinct.setdefault(selection.args(), selection)

    total_ids = _collect_node_ids([ROOT_TEST_DIR])
    covered: set[str] = set()
    for args in distinct:
        covered |= _collect_node_ids(args)

    toolchain = _toolchain_presence()
    rendered = False
    if RENDER_SCRIPT.is_file():
        completed = subprocess.run(
            ["uv", "run", "--no-sync", "python", str(RENDER_SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            _fail(
                "render path failed: scripts/render_manuscript_pdf.py exited "
                f"{completed.returncode}: {completed.stderr[-2000:]}"
            )
        rendered = True
    passing, collected = _run_render_dependent_tests()

    print(f"METRIC ci_root_tests_covered={len(covered)}")
    print(f"METRIC root_tests_total={len(total_ids)}")
    print(f"METRIC render_dependent_tests_local_passing={passing}")
    excluded = sorted(total_ids - covered)
    print(f"ASI workflows_parsed={len(selections)}")
    print(f"ASI distinct_selections={len(distinct)}")
    print(f"ASI render_path={'rendered' if rendered else 'absent'}")
    print(f"ASI render_dependent_collected={collected}")
    print(f"ASI expected_root_tests={EXPECTED_ROOT_TESTS}")
    print(f"ASI toolchain_xelatex={toolchain['xelatex']}")
    print(f"ASI toolchain_poppler={toolchain['pdftotext']}")
    print(f"ASI toolchain_pandoc={toolchain['pandoc']}")
    if excluded:
        preview = ", ".join(excluded[:8])
        more = "" if len(excluded) <= 8 else f" (+{len(excluded) - 8} more)"
        print(f"ASI excluded_from_ci={preview}{more}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
