"""Unit tests for the per-module coverage measurement script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "GEO-INFER-TEST" / "src" / "geo_infer_test" / "coverage.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_test.coverage_for_regression", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JUNIT_WITH_FAILURES = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="1" failures="1" skipped="0" tests="3">
    <testcase classname="tests.unit.test_sample" name="test_ok" file="tests/unit/test_sample.py" line="1" />
    <testcase classname="tests.unit.test_sample" name="test_bad[param]" file="tests/unit/test_sample.py" line="5">
      <failure message="assert 1 == 2">traceback</failure>
    </testcase>
    <testcase classname="tests.unit.test_sample" name="test_broken" file="tests/unit/test_sample.py" line="9">
      <error message="boom">traceback</error>
    </testcase>
  </testsuite>
</testsuites>
"""

JUNIT_ALL_PASS = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1">
    <testcase classname="tests.unit.test_sample" name="test_ok" file="tests/unit/test_sample.py" line="1" />
  </testsuite>
</testsuites>
"""

EXPECTED_FAILURES = [
    "tests.unit.test_sample::test_bad[param]",
    "tests.unit.test_sample::test_broken",
]


def make_fake_module(root: Path) -> None:
    (root / "GEO-INFER-SAMPLE" / "src" / "geo_infer_sample").mkdir(parents=True)
    test = root / "GEO-INFER-SAMPLE" / "tests" / "unit" / "test_sample.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_sample(): assert True\n")


def write_fake_reports(command: list[str], junit_text: str, coverage: float) -> None:
    junit_args = [arg for arg in command if arg.startswith("--junitxml=")]
    assert junit_args, "--junitxml must be part of the pytest command"
    Path(junit_args[0].removeprefix("--junitxml=")).write_text(
        junit_text, encoding="utf-8"
    )
    report_args = [arg for arg in command if arg.startswith("--cov-report=json:")]
    assert report_args, "--cov-report=json must be part of the pytest command"
    Path(report_args[0].removeprefix("--cov-report=json:")).write_text(
        json.dumps({"totals": {"percent_covered": coverage}}), encoding="utf-8"
    )


def test_junit_failure_names_extracts_failures_and_errors(tmp_path):
    module = load_module()
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT_WITH_FAILURES, encoding="utf-8")

    assert module.junit_failure_names(junit) == EXPECTED_FAILURES


def test_junit_failure_names_tolerates_missing_and_malformed(tmp_path):
    module = load_module()
    assert module.junit_failure_names(tmp_path / "absent.xml") == []

    malformed = tmp_path / "junit.xml"
    malformed.write_text("<testsuites><oops>", encoding="utf-8")
    assert module.junit_failure_names(malformed) == []


def test_measure_module_reports_failing_tests_from_junit(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    make_fake_module(tmp_path)

    def fake_run(command, **kwargs):
        write_fake_reports(command, JUNIT_WITH_FAILURES, 55.0)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    def fake_command(command, name, timeout, cwd, env_overrides):
        completed = fake_run(command)
        from geo_infer_test.execution import CommandResult

        return CommandResult(
            name,
            completed.returncode == 0,
            0.1,
            command,
            completed.stdout,
            completed.stderr,
            timeout,
            completed.returncode,
        )

    monkeypatch.setattr(module, "run_command", fake_command)

    result = module.measure_module("GEO-INFER-SAMPLE")

    assert result["status"] == "error"
    assert result["pytest_rc"] == 1
    assert result["failing_tests"] == EXPECTED_FAILURES


def test_measure_module_clean_suite_has_no_failing_tests_field(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    make_fake_module(tmp_path)

    def fake_run(command, **kwargs):
        write_fake_reports(command, JUNIT_ALL_PASS, 90.0)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_command(command, name, timeout, cwd, env_overrides):
        completed = fake_run(command)
        from geo_infer_test.execution import CommandResult

        return CommandResult(
            name,
            completed.returncode == 0,
            0.1,
            command,
            completed.stdout,
            completed.stderr,
            timeout,
            completed.returncode,
        )

    monkeypatch.setattr(module, "run_command", fake_command)

    result = module.measure_module("GEO-INFER-SAMPLE")

    assert result["status"] == "measured"
    assert result["pytest_rc"] == 0
    assert "failing_tests" not in result


def test_measure_module_error_status_carries_failing_tests(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    make_fake_module(tmp_path)

    def fake_run(command, **kwargs):
        write_fake_reports(command, JUNIT_WITH_FAILURES, 0.0)
        return subprocess.CompletedProcess(command, 4, stdout="", stderr="boom")

    def fake_command(command, name, timeout, cwd, env_overrides):
        completed = fake_run(command)
        from geo_infer_test.execution import CommandResult

        return CommandResult(
            name,
            completed.returncode == 0,
            0.1,
            command,
            completed.stdout,
            completed.stderr,
            timeout,
            completed.returncode,
        )

    monkeypatch.setattr(module, "run_command", fake_command)

    result = module.measure_module("GEO-INFER-SAMPLE")

    assert result["status"] == "error"
    assert result["failing_tests"] == EXPECTED_FAILURES
