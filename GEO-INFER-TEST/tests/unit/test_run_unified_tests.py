"""Unit tests for the unified test runner script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "GEO-INFER-TEST" / "src" / "geo_infer_test" / "execution.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_test.execution_for_regression", RUNNER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_module(root: Path, name: str) -> Path:
    src_path = root / f"GEO-INFER-{name}" / "src"
    src_path.mkdir(parents=True)
    return src_path


def make_test_module(root: Path, name: str) -> Path:
    module_path = root / f"GEO-INFER-{name}"
    (module_path / "tests" / "performance").mkdir(parents=True)
    (module_path / "tests" / "unit").mkdir()
    return module_path


def test_workspace_src_paths_are_sorted(tmp_path, monkeypatch):
    runner = load_runner_module()
    b_src = make_module(tmp_path, "B")
    a_src = make_module(tmp_path, "A")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    assert runner.workspace_src_paths() == [a_src, b_src]


def test_subprocess_env_prepends_workspace_src_paths(tmp_path, monkeypatch):
    runner = load_runner_module()
    a_src = make_module(tmp_path, "A")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    env = runner.build_subprocess_env()

    assert env["PYTHONPATH"].split(runner.os.pathsep)[:2] == [
        str(a_src),
        "/existing/path",
    ]


def test_module_discovery_ignores_non_test_files(tmp_path, monkeypatch):
    runner = load_runner_module()
    module_path = tmp_path / "GEO-INFER-SAMPLE"
    tests_path = module_path / "tests"
    tests_path.mkdir(parents=True)
    (tests_path / "conftest.py").write_text("# fixtures only\n")
    (tests_path / "README.md").write_text("Not a test file.\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    [module] = runner.discover_geo_infer_modules()

    assert module.has_tests is False


def test_unit_category_includes_root_test_files(tmp_path, monkeypatch):
    runner = load_runner_module()
    module_path = make_test_module(tmp_path, "SAMPLE")
    root_test = module_path / "tests" / "test_root_layout.py"
    root_test.write_text("def test_root_layout():\n    assert True\n")
    unit_test = module_path / "tests" / "unit" / "test_unit_layout.py"
    unit_test.write_text("def test_unit_layout():\n    assert True\n")
    # The retired tests/tools/ layout is no longer a unit-category source.
    tools_dir = module_path / "tests" / "tools"
    tools_dir.mkdir()
    (tools_dir / "test_tool_layout.py").write_text("def test_tool():\n    pass\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    module = runner.discover_geo_infer_modules()[0]

    assert runner.category_test_paths(module, "unit") == sorted([root_test, unit_test])
    assert runner.category_test_paths(module, "integration") == []
    assert runner.category_test_paths(module, "system") == []


def test_new_run_preserves_earlier_attempt_artifacts(tmp_path, monkeypatch):
    runner = load_runner_module()
    results_dir = tmp_path / "results"
    stale_nested = results_dir / "old-run"
    stale_nested.mkdir(parents=True)
    (stale_nested / "summary.json").write_text("{}\n")
    (results_dir / "stale.xml").write_text("<testsuites />\n")
    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)

    runner.ensure_results_dir(clean=True)

    assert (stale_nested / "summary.json").read_text() == "{}\n"
    assert (results_dir / "stale.xml").exists()
    assert runner.run_results_dir().is_dir()


def test_run_command_passes_workspace_env(tmp_path, monkeypatch):
    runner = load_runner_module()
    src_path = make_module(tmp_path, "A")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner, "run_process", fake_run)
    monkeypatch.setattr(runner, "runtime_receipt", lambda **_kwargs: {})

    result = runner.run_command(
        [sys.executable, "-c", "pass"],
        "sample",
        timeout=10,
        cwd=tmp_path,
        env_overrides={"COVERAGE_FILE": "/tmp/coverage-contract"},
    )

    assert result.success is True
    assert str(src_path) in captured["env"]["PYTHONPATH"]
    assert captured["env"]["COVERAGE_FILE"] == "/tmp/coverage-contract"


def test_pytest_no_tests_exit_is_a_failure(tmp_path, monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            runner.PYTEST_NO_TESTS_EXIT_CODE,
            stdout="collected 0 items\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "run_process", fake_run)
    monkeypatch.setattr(runner, "runtime_receipt", lambda **_kwargs: {})

    result = runner.run_command(
        [sys.executable, "-m", "pytest"],
        "empty tests",
        timeout=10,
        cwd=tmp_path,
    )

    assert result.success is False
    assert "pytest collected no tests" in result.stderr


def test_non_pytest_no_tests_exit_remains_failure(tmp_path, monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            runner.PYTEST_NO_TESTS_EXIT_CODE,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(runner, "run_process", fake_run)

    result = runner.run_command(
        ["python", "script.py"], "script", timeout=10, cwd=tmp_path
    )

    assert result.success is False


def test_write_summary_decodes_timeout_output_bytes(tmp_path, monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    report = runner.SuiteReport(
        results=[
            runner.CommandResult(
                name="timed out command",
                success=False,
                duration=10.0,
                command=["python", "script.py"],
                stdout=b"partial stdout",
                stderr=b"partial stderr\xff",
            )
        ]
    )

    runner.write_summary(report)

    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    result = summary["results"][0]
    assert result["stdout_tail"] == "partial stdout"
    assert result["stderr_tail"] == "partial stderr\ufffd"


def test_performance_category_uses_canonical_directory_only(tmp_path, monkeypatch):
    runner = load_runner_module()
    module_path = make_test_module(tmp_path, "SAMPLE")
    performance_file = module_path / "tests" / "performance" / "test_benchmark.py"
    performance_file.write_text("def test_benchmark():\n    assert True\n")
    unit_file = module_path / "tests" / "unit" / "test_performance_monitor.py"
    unit_file.write_text("def test_unit_monitor():\n    assert True\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ensure_results_dir", lambda clean=False: None)

    captured = []

    def fake_run(command, name, timeout, cwd=runner.PROJECT_ROOT, **kwargs):
        captured.append((name, command, timeout, cwd))
        return runner.CommandResult(
            name=name, success=True, duration=0.0, command=command
        )

    monkeypatch.setattr(runner, "run_command", fake_run)

    report = runner.run_performance_tests(timeout=42)

    assert report.success is True
    assert len(captured) == 1
    assert str(performance_file) in captured[0][1]
    assert str(unit_file) not in captured[0][1]


def test_coverage_category_isolates_modules_and_combines_data(tmp_path, monkeypatch):
    runner = load_runner_module()
    a_src = make_module(tmp_path, "A")
    b_src = make_module(tmp_path, "B")
    for module_name in ("A", "B"):
        test_file = (
            tmp_path / f"GEO-INFER-{module_name}" / "tests" / "unit" / "test_sample.py"
        )
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_sample():\n    assert True\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    runner.RESULTS_DIR.mkdir()
    stale_report = runner.RESULTS_DIR / "OLD_coverage_results.xml"
    stale_report.write_text("stale")
    captured = []

    def fake_run(
        command,
        name,
        timeout,
        cwd=runner.PROJECT_ROOT,
        env_overrides=None,
    ):
        captured.append((name, command, timeout, cwd, env_overrides))
        receipt = tmp_path / f"attempt-{len(captured)}" / "receipt.json"
        receipt.parent.mkdir()
        if "--cov=" in " ".join(command):
            (receipt.parent / ".coverage").write_bytes(b"bound fixture data")
        return runner.CommandResult(
            name=name, success=True, duration=0.0, command=command, receipt=str(receipt)
        )

    monkeypatch.setattr(runner, "run_command", fake_run)

    report = runner.run_coverage_analysis(timeout=42)

    assert report.success is True
    assert stale_report.read_text() == "stale"
    assert len(captured) == 5
    a_run, b_run, combine_run, json_run, terminal_run = captured
    assert f"--cov={a_src}" in a_run[1]
    assert f"--cov={b_src}" not in a_run[1]
    assert f"--cov={b_src}" in b_run[1]
    assert f"--cov={a_src}" not in b_run[1]
    assert "--cov-append" not in a_run[1]
    assert any(arg.startswith("--cov-report=json:") for arg in a_run[1])
    assert "--keep" in combine_run[1]
    assert len([arg for arg in combine_run[1] if arg.endswith("/.coverage")]) == 2
    expected_data = str(runner.run_results_dir() / ".coverage")
    assert a_run[4] == {"COVERAGE_FILE": expected_data}
    assert b_run[4] == {"COVERAGE_FILE": expected_data}
    assert json_run[1][-2:] == ["-o", str(runner.run_results_dir() / "coverage.json")]
    assert terminal_run[1][-2:] == ["report", "--show-missing"]


def test_parse_args_fail_fast_defaults_off(monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr("sys.argv", ["run_unified_tests.py"])
    assert runner.parse_args().fail_fast is False


def test_parse_args_fail_fast_flag_enables_stopping(monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr(
        "sys.argv", ["run_unified_tests.py", "--category", "unit", "--fail-fast"]
    )
    args = runner.parse_args()
    assert args.fail_fast is True
    assert args.category == "unit"


def test_fail_fast_stops_after_first_module_failure(tmp_path, monkeypatch):
    runner = load_runner_module()
    make_module(tmp_path, "A")
    make_module(tmp_path, "B")
    for module_name in ("A", "B"):
        test_file = (
            tmp_path / f"GEO-INFER-{module_name}" / "tests" / "unit" / "test_sample.py"
        )
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_sample():\n    assert True\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ensure_results_dir", lambda clean=False: None)
    captured = []

    def fake_run(command, name, timeout, cwd=runner.PROJECT_ROOT, **kwargs):
        captured.append(name)
        success = not any("GEO-INFER-A" in str(part) for part in command)
        return runner.CommandResult(
            name=name, success=success, duration=0.0, command=command
        )

    monkeypatch.setattr(runner, "run_command", fake_run)

    report = runner.run_module_category_tests("unit", timeout=1, fail_fast=True)

    assert report.success is False
    assert len(captured) == 1
    assert captured[0] == "A unit tests"


def test_default_behavior_runs_all_modules_despite_failure(tmp_path, monkeypatch):
    runner = load_runner_module()
    make_module(tmp_path, "A")
    make_module(tmp_path, "B")
    for module_name in ("A", "B"):
        test_file = (
            tmp_path / f"GEO-INFER-{module_name}" / "tests" / "unit" / "test_sample.py"
        )
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_sample():\n    assert True\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ensure_results_dir", lambda clean=False: None)
    captured = []

    def fake_run(command, name, timeout, cwd=runner.PROJECT_ROOT, **kwargs):
        captured.append(name)
        success = not any("GEO-INFER-A" in str(part) for part in command)
        return runner.CommandResult(
            name=name, success=success, duration=0.0, command=command
        )

    monkeypatch.setattr(runner, "run_command", fake_run)

    report = runner.run_module_category_tests("unit", timeout=1, fail_fast=False)

    assert report.success is False
    assert len(captured) == 2
    assert captured[0] == "A unit tests"
    assert captured[1] == "B unit tests"


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

EXPECTED_FAILURES = [
    "tests.unit.test_sample::test_bad[param]",
    "tests.unit.test_sample::test_broken",
]


def make_failed_result(runner, tmp_path, *, with_junit: bool = True):
    junit_path = tmp_path / "results" / "SEC_results.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.write_text(JUNIT_WITH_FAILURES, encoding="utf-8")
    command = [sys.executable, "-m", "pytest"]
    if with_junit:
        command.append(f"--junitxml={junit_path}")
    return runner.CommandResult(
        name="SEC tests",
        success=False,
        duration=1.0,
        command=command,
        stderr="1 failed, 2 passed in 0.5s",
    )


def test_junit_failure_names_extracts_failures_and_errors(tmp_path):
    runner = load_runner_module()
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT_WITH_FAILURES, encoding="utf-8")

    assert runner.junit_failure_names(junit) == EXPECTED_FAILURES


def test_junit_failure_names_tolerates_missing_and_malformed(tmp_path):
    runner = load_runner_module()
    assert runner.junit_failure_names(None) == []
    assert runner.junit_failure_names(tmp_path / "absent.xml") == []

    malformed = tmp_path / "junit.xml"
    malformed.write_text("<testsuites><oops>", encoding="utf-8")
    assert runner.junit_failure_names(malformed) == []


def test_parse_args_show_failures_defaults_off(monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr("sys.argv", ["run_unified_tests.py"])
    assert runner.parse_args().show_failures is False


def test_parse_args_show_failures_flag_enables_detail(monkeypatch):
    runner = load_runner_module()
    monkeypatch.setattr("sys.argv", ["run_unified_tests.py", "--show-failures"])
    assert runner.parse_args().show_failures is True


def test_write_summary_show_failures_prints_junit_failure_names(
    tmp_path, monkeypatch, capsys
):
    runner = load_runner_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)
    report = runner.SuiteReport()
    report.add(make_failed_result(runner, tmp_path))

    runner.write_summary(report, show_failures=True)

    out = capsys.readouterr().out
    assert "== Failing tests" in out
    assert "-- SEC tests" in out
    assert "FAILED tests.unit.test_sample::test_bad[param]" in out
    summary = json.loads((results_dir / "summary.json").read_text())
    assert summary["results"][0]["failures"] == EXPECTED_FAILURES


def test_write_summary_without_show_failures_keeps_verdict_compact(
    tmp_path, monkeypatch, capsys
):
    runner = load_runner_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)
    report = runner.SuiteReport()
    report.add(make_failed_result(runner, tmp_path))

    runner.write_summary(report)

    out = capsys.readouterr().out
    assert "== Failing tests" not in out
    assert "FAILED tests.unit.test_sample" not in out
    summary = json.loads((results_dir / "summary.json").read_text())
    assert summary["results"][0]["failures"] == EXPECTED_FAILURES


def test_write_summary_show_failures_falls_back_to_output_tail(
    tmp_path, monkeypatch, capsys
):
    runner = load_runner_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)
    report = runner.SuiteReport()
    report.add(make_failed_result(runner, tmp_path, with_junit=False))

    runner.write_summary(report, show_failures=True)

    out = capsys.readouterr().out
    assert "== Failing tests" in out
    assert "-- SEC tests" in out
    assert "1 failed, 2 passed in 0.5s" in out
