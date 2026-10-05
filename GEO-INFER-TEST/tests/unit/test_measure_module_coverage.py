"""Unit tests for the per-module coverage measurement script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import os
from pathlib import Path

import psutil
import pytest

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


def test_duplicate_coverage_modules_cannot_duplicate_evidence(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "discover_workspace_test_targets",
        lambda _: [
            module.Module("SAMPLE", Path("GEO-INFER-SAMPLE"), Path("tests"), True)
        ],
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["--modules", "GEO-INFER-SAMPLE,GEO-INFER-SAMPLE"])
    assert caught.value.code == 2


def test_completed_measurement_cannot_make_interrupted_fleet_pass(
    tmp_path, monkeypatch
):
    """Completed measurements remain truthful when reporting is interrupted."""
    from geo_infer_test import execution

    module = load_module()
    monkeypatch.setattr(
        module,
        "discover_workspace_test_targets",
        lambda _: [module.Module("DONE", Path("GEO-INFER-DONE"), Path("tests"), True)],
    )
    monkeypatch.setattr(
        module,
        "measure_module",
        lambda name: {
            "module": name,
            "status": "measured",
            "pytest_rc": 0,
            "coverage_percent": 100.0,
        },
    )

    def interrupted_after_completion(futures):
        for future in futures:
            future.result()
        raise KeyboardInterrupt()
        yield

    monkeypatch.setattr(execution, "as_completed", interrupted_after_completion)
    output = tmp_path / "results.json"
    assert (
        module.main(
            ["--modules", "GEO-INFER-DONE", "--workers", "2", "--json", str(output)]
        )
        == 130
    )
    rows = json.loads(output.read_text())
    assert rows[0]["status"] == "measured"
    assert rows[0]["pytest_rc"] == 0
    assert rows[0]["fleet_diagnostics"][0]["status"] == "INTERRUPTED"


@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("entrypoint", ["measurement", "floor"])
def test_interrupted_coverage_reaps_children_and_retains_every_module(
    tmp_path, cleanup_failure, entrypoint
):
    """Real cancellation retains output and prevents queued/root measurements."""
    script = tmp_path / "interrupt_coverage.py"
    script.write_text(
        """from pathlib import Path
from types import SimpleNamespace
import importlib.util, json, os, sys, time
from geo_infer_test import coverage, execution
root = Path(__file__).parent
execution.RESULTS_DIR = root / "attempts"
coverage.discover_workspace_test_targets = lambda _: [
    SimpleNamespace(name=name, path=Path("GEO-INFER-" + name))
    for name in ("FIRST", "SECOND", "THIRD", "ROOT")
]
def measure(name):
    marker = root / (name + ".pid")
    child = "import time; time.sleep(30)"
    code = f"from pathlib import Path; import os,subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=os.name=='posix'); Path({str(marker)!r}).write_text(str(child.pid)); print('retained '+{name!r},flush=True); time.sleep(30)"
    result = execution.run_command([sys.executable,"-c",code], name, 40, cwd=root)
    return {"module":name,"status":"measured" if result.success else "error",
            "pytest_rc":result.returncode,"receipt":result.receipt,
            "seconds":result.duration,"command_status":result.status}
coverage.measure_module = measure
if CLEANUP_FAILURE:
    original_cleanup = execution.terminate_running_processes
    def failed_cleanup():
        original_cleanup()
        raise RuntimeError("independent cleanup diagnosis")
    execution.terminate_running_processes = failed_cleanup
def interrupted(futures):
    deadline=time.monotonic()+15
    while not all((root/(name+".pid")).exists() for name in ("GEO-INFER-FIRST","GEO-INFER-SECOND")):
        if time.monotonic() >= deadline: raise AssertionError("Owned children did not start")
        time.sleep(.01)
    raise KeyboardInterrupt()
    yield
execution.as_completed = interrupted
arguments = ["--modules","GEO-INFER-FIRST,GEO-INFER-SECOND,GEO-INFER-THIRD,ROOT",
             "--workers","2","--json",str(root/"results.json")]
if ENTRYPOINT == "floor":
    spec = importlib.util.spec_from_file_location("interrupted_floor_gate", FLOOR_SCRIPT)
    floor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(floor)
    floor.REPO_ROOT = root
    floor.MANIFEST = root / "baseline.json"
    floor.MANIFEST.write_text(json.dumps({"modules": {
        "GEO-INFER-" + name: {"floor_percent":50}
        for name in ("FIRST", "SECOND", "THIRD")
    }}))
    floor.discover_workspace_test_targets = coverage.discover_workspace_test_targets
    floor.measure_module = measure
    status = floor.main(["--base","HEAD","--head","HEAD",*arguments])
else:
    status = coverage.main(arguments)
print(json.dumps({"exit_status":status}))
""".replace("CLEANUP_FAILURE", repr(cleanup_failure))
        .replace("ENTRYPOINT", repr(entrypoint))
        .replace(
            "FLOOR_SCRIPT",
            repr(str(REPO_ROOT / "GEO-INFER-TEST/check_coverage_floor.py")),
        ),
        encoding="utf-8",
    )
    from geo_infer_test.process import run_process

    completed = run_process(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=35,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout.splitlines()[-1]) == {"exit_status": 130}
    rows = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [row["module"] for row in rows] == [
        "GEO-INFER-FIRST",
        "GEO-INFER-SECOND",
        "GEO-INFER-THIRD",
        "ROOT",
    ]
    assert all(row["status"] == "error" for row in rows)
    assert all(
        any(item["status"] == "INTERRUPTED" for item in row["fleet_diagnostics"])
        for row in rows
    )
    if cleanup_failure:
        assert all(
            any(
                item["status"] == "FAIL"
                and "independent cleanup diagnosis" in item["stderr"]
                for item in row["fleet_diagnostics"]
            )
            for row in rows
        )
    for row in rows[:2]:
        receipt = json.loads(Path(row["receipt"]).read_text(encoding="utf-8"))
        assert receipt["success"] is False and receipt["returncode"] != 0
        assert "retained " + row["module"] in (
            Path(row["receipt"]).parent / "stdout.log"
        ).read_text(encoding="utf-8")
        pid = int((tmp_path / (row["module"] + ".pid")).read_text())
        try:
            child = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    assert not (tmp_path / "GEO-INFER-THIRD.pid").exists()
    assert not (tmp_path / "ROOT.pid").exists()
    assert all(
        "receipt" not in row and "interrupted" in row["reason"] for row in rows[2:]
    )
