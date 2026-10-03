"""Real subprocess regressions for test evidence, inventory, and deadlines."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from geo_infer_test import execution
from geo_infer_test.process import run_process


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "core.fsmonitor", "false"], cwd=tmp_path, check=True
    )
    (tmp_path / ".gitignore").write_text("receipts/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(execution, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setattr(execution, "RESULTS_DIR", tmp_path / "receipts")
    return execution


def test_validator_failure_without_junit_keeps_receipt(engine) -> None:
    result = engine.run_command(
        [sys.executable, "-c", "raise SystemExit(7)"], "validator", 5
    )
    assert not result.success
    assert result.returncode == 7
    receipt = json.loads(Path(result.receipt).read_text())
    assert receipt["returncode"] == 7
    assert len(list(Path(result.receipt).parent.parent.iterdir())) == 1


def test_missing_dirty_state_is_incomplete_source_custody(engine, monkeypatch):
    """A usable revision cannot replace a failed working-tree inventory."""
    original = engine.run_process

    def failed_status(command, **kwargs):
        if command[-2:] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 1, "", "status unavailable")
        return original(command, **kwargs)

    monkeypatch.setattr(engine, "run_process", failed_status)
    receipt = engine.runtime_receipt(timeout=10)
    assert receipt["revision"] and receipt["dirty"] is None
    assert receipt["custody_complete"] is False


def test_staged_only_source_bytes_are_bound_to_receipt(engine):
    """Two staged versions cannot share revision/status and source custody."""
    source = engine.PROJECT_ROOT / " staged source.py"
    source.write_text("value = 1\n")
    subprocess.run(
        ["git", "add", "--", source.name], cwd=engine.PROJECT_ROOT, check=True
    )
    first = engine.runtime_receipt(timeout=10)
    source.write_text("value = 2\n")
    subprocess.run(
        ["git", "add", "--", source.name], cwd=engine.PROJECT_ROOT, check=True
    )
    second = engine.runtime_receipt(timeout=10)
    assert first["custody_complete"] and second["custody_complete"]
    assert first["revision"] == second["revision"] and first["dirty"] == second["dirty"]
    assert source.name in first["dirty_sha256"]
    assert first["dirty_sha256"][source.name] != second["dirty_sha256"][source.name]


def test_native_invalid_output_bytes_preserve_receipt(engine) -> None:
    result = engine.run_command(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'valid stdout\\n\\xff\\n'); os.write(2, b'valid stderr\\n\\xfe\\n')",
        ],
        "native diagnostic bytes",
        5,
    )
    assert result.success and result.returncode == 0
    assert result.stdout == "valid stdout\n\ufffd\n"
    assert result.stderr == "valid stderr\n\ufffd\n"
    attempt = Path(result.receipt).parent
    assert (attempt / "stdout.log").read_text() == result.stdout
    assert (attempt / "stderr.log").read_text() == result.stderr


@pytest.mark.parametrize("kind", ["runtime", "value", "process-access"])
def test_unexpected_process_failure_retains_failed_receipt(
    engine, monkeypatch, kind
) -> None:
    import psutil

    error = {
        "runtime": RuntimeError("unexpected runtime failure"),
        "value": ValueError("unexpected value failure"),
        "process-access": psutil.AccessDenied(pid=123),
    }[kind]
    error.output = "retained target output"
    error.stderr = "retained target diagnostics"
    original = engine.run_process

    def fail_target(command, **kwargs):
        if command == [sys.executable, "-c", "pass"]:
            raise error
        return original(command, **kwargs)

    monkeypatch.setattr(engine, "run_process", fail_target)
    result = engine.run_command(
        [sys.executable, "-c", "pass"], "unexpected child failure", 10
    )
    assert not result.success and result.status == "FAIL"
    receipt = json.loads(Path(result.receipt).read_text())
    assert receipt["success"] is False and receipt["returncode"] is None
    assert result.stdout == "retained target output"
    assert "retained target diagnostics" in result.stderr
    assert type(error).__name__ in result.stderr


@pytest.mark.parametrize(
    "content", [None, "<broken>", '<testsuites><testsuite tests="0"/></testsuites>']
)
def test_zero_exit_needs_current_nonempty_valid_junit(
    engine, tmp_path, content
) -> None:
    stale = tmp_path / "junit.xml"
    stale.write_text(
        '<testsuites><testsuite tests="1"><testcase name="old"/></testsuite></testsuites>'
    )
    body = (
        "pass"
        if content is None
        else f"import pathlib,sys; pathlib.Path(sys.argv[1].split('=',1)[1]).write_text({content!r})"
    )
    result = engine.run_command(
        [sys.executable, "-c", body, f"--junitxml={stale}"], "missing evidence", 5
    )
    assert not result.success
    assert stale.read_text().find('name="old"') >= 0
    assert Path(result.receipt).is_file()


def _pytest_command(tmp_path: Path, body: str) -> list[str]:
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nfilterwarnings = error\n")
    test = tmp_path / "test_child.py"
    test.write_text(body)
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "geo_infer_test.selection",
        "-c",
        str(config),
        str(test),
        f"--junitxml={tmp_path / 'junit.xml'}",
    ]


def test_real_pytest_records_selection_and_immutable_attempts(engine, tmp_path) -> None:
    command = _pytest_command(tmp_path, "def test_value():\n    assert 2 + 3 == 5\n")
    first = engine.run_command(command, "pass", 20)
    original = Path(first.receipt).read_bytes()
    second = engine.run_command(command, "pass again", 20)
    assert first.success and second.success
    assert first.executed == second.executed == 1
    assert first.receipt != second.receipt
    assert Path(first.receipt).read_bytes() == original
    receipt = json.loads(original)
    assert receipt["selection"]["collected"] == receipt["selection"]["executed"]
    assert receipt["artifacts"]["junit.xml"]


def test_silent_collection_removal_fails(engine, tmp_path) -> None:
    (tmp_path / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items.pop()\n"
    )
    command = _pytest_command(
        tmp_path, "def test_one(): assert True\ndef test_two(): assert True\n"
    )
    result = engine.run_command(command, "silently omitted", 20)
    assert not result.success
    assert "unaccounted test deselection" in result.stderr


def test_junit_truncated_after_pytest_cannot_certify_complete_execution(
    engine, tmp_path
) -> None:
    (tmp_path / "conftest.py").write_text("""import atexit
from pathlib import Path
import xml.etree.ElementTree as ET
def pytest_configure(config):
    def truncate():
        path = Path(config.option.xmlpath)
        tree = ET.parse(path)
        suite = next(tree.getroot().iter('testsuite'))
        suite.remove(list(suite.iter('testcase'))[-1])
        suite.set('tests', '1')
        tree.write(path)
    atexit.register(truncate)
""")
    command = _pytest_command(
        tmp_path, "def test_one(): assert True\ndef test_two(): assert True\n"
    )
    result = engine.run_command(command, "truncated report", 30)
    assert result.returncode == 0 and not result.success
    assert "JUnit/selection executed count mismatch" in result.stderr
    assert "testcase identities do not match" in result.stderr


def test_unreadable_junit_enrichment_retains_failure(tmp_path, monkeypatch) -> None:
    report = tmp_path / "unreadable.xml"
    report.write_text(
        '<testsuites><testsuite tests="1"><testcase name="broken"><failure/></testcase></testsuite></testsuites>'
    )

    def unreadable(_path):
        raise PermissionError("report is unreadable")

    monkeypatch.setattr(execution.ET, "parse", unreadable)
    assert execution.junit_failure_names(report) == []
    assert any(
        "unreadable" in error for error in execution.junit_contract_errors(report)
    )


def test_empty_slow_complement_is_neutral_but_empty_suite_fails(
    engine, tmp_path
) -> None:
    command = _pytest_command(tmp_path, "def test_fast(): assert True\n")
    result = engine.run_command(
        [*command, "-m", "slow"], "empty complement", 20, allow_empty=True
    )
    assert result.status == "EMPTY"
    assert not engine.SuiteReport([result]).success
    assert not engine.SuiteReport().success


def _assert_recorded_process_dead(pidfile: Path) -> None:
    """Verify cleanup after return, independently of work inside its budget."""
    import psutil

    assert pidfile.is_file(), "the parent must have launched the real descendant"
    try:
        child = psutil.Process(int(pidfile.read_text()))
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_timeout_terminates_real_descendant(tmp_path) -> None:
    pidfile = tmp_path / "descendant.pid"
    child = "import time; time.sleep(30)"
    parent = f"from pathlib import Path; import subprocess,sys,time; child = subprocess.Popen([sys.executable,'-c',{child!r}]); Path({str(pidfile)!r}).write_text(str(child.pid)); time.sleep(30)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", parent], timeout=0.3, cwd=tmp_path)
    assert time.monotonic() - started < 5
    _assert_recorded_process_dead(pidfile)


def test_runtime_registry_covers_every_workspace_module() -> None:
    from geo_infer_test.core.test_discoverer import ALL_MODULES

    assert set(ALL_MODULES) == {
        module.name for module in execution.discover_geo_infer_modules()
    }
    assert "INSURANCE" in ALL_MODULES


def test_programmatic_discovery_includes_root_and_registered_extra_files() -> None:
    from geo_infer_test.core.test_runner import (
        GeoInferTestRunner,
        TestConfiguration as RunnerConfiguration,
    )

    runner = GeoInferTestRunner(
        RunnerConfiguration(["PLACE"], ["unit"], log_integration_enabled=False)
    )
    discovered = runner.discover_tests()["PLACE"]
    expected = execution.category_test_paths(execution.module_by_name("PLACE"), "unit")
    assert len(discovered) == len(expected)
    assert any("locations/cascadia/tests/unit" in name for name in discovered)


def test_root_manuscript_profiles_execute_and_account_for_render_complement(
    engine, tmp_path
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nfilterwarnings = ['error']\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / engine.ROOT_RENDER_FILE).write_text(
        "from pathlib import Path\ndef test_actual_render(): assert Path('render-ready').exists()\n"
    )
    (tests / "test_manuscript_paths.py").write_text("""from pathlib import Path
def test_source_paths(): assert 3 * 7 == 21
class TestFigurePathLiterals:
    def test_the_combined_document_keeps_the_rewrite_inside_image_targets(self):
        assert Path('render-ready').exists()
""")
    root = engine.module_by_name("ROOT")
    assert root.path == tmp_path
    assert engine.category_test_paths(root, "unit") == []
    assert len(engine.module_test_files(root)) == 2
    ordinary = engine.run_module_category_tests("manuscript", 30, workers=1)
    assert ordinary.success and sum(result.executed for result in ordinary.results) == 1
    ordinary_receipt = json.loads(Path(ordinary.results[0].receipt).read_text())
    assert len(ordinary_receipt["selection"]["deselected"]) == 1
    (tmp_path / "render-ready").touch()
    rendered = engine.run_module_category_tests("manuscript-render", 30, workers=1)
    assert rendered.success and sum(result.executed for result in rendered.results) == 2
    rendered_receipt = json.loads(Path(rendered.results[0].receipt).read_text())
    assert len(rendered_receipt["selection"]["deselected"]) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeouts_fail_before_spawning(engine, timeout) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        engine.run_command([sys.executable, "-c", "pass"], "bad timeout", timeout)


def test_xdist_cannot_hide_worker_collection_removal(engine, tmp_path) -> None:
    (tmp_path / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items.pop()\n"
    )
    command = _pytest_command(
        tmp_path, "def test_one(): assert True\ndef test_two(): assert True\n"
    )
    result = engine.run_command(
        [*command, "-n", "2", "-p", "xdist.plugin"], "worker silently omitted", 30
    )
    assert not result.success
    receipt = json.loads(Path(result.receipt).read_text())
    assert len(receipt["selection"]["collected"]) == 2
    assert len(receipt["selection"]["workers"]) == 2
    assert len(receipt["selection"]["unaccounted"]) == 1


def test_xdist_marker_deselection_is_explicit_and_complete(engine, tmp_path) -> None:
    command = _pytest_command(
        tmp_path,
        "import pytest\ndef test_fast(): assert True\n@pytest.mark.slow\ndef test_slow(): assert True\n",
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = slow: complement\nfilterwarnings = error\n"
    )
    result = engine.run_command(
        [*command, "-n", "2", "-m", "not slow", "-p", "xdist.plugin"],
        "worker explicit deselection",
        30,
    )
    assert result.success and result.executed == 1
    receipt = json.loads(Path(result.receipt).read_text())
    assert len(receipt["selection"]["collected"]) == 2
    assert len(receipt["selection"]["deselected"]) == 1
    assert receipt["selection"]["unaccounted"] == []


def test_coverage_report_bytes_belong_to_attempt_receipt(engine, tmp_path) -> None:
    command = _pytest_command(
        tmp_path,
        "def test_fast():\n    from geo_infer_test.execution import run_results_dir\n    assert run_results_dir().name\n",
    )
    result = engine.run_command(
        [
            *command,
            "-p",
            "pytest_cov.plugin",
            f"--cov={execution.__name__}",
            f"--cov-report=json:{tmp_path / 'unbound.json'}",
            "--cov-fail-under=0",
        ],
        "bound coverage",
        30,
    )
    assert result.success
    receipt = json.loads(Path(result.receipt).read_text())
    assert receipt["artifacts"]["coverage.json"]
    assert not (tmp_path / "unbound.json").exists()


def test_process_import_does_not_require_scientific_extras(tmp_path) -> None:
    code = "import sys; from geo_infer_test.process import run_process; assert 'numpy' not in sys.modules; assert 'geo_infer_test.core.validators' not in sys.modules"
    completed = run_process(
        [sys.executable, "-c", code],
        timeout=5,
        cwd=tmp_path,
        env=execution.build_subprocess_env(),
    )
    assert completed.returncode == 0, completed.stderr


def test_incomplete_source_custody_fails_before_child_launch(
    engine, tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "not-launched"
    monkeypatch.setattr(
        engine, "runtime_receipt", lambda **_kwargs: {"custody_complete": False}
    )
    result = engine.run_command(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        "custody unavailable",
        5,
    )
    assert not result.success and result.status == "METADATA_ERROR"
    assert not marker.exists()
    assert "terminated" not in result.stderr


def test_setup_and_child_share_one_command_deadline(
    engine, tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "not-launched"

    def slow_metadata(**_kwargs):
        time.sleep(0.15)
        return {}

    monkeypatch.setattr(engine, "runtime_receipt", slow_metadata)
    result = engine.run_command(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        "setup deadline",
        0.1,
    )
    assert result.status == "TIMEOUT" and not result.success
    assert not marker.exists()
    assert "before command launch" in result.stderr
    assert "tree terminated" not in result.stderr


def test_timeout_terminates_descendant_that_creates_its_own_session(tmp_path) -> None:
    pidfile = tmp_path / "escaped.pid"
    child = "import time; time.sleep(30)"
    parent = f"from pathlib import Path; import os,subprocess,sys,time; child = subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=os.name=='posix'); Path({str(pidfile)!r}).write_text(str(child.pid)); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", parent], timeout=0.3, cwd=tmp_path)
    _assert_recorded_process_dead(pidfile)


def test_timeout_terminates_detached_pipe_holder_after_parent_exit(tmp_path) -> None:
    """A retained descendant remains owned after its PPID and session change."""
    pidfile = tmp_path / "orphan.pid"
    child = "import time; print('detached child', flush=True); time.sleep(30)"
    parent = (
        "from pathlib import Path; import os,subprocess,sys,time; "
        f"child = subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=os.name=='posix'); Path({str(pidfile)!r}).write_text(str(child.pid)); "
        "print('parent exited', flush=True); time.sleep(0.3)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run_process([sys.executable, "-c", parent], timeout=0.6, cwd=tmp_path)
    assert time.monotonic() - started < 5
    assert "parent exited" in failure.value.output
    assert "detached child" in failure.value.output
    _assert_recorded_process_dead(pidfile)


def test_immediate_parent_exit_cannot_hide_detached_pipe_holder(tmp_path) -> None:
    pidfile = tmp_path / "early-orphan.pid"
    child = "import time; time.sleep(30)"
    parent = f"from pathlib import Path; import os,subprocess,sys; child = subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=os.name=='posix'); Path({str(pidfile)!r}).write_text(str(child.pid))"
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", parent], timeout=0.3, cwd=tmp_path)
    _assert_recorded_process_dead(pidfile)


def test_nominal_success_with_detached_background_process_fails_and_cleans(
    tmp_path,
) -> None:
    from geo_infer_test.process import OwnedProcessLeakError

    pidfile = tmp_path / "background.pid"
    child = "import time; time.sleep(30)"
    parent = f"from pathlib import Path; import os,subprocess,sys,time; child = subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=os.name=='posix', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); Path({str(pidfile)!r}).write_text(str(child.pid)); print('owned parent output', flush=True); time.sleep(0.3)"
    with pytest.raises(OwnedProcessLeakError) as failure:
        run_process([sys.executable, "-c", parent], timeout=3, cwd=tmp_path)
    assert "owned parent output" in failure.value.output
    _assert_recorded_process_dead(pidfile)


def test_fleet_cleanup_attempts_every_process_despite_one_failure(monkeypatch) -> None:
    import geo_infer_test.process as process_module

    calls = []
    processes = [object(), object()]
    monkeypatch.setattr(process_module, "_ACTIVE_PROCESSES", set(processes))

    def terminate(process):
        calls.append(process)
        if len(calls) == 1:
            raise OSError("first cleanup failed")

    monkeypatch.setattr(process_module, "terminate_tree", terminate)
    try:
        with pytest.raises(ExceptionGroup, match="Fleet process cleanup failed"):
            process_module.terminate_running_processes()
        assert set(calls) == set(processes)
    finally:
        process_module.reset_process_cancellation()


def test_keyboard_interrupt_retains_real_output_and_receipt(
    engine, tmp_path, monkeypatch
) -> None:
    import geo_infer_test.process as process_module

    ready = tmp_path / "ready"
    code = f"from pathlib import Path; import sys,time; print('before interrupt', flush=True); print('interrupt stderr', file=sys.stderr, flush=True); Path({str(ready)!r}).touch(); time.sleep(30)"
    original = subprocess.Popen.communicate
    interrupted = False

    def communicate(process, *args, **kwargs):
        nonlocal interrupted
        if code in process.args and not interrupted:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            interrupted = True
            raise KeyboardInterrupt
        return original(process, *args, **kwargs)

    monkeypatch.setattr(process_module.subprocess.Popen, "communicate", communicate)
    result = engine.run_command(
        [sys.executable, "-c", code], "interrupted real child", 30
    )
    assert result.status == "INTERRUPTED" and not result.success
    assert "before interrupt" in result.stdout
    assert "interrupt stderr" in result.stderr
    assert (
        "before interrupt" in (Path(result.receipt).parent / "stdout.log").read_text()
    )


def test_parallel_modules_preserve_conftest_isolation_and_inventory(
    engine, tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nfilterwarnings = ["error"]\n'
    )
    descriptors = []
    for name, other in (("A", "B"), ("B", "A")):
        tests = repository / f"GEO-INFER-{name}" / "tests" / "unit"
        tests.mkdir(parents=True)
        (tests / "conftest.py").write_text(
            f"import pytest\n@pytest.fixture\ndef module_identity(): return {name!r}\n"
        )
        (tests / "test_collision.py").write_text(f"""from pathlib import Path
import time

def test_conftest_identity(module_identity):
    assert module_identity == {name!r}
    Path({str(repository / (name + ".ready"))!r}).touch()
    deadline = time.monotonic() + 5
    while not Path({str(repository / (other + ".ready"))!r}).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert Path({str(repository / (other + ".ready"))!r}).exists()
""")
        descriptors.append(engine.Module(name, tests.parent.parent, tests.parent, True))
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "core.fsmonitor", "false"], cwd=repository, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    monkeypatch.setattr(engine, "PROJECT_ROOT", repository)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    report = engine.run_module_category_tests(
        "unit", 30, modules=descriptors, workers=2
    )
    assert report.success
    assert len(report.results) == 2 and all(
        result.executed == 1 for result in report.results
    )
    assert len({result.receipt for result in report.results}) == 2


@pytest.mark.parametrize("workers", [0, -1, 17, True, 1.5])
def test_worker_limits_reject_invalid_parallelism(engine, workers) -> None:
    with pytest.raises(ValueError, match="integer between 1 and 16"):
        engine.execute_module_tasks([], workers=workers, fail_fast=False)


def test_failed_native_tree_tool_still_kills_primary_process(
    tmp_path, monkeypatch
) -> None:
    import os
    import geo_infer_test.process as process_module

    original = subprocess.run

    def failed_census(command, **kwargs):
        if command[:1] == (["ps"] if os.name == "posix" else ["taskkill"]):
            raise subprocess.CalledProcessError(1, command)
        return original(command, **kwargs)

    monkeypatch.setattr(process_module.subprocess, "run", failed_census)
    marker = tmp_path / "census-failure-descendant"
    code = f"from pathlib import Path; import time; time.sleep(0.8); Path({str(marker)!r}).touch(); time.sleep(30)"
    started = time.monotonic()
    expected = (
        subprocess.CalledProcessError
        if os.name == "posix"
        else subprocess.TimeoutExpired
    )
    with pytest.raises(expected):
        process_module.run_process(
            [sys.executable, "-c", code], cwd=tmp_path, timeout=0.3
        )
    assert time.monotonic() - started < 5
    time.sleep(1)
    assert not marker.exists()


def test_environment_census_failure_does_not_expose_internal_listing(
    monkeypatch,
) -> None:
    from types import SimpleNamespace
    import geo_infer_test.process as process_module

    def failed_census(command, **kwargs):
        assert command[0] == "ps"
        raise subprocess.TimeoutExpired(
            command, 0.1, output="PRIVATE_ENV_VALUE", stderr="PRIVATE_ENV_ERROR"
        )

    monkeypatch.setattr(process_module, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(process_module.subprocess, "run", failed_census)
    census = process_module._DescendantCensus(0, "a" * 32)
    with pytest.raises(process_module.ProcessCensusError) as failure:
        census.refresh(timeout=1)
    assert failure.value.__cause__.output is None
    assert failure.value.__cause__.stderr is None
    assert "inspection budget" in str(failure.value)


def test_running_census_frequency_is_bounded(tmp_path, monkeypatch):
    """A real running child cannot trigger a native census every 50 ms."""
    import geo_infer_test.process as process_module

    scans = []

    def measured_scan(self, *, timeout):
        scans.append(time.monotonic())

    monkeypatch.setattr(process_module._DescendantCensus, "refresh", measured_scan)
    completed = run_process(
        [sys.executable, "-c", "import time; time.sleep(0.8); print('complete')"],
        timeout=10,
        cwd=tmp_path,
    )
    assert completed.stdout == "complete\n" and completed.returncode == 0
    assert len(scans) >= 2
    # Completion gets an immediate final scan; every running scan waits.
    assert all(right - left >= 0.20 for left, right in zip(scans, scans[1:-1]))


def test_census_timeout_is_not_reported_as_target_deadline(engine, monkeypatch):
    """A scanner failure cannot claim the command exhausted its long budget."""
    import geo_infer_test.process as process_module

    metadata = engine.runtime_receipt(timeout=10)
    monkeypatch.setattr(engine, "runtime_receipt", lambda **kwargs: metadata)

    def failed_refresh(self, *, timeout):
        raise process_module.ProcessCensusError(
            "Owned process census inspection budget"
        )

    monkeypatch.setattr(process_module._DescendantCensus, "refresh", failed_refresh)
    result = engine.run_command(
        [sys.executable, "-c", "print('target completed', flush=True)"],
        "census infrastructure failure",
        300,
    )
    assert not result.success and result.status == "FAIL"
    assert result.stdout == "target completed\n"
    assert result.duration < 10
    receipt = json.loads(Path(result.receipt).read_text())
    assert receipt["status"] == "FAIL"
    diagnostics = (Path(result.receipt).parent / "stderr.log").read_text()
    assert "ProcessCensusError" in diagnostics
    assert "Timed out after 300" not in diagnostics


def test_cleanup_scanner_failure_retains_target_output(tmp_path, monkeypatch) -> None:
    import os
    import geo_infer_test.process as process_module

    ready = tmp_path / "output-ready"
    original = subprocess.run

    def failed_scan(command, **kwargs):
        native = "ps" if os.name == "posix" else "taskkill"
        if command[:1] == [native]:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            raise subprocess.CalledProcessError(
                1, command, output="PRIVATE_SCAN_ENV", stderr="PRIVATE_SCAN_ERROR"
            )
        return original(command, **kwargs)

    monkeypatch.setattr(process_module.subprocess, "run", failed_scan)
    code = f"from pathlib import Path; import sys,time; print('target stdout', flush=True); print('target stderr', file=sys.stderr, flush=True); Path({str(ready)!r}).touch(); time.sleep(30)"
    expected = (
        subprocess.CalledProcessError
        if os.name == "posix"
        else subprocess.TimeoutExpired
    )
    with pytest.raises(expected) as failure:
        process_module.run_process(
            [sys.executable, "-c", code], timeout=0.3, cwd=tmp_path
        )
    assert "target stdout" in failure.value.output
    assert "target stderr" in failure.value.stderr
    assert "Process cleanup failed" in failure.value.stderr
    assert "PRIVATE_SCAN_ENV" not in failure.value.output
    assert "PRIVATE_SCAN_ERROR" not in failure.value.stderr


def test_cancellation_during_census_setup_prevents_launch(
    tmp_path, monkeypatch
) -> None:
    import threading
    import geo_infer_test.process as process_module

    process_module.reset_process_cancellation()
    entered, release = threading.Event(), threading.Event()
    original = process_module._DescendantCensus.__init__

    def gated_setup(self, pid, token):
        entered.set()
        assert release.wait(5)
        original(self, pid, token)

    monkeypatch.setattr(process_module._DescendantCensus, "__init__", gated_setup)
    marker = tmp_path / "must-not-launch"
    outcome = []

    def worker():
        try:
            process_module.run_process(
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                timeout=10,
                cwd=tmp_path,
            )
        except InterruptedError as exc:
            outcome.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(5)
        process_module.terminate_running_processes()
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(outcome) == 1
        assert not marker.exists()
    finally:
        release.set()
        thread.join(timeout=5)
        process_module.reset_process_cancellation()


def test_cancellation_cannot_miss_process_during_registration(
    tmp_path, monkeypatch
) -> None:
    import threading
    import geo_infer_test.process as process_module

    process_module.reset_process_cancellation()
    entered = threading.Event()
    release = threading.Event()
    original = subprocess.Popen

    def gated_spawn(command, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(command, **kwargs)

    monkeypatch.setattr(process_module.subprocess, "Popen", gated_spawn)
    pidfile = tmp_path / "registration-race.pid"
    child = "import time; time.sleep(30)"
    code = f"from pathlib import Path; import os,subprocess,sys,time; child = subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=os.name=='posix'); Path({str(pidfile)!r}).write_text(str(child.pid)); time.sleep(30)"
    outcome = []

    def worker():
        try:
            outcome.append(
                process_module.run_process(
                    [sys.executable, "-c", code], cwd=tmp_path, timeout=10
                )
            )
        except OSError as exc:
            outcome.append(exc)

    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    assert entered.wait(5)
    cancellation = threading.Thread(target=process_module.terminate_running_processes)
    cancellation.start()
    release.set()
    worker_thread.join(timeout=5)
    cancellation.join(timeout=5)
    process_module.reset_process_cancellation()
    assert not worker_thread.is_alive() and not cancellation.is_alive()
    assert len(outcome) == 1
    assert (
        not isinstance(outcome[0], subprocess.CompletedProcess)
        or outcome[0].returncode != 0
    )
    if pidfile.exists():
        _assert_recorded_process_dead(pidfile)


def test_sequential_interruption_does_not_launch_another_module(
    engine, tmp_path
) -> None:
    marker = tmp_path / "later-module-launched"

    def first():
        return engine.CommandResult(
            "interrupted module", False, 0, [], status="INTERRUPTED"
        )

    def later():
        return engine.run_command(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ],
            "later module",
            5,
        )

    report = engine.execute_module_tasks([first, later], workers=1, fail_fast=False)
    assert not report.success and len(report.results) == 1
    assert not marker.exists()


@pytest.mark.parametrize(
    "body",
    [
        "raise SystemExit(0)",
        "print('malformed')",
        'print(\'{"completion_token":"other","status":"ok"}\')',
    ],
)
def test_validator_requires_terminal_completion_evidence(engine, body) -> None:
    result = engine.run_command(
        [sys.executable, "-c", body],
        "early exit validator",
        10,
        completion_token="expected",
    )
    assert result.returncode == 0 and not result.success
    assert "terminal completion receipt" in result.stderr


def test_validator_retains_successful_terminal_completion_evidence(engine) -> None:
    result = engine.run_command(
        [
            sys.executable,
            "-c",
            "import json; assert 3*7 == 21; print(json.dumps({'completion_token':'expected','status':'ok'}))",
        ],
        "actual validator completion",
        10,
        completion_token="expected",
    )
    assert result.success
    assert json.loads(Path(result.receipt).read_text())["completion"] == {
        "completion_token": "expected",
        "status": "ok",
    }
