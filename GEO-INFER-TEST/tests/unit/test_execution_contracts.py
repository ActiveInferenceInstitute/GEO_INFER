"""Real subprocess regressions for test evidence, inventory, and deadlines."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
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
    receipt = json.loads(Path(result.receipt).read_text(encoding="utf-8"))
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
    assert (attempt / "stdout.log").read_text(encoding="utf-8") == result.stdout
    assert (attempt / "stderr.log").read_text(encoding="utf-8") == result.stderr


def test_restricted_console_preserves_unicode_failure_and_summary(engine, monkeypatch):
    """A strict Windows-style console cannot prevent retaining a failed attempt."""
    import hashlib
    import io

    displayed = io.BytesIO()
    console = io.TextIOWrapper(displayed, encoding="cp1252", write_through=True)
    with monkeypatch.context() as scope:
        scope.setattr(sys, "stdout", console)
        result = engine.run_command(
            [
                sys.executable,
                "-c",
                "import os; os.write(1, b'original\\xff'); "
                "os.write(2, bytes.fromhex('e6b8ace5ae9a')); raise SystemExit(7)",
            ],
            "Unicode failure \u0394",
            5,
        )
        engine.write_summary(engine.SuiteReport([result]), show_failures=True)
    assert result.returncode == 7 and result.status == "FAIL"
    assert result.stdout == "original\ufffd" and result.stderr == "\u6e2c\u5b9a"
    receipt_path = Path(result.receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for filename, expected in (
        ("stdout.log", result.stdout),
        ("stderr.log", result.stderr),
    ):
        artifact = receipt_path.parent / filename
        assert artifact.read_text(encoding="utf-8") == expected
        assert (
            receipt["artifacts"][filename]
            == hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
    summary = json.loads(
        (engine.RESULTS_DIR / "summary.json").read_text(encoding="utf-8")
    )
    assert not summary["success"] and summary["results"][0]["returncode"] == 7
    assert "\\ufffd" in displayed.getvalue().decode("cp1252")
    assert "\\u0394" in displayed.getvalue().decode("cp1252")


def _assert_large_timeout_output_retention(
    engine, monkeypatch, *, emission_delay: float = 0.0
):
    """Witness complete emission inside the one real command deadline."""
    import hashlib
    import threading
    import uuid

    import geo_infer_test.process as process_module
    import psutil

    ready_file = engine.RESULTS_DIR / f"large-output-ready-{uuid.uuid4().hex}.json"
    expected_stdout = "STDOUT_HEAD" + "x" * 1_050_000 + "STDOUT_TAIL\n"
    expected_stderr = "STDERR_HEAD" + "y" * 1_050_000 + "STDERR_TAIL\n"
    command_deadline, readiness_budget = 10, 8
    code = f"""
import json, os, sys, threading, time
from pathlib import Path
delay = {emission_delay!r}
if delay:
    threading.Event().wait(delay)
print('STDOUT_HEAD' + 'x'*1_050_000 + 'STDOUT_TAIL', flush=True)
print('STDERR_HEAD' + 'y'*1_050_000 + 'STDERR_TAIL', file=sys.stderr, flush=True)
ready = Path({str(ready_file)!r})
partial = ready.with_suffix('.partial')
partial.write_text(json.dumps({{'pid': os.getpid(), 'ready_at': time.monotonic(), 'status': 'both-streams-flushed', 'tokens': os.environ[{process_module._OWNERSHIP_ENV!r}].split(':')}}))
partial.replace(ready)
threading.Event().wait(30)
"""
    command = [sys.executable, "-c", code]
    primaries = []
    original_spawn = process_module.subprocess.Popen

    def record_primary(arguments, *args, **kwargs):
        process = original_spawn(arguments, *args, **kwargs)
        if arguments == command:
            # Observe the real handle without delaying registration or changing
            # the command, ownership chain, pipe behavior, or deadline.
            primaries.append((process, kwargs["env"][process_module._OWNERSHIP_ENV]))
        return process

    monkeypatch.setattr(process_module.subprocess, "Popen", record_primary)
    started = time.monotonic()
    stop = threading.Event()
    observations, observer_errors, owned_processes = [], [], []

    def observe_ready():
        try:
            while not stop.is_set():
                remaining = started + readiness_budget - time.monotonic()
                assert remaining > 0, (
                    "Emitter did not complete both streams within 8 seconds"
                )
                if ready_file.is_file() and primaries:
                    ready = json.loads(ready_file.read_text(encoding="utf-8"))
                    primary, tokens = primaries[0]
                    assert ready["pid"] == primary.pid and primary.poll() is None
                    assert ready["tokens"] == tokens.split(":")
                    process = psutil.Process(primary.pid)
                    assert process.environ()[process_module._OWNERSHIP_ENV] == tokens
                    assert (
                        process.ppid() == os.getpid()
                        and process.is_running()
                        and process.status() != psutil.STATUS_ZOMBIE
                    )
                    owned_processes.append(process)
                    observations.append(ready)
                    return
                if stop.wait(min(0.01, remaining)):
                    return
        except Exception as exc:
            observer_errors.append(f"{type(exc).__name__}: {exc}")

    observer = threading.Thread(
        target=observe_ready, name="large-output-ready", daemon=True
    )
    observer.start()
    try:
        result = engine.run_command(command, "large timeout", command_deadline)
        elapsed = time.monotonic() - started
        assert result.status == "TIMEOUT" and not result.success
        assert command_deadline <= elapsed < command_deadline + 10
        assert len(primaries) == 1, "The fixture must launch exactly one attempt"
        assert primaries[0][0].poll() is not None
        assert not observer_errors, observer_errors
        assert observations, "Emitter did not complete both streams within 8 seconds"
        assert observations[0]["status"] == "both-streams-flushed"
        assert started <= observations[0]["ready_at"] < started + readiness_budget
        assert not owned_processes[0].is_running()
        assert result.stdout == expected_stdout
        assert result.stderr == (
            expected_stderr
            + "\nTimed out after 10s; process-tree cleanup attempted; see retained diagnostics"
        )
        receipt_path = Path(result.receipt)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for name, prefix, suffix, retained in (
            ("stdout.log", "STDOUT_HEAD", "STDOUT_TAIL", result.stdout),
            ("stderr.log", "STDERR_HEAD", "STDERR_TAIL", result.stderr),
        ):
            artifact = receipt_path.parent / name
            assert retained.startswith(prefix) and suffix in retained
            assert len(retained) > 1_050_000
            assert artifact.read_text(encoding="utf-8") == retained
            assert (
                receipt["artifacts"][name]
                == hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
    finally:
        stop.set()
        observer.join(timeout=2)
        for primary, _ in primaries:
            if primary.poll() is None:
                primary.kill()
                primary.wait(timeout=2)
        assert not observer.is_alive(), "Emission observer failed its bounded join"


def test_large_real_timeout_retains_both_ends_of_output(engine, monkeypatch) -> None:
    """Timeout logs must preserve completely emitted diagnostic payloads."""
    _assert_large_timeout_output_retention(engine, monkeypatch)


@pytest.mark.parametrize("interrupted", [False, True])
def test_large_process_exception_retains_full_decoded_output(
    engine, monkeypatch, interrupted
) -> None:
    """Error and interruption artifacts share the full-output retention contract."""
    metadata = engine.runtime_receipt(timeout=10)
    monkeypatch.setattr(engine, "runtime_receipt", lambda **kwargs: metadata)
    error = KeyboardInterrupt() if interrupted else RuntimeError("process failure")
    error.output = b"OUTPUT_HEAD" + b"x" * 1_050_000 + b"\xffOUTPUT_TAIL"
    error.stderr = b"ERROR_HEAD" + b"y" * 1_050_000 + b"\xfeERROR_TAIL"

    def fail_process(*args, **kwargs):
        raise error

    monkeypatch.setattr(engine, "run_process", fail_process)
    result = engine.run_command([sys.executable, "-c", "pass"], "large error", 10)
    assert result.status == ("INTERRUPTED" if interrupted else "FAIL")
    assert result.stdout.startswith("OUTPUT_HEAD")
    assert "\ufffdOUTPUT_TAIL" in result.stdout
    assert result.stderr.startswith("ERROR_HEAD")
    assert "\ufffdERROR_TAIL" in result.stderr
    attempt = Path(result.receipt).parent
    assert (attempt / "stdout.log").read_text(encoding="utf-8") == result.stdout
    assert (attempt / "stderr.log").read_text(encoding="utf-8") == result.stderr


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
    receipt = json.loads(Path(result.receipt).read_text(encoding="utf-8"))
    assert receipt["success"] is False and receipt["returncode"] is None
    assert result.stdout == "retained target output"
    assert "retained target diagnostics" in result.stderr
    assert type(error).__name__ in result.stderr


@pytest.mark.parametrize(
    "content, diagnostic",
    [
        (None, "pytest produced no JUnit report"),
        ("<broken>", "invalid JUnit report"),
        (
            '<testsuites><testsuite tests="0"/></testsuites>',
            "pytest JUnit report contains no testcases",
        ),
    ],
)
def test_zero_exit_needs_current_nonempty_valid_junit(
    engine, tmp_path, content, diagnostic
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
        [sys.executable, "-c", body, f"--junitxml={stale}"], "missing evidence", 20
    )
    assert result.returncode == 0 and result.status == "FAIL" and not result.success
    assert diagnostic in result.stderr
    assert stale.read_text(encoding="utf-8").find('name="old"') >= 0
    assert Path(result.receipt).is_file()


def _pytest_command(tmp_path: Path, body: str, *, selection: bool = True) -> list[str]:
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nfilterwarnings = error\n")
    test = tmp_path / "test_child.py"
    test.write_text(body)
    command = [
        sys.executable,
        "-m",
        "pytest",
    ]
    if selection:
        command.extend(["-p", "geo_infer_test.selection"])
    return [
        *command,
        "-c",
        str(config),
        str(test),
        f"--junitxml={tmp_path / 'junit.xml'}",
    ]


def _pytest_console(entrypoint: str) -> str:
    executable = Path(sysconfig.get_path("scripts")) / (
        entrypoint + (".exe" if os.name == "nt" else "")
    )
    assert executable.is_file(), "The active pytest installation lacks its entrypoint"
    return str(executable)


@pytest.mark.parametrize("entrypoint", ["pytest", "py.test"])
@pytest.mark.parametrize(
    "report_form", [None, "--junitxml=", "--junit-xml=", "--junitxml", "--junit-xml"]
)
def test_real_pytest_console_requires_owned_junit(
    engine, tmp_path, entrypoint, report_form
) -> None:
    caller_report = tmp_path / "caller.xml"
    old_report = '<testsuites><testsuite tests="1"><testcase name="old"/></testsuite></testsuites>'
    caller_report.write_text(old_report)
    module_command = _pytest_command(tmp_path, "def test_one(): assert 2 + 3 == 5\n")
    command = [_pytest_console(entrypoint), *module_command[3:-1]]
    if report_form is not None:
        command.extend(
            [report_form + str(caller_report)]
            if report_form.endswith("=")
            else [report_form, str(caller_report)]
        )
    original_command = command.copy()
    result = engine.run_command(command, "real pytest console", 20)
    assert result.returncode == 0
    assert command == original_command
    assert caller_report.read_text() == old_report
    receipt = json.loads(Path(result.receipt).read_text())
    if report_form is None:
        assert not result.success and result.status == "FAIL"
        assert "1 passed" in result.stdout
        assert "omitted its required JUnit" in result.stderr
        assert result.executed == 0 and "junit.xml" not in receipt["artifacts"]
    else:
        assert result.success and result.executed == 1
        assert engine.junit_path(result.command) == Path(result.receipt).with_name(
            "junit.xml"
        )
        assert receipt["selection"]["collected"] == receipt["selection"]["executed"]
        assert len(receipt["selection"]["executed"]) == 1
        assert "junit.xml" in receipt["artifacts"]


@pytest.mark.parametrize(
    "interpreter_options", [["-m", "pytest"], ["-I", "-m", "pytest"], ["-Impytest"]]
)
@pytest.mark.parametrize("with_report", [False, True])
def test_real_python_module_pytest_requires_junit(
    engine, tmp_path, interpreter_options, with_report
) -> None:
    original = _pytest_command(
        tmp_path, "def test_one(): assert True\n", selection=False
    )
    command = [sys.executable, *interpreter_options, *original[3:]]
    if not with_report:
        command.pop()
    result = engine.run_command(command, "real Python module", 20)
    assert result.returncode == 0
    assert result.success is with_report
    assert result.executed == int(with_report)
    if not with_report:
        assert "1 passed" in result.stdout
        assert "omitted its required JUnit" in result.stderr


@pytest.mark.parametrize("entrypoint", [None, "pytest"])
def test_real_pytest_duplicate_reports_preserve_caller_files(
    engine, tmp_path, entrypoint
) -> None:
    first = tmp_path / "first.xml"
    last = tmp_path / "last.xml"
    first.write_text("first caller report")
    last.write_text("last caller report")
    command = _pytest_command(tmp_path, "def test_one(): assert True\n")[:-1]
    if entrypoint:
        command = [_pytest_console(entrypoint), *command[3:]]
    command.extend([f"--junitxml={first}", "--junit-xml", str(last)])
    assert engine.junit_path(command) == last
    original_command = command.copy()
    result = engine.run_command(command, "duplicate report options", 20)
    assert result.success and result.executed == 1
    assert command == original_command
    assert first.read_text() == "first caller report"
    assert last.read_text() == "last caller report"
    owned = Path(result.receipt).with_name("junit.xml")
    assert result.command.count(f"--junitxml={owned}") == 2


@pytest.mark.parametrize(
    "invalid_report",
    [
        ["--junitxml"],
        ["--junit-xml", ""],
        ["--junitxml="],
        ["--junit-xml", "--capture=no"],
        ["--junitxml", "--"],
    ],
)
def test_invalid_report_operands_cannot_be_repaired_into_success(
    engine, tmp_path, invalid_report
) -> None:
    command = _pytest_command(tmp_path, "def test_one(): assert True\n")[:-1]
    command = [_pytest_console("pytest"), *command[3:], *invalid_report]
    original_command = command.copy()
    result = engine.run_command(command, "invalid report operand", 20)
    assert not result.success and result.status == "FAIL"
    if "" in invalid_report or invalid_report == ["--junitxml="]:
        assert result.returncode == 0 and "1 passed" in result.stdout
    else:
        assert result.returncode == 4
    assert command == original_command
    assert result.command[-len(invalid_report) :] == invalid_report
    assert "JUnit report option" in result.stderr
    assert not Path(result.receipt).with_name("junit.xml").exists()


@pytest.mark.parametrize("relative", [False, True])
def test_existing_report_directory_keeps_real_pytest_configuration_failure(
    engine, tmp_path, relative
) -> None:
    child_cwd = tmp_path / "child"
    directory = child_cwd / "existing"
    directory.mkdir(parents=True)
    command = _pytest_command(tmp_path, "def test_one(): assert True\n")[:-1]
    operand = directory.name if relative else str(directory)
    command = [_pytest_console("pytest"), *command[3:], "--junit-xml", operand]
    original_command = command.copy()
    result = engine.run_command(command, "directory report operand", 20, cwd=child_cwd)
    assert not result.success and result.status == "FAIL" and result.returncode == 4
    assert result.executed == 0
    assert command == original_command and result.command[-2:] == command[-2:]
    assert "must be a filename" in result.stderr
    assert "file, not a directory" in result.stderr
    assert directory.is_dir() and not list(directory.iterdir())
    assert not Path(result.receipt).with_name("junit.xml").exists()


@pytest.mark.parametrize(
    "arguments", [["pytest"], ["-m", "pytest"], ["--", "--junitxml=unpromised.xml"]]
)
def test_validator_arguments_do_not_advertise_pytest_or_report_promises(
    engine, arguments
) -> None:
    result = engine.run_command(
        [sys.executable, "-Ic", "print('ordinary validator')", *arguments],
        "ordinary validator arguments",
        5,
    )
    assert result.success and result.returncode == 0 and result.executed == 0
    assert result.stdout == "ordinary validator\n" and not result.stderr
    assert not Path(result.receipt).with_name("junit.xml").exists()


def test_python_script_named_pytest_remains_an_ordinary_validator(
    engine, tmp_path
) -> None:
    validator = tmp_path / "pytest"
    validator.write_text("print('ordinary script')\n")
    result = engine.run_command(
        [sys.executable, str(validator), "-m", "pytest"], "ordinary script", 5
    )
    assert result.success and result.returncode == 0 and not result.stderr
    assert result.stdout == "ordinary script\n" and result.executed == 0


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
        child = psutil.Process(int(pidfile.read_text(encoding="utf-8")))
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def _assert_real_descendant_timeout(
    tmp_path,
    monkeypatch,
    *,
    detached=False,
    parent_exit="never",
    startup_delay=0,
) -> None:
    """Observe startup inside one real deadline, then verify owned cleanup.

    Coverage can instrument both newly launched interpreters. Their startup
    belongs to the five-second command deadline; readiness never resets it.
    The three-second observer allowance makes missing startup an explicit
    failure. A delayed-startup boundary can use ``startup_delay`` without
    replacing any clock, census, communicate call, or termination operation.
    """
    import threading

    import psutil
    import geo_infer_test.process as process_module

    assert parent_exit in {"never", "released", "immediate"}
    ready = tmp_path / "descendant-ready.json"
    release = tmp_path / "release-parent"
    child = f"""import json, os, sys, threading
from pathlib import Path
print("real descendant stdout", flush=True)
print("real descendant stderr", file=sys.stderr, flush=True)
ready = Path({str(ready)!r})
pending = ready.with_suffix(".pending")
pending.write_text(json.dumps({{
    "pid": os.getpid(), "parent_pid": int(sys.argv[1]),
    "tokens": os.environ[{process_module._OWNERSHIP_ENV!r}].split(":"),
}}), encoding="utf-8")
pending.replace(ready)
threading.Event().wait(30)
"""
    parent = f"""import os, subprocess, sys, threading, time
from pathlib import Path
print("real parent stdout", flush=True)
print("real parent stderr", file=sys.stderr, flush=True)
threading.Event().wait({startup_delay!r})
subprocess.Popen([sys.executable, "-c", {child!r}, str(os.getpid())],
                 start_new_session={detached!r} and os.name == "posix")
"""
    if parent_exit == "released":
        parent += f"""deadline = time.monotonic() + 5
poll = threading.Event()
while not Path({str(release)!r}).exists() and time.monotonic() < deadline:
    poll.wait(0.01)
assert Path({str(release)!r}).exists(), "Parent release was not observed"
print("parent exited", flush=True)
"""
    elif parent_exit == "immediate":
        # No readiness wait: the descendant may still be starting as its
        # original parent exits and it becomes an orphan/pipe holder.
        parent += 'print("parent exited", flush=True)\n'
    else:
        parent += "threading.Event().wait(30)\n"
    command = [sys.executable, "-c", parent]
    primaries, startup_errors = [], []
    observed = {}
    stopped = threading.Event()
    original_spawn = process_module.subprocess.Popen

    def record_primary(arguments, *args, **kwargs):
        process = original_spawn(arguments, *args, **kwargs)
        if arguments == command:
            # Only observe the real launch; do not block registration or
            # rewrite its deadline, command, environment, or pipe behavior.
            primaries.append((process, kwargs["env"][process_module._OWNERSHIP_ENV]))
        return process

    monkeypatch.setattr(process_module.subprocess, "Popen", record_primary)
    started = time.monotonic()

    def observe_readiness():
        deadline = started + 3
        try:
            while not stopped.is_set() and (not ready.exists() or not primaries):
                remaining = deadline - time.monotonic()
                assert remaining > 0, "Descendant did not publish output-ready identity"
                stopped.wait(min(0.01, remaining))
            assert ready.exists() and primaries, (
                "Command ended before descendant startup"
            )
            identity = json.loads(ready.read_text(encoding="utf-8"))
            primary, tokens = primaries[0]
            assert identity["parent_pid"] == primary.pid
            assert identity["tokens"] == tokens.split(":")
            descendant = psutil.Process(identity["pid"])
            assert descendant.environ()[process_module._OWNERSHIP_ENV] == tokens
            assert (
                descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE
            )
            observed["descendant"] = descendant
            if os.name == "posix":
                expected_session = descendant.pid if detached else primary.pid
                assert os.getsid(descendant.pid) == expected_session
            if parent_exit != "never":
                if parent_exit == "released":
                    release.touch()
                while primary.poll() is None and not stopped.is_set():
                    remaining = deadline - time.monotonic()
                    assert remaining > 0, (
                        "Primary did not exit before the command deadline"
                    )
                    stopped.wait(min(0.01, remaining))
                assert primary.returncode == 0, "Primary did not exit normally"
                assert (
                    descendant.is_running()
                    and descendant.status() != psutil.STATUS_ZOMBIE
                )
                observed["early_parent_exit"] = True
            else:
                assert primary.poll() is None, "Primary exited before the real timeout"
            observed["ready_elapsed"] = time.monotonic() - started
        except BaseException as error:
            startup_errors.append(f"{type(error).__name__}: {error}")

    watcher = threading.Thread(target=observe_readiness, daemon=True)
    watcher.start()
    try:
        try:
            with pytest.raises(subprocess.TimeoutExpired) as failure:
                run_process(command, timeout=5, cwd=tmp_path)
        finally:
            stopped.set()
            watcher.join(timeout=1)
        assert not watcher.is_alive(), "Readiness observer outlived bounded cleanup"
        assert not startup_errors, startup_errors
        assert len(primaries) == 1, "The fixture must launch exactly one attempt"
        assert 0 <= observed["ready_elapsed"] < 3
        assert 5 <= time.monotonic() - started < 10
        # The same command deadline can expire during its native ownership
        # census. Its TimeoutExpired may identify that inspection command;
        # elapsed time and retained target output establish this boundary.
        assert "real parent stdout" in failure.value.output
        assert "real parent stderr" in failure.value.stderr
        assert "real descendant stdout" in failure.value.output
        assert "real descendant stderr" in failure.value.stderr
        assert primaries[0][0].poll() is not None
        if parent_exit != "never":
            assert observed["early_parent_exit"]
            assert "parent exited" in failure.value.output
        descendant = observed["descendant"]
        try:
            assert (
                not descendant.is_running()
                or descendant.status() == psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            pass
    finally:
        # Preserve assertion/cleanup failures while preventing this owned
        # fixture from leaving native processes behind. These handles retain
        # the launch or birth identity; a PID file cannot authorize cleanup.
        stopped.set()
        watcher.join(timeout=1)
        descendant = observed.get("descendant")
        if descendant is not None:
            try:
                if (
                    descendant.is_running()
                    and descendant.status() != psutil.STATUS_ZOMBIE
                ):
                    descendant.kill()
                    psutil.wait_procs([descendant], timeout=2)
            except psutil.NoSuchProcess:
                pass
        for primary, _tokens in primaries:
            if primary.poll() is None:
                primary.kill()
                primary.wait(timeout=2)


def test_timeout_terminates_real_descendant(tmp_path, monkeypatch) -> None:
    _assert_real_descendant_timeout(tmp_path, monkeypatch)


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
    ordinary_receipt = json.loads(
        Path(ordinary.results[0].receipt).read_text(encoding="utf-8")
    )
    assert len(ordinary_receipt["selection"]["deselected"]) == 1
    (tmp_path / "render-ready").touch()
    rendered = engine.run_module_category_tests("manuscript-render", 30, workers=1)
    assert rendered.success and sum(result.executed for result in rendered.results) == 2
    rendered_receipt = json.loads(
        Path(rendered.results[0].receipt).read_text(encoding="utf-8")
    )
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
    receipt = json.loads(Path(result.receipt).read_text(encoding="utf-8"))
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
    receipt = json.loads(Path(result.receipt).read_text(encoding="utf-8"))
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
    receipt = json.loads(Path(result.receipt).read_text(encoding="utf-8"))
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


def test_timeout_terminates_descendant_that_creates_its_own_session(
    tmp_path, monkeypatch
) -> None:
    _assert_real_descendant_timeout(tmp_path, monkeypatch, detached=True)


def test_timeout_terminates_detached_pipe_holder_after_parent_exit(
    tmp_path, monkeypatch
) -> None:
    """A retained descendant remains owned after its PPID and session change."""
    _assert_real_descendant_timeout(
        tmp_path, monkeypatch, detached=True, parent_exit="released"
    )


def test_immediate_parent_exit_cannot_hide_detached_pipe_holder(
    tmp_path, monkeypatch
) -> None:
    _assert_real_descendant_timeout(
        tmp_path, monkeypatch, detached=True, parent_exit="immediate"
    )


def test_nested_native_observers_preserve_target_tokens_and_detached_cleanup(
    tmp_path,
) -> None:
    """A real nested owner excludes its observer while still reaping user work."""
    import geo_infer_test.process as process_module

    parent_environment = os.environ.copy()
    pidfile = tmp_path / "nested-descendant.pid"
    identity_file = tmp_path / "nested-descendant.json"
    descendant = f"""import json, os, threading
from pathlib import Path
Path({str(pidfile)!r}).write_text(str(os.getpid()), encoding="utf-8")
identity = Path({str(identity_file)!r})
pending = identity.with_suffix(".pending")
print("nested descendant output", flush=True)
pending.write_text(json.dumps({{
    "tokens": os.environ["GEO_INFER_PROCESS_TOKENS"].split(":"),
    "control": os.environ["GEO_INFER_OBSERVER_CONTROL"],
}}), encoding="utf-8")
pending.replace(identity)
threading.Event().wait(30)
"""
    inner_parent = f"""import os, subprocess, sys, threading, time
from pathlib import Path
print("nested parent output", flush=True)
print("nested parent stderr", file=sys.stderr, flush=True)
subprocess.Popen([sys.executable, "-c", {descendant!r}], start_new_session=True)
deadline = time.monotonic() + 5
while not Path({str(identity_file)!r}).exists() and time.monotonic() < deadline:
    time.sleep(0.001)
assert Path({str(identity_file)!r}).exists(), "Nested descendant did not start"
threading.Event().wait(30)
"""
    worker = tmp_path / "nested-owner.py"
    worker.write_text(
        f"""import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from geo_infer_test import process as module
parent_environment = os.environ.copy()
parent_tokens = parent_environment[module._OWNERSHIP_ENV].split(":")
original = subprocess.run
scans = []
def observe(command, **kwargs):
    if command[:1] == ["ps"]:
        assert module._OWNERSHIP_ENV not in kwargs["env"]
        assert kwargs["env"] == {{key: value for key, value in parent_environment.items()
                                  if key != module._OWNERSHIP_ENV}}
        scans.append(1)
    return original(command, **kwargs)
module.subprocess.run = observe
finished = threading.Event()
startup_errors = []
def interrupt_after_ready():
    deadline = time.monotonic() + 5
    while not Path({str(identity_file)!r}).exists():
        if finished.wait(0.001):
            return
        if time.monotonic() >= deadline:
            startup_errors.append("Nested descendant did not publish output-ready identity")
            return
    if not finished.is_set():
        os.kill(os.getpid(), signal.SIGINT)
watcher = threading.Thread(target=interrupt_after_ready)
watcher.start()
try:
    module.run_process([sys.executable, "-c", {inner_parent!r}],
                       timeout=10, cwd=Path({str(tmp_path)!r}))
except KeyboardInterrupt as error:
    assert "nested descendant output" in error.output
    assert "nested parent output" in error.output
    assert "nested parent stderr" in error.stderr
else:
    raise AssertionError("The live nested target was not interrupted")
finally:
    finished.set()
    watcher.join(timeout=2)
    assert not watcher.is_alive(), "Output-ready watcher outlived bounded cleanup"
    assert not startup_errors, startup_errors
assert os.environ == parent_environment
assert scans, "The nested owner never performed a real native census"
identity = json.loads(Path({str(identity_file)!r}).read_text(encoding="utf-8"))
assert identity["tokens"][:-1] == parent_tokens
assert identity["control"] == "preserved"
print(json.dumps({{"parent_tokens": parent_tokens, "descendant": identity,
                  "scans": len(scans)}}), flush=True)
""",
        encoding="utf-8",
    )
    env = execution.build_subprocess_env()
    env["GEO_INFER_OBSERVER_CONTROL"] = "preserved"
    completed = run_process(
        [sys.executable, str(worker)], timeout=20, cwd=tmp_path, env=env
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    inherited = env.get(process_module._OWNERSHIP_ENV, "").split(":")
    inherited = [token for token in inherited if token]
    assert observed["parent_tokens"][:-1] == inherited
    tokens = observed["descendant"]["tokens"]
    assert len(tokens) == len(inherited) + 2
    assert len(set(tokens)) == len(tokens)
    assert os.environ == parent_environment
    _assert_recorded_process_dead(pidfile)


def test_owned_cleanup_waits_for_real_process_exit(tmp_path, monkeypatch) -> None:
    """Cleanup must establish process exit, beyond requesting termination."""
    import psutil
    import geo_infer_test.process as process_module

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    census = process_module._DescendantCensus(0, "a" * 32)
    census.processes[child.pid] = psutil.Process(child.pid)
    original = psutil.wait_procs
    waited = []

    def observe_wait(processes, *, timeout):
        waited.append((tuple(process.pid for process in processes), timeout))
        return original(processes, timeout=timeout)

    monkeypatch.setattr(psutil, "wait_procs", observe_wait)
    try:
        census.kill()
        assert waited == [((child.pid,), process_module._DESCENDANT_REAP_SECONDS)]
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_owned_cleanup_preserves_signal_wait_and_status_errors(monkeypatch) -> None:
    """A later inspection error cannot erase earlier cleanup failures."""
    from types import SimpleNamespace
    import geo_infer_test.process as process_module

    def fail_signal():
        raise OSError("signal failure")

    def fail_wait(*args, **kwargs):
        raise TimeoutError("wait failure")

    def fail_status():
        raise PermissionError("status failure")

    census = process_module._DescendantCensus(0, "a" * 32)
    census.processes[1] = SimpleNamespace(
        pid=1, kill=fail_signal, is_running=lambda: True, status=fail_status
    )
    monkeypatch.setattr(census.psutil, "wait_procs", fail_wait)
    with pytest.raises(ExceptionGroup) as failure:
        census.kill()
    assert [str(error) for error in failure.value.exceptions] == [
        "signal failure",
        "wait failure",
        "status failure",
    ]


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
    assert "before interrupt" in (Path(result.receipt).parent / "stdout.log").read_text(
        encoding="utf-8"
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

    monkeypatch.setattr(
        process_module, "os", SimpleNamespace(name="posix", environ=os.environ)
    )
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


def test_census_lock_contention_obeys_deadline(monkeypatch):
    """A real lock holder cannot extend the census deadline or launch a late scan."""
    import threading
    import geo_infer_test.process as process_module

    census = process_module._DescendantCensus(0, "a" * 32)
    entered, release = threading.Event(), threading.Event()

    def hold_lock():
        with census.lock:
            entered.set()
            release.wait(timeout=5)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("Expired census launched a native scan")

    monkeypatch.setattr(process_module.subprocess, "run", unexpected_scan)
    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert entered.wait(timeout=5)
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            census.refresh(timeout=0.05)
        assert time.monotonic() - started < 1
    finally:
        release.set()
        holder.join(timeout=5)
        assert not holder.is_alive()


def test_late_empty_native_census_cannot_report_success(monkeypatch):
    """Even empty output must fail when a real producer returns after the deadline."""
    from types import SimpleNamespace
    import geo_infer_test.process as process_module

    original = subprocess.run

    def late_scan(*args, **kwargs):
        return original(
            [sys.executable, "-c", "import time; time.sleep(0.06)"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

    # Exercise the POSIX scan boundary without changing the platform's global os.
    monkeypatch.setattr(
        process_module, "os", SimpleNamespace(name="posix", environ=os.environ)
    )
    monkeypatch.setattr(process_module.subprocess, "run", late_scan)
    census = process_module._DescendantCensus(0, "a" * 32)
    with pytest.raises(subprocess.TimeoutExpired):
        census.refresh(timeout=0.03)


@pytest.mark.parametrize("target_exits", [False, True])
def test_census_timeout_is_not_reported_as_target_deadline(
    engine, monkeypatch, target_exits
):
    """A scanner failure cannot claim the command exhausted its long budget."""
    import threading

    import psutil
    import geo_infer_test.process as process_module

    metadata = engine.runtime_receipt(timeout=10)
    monkeypatch.setattr(engine, "runtime_receipt", lambda **kwargs: metadata)
    census_entered = engine.PROJECT_ROOT / "census-entered"
    output_ready = engine.PROJECT_ROOT / "target-output-ready"
    target_processes = []
    poll = threading.Event()

    def failed_refresh(self, *, timeout):
        if not target_processes:
            target_processes.append(psutil.Process(self.pid))
        census_entered.touch()
        deadline = time.monotonic() + min(timeout, 5)
        while not output_ready.exists() and time.monotonic() < deadline:
            poll.wait(min(0.01, max(0, deadline - time.monotonic())))
        assert output_ready.exists(), (
            "Target did not flush output within startup deadline"
        )
        if target_exits:
            target = target_processes[0]

            def exited():
                try:
                    return (
                        not target.is_running()
                        or target.status() == psutil.STATUS_ZOMBIE
                    )
                except psutil.NoSuchProcess:
                    return True

            while not exited() and time.monotonic() < deadline:
                poll.wait(min(0.01, max(0, deadline - time.monotonic())))
            assert exited(), "Target did not exit within startup deadline"
        raise process_module.ProcessCensusError(
            "Owned process census inspection budget"
        )

    monkeypatch.setattr(process_module._DescendantCensus, "refresh", failed_refresh)
    # Hold the real target before output until the first census begins. This
    # exercises slow startup deterministically, then publishes readiness only
    # after both streams are flushed. Census failure must retain that output
    # and reap either the still-running target or the target that has exited.
    code = f"""
import os, sys, threading, time
from pathlib import Path
deadline = time.monotonic() + 5
poll = threading.Event()
while not Path({str(census_entered)!r}).exists() and time.monotonic() < deadline:
    poll.wait(min(0.01, max(0, deadline - time.monotonic())))
assert Path({str(census_entered)!r}).exists(), "Census did not release target startup"
print('target completed', flush=True)
print('target stderr', file=sys.stderr, flush=True)
ready = Path({str(output_ready)!r})
pending = ready.with_suffix('.pending')
pending.write_text(str(os.getpid()), encoding='utf-8')
pending.replace(ready)
if not {target_exits!r}:
    threading.Event().wait(30)
"""
    result = engine.run_command(
        [sys.executable, "-c", code],
        "census infrastructure failure",
        300,
    )
    assert not result.success and result.status == "FAIL"
    assert result.stdout == "target completed\n"
    assert "target stderr\n" in result.stderr
    assert len(target_processes) == 1
    assert int(output_ready.read_text(encoding="utf-8")) == target_processes[0].pid
    assert not target_processes[0].is_running()
    assert result.duration < 10
    receipt = json.loads(Path(result.receipt).read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert (Path(result.receipt).parent / "stdout.log").read_text(
        encoding="utf-8"
    ) == result.stdout
    diagnostics = (Path(result.receipt).parent / "stderr.log").read_text(
        encoding="utf-8"
    )
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
    assert json.loads(Path(result.receipt).read_text(encoding="utf-8"))[
        "completion"
    ] == {
        "completion_token": "expected",
        "status": "ok",
    }
