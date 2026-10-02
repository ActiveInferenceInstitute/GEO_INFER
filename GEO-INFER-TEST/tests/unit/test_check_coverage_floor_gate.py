"""Regression tests for GS-293: check_coverage_floor.py must fail cleanly
(named-module message, exit 1) instead of raising KeyError when a diff-derived
module has no baseline entry."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "GEO-INFER-TEST" / "check_coverage_floor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_coverage_floor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_module_fails_cleanly(capsys):
    """A forced module absent from the baseline must produce the clean _fail
    message naming the module, not an unhandled KeyError traceback."""
    module = _load_module()
    try:
        module.main(
            ["--base", "HEAD", "--head", "HEAD", "--modules", "GEO-INFER-DOESNOTEXIST"]
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("gate should exit non-zero for an unknown module")
    err = capsys.readouterr().err
    assert "coverage floor gate failure" in err
    assert "GEO-INFER-DOESNOTEXIST" in err
    assert "KeyError" not in err


def test_unknown_module_via_cli_no_traceback():
    """End-to-end: the script exits 1 and prints no Python traceback."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            "HEAD",
            "--head",
            "HEAD",
            "--modules",
            "GEO-INFER-DOESNOTEXIST",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert "GEO-INFER-DOESNOTEXIST" in completed.stderr


def test_baseline_entries_are_dict_shaped():
    manifest = json.loads(
        (REPO_ROOT / "GEO-INFER-TEST" / "coverage_baseline.json").read_text()
    )
    assert isinstance(manifest.get("modules"), dict)


def _baseline_module() -> str:
    baseline = json.loads(
        (REPO_ROOT / "GEO-INFER-TEST" / "coverage_baseline.json").read_text()
    )
    return sorted(baseline["modules"])[0]


def test_failing_suite_during_measurement_fails_gate(monkeypatch, capsys):
    """GS-004: coverage measured while tests were failing must not pass."""
    module = _load_module()
    name = _baseline_module()

    def fake_measure(target):
        return {
            "module": target,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 1,
            "seconds": 0.1,
        }

    monkeypatch.setattr(module, "measure_module", fake_measure)

    try:
        module.main(["--base", "HEAD", "--head", "HEAD", "--modules", name])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("gate must fail when the measured suite had failures")

    captured = capsys.readouterr()
    assert "FAILED-SUITE" in captured.out
    assert "pytest rc=1" in captured.err


def test_clean_suite_measurement_passes_gate(monkeypatch, capsys):
    """GS-004: a measurement whose suite passed still passes the floor."""
    module = _load_module()
    name = _baseline_module()

    def fake_measure(target):
        return {
            "module": target,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 0,
            "seconds": 0.1,
        }

    monkeypatch.setattr(module, "measure_module", fake_measure)

    assert module.main(["--base", "HEAD", "--head", "HEAD", "--modules", name]) == 0
    assert "FAILED-SUITE" not in capsys.readouterr().out


def test_failed_suite_verdict_prints_failing_test_names(monkeypatch, capsys):
    """A FAILED-SUITE verdict names the failing tests from the JUnit report
    instead of hiding the per-test detail behind a one-line summary."""
    module = _load_module()
    name = _baseline_module()

    def fake_measure(target):
        return {
            "module": target,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 1,
            "failing_tests": [
                "tests.unit.test_sample::test_bad",
                "tests.unit.test_sample::test_broken",
            ],
            "seconds": 0.1,
        }

    monkeypatch.setattr(module, "measure_module", fake_measure)

    try:
        module.main(["--base", "HEAD", "--head", "HEAD", "--modules", name])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("gate must fail when the measured suite had failures")

    captured = capsys.readouterr()
    assert "FAILED tests.unit.test_sample::test_bad" in captured.out
    assert "FAILED tests.unit.test_sample::test_broken" in captured.out
    assert "pytest rc=1 (2 failing tests)" in captured.err


def test_failed_suite_verdict_truncates_long_failure_lists(monkeypatch, capsys):
    """Failure lists longer than 20 names are bounded in the verdict."""
    module = _load_module()
    name = _baseline_module()

    def fake_measure(target):
        return {
            "module": target,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 1,
            "failing_tests": [
                f"tests.unit.test_sample::test_bad_{index}" for index in range(25)
            ],
            "seconds": 0.1,
        }

    monkeypatch.setattr(module, "measure_module", fake_measure)

    try:
        module.main(["--base", "HEAD", "--head", "HEAD", "--modules", name])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("gate must fail when the measured suite had failures")

    out = capsys.readouterr().out
    printed = [line for line in out.splitlines() if line.startswith("  FAILED ")]
    assert len(printed) == 20
    assert "... and 5 more failing tests" in out


def _fake_git_diff(monkeypatch, name_only_output: str, per_file_output: str):
    """Patch subprocess.run used by _changed_modules: the name-only diff
    returns ``name_only_output``; per-file ``-U0`` diffs return
    ``per_file_output``."""

    def fake_run(cmd, **kwargs):
        if "-U0" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=per_file_output, stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=name_only_output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_version_only_bump_excluded_from_remeasurement(monkeypatch):
    """GS19-01: a module whose only src change is its __version__ literal is
    not re-measured by the gate."""
    module = _load_module()
    name_only = "GEO-INFER-AGENT/src/geo_infer_agent/__init__.py\n"
    version_hunk = (
        "--- a/GEO-INFER-AGENT/src/geo_infer_agent/__init__.py\n"
        "+++ b/GEO-INFER-AGENT/src/geo_infer_agent/__init__.py\n"
        '+__version__ = "0.3.0"\n'
        '-__version__ = "0.2.1"\n'
    )
    _fake_git_diff(monkeypatch, name_only, version_hunk)
    assert module._changed_modules("base", "head") == set()


def test_content_change_still_remeasured(monkeypatch):
    """GS19-01: non-version changes keep the module in the re-measurement set."""
    module = _load_module()
    name_only = "GEO-INFER-AGENT/src/geo_infer_agent/__init__.py\n"
    content_hunk = (
        "--- a/GEO-INFER-AGENT/src/geo_infer_agent/__init__.py\n"
        "+++ b/GEO-INFER-AGENT/src/geo_infer_agent/__init__.py\n"
        "+import os\n"
        "-import sys\n"
    )
    _fake_git_diff(monkeypatch, name_only, content_hunk)
    assert module._changed_modules("base", "head") == {"GEO-INFER-AGENT"}


def test_failed_suite_cannot_be_hidden_by_a_successful_retry(monkeypatch, capsys):
    """A first assertion failure is authoritative; no implicit retry may hide it."""
    module = _load_module()
    name = _baseline_module()
    calls: list[str] = []

    def fake_measure(target):
        calls.append(target)
        if len(calls) == 1:
            return {
                "module": target,
                "status": "measured",
                "coverage_percent": 100.0,
                "pytest_rc": 1,
                "failing_tests": ["tests.unit.test_sample::test_flaky"],
                "seconds": 0.1,
            }
        return {
            "module": target,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 0,
            "seconds": 0.1,
        }

    monkeypatch.setattr(module, "measure_module", fake_measure)
    import pytest

    with pytest.raises(SystemExit) as exc:
        module.main(["--base", "HEAD", "--head", "HEAD", "--modules", name])
    assert exc.value.code == 1
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "retrying once" not in out
    assert "FAILED-SUITE" in out


def test_failed_suite_keeps_first_failure(monkeypatch, capsys):
    """A failed measurement is attempted once and preserves its original verdict."""
    module = _load_module()
    name = _baseline_module()
    calls: list[str] = []

    def fake_measure(target):
        calls.append(target)
        return {
            "module": target,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 1,
            "failing_tests": ["tests.unit.test_sample::test_bad"],
            "seconds": 0.1,
        }

    monkeypatch.setattr(module, "measure_module", fake_measure)
    try:
        module.main(["--base", "HEAD", "--head", "HEAD", "--modules", name])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("gate must fail when the retry also fails")
    assert len(calls) == 1
    captured = capsys.readouterr()
    assert "FAILED-SUITE" in captured.out
    assert "pytest rc=1 (1 failing tests)" in captured.err


def test_deleted_test_triggers_real_git_diff_measurement(tmp_path, monkeypatch):
    """Deleting the only test cannot bypass the diff-scoped coverage floor."""
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    test = tmp_path / "GEO-INFER-SAMPLE" / "tests" / "unit" / "test_value.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_value(): assert True\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "core.fsmonitor", "false"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    commit = [
        "git",
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "-qm",
    ]
    subprocess.run([*commit, "initial test"], cwd=tmp_path, check=True)
    test.unlink()
    subprocess.run(["git", "add", "-u"], cwd=tmp_path, check=True)
    subprocess.run([*commit, "delete test"], cwd=tmp_path, check=True)
    assert module._changed_modules("HEAD^", "HEAD") == {"GEO-INFER-SAMPLE"}


def test_shared_lock_change_measures_all_modules(monkeypatch):
    module = _load_module()
    _fake_git_diff(monkeypatch, "uv.lock\n", "+dependency changed\n")
    expected = {
        path.name
        for path in REPO_ROOT.glob("GEO-INFER-*")
        if path.is_dir() and (path / "tests").is_dir()
    }
    expected.add("ROOT")
    assert module._changed_modules("base", "head") == expected


def test_root_manuscript_source_change_selects_root_profile(monkeypatch):
    module = _load_module()
    _fake_git_diff(
        monkeypatch,
        "manuscript/generate_research_artifacts.py\n",
        "+behavior changed\n",
    )
    assert module._changed_modules("base", "head") == {"ROOT"}


def test_registered_extra_test_root_changes_measure_place(monkeypatch):
    module = _load_module()
    _fake_git_diff(
        monkeypatch,
        "GEO-INFER-PLACE/locations/cascadia/tests/unit/test_import.py\n",
        "+assert value\n",
    )
    assert module._changed_modules("base", "head") == {"GEO-INFER-PLACE"}


def test_dependency_manifest_change_remeasures_consumers(monkeypatch):
    module = _load_module()
    _fake_git_diff(
        monkeypatch, "GEO-INFER-TIME/pyproject.toml\n", "+dependency changed\n"
    )
    names = module._changed_modules("base", "head")
    assert {
        "GEO-INFER-TIME",
        "GEO-INFER-SPACE",
        "GEO-INFER-DATA",
        "GEO-INFER-IOT",
        "ROOT",
    } <= names
    assert len(names) == 46


def test_explicit_empty_selection_fails():
    import pytest

    with pytest.raises(SystemExit) as failure:
        _load_module().main(["--base", "HEAD", "--head", "HEAD", "--modules", ", ,"])
    assert failure.value.code == 2


def test_worker_budget_and_root_after_package_writers(monkeypatch):
    import threading

    module = _load_module()
    names = sorted(json.loads(module.MANIFEST.read_text())["modules"])[:2]
    lock, both_started = threading.Lock(), threading.Event()
    active = peak = 0
    completed = []

    def measure(name):
        nonlocal active, peak
        if name == "ROOT":
            assert active == 0 and set(completed) == set(names)
        else:
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    both_started.set()
            assert both_started.wait(5)
            with lock:
                active -= 1
                completed.append(name)
        return {
            "module": name,
            "status": "measured",
            "coverage_percent": 100.0,
            "pytest_rc": 0,
        }

    monkeypatch.setattr(module, "measure_module", measure)
    assert (
        module.main(
            [
                "--base",
                "HEAD",
                "--head",
                "HEAD",
                "--modules",
                ",".join([*names, "ROOT"]),
                "--workers",
                "2",
            ]
        )
        == 0
    )
    assert peak == 2
