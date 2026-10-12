"""
Unit and property-based tests for the GeoInferTestRunner and helpers.
"""

import json
import string
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import psutil
import geo_infer_test.core.test_runner as runner_module
from geo_infer_test.execution import CommandResult
from geo_infer_test.core.test_runner import (
    GeoInferTestRunner as _GeoInferTestRunner,
)

# Alias imports to prevent pytest collection warnings
from geo_infer_test.core.test_runner import (
    TestConfiguration as _TestConfiguration,
)
from geo_infer_test.core.test_runner import (
    TestResult as _TestResult,
)
from hypothesis import given, settings
from hypothesis import strategies as st

# ============================================================================
# TestConfiguration dataclass tests
# ============================================================================


class TestTestConfiguration:
    """Tests for the TestConfiguration dataclass."""

    def test_defaults(self):
        cfg = _TestConfiguration(modules_to_test=["A"], test_types=["unit"])
        assert cfg.parallel_execution is True
        assert cfg.max_workers == 4
        assert cfg.timeout_seconds == 300
        assert cfg.fail_fast is False
        assert cfg.coverage_enabled is True

    @pytest.mark.parametrize(
        "modules",
        [
            ["A"],
            ["A", "B"],
            ["A", "B", "C", "D", "E"],
            [f"MOD_{i}" for i in range(20)],
        ],
    )
    def test_modules_list(self, modules):
        cfg = _TestConfiguration(modules_to_test=modules, test_types=["unit"])
        assert cfg.modules_to_test == modules

    @pytest.mark.parametrize(
        "types",
        [
            ["unit"],
            ["integration"],
            ["unit", "integration"],
            ["unit", "integration", "performance"],
        ],
    )
    def test_test_types(self, types):
        cfg = _TestConfiguration(modules_to_test=["A"], test_types=types)
        assert cfg.test_types == types

    @pytest.mark.parametrize("workers", [1, 2, 4, 8, 16])
    def test_max_workers(self, workers):
        cfg = _TestConfiguration(
            modules_to_test=["A"], test_types=["unit"], max_workers=workers
        )
        assert cfg.max_workers == workers

    @pytest.mark.parametrize("timeout", [10, 60, 300, 600, 3600])
    def test_timeout(self, timeout):
        cfg = _TestConfiguration(
            modules_to_test=["A"], test_types=["unit"], timeout_seconds=timeout
        )
        assert cfg.timeout_seconds == timeout

    @pytest.mark.parametrize(
        "parallel, fail_fast, coverage, perf",
        [
            (True, False, True, True),
            (False, True, False, False),
            (True, True, True, False),
            (False, False, False, True),
        ],
    )
    def test_boolean_flags(self, parallel, fail_fast, coverage, perf):
        cfg = _TestConfiguration(
            modules_to_test=["A"],
            test_types=["unit"],
            parallel_execution=parallel,
            fail_fast=fail_fast,
            coverage_enabled=coverage,
            performance_benchmarks=perf,
        )
        assert cfg.parallel_execution == parallel
        assert cfg.fail_fast == fail_fast
        assert cfg.coverage_enabled == coverage
        assert cfg.performance_benchmarks == perf


# ============================================================================
# TestResult dataclass tests
# ============================================================================


class TestTestResultDataclass:
    """Tests for the TestResult dataclass."""

    def test_basic_creation(self):
        r = _TestResult(
            test_id="t1",
            module="mod",
            test_name="test_foo",
            status="passed",
            duration=0.5,
            message="ok",
            details={},
        )
        assert r.test_id == "t1"
        assert r.status == "passed"
        assert r.performance_metrics is None

    @pytest.mark.parametrize("status", ["passed", "failed", "error", "skipped"])
    def test_status_values(self, status):
        r = _TestResult(
            test_id="t1",
            module="mod",
            test_name="test_x",
            status=status,
            duration=1.0,
            message="",
            details={},
        )
        assert r.status == status

    @pytest.mark.parametrize("duration", [0.0, 0.001, 1.0, 10.0, 100.0, 300.0])
    def test_duration_values(self, duration):
        r = _TestResult(
            test_id="t1",
            module="mod",
            test_name="test_x",
            status="passed",
            duration=duration,
            message="",
            details={},
        )
        assert r.duration == duration

    def test_with_performance_metrics(self):
        r = _TestResult(
            test_id="t1",
            module="mod",
            test_name="test_x",
            status="passed",
            duration=1.0,
            message="",
            details={},
            performance_metrics={"memory": 1024, "cpu": 0.5},
        )
        assert r.performance_metrics["memory"] == 1024


# ============================================================================
# GeoInferTestRunner tests
# ============================================================================


class TestGeoInferTestRunner:
    """Tests for GeoInferTestRunner core logic."""

    def test_runner_initialization(self):
        cfg = _TestConfiguration(modules_to_test=["SPACE"], test_types=["unit"])
        runner = _GeoInferTestRunner(cfg)
        assert runner.config == cfg

    def test_runner_setup_environment(self):
        cfg = _TestConfiguration(modules_to_test=["SPACE"], test_types=["unit"])
        runner = _GeoInferTestRunner(cfg)
        # Should not raise
        runner._setup_test_environment()

    def test_runner_initialization_does_not_create_test_tree(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        cfg = _TestConfiguration(
            modules_to_test=["SPACE"],
            test_types=["unit"],
            log_integration_enabled=False,
        )

        _GeoInferTestRunner(cfg)

        assert not (tmp_path / "tests").exists()

    def test_runner_uses_complete_module_registry(self):
        assert {"CLIMATE", "EDU", "EMERGENCY", "TRANSPORT"}.issubset(
            set(_GeoInferTestRunner.AVAILABLE_MODULES)
        )

    def test_runner_discovers_nested_test_files(self, tmp_path, monkeypatch):
        # Discovery is anchored to the repo root, not the CWD; point the
        # runner at tmp_path for the duration of the test.
        monkeypatch.setattr("geo_infer_test.core.test_runner._REPO_ROOT", tmp_path)
        test_file = (
            tmp_path
            / "GEO-INFER-SAMPLE"
            / "tests"
            / "unit"
            / "nested"
            / "test_nested.py"
        )
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_nested():\n    assert True\n")
        cfg = _TestConfiguration(
            modules_to_test=["SAMPLE"],
            test_types=["unit"],
            log_integration_enabled=False,
        )
        runner = _GeoInferTestRunner(cfg)

        assert runner._discover_module_tests("SAMPLE") == [
            "SAMPLE::unit::nested/test_nested"
        ]

    def test_core_exports_only_defined_names(self):
        import geo_infer_test.core as core

        assert all(hasattr(core, name) for name in core.__all__)

    def test_top_level_runner_exports_match_documented_example(self):
        import geo_infer_test

        assert geo_infer_test.GeoInferTestRunner is _GeoInferTestRunner

    @pytest.mark.parametrize(
        "module",
        [
            "SPACE",
            "TIME",
            "AI",
            "BAYES",
            "ACT",
            "AGENT",
            "SEC",
            "APP",
            "API",
            "LOG",
            "DATA",
            "OPS",
            "RISK",
        ],
    )
    def test_runner_discover_module(self, module):
        cfg = _TestConfiguration(modules_to_test=[module], test_types=["unit"])
        runner = _GeoInferTestRunner(cfg)
        # Discovery should return a structure even for non-existent modules
        tests = runner._discover_module_tests(module)
        assert isinstance(tests, (list, dict, type(None)))

    def test_runner_discover_tests_structure(self):
        cfg = _TestConfiguration(modules_to_test=["SPACE", "TIME"], test_types=["unit"])
        runner = _GeoInferTestRunner(cfg)
        discovered = runner.discover_tests()
        assert isinstance(discovered, dict)

    def test_report_generation(self):
        cfg = _TestConfiguration(modules_to_test=["SPACE"], test_types=["unit"])
        runner = _GeoInferTestRunner(cfg)
        # Add test results using the actual attribute name and status values
        runner.test_results = [
            _TestResult(
                test_id=f"t{i}",
                module="SPACE",
                test_name=f"test_{i}",
                status="PASS" if i % 2 == 0 else "FAIL",
                duration=float(i) * 0.1,
                message="",
                details={},
            )
            for i in range(10)
        ]
        report = runner._generate_execution_report(1.5)
        summary = report["execution_summary"]
        assert summary["total_tests"] == 10
        assert summary["passed"] == 5
        assert summary["failed"] == 5


# ============================================================================
# Property-Based Tests (Hypothesis)
# ============================================================================


class TestHypothesisTestRunner:
    """Property-based tests for test runner components."""

    @settings(max_examples=200)
    @given(
        st.lists(
            st.text(
                min_size=1,
                max_size=10,
                alphabet=string.ascii_uppercase,
            ),
            min_size=1,
            max_size=15,
            unique=True,
        )
    )
    def test_config_modules_preserved(self, modules):
        """All module names should survive config round-trip."""
        cfg = _TestConfiguration(modules_to_test=modules, test_types=["unit"])
        assert cfg.modules_to_test == modules

    @settings(max_examples=200)
    @given(
        st.text(min_size=1, max_size=20),
        st.text(min_size=1, max_size=20),
        st.text(min_size=1, max_size=30),
        st.sampled_from(["passed", "failed", "error", "skipped"]),
        st.floats(min_value=0.0, max_value=300.0),
    )
    def test_result_creation_never_crashes(
        self, test_id, module, name, status, duration
    ):
        """TestResult should handle any valid inputs."""
        r = _TestResult(
            test_id=test_id,
            module=module,
            test_name=name,
            status=status,
            duration=duration,
            message="",
            details={},
        )
        assert r.test_id == test_id
        assert r.status == status

    @settings(max_examples=200)
    @given(st.integers(min_value=1, max_value=16))
    def test_runner_worker_configs(self, workers):
        """Runner should accept any valid worker count."""
        cfg = _TestConfiguration(
            modules_to_test=["SPACE"], test_types=["unit"], max_workers=workers
        )
        runner = _GeoInferTestRunner(cfg)
        assert runner.config.max_workers == workers

    @settings(max_examples=200)
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.sampled_from(["passed", "failed", "error", "skipped"]),
                st.floats(min_value=0.0, max_value=10.0),
            ),
            min_size=1,
            max_size=50,
        )
    )
    def test_report_counts_correct(self, test_tuples):
        """Report passed/failed counts should match input."""
        cfg = _TestConfiguration(modules_to_test=["X"], test_types=["unit"])
        runner = _GeoInferTestRunner(cfg)

        # Map pytest-style statuses to runner's internal statuses
        status_map = {
            "passed": "PASS",
            "failed": "FAIL",
            "error": "ERROR",
            "skipped": "SKIP",
        }
        results = []
        for i, (name, status, dur) in enumerate(test_tuples):
            results.append(
                _TestResult(
                    test_id=f"t{i}",
                    module="X",
                    test_name=name,
                    status=status_map.get(status, status),
                    duration=dur,
                    message="",
                    details={},
                )
            )
        runner.test_results = results

        report = runner._generate_execution_report(1.0)
        summary = report["execution_summary"]
        assert summary["total_tests"] == len(test_tuples)
        expected_passed = sum(1 for _, s, _ in test_tuples if s == "passed")
        assert summary["passed"] == expected_passed


@pytest.fixture
def public_runner(tmp_path, monkeypatch):
    """Real discovery and runner; replace only the subprocess execution boundary."""
    monkeypatch.setattr(runner_module, "_REPO_ROOT", tmp_path)
    for relative in (
        "GEO-INFER-SPACE/tests/unit/test_first.py",
        "GEO-INFER-SPACE/tests/unit/nested/test_second.py",
        "GEO-INFER-SPACE/tests/integration/nested/test_link.py",
        "GEO-INFER-TIME/tests/unit/test_clock.py",
        "GEO-INFER-TIME/tests/integration/test_link.py",
        "tests/integration/test_root.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_local():\n    assert 2 + 2 == 4\n")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    monkeypatch.setattr(runner_module, "run_results_dir", lambda: receipts)
    calls = []

    def command(args, name, *, timeout, cwd):
        calls.append((args, name, timeout, cwd))
        receipt = receipts / f"attempt-{len(calls)}.json"
        receipt.write_text(json.dumps({"simulated_boundary": True, "name": name}))
        return CommandResult(
            name,
            True,
            0.125,
            args,
            timeout=timeout,
            returncode=0,
            executed=3,
            receipt=str(receipt),
        )

    monkeypatch.setattr(runner_module, "run_command", command)
    config = _TestConfiguration(
        modules_to_test=["SPACE", "TIME"],
        test_types=["unit"],
        parallel_execution=False,
        log_integration_enabled=False,
    )
    return _GeoInferTestRunner(config), calls, tmp_path


class TestPublicRunnerBehavior:
    def test_run_all_discovers_deduplicates_and_preserves_command_receipts(
        self, public_runner
    ):
        runner, calls, root = public_runner
        runner.config.modules_to_test = ["SPACE", "TIME", "SPACE"]
        runner.config.test_types = ["unit", "unit"]
        report = runner.run_all_tests()
        summary = report["execution_summary"]
        assert (summary["total_tests"], summary["passed"], summary["testcases"]) == (
            3,
            3,
            9,
        )
        assert summary["success"] is True
        assert summary["not_run"] == 0
        assert {c[1] for c in calls} == {
            "SPACE unit test_first",
            "SPACE unit nested/test_second",
            "TIME unit test_clock",
        }
        assert len(calls) == 3
        for args, _, timeout, cwd in calls:
            assert cwd == root
            assert 0 < timeout <= runner.config.timeout_seconds
            assert ["-m", "not slow"] == args[-4:-2]
        details = [result["details"] for result in report["test_results"]]
        assert {d["receipt"] for d in details} == {
            str(root / "receipts" / f"attempt-{i}.json") for i in (1, 2, 3)
        }
        assert all(d["testcases"] == 3 and d["returncode"] == 0 for d in details)
        assert all(
            json.loads((root / "receipts" / f"attempt-{i}.json").read_text())[
                "simulated_boundary"
            ]
            for i in (1, 2, 3)
        )
        # A second public run must discard previous result and receipt state.
        second = runner.run_all_tests()
        assert second["execution_summary"]["total_tests"] == 3
        assert all(
            "attempt-" in r["details"]["receipt"] for r in second["test_results"]
        )
        assert {r["details"]["receipt"] for r in second["test_results"]}.isdisjoint(
            {d["receipt"] for d in details}
        )

    @pytest.mark.parametrize(
        "modules,types", [([], ["unit"]), (["SPACE"], []), (["BIO"], ["unit"])]
    )
    def test_run_all_rejects_empty_selection(self, public_runner, modules, types):
        runner, calls, _ = public_runner
        runner.config.modules_to_test = modules
        runner.config.test_types = types
        with pytest.raises(ValueError, match="No tests discovered"):
            runner.run_all_tests()
        assert calls == []

    @pytest.mark.parametrize("entrypoint", ["run_all_tests", "run_module_tests"])
    def test_unknown_module_rejected_without_commands(self, public_runner, entrypoint):
        runner, calls, _ = public_runner
        original = runner.config.modules_to_test
        with pytest.raises(ValueError, match="Unknown module: UNKNOWN"):
            if entrypoint == "run_all_tests":
                runner.config.modules_to_test = ["UNKNOWN"]
                runner.run_all_tests()
            else:
                runner.run_module_tests("UNKNOWN")
        assert calls == []
        if entrypoint == "run_module_tests":
            assert runner.config.modules_to_test is original

    def test_module_run_restores_configuration_and_prior_discovery(self, public_runner):
        runner, calls, _ = public_runner
        original = runner.config.modules_to_test
        discovered = runner.discover_tests()
        report = runner.run_module_tests("TIME")
        assert report["configuration"]["modules_tested"] == ["TIME"]
        assert report["execution_summary"]["total_tests"] == 1
        assert runner.config.modules_to_test is original
        assert runner.discovered_tests == discovered
        assert runner.run_all_tests()["execution_summary"]["total_tests"] == 3
        assert len(calls) == 4

    def test_module_run_rejects_empty_and_restores_config(self, public_runner):
        runner, calls, _ = public_runner
        original = runner.config.modules_to_test
        with pytest.raises(ValueError, match="No tests discovered"):
            runner.run_module_tests("BIO")
        assert runner.config.modules_to_test is original
        assert calls == []

    @pytest.mark.parametrize("method", ["run_module_tests", "run_cross_module_tests"])
    def test_scoped_run_restores_config_on_interrupt(
        self, public_runner, monkeypatch, method
    ):
        runner, _, _ = public_runner
        modules, types = runner.config.modules_to_test, runner.config.test_types

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(runner, "_discover_module_tests", interrupt)
        with pytest.raises(KeyboardInterrupt):
            if method == "run_module_tests":
                runner.run_module_tests("SPACE")
            else:
                runner.run_cross_module_tests()
        assert runner.config.modules_to_test is modules
        assert runner.config.test_types is types

    def test_cross_module_run_includes_nested_integration_and_respects_root_profile(
        self, public_runner
    ):
        runner, calls, _ = public_runner
        modules, types = runner.config.modules_to_test, runner.config.test_types
        runner.discover_tests()  # Cached unit selection must not leak into integration.
        report = runner.run_cross_module_tests()
        assert report["execution_summary"]["total_tests"] == 2
        assert report["execution_summary"]["testcases"] == 6
        assert set(report["module_summaries"]) == {"SPACE", "TIME"}
        assert {c[1] for c in calls} == {
            "SPACE integration nested/test_link",
            "TIME integration test_link",
        }
        assert report["configuration"]["test_types"] == ["integration"]
        assert runner.config.modules_to_test is modules
        assert runner.config.test_types is types
        assert runner.discovered_tests == {}

    @pytest.mark.parametrize("failure", ["return-fail", "raise-error", "malformed"])
    @pytest.mark.parametrize("fail_fast", [False, True])
    def test_sequential_failure_reports_and_fail_fast(
        self, public_runner, monkeypatch, failure, fail_fast
    ):
        runner, calls, _ = public_runner
        runner.config.fail_fast = fail_fast
        runner.discovered_tests = {
            "SPACE": ["SPACE::unit::test_first", "SPACE::unit::nested/test_second"]
        }
        if failure == "malformed":
            runner.discovered_tests["SPACE"][0] = "invalid"
        real_command = runner_module.run_command

        def failing(args, name, **kwargs):
            if name == "SPACE unit test_first":
                if failure == "raise-error":
                    raise RuntimeError("receipt unavailable")
                result = real_command(args, name, **kwargs)
                result.success = False
                result.returncode = 1
                return result
            return real_command(args, name, **kwargs)

        monkeypatch.setattr(runner_module, "run_command", failing)
        report = runner.run_all_tests()
        summary = report["execution_summary"]
        assert summary["success"] is False
        assert summary["total_tests"] == (1 if fail_fast else 2)
        assert summary["not_run"] == (1 if fail_fast else 0)
        expected = "FAIL" if failure == "return-fail" else "ERROR"
        assert report["test_results"][0]["status"] == expected
        if not fail_fast:
            assert report["test_results"][1]["status"] == "PASS"
        if failure == "raise-error":
            assert (
                report["test_results"][0]["details"]["error"] == "receipt unavailable"
            )
        if failure == "malformed":
            assert "Malformed test identifier" in report["test_results"][0]["message"]

    def test_logging_context_wraps_public_execution(self, public_runner):
        runner, calls, _ = public_runner
        events = []
        integration = Mock()

        @contextmanager
        def context(test_id, module, name):
            events.append(("enter", module, name, len(calls)))
            yield
            events.append(("exit", module, name, len(calls)))

        integration.test_context.side_effect = context
        runner = _GeoInferTestRunner(
            _TestConfiguration(
                modules_to_test=["TIME"],
                test_types=["unit"],
                parallel_execution=False,
                log_integration=integration,
            )
        )
        report = runner.run_all_tests()
        assert events == [
            ("enter", "TIME", "unit_test_clock", 0),
            ("exit", "TIME", "unit_test_clock", 1),
        ]
        assert report["execution_summary"]["success"] is True
        assert integration.logger.info.call_count >= 4


@pytest.fixture
def controlled_executor(monkeypatch):
    """A lazy executor makes admission/cancellation ordering deterministic."""
    events, submitted = [], []

    class Executor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers

        def submit(self, function, *args):
            future = Future()
            future.work = (function, args)
            submitted.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures):
            events.append(("shutdown", wait, cancel_futures))
            if cancel_futures:
                for future in submitted:
                    future.cancel()

    def completed(futures):
        events.append(("window", len(futures)))
        for future in list(futures):
            function, args = future.work
            try:
                future.set_result(function(*args))
            except Exception as exc:
                future.set_exception(exc)
            yield future

    monkeypatch.setattr(runner_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(runner_module, "as_completed", completed)
    monkeypatch.setattr(
        runner_module, "reset_process_cancellation", lambda: events.append("reset")
    )
    monkeypatch.setattr(
        runner_module, "terminate_running_processes", lambda: events.append("terminate")
    )
    return events, submitted


class TestPublicParallelRunner:
    @pytest.mark.parametrize("join_returns_early", [False, True])
    def test_second_interrupt_retains_worker_receipt_before_reuse(
        self, public_runner, monkeypatch, join_returns_early
    ):
        from geo_infer_test import process as proc

        runner, calls, root = public_runner
        runner.config.parallel_execution = True
        runner.config.max_workers = 1
        runner.discovered_tests = {"SPACE": ["SPACE::unit::test_first"]}
        started, release, finished = (threading.Event() for _ in range(3))
        cancellation_seen = []
        executors, submitted = [], []
        native_executor = runner_module.ThreadPoolExecutor
        native_wait = runner_module.wait
        native_completed = runner_module.as_completed
        native_command = runner_module.run_command
        shutdown_calls = []

        class InterruptedExecutor(native_executor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                executors.append(self)

            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                submitted.append(future)
                return future

            def shutdown(self, *, wait, cancel_futures):
                shutdown_calls.append((wait, cancel_futures))
                if len(shutdown_calls) == 1:
                    assert started.wait(5)
                    assert proc._CANCELLED.is_set()
                    assert not submitted[0].done()
                    super().shutdown(wait=False, cancel_futures=cancel_futures)
                    raise KeyboardInterrupt("second interruption")
                assert proc._CANCELLED.is_set()
                assert runner.test_results == []
                if join_returns_early:
                    # Model a join marked complete by the interruption while
                    # its real worker is still finishing a receipt.
                    return
                release.set()
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        def command(*args, **kwargs):
            started.set()
            assert release.wait(5), "cleanup never released the worker"
            cancellation_seen.append(proc._CANCELLED.is_set())
            result = native_command(*args, **kwargs)
            finished.set()
            return result

        def interrupted_completion(futures):
            assert started.wait(5)
            raise KeyboardInterrupt("first interruption")

        def completion_barrier(futures):
            assert proc._CANCELLED.is_set()
            if join_returns_early:
                assert not finished.is_set()
                assert not submitted[0].done()
                release.set()
            return native_wait(futures)

        monkeypatch.setattr(runner_module, "ThreadPoolExecutor", InterruptedExecutor)
        monkeypatch.setattr(runner_module, "run_command", command)
        monkeypatch.setattr(runner_module, "as_completed", interrupted_completion)
        monkeypatch.setattr(runner_module, "wait", completion_barrier)
        try:
            with pytest.raises(KeyboardInterrupt, match="second interruption"):
                runner.run_all_tests()
            assert shutdown_calls == [(True, True), (True, True)]
            assert finished.is_set() and submitted[0].done()
            assert cancellation_seen == [True]
            assert not proc._CANCELLED.is_set()
            assert len(runner.test_results) == 1
            retained = runner.test_results[0]
            assert retained.details["abort_requested"] == "interruption"
            receipt = Path(retained.details["receipt"])
            assert receipt == root / "receipts" / "attempt-1.json"
            assert json.loads(receipt.read_text())["simulated_boundary"] is True
            assert len(runner._command_results) == 1

            monkeypatch.setattr(runner_module, "ThreadPoolExecutor", native_executor)
            monkeypatch.setattr(runner_module, "as_completed", native_completed)
            monkeypatch.setattr(runner_module, "wait", native_wait)
            report = runner.run_all_tests()
            assert report["execution_summary"]["success"] is True
            assert report["execution_summary"]["total_tests"] == 1
            assert cancellation_seen == [True, False]
            assert len(calls) == 2
            assert len(runner._command_results) == 1
            assert report["test_results"][0]["details"]["receipt"] == str(
                root / "receipts" / "attempt-2.json"
            )
            assert receipt.exists()
            assert not proc._CANCELLED.is_set()
        finally:
            release.set()
            for executor in executors:
                native_executor.shutdown(executor, wait=True, cancel_futures=True)
            proc.reset_process_cancellation()

    @pytest.mark.parametrize(
        "outcome", ["PASS", "FAIL", "exception", "interrupt", "no_result"]
    )
    def test_fail_fast_collects_already_running_and_completed_futures(
        self, public_runner, controlled_executor, monkeypatch, outcome
    ):
        runner, _, _ = public_runner
        events, submitted = controlled_executor
        runner.config.parallel_execution = True
        runner.config.max_workers = 2
        runner.config.fail_fast = True
        runner.discovered_tests = {"TEST": ["trigger", "admitted", "never"]}

        def worker(module, test):
            if test == "trigger":
                return _TestResult(test, module, test, "FAIL", 0, "trigger", {})
            if outcome == "exception":
                raise RuntimeError("late worker exception")
            if outcome == "interrupt":
                raise KeyboardInterrupt
            if outcome == "no_result":
                return None
            return _TestResult(test, module, test, outcome, 0, "finished", {})

        def completed(futures):
            # Both admitted futures have executed before the trigger is
            # observed. Neither can be cancelled, even though one is unread.
            for future in futures:
                future.set_running_or_notify_cancel()
                function, args = future.work
                try:
                    future.set_result(function(*args))
                except BaseException as exc:
                    future.set_exception(exc)
            yield submitted[0]

        monkeypatch.setattr(runner, "_execute_single_test", worker)
        monkeypatch.setattr(runner_module, "as_completed", completed)
        report = runner.run_all_tests()
        assert len(submitted) == 2
        assert report["execution_summary"]["total_tests"] == 2
        assert report["execution_summary"]["not_run"] == 1
        late = report["test_results"][1]
        assert late["status"] == (
            "ERROR" if outcome in {"exception", "interrupt", "no_result"} else outcome
        )
        if outcome == "exception":
            assert late["details"]["error"] == "late worker exception"
        if outcome == "interrupt":
            assert late["details"]["error"] == "KeyboardInterrupt"
        assert "abort_requested" not in late["details"]
        assert events[-2:] == [("shutdown", True, True), "reset"]

    @pytest.mark.parametrize(
        "stop_reason", ["fail_fast", "interruption", "runner_error"]
    )
    def test_real_parallel_stop_retains_ready_child_receipt_after_reap(
        self, tmp_path, monkeypatch, stop_reason
    ):
        from geo_infer_test import execution as engine, process as proc

        runner = _GeoInferTestRunner(
            _TestConfiguration(
                modules_to_test=["TEST"],
                test_types=["unit"],
                max_workers=2,
                fail_fast=True,
                timeout_seconds=15,
                log_integration_enabled=False,
            )
        )
        runner.discovered_tests = {
            "TEST": [
                "TEST::unit::trigger",
                "TEST::unit::running",
                "TEST::unit::never",
            ]
        }
        ready = tmp_path / "ready.json"
        release = threading.Event()
        commands, children = [], []
        monkeypatch.setattr(engine, "RESULTS_DIR", tmp_path / "engine")
        native_popen = proc.subprocess.Popen

        def observed_popen(*args, **kwargs):
            child = native_popen(*args, **kwargs)
            children.append(child)
            return child

        monkeypatch.setattr(proc.subprocess, "Popen", observed_popen)
        code = (
            "from pathlib import Path; import os,time,json,psutil; "
            "print('ready concurrent worker',flush=True); "
            f"Path({str(ready)!r}).write_text(json.dumps({{'pid':os.getpid(),"
            "'birth':psutil.Process().create_time()})); "
            "time.sleep(30)"
        )

        def await_ready():
            deadline = time.monotonic() + 8
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "concurrent command never became ready"

        def command(module, category, filename):
            commands.append(filename)
            if filename == "trigger":
                await_ready()
                if stop_reason != "fail_fast":
                    assert release.wait(8), "runner never initiated stop"
                return False
            result = engine.run_command(
                [sys.executable, "-c", code],
                "ready concurrent child",
                timeout=10,
                cwd=tmp_path,
            )
            runner._command_results[f"{module}::{category}::{filename}"] = result
            return result.success

        monkeypatch.setattr(runner, "_run_pytest_test", command)
        if stop_reason != "fail_fast":

            def interrupted_completion(futures):
                await_ready()
                release.set()
                if stop_reason == "interruption":
                    raise KeyboardInterrupt
                raise RuntimeError("coordinator failed")

            monkeypatch.setattr(runner_module, "as_completed", interrupted_completion)

        try:
            if stop_reason == "fail_fast":
                report = runner.run_all_tests()
            else:
                exception = (
                    KeyboardInterrupt if stop_reason == "interruption" else RuntimeError
                )
                with pytest.raises(exception):
                    runner.run_all_tests()
                report = runner._generate_execution_report(0)
            assert sorted(commands) == ["running", "trigger"]
            summary = report["execution_summary"]
            assert (summary["total_tests"], summary["not_run"], summary["success"]) == (
                2,
                1,
                False,
            )
            result = next(
                r for r in report["test_results"] if r["test_name"] == "unit_running"
            )
            assert result["status"] == "FAIL"
            assert result["details"]["abort_requested"] == stop_reason
            receipt = json.loads(Path(result["details"]["receipt"]).read_text())
            assert receipt["returncode"] == result["details"]["returncode"]
            assert receipt["status"] == result["details"]["command_status"]
            assert receipt["success"] is False
            assert receipt["process_evidence"]["target_returncode"] != 0
            assert (
                "ready concurrent worker"
                in Path(result["details"]["receipt"])
                .with_name("stdout.log")
                .read_text()
            )
            identity = json.loads(ready.read_text())
            matching = [p for p in children if p.pid == identity["pid"]]
            assert len(matching) == 1
            assert matching[0].poll() is not None
            assert matching[0].returncode != 0
            assert not proc._ACTIVE_PROCESSES
            assert not proc._CANCELLED.is_set()
            try:
                live = psutil.Process(identity["pid"])
                assert live.create_time() != identity["birth"] or not live.is_running()
            except psutil.NoSuchProcess:
                pass
            (tmp_path / "report.json").write_text(json.dumps(report, indent=2))
        finally:
            release.set()
            # Failure cleanup is limited to this test's observed children.
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

    def test_parallel_refills_bounded_window_and_reports_receipts(
        self, public_runner, controlled_executor
    ):
        runner, calls, _ = public_runner
        events, submitted = controlled_executor
        runner.config.parallel_execution = True
        runner.config.max_workers = 2
        report = runner.run_all_tests()
        assert report["execution_summary"]["passed"] == 3
        assert report["execution_summary"]["testcases"] == 9
        assert len(calls) == len(submitted) == 3
        assert (
            max(e[1] for e in events if isinstance(e, tuple) and e[0] == "window") == 2
        )
        assert events[0] == events[-1] == "reset"
        assert events[-2] == ("shutdown", True, True)

    def test_parallel_fail_fast_cancels_pending_and_stops_admission(
        self, public_runner, controlled_executor, monkeypatch
    ):
        runner, calls, _ = public_runner
        events, submitted = controlled_executor
        runner.config.parallel_execution = True
        runner.config.max_workers = 2
        runner.config.fail_fast = True
        runner.discovered_tests = {
            "SPACE": [
                "broken",
                "SPACE::unit::test_first",
                "SPACE::unit::nested/test_second",
            ]
        }
        report = runner.run_all_tests()
        assert report["execution_summary"]["errors"] == 1
        assert report["execution_summary"]["not_run"] == 2
        assert calls == []
        assert len(submitted) == 2
        assert submitted[1].cancelled()
        assert "terminate" in events
        assert events[-2:] == [("shutdown", True, True), "reset"]

    def test_parallel_interrupt_cleans_up_and_propagates(
        self, public_runner, controlled_executor, monkeypatch
    ):
        runner, _, _ = public_runner
        events, submitted = controlled_executor
        runner.config.parallel_execution = True
        runner.config.max_workers = 2

        def interrupt(futures):
            raise KeyboardInterrupt

        monkeypatch.setattr(runner_module, "as_completed", interrupt)
        with pytest.raises(KeyboardInterrupt):
            runner.run_all_tests()
        assert all(future.cancelled() for future in submitted)
        assert "terminate" in events
        assert events[-2:] == [("shutdown", True, True), "reset"]

    def test_parallel_worker_exception_is_reported(
        self, public_runner, controlled_executor, monkeypatch
    ):
        runner, _, _ = public_runner
        runner.config.parallel_execution = True

        def explode(module, test):
            raise RuntimeError("worker unavailable")

        monkeypatch.setattr(runner, "_execute_single_test", explode)
        report = runner.run_all_tests()
        assert report["execution_summary"]["errors"] == 3
        assert report["execution_summary"]["not_run"] == 0
        assert all(
            r["details"]["error"] == "worker unavailable"
            for r in report["test_results"]
        )


class TestPublicRunnerFailureBoundaries:
    @pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
    def test_invalid_timeout_rejected_before_discovery(self, timeout):
        with pytest.raises(
            ValueError, match="timeout_seconds must be finite and positive"
        ):
            _GeoInferTestRunner(
                _TestConfiguration(
                    modules_to_test=["SPACE"],
                    test_types=["unit"],
                    timeout_seconds=timeout,
                    log_integration_enabled=False,
                )
            )

    def test_invalid_worker_limit_rejected(self):
        with pytest.raises(ValueError, match="max_workers must be positive"):
            _GeoInferTestRunner(
                _TestConfiguration(
                    modules_to_test=["SPACE"],
                    test_types=["unit"],
                    max_workers=0,
                    log_integration_enabled=False,
                )
            )

    def test_shared_deadline_is_not_recharged_for_later_commands(
        self, public_runner, monkeypatch
    ):
        runner, calls, _ = public_runner
        runner.config.timeout_seconds = 10
        clock = [100.0]
        monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock[0])
        command = runner_module.run_command

        def use_budget(*args, **kwargs):
            result = command(*args, **kwargs)
            clock[0] = 111.0
            return result

        monkeypatch.setattr(runner_module, "run_command", use_budget)
        report = runner.run_all_tests()
        assert len(calls) == 1
        assert calls[0][2] == 10
        summary = report["execution_summary"]
        assert (summary["passed"], summary["failed"], summary["success"]) == (
            1,
            2,
            False,
        )
        assert summary["total_duration"] == 11
        assert all("receipt" not in r["details"] for r in report["test_results"][1:])

    @pytest.mark.parametrize(
        "filename", ["missing", "../../../../tests/integration/test_root"]
    )
    def test_missing_or_escaping_file_never_launches_command(
        self, public_runner, filename
    ):
        runner, calls, _ = public_runner
        runner.discovered_tests = {"SPACE": [f"SPACE::unit::{filename}"]}
        report = runner.run_all_tests()
        assert calls == []
        assert report["execution_summary"]["failed"] == 1
        assert report["execution_summary"]["success"] is False
        assert "receipt" not in report["test_results"][0]["details"]

    def test_cross_module_empty_selection_restores_config(self, public_runner):
        runner, calls, _ = public_runner
        original_modules, original_types = (
            runner.config.modules_to_test,
            runner.config.test_types,
        )
        for path in public_runner[2].rglob("test_*.py"):
            path.unlink()
        with pytest.raises(ValueError, match="No tests discovered"):
            runner.run_cross_module_tests()
        assert runner.config.modules_to_test is original_modules
        assert runner.config.test_types is original_types
        assert runner.discovered_tests == {}
        assert calls == []

    def test_parallel_submission_error_still_shuts_down_executor(
        self, public_runner, controlled_executor, monkeypatch
    ):
        runner, _, _ = public_runner
        events, submitted = controlled_executor
        runner.config.parallel_execution = True
        base = runner_module.ThreadPoolExecutor

        class BrokenExecutor(base):
            def submit(self, function, *args):
                if submitted:
                    raise RuntimeError("executor rejected work")
                return super().submit(function, *args)

        monkeypatch.setattr(runner_module, "ThreadPoolExecutor", BrokenExecutor)
        with pytest.raises(RuntimeError, match="executor rejected work"):
            runner.run_all_tests()
        assert len(submitted) == 1
        assert submitted[0].cancelled()
        assert events[-2:] == [("shutdown", True, True), "reset"]
