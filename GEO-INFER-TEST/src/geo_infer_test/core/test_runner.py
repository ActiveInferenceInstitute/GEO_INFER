"""
Main test runner for the GEO-INFER-TEST framework.

This module provides the core test execution engine that can run tests
across all GEO-INFER modules with comprehensive logging and reporting.
"""

import logging
import math
import os
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..process import terminate_running_processes, reset_process_cancellation
from .log_integration import LogIntegration
from .test_discoverer import ALL_MODULES
from ..execution import (
    Module,
    category_test_paths,
    pytest_base_args,
    run_command,
    run_results_dir,
    profile_selection_args,
)

logger = logging.getLogger(__name__)

# Module root (the GEO-INFER-TEST checkout); sibling module directories such as
# GEO-INFER-SPACE live beside it. Anchoring discovery and execution here keeps
# the runner independent of the current working directory, matching
# run_unified_tests.py's PROJECT_ROOT behavior.
_MODULE_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _MODULE_ROOT.parent


@dataclass
class TestConfiguration:
    """Configuration for test execution."""

    modules_to_test: list[str]
    test_types: list[str]  # ['unit', 'integration', 'performance', 'load']
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    fail_fast: bool = False
    coverage_enabled: bool = True
    performance_benchmarks: bool = True
    log_integration_enabled: bool = True
    log_integration: LogIntegration | None = None
    """Pre-built log integration injected by the caller (constructor
    injection preferred over post-construction attribute replacement)."""


@dataclass
class TestResult:
    """Result of a test execution."""

    test_id: str
    module: str
    test_name: str
    status: str
    duration: float
    message: str
    details: dict[str, Any]
    performance_metrics: dict[str, float] | None = None


class GeoInferTestRunner:
    """
    Main test runner for the GEO-INFER ecosystem.

    Provides comprehensive test execution capabilities across all modules
    with integration to GEO-INFER-LOG for detailed monitoring and reporting.
    """

    # Keep the programmatic runner aligned with the canonical discoverer.
    AVAILABLE_MODULES = (*ALL_MODULES, "ROOT")

    def __init__(self, config: TestConfiguration):
        """Initialize the test runner."""
        self.config = config
        self.log_integration = (
            config.log_integration
            if config.log_integration is not None
            else (LogIntegration() if config.log_integration_enabled else None)
        )
        self.test_results: list[TestResult] = []
        self.discovered_tests: dict[str, list[str]] = {}
        self._deadline: float | None = None
        self._command_results: dict[str, Any] = {}
        self._setup_test_environment()

    def _setup_test_environment(self) -> None:
        """Validate runner prerequisites without mutating the checkout."""
        if (
            not math.isfinite(self.config.timeout_seconds)
            or self.config.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.config.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        # Test discovery is intentionally read-only.  Creating a ``tests/``
        # tree here hides missing module fixtures and dirties the caller's
        # working directory before the first test is executed.
        if self.log_integration:
            self.log_integration.logger.info("GeoInferTestRunner initialized")

    def discover_tests(self) -> dict[str, list[str]]:
        """
        Discover all available tests across specified modules.

        Returns:
            Dictionary mapping module names to lists of discovered test functions
        """
        discovered = {}

        for module in self.config.modules_to_test:
            if module not in self.AVAILABLE_MODULES:
                raise ValueError(f"Unknown module: {module}")

            module_tests = self._discover_module_tests(module)
            if module_tests:
                discovered[module] = module_tests
                if self.log_integration:
                    self.log_integration.logger.info(
                        f"Discovered {len(module_tests)} tests for module {module}"
                    )

        self.discovered_tests = discovered
        return discovered

    def _discover_module_tests(self, module: str) -> list[str]:
        """Discover tests for a specific module."""
        tests: list[str] = []

        # Look for module test directory
        module_test_dir = (
            _REPO_ROOT if module == "ROOT" else _REPO_ROOT / f"GEO-INFER-{module}"
        ) / "tests"
        if not module_test_dir.exists():
            return tests

        descriptor = Module(module, module_test_dir.parent, module_test_dir, True)
        seen: set[Path] = set()
        for test_type in self.config.test_types:
            for test_file in category_test_paths(descriptor, test_type):
                if test_file in seen:
                    continue
                seen.add(test_file)
                base = module_test_dir / ("unit" if test_type == "slow" else test_type)
                relative = Path(os.path.relpath(test_file, base)).with_suffix("")
                tests.append(f"{module}::{test_type}::{relative.as_posix()}")

        return tests

    def run_all_tests(self) -> dict[str, Any]:
        """
        Execute all discovered tests with comprehensive logging and reporting.

        Returns:
            Comprehensive test execution report
        """
        if not self.discovered_tests:
            self.discover_tests()

        self.test_results = []
        self._command_results = {}
        start_time = time.monotonic()
        self._deadline = start_time + self.config.timeout_seconds
        if not any(self.discovered_tests.values()):
            raise ValueError("No tests discovered for the requested modules/categories")

        if self.log_integration:
            self.log_integration.logger.info("Starting comprehensive test execution")

        if self.config.parallel_execution:
            self._run_tests_parallel()
        else:
            self._run_tests_sequential()

        total_duration = time.monotonic() - start_time

        # Generate comprehensive report
        report = self._generate_execution_report(total_duration)

        if self.log_integration:
            self.log_integration.logger.info(
                f"Test execution completed in {total_duration:.2f}s"
            )

        return report

    def _run_tests_parallel(self) -> None:
        """Execute tests in parallel using a thread pool.

        Each test runs in its own subprocess (``_run_pytest_test``), so
        concurrent execution is safe. Admission is bounded by ``max_workers``
        and subprocesses share the run deadline. Worker failures are recorded
        as ERROR results instead of being silently dropped from the report.
        """
        reset_process_cancellation()
        executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        selections = iter(
            (module, test)
            for module, tests in self.discovered_tests.items()
            for test in tests
        )
        futures = {}
        abort_requested = {}
        stopped = False

        def collect(future, module, test):
            try:
                result = future.result()
            except BaseException as exc:
                result = TestResult(
                    uuid.uuid4().hex,
                    module,
                    test,
                    "ERROR",
                    0.0,
                    str(exc) or type(exc).__name__,
                    {
                        "error": str(exc) or type(exc).__name__,
                        "exception_type": type(exc).__name__,
                    },
                )
            if result is None:
                result = TestResult(
                    uuid.uuid4().hex,
                    module,
                    test,
                    "ERROR",
                    0.0,
                    "Worker returned no test result",
                    {"error": "Worker returned no test result"},
                )
            if future in abort_requested:
                # Record the stop request without replacing the worker's
                # actual outcome or claiming it caused an unrelated error.
                result.details["abort_requested"] = abort_requested[future]
            self.test_results.append(result)
            return result

        def stop(reason):
            nonlocal stopped
            stopped = True
            for pending in futures:
                if not pending.cancel() and not pending.done():
                    abort_requested[pending] = reason
            terminate_running_processes()

        try:
            while True:
                # Bound admitted work as well as active threads. Refill only
                # after a completion so fail-fast cannot queue the entire suite.
                while len(futures) < self.config.max_workers:
                    selection = next(selections, None)
                    if selection is None:
                        break
                    module, test = selection
                    futures[
                        executor.submit(self._execute_single_test, module, test)
                    ] = (module, test)
                if not futures:
                    break
                future = next(as_completed(futures))
                module, test = futures[future]
                result = collect(future, module, test)
                del futures[future]
                if (
                    self.config.fail_fast
                    and result is not None
                    and result.status != "PASS"
                ):
                    stop("fail_fast")
                    break
        except BaseException as exc:
            if not stopped:
                stop(
                    "interruption"
                    if isinstance(exc, KeyboardInterrupt)
                    else "runner_error"
                )
            raise
        finally:
            cleanup_interruption = None
            while True:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                    # An interrupted thread join can appear complete before
                    # its worker exits. Futures also fence worker-owned state.
                    wait([future for future in futures if not future.cancelled()])
                    break
                except (KeyboardInterrupt, SystemExit) as exc:
                    cleanup_interruption = exc
                    if not stopped:
                        stop("interruption")
            # Retain every admitted worker that actually ran, including
            # receipts and exceptions produced during stop, before reuse.
            for future, (module, test) in futures.items():
                if not future.cancelled():
                    collect(future, module, test)
            reset_process_cancellation()
            if cleanup_interruption is not None:
                raise cleanup_interruption

    def _run_tests_sequential(self) -> None:
        """Execute tests sequentially."""
        for module, tests in self.discovered_tests.items():
            for test in tests:
                try:
                    result = self._execute_single_test(module, test)
                    if result:
                        self.test_results.append(result)

                        # Check fail-fast
                        if self.config.fail_fast and result.status in ["FAIL", "ERROR"]:
                            if self.log_integration:
                                self.log_integration.logger.warning(
                                    "Stopping execution due to fail-fast mode"
                                )
                            return

                except Exception as e:
                    self.test_results.append(
                        TestResult(
                            uuid.uuid4().hex,
                            module,
                            test,
                            "ERROR",
                            0.0,
                            str(e),
                            {"error": str(e)},
                        )
                    )
                    if self.config.fail_fast:
                        return

    def _execute_single_test(self, module: str, test: str) -> TestResult | None:
        """Execute a single test with comprehensive logging."""
        test_id = f"{module}_{test}_{int(time.time())}"

        # Parse test information
        parts = test.split("::")
        if len(parts) != 3:
            raise ValueError(f"Malformed test identifier: {test}")

        module_name, test_type, test_file = parts
        test_name = f"{test_type}_{test_file}"

        start_time = time.time()

        try:
            if self.log_integration:
                with self.log_integration.test_context(test_id, module, test_name):
                    # Execute the actual test
                    result = self._run_pytest_test(module, test_type, test_file)

                    end_time = time.time()
                    duration = end_time - start_time

                    return TestResult(
                        test_id=test_id,
                        module=module,
                        test_name=test_name,
                        status="PASS" if result else "FAIL",
                        duration=duration,
                        message="Test execution completed",
                        details={
                            "test_type": test_type,
                            "test_file": test_file,
                            **self._command_details(module, test_type, test_file),
                        },
                    )
            else:
                # Execute without log integration
                result = self._run_pytest_test(module, test_type, test_file)
                end_time = time.time()
                duration = end_time - start_time

                return TestResult(
                    test_id=test_id,
                    module=module,
                    test_name=test_name,
                    status="PASS" if result else "FAIL",
                    duration=duration,
                    message="Test execution completed",
                    details={
                        "test_type": test_type,
                        "test_file": test_file,
                        **self._command_details(module, test_type, test_file),
                    },
                )

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            return TestResult(
                test_id=test_id,
                module=module,
                test_name=test_name,
                status="ERROR",
                duration=duration,
                message=f"Test execution failed: {str(e)}",
                details={
                    "error": str(e),
                    "test_type": test_type,
                    "test_file": test_file,
                },
            )

    def _command_details(self, module: str, category: str, filename: str) -> dict:
        result = self._command_results.get(f"{module}::{category}::{filename}")
        return (
            {
                "receipt": result.receipt,
                "testcases": result.executed,
                "returncode": result.returncode,
                "command_status": result.status,
            }
            if result
            else {}
        )

    def _run_pytest_test(self, module: str, test_type: str, test_file: str) -> bool:
        """Use the canonical execution engine, including JUnit and custody checks."""
        base_type = "unit" if test_type == "slow" else test_type
        module_root = (
            _REPO_ROOT if module == "ROOT" else _REPO_ROOT / f"GEO-INFER-{module}"
        )
        path = (module_root / "tests" / base_type / f"{test_file}.py").resolve()
        if not path.is_relative_to(module_root.resolve()) or not path.is_file():
            return False
        remaining = (
            self.config.timeout_seconds
            if self._deadline is None
            else self._deadline - time.monotonic()
        )
        if remaining <= 0:
            return False
        markers = (
            ["-m", "slow" if test_type == "slow" else "not slow"]
            if test_type in {"unit", "slow"}
            else []
        )
        result = run_command(
            [
                *pytest_base_args(),
                *markers,
                *profile_selection_args(
                    Module(module, module_root, module_root / "tests", True), test_type
                ),
                str(path),
                f"--junitxml={run_results_dir() / 'programmatic.xml'}",
            ],
            f"{module} {test_type} {test_file}",
            timeout=remaining,
            cwd=_REPO_ROOT,
        )
        self._command_results[f"{module}::{test_type}::{test_file}"] = result
        return result.success

    def _generate_execution_report(self, total_duration: float) -> dict[str, Any]:
        """Generate comprehensive test execution report."""
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == "PASS")
        failed = sum(1 for r in self.test_results if r.status == "FAIL")
        errors = sum(1 for r in self.test_results if r.status == "ERROR")

        module_summaries = {}
        for result in self.test_results:
            module = result.module
            if module not in module_summaries:
                module_summaries[module] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "duration": 0.0,
                }

            summary = module_summaries[module]
            summary["total"] += 1
            summary["duration"] += result.duration

            if result.status == "PASS":
                summary["passed"] += 1
            elif result.status == "FAIL":
                summary["failed"] += 1
            elif result.status == "ERROR":
                summary["errors"] += 1

        report = {
            "execution_summary": {
                "total_tests": total_tests,
                "not_run": max(
                    0, sum(map(len, self.discovered_tests.values())) - total_tests
                ),
                "testcases": sum(
                    r.details.get("testcases", 0) for r in self.test_results
                ),
                "success": bool(self.test_results) and passed == total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration,
            },
            "module_summaries": module_summaries,
            "test_results": [
                {
                    "test_id": r.test_id,
                    "module": r.module,
                    "test_name": r.test_name,
                    "status": r.status,
                    "duration": r.duration,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.test_results
            ],
            "configuration": {
                "modules_tested": self.config.modules_to_test,
                "test_types": self.config.test_types,
                "parallel_execution": self.config.parallel_execution,
                "max_workers": self.config.max_workers,
                "log_integration_enabled": self.config.log_integration_enabled,
            },
        }

        return report

    def run_module_tests(self, module: str) -> dict[str, Any]:
        """Run tests for a specific module only."""
        if module not in self.AVAILABLE_MODULES:
            raise ValueError(f"Unknown module: {module}")

        # Temporarily modify config to test only this module
        original_modules = self.config.modules_to_test
        original_discovered = self.discovered_tests
        self.config.modules_to_test = [module]

        try:
            # Discover and run tests for this module
            self.discovered_tests = {module: self._discover_module_tests(module)}
            report = self.run_all_tests()
            return report
        finally:
            # Restore original configuration
            self.config.modules_to_test = original_modules
            self.discovered_tests = original_discovered

    def run_cross_module_tests(self) -> dict[str, Any]:
        """Run canonical integration discovery without losing nested test roots."""
        original_types = self.config.test_types
        original_modules = self.config.modules_to_test
        try:
            self.config.test_types = ["integration"]
            self.config.modules_to_test = list(self.AVAILABLE_MODULES)
            self.discovered_tests = {}
            return self.run_all_tests()
        finally:
            self.config.test_types = original_types
            self.config.modules_to_test = original_modules
            self.discovered_tests = {}
