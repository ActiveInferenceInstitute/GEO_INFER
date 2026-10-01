"""
Main test runner for the GEO-INFER-TEST framework.

This module provides the core test execution engine that can run tests
across all GEO-INFER modules with comprehensive logging and reporting.
"""

import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .log_integration import LogIntegration
from .test_discoverer import ALL_MODULES

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
    AVAILABLE_MODULES = tuple(ALL_MODULES)

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
        self._setup_test_environment()

    def _setup_test_environment(self) -> None:
        """Validate runner prerequisites without mutating the checkout."""
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
                if self.log_integration:
                    self.log_integration.logger.warning(f"Unknown module: {module}")
                continue

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
        module_test_dir = _REPO_ROOT / f"GEO-INFER-{module}/tests"
        if not module_test_dir.exists():
            return tests

        # Discover test files
        for test_type in self.config.test_types:
            test_type_dir = module_test_dir / test_type
            if test_type_dir.exists():
                test_files = sorted(
                    {
                        *test_type_dir.rglob("test_*.py"),
                        *test_type_dir.rglob("*_test.py"),
                    }
                )
                for test_file in test_files:
                    relative_path = test_file.relative_to(test_type_dir).with_suffix("")
                    tests.append(f"{module}::{test_type}::{relative_path.as_posix()}")

        return tests

    def run_all_tests(self) -> dict[str, Any]:
        """
        Execute all discovered tests with comprehensive logging and reporting.

        Returns:
            Comprehensive test execution report
        """
        if not self.discovered_tests:
            self.discover_tests()

        start_time = time.time()

        if self.log_integration:
            self.log_integration.logger.info("Starting comprehensive test execution")

        if self.config.parallel_execution:
            self._run_tests_parallel()
        else:
            self._run_tests_sequential()

        end_time = time.time()
        total_duration = end_time - start_time

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
        concurrent execution is safe. ``as_completed`` applies a global wall
        clock deadline instead of recharging the full per-test timeout for
        every future, and worker failures are recorded as ERROR results
        instead of being silently dropped from the report.
        """
        futures = {
            executor.submit(self._execute_single_test, module, test): (
                module,
                test,
            )
            for executor in [ThreadPoolExecutor(max_workers=self.config.max_workers)]
            for module, tests in self.discovered_tests.items()
            for test in tests
        }

        try:
            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                module, test = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.exception("Test execution error for %s/%s", module, test)
                    result = TestResult(
                        test_id=f"{module}_{test}_{int(time.time())}",
                        module=module,
                        test_name=test,
                        status="ERROR",
                        duration=0.0,
                        message=f"Test execution failed: {e}",
                        details={"error": str(e)},
                    )
                if result:
                    self.test_results.append(result)
        except TimeoutError:
            logger.error(
                "Parallel test run exceeded the global timeout of %ss; "
                "remaining tests are recorded as ERROR",
                self.config.timeout_seconds,
            )
            for future in futures:
                if not future.done():
                    module, test = futures[future]
                    self.test_results.append(
                        TestResult(
                            test_id=f"{module}_{test}_{int(time.time())}",
                            module=module,
                            test_name=test,
                            status="ERROR",
                            duration=0.0,
                            message="Test execution timed out",
                            details={"timeout_seconds": self.config.timeout_seconds},
                        )
                    )
                future.cancel()

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
                    if self.log_integration:
                        self.log_integration.logger.error(f"Test execution error: {e}")

    def _execute_single_test(self, module: str, test: str) -> TestResult | None:
        """Execute a single test with comprehensive logging."""
        test_id = f"{module}_{test}_{int(time.time())}"

        # Parse test information
        parts = test.split("::")
        if len(parts) != 3:
            return None

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
                        details={"test_type": test_type, "test_file": test_file},
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
                    details={"test_type": test_type, "test_file": test_file},
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

    def _run_pytest_test(self, module: str, test_type: str, test_file: str) -> bool:
        """Execute a pytest test file in an isolated subprocess.

        ``pytest.main`` mutates global plugin/config state and is not safe to
        call concurrently (or repeatedly) in-process, so execution shells out
        to ``sys.executable -m pytest`` — the same model run_unified_tests.py
        uses per module.
        """
        test_path = _REPO_ROOT / f"GEO-INFER-{module}/tests/{test_type}/{test_file}.py"

        if not test_path.exists():
            return False

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                cwd=_REPO_ROOT,
                timeout=self.config.timeout_seconds,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            if self.log_integration:
                self.log_integration.logger.warning(
                    f"Test timed out after {self.config.timeout_seconds}s: {test_path}"
                )
            return False

        return proc.returncode == 0

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
        self.config.modules_to_test = [module]

        try:
            # Discover and run tests for this module
            self.discovered_tests = {module: self._discover_module_tests(module)}
            report = self.run_all_tests()
            return report
        finally:
            # Restore original configuration
            self.config.modules_to_test = original_modules

    def run_cross_module_tests(self) -> dict[str, Any]:
        """Run integration tests that verify cross-module interactions.

        Discovers and executes every test file located in a
        ``tests/integration/`` directory across all available GEO-INFER modules,
        then returns a consolidated report. Each file runs in an isolated
        pytest subprocess anchored to the repository root (not the CWD).
        """
        if self.log_integration:
            self.log_integration.logger.info("Starting cross-module integration tests")

        start_time = time.time()

        cross_results: dict[str, Any] = {}
        total_tests = 0
        total_passed = 0
        total_failed = 0

        for module in self.AVAILABLE_MODULES:
            integration_dir = _REPO_ROOT / f"GEO-INFER-{module}/tests/integration"
            if not integration_dir.exists():
                continue

            test_files = sorted(
                {
                    *integration_dir.glob("test_*.py"),
                    *integration_dir.glob("*_test.py"),
                }
            )
            if not test_files:
                continue

            module_results = []
            for test_file in test_files:
                t0 = time.time()
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            str(test_file),
                            "-q",
                            "--no-header",
                            "--disable-warnings",
                        ],
                        cwd=_REPO_ROOT,
                        timeout=self.config.timeout_seconds,
                        capture_output=True,
                    )
                    exit_code = proc.returncode
                except subprocess.TimeoutExpired:
                    exit_code = 1
                elapsed = time.time() - t0
                passed = exit_code == 0
                module_results.append(
                    {
                        "test_file": str(test_file),
                        "passed": passed,
                        "duration_s": round(elapsed, 3),
                    }
                )
                total_tests += 1
                if passed:
                    total_passed += 1
                else:
                    total_failed += 1

            cross_results[module] = module_results

        elapsed_total = time.time() - start_time
        if self.log_integration:
            self.log_integration.logger.info(
                f"Cross-module integration tests completed: {total_passed}/{total_tests} passed "
                f"in {elapsed_total:.2f}s"
            )

        return {
            "cross_module_tests": cross_results,
            "integration_status": "completed",
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": (
                    (total_passed / total_tests * 100) if total_tests else 0.0
                ),
                "duration_s": round(elapsed_total, 3),
            },
        }
