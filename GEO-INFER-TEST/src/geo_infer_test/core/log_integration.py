"""
GEO-INFER-TEST Log Integration Module.

Provides comprehensive logging, reporting, and analysis
for test execution across all GEO-INFER modules.
"""

import json
import logging
import statistics
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Iterator


# Check if GEO-INFER-LOG is available
try:
    import geo_infer_log  # noqa: F401 -- availability probe

    LOG_MODULE_AVAILABLE = True
except ImportError:
    LOG_MODULE_AVAILABLE = False


@dataclass
class TestLogEntry:
    """A single test execution log entry."""

    timestamp: datetime
    test_id: str
    module: str
    test_name: str
    status: str  # PASS, FAIL, ERROR, SKIP
    duration: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    error_info: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None


@dataclass
class ModuleTestSummary:
    """Aggregated test summary for a single module."""

    module_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float


class LogIntegration:
    """
    Context-managed test execution logger.

    Tracks individual test results and aggregates per-module summaries.
    Works with or without the GEO-INFER-LOG module.
    """

    def __init__(self, log_config: dict[str, Any] | None = None):
        self.log_config = log_config or {}
        self.test_entries: list[TestLogEntry] = []
        self.module_summaries: dict[str, ModuleTestSummary] = {}
        self.log_available: bool = LOG_MODULE_AVAILABLE

        # Set up a standard Python logger
        self.logger = logging.getLogger("geo_infer_test.log_integration")
        level_name = self.log_config.get("level", "INFO")
        self.logger.setLevel(getattr(logging, level_name, logging.INFO))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = self.log_config.get(
                "format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Context manager for individual tests
    # ------------------------------------------------------------------

    @contextmanager
    def test_context(
        self, test_id: str, module: str, test_name: str
    ) -> Iterator[TestLogEntry]:
        """
        Context manager that records a test's outcome and duration.

        Exceptions propagate to the caller after being logged.
        ``AssertionError`` → status FAIL; other exceptions → status ERROR.
        """
        start = time.time()
        entry = TestLogEntry(
            timestamp=datetime.now(UTC),
            test_id=test_id,
            module=module,
            test_name=test_name,
            status="PASS",
            duration=0.0,
            message="",
            details={},
        )

        try:
            yield entry
            entry.status = "PASS"
            entry.message = "Test passed"
            self.logger.info(
                "PASS %s::%s::%s (%.3fs)",
                module,
                test_name,
                test_id,
                time.time() - start,
            )
        except (AssertionError, ValueError) as exc:
            entry.status = "FAIL"
            entry.message = str(exc)
            entry.error_info = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            self.logger.warning(
                "FAIL %s::%s::%s – %s",
                module,
                test_name,
                test_id,
                exc,
            )
            raise
        except Exception as exc:
            entry.status = "ERROR"
            entry.message = str(exc)
            entry.error_info = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            self.logger.error(
                "ERROR %s::%s::%s – %s",
                module,
                test_name,
                test_id,
                exc,
            )
            raise
        finally:
            entry.duration = time.time() - start
            self.test_entries.append(entry)
            self._update_module_summary(entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_module_summary(self, entry: TestLogEntry) -> None:
        """Merge a single test entry into the per-module summary."""
        if entry.module not in self.module_summaries:
            self.module_summaries[entry.module] = ModuleTestSummary(
                module_name=entry.module,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=0,
                total_duration=0.0,
            )

        summary = self.module_summaries[entry.module]
        summary.total_tests += 1
        summary.total_duration += entry.duration

        if entry.status == "PASS":
            summary.passed += 1
        elif entry.status == "FAIL":
            summary.failed += 1
        elif entry.status == "SKIP":
            summary.skipped += 1
        elif entry.status == "ERROR":
            summary.errors += 1


class LoggingTestReporter:
    """
    Generates JSON and Markdown reports from a LogIntegration session.
    """

    def __init__(self, log_integration: LogIntegration):
        self.log_integration = log_integration
        self.logger = log_integration.logger

    def generate_test_report(self, output_dir: Path | None = None) -> dict[str, Any]:
        """
        Build a comprehensive report dict and, when *output_dir* is given,
        persist it as ``test_report_<timestamp>.json``.
        """
        entries = self.log_integration.test_entries
        total = len(entries)
        passed = sum(1 for e in entries if e.status == "PASS")
        failed = sum(1 for e in entries if e.status in ("FAIL", "ERROR"))

        report_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "skipped": sum(1 for e in entries if e.status == "SKIP"),
                "errors": sum(1 for e in entries if e.status == "ERROR"),
                "success_rate": (passed / total * 100) if total else 0.0,
                "total_duration": sum(e.duration for e in entries),
            },
            "module_summaries": {
                name: {
                    "module_name": s.module_name,
                    "total_tests": s.total_tests,
                    "passed": s.passed,
                    "failed": s.failed,
                    "skipped": s.skipped,
                    "errors": s.errors,
                    "total_duration": s.total_duration,
                }
                for name, s in self.log_integration.module_summaries.items()
            },
            "test_entries": [
                {
                    "test_id": e.test_id,
                    "module": e.module,
                    "test_name": e.test_name,
                    "status": e.status,
                    "duration": e.duration,
                    "message": e.message,
                }
                for e in entries
            ],
        }

        # Write to disk
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"test_report_{ts}.json"
            json_path.write_text(json.dumps(report_data, indent=2, default=str))

            self.logger.info("Report written to %s", json_path)

        return report_data


class _TestLoggerImpl:
    """
    Utility for logging ancillary test artefacts: performance metrics,
    module health, and cross-module interactions.

    NOTE: Named _TestLoggerImpl internally to avoid pytest collection
    warnings; exported as ``TestLogger`` at module level.
    """

    def __init__(self, log_integration: LogIntegration):
        self.log_integration = log_integration
        self.logger = log_integration.logger
        self._health_log: dict[str, dict[str, Any]] = {}
        self._interactions: list[dict[str, Any]] = []

    def log_performance_metrics(self, test_id: str, metrics: dict[str, Any]) -> None:
        """Attach performance metrics to an existing test entry."""
        for entry in self.log_integration.test_entries:
            if entry.test_id == test_id:
                entry.performance_metrics = metrics
                self.logger.info(
                    "Performance metrics attached to %s: %s", test_id, metrics
                )
                return
        self.logger.warning("Test entry %s not found for metrics", test_id)

    def log_module_health(self, module: str, health_data: dict[str, Any]) -> None:
        """Record module health snapshot."""
        self._health_log[module] = {
            "timestamp": datetime.now(UTC).isoformat(),
            **health_data,
        }
        self.logger.info(
            "Module %s health: %s", module, health_data.get("status", "unknown")
        )

    def log_cross_module_interaction(
        self,
        source_module: str,
        target_module: str,
        interaction_type: str,
        result: str,
    ) -> None:
        """Record a cross-module interaction event."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source": source_module,
            "target": target_module,
            "type": interaction_type,
            "result": result,
        }
        self._interactions.append(record)
        self.logger.info(
            "Cross-module %s→%s (%s): %s",
            source_module,
            target_module,
            interaction_type,
            result,
        )


class LogAnalyzer:
    """
    Analyses completed test execution data to surface patterns,
    reliability metrics, and performance bottlenecks.
    """

    def __init__(self, log_integration: LogIntegration):
        self.log_integration = log_integration
        self.logger = log_integration.logger

    def analyze_test_patterns(self) -> dict[str, Any]:
        """
        Compute per-module reliability (success rate) and overall stats.
        """
        entries = self.log_integration.test_entries

        module_groups: dict[str, list[TestLogEntry]] = {}
        for entry in entries:
            module_groups.setdefault(entry.module, []).append(entry)

        module_reliability: dict[str, dict[str, Any]] = {}
        for module, mod_entries in module_groups.items():
            total = len(mod_entries)
            passed = sum(1 for e in mod_entries if e.status == "PASS")
            module_reliability[module] = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": (passed / total * 100) if total else 0.0,
                "avg_duration": statistics.mean(e.duration for e in mod_entries),
            }

        return {
            "total_tests_analyzed": len(entries),
            "module_reliability": module_reliability,
            "overall_success_rate": (
                sum(1 for e in entries if e.status == "PASS") / len(entries) * 100
                if entries
                else 0.0
            ),
        }

    def identify_performance_bottlenecks(
        self, *, threshold_factor: float = 2.0
    ) -> list[dict[str, Any]]:
        """
        Find tests whose duration is *threshold_factor* × the mean,
        sorted by slowness factor descending.
        """
        entries = self.log_integration.test_entries
        if not entries:
            return []

        durations = [e.duration for e in entries]
        mean_dur = statistics.mean(durations)

        if mean_dur == 0:
            return []

        bottlenecks: list[dict[str, Any]] = []
        for entry in entries:
            factor = entry.duration / mean_dur
            if factor >= threshold_factor:
                bottlenecks.append(
                    {
                        "test_id": entry.test_id,
                        "test_name": entry.test_name,
                        "module": entry.module,
                        "duration": entry.duration,
                        "slowness_factor": factor,
                        "mean_duration": mean_dur,
                    }
                )

        bottlenecks.sort(key=lambda b: b["slowness_factor"], reverse=True)
        return bottlenecks


# Public alias kept so that ``from .log_integration import TestLogger`` works
# without triggering pytest collection warnings.
TestLogger = _TestLoggerImpl
