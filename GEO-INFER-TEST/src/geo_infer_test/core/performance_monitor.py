"""
GEO-INFER-TEST Performance Monitor.

Track execution time, memory usage, and throughput for test runs
and custom benchmarks.
"""

import json
import logging
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Callable


@dataclass
class _TimingRecord:
    """Internal timing snapshot."""

    label: str
    start_time: float
    end_time: float = 0.0
    peak_memory_bytes: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class PerformanceMonitor:
    """
    Tracks execution time and (optionally) peak memory for labelled
    code sections via a context-manager interface.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._records: list[_TimingRecord] = []
        self._section_stack: list[_TimingRecord] = []
        self._trace_depth: int = 0

    def start(self, label: str) -> None:
        """Begin timing a section.

        Nested sections are supported: ``tracemalloc`` is reference-counted,
        so an inner section's ``stop()`` does not kill tracing for the outer
        section. Overlapping sections each get their own record.
        """
        if self._trace_depth == 0:
            tracemalloc.start()
        self._trace_depth += 1
        self._section_stack.append(
            _TimingRecord(label=label, start_time=time.perf_counter())
        )
        self.logger.debug("⏱ Start: %s", label)

    def stop(self) -> dict[str, Any]:
        """Stop timing the active section and return metrics."""
        if not self._section_stack:
            raise RuntimeError("No active timing section")

        rec = self._section_stack.pop()
        rec.end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        rec.peak_memory_bytes = peak
        self._trace_depth = max(self._trace_depth - 1, 0)
        if self._trace_depth == 0:
            tracemalloc.stop()

        self._records.append(rec)

        metrics = {
            "label": rec.label,
            "duration_s": rec.duration,
            "peak_memory_bytes": rec.peak_memory_bytes,
        }
        self.logger.info(
            "⏱ Stop: %s – %.4fs, %.1f KB",
            rec.label,
            rec.duration,
            rec.peak_memory_bytes / 1024,
        )
        return metrics

    def get_all_records(self) -> list[dict[str, Any]]:
        return [
            {
                "label": r.label,
                "duration_s": r.duration,
                "peak_memory_bytes": r.peak_memory_bytes,
            }
            for r in self._records
        ]

    def reset(self) -> None:
        self._records.clear()
        self._section_stack.clear()
        self._trace_depth = 0


class BenchmarkRunner:
    """
    Run a callable *iterations* times (with optional warmup) and
    report min / max / mean / median / stdev timing.
    """

    def __init__(
        self,
        iterations: int = 10,
        warmup: int = 2,
        logger: logging.Logger | None = None,
    ):
        self.iterations = iterations
        self.warmup = warmup
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        func: Callable[..., Any],
        *args: Any,
        label: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute *func* and return timing statistics."""
        label = label or getattr(func, "__name__", "benchmark")

        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)

        durations: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            durations.append(time.perf_counter() - start)

        result = {
            "label": label,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "min_s": min(durations),
            "max_s": max(durations),
            "mean_s": statistics.mean(durations),
            "median_s": statistics.median(durations),
            "stdev_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "total_s": sum(durations),
        }
        self.logger.info(
            "Benchmark '%s': mean=%.4fs, stdev=%.4fs (%d iters)",
            label,
            result["mean_s"],
            result["stdev_s"],
            self.iterations,
        )
        return result


class LoadTester:
    """
    Tests concurrent load by running a callable across multiple threads
    and measuring aggregate throughput and error rate.
    """

    def __init__(
        self,
        concurrency: int = 4,
        total_requests: int = 20,
        logger: logging.Logger | None = None,
    ):
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        func: Callable[..., Any],
        *args: Any,
        label: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute load test and return throughput stats."""
        label = label or getattr(func, "__name__", "load_test")
        durations: list[float] = []
        errors: list[str] = []

        overall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = []
            for _ in range(self.total_requests):
                futures.append(pool.submit(self._timed_call, func, args, kwargs))

            for future in as_completed(futures):
                dur, err = future.result()
                durations.append(dur)
                if err:
                    errors.append(err)

        total_time = time.perf_counter() - overall_start

        result = {
            "label": label,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "completed": len(durations),
            "errors": len(errors),
            "error_rate": (
                len(errors) / self.total_requests if self.total_requests else 0.0
            ),
            "total_duration_s": total_time,
            "throughput_rps": self.total_requests / total_time if total_time else 0.0,
            "avg_latency_s": statistics.mean(durations) if durations else 0.0,
            "p95_latency_s": self._percentile(durations, 0.95),
        }
        self.logger.info(
            "Load test '%s': %d requests, %.1f rps, %.1f%% errors",
            label,
            self.total_requests,
            result["throughput_rps"],
            result["error_rate"] * 100,
        )
        return result

    @staticmethod
    def _timed_call(func: Callable, args: tuple, kwargs: dict) -> tuple:
        start = time.perf_counter()
        error: str | None = None
        try:
            func(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
        return time.perf_counter() - start, error

    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct)
        return sorted_data[min(idx, len(sorted_data) - 1)]


class MetricsCollector:
    """
    Collects, aggregates, and persists performance metrics across
    multiple benchmark and load-test runs.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._entries: list[dict[str, Any]] = []

    def add(self, metrics: dict[str, Any]) -> None:
        """Record a metrics snapshot with a timestamp."""
        self._entries.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                **metrics,
            }
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate summary across all collected entries."""
        if not self._entries:
            return {"total_entries": 0}

        durations = [e.get("duration_s", e.get("mean_s", 0.0)) for e in self._entries]
        return {
            "total_entries": len(self._entries),
            "duration_min": min(durations) if durations else 0.0,
            "duration_max": max(durations) if durations else 0.0,
            "duration_mean": statistics.mean(durations) if durations else 0.0,
        }

    def save(self, path: Path) -> None:
        """Persist all collected metrics as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._entries, indent=2, default=str))
        self.logger.info("Metrics saved to %s (%d entries)", path, len(self._entries))

    def reset(self) -> None:
        self._entries.clear()


class PerformanceAnalyzer:
    """
    Analyses historical performance data to detect trends and
    regressions compared to a baseline.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        logger: logging.Logger | None = None,
    ):
        self.collector = collector
        self.logger = logger or logging.getLogger(__name__)

    def detect_regression(
        self, baseline: dict[str, float], threshold: float = 1.5
    ) -> dict[str, Any]:
        """
        Compare the latest collected metrics against a *baseline*.
        Flag any metric whose value exceeds *threshold* × baseline.
        """
        if not self.collector._entries:
            return {"regressions": [], "status": "no_data"}

        latest = self.collector._entries[-1]
        regressions: list[dict[str, Any]] = []

        for key, base_val in baseline.items():
            current_val = latest.get(key)
            if current_val is None or base_val == 0:
                continue
            ratio = current_val / base_val
            if ratio >= threshold:
                regressions.append(
                    {
                        "metric": key,
                        "baseline": base_val,
                        "current": current_val,
                        "ratio": round(ratio, 3),
                    }
                )

        status = "regression" if regressions else "ok"
        self.logger.info("Regression check: %s (%d issues)", status, len(regressions))
        return {"regressions": regressions, "status": status}

    def trend_report(self) -> dict[str, Any]:
        """Return a simple trend report from the collected data."""
        entries = self.collector._entries
        if len(entries) < 2:
            return {"trend": "insufficient_data", "count": len(entries)}

        durations = [e.get("duration_s", e.get("mean_s", 0.0)) for e in entries]
        first_half = durations[: len(durations) // 2]
        second_half = durations[len(durations) // 2 :]

        first_mean = statistics.mean(first_half) if first_half else 0.0
        second_mean = statistics.mean(second_half) if second_half else 0.0

        if first_mean == 0:
            trend = "unknown"
        elif second_mean > first_mean * 1.1:
            trend = "degrading"
        elif second_mean < first_mean * 0.9:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "total_samples": len(entries),
        }
