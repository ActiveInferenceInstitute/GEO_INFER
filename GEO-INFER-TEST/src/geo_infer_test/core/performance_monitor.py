"""
GEO-INFER-TEST Performance Monitor.

Track execution time, memory usage, and throughput for test runs
and custom benchmarks.
"""

import asyncio
import inspect
import json
import math
import threading
from contextlib import contextmanager
import logging
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Callable


_TRACE_LOCK = threading.RLock()
_TRACE_USERS = 0
_TRACE_OWNED = False


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_sync_callable(func: Callable[..., Any]) -> None:
    if inspect.iscoroutinefunction(func) or (
        callable(func) and inspect.iscoroutinefunction(type(func).__call__)
    ):
        raise TypeError("Async work requires run_async()")


def _invoke_sync(func: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        # A wrapper can return a coroutine without being a coroutine function.
        # Close only that unstarted native coroutine; other awaitables retain
        # their caller's ownership and are never cancelled implicitly.
        if (
            inspect.iscoroutine(result)
            and inspect.getcoroutinestate(result) == inspect.CORO_CREATED
        ):
            result.close()
        raise TypeError("Async work requires run_async()")


def _raise_new_cancellation(task: asyncio.Task | None, baseline: int) -> int:
    """Retain a run's cancellation even if measured work swallows its delivery."""
    observed = task.cancelling() if task is not None else baseline
    if observed > baseline:
        raise asyncio.CancelledError()
    # Work may explicitly clear an older cancellation. Remember that observed
    # decrease so a later, distinct cancellation cannot hide below entry count.
    return observed


def _acquire_trace() -> None:
    global _TRACE_USERS, _TRACE_OWNED
    with _TRACE_LOCK:
        if _TRACE_USERS == 0:
            _TRACE_OWNED = not tracemalloc.is_tracing()
            if _TRACE_OWNED:
                tracemalloc.start()
        _TRACE_USERS += 1


def _release_trace() -> None:
    global _TRACE_USERS, _TRACE_OWNED
    with _TRACE_LOCK:
        _TRACE_USERS -= 1
        if _TRACE_USERS == 0:
            if _TRACE_OWNED:
                tracemalloc.stop()
            _TRACE_OWNED = False


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
        _acquire_trace()
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
        _release_trace()

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

    @contextmanager
    def section(self, label: str):
        """Balance timing and tracer ownership even when the body raises."""
        self.start(label)
        try:
            yield self
        finally:
            self.stop()

    def reset(self) -> None:
        if self._section_stack:
            raise RuntimeError("Stop active timing sections before reset")
        self._records.clear()
        self._section_stack.clear()
        self._trace_depth = 0


class BenchmarkRunner:
    """
    Run a callable *iterations* times (with optional warmup) and
    report min / max / mean / median / stdev timing. Use ``run_async``
    when the callable returns awaitable work.
    """

    def __init__(
        self,
        iterations: int = 10,
        warmup: int = 2,
        logger: logging.Logger | None = None,
    ):
        _positive_integer(iterations, "iterations")
        _positive_integer(warmup, "warmup", minimum=0)
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
        """Time synchronous calls; awaitable work requires ``run_async``."""
        _require_sync_callable(func)
        label = label or getattr(func, "__name__", "benchmark")

        # Warmup
        for _ in range(self.warmup):
            _invoke_sync(func, args, kwargs)

        durations: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            _invoke_sync(func, args, kwargs)
            durations.append(time.perf_counter() - start)

        return self._summarize(durations, label)

    async def run_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        label: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Time sequential awaited calls, including warmup and completion.

        Synchronous return values are accepted as completed work. Exceptions
        and cancellation propagate; no partial benchmark is reported as success.
        New outstanding cancellation stops measurement even if the callable
        catches its delivery. Explicit ``uncancel`` retains native semantics.
        """
        label = label or getattr(func, "__name__", "benchmark")
        owner = asyncio.current_task()
        cancellation_baseline = owner.cancelling() if owner is not None else 0
        durations: list[float] = []
        for index in range(self.warmup + self.iterations):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            finally:
                cancellation_baseline = _raise_new_cancellation(
                    owner, cancellation_baseline
                )
            if index >= self.warmup:
                durations.append(time.perf_counter() - start)
        return self._summarize(durations, label)

    def _summarize(self, durations: list[float], label: str) -> dict[str, Any]:
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
            "p95_s": LoadTester._percentile(durations, 0.95),
            "p99_s": LoadTester._percentile(durations, 0.99),
            "operations_per_second": self.iterations / sum(durations)
            if sum(durations)
            else 0.0,
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
    Measure concurrent throughput and error rate using threads for ``run``
    or bounded native async workers for ``run_async``.
    """

    def __init__(
        self,
        concurrency: int = 4,
        total_requests: int = 20,
        logger: logging.Logger | None = None,
    ):
        _positive_integer(concurrency, "concurrency")
        _positive_integer(total_requests, "total_requests")
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
        """Run synchronous threaded load; awaitables require ``run_async``."""
        _require_sync_callable(func)
        label = label or getattr(func, "__name__", "load_test")
        durations: list[float] = []
        errors: list[str] = []

        overall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            submitted = min(self.concurrency, self.total_requests)
            pending = {
                pool.submit(self._timed_call, func, args, kwargs)
                for _ in range(submitted)
            }
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    dur, err = future.result()
                    durations.append(dur)
                    if err is not None:
                        errors.append(err)
                    if submitted < self.total_requests:
                        pending.add(pool.submit(self._timed_call, func, args, kwargs))
                        submitted += 1

        total_time = time.perf_counter() - overall_start

        return self._summarize(durations, errors, total_time, label)

    async def run_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        label: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Await requests with at most ``concurrency`` owned worker tasks.

        Per-request exceptions count as errors. Cancelling the run cancels
        and awaits every worker before propagating cancellation to its caller.
        Swallowed delivery cannot start more requests while new cancellation
        remains outstanding on either the worker or the overall run task.
        """
        label = label or getattr(func, "__name__", "load_test")
        owner = asyncio.current_task()
        owner_cancellation_baseline = owner.cancelling() if owner is not None else 0
        durations: list[float] = []
        errors: list[str] = []
        requests = iter(range(self.total_requests))

        async def worker() -> None:
            nonlocal owner_cancellation_baseline
            task = asyncio.current_task()
            cancellation_baseline = task.cancelling() if task is not None else 0
            for _ in requests:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    errors.append(str(exc))
                cancellation_baseline = _raise_new_cancellation(
                    task, cancellation_baseline
                )
                owner_cancellation_baseline = _raise_new_cancellation(
                    owner, owner_cancellation_baseline
                )
                durations.append(time.perf_counter() - start)

        overall_start = time.perf_counter()
        async with asyncio.TaskGroup() as group:
            workers = []
            for _ in range(min(self.concurrency, self.total_requests)):
                workers.append(group.create_task(worker()))
            # Awaiting the workers also propagates a request's own
            # CancelledError; TaskGroup alone treats a cancelled child as a
            # normal exit and could otherwise report an incomplete run.
            await asyncio.gather(*workers)
        return self._summarize(
            durations, errors, time.perf_counter() - overall_start, label
        )

    def _summarize(
        self,
        durations: list[float],
        errors: list[str],
        total_time: float,
        label: str,
    ) -> dict[str, Any]:
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
            _invoke_sync(func, args, kwargs)
        except Exception as exc:
            error = str(exc)
        return time.perf_counter() - start, error

    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        if not math.isfinite(pct) or not 0 <= pct <= 1:
            raise ValueError("Percentile must be finite and between 0 and 1")
        if not data:
            return 0.0
        sorted_data = sorted(data)
        return sorted_data[max(0, math.ceil(len(sorted_data) * pct) - 1)]


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
