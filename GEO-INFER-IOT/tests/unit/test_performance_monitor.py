"""Regression and behavioral tests for IoT performance monitoring.

Covers summary aggregation, threshold detection, empty-history handling,
metric history windowing, and the metrics dataclass.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

from geo_infer_iot.performance import PerformanceMetrics, PerformanceMonitor


def _monitor_with_history(
    metrics: list[PerformanceMetrics], config: dict | None = None
) -> PerformanceMonitor:
    monitor = PerformanceMonitor(config or {"throughput_threshold": 0.0})
    monitor.metrics_history.extend(metrics)
    return monitor


def _sample(
    *,
    cpu: float = 20.0,
    memory: float = 50.0,
    latency: float = 5.0,
    error_rate: float = 0.0,
    throughput: float = 0.0,
    timestamp: datetime | None = None,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        timestamp=timestamp or datetime.now(UTC),
        cpu_percent=cpu,
        memory_percent=memory,
        processing_latency_ms=latency,
        error_rate=error_rate,
        measurements_per_second=throughput,
    )


class TestPerformanceSummary:
    """Tests for get_performance_summary."""

    def test_summary_aggregates_metrics(self) -> None:
        monitor = _monitor_with_history([_sample(cpu=20.0), _sample(cpu=40.0)])
        summary = monitor.get_performance_summary(minutes=5)
        assert summary["total_samples"] == 2
        assert summary["cpu_usage"]["mean"] == 30.0
        assert summary["cpu_usage"]["min"] == 20.0
        assert summary["cpu_usage"]["max"] == 40.0

    def test_summary_memory_stats(self) -> None:
        monitor = _monitor_with_history(
            [_sample(memory=30.0), _sample(memory=50.0), _sample(memory=70.0)]
        )
        summary = monitor.get_performance_summary()
        assert summary["memory_usage"]["mean"] == 50.0
        assert summary["memory_usage"]["max"] == 70.0

    def test_summary_empty_history_returns_error(self) -> None:
        monitor = PerformanceMonitor({})
        summary = monitor.get_performance_summary()
        assert "error" in summary

    def test_summary_respects_time_window(self) -> None:
        now = datetime.now(UTC)
        old = _sample(timestamp=now - timedelta(minutes=30))
        fresh = _sample(timestamp=now)
        monitor = _monitor_with_history([old, fresh])
        summary = monitor.get_performance_summary(minutes=5)
        assert summary["total_samples"] == 1


class TestThresholdExceedances:
    """Tests for threshold detection."""

    def test_cpu_threshold_exceedance_counted(self) -> None:
        monitor = _monitor_with_history(
            [_sample(cpu=90.0), _sample(cpu=10.0)],
            config={"cpu_threshold": 80.0},
        )
        summary = monitor.get_performance_summary()
        assert summary["threshold_exceedances"]["cpu_percent"] == 1

    def test_error_rate_threshold_exceedance_counted(self) -> None:
        monitor = _monitor_with_history(
            [_sample(error_rate=0.5)],
            config={"error_rate_threshold": 0.05},
        )
        summary = monitor.get_performance_summary()
        assert summary["threshold_exceedances"]["error_rate"] == 1

    def test_latency_threshold_exceedance_counted(self) -> None:
        monitor = _monitor_with_history(
            [_sample(latency=2000.0)],
            config={"latency_threshold": 1000.0},
        )
        summary = monitor.get_performance_summary()
        assert summary["threshold_exceedances"]["latency_ms"] == 1


class TestCurrentMetrics:
    """Tests for get_current_metrics and get_metrics_history."""

    def test_get_current_metrics_returns_latest(self) -> None:
        monitor = _monitor_with_history([_sample(cpu=10.0), _sample(cpu=60.0)])
        current = monitor.get_current_metrics()
        assert current is not None
        assert current.cpu_percent == 60.0

    def test_get_current_metrics_empty(self) -> None:
        monitor = PerformanceMonitor({})
        assert monitor.get_current_metrics() is None

    def test_get_metrics_history_window(self) -> None:
        now = datetime.now(UTC)
        monitor = _monitor_with_history(
            [_sample(timestamp=now - timedelta(minutes=10)), _sample(timestamp=now)]
        )
        history = monitor.get_metrics_history(minutes=1)
        assert len(history) == 1


class TestMetricsDataclass:
    """Tests for the PerformanceMetrics dataclass."""

    def test_defaults(self) -> None:
        metrics = PerformanceMetrics()
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.queue_size == 0
        assert metrics.thread_count == 0

    def test_full_construction(self) -> None:
        metrics = PerformanceMetrics(
            cpu_percent=42.5,
            memory_percent=63.2,
            memory_mb=1024.0,
            measurements_per_second=7.5,
            processing_latency_ms=12.3,
            error_rate=0.01,
            queue_size=4,
        )
        assert metrics.cpu_percent == 42.5
        assert metrics.measurements_per_second == 7.5
        assert metrics.error_rate == 0.01


class TestBenchmark:
    """Tests for run_benchmark error handling."""

    def test_unknown_benchmark_returns_failure(self) -> None:
        monitor = PerformanceMonitor({})
        result = monitor.run_benchmark("not_a_real_benchmark")
        assert result.success is False
        assert "Unknown benchmark type" in (result.error_message or "")

    def test_benchmark_recorded_in_history(self) -> None:
        monitor = PerformanceMonitor({})
        monitor.run_benchmark("not_a_real_benchmark")
        assert len(monitor.benchmark_history) == 1

    def test_ingestion_benchmark_without_system_returns_error(self) -> None:
        monitor = PerformanceMonitor({})
        result = monitor.run_benchmark("ingestion_throughput")
        assert result.success is True
        assert "error" in result.metrics
