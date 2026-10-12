"""
Performance Monitoring Module

This module provides comprehensive performance monitoring, benchmarking,
and profiling capabilities for the GEO-INFER-IOT system.
"""

import logging
import time
from typing import Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from collections import deque
import json
import numpy as np
import threading

try:
    import psutil
except ImportError:
    psutil = None

from geo_infer_iot.models.timestamps import utc_now
from geo_infer_time.core.timestamps import normalize_timestamp

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: datetime = field(default_factory=utc_now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_io_sent: float = 0.0
    network_io_recv: float = 0.0
    thread_count: int = 0
    open_files: int = 0

    # IoT-specific metrics
    measurements_per_second: float = 0.0
    processing_latency_ms: float = 0.0
    error_rate: float = 0.0
    queue_size: int = 0

    def __post_init__(self) -> None:
        self.timestamp = normalize_timestamp(self.timestamp)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""

    benchmark_id: str
    benchmark_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    metrics: dict[str, Any]
    success: bool
    error_message: str | None = None

    def __post_init__(self) -> None:
        self.start_time = normalize_timestamp(self.start_time)
        self.end_time = normalize_timestamp(self.end_time)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for IoT applications.

    Provides real-time monitoring of system resources, IoT-specific metrics,
    and performance benchmarking capabilities.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.is_monitoring: bool = False
        self.monitoring_thread: threading.Thread | None = None
        self.metrics_history: deque[PerformanceMetrics] = deque(
            maxlen=1000
        )  # Keep last 1000 metrics
        self.benchmark_history: list[BenchmarkResult] = []

        # Monitoring intervals
        self.system_metrics_interval = self.config.get(
            "system_metrics_interval_seconds", 5
        )
        self.iot_metrics_interval = self.config.get("iot_metrics_interval_seconds", 1)

        # Performance thresholds
        self.thresholds = {
            "cpu_percent": self.config.get("cpu_threshold", 80.0),
            "memory_percent": self.config.get("memory_threshold", 85.0),
            "measurements_per_second": self.config.get("throughput_threshold", 100.0),
            "latency_ms": self.config.get("latency_threshold", 1000.0),
            "error_rate": self.config.get("error_rate_threshold", 0.05),
        }

        # IoT system references for metrics collection
        self.iot_system: Any = None
        self.measurement_count: int = 0
        self.last_measurement_time: float | None = None

        logger.info("PerformanceMonitor initialized")

    def start_monitoring(self, iot_system: Any = None) -> None:
        """
        Start performance monitoring.

        Args:
            iot_system: Reference to IoT system for collecting IoT-specific metrics
        """
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.iot_system = iot_system
        self.is_monitoring = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        last_system_check = time.monotonic()
        last_iot_check = time.monotonic()

        while self.is_monitoring:
            current_time = time.monotonic()

            # Collect system metrics periodically
            if current_time - last_system_check >= self.system_metrics_interval:
                self._collect_system_metrics()
                last_system_check = current_time

            # Collect IoT-specific metrics more frequently
            if current_time - last_iot_check >= self.iot_metrics_interval:
                self._collect_iot_metrics()
                last_iot_check = current_time

            time.sleep(0.1)  # Small sleep to prevent busy waiting

    def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            if psutil is None:
                return
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            metrics = PerformanceMetrics(
                timestamp=datetime.now(UTC),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / (1024 * 1024),
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_io_sent=network_io.bytes_sent if network_io else 0,
                network_io_recv=network_io.bytes_recv if network_io else 0,
                thread_count=threading.active_count(),
                open_files=len(psutil.Process().open_files()),
            )

            self.metrics_history.append(metrics)

            # Check thresholds and log warnings
            self._check_performance_thresholds(metrics)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_iot_metrics(self) -> None:
        """Collect IoT-specific performance metrics."""
        if self.iot_system is None:
            return

        try:
            if not self.metrics_history:
                # Nothing to annotate yet; system metrics collection owns
                # history creation.
                return

            # Update measurement throughput
            current_measurements = len(self.iot_system.ingestion.measurements)
            current_time = time.monotonic()

            if self.last_measurement_time is not None:
                time_diff = current_time - self.last_measurement_time
                if time_diff > 0:
                    measurements_diff = current_measurements - self.measurement_count
                    self.metrics_history[-1].measurements_per_second = (
                        measurements_diff / time_diff
                    )

            self.measurement_count = current_measurements
            self.last_measurement_time = current_time

            # Real ingest latency, instrumented at IoTDataIngestion
            # .ingest_measurement entry/exit and surfaced here.
            metrics = self.metrics_history[-1]
            metrics.processing_latency_ms = float(
                getattr(self.iot_system.ingestion, "last_ingest_latency_ms", 0.0)
            )

            status = self.iot_system.get_system_status()
            metrics.error_rate = status.get("error_count", 0) / max(
                status.get("measurements", 1), 1
            )
            # Pending processing tasks are the ingestion queue.
            metrics.queue_size = len(
                getattr(self.iot_system.ingestion, "processing_tasks", [])
            )

        except Exception as e:
            logger.error(f"Error collecting IoT metrics: {e}")

    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> list[str]:
        """Check if performance metrics exceed thresholds."""
        warnings = []

        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.thresholds["memory_percent"]:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        if metrics.measurements_per_second < self.thresholds["measurements_per_second"]:
            warnings.append(
                f"Low measurement throughput: {metrics.measurements_per_second:.1f}/sec"
            )

        if metrics.processing_latency_ms > self.thresholds["latency_ms"]:
            warnings.append(
                f"High processing latency: {metrics.processing_latency_ms:.1f}ms"
            )

        if metrics.error_rate > self.thresholds["error_rate"]:
            warnings.append(f"High error rate: {metrics.error_rate:.2%}")

        if warnings:
            logger.warning(f"Performance thresholds exceeded: {', '.join(warnings)}")
        return warnings

    def get_current_metrics(self) -> PerformanceMetrics | None:
        """Get the most recent performance metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, minutes: int = 60) -> list[PerformanceMetrics]:
        """Get performance metrics history for the specified time window."""
        cutoff_time = datetime.now(UTC) - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_performance_summary(self, minutes: int = 60) -> dict[str, Any]:
        """Get performance summary for the specified time window."""
        history = self.get_metrics_history(minutes)

        if not history:
            return {"error": "No metrics data available"}

        # Calculate summary statistics
        cpu_percentages = [m.cpu_percent for m in history]
        memory_percentages = [m.memory_percent for m in history]
        measurements_per_second = [m.measurements_per_second for m in history]
        latencies = [m.processing_latency_ms for m in history]
        error_rates = [m.error_rate for m in history]

        summary = {
            "time_window_minutes": minutes,
            "total_samples": len(history),
            "cpu_usage": {
                "mean": np.mean(cpu_percentages),
                "max": np.max(cpu_percentages),
                "min": np.min(cpu_percentages),
                "std": np.std(cpu_percentages),
            },
            "memory_usage": {
                "mean": np.mean(memory_percentages),
                "max": np.max(memory_percentages),
                "min": np.min(memory_percentages),
                "std": np.std(memory_percentages),
            },
            "throughput": {
                "mean": np.mean(measurements_per_second),
                "max": np.max(measurements_per_second),
                "min": np.min(measurements_per_second),
                "std": np.std(measurements_per_second),
            },
            "latency": {
                "mean": np.mean(latencies),
                "max": np.max(latencies),
                "min": np.min(latencies),
                "std": np.std(latencies),
            },
            "error_rate": {
                "mean": np.mean(error_rates),
                "max": np.max(error_rates),
                "min": np.min(error_rates),
                "std": np.std(error_rates),
            },
            "threshold_exceedances": self._count_threshold_exceedances(history),
            "generated_at": datetime.now(UTC).isoformat(),
        }

        return summary

    def _count_threshold_exceedances(
        self, history: list[PerformanceMetrics]
    ) -> dict[str, int]:
        """Count how many times each threshold was exceeded."""
        exceedances = {
            "cpu_percent": 0,
            "memory_percent": 0,
            "measurements_per_second": 0,
            "latency_ms": 0,
            "error_rate": 0,
        }

        for metrics in history:
            if metrics.cpu_percent > self.thresholds["cpu_percent"]:
                exceedances["cpu_percent"] += 1
            if metrics.memory_percent > self.thresholds["memory_percent"]:
                exceedances["memory_percent"] += 1
            if (
                metrics.measurements_per_second
                < self.thresholds["measurements_per_second"]
            ):
                exceedances["measurements_per_second"] += 1
            if metrics.processing_latency_ms > self.thresholds["latency_ms"]:
                exceedances["latency_ms"] += 1
            if metrics.error_rate > self.thresholds["error_rate"]:
                exceedances["error_rate"] += 1

        return exceedances

    def run_benchmark(self, benchmark_type: str, **kwargs: Any) -> BenchmarkResult:
        """
        Run a performance benchmark.

        Args:
            benchmark_type: Type of benchmark to run
            **kwargs: Benchmark-specific parameters

        Returns:
            BenchmarkResult with results
        """
        benchmark_id = f"benchmark_{int(time.time())}_{benchmark_type}"

        try:
            start_time = datetime.now(UTC)
            started_monotonic = time.perf_counter()

            if benchmark_type == "ingestion_throughput":
                result = self._benchmark_ingestion_throughput(**kwargs)
            elif benchmark_type == "spatial_inference":
                result = self._benchmark_spatial_inference(**kwargs)
            elif benchmark_type == "memory_usage":
                result = self._benchmark_memory_usage(**kwargs)
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark_type}")

            end_time = datetime.now(UTC)
            duration = time.perf_counter() - started_monotonic

            benchmark_result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=benchmark_type,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metrics=result,
                success=True,
            )

        except Exception as e:
            benchmark_result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=benchmark_type,
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC),
                duration_seconds=0.0,
                metrics={},
                success=False,
                error_message=str(e),
            )

        self.benchmark_history.append(benchmark_result)
        return benchmark_result

    def _benchmark_ingestion_throughput(
        self, duration_seconds: int = 60, batch_size: int = 100
    ) -> dict:
        """Benchmark data ingestion throughput."""
        if self.iot_system is None:
            return {"error": "IoT system not available for benchmarking"}

        # Simulate data ingestion load
        start_time = time.perf_counter()

        # Generate and process test measurements
        test_measurements = []
        for i in range(batch_size):
            test_measurement = {
                "sensor_id": f"benchmark_sensor_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
                "variable": "temperature",
                "value": 25.0 + i * 0.1,
                "unit": "celsius",
                "latitude": 40.7128,
                "longitude": -74.0060,
            }
            test_measurements.append(test_measurement)

        # Process measurements
        processed_count = 0
        for measurement in test_measurements:
            try:
                # Simulate ingestion (without spatial inference for speed)
                self.iot_system.ingestion.measurements.append(
                    self.iot_system.ingestion._dict_to_measurement(measurement)
                )
                processed_count += 1
            except Exception as e:
                logger.warning(f"Benchmark measurement failed: {e}")

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        throughput = processed_count / actual_duration if actual_duration > 0 else 0

        return {
            "throughput_measurements_per_second": throughput,
            "measurements_processed": processed_count,
            "duration_seconds": actual_duration,
            "batch_size": batch_size,
            "success_rate": processed_count / batch_size,
        }

    def _benchmark_spatial_inference(
        self, num_sensors: int = 50, iterations: int = 10
    ) -> dict:
        """Benchmark spatial inference performance."""
        if self.iot_system is None or self.iot_system.spatial_inference is None:
            return {"error": "Spatial inference not available for benchmarking"}

        # Generate test data
        test_measurements = []
        for i in range(num_sensors):
            measurement = {
                "sensor_id": f"benchmark_sensor_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
                "variable": "temperature",
                "value": 20.0 + i * 0.1,
                "unit": "celsius",
                "latitude": 40.7128 + (i % 10) * 0.01,
                "longitude": -74.0060 + (i // 10) * 0.01,
            }
            test_measurements.append(measurement)

        # Run multiple inference iterations
        inference_times = []

        for _ in range(iterations):
            start_time = time.perf_counter()

            try:
                # Run spatial inference
                self.iot_system.spatial_inference.infer_spatial_distribution(
                    test_measurements, update_interval="1min"
                )
            except Exception as e:
                logger.warning(f"Inference benchmark failed: {e}")

            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)

        return {
            "average_inference_time_seconds": avg_inference_time,
            "inference_time_std_seconds": std_inference_time,
            "min_inference_time_seconds": np.min(inference_times),
            "max_inference_time_seconds": np.max(inference_times),
            "iterations": iterations,
            "sensor_count": num_sensors,
            "success_rate": len([t for t in inference_times if t > 0]) / iterations,
        }

    def _benchmark_memory_usage(
        self, operation: str = "ingestion", iterations: int = 100
    ) -> dict:
        """Benchmark memory usage for different operations."""
        if psutil is None:
            return {"error": "psutil not available"}
        process = psutil.Process()

        if operation == "ingestion" and self.iot_system:
            # Benchmark ingestion memory usage
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Perform ingestion operations
            for i in range(iterations):
                test_measurement = {
                    "sensor_id": f"benchmark_sensor_{i}",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "variable": "temperature",
                    "value": 25.0,
                    "unit": "celsius",
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                }

                try:
                    self.iot_system.ingestion._dict_to_measurement(test_measurement)
                except Exception as e:
                    logger.warning(f"Memory benchmark measurement failed: {e}")

            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory

            return {
                "operation": operation,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "memory_per_operation_mb": (
                    memory_increase / iterations if iterations > 0 else 0
                ),
                "iterations": iterations,
            }

        return {"error": f"Unsupported operation: {operation}"}

    def export_metrics(self, output_path: str, format: str = "json") -> dict:
        """Export performance metrics to file."""
        try:
            export_data = {
                "export_timestamp": datetime.now(UTC).isoformat(),
                "monitor_config": self.config,
                "thresholds": self.thresholds,
                "metrics_history": [m.__dict__ for m in self.metrics_history],
                "benchmark_history": [b.__dict__ for b in self.benchmark_history],
            }

            if format.lower() == "json":
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == "csv":
                # Export to CSV (simplified)
                import csv

                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "timestamp",
                            "cpu_percent",
                            "memory_percent",
                            "measurements_per_second",
                        ]
                    )
                    for metrics in self.metrics_history:
                        writer.writerow(
                            [
                                metrics.timestamp.isoformat(),
                                metrics.cpu_percent,
                                metrics.memory_percent,
                                metrics.measurements_per_second,
                            ]
                        )
            else:
                return {"error": f"Unsupported export format: {format}"}

            return {
                "success": True,
                "export_path": output_path,
                "format": format,
                "metrics_count": len(self.metrics_history),
                "benchmarks_count": len(self.benchmark_history),
            }

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return {"error": f"Export failed: {str(e)}"}

    def get_system_health_score(self) -> float:
        """Calculate overall system health score based on performance metrics."""
        if not self.metrics_history:
            return 0.0

        latest_metrics = self.get_current_metrics()
        if latest_metrics is None:
            return 0.0

        # Calculate health score based on key metrics
        health_factors = []

        # CPU health (lower is better for this component)
        cpu_health = max(0, 1.0 - (latest_metrics.cpu_percent / 100.0))
        health_factors.append(("cpu", cpu_health, 0.3))

        # Memory health (lower is better)
        memory_health = max(0, 1.0 - (latest_metrics.memory_percent / 100.0))
        health_factors.append(("memory", memory_health, 0.3))

        # Throughput health (higher is better)
        target_throughput = self.thresholds["measurements_per_second"]
        throughput_health = min(
            1.0, latest_metrics.measurements_per_second / target_throughput
        )
        health_factors.append(("throughput", throughput_health, 0.2))

        # Error rate health (lower is better)
        error_health = max(0, 1.0 - latest_metrics.error_rate)
        health_factors.append(("error_rate", error_health, 0.2))

        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0

        for _, score, weight in health_factors:
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def get_performance_report(self, hours: int = 24) -> dict:
        """Generate comprehensive performance report."""
        summary = self.get_performance_summary(hours * 60)  # Convert to minutes
        health_score = self.get_system_health_score()

        # Get benchmark summary if available
        benchmark_summary = {}
        if self.benchmark_history:
            recent_benchmarks = [
                b
                for b in self.benchmark_history
                if b.start_time >= datetime.now(UTC) - timedelta(hours=hours)
            ]

            if recent_benchmarks:
                benchmark_summary = {
                    "total_benchmarks": len(recent_benchmarks),
                    "successful_benchmarks": len(
                        [b for b in recent_benchmarks if b.success]
                    ),
                    "average_duration": np.mean(
                        [b.duration_seconds for b in recent_benchmarks]
                    ),
                    "benchmark_types": list(
                        set([b.benchmark_type for b in recent_benchmarks])
                    ),
                }

        return {
            "report_period_hours": hours,
            "performance_summary": summary,
            "system_health_score": health_score,
            "health_status": (
                "healthy"
                if health_score >= 0.8
                else "degraded"
                if health_score >= 0.5
                else "critical"
            ),
            "benchmark_summary": benchmark_summary,
            "thresholds": self.thresholds,
            "generated_at": datetime.now(UTC).isoformat(),
        }


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def start_performance_monitoring(iot_system: Any = None) -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring(iot_system)


def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


def get_current_performance_metrics() -> PerformanceMetrics | None:
    """Get current performance metrics."""
    monitor = get_performance_monitor()
    return monitor.get_current_metrics()


def run_performance_benchmark(benchmark_type: str, **kwargs: Any) -> BenchmarkResult:
    """Run a performance benchmark."""
    monitor = get_performance_monitor()
    return monitor.run_benchmark(benchmark_type, **kwargs)


def get_performance_report(hours: int = 24) -> dict[str, Any]:
    """Get comprehensive performance report."""
    monitor = get_performance_monitor()
    return monitor.get_performance_report(hours)
