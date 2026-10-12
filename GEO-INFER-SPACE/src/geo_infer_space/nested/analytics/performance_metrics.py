"""
Performance Metrics for H3 Nested Systems.

This module provides comprehensive performance analysis and metrics
for nested geospatial systems, including computational efficiency,
memory usage, and system optimization recommendations.
"""

import logging
import uuid
import time
import psutil
import threading
from types import TracebackType
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from typing import Any, cast
from collections.abc import Callable
from enum import Enum
from collections import defaultdict, deque

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics."""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    SCALABILITY = "scalability"
    EFFICIENCY = "efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"


class BenchmarkType(Enum):
    """Types of benchmarks."""

    OPERATION_SPEED = "operation_speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    SCALABILITY_TEST = "scalability_test"
    STRESS_TEST = "stress_test"
    LOAD_TEST = "load_test"
    ENDURANCE_TEST = "endurance_test"


class OptimizationTarget(Enum):
    """Optimization targets."""

    SPEED = "speed"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BALANCED = "balanced"


@dataclass
class PerformanceMeasurement:
    """
    Represents a single performance measurement.
    """

    measurement_id: str
    metric_type: PerformanceMetric

    # Measurement values
    value: float
    unit: str

    # Context
    operation_name: str
    system_context: dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration: timedelta | None = None

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert measurement to dictionary."""
        return {
            "measurement_id": self.measurement_id,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "operation_name": self.operation_name,
            "system_context": self.system_context,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration.total_seconds() if self.duration else None,
            "tags": self.tags,
        }


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark test.
    """

    benchmark_id: str
    benchmark_type: BenchmarkType

    # Measurements
    measurements: list[PerformanceMeasurement] = field(default_factory=list)

    # Summary statistics
    summary_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Performance scores
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    scalability_score: float = 0.0

    # System information
    system_info: dict[str, Any] = field(default_factory=dict)

    # Test configuration
    test_config: dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration: timedelta = field(default_factory=lambda: timedelta(0))

    def add_measurement(self, measurement: PerformanceMeasurement) -> None:
        """Add a measurement to the benchmark."""
        self.measurements.append(measurement)
        self._update_summary_stats()

    def _update_summary_stats(self) -> None:
        """Update summary statistics."""
        if not self.measurements:
            return

        # Group measurements by metric type
        metric_groups = defaultdict(list)
        for measurement in self.measurements:
            metric_groups[measurement.metric_type.value].append(measurement.value)

        # Calculate statistics for each metric
        for metric_type, values in metric_groups.items():
            if values:
                self.summary_stats[metric_type] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }


@dataclass
class PerformanceProfile:
    """
    Performance profile for a system or operation.
    """

    profile_id: str
    target_operation: str

    # Performance characteristics
    baseline_metrics: dict[PerformanceMetric, float] = field(default_factory=dict)
    current_metrics: dict[PerformanceMetric, float] = field(default_factory=dict)

    # Trends
    performance_trends: dict[PerformanceMetric, list[float]] = field(
        default_factory=dict
    )

    # Thresholds
    performance_thresholds: dict[PerformanceMetric, dict[str, float]] = field(
        default_factory=dict
    )

    # Optimization recommendations
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_metrics(self, metrics: dict[PerformanceMetric, float]) -> None:
        """Update current metrics and trends."""
        for metric, value in metrics.items():
            self.current_metrics[metric] = value

            if metric not in self.performance_trends:
                self.performance_trends[metric] = []

            self.performance_trends[metric].append(value)

            # Keep only recent trends (last 100 measurements)
            if len(self.performance_trends[metric]) > 100:
                self.performance_trends[metric] = self.performance_trends[metric][-100:]

        self.last_updated = datetime.now(UTC)


class PerformanceMonitor:
    """
    Context manager for monitoring performance.
    """

    def __init__(self, operation_name: str, analyzer: "H3PerformanceAnalyzer") -> None:
        self.operation_name = operation_name
        self.analyzer = analyzer
        self.start_time: float | None = None
        self.start_memory: int | None = None
        self.start_cpu: float | None = None

    def __enter__(self) -> "PerformanceMonitor":
        """Start monitoring."""
        self.start_time = time.time()

        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
            self.start_cpu = process.cpu_percent()
        except Exception:
            self.start_memory = 0
            self.start_cpu = 0

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop monitoring and record measurements."""
        end_time = time.time()
        execution_time = end_time - cast(float, self.start_time)

        # Record execution time
        self.analyzer.record_measurement(
            metric_type=PerformanceMetric.EXECUTION_TIME,
            value=execution_time,
            unit="seconds",
            operation_name=self.operation_name,
        )

        try:
            process = psutil.Process()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()

            # Record memory usage
            memory_delta = end_memory - cast(int, self.start_memory)
            self.analyzer.record_measurement(
                metric_type=PerformanceMetric.MEMORY_USAGE,
                value=memory_delta,
                unit="bytes",
                operation_name=self.operation_name,
            )

            # Record CPU usage
            cpu_usage = (cast(float, self.start_cpu) + end_cpu) / 2
            self.analyzer.record_measurement(
                metric_type=PerformanceMetric.CPU_USAGE,
                value=cpu_usage,
                unit="percent",
                operation_name=self.operation_name,
            )

        except Exception as e:
            logger.warning(f"Failed to record system metrics: {e}")


class H3PerformanceAnalyzer:
    """
    Advanced performance analyzer for H3 nested systems.

    Provides comprehensive performance analysis including:
    - Real-time performance monitoring
    - Benchmark testing and comparison
    - Performance profiling and optimization
    - Resource utilization analysis
    """

    def __init__(self, name: str = "H3PerformanceAnalyzer") -> None:
        """
        Initialize performance analyzer.

        Args:
            name: Analyzer name for identification
        """
        self.name = name

        # Measurement storage
        self.measurements: list[PerformanceMeasurement] = []
        self.benchmark_results: dict[str, BenchmarkResult] = {}
        self.performance_profiles: dict[str, PerformanceProfile] = {}

        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.monitoring_data: deque[dict[str, Any]] = deque(maxlen=1000)

        # Configuration
        self.config: dict[str, Any] = {
            "measurement_retention_days": 30,
            "monitoring_interval": 1.0,  # seconds
            "benchmark_iterations": 10,
            "memory_threshold_mb": 1000,
            "cpu_threshold_percent": 80,
        }

        # Statistics
        self.analysis_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def record_measurement(
        self,
        metric_type: PerformanceMetric,
        value: float,
        unit: str,
        operation_name: str,
        system_context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PerformanceMeasurement:
        """
        Record a performance measurement.

        Args:
            metric_type: Type of metric
            value: Measurement value
            unit: Unit of measurement
            operation_name: Name of operation being measured
            system_context: System context information
            tags: Additional tags

        Returns:
            Created PerformanceMeasurement instance
        """
        measurement = PerformanceMeasurement(
            measurement_id=f"measurement_{uuid.uuid4().hex[:8]}",
            metric_type=metric_type,
            value=value,
            unit=unit,
            operation_name=operation_name,
            system_context=system_context or {},
            tags=tags or {},
        )

        self.measurements.append(measurement)
        self.analysis_stats["measurements_recorded"] += 1
        self.updated_at = datetime.now(UTC)

        # Clean up old measurements
        self._cleanup_old_measurements()

        return measurement

    def monitor_operation(self, operation_name: str) -> PerformanceMonitor:
        """
        Create a performance monitor for an operation.

        Args:
            operation_name: Name of operation to monitor

        Returns:
            PerformanceMonitor context manager
        """
        return PerformanceMonitor(operation_name, self)

    def run_benchmark(
        self,
        benchmark_type: BenchmarkType,
        target_function: Callable[..., Any],
        test_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Run a benchmark test.

        Args:
            benchmark_type: Type of benchmark
            target_function: Function to benchmark
            test_config: Test configuration
            **kwargs: Additional parameters for the function

        Returns:
            BenchmarkResult instance
        """
        start_time = datetime.now(UTC)
        benchmark_id = f"benchmark_{uuid.uuid4().hex[:8]}"

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=benchmark_type,
            test_config=test_config or {},
            system_info=self._get_system_info(),
        )

        iterations = (
            test_config.get("iterations", self.config["benchmark_iterations"])
            if test_config
            else self.config["benchmark_iterations"]
        )

        # Run benchmark iterations
        for i in range(iterations):
            try:
                with self.monitor_operation(
                    f"benchmark_{benchmark_type.value}_iter_{i}"
                ):
                    target_function(**kwargs)

                # Get the latest measurements for this iteration
                recent_measurements = [
                    m
                    for m in self.measurements[-10:]
                    if m.operation_name.startswith(
                        f"benchmark_{benchmark_type.value}_iter_{i}"
                    )
                ]

                result.measurements.extend(recent_measurements)

            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
                continue

        # Calculate benchmark scores
        result.performance_score = self._calculate_performance_score(result)
        result.efficiency_score = self._calculate_efficiency_score(result)
        result.scalability_score = self._calculate_scalability_score(result)

        result.duration = datetime.now(UTC) - start_time

        # Store result
        self.benchmark_results[benchmark_id] = result
        self.analysis_stats["benchmarks_run"] += 1
        self.updated_at = datetime.now(UTC)

        return result

    def create_performance_profile(
        self,
        profile_id: str,
        target_operation: str,
        baseline_metrics: dict[PerformanceMetric, float] | None = None,
    ) -> PerformanceProfile:
        """
        Create a performance profile.

        Args:
            profile_id: Profile identifier
            target_operation: Operation being profiled
            baseline_metrics: Baseline performance metrics

        Returns:
            Created PerformanceProfile instance
        """
        profile = PerformanceProfile(
            profile_id=profile_id,
            target_operation=target_operation,
            baseline_metrics=baseline_metrics or {},
        )

        self.performance_profiles[profile_id] = profile
        self.updated_at = datetime.now(UTC)

        return profile

    def update_performance_profile(
        self, profile_id: str, metrics: dict[PerformanceMetric, float]
    ) -> None:
        """
        Update a performance profile with new metrics.

        Args:
            profile_id: Profile identifier
            metrics: New performance metrics
        """
        if profile_id not in self.performance_profiles:
            raise ValueError(f"Performance profile {profile_id} not found")

        profile = self.performance_profiles[profile_id]
        profile.update_metrics(metrics)

        # Generate recommendations
        profile.recommendations = self._generate_performance_recommendations(profile)

        self.updated_at = datetime.now(UTC)

    def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info(f"Performance monitoring started for {self.name}")

    def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        self.monitoring_active = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        logger.info(f"Performance monitoring stopped for {self.name}")

    def _monitoring_loop(self) -> None:
        """Real-time monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()

                # Store monitoring data
                self.monitoring_data.append(
                    {"timestamp": datetime.now(UTC), "metrics": system_metrics}
                )

                time.sleep(self.config["monitoring_interval"])

            except Exception as e:
                logger.warning(f"Monitoring loop error: {e}")
                time.sleep(self.config["monitoring_interval"])

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect current system metrics."""
        metrics = {}

        try:
            process = psutil.Process()

            # Memory metrics
            memory_info = process.memory_info()
            metrics["memory_rss"] = memory_info.rss
            metrics["memory_vms"] = memory_info.vms

            # CPU metrics
            metrics["cpu_percent"] = process.cpu_percent()

            # System-wide metrics
            metrics["system_cpu_percent"] = psutil.cpu_percent()
            metrics["system_memory_percent"] = psutil.virtual_memory().percent

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

        return metrics

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        info = {}

        try:
            info["cpu_count"] = psutil.cpu_count()
            info["memory_total"] = psutil.virtual_memory().total
            info["python_version"] = (
                f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}"
            )

        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")

        return info

    def _calculate_performance_score(self, result: BenchmarkResult) -> float:
        """Calculate overall performance score."""
        if not result.measurements:
            return 0.0

        # Simple scoring based on execution time
        execution_times = [
            m.value
            for m in result.measurements
            if m.metric_type == PerformanceMetric.EXECUTION_TIME
        ]

        if not execution_times:
            return 0.5

        avg_time = sum(execution_times) / len(execution_times)

        # Score inversely related to execution time
        # Assume 1 second is baseline (score = 0.5)
        score = 1.0 / (1.0 + avg_time)

        return min(1.0, max(0.0, score))

    def _calculate_efficiency_score(self, result: BenchmarkResult) -> float:
        """Calculate efficiency score."""
        if not result.measurements:
            return 0.0

        # Efficiency based on memory and CPU usage
        memory_measurements = [
            m.value
            for m in result.measurements
            if m.metric_type == PerformanceMetric.MEMORY_USAGE
        ]

        cpu_measurements = [
            m.value
            for m in result.measurements
            if m.metric_type == PerformanceMetric.CPU_USAGE
        ]

        efficiency_scores = []

        if memory_measurements:
            avg_memory = sum(abs(m) for m in memory_measurements) / len(
                memory_measurements
            )
            memory_score = 1.0 / (1.0 + avg_memory / (1024 * 1024))  # Normalize by MB
            efficiency_scores.append(memory_score)

        if cpu_measurements:
            avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
            cpu_score = 1.0 - (avg_cpu / 100.0)  # Lower CPU usage is better
            efficiency_scores.append(max(0.0, cpu_score))

        return (
            sum(efficiency_scores) / len(efficiency_scores)
            if efficiency_scores
            else 0.5
        )

    def _calculate_scalability_score(self, result: BenchmarkResult) -> float:
        """Calculate scalability score from execution time consistency.

        Scalability is gauged by how *predictably* execution time grows across
        the benchmark measurements.  A system that scales linearly will have
        consistent per-operation latency; one that degrades will show increasing
        variance.

        Score = max(0, 1 - CV) where CV = std(times) / mean(times).
        CV=0 (perfectly consistent) → score=1.0; CV≥1 (chaotic) → score≤0.
        Falls back to the speed score when fewer than 2 measurements exist.
        """
        if not result.measurements:
            return 0.5

        execution_times = [
            m.value
            for m in result.measurements
            if m.metric_type == PerformanceMetric.EXECUTION_TIME and m.value > 0
        ]

        if len(execution_times) < 2:
            # Only one data point — use inverse-time score as proxy
            return cast(
                float,
                self._calculate_speed_score(result),  # type: ignore[attr-defined]
            )

        mean_time = sum(execution_times) / len(execution_times)
        variance = sum((t - mean_time) ** 2 for t in execution_times) / len(
            execution_times
        )
        std_time = variance**0.5

        # Coefficient of variation: lower = more predictable = better scalability
        cv = std_time / mean_time if mean_time > 0 else 1.0
        return max(0.0, min(1.0, 1.0 - cv))

    def _generate_performance_recommendations(
        self, profile: PerformanceProfile
    ) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check memory usage trends
        if PerformanceMetric.MEMORY_USAGE in profile.performance_trends:
            memory_trend = profile.performance_trends[PerformanceMetric.MEMORY_USAGE]
            if len(memory_trend) >= 5:
                recent_memory = memory_trend[-5:]
                if all(
                    recent_memory[i] > recent_memory[i - 1]
                    for i in range(1, len(recent_memory))
                ):
                    recommendations.append(
                        "Memory usage is consistently increasing - check for memory leaks"
                    )

        # Check execution time trends
        if PerformanceMetric.EXECUTION_TIME in profile.performance_trends:
            time_trend = profile.performance_trends[PerformanceMetric.EXECUTION_TIME]
            if len(time_trend) >= 5:
                recent_times = time_trend[-5:]
                avg_time = sum(recent_times) / len(recent_times)
                if avg_time > 1.0:  # More than 1 second
                    recommendations.append(
                        "Consider optimizing algorithm or using caching for better performance"
                    )

        # Check CPU usage
        if PerformanceMetric.CPU_USAGE in profile.current_metrics:
            cpu_usage = profile.current_metrics[PerformanceMetric.CPU_USAGE]
            if cpu_usage > self.config["cpu_threshold_percent"]:
                recommendations.append(
                    "High CPU usage detected - consider parallel processing or optimization"
                )

        return recommendations

    def _cleanup_old_measurements(self) -> None:
        """Clean up old measurements."""
        cutoff_date = datetime.now(UTC) - timedelta(
            days=self.config["measurement_retention_days"]
        )

        self.measurements = [m for m in self.measurements if m.timestamp > cutoff_date]

    def get_performance_summary(
        self,
        operation_name: str | None = None,
        time_window: timedelta | None = None,
    ) -> dict[str, Any]:
        """
        Get performance summary.

        Args:
            operation_name: Filter by operation name
            time_window: Time window for analysis

        Returns:
            Performance summary dictionary
        """
        # Filter measurements
        filtered_measurements = self.measurements

        if operation_name:
            filtered_measurements = [
                m for m in filtered_measurements if m.operation_name == operation_name
            ]

        if time_window:
            cutoff_time = datetime.now(UTC) - time_window
            filtered_measurements = [
                m for m in filtered_measurements if m.timestamp > cutoff_time
            ]

        if not filtered_measurements:
            return {}

        # Calculate summary statistics
        metric_summaries = defaultdict(list)
        for measurement in filtered_measurements:
            metric_summaries[measurement.metric_type.value].append(measurement.value)

        summary = {}
        for metric_type, values in metric_summaries.items():
            if values:
                summary[metric_type] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

        return {
            "summary_statistics": summary,
            "total_measurements": len(filtered_measurements),
            "time_range": {
                "start": min(m.timestamp for m in filtered_measurements).isoformat(),
                "end": max(m.timestamp for m in filtered_measurements).isoformat(),
            },
        }

    def get_analyzer_statistics(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "analyzer_name": self.name,
            "total_measurements": len(self.measurements),
            "total_benchmarks": len(self.benchmark_results),
            "total_profiles": len(self.performance_profiles),
            "monitoring_active": self.monitoring_active,
            "analysis_stats": dict(self.analysis_stats),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
