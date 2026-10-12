"""
Performance monitoring utilities for GEO-INFER-DATA.

This module provides performance monitoring and optimization utilities
for data processing operations.
"""

import logging
from typing import Any
from datetime import datetime, UTC
import time
import psutil
import threading


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring for data operations.

    This class provides comprehensive performance monitoring including
    execution time tracking, memory usage monitoring, and bottleneck
    identification.

    Examples:
        >>> monitor = PerformanceMonitor()
        >>>
        >>> # Monitor data processing
        >>> with monitor.track_operation('data_processing'):
        ...     # Your data processing code here
        ...     result = process_data(data)
        >>>
        >>> # Get performance metrics
        >>> metrics = monitor.get_metrics()
        >>> print(f"Average processing time: {metrics['avg_execution_time']:.2f}s")
    """

    def __init__(
        self, enable_memory_monitoring: bool = True, enable_cpu_monitoring: bool = True
    ):
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_cpu_monitoring = enable_cpu_monitoring

        self.metrics: dict[str, Any] = {
            "operations": {},
            "start_time": datetime.now(UTC),
            "memory_usage": {},
            "cpu_usage": {},
        }

        self.active_operations: dict[str, Any] = {}
        self.memory_history: list[float] = []
        self.cpu_history: list[float] = []

        # Start background monitoring if enabled
        if self.enable_memory_monitoring or self.enable_cpu_monitoring:
            self._start_background_monitoring()

        logger.info("Initialized PerformanceMonitor")

    def track_operation(self, operation_name: str) -> "OperationTracker":
        """
        Track performance of an operation.

        Args:
            operation_name: Name of the operation to track

        Returns:
            OperationTracker context manager
        """
        return OperationTracker(self, operation_name)

    def record_metric(
        self, operation_name: str, metric_name: str, value: float
    ) -> None:
        """
        Record a custom metric.

        Args:
            operation_name: Operation name
            metric_name: Metric name
            value: Metric value
        """
        if operation_name not in self.metrics["operations"]:
            self.metrics["operations"][operation_name] = {}

        if metric_name not in self.metrics["operations"][operation_name]:
            self.metrics["operations"][operation_name][metric_name] = []

        self.metrics["operations"][operation_name][metric_name].append(value)

        logger.debug(f"Recorded metric {operation_name}.{metric_name}: {value}")

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Calculate summary statistics
        summary = {
            "monitoring_duration": datetime.now(UTC) - self.metrics["start_time"],
            "total_operations": len(self.metrics["operations"]),
            "operations": {},
        }

        for operation_name, operation_metrics in self.metrics["operations"].items():
            operation_summary = {}

            for metric_name, values in operation_metrics.items():
                if values:
                    operation_summary[metric_name] = {
                        "count": len(values),
                        "total": sum(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            summary["operations"][operation_name] = operation_summary

        # Add memory and CPU metrics
        if self.memory_history:
            summary["memory"] = {
                "current_mb": self.memory_history[-1] if self.memory_history else 0,
                "max_mb": max(self.memory_history) if self.memory_history else 0,
                "average_mb": (
                    sum(self.memory_history) / len(self.memory_history)
                    if self.memory_history
                    else 0
                ),
            }

        if self.cpu_history:
            summary["cpu"] = {
                "current_percent": self.cpu_history[-1] if self.cpu_history else 0,
                "max_percent": max(self.cpu_history) if self.cpu_history else 0,
                "average_percent": (
                    sum(self.cpu_history) / len(self.cpu_history)
                    if self.cpu_history
                    else 0
                ),
            }

        return summary

    def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        metrics = self.get_metrics()

        for operation_name, operation_metrics in metrics["operations"].items():
            # Check execution time
            if "execution_time" in operation_metrics:
                exec_time = operation_metrics["execution_time"]
                if exec_time["average"] > 60:  # More than 1 minute
                    bottlenecks.append(
                        {
                            "type": "slow_operation",
                            "operation": operation_name,
                            "average_time": exec_time["average"],
                            "severity": "high",
                        }
                    )
                elif exec_time["average"] > 10:  # More than 10 seconds
                    bottlenecks.append(
                        {
                            "type": "slow_operation",
                            "operation": operation_name,
                            "average_time": exec_time["average"],
                            "severity": "medium",
                        }
                    )

            # Check memory usage
            if "memory_peak" in operation_metrics:
                memory_mb = operation_metrics["memory_peak"]["average"]
                if memory_mb > 1000:  # More than 1GB
                    bottlenecks.append(
                        {
                            "type": "high_memory",
                            "operation": operation_name,
                            "memory_mb": memory_mb,
                            "severity": "high",
                        }
                    )
                elif memory_mb > 500:  # More than 500MB
                    bottlenecks.append(
                        {
                            "type": "high_memory",
                            "operation": operation_name,
                            "memory_mb": memory_mb,
                            "severity": "medium",
                        }
                    )

        # Check system resources
        if metrics.get("memory", {}).get(
            "max_mb", 0
        ) > 0.9 * psutil.virtual_memory().total / (1024**2):
            bottlenecks.append(
                {
                    "type": "system_memory",
                    "current_mb": metrics["memory"]["current_mb"],
                    "severity": "critical",
                }
            )

        if metrics.get("cpu", {}).get("max_percent", 0) > 90:
            bottlenecks.append(
                {
                    "type": "system_cpu",
                    "current_percent": metrics["cpu"]["current_percent"],
                    "severity": "high",
                }
            )

        return bottlenecks

    def _start_background_monitoring(self) -> None:
        """Start background system monitoring."""

        def monitor_system() -> None:
            while True:
                try:
                    # Monitor memory
                    if self.enable_memory_monitoring:
                        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                        self.memory_history.append(memory_mb)
                        self.metrics["memory_usage"][datetime.now(UTC)] = memory_mb

                        # Keep only last 1000 measurements
                        if len(self.memory_history) > 1000:
                            self.memory_history.pop(0)

                    # Monitor CPU
                    if self.enable_cpu_monitoring:
                        cpu_percent = psutil.cpu_percent(interval=1)
                        self.cpu_history.append(cpu_percent)
                        self.metrics["cpu_usage"][datetime.now(UTC)] = cpu_percent

                        # Keep only last 1000 measurements
                        if len(self.cpu_history) > 1000:
                            self.cpu_history.pop(0)

                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")

                time.sleep(1)  # Monitor every second

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        logger.info("Started background performance monitoring")

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {
            "operations": {},
            "start_time": datetime.now(UTC),
            "memory_usage": {},
            "cpu_usage": {},
        }
        self.memory_history.clear()
        self.cpu_history.clear()
        self.active_operations.clear()

        logger.info("Performance metrics reset")


class OperationTracker:
    """Context manager for tracking operation performance."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time: float | None = None
        self.start_memory: float | None = None

    def __enter__(self) -> "OperationTracker":
        """Start tracking operation."""
        self.start_time = time.time()

        if self.monitor.enable_memory_monitoring:
            try:
                self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                self.start_memory = 0

        self.monitor.active_operations[self.operation_name] = {
            "start_time": self.start_time,
            "start_memory": self.start_memory,
        }

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop tracking operation."""
        end_time = time.time()
        assert self.start_time is not None
        execution_time = end_time - self.start_time

        # Record execution time
        self.monitor.record_metric(
            self.operation_name, "execution_time", execution_time
        )

        # Record memory usage
        if self.monitor.enable_memory_monitoring and self.start_memory is not None:
            try:
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_used = end_memory - self.start_memory
                memory_peak = psutil.Process().memory_info().rss / (1024 * 1024)

                self.monitor.record_metric(
                    self.operation_name, "memory_used", memory_used
                )
                self.monitor.record_metric(
                    self.operation_name, "memory_peak", memory_peak
                )

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

        # Remove from active operations
        if self.operation_name in self.monitor.active_operations:
            del self.monitor.active_operations[self.operation_name]

        # Log operation completion
        logger.debug(
            f"Operation {self.operation_name} completed in {execution_time:.2f}s"
        )


class DataProcessingProfiler:
    """
    Profiler for data processing operations.

    This class provides detailed profiling for data processing operations
    including step-by-step timing and resource usage analysis.

    Examples:
        >>> profiler = DataProcessingProfiler()
        >>>
        >>> # Profile data loading
        >>> with profiler.profile_step('load_data'):
        ...     data = gpd.read_file('data.geojson')
        >>>
        >>> # Profile data processing
        >>> with profiler.profile_step('process_data'):
        ...     processed_data = process_geospatial_data(data)
        >>>
        >>> # Get profiling results
        >>> profile = profiler.get_profile()
        >>> print(f"Total time: {profile['total_time']:.2f}s")
    """

    def __init__(self) -> None:
        self.profile_data: dict[str, Any] = {
            "steps": {},
            "start_time": None,
            "end_time": None,
        }
        self.current_step: str | None = None

    def profile_step(self, step_name: str) -> "StepProfiler":
        """Profile a processing step."""
        return StepProfiler(self, step_name)

    def start_profiling(self) -> None:
        """Start profiling session."""
        self.profile_data["start_time"] = datetime.now(UTC)
        logger.info("Started data processing profiling")

    def end_profiling(self) -> None:
        """End profiling session."""
        self.profile_data["end_time"] = datetime.now(UTC)

        if self.profile_data["start_time"]:
            total_time = (
                self.profile_data["end_time"] - self.profile_data["start_time"]
            ).total_seconds()
            self.profile_data["total_time"] = total_time

        logger.info(
            f"Ended data processing profiling (total: {self.profile_data.get('total_time', 0):.2f}s)"
        )

    def get_profile(self) -> dict[str, Any]:
        """Get profiling results."""
        if not self.profile_data["start_time"]:
            return {"error": "Profiling not started"}

        if not self.profile_data["end_time"]:
            self.end_profiling()

        # Calculate step percentages
        total_time = self.profile_data["total_time"]

        for step_data in self.profile_data["steps"].values():
            step_data["percentage"] = (
                (step_data["duration"] / total_time) * 100 if total_time > 0 else 0
            )

        return self.profile_data


class StepProfiler:
    """Context manager for profiling individual steps."""

    def __init__(self, profiler: DataProcessingProfiler, step_name: str):
        self.profiler = profiler
        self.step_name = step_name
        self.start_time: float | None = None

    def __enter__(self) -> "StepProfiler":
        """Start profiling step."""
        self.start_time = time.time()
        self.profiler.current_step = self.step_name

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End profiling step."""
        end_time = time.time()
        assert self.start_time is not None
        duration = end_time - self.start_time

        # Record step metrics
        if self.step_name not in self.profiler.profile_data["steps"]:
            self.profiler.profile_data["steps"][self.step_name] = {
                "duration": 0,
                "calls": 0,
                "memory_start": 0,
                "memory_end": 0,
                "memory_used": 0,
            }

        step_data = self.profiler.profile_data["steps"][self.step_name]
        step_data["duration"] += duration
        step_data["calls"] += 1

        # Record memory usage
        try:
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            if step_data["calls"] == 1:
                step_data["memory_start"] = current_memory
            step_data["memory_end"] = current_memory
            step_data["memory_used"] = (
                step_data["memory_end"] - step_data["memory_start"]
            )
        except Exception as e:
            logger.error(f"Memory profiling error: {e}")

        self.profiler.current_step = None
