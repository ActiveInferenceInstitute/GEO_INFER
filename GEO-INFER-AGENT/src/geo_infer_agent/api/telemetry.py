#!/usr/bin/env python3

"""
Telemetry module for GEO-INFER-AGENT.

This module collects, processes, and reports metrics about agent performance,
resource usage, and activities to enable monitoring, debugging, and optimization.
"""

import os
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, cast
from collections.abc import Callable
from collections import defaultdict, deque

logger = logging.getLogger("geo_infer_agent.api.telemetry")


class MetricType:
    """Enum for different metric types."""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Point-in-time value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements


class Metric:
    """Base class for a telemetry metric."""

    def __init__(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """
        Initialize a metric.

        Args:
            name: Metric name
            description: Description of what the metric measures
            agent_id: ID of the agent this metric belongs to (None for system metrics)
            tags: Additional key-value tags for the metric
        """
        self.name = name
        self.description = description
        self.agent_id = agent_id
        self.tags = tags or {}
        self.metric_type: str | None = None
        self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "tags": self.tags,
            "type": self.metric_type,
            "created_at": self.created_at.isoformat(),
        }


class CounterMetric(Metric):
    """A metric that monotonically increases."""

    def __init__(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Initialize a counter metric."""
        super().__init__(name, description, agent_id, tags)
        self.metric_type = MetricType.COUNTER
        self.value = 0

    def increment(self, amount: int = 1) -> None:
        """
        Increment the counter.

        Args:
            amount: Amount to increment by (default 1)
        """
        self.value += amount

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result["value"] = self.value
        return result


class GaugeMetric(Metric):
    """A metric that can go up or down."""

    def __init__(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Initialize a gauge metric."""
        super().__init__(name, description, agent_id, tags)
        self.metric_type = MetricType.GAUGE
        self.value: float = 0

    def set(self, value: float) -> None:
        """
        Set the gauge value.

        Args:
            value: New value for the gauge
        """
        self.value = value

    def increment(self, amount: float = 1.0) -> None:
        """
        Increment the gauge.

        Args:
            amount: Amount to increment by
        """
        self.value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """
        Decrement the gauge.

        Args:
            amount: Amount to decrement by
        """
        self.value -= amount

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result["value"] = self.value
        return result


class HistogramMetric(Metric):
    """A metric that tracks the distribution of values."""

    def __init__(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
        max_samples: int = 1000,
    ):
        """
        Initialize a histogram metric.

        Args:
            name: Metric name
            description: Description of what the metric measures
            agent_id: ID of the agent this metric belongs to
            tags: Additional key-value tags for the metric
            max_samples: Maximum number of samples to store
        """
        super().__init__(name, description, agent_id, tags)
        self.metric_type = MetricType.HISTOGRAM
        self.values: deque[float] = deque(maxlen=max_samples)
        self.min: float | None = None
        self.max: float | None = None
        self.sum: float = 0
        self.count = 0

    def record(self, value: float) -> None:
        """
        Record a value in the histogram.

        Args:
            value: Value to record
        """
        self.values.append(value)
        self.sum += value
        self.count += 1

        if self.min is None or value < self.min:
            self.min = value

        if self.max is None or value > self.max:
            self.max = value

    def mean(self) -> float | None:
        """Calculate the mean of recorded values."""
        if self.count == 0:
            return None
        return self.sum / self.count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "count": self.count,
                "min": self.min,
                "max": self.max,
                "mean": self.mean(),
                "samples": list(self.values),
            }
        )
        return result


class TimerMetric(Metric):
    """A metric that measures duration."""

    def __init__(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Initialize a timer metric."""
        super().__init__(name, description, agent_id, tags)
        self.metric_type = MetricType.TIMER
        self.start_time: float | None = None
        self.histogram = HistogramMetric(
            f"{name}_histogram", f"Histogram for {description}", agent_id, tags
        )

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer and record the duration.

        Returns:
            Duration in seconds
        """
        if self.start_time is None:
            raise ValueError("Timer was not started")

        duration = time.time() - self.start_time
        self.histogram.record(duration)
        self.start_time = None
        return duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "is_running": self.start_time is not None,
                "histogram": self.histogram.to_dict(),
            }
        )
        return result


class TelemetryService:
    """
    Service for collecting and reporting agent telemetry.

    This service handles:
    - Metric collection and storage
    - Periodic reporting
    - Health checks
    - Resource monitoring
    """

    _instance: Optional["TelemetryService"] = None
    _initialized: bool = False

    def __new__(cls) -> "TelemetryService":
        """Singleton pattern to ensure a single telemetry service instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the telemetry service."""
        if self._initialized:
            return

        # Metrics storage
        self.metrics: dict[str, Metric] = {}

        # Callbacks for metric updates
        self.metric_callbacks: dict[str, list[Callable[[str, Metric], None]]] = (
            defaultdict(list)
        )

        # Agent health status
        self.agent_health: dict[str, dict[str, Any]] = {}

        # Configuration
        self.reporting_interval = 60  # seconds

        # Background tasks
        self.reporting_task: asyncio.Task | None = None
        self.resource_monitoring_task: asyncio.Task | None = None
        self.running = False

        self._initialized = True
        logger.info("Telemetry service initialized")

    async def start(self, reporting_interval: int = 60) -> None:
        """
        Start the telemetry service.

        Args:
            reporting_interval: How often to report metrics (in seconds)
        """
        if self.running:
            return

        self.reporting_interval = reporting_interval
        self.running = True

        # Start reporting task
        self.reporting_task = asyncio.create_task(self._periodic_reporting())

        # Start resource monitoring
        self.resource_monitoring_task = asyncio.create_task(self._monitor_resources())

        logger.info(
            f"Telemetry service started with reporting interval {reporting_interval}s"
        )

    async def stop(self) -> None:
        """Stop the telemetry service."""
        if not self.running:
            return

        self.running = False

        # Cancel background tasks
        if self.reporting_task:
            self.reporting_task.cancel()
            try:
                await self.reporting_task
            except asyncio.CancelledError:
                pass

        if self.resource_monitoring_task:
            self.resource_monitoring_task.cancel()
            try:
                await self.resource_monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Telemetry service stopped")

    def register_counter(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> CounterMetric:
        """
        Register a counter metric.

        Args:
            name: Metric name
            description: Description of what the metric measures
            agent_id: ID of the agent this metric belongs to
            tags: Additional key-value tags for the metric

        Returns:
            The created counter metric
        """
        metric_id = self._get_metric_id(name, agent_id, tags)
        if metric_id in self.metrics:
            return cast(CounterMetric, self.metrics[metric_id])

        metric = CounterMetric(name, description, agent_id, tags)
        self.metrics[metric_id] = metric
        logger.debug(f"Registered counter metric: {metric_id}")
        return metric

    def register_gauge(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> GaugeMetric:
        """
        Register a gauge metric.

        Args:
            name: Metric name
            description: Description of what the metric measures
            agent_id: ID of the agent this metric belongs to
            tags: Additional key-value tags for the metric

        Returns:
            The created gauge metric
        """
        metric_id = self._get_metric_id(name, agent_id, tags)
        if metric_id in self.metrics:
            return cast(GaugeMetric, self.metrics[metric_id])

        metric = GaugeMetric(name, description, agent_id, tags)
        self.metrics[metric_id] = metric
        logger.debug(f"Registered gauge metric: {metric_id}")
        return metric

    def register_histogram(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> HistogramMetric:
        """
        Register a histogram metric.

        Args:
            name: Metric name
            description: Description of what the metric measures
            agent_id: ID of the agent this metric belongs to
            tags: Additional key-value tags for the metric

        Returns:
            The created histogram metric
        """
        metric_id = self._get_metric_id(name, agent_id, tags)
        if metric_id in self.metrics:
            return cast(HistogramMetric, self.metrics[metric_id])

        metric = HistogramMetric(name, description, agent_id, tags)
        self.metrics[metric_id] = metric
        logger.debug(f"Registered histogram metric: {metric_id}")
        return metric

    def register_timer(
        self,
        name: str,
        description: str,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> TimerMetric:
        """
        Register a timer metric.

        Args:
            name: Metric name
            description: Description of what the metric measures
            agent_id: ID of the agent this metric belongs to
            tags: Additional key-value tags for the metric

        Returns:
            The created timer metric
        """
        metric_id = self._get_metric_id(name, agent_id, tags)
        if metric_id in self.metrics:
            return cast(TimerMetric, self.metrics[metric_id])

        metric = TimerMetric(name, description, agent_id, tags)
        self.metrics[metric_id] = metric
        logger.debug(f"Registered timer metric: {metric_id}")
        return metric

    def update_health(
        self,
        agent_id: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Update health status for an agent.

        Args:
            agent_id: ID of the agent
            status: Health status (e.g., 'healthy', 'degraded', 'failing')
            details: Additional health details
        """
        self.agent_health[agent_id] = {
            "status": status,
            "details": details or {},
            "updated_at": datetime.now().isoformat(),
        }
        logger.debug(f"Updated health for agent {agent_id}: {status}")

    def get_metrics(self, agent_id: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Get all metrics, optionally filtered by agent ID.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Dictionary of metrics data
        """
        result = {}
        for metric_id, metric in self.metrics.items():
            if agent_id is None or metric.agent_id == agent_id:
                result[metric_id] = metric.to_dict()
        return result

    def get_health_status(
        self, agent_id: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Get health status for agents.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Dictionary of agent health data
        """
        if agent_id:
            return {agent_id: self.agent_health.get(agent_id, {"status": "unknown"})}
        return self.agent_health

    def register_metric_callback(
        self, metric_name: str, callback: Callable[[str, Metric], None]
    ) -> None:
        """
        Register a callback to be called when a metric is updated.

        Args:
            metric_name: Name of the metric
            callback: Function to call when the metric is updated
        """
        self.metric_callbacks[metric_name].append(callback)
        logger.debug(f"Registered callback for metric {metric_name}")

    def _get_metric_id(
        self, name: str, agent_id: str | None, tags: dict[str, str] | None
    ) -> str:
        """
        Generate a unique ID for a metric.

        Args:
            name: Metric name
            agent_id: ID of the agent this metric belongs to
            tags: Additional key-value tags for the metric

        Returns:
            Unique metric ID
        """
        tags_str = ""
        if tags:
            tags_str = ";" + ";".join(f"{k}={v}" for k, v in sorted(tags.items()))

        agent_prefix = f"{agent_id}:" if agent_id else ""
        return f"{agent_prefix}{name}{tags_str}"

    async def _periodic_reporting(self) -> None:
        """Background task for periodic metric reporting."""
        while self.running:
            try:
                # Time to report metrics
                metrics_data = {
                    metric_id: metric.to_dict()
                    for metric_id, metric in self.metrics.items()
                }

                # Report metrics: log summary and write JSON snapshot to disk
                report_path = os.path.join(
                    os.environ.get("GEO_INFER_TELEMETRY_DIR", "/tmp"),
                    f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
                try:
                    with open(report_path, "w") as fp:
                        json.dump(metrics_data, fp, indent=2, default=str)
                except OSError:
                    pass  # Non-critical; directory may not exist
                logger.info(
                    f"Metrics summary: {len(metrics_data)} metrics collected, snapshot at {report_path}"
                )

                await asyncio.sleep(self.reporting_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric reporting: {str(e)}")
                await asyncio.sleep(10)  # Reduced interval on error

    async def _monitor_resources(self) -> None:
        """Background task for monitoring system and agent resources."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available, resource monitoring disabled")
            return

        # Register system metrics
        cpu_gauge = self.register_gauge(
            "system.cpu.usage", "System CPU usage percentage", None, {"type": "system"}
        )

        memory_gauge = self.register_gauge(
            "system.memory.usage",
            "System memory usage percentage",
            None,
            {"type": "system"},
        )

        while self.running:
            try:
                # Update system metrics
                cpu_gauge.set(psutil.cpu_percent())
                memory_gauge.set(psutil.virtual_memory().percent)

                # Track per-agent resource consumption from health registry
                for agent_id, health in self.agent_health.items():
                    agent_cpu = self.register_gauge(
                        f"agent.{agent_id}.cpu",
                        f"CPU usage for agent {agent_id}",
                        agent_id,
                        {"type": "agent"},
                    )
                    # Use health details if available, otherwise estimate from process
                    agent_cpu.set(health.get("details", {}).get("cpu_percent", 0.0))

                await asyncio.sleep(5)  # Check resources every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                await asyncio.sleep(10)  # Reduced interval on error


# Global instance
telemetry_service = TelemetryService()
