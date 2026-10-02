"""
GEO-INFER-LOG: Geospatial Logistics Optimization Module

This module provides logistics and supply-chain optimization for the
GEO-INFER framework: route optimization, fleet management, last-mile
delivery scheduling, multimodal transportation planning, and supply-chain
network design with geospatial intelligence.

Key Features:
- Route optimization and fleet management (network-based and network-free)
- Last-mile delivery routing, scheduling, and service-area analysis
- Multimodal transportation planning, traffic simulation, and emissions
- Supply-chain network design, facility location, and inventory management
- Enhanced structured logging with spatial context and performance metrics
"""

import atexit
import json
import time
import threading
import weakref
from datetime import datetime, timedelta, UTC
from typing import Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import logging
import logging.handlers
import uuid
import queue


__version__ = "0.4.0"
__all__ = [
    # Observability API
    "EnhancedLogger",
    "PerformanceMetrics",
    "LogAnalyzer",
    "SpatialLogContext",
    "get_logger",
    # Logistics API (lazy-loaded from core/)
    "LastMileRouter",
    "DeliveryScheduler",
    "ServiceAreaAnalyzer",
    "MultiModalPlanner",
    "TransportationNetworkAnalyzer",
    "EmissionsCalculator",
    "TrafficSimulator",
    "SupplyChainModel",
    "RouteOptimizer",
    "FleetManager",
    "VehicleRouter",
    "Vehicle",
    "VehicleType",
    "RoutingParameters",
    "TravelTimeEstimator",
    "MultiObjectiveOptimizer",
    "RealTimeTracker",
]


def _shutdown_logger_at_exit(ref: "weakref.ref[EnhancedLogger]") -> None:
    """Drain queued log entries at interpreter shutdown.

    Registered per async logger through a weak reference so the atexit
    registry never keeps a logger (or its worker thread) alive.
    """
    logger = ref()
    if logger is None:
        return
    try:
        logger.stop()
    except Exception:  # interpreter shutdown is best-effort
        pass


def __getattr__(name: str) -> Any:
    """Lazy-load logistics classes from core/ subpackages."""
    _logistics_map = {
        "LastMileRouter": "geo_infer_log.core.delivery",
        "DeliveryScheduler": "geo_infer_log.core.delivery",
        "ServiceAreaAnalyzer": "geo_infer_log.core.delivery",
        "MultiModalPlanner": "geo_infer_log.core.transport",
        "TransportationNetworkAnalyzer": "geo_infer_log.core.transport",
        "EmissionsCalculator": "geo_infer_log.core.transport",
        "TrafficSimulator": "geo_infer_log.core.transport",
        "SupplyChainModel": "geo_infer_log.core.supply_chain",
        "ResilienceAnalyzer": "geo_infer_log.core.supply_chain",
        "NetworkOptimizer": "geo_infer_log.core.supply_chain",
        "FacilityLocator": "geo_infer_log.core.supply_chain",
        "RouteOptimizer": "geo_infer_log.core.routing",
        "FleetManager": "geo_infer_log.core.routing",
        "VehicleRouter": "geo_infer_log.core.routing",
        "Vehicle": "geo_infer_log.core.routing",
        "VehicleType": "geo_infer_log.core.routing",
        "RoutingParameters": "geo_infer_log.core.routing",
        "TravelTimeEstimator": "geo_infer_log.core.routing",
        "MultiObjectiveOptimizer": "geo_infer_log.core.routing",
        "RealTimeTracker": "geo_infer_log.core.routing",
        "InventoryManager": "geo_infer_log.core.supply_chain",
    }
    if name in _logistics_map:
        import importlib

        mod = importlib.import_module(_logistics_map[name])
        return getattr(mod, name)
    # Lazy-load submodules (api, core, models, utils) on attribute access
    _submodules = {"api", "core", "models", "utils"}
    if name in _submodules:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class LogEntry:
    """Structured log entry with spatial and temporal context."""

    timestamp: str
    level: str
    module: str
    operation: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    spatial_context: "SpatialLogContext | None" = None
    performance_metrics: dict[str, float] | None = None
    trace_id: str | None = None
    span_id: str | None = None
    error_info: dict[str, Any] | None = None


@dataclass
class SpatialLogContext:
    """Spatial context for geospatial operations."""

    h3_index: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    resolution: int | None = None
    region: str | None = None
    bbox: list[float] | None = None
    coordinate_system: str = "EPSG:4326"


class PerformanceMetrics:
    """Performance metrics collection and analysis."""

    def __init__(self) -> None:
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: dict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = defaultdict(float)
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.start_times: dict[str, tuple[str, float]] = {}
        self.lock = threading.Lock()

    def start_timer(self, operation: str) -> str:
        """Start a timer for an operation."""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.start_times[timer_id] = (operation, time.time())
        return timer_id

    def end_timer(self, timer_id: str) -> float:
        """End a timer and record the duration under its full operation name."""
        with self.lock:
            entry = self.start_times.pop(timer_id, None)
        if entry is None:
            return 0.0
        operation, started = entry
        duration = time.time() - started
        self.record_duration(operation, duration)
        return duration

    def record_duration(self, operation: str, duration: float) -> None:
        """Record operation duration."""
        with self.lock:
            self.metrics[f"{operation}_duration"].append(duration)
            self.histograms[f"{operation}_duration"].append(duration)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric."""
        with self.lock:
            self.gauges[name] = value

    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        with self.lock:
            return self._stats_locked(operation)

    def _stats_locked(self, operation: str) -> dict[str, float]:
        """Compute operation statistics; the lock must already be held."""
        durations = sorted(self.metrics[f"{operation}_duration"])
        if not durations:
            return {}

        n = len(durations)

        return {
            "count": n,
            "mean": sum(durations) / n,
            "min": durations[0],
            "max": durations[-1],
            "p50": durations[n // 2],
            "p95": durations[int(n * 0.95)] if n > 0 else 0,
            "p99": durations[int(n * 0.99)] if n > 0 else 0,
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "performance_stats": {
                    op[: -len("_duration")]: self._stats_locked(op[: -len("_duration")])
                    for op in list(self.metrics.keys())
                    if op.endswith("_duration")
                },
            }


class EnhancedLogger:
    """Enhanced logger with spatial context and performance tracking."""

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}
        self.metrics = PerformanceMetrics()
        self.trace_id = None
        self.span_id = None

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.get("level", "INFO")))

        # Setup handlers
        self._setup_handlers()

        # Log queue for async processing
        self.log_queue: queue.Queue[LogEntry] = queue.Queue()
        self.log_processor_running = False
        # Stop latch + processor handle; stop() must never resurrect the
        # background processor once it has drained.
        self._log_processor_stopped = False
        self._log_processor_thread: threading.Thread | None = None

        # Start background log processor
        if self.config.get("async_logging", True):
            self._start_log_processor()

    def _setup_handlers(self) -> None:
        """Setup logging handlers based on configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()

        outputs = self.config.get("outputs", {"console": {"enabled": True}})

        # Console handler
        if outputs.get("console", {}).get("enabled", True):
            console_handler = logging.StreamHandler()
            console_format = outputs.get("console", {}).get("format", "text")

            formatter: logging.Formatter
            if console_format == "json":
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )

            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if outputs.get("file", {}).get("enabled", False):
            file_config = outputs["file"]
            log_file = Path(file_config.get("path", "logs/geo_infer.log"))
            log_file.parent.mkdir(parents=True, exist_ok=True)

            if file_config.get("rotation", "") == "daily":
                file_handler: logging.Handler = (
                    logging.handlers.TimedRotatingFileHandler(
                        log_file,
                        when="D",
                        interval=1,
                        backupCount=int(
                            file_config.get("retention", "30").replace("d", "")
                        ),
                    )
                )
            else:
                file_handler = logging.FileHandler(log_file)

            file_format = file_config.get("format", "json")
            if file_format == "json":
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _start_log_processor(self) -> None:
        """Start background log processor for async logging."""
        if not self.log_processor_running:
            self.log_processor_running = True
            self._log_processor_thread = threading.Thread(
                target=self._process_logs, daemon=True
            )
            self._log_processor_thread.start()
        atexit.register(_shutdown_logger_at_exit, weakref.ref(self))

    def _process_logs(self) -> None:
        """Process logs from the queue."""
        while self.log_processor_running:
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                self._write_log_entry(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.getLogger(__name__).error("Error processing log entry: %s", e)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background log processor, draining remaining entries.

        Sets the running flag, joins the worker with a deadline, then
        processes any entries still queued so exit-burst logs are not lost.
        Idempotent; after stop(), log() writes synchronously and never
        resurrects the worker.
        """
        self._log_processor_stopped = True
        if not self.log_processor_running:
            # Idempotent; also covers loggers created with async_logging=False.
            return
        self.log_processor_running = False
        if self._log_processor_thread is not None:
            self._log_processor_thread.join(timeout)
        # Drain entries the worker missed, processing them synchronously.
        while True:
            try:
                entry = self.log_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._write_log_entry(entry)
            finally:
                self.log_queue.task_done()

    def _write_log_entry(self, entry: LogEntry) -> None:
        """Write log entry to configured outputs."""
        level_method = getattr(self.logger, entry.level.lower(), self.logger.info)

        # Create extra fields for structured logging
        # "module" is a reserved LogRecord attribute; use "log_module" to
        # avoid the KeyError raised by logging for reserved keys.
        extra = {
            "log_module": entry.module,
            "operation": entry.operation,
            "context": entry.context,
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
        }

        if entry.spatial_context:
            extra["spatial_context"] = asdict(entry.spatial_context)

        if entry.performance_metrics:
            extra["performance_metrics"] = entry.performance_metrics

        if entry.error_info:
            extra["error_info"] = entry.error_info

        level_method(entry.message, extra=extra)

    def log(
        self,
        level: str,
        operation: str,
        message: str,
        context: dict | None = None,
        spatial_context: SpatialLogContext | None = None,
        module: str | None = None,
        performance_metrics: dict | None = None,
        error_info: dict | None = None,
    ) -> None:
        """Log a structured message."""

        entry = LogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            level=level.upper(),
            module=module or self.name,
            operation=operation,
            message=message,
            context=context or {},
            spatial_context=spatial_context,
            performance_metrics=performance_metrics,
            trace_id=self.trace_id,
            span_id=self.span_id,
            error_info=error_info,
        )

        # Update metrics
        self.metrics.increment_counter(f"{entry.module}_{entry.operation}")

        # Queue for async processing or process immediately. After stop()
        # the worker is gone: write synchronously so late entries are not
        # silently lost in the queue.
        if self.config.get("async_logging", True) and not self._log_processor_stopped:
            try:
                self.log_queue.put_nowait(entry)
            except queue.Full:
                # Fallback to immediate processing if queue is full
                self._write_log_entry(entry)
        else:
            self._write_log_entry(entry)

    def info(self, operation: str, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.log("INFO", operation, message, **kwargs)

    def debug(self, operation: str, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.log("DEBUG", operation, message, **kwargs)

    def warning(self, operation: str, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.log("WARNING", operation, message, **kwargs)

    def error(self, operation: str, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.log("ERROR", operation, message, **kwargs)

    def critical(self, operation: str, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.log("CRITICAL", operation, message, **kwargs)

    def start_operation(self, operation: str, **context: Any) -> str:
        """Start tracking an operation."""
        timer_id = self.metrics.start_timer(operation)

        self.info(
            f"{operation}_start",
            f"Started operation: {operation}",
            context=context,
            performance_metrics={"timer_id": timer_id},
        )

        return timer_id

    def end_operation(
        self, operation: str, timer_id: str, success: bool = True, **context: Any
    ) -> None:
        """End tracking an operation."""
        duration = self.metrics.end_timer(timer_id)

        level = "info" if success else "error"
        status = "completed" if success else "failed"

        getattr(self, level)(
            f"{operation}_end",
            f"Operation {status}: {operation}",
            context=context,
            performance_metrics={"duration_seconds": duration, "success": success},
        )

    def log_spatial_operation(
        self,
        operation: str,
        h3_index: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        resolution: int | None = None,
        **context: Any,
    ) -> None:
        """Log a spatial operation with geographic context."""
        spatial_context = SpatialLogContext(
            h3_index=h3_index, latitude=lat, longitude=lon, resolution=resolution
        )

        self.info(
            operation,
            f"Spatial operation: {operation}",
            context=context,
            spatial_context=spatial_context,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.get_all_metrics()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": getattr(record, "log_module", record.module),
            "message": record.getMessage(),
            "operation": getattr(record, "operation", "unknown"),
        }

        # Add extra fields
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        if hasattr(record, "spatial_context"):
            log_entry["spatial_context"] = record.spatial_context

        if hasattr(record, "performance_metrics"):
            log_entry["performance_metrics"] = record.performance_metrics

        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id

        if hasattr(record, "error_info"):
            log_entry["error_info"] = record.error_info

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class LogAnalyzer:
    """Analyze logs for patterns, anomalies, and insights."""

    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
        self.log_entries: list[dict] = []

        if self.log_file_path.exists():
            self._load_logs()

    def _load_logs(self) -> None:
        """Load logs from file."""
        try:
            with open(self.log_file_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            self.log_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logging.getLogger(__name__).error(
                "Error loading logs from %s: %s", self.log_file_path, e
            )

    def analyze_performance(self, operation: str | None = None) -> dict[str, Any]:
        """Analyze performance metrics from logs."""
        performance_data = []

        for entry in self.log_entries:
            if operation and entry.get("operation") != operation:
                continue

            if "performance_metrics" in entry:
                metrics = entry["performance_metrics"]
                if "duration_seconds" in metrics:
                    performance_data.append(
                        {
                            "timestamp": entry["timestamp"],
                            "operation": entry["operation"],
                            "duration": metrics["duration_seconds"],
                            "success": metrics.get("success", True),
                        }
                    )

        if not performance_data:
            return {"message": "No performance data found"}

        durations = [p["duration"] for p in performance_data]
        successes = [p for p in performance_data if p["success"]]

        return {
            "total_operations": len(performance_data),
            "successful_operations": len(successes),
            "success_rate": len(successes) / len(performance_data),
            "duration_stats": {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p95": (
                    sorted(durations)[int(len(durations) * 0.95)] if durations else 0
                ),
            },
        }

    def find_errors(self, hours: int = 24) -> list[dict]:
        """Find error entries in the last N hours."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        errors = []
        for entry in self.log_entries:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time > cutoff_time and entry["level"] in ["ERROR", "CRITICAL"]:
                    errors.append(entry)
            except (KeyError, ValueError):
                continue

        return sorted(errors, key=lambda x: x["timestamp"], reverse=True)

    def spatial_analysis(self) -> dict[str, Any]:
        """Analyze spatial operations from logs."""
        spatial_operations = []

        for entry in self.log_entries:
            if "spatial_context" in entry:
                spatial_operations.append(entry)

        if not spatial_operations:
            return {"message": "No spatial operations found"}

        h3_resolutions: dict[Any, int] = defaultdict(int)
        regions: dict[Any, int] = defaultdict(int)

        for op in spatial_operations:
            spatial = op["spatial_context"]
            if "resolution" in spatial:
                h3_resolutions[spatial["resolution"]] += 1
            if "region" in spatial:
                regions[spatial["region"]] += 1

        return {
            "total_spatial_operations": len(spatial_operations),
            "h3_resolution_usage": dict(h3_resolutions),
            "region_distribution": dict(regions),
        }


# Convenience logger factory
def get_logger(name: str, config: dict | None = None) -> EnhancedLogger:
    """Get an enhanced logger instance."""
    return EnhancedLogger(name, config)
