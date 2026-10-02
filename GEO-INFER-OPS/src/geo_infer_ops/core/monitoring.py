"""Monitoring configuration module."""

import logging
import socket
from contextlib import contextmanager
from typing import Any
from collections.abc import Iterator

import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

from .config import get_config

# Export REGISTRY as METRICS_REGISTRY for consistency
METRICS_REGISTRY = REGISTRY

# Define metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["module", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["module", "endpoint", "status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

ERROR_COUNT = Counter(
    "http_errors_total", "Total number of HTTP errors", ["module", "error_type"]
)

CACHE_SIZE = Gauge(
    "cache_size_bytes", "Current size of the cache in bytes", ["cache_name"]
)

QUEUE_SIZE = Gauge("queue_size", "Current number of items in the queue", ["queue_name"])

_STATIC_METRICS = [REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT, CACHE_SIZE, QUEUE_SIZE]
_CUSTOM_METRICS: dict[str, Any] = {}


def reset_metrics() -> None:
    """Reset custom and package-owned metrics without unregistering core collectors."""
    for metric in _STATIC_METRICS:
        if hasattr(metric, "_metrics"):
            metric._metrics.clear()

    for metric in list(_CUSTOM_METRICS.values()):
        try:
            METRICS_REGISTRY.unregister(metric)
        except (KeyError, ValueError):
            pass
    _CUSTOM_METRICS.clear()


def record_request(module: str, endpoint: str, status: int, duration: float) -> None:
    """Record a request metric.

    Args:
        module: Module or service name handling the request
        endpoint: Request endpoint
        status: HTTP status code
        duration: Request duration in seconds
    """
    labels = {"module": module, "endpoint": endpoint, "status": str(status)}
    REQUEST_COUNT.labels(**labels).inc()
    REQUEST_LATENCY.labels(**labels).observe(duration)


def record_error(module: str, error_type: str) -> None:
    """Record an error metric.

    Args:
        module: Module or service name raising the error
        error_type: Type of error
    """
    ERROR_COUNT.labels(module=module, error_type=error_type).inc()


def record_metric(
    name: str,
    value: float,
    metric_type: str = "counter",
    labels: dict[str, str] | None = None,
) -> None:
    """Record a metric value.

    Args:
        name: Metric name
        value: Metric value
        metric_type: Type of metric (counter, gauge, histogram)
        labels: Optional metric labels
    """
    labels = labels or {}
    label_names = tuple(labels.keys())

    existing = _CUSTOM_METRICS.get(name)
    if (
        existing is not None
        and tuple(getattr(existing, "_labelnames", ())) != label_names
    ):
        try:
            METRICS_REGISTRY.unregister(existing)
        except (KeyError, ValueError):
            pass
        existing = None

    if existing is None:
        if metric_type == "counter":
            existing = Counter(name, f"Counter metric {name}", label_names)
        elif metric_type == "gauge":
            existing = Gauge(name, f"Gauge metric {name}", label_names)
        elif metric_type == "histogram":
            existing = Histogram(name, f"Histogram metric {name}", label_names)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")
        _CUSTOM_METRICS[name] = existing

    metric = existing.labels(**labels) if labels else existing
    if isinstance(metric, Counter):
        metric.inc(value)
    elif isinstance(metric, Gauge):
        metric.set(value)
    elif isinstance(metric, Histogram):
        metric.observe(value)
    else:
        raise ValueError(f"Invalid metric type: {metric_type}")


def get_metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    """Get the value of a metric.

    Args:
        name: Metric name
        labels: Optional metric labels

    Returns:
        Metric value

    Raises:
        ValueError: If metric not found
    """
    labels = {key: str(value) for key, value in (labels or {}).items()}

    for collector in METRICS_REGISTRY._collector_to_names:
        for metric in collector.collect():
            for sample in metric.samples:
                if sample.name != name:
                    continue
                sample_labels = {
                    key: str(value) for key, value in sample.labels.items()
                }
                if sample_labels == labels:
                    return float(sample.value)

    raise ValueError(f"Metric {name} not found")


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use.

    Args:
        port: Port number

    Returns:
        bool: True if port is in use
    """
    if port:
        # macOS can permit a reusable wildcard bind beside a listener bound
        # to loopback (or vice versa). Detect the actual local listener first.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener_probe:
            listener_probe.settimeout(0.2)
            if listener_probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Match the HTTP server's reusable-address bind policy. A completed
        # request can leave TIME_WAIT sockets after a correctly closed server.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            # On macOS, reusable sockets can bind beside a listener but a
            # second listener still fails. Test the actual server operation.
            s.listen(1)
            return False
        except OSError:
            return True


@contextmanager
def start_metrics_server(port: int = 9090) -> Iterator[int]:
    """Start metrics server.

    Args:
        port: Port to start server on

    Yields:
        The port selected for the metrics server.
    """
    # Find available port if specified port is in use
    while is_port_in_use(port):
        port += 1

    server = None
    thread = None
    try:
        server, thread = prom.start_http_server(port, registry=METRICS_REGISTRY)
        port = server.server_port
        yield port
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)


def instrument_app(app: Any, metrics_path: str = "/metrics") -> None:
    """Instrument a FastAPI application.

    Args:
        app: FastAPI application instance
    """
    from prometheus_fastapi_instrumentator import Instrumentator

    instrumentator: Any = Instrumentator(
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
    )

    instrumentator.metrics_path = metrics_path
    instrumentator.instrument(app).expose(app)
    app.instrumentator = instrumentator


def setup_monitoring(
    app: Any | None = None,
    port: int | None = None,
    metrics_path: str = "/metrics",
) -> None:
    """Set up monitoring.

    Args:
        app: Optional FastAPI application to instrument
        port: Optional port for metrics server
        metrics_path: Path for metrics endpoint
    """
    config = get_config()
    if not config.monitoring.enabled:
        return

    # Instrument app if provided
    if app:
        instrument_app(app, metrics_path=metrics_path)

    # Start metrics server if port specified
    if port:
        if port == 9090:
            prom.start_http_server(port, registry=METRICS_REGISTRY)
        else:
            prom.start_http_server(port)

    logging.info("Monitoring setup complete")


# Export functions and registry
__all__ = [
    "METRICS_REGISTRY",
    "reset_metrics",
    "record_request",
    "record_error",
    "record_metric",
    "get_metric_value",
    "is_port_in_use",
    "start_metrics_server",
    "instrument_app",
    "setup_monitoring",
]
