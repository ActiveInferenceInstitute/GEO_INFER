"""
Tests for monitoring configuration.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY

from geo_infer_ops.core.monitoring import (
    setup_monitoring,
    reset_metrics,
    record_request,
    record_error,
    record_metric,
    get_metric_value,
    instrument_app,
    start_metrics_server,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
)
from geo_infer_ops.core.config import MonitoringConfig


@pytest.fixture
def mock_config():
    """Fixture providing a mock configuration."""
    return MonitoringConfig(
        enabled=True, metrics_port=9091, metrics_path="/test-metrics"
    )


@pytest.fixture
def mock_app():
    """Fixture providing a mock FastAPI application."""
    app = FastAPI()
    return app


def test_setup_monitoring_defaults(mock_config):
    """Test monitoring setup with default configuration."""
    with patch("geo_infer_ops.core.monitoring.get_config") as mock_get_config:
        mock_get_config.return_value.monitoring = mock_config
        setup_monitoring()

        # Verify metrics are registered
        assert REQUEST_COUNT in REGISTRY._collector_to_names
        assert REQUEST_LATENCY in REGISTRY._collector_to_names
        assert ERROR_COUNT in REGISTRY._collector_to_names


def test_setup_monitoring_with_app(mock_config, mock_app):
    """Test monitoring setup with FastAPI application."""
    with patch("geo_infer_ops.core.monitoring.get_config") as mock_get_config:
        mock_get_config.return_value.monitoring = mock_config
        setup_monitoring(app=mock_app)

        # Verify FastAPI instrumentation
        assert hasattr(mock_app, "instrumentator")
        assert mock_app.instrumentator is not None


def test_setup_monitoring_custom_port(mock_config):
    """Test monitoring setup with custom port."""
    with patch("geo_infer_ops.core.monitoring.get_config") as mock_get_config:
        mock_get_config.return_value.monitoring = mock_config
        with patch("prometheus_client.start_http_server") as mock_start_server:
            setup_monitoring(port=9092)
            mock_start_server.assert_called_once_with(9092)


def test_setup_monitoring_custom_path(mock_config, mock_app):
    """Test monitoring setup with custom metrics path."""
    with patch("geo_infer_ops.core.monitoring.get_config") as mock_get_config:
        mock_get_config.return_value.monitoring = mock_config
        setup_monitoring(app=mock_app, metrics_path="/custom-metrics")

        # Verify custom metrics path
        assert mock_app.instrumentator.metrics_path == "/custom-metrics"


def test_record_request():
    """Test recording request metrics."""
    # Reset metrics
    reset_metrics()

    # Record a request
    record_request("test_module", "/test", 200, 0.1)

    # Check request count
    assert (
        REQUEST_COUNT.labels(
            module="test_module", endpoint="/test", status=200
        )._value.get()
        == 1
    )

    # Check request latency
    assert (
        REQUEST_LATENCY.labels(
            module="test_module", endpoint="/test", status=200
        )._sum.get()
        == 0.1
    )


def test_record_error():
    """Test recording error metrics."""
    # Reset metrics
    reset_metrics()

    # Record an error
    record_error("test_module", "test_error")

    # Check error count
    assert (
        ERROR_COUNT.labels(module="test_module", error_type="test_error")._value.get()
        == 1
    )


def test_record_request_multiple():
    """Test recording multiple requests."""
    # Reset metrics
    reset_metrics()

    # Record multiple requests
    for _ in range(5):
        record_request("test_module", "/test", 200, 0.1)

    # Check request count
    assert (
        REQUEST_COUNT.labels(
            module="test_module", endpoint="/test", status=200
        )._value.get()
        == 5
    )

    # Check request latency sum
    assert (
        REQUEST_LATENCY.labels(
            module="test_module", endpoint="/test", status=200
        )._sum.get()
        == 0.5
    )


def test_record_error_multiple():
    """Test recording multiple errors."""
    # Reset metrics
    reset_metrics()

    # Record multiple errors
    for _ in range(3):
        record_error("test_module", "test_error")

    # Check error count
    assert (
        ERROR_COUNT.labels(module="test_module", error_type="test_error")._value.get()
        == 3
    )


def test_monitoring_disabled(mock_config):
    """Test monitoring setup when disabled."""
    mock_config.enabled = False
    with patch("geo_infer_ops.core.monitoring.get_config") as mock_get_config:
        mock_get_config.return_value.monitoring = mock_config
        with patch("prometheus_client.start_http_server") as mock_start_server:
            setup_monitoring()
            mock_start_server.assert_not_called()


def test_setup_monitoring():
    """Test monitoring setup."""
    with patch("prometheus_client.start_http_server") as mock_start_server:
        setup_monitoring(port=9090)
        mock_start_server.assert_called_once_with(9090, registry=REGISTRY)


def test_start_metrics_server_yields_port_and_cleans_up():
    """Test metrics context manager exposes the selected port and closes handles."""
    server = MagicMock()
    server.server_port = 9093
    thread = MagicMock()
    thread.is_alive.return_value = True

    with patch(
        "geo_infer_ops.core.monitoring.prom.start_http_server",
        return_value=(server, thread),
    ) as mock_start_server:
        with start_metrics_server(port=9093) as selected_port:
            assert selected_port == 9093

    mock_start_server.assert_called_once_with(9093, registry=REGISTRY)
    server.shutdown.assert_called_once()
    server.server_close.assert_called_once()
    thread.join.assert_called_once_with(timeout=1)


def test_reset_metrics():
    """Test metrics reset."""
    # Record some test metrics
    record_request("test_module", "/test", 200, 0.1)
    record_error("test_module", "test_error")

    # Reset metrics
    reset_metrics()

    # Verify metrics are reset
    assert (
        REQUEST_COUNT.labels(
            module="test_module", endpoint="/test", status=200
        )._value.get()
        == 0
    )

    assert (
        ERROR_COUNT.labels(module="test_module", error_type="test_error")._value.get()
        == 0
    )


def test_record_metric():
    """Test metric recording."""
    # Reset metrics
    reset_metrics()

    # Test counter
    record_metric("test_counter", 1, metric_type="counter")
    assert get_metric_value("test_counter_total") == 1

    # Test gauge
    record_metric("test_gauge", 42, metric_type="gauge")
    assert get_metric_value("test_gauge") == 42

    # Test histogram
    record_metric("test_histogram", 1.5, metric_type="histogram")
    assert get_metric_value("test_histogram_sum") == 1.5


def test_record_metric_with_labels():
    """Test metric recording with labels."""
    # Reset metrics
    reset_metrics()

    # Test counter with labels
    record_metric("test_counter", 1, metric_type="counter", labels={"label": "value"})
    assert get_metric_value("test_counter_total", {"label": "value"}) == 1

    # Test gauge with labels
    record_metric("test_gauge", 42, metric_type="gauge", labels={"label": "value"})
    assert get_metric_value("test_gauge", {"label": "value"}) == 42

    # Test histogram with labels
    record_metric(
        "test_histogram", 1.5, metric_type="histogram", labels={"label": "value"}
    )
    assert get_metric_value("test_histogram_sum", {"label": "value"}) == 1.5


def test_get_metric_value():
    """Test metric value retrieval."""
    # Reset metrics
    reset_metrics()

    # Setup test metric
    record_metric("test_counter", 42, metric_type="counter")

    # Test value retrieval
    assert get_metric_value("test_counter_total") == 42

    # Test with labels
    record_metric("test_counter", 1, metric_type="counter", labels={"label": "value"})
    assert get_metric_value("test_counter_total", {"label": "value"}) == 1

    # Test non-existent metric
    with pytest.raises(ValueError):
        get_metric_value("non_existent_metric")


def test_instrument_app():
    """Test FastAPI application instrumentation."""
    from fastapi import FastAPI

    app = FastAPI()

    with patch(
        "prometheus_fastapi_instrumentator.Instrumentator"
    ) as mock_instrumentator:
        mock_instance = MagicMock()
        mock_instrumentator.return_value = mock_instance

        instrument_app(app)

        mock_instrumentator.assert_called_once_with(
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/metrics"],
            env_var_name="ENABLE_METRICS",
        )
        mock_instance.instrument.assert_called_once_with(app)
        mock_instance.instrument.return_value.expose.assert_called_once_with(app)


def test_metric_types():
    """Test different metric types."""
    # Reset metrics
    reset_metrics()

    # Counter
    record_metric("test_counter", 1, metric_type="counter")
    assert get_metric_value("test_counter_total") == 1

    # Gauge
    record_metric("test_gauge", 42, metric_type="gauge")
    assert get_metric_value("test_gauge") == 42

    # Histogram
    record_metric("test_histogram", 1.5, metric_type="histogram")
    assert get_metric_value("test_histogram_sum") == 1.5

    # Invalid type
    with pytest.raises(ValueError):
        record_metric("test_invalid", 1, metric_type="invalid")


def test_metric_labels():
    """Test metric labels."""
    # Reset metrics
    reset_metrics()

    # Create labeled metric
    record_metric(
        "test_counter",
        1,
        metric_type="counter",
        labels={"label1": "value1", "label2": "value2"},
    )

    # Test value retrieval with labels
    assert (
        get_metric_value("test_counter_total", {"label1": "value1", "label2": "value2"})
        == 1
    )

    # Test different label values
    record_metric(
        "test_counter",
        2,
        metric_type="counter",
        labels={"label1": "value1", "label2": "value3"},
    )
    assert (
        get_metric_value("test_counter_total", {"label1": "value1", "label2": "value3"})
        == 2
    )

    # Test missing label
    with pytest.raises(ValueError):
        get_metric_value("test_counter_total", {"label1": "value1"})
