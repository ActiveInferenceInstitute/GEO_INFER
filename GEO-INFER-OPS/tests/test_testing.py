"""
Tests for testing utilities.
"""

import os
import pytest
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import Counter, Histogram

from geo_infer_ops.core.testing import (
    temporary_config,
    create_test_data_dir,
    setup_testing,
    create_test_client,
    assert_response_status,
    assert_response_json,
    assert_metric_value,
)
from geo_infer_ops.core.config import (
    Config,
    LoggingConfig,
    MonitoringConfig,
    TestingConfig as OpsTestingConfig,
)


@pytest.fixture
def mock_config_dict():
    """Fixture providing a mock configuration dictionary."""
    return {
        "logging": {"level": "DEBUG", "format": "console"},
        "monitoring": {"enabled": True, "metrics_port": 9090},
        "testing": {"coverage_threshold": 95, "timeout": 60, "parallel": False},
    }


@pytest.fixture
def mock_app():
    """Fixture providing a mock FastAPI application."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    return app


def test_mock_config(mock_config_dict):
    """Test configuration mocking."""
    with temporary_config(mock_config_dict) as config:
        assert isinstance(config, Config)
        assert config.logging.level == "DEBUG"
        assert config.monitoring.enabled is True
        assert config.testing.coverage_threshold == 95


def test_create_test_data_dir():
    """Test temporary directory creation."""
    prefix = "test_dir_"
    temp_dir = create_test_data_dir(prefix)

    try:
        assert os.path.exists(temp_dir)
        assert os.path.basename(temp_dir).startswith(prefix)
    finally:
        os.rmdir(temp_dir)


def test_setup_testing_defaults(mock_config_dict):
    """Test testing setup with default configuration."""
    with (
        patch("geo_infer_ops.core.testing.get_config") as mock_get_config,
        patch("pytest.main", return_value=0) as mock_pytest,
    ):
        mock_get_config.return_value = Config(
            logging=LoggingConfig(level="INFO"),
            monitoring=MonitoringConfig(enabled=True),
            testing=OpsTestingConfig(coverage_threshold=95, parallel=False, timeout=60),
        )

        exit_code = setup_testing()

        assert exit_code == 0
        args = mock_pytest.call_args[0][0]
        assert "--cov=geo_infer_ops" in args
        assert any(arg.startswith("--cov-fail-under=95") for arg in args)
        assert "--timeout=60" in args


def test_setup_testing_custom_config(mock_config_dict):
    """Test testing setup with custom configuration."""
    with (
        patch("geo_infer_ops.core.testing.get_config") as mock_get_config,
        patch("pytest.main", return_value=0) as mock_pytest,
    ):
        mock_get_config.return_value = Config(
            logging=LoggingConfig(level="DEBUG"),
            monitoring=MonitoringConfig(enabled=True),
            testing=OpsTestingConfig(coverage_threshold=95, parallel=False, timeout=60),
        )

        exit_code = setup_testing(
            test_dir="custom_tests",
            coverage_report=False,
            parallel=True,
            timeout=120,
            log_level="DEBUG",
            json_format=True,
        )

        assert exit_code == 0
        args = mock_pytest.call_args[0][0]
        assert "custom_tests" in args
        assert "--cov=geo_infer_ops" not in args
        assert "-n=auto" in args
        assert "--timeout=120" in args


def test_setup_testing_failure():
    """Test testing setup with test failure."""
    with (
        patch("geo_infer_ops.core.testing.get_config") as mock_get_config,
        patch("pytest.main", return_value=1),
    ):
        mock_get_config.return_value = Config(
            logging=LoggingConfig(level="INFO"),
            monitoring=MonitoringConfig(enabled=True),
            testing=OpsTestingConfig(coverage_threshold=95, parallel=False, timeout=60),
        )

        # Test without exit_on_failure
        exit_code = setup_testing()
        assert exit_code == 1

        # Test with exit_on_failure
        with pytest.raises(SystemExit) as exc_info:
            setup_testing(exit_on_failure=True)
        assert exc_info.value.code == 1


def test_create_test_client(mock_app):
    """Test test client creation."""
    client = create_test_client(mock_app)
    assert isinstance(client, TestClient)

    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "test"}


def test_assert_response_status():
    """Test response status assertion."""

    class MockResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    # Test successful assertion
    response = MockResponse(200)
    assert_response_status(response, 200)

    # Test failed assertion
    with pytest.raises(AssertionError):
        assert_response_status(response, 404)


def test_assert_response_json():
    """Test response JSON assertion."""

    class MockResponse:
        def __init__(self, data: dict[str, Any]):
            self._data = data

        def json(self):
            return self._data

    # Test successful assertion
    response = MockResponse({"key": "value"})
    assert_response_json(response, {"key": "value"})

    # Test failed assertion
    with pytest.raises(AssertionError):
        assert_response_json(response, {"key": "wrong"})


def test_assert_metric_value():
    """Test metric value assertion."""
    # Create test metrics
    counter = Counter("testing_assert_counter", "Test counter")
    counter.inc(5)

    histogram = Histogram(
        "testing_assert_histogram",
        "Test histogram",
        ["label"],
        buckets=[0.1, 1.0, 10.0],
    )
    histogram.labels(label="test").observe(2.5)

    # Test counter assertion
    assert_metric_value("testing_assert_counter_total", 5)
    with pytest.raises(AssertionError):
        assert_metric_value("testing_assert_counter_total", 10)

    # Test histogram assertion with labels
    assert_metric_value("testing_assert_histogram_sum", 2.5, {"label": "test"})
    with pytest.raises(AssertionError):
        assert_metric_value("testing_assert_histogram_sum", 5.0, {"label": "test"})
