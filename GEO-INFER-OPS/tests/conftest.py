"""
Common test fixtures and configuration for GEO-INFER-OPS.
"""

import os
import tempfile
import pytest
import structlog
from pathlib import Path
from typing import Any
from collections.abc import Generator
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from geo_infer_ops.core.config import Config
from geo_infer_ops.utils.shared_logging import configure_logging
from geo_infer_ops.core.monitoring import reset_metrics


@pytest.fixture(scope="session")
def test_dir() -> str:
    """Get the test directory path."""
    return str(Path(__file__).parent)


@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="session")
def mock_config_dict() -> dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "logging": {"level": "DEBUG", "format": "console", "file": None},
        "monitoring": {
            "enabled": True,
            "metrics_port": 9090,
            "metrics_path": "/metrics",
        },
        "testing": {"coverage_threshold": 95.0, "parallel": False, "timeout": 300},
        "security": {
            "tls": {"enabled": True, "cert_file": None, "key_file": None},
            "auth": {
                "enabled": True,
                "jwt_secret": "test-secret-for-pyjwt-hs256-tests-32-bytes",
                "jwt_algorithm": "HS256",
            },
        },
    }


@pytest.fixture(scope="session")
def config(mock_config_dict: dict[str, Any]) -> Config:
    """Create a test configuration instance."""
    return Config(**mock_config_dict)


@pytest.fixture(scope="session")
def mock_app() -> FastAPI:
    """Create a mock FastAPI application."""
    app = FastAPI(title="Test API")

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    return app


@pytest.fixture(scope="session")
def test_client(mock_app: FastAPI) -> TestClient:
    """Create a test client for the mock FastAPI application."""
    return TestClient(mock_app)


@pytest.fixture(scope="session")
def test_registry() -> CollectorRegistry:
    """Create a test Prometheus registry."""
    registry = CollectorRegistry()
    reset_metrics()
    return registry


@pytest.fixture(scope="session")
def mock_redis():
    """Create a mock Redis client."""
    mock = MagicMock()
    mock.pipeline.return_value.__enter__.return_value = MagicMock()
    return mock


@pytest.fixture(scope="session")
def test_logger():
    """Configure test logging."""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        log_file = f.name
    configure_logging(log_level="DEBUG", json_format=False, log_file=log_file)
    yield structlog.get_logger()
    os.unlink(log_file)
