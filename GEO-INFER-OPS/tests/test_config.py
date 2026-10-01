"""
Tests for configuration management.
"""

import os
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from geo_infer_ops.core.config import (
    Config,
    DeploymentConfig,
    DockerConfig,
    KubernetesConfig,
    LoggingConfig,
    MonitoringConfig,
    TestingConfig as OpsTestingConfig,
    SecurityConfig,
    TLSConfig,
    AuthConfig,
    load_config,
    get_config,
    update_config,
)


@pytest.fixture
def config_dict():
    """Fixture providing a test configuration dictionary."""
    return {
        "logging": {"level": "DEBUG", "format": "json", "file": "test.log"},
        "monitoring": {
            "enabled": True,
            "metrics_port": 9091,
            "metrics_path": "/test-metrics",
        },
        "testing": {"coverage_threshold": 90.0, "parallel": False, "timeout": 60},
        "environment": "test",
    }


@pytest.fixture
def config_file(tmp_path, config_dict):
    """Fixture creating a temporary config file."""
    import yaml

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    return config_path


def test_logging_config():
    """Test LoggingConfig creation and defaults."""
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.format == "json"
    assert config.file is None

    config = LoggingConfig(level="DEBUG", format="text", file="test.log")
    assert config.level == "DEBUG"
    assert config.format == "text"
    assert config.file == "test.log"


def test_monitoring_config():
    """Test MonitoringConfig creation and defaults."""
    config = MonitoringConfig()
    assert config.enabled is True
    assert config.metrics_port == 9090
    assert config.metrics_path == "/metrics"

    config = MonitoringConfig(enabled=False, metrics_port=9091, metrics_path="/test")
    assert config.enabled is False
    assert config.metrics_port == 9091
    assert config.metrics_path == "/test"


def test_testing_config():
    """Test TestingConfig creation and defaults."""
    config = OpsTestingConfig()
    assert config.coverage_threshold == 95.0
    assert config.parallel is True
    assert config.timeout == 300

    config = OpsTestingConfig(coverage_threshold=90.0, parallel=False, timeout=60)
    assert config.coverage_threshold == 90.0
    assert config.parallel is False
    assert config.timeout == 60


def test_config_creation(config_dict):
    """Test Config creation with custom values."""
    config = Config(**config_dict)
    assert config.logging.level == "DEBUG"
    assert config.monitoring.metrics_port == 9091
    assert config.testing.coverage_threshold == 90.0
    assert config.environment == "test"


def test_load_config(config_file):
    """Test loading configuration from file."""
    config = load_config(str(config_file))
    assert config.logging.level == "DEBUG"
    assert config.monitoring.metrics_port == 9091
    assert config.testing.coverage_threshold == 90.0
    assert config.environment == "test"


def test_get_config_singleton(config_file):
    """Test get_config returns singleton instance."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_load_config_environment_variable(tmp_path, config_dict):
    """Test loading configuration from environment variable."""
    import yaml

    config_path = tmp_path / "env_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    with patch.dict(os.environ, {"GEO_INFER_OPS_CONFIG": str(config_path)}):
        config = load_config()
        assert config.logging.level == "DEBUG"
        assert config.monitoring.metrics_port == 9091
        assert config.testing.coverage_threshold == 90.0
        assert config.environment == "test"


def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        LoggingConfig(level="INVALID")

    with pytest.raises(ValueError):
        LoggingConfig(format="INVALID")

    with pytest.raises(ValueError):
        MonitoringConfig(metrics_port=-1)

    with pytest.raises(ValueError):
        OpsTestingConfig(coverage_threshold=-1)

    with pytest.raises(ValueError):
        OpsTestingConfig(timeout=-1)


def test_deployment_config_uses_top_level_models():
    """Docker/Kubernetes models are top-level classes, not DeploymentConfig attributes."""
    assert not hasattr(DeploymentConfig, "DockerConfig")
    assert not hasattr(DeploymentConfig, "KubernetesConfig")
    deployment = DeploymentConfig(
        docker=DockerConfig(registry="r", tag="t"),
        kubernetes=KubernetesConfig(namespace="ns"),
    )
    assert deployment.docker.registry == "r"
    assert deployment.kubernetes.namespace == "ns"


def test_config_defaults():
    """Test default configuration values."""
    config = Config()

    assert isinstance(config.logging, LoggingConfig)
    assert config.logging.level == "INFO"
    assert config.logging.format == "console"
    assert config.logging.file is None

    assert isinstance(config.monitoring, MonitoringConfig)
    assert config.monitoring.enabled is True
    assert config.monitoring.metrics_port == 9090

    assert isinstance(config.testing, OpsTestingConfig)
    assert config.testing.coverage_threshold == 95.0
    assert config.testing.parallel is False
    assert config.testing.timeout == 300

    assert isinstance(config.security, SecurityConfig)
    assert isinstance(config.security.tls, TLSConfig)
    assert isinstance(config.security.auth, AuthConfig)


def test_config_custom_values(mock_config_dict: dict[str, Any]):
    """Test configuration with custom values."""
    config = Config(**mock_config_dict)

    assert config.logging.level == "DEBUG"
    assert config.monitoring.enabled is True
    assert config.testing.coverage_threshold == 95.0
    assert config.security.tls.enabled is True
    assert config.security.auth.enabled is True


def test_load_config_from_file(temp_dir: str):
    """Test loading configuration from file."""
    config_path = Path(temp_dir) / "test_config.yaml"
    config_dict = {
        "logging": {"level": "DEBUG"},
        "monitoring": {"enabled": True},
        "testing": {"coverage_threshold": 90.0},
    }

    with open(config_path, "w") as f:
        import yaml

        yaml.dump(config_dict, f)

    with patch("geo_infer_ops.core.config.get_config") as mock_get_config:
        mock_get_config.return_value = Config(**config_dict)
        config = load_config(str(config_path))

        assert config.logging.level == "DEBUG"
        assert config.monitoring.enabled is True
        assert config.testing.coverage_threshold == 90.0


def test_update_config():
    """Test configuration updates."""
    config = get_config()
    old_level = config.logging.level

    update_config({"logging": {"level": "DEBUG"}})
    assert get_config().logging.level == "DEBUG"

    update_config({"logging": {"level": old_level}})
    assert get_config().logging.level == old_level


def test_security_config():
    """Test security configuration."""
    config = Config(
        security={
            "tls": {
                "enabled": True,
                "cert_file": "/path/to/cert.pem",
                "key_file": "/path/to/key.pem",
            },
            "auth": {
                "enabled": True,
                "jwt_secret": "test-secret-for-pyjwt-hs256-tests-32-bytes",
                "jwt_algorithm": "HS256",
            },
        }
    )

    assert config.security.tls.enabled is True
    assert config.security.tls.cert_file == "/path/to/cert.pem"
    assert config.security.auth.enabled is True
    assert (
        config.security.auth.jwt_secret == "test-secret-for-pyjwt-hs256-tests-32-bytes"
    )
