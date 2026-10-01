"""Configuration management module."""

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="json", description="Log format (console, json, or text)"
    )
    file: str | None = Field(default=None, description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = ["console", "json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid log format. Must be one of {valid_formats}")
        return v.lower()


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

    @field_validator("metrics_port")
    @classmethod
    def validate_metrics_port(cls, v: int) -> int:
        """Validate metrics port."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class TestingConfig(BaseModel):
    """Testing configuration."""

    enabled: bool = Field(default=True, description="Enable testing")
    parallel: bool = Field(default=True, description="Enable parallel test execution")
    coverage_threshold: float = Field(
        default=95.0, description="Minimum test coverage threshold"
    )
    timeout: int = Field(default=300, description="Test timeout in seconds")

    @field_validator("coverage_threshold")
    @classmethod
    def validate_coverage_threshold(cls, v: float) -> float:
        """Validate coverage threshold."""
        if not 0 <= v <= 100:
            raise ValueError("Coverage threshold must be between 0 and 100")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout value."""
        if v < 1:
            raise ValueError("Timeout must be positive")
        return v


class DockerConfig(BaseModel):
    """Docker configuration."""

    registry: str = Field(default="localhost", description="Docker registry URL")
    tag: str = Field(default="latest", description="Docker image tag")
    username: str | None = Field(default=None, description="Registry username")
    password: str | None = Field(default=None, description="Registry password")
    timeout: int = Field(default=300, description="Docker operation timeout in seconds")
    build_timeout: int = Field(
        default=1800, description="Docker build subprocess timeout in seconds"
    )
    push_timeout: int = Field(
        default=900, description="Docker push subprocess timeout in seconds"
    )

    @field_validator("timeout", "build_timeout", "push_timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout value."""
        if v < 1:
            raise ValueError("Timeout must be positive")
        return v


class KubernetesConfig(BaseModel):
    """Kubernetes configuration."""

    context: str = Field(default="default", description="Kubernetes context")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    timeout: int = Field(
        default=300, description="Kubernetes operation timeout in seconds"
    )

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout value."""
        if v < 1:
            raise ValueError("Timeout must be positive")
        return v


class DeploymentConfig(BaseModel):
    """Deployment configuration."""

    replicas: int = Field(default=1, description="Number of replicas")
    resource_limits: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "500m", "memory": "512Mi"},
        description="Resource limits",
    )
    resource_requests: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "250m", "memory": "256Mi"},
        description="Resource requests",
    )
    timeout: int = Field(default=300, description="Deployment timeout in seconds")
    docker: DockerConfig = Field(
        default_factory=DockerConfig, description="Docker configuration"
    )
    kubernetes: KubernetesConfig = Field(
        default_factory=KubernetesConfig, description="Kubernetes configuration"
    )

    @field_validator("replicas")
    @classmethod
    def validate_replicas(cls, v: int) -> int:
        """Validate replicas count."""
        if v < 1:
            raise ValueError("Replicas must be at least 1")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout value."""
        if v < 1:
            raise ValueError("Timeout must be at least 1 second")
        return v


class TLSConfig(BaseModel):
    """TLS configuration."""

    enabled: bool = Field(default=True, description="Enable TLS")
    cert_file: str | None = Field(default=None, description="Certificate file path")
    key_file: str | None = Field(default=None, description="Private key file path")
    ca_file: str | None = Field(default=None, description="CA certificate file path")

    @field_validator("cert_file", "key_file", "ca_file")
    @classmethod
    def validate_file_paths(cls, v: Any, info: Any) -> Any:
        """Validate file path values without requiring files at config load time."""
        return v


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = Field(default=True, description="Enable authentication")
    jwt_secret: str | None = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    token_expiry: int = Field(default=3600, description="Token expiry in seconds")

    @field_validator("token_expiry")
    @classmethod
    def validate_token_expiry(cls, v: int) -> int:
        """Validate token expiry."""
        if v < 1:
            raise ValueError("Token expiry must be positive")
        return v


class SecurityConfig(BaseModel):
    """Security configuration."""

    enabled: bool = Field(default=True, description="Enable security features")
    tls: TLSConfig = Field(default_factory=TLSConfig, description="TLS configuration")
    auth: AuthConfig = Field(
        default_factory=AuthConfig, description="Authentication configuration"
    )


class RedisConfig(BaseModel):
    """Redis connection settings used by :class:`~geo_infer_ops.core.cache.CacheManager`."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database index")
    password: str | None = Field(default=None, description="Redis password")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("db")
    @classmethod
    def validate_db(cls, v: int) -> int:
        """Validate database index."""
        if v < 0:
            raise ValueError("Database index must be non-negative")
        return v


class CacheConfig(BaseModel):
    """Caching configuration."""

    enabled: bool = Field(default=True, description="Enable caching")
    type: str = Field(default="redis", description="Cache backend type")
    redis: RedisConfig = Field(
        default_factory=RedisConfig, description="Redis connection settings"
    )


class Config(BaseModel):
    """Main configuration."""

    environment: str = Field(default="development", description="Environment name")
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(format="console"),
        description="Logging configuration",
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    testing: TestingConfig = Field(
        default_factory=lambda: TestingConfig(parallel=False),
        description="Testing configuration",
    )
    docker: DockerConfig = Field(
        default_factory=DockerConfig, description="Docker configuration"
    )
    kubernetes: KubernetesConfig = Field(
        default_factory=KubernetesConfig, description="Kubernetes configuration"
    )
    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig, description="Deployment configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache configuration"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        valid_environments = ["development", "testing", "test", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(
                f"Invalid environment. Must be one of {valid_environments}"
            )
        return v.lower()


# Global configuration instance
_config: Config | None = None

# Backward-compatible nested aliases used by older callers and tests.
DeploymentConfig.DockerConfig = DockerConfig  # type: ignore[attr-defined]
DeploymentConfig.KubernetesConfig = KubernetesConfig  # type: ignore[attr-defined]


def load_config(config_file: str | None = None) -> Config:
    """Load configuration from file or environment variables."""
    global _config

    if not config_file:
        config_file = os.environ.get("GEO_INFER_OPS_CONFIG")

    if _config is not None and not config_file:
        return _config

    # Load from environment variables if available
    env_config = {}
    for key, value in os.environ.items():
        if key.startswith("GEO_INFER_"):
            env_key = key[10:].lower().replace("_", ".")
            env_config[env_key] = value

    # Load from file if provided
    file_config: dict[str, Any] = {}
    if config_file and os.path.exists(config_file):
        import yaml

        with open(config_file) as f:
            file_config = yaml.safe_load(f) or {}

    # Merge configurations
    config_dict = {**file_config, **env_config}

    # Create configuration instance
    _config = Config(**config_dict)
    return _config


def get_config() -> Config:
    """Get the current configuration instance."""
    if _config is None:
        return load_config()
    return _config


def update_config(config_dict: dict[str, Any]) -> Config:
    """Update configuration with new values.

    Args:
        config_dict: Dictionary of configuration updates

    Returns:
        Config: Updated configuration instance
    """
    global _config
    if _config is None:
        _config = Config()

    # Create a new config with the updated values
    updated_dict = _config.model_dump()
    for key, value in config_dict.items():
        if isinstance(value, dict) and key in updated_dict:
            updated_dict[key].update(value)
        else:
            updated_dict[key] = value

    _config = Config(**updated_dict)
    return _config
