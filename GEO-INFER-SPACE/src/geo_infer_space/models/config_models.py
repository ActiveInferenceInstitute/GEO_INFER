"""
Configuration models for GEO-INFER-SPACE.

This module defines Pydantic models for configuration schemas
including analysis parameters, indexing settings, API configuration,
and database connections.
"""

from typing import Any
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatabaseConfig(BaseModel):
    """Configuration for spatial database connections."""

    model_config = ConfigDict(populate_by_name=True)

    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    schema_name: str = Field("public", alias="schema", description="Database schema")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Maximum pool overflow")

    @field_validator("port")
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("pool_size")
    def validate_pool_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Pool size must be at least 1")
        return v

    def get_connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class IndexingConfig(BaseModel):
    """Configuration for spatial indexing systems."""

    default_index_type: str = Field("rtree", description="Default spatial index type")
    h3_default_resolution: int = Field(
        8, ge=0, le=15, description="Default H3 resolution"
    )
    rtree_leaf_capacity: int = Field(10, description="R-tree leaf node capacity")
    quadtree_max_depth: int = Field(20, description="QuadTree maximum depth")
    geohash_precision: int = Field(12, description="Geohash precision")
    enable_caching: bool = Field(True, description="Enable index caching")
    cache_size_mb: int = Field(100, description="Index cache size in MB")

    @field_validator("default_index_type")
    def validate_index_type(cls, v: str) -> str:
        valid_types = ["rtree", "quadtree", "h3", "geohash", "s2"]
        if v.lower() not in valid_types:
            raise ValueError(f"Index type must be one of {valid_types}")
        return v.lower()

    @field_validator("cache_size_mb")
    def validate_cache_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Cache size must be at least 1 MB")
        return v


class AnalysisConfig(BaseModel):
    """Configuration for spatial analysis operations."""

    # Vector analysis settings
    buffer_resolution: int = Field(
        16, description="Buffer resolution (segments per quarter circle)"
    )
    overlay_precision: float = Field(
        1e-6, description="Geometric precision for overlay operations"
    )
    simplify_tolerance: float = Field(
        0.0, description="Default simplification tolerance"
    )

    # Raster analysis settings
    raster_chunk_size: int = Field(1024, description="Raster processing chunk size")
    nodata_value: float = Field(-9999.0, description="Default NoData value")
    resampling_method: str = Field("bilinear", description="Default resampling method")

    # Network analysis settings
    network_tolerance: float = Field(1e-3, description="Network topology tolerance")
    max_routing_distance: float = Field(
        100000.0, description="Maximum routing distance"
    )

    # Geostatistics settings
    interpolation_method: str = Field("idw", description="Default interpolation method")
    idw_power: float = Field(2.0, description="IDW power parameter")
    kriging_variogram: str = Field("spherical", description="Kriging variogram model")

    # Performance settings
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_workers: int | None = Field(None, description="Maximum worker threads")
    memory_limit_gb: float | None = Field(None, description="Memory limit in GB")

    @field_validator("buffer_resolution")
    def validate_buffer_resolution(cls, v: int) -> int:
        if v < 4:
            raise ValueError("Buffer resolution must be at least 4")
        return v

    @field_validator("raster_chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        if v < 64 or v > 8192:
            raise ValueError("Raster chunk size must be between 64 and 8192")
        return v

    @field_validator("resampling_method")
    def validate_resampling_method(cls, v: str) -> str:
        valid_methods = ["nearest", "bilinear", "cubic", "average", "mode"]
        if v.lower() not in valid_methods:
            raise ValueError(f"Resampling method must be one of {valid_methods}")
        return v.lower()

    @field_validator("interpolation_method")
    def validate_interpolation_method(cls, v: str) -> str:
        valid_methods = ["idw", "kriging", "rbf", "nearest", "linear"]
        if v.lower() not in valid_methods:
            raise ValueError(f"Interpolation method must be one of {valid_methods}")
        return v.lower()


class APIConfig(BaseModel):
    """Configuration for REST API server."""

    host: str = Field("0.0.0.0", description="API server host")
    port: int = Field(8000, description="API server port")
    workers: int = Field(1, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload in development")

    # CORS settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins"
    )
    cors_methods: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS methods"
    )
    cors_headers: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS headers"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(False, description="Enable rate limiting")
    rate_limit_requests: int = Field(100, description="Requests per minute limit")

    # Authentication
    auth_enabled: bool = Field(False, description="Enable authentication")
    jwt_secret: str | None = Field(None, description="JWT secret key")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(24, description="JWT expiration time in hours")

    # Request limits
    max_request_size_mb: int = Field(100, description="Maximum request size in MB")
    request_timeout_seconds: int = Field(300, description="Request timeout in seconds")

    @field_validator("port")
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("workers")
    def validate_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Number of workers must be at least 1")
        return v

    @field_validator("max_request_size_mb")
    def validate_request_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Maximum request size must be at least 1 MB")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging system."""

    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file_enabled: bool = Field(True, description="Enable file logging")
    file_path: Path | None = Field(None, description="Log file path")
    file_max_size_mb: int = Field(10, description="Maximum log file size in MB")
    file_backup_count: int = Field(5, description="Number of backup log files")
    console_enabled: bool = Field(True, description="Enable console logging")

    @field_validator("level")
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("file_max_size_mb")
    def validate_file_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Log file size must be at least 1 MB")
        return v


class CacheConfig(BaseModel):
    """Configuration for caching system."""

    enabled: bool = Field(True, description="Enable caching")
    backend: str = Field("memory", description="Cache backend type")
    ttl_seconds: int = Field(3600, description="Default TTL in seconds")
    max_size_mb: int = Field(100, description="Maximum cache size in MB")

    # Redis settings (if backend is redis)
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_db: int = Field(0, description="Redis database number")
    redis_password: str | None = Field(None, description="Redis password")

    @field_validator("backend")
    def validate_backend(cls, v: str) -> str:
        valid_backends = ["memory", "redis", "memcached"]
        if v.lower() not in valid_backends:
            raise ValueError(f"Cache backend must be one of {valid_backends}")
        return v.lower()

    @field_validator("ttl_seconds")
    def validate_ttl(cls, v: int) -> int:
        if v < 1:
            raise ValueError("TTL must be at least 1 second")
        return v


class SpaceConfig(BaseModel):
    """Main configuration model for GEO-INFER-SPACE."""

    # Core settings
    debug: bool = Field(False, description="Enable debug mode")
    environment: str = Field("production", description="Environment name")
    data_directory: Path = Field(Path("./data"), description="Data directory path")
    temp_directory: Path = Field(
        Path("./temp"), description="Temporary files directory"
    )

    # Component configurations
    database: DatabaseConfig | None = Field(None, description="Database configuration")
    indexing: IndexingConfig = Field(
        default_factory=IndexingConfig,  # type: ignore[arg-type]
        description="Indexing configuration",
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig,  # type: ignore[arg-type]
        description="Analysis configuration",
    )
    api: APIConfig = Field(
        default_factory=APIConfig,  # type: ignore[arg-type]
        description="API configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,  # type: ignore[arg-type]
        description="Logging configuration",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,  # type: ignore[arg-type]
        description="Cache configuration",
    )
    # Custom settings
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Custom configuration parameters"
    )

    @field_validator("environment")
    def validate_environment(cls, v: str) -> str:
        valid_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v.lower()

    @model_validator(mode="after")
    def validate_directories(self) -> "SpaceConfig":
        """Ensure directories exist or can be created."""
        for dir_field in ["data_directory", "temp_directory"]:
            directory_value = getattr(self, dir_field)
            if directory_value:
                directory = Path(directory_value)
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create directory {directory}: {e}") from e
        return self

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SpaceConfig":
        """Load configuration from YAML or JSON file."""
        import yaml
        import json

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        return cls(**config_data)

    def to_file(self, config_path: str | Path, format: str = "yaml") -> None:
        """Save configuration to file."""
        import yaml
        import json

        config_path = Path(config_path)
        config_data = self.model_dump(by_alias=True)

        # Convert Path objects to strings for serialization
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        config_data = convert_paths(config_data)

        with open(config_path, "w") as f:
            if format.lower() == "yaml":
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""

    # Memory management
    memory_limit_gb: float | None = Field(None, description="Memory limit in GB")
    gc_threshold: int = Field(1000, description="Garbage collection threshold")

    # Parallel processing
    max_workers: int | None = Field(None, description="Maximum worker threads")
    chunk_size: int = Field(1000, description="Processing chunk size")
    use_multiprocessing: bool = Field(True, description="Enable multiprocessing")

    # I/O optimization
    io_buffer_size: int = Field(65536, description="I/O buffer size in bytes")
    async_io: bool = Field(True, description="Enable asynchronous I/O")

    # Spatial operations
    spatial_precision: float = Field(
        1e-6, description="Spatial precision for operations"
    )
    simplify_threshold: float = Field(
        0.0, description="Geometry simplification threshold"
    )

    @field_validator("chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Chunk size must be at least 1")
        return v

    @field_validator("io_buffer_size")
    def validate_buffer_size(cls, v: int) -> int:
        if v < 1024:
            raise ValueError("I/O buffer size must be at least 1024 bytes")
        return v
