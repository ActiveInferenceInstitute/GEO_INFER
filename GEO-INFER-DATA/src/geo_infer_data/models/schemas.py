"""
Pydantic data models for GEO-INFER-DATA validation and serialization.

This module defines comprehensive data models for geospatial datasets, metadata,
quality reports, and ETL processes using Pydantic for runtime validation.
"""

from typing import Any
from datetime import datetime, UTC
from enum import StrEnum
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pydantic import computed_field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class DataType(StrEnum):
    """Supported data types for datasets."""

    VECTOR = "vector"
    RASTER = "raster"
    POINT_CLOUD = "point_cloud"
    TIME_SERIES = "time_series"
    NETWORK = "network"
    TABULAR = "tabular"


class DataFormat(StrEnum):
    """Supported data formats."""

    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    GEOPACKAGE = "geopackage"
    GEOTIFF = "geotiff"
    NETCDF = "netcdf"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    ZARR = "zarr"
    KML = "kml"
    WKT = "wkt"
    WKB = "wkb"


class QualityStatus(StrEnum):
    """Quality check status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class ExecutionState(StrEnum):
    """ETL execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StorageBackend(StrEnum):
    """Available storage backends."""

    POSTGRESQL = "postgresql"
    MINIO = "minio"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    LOCAL = "local"


class CoordinateReferenceSystem(BaseModel):
    """Coordinate reference system information."""

    epsg_code: str | None = Field(default="EPSG:4326", description="EPSG code")
    proj_string: str | None = Field(default=None, description="PROJ string")
    wkt: str | None = Field(default=None, description="Well-known text representation")

    model_config = ConfigDict(validate_assignment=True)


class SpatialExtent(BaseModel):
    """Geographic extent of a dataset."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=6,
        description="Bounding box coordinates [min_lon, min_lat, max_lon, max_lat] or with elevation",
    )
    crs: str | dict[str, Any] | CoordinateReferenceSystem = Field(
        default="EPSG:4326", description="Coordinate reference system"
    )
    geometry_wkt: str | None = Field(default=None, description="Geometry in WKT format")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        """Validate bounding box coordinates."""
        if len(v) == 4:
            # [min_lon, min_lat, max_lon, max_lat]
            min_lon, min_lat, max_lon, max_lat = v
            if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if min_lon >= max_lon:
                raise ValueError("min_lon must be less than max_lon")
            if min_lat >= max_lat:
                raise ValueError("min_lat must be less than max_lat")
        elif len(v) == 6:
            # [min_lon, min_lat, min_elev, max_lon, max_lat, max_elev]
            min_lon, min_lat, min_elev, max_lon, max_lat, max_elev = v
            if min_elev >= max_elev:
                raise ValueError("min_elev must be less than max_elev")
        else:
            raise ValueError("Bounding box must have 4 or 6 coordinates")
        return v

    @field_validator("crs")
    @classmethod
    def validate_crs(cls, v: Any) -> Any:
        """Validate coordinate reference system."""
        if isinstance(v, str):
            # Handle string CRS like 'EPSG:4326'
            return CoordinateReferenceSystem(epsg_code=v)
        elif isinstance(v, dict):
            # Handle dictionary CRS
            return CoordinateReferenceSystem(**v)
        elif isinstance(v, CoordinateReferenceSystem):
            # Already a CoordinateReferenceSystem
            return v
        else:
            raise ValueError(
                "CRS must be a string, dictionary, or CoordinateReferenceSystem instance"
            )

    model_config = ConfigDict(validate_assignment=True)


class TemporalExtent(BaseModel):
    """Temporal extent of a dataset."""

    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")
    resolution: str | None = Field(
        default=None, description="Temporal resolution (ISO 8601 duration)"
    )

    @model_validator(mode="after")
    def validate_temporal_order(self) -> "TemporalExtent":
        """Ensure start time is before end time."""
        if self.start and self.end and self.start > self.end:
            raise ValueError("Start time must be before or equal to end time")
        return self

    model_config = ConfigDict(validate_assignment=True)


class DataLineage(BaseModel):
    """Data provenance and transformation history."""

    source: str = Field(..., description="Original data source")
    process: str = Field(..., description="Processing steps applied")
    created_by: str = Field(..., description="Entity that created this version")
    created_at: datetime = Field(default_factory=utc_now)
    parent_datasets: list[str] = Field(
        default_factory=list, description="Parent dataset IDs"
    )
    transformations: list[str] = Field(
        default_factory=list, description="Applied transformations"
    )

    model_config = ConfigDict(validate_assignment=True)


class QualityCheck(BaseModel):
    """Individual quality check result."""

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality score between 0 and 1"
    )
    status: QualityStatus = Field(..., description="Quality check status")
    issues: list[dict[str, Any]] = Field(
        default_factory=list, description="List of identified issues"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the check"
    )

    model_config = ConfigDict(validate_assignment=True)


class DatasetMetadata(BaseModel):
    """Comprehensive metadata for a geospatial dataset."""

    title: str = Field(..., description="Dataset title")
    description: str | None = Field(default=None, description="Dataset description")
    keywords: list[str] = Field(default_factory=list, description="Search keywords")
    spatial: SpatialExtent | None = Field(
        default=None, description="Spatial extent and CRS"
    )
    temporal: TemporalExtent | None = Field(
        default=None, description="Temporal extent and resolution"
    )
    lineage: DataLineage = Field(..., description="Data provenance information")
    quality: dict[str, QualityCheck] = Field(
        default_factory=dict, description="Quality check results by category"
    )
    contact: dict[str, str] = Field(
        default_factory=dict, description="Contact information"
    )
    license: str | None = Field(default=None, description="Data license")
    rights: str | None = Field(default=None, description="Usage rights")
    version: str = Field(default="1.0.0", description="Dataset version")
    checksum: str | None = Field(default=None, description="Data checksum")
    file_size: int | None = Field(default=None, description="File size in bytes")

    model_config = ConfigDict(validate_assignment=True)


class DatasetSummary(BaseModel):
    """Summary information for a dataset."""

    id: str = Field(..., description="Unique dataset identifier")
    title: str = Field(..., description="Dataset title")
    description: str | None = Field(default=None, description="Dataset description")
    type: DataType = Field(..., description="Dataset type")
    format: DataFormat = Field(..., description="Data format")
    size: int | None = Field(default=None, description="Size in bytes")
    bbox: list[float] = Field(
        default_factory=list,
        min_length=4,
        max_length=4,
        description="Spatial bounding box",
    )
    temporal_extent: TemporalExtent | None = Field(
        default=None, description="Temporal extent"
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overall quality score"
    )

    model_config = ConfigDict(validate_assignment=True)


class Dataset(BaseModel):
    """Complete dataset representation."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique dataset identifier",
    )
    title: str = Field(..., description="Dataset title")
    description: str | None = Field(default=None, description="Dataset description")
    type: DataType = Field(..., description="Dataset type")
    format: DataFormat = Field(..., description="Data format")
    metadata: DatasetMetadata = Field(..., description="Complete dataset metadata")
    storage_backend: StorageBackend = Field(
        default=StorageBackend.POSTGRESQL, description="Storage backend"
    )
    access_url: str | None = Field(default=None, description="Data access URL")
    download_url: str | None = Field(default=None, description="Download URL")
    permissions: dict[str, list[str]] = Field(
        default_factory=dict, description="Access permissions by role"
    )
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    version: str = Field(default="1.0.0", description="Dataset version")
    is_active: bool = Field(default=True, description="Whether dataset is active")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(validate_assignment=True)


class DataQualityReport(BaseModel):
    """Comprehensive data quality assessment report."""

    dataset_id: str = Field(..., description="Dataset being assessed")
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall quality score"
    )
    checks: dict[str, QualityCheck] = Field(
        ..., description="Quality checks by category"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    generated_at: datetime = Field(default_factory=utc_now)
    assessment_method: str | list[str] = Field(
        default="comprehensive", description="Assessment methodology used"
    )
    validation_rules: list[str] = Field(
        default_factory=list, description="Validation rules applied"
    )

    @computed_field
    def status(self) -> QualityStatus:
        """Overall pass/warning/fail status derived from the quality score."""
        if self.overall_score >= 0.8:
            return QualityStatus.PASS
        if self.overall_score >= 0.5:
            return QualityStatus.WARNING
        return QualityStatus.FAIL

    @computed_field
    def issues(self) -> list[dict[str, Any]]:
        """Flattened issues across all quality checks."""
        return [issue for check in self.checks.values() for issue in check.issues]

    model_config = ConfigDict(validate_assignment=True)


class DataSource(BaseModel):
    """Data source configuration for ETL pipelines."""

    type: str = Field(..., description="Source type (file, database, api, stream)")
    configuration: dict[str, Any] = Field(
        ..., description="Source-specific configuration"
    )
    credentials: dict[str, str] | None = Field(
        default=None, description="Authentication credentials"
    )
    format: DataFormat | None = Field(default=None, description="Data format")
    data_schema: dict[str, Any] | None = Field(
        default=None, alias="schema", description="Data schema"
    )

    model_config = ConfigDict(validate_assignment=True, populate_by_name=True)


class DataDestination(BaseModel):
    """Data destination configuration for ETL pipelines."""

    type: str = Field(
        ..., description="Destination type (dataset, database, file, api)"
    )
    configuration: dict[str, Any] = Field(
        ..., description="Destination-specific configuration"
    )
    format: DataFormat | None = Field(default=None, description="Output format")
    compression: str | None = Field(default=None, description="Compression method")

    model_config = ConfigDict(validate_assignment=True)


class Transformation(BaseModel):
    """Transformation step in ETL pipeline."""

    type: str = Field(..., description="Transformation type")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Transformation parameters"
    )
    order: int = Field(default=0, description="Execution order")
    enabled: bool = Field(default=True, description="Whether transformation is enabled")

    model_config = ConfigDict(validate_assignment=True)


class ETLPipeline(BaseModel):
    """ETL pipeline configuration."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Pipeline identifier"
    )
    name: str = Field(..., description="Pipeline name")
    description: str | None = Field(default=None, description="Pipeline description")
    source: DataSource = Field(..., description="Data source configuration")
    destination: DataDestination = Field(
        ..., description="Data destination configuration"
    )
    transformations: list[Transformation] = Field(
        default_factory=list, description="Transformation steps"
    )
    schedule: dict[str, str] | None = Field(
        default=None, description="Scheduling configuration"
    )
    status: str = Field(default="inactive", description="Pipeline status")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("transformations")
    @classmethod
    def sort_transformations(cls, v: list[Transformation]) -> list[Transformation]:
        """Sort transformations by execution order."""
        return sorted(v, key=lambda x: x.order)

    model_config = ConfigDict(validate_assignment=True)


class ExecutionStatus(BaseModel):
    """ETL execution status and progress."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Execution identifier"
    )
    pipeline_id: str = Field(..., description="Associated pipeline ID")
    status: ExecutionState = Field(..., description="Current execution state")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    message: str | None = Field(default=None, description="Status message")
    started_at: datetime | None = Field(default=None, description="Start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")
    created_at: datetime = Field(default_factory=utc_now)
    logs: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution logs"
    )

    model_config = ConfigDict(validate_assignment=True)


class Pagination(BaseModel):
    """Pagination information for API responses."""

    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    has_next: bool = Field(default=False, description="Whether there is a next page")
    has_prev: bool = Field(
        default=False, description="Whether there is a previous page"
    )


class HealthStatus(BaseModel):
    """System health status."""

    status: str = Field(..., description="Health status")
    message: str | None = Field(default=None, description="Health message")
    checked_at: datetime = Field(default_factory=utc_now)
    components: dict[str, Any] = Field(
        default_factory=dict, description="Component health details"
    )
