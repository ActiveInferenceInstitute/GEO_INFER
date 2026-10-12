"""
Pydantic schemas for API request/response validation.

This module defines data models for spatial analysis API endpoints
using Pydantic for automatic validation and documentation generation.
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator
from geojson_pydantic import Feature, FeatureCollection, MultiPolygon, Polygon


class SpatialAnalysisRequest(BaseModel):
    """Base request model for spatial analysis operations."""

    data: Feature | FeatureCollection = Field(
        ..., description="GeoJSON data for analysis"
    )
    crs: str | None = Field("EPSG:4326", description="Coordinate reference system")
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Analysis-specific parameters"
    )


class SpatialAnalysisResponse(BaseModel):
    """Base response model for spatial analysis results."""

    success: bool = Field(..., description="Whether the operation succeeded")
    result: Feature | FeatureCollection | dict[str, Any] | None = Field(
        None, description="Analysis results"
    )
    message: str | None = Field(None, description="Status or error message")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional metadata about the operation"
    )


class BufferAnalysisRequest(SpatialAnalysisRequest):
    """Request model for buffer analysis operations."""

    buffer_distance: float = Field(
        ..., gt=0, description="Buffer distance in CRS units"
    )
    dissolve: bool = Field(False, description="Whether to dissolve overlapping buffers")


class ProximityAnalysisRequest(BaseModel):
    """Request model for proximity analysis operations."""

    source_data: Feature | FeatureCollection = Field(
        ..., description="Source geometries"
    )
    target_data: Feature | FeatureCollection = Field(
        ..., description="Target geometries"
    )
    max_distance: float | None = Field(None, description="Maximum distance to consider")
    crs: str | None = Field("EPSG:4326", description="Coordinate reference system")


class InterpolationRequest(BaseModel):
    """Request model for spatial interpolation operations."""

    points: FeatureCollection = Field(
        ..., description="Point observations for interpolation"
    )
    value_column: str = Field(
        ..., description="Column name containing values to interpolate"
    )
    bounds: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Interpolation bounds [minx, miny, maxx, maxy]",
    )
    resolution: float = Field(
        ..., gt=0, description="Grid resolution for interpolation"
    )
    method: str = Field(
        "idw", description="Interpolation method (idw, kriging, rbf, nearest)"
    )
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Method-specific parameters"
    )
    crs: str | None = Field("EPSG:4326", description="Coordinate reference system")

    @field_validator("method")
    def validate_method(cls: Any, v: str) -> str:
        valid_methods = ["idw", "kriging", "rbf", "nearest"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class ClusteringRequest(BaseModel):
    """Request model for spatial clustering operations."""

    points: FeatureCollection = Field(..., description="Point data for clustering")
    method: str = Field(
        "dbscan", description="Clustering method (dbscan, kmeans, hierarchical)"
    )
    parameters: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Clustering parameters (eps, min_samples, n_clusters, etc.)",
    )
    crs: str | None = Field("EPSG:4326", description="Coordinate reference system")

    @field_validator("method")
    def validate_method(cls: Any, v: str) -> str:
        valid_methods = ["dbscan", "kmeans", "hierarchical"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class HotspotRequest(BaseModel):
    """Request model for hotspot detection operations."""

    points: FeatureCollection = Field(
        ..., description="Point data for hotspot analysis"
    )
    value_column: str | None = Field(
        None, description="Column with values (if None, uses point density)"
    )
    method: str = Field("getis_ord", description="Hotspot detection method")
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Method-specific parameters"
    )
    crs: str | None = Field("EPSG:4326", description="Coordinate reference system")

    @field_validator("method")
    def validate_method(cls: Any, v: str) -> str:
        valid_methods = ["getis_ord", "local_moran", "kernel_density"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class NetworkAnalysisRequest(BaseModel):
    """Request model for network analysis operations."""

    network: FeatureCollection = Field(
        ..., description="Network edges as LineString features"
    )
    origins: FeatureCollection | None = Field(
        None, description="Origin points for routing analysis"
    )
    destinations: FeatureCollection | None = Field(
        None, description="Destination points for routing analysis"
    )
    analysis_type: str = Field(..., description="Type of network analysis")
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Analysis-specific parameters"
    )
    crs: str | None = Field("EPSG:4326", description="Coordinate reference system")

    @field_validator("analysis_type")
    def validate_analysis_type(cls: Any, v: str) -> str:
        valid_types = [
            "shortest_path",
            "service_area",
            "connectivity",
            "routing",
            "accessibility",
        ]
        if v not in valid_types:
            raise ValueError(f"Analysis type must be one of {valid_types}")
        return v


class TerrainAnalysisRequest(BaseModel):
    """Request model for terrain analysis operations."""

    dem_data: str = Field(
        ..., description="Path to DEM raster or base64 encoded raster data"
    )
    analyses: list[str] = Field(
        ["slope", "aspect", "hillshade"],
        description="List of terrain analyses to perform",
    )
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Analysis-specific parameters"
    )

    @field_validator("analyses")
    def validate_analyses(cls: Any, v: list[str]) -> list[str]:
        valid_analyses = ["slope", "aspect", "hillshade", "curvature", "tpi"]
        for analysis in v:
            if analysis not in valid_analyses:
                raise ValueError(f"Analysis must be one of {valid_analyses}")
        return v


class H3AnalysisRequest(BaseModel):
    """Request model for H3 hexagonal grid operations."""

    geometry: Feature | FeatureCollection | Polygon | MultiPolygon | None = Field(
        None, description="Polygon geometry for polygon_to_cells"
    )
    resolution: int = Field(9, ge=0, le=15, description="H3 resolution (0-15)")
    operation: str = Field("polygon_to_cells", description="H3 operation to perform")
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Operation-specific parameters"
    )

    @field_validator("operation")
    def validate_operation(cls: Any, v: str) -> str:
        valid_ops = [
            "polygon_to_cells",
            "grid_disk",
            "compact_cells",
            "cell_to_boundary",
        ]
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of {valid_ops}")
        return v


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
