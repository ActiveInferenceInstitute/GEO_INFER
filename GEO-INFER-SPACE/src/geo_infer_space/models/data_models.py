"""
Pydantic data models for spatial data structures.

This module defines data models for representing spatial datasets,
geometries, coordinate systems, and analysis results with automatic
validation and serialization capabilities.
"""

from typing import Any, cast
from datetime import UTC, datetime
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from geojson_pydantic import Feature, FeatureCollection, Point, Polygon, LineString
from geo_infer_time import normalize_timestamp


class _UTCMetadataModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    @field_validator("created_at", "updated_at", check_fields=False)
    @classmethod
    def normalize_metadata_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_timestamp(value)


class GeometryType(StrEnum):
    """Enumeration of supported geometry types."""

    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOINT = "MultiPoint"
    MULTILINESTRING = "MultiLineString"
    MULTIPOLYGON = "MultiPolygon"
    GEOMETRYCOLLECTION = "GeometryCollection"


class CoordinateReferenceSystem(BaseModel):
    """Model for coordinate reference system information."""

    epsg_code: int | None = Field(None, description="EPSG code")
    proj4_string: str | None = Field(None, description="PROJ4 string")
    wkt: str | None = Field(None, description="Well-Known Text representation")
    name: str | None = Field(None, description="CRS name")
    is_projected: bool = Field(False, description="Whether CRS is projected")
    units: str | None = Field(None, description="Linear units")

    @field_validator("epsg_code")
    def validate_epsg_code(cls, v: int | None) -> int | None:
        if v is not None and (v < 1000 or v > 32767):
            raise ValueError("EPSG code must be between 1000 and 32767")
        return v


class GeometryModel(BaseModel):
    """Model for geometry objects with validation."""

    geometry: Point | Polygon | LineString = Field(..., description="GeoJSON geometry")
    crs: CoordinateReferenceSystem | None = Field(
        None, description="Coordinate reference system"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Geometry properties"
    )

    def to_feature(self) -> Feature:
        """Convert to GeoJSON Feature."""
        return Feature(  # type: ignore[call-arg]
            geometry=self.geometry,
            properties=self.properties,
        )


class SpatialBounds(BaseModel):
    """Model for spatial bounding box."""

    minx: float = Field(..., description="Minimum X coordinate")
    miny: float = Field(..., description="Minimum Y coordinate")
    maxx: float = Field(..., description="Maximum X coordinate")
    maxy: float = Field(..., description="Maximum Y coordinate")
    minz: float | None = Field(None, description="Minimum Z coordinate")
    maxz: float | None = Field(None, description="Maximum Z coordinate")

    @field_validator("maxx")
    def validate_x_bounds(cls, v: float, info: ValidationInfo) -> float:
        if "minx" in info.data and v < info.data["minx"]:
            raise ValueError("maxx must be greater than or equal to minx")
        return v

    @field_validator("maxy")
    def validate_y_bounds(cls, v: float, info: ValidationInfo) -> float:
        if "miny" in info.data and v < info.data["miny"]:
            raise ValueError("maxy must be greater than or equal to miny")
        return v

    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return self.maxy - self.miny

    @property
    def area(self) -> float:
        """Calculate area of bounding box."""
        return self.width * self.height


class SpatialIndex(_UTCMetadataModel):
    """Model for spatial index configuration."""

    index_type: str = Field(..., description="Type of spatial index")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Index parameters"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp (UTC)",
    )
    num_features: int = Field(0, description="Number of indexed features")

    @field_validator("index_type")
    def validate_index_type(cls, v: str) -> str:
        valid_types = ["rtree", "quadtree", "h3", "geohash", "s2"]
        if v.lower() not in valid_types:
            raise ValueError(f"Index type must be one of {valid_types}")
        return v.lower()


class SpatialMetadata(_UTCMetadataModel):
    """Model for spatial dataset metadata."""

    name: str = Field(..., description="Dataset name")
    description: str | None = Field(None, description="Dataset description")
    source: str | None = Field(None, description="Data source")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp (UTC)",
    )
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    bounds: SpatialBounds | None = Field(None, description="Spatial bounds")
    crs: CoordinateReferenceSystem | None = Field(
        None, description="Coordinate reference system"
    )
    num_features: int = Field(0, description="Number of features")
    geometry_types: list[GeometryType] = Field(
        default_factory=list, description="Geometry types present"
    )
    attributes: dict[str, str] = Field(
        default_factory=dict, description="Attribute column types"
    )
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    license: str | None = Field(None, description="Data license")

    @field_validator("num_features")
    def validate_num_features(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Number of features cannot be negative")
        return v


class SpatialDataset(BaseModel):
    """Model for complete spatial dataset."""

    metadata: SpatialMetadata = Field(..., description="Dataset metadata")
    features: FeatureCollection | list[Feature] = Field(
        ..., description="Spatial features"
    )
    spatial_index: SpatialIndex | None = Field(
        None, description="Spatial index information"
    )

    @field_validator("features", mode="after")
    def validate_features(
        cls,
        v: FeatureCollection | list[Feature],
        info: ValidationInfo,
    ) -> FeatureCollection:
        if isinstance(v, list):
            # Convert list of features to FeatureCollection
            v = FeatureCollection(features=v)  # type: ignore[call-arg]

        # Update metadata with feature count
        metadata = info.data.get("metadata")
        if metadata is not None:
            metadata.num_features = len(v.features)

        return v

    def get_bounds(self) -> SpatialBounds | None:
        """Calculate spatial bounds of the dataset."""
        feature_collection = cast(FeatureCollection, self.features)
        if not feature_collection.features:
            return None

        # Extract coordinates from all geometries
        all_coords = []
        for feature in feature_collection.features:
            geom = feature.geometry
            coords = self._extract_coordinates(geom.model_dump())
            all_coords.extend(coords)

        if not all_coords:
            return None

        # Calculate bounds
        xs, ys = zip(*all_coords)
        return SpatialBounds(  # type: ignore[call-arg]
            minx=min(xs),
            miny=min(ys),
            maxx=max(xs),
            maxy=max(ys),
        )

    def _extract_coordinates(
        self, geometry: dict[str, Any]
    ) -> list[tuple[float, float]]:
        """Extract coordinate pairs from geometry."""
        geom_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if geom_type == "Point":
            return [tuple(coordinates[:2])]
        elif geom_type in ["LineString", "MultiPoint"]:
            return [tuple(coord[:2]) for coord in coordinates]
        elif geom_type in ["Polygon", "MultiLineString"]:
            coords = []
            for ring in coordinates:
                coords.extend([tuple(coord[:2]) for coord in ring])
            return coords
        elif geom_type == "MultiPolygon":
            coords = []
            for polygon in coordinates:
                for ring in polygon:
                    coords.extend([tuple(coord[:2]) for coord in ring])
            return coords
        elif geom_type == "GeometryCollection":
            coords = []
            for geom in geometry.get("geometries", []):
                coords.extend(self._extract_coordinates(geom))
            return coords

        return []


class AnalysisResult(_UTCMetadataModel):
    """Model for spatial analysis results."""

    analysis_type: str = Field(..., description="Type of analysis performed")
    success: bool = Field(..., description="Whether analysis succeeded")
    result_data: FeatureCollection | dict[str, Any] | None = Field(
        None, description="Analysis result data"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Analysis parameters"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    execution_time: float | None = Field(None, description="Execution time in seconds")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp (UTC)",
    )
    error_message: str | None = Field(None, description="Error message if failed")

    @field_validator("execution_time")
    def validate_execution_time(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("Execution time cannot be negative")
        return v


class H3CellData(BaseModel):
    """Model for H3 hexagonal cell data."""

    h3_index: str = Field(..., description="H3 cell index")
    resolution: int = Field(..., ge=0, le=15, description="H3 resolution")
    center_lat: float = Field(..., description="Cell center latitude")
    center_lng: float = Field(..., description="Cell center longitude")
    boundary: list[tuple[float, float]] = Field(
        ..., description="Cell boundary coordinates"
    )
    area_km2: float | None = Field(None, description="Cell area in square kilometers")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Cell properties"
    )

    @field_validator("h3_index")
    def validate_h3_index(cls, v: str) -> str:
        # Basic H3 index validation (should be 15-character hex string for most resolutions)
        if not isinstance(v, str) or len(v) < 8 or len(v) > 16:
            raise ValueError("Invalid H3 index format")
        return v

    @field_validator("center_lat")
    def validate_latitude(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("center_lng")
    def validate_longitude(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v


class NetworkEdge(BaseModel):
    """Model for network edge data."""

    edge_id: str = Field(..., description="Unique edge identifier")
    source_node: str = Field(..., description="Source node identifier")
    target_node: str = Field(..., description="Target node identifier")
    geometry: LineString = Field(..., description="Edge geometry (LineString)")
    length: float = Field(..., gt=0, description="Edge length")
    weight: float | None = Field(None, description="Edge weight for routing")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Edge attributes"
    )

    @field_validator("geometry")
    def validate_geometry_type(cls, v: LineString) -> LineString:
        if v.type != "LineString":
            raise ValueError("Network edge geometry must be LineString")
        return v


class NetworkNode(BaseModel):
    """Model for network node data."""

    node_id: str = Field(..., description="Unique node identifier")
    geometry: Point = Field(..., description="Node geometry (Point)")
    degree: int = Field(0, ge=0, description="Node degree (number of connections)")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Node attributes"
    )

    @field_validator("geometry")
    def validate_geometry_type(cls, v: Point) -> Point:
        if v.type != "Point":
            raise ValueError("Network node geometry must be Point")
        return v


class SpatialNetwork(BaseModel):
    """Model for spatial network data."""

    name: str = Field(..., description="Network name")
    nodes: list[NetworkNode] = Field(..., description="Network nodes")
    edges: list[NetworkEdge] = Field(..., description="Network edges")
    crs: CoordinateReferenceSystem | None = Field(
        None, description="Coordinate reference system"
    )
    is_directed: bool = Field(False, description="Whether network is directed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Network metadata"
    )

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the network."""
        return len(self.edges)

    def get_bounds(self) -> SpatialBounds | None:
        """Calculate spatial bounds of the network."""
        if not self.nodes:
            return None

        coords = []
        for node in self.nodes:
            if node.geometry.type == "Point":
                coords.append(tuple(node.geometry.coordinates[:2]))

        if not coords:
            return None

        xs, ys = zip(*coords)
        return SpatialBounds(  # type: ignore[call-arg]
            minx=min(xs),
            miny=min(ys),
            maxx=max(xs),
            maxy=max(ys),
        )
