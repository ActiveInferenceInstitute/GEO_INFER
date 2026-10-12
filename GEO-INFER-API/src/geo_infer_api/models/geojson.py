"""
Pydantic models for GeoJSON data structures.

These models follow the GeoJSON specification (RFC 7946)
https://tools.ietf.org/html/rfc7946
"""

from enum import StrEnum
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GeoJSONType(StrEnum):
    """Valid GeoJSON types."""

    POINT = "Point"
    MULTI_POINT = "MultiPoint"
    LINE_STRING = "LineString"
    MULTI_LINE_STRING = "MultiLineString"
    POLYGON = "Polygon"
    MULTI_POLYGON = "MultiPolygon"
    GEOMETRY_COLLECTION = "GeometryCollection"
    FEATURE = "Feature"
    FEATURE_COLLECTION = "FeatureCollection"


def _validate_position(position: tuple[float, float]) -> tuple[float, float]:
    """Validate a GeoJSON longitude/latitude position."""
    if len(position) != 2:
        raise ValueError("Positions must be [longitude, latitude]")
    lon, lat = position
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise ValueError("Coordinates must be finite")
    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180")
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90")
    return position


# Base geometry models
class GeometryBase(BaseModel):
    """Base model for all GeoJSON geometry objects."""

    model_config = ConfigDict(extra="forbid")

    type: GeoJSONType
    coordinates: Any  # Will be validated by subclasses


class Point(GeometryBase):
    """GeoJSON Point geometry."""

    type: Literal[GeoJSONType.POINT] = GeoJSONType.POINT
    coordinates: tuple[float, float] = Field(..., description="[longitude, latitude]")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate point coordinates."""
        return _validate_position(v)


class LineString(GeometryBase):
    """GeoJSON LineString geometry."""

    type: Literal[GeoJSONType.LINE_STRING] = GeoJSONType.LINE_STRING
    coordinates: list[tuple[float, float]] = Field(
        ..., description="Array of [longitude, latitude] positions"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Validate LineString has at least 2 points."""
        if len(v) < 2:
            raise ValueError("LineString must have at least 2 points")
        for position in v:
            _validate_position(position)
        return v


class Polygon(GeometryBase):
    """GeoJSON Polygon geometry.

    The first array of coordinates represents the exterior ring.
    Any subsequent arrays represent interior rings (holes).
    """

    type: Literal[GeoJSONType.POLYGON] = GeoJSONType.POLYGON
    coordinates: list[list[tuple[float, float]]] = Field(
        ..., description="Array of linear rings"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: list[list[tuple[float, float]]]
    ) -> list[list[tuple[float, float]]]:
        """Validate Polygon rings."""
        if not v or len(v) < 1:
            raise ValueError("Polygon must have at least one linear ring")

        for ring in v:
            # Each linear ring must have at least 4 positions (first = last)
            if len(ring) < 4:
                raise ValueError("Each polygon ring must have at least 4 positions")

            # First and last positions must be the same (closed ring)
            if ring[0] != ring[-1]:
                raise ValueError(
                    "First and last positions in a polygon ring must be the same"
                )
            for position in ring:
                _validate_position(position)

        return v


class MultiPoint(GeometryBase):
    """GeoJSON MultiPoint geometry."""

    type: Literal[GeoJSONType.MULTI_POINT] = GeoJSONType.MULTI_POINT
    coordinates: list[tuple[float, float]] = Field(
        ..., description="Array of positions"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Validate every point in the collection."""
        for position in v:
            _validate_position(position)
        return v


class MultiLineString(GeometryBase):
    """GeoJSON MultiLineString geometry."""

    type: Literal[GeoJSONType.MULTI_LINE_STRING] = GeoJSONType.MULTI_LINE_STRING
    coordinates: list[list[tuple[float, float]]] = Field(
        ..., description="Array of line strings"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: list[list[tuple[float, float]]]
    ) -> list[list[tuple[float, float]]]:
        """Validate every line position in the collection."""
        for line in v:
            if len(line) < 2:
                raise ValueError("LineString must have at least 2 points")
            for position in line:
                _validate_position(position)
        return v


class MultiPolygon(GeometryBase):
    """GeoJSON MultiPolygon geometry."""

    type: Literal[GeoJSONType.MULTI_POLYGON] = GeoJSONType.MULTI_POLYGON
    coordinates: list[list[list[tuple[float, float]]]] = Field(
        ..., description="Array of polygons"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: list[list[list[tuple[float, float]]]]
    ) -> list[list[list[tuple[float, float]]]]:
        """Validate every ring and position in the collection."""
        for polygon in v:
            for ring in polygon:
                if len(ring) < 4:
                    raise ValueError("Each polygon ring must have at least 4 positions")
                if ring[0] != ring[-1]:
                    raise ValueError("Polygon rings must be closed")
                for position in ring:
                    _validate_position(position)
        return v


# Union of all geometry types
Geometry = Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon


class Feature(BaseModel):
    """GeoJSON Feature object."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[GeoJSONType.FEATURE] = GeoJSONType.FEATURE
    geometry: dict[str, Any] | None = None
    properties: dict[str, Any] | None = Field(default_factory=dict)
    id: str | int | None = None


class FeatureCollection(BaseModel):
    """GeoJSON FeatureCollection object."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[GeoJSONType.FEATURE_COLLECTION] = GeoJSONType.FEATURE_COLLECTION
    features: list[Feature] = Field(..., description="Array of features")


# Specialized models for specific API operations


class PolygonFeature(Feature):
    """A GeoJSON Feature with a Polygon geometry."""

    geometry: Polygon  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def ensure_polygon_geometry(cls, values: Any) -> Any:
        """Ensure the geometry is a Polygon."""
        geometry = values.get("geometry") if isinstance(values, dict) else None
        if geometry and isinstance(geometry, dict):
            if geometry.get("type") != GeoJSONType.POLYGON:
                raise ValueError("Geometry must be a Polygon")
        return values


class PolygonFeatureCollection(FeatureCollection):
    """A GeoJSON FeatureCollection containing only Polygon features."""

    features: list[PolygonFeature]  # type: ignore[assignment]
