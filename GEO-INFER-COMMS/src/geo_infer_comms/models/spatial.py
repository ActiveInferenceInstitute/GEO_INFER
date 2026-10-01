"""
Geospatial metadata and spatial data models for GEO-INFER-COMMS.

This module provides comprehensive geospatial data structures and utilities
for handling spatial information in communication systems, including coordinate
reference systems, spatial bounds, and geospatial filtering capabilities.
"""

from __future__ import annotations
from typing import Any, Literal, cast
from dataclasses import dataclass, field
from datetime import datetime, UTC
import math

from geo_infer_comms.utils.validation import (
    SUPPORTED_CRS,
    validate_coordinates,
    validate_crs,
    validate_geojson_geometry,
)


class CoordinateSystem(str):
    """Supported coordinate reference systems."""

    WGS84 = SUPPORTED_CRS["WGS84"]
    UTM = SUPPORTED_CRS["UTM"]
    WEB_MERCATOR = SUPPORTED_CRS["WEB_MERCATOR"]
    LOCAL = SUPPORTED_CRS["LOCAL"]


@dataclass
class GeospatialPoint:
    """Represents a geospatial point with coordinates and metadata."""

    longitude: float  # X coordinate
    latitude: float  # Y coordinate
    altitude: float | None = None
    crs: str = CoordinateSystem.WGS84

    def __post_init__(self) -> None:
        """Validate coordinates after initialization."""
        if not validate_coordinates(self.longitude, self.latitude):
            raise ValueError(
                f"Invalid coordinates: ({self.longitude}, {self.latitude})"
            )
        if not validate_crs(self.crs):
            raise ValueError(f"Invalid CRS: {self.crs}")

    def to_dict(self) -> dict[str, Any]:
        """Convert point to dictionary representation."""
        data = {"longitude": self.longitude, "latitude": self.latitude, "crs": self.crs}
        if self.altitude is not None:
            data["altitude"] = self.altitude
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeospatialPoint:
        """Create point from dictionary."""
        return cls(
            longitude=data["longitude"],
            latitude=data["latitude"],
            altitude=data.get("altitude"),
            crs=data.get("crs", CoordinateSystem.WGS84),
        )

    def distance_to(self, other: GeospatialPoint, method: str = "haversine") -> float:
        """Calculate distance to another point in meters."""
        if method == "haversine":
            return self._haversine_distance(other)
        elif method == "euclidean":
            return self._euclidean_distance(other)
        else:
            raise ValueError(f"Unknown distance method: {method}")

    def _haversine_distance(self, other: GeospatialPoint) -> float:
        """Calculate great circle distance using Haversine formula."""
        # Convert to radians
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Earth radius in meters
        earth_radius = 6371000
        return earth_radius * c

    def _euclidean_distance(self, other: GeospatialPoint) -> float:
        """Calculate Euclidean distance (for projected coordinates)."""
        dx = self.longitude - other.longitude
        dy = self.latitude - other.latitude
        return math.sqrt(dx * dx + dy * dy)

    def is_within_bounds(self, bounds: GeospatialBounds) -> bool:
        """Check if point is within given bounds."""
        return bounds.contains_point(self)


@dataclass
class GeospatialBounds:
    """Represents geospatial bounding box or area."""

    min_longitude: float
    min_latitude: float
    max_longitude: float
    max_latitude: float
    crs: str = CoordinateSystem.WGS84

    def __post_init__(self) -> None:
        """Validate bounds after initialization."""
        if self.min_longitude >= self.max_longitude:
            raise ValueError("min_longitude must be less than max_longitude")
        if self.min_latitude >= self.max_latitude:
            raise ValueError("min_latitude must be less than max_latitude")
        if not validate_crs(self.crs):
            raise ValueError(f"Invalid CRS: {self.crs}")

    def to_dict(self) -> dict[str, Any]:
        """Convert bounds to dictionary."""
        return {
            "min_longitude": self.min_longitude,
            "min_latitude": self.min_latitude,
            "max_longitude": self.max_longitude,
            "max_latitude": self.max_latitude,
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeospatialBounds:
        """Create bounds from dictionary."""
        return cls(
            min_longitude=data["min_longitude"],
            min_latitude=data["min_latitude"],
            max_longitude=data["max_longitude"],
            max_latitude=data["max_latitude"],
            crs=data.get("crs", CoordinateSystem.WGS84),
        )

    def contains_point(self, point: GeospatialPoint) -> bool:
        """Check if point is within these bounds."""
        if point.crs != self.crs:
            # Simple check - in production, would transform coordinates
            return False
        return (
            self.min_longitude <= point.longitude <= self.max_longitude
            and self.min_latitude <= point.latitude <= self.max_latitude
        )

    def intersects(self, other: GeospatialBounds) -> bool:
        """Check if these bounds intersect with another bounds."""
        return not (
            self.max_longitude < other.min_longitude
            or other.max_longitude < self.min_longitude
            or self.max_latitude < other.min_latitude
            or other.max_latitude < self.min_latitude
        )

    def area(self) -> float:
        """Calculate approximate area in square meters."""
        # Simple calculation for WGS84 - more accurate for small areas
        if self.crs != CoordinateSystem.WGS84:
            return 0.0  # Would need proper projection transformation

        # Convert to radians
        lat1, lon1 = math.radians(self.min_latitude), math.radians(self.min_longitude)
        lat2, lon2 = math.radians(self.max_latitude), math.radians(self.max_longitude)

        # Approximate area calculation
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        avg_lat = (lat1 + lat2) / 2

        earth_radius = 6371000  # meters
        area = (earth_radius**2) * dlon * math.sin(avg_lat) * dlat
        return abs(area)

    def center(self) -> GeospatialPoint:
        """Get center point of bounds."""
        center_lon = (self.min_longitude + self.max_longitude) / 2
        center_lat = (self.min_latitude + self.max_latitude) / 2
        return GeospatialPoint(center_lon, center_lat, crs=self.crs)


@dataclass
class GeospatialMetadata:
    """Comprehensive geospatial metadata for messages and data."""

    location: GeospatialPoint
    bounds: GeospatialBounds | None = None
    accuracy: float | None = None  # meters
    precision: float | None = None  # meters
    source: str | None = None  # GPS, network, user_input, etc.
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert geospatial metadata to dictionary."""
        data: dict[str, Any] = {
            "location": self.location.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }
        if self.bounds:
            data["bounds"] = self.bounds.to_dict()
        if self.accuracy is not None:
            data["accuracy"] = self.accuracy
        if self.precision is not None:
            data["precision"] = self.precision
        if self.source:
            data["source"] = self.source
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeospatialMetadata:
        """Create geospatial metadata from dictionary."""
        location_data = data["location"]
        location = GeospatialPoint.from_dict(location_data)

        bounds = None
        if "bounds" in data:
            bounds = GeospatialBounds.from_dict(data["bounds"])

        return cls(
            location=location,
            bounds=bounds,
            accuracy=data.get("accuracy"),
            precision=data.get("precision"),
            source=data.get("source"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def distance_to(self, other: GeospatialMetadata) -> float:
        """Calculate distance between two geospatial metadata objects."""
        return self.location.distance_to(other.location)

    def is_within_distance(
        self, other: GeospatialMetadata, distance_meters: float
    ) -> bool:
        """Check if this location is within distance of another."""
        return self.distance_to(other) <= distance_meters


class SpatialFilter:
    """Represents a spatial filter for message routing and filtering."""

    def __init__(
        self,
        filter_type: Literal["bounds", "radius", "polygon", "proximity"],
        parameters: dict[str, Any],
        crs: str = CoordinateSystem.WGS84,
    ):
        self.filter_type = filter_type
        self.parameters = parameters
        self.crs = crs

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary."""
        return {
            "filter_type": self.filter_type,
            "parameters": self.parameters,
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialFilter:
        """Create filter from dictionary."""
        return cls(
            filter_type=data["filter_type"],
            parameters=data["parameters"],
            crs=data.get("crs", CoordinateSystem.WGS84),
        )

    def matches_location(self, location: GeospatialPoint) -> bool:
        """Check if location matches this spatial filter."""
        if location.crs != self.crs:
            # In production, would transform coordinates
            return False

        if self.filter_type == "bounds":
            return self._matches_bounds(location)
        elif self.filter_type == "radius":
            return self._matches_radius(location)
        elif self.filter_type == "polygon":
            return self._matches_polygon(location)
        elif self.filter_type == "proximity":
            return self._matches_proximity(location)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def _matches_bounds(self, location: GeospatialPoint) -> bool:
        """Check if location matches bounds filter."""
        bounds_data = self.parameters.get("bounds", {})
        bounds = GeospatialBounds(
            min_longitude=bounds_data["min_longitude"],
            min_latitude=bounds_data["min_latitude"],
            max_longitude=bounds_data["max_longitude"],
            max_latitude=bounds_data["max_latitude"],
            crs=self.crs,
        )
        return bounds.contains_point(location)

    def _matches_radius(self, location: GeospatialPoint) -> bool:
        """Check if location matches radius filter."""
        center_data = self.parameters.get("center", {})
        center = GeospatialPoint(
            longitude=center_data["longitude"],
            latitude=center_data["latitude"],
            crs=self.crs,
        )
        radius = cast(float, self.parameters.get("radius_meters", 0))
        return center.distance_to(location) <= radius

    def _matches_polygon(self, location: GeospatialPoint) -> bool:
        """Check if location matches polygon filter using point-in-polygon test.

        Expects ``parameters["polygon"]`` to be either:
        - A list of ``[longitude, latitude]`` pairs (GeoJSON ring order), or
        - A dict with key ``"coordinates"`` holding such a list.

        Falls back to a bounding-box pre-check when Shapely is unavailable.
        """
        polygon_data = self.parameters.get("polygon", {})
        # Accept either a raw list or a GeoJSON-style dict
        coords = (
            polygon_data
            if isinstance(polygon_data, list)
            else polygon_data.get("coordinates", [])
        )
        if not coords or len(coords) < 3:
            return False

        try:
            from shapely.geometry import Point, Polygon

            poly = Polygon([(c[0], c[1]) for c in coords])
            point = Point(location.longitude, location.latitude)
            return cast(bool, poly.contains(point))
        except ImportError:
            # Bounding-box fallback when Shapely is not installed
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            in_bounds = min(lons) <= location.longitude <= max(lons) and min(
                lats
            ) <= location.latitude <= max(lats)
            return cast(bool, in_bounds)
        except Exception:
            return False

    def _matches_proximity(self, location: GeospatialPoint) -> bool:
        """Check if location matches proximity filter."""
        target_location_data = self.parameters.get("target_location", {})
        target_location = GeospatialPoint(
            longitude=target_location_data["longitude"],
            latitude=target_location_data["latitude"],
            crs=self.crs,
        )
        max_distance = cast(float, self.parameters.get("max_distance_meters", 0))
        return location.distance_to(target_location) <= max_distance


class SpatialIndex:
    """Spatial indexing for efficient geospatial queries."""

    def __init__(
        self, index_type: str = "quadtree", bounds: GeospatialBounds | None = None
    ):
        self.index_type = index_type
        self.bounds = bounds
        self._index: dict[str, list[str]] = {}  # Simplified in-memory index

    def insert(self, location: GeospatialPoint, data_id: str) -> None:
        """Insert location-data mapping into spatial index."""
        # Simplified indexing - in production would use proper spatial index
        key = f"{location.longitude:.6f},{location.latitude:.6f}"
        if key not in self._index:
            self._index[key] = []
        self._index[key].append(data_id)

    def query(self, filter_obj: SpatialFilter) -> list[str]:
        """Query spatial index for data matching filter."""
        results = []
        for key, data_ids in self._index.items():
            # Parse location from key
            parts = key.split(",")
            if len(parts) == 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    location = GeospatialPoint(lon, lat)
                    if filter_obj.matches_location(location):
                        results.extend(data_ids)
                except (ValueError, IndexError):
                    continue
        return results

    def remove(self, location: GeospatialPoint, data_id: str) -> None:
        """Remove data from spatial index."""
        key = f"{location.longitude:.6f},{location.latitude:.6f}"
        if key in self._index:
            try:
                self._index[key].remove(data_id)
                if not self._index[key]:
                    del self._index[key]
            except ValueError:
                pass

    def clear(self) -> None:
        """Clear all data from spatial index."""
        self._index.clear()


# Utility functions for spatial operations
def calculate_distance(point1: GeospatialPoint, point2: GeospatialPoint) -> float:
    """Calculate distance between two points in meters."""
    return point1.distance_to(point2)


def create_bounds_from_points(points: list[GeospatialPoint]) -> GeospatialBounds:
    """Create bounding box from list of points."""
    if not points:
        raise ValueError("Cannot create bounds from empty point list")

    lons = [p.longitude for p in points]
    lats = [p.latitude for p in points]

    return GeospatialBounds(
        min_longitude=min(lons),
        min_latitude=min(lats),
        max_longitude=max(lons),
        max_latitude=max(lats),
    )


def buffer_point(point: GeospatialPoint, distance_meters: float) -> GeospatialBounds:
    """Create a bounding box buffer around a point."""
    # Approximate conversion: 1 degree ≈ 111,000 meters at equator
    meters_per_degree = 111000

    # Adjust for latitude (rough approximation)
    lat_adjustment = math.cos(math.radians(point.latitude))
    meters_per_degree_lon = meters_per_degree * lat_adjustment

    delta_lat = distance_meters / meters_per_degree
    delta_lon = distance_meters / meters_per_degree_lon

    return GeospatialBounds(
        min_longitude=point.longitude - delta_lon,
        min_latitude=point.latitude - delta_lat,
        max_longitude=point.longitude + delta_lon,
        max_latitude=point.latitude + delta_lat,
        crs=point.crs,
    )


def geojson_to_geospatial_point(geojson: dict[str, Any]) -> GeospatialPoint:
    """Convert GeoJSON Point to GeospatialPoint."""
    if not validate_geojson_geometry(geojson):
        raise ValueError("Invalid GeoJSON geometry")

    if geojson["type"] != "Point":
        raise ValueError("Geometry must be a Point")

    coordinates = geojson["coordinates"]
    if len(coordinates) < 2:
        raise ValueError("Point must have at least longitude and latitude")

    longitude, latitude = coordinates[0], coordinates[1]
    altitude = coordinates[2] if len(coordinates) > 2 else None

    return GeospatialPoint(longitude, latitude, altitude)


def geospatial_point_to_geojson(point: GeospatialPoint) -> dict[str, Any]:
    """Convert GeospatialPoint to GeoJSON Point."""
    coordinates = [point.longitude, point.latitude]
    if point.altitude is not None:
        coordinates.append(point.altitude)

    return {"type": "Point", "coordinates": coordinates}
