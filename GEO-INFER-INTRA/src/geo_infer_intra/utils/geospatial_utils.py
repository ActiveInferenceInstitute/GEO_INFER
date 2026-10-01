"""Geospatial test helper utilities for GEO-INFER-INTRA."""

from typing import Any
from collections.abc import Callable
import json
import math
from pathlib import Path


def create_point(lon: float, lat: float) -> dict[str, Any]:
    """
    Create a GeoJSON Point.

    Args:
        lon: Longitude
        lat: Latitude

    Returns:
        GeoJSON Point object
    """
    return {"type": "Point", "coordinates": [lon, lat]}


def create_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[float]:
    """
    Create a GeoJSON bounding box.

    Args:
        min_lon: Minimum longitude
        min_lat: Minimum latitude
        max_lon: Maximum longitude
        max_lat: Maximum latitude

    Returns:
        GeoJSON bbox array [min_lon, min_lat, max_lon, max_lat]
    """
    return [min_lon, min_lat, max_lon, max_lat]


def create_polygon(coordinates: list[list[float]]) -> dict[str, Any]:
    """
    Create a GeoJSON Polygon from a list of coordinates.

    Args:
        coordinates: List of [lon, lat] coordinate pairs

    Returns:
        GeoJSON Polygon object
    """
    # Ensure the polygon is closed (first and last points are the same)
    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + [coordinates[0]]

    return {"type": "Polygon", "coordinates": [coordinates]}


def create_feature(
    geometry: dict[str, Any], properties: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Create a GeoJSON Feature.

    Args:
        geometry: GeoJSON geometry object
        properties: Feature properties

    Returns:
        GeoJSON Feature object
    """
    if properties is None:
        properties = {}

    return {"type": "Feature", "geometry": geometry, "properties": properties}


def create_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create a GeoJSON FeatureCollection.

    Args:
        features: List of GeoJSON Feature objects

    Returns:
        GeoJSON FeatureCollection object
    """
    return {"type": "FeatureCollection", "features": features}


def is_valid_geojson(geojson_obj: dict[str, Any]) -> bool:
    """
    Check if a dictionary is valid GeoJSON.

    Args:
        geojson_obj: Dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check type field exists
        if "type" not in geojson_obj:
            return False

        # Check based on type
        if geojson_obj["type"] == "Feature":
            if "geometry" not in geojson_obj or "properties" not in geojson_obj:
                return False
            if not isinstance(geojson_obj["properties"], dict):
                return False

            # Recursive check on geometry
            if geojson_obj["geometry"] is not None:
                return is_valid_geojson(geojson_obj["geometry"])

        elif geojson_obj["type"] == "FeatureCollection":
            if "features" not in geojson_obj:
                return False
            if not isinstance(geojson_obj["features"], list):
                return False

            # Recursive check on each feature
            for feature in geojson_obj["features"]:
                if not is_valid_geojson(feature):
                    return False

        elif geojson_obj["type"] in [
            "Point",
            "LineString",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
        ]:
            if "coordinates" not in geojson_obj and "geometries" not in geojson_obj:
                return False

        return True
    except Exception:
        return False


def load_geojson_file(file_path: str | Path) -> dict[str, Any]:
    """
    Load GeoJSON from file.

    Args:
        file_path: Path to GeoJSON file

    Returns:
        GeoJSON object
    """
    with open(file_path) as f:
        return json.load(f)


def save_geojson_file(geojson_obj: dict[str, Any], file_path: str | Path) -> None:
    """
    Save GeoJSON to file.

    Args:
        geojson_obj: GeoJSON object to save
        file_path: Path to save to
    """
    with open(file_path, "w") as f:
        json.dump(geojson_obj, f, indent=2)


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Args:
        lon1: Longitude of point 1
        lat1: Latitude of point 1
        lon2: Longitude of point 2
        lat2: Latitude of point 2

    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers

    return c * r


def create_sample_h3_data(
    h3_indexes: list[str], value_generator: Callable[[str], Any]
) -> dict[str, Any]:
    """
    Create sample data for H3 indexes.

    Args:
        h3_indexes: List of H3 indexes
        value_generator: Function to generate a value for each H3 index

    Returns:
        Dictionary mapping H3 indexes to generated values
    """
    return {h3_index: value_generator(h3_index) for h3_index in h3_indexes}
