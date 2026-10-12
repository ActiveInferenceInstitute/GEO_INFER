"""
Geographic utility functions for GEO-INFER-LOG.

This module provides geographic utility functions for distance calculation,
coordinate manipulation, and geospatial data conversion.
"""

import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd


def haversine_distance(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    """
    Calculate the great circle distance between two points on the earth.

    Args:
        point1: Tuple of (longitude, latitude) for point 1
        point2: Tuple of (longitude, latitude) for point 2

    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1 = point1
    lon2, lat2 = point2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def get_bbox(
    points: list[tuple[float, float]], buffer: float = 0.0
) -> tuple[float, float, float, float]:
    """
    Get the bounding box for a list of coordinates.

    Args:
        points: List of (longitude, latitude) points
        buffer: Buffer distance in degrees to add around the bbox

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    if not points:
        raise ValueError("Points list cannot be empty")

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]

    min_lon = min(lons) - buffer
    max_lon = max(lons) + buffer
    min_lat = min(lats) - buffer
    max_lat = max(lats) + buffer

    return (min_lon, min_lat, max_lon, max_lat)


def coords_to_geojson(
    coords: list[tuple[float, float]], geometry_type: str = "LineString"
) -> dict:
    """
    Convert a list of coordinates to GeoJSON format.

    Args:
        coords: List of (longitude, latitude) coordinates
        geometry_type: Type of geometry ("Point", "LineString", "Polygon")

    Returns:
        GeoJSON dictionary
    """
    if not coords:
        raise ValueError("Coordinates list cannot be empty")

    if geometry_type == "Point" and len(coords) != 1:
        raise ValueError("Point geometry requires exactly one coordinate")

    if geometry_type == "LineString" and len(coords) < 2:
        raise ValueError("LineString geometry requires at least two coordinates")

    if geometry_type == "Polygon" and len(coords) < 3:
        raise ValueError("Polygon geometry requires at least three coordinates")

    # Create GeoJSON
    if geometry_type == "Point":
        geometry = {"type": "Point", "coordinates": coords[0]}
    elif geometry_type == "LineString":
        geometry = {"type": "LineString", "coordinates": coords}
    elif geometry_type == "Polygon":
        # Ensure polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        geometry = {
            "type": "Polygon",
            "coordinates": [coords],  # Polygon requires an array of linear rings
        }
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

    geojson = {"type": "Feature", "geometry": geometry, "properties": {}}

    return geojson


def points_to_gdf(
    points: list[tuple[float, float]], properties: list[dict] | None = None
) -> gpd.GeoDataFrame:
    """
    Convert a list of points to a GeoDataFrame.

    Args:
        points: List of (longitude, latitude) coordinates
        properties: Optional list of property dictionaries for each point

    Returns:
        GeoDataFrame of points
    """
    if not points:
        raise ValueError("Points list cannot be empty")

    if properties and len(points) != len(properties):
        raise ValueError("Number of points and properties must match")

    # Create geometry list
    geometry = [Point(lon, lat) for lon, lat in points]

    # Create GeoDataFrame
    if properties:
        gdf = gpd.GeoDataFrame(properties, geometry=geometry, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    return gdf


def route_to_linestring(coords: list[tuple[float, float]]) -> LineString:
    """
    Convert route coordinates to a LineString geometry.

    Args:
        coords: List of (longitude, latitude) coordinates

    Returns:
        LineString geometry
    """
    if len(coords) < 2:
        raise ValueError("Route must have at least two coordinates")

    return LineString(coords)


def create_buffer(point: tuple[float, float], distance_km: float) -> Polygon:
    """
    Create a buffer around a point with a specified radius.

    Args:
        point: (longitude, latitude) coordinate
        distance_km: Buffer radius in kilometers

    Returns:
        Polygon representing the buffer
    """
    # Convert distance in km to degrees (approximately)
    # This is a rough conversion and will be more accurate near the equator
    distance_deg = distance_km / 111  # ~111 km per degree

    # Create point and buffer
    pt = Point(point)
    buffer = pt.buffer(distance_deg)

    return buffer


def calculate_route_distance(coords: list[tuple[float, float]]) -> float:
    """
    Calculate the total distance of a route using Haversine formula.

    Args:
        coords: List of (longitude, latitude) coordinates

    Returns:
        Total distance in kilometers
    """
    if len(coords) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(len(coords) - 1):
        total_distance += haversine_distance(coords[i], coords[i + 1])

    return total_distance


def get_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Calculate the geographic centroid of a set of points.

    Args:
        points: List of (longitude, latitude) coordinates

    Returns:
        (longitude, latitude) of the centroid
    """
    if not points:
        raise ValueError("Points list cannot be empty")

    # Convert to radians
    radians = np.radians(points)

    # Convert lat/lon to Cartesian coordinates for the centroid
    x = np.cos(radians[:, 1]) * np.cos(radians[:, 0])
    y = np.cos(radians[:, 1]) * np.sin(radians[:, 0])
    z = np.sin(radians[:, 1])

    # Compute the centroid in Cartesian coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Convert the centroid back to lat/lon
    lon = np.arctan2(y_mean, x_mean)
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    lat = np.arctan2(z_mean, hyp)

    # Convert to degrees
    return (np.degrees(lon), np.degrees(lat))


def within_distance(
    point: tuple[float, float], target: tuple[float, float], max_distance_km: float
) -> bool:
    """
    Check if a point is within a specified distance of a target.

    Args:
        point: (longitude, latitude) coordinate to check
        target: (longitude, latitude) target coordinate
        max_distance_km: Maximum distance in kilometers

    Returns:
        True if point is within the specified distance of the target
    """
    return haversine_distance(point, target) <= max_distance_km
