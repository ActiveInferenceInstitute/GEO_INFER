"""
H3 utility functions for GEO-INFER-SPACE.

This module provides utility functions for working with H3 grid data.
All functions use H3 4.x API directly.
"""

import logging
import json
import math
from typing import Any, cast

import h3

logger = logging.getLogger(__name__)

MIN_H3_VERSION = (4, 5, 0)


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    """Parse an H3 semantic version for the supported v4 API surface."""
    try:
        parts = version.lstrip("v").split(".")
        return cast(
            tuple[int, int, int],
            tuple(int(part.split("+")[0].split("-")[0]) for part in parts[:3])
            + (0,) * max(0, 3 - len(parts)),
        )
    except (AttributeError, TypeError, ValueError):
        return None


_h3_version = _version_tuple(cast(str, getattr(h3, "__version__", None)))
if _h3_version is None or _h3_version < MIN_H3_VERSION or _h3_version[0] >= 5:
    raise RuntimeError(
        "GEO-INFER-SPACE requires h3-py >=4.5.0,<5; "
        f"found {getattr(h3, '__version__', 'unknown')!r}"
    )


def latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
    """
    Convert lat/lng to H3 cell index using H3 v4 API.

    Args:
        lat: Latitude
        lng: Longitude
        resolution: H3 resolution (0-15)

    Returns:
        H3 cell index
    """
    return cast(str, h3.latlng_to_cell(lat, lng, resolution))


def cell_to_latlng(h3_index: str) -> tuple[float, float]:
    """
    Convert H3 cell index to lat/lng using H3 v4 API.

    Args:
        h3_index: H3 cell index

    Returns:
        (lat, lng) tuple
    """
    return cast(tuple[float, float], h3.cell_to_latlng(h3_index))


def cell_to_latlng_boundary(h3_index: str) -> list[tuple[float, float]]:
    """
    Get H3 cell boundary as list of lat/lng pairs using H3 v4 API.

    Args:
        h3_index: H3 cell index

    Returns:
        List of (lat, lng) tuples representing the boundary
    """
    return cast(list[tuple[float, float]], h3.cell_to_boundary(h3_index))


def polygon_to_cells(
    polygon: dict[str, Any] | list[list[float]], resolution: int
) -> list[str]:
    """
    Convert polygon to H3 cell indices using h3 v4 API.

    Args:
        polygon: Either a GeoJSON-like dict or list of [lng, lat] coordinate pairs
        resolution: H3 resolution (0-15)

    Returns:
        List of H3 cell indices covering the polygon
    """
    # Handle different input formats
    if isinstance(polygon, dict):
        # For GeoJSON Feature or FeatureCollection
        if polygon.get("type") == "Feature":
            return sorted(h3.geo_to_cells(polygon["geometry"], resolution))
        # For GeoJSON Geometry objects
        elif polygon.get("type") in ("Polygon", "MultiPolygon"):
            # Ensure coordinates are properly nested for H3 v4
            normalized_polygon = dict(polygon)
            coordinates = normalized_polygon.get("coordinates")
            if normalized_polygon.get("type") == "Polygon" and coordinates:
                if not isinstance(coordinates[0][0], (list, tuple)):
                    normalized_polygon["coordinates"] = [coordinates]
            # Do not mutate a caller-owned GeoJSON object while normalizing it.
            return sorted(h3.geo_to_cells(normalized_polygon, resolution))
        # For GeoJSON FeatureCollection
        elif polygon.get("type") == "FeatureCollection":
            all_cells = set()
            for feature in polygon.get("features", []):
                if "geometry" in feature:
                    cells = h3.geo_to_cells(feature["geometry"], resolution)
                    all_cells.update(cells)
            return sorted(all_cells)
        else:
            raise ValueError(f"Unsupported GeoJSON type: {polygon.get('type')}")
    elif isinstance(polygon, list):
        # For coordinate lists, create a proper GeoJSON structure
        if polygon and isinstance(polygon[0], (list, tuple)) and len(polygon[0]) >= 2:
            # Create a GeoJSON polygon
            geojson = {
                "type": "Polygon",
                "coordinates": [polygon],  # Wrap in an array as GeoJSON requires
            }
            return sorted(h3.geo_to_cells(geojson, resolution))
        else:
            raise ValueError(f"Invalid coordinate list format: {polygon}")
    else:
        raise ValueError(f"Unsupported polygon format: {type(polygon)}")


def cell_to_latlngjson(
    h3_indices: list[str], properties: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """
    Convert H3 indices to GeoJSON format (H3 4.x API).
    """
    features = []

    for h3_index in h3_indices:
        # H3 returns (lat, lng); GeoJSON requires [lng, lat].
        boundary = [[lng, lat] for lat, lng in h3.cell_to_boundary(h3_index)]

        # Add closing point to the polygon if needed
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])

        # Create the polygon geometry
        polygon_geometry = {"type": "Polygon", "coordinates": [boundary]}

        # Get properties for this H3 index
        feature_properties = dict(properties.get(h3_index, {})) if properties else {}
        feature_properties["h3_index"] = h3_index

        # Create the feature
        feature = {
            "type": "Feature",
            "geometry": polygon_geometry,
            "properties": feature_properties,
        }

        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def geojson_to_h3(
    geojson_data: str | dict[str, Any],
    resolution: int = 8,
    feature_properties: bool = True,
) -> dict[str, list[str] | dict[str, dict[str, Any]]]:
    """
    Convert GeoJSON to H3 indices (H3 4.x API).

    Args:
        geojson_data: GeoJSON data as a string or dictionary.
        resolution: H3 resolution (0-15).
        feature_properties: Whether to include feature properties in the result.

    Returns:
        Dictionary with H3 indices and properties.
    """
    # Parse GeoJSON if it's a string
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)

    geojson_dict = cast(dict[str, Any], geojson_data)

    # Get features from GeoJSON
    if "type" in geojson_dict and geojson_dict["type"] == "FeatureCollection":
        features = geojson_dict.get("features", [])
    elif "type" in geojson_dict and geojson_dict["type"] == "Feature":
        features = [geojson_dict]
    else:
        # Assume it's a geometry object
        features = [{"type": "Feature", "geometry": geojson_dict, "properties": {}}]

    h3_indices: list[str] = []
    properties_dict: dict[str, dict[str, Any]] = {}

    for feature in features:
        geometry = feature.get("geometry", {})
        props = feature.get("properties", {})

        # Skip features without geometry
        if not geometry:
            continue

        # Convert geometry to H3 indices
        try:
            cells = h3.geo_to_cells(geometry, resolution)

            # Add to results
            for cell in cells:
                h3_indices.append(cell)
                if feature_properties:
                    properties_dict[cell] = props
        except Exception as e:
            logger.error(f"Failed to convert geometry to H3: {e}")

    result: dict[str, list[str] | dict[str, dict[str, Any]]] = {
        "h3_indices": h3_indices
    }
    if feature_properties:
        result["properties"] = properties_dict

    return result


# Additional H3 v4 utility functions


def geo_to_cells(geojson: dict[str, Any], resolution: int) -> list[str]:
    """Convert GeoJSON to H3 cells using H3 v4 API."""
    return sorted(h3.geo_to_cells(geojson, resolution))


def grid_disk(h3_index: str, k: int) -> list[str]:
    """Get cells within grid distance k using the H3 v4 API."""
    return sorted(h3.grid_disk(h3_index, k))


def get_h3_neighbors(h3_index: str, ring_size: int = 1) -> list[str]:
    """Return H3 cells in the requested neighborhood ring, including center."""
    return grid_disk(h3_index, ring_size)


def h3_resolution_stats(resolution: int) -> dict[str, float]:
    """Return finite area statistics for an H3 resolution."""
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    return {
        "resolution": float(resolution),
        "area_km2": float(h3.average_hexagon_area(resolution, unit="km^2")),
    }


def grid_distance(h3_index1: str, h3_index2: str) -> int:
    """Get grid distance between two H3 indices using H3 v4 API."""
    return cast(int, h3.grid_distance(h3_index1, h3_index2))


def compact_cells(h3_indices: list[str]) -> list[str]:
    """Compact H3 cells using H3 v4 API."""
    return list(h3.compact_cells(h3_indices))


def uncompact_cells(h3_indices: list[str], resolution: int) -> list[str]:
    """Uncompact H3 cells using H3 v4 API."""
    return list(h3.uncompact_cells(h3_indices, resolution))


def cell_area(h3_index: str, unit: str = "km^2") -> float:
    """Get area of H3 cell using H3 v4 API."""
    return cast(float, h3.cell_area(h3_index, unit))


def get_resolution(h3_index: str) -> int:
    """Get resolution of H3 index using H3 v4 API."""
    return cast(int, h3.get_resolution(h3_index))


def is_valid_cell(h3_index: str) -> bool:
    """Check if H3 index is valid using H3 v4 API."""
    return cast(bool, h3.is_valid_cell(h3_index))


def are_neighbor_cells(h3_index1: str, h3_index2: str) -> bool:
    """Check if two H3 indices are neighbors using H3 v4 API."""
    return cast(bool, h3.are_neighbor_cells(h3_index1, h3_index2))


def haversine_distance(
    point1: tuple[float, float], point2: tuple[float, float], radius_km: float = 6371.0
) -> float:
    """
    Great-circle distance between two (lat, lng) points in kilometres.

    Single canonical scalar implementation for GEO-INFER-SPACE. Batch /
    GPU computation with chunking and validation is provided separately by
    ``geo_infer_space.backends.gpu.gpu_acceleration.pairwise_haversine_kernel``.

    Args:
        point1: (latitude, longitude) in degrees
        point2: (latitude, longitude) in degrees
        radius_km: Sphere radius (default Earth mean radius)

    Returns:
        Distance in kilometres
    """
    lat1, lng1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lng2 = math.radians(point2[0]), math.radians(point2[1])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))
