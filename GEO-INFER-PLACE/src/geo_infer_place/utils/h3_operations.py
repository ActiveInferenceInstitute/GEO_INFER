#!/usr/bin/env python3
"""
H3 Operations Module for GEO-INFER-PLACE

Provides unified H3 v4 geospatial operations for the entire module.
All H3 functions are consolidated here to avoid duplication across files.

Usage:
    from geo_infer_place.utils.h3_operations import (
        latlng_to_cell,
        cell_to_latlng,
        polygon_to_cells,
    )
"""

import logging
from typing import Any, cast

import h3

logger = logging.getLogger(__name__)

MIN_H3_VERSION = (4, 5, 0)


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    """Parse an H3 semantic version for the supported v4 API surface."""
    try:
        parts = version.lstrip("v").split(".")
        parsed = tuple(int(part.split("+")[0].split("-")[0]) for part in parts[:3])
        if len(parsed) < 3:
            parsed = parsed + (0,) * (3 - len(parsed))
        return parsed[0], parsed[1], parsed[2]
    except (AttributeError, TypeError, ValueError):
        return None


_h3_version = _version_tuple(str(getattr(h3, "__version__", "")))
if _h3_version is None or _h3_version < MIN_H3_VERSION or _h3_version[0] >= 5:
    raise RuntimeError(
        "GEO-INFER-PLACE requires h3-py >=4.5.0,<5; "
        f"found {getattr(h3, '__version__', 'unknown')!r}"
    )

# ============================================================================
# Core H3 v4 Operations
# ============================================================================


def latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
    """
    Convert latitude/longitude coordinates to an H3 cell index.

    Args:
        lat: Latitude in degrees
        lng: Longitude in degrees
        resolution: H3 resolution (0-15)

    Returns:
        H3 cell index as string
    """
    return cast(str, h3.latlng_to_cell(lat, lng, resolution))


def cell_to_latlng(cell: str) -> tuple[float, float]:
    """
    Get the center coordinates of an H3 cell.

    Args:
        cell: H3 cell index

    Returns:
        Tuple of (latitude, longitude)
    """
    return cast(tuple[float, float], h3.cell_to_latlng(cell))


def cell_to_latlng_boundary(cell: str) -> list[tuple[float, float]]:
    """
    Get the boundary vertices of an H3 cell.

    Args:
        cell: H3 cell index

    Returns:
        List of (lat, lng) tuples forming the cell boundary
    """
    return cast(list[tuple[float, float]], h3.cell_to_boundary(cell))


def geo_to_cells(geojson: Any, resolution: int) -> list[str]:
    """
    Convert a GeoJSON polygon to a set of H3 cells.

    Args:
        geojson: GeoJSON polygon geometry
        resolution: H3 resolution

    Returns:
        List of H3 cell indices
    """
    if hasattr(geojson, "__geo_interface__"):
        geojson = geojson.__geo_interface__
    if geojson.get("type") == "Feature":
        geojson = geojson.get("geometry")
    if geojson.get("type") == "FeatureCollection":
        cells: set[str] = set()
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                cells.update(h3.geo_to_cells(geometry, resolution))
        return sorted(cells)
    if geojson.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("geojson must contain a Polygon or MultiPolygon geometry")
    return sorted(h3.geo_to_cells(geojson, resolution))


def polygon_to_cells(polygon: Any, resolution: int) -> list[str]:
    """
    Convert a Shapely polygon to H3 cells.

    Args:
        polygon: Shapely Polygon or GeoJSON dict
        resolution: H3 resolution

    Returns:
        List of H3 cell indices
    """
    if hasattr(polygon, "__geo_interface__"):
        geojson = polygon.__geo_interface__
    elif isinstance(polygon, dict):
        geojson = polygon
    else:
        raise ValueError(f"Unsupported polygon type: {type(polygon)}")

    return geo_to_cells(geojson, resolution)


# ============================================================================
# Grid Operations
# ============================================================================


def grid_disk(cell: str, k: int = 1) -> list[str]:
    """
    Get all cells within k grid distance of the origin cell.

    Args:
        cell: Origin H3 cell index
        k: Distance (number of rings)

    Returns:
        List of H3 cell indices in the disk
    """
    return list(h3.grid_disk(cell, k))


def grid_distance(cell1: str, cell2: str) -> int:
    """
    Get the grid distance between two H3 cells.

    Args:
        cell1: First H3 cell index
        cell2: Second H3 cell index

    Returns:
        Grid distance (number of cells)
    """
    return cast(int, h3.grid_distance(cell1, cell2))


def grid_ring(cell: str, k: int) -> list[str]:
    """
    Get cells at exactly k grid distance from origin.

    Args:
        cell: Origin H3 cell index
        k: Distance (ring number)

    Returns:
        List of H3 cell indices in the ring
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return sorted(set(h3.grid_disk(cell, k)) - set(h3.grid_disk(cell, k - 1)))


# ============================================================================
# Cell Properties
# ============================================================================


def cell_area(cell: str, unit: str = "km^2") -> float:
    """
    Get the area of an H3 cell.

    Args:
        cell: H3 cell index
        unit: Area unit ('km^2' or 'm^2')

    Returns:
        Cell area in specified units
    """
    return cast(float, h3.cell_area(cell, unit=unit))


def get_resolution(cell: str) -> int:
    """
    Get the resolution of an H3 cell.

    Args:
        cell: H3 cell index

    Returns:
        Resolution (0-15)
    """
    return cast(int, h3.get_resolution(cell))


def is_valid_cell(cell: str) -> bool:
    """
    Check if a string is a valid H3 cell index.

    Args:
        cell: H3 cell index to validate

    Returns:
        True if valid, False otherwise
    """
    return cast(bool, h3.is_valid_cell(cell))


def are_neighbor_cells(cell1: str, cell2: str) -> bool:
    """
    Check if two H3 cells are neighbors.

    Args:
        cell1: First H3 cell index
        cell2: Second H3 cell index

    Returns:
        True if cells are neighbors
    """
    return cast(bool, h3.are_neighbor_cells(cell1, cell2))


def get_base_cell_number(cell: str) -> int:
    """
    Get the base cell number of an H3 index.

    Args:
        cell: H3 cell index

    Returns:
        Base cell number (0-121)
    """
    return cast(int, h3.get_base_cell_number(cell))


# ============================================================================
# Hierarchical Operations
# ============================================================================


def cell_to_parent(cell: str, parent_res: int) -> str:
    """
    Get the parent cell at a coarser resolution.

    Args:
        cell: H3 cell index
        parent_res: Parent resolution (must be less than cell's resolution)

    Returns:
        Parent H3 cell index
    """
    return cast(str, h3.cell_to_parent(cell, parent_res))


def cell_to_children(cell: str, child_res: int) -> list[str]:
    """
    Get all child cells at a finer resolution.

    Args:
        cell: H3 cell index
        child_res: Child resolution (must be greater than cell's resolution)

    Returns:
        List of child H3 cell indices
    """
    return list(h3.cell_to_children(cell, child_res))


def compact_cells(cells: list[str]) -> list[str]:
    """
    Compact a set of cells to their most compact representation.

    Args:
        cells: List of H3 cell indices

    Returns:
        Compacted list of H3 cell indices
    """
    return list(h3.compact_cells(cells))


def uncompact_cells(cells: list[str], resolution: int) -> list[str]:
    """
    Uncompact cells to a specified resolution.

    Args:
        cells: Compacted list of H3 cell indices
        resolution: Target resolution

    Returns:
        Uncompacted list of H3 cell indices
    """
    return list(h3.uncompact_cells(cells, resolution))


# ============================================================================
# Utility Functions
# ============================================================================


def cells_to_geodataframe(cells: list[str]) -> Any:
    """
    Convert a list of H3 cells to a GeoDataFrame with polygon geometries.

    Args:
        cells: List of H3 cell indices

    Returns:
        GeoDataFrame with cell polygons
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    geometries = []
    for cell in cells:
        boundary = cell_to_latlng_boundary(cell)
        # H3 returns (lat, lng) but Shapely needs (lng, lat)
        polygon = Polygon([(lng, lat) for lat, lng in boundary])
        geometries.append(polygon)

    return gpd.GeoDataFrame({"h3_cell": cells}, geometry=geometries, crs="EPSG:4326")


def estimate_cell_count(area_km2: float, resolution: int) -> int:
    """
    Estimate the number of H3 cells to cover an area.

    Args:
        area_km2: Area in square kilometers
        resolution: H3 resolution

    Returns:
        Estimated number of cells
    """
    avg_cell_area = h3.average_hexagon_area(resolution, unit="km^2")
    return max(1, int(area_km2 / avg_cell_area))


logger.debug("H3 operations module initialized with H3 v4 API")
