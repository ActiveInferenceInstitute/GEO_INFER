"""
H3 Operations module providing comprehensive hexagonal grid operations.

This module implements all H3 v4 API operations with enhanced functionality,
error handling, and real-world spatial analysis capabilities.
"""

import logging
import math
from typing import Any, cast

import h3

logger = logging.getLogger(__name__)

# Additional utility functions for comprehensive H3 operations


def get_resolution_info(resolution: int) -> dict[str, Any]:
    """
    Get detailed information about an H3 resolution level.

    Args:
        resolution: H3 resolution (0-15)

    Returns:
        Dictionary containing resolution information

    Example:
        >>> info = get_resolution_info(9)
        >>> print(f"Resolution 9 average area: {info['avg_area_km2']:.6f} km²")
    """
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError("Resolution must be an integer")
    if not 0 <= resolution <= 15:
        raise ValueError("Resolution must be between 0 and 15")

    avg_edge_length_km = float(h3.average_hexagon_edge_length(resolution, unit="km"))
    avg_area_km2 = float(h3.average_hexagon_area(resolution, unit="km^2"))

    return {
        "resolution": resolution,
        "avg_edge_length_km": avg_edge_length_km,
        "avg_edge_length_m": avg_edge_length_km * 1000,
        "avg_area_km2": avg_area_km2,
        "avg_area_m2": avg_area_km2 * 1_000_000,
        "description": (
            f"Resolution {resolution}: ~{avg_edge_length_km:.3f}km edge, "
            f"~{avg_area_km2:.6f}km² area"
        ),
    }


def find_optimal_resolution(
    area_km2: float, target_cells: int | None = None
) -> dict[str, Any]:
    """
    Find the optimal H3 resolution for a given area or target number of cells.

    Args:
        area_km2: Area in square kilometers
        target_cells: Target number of cells (optional)

    Returns:
        Dictionary with recommended resolution and analysis

    Example:
        >>> optimal = find_optimal_resolution(100.0)  # 100 km²
        >>> print(f"Recommended resolution: {optimal['recommended_resolution']}")
    """
    if isinstance(area_km2, bool) or not isinstance(area_km2, (int, float)):
        raise TypeError("Area must be a number")
    if not math.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError("Area must be a finite value greater than zero")
    if target_cells is not None:
        if isinstance(target_cells, bool) or not isinstance(target_cells, int):
            raise TypeError("Target cells must be an integer")
        if target_cells <= 0:
            raise ValueError("Target cells must be greater than zero")

    recommendations = []

    for resolution in range(16):
        res_info = get_resolution_info(resolution)

        # Estimate number of cells needed
        estimated_cells = area_km2 / res_info["avg_area_km2"]

        # Calculate efficiency score
        if target_cells is not None:
            efficiency = 1.0 / (
                1.0 + abs(estimated_cells - target_cells) / target_cells
            )
        else:
            # Prefer resolutions that give reasonable cell counts (10-10000)
            if 10 <= estimated_cells <= 10000:
                efficiency = 1.0
            elif estimated_cells < 10:
                efficiency = estimated_cells / 10
            else:
                efficiency = 10000 / estimated_cells

        recommendations.append(
            {
                "resolution": resolution,
                "estimated_cells": int(estimated_cells),
                "efficiency_score": efficiency,
                "avg_area_km2": res_info["avg_area_km2"],
                "avg_edge_length_km": res_info["avg_edge_length_km"],
            }
        )

    # Sort by efficiency score
    recommendations.sort(key=lambda x: x["efficiency_score"], reverse=True)

    return {
        "area_km2": area_km2,
        "target_cells": target_cells,
        "recommended_resolution": recommendations[0]["resolution"],
        "estimated_cells": recommendations[0]["estimated_cells"],
        "all_options": recommendations[:5],  # Top 5 options
    }


def create_h3_grid_for_bounds(
    min_lat: float, max_lat: float, min_lng: float, max_lng: float, resolution: int
) -> list[str]:
    """
    Create an H3 grid covering the specified bounding box.

    Args:
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lng: Minimum longitude
        max_lng: Maximum longitude
        resolution: H3 resolution

    Returns:
        List of H3 cell indices covering the bounding box

    Example:
        >>> grid = create_h3_grid_for_bounds(37.7, 37.8, -122.5, -122.4, 9)
        >>> print(f"Created grid with {len(grid)} cells")
    """
    # Validate bounds
    if not (-90 <= min_lat <= max_lat <= 90):
        raise ValueError("Invalid latitude bounds")
    if not (-180 <= min_lng <= max_lng <= 180):
        raise ValueError("Invalid longitude bounds")
    if not (0 <= resolution <= 15):
        raise ValueError("Resolution must be between 0 and 15")

    try:
        # Create bounding box polygon
        bbox_coords = [
            (min_lat, min_lng),
            (min_lat, max_lng),
            (max_lat, max_lng),
            (max_lat, min_lng),
        ]

        return polygon_to_cells(bbox_coords, resolution)

    except Exception as e:
        logger.error(f"Failed to create H3 grid for bounds: {e}")
        raise


# Core Coordinate Operations


def coordinate_to_cell(lat: float, lng: float, resolution: int) -> str:
    """
    Convert latitude/longitude coordinates to H3 cell index.

    Args:
        lat: Latitude in degrees (-90 to 90)
        lng: Longitude in degrees (-180 to 180)
        resolution: H3 resolution (0-15)

    Returns:
        H3 cell index string

    Raises:
        ValueError: If coordinates or resolution invalid

    Example:
        >>> cell = coordinate_to_cell(37.7749, -122.4194, 9)
        >>> print(cell)
        '89283082e3fffff'
    """
    # Validate inputs
    if not -90 <= lat <= 90:
        raise ValueError(f"Latitude {lat} must be between -90 and 90")

    if not -180 <= lng <= 180:
        raise ValueError(f"Longitude {lng} must be between -180 and 180")

    if not 0 <= resolution <= 15:
        raise ValueError(f"Resolution {resolution} must be between 0 and 15")

    try:
        return cast(str, h3.latlng_to_cell(lat, lng, resolution))
    except Exception as e:
        logger.error(f"Failed to convert coordinates ({lat}, {lng}) to H3 cell: {e}")
        raise


def cell_to_coordinates(h3_index: str) -> tuple[float, float]:
    """
    Convert H3 cell index to latitude/longitude coordinates.

    Args:
        h3_index: H3 cell index string

    Returns:
        Tuple of (latitude, longitude) coordinates

    Raises:
        ValueError: If H3 index is invalid

    Example:
        >>> coords = cell_to_coordinates('89283082e3fffff')
        >>> print(f"Lat: {coords[0]:.4f}, Lng: {coords[1]:.4f}")
    """
    try:
        return cast(tuple[float, float], h3.cell_to_latlng(h3_index))
    except Exception as e:
        logger.error(f"Failed to convert H3 index {h3_index} to coordinates: {e}")
        raise ValueError(f"Invalid H3 index: {h3_index}") from e


def cell_to_boundary(
    h3_index: str, geo_json: bool = False
) -> list[tuple[float, float]]:
    """
    Get the boundary coordinates of an H3 cell.

    Based on methods from Helsinki bike sharing analysis:
    https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2

    Args:
        h3_index: H3 cell index string
        geo_json: If True, return coordinates in GeoJSON format (lng, lat)

    Returns:
        List of boundary coordinate tuples

    Example:
        >>> boundary = cell_to_boundary('89283082e3fffff')
        >>> print(f"Hexagon has {len(boundary)} vertices")
    """
    try:
        # H3-py v4 removed the ``geo_json`` keyword from ``cell_to_boundary``.
        # Its native result is always ``(latitude, longitude)``; convert only
        # when ``geo_json`` requests GeoJSON order.
        boundary = h3.cell_to_boundary(h3_index)
        if geo_json:
            return [(lng, lat) for lat, lng in boundary]
        return [(lat, lng) for lat, lng in boundary]
    except Exception as e:
        logger.error(f"Failed to get boundary for H3 index {h3_index}: {e}")
        raise ValueError(f"Invalid H3 index: {h3_index}") from e


def cells_to_geojson(
    h3_indices: list[str], properties: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Convert H3 cell indices to GeoJSON FeatureCollection.

    Based on methods from UGRC's H3 analysis:
    https://gis.utah.gov/blog/2022-10-26-using-h3-hexes/

    Args:
        h3_indices: List of H3 cell indices
        properties: Optional properties to add to each feature

    Returns:
        GeoJSON FeatureCollection dictionary

    Example:
        >>> cells = ['89283082e3fffff', '89283082e7fffff']
        >>> geojson = cells_to_geojson(cells, {'type': 'analysis_area'})
        >>> print(f"Created FeatureCollection with {len(geojson['features'])} features")
    """
    features = []

    for h3_index in h3_indices:
        try:
            # Get boundary coordinates
            boundary = cell_to_boundary(h3_index, geo_json=True)

            # Create polygon coordinates (close the ring)
            ring = [list(point) for point in boundary]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            coordinates = [ring]

            # Create feature
            feature: dict[str, Any] = {
                "type": "Feature",
                "properties": {
                    "h3_index": h3_index,
                    "resolution": h3.get_resolution(h3_index),
                },
                "geometry": {"type": "Polygon", "coordinates": coordinates},
            }

            # Add additional properties if provided
            if properties:
                feature["properties"].update(properties)

            features.append(feature)

        except Exception as e:
            logger.warning(f"Failed to convert H3 index {h3_index} to GeoJSON: {e}")

    return {"type": "FeatureCollection", "features": features}


# Grid Operations


def grid_disk(h3_index: str, k: int) -> list[str]:
    """
    Get all H3 cells within grid distance k of the given cell.

    Based on methods from Foursquare's H3 guide:
    https://location.foursquare.com/resources/reports-and-insights/ebook/how-to-use-h3-for-geospatial-analytics/

    Args:
        h3_index: Center H3 cell index
        k: Number of rings (0 = just the center cell)

    Returns:
        Deterministically ordered list of H3 cells within grid distance k

    Example:
        >>> neighbors = grid_disk('89283082e3fffff', 2)
        >>> print(f"Found {len(neighbors)} cells within 2 rings")
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("Grid distance k must be an integer")
    if k < 0:
        raise ValueError("Grid distance k must be non-negative")

    try:
        return sorted(h3.grid_disk(h3_index, k))
    except Exception as e:
        logger.error(f"Failed to get grid disk for {h3_index} with k={k}: {e}")
        raise


def grid_ring(h3_index: str, k: int) -> list[str]:
    """
    Get H3 cells at exactly grid distance k from the given cell.

    Args:
        h3_index: Center H3 cell index
        k: Ring distance (must be > 0)

    Returns:
        Deterministically ordered list of H3 cells at grid distance k

    Example:
        >>> ring_cells = grid_ring('89283082e3fffff', 1)
        >>> print(f"Found {len(ring_cells)} cells at ring 1")
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("Ring distance k must be an integer")
    if k <= 0:
        raise ValueError("Ring distance k must be greater than 0")

    try:
        return sorted(h3.grid_ring(h3_index, k))
    except Exception as e:
        logger.error(f"Failed to get grid ring for {h3_index} with k={k}: {e}")
        raise


def grid_distance(h3_index1: str, h3_index2: str) -> int:
    """
    Calculate the grid distance between two H3 cells.

    Based on methods from Analytics Vidhya's H3 guide:
    https://www.analyticsvidhya.com/blog/2025/03/ubers-h3-for-spatial-indexing/

    Args:
        h3_index1: First H3 cell index
        h3_index2: Second H3 cell index

    Returns:
        Grid distance (number of steps)

    Example:
        >>> distance = grid_distance('89283082e3fffff', '89283082e7fffff')
        >>> print(f"Grid distance: {distance} steps")
    """
    try:
        return cast(int, h3.grid_distance(h3_index1, h3_index2))
    except Exception as e:
        logger.error(
            f"Failed to calculate grid distance between {h3_index1} and {h3_index2}: {e}"
        )
        raise


def grid_path(h3_index1: str, h3_index2: str) -> list[str]:
    """
    Find a path between two H3 cells.

    Args:
        h3_index1: Start H3 cell index
        h3_index2: End H3 cell index

    Returns:
        List of H3 cell indices forming a path

    Example:
        >>> path = grid_path('89283082e3fffff', '89283082e7fffff')
        >>> print(f"Path has {len(path)} steps")
    """
    try:
        return cast(list[str], h3.grid_path_cells(h3_index1, h3_index2))
    except Exception as e:
        logger.error(
            f"Failed to find grid path between {h3_index1} and {h3_index2}: {e}"
        )
        raise


# Hierarchy Operations


def cell_to_parent(h3_index: str, parent_resolution: int) -> str:
    """
    Get the parent cell at a coarser resolution.

    Args:
        h3_index: H3 cell index
        parent_resolution: Target parent resolution (must be < current resolution)

    Returns:
        Parent H3 cell index

    Example:
        >>> parent = cell_to_parent('89283082e3fffff', 8)
        >>> print(f"Parent cell: {parent}")
    """
    try:
        current_resolution = h3.get_resolution(h3_index)
        if parent_resolution >= current_resolution:
            raise ValueError(
                f"Parent resolution {parent_resolution} must be less than current resolution {current_resolution}"
            )

        return cast(str, h3.cell_to_parent(h3_index, parent_resolution))
    except Exception as e:
        logger.error(
            f"Failed to get parent for {h3_index} at resolution {parent_resolution}: {e}"
        )
        raise


def cell_to_children(h3_index: str, child_resolution: int) -> list[str]:
    """
    Get the children cells at a finer resolution.

    Args:
        h3_index: H3 cell index
        child_resolution: Target child resolution (must be > current resolution)

    Returns:
        Deterministically ordered list of child H3 cells

    Example:
        >>> children = cell_to_children('89283082e3fffff', 10)
        >>> print(f"Found {len(children)} children")
    """
    try:
        current_resolution = h3.get_resolution(h3_index)
        if child_resolution <= current_resolution:
            raise ValueError(
                f"Child resolution {child_resolution} must be greater than current resolution {current_resolution}"
            )

        return sorted(h3.cell_to_children(h3_index, child_resolution))
    except Exception as e:
        logger.error(
            f"Failed to get children for {h3_index} at resolution {child_resolution}: {e}"
        )
        raise


def compact_cells(h3_indices: set[str]) -> list[str]:
    """
    Compact a set of H3 cells by replacing clusters with their parents.

    Args:
        h3_indices: Set of H3 cell indices

    Returns:
        Deterministically ordered list of compacted H3 cells

    Example:
        >>> cells = {'89283082e3fffff', '89283082e7fffff', '89283082ebfffff'}
        >>> compacted = compact_cells(cells)
        >>> print(f"Compacted from {len(cells)} to {len(compacted)} cells")
    """
    try:
        return sorted(h3.compact_cells(h3_indices))
    except Exception as e:
        logger.error(f"Failed to compact cells: {e}")
        raise


def uncompact_cells(h3_indices: set[str], target_resolution: int) -> list[str]:
    """
    Uncompact a set of H3 cells to a target resolution.

    Args:
        h3_indices: Set of H3 cell indices
        target_resolution: Target resolution for uncompacting

    Returns:
        Deterministically ordered list of uncompacted H3 cells

    Example:
        >>> cells = {'89283082e3fffff'}
        >>> uncompacted = uncompact_cells(cells, 10)
        >>> print(f"Uncompacted to {len(uncompacted)} cells")
    """
    try:
        return sorted(h3.uncompact_cells(h3_indices, target_resolution))
    except Exception as e:
        logger.error(
            f"Failed to uncompact cells to resolution {target_resolution}: {e}"
        )
        raise


# Area Operations


def polygon_to_cells(
    polygon_coords: list[tuple[float, float]], resolution: int
) -> list[str]:
    """
    Get H3 cells that cover a polygon.

    Based on methods from UGRC's address point analysis:
    https://gis.utah.gov/blog/2022-10-26-using-h3-hexes/

    Args:
        polygon_coords: List of (lat, lng) coordinate tuples. This low-level
            helper uses H3's native ``LatLngPoly`` coordinate order; callers
            with GeoJSON ``[lng, lat]`` data should use ``geo_to_cells``.
        resolution: H3 resolution for the cells

    Returns:
        Deterministically ordered list of H3 cells covering the polygon

    Example:
        >>> coords = [(37.7749, -122.4194), (37.7849, -122.4194), (37.7849, -122.4094), (37.7749, -122.4094)]
        >>> cells = polygon_to_cells(coords, 9)
        >>> print(f"Polygon covered by {len(cells)} H3 cells")
    """
    try:
        # H3-py v4 expects an H3Shape (normally LatLngPoly), not the raw
        # nested coordinate list accepted by older releases.
        from h3 import LatLngPoly

        ring = list(polygon_coords)
        if ring and ring[0] == ring[-1]:
            ring = ring[:-1]
        return sorted(h3.polygon_to_cells(LatLngPoly(ring), resolution))
    except Exception as e:
        logger.error(f"Failed to convert polygon to H3 cells: {e}")
        raise


def cells_to_polygon(h3_indices: set[str]) -> list[tuple[float, float]]:
    """
    Create a polygon boundary from a set of H3 cells.

    Args:
        h3_indices: Set of H3 cell indices

    Returns:
        List of (lat, lng) coordinates forming the polygon boundary

    Example:
        >>> cells = {'89283082e3fffff', '89283082e7fffff'}
        >>> boundary = cells_to_polygon(cells)
        >>> print(f"Boundary has {len(boundary)} vertices")
    """
    try:
        geometry = h3.cells_to_geo(h3_indices)
        if geometry.get("type") != "Polygon":
            raise ValueError(
                "cells_to_polygon requires cells whose H3 coverage forms one "
                "GeoJSON Polygon; use cells_to_geo for disconnected coverage"
            )
        ring = geometry.get("coordinates", [[]])[0]
        return [(lat, lng) for lng, lat in ring]
    except Exception as e:
        logger.error(f"Failed to convert cells to polygon: {e}")
        raise


def cell_area(h3_index: str, unit: str = "km^2") -> float:
    """
    Calculate the area of an H3 cell.

    Delegates to :func:`geo_infer_space.utils.h3_utils.cell_area` — the
    canonical implementation.

    Args:
        h3_index: H3 cell index
        unit: Area unit ('km^2', 'm^2', 'rads^2')

    Returns:
        Cell area in specified units

    Example:
        >>> area = cell_area('89283082e3fffff', 'km^2')
        >>> print(f"Cell area: {area:.6f} km²")
    """
    from ...utils.h3_utils import cell_area as _canonical_cell_area

    return _canonical_cell_area(h3_index, unit)


def cells_area(h3_indices: set[str], unit: str = "km^2") -> float:
    """
    Calculate the total area of a set of H3 cells.

    Args:
        h3_indices: Set of H3 cell indices
        unit: Area unit ('km^2', 'm^2', 'rads^2')

    Returns:
        Total area in specified units

    Example:
        >>> cells = {'89283082e3fffff', '89283082e7fffff'}
        >>> total_area = cells_area(cells, 'km^2')
        >>> print(f"Total area: {total_area:.6f} km²")
    """
    try:
        total_area = 0.0
        for h3_index in h3_indices:
            total_area += h3.cell_area(h3_index, unit=unit)
        return total_area
    except Exception as e:
        logger.error(f"Failed to calculate total area: {e}")
        raise


# Analysis Operations


def neighbor_cells(h3_index: str) -> list[str]:
    """
    Get the immediate neighbors of an H3 cell.

    Args:
        h3_index: H3 cell index

    Returns:
        Deterministically ordered list of neighboring H3 cells

    Example:
        >>> neighbors = neighbor_cells('89283082e3fffff')
        >>> print(f"Cell has {len(neighbors)} neighbors")
    """
    try:
        disk = set(h3.grid_disk(h3_index, 1))
        disk.discard(h3_index)
        return sorted(disk)
    except Exception as e:
        logger.error(f"Failed to get neighbors for {h3_index}: {e}")
        raise


def cell_resolution(h3_index: str) -> int:
    """
    Get the resolution of an H3 cell.

    Args:
        h3_index: H3 cell index

    Returns:
        H3 resolution (0-15)

    Example:
        >>> resolution = cell_resolution('89283082e3fffff')
        >>> print(f"Cell resolution: {resolution}")
    """
    try:
        return cast(int, h3.get_resolution(h3_index))
    except Exception as e:
        logger.error(f"Failed to get resolution for {h3_index}: {e}")
        raise


def is_valid_cell(h3_index: str) -> bool:
    """
    Check if an H3 index is valid.

    Args:
        h3_index: H3 cell index to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> valid = is_valid_cell('89283082e3fffff')
        >>> print(f"Index is valid: {valid}")
    """
    try:
        return bool(h3.is_valid_cell(h3_index))
    except (TypeError, ValueError):
        return False
    except Exception:
        logger.debug("Unexpected error validating H3 cell %r", h3_index, exc_info=True)
        return False


def are_neighbor_cells(h3_index1: str, h3_index2: str) -> bool:
    """
    Check if two H3 cells are neighbors.

    Args:
        h3_index1: First H3 cell index
        h3_index2: Second H3 cell index

    Returns:
        True if cells are neighbors, False otherwise

    Example:
        >>> neighbors = are_neighbor_cells('89283082e3fffff', '89283082e7fffff')
        >>> print(f"Cells are neighbors: {neighbors}")
    """
    try:
        return bool(h3.are_neighbor_cells(h3_index1, h3_index2))
    except (TypeError, ValueError):
        return False
    except Exception:
        logger.debug(
            "Unexpected error checking H3 neighborship for %r, %r",
            h3_index1,
            h3_index2,
            exc_info=True,
        )
        return False


# Advanced Operations


def cells_intersection(cells1: set[str], cells2: set[str]) -> list[str]:
    """
    Find the intersection of two sets of H3 cells.

    Args:
        cells1: First set of H3 cell indices
        cells2: Second set of H3 cell indices

    Returns:
        Deterministically ordered list of H3 cell indices in both sets

    Example:
        >>> set1 = {'89283082e3fffff', '89283082e7fffff'}
        >>> set2 = {'89283082e7fffff', '89283082ebfffff'}
        >>> intersection = cells_intersection(set1, set2)
        >>> print(f"Intersection has {len(intersection)} cells")
    """
    return sorted(cells1.intersection(cells2))


def cells_union(cells1: set[str], cells2: set[str]) -> list[str]:
    """
    Find the union of two sets of H3 cells.

    Args:
        cells1: First set of H3 cell indices
        cells2: Second set of H3 cell indices

    Returns:
        Deterministically ordered list of H3 cell indices in either set

    Example:
        >>> set1 = {'89283082e3fffff', '89283082e7fffff'}
        >>> set2 = {'89283082e7fffff', '89283082ebfffff'}
        >>> union = cells_union(set1, set2)
        >>> print(f"Union has {len(union)} cells")
    """
    return sorted(cells1.union(cells2))


def cells_difference(cells1: set[str], cells2: set[str]) -> list[str]:
    """
    Find the difference between two sets of H3 cells.

    Args:
        cells1: First set of H3 cell indices
        cells2: Second set of H3 cell indices

    Returns:
        Deterministically ordered list of H3 cell indices in cells1 but not in cells2

    Example:
        >>> set1 = {'89283082e3fffff', '89283082e7fffff'}
        >>> set2 = {'89283082e7fffff', '89283082ebfffff'}
        >>> difference = cells_difference(set1, set2)
        >>> print(f"Difference has {len(difference)} cells")
    """
    return sorted(cells1.difference(cells2))


def grid_statistics(h3_indices: set[str]) -> dict[str, Any]:
    """
    Calculate comprehensive statistics for a set of H3 cells.

    Based on methods from multiple H3 analysis guides.

    Args:
        h3_indices: Set of H3 cell indices

    Returns:
        Dictionary containing grid statistics

    Example:
        >>> cells = {'89283082e3fffff', '89283082e7fffff', '89283082ebfffff'}
        >>> stats = grid_statistics(cells)
        >>> print(f"Total area: {stats['total_area_km2']:.2f} km²")
    """
    if not h3_indices:
        return {"error": "No cells provided"}

    try:
        # Basic counts
        total_cells = len(h3_indices)

        # Resolution analysis
        h3_list = sorted(h3_indices)
        resolutions = [h3.get_resolution(idx) for idx in h3_list]
        unique_resolutions = sorted(set(resolutions))

        # Area calculations
        total_area_km2 = sum(h3.cell_area(idx, "km^2") for idx in h3_list)
        total_area_m2 = sum(h3.cell_area(idx, "m^2") for idx in h3_list)

        # Connectivity analysis
        connected_pairs = 0
        total_possible_pairs = total_cells * (total_cells - 1) // 2

        for i in range(len(h3_list)):
            for j in range(i + 1, len(h3_list)):
                if h3.are_neighbor_cells(h3_list[i], h3_list[j]):
                    connected_pairs += 1

        connectivity_ratio = (
            connected_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
        )

        # Compactness (try to compact and see reduction)
        try:
            compacted = h3.compact_cells(h3_indices)
            compactness_ratio = len(compacted) / total_cells
        except Exception:
            compactness_ratio = 1.0  # No compaction possible

        # Bounding box
        if h3_indices:
            all_coords = [h3.cell_to_latlng(idx) for idx in h3_list]
            lats = [coord[0] for coord in all_coords]
            lngs = [coord[1] for coord in all_coords]

            bounding_box = {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lng": min(lngs),
                "max_lng": max(lngs),
            }

            # Calculate bounding box area
            bbox_area_km2 = (
                (bounding_box["max_lat"] - bounding_box["min_lat"])
                * (bounding_box["max_lng"] - bounding_box["min_lng"])
                * 111.32
                * 111.32
            )  # Rough conversion
        else:
            bounding_box = None
            bbox_area_km2 = 0

        return {
            "total_cells": total_cells,
            "unique_resolutions": list(unique_resolutions),
            "resolution_counts": {
                res: resolutions.count(res) for res in unique_resolutions
            },
            "total_area_km2": total_area_km2,
            "total_area_m2": total_area_m2,
            "average_area_km2": total_area_km2 / total_cells if total_cells > 0 else 0,
            "connected_pairs": connected_pairs,
            "connectivity_ratio": connectivity_ratio,
            "compactness_ratio": compactness_ratio,
            "bounding_box": bounding_box,
            "bbox_area_km2": bbox_area_km2,
            "coverage_efficiency": (
                (total_area_km2 / bbox_area_km2) if bbox_area_km2 > 0 else 0
            ),
        }

    except Exception as e:
        logger.error(f"Failed to calculate grid statistics: {e}")
        return {"error": str(e)}
