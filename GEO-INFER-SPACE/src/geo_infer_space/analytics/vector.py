"""
Vector operations module for advanced spatial analysis.

This module provides comprehensive vector-based spatial operations including
overlay analysis, buffer operations, proximity calculations, and geometric
computations using GeoPandas and Shapely.
"""

import logging
import numpy as np
import geopandas as gpd
from typing import Any
from shapely.ops import unary_union
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

logger = logging.getLogger(__name__)


def _reproject(frame: gpd.GeoDataFrame, target_crs: Any) -> gpd.GeoDataFrame:
    """Reproject without the array-to-scalar warning in older GeoPandas."""
    if frame.crs == target_crs:
        return frame
    transformer = Transformer.from_crs(frame.crs, target_crs, always_xy=True)
    result = frame.copy()
    result["geometry"] = result.geometry.map(
        lambda geometry: shapely_transform(transformer.transform, geometry)
    )
    return result.set_crs(target_crs, allow_override=True)


def buffer_and_intersect(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    buffer_distance_meters: int | float,
) -> gpd.GeoDataFrame:
    """Buffer points and intersect with polygons.

    Args:
        points_gdf: GeoDataFrame of points to buffer.
        polygons_gdf: GeoDataFrame of polygons to intersect with.
        buffer_distance_meters: Buffer distance in meters.

    Returns:
        GeoDataFrame of intersection results.

    Raises:
        ValueError: If CRS mismatch or empty inputs.
    """
    if points_gdf.empty or polygons_gdf.empty:
        raise ValueError("Input GeoDataFrames cannot be empty")

    if points_gdf.crs != polygons_gdf.crs:
        raise ValueError("CRS mismatch between points and polygons")

    # Project to metric CRS if necessary
    original_crs = points_gdf.crs
    if not original_crs.is_projected:
        metric_crs = "EPSG:3857"  # Web Mercator
        points_gdf = points_gdf.to_crs(metric_crs)
        polygons_gdf = polygons_gdf.to_crs(metric_crs)

    buffered = points_gdf.buffer(buffer_distance_meters)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=points_gdf.crs)

    intersection = gpd.overlay(buffered_gdf, polygons_gdf, how="intersection")

    # Reproject back to original CRS
    intersection = intersection.to_crs(original_crs)

    logger.info("Intersection completed: %d features", len(intersection))
    return intersection


def overlay_analysis(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    operation: str = "intersection",
    keep_geom_type: bool = True,
) -> gpd.GeoDataFrame:
    """
    Perform overlay operations between two GeoDataFrames.

    Args:
        gdf1: First GeoDataFrame
        gdf2: Second GeoDataFrame
        operation: Type of overlay ('intersection', 'union', 'difference', 'symmetric_difference')
        keep_geom_type: Whether to keep only geometries of the same type

    Returns:
        GeoDataFrame with overlay results

    Raises:
        ValueError: If invalid operation or CRS mismatch
    """
    valid_operations = ["intersection", "union", "difference", "symmetric_difference"]
    if operation not in valid_operations:
        raise ValueError(f"Operation must be one of {valid_operations}")

    if gdf1.crs != gdf2.crs:
        gdf2 = _reproject(gdf2, gdf1.crs)
        logger.warning("Reprojected gdf2 to match gdf1 CRS")

    try:
        result = gpd.overlay(gdf1, gdf2, how=operation, keep_geom_type=keep_geom_type)
        logger.info(f"Overlay {operation} completed: {len(result)} features")
        return result
    except Exception as e:
        logger.error(f"Overlay operation failed: {e}")
        raise


def proximity_analysis(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, max_distance: float | None = None
) -> gpd.GeoDataFrame:
    """
    Calculate proximity metrics between two sets of geometries.

    Args:
        gdf1: Source geometries
        gdf2: Target geometries
        max_distance: Maximum distance to consider (in CRS units)

    Returns:
        GeoDataFrame with distance calculations and nearest feature info
    """
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    metric_gdf1 = (
        gdf1.to_crs("EPSG:3857") if gdf1.crs and gdf1.crs.is_geographic else gdf1
    )
    metric_gdf2 = (
        gdf2.to_crs("EPSG:3857") if gdf2.crs and gdf2.crs.is_geographic else gdf2
    )
    result_data = []

    for idx1, geom1 in metric_gdf1.iterrows():
        distances = []
        nearest_idx = None
        min_distance = float("inf")

        for idx2, geom2 in metric_gdf2.iterrows():
            distance = geom1.geometry.distance(geom2.geometry)
            distances.append(distance)

            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx2

        # Skip if beyond max_distance
        if max_distance and min_distance > max_distance:
            continue

        result_data.append(
            {
                "source_id": idx1,
                "nearest_id": nearest_idx,
                "min_distance": min_distance,
                "mean_distance": np.mean(distances),
                "geometry": gdf1.loc[idx1].geometry,
            }
        )

    result_gdf = gpd.GeoDataFrame(result_data, crs=gdf1.crs)
    logger.info(f"Proximity analysis completed: {len(result_gdf)} features")
    return result_gdf


def spatial_join_analysis(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    predicate: str = "intersects",
    how: str = "inner",
) -> gpd.GeoDataFrame:
    """
    Perform spatial join between two GeoDataFrames.

    Args:
        left_gdf: Left GeoDataFrame
        right_gdf: Right GeoDataFrame
        predicate: Spatial relationship ('intersects', 'contains', 'within', 'touches', 'crosses', 'overlaps')
        how: Type of join ('left', 'right', 'inner')

    Returns:
        GeoDataFrame with joined attributes
    """
    valid_predicates = [
        "intersects",
        "contains",
        "within",
        "touches",
        "crosses",
        "overlaps",
    ]
    if predicate not in valid_predicates:
        raise ValueError(f"Predicate must be one of {valid_predicates}")

    if left_gdf.crs != right_gdf.crs:
        right_gdf = right_gdf.to_crs(left_gdf.crs)

    try:
        result = gpd.sjoin(left_gdf, right_gdf, predicate=predicate, how=how)
        logger.info(f"Spatial join completed: {len(result)} features")
        return result
    except Exception as e:
        logger.error(f"Spatial join failed: {e}")
        raise


def geometric_calculations(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate geometric properties for geometries.

    Args:
        gdf: Input GeoDataFrame

    Returns:
        GeoDataFrame with additional geometric columns
    """
    result = gdf.copy()
    metric_gdf = gdf.to_crs("EPSG:3857") if gdf.crs and gdf.crs.is_geographic else gdf

    # Calculate area for polygons
    polygon_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if any(polygon_mask):
        result.loc[polygon_mask, "area"] = metric_gdf.loc[polygon_mask, "geometry"].area
        result.loc[polygon_mask, "perimeter"] = metric_gdf.loc[
            polygon_mask, "geometry"
        ].length
    else:
        result["area"] = 0.0
        result["perimeter"] = 0.0

    # Calculate length for lines
    if any(gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])):
        result["length"] = metric_gdf.geometry.length

    # Calculate centroids for all geometries
    metric_centroids = metric_gdf.geometry.centroid
    result["centroid_x"] = metric_centroids.x
    result["centroid_y"] = metric_centroids.y

    # Calculate bounds
    bounds = gdf.bounds
    result["bbox_area"] = (bounds["maxx"] - bounds["minx"]) * (
        bounds["maxy"] - bounds["miny"]
    )

    # Calculate convex hull area ratio (compactness measure)
    result["convex_hull_area"] = metric_gdf.geometry.convex_hull.area
    # Avoid division by zero
    result["compactness"] = result["area"] / result["convex_hull_area"].replace(0, 1)

    logger.info("Geometric calculations completed")
    return result


def topology_operations(
    gdf: gpd.GeoDataFrame, operation: str, tolerance: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Perform topology operations on geometries.

    Args:
        gdf: Input GeoDataFrame
        operation: Operation type ('buffer', 'simplify', 'convex_hull', 'envelope', 'dissolve')
        tolerance: Tolerance for simplification operations

    Returns:
        GeoDataFrame with processed geometries
    """
    operations = {"buffer", "simplify", "convex_hull", "envelope", "dissolve"}
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    if operation in {"buffer", "simplify"} and (
        isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0
    ):
        raise ValueError(
            "Buffer distance and simplification tolerance must be positive"
        )
    metric_gdf = (
        _reproject(gdf, "EPSG:3857") if gdf.crs and gdf.crs.is_geographic else gdf
    )
    # Keep coordinate data and its CRS together through every operation.
    result = metric_gdf.copy()

    if operation == "buffer":
        result["geometry"] = metric_gdf.geometry.buffer(tolerance)

    elif operation == "simplify":
        result["geometry"] = metric_gdf.geometry.simplify(
            tolerance, preserve_topology=True
        )

    elif operation == "convex_hull":
        result["geometry"] = metric_gdf.geometry.convex_hull

    elif operation == "envelope":
        result["geometry"] = metric_gdf.geometry.envelope

    elif operation == "dissolve":
        # Dissolve all geometries into one
        dissolved_geom = unary_union(metric_gdf.geometry.tolist())
        result = gpd.GeoDataFrame([{"geometry": dissolved_geom}], crs=metric_gdf.crs)

    if gdf.crs and result.crs != gdf.crs:
        result = _reproject(result, gdf.crs)

    logger.info(f"Topology operation '{operation}' completed")
    return result
