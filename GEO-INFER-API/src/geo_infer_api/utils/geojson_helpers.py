"""
Utility functions for working with GeoJSON data.
"""

import math
from typing import cast

from geo_infer_api.models.geojson import GeoJSONType, Polygon, PolygonFeature


def validate_polygon_rings(coordinates: list[list[tuple[float, float]]]) -> bool:
    """
    Validate that a polygon's rings follow the GeoJSON specification.

    Args:
        coordinates: List of rings where each ring is a list of [lon, lat] coordinates

    Returns:
        bool: True if valid, False otherwise
    """
    if not coordinates or len(coordinates) < 1:
        return False

    for ring in coordinates:
        # Each ring must have at least 4 coordinates (closed loop)
        if len(ring) < 4:
            return False

        # First and last positions must be identical (closed loop)
        if ring[0] != ring[-1]:
            return False

        # Check coordinate bounds
        for pos in ring:
            lon, lat = pos
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                return False

    return True


def calculate_polygon_area(polygon: Polygon | dict) -> float:
    """
    Calculate the approximate area of a polygon in square kilometers.

    Uses a planar shoelace calculation that is approximate for small areas.

    Args:
        polygon: A GeoJSON Polygon object or dict

    Returns:
        float: Area in square kilometers
    """
    if isinstance(polygon, Polygon):
        coords = polygon.coordinates[0]  # Use exterior ring only
    elif isinstance(polygon, dict) and polygon.get("type") == GeoJSONType.POLYGON:
        coords = polygon.get("coordinates", [[]])[0]
    else:
        raise ValueError("Input must be a GeoJSON Polygon")

    n = len(coords) - 1  # Subtract 1 because first/last points are the same
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]

    # Convert degrees² to km²: 1 degree ≈ 111 km
    area = abs(area) * 0.5 * 111 * 111

    return area


def polygon_contains_point(polygon: Polygon | dict, point: tuple[float, float]) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.

    Args:
        polygon: A GeoJSON Polygon object or dict
        point: A (longitude, latitude) tuple

    Returns:
        bool: True if the point is inside the polygon, False otherwise
    """
    if isinstance(polygon, Polygon):
        exterior_ring = polygon.coordinates[0]
    elif isinstance(polygon, dict) and polygon.get("type") == GeoJSONType.POLYGON:
        exterior_ring = polygon.get("coordinates", [[]])[0]
    else:
        raise ValueError("Input must be a GeoJSON Polygon")

    x, y = point
    n = len(exterior_ring)
    inside = False

    x, y = point
    n = len(exterior_ring)
    inside = False
    j = n - 1
    for i in range(n):
        p1x, p1y = exterior_ring[i]
        p2x, p2y = exterior_ring[j]
        # Standard PNPOLY half-open test; horizontal edges never toggle.
        if (p1y > y) != (p2y > y):
            xinters = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
            if x < xinters:
                inside = not inside
        j = i

    return inside


def simplify_polygon(polygon: Polygon | dict, tolerance: float = 0.01) -> Polygon:
    """
    Simplify a polygon using the Ramer-Douglas-Peucker algorithm.

    Args:
        polygon: A GeoJSON Polygon object or dict
        tolerance: The simplification tolerance

    Returns:
        Polygon: A simplified Polygon
    """

    def rdp(points: list, epsilon: float) -> list:
        """Recursive Ramer-Douglas-Peucker simplification."""
        if len(points) <= 2:
            return points

        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            results1 = rdp(points[: index + 1], epsilon)
            results2 = rdp(points[index:], epsilon)
            return results1[:-1] + results2
        else:
            return [points[0], points[-1]]

    def perpendicular_distance(
        point: tuple, line_start: tuple, line_end: tuple
    ) -> float:
        """Calculate perpendicular distance from a point to a line segment."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        if x1 == x2:
            return float(abs(x - x1))

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return float(abs(slope * x - y + intercept) / ((slope**2 + 1) ** 0.5))

    rings = (
        polygon.coordinates if isinstance(polygon, Polygon) else polygon["coordinates"]
    )
    simplified_rings = []
    for ring in rings:
        simplified_ring = rdp(ring[:-1], tolerance)

        if len(simplified_ring) < 3:
            simplified_ring = ring[:-1]

        if simplified_ring[0] != simplified_ring[-1]:
            simplified_ring.append(simplified_ring[0])

        if len(simplified_ring) < 4:
            simplified_rings.append(ring)
        else:
            simplified_rings.append(simplified_ring)

    return Polygon(type=GeoJSONType.POLYGON, coordinates=simplified_rings)


def create_polygon_feature(
    coordinates: list[list[tuple[float, float]]],
    properties: dict | None = None,
    feature_id: str | None = None,
) -> PolygonFeature:
    """
    Create a GeoJSON PolygonFeature from coordinates.

    Args:
        coordinates: List of rings where each ring is a list of [lon, lat] coordinates
        properties: Optional properties to attach to the feature
        feature_id: Optional feature ID

    Returns:
        PolygonFeature: A GeoJSON Feature with Polygon geometry
    """
    if not validate_polygon_rings(coordinates):
        raise ValueError("Invalid polygon coordinates")

    # Convert coordinates from list of tuples to proper ring positions
    json_coordinates = [[(lon, lat) for lon, lat in ring] for ring in coordinates]

    polygon = Polygon(type=GeoJSONType.POLYGON, coordinates=json_coordinates)

    return PolygonFeature(
        type=GeoJSONType.FEATURE,
        geometry=polygon,
        properties=properties or {},
        id=feature_id,
    )


def _get_exterior_ring(polygon: Polygon | dict) -> list[tuple[float, float]]:
    """Extract exterior ring coordinates from a Polygon or dict."""
    if isinstance(polygon, Polygon):
        return polygon.coordinates[0]
    elif isinstance(polygon, dict) and polygon.get("type") == GeoJSONType.POLYGON:
        coords = polygon.get("coordinates", [[]])[0]
        return cast("list[tuple[float, float]]", coords)
    raise ValueError("Input must be a GeoJSON Polygon")


def create_buffer(
    polygon: Polygon | dict,
    distance: float,
    unit: str = "kilometers",
    segments: int = 16,
) -> Polygon:
    """
    Create an axis-aligned bounding-box buffer around a polygon.

    The buffer expands the polygon's bounding box by the specified distance
    on all sides. This is a fast approximation suitable for spatial indexing
    and coarse containment checks. For geodetically accurate buffers, use
    a library such as shapely with pyproj.

    Args:
        polygon: A GeoJSON Polygon object or dict
        distance: Buffer distance (in the specified unit)
        unit: Unit for the distance ("meters", "kilometers", or "miles")
        segments: Unused — retained for API compatibility

    Returns:
        Polygon: A new polygon representing the rectangular buffer zone
    """
    # Convert distance to kilometers
    if unit == "meters":
        distance_km = distance / 1000.0
    elif unit == "miles":
        distance_km = distance * 1.60934
    else:
        distance_km = distance

    exterior_ring = _get_exterior_ring(polygon)

    # Compute bounding box
    lons = [coord[0] for coord in exterior_ring]
    lats = [coord[1] for coord in exterior_ring]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Expand bounding box by distance_km on each side.
    # Approximate: 1 degree latitude ≈ 111.32 km everywhere;
    # 1 degree longitude ≈ 111.32 * cos(lat) km.
    avg_lat = (min_lat + max_lat) / 2.0
    lat_delta = distance_km / 111.32
    lon_delta = distance_km / (111.32 * max(math.cos(math.radians(avg_lat)), 1e-9))

    buf_min_lon = min_lon - lon_delta
    buf_max_lon = max_lon + lon_delta
    buf_min_lat = min_lat - lat_delta
    buf_max_lat = max_lat + lat_delta

    buffer_ring: list[tuple[float, float]] = [
        (buf_min_lon, buf_min_lat),
        (buf_max_lon, buf_min_lat),
        (buf_max_lon, buf_max_lat),
        (buf_min_lon, buf_max_lat),
        (buf_min_lon, buf_min_lat),  # Close the ring
    ]

    return Polygon(type=GeoJSONType.POLYGON, coordinates=[buffer_ring])


def calculate_intersection(polygons: list[Polygon | dict]) -> Polygon:
    """
    Calculate the bounding-box intersection of multiple polygons.

    Computes the axis-aligned bounding box (AABB) of each polygon, then
    returns the intersection of those boxes as a Polygon. Returns None
    if the bounding boxes do not overlap.

    For exact geometric intersection, use shapely.

    Args:
        polygons: List of GeoJSON Polygon objects or dicts (minimum 2)

    Returns:
        Polygon: The intersection bounding-box polygon

    Raises:
        ValueError: If fewer than 2 polygons are supplied, or if the
                    bounding boxes do not overlap.
    """
    if len(polygons) < 2:
        raise ValueError("At least 2 polygons required for intersection")

    # Compute bounding box for each polygon
    def bbox(poly: Polygon | dict) -> tuple[float, float, float, float]:
        ring = _get_exterior_ring(poly)
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        return min(lons), min(lats), max(lons), max(lats)

    bboxes = [bbox(p) for p in polygons]

    # Intersection of all bounding boxes
    inter_min_lon = max(b[0] for b in bboxes)
    inter_min_lat = max(b[1] for b in bboxes)
    inter_max_lon = min(b[2] for b in bboxes)
    inter_max_lat = min(b[3] for b in bboxes)

    if inter_min_lon >= inter_max_lon or inter_min_lat >= inter_max_lat:
        raise ValueError("Polygons do not overlap — intersection is empty")

    ring: list[tuple[float, float]] = [
        (inter_min_lon, inter_min_lat),
        (inter_max_lon, inter_min_lat),
        (inter_max_lon, inter_max_lat),
        (inter_min_lon, inter_max_lat),
        (inter_min_lon, inter_min_lat),
    ]
    return Polygon(type=GeoJSONType.POLYGON, coordinates=[ring])


def calculate_union(polygons: list[Polygon | dict]) -> Polygon:
    """
    Calculate the bounding-box union of multiple polygons.

    Returns the smallest axis-aligned bounding box that contains all
    input polygons. For exact geometric union, use shapely.

    Args:
        polygons: List of GeoJSON Polygon objects or dicts (minimum 2)

    Returns:
        Polygon: The union bounding-box polygon
    """
    if len(polygons) < 2:
        raise ValueError("At least 2 polygons required for union")

    all_lons: list[float] = []
    all_lats: list[float] = []
    for poly in polygons:
        ext_ring = _get_exterior_ring(poly)
        all_lons.extend(c[0] for c in ext_ring)
        all_lats.extend(c[1] for c in ext_ring)

    u_min_lon, u_max_lon = min(all_lons), max(all_lons)
    u_min_lat, u_max_lat = min(all_lats), max(all_lats)

    union_ring: list[tuple[float, float]] = [
        (u_min_lon, u_min_lat),
        (u_max_lon, u_min_lat),
        (u_max_lon, u_max_lat),
        (u_min_lon, u_max_lat),
        (u_min_lon, u_min_lat),
    ]
    return Polygon(type=GeoJSONType.POLYGON, coordinates=[union_ring])


def calculate_distance(
    polygon1: Polygon | dict,
    polygon2: Polygon | dict,
    method: str = "centroid",
) -> float:
    """
    Calculate the distance between two polygons.

    Args:
        polygon1: First GeoJSON Polygon object or dict
        polygon2: Second GeoJSON Polygon object or dict
        method: Distance calculation method ("centroid", "edge", "vertex")
                Currently only "centroid" is implemented.

    Returns:
        float: Distance in kilometers
    """

    def get_centroid(polygon: Polygon | dict) -> tuple[float, float]:
        """Calculate centroid of a polygon."""
        if isinstance(polygon, Polygon):
            coords = polygon.coordinates[0]
        elif isinstance(polygon, dict) and polygon.get("type") == GeoJSONType.POLYGON:
            coords = polygon.get("coordinates", [[]])[0]
        else:
            raise ValueError("Input must be a GeoJSON Polygon")

        centroid_x = sum(coord[0] for coord in coords) / len(coords)
        centroid_y = sum(coord[1] for coord in coords) / len(coords)
        return centroid_x, centroid_y

    c1 = get_centroid(polygon1)
    c2 = get_centroid(polygon2)

    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]

    # Haversine-lite approximation using avg latitude cosine for lon scaling
    avg_lat = (c1[1] + c2[1]) / 2.0
    km_per_deg_lon = 111.32 * math.cos(math.radians(avg_lat))
    km_per_deg_lat = 111.32

    dist_km = math.sqrt((dx * km_per_deg_lon) ** 2 + (dy * km_per_deg_lat) ** 2)
    return dist_km
