"""
Geometry Module

This module provides functions and classes for geometric operations
on geospatial data, including calculations for distances, areas,
intersections, and other geometric properties.
"""

import numpy as np
from typing import cast
from dataclasses import dataclass

# Constants for Earth calculations
EARTH_RADIUS_KM = 6371.0  # Mean radius in kilometers
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000  # Mean radius in meters
WGS84_SEMI_MAJOR_AXIS = 6378137.0  # WGS84 semi-major axis in meters
WGS84_SEMI_MINOR_AXIS = 6356752.314245  # WGS84 semi-minor axis in meters
WGS84_FLATTENING = 1 / 298.257223563  # WGS84 flattening


@dataclass
class Point:
    """Representation of a 2D point with optional z-coordinate."""

    x: float
    y: float
    z: float | None = None

    def distance_to(self, other: "Point") -> float:
        """Calculate Euclidean distance to another point."""
        if self.z is not None and other.z is not None:
            return float(
                np.sqrt(
                    (self.x - other.x) ** 2
                    + (self.y - other.y) ** 2
                    + (self.z - other.z) ** 2
                )
            )
        else:
            return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        if self.z is not None:
            return np.array([self.x, self.y, self.z])
        else:
            return np.array([self.x, self.y])


@dataclass
class LineString:
    """Representation of a line string (sequence of points)."""

    points: list[Point]

    def length(self) -> float:
        """Calculate the length of the line string."""
        if len(self.points) < 2:
            return 0.0

        length = 0.0
        for i in range(len(self.points) - 1):
            length += self.points[i].distance_to(self.points[i + 1])

        return length

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        if len(self.points) == 0:
            return np.array([])

        if self.points[0].z is not None:
            return np.array([[p.x, p.y, p.z] for p in self.points])
        else:
            return np.array([[p.x, p.y] for p in self.points])


@dataclass
class Polygon:
    """Representation of a polygon (exterior ring and optional interior rings)."""

    exterior: list[Point]
    interiors: list[list[Point]] | None = None

    def area(self) -> float:
        """Calculate the area of the polygon using Shoelace formula."""
        if len(self.exterior) < 3:
            return 0.0

        # Exterior ring area
        area = 0.0
        for i in range(len(self.exterior)):
            j = (i + 1) % len(self.exterior)
            area += self.exterior[i].x * self.exterior[j].y
            area -= self.exterior[j].x * self.exterior[i].y

        area = abs(area) / 2.0

        # Subtract interior rings
        if self.interiors:
            for interior in self.interiors:
                interior_area = 0.0
                for i in range(len(interior)):
                    j = (i + 1) % len(interior)
                    interior_area += interior[i].x * interior[j].y
                    interior_area -= interior[j].x * interior[i].y

                area -= abs(interior_area) / 2.0

        return area

    def centroid(self) -> Point:
        """Calculate the area-weighted centroid of the polygon (holes subtracted)."""
        if len(self.exterior) < 3:
            raise ValueError("Polygon must have at least 3 points")

        def ring_contribution(ring: list[Point]) -> tuple[float, float, float]:
            """Return (weighted_cx, weighted_cy, abs_area) for one ring."""
            cx = cy = a2 = 0.0
            n = len(ring)
            for i in range(n):
                j = (i + 1) % n
                cross = ring[i].x * ring[j].y - ring[j].x * ring[i].y
                a2 += cross
                cx += (ring[i].x + ring[j].x) * cross
                cy += (ring[i].y + ring[j].y) * cross
            area = abs(a2) / 2.0
            # Ring centroid; sign of a2 cancels in the ratio.
            if a2 == 0:
                return 0.0, 0.0, 0.0
            return cx / (3.0 * a2) * area, cy / (3.0 * a2) * area, area

        total_area = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        cx, cy, area = ring_contribution(self.exterior)
        total_area += area
        weighted_x += cx
        weighted_y += cy

        for interior in self.interiors or []:
            cx, cy, area = ring_contribution(interior)
            total_area -= area
            weighted_x -= cx
            weighted_y -= cy

        if total_area <= 0.0:
            raise ValueError("Polygon has zero area; centroid is undefined")

        return Point(x=weighted_x / total_area, y=weighted_y / total_area)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the Earth's surface.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance in kilometers
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    # Return distance in kilometers
    return float(EARTH_RADIUS_KM * c)


def vincenty_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> float:
    """
    Calculate the geodesic distance between two points using Vincenty's formula.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        max_iterations: Maximum number of iterations for convergence
        tolerance: Convergence tolerance

    Returns:
        Distance in meters

    Raises:
        ValueError: If the iteration loop exhausts without converging (classic
            Vincenty failure for near-antipodal points) instead of silently
            returning an unconverged result.
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    f = WGS84_FLATTENING

    L = lon2_rad - lon1_rad

    U1 = np.arctan((1 - f) * np.tan(lat1_rad))
    U2 = np.arctan((1 - f) * np.tan(lat2_rad))

    sin_U1 = np.sin(U1)
    cos_U1 = np.cos(U1)
    sin_U2 = np.sin(U2)
    cos_U2 = np.cos(U2)

    # Initial value
    lambda_old = L
    converged = False

    for _ in range(max_iterations):
        sin_lambda = np.sin(lambda_old)
        cos_lambda = np.cos(lambda_old)

        sin_sigma = np.sqrt(
            (cos_U2 * sin_lambda) ** 2
            + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2
        )

        if sin_sigma == 0:
            return 0.0  # Coincident points

        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2

        if cos_sq_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

        lambda_new = L + (1 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )

        if abs(lambda_new - lambda_old) < tolerance:
            converged = True
            break

        lambda_old = lambda_new

    if not converged:
        raise ValueError(
            "vincenty_distance failed to converge after "
            f"{max_iterations} iterations (tolerance={tolerance}); "
            "points are likely near-antipodal"
        )

    u2 = cos_sq_alpha * ((a**2 - b**2) / b**2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + B
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                - B
                / 6
                * cos_2sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_2sigma_m**2)
            )
        )
    )

    # Return distance in meters
    return float(b * A * (sigma - delta_sigma))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing from point 1 to point 2.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(lon2_rad - lon1_rad)

    bearing_rad = np.arctan2(y, x)

    # Convert to degrees and normalize to 0-360
    bearing_deg = np.degrees(bearing_rad)
    return float((bearing_deg + 360) % 360)


def destination_point(
    lat: float, lon: float, bearing: float, distance: float
) -> tuple[float, float]:
    """
    Calculate the destination point given a starting point, bearing, and distance.

    Args:
        lat: Latitude of starting point in degrees
        lon: Longitude of starting point in degrees
        bearing: Bearing in degrees
        distance: Distance in kilometers

    Returns:
        Tuple of (latitude, longitude) of destination point in degrees
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)

    # Angular distance
    angular_dist = distance / EARTH_RADIUS_KM

    # Calculate destination point
    sin_lat = np.sin(lat_rad) * np.cos(angular_dist) + np.cos(lat_rad) * np.sin(
        angular_dist
    ) * np.cos(bearing_rad)
    dest_lat_rad = np.arcsin(sin_lat)

    y = np.sin(bearing_rad) * np.sin(angular_dist) * np.cos(lat_rad)
    x = np.cos(angular_dist) - np.sin(lat_rad) * np.sin(dest_lat_rad)
    dest_lon_rad = lon_rad + np.arctan2(y, x)

    # Normalize longitude to -180 to 180
    dest_lon_rad = ((dest_lon_rad + 3 * np.pi) % (2 * np.pi)) - np.pi

    # Convert back to degrees
    dest_lat = np.degrees(dest_lat_rad)
    dest_lon = np.degrees(dest_lon_rad)

    return (dest_lat, dest_lon)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Determine if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: The point to test
        polygon: The polygon to test against

    Returns:
        True if the point is inside the polygon, False otherwise
    """
    inside = False
    j = len(polygon.exterior) - 1

    for i in range(len(polygon.exterior)):
        if (
            (polygon.exterior[i].y > point.y) != (polygon.exterior[j].y > point.y)
        ) and (
            point.x
            < (polygon.exterior[j].x - polygon.exterior[i].x)
            * (point.y - polygon.exterior[i].y)
            / (polygon.exterior[j].y - polygon.exterior[i].y)
            + polygon.exterior[i].x
        ):
            inside = not inside
        j = i

    # Check if point is in any holes (interior rings)
    if inside and polygon.interiors:
        for interior in polygon.interiors:
            j = len(interior) - 1
            in_hole = False

            for i in range(len(interior)):
                if ((interior[i].y > point.y) != (interior[j].y > point.y)) and (
                    point.x
                    < (interior[j].x - interior[i].x)
                    * (point.y - interior[i].y)
                    / (interior[j].y - interior[i].y)
                    + interior[i].x
                ):
                    in_hole = not in_hole
                j = i

            if in_hole:
                return False

    return inside


def points_in_polygon_vectorized(
    points_x: np.ndarray,
    points_y: np.ndarray,
    polygon_vertices_x: np.ndarray,
    polygon_vertices_y: np.ndarray,
) -> np.ndarray:
    """
    Vectorized SIMD ray-casting point-in-polygon containment test for N points simultaneously.

    Args:
        points_x: 1D array of point X coordinates (longitudes / eastings)
        points_y: 1D array of point Y coordinates (latitudes / northings)
        polygon_vertices_x: 1D array of polygon boundary vertex X coordinates
        polygon_vertices_y: 1D array of polygon boundary vertex Y coordinates

    Returns:
        1D boolean array of shape (N,) indicating True for points inside the polygon
    """
    px = np.asarray(points_x, dtype=float)
    py = np.asarray(points_y, dtype=float)
    vx = np.asarray(polygon_vertices_x, dtype=float)
    vy = np.asarray(polygon_vertices_y, dtype=float)

    n_points = len(px)
    n_vertices = len(vx)
    if n_vertices < 3 or n_points == 0:
        return np.zeros(n_points, dtype=bool)

    # Ensure closed ring
    if vx[0] != vx[-1] or vy[0] != vy[-1]:
        vx = np.append(vx, vx[0])
        vy = np.append(vy, vy[0])
        n_vertices += 1

    inside = np.zeros(n_points, dtype=bool)

    # Vectorized ray crossing across all polygon segments
    v0_x = vx[:-1]
    v0_y = vy[:-1]
    v1_x = vx[1:]
    v1_y = vy[1:]

    # Shape: (n_points, n_segments)
    py_2d = py[:, np.newaxis]
    px_2d = px[:, np.newaxis]

    cond1 = (v0_y > py_2d) != (v1_y > py_2d)
    denom = v1_y - v0_y
    # Avoid zero division on horizontal edges
    denom_safe = np.where(denom == 0, 1e-12, denom)
    x_intercept = (v1_x - v0_x) * (py_2d - v0_y) / denom_safe + v0_x
    cond2 = px_2d < x_intercept

    crossings = cond1 & cond2
    inside = np.sum(crossings, axis=1) % 2 == 1
    return cast(np.ndarray, inside)


def buffer_point(
    lat: float, lon: float, distance: float, segments: int = 32
) -> list[tuple[float, float]]:
    """
    Create a circular buffer around a point.

    Args:
        lat: Latitude of center point in degrees
        lon: Longitude of center point in degrees
        distance: Buffer distance in kilometers
        segments: Number of segments to use for the circle

    Returns:
        List of (latitude, longitude) tuples forming the buffer polygon
    """
    buffer = []

    for i in range(segments):
        angle = (360.0 / segments) * i
        buffer.append(destination_point(lat, lon, angle, distance))

    # Close the ring
    buffer.append(buffer[0])

    return buffer


def line_intersection(
    line1_start: Point, line1_end: Point, line2_start: Point, line2_end: Point
) -> Point | None:
    """
    Find the intersection point of two line segments.

    Args:
        line1_start: Start point of first line
        line1_end: End point of first line
        line2_start: Start point of second line
        line2_end: End point of second line

    Returns:
        Intersection point, or None if lines don't intersect
    """
    # Line 1 represented as a1x + b1y = c1
    a1 = line1_end.y - line1_start.y
    b1 = line1_start.x - line1_end.x
    c1 = a1 * line1_start.x + b1 * line1_start.y

    # Line 2 represented as a2x + b2y = c2
    a2 = line2_end.y - line2_start.y
    b2 = line2_start.x - line2_end.x
    c2 = a2 * line2_start.x + b2 * line2_start.y

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None  # Lines are parallel

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    # Check if intersection point lies within both line segments
    def is_on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            q.x <= max(p.x, r.x)
            and q.x >= min(p.x, r.x)
            and q.y <= max(p.y, r.y)
            and q.y >= min(p.y, r.y)
        )

    intersection = Point(x=x, y=y)

    if is_on_segment(line1_start, intersection, line1_end) and is_on_segment(
        line2_start, intersection, line2_end
    ):
        return intersection

    return None


def polygon_area_spherical(polygon: list[tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon on the Earth's surface.

    Args:
        polygon: List of (latitude, longitude) tuples in degrees

    Returns:
        Area in square kilometers
    """
    if len(polygon) < 3:
        return 0.0

    # Convert to radians
    polygon_rad = [(np.radians(lat), np.radians(lon)) for lat, lon in polygon]

    area = 0.0
    for i in range(len(polygon_rad)):
        j = (i + 1) % len(polygon_rad)

        lat1, lon1 = polygon_rad[i]
        lat2, lon2 = polygon_rad[j]

        area += (lon2 - lon1) * (2 + np.sin(lat1) + np.sin(lat2))

    area = abs(area * EARTH_RADIUS_KM**2 / 2.0)

    return float(area)


def great_circle_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Calculate the great circle distance between arrays of points.

    Args:
        coords1: Array of [lat, lon] coordinates in degrees (n x 2)
        coords2: Array of [lat, lon] coordinates in degrees (m x 2)

    Returns:
        Distance matrix (n x m) in kilometers
    """
    n = coords1.shape[0]
    m = coords2.shape[0]

    # Convert to radians
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)

    # Extract latitudes and longitudes
    lat1 = coords1_rad[:, 0].reshape(n, 1)
    lon1 = coords1_rad[:, 1].reshape(n, 1)
    lat2 = coords2_rad[:, 0].reshape(1, m)
    lon2 = coords2_rad[:, 1].reshape(1, m)

    # Compute components
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return cast(np.ndarray, EARTH_RADIUS_KM * c)


__all__ = [
    "Point",
    "LineString",
    "Polygon",
    "haversine_distance",
    "vincenty_distance",
    "bearing",
    "destination_point",
    "point_in_polygon",
    "buffer_point",
    "line_intersection",
    "polygon_area_spherical",
    "great_circle_distance",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_M",
    "WGS84_SEMI_MAJOR_AXIS",
    "WGS84_SEMI_MINOR_AXIS",
    "WGS84_FLATTENING",
]
