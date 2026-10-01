"""Geospatial helpers for GEO-INFER-SEC.

Small, dependency-light geometric utilities shared by the security
domain managers (physical zones, perimeter calculations).
"""

import math

from shapely.geometry import Point, Polygon

_EARTH_RADIUS_M = 6_371_000.0


def _destination_point(
    lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """Great-circle destination point from a start, bearing, and distance.

    Implements the standard spherical single-leg formula (same family as the
    haversine problem inverted). Accurate to well under a metre for the
    short radii used by security zones.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    angular = distance_m / _EARTH_RADIUS_M

    dest_lat = math.asin(
        math.sin(lat) * math.cos(angular)
        + math.cos(lat) * math.sin(angular) * math.cos(bearing)
    )
    dest_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat),
        math.cos(angular) - math.sin(lat) * math.sin(dest_lat),
    )
    return math.degrees(dest_lat), math.degrees(dest_lon)


class GeoSpatialUtils:
    """Geometric helpers for security-zone construction."""

    def create_circle(
        self,
        center: Point,
        radius_m: float,
        num_points: int | None = 64,
    ) -> Polygon:
        """Create a circle of ``radius_m`` metres around ``center``.

        Args:
            center: Center point in EPSG:4326 degrees (x = longitude,
                y = latitude).
            radius_m: Circle radius in metres. Must be finite and positive.
            num_points: Number of polygon vertices sampled around the circle.

        Returns:
            A shapely Polygon approximating the metric circle.

        Raises:
            ValueError: If ``radius_m`` is not a positive finite number or
                ``num_points`` is smaller than 3.
        """
        if not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("radius_m must be a positive finite number")
        if num_points is None or num_points < 3:
            raise ValueError("num_points must be at least 3")

        lat, lon = center.y, center.x
        step = 360.0 / num_points
        ring = [
            _destination_point(lat, lon, bearing, radius_m)
            for bearing in (i * step for i in range(num_points))
        ]
        # Close the ring explicitly.
        ring.append(ring[0])
        return Polygon([(lon, lat) for lat, lon in ring])
