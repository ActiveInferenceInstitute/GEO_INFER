"""Shared geodesic helpers.

Single owner for geo distance calculations used across the emergency
management core (resource travel-time estimation, SAR pattern metrics).
"""

import math

# Mean Earth radius in kilometres (spherical approximation).
EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    point_a: dict[str, float], point_b: dict[str, float]
) -> float:
    """
    Great-circle distance between two points in kilometres.

    Args:
        point_a: {"lat": float, "lon": float} in decimal degrees
        point_b: {"lat": float, "lon": float} in decimal degrees

    Returns:
        Distance in kilometres (Haversine formula, spherical Earth).
    """
    for name, point in (("point_a", point_a), ("point_b", point_b)):
        missing = [key for key in ("lat", "lon") if key not in point]
        if missing:
            raise ValueError(
                f"haversine_distance_km: {name} is missing coordinate key(s) "
                f"{missing}; found keys {sorted(point)}. Both 'lat' and 'lon' "
                "are required in decimal degrees."
            )

    lat1 = math.radians(point_a["lat"])
    lon1 = math.radians(point_a["lon"])
    lat2 = math.radians(point_b["lat"])
    lon2 = math.radians(point_b["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c
