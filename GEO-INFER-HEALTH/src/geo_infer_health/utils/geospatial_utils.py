import math
from geo_infer_health.models import Location

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(loc1: Location, loc2: Location) -> float:
    """Calculate the Haversine distance between two points on the Earth.

    Args:
        loc1: The first location.
        loc2: The second location.

    Returns:
        The distance in kilometers.
    """
    if (
        loc1.latitude is None
        or loc1.longitude is None
        or loc2.latitude is None
        or loc2.longitude is None
    ):
        raise TypeError("Locations must have non-null latitude and longitude")

    lat1_rad = math.radians(loc1.latitude)
    lon1_rad = math.radians(loc1.longitude)
    lat2_rad = math.radians(loc2.latitude)
    lon2_rad = math.radians(loc2.longitude)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS_KM * c
    return distance


def create_bounding_box(
    center_loc: Location, distance_km: float
) -> tuple[Location, Location]:
    """Creates a square bounding box around a central point.

    Args:
        center_loc: The center location.
        distance_km: The half side length of the square in kilometers.

    Returns:
        A tuple containing the bottom-left and top-right Location objects of the bounding box.
    """
    if center_loc.latitude is None or center_loc.longitude is None:
        raise TypeError("Center location must have non-null latitude and longitude")

    lat_rad = math.radians(center_loc.latitude)
    lon_rad = math.radians(center_loc.longitude)

    # Angular distance in radians
    angular_dist = distance_km / EARTH_RADIUS_KM

    # Calculate latitude bounds
    min_lat_rad = lat_rad - angular_dist
    max_lat_rad = lat_rad + angular_dist

    # Calculate longitude bounds (more complex due to convergence at poles)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-12:
        delta_lon_rad = math.pi
    else:
        ratio = math.sin(abs(angular_dist)) / abs(cos_lat)
        delta_lon_rad = math.asin(min(1.0, max(-1.0, ratio)))
    min_lon_rad = lon_rad - delta_lon_rad
    max_lon_rad = lon_rad + delta_lon_rad

    min_lat = math.degrees(min_lat_rad)
    max_lat = math.degrees(max_lat_rad)
    min_lon = math.degrees(min_lon_rad)
    max_lon = math.degrees(max_lon_rad)

    # Ensure latitudes are within valid range [-90, 90]
    min_lat = max(min_lat, -90.0)
    max_lat = min(max_lat, 90.0)

    # Ensure longitudes are within valid range [-180, 180] (can wrap around)
    min_lon = (min_lon + 180) % 360 - 180
    max_lon = (max_lon + 180) % 360 - 180

    # Handle cases where bounding box crosses the antimeridian
    if min_lon > max_lon:
        # This simple bounding box does not explicitly handle antimeridian crossing for complex queries.
        # For sophisticated geospatial queries crossing the antimeridian, a GIS library is recommended.
        min_lon, max_lon = -180.0, 180.0

    return (
        Location(latitude=min_lat, longitude=min_lon, crs=center_loc.crs),
        Location(latitude=max_lat, longitude=max_lon, crs=center_loc.crs),
    )


# Baseline for more advanced geo-utils, e.g., using geopandas or shapely if available
# def project_to_utm(location: Location) -> Tuple[float, float, str]:
#     pass

# def buffer_point(location: Location, radius_meters: float) -> Any:
#     pass # Returns a geometry object
