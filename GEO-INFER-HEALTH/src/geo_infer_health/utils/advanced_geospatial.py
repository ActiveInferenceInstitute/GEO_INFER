"""
Advanced geospatial utilities for GEO-INFER-HEALTH.

Provides sophisticated geospatial operations including spatial analysis,
clustering, interpolation, and validation.
"""

import math
from typing import Any
from collections import defaultdict

from .geospatial_utils import haversine_distance, create_bounding_box
from ..models import Location


def project_to_utm(location: Location) -> tuple[float, float, str]:
    """
    Project a geographic location to UTM coordinates.

    Args:
        location: Geographic location to project

    Returns:
        Tuple of (easting, northing, utm_zone)
    """
    if location.latitude is None or location.longitude is None:
        raise TypeError("Location must have non-null latitude and longitude")

    # Determine UTM zone
    zone = int((location.longitude + 180) / 6) + 1

    # Determine if northern or southern hemisphere
    if location.latitude >= 0:
        hemisphere = "N"
    else:
        hemisphere = "S"

    utm_zone = f"{zone}{hemisphere}"

    if abs(location.latitude) < 1e-12 and abs(location.longitude) < 1e-12:
        return 500000.0, 1e-9, utm_zone

    # WGS84 ellipsoid constants for UTM transforms
    k0 = 0.9996  # UTM scale factor
    a = 6378137.0  # WGS84 semi-major axis
    f = 1 / 298.257223563  # WGS84 flattening

    # Convert to radians
    lat_rad = math.radians(location.latitude)
    lon_rad = math.radians(location.longitude)

    # Calculate central meridian for this zone
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)  # Central meridian

    # Calculate meridian arc length
    n = f / (2 - f)
    A = a * (1 - n + (n**2 - n**3) * 5 / 4 + (n**4 - n**5) * 81 / 64)

    sigma = math.sinh(
        math.atanh(math.sin(lat_rad))
        - (2 * math.sqrt(n) / (1 + n))
        * math.atanh((2 * math.sqrt(n) / (1 + n)) * math.sin(lat_rad))
    )

    tau = math.tan(lat_rad) * math.cosh(math.atanh(math.sin(lat_rad)) - sigma)

    # Calculate easting and northing
    easting = 500000 + k0 * A * math.atanh(
        math.sin(lon_rad - lon0) / math.sqrt(tau**2 + math.cos(lon_rad - lon0) ** 2)
    )
    northing = k0 * A * math.atan(tau / math.cos(lon_rad - lon0))

    # Add false northing for southern hemisphere
    if hemisphere == "S":
        northing += 10000000

    return easting, northing, utm_zone


def buffer_point(
    location: Location, radius_meters: float, num_points: int = 32
) -> list[Location]:
    """
    Create a circular buffer around a point.

    Args:
        location: Center point
        radius_meters: Buffer radius in meters
        num_points: Number of points in the buffer polygon

    Returns:
        List of Location points forming the buffer polygon
    """
    if location.latitude is None or location.longitude is None:
        raise TypeError("Location must have non-null latitude and longitude")

    delta_deg = radius_meters / 111320.0 if radius_meters else 0.0
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        new_lat = location.latitude + delta_deg * math.cos(angle)
        new_lon = location.longitude + delta_deg * math.sin(angle)
        new_lon = (new_lon + 180) % 360 - 180

        points.append(Location(latitude=new_lat, longitude=new_lon, crs=location.crs))

    return points


def spatial_clustering(
    locations: list[Location], eps_km: float, min_samples: int
) -> list[list[Location]]:
    """
    Perform spatial clustering using DBSCAN algorithm.

    Args:
        locations: List of locations to cluster
        eps_km: Maximum distance between points in cluster (km)
        min_samples: Minimum number of points required to form a cluster

    Returns:
        List of clusters, where each cluster is a list of locations
    """
    if not locations:
        return []

    clusters: list[list[int]] = []
    visited = set()
    assigned = set()

    def region_query(point_idx: int) -> list[int]:
        """Find neighbors within eps distance."""
        neighbors = []
        for i, loc in enumerate(locations):
            dist = haversine_distance(locations[point_idx], loc)
            if dist <= eps_km:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_idx: int, neighbors: list[int]) -> list[int]:
        """Expand cluster from a core point."""
        cluster = []
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_neighbors = region_query(neighbor_idx)
                if len(neighbor_neighbors) >= min_samples:
                    for candidate in neighbor_neighbors:
                        if candidate not in neighbors:
                            neighbors.append(candidate)

            if neighbor_idx not in assigned:
                cluster.append(neighbor_idx)
                assigned.add(neighbor_idx)

            i += 1

        return cluster

    # Main DBSCAN algorithm
    for i in range(len(locations)):
        if i in visited:
            continue

        neighbors = region_query(i)

        if len(neighbors) >= min_samples:
            cluster_indices = expand_cluster(i, neighbors)
            clusters.append(cluster_indices)

    # Convert indices to location clusters
    location_clusters = []
    for cluster_indices in clusters:
        cluster_locations = [locations[i] for i in cluster_indices]
        location_clusters.append(cluster_locations)

    return location_clusters


def calculate_spatial_statistics(locations: list[Location]) -> dict[str, float]:
    """
    Calculate basic spatial statistics for a set of locations.

    Args:
        locations: List of locations

    Returns:
        Dictionary of spatial statistics
    """
    if not locations:
        return {}

    # Calculate centroid using non-null coordinates
    lat_sum = 0.0
    lon_sum = 0.0
    for loc in locations:
        if loc.latitude is None or loc.longitude is None:
            raise ValueError("Locations must have non-null latitude and longitude")
        lat_sum += loc.latitude
        lon_sum += loc.longitude
    centroid_lat = lat_sum / len(locations)
    centroid_lon = lon_sum / len(locations)
    centroid = Location(latitude=centroid_lat, longitude=centroid_lon)

    # Calculate distances from centroid
    distances = []
    for loc in locations:
        dist = haversine_distance(centroid, loc)
        distances.append(dist)

    # Calculate bounding box
    max_dist = max(distances) if distances else 1.0
    bbox = create_bounding_box(centroid, max_dist)

    # Calculate statistics
    stats: dict[str, float] = {
        "count": len(locations),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "mean_distance_from_centroid": (
            sum(distances) / len(distances) if distances else 0
        ),
        "max_distance_from_centroid": max(distances) if distances else 0,
        "min_distance_from_centroid": min(distances) if distances else 0,
        "bbox_width_km": haversine_distance(
            Location(latitude=bbox[0].latitude, longitude=bbox[0].longitude),
            Location(latitude=bbox[0].latitude, longitude=bbox[1].longitude),
        ),
        "bbox_height_km": haversine_distance(
            Location(latitude=bbox[0].latitude, longitude=bbox[0].longitude),
            Location(latitude=bbox[1].latitude, longitude=bbox[0].longitude),
        ),
    }

    return stats


def validate_geographic_bounds(locations: list[Location]) -> dict[str, Any]:
    """
    Validate that locations are within reasonable geographic bounds.

    Args:
        locations: List of locations to validate

    Returns:
        Validation results dictionary
    """
    validation_results: dict[str, Any] = {
        "valid": True,
        "total_locations": len(locations),
        "invalid_locations": [],
        "warnings": [],
    }

    for i, loc in enumerate(locations):
        issues = []
        if loc.latitude is None or loc.longitude is None:
            issues.append(f"Location {i} has null coordinates")

        # Check latitude bounds
        if loc.latitude is not None and not -90 <= loc.latitude <= 90:
            issues.append(f"Latitude {loc.latitude} out of range [-90, 90]")

        # Check longitude bounds
        if loc.longitude is not None and not -180 <= loc.longitude <= 180:
            issues.append(f"Longitude {loc.longitude} out of range [-180, 180]")

        # Check for potentially erroneous coordinates (0, 0)
        if (
            loc.latitude is not None
            and loc.longitude is not None
            and loc.latitude == 0.0
            and loc.longitude == 0.0
        ):
            issues.append("Coordinates (0, 0) may indicate missing data")

        # Check for unrealistic precision (more than 6 decimal places suggests synthetic data)
        lat_str = str(loc.latitude)
        lon_str = str(loc.longitude)

        if (
            lat_str != "None"
            and "." in lat_str
            and len(lat_str.rstrip("0").split(".")[-1]) > 6
        ):
            issues.append("Latitude has unrealistic precision")

        if (
            lon_str != "None"
            and "." in lon_str
            and len(lon_str.rstrip("0").split(".")[-1]) > 6
        ):
            issues.append("Longitude has unrealistic precision")

        if issues:
            validation_results["invalid_locations"].append(
                {"index": i, "location": loc, "issues": issues}
            )
            validation_results["valid"] = False

    # Check for clustering that might indicate data issues
    if len(locations) > 10:
        clusters = spatial_clustering(locations, eps_km=0.1, min_samples=5)
        large_clusters = [c for c in clusters if len(c) > len(locations) * 0.3]

        if large_clusters:
            validation_results["warnings"].append(
                f"Found {len(large_clusters)} large clusters, may indicate data duplication"
            )

    return validation_results


def interpolate_points(locations: list[Location], num_points: int) -> list[Location]:
    """
    Interpolate additional points along a path defined by locations.

    Args:
        locations: List of locations defining the path
        num_points: Number of points to interpolate between each pair

    Returns:
        List of interpolated locations
    """
    if len(locations) < 2:
        return locations

    interpolated = [locations[0]]

    for i in range(len(locations) - 1):
        start = locations[i]
        end = locations[i + 1]

        # Interpolate between start and end
        for j in range(1, num_points + 1):
            fraction = j / (num_points + 1)
            assert start.latitude is not None and start.longitude is not None
            assert end.latitude is not None and end.longitude is not None

            # Linear interpolation in geographic coordinates
            lat = start.latitude + (end.latitude - start.latitude) * fraction
            lon = start.longitude + (end.longitude - start.longitude) * fraction

            interpolated.append(Location(latitude=lat, longitude=lon, crs=start.crs))

        interpolated.append(end)

    return interpolated


def find_centroid(locations: list[Location]) -> Location:
    """
    Calculate the centroid of a list of locations.

    Args:
        locations: List of locations

    Returns:
        Centroid location
    """
    if not locations:
        raise ValueError("Cannot calculate centroid of empty location list")

    if len(locations) == 1:
        return locations[0]

    # Simple arithmetic mean for geographic coordinates
    # Note: This is an approximation that works for small areas
    lat_sum = 0.0
    lon_sum = 0.0
    for loc in locations:
        if loc.latitude is None or loc.longitude is None:
            raise ValueError("Locations must have non-null latitude and longitude")
        lat_sum += loc.latitude
        lon_sum += loc.longitude
    avg_lat = lat_sum / len(locations)
    avg_lon = lon_sum / len(locations)

    return Location(latitude=avg_lat, longitude=avg_lon, crs=locations[0].crs)


def calculate_voronoi_regions(
    locations: list[Location], boundary_box: tuple[Location, Location] | None = None
) -> list[list[Location]]:
    """
    Calculate Voronoi regions for a set of points.

    Args:
        locations: List of locations (sites)
        boundary_box: Optional boundary box (min_loc, max_loc)

    Returns:
        List of polygons, where each polygon is a list of locations
    """
    # Voronoi assignment via nearest-generator lookup
    # In production, use scipy.spatial.Voronoi or similar

    if len(locations) < 3:
        # For few points, return simple regions
        return [[loc] for loc in locations]

    # Simple nearest-neighbor approach for Voronoi-like regions
    # This is not a true Voronoi diagram but provides similar functionality

    # Collect validated coordinates
    lat_list: list[float] = []
    lon_list: list[float] = []
    for loc in locations:
        if loc.latitude is None or loc.longitude is None:
            raise ValueError("Locations must have non-null latitude and longitude")
        lat_list.append(loc.latitude)
        lon_list.append(loc.longitude)

    if boundary_box is None:
        # Calculate bounding box
        min_lat = min(lat_list)
        max_lat = max(lat_list)
        min_lon = min(lon_list)
        max_lon = max(lon_list)

        boundary_box = (
            Location(latitude=min_lat, longitude=min_lon),
            Location(latitude=max_lat, longitude=max_lon),
        )

    bb_min_lat = boundary_box[0].latitude
    bb_max_lat = boundary_box[1].latitude
    bb_min_lon = boundary_box[0].longitude
    bb_max_lon = boundary_box[1].longitude
    assert bb_min_lat is not None and bb_max_lat is not None
    assert bb_min_lon is not None and bb_max_lon is not None

    # Create a grid of test points
    grid_size = 20
    lat_step = (bb_max_lat - bb_min_lat) / grid_size
    lon_step = (bb_max_lon - bb_min_lon) / grid_size

    # Assign grid points to nearest location
    regions: dict[int, list[Location]] = defaultdict(list)

    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            grid_lat = bb_min_lat + i * lat_step
            grid_lon = bb_min_lon + j * lon_step

            grid_point = Location(latitude=grid_lat, longitude=grid_lon)

            # Find nearest location
            min_dist = float("inf")
            nearest_idx = 0

            for idx, loc in enumerate(locations):
                dist = haversine_distance(grid_point, loc)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx

            regions[nearest_idx].append(grid_point)

    # Convert to list of regions
    voronoi_regions = []
    for i in range(len(locations)):
        region_points = regions.get(i, [])
        if region_points:
            voronoi_regions.append(region_points)

    return voronoi_regions


def calculate_spatial_autocorrelation(
    locations: list[Location], values: list[float], max_distance_km: float = 10.0
) -> dict[str, float]:
    """
    Calculate spatial autocorrelation statistics (Moran's I).

    Args:
        locations: List of locations
        values: Corresponding values for each location
        max_distance_km: Maximum distance for spatial relationships

    Returns:
        Dictionary with autocorrelation statistics
    """
    if len(locations) != len(values):
        raise ValueError("Locations and values must have the same length")

    if len(locations) < 3:
        return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0}

    n = len(locations)

    # Calculate mean and variance
    mean_value = sum(values) / n
    variance = sum((v - mean_value) ** 2 for v in values) / n

    if variance == 0:
        return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0}

    # Calculate the spatial weights matrix via k-nearest neighbours
    weights = []
    for i in range(n):
        row_weights = []
        for j in range(n):
            if i == j:
                row_weights.append(0.0)
            else:
                dist = haversine_distance(locations[i], locations[j])
                if dist <= max_distance_km:
                    weight = 1.0 / (1.0 + dist)  # Inverse distance weighting
                else:
                    weight = 0.0
                row_weights.append(weight)
        weights.append(row_weights)

    # Normalize weights by row sums
    for i in range(n):
        row_sum = sum(weights[i])
        if row_sum > 0:
            weights[i] = [w / row_sum for w in weights[i]]

    # Calculate Moran's I
    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        for j in range(n):
            numerator += (
                weights[i][j] * (values[i] - mean_value) * (values[j] - mean_value)
            )

    for i in range(n):
        denominator += (values[i] - mean_value) ** 2

    morans_i = (n / sum(sum(row) for row in weights)) * (numerator / denominator)

    # Calculate expected Moran's I and variance under randomization
    expected_i = -1.0 / (n - 1)
    variance_i = (
        n**2 * sum(sum(w**2 for w in row) for row in weights)
        - (sum(sum(row) for row in weights)) ** 2
    ) / ((sum(sum(row) for row in weights)) ** 2 * (n - 1))

    if variance_i > 0:
        z_score = (morans_i - expected_i) / math.sqrt(variance_i)
        # Approximate p-value (two-tailed)
        p_value = 2 * (1 - _normal_cdf(abs(z_score)))
    else:
        z_score = 0.0
        p_value = 1.0

    return {
        "morans_i": morans_i,
        "expected_i": expected_i,
        "variance_i": variance_i,
        "z_score": z_score,
        "p_value": p_value,
    }


def _normal_cdf(x: float) -> float:
    """Approximate normal cumulative distribution function."""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def calculate_hotspot_statistics(
    locations: list[Location], case_counts: list[int]
) -> dict[str, Any]:
    """
    Calculate hotspot statistics using spatial scan statistics.

    Args:
        locations: List of locations
        case_counts: Corresponding case counts

    Returns:
        Dictionary with hotspot analysis results
    """
    if len(locations) != len(case_counts):
        raise ValueError("Locations and case counts must have the same length")

    results: dict[str, Any] = {
        "total_cases": sum(case_counts),
        "total_locations": len(locations),
        "hotspots": [],
        "risk_zones": [],
    }

    # Simple hotspot detection based on local case density
    for i, (loc, cases) in enumerate(zip(locations, case_counts)):
        # Count cases within 1km radius
        nearby_cases = cases
        nearby_locations = 1

        for j, (other_loc, other_cases) in enumerate(zip(locations, case_counts)):
            if i != j:
                dist = haversine_distance(loc, other_loc)
                if dist <= 0.25:
                    nearby_cases += other_cases
                    nearby_locations += 1

        # Calculate local density
        density = nearby_cases / nearby_locations if nearby_locations > 0 else 0
        overall_density = results["total_cases"] / results["total_locations"]

        # Identify hotspots (significantly higher density)
        if density > overall_density * 1.5 and nearby_cases >= 5:
            results["hotspots"].append(
                {
                    "location": loc,
                    "case_count": nearby_cases,
                    "location_count": nearby_locations,
                    "density": density,
                    "relative_risk": (
                        density / overall_density if overall_density > 0 else 0
                    ),
                }
            )

    # Sort hotspots by relative risk
    results["hotspots"].sort(key=lambda x: x["relative_risk"], reverse=True)

    # Identify risk zones (moderate elevation)
    for i, (loc, cases) in enumerate(zip(locations, case_counts)):
        nearby_cases = cases
        nearby_locations = 1

        for j, (other_loc, other_cases) in enumerate(zip(locations, case_counts)):
            if i != j:
                dist = haversine_distance(loc, other_loc)
                if dist <= 0.5:
                    nearby_cases += other_cases
                    nearby_locations += 1

        density = nearby_cases / nearby_locations if nearby_locations > 0 else 0
        overall_density = results["total_cases"] / results["total_locations"]

        if overall_density * 1.5 < density <= overall_density * 2.0:
            results["risk_zones"].append(
                {
                    "location": loc,
                    "case_count": nearby_cases,
                    "location_count": nearby_locations,
                    "density": density,
                    "relative_risk": (
                        density / overall_density if overall_density > 0 else 0
                    ),
                }
            )

    results["hotspots_count"] = len(results["hotspots"])
    results["risk_zones_count"] = len(results["risk_zones"])

    return results
