"""
Unit tests for advanced geospatial utilities.
"""

import pytest
import math
import numpy as np

from geo_infer_health.models import Location
from geo_infer_health.utils.advanced_geospatial import (
    project_to_utm,
    buffer_point,
    spatial_clustering,
    calculate_spatial_statistics,
    validate_geographic_bounds,
    interpolate_points,
    find_centroid,
    calculate_spatial_autocorrelation,
    calculate_hotspot_statistics,
)


def invalid_location(latitude, longitude):
    """Construct intentionally invalid coordinates for validation utility tests."""
    return Location.model_construct(
        latitude=latitude, longitude=longitude, crs="EPSG:4326"
    )


class TestUTMProjection:
    """Test UTM projection functionality."""

    def test_utm_projection_northern_hemisphere(self):
        """Test UTM projection for northern hemisphere."""
        location = Location(latitude=34.0522, longitude=-118.2437)  # Los Angeles

        easting, northing, zone = project_to_utm(location)

        assert isinstance(easting, float)
        assert isinstance(northing, float)
        assert isinstance(zone, str)
        assert "N" in zone  # Northern hemisphere

        # Check reasonable ranges for UTM coordinates
        assert easting > 0
        assert northing > 0

    def test_utm_projection_southern_hemisphere(self):
        """Test UTM projection for southern hemisphere."""
        location = Location(latitude=-33.8688, longitude=151.2093)  # Sydney

        easting, northing, zone = project_to_utm(location)

        assert isinstance(easting, float)
        assert isinstance(northing, float)
        assert isinstance(zone, str)
        assert "S" in zone  # Southern hemisphere

        # Check reasonable ranges
        assert easting > 0
        assert northing > 0

    def test_utm_projection_equator(self):
        """Test UTM projection at equator."""
        location = Location(latitude=0.0, longitude=0.0)

        easting, northing, zone = project_to_utm(location)

        assert easting == pytest.approx(
            500000, abs=1000
        )  # Should be close to central meridian
        assert northing > 0
        assert "N" in zone


class TestPointBuffering:
    """Test point buffering functionality."""

    def test_buffer_point_basic(self):
        """Test basic point buffering."""
        center = Location(latitude=34.0522, longitude=-118.2437)
        radius_meters = 1000  # 1km
        num_points = 8

        buffer_points = buffer_point(center, radius_meters, num_points)

        assert len(buffer_points) == num_points

        # Check that all points are approximately the right distance from center
        for point in buffer_points:
            distance = math.sqrt(
                (point.latitude - center.latitude) ** 2
                + (point.longitude - center.longitude) ** 2
            )
            # Convert to approximate meters (rough approximation)
            distance_meters = distance * 111320  # ~111km per degree
            assert distance_meters == pytest.approx(radius_meters, abs=100)

    def test_buffer_point_different_radii(self):
        """Test point buffering with different radii."""
        center = Location(latitude=34.0522, longitude=-118.2437)
        radii = [500, 1000, 2000]  # meters

        for radius in radii:
            buffer_points = buffer_point(center, radius, num_points=4)

            # Check average distance
            distances = []
            for point in buffer_points:
                # Rough distance calculation
                distance_meters = (
                    math.sqrt(
                        (point.latitude - center.latitude) ** 2
                        + (point.longitude - center.longitude) ** 2
                    )
                    * 111320
                )
                distances.append(distance_meters)

            avg_distance = sum(distances) / len(distances)
            assert avg_distance == pytest.approx(radius, abs=200)

    def test_buffer_point_zero_radius(self):
        """Test point buffering with zero radius."""
        center = Location(latitude=34.0522, longitude=-118.2437)

        buffer_points = buffer_point(center, 0, num_points=4)

        # Should return points very close to center
        for point in buffer_points:
            distance_meters = (
                math.sqrt(
                    (point.latitude - center.latitude) ** 2
                    + (point.longitude - center.longitude) ** 2
                )
                * 111320
            )
            assert distance_meters < 1  # Less than 1 meter


class TestSpatialClustering:
    """Test spatial clustering functionality."""

    def test_spatial_clustering_basic(self):
        """Test basic spatial clustering."""
        # Create clustered points
        base_loc = Location(latitude=34.0522, longitude=-118.2437)

        locations = [
            base_loc,
            Location(
                latitude=base_loc.latitude + 0.001, longitude=base_loc.longitude + 0.001
            ),
            Location(
                latitude=base_loc.latitude + 0.002, longitude=base_loc.longitude + 0.002
            ),
            # Far point
            Location(
                latitude=base_loc.latitude + 0.1, longitude=base_loc.longitude + 0.1
            ),
        ]

        clusters = spatial_clustering(locations, eps_km=0.5, min_samples=2)

        assert isinstance(clusters, list)
        assert len(clusters) > 0

        # Should have at least one cluster with the close points
        cluster_sizes = [len(cluster) for cluster in clusters]
        assert 2 in cluster_sizes or 3 in cluster_sizes

    def test_spatial_clustering_no_clusters(self):
        """Test spatial clustering with widely separated points."""
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            Location(latitude=40.7128, longitude=-74.0060),  # ~4000km away
            Location(latitude=41.8781, longitude=-87.6298),  # ~2800km away
        ]

        clusters = spatial_clustering(locations, eps_km=100, min_samples=2)

        # Should have individual clusters or no clusters
        assert isinstance(clusters, list)

    def test_spatial_clustering_empty_input(self):
        """Test spatial clustering with empty input."""
        clusters = spatial_clustering([], eps_km=1.0, min_samples=2)

        assert clusters == []

    def test_spatial_clustering_single_point(self):
        """Test spatial clustering with single point."""
        locations = [Location(latitude=34.0522, longitude=-118.2437)]

        clusters = spatial_clustering(locations, eps_km=1.0, min_samples=2)

        # Single point cannot form a cluster with min_samples=2
        assert clusters == []


class TestSpatialStatistics:
    """Test spatial statistics calculation."""

    def test_calculate_spatial_statistics(self):
        """Test spatial statistics calculation."""
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            Location(latitude=34.0523, longitude=-118.2438),
            Location(latitude=34.0524, longitude=-118.2439),
            Location(latitude=34.0525, longitude=-118.2440),
        ]

        stats = calculate_spatial_statistics(locations)

        assert isinstance(stats, dict)
        assert "count" in stats
        assert "centroid_lat" in stats
        assert "centroid_lon" in stats
        assert "mean_distance_from_centroid" in stats

        assert stats["count"] == 4
        assert isinstance(stats["centroid_lat"], float)
        assert isinstance(stats["centroid_lon"], float)

    def test_calculate_spatial_statistics_empty(self):
        """Test spatial statistics with empty input."""
        stats = calculate_spatial_statistics([])

        assert stats == {}

    def test_calculate_spatial_statistics_single_point(self):
        """Test spatial statistics with single point."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        stats = calculate_spatial_statistics([location])

        assert stats["count"] == 1
        assert stats["centroid_lat"] == location.latitude
        assert stats["centroid_lon"] == location.longitude
        assert stats["mean_distance_from_centroid"] == 0


class TestGeographicValidation:
    """Test geographic bounds validation."""

    def test_validate_valid_locations(self):
        """Test validation of valid locations."""
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            Location(latitude=40.7128, longitude=-74.0060),
            Location(latitude=-33.8688, longitude=151.2093),
        ]

        result = validate_geographic_bounds(locations)

        assert result["valid"] is True
        assert result["total_locations"] == 3
        assert len(result["invalid_locations"]) == 0

    def test_validate_invalid_latitude(self):
        """Test validation with invalid latitude."""
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            invalid_location(latitude=91.0, longitude=-74.0060),  # Invalid latitude
        ]

        result = validate_geographic_bounds(locations)

        assert result["valid"] is False
        assert len(result["invalid_locations"]) == 1
        assert (
            "Latitude 91.0 out of range" in result["invalid_locations"][0]["issues"][0]
        )

    def test_validate_invalid_longitude(self):
        """Test validation with invalid longitude."""
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            invalid_location(latitude=40.7128, longitude=181.0),  # Invalid longitude
        ]

        result = validate_geographic_bounds(locations)

        assert result["valid"] is False
        assert len(result["invalid_locations"]) == 1

    def test_validate_suspicious_coordinates(self):
        """Test validation of suspicious coordinates."""
        locations = [
            Location(latitude=0.0, longitude=0.0),  # Suspicious
            Location(latitude=34.0522, longitude=-118.2437),
        ]

        result = validate_geographic_bounds(locations)

        assert result["valid"] is False
        assert len(result["invalid_locations"]) == 1

    def test_validate_clustered_data_warning(self):
        """Test validation warning for clustered data."""
        # Create many points at the same location
        base_loc = Location(latitude=34.0522, longitude=-118.2437)
        locations = [base_loc] * 50  # 50 identical points

        result = validate_geographic_bounds(locations)

        assert "large clusters" in " ".join(result["warnings"])


class TestPointInterpolation:
    """Test point interpolation functionality."""

    def test_interpolate_points_basic(self):
        """Test basic point interpolation."""
        start = Location(latitude=34.0522, longitude=-118.2437)
        end = Location(latitude=34.0622, longitude=-118.2337)

        interpolated = interpolate_points([start, end], num_points=3)

        assert len(interpolated) == 5  # start + 3 interpolated + end
        assert interpolated[0] == start
        assert interpolated[-1] == end

        # Check that intermediate points are between start and end
        for point in interpolated[1:-1]:
            assert start.latitude <= point.latitude <= end.latitude
            assert start.longitude <= point.longitude <= end.longitude

    def test_interpolate_points_single_segment(self):
        """Test interpolation with single segment."""
        start = Location(latitude=34.0522, longitude=-118.2437)
        end = Location(latitude=34.0523, longitude=-118.2438)

        interpolated = interpolate_points([start, end], num_points=2)

        assert len(interpolated) == 4  # start + 2 interpolated + end

    def test_interpolate_points_no_interpolation(self):
        """Test interpolation with no intermediate points."""
        start = Location(latitude=34.0522, longitude=-118.2437)
        end = Location(latitude=34.0523, longitude=-118.2438)

        interpolated = interpolate_points([start, end], num_points=0)

        assert len(interpolated) == 2
        assert interpolated[0] == start
        assert interpolated[1] == end


class TestCentroidCalculation:
    """Test centroid calculation functionality."""

    def test_find_centroid_basic(self):
        """Test basic centroid calculation."""
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            Location(latitude=34.0524, longitude=-118.2439),
            Location(latitude=34.0526, longitude=-118.2441),
        ]

        centroid = find_centroid(locations)

        assert isinstance(centroid, Location)

        # Centroid should be within the bounds of the points
        lats = [loc.latitude for loc in locations]
        lons = [loc.longitude for loc in locations]

        assert min(lats) <= centroid.latitude <= max(lats)
        assert min(lons) <= centroid.longitude <= max(lons)

    def test_find_centroid_single_point(self):
        """Test centroid calculation with single point."""
        location = Location(latitude=34.0522, longitude=-118.2437)

        centroid = find_centroid([location])

        assert centroid.latitude == location.latitude
        assert centroid.longitude == location.longitude

    def test_find_centroid_empty_list(self):
        """Test centroid calculation with empty list."""
        with pytest.raises(ValueError):
            find_centroid([])


class TestSpatialAutocorrelation:
    """Test spatial autocorrelation calculation."""

    def test_calculate_spatial_autocorrelation(self):
        """Test spatial autocorrelation calculation."""
        # Create test data with some spatial pattern
        locations = [
            Location(latitude=34.0522 + i * 0.001, longitude=-118.2437 + i * 0.001)
            for i in range(20)
        ]
        values = [i + np.random.normal(0, 1) for i in range(20)]  # Trending with noise

        result = calculate_spatial_autocorrelation(
            locations, values, max_distance_km=1.0
        )

        assert isinstance(result, dict)
        assert "morans_i" in result
        assert "z_score" in result
        assert "p_value" in result

        # Moran's I should be between -1 and 1
        assert -1 <= result["morans_i"] <= 1

        # P-value should be between 0 and 1
        assert 0 <= result["p_value"] <= 1

    def test_calculate_spatial_autocorrelation_insufficient_data(self):
        """Test spatial autocorrelation with insufficient data."""
        locations = [Location(latitude=34.0522, longitude=-118.2437)]
        values = [1.0]

        result = calculate_spatial_autocorrelation(locations, values)

        assert result["morans_i"] == 0.0
        assert result["p_value"] == 1.0

    def test_calculate_spatial_autocorrelation_no_spatial_pattern(self):
        """Test spatial autocorrelation with random data."""
        np.random.seed(42)
        locations = [
            Location(
                latitude=34.0522 + np.random.uniform(-0.01, 0.01),
                longitude=-118.2437 + np.random.uniform(-0.01, 0.01),
            )
            for _ in range(30)
        ]
        values = np.random.normal(0, 1, 30)

        result = calculate_spatial_autocorrelation(locations, values)

        # With random data, autocorrelation should be low
        assert abs(result["morans_i"]) < 0.3


class TestHotspotStatistics:
    """Test hotspot statistics calculation."""

    def test_calculate_hotspot_statistics(self):
        """Test hotspot statistics calculation."""
        # Create test data with some hotspots
        base_loc = Location(latitude=34.0522, longitude=-118.2437)

        locations = []
        case_counts = []

        # Create a hotspot
        for _ in range(10):
            locations.append(
                Location(
                    latitude=base_loc.latitude + np.random.uniform(-0.001, 0.001),
                    longitude=base_loc.longitude + np.random.uniform(-0.001, 0.001),
                )
            )
            case_counts.append(5 + np.random.randint(0, 5))  # High case counts

        # Create some background points
        for _ in range(20):
            locations.append(
                Location(
                    latitude=base_loc.latitude + np.random.uniform(-0.01, 0.01),
                    longitude=base_loc.longitude + np.random.uniform(-0.01, 0.01),
                )
            )
            case_counts.append(1 + np.random.randint(0, 2))  # Low case counts

        result = calculate_hotspot_statistics(locations, case_counts)

        assert isinstance(result, dict)
        assert "total_cases" in result
        assert "total_locations" in result
        assert "hotspots" in result
        assert "risk_zones" in result

        assert result["total_cases"] == sum(case_counts)
        assert result["total_locations"] == len(locations)

        # Should identify at least one hotspot
        assert len(result["hotspots"]) >= 1

    def test_calculate_hotspot_statistics_no_hotspots(self):
        """Test hotspot statistics with uniform low case counts."""
        locations = [
            Location(latitude=34.0522 + i * 0.001, longitude=-118.2437 + i * 0.001)
            for i in range(20)
        ]
        case_counts = [1] * 20  # Uniform low case counts

        result = calculate_hotspot_statistics(locations, case_counts)

        assert result["total_cases"] == 20
        assert result["total_locations"] == 20
        # May or may not identify hotspots depending on algorithm
        assert isinstance(result["hotspots"], list)

    def test_calculate_hotspot_statistics_empty_data(self):
        """Test hotspot statistics with empty data."""
        result = calculate_hotspot_statistics([], [])

        assert result["total_cases"] == 0
        assert result["total_locations"] == 0
        assert result["hotspots"] == []
        assert result["risk_zones"] == []


class TestIntegration:
    """Test integration of advanced geospatial functions."""

    def test_clustering_and_statistics_integration(self):
        """Test integration of clustering and statistics."""
        # Create clustered data
        base_locs = [
            Location(latitude=34.0522, longitude=-118.2437),
            Location(latitude=34.0622, longitude=-118.2337),
            Location(latitude=34.0722, longitude=-118.2237),
        ]

        locations = []
        for base_loc in base_locs:
            # Create cluster around each base location
            for _ in range(5):
                locations.append(
                    Location(
                        latitude=base_loc.latitude + np.random.uniform(-0.001, 0.001),
                        longitude=base_loc.longitude + np.random.uniform(-0.001, 0.001),
                    )
                )

        # Perform clustering
        clusters = spatial_clustering(locations, eps_km=0.2, min_samples=3)

        # Calculate statistics for each cluster
        cluster_stats = []
        for cluster in clusters:
            if len(cluster) >= 3:
                stats = calculate_spatial_statistics(cluster)
                cluster_stats.append(stats)

        assert len(cluster_stats) > 0

        for stats in cluster_stats:
            assert stats["count"] >= 3
            assert "centroid_lat" in stats

    def test_validation_and_clustering_integration(self):
        """Test integration of validation and clustering."""
        # Create data with some invalid points
        locations = [
            Location(latitude=34.0522, longitude=-118.2437),
            invalid_location(latitude=91.0, longitude=-74.0060),  # Invalid
            Location(latitude=34.0523, longitude=-118.2438),
            Location(latitude=0.0, longitude=0.0),  # Suspicious
        ]

        # Validate data
        validation_result = validate_geographic_bounds(locations)

        # Extract valid locations
        invalid_by_index = {
            item["index"]: item for item in validation_result["invalid_locations"]
        }
        valid_locations = []
        for i, loc in enumerate(locations):
            invalid_entry = invalid_by_index.get(i)
            if invalid_entry is None or not any(
                "out of range" in issue for issue in invalid_entry["issues"]
            ):
                valid_locations.append(loc)

        # Perform clustering on valid locations
        if len(valid_locations) >= 3:
            clusters = spatial_clustering(valid_locations, eps_km=1.0, min_samples=2)
            assert isinstance(clusters, list)

        # Should have identified some invalid locations
        assert not validation_result["valid"]
