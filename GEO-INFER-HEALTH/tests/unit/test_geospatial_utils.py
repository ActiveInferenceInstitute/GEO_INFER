"""
Unit tests for geospatial utility functions.
"""

import pytest
import math

from geo_infer_health.models import Location
from geo_infer_health.utils.geospatial_utils import (
    EARTH_RADIUS_KM,
    haversine_distance,
    create_bounding_box,
)


class TestHaversineDistance:
    """Test cases for haversine distance calculation."""

    @pytest.mark.parametrize("latitude", [-87.5, 87.5])
    def test_antipodal_roundoff_stays_in_real_domain(self, latitude):
        first = Location(latitude=latitude, longitude=12.3)
        second = Location(latitude=-latitude, longitude=-167.7)

        assert haversine_distance(first, second) == math.pi * EARTH_RADIUS_KM
        assert haversine_distance(second, first) == math.pi * EARTH_RADIUS_KM

    def test_mutated_nan_coordinate_does_not_become_zero_distance(self):
        first = Location(latitude=0, longitude=0)
        second = Location(latitude=0, longitude=0)
        first.latitude = float("nan")

        assert math.isnan(haversine_distance(first, second))

    def test_same_location_distance(self):
        """Test distance between identical locations."""
        location = Location(latitude=34.0522, longitude=-118.2437)

        distance = haversine_distance(location, location)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_los_angeles_to_new_york(self):
        """Test distance between Los Angeles and New York."""
        la = Location(latitude=34.0522, longitude=-118.2437)
        ny = Location(latitude=40.7128, longitude=-74.0060)

        # Approximate distance: ~3935 km
        expected_distance = 3935.0  # km
        actual_distance = haversine_distance(la, ny)

        assert actual_distance == pytest.approx(
            expected_distance, abs=50
        )  # Allow 50km tolerance

    def test_symmetry(self):
        """Test that distance calculation is symmetric."""
        loc1 = Location(latitude=34.0522, longitude=-118.2437)
        loc2 = Location(latitude=40.7128, longitude=-74.0060)

        distance_1_to_2 = haversine_distance(loc1, loc2)
        distance_2_to_1 = haversine_distance(loc2, loc1)

        assert distance_1_to_2 == distance_2_to_1

    def test_short_distance(self):
        """Test distance calculation for short distances."""
        # Two points 1km apart (approximately)
        loc1 = Location(latitude=0.0, longitude=0.0)
        loc2 = Location(latitude=0.009, longitude=0.0)  # Roughly 1km north

        distance = haversine_distance(loc1, loc2)
        assert distance == pytest.approx(1.0, abs=0.1)  # Allow 100m tolerance

    def test_equator_vs_poles(self):
        """Test distance calculation at different latitudes."""
        # Points at equator
        equator1 = Location(latitude=0.0, longitude=0.0)
        equator2 = Location(latitude=0.0, longitude=1.0)

        # Points at high latitude
        arctic1 = Location(latitude=80.0, longitude=0.0)
        arctic2 = Location(latitude=80.0, longitude=1.0)

        distance_equator = haversine_distance(equator1, equator2)
        distance_arctic = haversine_distance(arctic1, arctic2)

        # Distance should be shorter at higher latitudes for same longitude difference
        assert distance_arctic < distance_equator

    def test_edge_cases(self):
        """Test edge cases for distance calculation."""
        # North Pole to South Pole
        north_pole = Location(latitude=90.0, longitude=0.0)
        south_pole = Location(latitude=-90.0, longitude=0.0)

        distance = haversine_distance(north_pole, south_pole)
        expected_distance = 20015.0  # Earth's circumference / 2
        assert distance == pytest.approx(expected_distance, abs=100)

    def test_distance_with_different_crs(self):
        """Test distance calculation with different CRS (should still work)."""
        loc1 = Location(latitude=34.0522, longitude=-118.2437, crs="EPSG:4326")
        loc2 = Location(latitude=40.7128, longitude=-74.0060, crs="EPSG:3857")

        # Distance should still be calculated correctly regardless of CRS
        distance = haversine_distance(loc1, loc2)
        assert distance > 0
        assert isinstance(distance, float)


class TestBoundingBox:
    """Test cases for bounding box creation."""

    def test_bounding_box_creation(self):
        """Test basic bounding box creation."""
        center = Location(latitude=34.0522, longitude=-118.2437)
        distance_km = 10.0

        bbox = create_bounding_box(center, distance_km)

        assert len(bbox) == 2
        assert isinstance(bbox[0], Location)
        assert isinstance(bbox[1], Location)

        # Check that corners are correct relative to center
        assert bbox[0].latitude < center.latitude  # Southwest corner
        assert bbox[0].longitude < center.longitude
        assert bbox[1].latitude > center.latitude  # Northeast corner
        assert bbox[1].longitude > center.longitude

    def test_bounding_box_symmetry(self):
        """Test that bounding box is symmetric around center."""
        center = Location(latitude=0.0, longitude=0.0)
        distance_km = 5.0

        bbox = create_bounding_box(center, distance_km)

        south_lat = bbox[0].latitude
        north_lat = bbox[1].latitude
        west_lon = bbox[0].longitude
        east_lon = bbox[1].longitude

        # Check symmetry
        assert abs(center.latitude - south_lat) == pytest.approx(
            abs(north_lat - center.latitude), abs=1e-6
        )
        assert abs(center.longitude - west_lon) == pytest.approx(
            abs(east_lon - center.longitude), abs=1e-6
        )

    def test_bounding_box_size(self):
        """Test bounding box size for different distances."""
        center = Location(latitude=0.0, longitude=0.0)

        # Small bounding box
        small_bbox = create_bounding_box(center, 1.0)
        small_lat_diff = small_bbox[1].latitude - small_bbox[0].latitude
        small_lon_diff = small_bbox[1].longitude - small_bbox[0].longitude

        # Large bounding box
        large_bbox = create_bounding_box(center, 10.0)
        large_lat_diff = large_bbox[1].latitude - large_bbox[0].latitude
        large_lon_diff = large_bbox[1].longitude - large_bbox[0].longitude

        # Large bbox should have larger differences
        assert large_lat_diff > small_lat_diff
        assert large_lon_diff > small_lon_diff

    def test_bounding_box_at_equator(self):
        """Test bounding box creation at equator."""
        center = Location(latitude=0.0, longitude=0.0)
        distance_km = 5.0

        bbox = create_bounding_box(center, distance_km)

        # At equator, longitude differences should be larger than latitude differences
        lat_diff = bbox[1].latitude - bbox[0].latitude
        lon_diff = bbox[1].longitude - bbox[0].longitude

        assert lon_diff > lat_diff

    def test_bounding_box_at_pole(self):
        """Test bounding box creation near poles."""
        center = Location(latitude=85.0, longitude=0.0)
        distance_km = 5.0

        bbox = create_bounding_box(center, distance_km)

        # Near poles, the bounding box should handle longitude convergence
        assert bbox[0].latitude < bbox[1].latitude
        assert bbox[0].longitude != bbox[1].longitude

        # Longitude differences should be handled properly
        lon_diff = abs(bbox[1].longitude - bbox[0].longitude)
        assert lon_diff > 0

    def test_bounding_box_crs_preservation(self):
        """Test that CRS is preserved in bounding box corners."""
        center = Location(latitude=34.0522, longitude=-118.2437, crs="EPSG:4326")
        distance_km = 10.0

        bbox = create_bounding_box(center, distance_km)

        assert bbox[0].crs == center.crs
        assert bbox[1].crs == center.crs

    def test_zero_distance_bbox(self):
        """Test bounding box with zero distance."""
        center = Location(latitude=34.0522, longitude=-118.2437)

        bbox = create_bounding_box(center, 0.0)

        # Should return the center point as both corners
        assert bbox[0].latitude == pytest.approx(center.latitude, abs=1e-6)
        assert bbox[0].longitude == pytest.approx(center.longitude, abs=1e-6)
        assert bbox[1].latitude == pytest.approx(center.latitude, abs=1e-6)
        assert bbox[1].longitude == pytest.approx(center.longitude, abs=1e-6)


class TestGeospatialIntegration:
    """Test integration of geospatial utilities."""

    def test_distance_and_bbox_consistency(self):
        """Test consistency between distance and bounding box calculations."""
        center = Location(latitude=34.0522, longitude=-118.2437)
        distance_km = 5.0

        bbox = create_bounding_box(center, distance_km)

        # Calculate actual distances to corners
        dist_sw = haversine_distance(center, bbox[0])
        dist_ne = haversine_distance(center, bbox[1])

        # Distances should be close to the specified distance
        # (allowing for approximation errors in bounding box calculation)
        assert dist_sw == pytest.approx(distance_km * math.sqrt(2), abs=0.5)
        assert dist_ne == pytest.approx(distance_km * math.sqrt(2), abs=0.5)

    def test_multiple_distance_calculations(self, sample_locations):
        """Test distance calculations between multiple locations."""
        distances = []

        for i in range(len(sample_locations)):
            for j in range(i + 1, len(sample_locations)):
                dist = haversine_distance(sample_locations[i], sample_locations[j])
                distances.append(dist)

                # All distances should be positive and reasonable
                assert dist > 0
                assert dist < 20000  # No distance should exceed Earth's circumference

        # Should have calculated some distances
        assert len(distances) > 0

    def test_bbox_coverage(self, sample_locations):
        """Test that bounding boxes properly cover point sets."""
        if len(sample_locations) < 2:
            pytest.fail("Need at least 2 locations for this test")

        # Create a bounding box that should contain all points
        center_lat = sum(loc.latitude for loc in sample_locations) / len(
            sample_locations
        )
        center_lon = sum(loc.longitude for loc in sample_locations) / len(
            sample_locations
        )
        center = Location(latitude=center_lat, longitude=center_lon)

        # Find maximum distance from center to any point
        max_distance = max(haversine_distance(center, loc) for loc in sample_locations)

        # Create bounding box with slightly larger distance
        bbox = create_bounding_box(center, max_distance * 1.1)

        # All points should be within the bounding box
        for loc in sample_locations:
            assert bbox[0].latitude <= loc.latitude <= bbox[1].latitude
            assert bbox[0].longitude <= loc.longitude <= bbox[1].longitude


class TestPerformance:
    """Test performance characteristics of geospatial utilities."""

    def test_distance_calculation_performance(self):
        """Test that distance calculations are reasonably fast."""
        import time

        loc1 = Location(latitude=34.0522, longitude=-118.2437)
        loc2 = Location(latitude=40.7128, longitude=-74.0060)

        # Time multiple calculations
        num_calculations = 1000
        start_time = time.time()

        for _ in range(num_calculations):
            haversine_distance(loc1, loc2)

        end_time = time.time()
        total_time = end_time - start_time

        # Should be very fast (< 0.1 seconds for 1000 calculations)
        assert total_time < 0.1

        avg_time = total_time / num_calculations
        # Average time per calculation should be very small
        assert avg_time < 0.0001  # Less than 0.1ms per calculation

    def test_bbox_creation_performance(self):
        """Test that bounding box creation is reasonably fast."""
        import time

        center = Location(latitude=34.0522, longitude=-118.2437)

        # Time multiple bbox creations
        num_calculations = 1000
        start_time = time.time()

        for _ in range(num_calculations):
            create_bounding_box(center, 10.0)

        end_time = time.time()
        total_time = end_time - start_time

        # Should be reasonably fast (< 0.5 seconds for 1000 calculations)
        assert total_time < 0.5

        avg_time = total_time / num_calculations
        # Average time per calculation should be reasonable
        assert avg_time < 0.0005  # Less than 0.5ms per calculation


class TestErrorHandling:
    """Test error handling in geospatial utilities."""

    def test_distance_with_none_values(self):
        """Test distance calculation with None values."""
        loc1 = Location(latitude=34.0522, longitude=-118.2437)
        loc2 = Location(latitude=None, longitude=-74.0060)

        # Should handle None values gracefully or raise appropriate error
        with pytest.raises((TypeError, AttributeError)):
            haversine_distance(loc1, loc2)

    def test_bbox_with_negative_distance(self):
        """Test bounding box creation with negative distance."""
        center = Location(latitude=34.0522, longitude=-118.2437)

        # Should handle negative distances
        bbox = create_bounding_box(center, -5.0)

        # The bounding box should still be created, but may not make geometric sense
        assert isinstance(bbox, tuple)
        assert len(bbox) == 2

    def test_bbox_with_extreme_coordinates(self):
        """Test bounding box with extreme coordinate values."""
        # Test with coordinates at the limits
        center = Location(latitude=89.999, longitude=179.999)

        bbox = create_bounding_box(center, 1.0)

        # Should handle extreme coordinates without errors
        assert isinstance(bbox, tuple)
        assert len(bbox) == 2

        # Coordinates should still be within valid ranges
        assert -90 <= bbox[0].latitude <= 90
        assert -90 <= bbox[1].latitude <= 90
        assert -180 <= bbox[0].longitude <= 180
        assert -180 <= bbox[1].longitude <= 180
