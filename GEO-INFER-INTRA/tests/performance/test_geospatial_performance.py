"""Performance tests for geospatial operations."""

import pytest
import time
import random
from typing import Any

from geo_infer_intra.utils.geospatial_utils import (
    create_point,
    create_polygon,
    create_feature,
    create_feature_collection,
    haversine_distance,
)


@pytest.mark.performance
@pytest.mark.geospatial
class TestGeospatialPerformance:
    """Test suite for geospatial performance."""

    @pytest.fixture
    def random_points(self, n: int = 10000) -> list[dict[str, Any]]:
        """Generate n random GeoJSON points."""
        return [
            create_point(lon=random.uniform(-180, 180), lat=random.uniform(-90, 90))
            for _ in range(n)
        ]

    @pytest.fixture
    def random_point_features(self, n: int = 10000) -> list[dict[str, Any]]:
        """Generate n random GeoJSON point features."""
        return [
            create_feature(
                create_point(
                    lon=random.uniform(-180, 180), lat=random.uniform(-90, 90)
                ),
                {"id": i, "value": random.random()},
            )
            for i in range(n)
        ]

    @pytest.mark.parametrize("n_points", [100, 1000, 10000])
    def test_feature_collection_creation(self, random_point_features, n_points):
        """Test performance of creating feature collections of different sizes."""
        # Use only n_points from the fixture
        features = random_point_features[:n_points]

        # Measure time to create feature collection
        start_time = time.time()
        fc = create_feature_collection(features)
        end_time = time.time()

        duration = end_time - start_time

        # Log performance data
        print(
            f"\nCreated FeatureCollection with {n_points} features in {duration:.4f} seconds"
        )

        # Add some basic assertions to validate the result
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == n_points

        # Performance threshold (adjust based on actual performance)
        max_duration = 0.1 * n_points / 1000  # Scale with number of points
        assert duration < max_duration, (
            f"Performance too slow: {duration:.4f}s > {max_duration:.4f}s"
        )

    @pytest.mark.parametrize("n_points", [100, 1000])
    def test_distance_calculation(self, random_points, n_points):
        """Test performance of calculating distances between many points."""
        # Use only n_points from the fixture
        points = random_points[:n_points]

        # Measure time to calculate distances
        start_time = time.time()

        # Calculate distances between each point and the first point
        reference_point = points[0]["coordinates"]
        distances = []

        for point in points:
            coords = point["coordinates"]
            dist = haversine_distance(
                reference_point[0], reference_point[1], coords[0], coords[1]
            )
            distances.append(dist)

        end_time = time.time()
        duration = end_time - start_time

        # Log performance data
        print(f"\nCalculated {n_points} distances in {duration:.4f} seconds")
        print(
            f"Average time per distance calculation: {(duration / n_points) * 1000:.4f} ms"
        )

        # Performance threshold (adjust based on actual performance)
        max_duration = 0.05 * n_points / 1000  # Scale with number of points
        assert duration < max_duration, (
            f"Performance too slow: {duration:.4f}s > {max_duration:.4f}s"
        )

    @pytest.mark.parametrize("n_coords", [10, 100, 1000])
    def test_polygon_creation(self, n_coords):
        """Test performance of creating polygons with different numbers of vertices."""
        # Generate n_coords random coordinates for a polygon
        # Make sure they form a roughly circular shape to avoid self-intersection
        import math

        coords = []
        radius = 1.0
        center_lon, center_lat = 0.0, 0.0

        for i in range(n_coords):
            angle = 2 * math.pi * i / n_coords
            x = center_lon + radius * math.cos(angle)
            y = center_lat + radius * math.sin(angle)
            coords.append([x, y])

        # Measure time to create polygon
        start_time = time.time()
        polygon = create_polygon(coords)
        end_time = time.time()

        duration = end_time - start_time

        # Log performance data
        print(f"\nCreated Polygon with {n_coords} vertices in {duration:.4f} seconds")

        # Add some basic assertions to validate the result
        assert polygon["type"] == "Polygon"
        assert len(polygon["coordinates"][0]) == n_coords + 1  # +1 for closure

        # Performance threshold (adjust based on actual performance)
        max_duration = 0.05 * n_coords / 1000  # Scale with number of coordinates
        assert duration < max_duration, (
            f"Performance too slow: {duration:.4f}s > {max_duration:.4f}s"
        )
