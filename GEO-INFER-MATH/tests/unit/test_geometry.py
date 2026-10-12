"""
Tests for GEO-INFER-MATH geometry module.

Tests cover: Point/LineString/Polygon dataclasses, haversine/vincenty distances,
bearing calculations, point-in-polygon, line intersection, and spherical area.
"""

import numpy as np

import pytest


from geo_infer_math.core.geometry import (
    Point,
    LineString,
    Polygon,
    haversine_distance,
    vincenty_distance,
    bearing,
    destination_point,
    point_in_polygon,
    buffer_point,
    line_intersection,
    polygon_area_spherical,
    great_circle_distance,
)


class TestPoint:
    """Tests for Point dataclass."""

    def test_point_2d_distance(self):
        p1 = Point(x=0.0, y=0.0)
        p2 = Point(x=3.0, y=4.0)
        assert abs(p1.distance_to(p2) - 5.0) < 1e-10

    def test_point_3d_distance(self):
        p1 = Point(x=0.0, y=0.0, z=0.0)
        p2 = Point(x=1.0, y=2.0, z=2.0)
        assert abs(p1.distance_to(p2) - 3.0) < 1e-10

    def test_point_to_array_2d(self):
        p = Point(x=1.5, y=2.5)
        arr = p.to_array()
        assert arr.shape == (2,)
        assert arr[0] == 1.5

    def test_point_to_array_3d(self):
        p = Point(x=1.0, y=2.0, z=3.0)
        arr = p.to_array()
        assert arr.shape == (3,)
        assert arr[2] == 3.0

    def test_point_self_distance_zero(self):
        p = Point(x=5.0, y=5.0)
        assert p.distance_to(p) == 0.0


class TestLineString:
    """Tests for LineString dataclass."""

    def test_linestring_length(self):
        points = [Point(x=0.0, y=0.0), Point(x=3.0, y=0.0), Point(x=3.0, y=4.0)]
        ls = LineString(points=points)
        assert abs(ls.length() - 7.0) < 1e-10

    def test_linestring_single_point_length(self):
        ls = LineString(points=[Point(x=0.0, y=0.0)])
        assert ls.length() == 0.0

    def test_linestring_to_array(self):
        points = [Point(x=0.0, y=0.0), Point(x=1.0, y=1.0)]
        ls = LineString(points=points)
        arr = ls.to_array()
        assert arr.shape == (2, 2)


class TestPolygon:
    """Tests for Polygon dataclass."""

    def test_polygon_area_unit_square(self):
        exterior = [
            Point(x=0.0, y=0.0),
            Point(x=1.0, y=0.0),
            Point(x=1.0, y=1.0),
            Point(x=0.0, y=1.0),
        ]
        poly = Polygon(exterior=exterior)
        assert abs(poly.area() - 1.0) < 1e-10

    def test_polygon_area_triangle(self):
        exterior = [
            Point(x=0.0, y=0.0),
            Point(x=4.0, y=0.0),
            Point(x=0.0, y=3.0),
        ]
        poly = Polygon(exterior=exterior)
        assert abs(poly.area() - 6.0) < 1e-10

    def test_polygon_with_hole(self):
        exterior = [
            Point(x=0.0, y=0.0),
            Point(x=10.0, y=0.0),
            Point(x=10.0, y=10.0),
            Point(x=0.0, y=10.0),
        ]
        hole = [
            Point(x=2.0, y=2.0),
            Point(x=4.0, y=2.0),
            Point(x=4.0, y=4.0),
            Point(x=2.0, y=4.0),
        ]
        poly = Polygon(exterior=exterior, interiors=[hole])
        assert abs(poly.area() - 96.0) < 1e-10

    def test_polygon_centroid(self):
        exterior = [
            Point(x=0.0, y=0.0),
            Point(x=2.0, y=0.0),
            Point(x=2.0, y=2.0),
            Point(x=0.0, y=2.0),
        ]
        poly = Polygon(exterior=exterior)
        centroid = poly.centroid()
        assert abs(centroid.x - 1.0) < 1e-10
        assert abs(centroid.y - 1.0) < 1e-10

    def test_polygon_centroid_non_convex(self):
        # L-shape: area centroid (1.5, 1.0), not the vertex mean (1.667, 1.333)
        exterior = [
            Point(x=0.0, y=0.0),
            Point(x=4.0, y=0.0),
            Point(x=4.0, y=1.0),
            Point(x=1.0, y=1.0),
            Point(x=1.0, y=3.0),
            Point(x=0.0, y=3.0),
        ]
        poly = Polygon(exterior=exterior)
        centroid = poly.centroid()
        assert abs(centroid.x - 1.5) < 1e-10
        assert abs(centroid.y - 1.0) < 1e-10

    def test_polygon_centroid_with_hole(self):
        exterior = [
            Point(x=0.0, y=0.0),
            Point(x=10.0, y=0.0),
            Point(x=10.0, y=10.0),
            Point(x=0.0, y=10.0),
        ]
        hole = [
            Point(x=2.0, y=2.0),
            Point(x=4.0, y=2.0),
            Point(x=4.0, y=4.0),
            Point(x=2.0, y=4.0),
        ]
        poly = Polygon(exterior=exterior, interiors=[hole])
        centroid = poly.centroid()
        # (100*(5,5) - 4*(3,3)) / 96
        assert abs(centroid.x - 488.0 / 96.0) < 1e-10
        assert abs(centroid.y - 488.0 / 96.0) < 1e-10


class TestHaversineDistance:
    """Tests for haversine distance calculations."""

    def test_same_point_distance_zero(self):
        dist = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert abs(dist) < 1e-10

    def test_new_york_to_london(self):
        # NYC to London approximately 5570 km
        dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < dist < 5700

    def test_equator_one_degree(self):
        # One degree of longitude at equator is approximately 111 km
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110 < dist < 112

    def test_symmetry(self):
        d1 = haversine_distance(40.0, -74.0, 51.0, -0.1)
        d2 = haversine_distance(51.0, -0.1, 40.0, -74.0)
        assert abs(d1 - d2) < 1e-10


class TestVincentyDistance:
    """Tests for vincenty distance calculations."""

    def test_same_point_zero(self):
        dist = vincenty_distance(40.0, -74.0, 40.0, -74.0)
        assert abs(dist) < 1e-6

    def test_known_distance(self):
        # NYC to London, vincenty returns meters
        dist = vincenty_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5.5e6 < dist < 5.7e6

    def test_near_antipodal_raises(self):
        # Near-antipodal pairs are Vincenty's classic non-convergence case;
        # the function must raise rather than return an unconverged result.
        with pytest.raises(ValueError, match="failed to converge"):
            vincenty_distance(40.7128, -74.0060, -40.7128, 105.994)


class TestBearing:
    """Tests for bearing calculations."""

    def test_due_north(self):
        b = bearing(0.0, 0.0, 1.0, 0.0)
        assert abs(b - 0.0) < 1.0

    def test_due_east(self):
        b = bearing(0.0, 0.0, 0.0, 1.0)
        assert abs(b - 90.0) < 1.0

    def test_due_south(self):
        b = bearing(1.0, 0.0, 0.0, 0.0)
        assert abs(b - 180.0) < 1.0


class TestDestinationPoint:
    """Tests for destination point calculations."""

    def test_destination_north(self):
        lat, lon = destination_point(0.0, 0.0, 0.0, 111.0)
        # Moving north ~111 km should give approx 1 degree of latitude
        assert abs(lat - 1.0) < 0.05
        assert abs(lon - 0.0) < 0.05

    def test_round_trip(self):
        # Go north 100 km, then south 100 km, should return to start
        lat1, lon1 = destination_point(0.0, 0.0, 0.0, 100.0)
        lat2, lon2 = destination_point(lat1, lon1, 180.0, 100.0)
        assert abs(lat2 - 0.0) < 0.01
        assert abs(lon2 - 0.0) < 0.01


class TestPointInPolygon:
    """Tests for point-in-polygon determination."""

    def test_point_inside_square(self):
        poly = Polygon(
            exterior=[
                Point(x=0.0, y=0.0),
                Point(x=10.0, y=0.0),
                Point(x=10.0, y=10.0),
                Point(x=0.0, y=10.0),
            ]
        )
        assert point_in_polygon(Point(x=5.0, y=5.0), poly) is True

    def test_point_outside_square(self):
        poly = Polygon(
            exterior=[
                Point(x=0.0, y=0.0),
                Point(x=10.0, y=0.0),
                Point(x=10.0, y=10.0),
                Point(x=0.0, y=10.0),
            ]
        )
        assert point_in_polygon(Point(x=15.0, y=5.0), poly) is False

    def test_point_in_hole_excluded(self):
        poly = Polygon(
            exterior=[
                Point(x=0.0, y=0.0),
                Point(x=10.0, y=0.0),
                Point(x=10.0, y=10.0),
                Point(x=0.0, y=10.0),
            ],
            interiors=[
                [
                    Point(x=3.0, y=3.0),
                    Point(x=7.0, y=3.0),
                    Point(x=7.0, y=7.0),
                    Point(x=3.0, y=7.0),
                ]
            ],
        )
        assert point_in_polygon(Point(x=5.0, y=5.0), poly) is False


class TestLineIntersection:
    """Tests for line intersection."""

    def test_crossing_lines(self):
        result = line_intersection(
            Point(x=0.0, y=0.0),
            Point(x=10.0, y=10.0),
            Point(x=0.0, y=10.0),
            Point(x=10.0, y=0.0),
        )
        assert result is not None
        assert abs(result.x - 5.0) < 1e-10
        assert abs(result.y - 5.0) < 1e-10

    def test_parallel_lines_no_intersection(self):
        result = line_intersection(
            Point(x=0.0, y=0.0),
            Point(x=10.0, y=0.0),
            Point(x=0.0, y=1.0),
            Point(x=10.0, y=1.0),
        )
        assert result is None

    def test_non_intersecting_segments(self):
        result = line_intersection(
            Point(x=0.0, y=0.0),
            Point(x=1.0, y=0.0),
            Point(x=2.0, y=1.0),
            Point(x=3.0, y=1.0),
        )
        assert result is None


class TestBufferPoint:
    """Tests for point buffering."""

    def test_buffer_creates_closed_ring(self):
        buf = buffer_point(0.0, 0.0, 10.0, segments=16)
        assert buf[0] == buf[-1]
        assert len(buf) == 17

    def test_buffer_points_correct_distance(self):
        buf = buffer_point(0.0, 0.0, 100.0, segments=8)
        for lat, lon in buf[:-1]:
            dist = haversine_distance(0.0, 0.0, lat, lon)
            assert abs(dist - 100.0) < 5.0


class TestGreatCircleDistance:
    """Tests for vectorized great circle distance."""

    def test_distance_matrix_shape(self):
        coords1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        coords2 = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        result = great_circle_distance(coords1, coords2)
        assert result.shape == (2, 3)


class TestVectorizedPointInPolygon:
    """Tests for SIMD/vectorized point-in-polygon containment."""

    def test_vectorized_matches_scalar_ray_casting(self):
        from geo_infer_math.core.geometry import (
            points_in_polygon_vectorized,
            point_in_polygon,
            Point,
            Polygon,
        )

        poly_x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
        poly_y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
        polygon = Polygon([Point(x=x, y=y) for x, y in zip(poly_x, poly_y)])

        test_x = np.array([5.0, 15.0, -2.0, 2.0, 8.0, 10.5])
        test_y = np.array([5.0, 5.0, 2.0, 8.0, 2.0, 10.0])

        vec_res = points_in_polygon_vectorized(test_x, test_y, poly_x, poly_y)
        scalar_res = np.array(
            [point_in_polygon(Point(x=x, y=y), polygon) for x, y in zip(test_x, test_y)]
        )

        assert np.array_equal(vec_res, scalar_res)
        assert np.array_equal(vec_res, [True, False, False, True, True, False])

    def test_self_distance_zero(self):
        coords = np.array([[40.0, -74.0]])
        result = great_circle_distance(coords, coords)
        assert abs(result[0, 0]) < 1e-10

    def test_distance_consistency_with_haversine(self):
        coords1 = np.array([[40.7128, -74.0060]])
        coords2 = np.array([[51.5074, -0.1278]])
        matrix_dist = great_circle_distance(coords1, coords2)[0, 0]
        scalar_dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert abs(matrix_dist - scalar_dist) < 1.0


class TestPolygonAreaSpherical:
    """Tests for spherical polygon area."""

    def test_small_polygon_positive_area(self):
        polygon = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
        ]
        area = polygon_area_spherical(polygon)
        assert area > 0
        # Area of ~1 degree square at equator is about 12,300 km^2
        assert 10000 < area < 15000

    def test_degenerate_polygon_zero_area(self):
        polygon = [(0.0, 0.0), (1.0, 0.0)]
        area = polygon_area_spherical(polygon)
        assert area == 0.0
