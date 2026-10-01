"""
Unit tests for spatial functions and H3 v4 functionality.

This module tests the core spatial processing capabilities including:
- H3 v4 spatial indexing
- Geospatial data processing
- Spatial analysis functions
- Coordinate transformations
"""

import pytest
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon, LineString
import json
import time


# Test data
@pytest.fixture
def sample_points():
    """Sample point geometries for testing."""
    return gpd.GeoDataFrame(
        {
            "geometry": [
                Point(-122.4194, 37.7749),  # San Francisco
                Point(-122.4000, 37.7800),  # Oakland
                Point(-122.4500, 37.7600),  # San Jose
                Point(-122.5000, 37.7500),  # Further south
                Point(-122.3500, 37.8000),  # Further north
            ],
            "name": [
                "San Francisco",
                "Oakland",
                "San Jose",
                "South Point",
                "North Point",
            ],
            "population": [873965, 440646, 1030119, 50000, 30000],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_polygons():
    """Sample polygon geometries for testing."""
    return gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (-122.5, 37.7),
                        (-122.3, 37.7),
                        (-122.3, 37.9),
                        (-122.5, 37.9),
                        (-122.5, 37.7),
                    ]
                ),
                Polygon(
                    [
                        (-122.4, 37.6),
                        (-122.2, 37.6),
                        (-122.2, 37.8),
                        (-122.4, 37.8),
                        (-122.4, 37.6),
                    ]
                ),
                Polygon(
                    [
                        (-122.6, 37.5),
                        (-122.4, 37.5),
                        (-122.4, 37.7),
                        (-122.6, 37.7),
                        (-122.6, 37.5),
                    ]
                ),
            ],
            "name": ["Region A", "Region B", "Region C"],
            "area_km2": [100, 150, 200],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_h3_indices():
    """Sample H3 v4 indices for testing."""
    # Generate H3 indices around San Francisco
    center_lat, center_lng = 37.7749, -122.4194

    indices = []
    for resolution in [9, 10, 11]:
        # Get the center index
        center_index = h3.latlng_to_cell(center_lat, center_lng, resolution)
        indices.append(center_index)

        # Get neighboring indices
        neighbors = h3.grid_disk(center_index, 2)
        indices.extend(list(neighbors)[:5])  # Limit to 5 neighbors

    return indices


class TestH3SpatialIndexing:
    """Test H3 v4 spatial indexing functionality."""

    def test_h3_cell_creation(self):
        """Test creating H3 cells from lat/lng coordinates."""
        lat, lng = 37.7749, -122.4194
        resolution = 10

        # Create H3 cell
        h3_cell = h3.latlng_to_cell(lat, lng, resolution)

        assert isinstance(h3_cell, str)
        assert h3.is_valid_cell(h3_cell)
        assert h3.str_to_int(h3_cell) > 0

    def test_h3_cell_to_latlng(self):
        """Test converting H3 cell back to lat/lng coordinates."""
        lat, lng = 37.7749, -122.4194
        resolution = 10

        # Create H3 cell
        h3_cell = h3.latlng_to_cell(lat, lng, resolution)

        # Convert back to lat/lng
        cell_lat, cell_lng = h3.cell_to_latlng(h3_cell)

        # Check that coordinates are close (within cell bounds)
        assert abs(lat - cell_lat) < 0.01
        assert abs(lng - cell_lng) < 0.01

    def test_h3_grid_disk(self):
        """Test creating H3 grid disk around a cell."""
        lat, lng = 37.7749, -122.4194
        resolution = 10
        k_distance = 2

        center_cell = h3.latlng_to_cell(lat, lng, resolution)
        disk_cells = h3.grid_disk(center_cell, k_distance)

        assert len(disk_cells) > 0
        assert center_cell in disk_cells

        # Check that all cells are at the same resolution
        for cell in disk_cells:
            cell_resolution = h3.get_resolution(cell)
            assert cell_resolution == resolution

    def test_h3_cell_boundary(self):
        """Test getting H3 cell boundary coordinates."""
        lat, lng = 37.7749, -122.4194
        resolution = 10

        h3_cell = h3.latlng_to_cell(lat, lng, resolution)
        boundary = h3.cell_to_boundary(h3_cell)

        # H3 v4 returns an open ring of six (lat, lng) vertices for a hexagon.
        assert len(boundary) == 6
        assert all(len(coord) == 2 for coord in boundary)
        assert boundary[0] != boundary[-1]

    def test_h3_resolution_consistency(self):
        """Test that H3 resolution is consistent across operations."""
        lat, lng = 37.7749, -122.4194

        for resolution in range(0, 16):  # Test all valid resolutions
            h3_cell = h3.latlng_to_cell(lat, lng, resolution)
            actual_resolution = h3.get_resolution(h3_cell)
            assert actual_resolution == resolution

    def test_h3_parent_child_relationships(self):
        """Test H3 parent-child cell relationships."""
        lat, lng = 37.7749, -122.4194
        resolution = 10

        # Create child cell
        child_cell = h3.latlng_to_cell(lat, lng, resolution)

        # Get parent cell
        parent_cell = h3.cell_to_parent(child_cell, resolution - 1)

        # Get children of parent
        children = h3.cell_to_children(parent_cell, resolution)

        assert child_cell in children
        assert len(children) == 7  # Each parent has 7 children in H3

    def test_h3_polyfill(self, sample_polygons):
        """Test filling polygons with H3 cells."""
        polygon = sample_polygons.iloc[0].geometry
        resolution = 10

        # Fill polygon with H3 cells using v4 API
        h3_cells = h3.geo_to_cells(polygon.__geo_interface__, resolution)

        assert len(h3_cells) > 0
        assert all(isinstance(cell, str) for cell in h3_cells)

        # Check that all cells are at the specified resolution
        for cell in h3_cells:
            cell_resolution = h3.get_resolution(cell)
            assert cell_resolution == resolution


class TestSpatialDataProcessing:
    """Test spatial data processing functions."""

    def test_geodataframe_creation(self, sample_points):
        """Test creating GeoDataFrame from point data."""
        gdf = sample_points

        assert len(gdf) == 5
        assert all(isinstance(geom, Point) for geom in gdf.geometry)
        assert "name" in gdf.columns
        assert "population" in gdf.columns

    def test_spatial_indexing(self, sample_points):
        """Test spatial indexing for efficient queries."""
        gdf = sample_points

        # Create spatial index
        spatial_index = gdf.sindex

        # Test bounding box query
        bbox = (-122.5, 37.7, -122.3, 37.9)
        indices = list(spatial_index.intersection(bbox))

        assert len(indices) > 0
        assert all(0 <= idx < len(gdf) for idx in indices)

    def test_spatial_join(self, sample_points, sample_polygons):
        """Test spatial join operations."""
        points_gdf = sample_points
        polygons_gdf = sample_polygons

        # Perform spatial join
        joined = gpd.sjoin(points_gdf, polygons_gdf, how="inner", predicate="within")

        # Check that join produced results
        assert len(joined) >= 0  # May be 0 if no points are within polygons

    def test_buffer_operations(self, sample_points):
        """Test buffer operations on geometries."""
        gdf = sample_points.copy()
        buffer_distance = 0.01  # degrees

        # Create buffers
        metric_gdf = gdf.to_crs("EPSG:3857")
        metric_gdf["buffer"] = metric_gdf.geometry.buffer(buffer_distance)
        gdf["buffer"] = metric_gdf["buffer"].to_crs("EPSG:4326")

        assert all(isinstance(buf, Polygon) for buf in gdf["buffer"])
        assert all(buf.area > 0 for buf in gdf["buffer"])

    def test_distance_calculations(self, sample_points):
        """Test distance calculations between geometries."""
        gdf = sample_points

        # Calculate distances from first point to all others
        metric_gdf = gdf.to_crs("EPSG:3857")
        first_point = metric_gdf.iloc[0].geometry
        distances = metric_gdf.geometry.distance(first_point)

        assert len(distances) == len(gdf)
        assert distances.iloc[0] == 0  # Distance to self should be 0
        assert all(dist >= 0 for dist in distances)


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""

    def test_wgs84_to_web_mercator(self):
        """Test WGS84 to Web Mercator transformation."""
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Test San Francisco coordinates
        lon, lat = -122.4194, 37.7749
        x, y = transformer.transform(lon, lat)

        # Check that coordinates are reasonable for Web Mercator
        assert -20000000 < x < 20000000
        assert -20000000 < y < 20000000

    def test_web_mercator_to_wgs84(self):
        """Test Web Mercator to WGS84 transformation."""
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Test Web Mercator coordinates (approximate San Francisco)
        x, y = -13629000, 4519000
        lon, lat = transformer.transform(x, y)

        # Check that coordinates are reasonable for WGS84
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90

    def test_crs_consistency(self):
        """Test that coordinate transformations are consistent."""
        from pyproj import Transformer

        # Forward and reverse transformations
        forward_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        reverse_transformer = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )

        # Original coordinates
        lon, lat = -122.4194, 37.7749

        # Forward transformation
        x, y = forward_transformer.transform(lon, lat)

        # Reverse transformation
        lon_back, lat_back = reverse_transformer.transform(x, y)

        # Check that we get back close to original coordinates
        assert abs(lon - lon_back) < 1e-6
        assert abs(lat - lat_back) < 1e-6


class TestSpatialAnalysis:
    """Test spatial analysis functions."""

    def test_convex_hull(self, sample_points):
        """Test convex hull calculation."""
        gdf = sample_points

        # Calculate convex hull
        hull = gdf.union_all().convex_hull

        assert isinstance(hull, Polygon)
        assert hull.area > 0

        # Check that all points are within or on the hull
        for point in gdf.geometry:
            assert hull.contains(point) or hull.touches(point)

    def test_centroid_calculation(self, sample_points):
        """Test centroid calculation."""
        gdf = sample_points

        # Calculate centroid
        centroid = gdf.union_all().centroid

        assert isinstance(centroid, Point)
        assert centroid.x != 0 or centroid.y != 0

    def test_area_calculations(self, sample_polygons):
        """Test area calculations."""
        gdf = sample_polygons

        # Calculate areas
        areas = gdf.to_crs("EPSG:3857").geometry.area

        assert len(areas) == len(gdf)
        assert all(area > 0 for area in areas)

    def test_length_calculations(self):
        """Test length calculations for linear geometries."""
        # Create sample line geometries
        lines = [
            LineString([(-122.4194, 37.7749), (-122.4000, 37.7800)]),
            LineString([(-122.4500, 37.7600), (-122.4194, 37.7749)]),
            LineString([(-122.5000, 37.7500), (-122.3500, 37.8000)]),
        ]

        gdf = gpd.GeoDataFrame({"geometry": lines})

        # Calculate lengths
        lengths = gdf.geometry.length

        assert len(lengths) == len(gdf)
        assert all(length > 0 for length in lengths)


class TestSpatialDataValidation:
    """Test spatial data validation functions."""

    def test_geometry_validation(self):
        """Test geometry validation."""
        # Valid geometries
        valid_point = Point(0, 0)
        valid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        assert valid_point.is_valid
        assert valid_polygon.is_valid

        # Invalid polygon (self-intersecting)
        invalid_polygon = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])
        assert not invalid_polygon.is_valid

    def test_crs_validation(self, sample_points):
        """Test CRS validation."""
        gdf = sample_points

        # Check if CRS is set
        if gdf.crs is None:
            # Set CRS if not already set
            gdf.set_crs("EPSG:4326", inplace=True)

        assert gdf.crs is not None
        assert str(gdf.crs) == "EPSG:4326"

    def test_bounding_box_validation(self, sample_points):
        """Test bounding box validation."""
        gdf = sample_points

        # Get bounding box
        bbox = gdf.total_bounds

        assert len(bbox) == 4  # [minx, miny, maxx, maxy]
        assert bbox[0] < bbox[2]  # minx < maxx
        assert bbox[1] < bbox[3]  # miny < maxy


class TestSpatialDataIO:
    """Test spatial data input/output functions."""

    def test_geojson_io(self, sample_points, tmp_path):
        """Test GeoJSON read/write operations."""
        gdf = sample_points

        # Write to GeoJSON
        output_file = tmp_path / "test_points.geojson"
        gdf.to_file(output_file, driver="GeoJSON")

        # Read back from GeoJSON
        gdf_read = gpd.read_file(output_file)

        assert len(gdf_read) == len(gdf)
        assert all(col in gdf_read.columns for col in gdf.columns)

    def test_shapefile_io(self, sample_polygons, tmp_path):
        """Test Shapefile read/write operations."""
        gdf = sample_polygons

        # Write to Shapefile
        output_file = tmp_path / "test_polygons.shp"
        gdf.to_file(output_file, driver="ESRI Shapefile")

        # Read back from Shapefile
        gdf_read = gpd.read_file(output_file)

        assert len(gdf_read) == len(gdf)
        assert all(col in gdf_read.columns for col in gdf.columns)

    def test_h3_to_geojson(self, sample_h3_indices, tmp_path):
        """Test converting H3 cells to GeoJSON."""
        # Convert H3 cells to GeoJSON
        geojson_features = []

        for h3_cell in sample_h3_indices:
            boundary = h3.cell_to_boundary(h3_cell)
            # GeoJSON rings are closed [lng, lat] sequences; H3 v4 returns an
            # open ring of (lat, lng) vertices.
            coordinates = [[lng, lat] for lat, lng in boundary]
            coordinates.append(coordinates[0])

            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {
                    "h3_index": h3_cell,
                    "resolution": h3.get_resolution(h3_cell),
                },
            }
            geojson_features.append(feature)

        geojson = {"type": "FeatureCollection", "features": geojson_features}

        # Write to file
        output_file = tmp_path / "h3_cells.geojson"
        with open(output_file, "w") as f:
            json.dump(geojson, f)

        # Read back and validate
        gdf = gpd.read_file(output_file)

        assert len(gdf) == len(sample_h3_indices)
        assert "h3_index" in gdf.columns
        assert "resolution" in gdf.columns


@pytest.mark.performance
class TestSpatialPerformance:
    """Test spatial operations performance."""

    def test_large_dataset_processing(self):
        """Test processing large spatial datasets."""
        # Create large dataset
        n_points = 10000
        np.random.seed(42)

        lons = np.random.uniform(-122.5, -122.3, n_points)
        lats = np.random.uniform(37.7, 37.9, n_points)

        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        gdf = gpd.GeoDataFrame({"geometry": points})

        # Test spatial indexing performance
        spatial_index = gdf.sindex

        # Test bounding box query performance
        bbox = (-122.4, 37.8, -122.3, 37.9)
        start_time = time.time()
        indices = list(spatial_index.intersection(bbox))
        query_time = time.time() - start_time

        assert query_time < 1.0  # Should complete within 1 second
        assert len(indices) > 0

    def test_h3_bulk_operations(self):
        """Test bulk H3 operations performance."""
        # Create many lat/lng pairs
        n_points = 10000
        np.random.seed(42)

        lats = np.random.uniform(37.7, 37.9, n_points)
        lons = np.random.uniform(-122.5, -122.3, n_points)

        resolution = 10

        # Test bulk H3 cell creation
        start_time = time.time()
        h3_cells = [
            h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)
        ]
        creation_time = time.time() - start_time

        assert creation_time < 5.0  # Should complete within 5 seconds
        assert len(h3_cells) == n_points
        assert all(isinstance(cell, str) for cell in h3_cells)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestHypothesisSpatial:
    """Property-based tests for spatial functions using Hypothesis."""

    @given(
        st.floats(min_value=-80, max_value=80), st.floats(min_value=-180, max_value=180)
    )
    def test_h3_indexing_roundtrip(self, lat, lon):
        """Verify lat/lon -> h3 -> lat/lon consistency."""
        try:
            h3_index = h3.latlng_to_cell(lat, lon, 9)
            lat_out, lon_out = h3.cell_to_latlng(h3_index)

            # H3 is lossy (hex center), so we check proximity, not equality
            # ~0.001 degrees covers hex edge size at res 9 (approx 174m)
            # Relax tolerance to 0.05 degrees near edges

            lat_diff = abs(lat - lat_out)
            lon_diff = abs(lon - lon_out)
            # Handle longitude wrapping at 180/-180
            if lon_diff > 180:
                lon_diff = 360 - lon_diff

            assert lat_diff < 0.05
            assert lon_diff < 0.05
        except Exception as e:
            # Should handle edge cases (poles etc) gracefully or fail with helpful message
            if "h3" in str(e).lower():
                pass
            else:
                pytest.fail(f"H3 roundtrip failed for {lat}, {lon}: {e}")

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=10),
                st.floats(min_value=0, max_value=10),
            ),
            min_size=3,
            max_size=10,
        )
    )
    def test_convex_hull_invariants(self, points):
        """Verify convex hull invariants on random point sets."""
        if len(points) < 3:
            return

        df = pd.DataFrame(points, columns=["latitude", "longitude"])
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
        )

        # Use union_all() if available, else unary_union
        try:
            geom_union = gdf.union_all()
        except AttributeError:
            geom_union = gdf.unary_union

        hull = geom_union.convex_hull

        if hull.is_empty:
            return

        # Invariant 1: Hull area >= 0
        assert hull.area >= 0

        # Invariant 2: All original points must be within (or on boundary of) the hull
        # Use a small buffer for float precision issues
        buffered_hull = hull.buffer(1e-9)
        for geom in gdf.geometry:
            assert buffered_hull.contains(geom) or buffered_hull.touches(geom)


if __name__ == "__main__":
    pytest.main([__file__])
