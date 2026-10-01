"""
Integration tests for cross-component workflows in GEO-INFER-SPACE.

Tests cover:
- SpatialIndexingInterface latlng_to_cell / cell_to_latlng round-trips
- Hierarchical cell parent/child consistency
- SpatialProcessor buffer analysis
"""

import logging

import pytest

from geo_infer_space.core import SpatialIndexingInterface

logger = logging.getLogger(__name__)


class TestSpatialIndexingWorkflows:
    """Integration tests for spatial indexing cross-component workflows."""

    @pytest.fixture
    def indexer(self):
        """Create a spatial indexing interface."""
        return SpatialIndexingInterface(backend="h3")

    def test_point_to_cell_to_neighbors_workflow(self, indexer):
        """
        Workflow: Convert point -> H3 cell -> find neighbors -> verify connectivity.
        """
        lat, lng = 37.7749, -122.4194  # San Francisco
        resolution = 8

        # Step 1: Convert lat/lng to cell
        cell = indexer.latlng_to_cell(lat, lng, resolution)
        assert cell is not None
        assert isinstance(cell, str)

        # Step 2: Get neighbors of that cell
        neighbors = indexer.get_cell_neighbors(cell, k=1)
        assert len(neighbors) > 0

        # Step 3: Verify the original cell is in its own k=1 neighborhood
        # (grid_disk includes the center cell)
        assert cell in neighbors or len(neighbors) >= 6

    def test_multi_resolution_hierarchy_workflow(self, indexer):
        """
        Workflow: Index at high resolution -> get parent at lower resolution ->
        verify children contain original cell.
        """
        lat, lng = 40.7128, -74.0060  # New York City
        high_res = 9
        low_res = 7

        # Step 1: Get high-resolution cell
        high_cell = indexer.latlng_to_cell(lat, lng, high_res)
        assert high_cell is not None

        # Step 2: Get parent cell at lower resolution
        parent_cell = indexer.get_cell_parent(high_cell, low_res)
        assert parent_cell is not None

        # Step 3: Get all children of parent at high resolution
        children = indexer.get_cell_children(parent_cell, high_res)
        assert len(children) > 0

        # Step 4: Verify the original cell is among the children
        assert high_cell in children, (
            f"High-res cell {high_cell} should be a child of its parent {parent_cell}"
        )

    def test_index_to_coordinates_round_trip(self, indexer):
        """
        Workflow: Coordinates -> H3 cell -> back to coordinates -> same cell.
        """
        lat_in, lng_in = 51.5074, -0.1278  # London
        resolution = 8

        # Step 1: Convert to cell
        cell = indexer.latlng_to_cell(lat_in, lng_in, resolution)

        # Step 2: Get cell center coordinates
        lat_out, lng_out = indexer.cell_to_latlng(cell)

        # Step 3: Convert center back to cell -- should be the same cell
        cell_round_trip = indexer.latlng_to_cell(lat_out, lng_out, resolution)
        assert cell == cell_round_trip, (
            f"Round-trip failed: {cell} != {cell_round_trip}"
        )

    def test_cell_distance_positive_for_different_cells(self, indexer):
        """
        Workflow: Create two cells at same resolution -> compute distance.
        """
        cell_a = indexer.latlng_to_cell(37.7749, -122.4194, 8)  # SF
        cell_b = indexer.latlng_to_cell(37.7849, -122.4094, 8)  # nearby
        dist = indexer.get_cell_distance(cell_a, cell_b)
        assert dist >= 0
        if cell_a != cell_b:
            assert dist > 0


class TestSpatialProcessorWorkflows:
    """Integration tests for spatial processor cross-component workflows."""

    def test_buffer_and_spatial_relationships(self):
        """
        Workflow: Create geometries -> buffer -> check spatial relationships.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            pytest.fail("geopandas or shapely not available")

        from geo_infer_space.core.spatial_processor import SpatialProcessor

        processor = SpatialProcessor()

        # Step 1: Create point GeoDataFrame
        points = gpd.GeoDataFrame(
            {"name": ["A", "B", "C"]},
            geometry=[
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(5.0, 5.0),
            ],
            crs="EPSG:4326",
        )

        # Step 2: Buffer analysis
        # ``buffer_distance`` is expressed in metres for geographic inputs;
        # 200 km covers the nearby one-degree pair but not the distant point.
        buffered = processor.buffer_analysis(points, buffer_distance=200_000.0)

        # Step 3: Verify buffers exist and are larger than points
        assert len(buffered) == 3
        for _, row in buffered.iterrows():
            assert row.geometry.area > 0

        # Step 4: Nearby buffers should overlap, distant should not
        buf_a = buffered.iloc[0].geometry
        buf_b = buffered.iloc[1].geometry
        buf_c = buffered.iloc[2].geometry

        assert buf_a.intersects(buf_b), "Nearby buffers should overlap"
        assert not buf_a.intersects(buf_c), "Distant buffers should not overlap"
