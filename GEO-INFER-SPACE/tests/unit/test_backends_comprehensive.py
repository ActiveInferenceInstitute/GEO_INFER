#!/usr/bin/env python3
"""
Comprehensive Backend Tests for GEO-INFER-SPACE.

This module provides full coverage testing for both H3 and SRAI backends
via the unified spatial interface. All tests use real methods - no mocks.
"""

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def h3_backend():
    """Create an H3 backend instance."""
    from geo_infer_space.backends.h3 import H3Backend

    return H3Backend()


@pytest.fixture
def srai_backend():
    """Create an SRAI backend instance."""
    from geo_infer_space.backends.srai import SraiBackend

    return SraiBackend()


@pytest.fixture
def sample_polygon():
    """Return a simple test polygon (San Francisco area)."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.5, 37.7],
                [-122.3, 37.7],
                [-122.3, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7],
            ]
        ],
    }


@pytest.fixture
def sample_points():
    """Return sample points for proximity analysis."""
    return [
        (37.7749, -122.4194),  # San Francisco
        (37.8044, -122.2712),  # Oakland
        (37.5485, -122.0590),  # Fremont
        (37.3382, -121.8863),  # San Jose
    ]


@pytest.fixture
def sample_hotspot_data():
    """Return sample data for hotspot analysis."""
    # Generate some cells around San Francisco
    import h3

    center_cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
    cells = list(h3.grid_disk(center_cell, 5))

    # Create values with some clear hotspots
    import random

    random.seed(42)
    values = [random.uniform(0, 50) for _ in range(len(cells))]

    # Make a few hotspots
    for i in [0, 10, 20]:
        if i < len(values):
            values[i] = 100 + random.uniform(0, 50)

    return {"cells": cells, "values": values}


# ============================================================================
# H3 Backend Tests
# ============================================================================


class TestH3BackendBasic:
    """Basic H3 backend functionality tests."""

    def test_h3_backend_initialization(self, h3_backend):
        """Test H3 backend initializes correctly."""
        assert h3_backend.name == "h3"
        assert h3_backend.is_available() is True
        assert "4." in h3_backend.version  # H3 v4.x

    def test_h3_backend_capabilities(self, h3_backend):
        """Test H3 backend reports correct capabilities."""
        caps = h3_backend.get_capabilities()

        assert "indexing" in caps
        assert "analytics" in caps
        assert caps["indexing"]["latlng_to_cell"] is True
        assert caps["indexing"]["cell_to_latlng"] is True
        assert caps["indexing"]["polygon_to_cells"] is True


class TestH3BackendIndexing:
    """H3 backend spatial indexing tests."""

    def test_latlng_to_cell(self, h3_backend):
        """Test coordinate to cell conversion."""
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)

        assert isinstance(cell, str)
        assert len(cell) == 15  # H3 cell IDs are 15 chars
        assert cell.startswith("89")  # Resolution 9 prefix

    def test_cell_to_latlng(self, h3_backend):
        """Test cell to coordinate conversion."""
        # First convert to cell
        original_lat, original_lng = 37.7749, -122.4194
        cell = h3_backend.latlng_to_cell(original_lat, original_lng, 9)

        # Then convert back
        lat, lng = h3_backend.cell_to_latlng(cell)

        # Should be within cell bounds (close to original)
        assert abs(lat - original_lat) < 0.01
        assert abs(lng - original_lng) < 0.01

    def test_polygon_to_cells(self, h3_backend, sample_polygon):
        """Test polygon to cells conversion."""
        cells = h3_backend.polygon_to_cells(sample_polygon, 8)

        assert isinstance(cells, list)
        assert len(cells) > 0
        # All cells should be valid H3 identifiers
        for cell in cells:
            assert isinstance(cell, str)
            assert len(cell) == 15

    def test_get_cell_neighbors(self, h3_backend):
        """Test neighbor cell retrieval."""
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        neighbors = h3_backend.get_cell_neighbors(cell, k=1)

        assert isinstance(neighbors, list)
        assert len(neighbors) == 6  # Hexagons have 6 neighbors
        assert cell not in neighbors  # Center should not be included

    def test_get_cell_neighbors_k2(self, h3_backend):
        """Test neighbor retrieval with k=2."""
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        neighbors = h3_backend.get_cell_neighbors(cell, k=2)

        assert isinstance(neighbors, list)
        assert len(neighbors) > 6  # k=2 has more neighbors

    def test_get_cell_distance(self, h3_backend):
        """Test distance calculation between cells."""
        cell1 = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        cell2 = h3_backend.latlng_to_cell(37.7850, -122.4094, 9)

        distance = h3_backend.get_cell_distance(cell1, cell2)

        assert isinstance(distance, int)
        assert distance >= 0

    def test_compact_cells(self, h3_backend):
        """Test cell compaction."""
        # Get a set of cells
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        neighbors = h3_backend.get_cell_neighbors(cell, k=1)
        all_cells = [cell] + neighbors

        compacted = h3_backend.compact_cells(all_cells)

        assert isinstance(compacted, list)
        assert len(compacted) <= len(all_cells)

    def test_uncompact_cells(self, h3_backend):
        """Test cell uncompaction."""
        # Get a cell at resolution 7
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 7)

        # Uncompact to resolution 8
        uncompacted = h3_backend.uncompact_cells([cell], 8)

        assert isinstance(uncompacted, list)
        assert len(uncompacted) > 1  # Should expand to more cells


class TestH3BackendNewMethods:
    """Tests for new spatial methods added to H3 backend."""

    def test_get_cell_resolution(self, h3_backend):
        """Test getting cell resolution."""
        for res in [5, 9, 12]:
            cell = h3_backend.latlng_to_cell(37.7749, -122.4194, res)
            detected_res = h3_backend.get_cell_resolution(cell)
            assert detected_res == res

    def test_get_cell_boundary(self, h3_backend):
        """Test getting cell boundary coordinates."""
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        boundary = h3_backend.get_cell_boundary(cell)

        assert isinstance(boundary, list)
        assert len(boundary) == 6  # Hexagons have 6 vertices

        for point in boundary:
            lat, lng = point
            assert -90 <= lat <= 90
            assert -180 <= lng <= 180

    def test_get_cell_area(self, h3_backend):
        """Test getting cell area."""
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        area = h3_backend.get_cell_area(cell)

        assert isinstance(area, float)
        assert area > 0
        # Resolution 9 cells are approximately 0.1 km²
        assert 0.01 < area < 1.0

    def test_cells_to_multipolygon(self, h3_backend):
        """Test converting cells to GeoJSON MultiPolygon."""
        cell = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)
        neighbors = h3_backend.get_cell_neighbors(cell, k=1)
        all_cells = [cell] + neighbors[:3]

        geojson = h3_backend.cells_to_multipolygon(all_cells)

        assert geojson["type"] == "MultiPolygon"
        assert geojson["coordinates"]

        # Native H3 topology dissolves shared edges, so adjacent cells may
        # produce fewer polygon parts than input cells.
        for poly in geojson["coordinates"]:
            assert poly
            for ring in poly:
                assert len(ring) >= 4
                assert ring[0] == ring[-1]

    def test_cells_to_multipolygon_empty(self, h3_backend):
        """Test converting empty cell list to MultiPolygon."""
        geojson = h3_backend.cells_to_multipolygon([])

        assert geojson["type"] == "MultiPolygon"
        assert geojson["coordinates"] == []


class TestH3BackendAnalytics:
    """H3 backend analytics tests."""

    def test_analyze_hotspots(self, h3_backend, sample_hotspot_data):
        """Test hotspot analysis."""
        result = h3_backend.analyze_hotspots(sample_hotspot_data)

        assert "hotspots" in result
        assert "threshold" in result
        assert "total_cells" in result
        assert "hotspot_count" in result

        assert result["total_cells"] == len(sample_hotspot_data["cells"])
        assert result["hotspot_count"] == len(result["hotspots"])

    def test_compute_proximity(self, h3_backend, sample_points):
        """Test proximity analysis."""
        result = h3_backend.compute_proximity(sample_points)

        assert "proximity_pairs" in result
        assert "total_points" in result
        assert "analyzed_pairs" in result

        assert result["total_points"] == len(sample_points)
        # n points should have n*(n-1)/2 pairs
        expected_pairs = len(sample_points) * (len(sample_points) - 1) // 2
        assert result["analyzed_pairs"] <= expected_pairs


# ============================================================================
# SRAI Backend Tests
# ============================================================================


class TestSRAIBackendBasic:
    """Basic SRAI backend functionality tests."""

    def test_srai_backend_initialization(self, srai_backend):
        """Test SRAI backend initializes correctly."""
        assert srai_backend.name == "srai"
        # SRAI may or may not be available
        assert isinstance(srai_backend.is_available(), bool)

    def test_srai_backend_capabilities(self, srai_backend):
        """Test SRAI backend reports correct capabilities."""
        caps = srai_backend.get_capabilities()

        assert "indexing" in caps
        assert "analytics" in caps
        assert "regionalizers" in caps
        assert "embedders" in caps


# Check if SRAI is available for conditional tests
try:
    import srai  # noqa: F401

    SRAI_INSTALLED = True
except ImportError:
    SRAI_INSTALLED = False


class TestSRAIBackendWithLibrary:
    """SRAI backend tests that require SRAI to be installed."""

    def test_srai_latlng_to_cell(self, srai_backend):
        """Test SRAI coordinate to cell conversion."""
        if not srai_backend.is_available():
            from geo_infer_space.core.interfaces import SRAIUnavailableError

            with pytest.raises(SRAIUnavailableError):
                srai_backend.latlng_to_cell(37.7749, -122.4194, 9)
            return

        cell = srai_backend.latlng_to_cell(37.7749, -122.4194, 9)
        assert isinstance(cell, str)

    def test_srai_cell_to_latlng(self, srai_backend):
        """Test SRAI cell to coordinate conversion."""
        if not srai_backend.is_available():
            from geo_infer_space.core.interfaces import SRAIUnavailableError

            with pytest.raises(SRAIUnavailableError):
                srai_backend.cell_to_latlng("8928308280fffff")
            return

        cell = srai_backend.latlng_to_cell(37.7749, -122.4194, 9)
        lat, lng = srai_backend.cell_to_latlng(cell)

        assert abs(lat - 37.7749) < 0.01
        assert abs(lng - (-122.4194)) < 0.01


# ============================================================================
# Unified Interface Tests
# ============================================================================


class TestUnifiedInterface:
    """Tests for unified spatial interface dispatching to backends."""

    def test_dispatch_to_h3(self):
        """Test dispatching operations to H3 backend."""
        from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

        interface = SpatialIndexingInterface(backend="h3")
        cell = interface.latlng_to_cell(37.7749, -122.4194, 9)

        assert isinstance(cell, str)
        assert len(cell) == 15

    def test_dispatch_default_backend(self):
        """Test using default backend."""
        from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

        interface = SpatialIndexingInterface()
        cell = interface.latlng_to_cell(37.7749, -122.4194, 9)

        assert isinstance(cell, str)

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        from geo_infer_space.core.spatial_indexing import latlng_to_cell, cell_to_latlng

        cell = latlng_to_cell(37.7749, -122.4194, 9)
        assert isinstance(cell, str)

        lat, lng = cell_to_latlng(cell)
        assert isinstance(lat, float)
        assert isinstance(lng, float)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for proper error handling."""

    def test_h3_invalid_cell(self, h3_backend):
        """Test H3 handles invalid cell gracefully."""
        with pytest.raises(ValueError):
            h3_backend.cell_to_latlng("invalid_cell_id")

    def test_h3_invalid_resolution(self, h3_backend):
        """Test H3 handles invalid resolution."""
        # H3 should handle resolution validation
        with pytest.raises((ValueError, Exception)):
            h3_backend.latlng_to_cell(37.7749, -122.4194, 20)  # Max is 15

    def test_srai_unavailable_error(self):
        """Test SRAIUnavailableError is properly defined."""
        from geo_infer_space.core.interfaces import SRAIUnavailableError

        error = SRAIUnavailableError("test operation")
        assert "test operation" in str(error)
        assert "SRAI" in str(error)

    def test_h3_unavailable_error(self):
        """Test H3UnavailableError is properly defined."""
        from geo_infer_space.core.interfaces import H3UnavailableError

        error = H3UnavailableError("test operation")
        assert "test operation" in str(error)
        assert "H3" in str(error)


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_h3_implements_indexing_protocol(self, h3_backend):
        """Test H3 backend implements IndexingBackendProtocol."""

        # Check required methods exist
        assert hasattr(h3_backend, "latlng_to_cell")
        assert hasattr(h3_backend, "cell_to_latlng")
        assert hasattr(h3_backend, "polygon_to_cells")
        assert hasattr(h3_backend, "get_cell_neighbors")
        assert hasattr(h3_backend, "get_cell_distance")
        assert hasattr(h3_backend, "compact_cells")
        assert hasattr(h3_backend, "uncompact_cells")
        assert hasattr(h3_backend, "get_cell_resolution")
        assert hasattr(h3_backend, "get_cell_boundary")
        assert hasattr(h3_backend, "get_cell_area")
        assert hasattr(h3_backend, "cells_to_multipolygon")

    def test_h3_implements_analytics_protocol(self, h3_backend):
        """Test H3 backend implements AnalyticsBackendProtocol."""
        assert hasattr(h3_backend, "analyze_hotspots")
        assert hasattr(h3_backend, "compute_proximity")

    def test_srai_implements_protocols(self, srai_backend):
        """Test SRAI backend implements required protocols."""
        assert hasattr(srai_backend, "latlng_to_cell")
        assert hasattr(srai_backend, "cell_to_latlng")
        assert hasattr(srai_backend, "analyze_hotspots")
        assert hasattr(srai_backend, "compute_proximity")


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full workflows."""

    def test_full_spatial_workflow(self, h3_backend, sample_polygon):
        """Test complete spatial analysis workflow."""
        # 1. Convert polygon to cells
        cells = h3_backend.polygon_to_cells(sample_polygon, 8)
        assert len(cells) > 0

        # 2. Get resolution of first cell
        res = h3_backend.get_cell_resolution(cells[0])
        assert res == 8

        # 3. Get area of cells
        total_area = sum(h3_backend.get_cell_area(c) for c in cells[:5])
        assert total_area > 0

        # 4. Convert to GeoJSON
        geojson = h3_backend.cells_to_multipolygon(cells[:5])
        assert geojson["type"] == "MultiPolygon"
        assert geojson["coordinates"]

    def test_neighbor_chain(self, h3_backend):
        """Test chaining neighbor operations."""
        # Start with a cell
        center = h3_backend.latlng_to_cell(37.7749, -122.4194, 9)

        # Get immediate neighbors
        ring1 = h3_backend.get_cell_neighbors(center, k=1)
        assert len(ring1) == 6

        # Get distance from center to each neighbor
        for neighbor in ring1:
            distance = h3_backend.get_cell_distance(center, neighbor)
            assert distance == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
