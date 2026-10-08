"""
Comprehensive tests for the GEO-INFER-SPACE dispatch system.

Tests the multi-backend dispatch abstraction to ensure operations
are correctly routed to the appropriate backend.
"""

import pytest

from geo_infer_space.core.dispatcher import (
    SpatialBackendDispatcher,
    get_backend_dispatcher,
    reset_dispatcher,
)
from geo_infer_space.core.analytics import SpatialAnalyticsInterface
from geo_infer_space.core.geometric_operations import GeometricOperationsInterface
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface


@pytest.fixture(autouse=True)
def reset_global_dispatcher():
    """Reset the global dispatcher before each test."""
    reset_dispatcher()
    yield
    reset_dispatcher()


class TestDispatcherInitialization:
    """Test dispatcher initialization and backend loading."""

    def test_dispatcher_creation(self):
        """Test basic dispatcher creation."""
        dispatcher = SpatialBackendDispatcher()
        assert dispatcher is not None
        assert isinstance(dispatcher.backends, dict)

    def test_h3_backend_auto_loaded(self):
        """Test that H3 backend is automatically loaded."""
        dispatcher = SpatialBackendDispatcher()

        # H3 should be available if the library is installed
        available = dispatcher.get_available_backends()
        assert "h3" in available

    def test_get_backend_by_name(self):
        """Test retrieving a specific backend."""
        dispatcher = SpatialBackendDispatcher()

        h3_backend = dispatcher.get_backend("h3")
        assert h3_backend is not None
        assert h3_backend.name == "h3"

    def test_get_nonexistent_backend(self):
        """Test retrieving a backend that doesn't exist."""
        dispatcher = SpatialBackendDispatcher()

        result = dispatcher.get_backend("nonexistent")
        assert result is None

    def test_backend_info(self):
        """Test getting backend information."""
        dispatcher = SpatialBackendDispatcher()

        info = dispatcher.get_backend_info()
        assert "h3" in info
        assert "available" in info["h3"]
        assert "capabilities" in info["h3"]


class TestGlobalDispatcher:
    """Test global dispatcher singleton pattern."""

    def test_get_global_dispatcher(self):
        """Test getting the global dispatcher instance."""
        dispatcher1 = get_backend_dispatcher()
        dispatcher2 = get_backend_dispatcher()

        # Should be the same instance
        assert dispatcher1 is dispatcher2

    def test_reset_dispatcher(self):
        """Test resetting the global dispatcher."""
        dispatcher1 = get_backend_dispatcher()
        reset_dispatcher()
        dispatcher2 = get_backend_dispatcher()

        # Should be different instances after reset
        assert dispatcher1 is not dispatcher2


class TestDefaultBackendConfiguration:
    """Test default backend configuration."""

    def test_set_default_backend(self):
        """Test setting default backend for operation type."""
        dispatcher = SpatialBackendDispatcher()

        dispatcher.set_default_backend("indexing", "h3")
        assert dispatcher.get_default_backend("indexing") == "h3"

    def test_no_implicit_default_fallback(self):
        """Without a registered default, get_default_backend returns None."""
        dispatcher = SpatialBackendDispatcher()

        # Operation types standardly covered by a successfully loaded h3
        # backend get a registered default; anything else returns None.
        assert dispatcher.get_default_backend("some_operation") is None

    def test_dispatch_without_usable_default_raises_precise_error(self):
        """Dispatching with no usable default lists available backends."""
        dispatcher = SpatialBackendDispatcher()
        dispatcher.backends.clear()
        dispatcher.default_backends.clear()

        with pytest.raises(ValueError, match="available backends"):
            dispatcher.dispatch_indexing_operation("latlng_to_cell", 37.7, -122.4, 8)

    def test_set_invalid_default_backend(self):
        """Test setting a non-existent backend as default."""
        dispatcher = SpatialBackendDispatcher()

        with pytest.raises(ValueError, match="not registered"):
            dispatcher.set_default_backend("indexing", "nonexistent")


class TestIndexingOperationDispatch:
    """Test dispatching indexing operations."""

    @pytest.fixture
    def dispatcher(self):
        return SpatialBackendDispatcher()

    def test_dispatch_latlng_to_cell(self, dispatcher):
        """Test dispatching latlng_to_cell operation."""
        result = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_dispatch_cell_to_latlng(self, dispatcher):
        """Test dispatching cell_to_latlng operation."""
        # First get a valid cell
        cell = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8
        )

        # Then convert back
        lat, lng = dispatcher.dispatch_indexing_operation("cell_to_latlng", cell)

        assert -90 <= lat <= 90
        assert -180 <= lng <= 180

    def test_dispatch_get_neighbors(self, dispatcher):
        """Test dispatching get_neighbors operation."""
        cell = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8
        )

        neighbors = dispatcher.dispatch_indexing_operation("get_neighbors", cell, k=1)

        assert isinstance(neighbors, list)
        assert len(neighbors) == 6  # Hexagon has 6 neighbors

    def test_dispatch_get_cell_parent(self, dispatcher):
        """Test dispatching get_cell_parent operation."""
        cell = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8
        )

        parent = dispatcher.dispatch_indexing_operation("get_cell_parent", cell, 5)

        assert isinstance(parent, str)
        assert len(parent) > 0

    def test_dispatch_get_cell_children(self, dispatcher):
        """Test dispatching get_cell_children operation."""
        cell = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 5
        )

        children = dispatcher.dispatch_indexing_operation("get_cell_children", cell, 6)

        assert isinstance(children, list)
        assert len(children) == 7  # Each cell has 7 children

    def test_dispatch_compact_uncompact(self, dispatcher):
        """Test compact and uncompact operations."""
        # Get a parent cell and its children
        parent = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 5
        )
        children = dispatcher.dispatch_indexing_operation(
            "get_cell_children", parent, 6
        )

        # Compact should reduce back
        compacted = dispatcher.dispatch_indexing_operation("compact_cells", children)
        assert len(compacted) <= len(children)

        # Uncompact should expand
        uncompacted = dispatcher.dispatch_indexing_operation(
            "uncompact_cells", compacted, 6
        )
        assert len(uncompacted) == len(children)

    def test_dispatch_unknown_operation(self, dispatcher):
        """Test dispatching an unknown operation."""
        with pytest.raises(ValueError, match="Unknown indexing operation"):
            dispatcher.dispatch_indexing_operation("unknown_operation", "arg1")

    def test_dispatch_with_explicit_backend(self, dispatcher):
        """Test dispatching with explicit backend specification."""
        result = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8, backend="h3"
        )

        assert isinstance(result, str)


class TestAnalyticsOperationDispatch:
    """Test dispatching analytics operations."""

    @pytest.fixture
    def dispatcher(self):
        return SpatialBackendDispatcher()

    @pytest.fixture
    def test_cells(self, dispatcher):
        """Generate test cells for analytics."""
        # Create a cluster of cells
        center = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8
        )
        neighbors = dispatcher.dispatch_indexing_operation("get_neighbors", center, k=2)
        neighbors.insert(0, center)
        return neighbors

    def test_dispatch_analyze_hotspots(self, dispatcher, test_cells):
        """Test dispatching analyze_hotspots operation."""
        values = [i * 10 for i in range(len(test_cells))]

        result = dispatcher.dispatch_analytics_operation(
            "analyze_hotspots", {"cells": test_cells, "values": values}
        )

        assert "hotspots" in result
        assert "threshold" in result
        assert "total_cells" in result

    def test_dispatch_compute_proximity(self, dispatcher):
        """Test dispatching compute_proximity operation."""
        points = [
            (37.7749, -122.4194),
            (37.7849, -122.4094),
            (37.7649, -122.4294),
        ]

        result = dispatcher.dispatch_analytics_operation("compute_proximity", points)

        assert "proximity_pairs" in result
        assert "total_points" in result
        assert result["total_points"] == 3

    def test_dispatch_find_clusters(self, dispatcher, test_cells):
        """Test dispatching find_clusters operation."""
        values = [1.0] * len(test_cells)

        result = dispatcher.dispatch_analytics_operation(
            "find_clusters",
            test_cells,
            values,
            min_cluster_size=3,
            distance_threshold=1,
        )

        assert "clusters" in result
        assert "num_clusters" in result
        assert "noise_cells" in result

    def test_dispatch_calculate_density(self, dispatcher, test_cells):
        """Test dispatching calculate_density operation."""
        values = [1.0 + i * 0.5 for i in range(len(test_cells))]

        result = dispatcher.dispatch_analytics_operation(
            "calculate_density", test_cells, values, kernel_radius=1
        )

        assert "densities" in result
        assert "statistics" in result
        assert len(result["densities"]) == len(test_cells)

    def test_dispatch_spatial_join(self, dispatcher, test_cells):
        """Test dispatching spatial_join operation."""
        cells_a = test_cells[:5]
        cells_b = test_cells[3:8]  # Overlapping

        result = dispatcher.dispatch_analytics_operation(
            "spatial_join", cells_a, cells_b, join_type="intersects"
        )

        assert "matches" in result
        assert "match_count" in result
        assert result["match_count"] > 0  # Should have overlapping matches

    def test_dispatch_interpolate_values(self, dispatcher, test_cells):
        """Test dispatching interpolate_values operation."""
        source_cells = test_cells[:5]
        source_values = [10.0, 20.0, 30.0, 40.0, 50.0]
        target_cells = test_cells[3:8]

        result = dispatcher.dispatch_analytics_operation(
            "interpolate_values",
            source_cells,
            source_values,
            target_cells,
            method="idw",
        )

        assert "interpolated" in result
        assert "method" in result
        assert result["method"] == "idw"
        assert len(result["interpolated"]) == len(target_cells)

    def test_dispatch_unknown_analytics_operation(self, dispatcher):
        """Test dispatching an unknown analytics operation."""
        with pytest.raises(ValueError, match="Unknown analytics operation"):
            dispatcher.dispatch_analytics_operation("unknown_analytics", "arg1")


class TestPublicInterfaces:
    """Exercise the public facades against the real H3 backend."""

    def test_spatial_indexing_radius_adapter_returns_a_disk(self):
        interface = SpatialIndexingInterface(backend="h3")
        center = interface.latlng_to_cell(37.7749, -122.4194, 9)
        neighbors = interface.get_neighbors((37.7749, -122.4194), 500, resolution=9)

        assert neighbors
        assert center not in neighbors
        assert all(
            interface.dispatcher.get_backend("h3").is_valid_cell(cell)
            for cell in neighbors
        )

    def test_geometric_facade_dispatches_to_native_shapely_operations(self):
        interface = GeometricOperationsInterface(backend="h3")
        square = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
        }

        assert interface.calculate_area(square) == 4.0
        assert interface.calculate_centroid(square) == (1.0, 1.0)
        buffered = interface.buffer_geometry(square, 0.1)
        assert buffered["type"] == "Polygon"

    def test_analytics_facade_adapts_points_and_aliases(self):
        indexer = SpatialIndexingInterface(backend="h3")
        center = indexer.latlng_to_cell(37.7749, -122.4194, 9)
        analytics = SpatialAnalyticsInterface(backend="h3")

        hotspots = analytics.find_hotspots(
            {"cells": [center], "values": [2.0], "threshold": 1.0}
        )
        assert hotspots["hotspot_count"] == 1

        interpolation = analytics.interpolate_values(
            [(37.7749, -122.4194, 4.0)],
            target_points=[(37.7749, -122.4194)],
        )
        assert interpolation["interpolated"][center] == 4.0


class TestBackendSwitching:
    """Test switching between backends."""

    def test_operation_same_across_backends(self):
        """Test that operations produce the same result type across backends."""
        dispatcher = SpatialBackendDispatcher()

        # Test with H3 backend
        result_h3 = dispatcher.dispatch_indexing_operation(
            "latlng_to_cell", 37.7749, -122.4194, 8, backend="h3"
        )

        assert isinstance(result_h3, str)

    def test_backend_capabilities_match_operations(self):
        """Test that backend capabilities match available operations."""
        dispatcher = SpatialBackendDispatcher()
        h3_backend = dispatcher.get_backend("h3")

        capabilities = h3_backend.get_capabilities()

        # Check indexing capabilities
        assert "indexing" in capabilities
        assert capabilities["indexing"]["latlng_to_cell"] is True
        assert capabilities["indexing"]["polygon_to_cells"] is True

        # Check analytics capabilities
        assert "analytics" in capabilities
        assert capabilities["analytics"]["analyze_hotspots"] is True


class TestEndToEndDispatchWorkflow:
    """End-to-end tests for complete dispatch workflows."""

    def test_complete_spatial_analysis_workflow(self):
        """Test a complete spatial analysis workflow through dispatch."""
        dispatcher = SpatialBackendDispatcher()

        # Step 1: Convert locations to cells
        locations = [
            (37.7749, -122.4194),  # San Francisco
            (37.7849, -122.4294),
            (37.7649, -122.4094),
            (37.7749, -122.4094),
            (37.7849, -122.4194),
        ]

        cells = []
        for lat, lng in locations:
            cell = dispatcher.dispatch_indexing_operation("latlng_to_cell", lat, lng, 8)
            cells.append(cell)

        assert len(cells) == 5

        # Step 2: Analyze hotspots
        values = [100, 50, 75, 200, 25]
        hotspot_result = dispatcher.dispatch_analytics_operation(
            "analyze_hotspots", {"cells": cells, "values": values}
        )

        assert hotspot_result["total_cells"] == 5

        # Step 3: Find clusters
        cluster_result = dispatcher.dispatch_analytics_operation(
            "find_clusters", cells, values, min_cluster_size=2, distance_threshold=2
        )

        assert "clusters" in cluster_result

        # Step 4: Calculate density
        density_result = dispatcher.dispatch_analytics_operation(
            "calculate_density", cells, values, kernel_radius=1
        )

        assert len(density_result["densities"]) == 5

        # Step 5: Get cell boundaries for visualization
        for cell in cells[:2]:
            boundary = dispatcher.dispatch_indexing_operation("get_cell_boundary", cell)
            assert len(boundary) >= 6  # Hexagon has at least 6 vertices


class TestOptionalBackendAdmission:
    """Default H3 operations do not import unrelated optional libraries."""

    def test_h3_cold_process_does_not_import_srai(self):
        import subprocess
        import sys

        script = """
import sys
from geo_infer_space.core.dispatcher import SpatialBackendDispatcher
assert 'srai' not in sys.modules
dispatcher = SpatialBackendDispatcher()
cell = dispatcher.dispatch_indexing_operation('latlng_to_cell', 37.7, -122.4, 8)
assert dispatcher.get_backend('h3').is_valid_cell(cell)
assert 'srai' not in sys.modules
assert 'torch' not in sys.modules
"""
        with subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            assert process.returncode == 0, (stdout, stderr)

    def test_broken_optional_backend_propagates_only_when_requested(self, monkeypatch):
        def broken_loader(self):
            raise ModuleNotFoundError("broken installed SRAI dependency", name="torch")

        monkeypatch.setattr(SpatialBackendDispatcher, "_load_srai_backend", broken_loader)
        dispatcher = SpatialBackendDispatcher()
        assert dispatcher.dispatch_indexing_operation('latlng_to_cell', 37.7, -122.4, 8)
        with pytest.raises(ModuleNotFoundError, match="broken installed SRAI"):
            dispatcher.get_backend("srai")
        with pytest.raises(ModuleNotFoundError, match="broken installed SRAI"):
            dispatcher.get_available_backends()

    def test_optional_backend_realized_once_with_real_capabilities(self, monkeypatch):
        original_loader = SpatialBackendDispatcher._load_srai_backend
        calls = []

        def counted_loader(self):
            calls.append("srai")
            return original_loader(self)

        monkeypatch.setattr(SpatialBackendDispatcher, "_load_srai_backend", counted_loader)
        dispatcher = SpatialBackendDispatcher()
        assert calls == []
        backend = dispatcher.get_backend("srai")
        assert dispatcher.get_backend("srai") is backend
        assert "srai" in dispatcher.get_backend_info()
        assert dispatcher.backend_capabilities["srai"] == backend.get_capabilities()
        assert calls == ["srai"]
