"""
Tests for Unified ML Integration using SpatialIndexingInterface.

Verifies that ML feature engineering workflows can be executed using the
unified spatial architecture (``SpatialIndexingInterface``).
"""

import pytest
from datetime import datetime
import logging

# Unified Spatial Architecture
from geo_infer_space.core import SpatialIndexingInterface

logger = logging.getLogger(__name__)


class TestUnifiedMLIntegration:
    """Test ML integration using unified interfaces."""

    @pytest.fixture
    def indexer(self):
        """Fixture for spatial indexer."""
        return SpatialIndexingInterface(backend="h3")

    def test_feature_engineering_workflow(self, indexer):
        """
        Test the ML feature engineering workflow demonstrated in advanced applications.
        Verifies we can generate spatial features from raw coordinates.
        """
        # 1. Setup Test Data (San Francisco Area)
        sf_area = [
            (37.7749, -122.4194),  # Downtown
            (37.7849, -122.4094),  # North Beach
        ]

        # 2. Generate Grid and Features
        features = []
        for lat, lng in sf_area:
            # Convert to cell
            cell_index = indexer.latlng_to_cell(lat, lng, 9)
            assert isinstance(cell_index, str)

            # Get spatial context (neighbors)
            neighbors = indexer.get_cell_neighbors(cell_index, k=1)
            assert len(neighbors) > 0

            # Create feature vector (Simulation)
            feature_vector = {
                "cell_id": cell_index,
                "lat": lat,
                "lng": lng,
                "neighbor_count": len(neighbors),
                "hour_of_day": datetime.now().hour,
                # Verify we can perform distance calc for features
                "dist_to_center": indexer.get_cell_distance(cell_index, neighbors[0]),
            }
            features.append(feature_vector)

        # 3. Assertions
        assert len(features) == 2
        for f in features:
            assert "cell_id" in f
            assert f["neighbor_count"] > 0
            assert f["dist_to_center"] >= 0

    def test_spatial_context_features(self, indexer):
        """Test extraction of spatial context features (rings/neighbors)."""
        # Center point
        lat, lng = 37.7749, -122.4194
        center_cell = indexer.latlng_to_cell(lat, lng, 9)

        # Get 2-ring neighborhood
        k2_neighbors = indexer.get_cell_neighbors(center_cell, k=2)
        k1_neighbors = indexer.get_cell_neighbors(center_cell, k=1)

        # Verify ring properties for feature engineering
        assert len(k2_neighbors) > len(k1_neighbors)

        # Simulate aggregation feature (e.g., density of POIs in ring 2)
        # This confirms we can perform the spatial lookups needed for ML
        assert isinstance(k2_neighbors, (list, set))

    def test_backend_switch_consistency(self):
        """Verify features can be generated consistently across backends (where available)."""
        # This test mainly ensures the interface holds up
        h3_indexer = SpatialIndexingInterface(backend="h3")

        lat, lng = 37.7749, -122.4194
        cell = h3_indexer.latlng_to_cell(lat, lng, 9)

        # If we had SRAI enabled and configured, we would test equivalence here
        # For now, just asserting the H3 backend is functional for ML pipelines
        assert cell is not None
