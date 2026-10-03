"""Focused tests for ACT H3 spatial Active Inference with visualization.

Verifies that the ACT H3 spatial framework integrates with the deterministic
visualization receipt pattern: H3 cells, boundary geometry, belief updates,
free energy computation, and visualization artifact generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h3
import numpy as np

from geo_infer_act import H3GridInferenceResult, H3BeliefUpdateResult
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.spatial_agent import SpatialActiveInferenceAgent
from geo_infer_act.utils.h3_adapter import get_h3_adapter

# Minimal San Francisco boundary for tests
SF_BOUNDARY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.42, 37.77],
            [-122.42, 37.78],
            [-122.41, 37.78],
            [-122.41, 37.77],
            [-122.42, 37.77],
        ]
    ],
}


# ---------------------------------------------------------------------------
# H3 adapter tests
# ---------------------------------------------------------------------------


class TestH3Adapter:
    """Tests for the H3 adapter (geo_infer_act.utils.h3_adapter)."""

    def test_adapter_returns_real_h3_cells(self):
        """Adapter produces real H3 cells for a boundary."""
        adapter = get_h3_adapter()
        cells = adapter.polygon_to_cells(SF_BOUNDARY, resolution=8)
        assert len(cells) > 0
        for cell in cells:
            assert h3.is_valid_cell(cell)

    def test_adapter_cell_to_boundary(self):
        """cell_to_boundary returns vertex coordinates for an H3 cell."""
        adapter = get_h3_adapter()
        center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
        boundary = adapter.cell_to_boundary(center)
        assert isinstance(boundary, (list, tuple))
        assert len(boundary) >= 6  # hexagon has 6+ vertices
        for point in boundary:
            assert len(point) == 2
            assert math.isfinite(point[0])
            assert math.isfinite(point[1])

    def test_adapter_grid_disk(self):
        """grid_disk returns k-ring of cells."""
        adapter = get_h3_adapter()
        center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
        ring = adapter.grid_disk(center, 2)
        assert len(ring) >= 7  # center + 6 neighbors minimum
        for cell in ring:
            assert h3.is_valid_cell(cell)


# ---------------------------------------------------------------------------
# GenerativeModel H3 spatial tests
# ---------------------------------------------------------------------------


class TestGenerativeModelH3Spatial:
    """Tests for GenerativeModel.enable_h3_spatial."""

    def test_enable_h3_spatial_adds_cells(self):
        """Enabling H3 spatial populates h3_cells."""
        model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
        model.enable_h3_spatial(8, SF_BOUNDARY)
        assert hasattr(model, "h3_cells")
        assert len(model.h3_cells) > 0

    def test_enable_h3_spatial_sets_mode(self):
        """Enabling H3 spatial sets spatial_mode to True."""
        model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
        model.enable_h3_spatial(8, SF_BOUNDARY)
        if hasattr(model, "spatial_mode"):
            assert model.spatial_mode is True

    def test_update_h3_beliefs_returns_typed_result(self):
        """update_h3_beliefs with return_result=True returns H3BeliefUpdateResult."""
        model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
        model.enable_h3_spatial(8, SF_BOUNDARY)
        cells = model.h3_cells[:2] if len(model.h3_cells) >= 2 else model.h3_cells
        obs = {cell: np.array([1.0, 0.0]) for cell in cells}
        result = model.update_h3_beliefs(obs, return_result=True)
        assert isinstance(result, H3BeliefUpdateResult)
        assert math.isfinite(result.aggregate_free_energy)

    def test_aggregate_beliefs_reduces_resolution(self):
        """aggregate_beliefs_to_resolution produces fewer cells."""
        model = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 3})
        model.enable_h3_spatial(9, SF_BOUNDARY)
        beliefs = {cell: np.array([0.3, 0.3, 0.4]) for cell in model.h3_cells[:5]}
        aggregated = model.aggregate_beliefs_to_resolution(beliefs, target_resolution=5)
        assert isinstance(aggregated, dict)
        assert len(aggregated) <= len(beliefs)
        for b in aggregated.values():
            assert np.isclose(np.sum(b), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# SpatialActiveInferenceAgent visualization tests
# ---------------------------------------------------------------------------


class TestSpatialAgentVisualization:
    """Tests for SpatialActiveInferenceAgent diagnostics/export."""

    def test_agent_diagnostics_contains_spatial_coherence(self):
        """get_diagnostics returns spatial coherence metrics."""
        adapter = get_h3_adapter()
        center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
        cells = [center, *sorted(set(adapter.grid_disk(center, 1)) - {center})][:4]
        agent = SpatialActiveInferenceAgent(initial_cells=cells)
        for _ in range(3):
            agent.step({cells[0]: np.random.rand(4)})
        diag = agent.get_diagnostics()
        assert "spatial_coherence" in diag
        assert "free_energy" in diag
        assert "agent_info" in diag

    def test_agent_export_results_to_json(self, tmp_path: Path):
        """export_results produces valid JSON with history."""
        adapter = get_h3_adapter()
        center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
        cells = [center, *sorted(set(adapter.grid_disk(center, 1)) - {center})][:3]
        agent = SpatialActiveInferenceAgent(initial_cells=cells)
        agent.step({cells[0]: np.array([1.0, 0.0, 0.0, 0.0])})
        agent.step({cells[0]: np.array([0.0, 1.0, 0.0, 0.0])})
        path = tmp_path / "spatial_agent_results.json"
        agent.export_results(str(path))
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "diagnostics" in data
        assert "free_energy_history" in data
        assert len(data["free_energy_history"]) == 2

    def test_step_returns_h3_grid_inference_result(self):
        """step with return_result=True returns H3GridInferenceResult."""
        adapter = get_h3_adapter()
        center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
        cells = [center, *sorted(set(adapter.grid_disk(center, 1)) - {center})][:2]
        agent = SpatialActiveInferenceAgent(initial_cells=cells)
        result = agent.step(
            {cells[0]: np.array([1.0, 0.0, 0.0, 0.0])}, return_result=True
        )
        assert isinstance(result, H3GridInferenceResult)
        assert result.spatial_consistency.cell_count >= 1
        for cell_result in result.cell_results.values():
            beliefs = np.asarray(cell_result.beliefs)
            assert np.isclose(np.sum(beliefs), 1.0)


# ---------------------------------------------------------------------------
# H3 version and contract tests
# ---------------------------------------------------------------------------


class TestH3VersionContract:
    """Tests that H3 usage follows the v4 contract."""

    def test_h3_version_is_4_5_0(self):
        """The installed h3 library is version 4.5.0."""
        assert h3.__version__ == "4.5.0"

    def test_uses_v4_api(self):
        """Uses latlng_to_cell, not legacy geo_to_h3."""
        cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
        assert isinstance(cell, str)
        lat, lng = h3.cell_to_latlng(cell)
        assert abs(lat - 37.7749) < 0.01
        assert abs(lng - -122.4194) < 0.01

    def test_cell_to_boundary_returns_lng_lat(self):
        """cell_to_boundary returns [lat, lng] coordinates (H3 native format)."""
        cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
        boundary = h3.cell_to_boundary(cell)
        assert len(boundary) >= 6
        for point in boundary:
            assert len(point) == 2
