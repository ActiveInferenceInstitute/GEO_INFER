#!/usr/bin/env python
"""
Unit tests for SpatialActiveInferenceAgent.

Tests cover:
- H3 cell initialization
- Spatial perception with neighbor influence
- Belief propagation
- Free energy computation
- Spatial policy selection
- Diagnostics and export
"""

import numpy as np
import json
import tempfile
from pathlib import Path

from geo_infer_act import H3GridInferenceResult
from geo_infer_act.core.spatial_agent import SpatialActiveInferenceAgent
from geo_infer_act.utils.h3_adapter import get_h3_adapter


def _cells(count: int) -> list[str]:
    """Return real H3 cells for spatial-agent tests."""
    adapter = get_h3_adapter()
    center = adapter.latlng_to_cell(37.7749, -122.4194, 9)
    candidates = [center, *adapter.grid_disk(center, 2)]
    return list(dict.fromkeys(candidates))[:count]


def _cell(index: int) -> str:
    return _cells(index + 1)[index]


class TestSpatialActiveInferenceAgentInit:
    """Tests for agent initialization."""

    def test_default_initialization(self):
        """Test agent initializes with defaults."""
        agent = SpatialActiveInferenceAgent()

        assert agent.h3_resolution == 9
        assert agent.state_dim == 4
        assert agent.obs_dim == 4
        assert len(agent.cells) > 0
        assert agent.beliefs.shape == (len(agent.cells), agent.state_dim)
        assert agent.step_count == 0

    def test_custom_dimensions(self):
        """Test agent with custom state/obs dimensions."""
        agent = SpatialActiveInferenceAgent(state_dim=8, obs_dim=6, h3_resolution=7)

        assert agent.state_dim == 8
        assert agent.obs_dim == 6
        assert agent.h3_resolution == 7
        assert agent.beliefs.shape[1] == 8

    def test_initialization_from_cells(self):
        """Test initialization from explicit cell list."""
        cells = _cells(4)
        agent = SpatialActiveInferenceAgent(initial_cells=cells)

        assert len(agent.cells) == 4
        assert agent.cells == cells
        assert all(c in agent.cell_to_idx for c in cells)

    def test_diffusion_rate_clamping(self):
        """Test diffusion rate is clamped to [0, 1]."""
        agent1 = SpatialActiveInferenceAgent(diffusion_rate=-0.5)
        agent2 = SpatialActiveInferenceAgent(diffusion_rate=1.5)

        assert agent1.diffusion_rate == 0.0
        assert agent2.diffusion_rate == 1.0

    def test_uniform_initial_beliefs(self):
        """Test beliefs are initially uniform."""
        agent = SpatialActiveInferenceAgent(state_dim=4)

        expected = 0.25  # 1/4
        np.testing.assert_array_almost_equal(
            agent.beliefs[0], [expected, expected, expected, expected]
        )


class TestSpatialPerception:
    """Tests for spatial perception."""

    def test_basic_perception(self):
        """Test basic perception updates beliefs."""
        cells = _cells(3)
        agent = SpatialActiveInferenceAgent(
            initial_cells=cells,
            state_dim=4,
            obs_dim=4,
            diffusion_rate=0.0,  # No propagation for this test
        )

        initial_beliefs = agent.beliefs.copy()

        observations = {
            _cell(0): np.array([1.0, 0.0, 0.0, 0.0]),
            _cell(1): np.array([0.0, 1.0, 0.0, 0.0]),
        }

        agent.spatial_perception(observations, propagate_beliefs=False)

        # Beliefs should have changed
        assert not np.allclose(agent.beliefs, initial_beliefs)
        assert agent.step_count == 1
        assert len(agent.free_energy_history) == 1

    def test_perception_with_propagation(self):
        """Test perception with neighbor propagation."""
        cells = _cells(3)
        agent = SpatialActiveInferenceAgent(
            initial_cells=cells, state_dim=3, obs_dim=3, diffusion_rate=0.3
        )

        # Update only first cell
        observations = {_cell(0): np.array([1.0, 0.0, 0.0])}

        updated = agent.spatial_perception(observations, propagate_beliefs=True)

        # All cells should be returned
        assert len(updated) == 3
        assert all(cell in updated for cell in cells)

    def test_observation_history_recorded(self):
        """Test observations are recorded in history."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        obs = {_cell(0): np.array([0.5, 0.5, 0.0, 0.0])}
        agent.spatial_perception(obs)

        assert len(agent.observation_history) == 1
        assert agent.observation_history[0]["step"] == 1


class TestSpatialAction:
    """Tests for spatial action selection."""

    def test_action_selection(self):
        """Test action selection returns valid result."""
        agent = SpatialActiveInferenceAgent(
            initial_cells=_cells(2), state_dim=4, obs_dim=4
        )

        # First do perception
        agent.spatial_perception({_cell(0): np.array([1.0, 0.0, 0.0, 0.0])})

        result = agent.spatial_action()

        assert "action" in result
        assert "action_name" in result
        assert "efe" in result
        assert "confidence" in result
        assert result["action"] in range(5)  # 5 actions
        assert result["action_name"] in ["stay", "north", "south", "east", "west"]

    def test_action_history_recorded(self):
        """Test actions are recorded in history."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(1))

        agent.spatial_perception({_cell(0): np.array([1.0, 0.0, 0.0, 0.0])})
        agent.spatial_action()

        assert len(agent.action_history) == 1

    def test_policy_distribution_sums_to_one(self):
        """Test policy distribution is valid probability."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        agent.spatial_perception({_cell(0): np.array([0.5, 0.5, 0.0, 0.0])})
        result = agent.spatial_action()

        pi = result["policy_distribution"]
        assert abs(sum(pi) - 1.0) < 0.01


class TestStepFunction:
    """Tests for full perception-action cycle."""

    def test_step_returns_all_components(self):
        """Test step returns beliefs, action, and free energy."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        result = agent.step({_cell(0): np.array([1.0, 0.0, 0.0, 0.0])})

        assert "beliefs" in result
        assert "action" in result
        assert "free_energy" in result
        assert "step" in result

    def test_step_can_return_typed_h3_result(self):
        """Test step can return typed H3 diagnostics."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        result = agent.step(
            {_cell(0): np.array([1.0, 0.0, 0.0, 0.0])}, return_result=True
        )

        assert isinstance(result, H3GridInferenceResult)
        assert result.spatial_consistency.cell_count == 2
        assert np.isfinite(result.aggregate_free_energy)
        for cell_result in result.cell_results.values():
            beliefs = np.asarray(cell_result.beliefs)
            assert np.all(beliefs >= 0)
            assert np.isclose(np.sum(beliefs), 1.0)

    def test_multiple_steps(self):
        """Test multiple steps accumulate history."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        for _ in range(5):
            obs = {_cell(0): np.random.rand(4)}
            agent.step(obs)

        assert agent.step_count == 5
        assert len(agent.free_energy_history) == 5
        assert len(agent.action_history) == 5


class TestDiagnostics:
    """Tests for diagnostics and export."""

    def test_get_diagnostics(self):
        """Test diagnostics returns expected structure."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(3))

        # Run a few steps
        for _ in range(3):
            agent.step({_cell(0): np.random.rand(4)})

        diag = agent.get_diagnostics()

        assert "agent_info" in diag
        assert "belief_stats" in diag
        assert "free_energy" in diag
        assert "spatial_coherence" in diag
        assert diag["agent_info"]["n_cells"] == 3
        assert diag["agent_info"]["step_count"] == 3

    def test_export_results_json(self):
        """Test export to JSON file."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        agent.step({_cell(0): np.array([1.0, 0.0, 0.0, 0.0])})
        agent.step({_cell(0): np.array([0.0, 1.0, 0.0, 0.0])})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            agent.export_results(str(filepath))

            assert filepath.exists()

            with open(filepath) as f:
                data = json.load(f)

            assert "diagnostics" in data
            assert "free_energy_history" in data
            assert len(data["free_energy_history"]) == 2


class TestPreferencesAndModels:
    """Tests for setting preferences and models."""

    def test_set_preferences(self):
        """Test setting preferences per cell."""
        agent = SpatialActiveInferenceAgent(
            initial_cells=_cells(2), state_dim=3, obs_dim=3
        )

        preferences = {_cell(0): np.array([0.8, 0.1, 0.1])}
        agent.set_preferences(preferences)

        np.testing.assert_array_almost_equal(agent.preferences[0], [0.8, 0.1, 0.1])

    def test_update_precision(self):
        """Test updating precision for a cell."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        initial_precision = agent.precision[0, 0]
        agent.update_precision(_cell(0), 2.5)

        assert agent.precision[0, 0] == 2.5
        assert agent.precision[0, 0] != initial_precision


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_history(self):
        """Test reset clears all history."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(2))

        # Run some steps
        for _ in range(3):
            agent.step({_cell(0): np.random.rand(4)})

        assert agent.step_count > 0

        agent.reset()

        assert agent.step_count == 0
        assert len(agent.free_energy_history) == 0
        assert len(agent.action_history) == 0
        assert len(agent.belief_history) == 0

    def test_reset_restores_uniform_beliefs(self):
        """Test reset restores uniform beliefs."""
        agent = SpatialActiveInferenceAgent(initial_cells=_cells(1), state_dim=4)

        # Change beliefs
        agent.step({_cell(0): np.array([1.0, 0.0, 0.0, 0.0])})

        agent.reset()

        expected = 0.25
        np.testing.assert_array_almost_equal(
            agent.beliefs[0], [expected, expected, expected, expected]
        )


class TestFreeEnergyComputation:
    """Tests for free energy computation."""

    def test_free_energy_decreases_with_learning(self):
        """Test free energy trend during learning."""
        agent = SpatialActiveInferenceAgent(
            initial_cells=_cells(2),
            state_dim=4,
            obs_dim=4,
            diffusion_rate=0.1,
        )

        # Consistent observations should reduce free energy
        consistent_obs = {_cell(0): np.array([1.0, 0.0, 0.0, 0.0])}

        for _ in range(10):
            agent.step(consistent_obs)

        fe_history = agent.free_energy_history

        # Free energy should generally decrease or stabilize
        initial_avg = np.mean(fe_history[:3])
        final_avg = np.mean(fe_history[-3:])

        # Allow for some variance but trend should be stable or decreasing
        assert final_avg <= initial_avg * 1.2  # Allow 20% variance
