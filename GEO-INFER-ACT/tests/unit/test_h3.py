"""
Comprehensive H3 geospatial active inference tests.

Tests the H3 hexagonal grid functionality including spatial modeling,
belief updating, multi-agent coordination, and visualization capabilities.
"""

import unittest
import copy
import numpy as np
import os
import tempfile
import logging
from pathlib import Path

# Set matplotlib backend before any imports that might use it
import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.types import H3BeliefUpdateResult, H3GridInferenceResult
from geo_infer_act.models.multi_agent import MultiAgentModel
from geo_infer_act.utils.visualization import (
    plot_h3_grid_static,
    create_h3_gif,
    create_interactive_h3_slider,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestH3Methods(unittest.TestCase):
    """Test suite for H3 geospatial active inference methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.boundary = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.435, 37.765],
                    [-122.435, 37.795],
                    [-122.405, 37.795],
                    [-122.405, 37.765],
                    [-122.435, 37.765],
                ]
            ],
        }
        self.simple_boundary = {
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
        logger.info("H3 test setup complete")

    def test_enable_h3_spatial(self):
        """Test enabling H3 spatial functionality in generative models."""
        model = GenerativeModel("categorical", {"state_dim": 2})

        # Test H3 spatial enablement
        model.enable_h3_spatial(8, self.boundary)

        # Verify spatial mode is enabled
        self.assertTrue(model.spatial_mode)
        self.assertGreater(len(model.h3_cells), 0)
        self.assertIsNotNone(model.spatial_graph)

        # Verify state dimension expansion for spatial modeling
        expected_dim = 2 * len(model.h3_cells)
        self.assertEqual(model.state_dim, expected_dim)

        logger.info(f"H3 spatial enabled with {len(model.h3_cells)} cells")

    def test_update_h3_beliefs(self):
        """Test H3 belief updating with spatial observations."""
        model = GenerativeModel("categorical", {"state_dim": 2})
        model.enable_h3_spatial(8, self.simple_boundary)

        # Create test observations for H3 cells
        h3_cells = model.h3_cells[:2]  # Test with first 2 cells
        observations = {}
        for i, cell in enumerate(h3_cells):
            obs = np.zeros(2)
            obs[i % 2] = 1.0  # Create distinct observations
            observations[cell] = obs

        # Update beliefs
        updated = model.update_h3_beliefs(observations)
        typed = model.update_h3_beliefs(observations, return_result=True)

        # Verify belief update structure
        self.assertIn("h3_beliefs", updated)
        self.assertIn("spatial_consistency", updated)
        self.assertIn("aggregate_free_energy", updated)
        self.assertIsInstance(typed, H3BeliefUpdateResult)
        self.assertTrue(np.isfinite(typed.aggregate_free_energy))
        self.assertGreaterEqual(typed.spatial_consistency.cell_count, 1)

        # Verify belief normalization
        for belief in updated["h3_beliefs"].values():
            self.assertAlmostEqual(np.sum(belief), 1.0, places=6)
            self.assertTrue(np.all(belief >= 0))

        # Verify spatial consistency measures
        consistency = updated["spatial_consistency"]
        self.assertIn("global_coherence", consistency)
        self.assertIn("neighbor_correlations", consistency)

        logger.info("H3 belief updating test passed")

    def test_expanded_h3_model_accepts_full_grid_observation(self):
        """The generic update path remains valid after H3 expands state space."""
        model = GenerativeModel("categorical", {"state_dim": 2})
        small_boundary = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.42, 37.77],
                    [-122.42, 37.771],
                    [-122.419, 37.771],
                    [-122.419, 37.77],
                    [-122.42, 37.77],
                ]
            ],
        }
        model.enable_h3_spatial(11, small_boundary)  # r=11: tiny box yields 4 cells
        observation = np.zeros(model.state_dim)
        observation[0] = 1.0

        beliefs = model.update_beliefs({"observations": observation})["states"]

        self.assertEqual(beliefs.shape, (model.state_dim,))
        self.assertAlmostEqual(float(np.sum(beliefs)), 1.0, places=6)

    def test_infer_over_h3_grid(self):
        """Test active inference over H3 grid."""
        aim = ActiveInferenceModel()
        gen = GenerativeModel("categorical", {"state_dim": 3})
        gen.enable_h3_spatial(8, self.simple_boundary)
        aim.set_generative_model(gen)
        original_model_beliefs = np.array(gen.beliefs["states"], copy=True)
        original_policy_rng_state = copy.deepcopy(
            aim.policy_selector.rng.bit_generator.state
        )

        # Create grid observations
        grid_observations = {}
        for cell in gen.h3_cells[:3]:  # Test with first 3 cells
            obs = np.random.rand(3)
            obs = obs / np.sum(obs)  # Normalize
            grid_observations[cell] = obs

        # Perform inference
        results = aim.infer_over_h3_grid(grid_observations)
        typed = aim.infer_over_h3_grid(grid_observations, return_result=True)

        np.testing.assert_allclose(gen.beliefs["states"], original_model_beliefs)
        self.assertEqual(
            aim.policy_selector.rng.bit_generator.state, original_policy_rng_state
        )

        # Verify results structure
        self.assertEqual(len(results), len(grid_observations))
        self.assertIsInstance(typed, H3GridInferenceResult)
        self.assertEqual(len(typed.cell_results), len(grid_observations))
        self.assertTrue(np.isfinite(typed.aggregate_free_energy))

        for result in results.values():
            self.assertIn("beliefs", result)
            self.assertIn("free_energy", result)
            self.assertIn("precision", result)

            # Verify belief properties — beliefs may be a dict {'states': array}
            beliefs_raw = result["beliefs"]
            beliefs = (
                beliefs_raw["states"] if isinstance(beliefs_raw, dict) else beliefs_raw
            )
            beliefs = np.asarray(beliefs, dtype=float).flatten()
            self.assertAlmostEqual(np.sum(beliefs), 1.0, places=6)
            self.assertTrue(np.all(beliefs >= 0))

            # Verify free energy is finite
            self.assertFalse(np.isinf(result["free_energy"]))
            self.assertFalse(np.isnan(result["free_energy"]))

        logger.info(f"H3 grid inference completed for {len(results)} cells")

    def test_multi_agent_h3(self):
        """Test multi-agent modeling on H3 grid."""
        model = MultiAgentModel(n_agents=10)

        # Enable H3 spatial modeling
        model.enable_h3_spatial(8, self.boundary)

        # Verify H3 integration
        self.assertTrue(hasattr(model, "h3_cells"))
        self.assertGreater(len(model.h3_cells), 0)
        self.assertEqual(model.n_locations, len(model.h3_cells))

        # Verify agent-location mapping
        self.assertTrue(hasattr(model, "agent_location_map"))

        # Test multi-agent coordination
        coordination_result = model.coordinate_agents()
        self.assertIn("coordination_matrix", coordination_result)
        self.assertIn("average_coordination", coordination_result)

        # Verify coordination matrix properties
        coord_matrix = coordination_result["coordination_matrix"]
        n_agents = len(model.agent_models)
        self.assertEqual(coord_matrix.shape, (n_agents, n_agents))

        # Test agent communication
        for agent_id in range(min(3, n_agents)):  # Test first 3 agents
            messages = model.get_agent_messages(agent_id)
            self.assertIsInstance(messages, dict)

        logger.info(
            f"Multi-agent H3 test completed with {n_agents} agents on {len(model.h3_cells)} cells"
        )

    def test_spatial_belief_diffusion(self):
        """Test belief diffusion across H3 spatial neighbors."""
        model = GenerativeModel("categorical", {"state_dim": 2})
        model.enable_h3_spatial(8, self.boundary)

        # Create initial belief concentrations
        initial_beliefs = {}
        for cell in model.h3_cells:
            initial_beliefs[cell] = np.array([0.5, 0.5])  # Uniform initial beliefs

        # Set high confidence belief in one cell
        if model.h3_cells:
            source_cell = model.h3_cells[0]
            initial_beliefs[source_cell] = np.array([0.9, 0.1])

            # Perform belief diffusion
            diffused_beliefs = model.diffuse_beliefs(
                initial_beliefs, diffusion_rate=0.1
            )

            # Check that beliefs are still normalized
            for beliefs in diffused_beliefs.values():
                self.assertAlmostEqual(np.sum(beliefs), 1.0, places=6)
                self.assertTrue(np.all(beliefs >= 0))

            # Check for spatial influence (if there are neighbors)
            if (
                hasattr(model.spatial_graph, "neighbors")
                and source_cell in model.spatial_graph.neighbors
            ):
                neighbors = model.spatial_graph.neighbors.get(source_cell, {}).get(
                    1, set()
                )

                if neighbors:  # Only check if there are actual neighbors
                    for neighbor in neighbors:
                        if neighbor in diffused_beliefs:
                            # Neighbor should have some influence from source
                            neighbor_belief = diffused_beliefs[neighbor]
                            # Should be slightly more confident than uniform
                            self.assertNotEqual(neighbor_belief[0], 0.5)
                else:
                    logger.info("No neighbors found for belief diffusion test")
            else:
                logger.info(
                    "Spatial graph neighbors not available or source cell not in graph"
                )

        logger.info("Spatial belief diffusion test completed")

    def test_h3_visualizations(self):
        """Test H3 visualization capabilities."""
        model = GenerativeModel("categorical", {"state_dim": 2})
        model.enable_h3_spatial(8, self.boundary)

        # Create test data
        test_data = {}
        for i, cell in enumerate(model.h3_cells[:3]):
            test_data[cell] = {
                "fe": 1.0 + 0.1 * i,
                "beliefs": np.array([0.6 - 0.1 * i, 0.4 + 0.1 * i]),
                "precision": 2.0 + 0.5 * i,
            }

        # Test static plotting
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test static H3 grid plot
            fig = plot_h3_grid_static(test_data, metric="fe")
            self.assertIsNotNone(fig)

            # Test GIF creation
            history = [test_data] * 5  # 5 timesteps
            gif_path = Path(tmpdir) / "test_h3.gif"
            create_h3_gif(history, str(gif_path))
            self.assertTrue(gif_path.exists())

            # Test interactive slider
            slider_fig = create_interactive_h3_slider(history)
            if slider_fig is not None:
                self.assertIsNotNone(slider_fig)

                # Save interactive HTML
                html_path = Path(tmpdir) / "test_interactive.html"
                slider_fig.write_html(str(html_path))
                self.assertTrue(html_path.exists())

        logger.info("H3 visualization tests completed")

    def test_h3_hierarchical_analysis(self):
        """Test hierarchical spatial analysis across H3 resolutions."""
        # Test multi-resolution analysis
        resolutions = [6, 7, 8]
        models = {}

        for res in resolutions:
            model = GenerativeModel("categorical", {"state_dim": 2})
            model.enable_h3_spatial(res, self.boundary)
            models[res] = model

        # Verify different resolutions produce different cell counts
        cell_counts = {res: len(model.h3_cells) for res, model in models.items()}

        # Higher resolution should generally have more cells
        self.assertGreaterEqual(cell_counts[8], cell_counts[6])

        # Test cross-resolution belief aggregation
        finest_model = models[8]
        if len(finest_model.h3_cells) > 0:
            # Create beliefs at finest resolution
            fine_beliefs = {}
            for cell in finest_model.h3_cells:
                fine_beliefs[cell] = np.array([0.6, 0.4])

            # Aggregate to coarser resolution
            coarse_beliefs = finest_model.aggregate_beliefs_to_resolution(
                fine_beliefs, target_resolution=6
            )

            # Verify aggregated beliefs are normalized
            for beliefs in coarse_beliefs.values():
                self.assertAlmostEqual(np.sum(beliefs), 1.0, places=6)

        logger.info(f"Hierarchical H3 analysis tested across resolutions {resolutions}")

    def test_environmental_active_inference(self):
        """Test environmental modeling integration with H3."""
        from geo_infer_ai.models.predictive.geospatial_ai import (
            EnvironmentalActiveInferenceEngine,
        )

        # Initialize environmental engine
        engine = EnvironmentalActiveInferenceEngine(h3_resolution=8)
        engine.initialize_spatial_domain(self.boundary)

        # Verify environmental initialization
        self.assertGreater(len(engine.environmental_states), 0)

        # Test environmental observations
        observations = {}
        for cell in list(engine.environmental_states.keys())[:2]:
            observations[cell] = {
                "temperature": 22.0 + np.random.normal(0, 1),
                "humidity": 0.6 + np.random.normal(0, 0.05),
                "vegetation_density": 0.7 + np.random.normal(0, 0.03),
            }

        engine.observe_environment(observations, 1.0)

        # Test environmental summary
        summary = engine.get_environmental_summary()
        self.assertIn("spatial_domain", summary)
        self.assertIn("temporal_domain", summary)
        self.assertEqual(summary["spatial_domain"]["h3_resolution"], 8)

        logger.info("Environmental active inference H3 integration test completed")


class TestH3Integration(unittest.TestCase):
    """Integration tests for H3 with other system components."""

    def test_h3_with_resource_optimization(self):
        """Test H3 integration with resource optimization."""
        from geo_infer_ai.models.predictive.geospatial_ai import (
            EnvironmentalActiveInferenceEngine,
        )

        boundary = {
            "coordinates": [
                [
                    [
                        [-122.45, 37.75],
                        [-122.45, 37.76],
                        [-122.44, 37.76],
                        [-122.44, 37.75],
                        [-122.45, 37.75],
                    ]
                ]
            ]
        }

        # Initialize environmental engine
        engine = EnvironmentalActiveInferenceEngine(h3_resolution=9)
        engine.initialize_spatial_domain(boundary)

        # Add environmental observations
        observations = {}
        for cell in engine.environmental_states:
            observations[cell] = {
                "biodiversity_index": np.random.uniform(0.3, 0.8),
                "vegetation_density": np.random.uniform(0.2, 0.9),
                "water_availability": np.random.uniform(0.1, 0.7),
            }

        engine.observe_environment(observations, 0.0)

        # Test resource allocation optimization
        allocations = engine.optimize_resource_allocation(
            resource_budget=100.0,
            resource_types=["vegetation_restoration", "water_conservation"],
            optimization_objective="biodiversity",
        )

        # Verify allocations
        self.assertIsInstance(allocations, list)
        if allocations:
            total_allocated = sum(alloc.allocation_amount for alloc in allocations)
            self.assertLessEqual(total_allocated, 100.0)

            for alloc in allocations:
                self.assertIn(alloc.location, engine.environmental_states)
                self.assertGreater(alloc.allocation_amount, 0)
                self.assertGreater(alloc.priority_score, 0)

        logger.info(
            f"H3 resource optimization test completed with {len(allocations)} allocations"
        )

    def test_h3_multi_scale_analysis(self):
        """Test multi-scale analysis with H3."""
        from geo_infer_ai.models.predictive.geospatial_ai import (
            MultiScaleHierarchicalAnalyzer,
        )

        boundary = {
            "coordinates": [
                [
                    [
                        [-122.435, 37.765],
                        [-122.435, 37.795],
                        [-122.405, 37.795],
                        [-122.405, 37.765],
                        [-122.435, 37.765],
                    ]
                ]
            ]
        }

        # Initialize hierarchical analyzer
        analyzer = MultiScaleHierarchicalAnalyzer(base_resolution=8, hierarchy_levels=2)
        analyzer.initialize_hierarchy(boundary)

        # Verify hierarchical structure
        self.assertGreater(len(analyzer.hierarchical_graphs), 0)
        self.assertGreater(len(analyzer.hierarchical_beliefs), 0)

        # Test belief propagation
        finest_level = list(analyzer.hierarchical_graphs.keys())[0]
        bottom_up_evidence = {finest_level: {}}

        # Add evidence for some cells
        for cell in list(analyzer.hierarchical_beliefs[finest_level].keys())[:2]:
            bottom_up_evidence[finest_level][cell] = np.array([0.7, 0.2, 0.08, 0.02])

        # Propagate beliefs
        updated_beliefs = analyzer.propagate_beliefs_hierarchically(bottom_up_evidence)

        # Verify belief propagation worked
        self.assertIsInstance(updated_beliefs, dict)
        self.assertEqual(len(updated_beliefs), len(analyzer.hierarchical_beliefs))

        logger.info("H3 multi-scale analysis test completed")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestH3Methods))
    test_suite.addTest(unittest.makeSuite(TestH3Integration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print("H3 GEOSPATIAL TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )
