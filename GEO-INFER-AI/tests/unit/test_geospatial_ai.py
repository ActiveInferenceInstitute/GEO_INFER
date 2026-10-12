"""
Comprehensive tests for advanced geospatial active inference methods.

This module tests the sophisticated geospatial AI functionality including
environmental modeling, resource optimization, multi-scale analysis, and
predictive environmental dynamics.
"""

import logging
import unittest

import matplotlib
import numpy as np

from geo_infer_ai.models.predictive.geospatial_ai import (
    EnvironmentalActiveInferenceEngine,
    MultiScaleHierarchicalAnalyzer,
    EnvironmentalState,
    ResourceAllocation,
    SpatialPrediction,
    analyze_multi_scale_patterns,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class TestEnvironmentalActiveInferenceEngine(unittest.TestCase):
    """Test suite for EnvironmentalActiveInferenceEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.h3_resolution = 8
        self.engine = EnvironmentalActiveInferenceEngine(
            h3_resolution=self.h3_resolution,
            prediction_horizon=5,
            uncertainty_threshold=0.2,
        )

        # Create test boundary
        self.boundary = {
            "coordinates": [
                [
                    [
                        [-122.5, 37.7],
                        [-122.5, 37.8],
                        [-122.3, 37.8],
                        [-122.3, 37.7],
                        [-122.5, 37.7],
                    ]
                ]
            ]
        }

        logger.info(f"Test setup with boundary: {self.boundary}")

    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.h3_resolution, self.h3_resolution)
        self.assertEqual(len(self.engine.environmental_variables), 7)
        self.assertIsNotNone(self.engine.gp_models)
        self.assertEqual(len(self.engine.gp_models), 7)

        logger.info("Engine initialization test passed")

    def test_spatial_domain_initialization(self):
        """Test spatial domain initialization."""
        self.engine.initialize_spatial_domain(self.boundary)

        self.assertGreater(len(self.engine.environmental_states), 0)
        self.assertIsNotNone(self.engine.spatial_graph)

        # Check that environmental states are properly initialized
        for cell, env_state in self.engine.environmental_states.items():
            self.assertIsInstance(env_state, EnvironmentalState)
            self.assertEqual(env_state.location, cell)
            self.assertEqual(env_state.timestamp, 0.0)

        logger.info(
            f"Spatial domain initialized with {len(self.engine.environmental_states)} cells"
        )

    def test_environmental_observations(self):
        """Test environmental observation processing."""
        self.engine.initialize_spatial_domain(self.boundary)

        # Create test observations
        observations = {}
        for cell in list(self.engine.environmental_states.keys())[:2]:
            observations[cell] = {
                "temperature": 25.0,
                "humidity": 0.6,
                "vegetation_density": 0.7,
                "water_availability": 0.5,
                "soil_quality": 0.8,
                "biodiversity_index": 0.6,
                "carbon_flux": 0.1,
            }

        # Update observations
        self.engine.observe_environment(observations, 1.0)

        # Check that observations were processed
        self.assertEqual(len(self.engine.observation_history), 1)
        self.assertEqual(self.engine.observation_history[0]["timestamp"], 1.0)

        # Check environmental state updates
        for cell in observations:
            env_state = self.engine.environmental_states[cell]
            self.assertEqual(env_state.temperature, 25.0)
            self.assertEqual(env_state.humidity, 0.6)
            self.assertEqual(env_state.timestamp, 1.0)

        logger.info("Environmental observations test passed")

    def test_prediction_generation(self):
        """Test environmental prediction generation."""
        self.engine.initialize_spatial_domain(self.boundary)

        # Add multiple observation timesteps
        for t in range(8):  # Need enough for GP training
            observations = {}
            for cell in list(self.engine.environmental_states.keys())[:3]:
                observations[cell] = {
                    "temperature": 20.0 + t + np.random.normal(0, 0.5),
                    "humidity": 0.5 + 0.1 * np.sin(t) + np.random.normal(0, 0.02),
                    "vegetation_density": 0.6 + np.random.normal(0, 0.02),
                }

            self.engine.observe_environment(observations, float(t))

        # Generate predictions
        predictions = self.engine.predict_environmental_dynamics(forecast_timesteps=3)

        # Check predictions structure
        self.assertIsInstance(predictions, dict)

        for var_predictions in predictions.values():
            self.assertIsInstance(var_predictions, list)
            if var_predictions:  # May be empty if GP training failed
                for pred in var_predictions:
                    self.assertIsInstance(pred, SpatialPrediction)
                    self.assertIn(pred.location, self.engine.environmental_states)
                    self.assertIsInstance(pred.predicted_value, float)
                    self.assertIsInstance(pred.uncertainty, float)
                    self.assertGreater(pred.uncertainty, 0)

        logger.info(f"Generated predictions for {len(predictions)} variables")

    def test_resource_allocation_optimization(self):
        """Test resource allocation optimization."""
        self.engine.initialize_spatial_domain(self.boundary)

        # Add observations for resource optimization
        observations = {}
        for cell in self.engine.environmental_states:
            observations[cell] = {
                "biodiversity_index": np.random.uniform(0.2, 0.8),
                "vegetation_density": np.random.uniform(0.3, 0.9),
                "soil_quality": np.random.uniform(0.4, 0.9),
                "water_availability": np.random.uniform(0.2, 0.8),
                "carbon_flux": np.random.normal(0, 0.3),
            }

        self.engine.observe_environment(observations, 0.0)

        # Test resource allocation optimization
        resource_types = ["vegetation_restoration", "water_conservation"]
        allocations = self.engine.optimize_resource_allocation(
            resource_budget=100.0,
            resource_types=resource_types,
            optimization_objective="biodiversity",
        )

        # Check allocations
        self.assertIsInstance(allocations, list)

        total_allocation = sum(alloc.allocation_amount for alloc in allocations)
        self.assertLessEqual(total_allocation, 100.0)  # Within budget

        for alloc in allocations:
            self.assertIsInstance(alloc, ResourceAllocation)
            self.assertIn(alloc.location, self.engine.environmental_states)
            self.assertIn(alloc.resource_type, resource_types)
            self.assertGreater(alloc.allocation_amount, 0)
            self.assertGreater(alloc.priority_score, 0)

        logger.info(f"Generated {len(allocations)} resource allocations")

    def test_uncertainty_analysis(self):
        """Test environmental uncertainty analysis."""
        self.engine.initialize_spatial_domain(self.boundary)

        # Add uncertainty to environmental states
        for env_state in self.engine.environmental_states.values():
            env_state.uncertainty = {
                "temperature": np.random.uniform(0.1, 0.3),
                "humidity": np.random.uniform(0.05, 0.2),
                "vegetation_density": np.random.uniform(0.02, 0.15),
            }

        # Analyze uncertainty
        uncertainty_analysis = self.engine.analyze_environmental_uncertainty()

        # Check analysis structure
        self.assertIn("global_uncertainty", uncertainty_analysis)
        self.assertIn("high_uncertainty_regions", uncertainty_analysis)

        # Check global uncertainty metrics
        global_uncertainty = uncertainty_analysis["global_uncertainty"]
        for var in ["temperature", "humidity", "vegetation_density"]:
            if var in global_uncertainty:
                self.assertIn("mean", global_uncertainty[var])
                self.assertIn("std", global_uncertainty[var])
                self.assertGreater(global_uncertainty[var]["mean"], 0)

        logger.info("Uncertainty analysis test passed")

    def test_environmental_free_energy(self):
        """Test environmental free energy computation."""
        self.engine.initialize_spatial_domain(self.boundary)

        # Add observations
        observations = {}
        for cell in list(self.engine.environmental_states.keys())[:2]:
            observations[cell] = {"temperature": 22.0, "humidity": 0.65}

        self.engine.observe_environment(observations, 1.0)

        # Compute free energy
        free_energy = self.engine.compute_environmental_free_energy()

        # Check free energy structure
        self.assertIn("total_free_energy", free_energy)
        self.assertIn("components", free_energy)

        # Should be finite (not infinite) if we have observations
        if self.engine.observation_history:
            self.assertNotEqual(free_energy["total_free_energy"], np.inf)

        logger.info(f"Environmental free energy: {free_energy['total_free_energy']}")

    def test_environmental_summary(self):
        """Test environmental summary generation."""
        self.engine.initialize_spatial_domain(self.boundary)

        summary = self.engine.get_environmental_summary()

        # Check summary structure
        self.assertIn("spatial_domain", summary)
        self.assertIn("temporal_domain", summary)
        self.assertIn("environmental_variables", summary)
        self.assertIn("model_status", summary)

        # Check spatial domain info
        spatial_info = summary["spatial_domain"]
        self.assertEqual(spatial_info["n_cells"], len(self.engine.environmental_states))
        self.assertEqual(spatial_info["h3_resolution"], self.h3_resolution)
        self.assertGreater(spatial_info["coverage_area_km2"], 0)

        logger.info("Environmental summary test passed")


class TestMultiScaleHierarchicalAnalyzer(unittest.TestCase):
    """Test suite for MultiScaleHierarchicalAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiScaleHierarchicalAnalyzer(
            base_resolution=8, hierarchy_levels=3
        )

        self.boundary = {
            "coordinates": [
                [
                    [
                        [-122.5, 37.7],
                        [-122.5, 37.8],
                        [-122.3, 37.8],
                        [-122.3, 37.7],
                        [-122.5, 37.7],
                    ]
                ]
            ]
        }

        logger.info("MultiScaleHierarchicalAnalyzer test setup complete")

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.base_resolution, 8)
        self.assertEqual(self.analyzer.hierarchy_levels, 3)
        self.assertEqual(self.analyzer.scale_factor, 3)

        logger.info("Hierarchical analyzer initialization test passed")

    def test_hierarchy_initialization(self):
        """Test hierarchical structure initialization."""
        self.analyzer.initialize_hierarchy(self.boundary)

        # Check that hierarchical graphs are created
        self.assertGreater(len(self.analyzer.hierarchical_graphs), 0)
        self.assertGreater(len(self.analyzer.hierarchical_beliefs), 0)

        # Check that beliefs are properly initialized
        for beliefs in self.analyzer.hierarchical_beliefs.values():
            self.assertIsInstance(beliefs, dict)
            for belief in beliefs.values():
                self.assertIsInstance(belief, np.ndarray)
                self.assertEqual(len(belief), 4)  # 4-state categorical model
                self.assertAlmostEqual(np.sum(belief), 1.0, places=6)

        logger.info(
            f"Hierarchy initialized with {len(self.analyzer.hierarchical_graphs)} levels"
        )

    def test_belief_propagation(self):
        """Test hierarchical belief propagation."""
        self.analyzer.initialize_hierarchy(self.boundary)

        # Create test bottom-up evidence
        finest_level = list(self.analyzer.hierarchical_graphs.keys())[0]
        bottom_up_evidence = {finest_level: {}}

        # Add evidence for some cells
        for cell in list(self.analyzer.hierarchical_beliefs[finest_level].keys())[:2]:
            bottom_up_evidence[finest_level][cell] = np.array([0.1, 0.3, 0.4, 0.2])

        # Propagate beliefs
        updated_beliefs = self.analyzer.propagate_beliefs_hierarchically(
            bottom_up_evidence
        )

        # Check that beliefs are updated
        self.assertIsInstance(updated_beliefs, dict)
        self.assertEqual(len(updated_beliefs), len(self.analyzer.hierarchical_beliefs))

        # Check belief normalization
        for level_beliefs in updated_beliefs.values():
            for belief in level_beliefs.values():
                self.assertAlmostEqual(np.sum(belief), 1.0, places=6)
                self.assertTrue(np.all(belief >= 0))

        logger.info("Belief propagation test passed")

    def test_cross_scale_interactions(self):
        """Test cross-scale interaction analysis."""
        self.analyzer.initialize_hierarchy(self.boundary)

        # Analyze cross-scale interactions
        interactions = self.analyzer.analyze_cross_scale_interactions()

        # Check analysis structure
        self.assertIn("scale_coherence", interactions)
        self.assertIn("information_flow", interactions)

        # Check information flow analysis
        info_flow = interactions["information_flow"]
        for level_name in self.analyzer.hierarchical_graphs.keys():
            self.assertIn(level_name, info_flow)
            level_info = info_flow[level_name]
            self.assertIn("entropy", level_info)
            self.assertIn("n_cells", level_info)
            self.assertGreater(level_info["entropy"], 0)
            self.assertGreater(level_info["n_cells"], 0)

        logger.info("Cross-scale interaction analysis test passed")

    def test_emergent_pattern_detection(self):
        """Test emergent pattern detection."""
        self.analyzer.initialize_hierarchy(self.boundary)

        # Create diverse belief patterns
        for beliefs in self.analyzer.hierarchical_beliefs.values():
            cells = list(beliefs.keys())
            if len(cells) >= 4:
                # Create a cluster pattern
                beliefs[cells[0]] = np.array([0.8, 0.1, 0.05, 0.05])
                beliefs[cells[1]] = np.array([0.7, 0.15, 0.1, 0.05])
                if len(cells) > 2:
                    beliefs[cells[2]] = np.array([0.1, 0.1, 0.1, 0.7])
                if len(cells) > 3:
                    beliefs[cells[3]] = np.array([0.05, 0.05, 0.1, 0.8])

        # Detect patterns
        patterns = self.analyzer.detect_emergent_patterns()

        # Check patterns
        self.assertIsInstance(patterns, list)

        for pattern in patterns:
            self.assertIn("type", pattern)
            self.assertIn("level", pattern)
            self.assertIn("cells", pattern)
            self.assertIn("size", pattern)
            self.assertGreater(pattern["size"], 1)

            if "spatial_extent" in pattern:
                extent = pattern["spatial_extent"]
                self.assertIn("area_km2", extent)
                self.assertGreaterEqual(extent["area_km2"], 0)

        logger.info(f"Detected {len(patterns)} emergent patterns")

    def test_multi_scale_pattern_analysis(self):
        """Test the multi-scale pattern analysis function."""
        self.analyzer.initialize_hierarchy(self.boundary)

        # Test the standalone analysis function
        analysis = analyze_multi_scale_patterns(
            self.analyzer.hierarchical_graphs, self.analyzer.hierarchical_beliefs
        )

        # Check analysis structure
        self.assertIn("cross_scale_interactions", analysis)
        self.assertIn("emergent_patterns", analysis)
        self.assertIn("scale_statistics", analysis)

        # Check scale statistics
        scale_stats = analysis["scale_statistics"]
        for level_name in self.analyzer.hierarchical_graphs.keys():
            self.assertIn(level_name, scale_stats)
            level_stats = scale_stats[level_name]
            self.assertIn("n_cells", level_stats)
            self.assertIn("mean_entropy", level_stats)
            self.assertGreater(level_stats["n_cells"], 0)
            self.assertGreater(level_stats["mean_entropy"], 0)

        logger.info("Multi-scale pattern analysis test passed")

    def test_pattern_diversity_matches_pairwise_bruteforce(self):
        """Vectorized pattern diversity must equal the pairwise-distance mean."""
        rng = np.random.default_rng(5)
        beliefs = {f"cell_{index}": rng.dirichlet(np.ones(4)) for index in range(30)}

        analysis = analyze_multi_scale_patterns(
            {"res_8": {"cells": list(beliefs.keys())}}, {"res_8": beliefs}
        )

        belief_list = list(beliefs.values())
        bruteforce = np.mean(
            [
                np.linalg.norm(belief_list[i] - belief_list[j])
                for i in range(len(belief_list))
                for j in range(i + 1, len(belief_list))
            ]
        )

        diversity = analysis["scale_statistics"]["res_8"]["pattern_diversity"]
        # 1-ULP slack: the vectorized reduce differs from BLAS ddot in the
        # last bit for >=3-element norms.
        self.assertAlmostEqual(float(diversity), float(bruteforce), delta=1e-12)

        single = analyze_multi_scale_patterns(
            {"res_8": {"cells": ["c0"]}}, {"res_8": {"c0": beliefs["cell_0"]}}
        )
        self.assertEqual(single["scale_statistics"]["res_8"]["pattern_diversity"], 0.0)


class TestDataStructures(unittest.TestCase):
    """Test suite for data structures."""

    def test_environmental_state(self):
        """Test EnvironmentalState data structure."""
        env_state = EnvironmentalState(
            location="test_cell", temperature=25.0, humidity=0.6, vegetation_density=0.7
        )

        self.assertEqual(env_state.location, "test_cell")
        self.assertEqual(env_state.temperature, 25.0)
        self.assertEqual(env_state.humidity, 0.6)
        self.assertEqual(env_state.vegetation_density, 0.7)

        # Test defaults
        self.assertEqual(env_state.biodiversity_index, 0.5)
        self.assertEqual(env_state.carbon_flux, 0.0)

        logger.info("EnvironmentalState test passed")

    def test_resource_allocation(self):
        """Test ResourceAllocation data structure."""
        allocation = ResourceAllocation(
            location="test_cell",
            resource_type="vegetation_restoration",
            allocation_amount=50.0,
            priority_score=0.8,
            expected_benefit=0.7,
            uncertainty=0.2,
        )

        self.assertEqual(allocation.location, "test_cell")
        self.assertEqual(allocation.resource_type, "vegetation_restoration")
        self.assertEqual(allocation.allocation_amount, 50.0)
        self.assertEqual(allocation.priority_score, 0.8)

        logger.info("ResourceAllocation test passed")

    def test_spatial_prediction(self):
        """Test SpatialPrediction data structure."""
        prediction = SpatialPrediction(
            location="test_cell",
            predicted_value=0.65,
            uncertainty=0.1,
            confidence_interval=(0.55, 0.75),
            prediction_horizon=3,
        )

        self.assertEqual(prediction.location, "test_cell")
        self.assertEqual(prediction.predicted_value, 0.65)
        self.assertEqual(prediction.uncertainty, 0.1)
        self.assertEqual(prediction.confidence_interval, (0.55, 0.75))
        self.assertEqual(prediction.prediction_horizon, 3)

        logger.info("SpatialPrediction test passed")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.boundary = {
            "coordinates": [
                [
                    [
                        [-122.5, 37.7],
                        [-122.5, 37.8],
                        [-122.3, 37.8],
                        [-122.3, 37.7],
                        [-122.5, 37.7],
                    ]
                ]
            ]
        }

    def test_complete_environmental_workflow(self):
        """Test complete environmental modeling workflow."""
        logger.info("Starting complete environmental workflow test")

        # Initialize engine
        engine = EnvironmentalActiveInferenceEngine(h3_resolution=8)
        engine.initialize_spatial_domain(self.boundary)

        # Simulate environmental observations over time
        for t in range(10):
            observations = {}
            for cell in list(engine.environmental_states.keys())[:3]:
                observations[cell] = {
                    "temperature": 20.0 + 3 * np.sin(t / 3) + np.random.normal(0, 0.5),
                    "humidity": 0.6 + 0.1 * np.cos(t / 2) + np.random.normal(0, 0.02),
                    "vegetation_density": 0.5
                    + 0.2 * np.sin(t / 4)
                    + np.random.normal(0, 0.02),
                    "biodiversity_index": 0.6 + np.random.normal(0, 0.03),
                    "carbon_flux": np.random.normal(0, 0.2),
                }

            engine.observe_environment(observations, float(t))

        # Generate predictions
        predictions = engine.predict_environmental_dynamics(forecast_timesteps=3)

        # Optimize resource allocation
        allocations = engine.optimize_resource_allocation(
            resource_budget=200.0,
            resource_types=["vegetation_restoration", "carbon_sequestration"],
            optimization_objective="biodiversity",
        )

        # Analyze uncertainty
        uncertainty_analysis = engine.analyze_environmental_uncertainty()

        # Compute free energy
        free_energy = engine.compute_environmental_free_energy()

        # Verify workflow completed successfully
        self.assertGreater(len(engine.observation_history), 0)
        self.assertIsInstance(predictions, dict)
        self.assertIsInstance(allocations, list)
        self.assertIsInstance(uncertainty_analysis, dict)
        self.assertIsInstance(free_energy, dict)

        logger.info("Complete environmental workflow test passed")

    def test_hierarchical_environmental_integration(self):
        """Test integration between environmental engine and hierarchical analyzer."""
        logger.info("Starting hierarchical environmental integration test")

        # Initialize both components
        engine = EnvironmentalActiveInferenceEngine(h3_resolution=8)
        engine.initialize_spatial_domain(self.boundary)

        analyzer = MultiScaleHierarchicalAnalyzer(base_resolution=8, hierarchy_levels=2)
        analyzer.initialize_hierarchy(self.boundary)

        # Add environmental observations
        observations = {}
        for cell in engine.environmental_states:
            observations[cell] = {
                "biodiversity_index": np.random.uniform(0.3, 0.8),
                "vegetation_density": np.random.uniform(0.4, 0.9),
                "soil_quality": np.random.uniform(0.5, 0.9),
            }

        engine.observe_environment(observations, 0.0)

        # Convert environmental states to hierarchical beliefs
        finest_level = list(analyzer.hierarchical_graphs.keys())[0]
        bottom_up_evidence = {finest_level: {}}

        for cell in engine.environmental_states:
            env_state = engine.environmental_states[cell]
            quality_score = (
                env_state.biodiversity_index
                + env_state.vegetation_density
                + env_state.soil_quality
            ) / 3.0

            # Convert to belief distribution
            if quality_score < 0.5:
                belief = np.array([0.6, 0.3, 0.08, 0.02])
            else:
                belief = np.array([0.1, 0.2, 0.4, 0.3])

            bottom_up_evidence[finest_level][cell] = belief

        # Propagate beliefs hierarchically
        hierarchical_beliefs = analyzer.propagate_beliefs_hierarchically(
            bottom_up_evidence
        )

        # Analyze multi-scale patterns
        analysis = analyze_multi_scale_patterns(
            analyzer.hierarchical_graphs, hierarchical_beliefs
        )

        # Verify integration worked
        self.assertIsInstance(hierarchical_beliefs, dict)
        self.assertIsInstance(analysis, dict)
        self.assertIn("cross_scale_interactions", analysis)
        self.assertIn("emergent_patterns", analysis)

        logger.info("Hierarchical environmental integration test passed")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_empty_boundary_handling(self):
        """Test handling of empty or invalid boundaries."""
        engine = EnvironmentalActiveInferenceEngine()

        # Test with empty boundary
        empty_boundary = {"coordinates": [[[]]]}

        try:
            engine.initialize_spatial_domain(empty_boundary)
            # Should handle gracefully
            self.assertEqual(len(engine.environmental_states), 0)
        except Exception as e:
            # Should not crash completely
            self.assertIsInstance(e, (ValueError, IndexError, KeyError))

        logger.info("Empty boundary handling test passed")

    def test_invalid_observations(self):
        """Test handling of invalid observation data."""
        engine = EnvironmentalActiveInferenceEngine()
        boundary = {
            "coordinates": [
                [
                    [
                        [-122.5, 37.7],
                        [-122.5, 37.8],
                        [-122.3, 37.8],
                        [-122.3, 37.7],
                        [-122.5, 37.7],
                    ]
                ]
            ]
        }
        engine.initialize_spatial_domain(boundary)

        # Test with invalid cell ID
        invalid_observations = {"invalid_cell_id": {"temperature": 25.0}}

        # Should handle gracefully
        engine.observe_environment(invalid_observations, 1.0)
        self.assertEqual(len(engine.observation_history), 1)

        logger.info("Invalid observations handling test passed")

    def test_insufficient_data_for_prediction(self):
        """Test prediction generation with insufficient data."""
        engine = EnvironmentalActiveInferenceEngine()
        boundary = {
            "coordinates": [
                [
                    [
                        [-122.5, 37.7],
                        [-122.5, 37.8],
                        [-122.3, 37.8],
                        [-122.3, 37.7],
                        [-122.5, 37.7],
                    ]
                ]
            ]
        }
        engine.initialize_spatial_domain(boundary)

        # Add minimal observations
        observations = {}
        for cell in list(engine.environmental_states.keys())[:1]:
            observations[cell] = {"temperature": 20.0}

        engine.observe_environment(observations, 0.0)

        # Try to generate predictions (should handle gracefully)
        predictions = engine.predict_environmental_dynamics()

        # Should return empty or minimal predictions without crashing
        self.assertIsInstance(predictions, dict)

        logger.info("Insufficient data prediction test passed")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEnvironmentalActiveInferenceEngine))
    test_suite.addTest(unittest.makeSuite(TestMultiScaleHierarchicalAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestDataStructures))
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    test_suite.addTest(unittest.makeSuite(TestErrorHandling))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print("GEOSPATIAL AI TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
