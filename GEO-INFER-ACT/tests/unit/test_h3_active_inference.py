"""
Comprehensive Tests for H3 Active Inference Example Script

This module tests the H3 scenario helpers in ``geo_infer_act.runners.h3``
(re-exported by ``examples/h3_active_inference.py``) to ensure the robustness and correctness of the geospatial active inference simulation.
"""

import unittest
import numpy as np
import h3
from pathlib import Path
import tempfile
import json
import pytest

from geo_infer_act.runners.h3 import (
    generate_realistic_environmental_observations,
    run_h3_active_inference,
    setup_san_francisco_boundary,
)
from geo_infer_act.models.multi_agent import MultiAgentModel


@pytest.mark.slow
class TestH3ActiveInference(unittest.TestCase):
    """Comprehensive test suite for H3 Active Inference example script."""

    def setUp(self):
        """Set up test fixtures."""
        self.boundary = setup_san_francisco_boundary()
        self.h3_resolution = 8

    def test_generate_environmental_observations(self):
        """Test advanced environmental observations generation."""
        # Create a few H3 cells
        model = MultiAgentModel(n_agents=2)
        model.enable_h3_spatial(self.h3_resolution, self.boundary)

        # Test with and without spatial seed
        observations_no_seed = generate_realistic_environmental_observations(
            model.h3_cells, timestep=1.0
        )
        observations_seed = generate_realistic_environmental_observations(
            model.h3_cells, timestep=1.0, spatial_seed=42
        )

        # Verify observations
        self.assertEqual(len(observations_no_seed), len(model.h3_cells))
        self.assertEqual(len(observations_seed), len(model.h3_cells))

        # Check reproducibility with seed
        for cell in model.h3_cells:
            for var in observations_no_seed[cell].keys():
                # Verify all expected variables are present
                self.assertIn(var, observations_seed[cell])

                # Check value ranges
                if var not in ["temperature", "carbon_flux"]:
                    self.assertTrue(0 <= observations_seed[cell][var] <= 1)

        # Test seed reproducibility
        observations_seed_2 = generate_realistic_environmental_observations(
            model.h3_cells, timestep=1.0, spatial_seed=42
        )
        for cell in model.h3_cells:
            for var in observations_seed[cell].keys():
                self.assertAlmostEqual(
                    observations_seed[cell][var],
                    observations_seed_2[cell][var],
                    places=6,
                    msg=f"Seed reproducibility failed for {var} in cell {cell}",
                )

    @pytest.mark.slow
    def test_advanced_h3_active_inference(self):
        """Test comprehensive H3 active inference simulation."""
        # Create a temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run simulation with various parameters
            results = run_h3_active_inference(
                output_dir, h3_resolution=8, timesteps=10, n_agents=3, spatial_seed=42
            )

            # Verify comprehensive results structure
            self.assertIn("simulation_params", results)
            self.assertIn("history", results)
            self.assertIn("metrics", results)
            self.assertIn("environmental_observations", results)
            self.assertIn("agent_coordination_history", results)

            # Check simulation parameters
            params = results["simulation_params"]
            self.assertEqual(params["h3_resolution"], 8)
            self.assertGreater(params["n_cells"], 0)
            self.assertEqual(params["n_agents"], 3)
            self.assertEqual(params["timesteps"], 10)
            self.assertEqual(params["spatial_seed"], 42)

            # Check metrics
            metrics = results["metrics"]
            self.assertIn("free_energy_evolution", metrics)
            self.assertIn("final_free_energy", metrics)
            self.assertIn("free_energy_change", metrics)
            self.assertIn("total_processing_time", metrics)

            # Verify free energy evolution
            fe_evolution = metrics["free_energy_evolution"]
            self.assertEqual(len(fe_evolution), 10)  # 10 timesteps

            # Check history
            history = results["history"]
            self.assertEqual(len(history), 10)  # 10 timesteps

            # Check environmental observations
            env_obs_history = results["environmental_observations"]
            self.assertEqual(len(env_obs_history), 10)

            # Verify each timestep's data
            for timestep_data in history:
                self.assertIn("timestep", timestep_data)
                self.assertIn("cells", timestep_data)
                self.assertIn("global_metrics", timestep_data)

                # Check global metrics
                global_metrics = timestep_data["global_metrics"]
                self.assertIn("average_free_energy", global_metrics)
                self.assertIn("total_free_energy", global_metrics)
                self.assertIn("coordination_coherence", global_metrics)
                self.assertIn("processing_time", global_metrics)

    def test_spatial_variation_complexity(self):
        """Test the complexity of spatial variations in environmental observations."""
        # Create a few H3 cells
        model = MultiAgentModel(n_agents=2)
        model.enable_h3_spatial(self.h3_resolution, self.boundary)

        # Custom base patterns to test specific variations
        custom_patterns = {
            "temperature": {
                "base": 25.0,
                "amplitude": 3.0,
                "spatial_scale": 50,
                "temporal_period": 10,
            },
            "vegetation_density": {
                "base": 0.5,
                "amplitude": 0.3,
                "spatial_scale": 40,
                "temporal_period": 12,
                "coastal_gradient": True,
            },
        }

        # Generate observations for multiple timesteps
        observations_t1 = generate_realistic_environmental_observations(
            model.h3_cells, timestep=1.0, base_patterns=custom_patterns
        )
        observations_t2 = generate_realistic_environmental_observations(
            model.h3_cells, timestep=2.0, base_patterns=custom_patterns
        )

        # Analyze spatial and temporal variations
        for var in ["temperature", "vegetation_density"]:
            # Compute spatial variation
            var_values_t1 = [obs[var] for obs in observations_t1.values()]
            var_values_t2 = [obs[var] for obs in observations_t2.values()]

            # Check temporal variation
            self.assertNotEqual(
                np.mean(var_values_t1),
                np.mean(var_values_t2),
                f"No temporal variation detected for {var}",
            )

            # Check spatial variation
            if var == "vegetation_density":
                # For vegetation, check coastal gradient effect
                coastal_cells = [
                    cell
                    for cell, obs in observations_t1.items()
                    if abs(h3.cell_to_latlng(cell)[1] + 122.4) > 0.1
                ]
                non_coastal_cells = [
                    cell
                    for cell, obs in observations_t1.items()
                    if abs(h3.cell_to_latlng(cell)[1] + 122.4) <= 0.1
                ]

                if coastal_cells and non_coastal_cells:
                    coastal_values = [
                        observations_t1[cell][var] for cell in coastal_cells
                    ]
                    non_coastal_values = [
                        observations_t1[cell][var] for cell in non_coastal_cells
                    ]

                    # Coastal cells should have different vegetation density
                    self.assertNotAlmostEqual(
                        np.mean(coastal_values),
                        np.mean(non_coastal_values),
                        msg="Coastal gradient not reflected in vegetation density",
                    )

    @pytest.mark.slow
    def test_export_and_reproducibility(self):
        """Test simulation export and reproducibility."""
        # Create a temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run simulation with fixed seed
            results_1 = run_h3_active_inference(
                output_dir, h3_resolution=8, timesteps=10, n_agents=3, spatial_seed=42
            )

            # Run again with same seed
            results_2 = run_h3_active_inference(
                output_dir, h3_resolution=8, timesteps=10, n_agents=3, spatial_seed=42
            )

            # Compare key metrics for reproducibility
            metrics_1 = results_1["metrics"]
            metrics_2 = results_2["metrics"]

            # Check free energy evolution
            self.assertEqual(
                metrics_1["free_energy_evolution"],
                metrics_2["free_energy_evolution"],
                "Free energy evolution not reproducible with same seed",
            )

            # Check environmental observations
            env_obs_1 = results_1["environmental_observations"]
            env_obs_2 = results_2["environmental_observations"]

            # Compare each timestep's observations
            for t in range(len(env_obs_1)):
                for cell in env_obs_1[t].keys():
                    for var in env_obs_1[t][cell].keys():
                        self.assertAlmostEqual(
                            env_obs_1[t][cell][var],
                            env_obs_2[t][cell][var],
                            places=6,
                            msg=f"Observation not reproducible for {var} in cell {cell} at timestep {t}",
                        )

            # Optional: Export results to JSON for further analysis
            export_path = output_dir / "simulation_results.json"
            with open(export_path, "w") as f:
                json.dump(results_1, f, indent=2, default=str)

            self.assertTrue(export_path.exists(), "Results export failed")


if __name__ == "__main__":
    unittest.main()
