#!/usr/bin/env python
"""
Unit tests for the ProceduralArt class in geo_infer_art.core.generation.procedural_art.
"""

import os
import numpy as np
import tempfile
import unittest
import pytest
from pathlib import Path
from PIL import Image

from geo_infer_art.core.generation.procedural_art import ProceduralArt


class TestProceduralArt(unittest.TestCase):
    """Test suite for the ProceduralArt class."""

    def setUp(self):
        """Set up test fixtures."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.test_dir = self._tmpdir.name

        # Define test coordinates
        self.test_lat = 40.7128  # New York
        self.test_lon = -74.0060

    def tearDown(self):
        """Clean up after tests."""
        self._tmpdir.cleanup()

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        proc_art = ProceduralArt()

        self.assertEqual(proc_art.algorithm, "noise_field")
        self.assertIsInstance(proc_art.params, dict)
        self.assertEqual(proc_art.resolution, (800, 800))
        self.assertIsNone(proc_art.image)

    def test_output_directory_is_temporary(self):
        """Test output directory does not use root-relative test_output."""
        repo_root = Path(__file__).resolve().parents[3]
        self.assertFalse(Path(self.test_dir).resolve().is_relative_to(repo_root))

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        algorithm = "l_system"
        params = {"iterations": 5, "angle": 25.0, "axiom": "F", "rules": {"F": "F+F-F"}}
        resolution = (600, 400)

        proc_art = ProceduralArt(
            algorithm=algorithm, params=params, resolution=resolution
        )

        self.assertEqual(proc_art.algorithm, algorithm)
        self.assertEqual(proc_art.params, params)
        self.assertEqual(proc_art.resolution, resolution)

    def test_from_geo_coordinates(self):
        """Test creating ProceduralArt from geographic coordinates."""
        # Test basic creation
        proc_art = ProceduralArt.from_geo_coordinates(
            lat=self.test_lat, lon=self.test_lon
        )

        self.assertEqual(proc_art.algorithm, "noise_field")  # Default algorithm
        self.assertIn("seed", proc_art.params)
        self.assertIsNotNone(proc_art.image)

        # Test with different algorithm and params
        additional_params = {"color_palette": "sunset", "iterations": 3}

        proc_art = ProceduralArt.from_geo_coordinates(
            lat=self.test_lat,
            lon=self.test_lon,
            algorithm="l_system",
            additional_params=additional_params,
        )

        self.assertEqual(proc_art.algorithm, "l_system")
        self.assertEqual(proc_art.params["color_palette"], "sunset")
        self.assertEqual(proc_art.params["iterations"], 3)
        self.assertIsNotNone(proc_art.image)

    @pytest.mark.slow
    def test_from_geo_features(self):
        """Test creating ProceduralArt from geographic features."""
        # Test basic creation
        proc_art = ProceduralArt.from_geo_features(
            feature_type="rivers", feature_count=3
        )

        self.assertEqual(proc_art.algorithm, "l_system")  # Default for features
        self.assertIn("feature_type", proc_art.params)
        self.assertEqual(proc_art.params["feature_type"], "rivers")
        self.assertEqual(proc_art.params["feature_count"], 3)
        self.assertIsNotNone(proc_art.image)

        # Test with different algorithm and params
        additional_params = {"color_palette": "ocean", "complexity": 0.7}

        proc_art = ProceduralArt.from_geo_features(
            feature_type="coastlines",
            feature_count=1,
            algorithm="cellular_automata",
            additional_params=additional_params,
        )

        self.assertEqual(proc_art.algorithm, "cellular_automata")
        self.assertEqual(proc_art.params["color_palette"], "ocean")
        self.assertEqual(proc_art.params["complexity"], 0.7)
        self.assertIsNotNone(proc_art.image)

    def test_generate(self):
        """Test the generate method."""
        # Initialize without generating
        proc_art = ProceduralArt(
            algorithm="noise_field",
            params={"color_palette": "viridis"},
            resolution=(400, 400),
        )

        # Image should not exist yet
        self.assertIsNone(proc_art.image)

        # Generate the image
        proc_art.generate()

        # Now the image should exist
        self.assertIsNotNone(proc_art.image)
        self.assertIsInstance(proc_art.image, Image.Image)
        self.assertEqual(proc_art.image.size, (400, 400))

    def test_save_and_show(self):
        """Test saving and showing procedural art."""
        # Create procedural art
        proc_art = ProceduralArt.from_geo_coordinates(
            lat=self.test_lat, lon=self.test_lon, algorithm="noise_field"
        )

        # Test save method
        output_path = os.path.join(self.test_dir, "procedural_art_output.png")
        saved_path = proc_art.save(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(saved_path, output_path)

        # Test show method - can only check that it doesn't raise an error
        try:
            proc_art.show()
        except Exception as e:
            self.fail(f"show() method raised an error: {str(e)}")

    # Small parameters so every algorithm runs fast at low resolution.
    FAST_PARAMS = {
        "boids": {"num_boids": 10, "iterations": 5},
        "particle_system": {"num_particles": 100, "iterations": 5},
        "diffusion_limited_aggregation": {"num_particles": 30, "iterations": 2},
        "turtle_graphics": {"iterations": 2},
        "sierpinski": {"iterations": 3},
        "dragon_curve": {"iterations": 5},
        "hilbert_curve": {"order": 2},
        "koch_snowflake": {"iterations": 2},
        "barnsley_fern": {"num_points": 500},
        "ifs_fractal": {"num_points": 500, "iterations": 2},
    }

    def test_different_algorithms(self):
        """Test generating art with every advertised algorithm (all of ALGORITHMS)."""
        for algorithm in ProceduralArt.ALGORITHMS:
            try:
                params = {"color_palette": "viridis", "seed": 42}
                params.update(self.FAST_PARAMS.get(algorithm, {}))

                proc_art = ProceduralArt(
                    algorithm=algorithm,
                    params=params,
                    resolution=(200, 200),  # Smaller for faster tests
                )

                proc_art.generate()
                self.assertIsNotNone(proc_art.image)
                self.assertEqual(proc_art.image.size, (200, 200))

                # Save the output for this algorithm
                output_path = os.path.join(self.test_dir, f"procedural_{algorithm}.png")
                proc_art.save(output_path)
                self.assertTrue(os.path.exists(output_path))

            except Exception as e:
                self.fail(f"Generation with algorithm {algorithm} failed: {str(e)}")

    def test_simplex_noise_is_distinct_algorithm(self):
        """GS-271: simplex_noise must differ from noise_field for the same seed."""
        noise_field = ProceduralArt(
            algorithm="noise_field", params={"seed": 42}, resolution=(200, 200)
        ).generate()
        simplex = ProceduralArt(
            algorithm="simplex_noise", params={"seed": 42}, resolution=(200, 200)
        ).generate()
        self.assertIsNotNone(simplex.image)
        self.assertFalse(
            np.array_equal(np.asarray(noise_field.image), np.asarray(simplex.image))
        )

    def test_simplex_noise_is_seed_deterministic(self):
        """GS-271: the same seed must reproduce identical simplex output."""
        first = ProceduralArt(
            algorithm="simplex_noise", params={"seed": 42}, resolution=(200, 200)
        ).generate()
        second = ProceduralArt(
            algorithm="simplex_noise", params={"seed": 42}, resolution=(200, 200)
        ).generate()
        self.assertTrue(
            np.array_equal(np.asarray(first.image), np.asarray(second.image))
        )

    def test_seeded_generation_uses_instance_generator(self):
        """Seeded voronoi output is reproducible and leaves numpy.random untouched."""
        np.random.seed(1234)
        expected_global = np.random.random()
        np.random.seed(1234)

        images = [
            np.asarray(
                ProceduralArt(
                    algorithm="voronoi", params={"seed": 5}, resolution=(64, 64)
                )
                .generate()
                .image
            )
            for _ in range(2)
        ]

        self.assertTrue(np.array_equal(images[0], images[1]))
        self.assertEqual(np.random.random(), expected_global)

    def test_dla_degenerate_structure(self):
        """DLA with no stuck particles must not divide by zero (NaN -> ValueError)."""
        proc_art = ProceduralArt(
            algorithm="diffusion_limited_aggregation",
            params={"seed": 7, "num_particles": 10, "iterations": 1, "stickiness": 0.0},
            resolution=(100, 100),
        )
        proc_art.generate()
        self.assertIsNotNone(proc_art.image)

    def test_koch_snowflake_default_palette(self):
        """Koch snowflake defaults must use a palette that actually exists."""
        proc_art = ProceduralArt(
            algorithm="koch_snowflake",
            params={"seed": 3, "iterations": 2},
            resolution=(200, 200),
        )
        proc_art.generate()
        self.assertIsNotNone(proc_art.image)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid algorithm
        with self.assertRaises(ValueError):
            ProceduralArt(algorithm="invalid_algorithm")

        # Test invalid coordinates
        with self.assertRaises(ValueError):
            ProceduralArt.from_geo_coordinates(
                lat=100.0,
                lon=self.test_lon,  # Invalid latitude
            )

        # Test invalid feature type
        with self.assertRaises(ValueError):
            ProceduralArt.from_geo_features(
                feature_type="invalid_feature_type", feature_count=1
            )

        # Test invalid resolution
        with self.assertRaises(ValueError):
            ProceduralArt(resolution=(-100, 300))  # Negative resolution


if __name__ == "__main__":
    unittest.main()
