"""
Unit tests for helper functions
"""

import numpy as np
import pytest

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.utils.helpers import (
    create_design_matrix,
    generate_coordinates,
    generate_synthetic_data,
    create_spatial_basis_functions,
    compute_power_analysis,
)


class TestDesignMatrixCreation:
    """Test design matrix creation functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_points = 50

        self.coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, n_points),
                np.random.uniform(-90, 90, n_points),
            ]
        )

        self.data = np.random.randn(n_points)

        # Create covariates
        self.covariates = {
            "elevation": np.random.normal(500, 100, n_points),
            "temperature": np.random.normal(20, 5, n_points),
            "urban": np.random.choice([0, 1], n_points),
        }

        self.spm_data = SPMData(
            data=self.data,
            coordinates=self.coordinates,
            covariates=self.covariates,
            crs="EPSG:4326",
        )

    def test_basic_design_matrix_creation(self):
        """Test basic design matrix creation."""
        design = create_design_matrix(self.spm_data, covariates=["elevation"])

        assert isinstance(design, DesignMatrix)
        assert design.matrix.shape[0] == len(self.data)
        assert design.matrix.shape[1] == 2  # intercept + elevation
        assert design.names == ["intercept", "elevation"]

    def test_design_matrix_with_multiple_covariates(self):
        """Test design matrix with multiple covariates."""
        design = create_design_matrix(
            self.spm_data, covariates=["elevation", "temperature"]
        )

        assert design.matrix.shape[1] == 3  # intercept + 2 covariates
        assert set(design.names) == {"intercept", "elevation", "temperature"}

    def test_design_matrix_without_intercept(self):
        """Test design matrix without intercept."""
        design = create_design_matrix(
            self.spm_data, covariates=["elevation"], intercept=False
        )

        assert design.matrix.shape[1] == 1  # no intercept
        assert design.names == ["elevation"]

    def test_design_matrix_with_factors(self):
        """Test design matrix with categorical factors."""
        # Create data with factor
        factor_data = self.spm_data.copy()
        factor_data.covariates["land_use"] = np.random.choice(
            ["urban", "rural", "forest"], 50
        )

        design = create_design_matrix(
            factor_data,
            covariates=["elevation"],
            factors={"land_use": ["urban", "rural", "forest"]},
        )

        # Should have intercept + elevation + 2 dummy variables (3 levels - 1)
        assert design.matrix.shape[1] == 4
        assert "intercept" in design.names
        assert "elevation" in design.names

    def test_invalid_covariate_name(self):
        """Test error handling for invalid covariate names."""
        with pytest.raises(ValueError, match="Covariate.*not found"):
            create_design_matrix(self.spm_data, covariates=["invalid_covariate"])

    def test_formula_based_design(self):
        """Test formula-based design matrix creation."""
        # This would require implementing formula parsing
        # For now, test basic functionality
        design = create_design_matrix(self.spm_data, covariates=["elevation"])

        assert design is not None


class TestCoordinateGeneration:
    """Test coordinate generation functionality."""

    def test_regular_grid_generation(self):
        """Test regular grid coordinate generation."""
        coordinates = generate_coordinates(
            "regular", n_points=25, bounds=(-10, 10, -5, 5)
        )

        assert coordinates.shape == (25, 2)
        assert coordinates[:, 0].min() >= -10
        assert coordinates[:, 0].max() <= 10
        assert coordinates[:, 1].min() >= -5
        assert coordinates[:, 1].max() <= 5

    def test_random_coordinate_generation(self):
        """Test random coordinate generation."""
        coordinates = generate_coordinates(
            "random", n_points=100, bounds=(-180, 180, -90, 90), random_seed=0
        )

        assert coordinates.shape == (100, 2)
        assert coordinates[:, 0].min() >= -180
        assert coordinates[:, 0].max() <= 180
        assert coordinates[:, 1].min() >= -90
        assert coordinates[:, 1].max() <= 90

    def test_clustered_coordinate_generation(self):
        """Test clustered coordinate generation."""
        coordinates = generate_coordinates(
            "clustered",
            n_points=60,
            bounds=(0, 100, 0, 100),
            n_clusters=3,
            random_seed=0,
        )

        assert coordinates.shape == (60, 2)

        # Should have some clustering (though hard to test precisely)
        # Check that coordinates are within bounds
        assert coordinates[:, 0].min() >= 0
        assert coordinates[:, 0].max() <= 100
        assert coordinates[:, 1].min() >= 0
        assert coordinates[:, 1].max() <= 100

    def test_invalid_coordinate_method(self):
        """Test error handling for invalid coordinate generation methods."""
        with pytest.raises(ValueError, match="Unknown grid type"):
            generate_coordinates("invalid_method", n_points=10)


class TestSyntheticDataGeneration:
    """Test synthetic data generation functionality."""

    def setup_method(self):
        """Set up coordinate data for testing."""
        self.coordinates = np.column_stack(
            [np.linspace(-10, 10, 20), np.linspace(-5, 5, 20)]
        )

    def test_basic_synthetic_data_generation(self):
        """Test basic synthetic data generation."""
        spm_data = generate_synthetic_data(self.coordinates, random_seed=0)

        assert isinstance(spm_data, SPMData)
        assert len(spm_data.data) == 20
        assert spm_data.coordinates.shape == (20, 2)
        assert spm_data.crs == "EPSG:4326"

    def test_synthetic_data_with_effects(self):
        """Test synthetic data with specific effects."""
        effects = {
            "trend": "east_west",
            "clusters": {"n_clusters": 2, "effect_size": 1.0},
        }

        spm_data = generate_synthetic_data(
            self.coordinates, effects=effects, noise_level=0.1, random_seed=0
        )

        assert spm_data.metadata["effects"] == effects
        assert spm_data.metadata["noise_level"] == 0.1
        assert "synthetic" in spm_data.metadata

    def test_temporal_synthetic_data(self):
        """Test generation of temporal synthetic data."""
        coordinates = self.coordinates[:5]  # Fewer spatial points

        spm_data = generate_synthetic_data(
            coordinates, temporal=True, n_timepoints=10, random_seed=0
        )

        assert spm_data.has_temporal
        assert spm_data.data.shape == (5, 10)  # (space, time) — data stored as data.T
        assert len(spm_data.time) == 10

    def test_synthetic_data_covariates(self):
        """Test that synthetic data includes appropriate covariates."""
        spm_data = generate_synthetic_data(self.coordinates, random_seed=0)

        # Should have elevation and temperature covariates
        assert "elevation" in spm_data.covariates
        assert "temperature" in spm_data.covariates

        assert len(spm_data.covariates["elevation"]) == 20
        assert len(spm_data.covariates["temperature"]) == 20


class TestSpatialBasisFunctions:
    """Test spatial basis function generation."""

    def setup_method(self):
        """Set up coordinate data."""
        np.random.seed(42)
        self.coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 30), np.random.uniform(-89, 89, 30)]
        )

    def test_gaussian_basis_functions(self):
        """Test Gaussian basis function generation."""
        basis = create_spatial_basis_functions(
            self.coordinates, n_basis=5, method="gaussian"
        )

        assert basis.shape == (30, 5)
        assert np.all(basis >= 0)  # Gaussian basis should be non-negative

        # Check normalization (each basis function should have reasonable scale)
        for i in range(basis.shape[1]):
            assert 0.1 < np.std(basis[:, i]) < 5.0

    def test_polynomial_basis_functions(self):
        """Test polynomial basis function generation."""
        basis = create_spatial_basis_functions(
            self.coordinates, n_basis=4, method="polynomial"
        )

        assert basis.shape == (30, 4)

        # First column should be constant (intercept)
        assert np.allclose(basis[:, 0], 1.0)

    def test_fourier_basis_functions(self):
        """Test Fourier basis function generation."""
        basis = create_spatial_basis_functions(
            self.coordinates, n_basis=6, method="fourier"
        )

        assert basis.shape == (30, 6)

        # Should include constant term
        assert np.allclose(basis[:, 0], 1.0, atol=0.1)

    def test_invalid_basis_method(self):
        """Test error handling for invalid basis methods."""
        with pytest.raises(ValueError, match="Unknown basis method"):
            create_spatial_basis_functions(self.coordinates, method="invalid")

    def test_basis_function_properties(self):
        """Test mathematical properties of basis functions."""
        basis = create_spatial_basis_functions(
            self.coordinates, n_basis=10, method="gaussian"
        )

        # Basis functions should be linearly independent (approximately)
        # Check condition number of basis matrix
        condition_number = np.linalg.cond(basis)
        assert condition_number < 1000  # Reasonable condition number


class TestPowerAnalysis:
    """Test statistical power analysis functionality."""

    def test_power_analysis_calculation(self):
        """Test power analysis computation."""
        results = compute_power_analysis(
            effect_size=0.5, n_points=100, alpha=0.05, n_simulations=100, random_seed=0
        )

        assert isinstance(results, dict)
        assert "power" in results
        assert "effect_size" in results
        assert "n_points" in results
        assert "alpha" in results

        assert 0 <= results["power"] <= 1
        assert results["effect_size"] == 0.5
        assert results["n_points"] == 100

    def test_power_vs_effect_size(self):
        """Test that power increases with effect size."""
        power_small = compute_power_analysis(
            effect_size=0.2, n_points=50, alpha=0.05, n_simulations=50, random_seed=0
        )
        power_large = compute_power_analysis(
            effect_size=0.8, n_points=50, alpha=0.05, n_simulations=50, random_seed=0
        )

        assert power_large["power"] > power_small["power"]

    def test_power_vs_sample_size(self):
        """Test that power increases with sample size."""
        power_small_n = compute_power_analysis(
            effect_size=0.5, n_points=30, alpha=0.05, n_simulations=50, random_seed=0
        )
        power_large_n = compute_power_analysis(
            effect_size=0.5, n_points=100, alpha=0.05, n_simulations=50, random_seed=0
        )

        assert power_large_n["power"] > power_small_n["power"]

    def test_power_vs_alpha(self):
        """Test that power decreases with stricter alpha."""
        power_liberal = compute_power_analysis(
            effect_size=0.5, n_points=50, alpha=0.10, n_simulations=50, random_seed=0
        )
        power_conservative = compute_power_analysis(
            effect_size=0.5, n_points=50, alpha=0.01, n_simulations=50, random_seed=0
        )

        assert power_liberal["power"] > power_conservative["power"]


class TestHelperFunctionEdgeCases:
    """Test edge cases in helper functions."""

    def test_empty_covariates(self):
        """Test design matrix creation with no covariates."""
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 10), np.random.uniform(-89, 89, 10)]
        )
        data = np.random.randn(10)
        spm_data = SPMData(data=data, coordinates=coordinates, crs="EPSG:4326")

        design = create_design_matrix(spm_data)

        # Should have only intercept
        assert design.matrix.shape[1] == 1
        assert design.names == ["intercept"]

    def test_single_coordinate_generation(self):
        """Test coordinate generation for single point."""
        coordinates = generate_coordinates("regular", n_points=1)

        assert coordinates.shape == (1, 2)

    def test_zero_effect_size_power(self):
        """Test power analysis with zero effect size."""
        results = compute_power_analysis(
            effect_size=0.0, n_points=50, alpha=0.05, n_simulations=20, random_seed=0
        )

        # Power should be approximately alpha (false positive rate)
        assert abs(results["power"] - 0.05) < 0.1

    def test_large_basis_function_request(self):
        """Test basis function generation with many functions."""
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 20), np.random.uniform(-89, 89, 20)]
        )

        # Request more basis functions than data points
        basis = create_spatial_basis_functions(
            coordinates, n_basis=15, method="gaussian"
        )

        assert basis.shape[0] == 20
        assert basis.shape[1] == 15

    def test_design_matrix_with_nan_covariates(self):
        """Test design matrix creation with NaN covariates."""
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 10), np.random.uniform(-89, 89, 10)]
        )
        data = np.random.randn(10)

        covariates = {
            "elevation": np.random.normal(500, 100, 10),
            "temperature": np.full(10, np.nan),  # All NaN
        }

        spm_data = SPMData(
            data=data, coordinates=coordinates, covariates=covariates, crs="EPSG:4326"
        )

        # Should still create design matrix (though with NaN values)
        design = create_design_matrix(spm_data, covariates=["elevation", "temperature"])

        assert design.matrix.shape[1] == 3  # intercept + 2 covariates
        assert np.any(np.isnan(design.matrix))  # Will contain NaN values

    def test_synthetic_data_extreme_parameters(self):
        """Test synthetic data generation with extreme parameters."""
        coordinates = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])

        # Very high noise: the sample std over 50 points tracks noise_level=10.
        spm_data = generate_synthetic_data(
            coordinates, noise_level=10.0, effects={"intercept": 100.0}, random_seed=0
        )

        assert len(spm_data.data) == 50
        assert spm_data.data.std() > 5.0  # Should have high variance due to noise
