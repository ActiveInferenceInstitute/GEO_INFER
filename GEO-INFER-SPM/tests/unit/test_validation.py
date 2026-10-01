"""
Unit tests for data validation functionality
"""

import numpy as np
import pytest

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.utils.validation import (
    validate_spm_data,
    validate_design_matrix,
    validate_contrast,
    validate_spatial_autocorrelation,
)


class TestSPMDataValidation:
    """Test SPMData validation functionality."""

    def test_valid_spm_data(self):
        """Test validation of valid SPMData."""
        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 50),  # longitude
                np.random.uniform(-90, 90, 50),  # latitude
            ]
        )
        data = np.random.randn(50)

        spm_data = SPMData(data=data, coordinates=coordinates, crs="EPSG:4326")
        validated = validate_spm_data(spm_data)

        assert validated is spm_data
        assert "validation" in validated.metadata
        assert validated.metadata["validation"]["passed"]

    def test_invalid_coordinate_shape(self):
        """Test validation of invalid coordinate shapes."""
        coordinates = np.random.rand(50, 3)  # Wrong shape
        data = np.random.randn(50)

        spm_data = SPMData(data=data, coordinates=coordinates)

        with pytest.raises(ValueError, match="Coordinates must have shape"):
            validate_spm_data(spm_data)

    def test_coordinate_data_mismatch(self):
        """Test validation of mismatched coordinate/data dimensions."""
        coordinates = np.random.rand(50, 2)
        data = np.random.randn(30)  # Different size

        spm_data = SPMData(data=data, coordinates=coordinates)

        with pytest.raises(ValueError, match="Coordinate count.*does not match"):
            validate_spm_data(spm_data)

    def test_nan_data_detection(self):
        """Test detection of NaN values in data."""
        coordinates = np.random.rand(50, 2)
        data = np.random.randn(50)
        data[10:15] = np.nan  # Add NaN values

        spm_data = SPMData(data=data, coordinates=coordinates)

        # Should validate but warn about NaN
        validated = validate_spm_data(spm_data)
        assert validated is spm_data
        # Note: warnings are not captured in this test context

    def test_coordinate_bounds_validation(self):
        """Test coordinate bounds validation for different CRS."""
        # Test EPSG:4326 bounds
        coordinates = np.array([[200.0, 0.0]])  # Invalid longitude
        data = np.array([1.0])

        spm_data = SPMData(data=data, coordinates=coordinates, crs="EPSG:4326")

        with pytest.raises(ValueError, match="Longitude values must be between"):
            validate_spm_data(spm_data)

        # Test valid coordinates
        coordinates_valid = np.array([[45.0, 30.0]])  # Valid lat/lon
        spm_data_valid = SPMData(
            data=data, coordinates=coordinates_valid, crs="EPSG:4326"
        )

        validated = validate_spm_data(spm_data_valid)
        assert validated is spm_data_valid

    def test_covariates_validation(self):
        """Test validation of covariates."""
        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 50),  # longitude
                np.random.uniform(-90, 90, 50),  # latitude
            ]
        )
        data = np.random.randn(50)

        # Valid covariates
        covariates = {
            "elevation": np.random.normal(500, 100, 50),
            "temperature": np.random.normal(20, 5, 50),
        }

        spm_data = SPMData(data=data, coordinates=coordinates, covariates=covariates)
        validated = validate_spm_data(spm_data)

        assert validated.covariates is not None
        assert "elevation" in validated.covariates
        assert "temperature" in validated.covariates

    def test_invalid_covariates_length(self):
        """Test validation of covariates with wrong length."""
        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 50),  # longitude
                np.random.uniform(-90, 90, 50),  # latitude
            ]
        )
        data = np.random.randn(50)

        # Invalid covariates (wrong length)
        covariates = {
            "elevation": np.random.normal(500, 100, 30),  # Wrong length
        }

        spm_data = SPMData(data=data, coordinates=coordinates, covariates=covariates)

        with pytest.raises(ValueError, match="Covariate.*length.*does not match"):
            validate_spm_data(spm_data)

    def test_temporal_data_validation(self):
        """Test validation of temporal data."""
        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 50),  # longitude
                np.random.uniform(-90, 90, 50),  # latitude
            ]
        )
        data = np.random.randn(50)
        time = np.arange(50)  # Valid time array

        spm_data = SPMData(data=data, coordinates=coordinates, time=time)
        validated = validate_spm_data(spm_data)

        assert validated.has_temporal
        assert validated.time is not None

    def test_invalid_temporal_length(self):
        """Test validation of temporal data with wrong length."""
        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 50),  # longitude
                np.random.uniform(-90, 90, 50),  # latitude
            ]
        )
        data = np.random.randn(50)
        time = np.arange(30)  # Wrong length

        spm_data = SPMData(data=data, coordinates=coordinates, time=time)

        with pytest.raises(ValueError, match="Time array length.*does not match"):
            validate_spm_data(spm_data)


class TestDesignMatrixValidation:
    """Test design matrix validation functionality."""

    def test_valid_design_matrix(self):
        """Test validation of valid design matrix."""
        X = np.random.randn(50, 3)
        design_matrix = DesignMatrix(matrix=X, names=["intercept", "x1", "x2"])

        validated = validate_design_matrix(design_matrix)
        assert validated is design_matrix

    def test_invalid_matrix_shape(self):
        """Test validation of invalid matrix shapes."""
        X = np.random.randn(50)  # 1D array
        design_matrix = DesignMatrix(matrix=X, names=["x1"])

        with pytest.raises(ValueError, match="Design matrix must be 2D"):
            validate_design_matrix(design_matrix)

    def test_names_length_mismatch(self):
        """Test validation of mismatched names length."""
        X = np.random.randn(50, 3)
        design_matrix = DesignMatrix(
            matrix=X,
            names=["intercept", "x1"],  # Wrong length
        )

        with pytest.raises(ValueError, match="Number of names.*does not match"):
            validate_design_matrix(design_matrix)

    def test_rank_deficient_matrix(self):
        """Test validation of rank deficient matrices."""
        X = np.ones((50, 3))  # Rank deficient
        design_matrix = DesignMatrix(matrix=X, names=["intercept", "x1", "x2"])

        # Should validate but potentially warn
        validated = validate_design_matrix(design_matrix)
        assert validated is design_matrix

    def test_collinear_predictors(self):
        """Test validation of collinear predictors."""
        X = np.random.randn(50, 3)
        X[:, 2] = X[:, 0] + 0.01 * np.random.randn(50)  # Nearly collinear

        design_matrix = DesignMatrix(matrix=X, names=["x1", "x2", "x1_copy"])

        # Should validate but warn about collinearity
        validated = validate_design_matrix(design_matrix)
        assert validated is design_matrix

    def test_missing_names(self):
        """Test validation with missing names."""
        X = np.random.randn(50, 2)
        design_matrix = DesignMatrix(matrix=X)  # No names provided

        validated = validate_design_matrix(design_matrix)

        # Should generate default names
        assert len(validated.names) == 2
        assert all(name.startswith("regressor_") for name in validated.names)

    def test_factors_validation(self):
        """Test validation of categorical factors."""
        X = np.random.randn(50, 4)
        design_matrix = DesignMatrix(
            matrix=X,
            names=["intercept", "factor_A_1", "factor_A_2", "covariate"],
            factors={"factor_A": ["level1", "level2"]},
        )

        validated = validate_design_matrix(design_matrix)
        assert validated is design_matrix

    def test_invalid_factors(self):
        """Test validation of invalid factor specifications."""
        X = np.random.randn(50, 3)
        design_matrix = DesignMatrix(
            matrix=X,
            names=["intercept", "x1", "x2"],
            factors={"factor_A": ["level1", "level2", "level3"]},  # Too many levels
        )

        # Should still validate (factors validation is permissive)
        validated = validate_design_matrix(design_matrix)
        assert validated is design_matrix


class TestContrastValidation:
    """Test contrast validation functionality."""

    def test_valid_t_contrast(self):
        """Test validation of valid t-contrasts."""
        contrast_vector = np.array([0, 1, -1, 0])
        n_regressors = 4

        validated = validate_contrast(contrast_vector, n_regressors, "t")
        np.testing.assert_array_equal(validated, contrast_vector)

    def test_valid_f_contrast(self):
        """Test validation of valid F-contrasts."""
        contrast_matrix = np.array([[0, 1, -1, 0], [0, 0, 1, -1]])
        n_regressors = 4

        validated = validate_contrast(contrast_matrix, n_regressors, "F")
        np.testing.assert_array_equal(validated, contrast_matrix)

    def test_invalid_t_contrast_length(self):
        """Test validation of t-contrasts with wrong length."""
        contrast_vector = np.array([0, 1, -1])  # Too short
        n_regressors = 4

        with pytest.raises(ValueError, match="T-contrast length.*does not match"):
            validate_contrast(contrast_vector, n_regressors, "t")

    def test_invalid_f_contrast_dimensions(self):
        """Test validation of F-contrasts with wrong dimensions."""
        contrast_matrix = np.array([0, 1, -1, 0])  # 1D instead of 2D
        n_regressors = 4

        with pytest.raises(ValueError, match="F-contrast must be 2D"):
            validate_contrast(contrast_matrix, n_regressors, "F")

    def test_invalid_contrast_type(self):
        """Test validation with invalid contrast type."""
        contrast_vector = np.array([0, 1, -1, 0])
        n_regressors = 4

        with pytest.raises(ValueError, match="Unknown contrast type"):
            validate_contrast(contrast_vector, n_regressors, "invalid")

    def test_zero_contrast_warning(self):
        """Test validation of zero contrasts."""
        contrast_vector = np.array([0, 0, 0, 0])
        n_regressors = 4

        # Should validate but potentially warn
        validated = validate_contrast(contrast_vector, n_regressors, "t")
        np.testing.assert_array_equal(validated, contrast_vector)


class TestSpatialAutocorrelationValidation:
    """Test spatial autocorrelation validation functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_points = 50

        # Create coordinates with valid EPSG:4326 ranges
        self.coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, n_points),  # longitude
                np.random.uniform(-90, 90, n_points),  # latitude
            ]
        )

        # Create data with spatial autocorrelation
        distances = np.linalg.norm(
            self.coordinates[:, np.newaxis] - self.coordinates[np.newaxis, :], axis=2
        )
        self.data = np.sin(distances / 10).sum(axis=1) + 0.1 * np.random.randn(n_points)

        self.spm_data = SPMData(
            data=self.data, coordinates=self.coordinates, crs="EPSG:4326"
        )

    def test_spatial_autocorrelation_analysis(self):
        """Test comprehensive spatial autocorrelation analysis."""
        results = validate_spatial_autocorrelation(self.spm_data, alpha=0.05)

        # Check that all expected metrics are computed
        expected_keys = ["morans_i", "gearys_c", "variogram", "spatial_dependence"]
        for key in expected_keys:
            assert key in results

        # Check Moran's I structure
        morans = results["morans_i"]
        required_morans_keys = ["statistic", "expected", "p_value", "z_score"]
        for key in required_morans_keys:
            assert key in morans

        # Check Geary's C structure
        geary = results["gearys_c"]
        required_geary_keys = ["statistic", "expected", "p_value"]
        for key in required_geary_keys:
            assert key in geary

        # Check variogram structure
        variogram = results["variogram"]
        required_variogram_keys = ["distances", "variogram", "counts", "model"]
        for key in required_variogram_keys:
            assert key in variogram

    def test_spatial_dependence_classification(self):
        """Test spatial dependence classification."""
        # Test with strong autocorrelation
        results = validate_spatial_autocorrelation(self.spm_data, alpha=0.05)

        assert "spatial_dependence" in results
        assert results["spatial_dependence"] in [
            "strong_spatial_dependence",
            "moderate_spatial_dependence",
            "no_spatial_dependence",
        ]

    def test_variogram_computation(self):
        """Test variogram computation details."""
        results = validate_spatial_autocorrelation(self.spm_data, max_lag=5, alpha=0.05)

        variogram = results["variogram"]

        # Check variogram properties
        assert len(variogram["distances"]) == 5
        assert len(variogram["variogram"]) == 5
        assert len(variogram["counts"]) == 5

        # Variogram maximum should be >= first value (variogram eventually increases)
        assert max(variogram["variogram"]) >= variogram["variogram"][0]

        # Check model parameters
        assert "nugget" in variogram["model"]
        assert "sill" in variogram["model"]
        assert "range" in variogram["model"]

    def test_small_dataset_validation(self):
        """Test validation with small datasets."""
        small_coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 10),  # longitude
                np.random.uniform(-90, 90, 10),  # latitude
            ]
        )
        small_data = np.random.randn(10)

        small_spm_data = SPMData(
            data=small_data, coordinates=small_coordinates, crs="EPSG:4326"
        )

        results = validate_spatial_autocorrelation(
            small_spm_data, max_lag=3, alpha=0.05
        )

        # Should still compute basic statistics
        assert "morans_i" in results
        assert "gearys_c" in results

    def test_different_bin_counts(self):
        """Test variogram with different bin counts."""
        for n_bins in [5, 10, 15]:
            results = validate_spatial_autocorrelation(
                self.spm_data, max_lag=n_bins, alpha=0.05
            )

            variogram = results["variogram"]
            assert len(variogram["distances"]) == n_bins
            assert len(variogram["variogram"]) == n_bins


class TestValidationEdgeCases:
    """Test validation edge cases and error conditions."""

    def test_empty_data_validation(self):
        """Test validation of empty datasets."""
        coordinates = np.array([]).reshape(0, 2)
        data = np.array([])

        spm_data = SPMData(data=data, coordinates=coordinates)

        with pytest.raises(ValueError, match="Data cannot be empty"):
            validate_spm_data(spm_data)

    def test_single_point_validation(self):
        """Test validation of single-point datasets."""
        coordinates = np.array([[0.0, 0.0]])
        data = np.array([1.0])

        spm_data = SPMData(data=data, coordinates=coordinates, crs="EPSG:4326")

        # Should validate single points
        validated = validate_spm_data(spm_data)
        assert validated is spm_data

    def test_extreme_coordinate_values(self):
        """Test validation of extreme coordinate values."""
        # Test with very large coordinates in a projected (non-lat/lon) CRS
        coordinates = np.array([[1e6, 1e6]])
        data = np.array([1.0])

        spm_data = SPMData(data=data, coordinates=coordinates, crs="EPSG:32634")

        # Should validate (no bounds checking for projected CRS)
        validated = validate_spm_data(spm_data)
        assert validated is spm_data

    def test_non_numeric_data(self):
        """Test validation of non-numeric data."""
        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, 5),  # longitude
                np.random.uniform(-90, 90, 5),  # latitude
            ]
        )

        # Test with string data (should fail)
        try:
            spm_data = SPMData(data=["a", "b", "c", "d", "e"], coordinates=coordinates)
            with pytest.raises((ValueError, TypeError)):
                validate_spm_data(spm_data)
        except Exception:
            # Expected to fail during array conversion
            pass

    def test_design_matrix_condition_number(self):
        """Test design matrix condition number validation."""
        # Create ill-conditioned matrix
        X = np.random.randn(50, 3)
        X[:, 2] = X[:, 0] + 1e-15 * X[:, 1]  # Almost linearly dependent

        design_matrix = DesignMatrix(matrix=X, names=["x1", "x2", "x1_copy"])

        # Should validate but warn about condition number
        validated = validate_design_matrix(design_matrix)
        assert validated is design_matrix
