"""
Unit tests for data input/output functionality
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import json
import os

from geo_infer_spm.models.data_models import SPMData
from geo_infer_spm.utils.data_io import (
    load_data,
    save_spm,
    load_csv_with_coords,
    load_json_data,
    _save_spm_json,
    _save_spm_csv,
)
from geo_infer_spm.models.data_models import DesignMatrix


class TestDataLoading:
    """Test data loading functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_points = 50
        self.coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, self.n_points),
                np.random.uniform(-89, 89, self.n_points),
            ]
        )
        self.data = np.random.randn(self.n_points)
        self.covariates = {
            "elevation": np.random.normal(500, 100, self.n_points),
            "temperature": self.data + np.random.normal(0, 1, self.n_points),
        }

    def test_load_json_data(self):
        """Test loading JSON data."""
        json_data = {
            "data": self.data.tolist(),
            "coordinates": self.coordinates.tolist(),
            "covariates": {k: v.tolist() for k, v in self.covariates.items()},
            "metadata": {"test": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            spm_data = load_json_data(temp_path, "data", "coordinates")

            np.testing.assert_array_equal(spm_data.data, self.data)
            np.testing.assert_array_equal(spm_data.coordinates, self.coordinates)
            assert spm_data.metadata["test"]

        finally:
            os.unlink(temp_path)

    def test_load_csv_with_coords(self):
        """Test loading CSV data with coordinates."""
        df = pd.DataFrame(
            {
                "longitude": self.coordinates[:, 0],
                "latitude": self.coordinates[:, 1],
                "value": self.data,
                "elevation": self.covariates["elevation"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            spm_data = load_csv_with_coords(
                temp_path,
                x_column="longitude",
                y_column="latitude",
                value_column="value",
            )

            np.testing.assert_allclose(spm_data.data, self.data, rtol=1e-5)
            np.testing.assert_allclose(
                spm_data.coordinates, self.coordinates[:, [0, 1]], rtol=1e-5
            )

        finally:
            os.unlink(temp_path)

    def test_load_data_dispatch(self):
        """Test automatic format detection in load_data."""
        # Test JSON
        json_data = {
            "data": self.data.tolist(),
            "coordinates": self.coordinates.tolist(),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            spm_data = load_data(temp_path)
            np.testing.assert_array_equal(spm_data.data, self.data)
            np.testing.assert_array_equal(spm_data.coordinates, self.coordinates)

        finally:
            os.unlink(temp_path)

    def test_invalid_file_format(self):
        """Test error handling for invalid file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data("test.invalid")

    def test_missing_file(self):
        """Test error handling for missing files."""
        with pytest.raises(ValueError, match="path cannot be empty"):
            load_data("")


class TestDataSaving:
    """Test data saving functionality."""

    def setup_method(self):
        """Set up test SPM result."""
        np.random.seed(42)
        n_points = 50

        # Create mock SPM result
        coordinates = np.random.rand(n_points, 2)
        data = np.random.randn(n_points)
        X = np.random.randn(n_points, 3)

        spm_data = SPMData(data=data, coordinates=coordinates)
        design_matrix = DesignMatrix(matrix=X, names=["intercept", "x1", "x2"])

        from geo_infer_spm.core.glm import GeneralLinearModel

        glm = GeneralLinearModel(design_matrix)
        self.spm_result = glm.fit(spm_data)

    def test_save_json(self):
        """Test saving SPM results as JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            _save_spm_json(self.spm_result, temp_path)

            # Load and verify
            with open(temp_path) as f:
                saved_data = json.load(f)

            assert "beta_coefficients" in saved_data
            assert "residuals" in saved_data
            assert "model_diagnostics" in saved_data
            np.testing.assert_array_equal(
                saved_data["beta_coefficients"],
                self.spm_result.beta_coefficients.tolist(),
            )

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_csv(self):
        """Test saving SPM results as CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            _save_spm_csv(self.spm_result, temp_path)

            # Load and verify
            df = pd.read_csv(temp_path)

            assert "longitude" in df.columns
            assert "latitude" in df.columns
            assert "residuals" in df.columns
            assert len(df) == len(self.spm_result.spm_data.data)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_spm_dispatch(self):
        """Test save_spm format dispatch."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_spm(self.spm_result, temp_path, format="json")

            # Verify file was created and has content
            assert os.path.exists(temp_path)
            with open(temp_path) as f:
                data = json.load(f)
            assert "beta_coefficients" in data

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_invalid_save_format(self):
        """Test error handling for invalid save formats."""
        with pytest.raises(ValueError, match="Unsupported save format"):
            save_spm(self.spm_result, "test.invalid", format="invalid")


class TestSPMDataValidation:
    """Test SPMData validation."""

    def test_valid_spm_data(self):
        """Test validation of valid SPMData."""
        from geo_infer_spm.utils.validation import validate_spm_data

        coordinates = np.random.rand(50, 2)
        data = np.random.randn(50)

        spm_data = SPMData(data=data, coordinates=coordinates)
        validated = validate_spm_data(spm_data)

        assert validated is spm_data
        assert "validation" in validated.metadata

    def test_invalid_coordinates_shape(self):
        """Test validation of invalid coordinate shapes."""
        from geo_infer_spm.utils.validation import validate_spm_data

        coordinates = np.random.rand(50, 3)  # Wrong shape
        data = np.random.randn(50)

        spm_data = SPMData(data=data, coordinates=coordinates)

        with pytest.raises(ValueError, match="Coordinates must have shape"):
            validate_spm_data(spm_data)

    def test_coordinate_data_mismatch(self):
        """Test validation of mismatched coordinate/data dimensions."""
        from geo_infer_spm.utils.validation import validate_spm_data

        coordinates = np.random.rand(50, 2)
        data = np.random.randn(30)  # Different size

        spm_data = SPMData(data=data, coordinates=coordinates)

        with pytest.raises(ValueError, match="Coordinate count.*does not match"):
            validate_spm_data(spm_data)

    def test_nan_data_handling(self):
        """Test handling of NaN values in data."""
        from geo_infer_spm.utils.validation import validate_spm_data

        coordinates = np.random.rand(50, 2)
        data = np.random.randn(50)
        data[10:15] = np.nan  # Add NaN values

        spm_data = SPMData(data=data, coordinates=coordinates)
        validated = validate_spm_data(spm_data)

        # Should warn about NaN values but not fail
        assert validated is spm_data
