"""
Unit tests for contrast analysis functionality
"""

import numpy as np
import pytest

from geo_infer_spm.models.data_models import (
    SPMData,
    DesignMatrix,
    SPMResult,
    ContrastResult,
)
from geo_infer_spm.core.contrasts import Contrast, contrast


class TestContrast:
    """Test Contrast class functionality."""

    def setup_method(self):
        """Set up test data."""
        self.design_names = ["intercept", "condition_A", "condition_B", "covariate"]

    def test_contrast_initialization(self):
        """Test contrast initialization."""
        vector = np.array([0, 1, -1, 0])
        contrast_obj = Contrast(vector, name="A_vs_B")

        assert np.array_equal(contrast_obj.vector, vector)
        assert contrast_obj.name == "A_vs_B"
        assert contrast_obj.type == "t"

    def test_contrast_from_string_simple(self):
        """Test parsing simple contrast strings."""
        contrast_obj = Contrast.from_string(
            "condition_A > condition_B", self.design_names
        )

        expected_vector = np.array([0, 1, -1, 0])
        np.testing.assert_array_equal(contrast_obj.vector, expected_vector)
        assert contrast_obj.name == "condition_A > condition_B"

    def test_contrast_from_string_complex(self):
        """Test parsing complex contrast strings."""
        contrast_obj = Contrast.from_string(
            "condition_A + condition_B > 2*covariate", self.design_names
        )

        # A + B - 2*C = [0, 1, 1, -2]
        expected_vector = np.array([0, 1, 1, -2])
        np.testing.assert_array_equal(contrast_obj.vector, expected_vector)

    def test_contrast_from_string_coefficients(self):
        """Test parsing contrast strings with coefficients."""
        contrast_obj = Contrast.from_string(
            "2*condition_A - condition_B", self.design_names
        )

        expected_vector = np.array([0, 2, -1, 0])
        np.testing.assert_array_equal(contrast_obj.vector, expected_vector)

    def test_f_contrast(self):
        """Test F-contrast initialization."""
        # Two contrasts: A>B and A>C
        f_matrix = np.array(
            [
                [0, 1, -1, 0],  # A > B
                [0, 1, 0, -1],  # A > C (assuming C is another condition)
            ]
        )

        contrast_obj = Contrast(f_matrix, name="A_tests", contrast_type="F")

        assert contrast_obj.type == "f"
        assert contrast_obj.vector.shape == (2, 4)

    def test_invalid_contrast_string(self):
        """Test error handling for invalid contrast strings."""
        with pytest.raises(ValueError):
            Contrast.from_string("invalid > contrast", self.design_names)


class TestContrastAnalysis:
    """Test contrast computation functionality."""

    def setup_method(self):
        """Set up test data with known contrasts."""
        np.random.seed(42)
        n_points = 100

        # Create design matrix for 2x2 factorial design
        # Conditions: A1, A2, B1, B2
        self.design_matrix = DesignMatrix(
            matrix=np.random.randn(n_points, 4), names=["intercept", "A", "B", "A:B"]
        )

        # Known beta coefficients
        beta_true = np.array([1.0, 2.0, 1.5, 0.5])
        X = self.design_matrix.matrix
        self.y = X @ beta_true + 0.1 * np.random.randn(n_points)

        # Create mock SPM result
        self.spm_data = SPMData(data=self.y, coordinates=np.random.rand(n_points, 2))

        self.spm_result = SPMResult(
            spm_data=self.spm_data,
            design_matrix=self.design_matrix,
            beta_coefficients=beta_true,
            residuals=self.y - X @ beta_true,
            model_diagnostics={"r_squared": 0.95},
        )

    def test_t_contrast_computation(self):
        """Test t-contrast computation."""
        # Test main effect of A
        contrast_result = contrast(self.spm_result, [0, 1, 0, 0])

        assert isinstance(contrast_result, ContrastResult)
        assert hasattr(contrast_result, "t_statistic")
        assert hasattr(contrast_result, "p_values")
        assert hasattr(contrast_result, "effect_size")
        assert hasattr(contrast_result, "standard_error")

        # Effect size should be close to true beta (2.0)
        np.testing.assert_allclose(contrast_result.effect_size, 2.0, atol=0.2)

    def test_contrast_from_string_integration(self):
        """Test contrast computation from string."""
        # Test using string specification
        contrast_result = contrast(self.spm_result, "A > intercept")

        # Should have positive effect (A coefficient is 2.0, intercept is 1.0)
        assert contrast_result.effect_size > 0
        assert contrast_result.p_values < 0.05  # Should be significant

    def test_multiple_contrasts(self):
        """Test multiple contrast computation."""
        contrasts = [
            [0, 1, 0, 0],  # A effect
            [0, 0, 1, 0],  # B effect
            [0, 0, 0, 1],  # Interaction
        ]

        results = []
        for c_vec in contrasts:
            result = contrast(self.spm_result, c_vec)
            results.append(result)

        # All should be significant
        for result in results:
            assert result.p_values < 0.05

        # Effects should match true betas
        np.testing.assert_allclose(results[0].effect_size, 2.0, atol=0.2)  # A
        np.testing.assert_allclose(results[1].effect_size, 1.5, atol=0.2)  # B
        np.testing.assert_allclose(results[2].effect_size, 0.5, atol=0.2)  # A:B

    def test_contrast_significance_mask(self):
        """Test significance masking."""
        contrast_result = contrast(self.spm_result, [0, 1, 0, 0])

        # With default threshold, should be significant
        assert contrast_result.p_values < 0.05
        if contrast_result.significance_mask is not None:
            assert np.all(contrast_result.significance_mask)

    def test_invalid_contrast_dimensions(self):
        """Test error handling for invalid contrast dimensions."""
        # Wrong length contrast vector
        with pytest.raises(ValueError, match="length.*does not match"):
            contrast(self.spm_result, [0, 1])  # Too short

        with pytest.raises(ValueError, match="length.*does not match"):
            contrast(self.spm_result, [0, 1, 0, 0, 1])  # Too long


class TestCommonContrasts:
    """Test generation of common contrast patterns."""

    def test_categorical_contrasts(self):
        """Test automatic generation of categorical contrasts."""
        from geo_infer_spm.core.contrasts import generate_common_contrasts

        design_matrix = DesignMatrix(
            matrix=np.random.randn(50, 4),
            names=["intercept", "group1", "group2", "group3"],
        )

        contrasts = generate_common_contrasts(design_matrix, design_type="categorical")

        # Should generate pairwise comparisons
        assert len(contrasts) >= 3  # 3 pairwise comparisons for 3 groups

        # Check contrast vectors
        for c in contrasts:
            assert len(c.vector) == 4  # Match design matrix
            assert c.vector[0] == 0  # Intercept should be zero

    def test_trend_contrasts(self):
        """Test automatic generation of trend contrasts."""
        from geo_infer_spm.core.contrasts import generate_common_contrasts

        design_matrix = DesignMatrix(
            matrix=np.random.randn(50, 5),
            names=["intercept", "time1", "time2", "time3", "time4"],
        )

        contrasts = generate_common_contrasts(design_matrix, design_type="trend")

        # Should generate trend contrast
        assert len(contrasts) >= 1

        trend_contrast = contrasts[0]
        assert len(trend_contrast.vector) == 5
        assert trend_contrast.vector[0] == 0  # Intercept zero

        # Trend should be monotonic (increasing or decreasing)
        time_weights = trend_contrast.vector[1:]
        assert np.all(np.diff(np.abs(time_weights)) >= 0)  # Monotonic trend
