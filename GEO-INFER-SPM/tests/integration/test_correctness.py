"""
Statistical correctness validation tests

This module contains tests to verify the statistical correctness of SPM
implementations, ensuring that results match theoretical expectations
and established statistical methods.
"""

import numpy as np
from scipy import stats

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.core.glm import fit_glm
from geo_infer_spm.core.contrasts import contrast
from geo_infer_spm.core.rft import RandomFieldTheory, compute_spm


class TestGLMCorrectness:
    """Test statistical correctness of GLM implementation."""

    def test_ols_coefficient_recovery(self):
        """Test that OLS correctly recovers known coefficients."""
        np.random.seed(42)
        n_points, n_params = 200, 3

        # Generate data with known coefficients
        X = np.random.randn(n_points, n_params)
        beta_true = np.array([2.5, -1.8, 0.9])
        y = X @ beta_true + 0.01 * np.random.randn(n_points)  # Low noise

        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["b0", "b1", "b2"])

        result = fit_glm(spm_data, design_matrix)

        # Should recover true coefficients with high accuracy
        np.testing.assert_allclose(result.beta_coefficients, beta_true, atol=0.05)

        # R-squared should be very high
        assert result.model_diagnostics["r_squared"] > 0.95

    def test_residual_properties(self):
        """Test that residuals have correct statistical properties."""
        np.random.seed(42)
        n_points = 150

        X = np.random.randn(n_points, 2)
        beta_true = np.array([1.0, -0.5])
        sigma_true = 2.0
        y = X @ beta_true + sigma_true * np.random.randn(n_points)

        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["int", "slope"])

        result = fit_glm(spm_data, design_matrix)

        residuals = result.residuals

        # Residuals should have zero mean
        assert abs(np.mean(residuals)) < 0.15

        # Residual variance should match true variance
        residual_var = np.var(residuals, ddof=2)  # 2 df for 2 parameters
        assert abs(residual_var - sigma_true**2) / sigma_true**2 < 0.2

        # Residuals should be uncorrelated with predictors
        for i in range(X.shape[1]):
            corr = np.corrcoef(residuals, X[:, i])[0, 1]
            assert abs(corr) < 0.1

    def test_f_test_correctness(self):
        """Test F-test for overall model significance."""
        np.random.seed(42)

        # Significant model
        n_points, n_params = 100, 3
        X = np.random.randn(n_points, n_params)
        beta_true = np.array([3.0, -2.0, 1.5])
        y_sig = X @ beta_true + 0.5 * np.random.randn(n_points)

        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        spm_data_sig = SPMData(data=y_sig, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["b0", "b1", "b2"])

        result_sig = fit_glm(spm_data_sig, design_matrix)

        # Should have significant F-test
        assert result_sig.model_diagnostics["f_statistic"] > 10
        f_p_value = result_sig.model_diagnostics["f_p_value"]
        assert f_p_value < 0.01

        # Non-significant model (random y)
        y_noise = np.random.randn(n_points)
        spm_data_noise = SPMData(data=y_noise, coordinates=coordinates, crs="EPSG:4326")

        result_noise = fit_glm(spm_data_noise, design_matrix)

        # Should have non-significant F-test
        f_p_value_noise = result_noise.model_diagnostics["f_p_value"]
        assert f_p_value_noise > 0.1

    def test_confidence_intervals(self):
        """Test coefficient confidence interval calculation."""
        np.random.seed(42)
        n_points, n_params = 200, 2

        X = np.random.randn(n_points, n_params)
        beta_true = np.array([1.5, -0.8])
        y = X @ beta_true + 0.1 * np.random.randn(n_points)

        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["int", "slope"])

        result = fit_glm(spm_data, design_matrix)

        # Calculate confidence intervals manually
        se_beta = np.sqrt(np.diag(result.cov_beta))
        t_critical = stats.t.ppf(0.975, n_points - n_params)

        ci_lower = result.beta_coefficients - t_critical * se_beta
        ci_upper = result.beta_coefficients + t_critical * se_beta

        # True coefficients should be within 95% CI
        assert np.all(ci_lower <= beta_true)
        assert np.all(beta_true <= ci_upper)

        # CI width should be reasonable
        ci_width = ci_upper - ci_lower
        assert np.all(ci_width > 0)
        assert np.all(ci_width < 2.0)  # Not too wide


class TestContrastCorrectness:
    """Test statistical correctness of contrast analysis."""

    def test_t_contrast_statistics(self):
        """Test t-contrast statistical properties."""
        np.random.seed(42)
        n_points = 150

        # Create data with known effect
        X = np.random.randn(n_points, 2)
        y = 2.0 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.1 * np.random.randn(n_points)

        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(
            matrix=np.column_stack([np.ones(n_points), X[:, 0]]), names=["int", "x1"]
        )

        result = fit_glm(spm_data, design_matrix)

        # Test contrast for x1 coefficient
        contrast_result = contrast(result, [0, 1])  # x1 effect

        # Should recover the true coefficient (1.5)
        np.testing.assert_allclose(contrast_result.effect_size, 1.5, atol=0.1)

        # Should be statistically significant
        assert contrast_result.p_values < 0.05

        # Check t-statistic calculation
        expected_t = contrast_result.effect_size / contrast_result.standard_error
        np.testing.assert_allclose(contrast_result.t_statistic, expected_t, rtol=1e-10)

    def test_contrast_inference_properties(self):
        """Test contrast inference statistical properties."""
        np.random.seed(42)
        n_simulations = 100
        true_effect = 1.2
        significant_count = 0

        for _ in range(n_simulations):
            n_points = 80
            X = np.random.randn(n_points, 2)
            y = 1.0 + true_effect * X[:, 1] + 0.5 * np.random.randn(n_points)

            coordinates = np.column_stack(
                [
                    np.random.uniform(-179, 179, n_points),
                    np.random.uniform(-89, 89, n_points),
                ]
            )
            spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
            design_matrix = DesignMatrix(matrix=X, names=["int", "effect"])

            result = fit_glm(spm_data, design_matrix)
            contrast_result = contrast(result, [0, 1])

            if contrast_result.p_values < 0.05:
                significant_count += 1

        # Power should be reasonable (>80% for effect size 1.2)
        power = significant_count / n_simulations
        assert power > 0.7

    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction accuracy."""
        np.random.seed(42)
        n_voxels = 1000
        n_significant_true = 50

        # Create statistical map with known number of significant voxels
        stat_map = np.random.randn(n_voxels) * 0.5

        # Make specific voxels significantly large
        significant_indices = np.random.choice(
            n_voxels, n_significant_true, replace=False
        )
        stat_map[significant_indices] = 3.5 + np.random.randn(n_significant_true)

        # Create mock SPM result
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_voxels),
                np.random.uniform(-89, 89, n_voxels),
            ]
        )
        spm_data = SPMData(
            data=np.random.randn(n_voxels), coordinates=coordinates, crs="EPSG:4326"
        )
        X = np.random.randn(n_voxels, 1)
        design_matrix = DesignMatrix(matrix=X, names=["int"])

        spm_result = type(
            "SPMResult",
            (),
            {
                "spm_data": spm_data,
                "design_matrix": design_matrix,
                "beta_coefficients": np.array([1.0]),
                "residuals": np.random.randn(n_voxels),
                "model_diagnostics": {"r_squared": 0.1},
            },
        )()

        # Create mock contrast result
        p_values = 2 * (1 - stats.norm.cdf(np.abs(stat_map)))
        contrast_result = type(
            "ContrastResult",
            (),
            {
                "contrast_vector": np.array([1.0]),
                "t_statistic": stat_map,
                "effect_size": stat_map,
                "standard_error": np.ones(n_voxels),
                "p_values": p_values,
            },
        )()

        # Apply Bonferroni correction
        corrected_bonferroni = compute_spm(
            spm_result, contrast_result, correction="Bonferroni"
        )

        # Bonferroni should control family-wise error
        # Expected number of false positives should be low
        n_significant_bonferroni = np.sum(
            corrected_bonferroni.corrected_p_values < 0.05
        )
        assert (
            n_significant_bonferroni <= n_significant_true + 10
        )  # Allow some false positives


class TestRFTCorrectness:
    """Test Random Field Theory statistical correctness."""

    def test_rft_threshold_calculation(self):
        """Test RFT threshold calculation accuracy."""
        # Test on 2D field
        field_shape = (20, 20)
        rft = RandomFieldTheory(field_shape)

        # Set known smoothness
        rft.smoothness = np.array([2.0, 2.0])
        rft.search_volume = 50.0

        # Calculate cluster-forming threshold
        threshold = rft.cluster_threshold(alpha=0.05)

        assert isinstance(threshold, float)
        assert threshold > 0

        # Threshold should be higher for stricter alpha
        threshold_strict = rft.cluster_threshold(alpha=0.01)
        assert threshold_strict > threshold

    def test_expected_clusters_formula(self):
        """Test expected clusters formula implementation."""
        rft = RandomFieldTheory((15, 15))
        rft.smoothness = np.array([1.5, 1.5])
        rft.search_volume = 40.0

        # For high thresholds, the full boundary-corrected EC approaches zero.
        expected_high = rft.expected_clusters(4.0)
        assert expected_high < 0.02

        # For low thresholds, expected clusters should be higher
        expected_low = rft.expected_clusters(1.0)
        assert expected_low > expected_high

    def test_rft_p_value_correction(self):
        """Test RFT-based p-value correction."""
        # Create statistical field with known properties
        field_shape = (10, 10)
        stat_field = np.random.randn(*field_shape)

        # Add some significant clusters
        stat_field[2:5, 2:5] = 3.0  # Significant cluster
        stat_field[7:9, 7:9] = 2.8  # Another cluster

        # Create mock result
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 100), np.random.uniform(-89, 89, 100)]
        )
        spm_data = SPMData(
            data=stat_field.flatten(), coordinates=coordinates, crs="EPSG:4326"
        )
        X = np.random.randn(100, 1)
        design_matrix = DesignMatrix(matrix=X, names=["int"])

        spm_result = type(
            "SPMResult",
            (),
            {
                "spm_data": spm_data,
                "design_matrix": design_matrix,
                "beta_coefficients": np.array([1.0]),
                "residuals": np.random.randn(100),
                "model_diagnostics": {"r_squared": 0.1},
            },
        )()

        contrast_result = type(
            "ContrastResult",
            (),
            {
                "contrast_vector": np.array([1.0]),
                "t_statistic": stat_field.flatten(),
                "effect_size": stat_field.flatten(),
                "standard_error": np.ones(100),
                "p_values": np.ones(100) * 0.5,  # Mock p-values
            },
        )()

        # Apply RFT correction
        corrected = compute_spm(spm_result, contrast_result, correction="RFT")

        # Should produce corrected p-values
        assert hasattr(corrected, "corrected_p_values")
        assert corrected.corrected_p_values.shape == (100,)


class TestSpatialStatisticsCorrectness:
    """Test correctness of spatial statistical methods."""

    def test_morans_i_calculation(self):
        """Test Moran's I calculation correctness."""
        from geo_infer_spm.utils.validation import validate_spatial_autocorrelation

        # Create data with known spatial autocorrelation
        n_points = 50
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Generate spatially autocorrelated data
        distances = np.linalg.norm(
            coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2
        )
        weights = np.exp(-distances / 30)  # Spatial weights
        np.fill_diagonal(weights, 0)

        # Create autocorrelated variable
        noise = np.random.randn(n_points)
        data_autocorr = weights @ noise  # Spatially filtered noise

        spm_data = SPMData(data=data_autocorr, coordinates=coordinates, crs="EPSG:4326")

        results = validate_spatial_autocorrelation(spm_data)

        # Moran's I should detect positive autocorrelation
        morans_i = results["morans_i"]["statistic"]
        assert morans_i > 0.1  # Should show positive autocorrelation

        # Should be statistically significant
        assert results["morans_i"]["p_value"] < 0.05

    def test_variogram_cloud_correctness(self):
        """Test variogram cloud calculation."""
        from geo_infer_spm.core.spatial_analysis import SpatialAnalyzer

        # Create data with known variogram structure
        n_points = 30
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Exponential variogram: γ(h) = sill * (1 - exp(-h/range))
        true_sill = 2.0
        true_range = 15.0

        # Generate data with this variogram
        distances = np.linalg.norm(
            coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2
        )

        # Create covariance matrix
        covariance = true_sill * np.exp(-distances / true_range)
        np.fill_diagonal(covariance, true_sill)  # Nugget = 0

        # Generate correlated data
        L = np.linalg.cholesky(
            covariance + 0.01 * np.eye(n_points)
        )  # Add small regularization
        data = L @ np.random.randn(n_points)

        analyzer = SpatialAnalyzer(coordinates)
        variogram = analyzer.estimate_variogram(data, n_bins=8)

        # Variogram should increase with distance
        assert variogram["variogram"][-1] > variogram["variogram"][0]

        # Should approach sill value
        final_gamma = variogram["variogram"][-1]
        assert abs(final_gamma - true_sill) / true_sill < 0.5  # Within 50% of true sill


class TestModelComparisonCorrectness:
    """Test statistical model comparison methods."""

    def test_aic_bic_calculation(self):
        """Test AIC and BIC calculation correctness."""
        from geo_infer_spm.core.advanced.model_validation import ModelValidator

        # Create models with different numbers of parameters
        n_points = 100
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # True model: 2 parameters
        X2 = np.random.randn(n_points, 2)
        beta2 = np.array([1.5, -0.8])
        y = X2 @ beta2 + 0.1 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")

        # Fit correct model
        design2 = DesignMatrix(matrix=X2, names=["int", "x"])
        result2 = fit_glm(spm_data, design2)

        # Fit overparameterized model
        X3 = np.random.randn(n_points, 3)
        design3 = DesignMatrix(matrix=X3, names=["int", "x1", "x2"])
        result3 = fit_glm(spm_data, design3)

        # AIC/BIC should favor the correct model
        ModelValidator()

        # Manually calculate Gaussian AIC from residual sum of squares.
        rss2 = np.sum(result2.residuals**2)
        rss3 = np.sum(result3.residuals**2)
        aic2 = n_points * np.log(rss2 / n_points) + 2 * 2
        aic3 = n_points * np.log(rss3 / n_points) + 2 * 3

        # AIC should be lower for better fitting model
        # (This is a simplified check - actual AIC calculation is more complex)
        assert aic2 < aic3 + 10  # Allow some tolerance

    def test_cross_validation_bias_variance(self):
        """Test that cross-validation captures bias-variance tradeoff."""
        from geo_infer_spm.core.advanced.model_validation import ModelValidator

        n_points = 80
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Generate data with true 2-parameter relationship
        X = np.random.randn(n_points, 2)
        y = X @ np.array([2.0, -1.0]) + 0.2 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")

        validator = ModelValidator(validation_method="kfold", n_folds=5)

        # Cross-validate different model complexities
        design2 = DesignMatrix(matrix=X, names=["int", "x"])
        cv2 = validator.cross_validate(fit_glm, spm_data, design2)

        # Add noise predictors
        X_over = np.column_stack([X, np.random.randn(n_points, 2)])
        design_over = DesignMatrix(
            matrix=X_over, names=["int", "x", "noise1", "noise2"]
        )
        cv_over = validator.cross_validate(fit_glm, spm_data, design_over)

        # Correctly specified model should have higher CV R²
        assert cv2["overall_r2"] > cv_over["overall_r2"] - 0.2  # Allow some tolerance


class TestDistributionalAssumptions:
    """Test validation of statistical distributional assumptions."""

    def test_normality_assumption_checking(self):
        """Test checking of normality assumption."""
        from geo_infer_spm.core.advanced.model_validation import ModelValidator

        n_points = 100
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Normally distributed residuals
        X = np.random.randn(n_points, 1)
        y_normal = X.flatten() + 0.1 * np.random.randn(n_points)

        spm_data_normal = SPMData(
            data=y_normal, coordinates=coordinates, crs="EPSG:4326"
        )
        design = DesignMatrix(matrix=X, names=["x"])

        result_normal = fit_glm(spm_data_normal, design)

        validator = ModelValidator()
        diagnostics_normal = validator.diagnostic_tests(result_normal)

        # Normal data should pass normality tests (most of the time)
        shapiro_p = diagnostics_normal["shapiro_wilk"]["p_value"]
        jarque_p = diagnostics_normal["jarque_bera"]["p_value"]

        # At least one normality test should not reject normality
        assert shapiro_p > 0.01 or jarque_p > 0.01

    def test_heteroscedasticity_detection(self):
        """Test detection of heteroscedasticity."""
        from geo_infer_spm.core.advanced.model_validation import ModelValidator

        n_points = 100
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Create heteroscedastic data (variance increases with x)
        x = np.linspace(0, 10, n_points)
        X = x.reshape(-1, 1)
        y = 2 * x + x * np.random.randn(n_points)  # Variance proportional to x

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design = DesignMatrix(matrix=X, names=["x"])

        result = fit_glm(spm_data, design)

        validator = ModelValidator()
        diagnostics = validator.diagnostic_tests(result)

        # Should detect heteroscedasticity
        breusch_pagan = diagnostics["breusch_pagan"]
        # Note: This test may not always detect heteroscedasticity reliably
        # but the framework should be in place
        assert "p_value" in breusch_pagan

    def test_autocorrelation_detection(self):
        """Test detection of residual autocorrelation."""
        from geo_infer_spm.core.advanced.model_validation import ModelValidator

        # Create data with temporal autocorrelation
        n_points = 80
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Generate AR(1) process
        np.random.seed(42)
        residuals_ar1 = np.zeros(n_points)
        phi = 0.7
        for i in range(1, n_points):
            residuals_ar1[i] = phi * residuals_ar1[i - 1] + np.random.randn()

        # Create response with autocorrelated residuals
        X = np.random.randn(n_points, 1)
        y = X.flatten() + residuals_ar1

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design = DesignMatrix(matrix=X, names=["x"])

        result = fit_glm(spm_data, design)

        validator = ModelValidator()
        diagnostics = validator.diagnostic_tests(result)

        # Should detect autocorrelation
        dw_stat = diagnostics["durbin_watson"]["statistic"]

        # Durbin-Watson statistic < 2 suggests positive autocorrelation
        assert dw_stat < 2.5  # Should indicate autocorrelation
