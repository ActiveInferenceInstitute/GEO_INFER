"""
Integration tests for complete SPM analysis pipeline
"""

import numpy as np
import tempfile
import json

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.core.glm import fit_glm
from geo_infer_spm.core.contrasts import contrast
from geo_infer_spm.core.rft import compute_spm
from geo_infer_spm.utils.data_io import load_data, save_spm
from geo_infer_spm.utils.preprocessing import preprocess_data
from geo_infer_spm.utils.helpers import generate_synthetic_data, create_design_matrix


class TestFullSPMPipeline:
    """Test complete SPM analysis pipeline."""

    def setup_method(self):
        """Set up synthetic geospatial data for testing."""
        np.random.seed(42)

        # Generate synthetic geospatial data
        self.coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 200), np.random.uniform(-89, 89, 200)]
        )  # 200 spatial points
        self.spm_data = generate_synthetic_data(
            self.coordinates,
            effects={
                "trend": "east_west",
                "clusters": {"n_clusters": 3, "effect_size": 2.0},
            },
            noise_level=0.5,
        )

    def test_complete_spatial_analysis_pipeline(self):
        """Test complete spatial SPM analysis pipeline."""
        # Step 1: Preprocess data
        processed_data = preprocess_data(self.spm_data, steps=["validate", "normalize"])

        # Step 2: Create design matrix
        design_matrix = create_design_matrix(
            processed_data, covariates=["elevation"], intercept=True
        )

        # Step 3: Fit GLM
        spm_result = fit_glm(processed_data, design_matrix, method="OLS")

        # Verify GLM fit
        assert spm_result.model_diagnostics["r_squared"] > 0.5
        assert spm_result.beta_coefficients.shape[0] == design_matrix.n_regressors

        # Step 4: Define and compute contrasts
        elevation_contrast = contrast(spm_result, "elevation")
        assert elevation_contrast.p_values < 0.05  # Should be significant

        # Step 5: Apply multiple comparison correction
        corrected_result = compute_spm(spm_result, elevation_contrast, correction="FDR")

        # Verify correction
        assert hasattr(corrected_result, "corrected_p_values")
        assert corrected_result.correction_method == "FDR"

        # Step 6: Save results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_spm(spm_result, temp_path, format="json")

            # Verify save worked
            with open(temp_path) as f:
                saved_data = json.load(f)
            assert "beta_coefficients" in saved_data

        finally:
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_temporal_analysis_pipeline(self):
        """Test temporal SPM analysis pipeline."""
        # Create temporal data
        n_timepoints = 50
        time_coords = np.arange(n_timepoints)

        # Generate temporal data with trend
        temporal_data = np.zeros((n_timepoints, 50))
        for t in range(n_timepoints):
            temporal_data[t] = (
                10 + 0.2 * t + np.random.normal(0, 1, 50)
            )  # Trend + noise

        # Create SPMData with temporal dimension
        temporal_spm_data = SPMData(
            data=temporal_data, coordinates=self.coordinates, time=time_coords
        )

        # Create design matrix with temporal trend
        design_matrix = DesignMatrix(
            matrix=np.column_stack(
                [
                    np.ones(n_timepoints),  # intercept
                    time_coords,  # linear trend
                    time_coords**2,  # quadratic trend
                ]
            ),
            names=["intercept", "linear_trend", "quadratic_trend"],
        )

        # Fit GLM
        spm_result = fit_glm(temporal_spm_data, design_matrix, method="OLS")

        # Test trend contrast
        trend_contrast = contrast(spm_result, "linear_trend")
        assert trend_contrast.p_values < 0.01  # Strong trend should be significant

    def test_spatial_temporal_analysis(self):
        """Test spatio-temporal analysis pipeline."""
        # Create spatio-temporal data
        n_timepoints = 20
        n_spatial = 100

        spatiotemporal_data = np.random.randn(n_timepoints, n_spatial)

        # Add spatial pattern that changes over time
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_spatial),
                np.random.uniform(-89, 89, n_spatial),
            ]
        )
        for t in range(n_timepoints):
            # East-west gradient that strengthens over time
            east_west_effect = (coordinates[:, 0] / 100) * (t / n_timepoints)
            spatiotemporal_data[t] += east_west_effect

        # Create SPMData
        st_spm_data = SPMData(
            data=spatiotemporal_data,
            coordinates=coordinates,
            time=np.arange(n_timepoints),
        )

        # This would typically involve more complex spatio-temporal modeling
        # For now, just verify data structure
        assert st_spm_data.has_temporal
        assert st_spm_data.data.shape[0] == n_timepoints
        assert st_spm_data.data.shape[1] == n_spatial

    def test_data_io_pipeline(self):
        """Test data loading and saving pipeline."""
        # Create test data
        test_data = {
            "data": self.spm_data.data.tolist(),
            "coordinates": self.spm_data.coordinates.tolist(),
            "covariates": {k: v.tolist() for k, v in self.spm_data.covariates.items()},
            "metadata": {"test": True, "pipeline": "integration_test"},
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Load data
            loaded_data = load_data(temp_path)

            # Verify data integrity
            np.testing.assert_array_equal(loaded_data.data, self.spm_data.data)
            np.testing.assert_array_equal(
                loaded_data.coordinates, self.spm_data.coordinates
            )
            assert loaded_data.metadata["test"] == True

            # Run analysis on loaded data
            design_matrix = create_design_matrix(loaded_data, covariates=["elevation"])
            result = fit_glm(loaded_data, design_matrix)

            # Verify analysis worked
            assert result.model_diagnostics["r_squared"] > 0.5

        finally:
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_robustness_to_missing_data(self):
        """Test pipeline robustness to missing data."""
        # Introduce missing data
        data_with_missing = self.spm_data.data.copy()
        missing_indices = np.random.choice(
            len(data_with_missing), size=10, replace=False
        )
        data_with_missing[missing_indices] = np.nan

        spm_data_missing = SPMData(
            data=data_with_missing,
            coordinates=self.spm_data.coordinates,
            covariates=self.spm_data.covariates,
        )

        # Preprocess to handle missing data
        processed_data = preprocess_data(
            spm_data_missing,
            steps=["handle_missing", "validate"],
            missing_params={"method": "interpolate"},
        )

        # Verify no NaN values remain
        assert not np.any(np.isnan(processed_data.data))

        # Run analysis
        design_matrix = create_design_matrix(processed_data, covariates=["elevation"])
        result = fit_glm(processed_data, design_matrix)

        # Should still produce valid results
        assert result.model_diagnostics["r_squared"] > 0.4

    def test_multiple_contrast_analysis(self):
        """Test analysis with multiple contrasts."""
        # Fit model
        design_matrix = create_design_matrix(
            self.spm_data, covariates=["elevation", "temperature"], intercept=True
        )

        spm_result = fit_glm(self.spm_data, design_matrix)

        # Define multiple contrasts
        contrasts_to_test = [
            "elevation",  # Single covariate effect
            "temperature",  # Another covariate effect
            ["intercept", "elevation"],  # Custom vector contrast
        ]

        results = []
        for contrast_spec in contrasts_to_test:
            contrast_result = contrast(spm_result, contrast_spec)
            corrected_result = compute_spm(
                spm_result, contrast_result, correction="FDR"
            )
            results.append(corrected_result)

        # Verify all results are valid
        for result in results:
            assert hasattr(result, "corrected_p_values")
            assert hasattr(result, "significance_mask")

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger synthetic dataset
        n_points_large = 1000
        coordinates_large = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points_large),
                np.random.uniform(-89, 89, n_points_large),
            ]
        )

        large_spm_data = generate_synthetic_data(
            coordinates_large, effects={"trend": "east_west"}, noise_level=0.3
        )

        # Create design matrix
        design_matrix = create_design_matrix(large_spm_data, covariates=["elevation"])

        # Fit model (should complete in reasonable time)
        import time

        start_time = time.time()
        spm_result = fit_glm(large_spm_data, design_matrix, method="OLS")
        end_time = time.time()

        # Verify results
        assert spm_result.model_diagnostics["r_squared"] > 0.5
        assert end_time - start_time < 10  # Should complete within 10 seconds
