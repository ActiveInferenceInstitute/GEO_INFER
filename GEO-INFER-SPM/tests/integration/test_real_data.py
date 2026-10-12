"""
Integration tests with real geospatial data formats

This module tests the SPM pipeline with various real-world geospatial
data formats and sources to ensure robust handling of diverse inputs.
"""

import numpy as np
import pytest
import tempfile
import json
import os

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.core.glm import fit_glm
from geo_infer_spm.core.spatial_analysis import SpatialAnalyzer
from geo_infer_spm.utils.data_io import load_data, save_spm
from geo_infer_spm.utils.preprocessing import preprocess_data


class TestGeospatialDataFormats:
    """Test loading and processing various geospatial data formats."""

    def test_json_data_integration(self):
        """Test complete pipeline with JSON data."""
        # Create realistic JSON data
        n_points = 100
        coordinates = (
            np.column_stack(
                [
                    np.random.uniform(-179, 179, n_points),
                    np.random.uniform(-89, 89, n_points),
                ]
            )
            - 90
        )  # Global extent
        data = np.random.normal(25, 5, n_points)  # Temperature-like data

        json_data = {
            "data": data.tolist(),
            "coordinates": coordinates.tolist(),
            "metadata": {
                "data_type": "temperature",
                "units": "celsius",
                "source": "synthetic_weather_stations",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            # Load data
            spm_data = load_data(temp_path)

            # Validate loaded data
            assert len(spm_data.data) == n_points
            assert spm_data.coordinates.shape == (n_points, 2)

            # Preprocess
            processed = preprocess_data(spm_data, steps=["validate", "normalize"])

            # Create design matrix
            from geo_infer_spm.utils.helpers import create_design_matrix

            design = create_design_matrix(processed, covariates=[])

            # Fit model
            result = fit_glm(processed, design)

            # Verify results
            assert result.model_diagnostics["r_squared"] >= 0
            assert len(result.beta_coefficients) == 1  # Just intercept

        finally:
            os.unlink(temp_path)

    def test_csv_data_integration(self):
        """Test complete pipeline with CSV data."""
        import pandas as pd

        n_points = 50
        df_data = {
            "longitude": np.random.uniform(-180, 180, n_points),
            "latitude": np.random.uniform(-90, 90, n_points),
            "pollution_level": np.random.normal(50, 15, n_points),
            "population_density": np.random.normal(1000, 500, n_points),
            "industrial_activity": np.random.choice([0, 1], n_points),
        }

        df = pd.DataFrame(df_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load CSV data
            spm_data = load_data(temp_path)

            # Verify data loading
            assert len(spm_data.data) == n_points
            assert "population_density" in spm_data.covariates
            assert "industrial_activity" in spm_data.covariates

            # Preprocess
            processed = preprocess_data(spm_data, steps=["validate", "normalize"])

            # Create design matrix with covariates
            from geo_infer_spm.utils.helpers import create_design_matrix

            design = create_design_matrix(
                processed, covariates=["population_density", "industrial_activity"]
            )

            # Fit model
            result = fit_glm(processed, design)

            # Should find relationships
            assert result.model_diagnostics["r_squared"] >= 0

        finally:
            os.unlink(temp_path)


class TestSpatialDataIntegration:
    """Test integration with spatial analysis on real data patterns."""

    def test_spatial_autocorrelation_workflow(self):
        """Test spatial autocorrelation analysis workflow."""
        # Create data with known spatial patterns
        n_points = 200
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Create spatially correlated data
        distances = np.linalg.norm(
            coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2
        )
        spatial_covariance = np.exp(-distances / 50)  # Exponential decay

        # Generate correlated response
        L = np.linalg.cholesky(spatial_covariance + 0.1 * np.eye(n_points))
        spatial_effect = L @ np.random.randn(n_points)

        X = np.random.randn(n_points, 2)
        y = (
            X @ np.array([1.0, -0.5])
            + spatial_effect * 0.5
            + 0.1 * np.random.randn(n_points)
        )

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["int", "x"])

        # Fit GLM
        result = fit_glm(spm_data, design_matrix)

        # Analyze residuals for spatial autocorrelation
        analyzer = SpatialAnalyzer(coordinates)
        variogram = analyzer.estimate_variogram(result.residuals)

        # Should detect spatial structure in residuals
        assert (
            variogram["variogram"][-1] > variogram["variogram"][0]
        )  # Increases with distance

        # Test cluster detection
        clusters = analyzer.detect_clusters(result.residuals, threshold=2.0)
        assert "n_clusters" in clusters

    def test_geographically_weighted_analysis(self):
        """Test GWR on spatially varying relationships."""
        n_points = 150
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )

        # Create spatially varying relationships
        x_coord = coordinates[:, 0]
        y_coord = coordinates[:, 1]

        # Coefficients vary spatially
        beta0 = 10 + 0.01 * x_coord  # Intercept increases east
        beta1 = 2.0 - 0.005 * y_coord  # Slope decreases north

        X = np.random.randn(n_points, 1)
        y = beta0 + beta1 * X.flatten() + 0.2 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(
            matrix=np.column_stack([np.ones(n_points), X.flatten()]), names=["int", "x"]
        )

        # Fit GWR
        analyzer = SpatialAnalyzer(coordinates)
        gwr_result = analyzer.geographically_weighted_regression(
            spm_data, design_matrix, bandwidth=30.0
        )

        # Should capture spatial variation
        assert "local_coefficients" in gwr_result.model_diagnostics
        local_coeffs = gwr_result.model_diagnostics["local_coefficients"]

        # Coefficients should vary across space
        coeff_variation = np.std(local_coeffs, axis=0)
        assert np.any(coeff_variation > 0.1)  # Some variation detected


class TestTemporalDataIntegration:
    """Test integration with temporal data patterns."""

    def test_time_series_spm_workflow(self):
        """Test SPM analysis on temporal data."""
        n_stations, n_years = 20, 15
        time_points = np.arange(n_years)

        # Create spatial locations
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_stations),
                np.random.uniform(-89, 89, n_stations),
            ]
        )

        # Create spatio-temporal data
        spatiotemporal_data = np.zeros((n_years, n_stations))

        for year_idx, year in enumerate(time_points):
            # Global trend
            trend = 0.1 * year

            # Spatial variation
            lat_effect = coordinates[:, 1] / 500 * 2  # Northern stations warmer
            lon_effect = coordinates[:, 0] / 500 * 1  # Eastern stations warmer

            # Seasonal variation
            seasonal = 3 * np.sin(2 * np.pi * year / 5)

            # Station-specific noise
            noise = 0.5 * np.random.randn(n_stations)

            spatiotemporal_data[year_idx] = (
                trend + lat_effect + lon_effect + seasonal + noise
            )

        # Create SPMData with temporal dimension
        spm_data = SPMData(
            data=spatiotemporal_data,
            coordinates=coordinates,
            time=time_points,
            crs="EPSG:4326",
        )

        # Create design matrix for temporal analysis
        # Include time, latitude, longitude effects
        n_total = n_years * n_stations
        time_design = np.repeat(time_points, n_stations)
        lat_design = np.tile(coordinates[:, 1], n_years)
        lon_design = np.tile(coordinates[:, 0], n_years)

        X = np.column_stack(
            [
                np.ones(n_total),  # intercept
                time_design,  # temporal trend
                lat_design,  # latitude effect
                lon_design,  # longitude effect
                np.sin(2 * np.pi * time_design / 5),
                np.cos(2 * np.pi * time_design / 5),
            ]
        )

        design_matrix = DesignMatrix(
            matrix=X, names=["int", "time", "lat", "lon", "season_sin", "season_cos"]
        )

        # Fit spatio-temporal model
        result = fit_glm(spm_data, design_matrix)

        # Should detect temporal trend
        time_coefficient = result.beta_coefficients[1]  # time coefficient
        assert abs(time_coefficient - 0.1) < 0.05  # Should recover true trend

        # Should detect spatial effects
        lat_coefficient = result.beta_coefficients[2]
        lon_coefficient = result.beta_coefficients[3]

        assert abs(lat_coefficient) > 0.001  # Should detect latitude effect
        assert abs(lon_coefficient) > 0.001  # Should detect longitude effect

    def test_seasonal_temporal_analysis(self):
        """Test seasonal analysis in temporal SPM."""
        from geo_infer_spm.core.temporal_analysis import TemporalAnalyzer

        # Create seasonal time series
        n_years = 20
        time_points = np.arange(n_years)
        seasonal_data = (
            10 + 5 * np.sin(2 * np.pi * time_points / 5) + np.random.randn(n_years)
        )

        # Create minimal spatial data
        coordinates = np.array([[0.0, 0.0]])
        SPMData(
            data=seasonal_data,
            coordinates=coordinates,
            time=time_points,
            crs="EPSG:4326",
        )

        # Test temporal analysis
        analyzer = TemporalAnalyzer(time_points, seasonal_data)

        try:
            # Seasonal decomposition
            decomposition = analyzer.seasonal_decomposition(seasonal_data, period=5)

            assert "trend" in decomposition
            assert "seasonal" in decomposition
            assert "residual" in decomposition

            # Seasonal component should capture the sinusoidal pattern
            seasonal_amplitude = np.std(decomposition["seasonal"])
            assert seasonal_amplitude > 2.0

        except ImportError:
            pytest.fail("Statsmodels not available for seasonal decomposition")


class TestLargeScaleDataHandling:
    """Test handling of larger-scale geospatial datasets."""

    def test_medium_scale_geospatial_analysis(self):
        """Test analysis on medium-scale geospatial data."""
        n_points = 1000  # Medium scale

        # Generate realistic geospatial data
        coordinates = np.column_stack(
            [
                np.random.uniform(-125, -65, n_points),  # Continental US longitude
                np.random.uniform(25, 50, n_points),  # Continental US latitude
            ]
        )

        # Simulate environmental variables
        elevation = np.random.normal(500, 200, n_points)
        temperature = 25 - 0.005 * elevation + np.random.normal(0, 3, n_points)
        precipitation = 800 + 0.001 * elevation + np.random.normal(0, 100, n_points)

        # Response variable (e.g., vegetation index)
        vegetation = (
            0.5
            + 0.002 * temperature
            - 0.00001 * precipitation
            + 0.1 * np.random.randn(n_points)
        )

        spm_data = SPMData(
            data=vegetation,
            coordinates=coordinates,
            covariates={
                "elevation": elevation,
                "temperature": temperature,
                "precipitation": precipitation,
            },
            crs="EPSG:4326",
        )

        # Preprocess
        processed = preprocess_data(spm_data, steps=["validate", "normalize"])

        # Create design matrix
        from geo_infer_spm.utils.helpers import create_design_matrix

        design = create_design_matrix(
            processed, covariates=["elevation", "temperature", "precipitation"]
        )

        # Fit model
        result = fit_glm(processed, design)

        # Should produce reasonable results
        assert result.model_diagnostics["r_squared"] >= 0.0
        assert len(result.beta_coefficients) == 4  # intercept + 3 covariates

        # Spatial analysis on residuals
        analyzer = SpatialAnalyzer(coordinates)
        variogram = analyzer.estimate_variogram(result.residuals, n_bins=10)

        assert len(variogram["variogram"]) == 10


class TestDataExportImportCycle:
    """Test complete data export/import cycle."""

    def test_json_export_import_cycle(self):
        """Test exporting results to JSON and re-importing."""
        # Create analysis results
        n_points = 50
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        X = np.random.randn(n_points, 2)
        y = X @ np.array([1.5, -0.8]) + 0.1 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["int", "slope"])

        result = fit_glm(spm_data, design_matrix)

        # Export to JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_spm(result, temp_path, format="json")

            # Verify file was created and contains expected data
            with open(temp_path) as f:
                saved_data = json.load(f)

            assert "beta_coefficients" in saved_data
            assert "residuals" in saved_data
            assert "model_diagnostics" in saved_data

            # Verify data integrity
            np.testing.assert_allclose(
                saved_data["beta_coefficients"], result.beta_coefficients.tolist()
            )

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_csv_export_cycle(self):
        """Test CSV export functionality."""
        # Create results with spatial coordinates
        n_points = 30
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_points),
                np.random.uniform(-89, 89, n_points),
            ]
        )
        data = np.random.randn(n_points)

        spm_data = SPMData(data=data, coordinates=coordinates, crs="EPSG:4326")
        X = np.random.randn(n_points, 1)
        design_matrix = DesignMatrix(matrix=X, names=["intercept"])

        result = fit_glm(spm_data, design_matrix)

        # Export to CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_spm(result, temp_path, format="csv")

            # Verify CSV structure
            import pandas as pd

            df = pd.read_csv(temp_path)

            assert "longitude" in df.columns
            assert "latitude" in df.columns
            assert "residuals" in df.columns
            assert len(df) == n_points

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestRealWorldScenarios:
    """Test scenarios inspired by real-world geospatial analysis."""

    def test_environmental_monitoring_scenario(self):
        """Test scenario: environmental monitoring network."""
        n_stations = 100

        # Realistic station locations (e.g., air quality monitors)
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_stations),
                np.random.uniform(-89, 89, n_stations),
            ]
        ) + np.array([-100, 30])  # Western US

        # Environmental variables
        urban_proximity = np.random.exponential(
            50, n_stations
        )  # Distance to urban center
        elevation = np.random.normal(800, 300, n_stations)
        wind_speed = np.random.normal(5, 2, n_stations)

        # Pollution response with realistic relationships
        baseline_pollution = 20
        urban_effect = 15 * np.exp(-urban_proximity / 20)  # Higher near urban areas
        elevation_effect = -0.01 * elevation  # Lower at higher elevations
        noise = np.random.normal(0, 3, n_stations)

        pollution = baseline_pollution + urban_effect + elevation_effect + noise

        spm_data = SPMData(
            data=pollution,
            coordinates=coordinates,
            covariates={
                "urban_proximity": urban_proximity,
                "elevation": elevation,
                "wind_speed": wind_speed,
            },
            crs="LOCAL",
        )

        # Preprocess
        processed = preprocess_data(spm_data, steps=["validate", "remove_outliers"])

        # Create design matrix
        from geo_infer_spm.utils.helpers import create_design_matrix

        design = create_design_matrix(
            processed, covariates=["urban_proximity", "elevation", "wind_speed"]
        )

        # Fit model
        result = fit_glm(processed, design)

        # Should detect urban proximity effect
        urban_coeff_idx = design.names.index("urban_proximity")
        urban_coefficient = result.beta_coefficients[urban_coeff_idx]

        # Urban proximity should have negative effect (closer = more pollution)
        assert urban_coefficient < 0

        # Model should explain reasonable variance
        assert result.model_diagnostics["r_squared"] > 0.3

    def test_agricultural_yield_scenario(self):
        """Test scenario: agricultural yield prediction."""
        n_fields = 150

        # Field locations
        coordinates = np.column_stack(
            [
                np.random.uniform(-179, 179, n_fields),
                np.random.uniform(-89, 89, n_fields),
            ]
        )

        # Agricultural variables
        soil_quality = np.random.beta(2, 2, n_fields) * 100  # Soil quality score
        irrigation = np.random.choice(
            [0, 1], n_fields, p=[0.3, 0.7]
        )  # Irrigation access
        fertilizer = np.random.normal(50, 20, n_fields)  # Fertilizer amount
        precipitation = np.random.normal(600, 150, n_fields)  # Annual precipitation

        # Yield response
        yield_response = (
            50  # baseline
            + 0.5 * soil_quality
            + 15 * irrigation
            + 0.2 * fertilizer
            + 0.05 * precipitation
            + 5 * np.random.randn(n_fields)
        )

        spm_data = SPMData(
            data=yield_response,
            coordinates=coordinates,
            covariates={
                "soil_quality": soil_quality,
                "irrigation": irrigation,
                "fertilizer": fertilizer,
                "precipitation": precipitation,
            },
            crs="EPSG:4326",
        )

        # Analyze
        from geo_infer_spm.utils.helpers import create_design_matrix

        design = create_design_matrix(
            spm_data,
            covariates=["soil_quality", "irrigation", "fertilizer", "precipitation"],
        )

        result = fit_glm(spm_data, design)

        # Should detect important agricultural factors
        irrigation_coeff = result.beta_coefficients[design.names.index("irrigation")]
        soil_coeff = result.beta_coefficients[design.names.index("soil_quality")]

        assert irrigation_coeff > 5  # Irrigation should have strong positive effect
        assert soil_coeff > 0.1  # Soil quality should help

        assert result.model_diagnostics["r_squared"] > 0.4
