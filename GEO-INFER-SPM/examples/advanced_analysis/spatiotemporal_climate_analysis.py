#!/usr/bin/env python3
"""
Advanced SPM Analysis: Spatio-Temporal Climate Change Detection

This example demonstrates advanced Statistical Parametric Mapping for detecting
climate change patterns across space and time. We'll analyze simulated
temperature data to identify regions with significant warming trends while
accounting for spatial autocorrelation and temporal dependencies.

The analysis incorporates:
- Spatio-temporal General Linear Modeling
- Random Field Theory for multiple comparison correction
- Cluster-based inference for regional patterns
- Bayesian uncertainty quantification
- Interactive visualization of results

This showcases the full power of SPM for complex geospatial time series analysis.
"""

import numpy as np
import warnings

# Import GEO-INFER-SPM modules
from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.core.glm import fit_glm
from geo_infer_spm.core.contrasts import contrast
from geo_infer_spm.core.rft import compute_spm, RandomFieldTheory
from geo_infer_spm.core.spatial_analysis import SpatialAnalyzer
from geo_infer_spm.core.temporal_analysis import TemporalAnalyzer
from geo_infer_spm.core.bayesian import BayesianSPM
from geo_infer_spm.utils.preprocessing import preprocess_data
from geo_infer_spm.visualization.interactive import create_interactive_map

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)


def generate_climate_data(n_years=30, n_stations=150):
    """
    Generate realistic climate monitoring data with spatio-temporal patterns.

    Parameters:
        n_years: Number of years of data
        n_stations: Number of monitoring stations

    Returns:
        SPMData with temperature time series
    """
    print(f"Generating {n_years} years of climate data from {n_stations} stations...")

    # Create spatial grid (climate stations)
    coordinates = np.random.rand(n_stations, 2) * 2000  # 2000x2000 km area

    # Initialize temperature data (n_years x n_stations)
    temperature_data = np.zeros((n_years, n_stations))

    # Base temperature (spatial variation)
    base_temp = 15 - 0.005 * coordinates[:, 1]  # Cooler in north
    base_temp += 0.002 * coordinates[:, 0]  # Warmer in east (continental effect)

    # Temporal patterns
    years = np.arange(n_years) + 1990

    for year_idx, year in enumerate(years):
        # Global warming trend (+0.02°C per year)
        warming_trend = 0.02 * (year - 1990)

        # Spatial variation in warming rate
        # Faster warming in northern latitudes
        lat_effect = 0.001 * coordinates[:, 1] * (year - 1990)

        # Regional climate patterns
        # Create some spatial clusters with different warming rates
        cluster_centers = np.array(
            [
                [500, 1500],  # Northern cluster (fast warming)
                [1500, 500],  # Southern cluster (slow warming)
                [1200, 1200],  # Central cluster (moderate warming)
            ]
        )

        regional_warming = np.zeros(n_stations)
        for center in cluster_centers:
            distances = np.linalg.norm(coordinates - center, axis=1)
            # Exponential decay from cluster centers
            regional_warming += 0.01 * np.exp(-distances / 300) * (year - 1990)

        # Seasonal variation (simplified)
        seasonal_temp = 5 * np.sin(2 * np.pi * (year_idx / n_years))

        # Combine all effects
        annual_temp = (
            base_temp + warming_trend + lat_effect + regional_warming + seasonal_temp
        )

        # Add realistic noise (measurement error + weather variability)
        noise = np.random.normal(0, 1.5, n_stations)

        temperature_data[year_idx] = annual_temp + noise

    # Create covariates
    covariates = {
        "latitude": coordinates[:, 1],
        "longitude": coordinates[:, 0],
        "elevation": np.random.normal(300, 150, n_stations),  # Simulated elevation
        "urbanization": np.random.beta(2, 5, n_stations),  # Urbanization index
    }

    # Create SPMData
    climate_data = SPMData(
        data=temperature_data,
        coordinates=coordinates,
        time=years,
        covariates=covariates,
        metadata={
            "data_type": "climate_temperature",
            "units": "celsius",
            "temporal_resolution": "annual",
            "spatial_resolution": "station_network",
            "n_years": n_years,
            "n_stations": n_stations,
        },
    )

    return climate_data


def main():
    """Run advanced spatio-temporal climate analysis."""

    print("=== Advanced SPM Analysis: Climate Change Detection ===\n")

    # Step 1: Generate synthetic climate data
    climate_data = generate_climate_data(n_years=25, n_stations=120)

    print(
        f"Data generated: {climate_data.data.shape[0]} years × {climate_data.data.shape[1]} stations"
    )
    print(
        f"Temperature range: {np.min(climate_data.data):.1f} to "
        f"{np.max(climate_data.data):.1f} °C"
    )
    print()

    # Step 2: Preprocess data
    print("2. Preprocessing climate data...")

    processed_data = preprocess_data(
        climate_data,
        steps=["validate", "temporal_filter", "spatial_filter"],
        temporal_params={"method": "moving_average", "window_size": 3},
        spatial_params={"method": "gaussian", "sigma": 50.0},
    )

    print("   ✓ Data validation completed")
    print("   ✓ Temporal smoothing applied (3-year moving average)")
    print("   ✓ Spatial smoothing applied (50km Gaussian filter)")
    print()

    # Step 3: Create spatio-temporal design matrix
    print("3. Creating spatio-temporal design matrix...")

    # Design includes the latitude, longitude, and urbanization regressors;
    # temporal trends are tested separately in step 9.

    # Spatial regressors
    spatial_regressors = np.column_stack(
        [
            processed_data.covariates["latitude"] / 1000,  # Latitude (scaled)
            processed_data.covariates["longitude"] / 1000,  # Longitude (scaled)
            processed_data.covariates["urbanization"],  # Urban effect
        ]
    )

    design_matrix = DesignMatrix(
        matrix=spatial_regressors, names=["latitude", "longitude", "urbanization"]
    )

    print(f"   Design matrix: {design_matrix.matrix.shape}")
    print(f"   Regressors: {design_matrix.names}")
    print()

    # Step 4: Fit GLM with spatial regularization
    print("4. Fitting GLM with spatial regularization...")

    spm_result = fit_glm(
        processed_data,
        design_matrix,
        method="spatial",
        spatial_regularization={
            "lambda": 0.1,
            "spatial_weights": None,  # Will use distance-based weights
        },
    )

    print(
        f"   Model R²: {spm_result.model_diagnostics.get('r_squared', float('nan')):.3f}"
    )
    print("   Spatial regularization: λ = 0.1")
    print()

    # Step 5: Test climate change hypotheses
    print("5. Testing climate change hypotheses...")

    # Test for latitudinal effect on temperature
    latitude_contrast = contrast(spm_result, "latitude")
    print(
        f"   Latitude mean t-statistic: {float(np.mean(latitude_contrast.t_statistic)):.3f}"
    )

    # Test for urbanization effect
    urban_contrast = contrast(spm_result, "urbanization")
    print(
        f"   Urbanization mean t-statistic: {float(np.mean(urban_contrast.t_statistic)):.3f}"
    )

    print()

    # Step 6: Apply Random Field Theory correction
    print("6. Applying Random Field Theory correction...")

    # Initialize RFT for 2D spatial field
    field_shape = (
        int(np.sqrt(processed_data.n_points)),
        int(np.sqrt(processed_data.n_points)),
    )
    rft = RandomFieldTheory(field_shape)

    # Estimate spatial smoothness
    smoothness = rft.estimate_smoothness(spm_result.residuals.reshape(field_shape))
    print(f"   Estimated FWHM per axis: {np.round(smoothness, 2).tolist()}")

    # Compute search volume
    search_volume = rft.compute_search_volume()
    print(f"   Search volume: {search_volume:.1f} resels")

    # Apply RFT correction
    rft_corrected_lat = compute_spm(spm_result, latitude_contrast, correction="RFT")
    rft_corrected_urban = compute_spm(spm_result, urban_contrast, correction="RFT")

    print(
        f"   RFT-corrected latitude effect significant: {rft_corrected_lat.n_significant > 0}"
    )
    print(
        f"   RFT-corrected urban effect significant: {rft_corrected_urban.n_significant > 0}"
    )
    print()

    # Step 7: Spatial analysis and clustering
    print("7. Performing spatial pattern analysis...")

    spatial_analyzer = SpatialAnalyzer(processed_data.coordinates)

    # Estimate variogram
    variogram = spatial_analyzer.estimate_variogram(spm_result.residuals)
    print(f"   Variogram sill: {float(variogram['sill']):.2f}")
    print(f"   Variogram range: {float(variogram['range']):.1f}")

    # Detect significant clusters
    clusters = spatial_analyzer.detect_clusters(
        latitude_contrast.t_statistic.reshape(field_shape),
        threshold=2.0,  # Approximate 2 SD threshold
        min_cluster_size=5,
    )

    print(f"   Detected {clusters['n_clusters']} significant clusters")
    if clusters["n_clusters"] > 0:
        largest_cluster = max(clusters["clusters"], key=lambda x: x["size"])
        print(f"   Largest cluster: {largest_cluster['size']} stations")
    print()

    # Step 8: Bayesian analysis (if available)
    print("8. Bayesian uncertainty quantification...")

    try:
        bayesian_spm = BayesianSPM()

        # Uses PyMC >= 5 when installed, otherwise empirical Bayes
        bayesian_result = bayesian_spm.fit_bayesian_glm(
            processed_data, design_matrix.matrix, n_samples=500, n_tune=200
        )

        print("   ✓ Bayesian GLM fitted")
        print(
            "   Mean posterior coefficient: "
            f"{float(np.mean(bayesian_result.beta_coefficients)):.3f}"
        )

    except Exception as e:
        print(f"   Note: Bayesian analysis skipped ({e})")

    print()

    # Step 9: Temporal trend analysis
    print("9. Analyzing temporal trends...")

    temporal_analyzer = TemporalAnalyzer(processed_data.time)

    # Test for temporal trends
    trends = temporal_analyzer.detect_trends(
        processed_data.data.T,  # Transpose for (n_stations, n_years)
        method="mann_kendall",
        alpha=0.05,
    )

    significant_trends = sum(1 for t in trends["trends"] if t["significant"])
    print(
        f"   Stations with significant trends: {significant_trends}/{len(trends['trends'])}"
    )

    # Seasonal decomposition (simplified annual data)
    if len(processed_data.time) >= 5:  # Need minimum data for decomposition
        try:
            decomposition = temporal_analyzer.seasonal_decomposition(
                processed_data.data.mean(axis=1),  # Mean across stations
                period=3,  # Simplified for annual data
            )
            print(f"   ✓ Seasonal decomposition completed: {sorted(decomposition)}")
        except Exception:
            print("   Note: Seasonal decomposition skipped (insufficient data)")

    print()

    # Step 10: Create visualizations
    print("10. Generating statistical maps...")

    # Create interactive statistical map
    try:
        interactive_map = create_interactive_map(
            spm_result,
            contrast_idx=0,  # latitude effect
            map_type="scattergeo",
        )

        if interactive_map:
            print("   ✓ Interactive statistical map created")
            print("   Note: Use interactive_map.show() to display in Jupyter")
        else:
            print("   Note: Interactive visualization not available")

    except Exception as e:
        print(f"   Note: Interactive map creation failed ({e})")

    # Step 11: Results summary
    print("\n=== CLIMATE CHANGE ANALYSIS RESULTS ===")
    print("Spatial Patterns:")
    print(
        f"  • Latitudinal temperature gradient: {'DETECTED' if rft_corrected_lat.n_significant > 0 else 'NOT DETECTED'}"
    )
    print(
        f"  • Urban heat island effect: {'DETECTED' if rft_corrected_urban.n_significant > 0 else 'NOT DETECTED'}"
    )
    print(f"  • Significant spatial clusters: {clusters['n_clusters']}")

    print("\nTemporal Patterns:")
    print(f"  • Stations with significant trends: {significant_trends}")
    print(
        f"  • Fraction of stations trending: {significant_trends / len(trends['trends']):.3f}"
    )

    print("\nModel Performance:")
    print(
        f"  • Model R²: {spm_result.model_diagnostics.get('r_squared', float('nan')):.3f}"
    )
    print(f"  • Spatial autocorrelation range: {variogram.get('range', 'N/A'):.1f} km")

    print("\n=== ANALYSIS COMPLETE ===")
    print("This advanced analysis demonstrates SPM's capability for")
    print("detecting complex spatio-temporal patterns in climate data")
    print("while properly controlling for spatial and temporal dependencies.")

    return {
        "spm_result": spm_result,
        "rft_results": {"latitude": rft_corrected_lat, "urban": rft_corrected_urban},
        "clusters": clusters,
        "variogram": variogram,
        "trends": trends,
    }


if __name__ == "__main__":
    # Run the advanced analysis
    results = main()

    # Optionally save comprehensive results
    save_results = input("\nSave comprehensive results? (y/n): ").lower().strip()
    if save_results == "y":
        from geo_infer_spm.utils.data_io import save_spm

        output_file = "advanced_climate_analysis_results.json"
        save_spm(results["spm_result"], output_file, format="json")
        print(f"Results saved to {output_file}")

        # Save additional analysis results
        import json

        extended_results = {
            "rft_corrected_latitude_significant": results["rft_results"][
                "latitude"
            ].n_significant
            > 0,
            "rft_corrected_urban_significant": results["rft_results"][
                "urban"
            ].n_significant
            > 0,
            "n_clusters": results["clusters"]["n_clusters"],
            "variogram_range": results["variogram"].get("range"),
            "stations_with_trends": sum(
                1 for t in results["trends"]["trends"] if t["significant"]
            ),
        }

        with open("climate_analysis_summary.json", "w") as f:
            json.dump(extended_results, f, indent=2)

        print("Extended analysis summary saved to climate_analysis_summary.json")
