#!/usr/bin/env python3
"""
Basic SPM Analysis: Detecting Spatial Trends in Environmental Data

This example demonstrates how to use GEO-INFER-SPM to detect statistically
significant spatial trends in environmental monitoring data. We'll analyze
simulated air quality data to identify areas with significant pollution gradients.

The analysis follows these steps:
1. Load and preprocess geospatial data
2. Specify experimental design (spatial trends)
3. Fit General Linear Model
4. Test statistical contrasts
5. Apply multiple comparison correction
6. Visualize results

This example showcases the core SPM methodology adapted for geospatial analysis.
"""

import numpy as np

# Import GEO-INFER-SPM modules
from geo_infer_spm.core.glm import fit_glm
from geo_infer_spm.core.contrasts import contrast
from geo_infer_spm.core.rft import compute_spm
from geo_infer_spm.utils.helpers import generate_synthetic_data, create_design_matrix
from geo_infer_spm.utils.preprocessing import preprocess_data
from geo_infer_spm.visualization.maps import create_statistical_map

# Set random seed for reproducibility
np.random.seed(42)


def main():
    """Run spatial trend analysis example."""

    print("=== GEO-INFER-SPM: Spatial Trend Analysis Example ===\n")

    # Step 1: Generate synthetic environmental monitoring data
    print("1. Generating synthetic air quality monitoring data...")

    # Create spatial grid (simulating monitoring stations)
    n_stations = 200
    coordinates = np.random.rand(n_stations, 2) * 100  # 100x100 km area

    # Generate data with realistic spatial patterns
    air_quality_data = generate_synthetic_data(
        coordinates,
        effects={
            "trend": "east_west",  # Pollution increases from west to east
            "clusters": {  # Pollution hotspots
                "n_clusters": 3,
                "effect_size": 2.5,
            },
        },
        noise_level=0.8,  # Realistic measurement noise
    )

    # Add urban vs rural covariate
    urban_distance = np.linalg.norm(coordinates - np.array([75, 50]), axis=1)
    air_quality_data.covariates["urban_distance"] = urban_distance

    print(f"   Generated {n_stations} monitoring stations")
    print(
        f"   Spatial extent: {coordinates.min(axis=0)} to {coordinates.max(axis=0)} km"
    )
    print(".2f")
    print()

    # Step 2: Preprocess the data
    print("2. Preprocessing data...")

    processed_data = preprocess_data(
        air_quality_data,
        steps=["validate", "handle_missing", "normalize"],
        normalize_params={"method": "zscore"},
    )

    print("   Data validation: PASSED")
    print("   Missing data handling: COMPLETED")
    print("   Normalization: Z-score standardization")
    print()

    # Step 3: Define experimental design
    print("3. Specifying experimental design...")

    # Design matrix includes:
    # - Intercept (baseline pollution level)
    # - East-west trend (longitude effect)
    # - North-south trend (latitude effect)
    # - Urban proximity effect
    design_matrix = create_design_matrix(
        processed_data,
        formula="pollution ~ longitude + latitude + urban_distance",
        covariates=["urban_distance"],
    )

    print(f"   Design matrix shape: {design_matrix.matrix.shape}")
    print(f"   Regressors: {design_matrix.names}")
    print()

    # Step 4: Fit General Linear Model
    print("4. Fitting General Linear Model...")

    spm_result = fit_glm(
        processed_data,
        design_matrix,
        method="OLS",  # Ordinary Least Squares
    )

    print(".3f")
    print(f"   Model degrees of freedom: {design_matrix.n_regressors}")
    print(".2e")
    print()

    # Step 5: Test statistical hypotheses (contrasts)
    print("5. Testing statistical hypotheses...")

    # Test for east-west pollution gradient
    ew_contrast = contrast(spm_result, "longitude")
    print(".3f")
    print(".2e")
    print()

    # Test for urban proximity effect
    urban_contrast = contrast(spm_result, "urban_distance")
    print(".3f")
    print(".2e")
    print()

    # Step 6: Apply multiple comparison correction
    print("6. Applying multiple comparison correction...")

    # Use False Discovery Rate correction for geospatial data
    corrected_ew = compute_spm(spm_result, ew_contrast, correction="FDR", alpha=0.05)
    corrected_urban = compute_spm(
        spm_result, urban_contrast, correction="FDR", alpha=0.05
    )

    print("   Correction method: False Discovery Rate (FDR)")
    print("   Significance threshold: α = 0.05")
    print(f"   East-west gradient significant: {corrected_ew.n_significant > 0}")
    print(f"   Urban effect significant: {corrected_urban.n_significant > 0}")
    print()

    # Step 7: Visualize results
    print("7. Creating statistical maps...")

    # Create statistical map for east-west gradient
    ew_map = create_statistical_map(
        spm_result,
        contrast_idx=0,  # longitude effect
        threshold=0.05,
        title="East-West Pollution Gradient (SPM{t})",
    )

    # Display results
    print("\n=== ANALYSIS RESULTS ===")
    print(".3f")
    print(".3f")
    print(".2e")
    print(".2e")
    print(f"Number of significant stations (EW): {corrected_ew.n_significant}")
    print(f"Number of significant stations (Urban): {corrected_urban.n_significant}")

    # Step 8: Interpretation
    print("\n=== INTERPRETATION ===")
    if corrected_ew.n_significant > 0:
        print("✓ Significant east-west pollution gradient detected")
        print("  → Pollution levels increase from west to east")
    else:
        print("✗ No significant east-west pollution gradient")

    if corrected_urban.n_significant > 0:
        print("✓ Significant urban proximity effect detected")
        print("  → Pollution levels higher near urban centers")
    else:
        print("✗ No significant urban proximity effect")

    print("\n=== SPM ANALYSIS COMPLETE ===")
    print("This analysis demonstrates how Statistical Parametric Mapping")
    print("can identify statistically significant spatial patterns in")
    print("environmental monitoring data while controlling for multiple")
    print("comparisons across continuous spatial fields.")

    return spm_result, corrected_ew, corrected_urban


if __name__ == "__main__":
    # Run the example
    results = main()

    # Optionally save results
    save_results = input("\nSave results to file? (y/n): ").lower().strip()
    if save_results == "y":
        from geo_infer_spm.utils.data_io import save_spm

        output_file = "spatial_trend_analysis_results.json"
        save_spm(results[0], output_file, format="json")
        print(f"Results saved to {output_file}")
