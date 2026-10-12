#!/usr/bin/env python3
"""
Advanced Geospatial Analysis Example

This example demonstrates the comprehensive capabilities of GEO-INFER-MATH
by performing a complete spatial analysis workflow including:
- Coordinate transformations
- Spatial statistics and autocorrelation
- Interpolation and modeling
- Clustering and pattern analysis
- Parallel processing
- Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

# Import GEO-INFER-MATH modules
from geo_infer_math.core.spatial_statistics import (
    MoranI,
    getis_ord_g,
    local_indicators_spatial_association,
)
from geo_infer_math.core.interpolation import SpatialInterpolator
from geo_infer_math.core.geometry import haversine_distance
from geo_infer_math.core.transforms import (
    geographic_to_projected,
    CoordinateTransformer,
)
from geo_infer_math.models.regression import GeographicallyWeightedRegression
from geo_infer_math.models.clustering import SpatialKMeans
from geo_infer_math.utils.validation import validate_coordinates, validate_values_array
from geo_infer_math.utils.parallel import parallel_compute

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_points=200, seed=42):
    """Generate synthetic geospatial data with spatial patterns."""
    np.random.seed(seed)

    # Create base coordinates (longitude, latitude)
    # Focus on a region around Portland, OR
    center_lon, center_lat = -122.6765, 45.5231

    # Generate clustered points
    coordinates = []

    # Urban core cluster
    urban_core = np.random.normal(
        [center_lon, center_lat], [0.02, 0.02], (n_points // 3, 2)
    )
    coordinates.extend(urban_core)

    # Suburban cluster
    suburban = np.random.normal(
        [center_lon + 0.1, center_lat + 0.05], [0.03, 0.03], (n_points // 3, 2)
    )
    coordinates.extend(suburban)

    # Rural/outlier cluster
    rural = np.random.normal(
        [center_lon - 0.15, center_lat - 0.1], [0.04, 0.04], (n_points // 3, 2)
    )
    coordinates.extend(rural)

    coordinates = np.array(coordinates)

    # Generate values with spatial autocorrelation
    # Higher values in urban core, lower in rural areas
    values = np.zeros(n_points)

    for i, coord in enumerate(coordinates):
        # Distance from urban core
        dist_to_center = haversine_distance(coord[1], coord[0], center_lat, center_lon)

        # Base value decreases with distance
        base_value = 100 - dist_to_center * 10

        # Add some random variation
        values[i] = base_value + np.random.normal(0, 5)

        # Add spatial trend
        if i < n_points // 3:  # Urban core
            values[i] += 20
        elif i < 2 * n_points // 3:  # Suburban
            values[i] += 10

    return coordinates, values


def coordinate_transformation_analysis(coordinates):
    """Demonstrate coordinate transformations."""
    logger.info("Performing coordinate transformation analysis...")

    # Transform to UTM
    utm_coords = []
    for lon, lat in coordinates:
        utm_x, utm_y = geographic_to_projected(lon, lat, "utm")
        utm_coords.append([utm_x, utm_y])

    utm_coords = np.array(utm_coords)

    # Transform to Web Mercator
    transformer = CoordinateTransformer("EPSG:4326", "EPSG:3857")
    mercator_coords = transformer.transform_points(coordinates)

    return utm_coords, mercator_coords


def spatial_statistics_analysis(coordinates, values):
    """Perform comprehensive spatial statistics analysis."""
    logger.info("Performing spatial statistics analysis...")

    # Validate inputs
    validate_coordinates(coordinates)
    validate_values_array(values)

    # Create spatial weights matrix
    from geo_infer_math.core.linalg_tensor import MatrixOperations

    weights_matrix = MatrixOperations.spatial_weights_matrix(coordinates, k=8)

    # Global Moran's I
    moran = MoranI(weights_matrix)
    moran_result = moran.compute(values)

    logger.info("Computed global Moran's I metrics")

    # Local Indicators of Spatial Association (LISA)
    lisa_result = local_indicators_spatial_association(values, weights_matrix)

    # Hot spot analysis
    g_result = getis_ord_g(values, weights_matrix)

    return moran_result, lisa_result, g_result


def interpolation_analysis(coordinates, values, utm_coords):
    """Perform spatial interpolation analysis."""
    logger.info("Performing spatial interpolation analysis...")

    # Create prediction grid
    x_min, x_max = utm_coords[:, 0].min(), utm_coords[:, 0].max()
    y_min, y_max = utm_coords[:, 1].min(), utm_coords[:, 1].max()

    # Create regular grid
    grid_size = 50
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.flatten(), yy.flatten()])

    # Convert back to geographic for interpolation
    grid_geo = []
    transformer = CoordinateTransformer("UTM", "EPSG:4326")
    for point in grid_points:
        geo_point = transformer.transform_point(point)
        grid_geo.append(geo_point[:2])
    grid_geo = np.array(grid_geo)

    # Perform interpolation
    interpolator = SpatialInterpolator(method="idw", power=2)
    interpolator.fit(coordinates, values)
    interpolated_values = interpolator.predict(grid_geo)

    # Reshape for plotting
    interpolated_grid = interpolated_values.reshape(xx.shape)

    return xx, yy, interpolated_grid


def regression_analysis(coordinates, values):
    """Perform geographically weighted regression."""
    logger.info("Performing GWR analysis...")

    # Create synthetic independent variable
    X = np.random.randn(len(values), 1) * 10 + 50

    # Fit GWR model
    gwr = GeographicallyWeightedRegression(bandwidth=0.05)  # 5km bandwidth
    gwr.fit(X, values, coordinates)

    logger.info(f"GWR model training complete: {gwr.__class__.__name__}")

    return gwr


def clustering_analysis(coordinates, values):
    """Perform spatial clustering analysis."""
    logger.info("Performing spatial clustering analysis...")

    # Prepare feature matrix
    X = np.column_stack([coordinates, values.reshape(-1, 1)])

    # Perform spatial K-means
    kmeans = SpatialKMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X, coordinates)

    logger.info(f"Clustering completed with {len(set(labels))} clusters")

    return labels, kmeans.cluster_centers_


def parallel_processing_example(coordinates, values):
    """Demonstrate parallel processing capabilities."""
    logger.info("Demonstrating parallel processing...")

    # Function to compute distances from a point
    def compute_distances(query_point, all_points=coordinates):
        distances = []
        for point in all_points:
            dist = haversine_distance(
                query_point[1], query_point[0], point[1], point[0]
            )
            distances.append(dist)
        return distances

    # Use parallel processing to compute distance matrices

    # Select subset of points for demonstration
    query_points = coordinates[:10]

    # Parallel computation
    results = parallel_compute(
        compute_distances, query_points, num_workers=4, use_processes=True
    )

    logger.info(f"Parallel processing completed for {len(results)} queries")

    return results


def create_comprehensive_visualization(
    coordinates, values, utm_coords, interpolated_grid, xx, yy, labels
):
    """Create comprehensive visualization of all analyses."""
    logger.info("Creating comprehensive visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Advanced Geospatial Analysis - GEO-INFER-MATH", fontsize=16)

    # Plot 1: Original data
    scatter = axes[0, 0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=values, cmap="viridis", s=50, alpha=0.7
    )
    axes[0, 0].set_title("Original Data\n(Latitude vs Longitude)")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    plt.colorbar(scatter, ax=axes[0, 0], label="Value")

    # Plot 2: UTM coordinates
    scatter_utm = axes[0, 1].scatter(
        utm_coords[:, 0], utm_coords[:, 1], c=values, cmap="viridis", s=50, alpha=0.7
    )
    axes[0, 1].set_title("UTM Projection")
    axes[0, 1].set_xlabel("UTM Easting (m)")
    axes[0, 1].set_ylabel("UTM Northing (m)")
    plt.colorbar(scatter_utm, ax=axes[0, 1], label="Value")

    # Plot 3: Interpolation surface
    im = axes[0, 2].contourf(xx, yy, interpolated_grid, levels=20, cmap="viridis")
    axes[0, 2].scatter(
        utm_coords[:, 0],
        utm_coords[:, 1],
        c=values,
        edgecolors="white",
        s=30,
        alpha=0.8,
    )
    axes[0, 2].set_title("Spatial Interpolation\n(IDW Surface)")
    axes[0, 2].set_xlabel("UTM Easting (m)")
    axes[0, 2].set_ylabel("UTM Northing (m)")
    plt.colorbar(im, ax=axes[0, 2], label="Interpolated Value")

    # Plot 4: Clustering results
    colors = ["red", "blue", "green"]
    for i in range(3):
        mask = labels == i
        axes[1, 0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=colors[i],
            label=f"Cluster {i + 1}",
            s=50,
            alpha=0.7,
        )
    axes[1, 0].set_title("Spatial Clustering\n(K-means)")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    axes[1, 0].legend()

    # Plot 5: Statistical summary
    axes[1, 1].hist(values, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    axes[1, 1].axvline(
        np.mean(values), color="red", linestyle="--", linewidth=2, label="Mean"
    )
    axes[1, 1].axvline(
        np.median(values), color="green", linestyle="--", linewidth=2, label="Median"
    )
    axes[1, 1].set_title("Value Distribution")
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()

    # Plot 6: Analysis summary
    axes[1, 2].axis("off")
    summary_text = f"""
    Analysis Summary:

    • Dataset: {len(coordinates)} points
    • Value Range: {values.min():.1f} - {values.max():.1f}
    • Spatial Extent: {coordinates[:, 0].min():.3f}° to {coordinates[:, 0].max():.3f}° lon
                     {coordinates[:, 1].min():.3f}° to {coordinates[:, 1].max():.3f}° lat
    • Interpolation: IDW with power=2
    • Clustering: K-means with k=3
    • Coordinate Systems: WGS84, UTM, Web Mercator

    GEO-INFER-MATH provides comprehensive
    geospatial analysis capabilities including
    spatial statistics, interpolation, clustering,
    and coordinate transformations.
    """
    axes[1, 2].text(
        0.05,
        0.95,
        summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("advanced_geospatial_analysis.png", dpi=300, bbox_inches="tight")
    logger.info("Visualization saved as 'advanced_geospatial_analysis.png'")
    plt.close()


def main():
    """Main analysis workflow."""
    logger.info("Starting Advanced Geospatial Analysis with GEO-INFER-MATH")
    logger.info("=" * 60)

    # Generate synthetic data
    logger.info("Generating synthetic geospatial data...")
    coordinates, values = generate_synthetic_data(n_points=200)

    # Coordinate transformation analysis
    utm_coords, mercator_coords = coordinate_transformation_analysis(coordinates)

    # Spatial statistics analysis
    moran_result, lisa_result, g_result = spatial_statistics_analysis(
        coordinates, values
    )

    # Interpolation analysis
    xx, yy, interpolated_grid = interpolation_analysis(coordinates, values, utm_coords)

    # Regression analysis
    gwr_model = regression_analysis(coordinates, values)

    # Clustering analysis
    labels, cluster_centers = clustering_analysis(coordinates, values)

    # Parallel processing demonstration
    parallel_results = parallel_processing_example(coordinates, values)

    # Create comprehensive visualization
    create_comprehensive_visualization(
        coordinates, values, utm_coords, interpolated_grid, xx, yy, labels
    )

    # Print comprehensive results
    print("\n" + "=" * 60)
    print("ADVANCED GEOSPATIAL ANALYSIS RESULTS")
    print("=" * 60)

    if isinstance(moran_result, dict) and "moran_i" in moran_result:
        print(f"Global Moran's I: {moran_result['moran_i']:.4f}")
    else:
        print(f"Global Moran's I: {moran_result}")

    print("LISA Analysis:")
    print(f"  High-High clusters: {np.sum(lisa_result['classifications'] == 1)}")
    print(f"  Low-Low clusters: {np.sum(lisa_result['classifications'] == 2)}")
    print(f"  Outliers: {np.sum(lisa_result['classifications'] > 2)}")

    print("Hot Spot Analysis:")
    if isinstance(g_result, dict):
        print(f"  Z-score: {g_result.get('z', 0):.4f}")
        print(f"  P-value: {g_result.get('p', 0):.1f}")
    else:
        print(f"  Result: {g_result}")

    print("Clustering Results:")
    for i in range(3):
        count = np.sum(labels == i)
        print(f"  Cluster {i + 1}: {count} points")

    print("GWR Results:")
    print(f"  Completed model: {type(gwr_model).__name__}")
    print(f"  Bandwidth: {gwr_model.bandwidth}")

    print("Parallel Processing:")
    print(f"  Processed {len(parallel_results)} distance queries")

    print("Coordinate Transformations:")
    print(f"  Geographic → UTM: {len(coordinates)} points")
    print(f"  Geographic → Web Mercator: {len(coordinates)} points")

    print("Interpolation:")
    print(f"  IDW interpolation on {xx.shape[0]}x{xx.shape[1]} grid")
    print(
        f"  Interpolated range: {interpolated_grid.min():.1f} to {interpolated_grid.max():.1f}"
    )

    print("\nAnalysis completed successfully!")
    print("Visualization saved as 'advanced_geospatial_analysis.png'")


if __name__ == "__main__":
    main()
