#!/usr/bin/env python3
"""
Spatial Statistics Example

This example demonstrates the use of spatial statistics functions from
the GEO-INFER-MATH package, including Moran's I for spatial autocorrelation
and hot spot analysis using Getis-Ord G*.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from geo_infer_math.core.spatial_statistics import (
    MoranI,
    getis_ord_g,
    local_indicators_spatial_association,
    spatial_descriptive_statistics,
)


def generate_clustered_data(n=100, clusters=3, seed=42):
    """Generate spatially clustered data for demonstration."""
    np.random.seed(seed)

    # Create clusters
    coords = []
    values = []

    cluster_centers = np.random.uniform(0, 100, size=(clusters, 2))
    cluster_values = np.random.uniform(50, 150, size=clusters)

    for _ in range(n):
        # Randomly select a cluster
        cluster_idx = np.random.randint(0, clusters)

        # Generate point near cluster center with some noise
        center = cluster_centers[cluster_idx]
        point = center + np.random.normal(0, 5, size=2)

        # Generate value similar to cluster value with some noise
        value = cluster_values[cluster_idx] + np.random.normal(0, 10)

        coords.append(point)
        values.append(value)

    return np.array(coords), np.array(values)


def create_weight_matrix(coords, k=5):
    """Create a spatial weights matrix using k-nearest neighbors."""
    n = len(coords)
    weights = np.zeros((n, n))

    # Calculate all pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

    # For each point, find k nearest neighbors
    for i in range(n):
        # Get indices of k nearest neighbors
        nearest = np.argsort(distances[i])[1 : k + 1]  # Skip first (self)
        weights[i, nearest] = 1 / distances[i, nearest]

    # Row-standardize
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums[:, np.newaxis]

    return weights


def main():
    """Run the spatial statistics example."""
    # Generate sample data
    print("Generating sample data...")
    coords, values = generate_clustered_data(n=100, clusters=3)

    # Create spatial weights matrix
    weights_matrix = create_weight_matrix(coords, k=5)

    # 1. Calculate spatial descriptive statistics
    print("\nCalculating spatial descriptive statistics...")
    stats = spatial_descriptive_statistics(coords, values)
    print(f"Mean: {stats.mean:.2f}")
    print(f"Median: {stats.median:.2f}")
    print(f"Standard deviation: {stats.stdev:.2f}")
    print(f"Centroid: ({stats.centroid[0]:.2f}, {stats.centroid[1]:.2f})")
    print(f"Spatial dispersion: {stats.dispersion:.2f}")

    # 2. Calculate global Moran's I
    print("\nCalculating global Moran's I...")
    moran = MoranI(weights_matrix)
    moran_result = moran.compute(values)

    print(f"Moran's I: {moran_result['I']:.4f}")
    print(f"Expected I: {moran_result['expected_I']:.4f}")
    print(f"Z-score: {moran_result['z_score']:.4f}")
    print(f"P-value: {moran_result['p_value']:.4f}")

    if moran_result["p_value"] < 0.05:
        if moran_result["I"] > 0:
            print("Significant positive spatial autocorrelation (clustered pattern)")
        else:
            print("Significant negative spatial autocorrelation (dispersed pattern)")
    else:
        print("No significant spatial autocorrelation (random pattern)")

    # 3. Calculate Local Indicators of Spatial Association (LISA)
    print("\nCalculating LISA statistics...")
    lisa_result = local_indicators_spatial_association(values, weights_matrix)

    high_high = np.sum(lisa_result["classifications"] == 1)
    low_low = np.sum(lisa_result["classifications"] == 2)
    high_low = np.sum(lisa_result["classifications"] == 3)
    low_high = np.sum(lisa_result["classifications"] == 4)

    print(f"High-High clusters: {high_high}")
    print(f"Low-Low clusters: {low_low}")
    print(f"High-Low outliers: {high_low}")
    print(f"Low-High outliers: {low_high}")

    # 4. Calculate Getis-Ord G* hot spot analysis
    print("\nPerforming hot spot analysis...")
    g_result = getis_ord_g(values, weights_matrix)

    # Count significant hot and cold spots
    hot_spots = np.sum(g_result["z_scores"] > 1.96)
    cold_spots = np.sum(g_result["z_scores"] < -1.96)

    print(f"Hot spots (95% confidence): {hot_spots}")
    print(f"Cold spots (95% confidence): {cold_spots}")
    print(f"Global G: {g_result['global_g']:.4f}")

    # 5. Plot results
    plot_results(coords, values, lisa_result, g_result["z_scores"])


def plot_results(coords, values, lisa_result, g_star_zscores):
    """Plot spatial statistics results."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Raw values
    scatter = axs[0, 0].scatter(
        coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=50
    )
    axs[0, 0].set_title("Raw Values")
    fig.colorbar(scatter, ax=axs[0, 0], label="Value")

    # Plot 2: LISA clusters
    # Create custom colormap for LISA clusters
    class_cmap = ListedColormap(["#ffffff", "#ff0000", "#0000ff", "#ff00ff", "#00ffff"])
    axs[0, 1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=lisa_result["classifications"],
        cmap=class_cmap,
        s=50,
        vmin=0,
        vmax=4,
    )
    axs[0, 1].set_title("LISA Clusters")

    # Create a custom legend for LISA clusters
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ffffff",
            markersize=10,
            label="Not Significant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ff0000",
            markersize=10,
            label="High-High",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#0000ff",
            markersize=10,
            label="Low-Low",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ff00ff",
            markersize=10,
            label="High-Low",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00ffff",
            markersize=10,
            label="Low-High",
        ),
    ]
    axs[0, 1].legend(handles=legend_elements, loc="upper right")

    # Plot 3: Moran Scatterplot
    # Calculate spatially lagged values
    z = (values - np.mean(values)) / np.std(values)

    # Create weight matrix from LISA result
    n = len(values)
    w_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (
                lisa_result["classifications"][i] > 0
                and lisa_result["classifications"][j] > 0
            ):
                w_mat[i, j] = 1.0

    # Row standardize
    row_sums = w_mat.sum(axis=1)
    nonzero_rows = row_sums > 0
    w_mat[nonzero_rows] = w_mat[nonzero_rows] / row_sums[nonzero_rows, np.newaxis]

    # Calculate spatially lagged values
    lag_z = np.zeros(n)
    for i in range(n):
        lag_z[i] = np.sum(w_mat[i] * z)

    # Moran scatterplot
    axs[1, 0].scatter(z, lag_z, c=lisa_result["classifications"], cmap=class_cmap, s=50)
    axs[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axs[1, 0].axvline(x=0, color="k", linestyle="--", alpha=0.3)
    axs[1, 0].set_title("Moran's Scatterplot")
    axs[1, 0].set_xlabel("Standardized Values")
    axs[1, 0].set_ylabel("Spatially Lagged Values")

    # Add quadrant labels
    axs[1, 0].text(2, 2, "HH", fontsize=15, ha="center")
    axs[1, 0].text(-2, -2, "LL", fontsize=15, ha="center")
    axs[1, 0].text(-2, 2, "LH", fontsize=15, ha="center")
    axs[1, 0].text(2, -2, "HL", fontsize=15, ha="center")

    # Plot 4: Getis-Ord G* Hot Spot Analysis
    hot_cold_cmap = plt.cm.RdBu_r
    g_scatter = axs[1, 1].scatter(
        coords[:, 0], coords[:, 1], c=g_star_zscores, cmap=hot_cold_cmap, s=50
    )
    axs[1, 1].set_title("Hot Spot Analysis (G*)")
    g_cbar = fig.colorbar(g_scatter, ax=axs[1, 1], label="G* Z-Score")

    # Add significance thresholds to colorbar
    g_cbar.ax.axhline(
        y=g_cbar.ax.get_position().height * 0.95,
        xmin=0.25,
        xmax=0.75,
        color="k",
        linestyle="--",
    )
    g_cbar.ax.axhline(
        y=g_cbar.ax.get_position().height * 0.05,
        xmin=0.25,
        xmax=0.75,
        color="k",
        linestyle="--",
    )
    g_cbar.ax.text(1.5, g_cbar.ax.get_position().height * 0.95, "1.96", va="center")
    g_cbar.ax.text(1.5, g_cbar.ax.get_position().height * 0.05, "-1.96", va="center")

    plt.tight_layout()
    plt.savefig("spatial_statistics_results.png", dpi=300)
    plt.show()
    print("Plot saved as 'spatial_statistics_results.png'")


if __name__ == "__main__":
    main()
