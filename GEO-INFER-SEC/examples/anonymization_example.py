#!/usr/bin/env python3
"""
Example script demonstrating the use of geospatial anonymization techniques.

This example shows how to use the GeospatialAnonymizer class to apply
different anonymization methods to point data.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

from geo_infer_sec.core.anonymization import GeospatialAnonymizer


def create_sample_data(n_points=100, seed=42):
    """Create a sample GeoDataFrame with random points."""
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate random points in New York City area
    latitudes = np.random.uniform(40.70, 40.80, n_points)
    longitudes = np.random.uniform(-74.02, -73.92, n_points)

    # Generate some random attributes
    values = np.random.randint(0, 100, n_points)
    categories = np.random.choice(["A", "B", "C"], n_points)

    # Generate some PII-like data
    names = [f"Person {i}" for i in range(1, n_points + 1)]
    emails = [f"person{i}@example.com" for i in range(1, n_points + 1)]

    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        {
            "person_id": range(1, n_points + 1),
            "name": names,
            "email": emails,
            "value": values,
            "category": categories,
            "geometry": [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)],
        },
        crs="EPSG:4326",
    )

    return points_gdf


def plot_comparison(original, anonymized, title):
    """Plot original and anonymized data side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot original data
    original.plot(ax=ax1, color="blue", alpha=0.7, markersize=50)
    ax1.set_title("Original Data")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Plot anonymized data
    anonymized.plot(ax=ax2, color="red", alpha=0.7, markersize=50)
    ax2.set_title(f"Anonymized Data ({title})")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # Add grid lines
    ax1.grid(True)
    ax2.grid(True)

    plt.tight_layout()
    return fig


def demonstrate_location_perturbation(gdf):
    """Demonstrate location perturbation anonymization."""
    print("\n=== Location Perturbation ===")
    print("This method adds random noise to coordinate values.")

    # Create anonymizer
    anonymizer = GeospatialAnonymizer(seed=42)

    # Apply perturbation with different epsilon values
    epsilons = [50, 200, 500]  # meters

    for epsilon in epsilons:
        print(f"\nApplying perturbation with epsilon={epsilon} meters...")
        perturbed = anonymizer.location_perturbation(gdf, epsilon=epsilon)

        # Calculate the actual displacement distances
        distances = []
        for i, (orig, pert) in enumerate(zip(gdf.geometry, perturbed.geometry)):
            # Convert degrees to approximate meters (rough approximation)
            lat_meters = 111000  # 1 degree latitude ≈ 111 km
            lon_meters = 111000 * np.cos(
                np.radians(orig.y)
            )  # 1 degree longitude depends on latitude

            # Calculate displacement in meters
            dx = (pert.x - orig.x) * lon_meters
            dy = (pert.y - orig.y) * lat_meters
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)

        print(f"  Average displacement: {np.mean(distances):.2f} meters")
        print(f"  Maximum displacement: {np.max(distances):.2f} meters")

        # Plot the results
        fig = plot_comparison(gdf, perturbed, f"Perturbation (ε={epsilon}m)")
        plt.savefig(f"perturbation_epsilon_{epsilon}.png")
        plt.close(fig)


def demonstrate_spatial_k_anonymity(gdf):
    """Demonstrate spatial k-anonymity anonymization."""
    print("\n=== Spatial K-Anonymity ===")
    print("This method aggregates points into H3 cells to ensure k-anonymity.")

    # Create anonymizer
    anonymizer = GeospatialAnonymizer(seed=42)

    # Apply k-anonymity with different k values and resolutions
    params = [
        (5, 9),  # k=5, resolution=9 (~ 0.1 km cells)
        (10, 9),  # k=10, resolution=9
        (5, 8),  # k=5, resolution=8 (~ 0.5 km cells)
    ]

    for k, resolution in params:
        print(f"\nApplying k-anonymity with k={k}, resolution={resolution}...")
        anonymized = anonymizer.spatial_k_anonymity(gdf, k=k, h3_resolution=resolution)

        # Count unique locations
        unique_locations = set((p.x, p.y) for p in anonymized.geometry)

        print(
            f"  Original unique locations: {len(set((p.x, p.y) for p in gdf.geometry))}"
        )
        print(f"  Anonymized unique locations: {len(unique_locations)}")

        # Plot the results
        fig = plot_comparison(gdf, anonymized, f"K-Anonymity (k={k}, res={resolution})")
        plt.savefig(f"k_anonymity_k_{k}_res_{resolution}.png")
        plt.close(fig)


def create_admin_boundaries():
    """Create some artificial administrative boundaries for demonstration."""
    # Create a grid of 4 administrative areas
    admin_polygons = []
    admin_ids = []
    admin_names = []

    # New York grid (roughly divided into 4 areas)
    lat_mid = 40.75
    lon_mid = -73.97

    # Northwest
    from shapely.geometry import Polygon

    admin_polygons.append(
        Polygon(
            [(lon_mid, lat_mid), (lon_mid, 40.80), (-74.02, 40.80), (-74.02, lat_mid)]
        )
    )
    admin_ids.append(1)
    admin_names.append("Northwest District")

    # Northeast
    admin_polygons.append(
        Polygon(
            [(lon_mid, lat_mid), (lon_mid, 40.80), (-73.92, 40.80), (-73.92, lat_mid)]
        )
    )
    admin_ids.append(2)
    admin_names.append("Northeast District")

    # Southwest
    admin_polygons.append(
        Polygon(
            [(lon_mid, lat_mid), (lon_mid, 40.70), (-74.02, 40.70), (-74.02, lat_mid)]
        )
    )
    admin_ids.append(3)
    admin_names.append("Southwest District")

    # Southeast
    admin_polygons.append(
        Polygon(
            [(lon_mid, lat_mid), (lon_mid, 40.70), (-73.92, 40.70), (-73.92, lat_mid)]
        )
    )
    admin_ids.append(4)
    admin_names.append("Southeast District")

    # Create GeoDataFrame
    admin_gdf = gpd.GeoDataFrame(
        {"admin_id": admin_ids, "name": admin_names, "geometry": admin_polygons},
        crs="EPSG:4326",
    )

    return admin_gdf


def demonstrate_geographic_masking(gdf):
    """Demonstrate geographic masking anonymization."""
    print("\n=== Geographic Masking ===")
    print("This method aggregates data to administrative boundaries.")

    # Create anonymizer
    anonymizer = GeospatialAnonymizer(seed=42)

    # Create admin boundaries
    admin_boundaries = create_admin_boundaries()

    # Apply geographic masking
    print("\nApplying geographic masking...")

    # Define attributes to aggregate
    attribute_cols = ["value", "category"]

    masked = anonymizer.geographic_masking(
        gdf, attribute_cols=attribute_cols, admin_boundaries=admin_boundaries
    )

    print(f"  Original records: {len(gdf)}")
    print(f"  Masked records: {len(masked)} (1 per admin area)")

    # Plot the results (different from other methods due to polygon output)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot original points
    gdf.plot(ax=ax1, color="blue", alpha=0.7, markersize=50)
    admin_boundaries.boundary.plot(ax=ax1, color="black", linewidth=1)
    ax1.set_title("Original Data with Admin Boundaries")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Plot masked data
    masked.plot(ax=ax2, column="value", legend=True, cmap="viridis", alpha=0.7)
    masked.boundary.plot(ax=ax2, color="black", linewidth=1)
    ax2.set_title("Geographic Masking (Aggregated to Admin Areas)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # Add grid lines
    ax1.grid(True)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("geographic_masking.png")
    plt.close(fig)


def main():
    """Main function to run all demonstrations."""
    print("Geospatial Anonymization Techniques Demonstration")
    print("================================================")

    # Create sample data
    print("\nCreating sample point data...")
    gdf = create_sample_data(n_points=100)
    print(f"Created {len(gdf)} sample points")

    # Save original data
    gdf.to_file("original_data.geojson", driver="GeoJSON")

    # Demonstrate each anonymization method
    demonstrate_location_perturbation(gdf)
    demonstrate_spatial_k_anonymity(gdf)
    demonstrate_geographic_masking(gdf)

    print("\nDemonstration complete. Check the output files in the current directory.")


if __name__ == "__main__":
    main()
