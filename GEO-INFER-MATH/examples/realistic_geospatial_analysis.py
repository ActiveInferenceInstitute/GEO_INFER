#!/usr/bin/env python3
"""
Realistic Geospatial Data Processing Example

This example demonstrates the comprehensive capabilities of GEO-INFER-MATH
using realistic geospatial datasets and workflows, including:

- Environmental monitoring data analysis
- Urban planning spatial statistics
- Public health data spatial analysis
- Natural disaster risk assessment
- Transportation network analysis
- Climate data interpolation

The example uses synthetically generated but realistic data to demonstrate
real-world geospatial analysis workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from pathlib import Path

# Import GEO-INFER-MATH modules
from geo_infer_math.core.spatial_statistics import (
    MoranI,
    getis_ord_g,
    spatial_descriptive_statistics,
    local_indicators_spatial_association,
)
from geo_infer_math.core.interpolation import SpatialInterpolator
from geo_infer_math.models.clustering import spatial_clustering_analysis
from geo_infer_math.api.spatial_analysis import SpatialAnalysisAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RealisticGeospatialAnalyzer:
    """Comprehensive geospatial analysis using realistic datasets."""

    def __init__(self, output_dir: str = "geospatial_analysis_output"):
        """
        Initialize the analyzer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.api = SpatialAnalysisAPI(verbose=True)

    def generate_environmental_monitoring_data(self, n_stations: int = 50) -> tuple:
        """
        Generate realistic environmental monitoring data.

        Args:
            n_stations: Number of monitoring stations

        Returns:
            Tuple of (coordinates, air_quality_data, temperature_data, humidity_data)
        """
        logger.info(
            f"Generating environmental monitoring data for {n_stations} stations..."
        )

        # Create realistic station locations (urban area)
        np.random.seed(42)

        # Urban center coordinates (simulating a city like Seattle)
        center_lat, center_lon = 47.6062, -122.3321

        # Generate station coordinates with some clustering in urban areas
        coords = []
        for i in range(n_stations):
            # Most stations in urban core, some in suburbs
            if i < n_stations * 0.7:  # 70% in urban core
                lat = np.random.normal(center_lat, 0.05)
                lon = np.random.normal(center_lon, 0.05)
            else:  # 30% in suburbs
                lat = np.random.normal(center_lat, 0.15)
                lon = np.random.normal(center_lon, 0.15)

            coords.append([lat, lon])

        coordinates = np.array(coords)

        # Generate realistic environmental data
        # Air quality (PM2.5) - higher in urban core due to traffic
        urban_effect = np.exp(
            -0.5
            * (
                (coordinates[:, 0] - center_lat) ** 2
                + (coordinates[:, 1] - center_lon) ** 2
            )
            / 0.01
        )
        air_quality = 25 + 15 * urban_effect + np.random.normal(0, 3, n_stations)

        # Temperature - slight urban heat island effect
        temperature = 18 + 2 * urban_effect + np.random.normal(0, 1.5, n_stations)

        # Humidity - lower in urban areas
        humidity = 65 - 5 * urban_effect + np.random.normal(0, 4, n_stations)

        return coordinates, air_quality, temperature, humidity

    def analyze_air_quality_patterns(
        self, coordinates: np.ndarray, air_quality: np.ndarray
    ) -> dict:
        """
        Analyze spatial patterns in air quality data.

        Args:
            coordinates: Station coordinates
            air_quality: Air quality measurements

        Returns:
            Analysis results
        """
        logger.info("Analyzing air quality spatial patterns...")

        # Spatial autocorrelation analysis
        from geo_infer_math.core.linalg_tensor import MatrixOperations

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            coordinates, method="inverse_distance", k=8
        )

        moran = MoranI(weights_matrix)
        moran_result = moran.compute(air_quality, coordinates)

        # Hot spot analysis
        g_result = getis_ord_g(air_quality, weights_matrix)

        # Local indicators of spatial association
        lisa_result = local_indicators_spatial_association(air_quality, weights_matrix)

        return {
            "moran_i": moran_result,
            "hotspots": g_result,
            "lisa": lisa_result,
            "descriptive_stats": spatial_descriptive_statistics(
                coordinates, air_quality
            ),
        }

    def analyze_urban_heat_patterns(
        self, coordinates: np.ndarray, temperature: np.ndarray
    ) -> dict:
        """
        Analyze urban heat island patterns.

        Args:
            coordinates: Station coordinates
            temperature: Temperature measurements

        Returns:
            Analysis results
        """
        logger.info("Analyzing urban heat island patterns...")

        # Create spatial weights for temperature analysis
        from geo_infer_math.core.linalg_tensor import MatrixOperations

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            coordinates, method="gaussian"
        )

        # Temperature autocorrelation
        moran = MoranI(weights_matrix)
        moran_result = moran.compute(temperature, coordinates)

        # Identify heat islands using clustering
        # Combine coordinates with temperature for clustering
        features = np.column_stack([coordinates, temperature.reshape(-1, 1)])
        clustering_result = spatial_clustering_analysis(
            features, coordinates, method="kmeans", n_clusters=3
        )

        return {"moran_i": moran_result, "clustering": clustering_result}

    def public_health_spatial_analysis(self) -> dict:
        """
        Analyze public health data with spatial patterns.

        Returns:
            Public health analysis results
        """
        logger.info("Analyzing public health spatial patterns...")

        # Generate realistic health data (asthma rates by neighborhood)
        n_neighborhoods = 30

        # Neighborhood centroids
        np.random.seed(123)
        neighborhood_coords = np.random.rand(n_neighborhoods, 2) * 10 + np.array(
            [47.5, -122.4]
        )

        # Asthma prevalence (higher in urban areas with poor air quality)
        urban_distance = np.sqrt(
            np.sum((neighborhood_coords - np.array([47.6062, -122.3321])) ** 2, axis=1)
        )
        asthma_rates = (
            0.08
            + 0.02 * np.exp(-urban_distance / 2)
            + np.random.normal(0, 0.005, n_neighborhoods)
        )
        asthma_rates = np.clip(asthma_rates, 0.02, 0.20)  # Realistic range

        # Analyze spatial patterns
        from geo_infer_math.core.linalg_tensor import MatrixOperations

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            neighborhood_coords, method="knn", k=5
        )

        moran = MoranI(weights_matrix)
        moran_result = moran.compute(asthma_rates, neighborhood_coords)

        # Hot spot analysis for asthma prevalence
        g_result = getis_ord_g(asthma_rates, weights_matrix)

        return {
            "coordinates": neighborhood_coords,
            "asthma_rates": asthma_rates,
            "moran_i": moran_result,
            "hotspots": g_result,
        }

    def disaster_risk_assessment(self) -> dict:
        """
        Assess natural disaster risks using spatial analysis.

        Returns:
            Disaster risk analysis results
        """
        logger.info("Assessing natural disaster risks...")

        # Generate earthquake monitoring stations
        n_stations = 40
        np.random.seed(456)

        # Pacific Northwest region (earthquake prone area)
        station_coords = np.random.rand(n_stations, 2) * 5 + np.array([45.0, -123.0])

        # Seismic activity (simulated readings)
        distance_from_fault = np.sqrt(
            np.sum((station_coords - np.array([46.0, -122.0])) ** 2, axis=1)
        )
        seismic_activity = (
            2.0
            + 1.5 * np.exp(-distance_from_fault / 1.5)
            + np.random.normal(0, 0.3, n_stations)
        )

        # Risk assessment using spatial statistics
        from geo_infer_math.core.linalg_tensor import MatrixOperations

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            station_coords, method="inverse_distance"
        )

        # High-risk areas identification
        risk_threshold = np.percentile(seismic_activity, 75)
        high_risk_mask = seismic_activity > risk_threshold

        # Spatial clustering of high-risk areas
        risk_coords = station_coords[high_risk_mask]
        if len(risk_coords) > 3:  # Need minimum points for clustering
            risk_features = np.column_stack(
                [risk_coords, seismic_activity[high_risk_mask]]
            )
            clustering_result = spatial_clustering_analysis(
                risk_features, risk_coords, method="dbscan"
            )
        else:
            clustering_result = None

        return {
            "station_coordinates": station_coords,
            "seismic_activity": seismic_activity,
            "risk_threshold": risk_threshold,
            "high_risk_coordinates": risk_coords,
            "clustering": clustering_result,
        }

    def transportation_network_analysis(self) -> dict:
        """
        Analyze transportation network using graph theory.

        Returns:
            Transportation network analysis results
        """
        logger.info("Analyzing transportation network...")

        # Generate realistic road network (simplified grid with some randomness)
        n_intersections = 25
        np.random.seed(789)

        # Create grid-like network with some perturbations
        grid_size = int(np.sqrt(n_intersections))
        intersections = []

        for i in range(grid_size):
            for j in range(grid_size):
                # Add some random perturbation
                x = i + np.random.normal(0, 0.1)
                y = j + np.random.normal(0, 0.1)
                intersections.append([x, y])

        intersections = np.array(intersections)

        # Create graph representation
        from geo_infer_math.core.graph_theory import SpatialGraph

        graph = SpatialGraph(directed=False)

        # Add intersections as nodes
        for i, coord in enumerate(intersections):
            graph.add_node(f"intersection_{i}", coord, intersection_id=i)

        # Add edges between nearby intersections
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                distance = np.sqrt(np.sum((intersections[i] - intersections[j]) ** 2))
                if distance < 0.5:  # Connect nearby intersections
                    graph.add_edge(f"intersection_{i}", f"intersection_{j}", distance)

        # Analyze network properties
        network_analysis = graph.spatial_network_analysis()

        # Find shortest paths between random points
        if len(intersections) > 3:
            start_idx = 0
            end_idx = len(intersections) - 1
            path, distance = graph.shortest_path(
                f"intersection_{start_idx}", f"intersection_{end_idx}"
            )
        else:
            path, distance = [], 0

        return {
            "intersections": intersections,
            "graph": graph,
            "network_analysis": network_analysis,
            "sample_path": path,
            "sample_distance": distance,
        }

    def climate_data_interpolation(self) -> dict:
        """
        Interpolate climate data across a region.

        Returns:
            Climate data interpolation results
        """
        logger.info("Interpolating climate data...")

        # Generate weather station data
        n_stations = 20
        np.random.seed(101)

        # Regional coordinates (Pacific Northwest)
        station_coords = np.random.rand(n_stations, 2) * 8 + np.array([45.0, -124.0])

        # Temperature readings (realistic seasonal variation)
        base_temp = 15  # Base temperature
        elevation_effect = np.random.rand(n_stations) * 5  # Elevation effect
        precipitation = np.random.exponential(2, n_stations)  # Precipitation in mm

        # Temperature influenced by latitude and elevation
        lat_effect = (station_coords[:, 0] - 45.0) * -2  # Cooler at higher latitudes
        temperature = (
            base_temp
            + lat_effect
            + elevation_effect
            + np.random.normal(0, 1, n_stations)
        )

        # Create interpolation grid
        lat_min, lat_max = (
            station_coords[:, 0].min() - 0.5,
            station_coords[:, 0].max() + 0.5,
        )
        lon_min, lon_max = (
            station_coords[:, 1].min() - 0.5,
            station_coords[:, 1].max() + 0.5,
        )

        grid_resolution = 0.1
        lat_grid = np.arange(lat_min, lat_max, grid_resolution)
        lon_grid = np.arange(lon_min, lon_max, grid_resolution)

        # Create meshgrid for interpolation
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([lat_mesh.flatten(), lon_mesh.flatten()])

        # Perform interpolation using different methods
        methods = ["idw", "kriging"]
        interpolation_results = {}

        for method in methods:
            try:
                interpolator = SpatialInterpolator(method=method)
                if method == "idw":
                    interpolator.fit(station_coords, temperature)
                else:  # kriging
                    interpolator.fit(station_coords, temperature)

                interpolated_temps = interpolator.predict(grid_points)
                interpolation_results[method] = {
                    "grid_points": grid_points,
                    "temperatures": interpolated_temps,
                    "grid_shape": (len(lat_grid), len(lon_grid)),
                }

            except Exception as e:
                logger.warning(f"Interpolation method {method} failed: {e}")
                interpolation_results[method] = None

        return {
            "station_coordinates": station_coords,
            "station_temperatures": temperature,
            "interpolation_results": interpolation_results,
            "grid_bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        }

    def run_comprehensive_analysis(self) -> dict:
        """
        Run comprehensive geospatial analysis on all datasets.

        Returns:
            Complete analysis results
        """
        logger.info("Starting comprehensive geospatial analysis...")

        results = {}

        # 1. Environmental Monitoring Analysis
        logger.info("=== Environmental Monitoring Analysis ===")
        env_coords, air_quality, temperature, humidity = (
            self.generate_environmental_monitoring_data()
        )

        results["environmental"] = {
            "air_quality": self.analyze_air_quality_patterns(env_coords, air_quality),
            "urban_heat": self.analyze_urban_heat_patterns(env_coords, temperature),
            "humidity_patterns": spatial_descriptive_statistics(env_coords, humidity),
        }

        # 2. Public Health Analysis
        logger.info("=== Public Health Analysis ===")
        health_results = self.public_health_spatial_analysis()
        results["public_health"] = health_results

        # 3. Disaster Risk Assessment
        logger.info("=== Disaster Risk Assessment ===")
        disaster_results = self.disaster_risk_assessment()
        results["disaster_risk"] = disaster_results

        # 4. Transportation Network Analysis
        logger.info("=== Transportation Network Analysis ===")
        transport_results = self.transportation_network_analysis()
        results["transportation"] = transport_results

        # 5. Climate Data Interpolation
        logger.info("=== Climate Data Interpolation ===")
        climate_results = self.climate_data_interpolation()
        results["climate"] = climate_results

        return results

    def create_visualization_summary(self, results: dict) -> None:
        """
        Create comprehensive visualization of all analyses.

        Args:
            results: Analysis results from run_comprehensive_analysis
        """
        logger.info("Creating comprehensive visualization...")

        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        fig.suptitle("Comprehensive Geospatial Analysis - GEO-INFER-MATH", fontsize=20)

        # 1. Air Quality Hot Spots
        ax1 = axes[0, 0]
        env_data = results["environmental"]
        air_quality_data = env_data["air_quality"]
        coords = air_quality_data[
            "descriptive_stats"
        ].centroid  # Use centroid as reference

        # Plot air quality hot spots
        z_scores = air_quality_data["hotspots"]["z_scores"]
        scatter = ax1.scatter(
            coords[0], coords[1], c=z_scores, cmap="RdYlBu_r", s=100, alpha=0.7
        )
        ax1.set_title("Air Quality Hot Spots\n(Getis-Ord G* Z-Scores)")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        plt.colorbar(scatter, ax=ax1, label="Z-Score")

        # 2. Urban Heat Island Clustering
        ax2 = axes[0, 1]
        heat_data = env_data["urban_heat"]
        # Simplified visualization - in practice would use actual coordinates
        ax2.text(
            0.5,
            0.5,
            "Urban Heat Island\nClustering Results\n(Moran I: {:.3f})".format(
                heat_data["moran_i"]["I"]
            ),
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Urban Heat Island Analysis")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # 3. Public Health Spatial Patterns
        ax3 = axes[1, 0]
        health_data = results["public_health"]
        coords = health_data["coordinates"]
        asthma_rates = health_data["asthma_rates"]

        scatter = ax3.scatter(
            coords[:, 1], coords[:, 0], c=asthma_rates, cmap="Reds", s=80, alpha=0.8
        )
        ax3.set_title(
            "Asthma Prevalence by Neighborhood\n(Moran I: {:.3f})".format(
                health_data["moran_i"]["I"]
            )
        )
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        plt.colorbar(scatter, ax=ax3, label="Asthma Rate")

        # 4. Disaster Risk Assessment
        ax4 = axes[1, 1]
        disaster_data = results["disaster_risk"]
        coords = disaster_data["station_coordinates"]
        activity = disaster_data["seismic_activity"]

        scatter = ax4.scatter(
            coords[:, 1], coords[:, 0], c=activity, cmap="YlOrRd", s=60, alpha=0.8
        )
        ax4.set_title("Seismic Activity Risk Assessment")
        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        plt.colorbar(scatter, ax=ax4, label="Seismic Activity")

        # 5. Transportation Network
        ax5 = axes[2, 0]
        transport_data = results["transportation"]
        coords = transport_data["intersections"]

        scatter = ax5.scatter(coords[:, 1], coords[:, 0], c="blue", s=50, alpha=0.7)
        ax5.set_title(
            "Transportation Network\n({} intersections, {} connections)".format(
                len(coords), transport_data["network_analysis"]["n_edges"]
            )
        )
        ax5.set_xlabel("Longitude")
        ax5.set_ylabel("Latitude")

        # 6. Climate Data Interpolation
        ax6 = axes[2, 1]
        climate_data = results["climate"]
        interp_results = climate_data["interpolation_results"]

        if "idw" in interp_results and interp_results["idw"] is not None:
            idw_result = interp_results["idw"]
            grid_shape = idw_result["grid_shape"]
            temps = idw_result["temperatures"].reshape(grid_shape)

            im = ax6.imshow(
                temps,
                cmap="coolwarm",
                origin="lower",
                extent=[
                    climate_data["grid_bounds"]["lon_min"],
                    climate_data["grid_bounds"]["lon_max"],
                    climate_data["grid_bounds"]["lat_min"],
                    climate_data["grid_bounds"]["lat_max"],
                ],
            )
            ax6.scatter(
                climate_data["station_coordinates"][:, 1],
                climate_data["station_coordinates"][:, 0],
                c="red",
                s=30,
                marker="^",
                edgecolors="white",
                linewidth=0.5,
            )
            ax6.set_title("Temperature Interpolation (IDW)")
            ax6.set_xlabel("Longitude")
            ax6.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax6, label="Temperature (°C)")

        plt.tight_layout()

        # Save comprehensive visualization
        output_path = self.output_dir / "comprehensive_geospatial_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Comprehensive visualization saved to {output_path}")

        # Create summary report
        self.create_analysis_summary(results)

    def create_analysis_summary(self, results: dict) -> None:
        """
        Create a comprehensive analysis summary.

        Args:
            results: Analysis results
        """
        summary_path = self.output_dir / "geospatial_analysis_summary.txt"

        with open(summary_path, "w") as f:
            f.write("GEO-INFER-MATH: Comprehensive Geospatial Analysis Summary\n")
            f.write("=" * 60 + "\n\n")

            # Environmental Analysis Summary
            f.write("1. ENVIRONMENTAL MONITORING ANALYSIS\n")
            f.write("-" * 40 + "\n")
            env_data = results["environmental"]
            air_quality = env_data["air_quality"]

            f.write(".3f.3f.3f")

            # Public Health Summary
            f.write("\n\n2. PUBLIC HEALTH ANALYSIS\n")
            f.write("-" * 40 + "\n")
            health_data = results["public_health"]
            f.write(".3f.3f")

            # Disaster Risk Summary
            f.write("\n\n3. DISASTER RISK ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            disaster_data = results["disaster_risk"]
            f.write(".2f.0f")

            # Transportation Summary
            f.write("\n\n4. TRANSPORTATION NETWORK ANALYSIS\n")
            f.write("-" * 40 + "\n")
            transport_data = results["transportation"]
            network_analysis = transport_data["network_analysis"]
            f.write(".0f.0f.3f.1f")

            # Climate Summary
            f.write("\n\n5. CLIMATE DATA INTERPOLATION\n")
            f.write("-" * 40 + "\n")
            climate_data = results["climate"]
            station_temps = climate_data["station_temperatures"]
            f.write(".1f.1f.1f.1f")

            f.write("\n\nANALYSIS COMPLETED SUCCESSFULLY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"Analysis summary saved to {summary_path}")


def main():
    """Run the comprehensive geospatial analysis example."""
    logger.info("Starting realistic geospatial data processing example...")

    # Create analyzer
    analyzer = RealisticGeospatialAnalyzer()

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()

    # Create visualizations and summary
    analyzer.create_visualization_summary(results)

    logger.info("Comprehensive geospatial analysis completed successfully!")
    logger.info(f"Check output directory: {analyzer.output_dir}")

    return results


if __name__ == "__main__":
    main()
