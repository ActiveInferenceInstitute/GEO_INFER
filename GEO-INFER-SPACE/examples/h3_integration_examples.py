"""
H3 Integration Examples.

Examples showing H3 integration with other GEO-INFER-SPACE modules
for comprehensive spatial analysis workflows.
"""

import logging
from datetime import datetime, timedelta

# Unified Spatial Architecture
import geopandas as gpd
from shapely.geometry import Polygon

from geo_infer_space.analytics.temporal import TemporalAnalyzer
from geo_infer_space.analytics.vector import geometric_calculations
from geo_infer_space.core import SpatialAnalyticsInterface, SpatialIndexingInterface

logger = logging.getLogger(__name__)


def example_h3_vector_integration():
    """
    Example: Integrate H3 with vector analytics.

    Shows how to use H3 hexagonal grids with traditional vector operations
    for comprehensive spatial analysis.
    """
    print("H3-Vector Integration Example")
    print("=" * 40)

    # Create H3 grid for San Francisco area
    sf_bounds = {"min_lat": 37.7, "max_lat": 37.8, "min_lng": -122.5, "max_lng": -122.4}

    # Generate H3 cells for the area
    polygon_coords = {
        "type": "Polygon",
        "coordinates": [
            [
                [sf_bounds["min_lng"], sf_bounds["min_lat"]],
                [sf_bounds["max_lng"], sf_bounds["min_lat"]],
                [sf_bounds["max_lng"], sf_bounds["max_lat"]],
                [sf_bounds["min_lng"], sf_bounds["max_lat"]],
                [sf_bounds["min_lng"], sf_bounds["min_lat"]],
            ]
        ],
    }

    indexer = SpatialIndexingInterface(backend="h3")

    # Using H3 backend explicitly
    h3_cells = indexer.polygon_to_cells(polygon_coords, resolution=9)
    print(f"Generated {len(h3_cells)} H3 cells for San Francisco area")

    # Create H3 grid with synthetic data
    grid_cells = []

    for i, cell_index in enumerate(list(h3_cells)[:20]):  # Limit for example
        # Add synthetic properties
        properties = {
            "population": 1000 + (i * 50),
            "poi_count": 5 + (i % 10),
            "crime_incidents": 2 + (i % 5),
            "avg_income": 50000 + (i * 1000),
        }

        grid_cells.append({"index": cell_index, "properties": properties})

    # Convert H3 grid to GeoDataFrame for vector operations
    geometries = []
    properties_list = []

    for cell_data in grid_cells:
        cell_index = cell_data["index"]
        # Get cell boundary as polygon
        try:
            boundary = indexer.get_cell_boundary(cell_index)
            # boundary is usually [(lat, lng), ...]
            # Shapely expects (x, y) = (lng, lat)
            polygon = Polygon([(lng, lat) for lat, lng in boundary])

            geometries.append(polygon)
            props = cell_data["properties"]
            properties_list.append(
                {
                    "h3_index": cell_index,
                    "population": props.get("population", 0),
                    "poi_count": props.get("poi_count", 0),
                    "crime_incidents": props.get("crime_incidents", 0),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to process cell {cell_index}: {e}")

    if geometries:
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")

        # Apply vector analytics
        # Apply the configured vector analytics implementation.
        try:
            gdf_with_calcs = geometric_calculations(gdf)
            print(
                f"Applied geometric calculations to {len(gdf_with_calcs)} hexagonal cells"
            )

            # Calculate density metrics
            if "area" in gdf_with_calcs.columns:
                gdf_with_calcs["population_density"] = (
                    gdf_with_calcs["population"] / gdf_with_calcs["area"]
                )
            else:
                # Fallback area calculation if vector module didn't provide it
                # H3 res 9 area is approx 0.1 sq km
                gdf_with_calcs["population_density"] = (
                    gdf_with_calcs["population"] / 0.1
                )

            gdf_with_calcs["crime_rate"] = (
                gdf_with_calcs["crime_incidents"] / gdf_with_calcs["population"] * 1000
            )

            print(
                f"Average population density: {gdf_with_calcs['population_density'].mean():.2f}"
            )
            print(
                f"Average crime rate: {gdf_with_calcs['crime_rate'].mean():.2f} per 1000 residents"
            )

            # Find high-density areas using Spatial analysis
            analytics = SpatialAnalyticsInterface(backend="h3")

            # Prepare data for analytics (cells and values)
            data_for_analytics = {
                "cells": gdf["h3_index"].tolist(),
                "values": gdf["population"].tolist(),
            }

            hotspots = analytics.analyze_hotspots(
                data_for_analytics, method="getis_ord"
            )

            # Note: Result format depends on backend implementation
            print(
                f"Hotspot analysis completed: {hotspots.get('hotspot_count', 0)} hotspots detected"
            )

        except Exception as e:
            print(f"Vector/Spatial operations failed: {e}")

    print("H3-Vector integration completed successfully\n")


def example_h3_density_clustering():
    """
    Example: H3 density analysis with clustering.

    Demonstrates density-based spatial analysis using H3 grids
    combined with clustering algorithms.
    """
    print("H3 Density-Clustering Analysis Example")
    print("=" * 40)

    indexer = SpatialIndexingInterface(backend="h3")

    # Simulate urban density patterns
    city_centers = [
        (37.7749, -122.4194),  # San Francisco
        (37.7849, -122.4094),  # North area
        (37.7649, -122.4294),  # South area
    ]

    grid_cells = {}  # Map index to properties

    cell_count = 0
    for center_lat, center_lng in city_centers:
        # Create density gradient around each center
        center_cell = indexer.latlng_to_cell(center_lat, center_lng, 8)

        # Get cells within 3 rings of center
        area_cells = indexer.get_cell_neighbors(center_cell, k=3)

        for cell_index in area_cells:
            if len(grid_cells) >= 50:  # Limit for example
                break

            # Calculate distance from center for density gradient
            try:
                distance = indexer.get_cell_distance(center_cell, cell_index)

                # Higher density closer to center
                base_density = 1000
                density = max(100, base_density - (distance * 150))

                properties = {
                    "population_density": density,
                    "business_count": max(5, 50 - (distance * 8)),
                    "traffic_volume": max(100, 1000 - (distance * 100)),
                    "center_distance": distance,
                }

                grid_cells[cell_index] = properties
                cell_count += 1

            except Exception as e:
                logger.warning(f"Failed to process cell {cell_index}: {e}")

    print(f"Created H3 grid with {len(grid_cells)} unique cells")

    # Analytics interface
    analytics = SpatialAnalyticsInterface(backend="h3")

    # Prepare data for analytics
    cells_list = list(grid_cells.keys())
    density_values = [props["population_density"] for props in grid_cells.values()]

    data = {
        "cells": cells_list,
        "values": density_values,
        "label": "population_density",
    }

    # Calculate density patterns
    # Use the H3 cell/value representation supported by the analytics backend.
    try:
        density_result = analytics.compute_density(
            [indexer.cell_to_latlng(cell) for cell in cells_list], data=data
        )
        print(
            f"Density analysis completed for {len(cells_list)} cells "
            f"({len(density_result)} result fields)"
        )
    except Exception as e:
        print(f"Density analysis skipped: {e}")

    # Clustering analysis
    try:
        # Group the observed density values into three interpretable bands.
        high_density = [
            c for c, p in grid_cells.items() if p["population_density"] > 800
        ]
        medium_density = [
            c for c, p in grid_cells.items() if 400 < p["population_density"] <= 800
        ]
        low_density = [
            c for c, p in grid_cells.items() if p["population_density"] <= 400
        ]

        print("Density-band clustering:")
        print(f"  High Density Cluster: {len(high_density)} cells")
        print(f"  Medium Density Cluster: {len(medium_density)} cells")
        print(f"  Low Density Cluster: {len(low_density)} cells")

    except Exception as e:
        print(f"Clustering analysis failed: {e}")

    print("H3 density-clustering analysis completed successfully\n")


def example_h3_temporal_analysis():
    """
    Example: H3 temporal pattern analysis.

    Shows temporal analysis of spatial data using H3 grids
    for time-series spatial analytics.
    """
    print("H3 Temporal Analysis Example")
    print("=" * 40)

    indexer = SpatialIndexingInterface(backend="h3")

    # Simulate 24 hours of activity data
    base_time = datetime(2023, 6, 15, 0, 0, 0)  # June 15, 2023
    sf_center = (37.7749, -122.4194)

    # Create cells around SF with hourly data
    center_cell = indexer.latlng_to_cell(sf_center[0], sf_center[1], 9)
    area_cells = indexer.get_cell_neighbors(center_cell, k=2)[:15]  # Limit for example

    temporal_data = []

    for hour in range(24):
        for i, cell_index in enumerate(area_cells):
            # Create realistic temporal patterns
            if 7 <= hour <= 9:  # Morning rush
                activity_level = 80 + (i * 5)
            elif 17 <= hour <= 19:  # Evening rush
                activity_level = 90 + (i * 4)
            elif 12 <= hour <= 14:  # Lunch time
                activity_level = 60 + (i * 3)
            elif 22 <= hour or hour <= 5:  # Night
                activity_level = 10 + (i * 2)
            else:  # Regular hours
                activity_level = 40 + (i * 3)

            # Add some randomness
            import random

            activity_level += random.randint(-10, 10)
            activity_level = max(0, activity_level)

            timestamp = base_time + timedelta(hours=hour)

            record = {
                "timestamp": timestamp.isoformat(),
                "activity_level": activity_level,
                "hour": hour,
                "trip_count": activity_level // 2,
                "cell_id": cell_index,
            }

            temporal_data.append(record)

    print(f"Created temporal data with {len(temporal_data)} observations")

    # Temporal analysis
    temporal_analyzer = TemporalAnalyzer()

    # Analyze temporal patterns
    patterns = temporal_analyzer.analyze_temporal_patterns(
        temporal_data, "timestamp", "activity_level", temporal_resolution="hour"
    )

    if "temporal_patterns" in patterns:
        print("Temporal pattern analysis:")
        temporal_patterns = patterns["temporal_patterns"]

        # Show peak periods
        if "peak_periods" in temporal_patterns:
            print("Top 3 peak activity periods:")
            for i, period in enumerate(temporal_patterns["peak_periods"][:3]):
                print(
                    f"  {i + 1}. Hour {period['period']}: {period['mean_value']:.1f} avg activity"
                )

    print("H3 temporal analysis completed successfully\n")


def example_h3_multi_resolution_analysis():
    """
    Example: Multi-resolution H3 analysis.

    Demonstrates analysis across multiple H3 resolutions
    for hierarchical spatial understanding.
    """
    print("H3 Multi-Resolution Analysis Example")
    print("=" * 40)

    indexer = SpatialIndexingInterface(backend="h3")

    # Define analysis area (San Francisco Bay Area)
    bay_area_bounds = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.6, 37.4],
                [-122.0, 37.4],
                [-122.0, 37.9],
                [-122.6, 37.9],
                [-122.6, 37.4],
            ]
        ],
    }

    # Analyze at multiple resolutions
    resolutions = [6, 7, 8, 9]
    resolution_results = {}

    for resolution in resolutions:
        print(f"Analyzing at resolution {resolution}...")

        # Get H3 cells for the area
        cells = indexer.polygon_to_cells(bay_area_bounds, resolution)

        # Create synthetic data
        cell_data = []
        for i, cell_index in enumerate(
            list(cells)[: min(50, len(cells))]
        ):  # Limit for example
            # Simulate population data (higher resolution = more detailed)
            base_pop = 1000 if resolution <= 7 else 100
            population = base_pop + (i * (10 if resolution <= 7 else 5))

            area_km2 = 10 / (2 ** (resolution - 6))  # Approximate area scaling

            cell_data.append(
                {
                    "cell_index": cell_index,
                    "population": population,
                    "area_km2": area_km2,
                    "density": population / area_km2,
                }
            )

        # Calculate statistics
        total_population = sum(d["population"] for d in cell_data)
        total_area = sum(d["area_km2"] for d in cell_data)
        avg_density = total_population / total_area if total_area > 0 else 0

        resolution_results[resolution] = {
            "num_cells": len(cell_data),
            "total_population": total_population,
            "total_area_km2": total_area,
            "avg_density": avg_density,
        }

        print(
            f"  Resolution {resolution}: {len(cell_data)} cells, "
            f"density: {avg_density:.1f} pop/km²"
        )

    # Compare across resolutions
    print("\nMulti-resolution comparison:")
    print("Resolution | Cells | Avg Density")
    print("-" * 50)

    for res in resolutions:
        result = resolution_results[res]
        print(
            f"    {res:2d}     | {result['num_cells']:5d} | "
            f"{result['avg_density']:8.1f}  | {result['spatial_autocorrelation']:8.3f}"
        )

    print("Multi-resolution analysis completed successfully\n")


def example_h3_orchestration_workflow():
    """
    Example: Complete H3 orchestration workflow.

    Demonstrates a complete spatial analysis workflow using H3
    integrated with multiple SPACE modules.
    """
    print("H3 Orchestration Workflow Example")
    print("=" * 40)

    indexer = SpatialIndexingInterface(backend="h3")

    # Step 1: Data preparation
    print("Step 1: Data Preparation")

    # Define study area
    # Poly in GeoJSON: [lng, lat]
    study_area_poly = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.43, 37.77],
                [-122.41, 37.77],
                [-122.41, 37.79],
                [-122.43, 37.79],
                [-122.43, 37.77],
            ]
        ],
    }

    # Generate H3 grid
    h3_cells = indexer.polygon_to_cells(study_area_poly, resolution=9)

    grid_data = []  # List of dicts

    # Add synthetic urban data
    for i, cell_index in enumerate(list(h3_cells)[:30]):  # Limit for example
        properties = {
            "population": 800 + (i * 25),
            "employment": 400 + (i * 15),
            "retail_sqft": 5000 + (i * 200),
            "green_space_pct": max(5, 30 - (i * 0.8)),
            "transit_access": min(10, 3 + (i * 0.2)),
            "housing_units": 300 + (i * 10),
            "avg_rent": 3000 + (i * 50),
        }

        grid_data.append({"index": cell_index, "properties": properties})

    print(f"Created data for {len(grid_data)} cells")

    # Step 2: Spatial analysis
    print("\nStep 2: Spatial Analysis")

    analytics = SpatialAnalyticsInterface(backend="h3")

    # Population clustering
    data_pop = {
        "cells": [d["index"] for d in grid_data],
        "values": [d["properties"]["population"] for d in grid_data],
    }
    pop_hotspots = analytics.analyze_hotspots(data_pop, method="getis_ord")
    print(f"Population hotspots: {pop_hotspots.get('hotspot_count', 0)}")

    # Step 5: Integrated metrics (Manual Calculation)
    print("\nStep 5: Integrated Urban Metrics")

    # Calculate composite indicators
    composite_scores = []

    for item in grid_data:
        cell_index = item["index"]
        props = item["properties"]

        # Livability score (0-100)
        livability = (
            (props.get("green_space_pct", 0) / 30 * 25)  # Green space (25 points)
            + (min(props.get("transit_access", 0), 10) / 10 * 25)  # Transit (25 points)
            + (min(props.get("employment", 0), 1000) / 1000 * 25)  # Jobs (25 points)
            + (
                max(0, 100 - props.get("avg_rent", 3000) / 50) / 100 * 25
            )  # Affordability (25 points)
        )

        # Economic vitality score (0-100)
        vitality = (
            (
                min(props.get("employment", 0), 1000) / 1000 * 40
            )  # Employment (40 points)
            + (
                min(props.get("retail_sqft", 0), 10000) / 10000 * 30
            )  # Retail (30 points)
            + (
                min(props.get("population", 0), 1500) / 1500 * 30
            )  # Population (30 points)
        )

        composite_scores.append(
            {
                "cell_index": cell_index,
                "livability_score": livability,
                "vitality_score": vitality,
                "composite_score": (livability + vitality) / 2,
            }
        )

    # Summary statistics
    avg_livability = sum(s["livability_score"] for s in composite_scores) / len(
        composite_scores
    )
    avg_vitality = sum(s["vitality_score"] for s in composite_scores) / len(
        composite_scores
    )
    avg_composite = sum(s["composite_score"] for s in composite_scores) / len(
        composite_scores
    )

    print(f"Average livability score: {avg_livability:.1f}/100")
    print(f"Average economic vitality: {avg_vitality:.1f}/100")
    print(f"Average composite score: {avg_composite:.1f}/100")

    print("\nH3 orchestration workflow completed successfully!")
    print("=" * 40)


def main():
    """Run all H3 integration examples."""
    print("GEO-INFER-SPACE H3 Integration Examples")
    print("=" * 50)
    print()

    try:
        example_h3_vector_integration()
    except Exception as e:
        print(f"H3-Vector integration example failed: {e}\n")

    try:
        example_h3_density_clustering()
    except Exception as e:
        print(f"H3 density-clustering example failed: {e}\n")

    try:
        example_h3_temporal_analysis()
    except Exception as e:
        print(f"H3 temporal analysis example failed: {e}\n")

    try:
        example_h3_multi_resolution_analysis()
    except Exception as e:
        print(f"H3 multi-resolution example failed: {e}\n")

    try:
        example_h3_orchestration_workflow()
    except Exception as e:
        print(f"H3 orchestration workflow example failed: {e}\n")

    print("All H3 integration examples completed!")


if __name__ == "__main__":
    main()
