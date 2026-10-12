#!/usr/bin/env python3
"""
Comprehensive H3 Examples for GEO-INFER-SPACE.

This script demonstrates real-world applications of H3 hexagonal grid operations
with visualizations, analytics, and practical use cases using the unified spatial architecture.

Run with: python examples/h3_comprehensive_examples.py
"""

import sys
import numpy as np
from pathlib import Path
import logging

import geopandas as gpd
import h3
from shapely.geometry import Polygon

from geo_infer_space.core import SpatialIndexingInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_basic_h3_operations():
    """
    Example 1: Basic H3 Operations
    Demonstrates fundamental H3 operations with real coordinates using the unified interface.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: BASIC H3 OPERATIONS")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Real-world locations
    locations = {
        "San Francisco": (37.7749, -122.4194),
        "New York": (40.7128, -74.0060),
        "London": (51.5074, -0.1278),
        "Tokyo": (35.6762, 139.6503),
    }

    print("Converting coordinates to H3 cells at different resolutions:")
    print("-" * 60)

    for city, (lat, lng) in locations.items():
        print(f"\n{city}: ({lat}, {lng})")

        for resolution in [7, 8, 9, 10]:
            # Convert to H3 cell
            cell_index = indexer.latlng_to_cell(lat, lng, resolution)

            # Get cell properties
            try:
                cell_lat, cell_lng = indexer.cell_to_latlng(cell_index)
                # Approximate area: Res 9 is ~0.1 km2, scaling by 7 for each resolution step roughly
                # Or use library if available
                area_est = 0.1 * (7 ** (9 - resolution))

                print(
                    f"  Resolution {resolution:2d}: {cell_index} | Est. Area: {area_est:.6f} km²"
                )
            except Exception as e:
                print(f"  Resolution {resolution:2d}: Error - {e}")

    print("\nTesting grid operations:")
    print("-" * 30)

    # Use San Francisco for grid operations
    sf_cell = indexer.latlng_to_cell(37.7749, -122.4194, 9)
    print(f"SF Center Cell: {sf_cell}")

    # Get neighbors
    neighbors = indexer.get_cell_neighbors(sf_cell, k=1)
    print(f"Direct neighbors: {len(neighbors)}")

    # Get the grid disk within distance 2.
    disk_radius_2 = indexer.get_cell_neighbors(sf_cell, k=2)
    print(f"Distance-2 grid disk: {len(disk_radius_2)} cells")

    # Test distance calculation
    if neighbors:
        distance = indexer.get_cell_distance(sf_cell, neighbors[0])
        print(f"Distance to first neighbor: {distance} cells")


def example_2_city_coverage_analysis():
    """
    Example 2: City Coverage Analysis
    Demonstrates H3 grid creation and analysis for urban areas.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CITY COVERAGE ANALYSIS")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Define city boundaries (GeoJSON-like Polygons)
    cities = {
        "San Francisco": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.5144, 37.8044],  # Northwest
                    [-122.3549, 37.8044],  # Northeast
                    [-122.3549, 37.7049],  # Southeast
                    [-122.5144, 37.7049],  # Southwest
                    [-122.5144, 37.8044],  # Close loop
                ]
            ],
        },
        "Manhattan": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-73.9442, 40.8176],  # North
                    [-73.9734, 40.8176],  # Northwest
                    [-73.9734, 40.7047],  # Southwest
                    [-73.9442, 40.7047],  # Southeast
                    [-73.9442, 40.8176],  # Close loop
                ]
            ],
        },
    }

    for city_name, boundary in cities.items():
        print(f"\nAnalyzing {city_name}:")
        print("-" * 40)

        # Create H3 grids at different resolutions
        for resolution in [7, 8, 9]:
            cells = indexer.polygon_to_cells(boundary, resolution)

            print(f"Resolution {resolution}: {len(cells)} cells")

            # Simple stats
            area_est = len(cells) * (0.1 * (7 ** (9 - resolution)))
            print(f"  Est. Total area: {area_est:.2f} km²")

        print()


def example_3_transportation_corridor():
    """
    Example 3: Transportation Corridor Analysis
    Demonstrates H3 for transportation network analysis.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: TRANSPORTATION CORRIDOR ANALYSIS")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Define transportation corridors
    corridors = {
        "SF_to_Oakland": {
            "start": (37.7749, -122.4194),  # San Francisco
            "end": (37.8044, -122.2712),  # Oakland
            "name": "SF-Oakland Corridor",
        }
    }

    for corridor_data in corridors.values():
        print(f"\nAnalyzing {corridor_data['name']}:")
        print("-" * 50)

        start_coords = corridor_data["start"]
        end_coords = corridor_data["end"]

        # Create cells for start and end points
        start_cell = indexer.latlng_to_cell(*start_coords, 9)
        end_cell = indexer.latlng_to_cell(*end_coords, 9)

        print(f"Start cell: {start_cell}")
        print(f"End cell: {end_cell}")

        # Calculate direct distance
        try:
            direct_distance = indexer.get_cell_distance(start_cell, end_cell)
            print(f"Grid distance: {direct_distance} cells")

            # Simple path finding simulation (line between cells)
            # In a real scenario, use h3.grid_path_cells if wrapped, or implement generic A*
            try:
                path_cells = h3.grid_path_cells(start_cell, end_cell)
                print(f"Path length: {len(path_cells)} cells")

                # Buffer logic simulated by neighbors
                corridor_cells = set()
                for cell in path_cells:
                    neighbors = indexer.get_cell_neighbors(cell, k=2)
                    corridor_cells.update(neighbors)

                print(f"Corridor cells (with buffer): {len(corridor_cells)}")
            except Exception as e:
                print(f"H3 path finding error: {e}")

        except Exception as e:
            print(f"Distance/Path failed: {e}")


def example_4_retail_catchment_analysis():
    """
    Example 4: Retail Catchment Analysis
    Demonstrates H3 for retail location analysis and market coverage.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: RETAIL CATCHMENT ANALYSIS")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Define store locations in San Francisco
    stores = {
        "Downtown_SF": {
            "location": (37.7749, -122.4194),
            "type": "flagship",
            "catchment_km": 3,
        },
        "Mission_District": {
            "location": (37.7599, -122.4148),
            "type": "neighborhood",
            "catchment_km": 2,
        },
    }

    all_catchment_cells = set()

    print("Creating catchment areas for stores:")
    print("-" * 40)

    for store_id, store_data in stores.items():
        lat, lng = store_data["location"]
        catchment_km = store_data["catchment_km"]

        # Use a simple three-grid-steps-per-kilometer catchment heuristic.
        # This is illustrative rather than a geodesic distance guarantee.
        disk_radius = max(1, int(catchment_km * 3))

        center_cell = indexer.latlng_to_cell(lat, lng, 9)
        catchment_cells = indexer.get_cell_neighbors(center_cell, k=disk_radius)

        all_catchment_cells.update(catchment_cells)

        print(f"{store_id}:")
        print(f"  Location: ({lat:.4f}, {lng:.4f})")
        print(f"  Catchment cells: {len(catchment_cells)}")
        print(f"  Store type: {store_data['type']}")

    print(f"\nTotal unique catchment cells: {len(all_catchment_cells)}")


def example_5_environmental_monitoring():
    """
    Example 5: Environmental Monitoring Grid
    Demonstrates H3 for environmental data collection and analysis.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: ENVIRONMENTAL MONITORING GRID")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Define monitoring area (San Francisco Bay) - simplified box
    bay_area_polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.6, 37.9],
                [-121.9, 37.9],
                [-121.9, 37.3],
                [-122.6, 37.3],
                [-122.6, 37.9],
            ]
        ],
    }

    print("Creating environmental monitoring grids:")
    print("-" * 45)

    # Create monitoring grids at different resolutions
    monitoring_data = {}

    for resolution in [6, 7]:
        cells = indexer.polygon_to_cells(bay_area_polygon, resolution)
        monitoring_data[resolution] = []

        print(f"Resolution {resolution}: {len(cells)} monitoring stations")

        # Simulate environmental data
        np.random.seed(42)  # For reproducible results

        for cell_index in cells:
            data = {
                "cell_index": cell_index,
                "pm25": np.random.normal(15, 5),
                "ozone": np.random.normal(0.08, 0.02),
            }
            monitoring_data[resolution].append(data)

        # Simple analysis
        pm25_values = [d["pm25"] for d in monitoring_data[resolution]]
        print(f"  PM2.5 range: {min(pm25_values):.1f} - {max(pm25_values):.1f} μg/m³")


def example_6_disaster_response_planning():
    """
    Example 6: Disaster Response Planning
    Demonstrates H3 for emergency response and evacuation planning.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: DISASTER RESPONSE PLANNING")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Emergency response center
    emergency_center = (37.7749, -122.4194)  # San Francisco City Hall
    center_cell = indexer.latlng_to_cell(*emergency_center, 8)

    print(f"Emergency Response Center: ({emergency_center[0]}, {emergency_center[1]})")
    print("-" * 60)

    # Define response zones
    response_zones = {"immediate": {"k": 2, "time": 5}, "primary": {"k": 5, "time": 15}}

    zone_cells = {}

    for zone_name, config in response_zones.items():
        print(f"\n{zone_name.upper()} RESPONSE ZONE:")

        cells = indexer.get_cell_neighbors(center_cell, k=config["k"])
        zone_cells[zone_name] = set(cells)

        print(f"Zone cells: {len(cells)}")
        print(f"Response time: {config['time']} minutes")

    # Zone overlap analysis
    immediate = zone_cells["immediate"]
    primary = zone_cells["primary"]

    # Immediate should be subset of primary
    overlap = len(immediate.intersection(primary))
    print(f"\nOverlap (Immediate in Primary): {overlap}/{len(immediate)}")


def example_7_visualization_showcase():
    """
    Example 7: Visualization Showcase
    Demonstrates various data export for visualization.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: VISUALIZATION SHOWCASE")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Create sample grid with data
    center_coords = (37.7749, -122.4194)  # San Francisco
    center_cell = indexer.latlng_to_cell(*center_coords, 9)
    sample_cells = indexer.get_cell_neighbors(center_cell, k=3)

    data = []
    np.random.seed(42)

    for i, cell_index in enumerate(sample_cells):
        data.append(
            {
                "cell_index": cell_index,
                "value": np.random.normal(100, 20),
                "category": ["A", "B", "C"][i % 3],
            }
        )

    print(f"Created sample data with {len(data)} cells")

    # Create output directory
    output_dir = Path("output/h3_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. GeoJSON/Shapefile Export via GeoPandas
    print("\n1. Exporting to GeoJSON via GeoPandas...")

    geometries = []
    rows = []
    for item in data:
        try:
            boundary = indexer.get_cell_boundary(item["cell_index"])
            # Shapely polygon: (lng, lat)
            poly = Polygon([(lng, lat) for lat, lng in boundary])
            geometries.append(poly)
            rows.append(item)
        except Exception as e:
            print(f"Error creating geometry for {item['cell_index']}: {e}")

    if geometries:
        gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
        geojson_path = output_dir / "sample_grid.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"   Saved: {geojson_path}")

        # Simple static plot
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 10))
            gdf.plot(column="value", ax=ax, legend=True)
            plt.title("H3 Grid Visualization")
            png_path = output_dir / "grid_plot.png"
            plt.savefig(png_path)
            print(f"   Saved: {png_path}")
        except ImportError:
            print("   Matplotlib not available for plotting.")

    print(f"\nVisualization showcase complete! Check {output_dir} for outputs.")


def example_8_performance_benchmarks():
    """
    Example 8: Performance Benchmarks
    Demonstrates H3 performance with large datasets using unified interface.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 8: PERFORMANCE BENCHMARKS")
    print("=" * 60)

    import time

    indexer = SpatialIndexingInterface(backend="h3")

    # Performance test configurations
    test_configs = [
        {"name": "Medium Ring", "k": 10, "resolution": 8},
        {"name": "Large Ring", "k": 20, "resolution": 8},
    ]

    center_coords = (37.7749, -122.4194)
    center_cell = indexer.latlng_to_cell(*center_coords, 8)

    print("Grid Creation Performance:")
    print("-" * 30)

    for config in test_configs:
        print(f"\n{config['name']} (k={config['k']}, res={config['resolution']}):")

        # Time neighbor finding
        start_time = time.time()
        cells = indexer.get_cell_neighbors(center_cell, k=config["k"])
        creation_time = time.time() - start_time

        print(f"  Cells found: {len(cells)}")
        print(f"  Time: {creation_time:.4f} seconds")


def main():
    """Run all H3 comprehensive examples."""
    print("GEO-INFER-SPACE H3 Comprehensive Examples")
    print("=" * 60)

    try:
        example_1_basic_h3_operations()
        example_2_city_coverage_analysis()
        example_3_transportation_corridor()
        example_4_retail_catchment_analysis()
        example_5_environmental_monitoring()
        example_6_disaster_response_planning()
        example_7_visualization_showcase()
        example_8_performance_benchmarks()

        print("\n" + "=" * 60)
        print("✅ All comprehensive examples completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
