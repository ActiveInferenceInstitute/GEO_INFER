#!/usr/bin/env python3
"""
H3 Advanced Applications Examples.

Real-world examples demonstrating H3 machine learning integration,
disaster response, and performance optimization based on unified spatial architecture.
"""

import logging
import random
from datetime import datetime

import pandas as pd

from geo_infer_space.core import SpatialIndexingInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_demand_forecasting_ml():
    """
    Example: H3-based demand forecasting for ride-sharing.

    Demonstrates ML feature engineering using H3 hexagonal grids
    for demand prediction using unified spatial interfaces.
    """
    print("\n" + "=" * 60)
    print("H3 Demand Forecasting ML Example")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Simulate ride demand data across SF
    sf_area = [
        (37.7749, -122.4194),  # Downtown SF
        (37.7849, -122.4094),  # North Beach
        (37.7649, -122.4294),  # Mission
        (37.7949, -122.3994),  # Financial District
        (37.7549, -122.4394),  # Castro
        (37.7449, -122.4494),  # Sunset
        (37.7349, -122.4594),  # Richmond
        (37.8049, -122.4194),  # Russian Hill
    ]

    grid_cells = {}

    print("Creating demand forecasting grid...")

    # Create cells with realistic demand patterns
    for i, (lat, lng) in enumerate(sf_area):
        cell_index = indexer.latlng_to_cell(lat, lng, 9)

        # Simulate demand based on area characteristics
        base_demand = 50
        if i in [0, 3]:  # Downtown and Financial District
            demand_multiplier = 3.0
        elif i in [1, 7]:  # North Beach and Russian Hill
            demand_multiplier = 2.0
        else:
            demand_multiplier = 1.0

        # Add time-based variation
        hour = (datetime.now().hour + i) % 24
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            time_multiplier = 1.5
        elif 22 <= hour or hour <= 5:  # Night hours
            time_multiplier = 0.3
        else:
            time_multiplier = 1.0

        demand = int(base_demand * demand_multiplier * time_multiplier)
        supply = int(
            demand * 0.8 + random.randint(-10, 10)
        )  # Supply slightly less than demand

        grid_cells[cell_index] = {
            "demand": demand,
            "supply": supply,
            "gap": demand - supply,
            "surge_multiplier": (
                max(1.0, 1.0 + ((demand - supply) / 100)) if demand > supply else 1.0
            ),
        }

    # ML Feature Engineering simulation
    features = []
    print("\nFeature Engineering for ML:")

    for cell_index, data in grid_cells.items():
        # Get spatial context (neighbors)
        neighbors = indexer.get_cell_neighbors(cell_index, k=1)

        # Create features
        feature_vector = {
            "cell_id": cell_index,
            "hour_of_day": datetime.now().hour,
            "historical_demand": data["demand"] * 0.9,
            "weather_condition": "Clear",
            "is_holiday": False,
            "poi_density": len(neighbors) * 5,
            "nearby_supply": sum(
                [grid_cells.get(n, {}).get("supply", 0) for n in neighbors]
            ),
        }

        features.append(feature_vector)

    df = pd.DataFrame(features)
    print(f"Generated {len(df)} feature vectors")
    print("Sample Features:")
    print(df.head(2))

    print("\nDemand forecasting example completed successfully")


def example_disaster_response_system():
    """
    Example: H3-based disaster response coordination.

    Uses H3 grids to coordinate emergency response units
    and identify impact zones.
    """
    print("\n" + "=" * 60)
    print("H3 Disaster Response System Example")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Simulate an incident (e.g., fire)
    incident_loc = (37.7649, -122.4294)  # Mission district
    print(f"🚨 Incident reported at {incident_loc}")

    # Identify impact zone (k=2 rings)
    center_cell = indexer.latlng_to_cell(incident_loc[0], incident_loc[1], 9)
    impact_zone = indexer.get_cell_neighbors(center_cell, k=2)

    print(f"⚠️  Impact zone established: {len(impact_zone)} cells affected")

    # Resource allocation
    resources = {
        "Fire Station 1": (37.7749, -122.4194),
        "Fire Station 2": (37.7549, -122.4394),
        "Hospital": (37.7849, -122.4094),
    }

    print("\nDispatch Analysis:")
    for name, loc in resources.items():
        res_cell = indexer.latlng_to_cell(loc[0], loc[1], 9)
        distance = indexer.get_cell_distance(center_cell, res_cell)

        status = "🟢 Within range" if distance < 5 else "🔴 Out of range"
        print(f"  {name}: {distance} hops away - {status}")

    print("\nDisaster response example completed successfully")


def example_performance_optimization():
    """
    Example: Performance optimization strategies.

    Demonstrates resolution tuning and batch processing.
    """
    print("\n" + "=" * 60)
    print("H3 Performance Optimization (Simulation)")
    print("=" * 60)

    # Resolution optimization scenarios
    print("\nResolution Recommendations:")

    scenarios = [
        {"name": "City-wide Analysis", "area_km2": 600, "type": "general"},
        {"name": "ML Demand Forecasting", "area_km2": 200, "type": "ml"},
        {"name": "Real-time Visualization", "area_km2": 100, "type": "visualization"},
        {"name": "Routing", "area_km2": 50, "type": "routing"},
    ]

    # Heuristic based recommendations
    # Res 7: ~5 km2, Res 8: ~0.7 km2, Res 9: ~0.1 km2, Res 10: ~0.015 km2

    for scenario in scenarios:
        rec_res = 8
        if scenario["type"] == "routing":
            rec_res = 10
        elif scenario["type"] == "ml":
            rec_res = 9
        elif scenario["type"] == "visualization":
            rec_res = 8
        else:  # general
            rec_res = 7

        est_cells = int(scenario["area_km2"] / (0.1 * (7 ** (9 - rec_res))))

        print(f"  {scenario['name']} ({scenario['area_km2']} km²):")
        print(f"    Recommended Resolution: {rec_res}")
        print(f"    Est. Cell Count: {est_cells:,}")


def example_integrated_smart_city():
    """
    Example: Integrated smart city application.

    Combines analytics for a holistic view.
    """
    print("\n" + "=" * 60)
    print("H3 Integrated Smart City Example")
    print("=" * 60)

    indexer = SpatialIndexingInterface(backend="h3")

    # Define city districts
    city_districts = [
        {"name": "Downtown", "coords": (37.7749, -122.4194), "type": "business"},
        {
            "name": "Residential North",
            "coords": (37.7849, -122.4094),
            "type": "residential",
        },
        {"name": "Industrial", "coords": (37.7649, -122.4294), "type": "industrial"},
        {"name": "Suburbs", "coords": (37.7549, -122.4394), "type": "suburban"},
    ]

    grid_cells = {}

    for district in city_districts:
        lat, lng = district["coords"]
        cell_index = indexer.latlng_to_cell(lat, lng, 9)

        # Base stats
        if district["type"] == "business":
            pop = 800
            energy = 200
            traffic = 0.8
        elif district["type"] == "residential":
            pop = 2000
            energy = 80
            traffic = 0.4
        elif district["type"] == "industrial":
            pop = 200
            energy = 300
            traffic = 0.5
        else:
            pop = 1500
            energy = 100
            traffic = 0.3

        grid_cells[cell_index] = {
            "name": district["name"],
            "type": district["type"],
            "population": pop,
            "energy_consumption": energy,  # kWh per capita
            "traffic_congestion": traffic,
            "cell_index": cell_index,
        }

    print(f"Initialized smart city grid with {len(grid_cells)} districts")

    # Dashboard Metrics
    print("\nSmart City Dashboard Metrics:")

    total_pop = sum(c["population"] for c in grid_cells.values())
    avg_traffic = sum(c["traffic_congestion"] for c in grid_cells.values()) / len(
        grid_cells
    )
    total_energy = sum(
        c["population"] * c["energy_consumption"] for c in grid_cells.values()
    )

    print(f"  Total Population: {total_pop:,}")
    print(f"  Avg Traffic Congestion: {avg_traffic:.2f} (0-1)")
    print(f"  Total Energy Demand: {total_energy:,} kWh")

    # District Recommendations
    print("\nDistrict Recommendations:")
    for cell in grid_cells.values():
        print(f"  {cell['name']} ({cell['type']}):")
        if cell["traffic_congestion"] > 0.6:
            print("    - ⚠️ High traffic: Optimize signal timing")
        if cell["energy_consumption"] > 150:
            print("    - ⚡ High energy use: Promote efficiency programs")
        if cell["population"] > 1800:
            print("    - 👥 Dense population: Ensure adequate services")

    print("\nSmart city example completed successfully")


def main():
    """Run all advanced H3 application examples."""
    print("GEO-INFER-SPACE H3 Advanced Applications")
    print("=" * 60)
    print()

    try:
        example_demand_forecasting_ml()
    except Exception as e:
        print(f"Demand forecasting ML example failed: {e}\n")

    try:
        example_disaster_response_system()
    except Exception as e:
        print(f"Disaster response example failed: {e}\n")

    try:
        example_performance_optimization()
    except Exception as e:
        print(f"Performance optimization example failed: {e}\n")

    try:
        example_integrated_smart_city()
    except Exception as e:
        print(f"Integrated smart city example failed: {e}\n")

    print("All H3 advanced application examples completed!")


if __name__ == "__main__":
    main()
