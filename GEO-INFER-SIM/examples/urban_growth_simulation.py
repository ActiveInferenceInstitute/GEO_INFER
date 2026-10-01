#!/usr/bin/env python3
"""
GEO-INFER-SIM Example: Urban Growth Simulation

This example demonstrates an agent-based urban growth simulation
with population dynamics, land use change, and policy analysis.
"""

import numpy as np

from geo_infer_sim import SimulationEngine, Agent, Environment


def main():
    print("=" * 60)
    print("GEO-INFER-SIM: Urban Growth Simulation")
    print("=" * 60)

    # 1. Define Urban Environment
    print("\n1. Setting Up Urban Environment...")

    # Create a grid-based environment (100x100 cells, 1km resolution)
    grid_size = 50
    environment = Environment(
        name="MetroCity", dimensions=(grid_size, grid_size), cell_size_km=1.0
    )

    # Initialize land use (0=undeveloped, 1=residential, 2=commercial, 3=industrial, 4=protected)
    land_use = np.zeros((grid_size, grid_size))

    # Create city center
    center_x, center_y = grid_size // 2, grid_size // 2
    for i in range(center_x - 5, center_x + 5):
        for j in range(center_y - 5, center_y + 5):
            if 0 <= i < grid_size and 0 <= j < grid_size:
                land_use[i, j] = 2  # Commercial

    # Add some initial residential
    for i in range(center_x - 10, center_x + 10):
        for j in range(center_y - 10, center_y + 10):
            if land_use[i, j] == 0:
                if np.random.random() < 0.3:
                    land_use[i, j] = 1  # Residential

    environment.set_grid("land_use", land_use)

    # Calculate initial development
    developed = np.sum(land_use > 0)
    print(f"   Grid size: {grid_size}x{grid_size} ({grid_size**2} cells)")
    print(
        f"   Initial developed: {developed} cells ({100 * developed / grid_size**2:.1f}%)"
    )

    # 2. Create Simulation Engine
    print("\n2. Initializing Simulation Engine...")

    config = {"name": "UrbanGrowthSimulation", "time_step": "year", "random_seed": 42}

    engine = SimulationEngine(config)

    print(f"   Simulation: {config['name']}")
    print(f"   Time step: {config['time_step']}")

    # 3. Define Agent Classes
    print("\n3. Defining Agent Types...")

    class Developer(Agent):
        """Property development agent."""

        def __init__(self, agent_id, capital=1000000):
            self.id = agent_id
            self.capital = capital
            self.properties_developed = 0

        def step(self, environment):
            """Choose locations to develop."""
            land_use = environment.get_grid("land_use")
            undeveloped = np.where(land_use == 0)

            if len(undeveloped[0]) > 0:
                # Find locations near existing development
                idx = np.random.randint(len(undeveloped[0]))
                i, j = undeveloped[0][idx], undeveloped[1][idx]

                # Check if near developed area
                if self._has_neighbor_development(land_use, i, j):
                    land_use[i, j] = 1  # Build residential
                    self.properties_developed += 1
                    self.capital -= 100000

            environment.set_grid("land_use", land_use)

        def _has_neighbor_development(self, land_use, i, j):
            """Check if cell has developed neighbors."""
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < land_use.shape[0] and 0 <= nj < land_use.shape[1]:
                        if land_use[ni, nj] > 0:
                            return True
            return False

    class Household(Agent):
        """Household agent with location preferences."""

        def __init__(self, agent_id, income=50000):
            self.id = agent_id
            self.income = income
            self.location = None

        def step(self, environment):
            """Move to preferred location."""
            land_use = environment.get_grid("land_use")
            residential = np.where(land_use == 1)

            if len(residential[0]) > 0:
                idx = np.random.randint(len(residential[0]))
                self.location = (residential[0][idx], residential[1][idx])

    # Create agents
    developers = [Developer(f"DEV_{i}", capital=5000000) for i in range(10)]
    households = [
        Household(f"HH_{i}", income=np.random.uniform(30000, 150000))
        for i in range(500)
    ]

    print(f"   Developers: {len(developers)}")
    print(f"   Households: {len(households)}")

    # 4. Run Simulation
    print("\n4. Running Urban Growth Simulation...")

    years = 20
    history = []

    for year in range(years):
        # Developers act
        for developer in developers:
            developer.step(environment)

        # Households choose locations
        for household in households:
            household.step(environment)

        # Record state
        land_use = environment.get_grid("land_use")
        developed = np.sum(land_use > 0)
        residential = np.sum(land_use == 1)
        commercial = np.sum(land_use == 2)

        history.append(
            {
                "year": year + 1,
                "developed": developed,
                "residential": residential,
                "commercial": commercial,
            }
        )

        if (year + 1) % 5 == 0:
            print(
                f"   Year {year + 1}: {developed} cells developed ({100 * developed / grid_size**2:.1f}%)"
            )

    # 5. Analyze Results
    print("\n5. Analyzing Simulation Results...")

    final_land_use = environment.get_grid("land_use")

    # Land use breakdown
    land_use_counts = {
        "undeveloped": np.sum(final_land_use == 0),
        "residential": np.sum(final_land_use == 1),
        "commercial": np.sum(final_land_use == 2),
        "industrial": np.sum(final_land_use == 3),
        "protected": np.sum(final_land_use == 4),
    }

    print("\n   Final Land Use Distribution:")
    for use_type, count in land_use_counts.items():
        pct = 100 * count / grid_size**2
        print(f"   - {use_type.title()}: {count} cells ({pct:.1f}%)")

    # Growth statistics
    initial_developed = history[0]["developed"]
    final_developed = history[-1]["developed"]
    growth_rate = (final_developed - initial_developed) / years

    print("\n   Growth Statistics:")
    print(f"   - Total growth: {final_developed - initial_developed} cells")
    print(f"   - Annual growth rate: {growth_rate:.1f} cells/year")
    print(f"   - Population housed: ~{int(history[-1]['residential'] * 50)}")

    # Developer performance
    total_properties = sum(d.properties_developed for d in developers)
    print(f"   - Properties developed: {total_properties}")

    # 6. Spatial Metrics
    print("\n6. Computing Spatial Metrics...")

    # Calculate urban sprawl index (ratio of developed to footprint)
    developed_mask = final_land_use > 0
    if np.any(developed_mask):
        rows = np.any(developed_mask, axis=1)
        cols = np.any(developed_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bounding_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        sprawl_index = np.sum(developed_mask) / bounding_area
    else:
        sprawl_index = 0

    print(f"   Urban compactness index: {sprawl_index:.3f}")
    print(f"   Development density: {final_developed / (grid_size**2):.3f}")

    print("\n" + "=" * 60)
    print("Urban Growth Simulation Complete!")
    print("=" * 60)

    # Summary
    print("\nSimulation Summary:")
    print(f"  - Duration: {years} years")
    print(f"  - Final development: {100 * final_developed / grid_size**2:.1f}%")
    print(f"  - Annual growth: {growth_rate:.1f} cells/year")
    print(f"  - Urban compactness: {sprawl_index:.2f}")


if __name__ == "__main__":
    main()
