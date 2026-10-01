#!/usr/bin/env python3
"""
GEO-INFER-SIM Example: Urban Growth Simulation

A grid land-use model driven by ``SimulationEngine``: each yearly step lets
developers convert undeveloped cells adjacent to existing development into
residential land and households choose a residential cell. All randomness
comes from the engine's seeded generator, so a run is reproducible.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from geo_infer_sim import SimulationConfig, SimulationEngine

GRID_SIZE = 50
YEARS = 20
UNDEVELOPED, RESIDENTIAL, COMMERCIAL = 0, 1, 2


@dataclass
class Developer:
    """Development agent converting land next to existing development."""

    developer_id: str
    capital: float
    properties_developed: int = 0

    def act(self, land_use: np.ndarray, rng: np.random.Generator) -> None:
        """Develop one randomly chosen undeveloped cell if it borders development."""
        rows, cols = np.nonzero(land_use == UNDEVELOPED)
        if rows.size == 0:
            return
        idx = int(rng.integers(rows.size))
        i, j = int(rows[idx]), int(cols[idx])
        neighbourhood = land_use[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
        if np.any(neighbourhood > UNDEVELOPED):
            land_use[i, j] = RESIDENTIAL
            self.properties_developed += 1
            self.capital -= 100_000


@dataclass
class Household:
    """Household agent choosing a residential cell."""

    household_id: str
    income: float
    location: tuple[int, int] | None = None

    def act(self, land_use: np.ndarray, rng: np.random.Generator) -> None:
        """Move to a randomly chosen residential cell."""
        rows, cols = np.nonzero(land_use == RESIDENTIAL)
        if rows.size:
            idx = int(rng.integers(rows.size))
            self.location = (int(rows[idx]), int(cols[idx]))


def initial_land_use(rng: np.random.Generator) -> np.ndarray:
    """Commercial core with a 30 % residential ring around it."""
    land_use = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    centre = GRID_SIZE // 2
    ring = land_use[centre - 10 : centre + 10, centre - 10 : centre + 10]
    ring[rng.random(ring.shape) < 0.3] = RESIDENTIAL
    land_use[centre - 5 : centre + 5, centre - 5 : centre + 5] = COMMERCIAL
    return land_use


def main() -> None:
    print("=" * 60)
    print("GEO-INFER-SIM: Urban Growth Simulation")
    print("=" * 60)

    engine = SimulationEngine(
        SimulationConfig(time_step=1.0, max_time=float(YEARS), random_seed=42)
    )
    rng = engine.rng

    print("\n1. Setting Up Urban Environment...")
    land_use = initial_land_use(rng)
    developed = int(np.sum(land_use > UNDEVELOPED))
    print(f"   Grid: {GRID_SIZE}x{GRID_SIZE} cells (1 km)")
    print(
        f"   Initial developed: {developed} cells "
        f"({100 * developed / GRID_SIZE**2:.1f}%)"
    )

    print("\n2. Creating Agents...")
    developers = [Developer(f"DEV_{i}", capital=5_000_000) for i in range(10)]
    households = [
        Household(f"HH_{i}", income=float(rng.uniform(30_000, 150_000)))
        for i in range(500)
    ]
    print(f"   Developers: {len(developers)}")
    print(f"   Households: {len(households)}")

    def step(time: float, state: dict[str, Any]) -> dict[str, Any]:
        for developer in developers:
            developer.act(land_use, rng)
        for household in households:
            household.act(land_use, rng)
        return {
            "year": int(time) + 1,
            "developed": int(np.sum(land_use > UNDEVELOPED)),
            "residential": int(np.sum(land_use == RESIDENTIAL)),
            "commercial": int(np.sum(land_use == COMMERCIAL)),
        }

    print("\n3. Running Urban Growth Simulation...")
    engine.initialize({"year": 0, "developed": developed})
    engine.run(step)
    history = [entry["state"] for entry in engine.state_history[1:]]
    for record in history:
        if record["year"] % 5 == 0:
            print(
                f"   Year {record['year']}: {record['developed']} cells developed "
                f"({100 * record['developed'] / GRID_SIZE**2:.1f}%)"
            )

    print("\n4. Analyzing Simulation Results...")
    names = {
        UNDEVELOPED: "undeveloped",
        RESIDENTIAL: "residential",
        COMMERCIAL: "commercial",
    }
    print("\n   Final Land Use Distribution:")
    for code, name in names.items():
        count = int(np.sum(land_use == code))
        print(f"   - {name.title()}: {count} cells ({100 * count / GRID_SIZE**2:.1f}%)")

    growth = history[-1]["developed"] - developed
    print("\n   Growth Statistics:")
    print(f"   - Total growth: {growth} cells")
    print(f"   - Annual growth rate: {growth / YEARS:.1f} cells/year")
    print(f"   - Population housed: ~{history[-1]['residential'] * 50}")
    housed = sum(household.location is not None for household in households)
    print(f"   - Households with a residential location: {housed}")
    print(
        f"   - Properties developed: {sum(d.properties_developed for d in developers)}"
    )

    print("\n" + "=" * 60)
    print("Urban growth simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
