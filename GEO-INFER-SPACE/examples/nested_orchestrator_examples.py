#!/usr/bin/env python3
"""
Comprehensive Nested H3 Orchestrator Examples.

This module demonstrates the full capabilities of the system using the unified
spatial architecture, focusing on real-world scenarios like urban planning,
environmental monitoring, and supply chain optimization.
"""

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np


# Unified Spatial Architecture
from geo_infer_space.core import (
    SpatialIndexingInterface,
    SpatialAnalyticsInterface,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Visualization imports
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - visualizations will be limited")

RNG = np.random.default_rng(42)


class NestedH3Orchestrator:
    """
    Comprehensive orchestrator for spatial systems.

    Demonstrates full workflow capabilities including:
    - System creation and management via Unified Interfaces
    - Spatial Analysis
    - Data Aggregation and Messaging Simulation
    """

    def __init__(self, name: str = "NestedH3Orchestrator"):
        """Initialize the orchestrator."""
        self.name = name
        self.created_at = datetime.now()

        # Core interfaces
        self.indexer = SpatialIndexingInterface(backend="h3")
        try:
            self.analyzer = SpatialAnalyticsInterface(backend="h3")
        except Exception:
            self.analyzer = None

        # Results storage
        self.results = {}

        print(f"🚀 {self.name} initialized at {self.created_at}")

    def scenario_1_urban_planning(self) -> dict[str, Any]:
        """
        Scenario 1: Urban Planning and Development Analysis
        """
        print("\n" + "=" * 60)
        print("🏙️  SCENARIO 1: URBAN PLANNING ANALYSIS")
        print("=" * 60)

        scenario_results = {
            "scenario": "urban_planning",
            "start_time": datetime.now().isoformat(),
            "components_used": ["SpatialIndexingInterface"],
            "data_outputs": {},
            "visualizations": [],
        }

        # 1. Create urban grid system
        print("📍 Creating urban grid system...")

        # Simulation of an urban area (San Francisco)
        urban_data = {}
        center_lat, center_lng = 37.7749, -122.4194

        # Create a grid around center
        # We simulate ~50 cells
        cells = []
        for i in range(50):
            lat = center_lat + RNG.uniform(-0.05, 0.05)
            lng = center_lng + RNG.uniform(-0.05, 0.05)
            cell_index = self.indexer.latlng_to_cell(lat, lng, 9)

            # Attributes
            pop = int(RNG.integers(100, 5000))
            density = pop / 0.1  # Approx density
            d_type = ["residential", "commercial", "industrial", "mixed", "park"][i % 5]

            urban_data[cell_index] = {
                "population": pop,
                "density": density,
                "type": d_type,
                "infrastructure": RNG.uniform(0.3, 1.0),
                "lat": lat,
                "lng": lng,
            }
            cells.append(cell_index)

        print(f"✅ Created urban system with {len(urban_data)} cells")

        # Stats
        total_pop = sum(d["population"] for d in urban_data.values())
        avg_density = sum(d["density"] for d in urban_data.values()) / len(urban_data)

        scenario_results["data_outputs"]["urban_system"] = {
            "cell_count": len(urban_data),
            "population": total_pop,
            "avg_density": avg_density,
        }

        # 2. Boundary / Clustering Analysis (Simulated via Unified Interface if available)
        # Using simple attribute based grouping (Lumping simulation)
        print("🔄 Lumping similar urban areas (Simulation)...")
        zones = {}
        for c, data in urban_data.items():
            t = data["type"]
            if t not in zones:
                zones[t] = []
            zones[t].append(c)

        print(f"✅ Grouped cells into {len(zones)} zones based on type")
        scenario_results["data_outputs"]["zoning"] = {
            k: len(v) for k, v in zones.items()
        }

        # 3. Visualization
        if MATPLOTLIB_AVAILABLE:
            print("📊 Creating urban planning visualizations...")
            try:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # Pop Density
                lats = [d["lat"] for d in urban_data.values()]
                lngs = [d["lng"] for d in urban_data.values()]
                pops = [d["population"] for d in urban_data.values()]

                sc = axes[0].scatter(
                    lngs, lats, c=pops, cmap="YlOrRd", s=100, alpha=0.7
                )
                plt.colorbar(sc, ax=axes[0], label="Population")
                axes[0].set_title("Population Distribution")
                axes[0].set_xlabel("Longitude")
                axes[0].set_ylabel("Latitude")

                # Zoning
                types = [d["type"] for d in urban_data.values()]
                unique_types = list(set(types))
                type_map = {t: i for i, t in enumerate(unique_types)}
                colors = [type_map[t] for t in types]

                axes[1].scatter(lngs, lats, c=colors, cmap="tab10", s=100, alpha=0.7)
                axes[1].set_title("Zoning Types")

                viz_filename = (
                    f"urban_planning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.tight_layout()
                plt.savefig(viz_filename)
                scenario_results["visualizations"].append(viz_filename)
                print(f"✅ Saved visualization: {viz_filename}")

            except Exception as e:
                logger.error(f"Visualization failed: {e}")

        self.results["scenario_1"] = scenario_results
        return scenario_results

    def scenario_2_environmental_monitoring(self) -> dict[str, Any]:
        """Scenario 2: Environmental Monitoring"""
        print("\n" + "=" * 60)
        print("🌍 SCENARIO 2: ENVIRONMENTAL MONITORING")
        print("=" * 60)

        scenario_results = {
            "scenario": "env_mon",
            "start_time": datetime.now().isoformat(),
            "components_used": ["SpatialIndexingInterface"],
            "data_outputs": {},
            "visualizations": [],
        }

        # 1. Create Sensor Grid
        print("️  Creating sensor network...")
        sensor_data = {}
        center_lat, center_lng = 37.7749, -122.4194

        for _ in range(30):
            # Hex grid around center
            lat = center_lat + RNG.uniform(-0.1, 0.1)
            lng = center_lng + RNG.uniform(-0.1, 0.1)
            cell = self.indexer.latlng_to_cell(lat, lng, 8)

            sensor_data[cell] = {
                "temperature": 20 + RNG.normal(0, 2),
                "aqi": 50 + int(RNG.integers(0, 100)),
                "humidity": 60 + RNG.normal(0, 10),
                "lat": lat,
                "lng": lng,
            }

        print(f"✅ Deployed {len(sensor_data)} virtual sensors")

        # 2. Aggregation (Simulation)
        avg_temp = sum(d["temperature"] for d in sensor_data.values()) / len(
            sensor_data
        )
        max_aqi = max(d["aqi"] for d in sensor_data.values())

        print(f"📊 Network Stats: Avg Temp {avg_temp:.1f}°C, Max AQI {max_aqi}")
        scenario_results["data_outputs"]["stats"] = {
            "avg_temp": avg_temp,
            "max_aqi": max_aqi,
        }

        # 3. Anomaly Detection (Simulation)
        print("🔍 Scanning for anomalies...")
        anomalies = [ID for ID, d in sensor_data.items() if d["aqi"] > 120]
        print(f"⚠️  Found {len(anomalies)} high pollution zones")

        # 4. Viz
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            lats = [d["lat"] for d in sensor_data.values()]
            lngs = [d["lng"] for d in sensor_data.values()]
            aqis = [d["aqi"] for d in sensor_data.values()]

            sc = ax.scatter(lngs, lats, c=aqis, cmap="RdYlGn_r", s=150, alpha=0.8)
            plt.colorbar(sc, label="Air Quality Index")
            ax.set_title("Environmental Sensor Network")

            viz_name = f"env_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_name)
            scenario_results["visualizations"].append(viz_name)
            print(f"✅ Saved visualization: {viz_name}")

        self.results["scenario_2"] = scenario_results
        return scenario_results

    def scenario_3_supply_chain(self) -> dict[str, Any]:
        """Scenario 3: Supply Chain Optimization"""
        print("\n" + "=" * 60)
        print("🚚 SCENARIO 3: SUPPLY CHAIN OPTIMIZATION")
        print("=" * 60)

        scenario_results = {
            "scenario": "supply_chain",
            "start_time": datetime.now().isoformat(),
            "data_outputs": {},
            "visualizations": [],
        }

        # 1. Supply Nodes
        print("📦 Initializing supply chain nodes...")
        nodes = {}
        node_types = ["warehouse", "retail", "factory"]

        for i in range(20):
            lat = 37.7 + (i * 0.02)
            lng = -122.4 + (i * 0.01)
            cell = self.indexer.latlng_to_cell(lat, lng, 7)

            ntype = node_types[i % 3]
            nodes[cell] = {
                "type": ntype,
                "capacity": 1000 if ntype == "warehouse" else 100,
                "demand": 0 if ntype == "warehouse" else 50,
                "lat": lat,
                "lng": lng,
            }

        # 2. Flow Analysis (Simulation)
        print("🌊 Analyzing network flows...")
        flows = []
        for c, data in nodes.items():
            if data["type"] == "retail":
                # Assign the nearest warehouse from the configured node set.
                flows.append(
                    {"from": "nearest_warehouse", "to": c, "amount": data["demand"]}
                )

        total_flow = sum(f["amount"] for f in flows)
        print(f"✅ Total flow volume: {total_flow} units")

        # 3. Viz
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 8))

            for d in nodes.values():
                color = "blue" if d["type"] == "warehouse" else "green"
                marker = "s" if d["type"] == "warehouse" else "o"
                ax.scatter(
                    d["lng"], d["lat"], c=color, marker=marker, s=100, label=d["type"]
                )

            # Deduplicate labels helper
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            ax.set_title("Supply Chain Network")
            viz_name = f"supply_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_name)
            scenario_results["visualizations"].append(viz_name)
            print(f"✅ Saved visualization: {viz_name}")

        self.results["scenario_3"] = scenario_results
        return scenario_results

    def generate_report(self):
        """Generates a JSON report."""
        print("\n" + "=" * 60)
        print("📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)

        report = {"timestamp": datetime.now().isoformat(), "scenarios": self.results}

        fname = f"nested_orchestrator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved to {fname}")
        return report


def main():
    orchestrator = NestedH3Orchestrator()
    orchestrator.scenario_1_urban_planning()
    orchestrator.scenario_2_environmental_monitoring()
    orchestrator.scenario_3_supply_chain()
    orchestrator.generate_report()
    print("\n🎉 Orchestration Complete!")


if __name__ == "__main__":
    main()
