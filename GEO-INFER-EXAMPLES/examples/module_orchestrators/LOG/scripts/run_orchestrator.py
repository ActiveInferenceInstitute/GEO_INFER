#!/usr/bin/env python3
"""GEO-INFER-LOG module orchestrator.

Runs one documented end-to-end LOG operation on synthetic data: build a tiny
synthetic road network (12 intersections with distance and travel-time edge
weights), load it into the ``RouteOptimizer``, and solve a multi-stop route
with two waypoints between a depot and a delivery destination. All work goes
through the real ``geo_infer_log`` public API.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import math

    import networkx as nx
    import numpy as np
    from geo_infer_log import RouteOptimizer
    from geo_infer_log.core.routing import RoutingParameters, save_gpickle

    rng = np.random.default_rng(7)

    # Tiny synthetic street grid: 3 rows x 4 columns of intersections around
    # a fictional coastal town, with jittered WGS84 positions.
    node_coords: dict[str, tuple[float, float]] = {}
    for row in range(3):
        for col in range(4):
            name = f"I{row}{col}"
            lon = -124.21 + col * 0.01 + float(rng.uniform(-0.001, 0.001))
            lat = 41.74 + row * 0.01 + float(rng.uniform(-0.001, 0.001))
            node_coords[name] = (lon, lat)

    graph = nx.Graph()
    for name, (lon, lat) in node_coords.items():
        graph.add_node(name, x=lon, y=lat)

    def _add_road(u: str, v: str) -> None:
        u_lon, u_lat = node_coords[u]
        v_lon, v_lat = node_coords[v]
        distance_km = math.hypot(
            (v_lon - u_lon) * math.cos(math.radians(u_lat)) * 111.32,
            (v_lat - u_lat) * 110.57,
        )
        speed_kmh = float(rng.uniform(35.0, 55.0))
        graph.add_edge(
            u,
            v,
            distance=round(distance_km, 4),
            time=round(distance_km / speed_kmh * 60.0, 4),
        )

    for row in range(3):
        for col in range(3):
            _add_road(f"I{row}{col}", f"I{row}{col + 1}")
    for row in range(2):
        for col in range(4):
            _add_road(f"I{row}{col}", f"I{row + 1}{col}")
    _add_road("I00", "I11")  # diagonal shortcut

    with tempfile.TemporaryDirectory() as tmp_dir:
        network_path = Path(tmp_dir) / "synthetic_street_grid.gpickle"
        # Persist through the authenticated GISP1 envelope writer: LOG's
        # loader refuses unsigned payloads by design (GS19-07).
        save_gpickle(str(network_path), graph)

        optimizer = RouteOptimizer(RoutingParameters(weight_factor="time"))
        optimizer.load_network(str(network_path))

        depot_lon, depot_lat = node_coords["I00"]
        dest_lon, dest_lat = node_coords["I21"]
        waypoint_names = ["I02", "I12"]
        route = optimizer.optimize_route(
            origin=(depot_lon, depot_lat),
            destination=(dest_lon, dest_lat),
            waypoints=[node_coords[name] for name in waypoint_names],
        )

    if "error" in route:
        raise RuntimeError(f"route optimization failed: {route['error']}")

    return {
        "operation": "multi_stop_route_optimization_on_synthetic_network",
        "weight_factor": "time",
        "network_nodes": int(graph.number_of_nodes()),
        "network_edges": int(graph.number_of_edges()),
        "n_waypoints": len(waypoint_names),
        "route_stops": [str(node) for node in route["path"]],
        "total_distance_km": round(float(route["distance"]), 3),
        "total_travel_time_minutes": round(float(route["travel_time"]), 3),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("LOG", _operation))
