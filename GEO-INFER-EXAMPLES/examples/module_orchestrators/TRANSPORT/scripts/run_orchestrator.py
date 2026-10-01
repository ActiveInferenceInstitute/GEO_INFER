#!/usr/bin/env python3
"""GEO-INFER-TRANSPORT module orchestrator.

Runs one documented end-to-end TRANSPORT operation on synthetic data: build a
small road network via ``TransportNetwork.build_from_edges``, analyze its
connectivity and centrality, compute a car route between two intersections
with ``RoutingEngine.route``, model traffic flow on one segment with
``TrafficAnalyzer.analyze_flow``, and generate service-area isochrones with
``AccessibilityAnalyzer.calculate_isochrone``. All work goes through the real
``geo_infer_transport`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_transport import (
        AccessibilityAnalyzer,
        RoutingEngine,
        TrafficAnalyzer,
        TransportNetwork,
    )

    nodes = [
        {
            "id": "n1",
            "location": {"lat": 37.7749, "lon": -122.4194},
            "type": "intersection",
        },
        {
            "id": "n2",
            "location": {"lat": 37.7759, "lon": -122.4184},
            "type": "intersection",
        },
        {
            "id": "n3",
            "location": {"lat": 37.7769, "lon": -122.4174},
            "type": "intersection",
        },
        {
            "id": "n4",
            "location": {"lat": 37.7739, "lon": -122.4184},
            "type": "intersection",
        },
        {
            "id": "n5",
            "location": {"lat": 37.7729, "lon": -122.4174},
            "type": "destination",
        },
        {
            "id": "n6",
            "location": {"lat": 37.7749, "lon": -122.4164},
            "type": "intersection",
        },
    ]
    edges = [
        {
            "id": "e1",
            "from": "n1",
            "to": "n2",
            "road_class": "primary",
            "length_m": 1200,
            "speed_limit": 50,
            "lanes": 2,
        },
        {
            "id": "e2",
            "from": "n2",
            "to": "n3",
            "road_class": "primary",
            "length_m": 1400,
            "speed_limit": 50,
            "lanes": 2,
        },
        {
            "id": "e3",
            "from": "n1",
            "to": "n4",
            "road_class": "secondary",
            "length_m": 900,
            "speed_limit": 40,
            "lanes": 1,
        },
        {
            "id": "e4",
            "from": "n4",
            "to": "n5",
            "road_class": "secondary",
            "length_m": 1100,
            "speed_limit": 40,
            "lanes": 1,
        },
        {
            "id": "e5",
            "from": "n5",
            "to": "n6",
            "road_class": "residential",
            "length_m": 800,
            "speed_limit": 30,
            "lanes": 1,
        },
        {
            "id": "e6",
            "from": "n6",
            "to": "n3",
            "road_class": "residential",
            "length_m": 1000,
            "speed_limit": 30,
            "lanes": 1,
        },
        {
            "id": "e7",
            "from": "n2",
            "to": "n4",
            "road_class": "tertiary",
            "length_m": 700,
            "speed_limit": 35,
            "lanes": 1,
        },
    ]

    network = TransportNetwork(
        network_type="road", modes=["car", "bicycle", "pedestrian"]
    )
    build_summary = network.build_from_edges(edges, nodes=nodes)

    connectivity = network.analyze_connectivity(
        method="reachability", origin="n1", destinations=["n3", "n5", "n6"]
    )
    centrality = network.calculate_centrality(
        centrality_type="betweenness", weight="length", top_n=3
    )

    router = RoutingEngine(network=network, algorithm="dijkstra", modes=["car"])
    route = router.route(
        {"node_id": "n1"}, {"node_id": "n3"}, mode="car", optimization="time"
    )

    traffic = TrafficAnalyzer(model_type="bpr", time_resolution="15min")
    flow = traffic.analyze_flow(
        {"id": "e1", "capacity": 1800, "speed_limit": 50},
        counts=[
            {"count": 560, "speed_kmh": 42.0},
            {"count": 610, "speed_kmh": 39.0},
            {"count": 590, "speed_kmh": 40.5},
            {"count": 640, "speed_kmh": 37.5},
        ],
        time_period="peak",
    )

    accessibility = AccessibilityAnalyzer(network=network, default_mode="car")
    isochrones = accessibility.calculate_isochrone(
        {"node_id": "n1", "location": {"lat": 37.7749, "lon": -122.4194}},
        travel_times=[5, 10],
    )

    return {
        "operation": "network_built_routed_with_flow_and_isochrones",
        "network": {
            "network_type": str(build_summary["network_type"]),
            "nodes_created": int(build_summary["nodes_created"]),
            "edges_created": int(build_summary["edges_created"]),
            "is_connected": bool(build_summary["is_connected"]),
        },
        "connectivity": {
            "method": str(connectivity["method"]),
            "reachable_nodes": int(connectivity["reachable_nodes"]),
            "reachability_ratio": round(float(connectivity["reachability_ratio"]), 6),
            "destinations_reachable": sorted(connectivity["destinations_reachable"]),
        },
        "centrality": {
            "centrality_type": str(centrality["centrality_type"]),
            "top_nodes": [
                {
                    "node_id": str(node["node_id"]),
                    "centrality": round(float(node["centrality"]), 4),
                }
                for node in centrality["top_nodes"]
            ],
        },
        "route": {
            "path": list(route.path),
            "total_distance_m": round(float(route.total_distance_m), 6),
            "total_time_s": round(float(route.total_time_s), 6),
            "route_source": str(route.route_source),
        },
        "traffic_flow": {
            "segment_id": str(flow.segment_id),
            "volume_vph": int(flow.volume),
            "density_vpk": round(float(flow.density), 2),
            "speed_kmh": round(float(flow.speed), 1),
            "level_of_service": str(flow.level_of_service),
        },
        "isochrones": [
            {
                "time_minutes": int(iso.time_minutes),
                "mode": str(iso.mode),
                "area_sq_km": round(float(iso.area_sq_km), 2),
                "reachable_nodes": sorted(iso.reachable_nodes),
            }
            for iso in isochrones
        ],
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("TRANSPORT", _operation))
