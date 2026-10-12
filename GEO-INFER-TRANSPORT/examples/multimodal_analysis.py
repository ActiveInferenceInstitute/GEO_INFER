#!/usr/bin/env python3
"""
GEO-INFER-TRANSPORT Example: Network, Routing, and Traffic Analysis

Demonstrates the real public API end to end: building a transport
network from edges, routing, origin-destination matrices, alternative
routes, traffic flow/congestion/forecast analysis, accessibility
isochrones, and transit coverage/scenario evaluation.
"""

from geo_infer_transport import (
    RoutingEngine,
    TrafficAnalyzer,
    TransportNetwork,
    TransitOptimizer,
)


def build_sample_network() -> TransportNetwork:
    """Build a small road network with coordinates."""
    network = TransportNetwork(network_type="road", modes=["car"])
    network.build_from_edges(
        nodes=[
            {"id": "n1", "location": {"lat": 34.050, "lon": -118.250}},
            {"id": "n2", "location": {"lat": 34.052, "lon": -118.247}},
            {"id": "n3", "location": {"lat": 34.055, "lon": -118.243}},
            {"id": "n4", "location": {"lat": 34.058, "lon": -118.240}},
            {"id": "n5", "location": {"lat": 34.060, "lon": -118.245}},
        ],
        edges=[
            {"id": "e1", "from": "n1", "to": "n2", "length_m": 300, "speed_limit": 50},
            {"id": "e2", "from": "n2", "to": "n3", "length_m": 450, "speed_limit": 50},
            {"id": "e3", "from": "n3", "to": "n4", "length_m": 400, "speed_limit": 40},
            {"id": "e4", "from": "n1", "to": "n5", "length_m": 250, "speed_limit": 50},
            {"id": "e5", "from": "n5", "to": "n3", "length_m": 350, "speed_limit": 50},
        ],
    )
    return network


def routing_example(network: TransportNetwork) -> None:
    """Route, build an OD matrix, and find alternative routes."""
    router = RoutingEngine(network=network, algorithm="dijkstra")

    route = router.route({"node_id": "n1"}, {"node_id": "n4"})
    print("--- Routing ---")
    print(f"Path: {' -> '.join(route.path)}")
    print(f"Distance: {route.total_distance_m:.0f} m, time: {route.total_time_s:.0f} s")
    print(f"Route source: {route.route_source}")

    matrix = router.calculate_matrix(
        origins=[{"id": "n1", "node_id": "n1"}, {"id": "n5", "node_id": "n5"}],
        destinations=[{"id": "n4", "node_id": "n4"}],
        metric="time",
    )
    print(f"OD matrix: {matrix['matrix']}")

    alternatives = router.find_alternatives(
        {"node_id": "n1"}, {"node_id": "n3"}, count=3
    )
    print(f"Alternatives found: {len(alternatives)} (including primary)")


def traffic_example(network: TransportNetwork) -> None:
    """Analyze flow, model congestion, and forecast traffic."""
    analyzer = TrafficAnalyzer(model_type="bpr", time_resolution="15min")

    flow = analyzer.analyze_flow(
        segment={"id": "e2", "capacity": 1800, "speed_limit": 50},
        counts=[
            {"count": 700, "speed_kmh": 42},
            {"count": 750, "speed_kmh": 38},
            {"count": 690, "speed_kmh": 44},
        ],
    )
    print("\n--- Traffic Flow ---")
    print(f"Volume: {flow.volume} veh/h, LOS: {flow.level_of_service}")

    congestion = analyzer.model_congestion(
        network_flows={"e1": 1200, "e2": 2100, "e3": 900},
        capacity_data={"e1": 2000, "e2": 2000, "e3": 2000},
    )
    print(f"Congested segments: {congestion['summary']['congested_segments']}")

    forecast = analyzer.forecast_traffic(
        [{"volume": 1000}, {"volume": 1100}, {"volume": 1200}, {"volume": 1150}],
        forecast_horizon="1h",
    )
    print(f"Forecast points (1h / 15min steps): {len(forecast['forecasts'])}")


def transit_example() -> None:
    """Evaluate transit coverage and a service-change scenario."""
    optimizer = TransitOptimizer()

    coverage = optimizer.analyze_coverage(
        stops=[{"id": "s1", "location": {"lat": 34.050, "lon": -118.250}}],
        population_zones=[
            {
                "id": "z1",
                "centroid": {"lat": 34.050, "lon": -118.250},
                "population": 4000,
                "demographics": {"low_income": 0.4, "other": 0.6},
            },
            {
                "id": "z2",
                "centroid": {"lat": 34.100, "lon": -118.300},
                "population": 2000,
                "demographic_group": "low_income",
            },
        ],
        walk_radius_m=400,
    )
    print("\n--- Transit Coverage ---")
    print(f"Coverage rate: {coverage['coverage_rate']:.1%}")
    if "equity_analysis" in coverage:
        print(f"Group coverage: {coverage['equity_analysis']['group_coverage']}")

    scenario = optimizer.evaluate_scenario(
        base_network={},
        proposed_changes=[
            {"type": "add_route", "expected_ridership": 1200},
            {"type": "increase_frequency"},
        ],
    )
    print(f"Scenario impacts: {scenario['impacts']['ridership_change']} new riders")
    print(f"Recommendation: {scenario['recommendation']}")


def main() -> None:
    """Run the multimodal transportation analysis example."""
    print("=" * 60)
    print("GEO-INFER-TRANSPORT: Network & Traffic Analysis Example")
    print("=" * 60)
    network = build_sample_network()
    routing_example(network)
    traffic_example(network)
    transit_example()
    print("\nExample complete.")


if __name__ == "__main__":
    main()
