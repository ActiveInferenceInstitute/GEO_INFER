#!/usr/bin/env python3
"""
GEO-INFER-EMERGENCY Example: Emergency Response Simulation

This example demonstrates a complete emergency response workflow
including incident coordination, resource deployment optimization,
evacuation planning, situational awareness, and search and rescue.
"""

from geo_infer_emergency import (
    EmergencyCoordinator,
    ResourceDeployer,
    EvacuationPlanner,
    SituationalAwareness,
    SearchAndRescue,
)


def main():
    print("=" * 60)
    print("GEO-INFER-EMERGENCY: Emergency Response Simulation")
    print("=" * 60)

    # 1. Initialize Emergency Coordinator
    print("\n1. Initializing Emergency Command Structure...")
    coordinator = EmergencyCoordinator(
        command_structure="ics",  # Incident Command System
        agencies=["fire", "medical", "police", "public_works"],
    )

    # Coordinate the multi-agency response to an incident
    coordination = coordinator.coordinate(
        incident={
            "id": "wildfire_2024_001",
            "type": "wildfire",
            "name": "Canyon Wildfire",
            "location": {"latitude": 34.05, "longitude": -118.25},
            "scale": "type_2",
        },
        agencies=["agency_fire", "agency_police", "agency_medical"],
        resources={
            "engines": ["eng_1", "eng_2", "eng_3", "eng_4"],
            "personnel": [f"p_{i}" for i in range(40)],
            "patrol_units": ["pat_1", "pat_2"],
        },
    )

    print(f"   Incident: {coordination['incident_name']}")
    print(f"   Responding agencies: {len(coordination['responding_agencies'])}")
    print(f"   Channels: {list(coordination['communication_channels'])}")
    for assignment in coordination["resource_assignments"]:
        print(
            f"   - {assignment['agency']} -> sector {assignment['sector']} "
            f"({len(assignment['resources'])} resource lines)"
        )

    # Establish the ICS command structure
    command = coordinator.establish_command(
        incident_type="wildfire",
        location={"lat": 34.05, "lon": -118.25},
        scale="type_2",
        command_structure={
            "incident_commander": "Chief Alvarez",
            "operations": "Captain Boone",
            "planning": "Lt. Chen",
        },
    )
    print(f"   Command established: {command['incident_id']}")
    active = coordinator.get_active_incidents()
    print(f"   Active incidents tracked: {len(active)}")

    # 2. Build Situational Awareness
    print("\n2. Building Situational Awareness (Common Operating Picture)...")
    sa = SituationalAwareness(data_sources=["sensors", "field_reports", "satellite"])

    cop = sa.build_cop(
        layers=[
            {
                "id": "fire_perimeter",
                "name": "Fire Perimeter",
                "source": "satellite",
                "type": "polygon",
            },
            {
                "id": "population",
                "name": "Population",
                "source": "census",
                "type": "points",
            },
            {
                "id": "weather",
                "name": "Weather",
                "source": "station_network",
                "type": "raster",
            },
        ],
        extent={"lat_min": 33.9, "lat_max": 34.2, "lon_min": -118.4, "lon_max": -118.1},
        symbology={"polygon": {"fill": "red"}, "points": {"marker": "triangle"}},
        refresh_rate=30,
    )

    print(
        f"   COP established with {len(cop['layers'])} layers "
        f"(refresh every {cop['refresh_rate_seconds']}s)"
    )

    threat = sa.assess_threat(
        hazard={"type": "wildfire", "intensity": 0.8},
        affected_area={"population": 25000, "size_sq_km": 50.0},
        assets_at_risk=[{"id": "subdivision_1", "type": "residential", "value": 500e6}],
    )
    print(f"   Threat level: {sa.get_current_threat_level()}")

    # 3. Optimize Resource Deployment
    print("\n3. Optimizing Emergency Resource Deployment...")
    deployer = ResourceDeployer(
        optimization_algorithm="greedy_nearest_resource",
        resource_types=["engines", "ambulances", "rescue_units"],
    )

    deployment = deployer.optimize_allocation(
        resources=[
            {
                "id": "eng_1",
                "type": "engine",
                "location": {"lat": 34.10, "lon": -118.30},
            },
            {
                "id": "eng_2",
                "type": "engine",
                "location": {"lat": 34.00, "lon": -118.20},
            },
            {
                "id": "amb_1",
                "type": "ambulance",
                "location": {"lat": 34.08, "lon": -118.15},
            },
        ],
        demand_points=[
            {"id": "dp_1", "location": {"lat": 34.05, "lon": -118.25}},
            {"id": "dp_2", "location": {"lat": 34.02, "lon": -118.22}},
        ],
        constraints={"response_time": 15, "coverage": 0.8},
        objectives=["minimize_response_time", "maximize_coverage"],
    )

    print(f"   Coverage: {deployment['metrics']['coverage_rate']:.0%}")
    for allocation in deployment["allocations"]:
        print(
            f"   - {allocation['resource_id']} -> {allocation['demand_id']} "
            f"({allocation['estimated_response_time']:.1f} min)"
        )

    # 4. Plan Evacuation
    print("\n4. Planning Evacuation...")
    import networkx as nx

    road_network = nx.Graph()
    road_network.add_edge("zone_a", "junction_1", travel_time=12, capacity=1500)
    road_network.add_edge("zone_a", "junction_2", travel_time=18, capacity=900)
    road_network.add_edge("junction_1", "shelter_1", travel_time=10, capacity=1200)
    road_network.add_edge("junction_2", "shelter_2", travel_time=10, capacity=1200)
    road_network.add_edge("junction_1", "shelter_2", travel_time=20, capacity=700)

    evac_planner = EvacuationPlanner(
        road_network=road_network,
        shelters=[
            {
                "id": "shelter_1",
                "name": "North Shelter",
                "capacity": 5000,
                "location": {"lat": 34.10, "lon": -118.05},
            },
            {
                "id": "shelter_2",
                "name": "West Shelter",
                "capacity": 8000,
                "location": {"lat": 34.00, "lon": -118.35},
            },
        ],
    )

    evac_plan = evac_planner.plan(
        affected_zone={"id": "zone_a", "name": "Canyon Zone", "level": "order"},
        population={"total": 25000},
        destinations=[
            {"id": "shelter_1", "name": "North Shelter", "capacity": 5000},
            {"id": "shelter_2", "name": "West Shelter", "capacity": 8000},
        ],
        contraflow=True,
    )

    print(f"   Routes planned: {len(evac_plan['routes'])}")
    print(
        f"   Estimated clearance time: "
        f"{evac_plan['estimated_clearance_time_hours']:.1f} hours"
    )

    # 5. Search and Rescue Operations
    print("\n5. Planning Search and Rescue Operations...")
    sar = SearchAndRescue(
        terrain_data={"type": "mixed_forest"},
        statistical_data={"hiker": {"median_distance_km": 3.2}},
    )

    sar_plan = sar.plan_mission(
        subject={"id": "subject_1", "type": "hiker", "name": "Missing Hiker"},
        last_known_point={"lat": 34.05, "lon": -118.25},
        search_radius=5.0,
    )

    print(f"   Mission: {sar_plan['mission_id']}")
    print(
        f"   Recommended pattern: "
        f"{sar_plan['recommended_pattern']['pattern_type']} "
        f"({len(sar_plan['recommended_pattern']['waypoints'])} waypoints)"
    )
    print(f"   Ground teams needed: {sar_plan['resource_estimate']['ground_teams']}")

    print("\n" + "=" * 60)
    print("Emergency Response Simulation Complete!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(
        f"  - Incident: {coordination['incident_name']} "
        f"(command: {command['incident_id']})"
    )
    print(
        f"  - Resources deployed: {deployment['metrics']['resources_allocated']} units"
    )
    print("  - Population to evacuate: 25,000")
    print(f"  - SAR search radius: {sar_plan['search_radius_km']} km")


if __name__ == "__main__":
    main()
