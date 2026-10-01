#!/usr/bin/env python3
"""GEO-INFER-EMERGENCY module orchestrator.

Runs one documented end-to-end EMERGENCY operation on synthetic data: manage
a synthetic wildfire from detection to full response. ``SituationalAwareness``
assesses the fire threat and builds the common operating picture,
``EmergencyCoordinator`` establishes ICS command, coordinates the responding
agencies, requests mutual aid and issues an ICS-209 situation report,
``ResourceDeployer`` optimizes engine and ambulance allocation to demand
points, ``EvacuationPlanner`` plans a staged evacuation over a synthetic road
network with shelter operations, and ``SearchAndRescue`` opens a mission for
a missing hiker with an expanding-square search pattern. All work goes
through the real ``geo_infer_emergency`` public API.
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
    import networkx as nx

    from geo_infer_emergency import (
        EmergencyCoordinator,
        EvacuationPlanner,
        ResourceDeployer,
        SearchAndRescue,
        SituationalAwareness,
    )

    # 1. Threat assessment and common operating picture.
    awareness = SituationalAwareness()
    assessment = awareness.assess_threat(
        hazard={
            "type": "wildfire",
            "intensity": 0.7,
            "speed": 30,
            "direction": "northeast",
        },
        affected_area={"area_sq_km": 62.5, "geometry": {"type": "Polygon"}},
        assets_at_risk=[
            {"name": "Riverside District", "population": 24_000},
            {"name": "Pine Ridge Substation", "population": 0, "critical": True},
            {"name": "Cedar Valley Care Home", "population": 4_000},
        ],
        projection_hours=24,
    )
    integration = awareness.integrate_sensors(
        sensor_network={
            "sensors": [
                {
                    "id": "SENSOR_TOWER_1",
                    "type": "camera",
                    "location": {"lat": 38.57, "lon": -121.81},
                    "readings": {"smoke_detected": True},
                    "confidence": 0.9,
                },
                {
                    "id": "SENSOR_WX_2",
                    "type": "weather",
                    "location": {"lat": 38.59, "lon": -121.78},
                    "readings": {"wind_kmh": 30, "humidity_pct": 18},
                    "confidence": 0.95,
                },
            ]
        },
        data_types=["imagery", "weather"],
    )
    cop = awareness.build_cop(
        layers=[
            {
                "id": "layer_fire_perimeter",
                "name": "Fire Perimeter",
                "type": "hazard",
                "z_order": 1,
            },
            {
                "id": "layer_sensors",
                "name": "Sensor Feeds",
                "type": "sensors",
                "z_order": 2,
            },
            {
                "id": "layer_resources",
                "name": "Committed Resources",
                "type": "resources",
                "z_order": 3,
            },
        ],
        extent={
            "lat_min": 38.50,
            "lat_max": 38.62,
            "lon_min": -121.90,
            "lon_max": -121.70,
        },
        symbology={},
    )

    # 2. ICS command establishment, multi-agency coordination, mutual aid,
    #    and situation reporting.
    coordinator = EmergencyCoordinator(
        command_structure="ics",
        agencies=["fire", "police", "medical", "public_works"],
    )
    command = coordinator.establish_command(
        incident_type="wildfire",
        location={"name": "Cedar Creek Command Post", "lat": 38.55, "lon": -121.82},
        scale="type_3",
        command_structure={
            "name": "Cedar Creek Fire",
            "incident_commander": "Chief A. Reyes",
            "operations": "Battalion Chief M. Okafor",
            "planning": "Planner S. Duval",
            "logistics": "Logistics Lead T. Nguyen",
            "finance": "Finance Lead R. Ellison",
            "safety": "Safety Officer K. Brandt",
            "liaison": "Liaison J. Alvarez",
            "pio": "Public Info Officer D. Whitfield",
        },
    )
    coordination = coordinator.coordinate(
        incident={
            "id": "WF-2026-0914",
            "type": "wildfire",
            "name": "Cedar Creek Fire",
            "location": {"lat": 38.56, "lon": -121.80},
            "scale": "type_3",
            "description": "Wind-driven wildfire in grass and oak woodland",
        },
        agencies=["agency_fire", "agency_police", "agency_medical"],
        resources={
            "engines": 12,
            "trucks": 4,
            "personnel": 140,
            "patrol_units": 8,
            "ambulances": 6,
        },
        incident_action_plan={
            "period": "2026-09-14 day shift",
            "objectives": ["life safety", "perimeter control"],
        },
    )
    mutual_aid = coordinator.request_mutual_aid(
        requesting_agency="agency_fire",
        resource_needs=["heavy_equipment"],
        duration_hours=12,
        staging_areas=[{"name": "County Staging Yard", "lat": 38.52, "lon": -121.72}],
    )
    sitrep = coordinator.generate_sitrep(
        incident={
            "id": "WF-2026-0914",
            "name": "Cedar Creek Fire",
            "status": "active",
            "percent_contained": 15,
            "threat_level": "high",
            "personnel_count": 140,
            "engines": 12,
            "helicopters": 2,
            "evacuation_ordered": True,
            "population_affected": 24_000,
        },
        update_frequency="hourly",
        distribution=["eoc", "unified_command"],
    )

    # 3. Resource deployment optimization.
    deployer = ResourceDeployer(resource_types=["engines", "ambulances"])
    allocation = deployer.optimize_allocation(
        resources=[
            {
                "id": "ENG-01",
                "type": "engine",
                "name": "Engine 1",
                "location": {"lat": 38.5500, "lon": -121.8000},
                "agency": "fire",
            },
            {
                "id": "AMB-01",
                "type": "ambulance",
                "name": "Ambulance 1",
                "location": {"lat": 38.6020, "lon": -121.8620},
                "agency": "medical",
            },
        ],
        demand_points=[
            {"id": "DP-RIVERSIDE", "location": {"lat": 38.5540, "lon": -121.8060}},
            {"id": "DP-RIDGE", "location": {"lat": 38.6040, "lon": -121.8640}},
        ],
        constraints={"response_time": 15, "coverage": 0.8},
        objectives=["minimize_response_time", "maximize_coverage"],
    )

    # 4. Staged evacuation over a synthetic road network plus shelter
    #    operations planning.
    road_network = nx.Graph()
    road_network.add_edge(
        "ZONE_RIVERSIDE", "JCT_MAIN", travel_time=6.0, distance=4.8, capacity=1200
    )
    road_network.add_edge(
        "JCT_MAIN", "JCT_RIDGE", travel_time=9.0, distance=7.1, capacity=900
    )
    road_network.add_edge(
        "JCT_RIDGE", "SHELTER_FAIRGROUNDS", travel_time=5.0, distance=3.6, capacity=1000
    )
    road_network.add_edge(
        "JCT_MAIN", "SHELTER_EASTSIDE", travel_time=7.0, distance=5.9, capacity=800
    )
    road_network.add_edge(
        "ZONE_RIVERSIDE",
        "SHELTER_EASTSIDE",
        travel_time=14.0,
        distance=11.2,
        capacity=600,
    )

    planner = EvacuationPlanner(road_network=road_network)
    destinations = [
        {
            "id": "SHELTER_FAIRGROUNDS",
            "name": "County Fairgrounds",
            "location": {"lat": 38.58, "lon": -121.77},
            "capacity": 2500,
            "services": ["medical", "food", "cots"],
        },
        {
            "id": "SHELTER_EASTSIDE",
            "name": "Eastside Community Center",
            "location": {"lat": 38.56, "lon": -121.74},
            "capacity": 900,
            "services": ["food", "cots"],
        },
    ]
    evac_plan = planner.plan(
        affected_zone={
            "id": "ZONE_RIVERSIDE",
            "name": "Riverside District",
            "level": "order",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-121.83, 38.53],
                        [-121.79, 38.53],
                        [-121.79, 38.57],
                        [-121.83, 38.57],
                        [-121.83, 38.53],
                    ]
                ],
            },
        },
        population={"total": 3200, "special_populations": ["Cedar Valley Care Home"]},
        destinations=destinations,
        phasing="staged",
    )
    shelter_plan = planner.plan_shelters(
        shelter_locations=destinations,
        population_estimate=3200,
        duration_days=3,
        services=["medical", "food", "cots"],
    )

    # 5. Search and rescue mission for a missing hiker.
    sar = SearchAndRescue()
    mission = sar.plan_mission(
        subject={
            "id": "SUBJ-0042",
            "type": "hiker",
            "name": "Riley Chen",
            "age": 34,
            "last_known_location": {"lat": 38.92, "lon": -120.55},
            "experience": "intermediate",
        },
        last_known_point={"lat": 38.92, "lon": -120.55},
        terrain_type="forested",
        weather={"conditions": "clear", "wind_kmh": 12},
    )
    pattern = mission["recommended_pattern"]

    return {
        "operation": "wildfire_response_cycle_with_evacuation_and_sar",
        "situational_awareness": {
            "threat_level": assessment["threat_level"],
            "threat_score": round(float(assessment["threat_score"]), 6),
            "projection_hours": assessment["projection"]["hours"],
            "expected_expansion": assessment["projection"]["expected_expansion"],
            "recommendation_count": len(assessment["recommendations"]),
            "sensors_integrated": integration["sensor_count"],
            "cop_layers": len(cop["layers"]),
        },
        "command_and_coordination": {
            "command_structure": command["command_structure"],
            "responding_agencies": coordination["responding_agencies"],
            "communication_channels": coordination["communication_channels"],
            "operational_period": coordination["operational_period"],
            "resource_assignments": [
                {
                    "agency": entry["agency"],
                    "sector": entry["sector"],
                    "resources": [
                        {"type": unit["type"], "quantity": unit["quantity"]}
                        for unit in entry["resources"]
                    ],
                }
                for entry in coordination["resource_assignments"]
            ],
            "mutual_aid": {
                "status": mutual_aid["status"],
                "providing_agencies": [
                    assignment["providing_agency"]
                    for assignment in mutual_aid["assignments"]
                ],
                "eta_hours": mutual_aid["assignments"][0]["eta_hours"]
                if mutual_aid["assignments"]
                else None,
            },
            "active_incidents": len(coordinator.get_active_incidents()),
            "sitrep": {
                "format": sitrep["format"],
                "current_status": sitrep["current_status"],
                "resources_committed": sitrep["resources_committed"],
                "evacuations": sitrep["evacuations"],
            },
        },
        "resource_deployment": {
            "optimization_algorithm": allocation["optimization_algorithm"],
            "allocations": [
                {
                    "demand_id": item["demand_id"],
                    "resource_id": item["resource_id"],
                    "resource_type": item["resource_type"],
                    "estimated_response_time_min": round(
                        float(item["estimated_response_time"]), 3
                    ),
                }
                for item in allocation["allocations"]
            ],
            "coverage_rate": round(float(allocation["metrics"]["coverage_rate"]), 6),
            "average_response_time_min": round(
                float(allocation["metrics"]["average_response_time"]), 3
            ),
            "feasible": allocation["feasible"],
        },
        "evacuation": {
            "affected_zone": evac_plan["affected_zone"],
            "routes": [
                {
                    "route_id": route["route_id"],
                    "path": route["path"],
                    "distance_km": route["distance_km"],
                    "estimated_time_minutes": route["estimated_time_minutes"],
                    "capacity_vehicles_per_hour": route["capacity_vehicles_per_hour"],
                }
                for route in evac_plan["routes"]
            ],
            "phasing_strategy": evac_plan["phasing"]["strategy"],
            "phases": evac_plan["phasing"]["phases"],
            "estimated_clearance_time_hours": evac_plan[
                "estimated_clearance_time_hours"
            ],
            "special_populations": evac_plan["special_populations"],
        },
        "shelter_operations": {
            "total_shelter_capacity": shelter_plan["total_shelter_capacity"],
            "capacity_sufficient": shelter_plan["capacity_sufficient"],
            "overflow_population": shelter_plan["overflow_population"],
            "utilization": [
                {
                    "shelter_id": shelter["shelter_id"],
                    "assigned_population": shelter["assigned_population"],
                    "utilization": round(float(shelter["utilization"]), 6),
                }
                for shelter in shelter_plan["shelters"]
            ],
            "resource_requirements": shelter_plan["resource_requirements"],
            "volunteers_per_shift": shelter_plan["staffing_needs"][
                "volunteers_per_shift"
            ],
        },
        "search_and_rescue": {
            "subject": mission["subject"],
            "search_radius_km": mission["search_radius_km"],
            "terrain_type": mission["terrain_type"],
            "resource_estimate": mission["resource_estimate"],
            "priority_areas": mission["priority_areas"],
            "pattern": {
                "pattern_type": pattern["pattern_type"],
                "waypoints": len(pattern["waypoints"]),
                "track_spacing_m": pattern["track_spacing_m"],
                "estimated_distance_km": round(
                    float(pattern["estimated_distance_km"]), 6
                ),
                "estimated_search_time_hours": round(
                    float(pattern["estimated_search_time_hours"]), 6
                ),
            },
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("EMERGENCY", _operation))
