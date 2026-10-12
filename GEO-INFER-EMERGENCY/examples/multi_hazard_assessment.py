#!/usr/bin/env python3
"""
GEO-INFER-EMERGENCY Example: Multi-Hazard Situational Assessment

Demonstrates multi-hazard monitoring using the real GEO-INFER-EMERGENCY
public API: per-hazard threat assessment via SituationalAwareness,
multi-source data fusion, and consequence-aware resource allocation via
ResourceDeployer.
"""

from geo_infer_emergency import SituationalAwareness, ResourceDeployer


# Hypothetical hazards threatening the same region, each with an intensity
# estimate and an affected-area profile.
HAZARDS = [
    {
        "name": "Wildfire",
        "hazard": {"type": "wildfire", "intensity": 0.8},
        "affected_area": {"area_sq_km": 50.0},
        "assets_at_risk": [
            {
                "id": "subdivision_1",
                "type": "residential",
                "population": 25000,
                "value": 500e6,
            },
        ],
    },
    {
        "name": "Flood",
        "hazard": {"type": "flood", "intensity": 0.6},
        "affected_area": {"area_sq_km": 30.0},
        "assets_at_risk": [
            {
                "id": "industrial_park",
                "type": "industrial",
                "population": 18000,
                "value": 300e6,
            },
        ],
    },
    {
        "name": "Earthquake",
        "hazard": {"type": "earthquake", "intensity": 0.5},
        "affected_area": {"area_sq_km": 120.0},
        "assets_at_risk": [
            {
                "id": "hospital_1",
                "type": "critical_infrastructure",
                "population": 40000,
                "value": 800e6,
            },
        ],
    },
]


def main():
    print("=" * 60)
    print("GEO-INFER-EMERGENCY: Multi-Hazard Situational Assessment")
    print("=" * 60)

    sa = SituationalAwareness(
        data_sources=["sensors", "field_reports", "satellite", "weather"],
        fusion_algorithms=["bayesian"],
    )

    hazard_results = {}
    for item in HAZARDS:
        assessment = sa.assess_threat(
            hazard=item["hazard"],
            affected_area=item["affected_area"],
            assets_at_risk=item["assets_at_risk"],
        )
        hazard_results[item["name"]] = assessment
        print(f"\nHazard: {item['name']}")
        print(f"  Threat level : {sa.get_current_threat_level()}")
        print(f"  Score        : {assessment.get('threat_score', 'n/a')}")
        recommendations = assessment.get("recommendations", [])
        for rec in recommendations[:3]:
            print(f"  - {rec}")

    # Fuse field observations from independent reporting sources for the
    # highest-threat hazard scenario.
    print("\n" + "-" * 60)
    print("Fusing multi-source observations (weighted average)...")
    fused = sa.fuse_data(
        sources=[
            {
                "source": "sensor_network",
                "data": {"temperature_c": 41.0, "wind_speed_kmh": 35.0},
                "confidence": 0.9,
            },
            {
                "source": "field_report",
                "data": {"temperature_c": 39.0, "wind_speed_kmh": 42.0},
                "confidence": 0.7,
            },
            {
                "source": "satellite_retrieval",
                "data": {"temperature_c": 40.5, "wind_speed_kmh": 38.0},
                "confidence": 0.8,
            },
        ],
        fusion_method="weighted_average",
        confidence_weighting=True,
    )
    print(f"  Fused fields : {fused['fused_data']}")
    print(f"  Confidence   : {fused['confidence']}")

    # Allocate the response inventory against the joint demand points of the
    # two highest-priority hazards.
    print("\n" + "-" * 60)
    print("Optimizing resource allocation across hazards...")
    deployer = ResourceDeployer(optimization_algorithm="greedy_nearest_resource")
    allocation = deployer.optimize_allocation(
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
                "location": {"lat": 34.05, "lon": -118.10},
            },
            {
                "id": "res_1",
                "type": "rescue_unit",
                "location": {"lat": 34.12, "lon": -118.22},
            },
        ],
        demand_points=[
            {"id": "wildfire_perimeter", "location": {"lat": 34.08, "lon": -118.27}},
            {"id": "flood_zone", "location": {"lat": 34.01, "lon": -118.18}},
            {"id": "field_hospital", "location": {"lat": 34.06, "lon": -118.12}},
        ],
        constraints={"response_time": 20, "coverage": 0.8},
        objectives=["minimize_response_time", "maximize_coverage"],
    )

    print(
        f"  Allocated    : {allocation['metrics']['resources_allocated']} / "
        f"{allocation['metrics']['total_demands']} demand points"
    )
    print(f"  Coverage     : {allocation['metrics']['coverage_rate']:.0%}")
    for entry in allocation["allocations"]:
        print(
            f"  - {entry['resource_id']} -> {entry['demand_id']} "
            f"({entry['estimated_response_time']:.1f} min)"
        )
    if allocation["unallocated_demands"]:
        print(f"  Unallocated  : {allocation['unallocated_demands']}")

    print("\n" + "=" * 60)
    print("Multi-Hazard Situational Assessment Complete!")
    print("=" * 60)

    highest = max(
        hazard_results,
        key=lambda name: hazard_results[name].get("threat_score", 0) or 0,
    )
    print(f"\n  - Highest-priority hazard: {highest}")
    print(f"  - Fused temperature: {fused['fused_data'].get('temperature_c')} C")
    print(f"  - Response coverage: {allocation['metrics']['coverage_rate']:.0%}")


if __name__ == "__main__":
    main()
