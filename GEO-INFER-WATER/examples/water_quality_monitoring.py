#!/usr/bin/env python3
"""
GEO-INFER-WATER Example: Water Quality Monitoring System

This example demonstrates comprehensive water quality monitoring including
WQI calculation, pollution tracking, trend analysis, and regulatory compliance.
"""

from datetime import datetime, timedelta
import numpy as np

from geo_infer_water import (
    WaterQualityAssessor,
    WaterSample,
    WaterBodyType,
    PollutantType,
)


def main():
    print("=" * 60)
    print("GEO-INFER-WATER: Water Quality Monitoring System")
    print("=" * 60)

    # 1. Initialize Water Quality Assessor
    print("\n1. Setting Up Water Quality Monitoring...")

    assessor = WaterQualityAssessor()

    monitoring_network = {
        "name": "River Basin Monitoring Network",
        "region": "Sacramento River Basin",
        "stations": 12,
        "parameters_monitored": ["pH", "DO", "Turbidity", "Nitrate", "E. coli"],
    }

    print(f"   Network: {monitoring_network['name']}")
    print(f"   Region: {monitoring_network['region']}")
    print(f"   Stations: {monitoring_network['stations']}")

    # 2. Collect Water Samples
    print("\n2. Collecting Water Samples...")

    samples = []
    base_time = datetime.now()

    # Generate realistic samples across stations
    for station in range(5):
        for day in range(7):
            sample = WaterSample(
                sample_id=f"WQ_S{station + 1}_D{day + 1}",
                location=(-121.5 + station * 0.1, 38.5 + station * 0.05),
                timestamp=(base_time - timedelta(days=6 - day)).isoformat(),
                ph=7.0 + np.random.uniform(-0.5, 0.5),
                dissolved_oxygen=8.5 - station * 0.3 + np.random.uniform(-0.5, 0.5),
                turbidity=0.5 + station * 0.2 + np.random.uniform(0, 0.5),
                temperature=18 + np.random.uniform(-2, 2),
                nitrate=2.0 + station * 1.5 + np.random.uniform(0, 1),
                e_coli=10 * (station + 1) + np.random.uniform(0, 20),
            )
            samples.append(sample)

    print(f"   Samples collected: {len(samples)}")
    print("   Stations sampled: 5")
    print("   Days covered: 7")

    # 3. Calculate Water Quality Index
    print("\n3. Calculating Water Quality Index...")

    wqi_results = []
    for sample in samples[:5]:  # First sample from each station
        wqi = assessor.calculate_wqi(sample)
        wqi_results.append(wqi)

    print("\n   Station WQI Summary:")
    print(f"   {'Station':<12} {'WQI':>8} {'Classification':>15}")
    print(f"   {'-' * 37}")

    for i, wqi in enumerate(wqi_results):
        print(f"   Station {i + 1:<5} {wqi['wqi']:>8.1f} {wqi['classification']:>15}")

    avg_wqi = np.mean([w["wqi"] for w in wqi_results])
    print(f"\n   Network Average WQI: {avg_wqi:.1f}")

    # 4. Analyze Trends
    print("\n4. Analyzing Water Quality Trends...")

    # Get samples for one station
    station_samples = [s for s in samples if "S1" in s.sample_id]

    for param in ["ph", "dissolved_oxygen", "nitrate"]:
        trend = assessor.analyze_trends(station_samples, param)
        if "error" not in trend:
            print(
                f"   {param.upper()}: {trend['trend_direction']} "
                f"(mean={trend['mean']:.2f}, slope={trend['trend_slope']:.4f})"
            )

    # 5. Track Pollution Plume
    print("\n5. Modeling Pollution Plume Dispersion...")

    plume = assessor.track_pollution_plume(
        initial_location=(-121.5, 38.5),
        pollutant_type=PollutantType.NUTRIENT,
        flow_velocity=(0.2, 0.05),
        diffusion_coefficient=15.0,
        time_hours=12,
    )

    print(f"   Source location: {plume['initial_location']}")
    print(f"   Pollutant type: {plume['pollutant_type']}")
    print(f"   Time elapsed: {plume['time_hours']} hours")
    print(
        f"   Plume center: ({plume['plume_center'][0]:.4f}, {plume['plume_center'][1]:.4f})"
    )
    print(f"   Plume area: {plume['plume_area_km2']:.2f} km²")
    print(f"   Max dispersion extent: {plume['max_extent_km']:.2f} km")

    # 6. Assess Risk
    print("\n6. Assessing Water Quality Risk...")

    for usage in ["drinking", "recreation", "irrigation"]:
        risk = assessor.assess_risk(
            samples=samples[:10], water_body_type=WaterBodyType.RIVER, usage_type=usage
        )

        print(f"\n   {usage.upper()} Water Use:")
        print(f"   - Risk level: {risk['risk_level']}")
        print(f"   - Violations: {risk['violation_count']}")
        print(f"   - Recommendation: {risk['recommendation'][:60]}...")

    # 7. Check Regulatory Compliance
    print("\n7. Checking Regulatory Compliance...")
    compliance_results = {}
    for regulation in ["EPA", "WHO", "EU"]:
        compliance = assessor.check_regulatory_compliance(samples[:10], regulation)
        compliance_results[regulation] = compliance

        status = "✓ COMPLIANT" if compliance["overall_compliant"] else "✗ NON-COMPLIANT"
        print(
            f"   {regulation}: {status} ({compliance['compliance_rate']:.0%} parameters pass)"
        )
    # 8. Calculate Pollutant Load
    print("\n8. Calculating Pollutant Load...")

    # Example: Nitrate load calculation
    avg_nitrate = np.mean([s.nitrate for s in samples if s.nitrate])

    load = assessor.calculate_pollutant_load(
        concentration_mg_l=avg_nitrate,
        flow_rate_m3_s=50.0,  # Moderate river flow
        time_period_hours=24.0,
    )

    print(f"   Average nitrate concentration: {load['concentration_mg_l']:.2f} mg/L")
    print(f"   River flow rate: {load['flow_rate_m3_s']:.1f} m³/s")
    print(f"   Daily nitrate load: {load['load_kg']:.0f} kg/day")
    print(f"   Annual nitrate load: {load['load_kg'] * 365 / 1000:.1f} tonnes/year")

    # 9. Generate Summary Report
    print("\n" + "=" * 60)
    print("Water Quality Monitoring Report Summary")
    print("=" * 60)

    print(f"\n   Network: {monitoring_network['name']}")
    print("   Period: Last 7 days")
    print(f"   Samples analyzed: {len(samples)}")

    print("\n   Key Findings:")
    print(f"   • Average WQI: {avg_wqi:.1f} ({wqi_results[0]['classification']})")
    print(
        f"   • EPA Compliance: {'Pass' if compliance_results['EPA']['overall_compliant'] else 'Fail'}"
    )
    print(f"   • Daily nitrate load: {load['load_kg']:.0f} kg/day")
    print(f"   • Pollution plume extent: {plume['plume_area_km2']:.1f} km²")

    print("\n   Recommendations:")
    print("   1. Monitor downstream stations for elevated nutrients")
    print("   2. Investigate pollution source at Station 5")
    print("   3. Increase sampling frequency during storm events")


if __name__ == "__main__":
    main()
