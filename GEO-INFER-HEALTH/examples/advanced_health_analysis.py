#!/usr/bin/env python3
"""
Advanced GEO-INFER-HEALTH Analysis Example

This example demonstrates the comprehensive capabilities of the GEO-INFER-HEALTH module,
including Active Inference-based disease surveillance, advanced geospatial analysis,
and integrated health analytics.
"""

from datetime import datetime, timedelta, UTC
import numpy as np

from geo_infer_health.core.enhanced_disease_surveillance import (
    ActiveInferenceDiseaseAnalyzer,
)
from geo_infer_health.core.healthcare_accessibility import (
    HealthcareAccessibilityAnalyzer,
)
from geo_infer_health.core.environmental_health import EnvironmentalHealthAnalyzer
from geo_infer_health.models import (
    DiseaseReport,
    HealthFacility,
    PopulationData,
    EnvironmentalData,
    Location,
)
from geo_infer_health.utils.advanced_geospatial import (
    spatial_clustering,
    calculate_spatial_statistics,
    calculate_spatial_autocorrelation,
    calculate_hotspot_statistics,
    validate_geographic_bounds,
)
from geo_infer_health.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def _coordinate_value(location, coordinate: str, default: float = 0.0) -> float:
    """Read a coordinate from either model-like objects or dictionaries."""
    if isinstance(location, dict):
        return location.get(coordinate, default)
    return getattr(location, coordinate, default)


def create_sample_disease_data():
    """Create comprehensive sample disease surveillance data."""
    logger.info("Creating sample disease surveillance data...")

    base_time = datetime.now(UTC)

    # Create multiple disease types
    diseases = ["COVID-19", "Influenza", "RSV", "Pertussis"]
    locations = [
        Location(latitude=34.0522, longitude=-118.2437),  # Los Angeles
        Location(latitude=34.0622, longitude=-118.2337),  # Nearby area
        Location(latitude=34.0722, longitude=-118.2237),  # Another nearby area
        Location(latitude=33.9522, longitude=-118.3437),  # Distant area
    ]

    reports = []

    # Create disease reports over time
    for day in range(30):  # 30 days of data
        current_date = base_time - timedelta(days=29 - day)

        for loc_idx, location in enumerate(locations):
            for disease_idx, disease in enumerate(diseases):
                # Simulate disease patterns
                base_cases = 1

                # Add temporal patterns
                if disease == "Influenza":
                    # Seasonal pattern
                    seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day / 365)
                    base_cases *= seasonal_factor
                elif disease == "COVID-19":
                    # Wave pattern
                    wave_factor = 1 + np.exp(-((day - 15) ** 2) / 50)
                    base_cases *= wave_factor

                # Add spatial clustering for some diseases
                if (
                    disease == "COVID-19" and loc_idx < 2
                ):  # Cluster in first two locations
                    base_cases *= 2

                # Generate case count with noise
                case_count = max(1, int(np.random.poisson(base_cases)))

                report = DiseaseReport(
                    report_id=f"{disease}_{loc_idx}_{day}",
                    disease_code=disease,
                    location=location,
                    report_date=current_date,
                    case_count=case_count,
                    source="Sample Data Generator",
                )
                reports.append(report)

    logger.info(f"Created {len(reports)} disease reports")
    return reports


def create_sample_healthcare_data():
    """Create comprehensive sample healthcare facility data."""
    logger.info("Creating sample healthcare facility data...")

    # Create facilities of different types
    facility_templates = [
        {
            "type": "Hospital",
            "capacity": 500,
            "services": [
                "Emergency",
                "Surgery",
                "Cardiology",
                "Pediatrics",
                "Oncology",
            ],
        },
        {
            "type": "Clinic",
            "capacity": 100,
            "services": ["General Checkup", "Vaccinations", "Pediatrics"],
        },
        {"type": "Emergency", "capacity": 50, "services": ["Emergency", "Trauma"]},
        {
            "type": "Specialist",
            "capacity": 75,
            "services": ["Cardiology", "Neurology", "Orthopedics"],
        },
    ]

    facilities = []
    base_location = Location(latitude=34.0522, longitude=-118.2437)

    # Create facilities spread across the area
    for i in range(25):
        # Spread facilities
        lat_offset = (i % 5 - 2) * 0.02
        lon_offset = (i // 5 - 2) * 0.02

        location = Location(
            latitude=base_location.latitude + lat_offset,
            longitude=base_location.longitude + lon_offset,
        )

        template = facility_templates[i % len(facility_templates)]

        facility = HealthFacility(
            facility_id=f"facility_{i}",
            name=f"{template['type']} {i}",
            facility_type=template["type"],
            location=location,
            capacity=template["capacity"],
            services_offered=template["services"],
        )
        facilities.append(facility)

    logger.info(f"Created {len(facilities)} healthcare facilities")
    return facilities


def create_sample_environmental_data():
    """Create comprehensive sample environmental health data."""
    logger.info("Creating sample environmental health data...")

    base_time = datetime.now(UTC)
    base_location = Location(latitude=34.0522, longitude=-118.2437)

    parameters = ["PM2.5", "PM10", "NO2", "Temperature", "Humidity", "O3"]
    readings = []

    # Create readings across space and time
    for day in range(7):  # 7 days
        current_date = base_time - timedelta(days=6 - day)

        for hour in range(24):  # Hourly readings
            timestamp = current_date + timedelta(hours=hour)

            for loc_idx in range(10):  # 10 locations
                # Spread locations
                lat_offset = (loc_idx % 4 - 2) * 0.01
                lon_offset = (loc_idx // 4 - 1) * 0.01

                location = Location(
                    latitude=base_location.latitude + lat_offset,
                    longitude=base_location.longitude + lon_offset,
                )

                for param in parameters:
                    # Generate realistic values with temporal and spatial variation
                    base_value = {
                        "PM2.5": 15,
                        "PM10": 25,
                        "NO2": 20,
                        "Temperature": 22,
                        "Humidity": 60,
                        "O3": 30,
                    }[param]

                    # Add temporal patterns
                    if param in ["Temperature", "Humidity"]:
                        # Diurnal variation
                        time_factor = np.sin(2 * np.pi * hour / 24)
                        base_value += 5 * time_factor
                    elif param in ["PM2.5", "PM10", "NO2"]:
                        # Traffic/morning peak
                        time_factor = 1 + 0.5 * np.exp(-((hour - 8) ** 2) / 4)
                        base_value *= time_factor

                    # Add spatial variation
                    spatial_factor = 1 + 0.2 * np.sin(loc_idx)
                    base_value *= spatial_factor

                    # Add noise
                    value = base_value + np.random.normal(0, base_value * 0.1)

                    # Ensure positive values
                    value = max(0.1, value)

                    reading = EnvironmentalData(
                        data_id=f"{param}_{loc_idx}_{day}_{hour}",
                        parameter_name=param,
                        value=round(value, 2),
                        unit={
                            "PM2.5": "µg/m³",
                            "PM10": "µg/m³",
                            "NO2": "ppb",
                            "Temperature": "°C",
                            "Humidity": "%",
                            "O3": "ppb",
                        }[param],
                        location=location,
                        timestamp=timestamp,
                    )
                    readings.append(reading)

    logger.info(f"Created {len(readings)} environmental readings")
    return readings


def create_sample_population_data():
    """Create sample population data."""
    logger.info("Creating sample population data...")

    base_location = Location(latitude=34.0522, longitude=-118.2437)

    population_areas = []
    for i in range(5):
        lat_offset = (i % 3 - 1) * 0.03
        lon_offset = (i // 3) * 0.03

        location = Location(
            latitude=base_location.latitude + lat_offset,
            longitude=base_location.longitude + lon_offset,
        )

        population = PopulationData(
            area_id=f"area_{i}",
            population_count=50000 + i * 20000,
            age_distribution={
                "0-18": int((50000 + i * 20000) * (0.25 - i * 0.02)),
                "19-65": int((50000 + i * 20000) * (0.60 - i * 0.01)),
                "65+": int((50000 + i * 20000) * (0.15 + i * 0.03)),
            },
        )
        population_areas.append(population)

    logger.info(f"Created {len(population_areas)} population areas")
    return population_areas


def demonstrate_active_inference_disease_analysis():
    """Demonstrate Active Inference-based disease surveillance."""
    logger.info("\n" + "=" * 60)
    logger.info("ACTIVE INFERENCE DISEASE SURVEILLANCE ANALYSIS")
    logger.info("=" * 60)

    # Create data
    disease_reports = create_sample_disease_data()
    population_data = create_sample_population_data()

    # Initialize Active Inference analyzer
    analyzer = ActiveInferenceDiseaseAnalyzer(
        reports=disease_reports, population_data=population_data
    )

    # Perform comprehensive analysis
    logger.info("Running Active Inference analysis...")
    results = analyzer.analyze_with_active_inference(time_window_days=7)

    # Display results
    logger.info("BELIEF STATES:")
    for state, value in results["belief_states"].items():
        precision = results["belief_precisions"][state]
        logger.info(f"  {state}: {value:.3f} (precision: {precision:.3f})")
    logger.info("\nOBSERVATIONS:")
    for obs, value in results["observations"].items():
        logger.info(f"  {obs}: {value:.3f}")
    logger.info(
        "\nTRADITIONAL HOTSPOTS FOUND: {}".format(len(results["traditional_hotspots"]))
    )
    for hotspot in results["traditional_hotspots"][:3]:  # Show first 3
        location = hotspot.get("location", {})
        latitude = _coordinate_value(location, "latitude")
        longitude = _coordinate_value(location, "longitude")
        logger.info(
            f"  {hotspot.get('comment', 'Hotspot')} @ ({latitude:.2f}, {longitude:.2f}) "
            f"- Cases: {hotspot.get('case_count', 0):.0f}"
        )
    logger.info(
        "\nENHANCED HOTSPOTS FOUND: {}".format(len(results["enhanced_hotspots"]))
    )
    for hotspot in results["enhanced_hotspots"][:3]:  # Show first 3
        location = hotspot.get("location", {})
        latitude = _coordinate_value(location, "latitude")
        longitude = _coordinate_value(location, "longitude")
        logger.info(
            f"  {hotspot.get('comment', 'Enhanced hotspot')} @ ({latitude:.2f}, {longitude:.2f}) "
            f"- Cases: {hotspot.get('case_count', 0):.0f}, "
            f"Radius: {hotspot.get('radius_km', 0.0):.2f} km, "
            f"Confidence: {hotspot.get('confidence', 0.0):.2f}"
        )
    logger.info("\nPREDICTIONS:")
    predictions = results["predictions"]
    logger.info(f"  Short-term risk: {predictions['short_term_risk']:.3f}")
    logger.info(f"  Confidence: {predictions.get('confidence', 0):.3f}")
    logger.info("Trend: {}".format(predictions["trend"]))

    logger.info("\nRISK ASSESSMENT:")
    risk = results["risk_assessment"]
    logger.info("Risk Level: {}".format(risk["risk_level"]))
    logger.info("Risk Score: {:.3f}".format(risk["score"]))
    logger.info("Risk Factors:")
    for factor, value in risk["factors"].items():
        logger.info(f"  {factor}: {value:.3f}")

    logger.info("\nRECOMMENDATIONS:")
    for rec in results["recommendations"]:
        logger.info(f"  • {rec}")

    return results


def demonstrate_healthcare_accessibility_analysis():
    """Demonstrate healthcare accessibility analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("HEALTHCARE ACCESSIBILITY ANALYSIS")
    logger.info("=" * 60)

    # Create data
    facilities = create_sample_healthcare_data()
    population_data = create_sample_population_data()

    # Initialize analyzer
    analyzer = HealthcareAccessibilityAnalyzer(
        facilities=facilities, population_data=population_data
    )

    # Test location for accessibility analysis
    test_location = Location(latitude=34.0522, longitude=-118.2437)

    # Find nearby facilities
    logger.info("Finding facilities within 5km...")
    nearby_facilities = analyzer.find_facilities_in_radius(
        center_loc=test_location, radius_km=5.0
    )

    logger.info(f"Found {len(nearby_facilities)} facilities within 5km:")
    for facility in nearby_facilities[:5]:  # Show first 5
        distance = analyzer._calculate_distance(test_location, facility.location)
        logger.info(
            f"  {facility.name} ({distance:.2f} km): {facility.facility_type} (capacity {facility.capacity})"
        )
    # Find nearest facility
    logger.info("\nFinding nearest facility...")
    nearest_result = analyzer.get_nearest_facility(loc=test_location)

    if nearest_result:
        facility, distance = nearest_result
        logger.info(f"Nearest facility: {facility.name} ({distance:.2f} km)")
        logger.info(f"  Type: {facility.facility_type}")
        logger.info(f"  Capacity: {facility.capacity}")
        logger.info("  Services: {}".format(", ".join(facility.services_offered)))

    # Calculate facility-to-population ratios
    logger.info("\nCalculating facility-to-population ratios...")
    for pop_area in population_data:
        ratio_result = analyzer.calculate_facility_to_population_ratio(
            area_id=pop_area.area_id
        )

        if ratio_result:
            logger.info(
                "Area {}: {:.2f} facilities per 1000 people".format(
                    pop_area.area_id, ratio_result["ratio_per_1000_pop"]
                )
            )
            logger.info(f"  Population: {pop_area.population_count}")
            logger.info("  Facilities: {}".format(ratio_result["facility_count"]))

    return {
        "nearby_facilities": nearby_facilities,
        "nearest_facility": nearest_result,
        "ratios": [
            analyzer.calculate_facility_to_population_ratio(p.area_id)
            for p in population_data
        ],
    }


def demonstrate_environmental_health_analysis():
    """Demonstrate environmental health analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("ENVIRONMENTAL HEALTH ANALYSIS")
    logger.info("=" * 60)

    # Create data
    environmental_readings = create_sample_environmental_data()

    # Initialize analyzer
    analyzer = EnvironmentalHealthAnalyzer(
        environmental_readings=environmental_readings
    )

    # Test location for analysis
    test_location = Location(latitude=34.0522, longitude=-118.2437)

    # Get readings near location
    logger.info("Getting environmental readings within 2km...")
    nearby_readings = analyzer.get_environmental_readings_near_location(
        center_loc=test_location, radius_km=2.0, parameter_name="PM2.5"
    )

    logger.info(f"Found {len(nearby_readings)} PM2.5 readings within 2km")

    if nearby_readings:
        values = [r.value for r in nearby_readings]
        logger.info(f"  Average PM2.5: {np.mean(values):.2f} µg/m³")
        logger.info(f"  Min PM2.5: {np.min(values):.2f} µg/m³")
        logger.info(f"  Max PM2.5: {np.max(values):.2f} µg/m³")

    # Calculate average exposure for multiple locations
    logger.info("\nCalculating average exposure for multiple locations...")
    target_locations = [
        test_location,
        Location(
            latitude=test_location.latitude + 0.01,
            longitude=test_location.longitude + 0.01,
        ),
        Location(
            latitude=test_location.latitude - 0.01,
            longitude=test_location.longitude - 0.01,
        ),
    ]

    exposure_results = analyzer.calculate_average_exposure(
        target_locations=target_locations,
        radius_km=1.0,
        parameter_name="PM2.5",
        time_window_days=1,
    )

    logger.info("Average PM2.5 exposure (last 24 hours):")
    for key, value in exposure_results.items():
        logger.info(f"  {key}: {value if value else 0:.2f} µg/m³")

    # Get readings with time filter
    logger.info("\nGetting recent readings (last 12 hours)...")
    recent_readings = analyzer.get_environmental_readings_near_location(
        center_loc=test_location,
        radius_km=5.0,
        parameter_name="Temperature",
        start_time=datetime.now(UTC) - timedelta(hours=12),
        end_time=datetime.now(UTC),
    )

    logger.info(f"Found {len(recent_readings)} temperature readings in last 12 hours")

    if recent_readings:
        temps = [r.value for r in recent_readings]
        logger.info(f"  Average temperature: {np.mean(temps):.1f}°C")
        logger.info(
            f"  Temperature range: {np.min(temps):.1f}°C - {np.max(temps):.1f}°C"
        )

    return {
        "nearby_readings": nearby_readings,
        "exposure_results": exposure_results,
        "recent_readings": recent_readings,
    }


def demonstrate_advanced_geospatial_analysis():
    """Demonstrate advanced geospatial analysis capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("ADVANCED GEOSPATIAL ANALYSIS")
    logger.info("=" * 60)

    # Create sample location data
    base_location = Location(latitude=34.0522, longitude=-118.2437)

    # Create clustered locations
    locations = []

    # Cluster 1
    for _ in range(15):
        locations.append(
            Location(
                latitude=base_location.latitude + np.random.uniform(-0.005, 0.005),
                longitude=base_location.longitude + np.random.uniform(-0.005, 0.005),
            )
        )

    # Cluster 2
    for _ in range(10):
        locations.append(
            Location(
                latitude=base_location.latitude
                + 0.02
                + np.random.uniform(-0.003, 0.003),
                longitude=base_location.longitude
                + 0.02
                + np.random.uniform(-0.003, 0.003),
            )
        )

    # Isolated points
    for _ in range(5):
        locations.append(
            Location(
                latitude=base_location.latitude + np.random.uniform(-0.05, 0.05),
                longitude=base_location.longitude + np.random.uniform(-0.05, 0.05),
            )
        )

    # Validate geographic bounds
    logger.info("Validating geographic bounds...")
    validation_result = validate_geographic_bounds(locations)

    logger.info(
        "Validation result: {}".format(
            "Valid" if validation_result["valid"] else "Invalid"
        )
    )
    logger.info("Total locations: {}".format(validation_result["total_locations"]))
    logger.info(
        "Invalid locations: {}".format(len(validation_result["invalid_locations"]))
    )

    if validation_result["warnings"]:
        logger.info("Warnings:")
        for warning in validation_result["warnings"]:
            logger.info(f"  • {warning}")

    # Perform spatial clustering
    logger.info("\nPerforming spatial clustering...")
    clusters = spatial_clustering(locations, eps_km=1.0, min_samples=3)

    logger.info(f"Found {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        logger.info(f"  Cluster {i + 1}: {len(cluster)} points")

    # Calculate spatial statistics
    logger.info("\nCalculating spatial statistics...")
    stats = calculate_spatial_statistics(locations)

    logger.info("Spatial statistics:")
    logger.info("  Total points: {}".format(stats["count"]))
    logger.info(
        "  Centroid: {:.4f}, {:.4f}".format(
            stats["centroid_lat"], stats["centroid_lon"]
        )
    )
    logger.info(
        "  Mean distance from centroid: {:.3f} km".format(
            stats["mean_distance_from_centroid"]
        )
    )
    logger.info(
        "  Bounding box: {:.3f} x {:.3f} km".format(
            stats["bbox_width_km"], stats["bbox_height_km"]
        )
    )

    # Calculate spatial autocorrelation
    logger.info("\nCalculating spatial autocorrelation...")
    # Create synthetic values for autocorrelation analysis
    values = [10 + i * 0.5 + np.random.normal(0, 2) for i in range(len(locations))]

    autocorr_result = calculate_spatial_autocorrelation(
        locations, values, max_distance_km=5.0
    )

    logger.info("Spatial autocorrelation (Moran's I):")
    logger.info("  Moran's I: {:.3f}".format(autocorr_result["morans_i"]))
    logger.info("  Expected I: {:.3f}".format(autocorr_result["expected_i"]))
    logger.info("  Z-score: {:.3f}".format(autocorr_result["z_score"]))
    logger.info("  P-value: {:.3f}".format(autocorr_result["p_value"]))

    if autocorr_result["p_value"] < 0.05:
        logger.info("  → Significant spatial autocorrelation detected!")
    else:
        logger.info("  → No significant spatial autocorrelation")

    # Calculate hotspot statistics
    logger.info("\nCalculating hotspot statistics...")
    case_counts = [int(5 + v * 0.1) for v in values]  # Convert values to case counts

    hotspot_stats = calculate_hotspot_statistics(locations, case_counts)

    logger.info("Hotspot analysis:")
    logger.info("  Total cases: {}".format(hotspot_stats["total_cases"]))
    logger.info("  Total locations: {}".format(hotspot_stats["total_locations"]))
    logger.info("  Hotspots identified: {}".format(hotspot_stats["hotspots_count"]))
    logger.info("  Risk zones identified: {}".format(hotspot_stats["risk_zones_count"]))

    if hotspot_stats["hotspots"]:
        logger.info("  Top hotspot:")
        top_hotspot = hotspot_stats["hotspots"][0]
        logger.info(
            "    Location: {:.4f}, {:.4f}".format(
                top_hotspot["location"].latitude, top_hotspot["location"].longitude
            )
        )
        logger.info("    Cases: {}".format(top_hotspot["case_count"]))
        logger.info("    Relative risk: {:.2f}".format(top_hotspot["relative_risk"]))

    return {
        "validation": validation_result,
        "clusters": clusters,
        "statistics": stats,
        "autocorrelation": autocorr_result,
        "hotspots": hotspot_stats,
    }


def main():
    """Main function demonstrating all HEALTH module capabilities."""
    logger.info("GEO-INFER-HEALTH Advanced Analysis Demonstration")
    logger.info("=" * 60)

    try:
        # Run all analyses
        disease_results = demonstrate_active_inference_disease_analysis()
        healthcare_results = demonstrate_healthcare_accessibility_analysis()
        environmental_results = demonstrate_environmental_health_analysis()
        geospatial_results = demonstrate_advanced_geospatial_analysis()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)

        logger.info("Disease Surveillance:")
        logger.info(
            "  • Analyzed {} disease reports".format(
                len(disease_results.get("traditional_hotspots", [])) * 10  # Estimate
            )
        )
        logger.info(
            "  • Identified {} hotspots using Active Inference".format(
                len(disease_results.get("enhanced_hotspots", []))
            )
        )
        logger.info(
            "  • Risk level: {}".format(
                disease_results.get("risk_assessment", {}).get("risk_level", "Unknown")
            )
        )

        logger.info("\nHealthcare Accessibility:")
        logger.info(
            "  • Analyzed {} healthcare facilities".format(
                len(healthcare_results.get("nearby_facilities", []))
            )
        )
        if healthcare_results.get("nearest_facility"):
            facility, distance = healthcare_results["nearest_facility"]
            logger.info(f"  • Nearest facility: {facility.name} ({distance:.1f} km)")

        logger.info("\nEnvironmental Health:")
        logger.info(
            "  • Analyzed {} environmental readings".format(
                len(environmental_results.get("nearby_readings", []))
            )
        )
        exposure = environmental_results.get("exposure_results", {})
        if exposure:
            avg_exposure = [v for v in exposure.values() if v]
            if avg_exposure:
                logger.info(
                    f"  • Average PM2.5 exposure: {sum(avg_exposure) / len(avg_exposure):.1f} µg/m³"
                )

        logger.info("\nAdvanced Geospatial:")
        logger.info(
            "  • Validated {} geographic locations".format(
                geospatial_results.get("validation", {}).get("total_locations", 0)
            )
        )
        logger.info(
            "  • Identified {} spatial clusters".format(
                len(geospatial_results.get("clusters", []))
            )
        )
        logger.info(
            "  • Found {} disease hotspots".format(
                geospatial_results.get("hotspots", {}).get("hotspots_count", 0)
            )
        )

        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        # Save results summary
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "disease_analysis": {
                "hotspots": len(disease_results.get("enhanced_hotspots", [])),
                "risk_level": disease_results.get("risk_assessment", {}).get(
                    "risk_level"
                ),
            },
            "healthcare_analysis": {
                "facilities": len(healthcare_results.get("nearby_facilities", [])),
                "nearest_distance": healthcare_results.get(
                    "nearest_facility", [None, None]
                )[1],
            },
            "environmental_analysis": {
                "readings": len(environmental_results.get("nearby_readings", []))
            },
            "geospatial_analysis": {
                "clusters": len(geospatial_results.get("clusters", [])),
                "hotspots": geospatial_results.get("hotspots", {}).get(
                    "hotspots_count", 0
                ),
            },
        }

        # Save to file
        import json

        with open("health_analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Results summary saved to health_analysis_summary.json")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
