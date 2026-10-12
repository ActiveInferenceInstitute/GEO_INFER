"""
Integration tests for full GEO-INFER-HEALTH workflows.
"""

import pytest
from datetime import datetime, timedelta, UTC

from geo_infer_health.core import (
    DiseaseHotspotAnalyzer,
    HealthcareAccessibilityAnalyzer,
    EnvironmentalHealthAnalyzer,
)
from geo_infer_health.models import (
    DiseaseReport,
    HealthFacility,
    PopulationData,
    EnvironmentalData,
    Location,
)


class TestDiseaseSurveillanceWorkflow:
    """Test complete disease surveillance workflow."""

    def test_complete_disease_analysis_workflow(self):
        """Test end-to-end disease surveillance analysis."""
        # Create comprehensive test dataset
        base_time = datetime.now(UTC)
        center_location = Location(latitude=34.0522, longitude=-118.2437)

        # Create disease reports around center
        reports = []
        for i in range(50):
            # Create reports in a cluster around center
            lat_offset = (i % 7 - 3) * 0.01  # Spread over ~0.6km
            lon_offset = (i // 7 - 3) * 0.01

            location = Location(
                latitude=center_location.latitude + lat_offset,
                longitude=center_location.longitude + lon_offset,
            )

            report = DiseaseReport(
                report_id=f"workflow_report_{i}",
                disease_code="COVID-19" if i % 3 == 0 else "FLU",
                location=location,
                report_date=base_time - timedelta(days=i % 14),
                case_count=1 + (i % 5),  # 1-5 cases per report
                source="Test Surveillance System",
            )
            reports.append(report)

        # Create population data
        population_data = [
            PopulationData(
                area_id="test_area",
                population_count=100000,
                age_distribution={"0-18": 25000, "19-65": 60000, "65+": 15000},
            )
        ]

        # Initialize analyzer
        analyzer = DiseaseHotspotAnalyzer(
            reports=reports, population_data=population_data
        )

        # Test hotspot detection
        hotspots = analyzer.identify_simple_hotspots(
            threshold_case_count=5, scan_radius_km=1.0
        )

        assert len(hotspots) > 0

        # Test incidence rate calculation
        rate, cases, population, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                center_loc=center_location, radius_km=2.0, time_window_days=7
            )
        )

        assert rate >= 0
        assert cases > 0
        assert population > 0

        # Verify that hotspots are in high-incidence areas
        for hotspot in hotspots:
            hotspot_location = Location(
                latitude=hotspot["location"]["latitude"],
                longitude=hotspot["location"]["longitude"],
            )

            hotspot_rate, _, _, population_estimated = (
                analyzer.calculate_local_incidence_rate(
                    center_loc=hotspot_location, radius_km=hotspot["radius_km"]
                )
            )

            # Hotspot should have relatively high incidence
            assert hotspot_rate > 0

    def test_disease_data_integration(self):
        """Test integration of disease data with different sources."""
        # Create reports from different sources
        sources = ["Hospital A", "Clinic B", "Public Health Dept", "Private Lab"]
        diseases = ["COVID-19", "Influenza", "RSV", "Pertussis"]

        reports = []
        base_location = Location(latitude=34.0522, longitude=-118.2437)
        base_time = datetime.now(UTC)

        for i in range(40):
            location = Location(
                latitude=base_location.latitude + (i % 6 - 3) * 0.005,
                longitude=base_location.longitude + (i // 6 - 3) * 0.005,
            )

            report = DiseaseReport(
                report_id=f"integration_report_{i}",
                disease_code=diseases[i % len(diseases)],
                location=location,
                report_date=base_time - timedelta(hours=i * 2),
                case_count=1 + (i % 3),
                source=sources[i % len(sources)],
            )
            reports.append(report)

        analyzer = DiseaseHotspotAnalyzer(reports=reports, population_data=[])

        # Test that all sources are represented
        sources_in_data = set(report.source for report in reports)
        assert len(sources_in_data) == len(sources)

        # Test temporal analysis
        recent_reports = [
            r for r in reports if r.report_date > base_time - timedelta(hours=24)
        ]
        recent_analyzer = DiseaseHotspotAnalyzer(
            reports=recent_reports, population_data=[]
        )

        # Recent data should have fewer hotspots (less data)
        all_hotspots = analyzer.identify_simple_hotspots(threshold_case_count=3)
        recent_hotspots = recent_analyzer.identify_simple_hotspots(
            threshold_case_count=3
        )

        # Recent should have fewer or equal hotspots
        assert len(recent_hotspots) <= len(all_hotspots)


class TestHealthcareAccessibilityWorkflow:
    """Test complete healthcare accessibility workflow."""

    def test_complete_accessibility_analysis_workflow(self):
        """Test end-to-end healthcare accessibility analysis."""
        # Create comprehensive healthcare facilities dataset
        base_location = Location(latitude=34.0522, longitude=-118.2437)

        facilities = []
        facility_types = ["Hospital", "Clinic", "Emergency", "Specialist"]
        services = [
            ["Emergency", "Surgery", "Cardiology"],
            ["General Checkup", "Vaccinations", "Pediatrics"],
            ["Emergency", "Trauma"],
            ["Cardiology", "Neurology", "Oncology"],
        ]

        for i in range(20):
            # Spread facilities around center
            lat_offset = (i % 5 - 2) * 0.02  # Spread over ~2km
            lon_offset = (i // 5 - 2) * 0.02

            location = Location(
                latitude=base_location.latitude + lat_offset,
                longitude=base_location.longitude + lon_offset,
            )

            facility = HealthFacility(
                facility_id=f"workflow_facility_{i}",
                name=f"Test Facility {i}",
                facility_type=facility_types[i % len(facility_types)],
                location=location,
                capacity=50 + (i % 4) * 50,
                services_offered=services[i % len(services)],
            )
            facilities.append(facility)

        # Create population data
        population_data = [
            PopulationData(
                area_id="test_city",
                population_count=500000,
                age_distribution={"0-18": 125000, "19-65": 300000, "65+": 75000},
            )
        ]

        # Initialize analyzer
        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=population_data
        )

        # Test facility search
        center = Location(latitude=34.0522, longitude=-118.2437)
        nearby_facilities = analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=5.0
        )

        assert len(nearby_facilities) > 0

        # Test nearest facility
        nearest_result = analyzer.get_nearest_facility(loc=center)
        assert nearest_result is not None

        nearest_facility, distance = nearest_result
        assert distance >= 0

        # Test facility-to-population ratio
        ratio_result = analyzer.calculate_facility_to_population_ratio(
            area_id="test_city"
        )

        assert ratio_result is not None
        assert ratio_result["facility_count"] == len(facilities)
        assert ratio_result["population"] == 500000

        # Verify ratio calculation
        expected_ratio = len(facilities) / 500000 * 1000
        assert ratio_result["ratio_per_1000_pop"] == pytest.approx(expected_ratio)

    def test_accessibility_service_filtering(self):
        """Test accessibility analysis with service filtering."""
        base_location = Location(latitude=34.0522, longitude=-118.2437)

        # Create facilities with different service offerings
        facilities = [
            HealthFacility(
                facility_id="hospital_1",
                name="General Hospital",
                facility_type="Hospital",
                location=base_location,
                capacity=200,
                services_offered=["Emergency", "Surgery", "Cardiology", "Pediatrics"],
            ),
            HealthFacility(
                facility_id="clinic_1",
                name="Specialty Clinic",
                facility_type="Clinic",
                location=Location(
                    latitude=base_location.latitude + 0.01,
                    longitude=base_location.longitude + 0.01,
                ),
                capacity=50,
                services_offered=["Cardiology", "Neurology"],
            ),
            HealthFacility(
                facility_id="urgent_care",
                name="Urgent Care Center",
                facility_type="Clinic",
                location=Location(
                    latitude=base_location.latitude - 0.01,
                    longitude=base_location.longitude - 0.01,
                ),
                capacity=30,
                services_offered=["Emergency", "General Checkup"],
            ),
        ]

        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=[]
        )

        # Test emergency service search
        emergency_facilities = analyzer.find_facilities_in_radius(
            center_loc=base_location, radius_km=10.0, required_services=["Emergency"]
        )

        assert len(emergency_facilities) == 2  # Hospital and urgent care

        # Test cardiology service search
        cardiology_facilities = analyzer.find_facilities_in_radius(
            center_loc=base_location, radius_km=10.0, required_services=["Cardiology"]
        )

        assert len(cardiology_facilities) == 2  # Hospital and specialty clinic

        # Test nearest facility with service requirements
        nearest_emergency = analyzer.get_nearest_facility(
            loc=base_location, required_services=["Emergency"]
        )

        assert nearest_emergency is not None
        emergency_facility, _ = nearest_emergency
        assert "Emergency" in emergency_facility.services_offered


class TestEnvironmentalHealthWorkflow:
    """Test complete environmental health workflow."""

    def test_complete_environmental_analysis_workflow(self):
        """Test end-to-end environmental health analysis."""
        # Create comprehensive environmental dataset
        base_location = Location(latitude=34.0522, longitude=-118.2437)
        base_time = datetime.now(UTC)

        readings = []
        parameters = ["PM2.5", "PM10", "NO2", "Temperature", "Humidity"]

        for i in range(100):
            # Spread readings around center
            lat_offset = (i % 10 - 5) * 0.005  # Spread over ~0.5km
            lon_offset = (i // 10 - 5) * 0.005

            location = Location(
                latitude=base_location.latitude + lat_offset,
                longitude=base_location.longitude + lon_offset,
            )

            # Cycle through parameters
            param_idx = i % len(parameters)
            parameter = parameters[param_idx]

            # Generate realistic values
            if parameter in ["PM2.5", "PM10"]:
                value = 5 + (i % 20)  # 5-24 µg/m³
                unit = "µg/m³"
            elif parameter == "NO2":
                value = 10 + (i % 30)  # 10-39 ppb
                unit = "ppb"
            elif parameter == "Temperature":
                value = 15 + (i % 20)  # 15-34°C
                unit = "°C"
            else:  # Humidity
                value = 30 + (i % 40)  # 30-69%
                unit = "%"

            reading = EnvironmentalData(
                data_id=f"workflow_reading_{i}",
                parameter_name=parameter,
                value=value,
                unit=unit,
                location=location,
                timestamp=base_time - timedelta(hours=i % 48),  # Last 48 hours
            )
            readings.append(reading)

        # Initialize analyzer
        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings)

        # Test readings query
        center = Location(latitude=34.0522, longitude=-118.2437)
        nearby_readings = analyzer.get_environmental_readings_near_location(
            center_loc=center, radius_km=2.0, parameter_name="PM2.5"
        )

        assert len(nearby_readings) > 0

        # All should be PM2.5 readings
        for reading in nearby_readings:
            assert reading.parameter_name == "PM2.5"

        # Test average exposure calculation
        target_locations = [
            center,
            Location(
                latitude=center.latitude + 0.01, longitude=center.longitude + 0.01
            ),
            Location(
                latitude=center.latitude - 0.01, longitude=center.longitude - 0.01
            ),
        ]

        exposure_results = analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=1.0,
            parameter_name="PM2.5",
            time_window_days=2,
        )

        assert len(exposure_results) == len(target_locations)

        # Check that results are reasonable
        for loc in target_locations:
            key = f"{loc.latitude},{loc.longitude}"
            exposure = exposure_results[key]

            if exposure is not None:
                # Should be within expected range for PM2.5
                assert 5 <= exposure <= 25

    def test_environmental_temporal_analysis(self):
        """Test environmental data temporal analysis."""
        base_location = Location(latitude=34.0522, longitude=-118.2437)
        base_time = datetime.now(UTC)

        # Create readings over time
        readings = []
        for i in range(24):  # 24 hours of data
            reading = EnvironmentalData(
                data_id=f"temp_reading_{i}",
                parameter_name="PM2.5",
                value=10 + i % 10,  # Varying values
                unit="µg/m³",
                location=base_location,
                timestamp=base_time - timedelta(hours=23 - i),  # Chronological order
            )
            readings.append(reading)

        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings)

        # Test time-filtered queries
        recent_readings = analyzer.get_environmental_readings_near_location(
            center_loc=base_location,
            radius_km=1.0,
            parameter_name="PM2.5",
            start_time=base_time - timedelta(hours=6),
            end_time=base_time,
        )

        # Should have readings from last 6 hours
        assert len(recent_readings) <= 6

        # Test exposure with different time windows
        short_window_exposure = analyzer.calculate_average_exposure(
            target_locations=[base_location],
            radius_km=1.0,
            parameter_name="PM2.5",
            time_window_days=0.25,  # 6 hours
        )

        long_window_exposure = analyzer.calculate_average_exposure(
            target_locations=[base_location],
            radius_km=1.0,
            parameter_name="PM2.5",
            time_window_days=1.0,  # 24 hours
        )

        # Both should have values
        short_value = short_window_exposure[
            f"{base_location.latitude},{base_location.longitude}"
        ]
        long_value = long_window_exposure[
            f"{base_location.latitude},{base_location.longitude}"
        ]

        assert short_value is not None
        assert long_value is not None


class TestCrossModuleIntegration:
    """Test integration across different health modules."""

    def test_disease_and_environmental_integration(self):
        """Test integration between disease surveillance and environmental health."""
        base_location = Location(latitude=34.0522, longitude=-118.2437)
        base_time = datetime.now(UTC)

        # Create disease reports
        disease_reports = []
        for i in range(30):
            location = Location(
                latitude=base_location.latitude + (i % 6 - 3) * 0.01,
                longitude=base_location.longitude + (i // 6 - 2) * 0.01,
            )

            report = DiseaseReport(
                report_id=f"env_disease_{i}",
                disease_code="RESPIRATORY_INFECTION",
                location=location,
                report_date=base_time - timedelta(days=i % 7),
                case_count=1 + (i % 3),
                source="Hospital Network",
            )
            disease_reports.append(report)

        # Create environmental readings (air quality)
        env_readings = []
        for i in range(50):
            location = Location(
                latitude=base_location.latitude + (i % 7 - 3) * 0.008,
                longitude=base_location.longitude + (i // 7 - 3) * 0.008,
            )

            reading = EnvironmentalData(
                data_id=f"air_quality_{i}",
                parameter_name="PM2.5",
                value=8 + (i % 15),  # 8-22 µg/m³
                unit="µg/m³",
                location=location,
                timestamp=base_time - timedelta(hours=i % 48),
            )
            env_readings.append(reading)

        # Initialize analyzers
        disease_analyzer = DiseaseHotspotAnalyzer(
            reports=disease_reports, population_data=[]
        )
        env_analyzer = EnvironmentalHealthAnalyzer(environmental_readings=env_readings)

        # Find disease hotspots
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=3, scan_radius_km=1.0
        )

        # Check environmental conditions at hotspots
        for hotspot in hotspots:
            hotspot_location = Location(
                latitude=hotspot["location"]["latitude"],
                longitude=hotspot["location"]["longitude"],
            )

            # Get air quality near hotspot
            air_quality_readings = (
                env_analyzer.get_environmental_readings_near_location(
                    center_loc=hotspot_location,
                    radius_km=0.5,
                    parameter_name="PM2.5",
                    start_time=base_time - timedelta(days=7),
                )
            )

            # Calculate average air quality
            if air_quality_readings:
                avg_pm25 = sum(r.value for r in air_quality_readings) / len(
                    air_quality_readings
                )

                # Store for analysis - in real implementation, this could be used
                # to correlate disease incidence with air quality
                assert avg_pm25 > 0

        # Test combined spatial analysis
        # Both disease and environmental data should cover similar geographic areas
        disease_locations = [
            (r.location.latitude, r.location.longitude) for r in disease_reports
        ]
        env_locations = [
            (r.location.latitude, r.location.longitude) for r in env_readings
        ]

        # Check spatial overlap (simplified)
        disease_bbox = self._calculate_bbox(disease_locations)
        env_bbox = self._calculate_bbox(env_locations)

        # Bounding boxes should overlap significantly
        overlap = self._calculate_bbox_overlap(disease_bbox, env_bbox)
        assert overlap > 0.5  # At least 50% overlap

    def test_healthcare_and_population_integration(self):
        """Test integration between healthcare accessibility and population data."""
        base_location = Location(latitude=34.0522, longitude=-118.2437)

        # Create facilities
        facilities = []
        for i in range(15):
            location = Location(
                latitude=base_location.latitude + (i % 4 - 2) * 0.015,
                longitude=base_location.longitude + (i // 4 - 2) * 0.015,
            )

            facility = HealthFacility(
                facility_id=f"pop_facility_{i}",
                name=f"Test Facility {i}",
                facility_type="Hospital" if i % 3 == 0 else "Clinic",
                location=location,
                capacity=100 + (i % 5) * 50,
                services_offered=["Emergency", "General Checkup"],
            )
            facilities.append(facility)

        # Create population data for different areas
        population_data = []
        area_centers = [
            (base_location.latitude + 0.01, base_location.longitude + 0.01),
            (base_location.latitude - 0.01, base_location.longitude - 0.01),
            (base_location.latitude + 0.005, base_location.longitude - 0.005),
        ]

        for i in range(len(area_centers)):
            pop_data = PopulationData(
                area_id=f"area_{i}",
                population_count=50000 + i * 25000,
                age_distribution={
                    "0-18": int((50000 + i * 25000) * 0.25),
                    "19-65": int((50000 + i * 25000) * 0.6),
                    "65+": int((50000 + i * 25000) * 0.15),
                },
            )
            population_data.append(pop_data)

        # Initialize analyzer
        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=population_data
        )

        # Test accessibility analysis for different population areas
        accessibility_results = {}
        for pop_area, (lat, lon) in zip(population_data, area_centers, strict=True):
            # The area center stands in for the population-weighted centroid.
            area_center = Location(latitude=lat, longitude=lon)

            nearest = analyzer.get_nearest_facility(loc=area_center)
            if nearest:
                facility, distance = nearest
                accessibility_results[pop_area.area_id] = {
                    "distance_km": distance,
                    "population": pop_area.population_count,
                    "facility_type": facility.facility_type,
                }

        # Verify results
        assert len(accessibility_results) == len(population_data)

        for result in accessibility_results.values():
            assert result["distance_km"] >= 0
            assert result["population"] > 0
            assert result["facility_type"] in ["Hospital", "Clinic"]

    def _calculate_bbox(self, locations):
        """Calculate bounding box for a list of (lat, lon) tuples."""
        if not locations:
            return None

        lats, lons = zip(*locations)
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }

    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap area between two bounding boxes."""
        if not bbox1 or not bbox2:
            return 0

        # Calculate intersection
        min_lat = max(bbox1["min_lat"], bbox2["min_lat"])
        max_lat = min(bbox1["max_lat"], bbox2["max_lat"])
        min_lon = max(bbox1["min_lon"], bbox2["min_lon"])
        max_lon = min(bbox1["max_lon"], bbox2["max_lon"])

        if min_lat >= max_lat or min_lon >= max_lon:
            return 0  # No overlap

        # Calculate overlap ratio relative to the smaller bounding box.
        lat_overlap = max_lat - min_lat
        lon_overlap = max_lon - min_lon
        overlap_area = lat_overlap * lon_overlap
        area1 = (bbox1["max_lat"] - bbox1["min_lat"]) * (
            bbox1["max_lon"] - bbox1["min_lon"]
        )
        area2 = (bbox2["max_lat"] - bbox2["min_lat"]) * (
            bbox2["max_lon"] - bbox2["min_lon"]
        )
        smaller_area = min(area1, area2)
        return overlap_area / smaller_area if smaller_area else 0


class TestPerformanceIntegration:
    """Test performance of integrated workflows."""

    def test_large_scale_integration_performance(self):
        """Test performance with large-scale integrated dataset."""
        import time

        # Create large integrated dataset
        base_location = Location(latitude=34.0522, longitude=-118.2437)
        base_time = datetime.now(UTC)

        # Generate disease reports
        disease_reports = []
        for i in range(200):
            location = Location(
                latitude=base_location.latitude + (i % 14 - 7) * 0.005,
                longitude=base_location.longitude + (i // 14 - 7) * 0.005,
            )

            report = DiseaseReport(
                report_id=f"large_disease_{i}",
                disease_code="TEST_DISEASE",
                location=location,
                report_date=base_time - timedelta(hours=i % 168),  # Last week
                case_count=1 + (i % 4),
                source="Large Test Dataset",
            )
            disease_reports.append(report)

        # Generate healthcare facilities
        facilities = []
        for i in range(50):
            location = Location(
                latitude=base_location.latitude + (i % 7 - 3) * 0.02,
                longitude=base_location.longitude + (i // 7 - 3) * 0.02,
            )

            facility = HealthFacility(
                facility_id=f"large_facility_{i}",
                name=f"Large Test Facility {i}",
                facility_type="Hospital" if i % 4 == 0 else "Clinic",
                location=location,
                capacity=100 + (i % 5) * 30,
                services_offered=["Emergency", "General Checkup"],
            )
            facilities.append(facility)

        # Generate environmental readings
        env_readings = []
        for i in range(300):
            location = Location(
                latitude=base_location.latitude + (i % 17 - 8) * 0.003,
                longitude=base_location.longitude + (i // 17 - 8) * 0.003,
            )

            reading = EnvironmentalData(
                data_id=f"large_env_{i}",
                parameter_name="PM2.5" if i % 2 == 0 else "Temperature",
                value=10 + (i % 20),
                unit="µg/m³" if i % 2 == 0 else "°C",
                location=location,
                timestamp=base_time - timedelta(minutes=i % 1440),  # Last 24 hours
            )
            env_readings.append(reading)

        # Time integrated analysis
        start_time = time.time()

        # Disease analysis
        disease_analyzer = DiseaseHotspotAnalyzer(
            reports=disease_reports, population_data=[]
        )
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=5, scan_radius_km=1.0
        )

        # Healthcare analysis
        healthcare_analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=[]
        )
        nearest = healthcare_analyzer.get_nearest_facility(loc=base_location)

        # Environmental analysis
        env_analyzer = EnvironmentalHealthAnalyzer(environmental_readings=env_readings)
        exposure = env_analyzer.calculate_average_exposure(
            target_locations=[base_location],
            radius_km=2.0,
            parameter_name="PM2.5",
            time_window_days=1,
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (< 2 seconds for this dataset size)
        assert total_time < 2.0

        # Verify results
        assert len(hotspots) > 0
        assert nearest is not None
        assert len(exposure) == 1

        print(f"Large-scale integration test completed in {total_time:.2f}s")
