"""
Unit tests for healthcare accessibility functionality.
"""

import pytest

from geo_infer_health.core.healthcare_accessibility import (
    HealthcareAccessibilityAnalyzer,
)
from geo_infer_health.models import HealthFacility, Location, PopulationData


@pytest.mark.parametrize(
    "population, facility_type, expected_count",
    [(0, None, 2), (0, "hOSPiTaL", 1), (0, "Absent", 0), (1000, "Hospital", 1)],
)
def test_ratio_count_respects_filter_even_with_zero_population(
    population, facility_type, expected_count
):
    facilities = [
        HealthFacility(
            facility_id=kind,
            name=kind,
            facility_type=kind,
            location=Location(latitude=0, longitude=0),
        )
        for kind in ("Hospital", "Clinic")
    ]
    analyzer = HealthcareAccessibilityAnalyzer(
        facilities,
        [PopulationData(area_id="region", population_count=population)],
    )

    result = analyzer.calculate_facility_to_population_ratio("region", facility_type)

    assert result["facility_count"] == expected_count
    assert result["facility_type_filter"] == facility_type
    assert result["ratio_per_1000_pop"] == (
        float("inf") if population == 0 else expected_count / population * 1000
    )


def test_radius_search_measures_each_eligible_facility_once_and_keeps_ties(monkeypatch):
    from geo_infer_health.core import healthcare_accessibility as module

    def facility(identifier, latitude, kind="Hospital", services=("Emergency",)):
        return HealthFacility(
            facility_id=identifier,
            name=identifier,
            facility_type=kind,
            location=Location(latitude=latitude, longitude=0),
            services_offered=list(services),
        )

    eligible = [
        facility("outside", 1),
        facility("tie_first", 0),
        facility("near", 0.01),
        facility("tie_second", 0),
    ]
    facilities = eligible + [
        facility("clinic", 0, "Clinic"),
        facility("no_service", 0, services=()),
    ]
    calls = []
    original = module.haversine_distance

    def measured(first, second):
        calls.append(id(first))
        return original(first, second)

    monkeypatch.setattr(module, "haversine_distance", measured)
    analyzer = HealthcareAccessibilityAnalyzer(facilities)

    found = analyzer.find_facilities_in_radius(
        Location(latitude=0, longitude=0),
        2,
        facility_type="hospital",
        required_services=["Emergency"],
    )

    assert [item.facility_id for item in found] == ["tie_first", "tie_second", "near"]
    assert calls == [id(item.location) for item in eligible]


class TestHealthcareAccessibilityAnalyzer:
    """Test cases for HealthcareAccessibilityAnalyzer class."""

    def test_analyzer_creation(self, sample_health_facilities, sample_population_data):
        """Test creating a HealthcareAccessibilityAnalyzer instance."""
        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=sample_health_facilities, population_data=sample_population_data
        )

        assert len(analyzer.facilities) == len(sample_health_facilities)
        assert len(analyzer.population_data) == len(sample_population_data)

    def test_analyzer_creation_empty_data(self):
        """Test creating analyzer with empty data."""
        analyzer = HealthcareAccessibilityAnalyzer(facilities=[], population_data=[])

        assert len(analyzer.facilities) == 0
        assert len(analyzer.population_data) == 0

    def test_analyzer_creation_no_population_data(self, sample_health_facilities):
        """Test creating analyzer without population data."""
        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=sample_health_facilities, population_data=None
        )

        assert len(analyzer.facilities) == len(sample_health_facilities)
        assert analyzer.population_data == []


class TestFacilitySearch:
    """Test cases for finding facilities within radius."""

    def test_find_facilities_nearby(self, healthcare_analyzer, sample_locations):
        """Test finding facilities within radius."""
        center = sample_locations[0]
        radius_km = 10.0

        facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=radius_km
        )

        assert isinstance(facilities, list)

        # All returned facilities should be within radius
        for facility in facilities:
            distance = healthcare_analyzer._calculate_distance(
                center, facility.location
            )
            assert distance <= radius_km

    def test_find_facilities_with_type_filter(
        self, healthcare_analyzer, sample_locations
    ):
        """Test finding facilities with type filter."""
        center = sample_locations[0]
        radius_km = 1000.0  # Large radius to include all facilities
        facility_type = "Hospital"

        facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=radius_km, facility_type=facility_type
        )

        # All returned facilities should match the type
        for facility in facilities:
            assert facility.facility_type.lower() == facility_type.lower()

    def test_find_facilities_with_service_filter(
        self, healthcare_analyzer, sample_locations
    ):
        """Test finding facilities with service filter."""
        center = sample_locations[0]
        radius_km = 1000.0
        required_services = ["Emergency"]

        facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=radius_km, required_services=required_services
        )

        # All returned facilities should offer the required services
        for facility in facilities:
            assert all(
                service in facility.services_offered for service in required_services
            )

    def test_find_facilities_zero_radius(self, healthcare_analyzer, sample_locations):
        """Test finding facilities with zero radius."""
        center = sample_locations[0]
        radius_km = 0.0

        facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=radius_km
        )

        # Should find facilities at exact location (within floating point precision)
        # This might be zero or very few depending on exact coordinate matches
        assert isinstance(facilities, list)

    def test_find_facilities_large_radius(self, healthcare_analyzer, sample_locations):
        """Test finding facilities with large radius."""
        center = sample_locations[0]
        radius_km = 10000.0  # Very large radius

        facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=radius_km
        )

        # Should find all facilities
        assert len(facilities) == len(healthcare_analyzer.facilities)

    def test_find_facilities_no_facilities(self, sample_locations):
        """Test finding facilities when no facilities exist."""
        analyzer = HealthcareAccessibilityAnalyzer(facilities=[], population_data=[])
        center = sample_locations[0]
        radius_km = 1.0

        facilities = analyzer.find_facilities_in_radius(
            center_loc=center, radius_km=radius_km
        )

        assert facilities == []


class TestNearestFacility:
    """Test cases for finding nearest facility."""

    def test_get_nearest_facility(self, healthcare_analyzer, sample_locations):
        """Test finding nearest facility to a location."""
        test_location = sample_locations[0]

        result = healthcare_analyzer.get_nearest_facility(loc=test_location)

        if result is not None:
            facility, distance = result
            assert isinstance(facility, HealthFacility)
            assert isinstance(distance, float)
            assert distance >= 0

            # Verify this is actually the nearest
            for other_facility in healthcare_analyzer.facilities:
                if other_facility != facility:
                    other_distance = healthcare_analyzer._calculate_distance(
                        test_location, other_facility.location
                    )
                    assert distance <= other_distance

    def test_get_nearest_facility_with_filters(
        self, healthcare_analyzer, sample_locations
    ):
        """Test finding nearest facility with type and service filters."""
        test_location = sample_locations[0]
        facility_type = "Hospital"
        required_services = ["Emergency"]

        result = healthcare_analyzer.get_nearest_facility(
            loc=test_location,
            facility_type=facility_type,
            required_services=required_services,
        )

        if result is not None:
            facility, distance = result
            assert isinstance(facility, HealthFacility)
            assert facility.facility_type.lower() == facility_type.lower()
            assert all(
                service in facility.services_offered for service in required_services
            )

    def test_get_nearest_facility_no_matches(
        self, healthcare_analyzer, sample_locations
    ):
        """Test finding nearest facility with impossible filters."""
        test_location = sample_locations[0]

        # Use filters that no facility matches
        result = healthcare_analyzer.get_nearest_facility(
            loc=test_location,
            facility_type="NonExistentType",
            required_services=["ImpossibleService"],
        )

        assert result is None

    def test_get_nearest_facility_no_facilities(self, sample_locations):
        """Test finding nearest facility when no facilities exist."""
        analyzer = HealthcareAccessibilityAnalyzer(facilities=[], population_data=[])
        test_location = sample_locations[0]

        result = analyzer.get_nearest_facility(loc=test_location)

        assert result is None

    def test_get_nearest_facility_at_facility_location(self, healthcare_analyzer):
        """Test finding nearest facility when location is at a facility."""
        # Use location of first facility
        facility_location = healthcare_analyzer.facilities[0].location

        result = healthcare_analyzer.get_nearest_facility(loc=facility_location)

        assert result is not None
        facility, distance = result

        # Distance should be very small (essentially zero)
        assert distance < 0.001  # Less than 1 meter

        # Should find the facility at that location
        assert facility.location.latitude == pytest.approx(
            facility_location.latitude, abs=1e-6
        )
        assert facility.location.longitude == pytest.approx(
            facility_location.longitude, abs=1e-6
        )


class TestFacilityPopulationRatio:
    """Test cases for facility-to-population ratio calculations."""

    def test_calculate_ratio_with_data(self, healthcare_analyzer):
        """Test calculating facility-to-population ratio with data."""
        area_id = "area_1"  # Should match sample data

        result = healthcare_analyzer.calculate_facility_to_population_ratio(
            area_id=area_id
        )

        if result is not None:
            assert isinstance(result, dict)
            assert "area_id" in result
            assert "facility_count" in result
            assert "population" in result
            assert "ratio_per_1000_pop" in result

            # Verify calculations
            facility_count = result["facility_count"]
            population = result["population"]

            if population > 0:
                expected_ratio = (facility_count / population) * 1000
                assert result["ratio_per_1000_pop"] == pytest.approx(expected_ratio)

    def test_calculate_ratio_with_type_filter(self, healthcare_analyzer):
        """Test calculating ratio with facility type filter."""
        area_id = "area_1"
        facility_type = "Hospital"

        result = healthcare_analyzer.calculate_facility_to_population_ratio(
            area_id=area_id, facility_type=facility_type
        )

        if result is not None:
            assert result["facility_type_filter"] == facility_type

    def test_calculate_ratio_nonexistent_area(self, healthcare_analyzer):
        """Test calculating ratio for nonexistent area."""
        area_id = "nonexistent_area"

        result = healthcare_analyzer.calculate_facility_to_population_ratio(
            area_id=area_id
        )

        assert result is None

    def test_calculate_ratio_zero_population(self, healthcare_analyzer):
        """Test calculating ratio with zero population."""
        # Create analyzer with zero population
        zero_pop_data = [PopulationData(area_id="zero_area", population_count=0)]

        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=healthcare_analyzer.facilities, population_data=zero_pop_data
        )

        result = analyzer.calculate_facility_to_population_ratio(area_id="zero_area")

        if result is not None:
            assert result["population"] == 0
            # Ratio should be infinity or handled appropriately
            assert result["ratio_per_1000_pop"] == float("inf")

    def test_calculate_ratio_no_facilities(self):
        """Test calculating ratio when no facilities exist."""
        pop_data = [PopulationData(area_id="test_area", population_count=1000)]
        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=[], population_data=pop_data
        )

        result = analyzer.calculate_facility_to_population_ratio(area_id="test_area")

        if result is not None:
            assert result["facility_count"] == 0
            assert result["ratio_per_1000_pop"] == 0


class TestAccessibilityIntegration:
    """Test integration of accessibility analysis components."""

    def test_facility_search_and_nearest_consistency(
        self, healthcare_analyzer, sample_locations
    ):
        """Test consistency between facility search and nearest facility methods."""
        test_location = sample_locations[0]
        radius_km = 50.0

        # Find facilities within radius
        nearby_facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=test_location, radius_km=radius_km
        )

        # Find nearest facility
        nearest_result = healthcare_analyzer.get_nearest_facility(loc=test_location)

        if nearest_result is not None and nearby_facilities:
            nearest_facility, nearest_distance = nearest_result

            # Nearest facility should be in the nearby facilities list
            facility_ids = [f.facility_id for f in nearby_facilities]
            assert nearest_facility.facility_id in facility_ids

            # Distance to nearest should be less than or equal to radius
            assert nearest_distance <= radius_km

    def test_filter_consistency(self, healthcare_analyzer, sample_locations):
        """Test consistency of filtering across methods."""
        test_location = sample_locations[0]
        facility_type = "Hospital"
        required_services = ["Emergency"]

        # Test find_facilities_in_radius with filters
        filtered_facilities = healthcare_analyzer.find_facilities_in_radius(
            center_loc=test_location,
            radius_km=1000.0,
            facility_type=facility_type,
            required_services=required_services,
        )

        # Test get_nearest_facility with same filters
        nearest_result = healthcare_analyzer.get_nearest_facility(
            loc=test_location,
            facility_type=facility_type,
            required_services=required_services,
        )

        if nearest_result is not None and filtered_facilities:
            nearest_facility, _ = nearest_result

            # Nearest facility should be in the filtered list
            facility_ids = [f.facility_id for f in filtered_facilities]
            assert nearest_facility.facility_id in facility_ids

    def test_population_data_integration(self, healthcare_analyzer):
        """Test integration with population data."""
        # Test that population data is properly used in ratio calculations
        for pop_data in healthcare_analyzer.population_data:
            result = healthcare_analyzer.calculate_facility_to_population_ratio(
                area_id=pop_data.area_id
            )

            if result is not None:
                assert result["population"] == pop_data.population_count
                assert result["area_id"] == pop_data.area_id


class TestPerformance:
    """Test performance characteristics of accessibility functions."""

    def test_large_dataset_performance(self):
        """Test performance with larger facility dataset."""
        import time

        # Create larger dataset
        facilities = []
        locations = []

        for i in range(100):  # 100 facilities
            lat = 30 + (i % 10) * 0.5
            lon = -120 + (i // 10) * 0.5
            location = Location(latitude=lat, longitude=lon)
            locations.append(location)

            facility = HealthFacility(
                facility_id=f"perf_facility_{i}",
                name=f"Performance Facility {i}",
                facility_type="Clinic" if i % 2 == 0 else "Hospital",
                location=location,
                capacity=50 + i * 10,
                services_offered=["General Checkup", "Emergency"]
                if i % 3 == 0
                else ["General Checkup"],
            )
            facilities.append(facility)

        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=[]
        )

        # Time nearest facility search
        test_location = Location(latitude=35.0, longitude=-115.0)

        start_time = time.time()
        result = analyzer.get_nearest_facility(loc=test_location)
        end_time = time.time()

        # Should complete quickly (< 0.1 seconds for 100 facilities)
        duration = end_time - start_time
        assert duration < 0.1

        # Should find a result
        assert result is not None

    def test_facility_search_performance(self):
        """Test performance of facility search with filters."""
        import time

        # Create dataset
        facilities = []
        for i in range(50):
            facility = HealthFacility(
                facility_id=f"facility_{i}",
                name=f"Facility {i}",
                facility_type="Hospital" if i % 2 == 0 else "Clinic",
                location=Location(latitude=34.0 + i * 0.1, longitude=-118.0 + i * 0.1),
                capacity=100,
                services_offered=["Emergency"] if i % 3 == 0 else ["General Checkup"],
            )
            facilities.append(facility)

        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=[]
        )

        # Time search with filters
        test_location = Location(latitude=34.0, longitude=-118.0)

        start_time = time.time()
        results = analyzer.find_facilities_in_radius(
            center_loc=test_location,
            radius_km=10.0,
            facility_type="Hospital",
            required_services=["Emergency"],
        )
        end_time = time.time()

        # Should complete quickly
        duration = end_time - start_time
        assert duration < 0.05

        # Results should be filtered correctly
        for facility in results:
            assert facility.facility_type == "Hospital"
            assert "Emergency" in facility.services_offered


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_analyzer_operations(self):
        """Test operations on empty analyzer."""
        analyzer = HealthcareAccessibilityAnalyzer(facilities=[], population_data=[])

        # Should handle empty data gracefully
        facilities = analyzer.find_facilities_in_radius(
            Location(latitude=0, longitude=0), 1.0
        )
        assert facilities == []

        nearest = analyzer.get_nearest_facility(Location(latitude=0, longitude=0))
        assert nearest is None

    def test_single_facility_analysis(self):
        """Test analysis with single facility."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        facility = HealthFacility(
            facility_id="single_facility",
            name="Single Test Facility",
            facility_type="Hospital",
            location=location,
            capacity=100,
            services_offered=["Emergency", "Surgery"],
        )

        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=[facility], population_data=[]
        )

        # Test various operations
        facilities = analyzer.find_facilities_in_radius(location, 1.0)
        assert len(facilities) == 1

        nearest = analyzer.get_nearest_facility(location)
        assert nearest is not None
        nearest_facility, distance = nearest
        assert distance < 0.001  # Very close

    def test_identical_facility_locations(self):
        """Test handling of facilities at identical locations."""
        location = Location(latitude=34.0522, longitude=-118.2437)

        facilities = []
        for i in range(3):
            facility = HealthFacility(
                facility_id=f"identical_{i}",
                name=f"Identical Facility {i}",
                facility_type="Clinic",
                location=location,  # Same location
                capacity=50,
                services_offered=["General Checkup"],
            )
            facilities.append(facility)

        analyzer = HealthcareAccessibilityAnalyzer(
            facilities=facilities, population_data=[]
        )

        # Should find all facilities
        found_facilities = analyzer.find_facilities_in_radius(location, 0.1)
        assert len(found_facilities) == 3

        # Nearest facility should be one of them with zero distance
        nearest = analyzer.get_nearest_facility(location)
        assert nearest is not None
        _, distance = nearest
        assert distance < 0.001

    def test_distance_calculation_helper(self, healthcare_analyzer):
        """Test the internal distance calculation method."""
        loc1 = Location(latitude=34.0522, longitude=-118.2437)
        loc2 = Location(latitude=40.7128, longitude=-74.0060)

        # Test that distance is symmetric
        dist1 = healthcare_analyzer._calculate_distance(loc1, loc2)
        dist2 = healthcare_analyzer._calculate_distance(loc2, loc1)

        assert dist1 == dist2
        assert dist1 > 0

        # Test distance to self
        dist_self = healthcare_analyzer._calculate_distance(loc1, loc1)
        assert dist_self == pytest.approx(0.0, abs=1e-6)
