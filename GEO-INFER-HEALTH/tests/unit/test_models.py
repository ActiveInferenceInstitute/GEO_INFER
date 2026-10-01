"""
Unit tests for GEO-INFER-HEALTH data models.
"""

import pytest
from datetime import datetime, UTC

from geo_infer_health.models import (
    Location,
    HealthFacility,
    DiseaseReport,
    PopulationData,
    EnvironmentalData,
)


class TestLocation:
    """Test cases for Location model."""

    def test_location_creation(self):
        """Test creating a Location object."""
        location = Location(latitude=34.0522, longitude=-118.2437, crs="EPSG:4326")

        assert location.latitude == 34.0522
        assert location.longitude == -118.2437
        assert location.crs == "EPSG:4326"

    def test_location_default_crs(self):
        """Test default CRS value."""
        location = Location(latitude=0.0, longitude=0.0)

        assert location.crs == "EPSG:4326"

    def test_location_validation(self):
        """Test Location validation."""
        # Valid coordinates
        Location(latitude=90.0, longitude=180.0)
        Location(latitude=-90.0, longitude=-180.0)

        # Invalid coordinates should raise validation errors
        with pytest.raises(ValueError):
            Location(latitude=91.0, longitude=0.0)  # Latitude too high

        with pytest.raises(ValueError):
            Location(latitude=-91.0, longitude=0.0)  # Latitude too low

        with pytest.raises(ValueError):
            Location(latitude=0.0, longitude=181.0)  # Longitude too high

        with pytest.raises(ValueError):
            Location(latitude=0.0, longitude=-181.0)  # Longitude too low


class TestHealthFacility:
    """Test cases for HealthFacility model."""

    def test_health_facility_creation(self, sample_locations):
        """Test creating a HealthFacility object."""
        location = sample_locations[0]

        facility = HealthFacility(
            facility_id="test_hospital_001",
            name="Test General Hospital",
            facility_type="Hospital",
            location=location,
            capacity=500,
            services_offered=["Emergency", "Surgery", "Cardiology"],
        )

        assert facility.facility_id == "test_hospital_001"
        assert facility.name == "Test General Hospital"
        assert facility.facility_type == "Hospital"
        assert facility.capacity == 500
        assert facility.services_offered == ["Emergency", "Surgery", "Cardiology"]

    def test_health_facility_defaults(self, sample_locations):
        """Test HealthFacility default values."""
        location = sample_locations[0]

        facility = HealthFacility(
            facility_id="test_clinic_001",
            name="Test Clinic",
            facility_type="Clinic",
            location=location,
        )

        assert facility.capacity is None
        assert facility.services_offered == []
        assert facility.operating_hours is None
        assert facility.contact_info is None

    def test_health_facility_validation(self, sample_locations):
        """Test HealthFacility validation."""
        location = sample_locations[0]

        # Valid facility
        HealthFacility(
            facility_id="valid_id",
            name="Valid Name",
            facility_type="Hospital",
            location=location,
        )

        # Invalid capacity (negative)
        with pytest.raises(ValueError):
            HealthFacility(
                facility_id="invalid_capacity",
                name="Test",
                facility_type="Hospital",
                location=location,
                capacity=-1,
            )


class TestDiseaseReport:
    """Test cases for DiseaseReport model."""

    def test_disease_report_creation(self, sample_locations):
        """Test creating a DiseaseReport object."""
        location = sample_locations[0]
        report_date = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        report = DiseaseReport(
            report_id="covid_report_001",
            disease_code="COVID-19",
            location=location,
            report_date=report_date,
            case_count=25,
            source="CDC",
        )

        assert report.report_id == "covid_report_001"
        assert report.disease_code == "COVID-19"
        assert report.case_count == 25
        assert report.source == "CDC"

    def test_disease_report_defaults(self, sample_locations):
        """Test DiseaseReport default values."""
        location = sample_locations[0]
        report_date = datetime.now(UTC)

        report = DiseaseReport(
            report_id="minimal_report",
            disease_code="FLU",
            location=location,
            report_date=report_date,
            case_count=1,
        )

        assert report.source is None
        assert report.demographics is None
        assert report.notes is None

    def test_disease_report_validation(self, sample_locations):
        """Test DiseaseReport validation."""
        location = sample_locations[0]
        report_date = datetime.now(UTC)

        # Valid report
        DiseaseReport(
            report_id="valid_report",
            disease_code="VALID",
            location=location,
            report_date=report_date,
            case_count=5,
        )

        # Invalid case count (zero)
        with pytest.raises(ValueError):
            DiseaseReport(
                report_id="invalid_cases",
                disease_code="TEST",
                location=location,
                report_date=report_date,
                case_count=0,
            )

        # Invalid case count (negative)
        with pytest.raises(ValueError):
            DiseaseReport(
                report_id="invalid_cases_neg",
                disease_code="TEST",
                location=location,
                report_date=report_date,
                case_count=-1,
            )


class TestPopulationData:
    """Test cases for PopulationData model."""

    def test_population_data_creation(self):
        """Test creating a PopulationData object."""
        age_distribution = {"0-18": 30000, "19-65": 50000, "65+": 20000}

        pop_data = PopulationData(
            area_id="test_area_001",
            population_count=100000,
            age_distribution=age_distribution,
            other_demographics={"gender": {"male": 48000, "female": 52000}},
        )

        assert pop_data.area_id == "test_area_001"
        assert pop_data.population_count == 100000
        assert pop_data.age_distribution == age_distribution
        assert pop_data.other_demographics["gender"]["female"] == 52000

    def test_population_data_defaults(self):
        """Test PopulationData default values."""
        pop_data = PopulationData(area_id="minimal_area", population_count=50000)

        assert pop_data.age_distribution is None
        assert pop_data.other_demographics is None

    def test_population_data_validation(self):
        """Test PopulationData validation."""
        # Valid population data
        PopulationData(area_id="valid_area", population_count=1000)

        # Invalid population count (negative)
        with pytest.raises(ValueError):
            PopulationData(area_id="invalid_pop", population_count=-1)


class TestEnvironmentalData:
    """Test cases for EnvironmentalData model."""

    def test_environmental_data_creation(self, sample_locations):
        """Test creating an EnvironmentalData object."""
        location = sample_locations[0]
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        env_data = EnvironmentalData(
            data_id="pm25_reading_001",
            parameter_name="PM2.5",
            value=15.5,
            unit="µg/m³",
            location=location,
            timestamp=timestamp,
        )

        assert env_data.data_id == "pm25_reading_001"
        assert env_data.parameter_name == "PM2.5"
        assert env_data.value == 15.5
        assert env_data.unit == "µg/m³"

    def test_environmental_data_validation(self, sample_locations):
        """Test EnvironmentalData validation."""
        location = sample_locations[0]
        timestamp = datetime.now(UTC)

        # Valid environmental data
        EnvironmentalData(
            data_id="valid_reading",
            parameter_name="Temperature",
            value=25.0,
            unit="°C",
            location=location,
            timestamp=timestamp,
        )

        # Test with different parameter types
        EnvironmentalData(
            data_id="co_reading",
            parameter_name="CO",
            value=0.5,
            unit="ppm",
            location=location,
            timestamp=timestamp,
        )


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_location_serialization(self, sample_locations):
        """Test Location model serialization."""
        location = sample_locations[0]

        # Test Pydantic serialization
        data = location.model_dump()
        assert data["latitude"] == location.latitude
        assert data["longitude"] == location.longitude
        assert data["crs"] == location.crs

        # Test deserialization
        new_location = Location(**data)
        assert new_location == location

    def test_health_facility_serialization(self, sample_health_facilities):
        """Test HealthFacility model serialization."""
        facility = sample_health_facilities[0]

        data = facility.model_dump()
        assert data["facility_id"] == facility.facility_id
        assert data["name"] == facility.name

        # Test deserialization
        new_facility = HealthFacility(**data)
        assert new_facility.facility_id == facility.facility_id

    def test_disease_report_serialization(self, sample_disease_reports):
        """Test DiseaseReport model serialization."""
        report = sample_disease_reports[0]

        data = report.model_dump()
        assert data["report_id"] == report.report_id
        assert data["disease_code"] == report.disease_code

        # Test JSON serialization (datetime handling)
        json_str = report.model_dump_json()
        assert report.report_id in json_str

    def test_environmental_data_serialization(self, sample_environmental_data):
        """Test EnvironmentalData model serialization."""
        env_data = sample_environmental_data[0]

        data = env_data.model_dump()
        assert data["data_id"] == env_data.data_id
        assert data["parameter_name"] == env_data.parameter_name

        # Test deserialization
        new_env_data = EnvironmentalData(**data)
        assert new_env_data.data_id == env_data.data_id


class TestModelIntegration:
    """Test integration between different models."""

    def test_facility_location_integration(self, sample_health_facilities):
        """Test that facilities properly integrate with locations."""
        facility = sample_health_facilities[0]

        # Check that location is properly embedded
        assert hasattr(facility.location, "latitude")
        assert hasattr(facility.location, "longitude")

        # Check that we can access location properties
        assert isinstance(facility.location.latitude, float)
        assert isinstance(facility.location.longitude, float)

    def test_report_location_integration(self, sample_disease_reports):
        """Test that reports properly integrate with locations."""
        report = sample_disease_reports[0]

        # Check location integration
        assert hasattr(report.location, "latitude")
        assert hasattr(report.location, "longitude")

        # Check CRS consistency
        assert report.location.crs == "EPSG:4326"

    def test_environmental_location_integration(self, sample_environmental_data):
        """Test that environmental data properly integrates with locations."""
        env_data = sample_environmental_data[0]

        # Check location integration
        assert hasattr(env_data.location, "latitude")
        assert hasattr(env_data.location, "longitude")

        # Check coordinate system
        assert env_data.location.crs == "EPSG:4326"
