"""
Pytest configuration and fixtures for GEO-INFER-HEALTH tests.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Import modules under test
try:
    from geo_infer_health.models import (
        Location,
        HealthFacility,
        DiseaseReport,
        PopulationData,
        EnvironmentalData,
    )
    from geo_infer_health.core import (
        DiseaseHotspotAnalyzer,
        HealthcareAccessibilityAnalyzer,
        EnvironmentalHealthAnalyzer,
    )
    from geo_infer_health.utils import (
        haversine_distance,  # noqa: F401
        create_bounding_box,  # noqa: F401
    )  # noqa: F401
    from geo_infer_health.utils.config import HealthConfig  # noqa: F401
except ImportError as e:
    pytest.fail(f"Cannot import geo_infer_health modules: {e}")


@pytest.fixture
def sample_locations():
    """Create sample Location objects for testing."""
    return [
        Location(latitude=34.0522, longitude=-118.2437, crs="EPSG:4326"),  # Los Angeles
        Location(latitude=40.7128, longitude=-74.0060, crs="EPSG:4326"),  # New York
        Location(latitude=41.8781, longitude=-87.6298, crs="EPSG:4326"),  # Chicago
        Location(latitude=29.7604, longitude=-95.3698, crs="EPSG:4326"),  # Houston
        Location(latitude=33.4484, longitude=-112.0740, crs="EPSG:4326"),  # Phoenix
    ]


@pytest.fixture
def sample_health_facilities(sample_locations):
    """Create sample HealthFacility objects for testing."""
    facilities = []
    facility_types = ["Hospital", "Clinic", "Emergency", "Specialist"]
    services = [
        ["Emergency", "Surgery", "Cardiology"],
        ["General Checkup", "Vaccinations", "Pediatrics"],
        ["Emergency", "Trauma"],
        ["Cardiology", "Neurology", "Oncology"],
    ]

    for i, location in enumerate(sample_locations[:4]):
        facility = HealthFacility(
            facility_id=f"facility_{i + 1}",
            name=f"Test Facility {i + 1}",
            facility_type=facility_types[i % len(facility_types)],
            location=location,
            capacity=(i + 1) * 100,
            services_offered=services[i % len(services)],
        )
        facilities.append(facility)

    return facilities


@pytest.fixture
def sample_disease_reports(sample_locations):
    """Create sample DiseaseReport objects for testing."""
    reports = []
    diseases = ["COVID-19", "Influenza", "RSV", "Pertussis"]
    base_date = datetime(2024, 1, 1, tzinfo=UTC)

    for i, location in enumerate(sample_locations):
        for j in range(5):  # 5 reports per location
            report = DiseaseReport(
                report_id=f"report_{i * 5 + j + 1}",
                disease_code=diseases[(i + j) % len(diseases)],
                location=location,
                report_date=base_date,
                case_count=np.random.randint(1, 20),
                source="Test Source",
            )
            reports.append(report)

    return reports


@pytest.fixture
def sample_population_data():
    """Create sample PopulationData objects for testing."""
    population_data = []
    age_distributions = [
        {"0-18": 30000, "19-65": 50000, "65+": 20000},
        {"0-18": 25000, "19-65": 45000, "65+": 30000},
        {"0-18": 35000, "19-65": 55000, "65+": 15000},
    ]

    for i, age_distribution in enumerate(age_distributions):
        pop_data = PopulationData(
            area_id=f"area_{i + 1}",
            population_count=sum(age_distribution.values()),
            age_distribution=age_distribution,
        )
        population_data.append(pop_data)

    return population_data


@pytest.fixture
def sample_environmental_data(sample_locations):
    """Create sample EnvironmentalData objects for testing."""
    env_data = []
    parameters = ["PM2.5", "PM10", "NO2", "Temperature", "Humidity"]
    units = ["µg/m³", "µg/m³", "ppb", "°C", "%"]
    base_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

    for i, location in enumerate(sample_locations):
        for j, (param, unit) in enumerate(zip(parameters, units)):
            data_point = EnvironmentalData(
                data_id=f"env_{i * len(parameters) + j + 1}",
                parameter_name=param,
                value=np.random.uniform(10, 50),
                unit=unit,
                location=location,
                timestamp=base_timestamp,
            )
            env_data.append(data_point)

    return env_data


@pytest.fixture
def disease_analyzer(sample_disease_reports, sample_population_data):
    """Create a DiseaseHotspotAnalyzer instance for testing."""
    return DiseaseHotspotAnalyzer(
        reports=sample_disease_reports, population_data=sample_population_data
    )


@pytest.fixture
def healthcare_analyzer(sample_health_facilities, sample_population_data):
    """Create a HealthcareAccessibilityAnalyzer instance for testing."""
    return HealthcareAccessibilityAnalyzer(
        facilities=sample_health_facilities, population_data=sample_population_data
    )


@pytest.fixture
def environmental_analyzer(sample_environmental_data):
    """Create an EnvironmentalHealthAnalyzer instance for testing."""
    return EnvironmentalHealthAnalyzer(environmental_readings=sample_environmental_data)


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_data = {
        "module": {
            "name": "GEO-INFER-HEALTH-Test",
            "version": "1.0.0",
            "description": "Test configuration",
        },
        "api": {"host": "127.0.0.1", "port": 8001, "workers": 1},
        "database": {"type": "memory"},
        "analysis": {
            "disease_surveillance": {
                "default_scan_radius_km": 1.0,
                "hotspot_threshold_cases": 5,
            }
        },
        "development": {"debug_mode": True},
    }

    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_geodataframe(sample_locations):
    """Create a sample GeoDataFrame for testing."""
    # Create sample points
    geometry = [Point(loc.longitude, loc.latitude) for loc in sample_locations]

    # Create sample data
    data = {
        "id": [f"point_{i}" for i in range(len(sample_locations))],
        "name": [f"Location {i}" for i in range(len(sample_locations))],
        "value": np.random.uniform(0, 100, len(sample_locations)),
    }

    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
    return gdf


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Create sample CSV file
    csv_data = pd.DataFrame(
        {
            "id": range(10),
            "latitude": np.random.uniform(30, 40, 10),
            "longitude": np.random.uniform(-120, -110, 10),
            "value": np.random.uniform(0, 100, 10),
        }
    )
    csv_data.to_csv(test_dir / "sample_data.csv", index=False)

    # Create sample GeoJSON file
    sample_gdf = gpd.GeoDataFrame(
        {
            "id": range(5),
            "name": [f"Feature {i}" for i in range(5)],
            "value": np.random.uniform(0, 100, 5),
        },
        geometry=[
            Point(np.random.uniform(-120, -110), np.random.uniform(30, 40))
            for _ in range(5)
        ],
        crs="EPSG:4326",
    )
    sample_gdf.to_file(test_dir / "sample_data.geojson", driver="GeoJSON")

    return test_dir


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and configurations."""
    # Set test environment variables
    import os

    original_env = os.environ.copy()

    os.environ["GEO_INFER_HEALTH_TESTING"] = "true"
    os.environ["GEO_INFER_HEALTH_DEBUG"] = "true"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing."""

    class MockAPIClient:
        def __init__(self, base_url="http://testserver"):
            self.base_url = base_url
            self.requests = []

        def get(self, endpoint, **kwargs):
            self.requests.append(("GET", endpoint, kwargs))
            return MockResponse({"status": "success"})

        def post(self, endpoint, **kwargs):
            self.requests.append(("POST", endpoint, kwargs))
            return MockResponse({"status": "created"})

        def put(self, endpoint, **kwargs):
            self.requests.append(("PUT", endpoint, kwargs))
            return MockResponse({"status": "updated"})

        def delete(self, endpoint, **kwargs):
            self.requests.append(("DELETE", endpoint, kwargs))
            return MockResponse({"status": "deleted"})

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    return MockAPIClient()


# Custom pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "spatial: Tests involving spatial operations")
    config.addinivalue_line("markers", "api: API endpoint tests")


# Test utilities
def assert_geospatial_objects_equal(obj1, obj2, tolerance=1e-6):
    """Assert that two geospatial objects are approximately equal."""
    if hasattr(obj1, "latitude") and hasattr(obj2, "latitude"):
        assert abs(obj1.latitude - obj2.latitude) < tolerance
        assert abs(obj1.longitude - obj2.longitude) < tolerance
    elif hasattr(obj1, "__iter__") and hasattr(obj2, "__iter__"):
        for a, b in zip(obj1, obj2):
            assert_geospatial_objects_equal(a, b, tolerance)
    else:
        assert obj1 == obj2


def create_test_grid(
    center_lat=34.0522, center_lon=-118.2437, size_km=10, resolution=100
):
    """
    Create a test grid of points around a center location.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        size_km: Size of grid in km
        resolution: Number of points per side

    Returns:
        List of Location objects forming a grid
    """
    # Convert km to approximate degrees (rough approximation)
    km_to_deg = 0.009  # Approximate conversion

    half_size_deg = (size_km * km_to_deg) / 2
    step = (size_km * km_to_deg) / resolution

    locations = []
    for i in range(resolution):
        for j in range(resolution):
            lat = center_lat - half_size_deg + i * step
            lon = center_lon - half_size_deg + j * step
            locations.append(Location(latitude=lat, longitude=lon))

    return locations
