"""
Pytest fixtures for GEO-INFER-EMERGENCY tests.

Provides hazard zone GeoDataFrames, shelter locations,
emergency response configurations, and standard spatial fixtures.

Geo-stack imports (numpy, geopandas, shapely) are performed lazily inside
the fixtures that need them, so pure-logic tests run without the heavy
optional geo dependencies.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    # Annotation-only: the real import happens lazily inside fixtures so
    # pure-logic tests run without the heavy geo dependencies.
    import geopandas as gpd


@pytest.fixture(scope="session")
def sample_coordinates() -> list[tuple[float, float]]:
    """Standard (lat, lng) coordinate pairs for spatial tests."""
    return [
        (47.6062, -122.3321),  # Seattle
        (45.5152, -122.6784),  # Portland
        (48.4284, -122.3321),  # Whidbey Island
        (47.2529, -120.8398),  # Wenatchee
        (46.9753, -120.5408),  # Ellensburg
    ]


@pytest.fixture(scope="function")
def sample_geodataframe() -> "gpd.GeoDataFrame":
    """Standard GeoDataFrame with EPSG:4326 for spatial tests."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"id": range(5), "value": np.random.uniform(0, 100, 5)},
        geometry=[Point(-122.33 + i * 0.01, 47.61 + i * 0.01) for i in range(5)],
        crs="EPSG:4326",
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def hazard_zone_gdf() -> "gpd.GeoDataFrame":
    """GeoDataFrame with 3 hazard risk zones for emergency planning.

    Contains polygon geometries representing flood, earthquake, and
    wildfire risk zones with severity levels and affected population
    estimates in the Pacific Northwest.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    zones = [
        Polygon(
            [
                (-122.4, 47.5),
                (-122.3, 47.5),
                (-122.3, 47.6),
                (-122.4, 47.6),
            ]
        ),
        Polygon(
            [
                (-122.3, 47.55),
                (-122.2, 47.55),
                (-122.2, 47.65),
                (-122.3, 47.65),
            ]
        ),
        Polygon(
            [
                (-122.35, 47.45),
                (-122.25, 47.45),
                (-122.25, 47.55),
                (-122.35, 47.55),
            ]
        ),
    ]

    return gpd.GeoDataFrame(
        {
            "zone_id": ["FLD_001", "EQ_001", "WF_001"],
            "hazard_type": ["flood", "earthquake", "wildfire"],
            "severity": ["high", "moderate", "high"],
            "affected_population": [12000, 45000, 3500],
            "probability_annual": [0.05, 0.02, 0.10],
        },
        geometry=zones,
        crs="EPSG:4326",
    )


@pytest.fixture
def shelter_locations_gdf() -> "gpd.GeoDataFrame":
    """GeoDataFrame with 5 emergency shelter point locations.

    Contains shelter positions with capacity, type (school, community
    center, stadium), and current supply levels for evacuation planning.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {
            "shelter_id": [f"SHL_{i:03d}" for i in range(5)],
            "name": [
                "Lincoln High School",
                "Community Center A",
                "Stadium Complex",
                "Church of the Valley",
                "Civic Auditorium",
            ],
            "type": ["school", "community_center", "stadium", "religious", "civic"],
            "capacity": [500, 200, 2000, 150, 800],
            "supply_days": [3, 2, 7, 1, 5],
        },
        geometry=[
            Point(-122.35, 47.55),
            Point(-122.28, 47.58),
            Point(-122.32, 47.50),
            Point(-122.25, 47.52),
            Point(-122.30, 47.60),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def emergency_config() -> dict[str, Any]:
    """Configuration dict for emergency response modeling.

    Specifies response time targets, resource allocation parameters,
    and evacuation routing constraints.
    """
    return {
        "response_time_target_minutes": 15,
        "evacuation_speed_kmh": 30,
        "max_shelter_distance_km": 10,
        "resource_types": ["medical", "food", "water", "shelter"],
        "priority_populations": ["elderly", "disabled", "children"],
        "communication_channels": ["siren", "sms", "radio", "app"],
        "staging_area_min_size_m2": 5000,
    }
