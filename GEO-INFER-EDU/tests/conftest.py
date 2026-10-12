"""
Pytest fixtures for GEO-INFER-EDU tests.

Provides school location GeoDataFrames, population density data,
education configurations, and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from typing import Any


@pytest.fixture(scope="session")
def sample_coordinates() -> list[tuple[float, float]]:
    """Standard (lat, lng) coordinate pairs for spatial tests."""
    return [
        (47.6062, -122.3321),
        (37.7749, -122.4194),
        (40.7128, -74.0060),
        (51.5074, -0.1278),
        (35.6762, 139.6503),
    ]


@pytest.fixture(scope="function")
def sample_geodataframe() -> gpd.GeoDataFrame:
    """Standard GeoDataFrame with EPSG:4326 for spatial tests."""
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
def school_locations_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with 10 school point locations for accessibility tests.

    Contains schools in the Seattle metro area with type (elementary,
    middle, high), enrollment count, and student-teacher ratio attributes.
    """
    rng = np.random.default_rng(seed=42)
    n = 10
    lats = 47.5 + rng.uniform(0, 0.3, n)
    lngs = -122.4 + rng.uniform(0, 0.3, n)
    school_types = ["elementary"] * 5 + ["middle"] * 3 + ["high"] * 2

    return gpd.GeoDataFrame(
        {
            "school_id": [f"SCH_{i:03d}" for i in range(n)],
            "name": [f"School {i}" for i in range(n)],
            "type": school_types,
            "enrollment": rng.integers(150, 1200, n),
            "student_teacher_ratio": rng.uniform(15, 28, n).round(1),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )


@pytest.fixture
def population_density_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with population data for school catchment analysis.

    Contains 20 census block centroids with population count,
    school-age children count, and density values.
    """
    rng = np.random.default_rng(seed=42)
    n = 20
    lats = 47.45 + rng.uniform(0, 0.4, n)
    lngs = -122.5 + rng.uniform(0, 0.4, n)

    population = rng.integers(500, 8000, n)
    school_age_fraction = rng.uniform(0.10, 0.25, n)

    return gpd.GeoDataFrame(
        {
            "block_id": [f"BLK_{i:04d}" for i in range(n)],
            "population": population,
            "school_age_children": (population * school_age_fraction).astype(int),
            "density_per_km2": rng.uniform(200, 5000, n),
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )


@pytest.fixture
def education_config() -> dict[str, Any]:
    """Configuration dict for education accessibility analysis.

    Specifies travel time thresholds, catchment parameters, and
    equity weighting for school accessibility modeling.
    """
    return {
        "max_travel_time_minutes": 30,
        "travel_mode": "walking",
        "catchment_method": "network",
        "h3_resolution": 9,
        "equity_weights": {
            "low_income": 1.5,
            "minority": 1.3,
            "disability": 1.4,
        },
        "minimum_enrollment": 100,
        "overcrowding_threshold": 1.1,
    }
