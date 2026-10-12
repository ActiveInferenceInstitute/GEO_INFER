"""
Pytest fixtures for GEO-INFER-FOREST tests.

Provides forest stand GeoDataFrames, biomass allometric parameters,
forest management configurations, and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
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
def forest_stand_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with polygon forest stands and species/height attributes.

    Contains 6 rectangular forest stand polygons in the Cascade Range with
    dominant species, mean height, diameter at breast height (DBH), stem
    density, and age class attributes.
    """
    rng = np.random.default_rng(seed=42)
    stands = []
    base_lat, base_lng = 47.0, -121.5

    for i in range(6):
        row = i // 3
        col = i % 3
        lat0 = base_lat + row * 0.05
        lng0 = base_lng + col * 0.05
        stands.append(
            Polygon(
                [
                    (lng0, lat0),
                    (lng0 + 0.04, lat0),
                    (lng0 + 0.04, lat0 + 0.04),
                    (lng0, lat0 + 0.04),
                ]
            )
        )

    species = [
        "Pseudotsuga menziesii",
        "Tsuga heterophylla",
        "Thuja plicata",
        "Picea sitchensis",
        "Abies procera",
        "Pseudotsuga menziesii",
    ]

    return gpd.GeoDataFrame(
        {
            "stand_id": [f"STAND_{i:03d}" for i in range(6)],
            "dominant_species": species,
            "mean_height_m": rng.uniform(15, 60, 6).round(1),
            "mean_dbh_cm": rng.uniform(20, 80, 6).round(1),
            "stems_per_ha": rng.integers(200, 1200, 6),
            "age_years": rng.integers(20, 200, 6),
        },
        geometry=stands,
        crs="EPSG:4326",
    )


@pytest.fixture
def biomass_allometric_params() -> dict[str, dict[str, float]]:
    """Allometric equation parameters for biomass estimation by species.

    Each species has coefficients a and b for the equation:
    biomass_kg = a * (DBH_cm ** b)
    Based on published allometric relationships for PNW conifers.
    """
    return {
        "Pseudotsuga menziesii": {"a": 0.0436, "b": 2.8837},
        "Tsuga heterophylla": {"a": 0.0517, "b": 2.7574},
        "Thuja plicata": {"a": 0.0353, "b": 2.9046},
        "Picea sitchensis": {"a": 0.0467, "b": 2.8121},
        "Abies procera": {"a": 0.0498, "b": 2.7935},
    }


@pytest.fixture
def forest_config() -> dict[str, Any]:
    """Configuration dict for forest analysis operations.

    Specifies canopy analysis parameters, fire risk thresholds,
    and carbon accounting settings.
    """
    return {
        "canopy_analysis": {
            "min_height_m": 2.0,
            "resolution_m": 1.0,
            "gap_threshold_m2": 25.0,
        },
        "fire_risk": {
            "fuel_moisture_threshold": 0.12,
            "wind_speed_critical_kmh": 40,
            "slope_critical_degrees": 30,
        },
        "carbon": {
            "wood_density_kg_m3": 450,
            "carbon_fraction": 0.5,
            "root_to_shoot_ratio": 0.26,
        },
    }
