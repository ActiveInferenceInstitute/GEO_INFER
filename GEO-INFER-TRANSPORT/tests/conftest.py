"""
Pytest fixtures for GEO-INFER-TRANSPORT tests.

Provides road network GeoDataFrames, origin-destination matrices,
transport configurations, and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
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
def road_network_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with 10 road segment line geometries for network tests.

    Contains road segments in a grid pattern in the Seattle area with
    road type, speed limit, number of lanes, and length attributes.
    """
    rng = np.random.default_rng(seed=42)
    segments = []
    base_lat, base_lng = 47.60, -122.33

    # 5 east-west segments
    for i in range(5):
        lat = base_lat + i * 0.01
        segments.append(
            LineString(
                [
                    (base_lng, lat),
                    (base_lng + 0.05, lat),
                ]
            )
        )

    # 5 north-south segments
    for i in range(5):
        lng = base_lng + i * 0.0125
        segments.append(
            LineString(
                [
                    (lng, base_lat),
                    (lng, base_lat + 0.04),
                ]
            )
        )

    road_types = ["arterial", "arterial", "collector", "local", "local"] + [
        "arterial",
        "collector",
        "collector",
        "local",
        "local",
    ]

    return gpd.GeoDataFrame(
        {
            "road_id": [f"RD_{i:03d}" for i in range(10)],
            "road_type": road_types,
            "speed_limit_kmh": [50, 50, 40, 30, 30, 50, 40, 40, 30, 30],
            "lanes": [4, 4, 2, 2, 2, 4, 2, 2, 2, 2],
            "length_m": rng.uniform(500, 3000, 10).round(0).astype(int),
            "oneway": [False] * 8 + [True, True],
        },
        geometry=segments,
        crs="EPSG:4326",
    )


@pytest.fixture
def od_matrix() -> np.ndarray:
    """5x5 origin-destination trip matrix for traffic assignment tests.

    Rows are origins, columns are destinations. Values represent daily
    trip counts between 5 traffic analysis zones. Diagonal is zero
    (no intra-zone trips).
    """
    return np.array(
        [
            [0, 120, 80, 45, 200],
            [110, 0, 150, 60, 90],
            [75, 140, 0, 180, 55],
            [50, 65, 170, 0, 130],
            [190, 85, 50, 120, 0],
        ]
    )


@pytest.fixture
def transport_config() -> dict[str, Any]:
    """Configuration dict for transport network analysis.

    Specifies routing algorithm, impedance function, mode parameters,
    and analysis settings.
    """
    return {
        "routing": {
            "algorithm": "dijkstra",
            "impedance": "travel_time",
            "turn_penalties": True,
        },
        "modes": {
            "car": {"speed_factor": 1.0, "capacity_pcu": 1.0},
            "bus": {"speed_factor": 0.7, "capacity_pcu": 2.5},
            "bicycle": {"speed_factor": 0.3, "capacity_pcu": 0.2},
        },
        "assignment": {
            "method": "user_equilibrium",
            "convergence_threshold": 0.001,
            "max_iterations": 100,
        },
    }
