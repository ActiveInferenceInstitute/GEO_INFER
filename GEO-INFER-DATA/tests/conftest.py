"""
Pytest fixtures for GEO-INFER-DATA tests.

Provides sample CSV files, GeoJSON files, data source configurations,
and standard spatial fixtures.
"""

import asyncio
import json
from collections.abc import Iterator

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from typing import Any


@pytest.fixture(autouse=True)
def owned_event_loop() -> Iterator[None]:
    """Give each test an explicitly closed current-thread event loop.

    pytest-asyncio snapshots the current loop around async tests through
    ``asyncio.get_event_loop()``, which implicitly creates (and never closes)
    a loop when none is set; under ``-W error`` that loop's ResourceWarning
    would fail whichever later test triggers garbage collection.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        asyncio.set_event_loop(None)
        loop.close()


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
def sample_csv_path(tmp_path: Path) -> Path:
    """Path to a temporary CSV file with sample tabular data.

    Contains 20 rows of location data with id, lat, lng, value,
    and category columns, suitable for CSV ingestion tests.
    """
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        {
            "id": range(20),
            "lat": 47.0 + rng.uniform(0, 1, 20),
            "lng": -122.0 - rng.uniform(0, 1, 20),
            "value": rng.uniform(0, 100, 20),
            "category": [f"cat_{i % 4}" for i in range(20)],
        }
    )
    csv_path = tmp_path / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_geojson_path(tmp_path: Path) -> Path:
    """Path to a temporary GeoJSON file with sample point features.

    Contains 5 point features with id and value properties in
    EPSG:4326, suitable for GeoJSON loading and validation tests.
    """
    features = []
    for i in range(5):
        features.append(
            {
                "type": "Feature",
                "properties": {"id": i, "value": float(i * 10)},
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.33 + i * 0.01, 47.61 + i * 0.01],
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    geojson_path = tmp_path / "sample_data.geojson"
    geojson_path.write_text(json.dumps(geojson))
    return geojson_path


@pytest.fixture
def data_source_config() -> dict[str, Any]:
    """Configuration dict for a data source connection.

    Specifies connection parameters for a local file-based data source
    with format, CRS, and column mapping metadata.
    """
    return {
        "source_type": "file",
        "format": "geojson",
        "crs": "EPSG:4326",
        "encoding": "utf-8",
        "column_mapping": {
            "latitude": "lat",
            "longitude": "lng",
            "identifier": "id",
        },
        "cache_enabled": True,
        "cache_ttl_seconds": 3600,
    }
