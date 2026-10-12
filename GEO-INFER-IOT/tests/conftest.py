"""
Pytest fixtures for GEO-INFER-IOT tests.

Provides sensor readings, IoT configurations, sensor network
GeoDataFrames, and standard spatial fixtures.
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
def sensor_readings() -> list[dict[str, Any]]:
    """List of IoT sensor reading dicts for ingestion and processing tests.

    Contains 15 readings from 3 devices with device_id, lat, lng,
    timestamp (ISO 8601 UTC), value (temperature in Celsius), and
    battery_level fields.
    """
    rng = np.random.default_rng(seed=42)
    readings = []
    devices = ["DEV_001", "DEV_002", "DEV_003"]
    base_lats = [47.60, 47.61, 47.62]
    base_lngs = [-122.33, -122.34, -122.35]

    for i in range(15):
        dev_idx = i % 3
        hour = i
        readings.append(
            {
                "device_id": devices[dev_idx],
                "lat": base_lats[dev_idx] + rng.uniform(-0.001, 0.001),
                "lng": base_lngs[dev_idx] + rng.uniform(-0.001, 0.001),
                "timestamp": f"2024-06-15T{hour:02d}:00:00Z",
                "value": float(18.0 + rng.normal(0, 2)),
                "unit": "celsius",
                "battery_level": float(rng.uniform(0.3, 1.0)),
            }
        )

    return readings


@pytest.fixture
def iot_config() -> dict[str, Any]:
    """Configuration dict for IoT sensor network management.

    Specifies data ingestion, anomaly detection thresholds,
    and spatial aggregation parameters.
    """
    return {
        "ingestion": {
            "batch_size": 100,
            "max_latency_seconds": 30,
            "deduplication": True,
        },
        "anomaly_detection": {
            "method": "z_score",
            "threshold": 3.0,
            "window_size": 24,
        },
        "spatial_aggregation": {
            "h3_resolution": 9,
            "temporal_window": "1H",
            "aggregation_method": "mean",
        },
        "quality_control": {
            "min_battery_level": 0.1,
            "max_value": 60.0,
            "min_value": -40.0,
        },
    }


@pytest.fixture
def sensor_network_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame representing an IoT sensor network deployment.

    Contains 8 sensor locations with device metadata including type,
    installation date, and communication protocol.
    """
    rng = np.random.default_rng(seed=42)
    n = 8
    lats = 47.58 + rng.uniform(0, 0.06, n)
    lngs = -122.36 + rng.uniform(0, 0.06, n)

    return gpd.GeoDataFrame(
        {
            "device_id": [f"DEV_{i:03d}" for i in range(n)],
            "sensor_type": ["temperature"] * 3 + ["humidity"] * 2 + ["air_quality"] * 3,
            "protocol": ["lorawan"] * 4 + ["nbiot"] * 4,
            "install_date": ["2024-01-15"] * 4 + ["2024-03-01"] * 4,
            "status": ["active"] * 7 + ["maintenance"],
        },
        geometry=[Point(lng, lat) for lat, lng in zip(lats, lngs)],
        crs="EPSG:4326",
    )


@pytest.fixture(autouse=True)
def _close_stray_event_loops():
    """Close any asyncio event loops a test leaves running.

    Some IOT ingestion tests construct real event loops; if a test exits
    before closing its loop (or a library task still holds it), pytest's
    unraisable-exception collector turns the eventual ResourceWarning into
    a hard failure.  This teardown cancels pending tasks, closes every open
    loop, and clears the current-loop reference so nothing leaks between
    tests.
    """
    import asyncio

    yield

    import warnings

    with warnings.catch_warnings():
        # In Python 3.12 policy.get_event_loop() warns when no current loop
        # is set; that is exactly the stray-loop condition this teardown
        # cleans up, so the deprecation noise is expected and suppressed.
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None
    if loop is not None and not loop.is_closed():
        try:
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError:
            pass
        loop.close()
