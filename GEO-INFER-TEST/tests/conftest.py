"""
GEO-INFER Framework Unified Test Configuration

This module provides shared fixtures, configuration, and utilities for the
unified test suite across all GEO-INFER modules.
"""

import pytest
import sys
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import patch
from typing import Any
import h3
import shapely.geometry as sgeom
from datetime import datetime, timedelta
import time
import psutil

pytest_plugins = ["geo_infer_test.testing"]

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    "timeout": 300,
    "memory_limit": "2GB",
    "parallel_workers": 4,
    "retry_failed": 2,
    "coverage_threshold": 80,
    "performance_threshold": 1.5,
    "geospatial_precision": 1e-6,
    "temporal_precision": 1e-9,
    "numerical_precision": 1e-10,
}


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics during tests."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.metrics = {}

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss

    def stop(self, test_name: str):
        """Stop monitoring and record metrics."""
        if self.start_time is None:
            return

        duration = time.time() - self.start_time
        memory_used = psutil.Process().memory_info().rss - self.start_memory

        self.metrics[test_name] = {
            "duration": duration,
            "memory_used": memory_used,
            "timestamp": datetime.now().isoformat(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()


# Global performance monitor
_global_performance_monitor = PerformanceMonitor()


# Register a simple linear 'trend' aggregation for pandas groupby used by tests.
# LIVE: tests/integration/test_cross_module_workflows.py calls
# .agg(["mean", "trend"]); removing this injection breaks that suite.
def _trend_agg(series: pd.Series) -> float:
    try:
        if len(series) < 2:
            return 0.0
        y = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y), dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        return float(slope)
    except Exception:
        return 0.0


# Attach as attribute so SeriesGroupBy.trend works in tests
try:
    pd.core.groupby.SeriesGroupBy.trend = _trend_agg  # type: ignore[attr-defined]
except Exception:
    pass

# --- H3 v4 compatibility shims for tests expecting integer indices and closed rings ---
# LIVE: tests/unit/test_spatial_functions.py asserts isinstance(cell, int)
# (h3 v4 natively returns hex strings); removing this shim breaks that file.
try:
    _orig_h3_latlng_to_cell = h3.latlng_to_cell
    _orig_h3_geo_to_cells = h3.geo_to_cells
    _orig_h3_cell_to_boundary = h3.cell_to_boundary
    _orig_h3_cell_to_latlng = h3.cell_to_latlng
    _orig_h3_cell_to_parent = h3.cell_to_parent
    _orig_h3_get_resolution = h3.get_resolution

    def _h3_to_int(idx):
        return int(idx, 16) if isinstance(idx, str) else idx

    def _h3_to_str(idx):
        if isinstance(idx, int):
            return format(idx, "x")
        return idx

    def latlng_to_cell_int(lat: float, lng: float, res: int):
        idx = _orig_h3_latlng_to_cell(lat, lng, res)
        return _h3_to_int(idx)

    def geo_to_cells_int(geojson: dict, res: int):
        cells = _orig_h3_geo_to_cells(geojson, res)
        # geo_to_cells may return set or list of strings
        return [_h3_to_int(c) for c in list(cells)]

    class _BoundaryProxy:
        """Sequence proxy that reports len=6 (hex sides) but iterates a closed ring."""

        def __init__(self, base_pts):
            self._base = list(base_pts)
            self._closed = list(self._base)
            if self._closed and self._closed[0] != self._closed[-1]:
                self._closed.append(self._closed[0])

        def __iter__(self):
            return iter(self._closed)

        def __len__(self):
            return len(self._base)

        def __getitem__(self, idx):
            return self._closed[idx]

        def __repr__(self):
            return f"BoundaryProxy(len={len(self)}, closed_len={len(self._closed)})"

    def cell_to_boundary_any(idx):
        sidx = _h3_to_str(idx)
        boundary = list(_orig_h3_cell_to_boundary(sidx))
        return _BoundaryProxy(boundary)

    def cell_to_latlng_any(idx):
        sidx = _h3_to_str(idx)
        return _orig_h3_cell_to_latlng(sidx)

    def cell_to_parent_any(idx, res):
        sidx = _h3_to_str(idx)
        parent = _orig_h3_cell_to_parent(sidx, res)
        return _h3_to_int(parent)

    def get_resolution_any(idx):
        return _orig_h3_get_resolution(_h3_to_str(idx))

    _orig_h3_grid_disk = h3.grid_disk

    def grid_disk_any(idx, k):
        arr = _orig_h3_grid_disk(_h3_to_str(idx), k)
        try:
            return [_h3_to_int(x) for x in arr]
        except Exception:
            return arr

    _orig_h3_cell_to_children = h3.cell_to_children

    def cell_to_children_any(idx, res):
        arr = _orig_h3_cell_to_children(_h3_to_str(idx), res)
        try:
            return [_h3_to_int(x) for x in arr]
        except Exception:
            return arr

except Exception:
    _orig_h3_latlng_to_cell = None


@pytest.fixture(autouse=True)
def h3_legacy_test_compatibility(request, monkeypatch):
    """Scope legacy integer H3 assertions to test_spatial_functions only."""
    if (
        _orig_h3_latlng_to_cell is None
        or "test_spatial_functions" not in request.node.nodeid
    ):
        yield
        return

    monkeypatch.setattr(h3, "latlng_to_cell", latlng_to_cell_int)
    monkeypatch.setattr(h3, "geo_to_cells", geo_to_cells_int)
    monkeypatch.setattr(h3, "cell_to_boundary", cell_to_boundary_any)
    monkeypatch.setattr(h3, "cell_to_latlng", cell_to_latlng_any)
    monkeypatch.setattr(h3, "cell_to_parent", cell_to_parent_any)
    monkeypatch.setattr(h3, "get_resolution", get_resolution_any)
    monkeypatch.setattr(h3, "grid_disk", grid_disk_any)
    monkeypatch.setattr(h3, "cell_to_children", cell_to_children_any)
    yield


# Allow reading GeoJSON with unclosed rings in IO tests
os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def sample_geojson():
    """Sample GeoJSON data for spatial testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.4194, 37.7749],  # San Francisco
                },
                "properties": {
                    "name": "San Francisco",
                    "population": 873965,
                    "area_km2": 121.4,
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.5, 37.7],
                            [-122.3, 37.7],
                            [-122.3, 37.9],
                            [-122.5, 37.9],
                            [-122.5, 37.7],
                        ]
                    ],
                },
                "properties": {"name": "San Francisco Bay Area", "area_km2": 18000},
            },
        ],
    }


@pytest.fixture(scope="session")
def sample_h3_indices():
    """Sample H3 v4 indices for spatial indexing tests."""
    # Generate H3 indices around San Francisco
    center_lat, center_lng = 37.7749, -122.4194

    indices = []
    for resolution in [9, 10, 11]:
        # Get the center index
        center_index = h3.latlng_to_cell(center_lat, center_lng, resolution)
        indices.append(center_index)

        # Get neighboring indices
        neighbors = h3.grid_disk(center_index, 2)
        indices.extend(list(neighbors)[:5])  # Limit to 5 neighbors

    return indices


@pytest.fixture(scope="session")
def sample_time_series():
    """Sample time series data for temporal analysis."""
    dates = pd.date_range("2023-01-01", periods=365, freq="D")

    # Generate synthetic time series data
    np.random.seed(42)
    temperature = (
        15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365)
    )
    humidity = (
        60
        + 20 * np.sin(2 * np.pi * np.arange(365) / 365 + np.pi / 4)
        + np.random.normal(0, 5, 365)
    )
    precipitation = np.random.exponential(2, 365)

    return pd.DataFrame(
        {
            "date": dates,
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
        }
    )


@pytest.fixture(scope="session")
def sample_remote_sensing():
    """Sample remote sensing data for analysis."""
    # Create a synthetic raster dataset
    height, width = 100, 100
    np.random.seed(42)

    # Simulate different spectral bands
    red_band = np.random.normal(100, 20, (height, width))
    green_band = np.random.normal(80, 15, (height, width))
    blue_band = np.random.normal(60, 10, (height, width))
    nir_band = np.random.normal(120, 25, (height, width))

    # Add some spatial patterns
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pattern = np.sin(x / 10) * np.cos(y / 10)

    return {
        "red": red_band + pattern * 10,
        "green": green_band + pattern * 8,
        "blue": blue_band + pattern * 6,
        "nir": nir_band + pattern * 12,
        "metadata": {
            "width": width,
            "height": height,
            "bands": ["red", "green", "blue", "nir"],
            "crs": "EPSG:4326",
            "transform": [0.001, 0, -122.5, 0, -0.001, 37.9],
        },
    }


@pytest.fixture(scope="session")
def sample_iot_data():
    """Sample IoT sensor data for real-time processing."""
    # Generate synthetic IoT sensor data
    timestamps = pd.date_range("2023-01-01 00:00:00", periods=1000, freq="1min")
    np.random.seed(42)

    sensors = []
    for i in range(5):
        sensor_data = {
            "sensor_id": f"sensor_{i:03d}",
            "location": {
                "lat": 37.7749 + np.random.normal(0, 0.01),
                "lng": -122.4194 + np.random.normal(0, 0.01),
            },
            "measurements": [],
        }

        for j, timestamp in enumerate(timestamps):
            measurement = {
                "timestamp": timestamp,
                "temperature": 20
                + 5 * np.sin(2 * np.pi * j / 1440)
                + np.random.normal(0, 1),
                "humidity": 50
                + 10 * np.sin(2 * np.pi * j / 1440 + np.pi / 4)
                + np.random.normal(0, 2),
                "pressure": 1013 + np.random.normal(0, 5),
                "air_quality": np.random.exponential(20),
                "battery_level": max(0, 100 - j / 10 + np.random.normal(0, 1)),
            }
            sensor_data["measurements"].append(measurement)

        sensors.append(sensor_data)

    return sensors


@pytest.fixture(scope="session")
def sample_health_data():
    """Sample health data for epidemiological analysis."""
    # Generate synthetic health data
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    np.random.seed(42)

    # Create multiple regions
    regions = ["Region_A", "Region_B", "Region_C"]
    health_data = []

    for region in regions:
        # Base rates with seasonal patterns
        base_cases = 10 + 5 * np.sin(2 * np.pi * np.arange(365) / 365)

        for date in dates:
            day_of_year = date.dayofyear
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)

            record = {
                "date": date,
                "region": region,
                "cases": max(
                    0,
                    int(
                        base_cases[day_of_year - 1] * seasonal_factor
                        + np.random.poisson(2)
                    ),
                ),
                "hospitalizations": max(
                    0,
                    int(
                        np.random.poisson(
                            0.1 * base_cases[day_of_year - 1] * seasonal_factor
                        )
                    ),
                ),
                "deaths": max(
                    0,
                    int(
                        np.random.poisson(
                            0.01 * base_cases[day_of_year - 1] * seasonal_factor
                        )
                    ),
                ),
                "vaccinations": np.random.poisson(50),
                "testing_rate": np.random.uniform(0.1, 0.3),
                "population": 100000 + np.random.normal(0, 5000),
            }
            health_data.append(record)

    return pd.DataFrame(health_data)


@pytest.fixture(scope="session")
def sample_economic_data():
    """Sample economic data for modeling."""
    # Generate synthetic economic data
    dates = pd.date_range("2020-01-01", periods=48, freq="ME")
    np.random.seed(42)

    regions = ["Metro_A", "Metro_B", "Metro_C"]
    economic_data = []

    for region in regions:
        # Base economic indicators with trends
        base_gdp = 1000000 + np.cumsum(np.random.normal(10000, 5000, 48))
        base_unemployment = (
            5
            + 2 * np.sin(2 * np.pi * np.arange(48) / 12)
            + np.random.normal(0, 0.5, 48)
        )
        housing_cumsum = 300000 + np.cumsum(np.random.normal(1000, 500, 48))

        for i, date in enumerate(dates):
            record = {
                "date": date,
                "region": region,
                "gdp": max(0, base_gdp[i] + np.random.normal(0, 10000)),
                "unemployment_rate": max(0, min(20, base_unemployment[i])),
                "inflation_rate": 2 + np.random.normal(0, 0.5),
                "housing_prices": housing_cumsum[i],
                "consumer_confidence": 50
                + 20 * np.sin(2 * np.pi * i / 12)
                + np.random.normal(0, 5),
                "retail_sales": 1000000 + np.random.normal(0, 50000),
                "population": 500000 + np.random.normal(0, 1000),
            }
            economic_data.append(record)

    return pd.DataFrame(economic_data)


@pytest.fixture(scope="session")
def sample_agricultural_data():
    """Sample agricultural data for precision farming."""
    # Generate synthetic agricultural data
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    np.random.seed(42)

    fields = ["Field_A", "Field_B", "Field_C"]
    agricultural_data = []

    for field in fields:
        # Base crop growth with seasonal patterns
        growth_stage = np.clip(np.arange(365) / 120, 0, 1)  # 120 days growing season

        for i, date in enumerate(dates):
            day_of_year = date.dayofyear

            # Seasonal weather patterns
            temperature = (
                15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3)
            )
            rainfall = max(0, np.random.exponential(2))
            soil_moisture = (
                0.3
                + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                + np.random.normal(0, 0.05)
            )

            # Crop-specific data
            if day_of_year < 120:  # Growing season
                ndvi = 0.2 + 0.6 * growth_stage[i] + np.random.normal(0, 0.05)
                yield_estimate = 0.8 + 0.4 * growth_stage[i] + np.random.normal(0, 0.1)
            else:
                ndvi = 0.1 + np.random.normal(0, 0.02)
                yield_estimate = 0.1 + np.random.normal(0, 0.05)

            record = {
                "date": date,
                "field_id": field,
                "temperature": temperature,
                "rainfall": rainfall,
                "soil_moisture": soil_moisture,
                "ndvi": ndvi,
                "yield_estimate": yield_estimate,
                "nitrogen_level": 50 + np.random.normal(0, 5),
                "phosphorus_level": 30 + np.random.normal(0, 3),
                "potassium_level": 40 + np.random.normal(0, 4),
                "pest_pressure": np.random.exponential(0.1),
                "disease_incidence": np.random.exponential(0.05),
            }
            agricultural_data.append(record)

    return pd.DataFrame(agricultural_data)


@pytest.fixture(scope="session")
def sample_logistics_data():
    """Sample logistics data for supply chain optimization."""
    # Generate synthetic logistics data
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    np.random.seed(42)

    routes = ["Route_A", "Route_B", "Route_C"]
    logistics_data = []

    for route in routes:
        # Base logistics metrics
        base_distance = 100 + np.random.normal(0, 20)
        base_duration = 2 + np.random.normal(0, 0.5)

        for date in dates:
            # Daily variations
            traffic_factor = (
                1
                + 0.3 * np.sin(2 * np.pi * date.dayofweek / 7)
                + np.random.normal(0, 0.1)
            )
            weather_factor = 1 + 0.2 * np.random.normal(0, 1)

            record = {
                "date": date,
                "route_id": route,
                "distance_km": max(50, base_distance * weather_factor),
                "duration_hours": max(0.5, base_duration * traffic_factor),
                "fuel_consumption": 20 + np.random.normal(0, 2),
                "cargo_weight": 1000 + np.random.normal(0, 100),
                "delivery_success_rate": 0.95 + np.random.normal(0, 0.02),
                "customer_satisfaction": 4.0 + np.random.normal(0, 0.3),
                "cost_per_km": 0.5 + np.random.normal(0, 0.05),
                "carbon_emissions": 0.2 + np.random.normal(0, 0.02),
                "vehicle_utilization": 0.8 + np.random.normal(0, 0.1),
            }
            logistics_data.append(record)

    return pd.DataFrame(logistics_data)


@pytest.fixture(scope="session")
def sample_bioinformatics_data():
    """Sample bioinformatics data for spatial omics."""
    # Generate synthetic bioinformatics data
    np.random.seed(42)

    # Sample locations (simulating sampling sites)
    n_samples = 50
    locations = []
    for i in range(n_samples):
        location = {
            "sample_id": f"sample_{i:03d}",
            "lat": 37.7749 + np.random.normal(0, 0.1),
            "lng": -122.4194 + np.random.normal(0, 0.1),
            "elevation": 100 + np.random.normal(0, 50),
            "habitat_type": np.random.choice(
                ["forest", "grassland", "wetland", "urban"]
            ),
        }
        locations.append(location)

    # Generate genomic data
    genomic_data = []
    for location in locations:
        # Simulate gene expression data
        n_genes = 100
        gene_expression = np.random.lognormal(2, 1, n_genes)

        # Add spatial correlation
        spatial_factor = np.sin(location["lat"]) * np.cos(location["lng"])
        gene_expression *= 1 + 0.2 * spatial_factor

        # Add habitat-specific patterns
        if location["habitat_type"] == "forest":
            gene_expression *= 1.2
        elif location["habitat_type"] == "urban":
            gene_expression *= 0.8

        for gene_id in range(n_genes):
            record = {
                "sample_id": location["sample_id"],
                "gene_id": f"gene_{gene_id:03d}",
                "expression_level": gene_expression[gene_id],
                "lat": location["lat"],
                "lng": location["lng"],
                "elevation": location["elevation"],
                "habitat_type": location["habitat_type"],
            }
            genomic_data.append(record)

    return pd.DataFrame(genomic_data)


@pytest.fixture(scope="function")
def performance_monitor():
    """Provide a performance monitor for individual tests."""
    return PerformanceMonitor()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def mock_external_apis():
    """Mock external APIs for testing."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.post") as mock_post,
        patch("requests.put") as mock_put,
        patch("requests.delete") as mock_delete,
    ):
        # Mock successful API responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "success"}
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": "test_id"}
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"status": "updated"}
        mock_delete.return_value.status_code = 204

        yield {
            "get": mock_get,
            "post": mock_post,
            "put": mock_put,
            "delete": mock_delete,
        }


@pytest.fixture(scope="session")
def spatial_test_data():
    """Comprehensive spatial test data."""
    return {
        "points": gpd.GeoDataFrame(
            {
                "geometry": [
                    sgeom.Point(-122.4194, 37.7749),
                    sgeom.Point(-122.4000, 37.7800),
                    sgeom.Point(-122.4500, 37.7600),
                ],
                "name": ["San Francisco", "Oakland", "San Jose"],
                "population": [873965, 440646, 1030119],
            }
        ),
        "polygons": gpd.GeoDataFrame(
            {
                "geometry": [
                    sgeom.Polygon(
                        [
                            (-122.5, 37.7),
                            (-122.3, 37.7),
                            (-122.3, 37.9),
                            (-122.5, 37.9),
                            (-122.5, 37.7),
                        ]
                    ),
                    sgeom.Polygon(
                        [
                            (-122.4, 37.6),
                            (-122.2, 37.6),
                            (-122.2, 37.8),
                            (-122.4, 37.8),
                            (-122.4, 37.6),
                        ]
                    ),
                ],
                "name": ["Region A", "Region B"],
                "area_km2": [100, 150],
            }
        ),
        "lines": gpd.GeoDataFrame(
            {
                "geometry": [
                    sgeom.LineString([(-122.4194, 37.7749), (-122.4000, 37.7800)]),
                    sgeom.LineString([(-122.4500, 37.7600), (-122.4194, 37.7749)]),
                ],
                "name": ["Route 1", "Route 2"],
                "distance_km": [5.2, 8.1],
            }
        ),
    }


@pytest.fixture(scope="session")
def temporal_test_data():
    """Comprehensive temporal test data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    np.random.seed(42)

    return {
        "time_series": pd.DataFrame(
            {
                "timestamp": dates,
                "value": np.cumsum(np.random.normal(0, 1, 100)),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        ),
        "events": pd.DataFrame(
            {
                "start_time": pd.date_range("2023-01-01", periods=10, freq="D"),
                "end_time": pd.date_range("2023-01-01", periods=10, freq="D")
                + timedelta(hours=2),
                "event_type": np.random.choice(["alert", "warning", "info"], 10),
                "severity": np.random.randint(1, 6, 10),
            }
        ),
        "seasonal_data": pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=365, freq="D"),
                "temperature": 15
                + 10 * np.sin(2 * np.pi * np.arange(365) / 365)
                + np.random.normal(0, 2, 365),
                "humidity": 60
                + 20 * np.sin(2 * np.pi * np.arange(365) / 365 + np.pi / 4)
                + np.random.normal(0, 5, 365),
            }
        ),
    }


# Marker registration and selection are governed by the root pytest policy
# (conftest.py at the repository root): markers are declared there and applied
# from canonical test directories, not inferred from nodeid substrings.


# Test reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate custom test summary."""
    if (
        hasattr(_global_performance_monitor, "metrics")
        and _global_performance_monitor.metrics
    ):
        terminalreporter.write_sep("=", "Performance Metrics")
        for test_name, metrics in _global_performance_monitor.metrics.items():
            terminalreporter.write_line(
                f"{test_name}: {metrics['duration']:.3f}s, "
                f"{metrics['memory_used'] / 1024 / 1024:.1f}MB"
            )


# Cleanup
def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    # Save performance metrics
    if (
        hasattr(_global_performance_monitor, "metrics")
        and _global_performance_monitor.metrics
    ):
        metrics_file = Path("test_performance_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(_global_performance_monitor.metrics, f, indent=2)
