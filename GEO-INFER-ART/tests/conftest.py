"""
Pytest fixtures for GEO-INFER-ART tests.

Provides sample image arrays, spatial art configurations,
color palettes, and standard spatial fixtures.
"""

import importlib.util
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import Point

matplotlib.use("Agg")  # Non-interactive backend for CI/headless testing


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
def sample_image_array() -> np.ndarray:
    """100x100x3 uint8 RGB image array for art generation tests.

    Contains a gradient pattern from top-left (dark) to bottom-right
    (bright) with distinct red, green, and blue channel ramps to
    verify color manipulation operations.
    """
    rng = np.random.default_rng(seed=42)
    height, width = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Red channel: horizontal gradient
    image[:, :, 0] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    # Green channel: vertical gradient
    image[:, :, 1] = np.tile(
        np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width)
    )
    # Blue channel: random noise
    image[:, :, 2] = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return image


@pytest.fixture
def spatial_art_config() -> dict[str, Any]:
    """Configuration dict for spatial art generation.

    Specifies output resolution, projection, style parameters, and
    rendering options for geospatial art compositions.
    """
    return {
        "width": 1024,
        "height": 1024,
        "dpi": 150,
        "projection": "EPSG:4326",
        "style": "terrain",
        "background_color": "#1a1a2e",
        "line_width": 1.5,
        "opacity": 0.8,
        "antialiasing": True,
        "output_format": "png",
    }


@pytest.fixture
def color_palette() -> list[str]:
    """Hex color palette for generative art tests.

    A curated 8-color palette suitable for geospatial data
    visualization with good perceptual contrast.
    """
    return [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#606c38",
        "#283618",
        "#dda15e",
    ]


@pytest.fixture
def sample_terrain_data() -> np.ndarray:
    """50x50 float32 elevation grid for terrain art rendering.

    Simulates a mountain ridgeline with a Gaussian peak centered
    at (25, 25) and elevation values in meters (0-3000m range).
    """
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    xx, yy = np.meshgrid(x, y)
    elevation = 3000.0 * np.exp(-(xx**2 + yy**2) / 1.5)
    return elevation.astype(np.float32)


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect TensorFlow-gated style-transfer tests when TF is not installed.

    The root test contract forbids runtime skips, so the heavy optional
    dependency is handled at collection time instead: without tensorflow the
    style-transfer tests are deselected (they run wherever the declared test
    dependency group is installed).
    """
    if importlib.util.find_spec("tensorflow") is None:
        items[:] = [
            item for item in items if "test_style_transfer" not in item.path.name
        ]
