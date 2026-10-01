"""Shared pytest fixtures for GEO-INFER-PLACE tests."""

from pathlib import Path
from typing import Any

import h3
import pytest


@pytest.fixture
def del_norte_bbox() -> tuple:
    """Bounding box for Del Norte County, CA."""
    return (-124.408, 41.458, -123.536, 42.006)


@pytest.fixture
def sample_h3_cells() -> list[str]:
    """A small set of valid H3 resolution-8 cells over Del Norte County."""
    center_lat, center_lon = 41.75, -124.2
    center_cell = h3.latlng_to_cell(center_lat, center_lon, 8)
    return list(h3.grid_disk(center_cell, 1))


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for analysis outputs."""
    out = tmp_path / "place_output"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    """Minimal PlaceInterface config dict."""
    return {
        "location": {
            "bounds": {
                "west": -124.408,
                "south": 41.458,
                "east": -123.536,
                "north": 42.006,
            }
        },
        "spatial": {"h3_resolution": 8},
        "analyses": {
            "forest_health": {},
            "seismic_hazard": {},
        },
    }
