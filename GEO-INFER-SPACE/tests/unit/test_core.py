import pytest
import json
from pathlib import Path
from geo_infer_space.core.unified_backend import UnifiedH3Backend
import shutil
from importlib.metadata import version

import geo_infer_space


def test_package_version_matches_distribution_metadata():
    """The importable version matches the workspace distribution metadata."""
    assert geo_infer_space.__version__ == version("geo-infer-space")


@pytest.mark.core
def test_backend_init():
    """Test UnifiedH3Backend initialization with real parameters."""
    temp_config_dir = Path("config")
    temp_config_dir.mkdir(exist_ok=True)
    config_path = temp_config_dir / "target_areas.geojson"
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"area": "Test", "subarea": "all"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    with open(config_path, "w") as f:
        json.dump(sample_geojson, f)
    backend = UnifiedH3Backend(
        modules={}, resolution=8, target_region="Test", base_data_dir=Path("test_data")
    )
    assert backend.resolution == 8
    assert backend.target_region == "Test"
    assert isinstance(backend.base_data_dir, Path)
    shutil.rmtree(temp_config_dir)


@pytest.mark.core
def test_calculate_scores():
    """Test score calculation with real small data."""
    temp_config_dir = Path("config")
    temp_config_dir.mkdir(exist_ok=True)
    config_path = temp_config_dir / "target_areas.geojson"
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"area": "Test", "subarea": "all"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    with open(config_path, "w") as f:
        json.dump(sample_geojson, f)
    backend = UnifiedH3Backend(modules={}, resolution=8)
    backend.unified_data = {
        "hex1": {"module1": {"score": 0.5}, "module2": {"score": 0.7}}
    }
    scores = backend.calculate_analysis_scores()
    assert "hex1" in scores
    assert scores["hex1"]["composite_score"] == 0.6
    shutil.rmtree(temp_config_dir)
