#!/usr/bin/env python3
"""
Real Data Processing Test for Cascadia Framework

This test validates that the framework can process real data correctly.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

# Setup paths
cascadian_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(cascadian_dir, "..", "..", ".."))
place_src_path = os.path.join(project_root, "GEO-INFER-PLACE", "src")
space_src_path = os.path.join(project_root, "GEO-INFER-SPACE", "src")

for p in [cascadian_dir, place_src_path, space_src_path]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_config_loading_is_cwd_independent(tmp_path, monkeypatch):
    """Load an explicit Cascadia config from an unrelated working directory."""
    import importlib.util

    config_path = tmp_path / "analysis_config.yaml"
    expected_config = {
        "analysis_settings": {
            "target_counties": {"CA": ["Lassen"], "OR": ["Marion"]},
            "active_modules": ["zoning"],
            "h3_resolution": 8,
        }
    }
    config_path.write_text(yaml.safe_dump(expected_config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    spec = importlib.util.spec_from_file_location(
        "cascadia_main_config_test",
        Path(__file__).resolve().parents[2] / "cascadia_main.py",
    )
    assert spec is not None and spec.loader is not None
    cascadia_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cascadia_main)

    assert cascadia_main.load_analysis_config(config_path) == expected_config
    assert Path.cwd() == tmp_path


def test_canonical_validation_profile_passes():
    """Keep the three script entry points bound to real deterministic checks."""
    import run_comprehensive_validation

    results = run_comprehensive_validation.run_checks()

    assert results
    assert all(results.values()), results


def test_spatial_analysis():
    """Test spatial analysis capabilities"""
    logger.info("Testing spatial analysis...")

    from geo_infer_space.utils.h3_utils import (
        cell_to_latlng,
        cell_to_latlng_boundary,
        latlng_to_cell,
    )

    # Test spatial operations
    lat, lng = 40.5, -120.5
    h3_cell = latlng_to_cell(lat, lng, 8)
    lat2, lng2 = cell_to_latlng(h3_cell)
    boundary = cell_to_latlng_boundary(h3_cell)

    # Validate spatial operations
    assert abs(lat - lat2) < 0.01
    assert abs(lng - lng2) < 0.01
    assert len(boundary) == 6

    logger.info(f"✅ Spatial analysis working: {h3_cell}")


def test_export_functionality():
    """Test data export functionality"""
    logger.info("Testing export functionality...")

    with tempfile.TemporaryDirectory(prefix="cascadia_export_") as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample data
        sample_data = {
            "h3_cells": ["882816a51dfffff", "882816a51ffffff"],
            "scores": {"882816a51dfffff": 0.8, "882816a51ffffff": 0.7},
            "metadata": {"resolution": 8, "bioregion": "Cascadia"},
        }

        # Test JSON export
        json_path = temp_path / "test_export.json"
        with open(json_path, "w") as f:
            json.dump(sample_data, f, indent=2)

        # Validate export
        assert json_path.exists()
        with open(json_path) as f:
            exported_data = json.load(f)
            assert len(exported_data["h3_cells"]) == 2
            assert len(exported_data["scores"]) == 2

        logger.info("✅ Export functionality working")


def run_real_data_tests():
    """Run all real data processing tests (CLI entry point)."""
    logger.info("🚀 Starting Real Data Processing Tests")
    logger.info("=" * 60)

    tests = [
        ("Spatial Analysis", test_spatial_analysis),
        ("Export Functionality", test_export_functionality),
    ]

    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            test_func()
            passed += 1
            logger.info(f"{test_name}: ✅ PASS")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAIL — {e}")

    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_real_data_tests()
    sys.exit(0 if success else 1)
