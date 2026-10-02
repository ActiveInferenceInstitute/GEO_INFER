#!/usr/bin/env python3
"""
Test script for Cascadia modules

This script tests the converted modules to ensure they work correctly
with the BaseAnalysisModule pattern and SPACE integration.
"""

import logging
import sys
import tempfile
from pathlib import Path

import yaml

CASCADIA_ROOT = Path(__file__).resolve().parents[2]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_h3_integration():
    """Test basic H3 integration from SPACE"""
    logger.info("Testing H3 integration...")

    from geo_infer_space.utils.h3_utils import (
        cell_to_latlng,
        cell_to_latlng_boundary,
        latlng_to_cell,
    )

    # Test basic H3 functions
    lat, lng = 40.5, -120.5  # Lassen County area
    h3_cell = latlng_to_cell(lat, lng, 8)
    assert h3_cell, "H3 cell creation failed"
    logger.info(f"✅ H3 cell for ({lat}, {lng}): {h3_cell}")

    # Test reverse conversion
    lat2, lng2 = cell_to_latlng(h3_cell)
    assert abs(lat - lat2) < 0.1, "Reverse lat conversion inaccurate"
    assert abs(lng - lng2) < 0.1, "Reverse lng conversion inaccurate"
    logger.info(f"✅ Reverse conversion: ({lat2:.3f}, {lng2:.3f})")

    # Test boundary
    boundary = cell_to_latlng_boundary(h3_cell)
    assert len(boundary) > 0, "Boundary has no points"
    logger.info(f"✅ Boundary has {len(boundary)} points")


def _init_backend_and_modules(base_data_dir: Path):
    """Helper: create backend and initialize modules. Returns (backend, modules)."""
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

    from geo_infer_place.locations.cascadia.data_modules.current_use.geo_infer_current_use import (
        GeoInferCurrentUse,
    )
    from geo_infer_place.locations.cascadia.data_modules.improvements.geo_infer_improvements import (
        GeoInferImprovements,
    )
    from geo_infer_place.locations.cascadia.data_modules.ownership.geo_infer_ownership import (
        GeoInferOwnership,
    )
    from geo_infer_place.locations.cascadia.data_modules.zoning.geo_infer_zoning import (
        GeoInferZoning,
    )

    config_path = CASCADIA_ROOT / "config" / "analysis_config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config.get("analysis_settings"), "Cascadia analysis config is empty"

    base_data_dir.mkdir(parents=True, exist_ok=True)
    backend = CascadianAgriculturalH3Backend(
        modules={},
        resolution=8,
        bioregion="Cascadia",
        target_counties={"CA": ["Lassen"]},
        base_data_dir=base_data_dir,
    )
    logger.info(
        f"✅ Backend created with {len(backend.target_hexagons)} target hexagons"
    )

    modules = {}
    modules["zoning"] = GeoInferZoning(backend)
    logger.info("✅ Zoning module initialized")
    modules["current_use"] = GeoInferCurrentUse(backend)
    logger.info("✅ Current use module initialized")
    modules["ownership"] = GeoInferOwnership(backend)
    logger.info("✅ Ownership module initialized")
    modules["improvements"] = GeoInferImprovements(backend)
    logger.info("✅ Improvements module initialized")

    return backend, modules


def test_module_initialization(tmp_path):
    """Test module initialization with backend"""
    logger.info("Testing module initialization...")
    backend, modules = _init_backend_and_modules(tmp_path / "initialization")
    assert backend is not None, "Backend creation failed"
    assert len(modules) == 4, f"Expected 4 modules, got {len(modules)}"
    logger.info(f"✅ Successfully initialized {len(modules)} modules")


def test_module_workflow(tmp_path):
    """Test the BaseAnalysisModule workflow"""
    logger.info("Testing module workflow...")
    backend, modules = _init_backend_and_modules(tmp_path / "workflow")

    zoning_module = modules["zoning"]

    # Test that module has required methods
    required_methods = ["acquire_raw_data", "run_final_analysis", "run_analysis"]
    for method_name in required_methods:
        assert hasattr(zoning_module, method_name), f"{method_name} method missing"
        logger.info(f"✅ {method_name} method exists")

    # Test data directory creation
    assert zoning_module.data_dir.exists(), "Data directory not created"
    logger.info(f"✅ Data directory created: {zoning_module.data_dir}")


def main():
    """Run all tests (CLI entry point)."""
    logger.info("🚀 Starting Cascadia module tests...")

    with tempfile.TemporaryDirectory(prefix="cascadia_modules_") as temp_dir:
        temp_path = Path(temp_dir)
        tests = [
            ("H3 Integration", test_h3_integration),
            (
                "Module Initialization",
                lambda: test_module_initialization(temp_path),
            ),
            ("Module Workflow", lambda: test_module_workflow(temp_path)),
        ]

        passed = 0
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                test_func()
                passed += 1
                logger.info(f"{test_name}: ✅ PASS")
            except Exception as e:
                logger.error(f"{test_name}: ❌ FAIL — {e}")

    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
