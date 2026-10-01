#!/usr/bin/env python3
"""Run deterministic integration checks for the Cascadia framework."""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

CASCADIA_ROOT = Path(__file__).resolve().parents[2]
CASCADIA_SRC = CASCADIA_ROOT / "src"

# Run directly, this script's directory is tests/integration; the Cascadia
# location's own ``src`` package (not a workspace member) lives at CASCADIA_ROOT.
if str(CASCADIA_ROOT) not in sys.path:
    sys.path.insert(0, str(CASCADIA_ROOT))

logger = logging.getLogger(__name__)
Check = tuple[str, Callable[[], None]]


def _load_cascadia_main() -> ModuleType:
    """Import the tracked entry point without changing the process CWD."""
    module_path = CASCADIA_ROOT / "cascadia_main.py"
    spec = importlib.util.spec_from_file_location("cascadia_validation_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Cascadia entry point: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_main_script_contract() -> None:
    """Verify the current entry point and its required orchestration functions."""
    entry_cwd = Path.cwd()
    module = _load_cascadia_main()
    required_functions = (
        "parse_counties",
        "initialize_analysis",
        "initialize_modules_with_enhanced_data_management",
        "generate_reports",
        "load_analysis_config",
        "main",
    )
    missing = [name for name in required_functions if not callable(getattr(module, name, None))]
    assert not missing, f"missing Cascadia entry-point functions: {missing}"
    assert Path.cwd() == entry_cwd, "importing cascadia_main changed the process CWD"


def check_configuration_contract() -> None:
    """Load the tracked analysis config through its explicit path seam."""
    module = _load_cascadia_main()
    config_path = CASCADIA_ROOT / "config" / "analysis_config.yaml"
    config = module.load_analysis_config(config_path)
    assert isinstance(config, dict), "analysis config must be a mapping"
    settings = config.get("analysis_settings")
    assert isinstance(settings, dict) and settings, (
        "analysis config must define non-empty analysis_settings"
    )


def check_module_structure() -> None:
    """Verify each configured core data module has an importable source package."""
    for module_name in ("zoning", "current_use", "ownership", "improvements"):
        module_dir = CASCADIA_SRC / "data_modules" / module_name
        expected = module_dir / f"geo_infer_{module_name}.py"
        assert (module_dir / "__init__.py").is_file(), f"missing package initializer: {module_dir}"
        assert expected.is_file(), f"missing module implementation: {expected}"


def check_h3_integration() -> None:
    """Exercise the SPACE H3 wrapper with a real round trip."""
    from geo_infer_space.utils.h3_utils import (
        cell_to_latlng,
        cell_to_latlng_boundary,
        latlng_to_cell,
    )

    latitude, longitude = 40.5, -120.5
    cell = latlng_to_cell(latitude, longitude, 8)
    roundtrip_latitude, roundtrip_longitude = cell_to_latlng(cell)
    boundary = cell_to_latlng_boundary(cell)

    assert abs(latitude - roundtrip_latitude) < 0.01
    assert abs(longitude - roundtrip_longitude) < 0.01
    assert len(boundary) == 6


def _build_backend(base_data_dir: Path):
    """Create a real bounded backend over one tracked county boundary."""
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

    return CascadianAgriculturalH3Backend(
        modules={},
        resolution=8,
        bioregion="Cascadia",
        target_counties={"CA": ["Lassen"]},
        base_data_dir=base_data_dir,
        enable_caching=False,
    )


def check_backend_initialization() -> None:
    """Initialize the production backend without external data services."""
    with tempfile.TemporaryDirectory(prefix="cascadia_backend_") as temp_dir:
        backend = _build_backend(Path(temp_dir))
        assert backend.bioregion == "Cascadia"
        assert backend.resolution == 8
        assert backend.target_hexagons


def check_data_module_initialization() -> None:
    """Instantiate the four core data modules against the production backend."""
    from src.data_modules.current_use.geo_infer_current_use import GeoInferCurrentUse
    from src.data_modules.improvements.geo_infer_improvements import (
        GeoInferImprovements,
    )
    from src.data_modules.ownership.geo_infer_ownership import GeoInferOwnership
    from src.data_modules.zoning.geo_infer_zoning import GeoInferZoning

    module_classes = (
        GeoInferCurrentUse,
        GeoInferImprovements,
        GeoInferOwnership,
        GeoInferZoning,
    )
    with tempfile.TemporaryDirectory(prefix="cascadia_modules_") as temp_dir:
        backend = _build_backend(Path(temp_dir))
        for module_class in module_classes:
            module = module_class(backend)
            assert module.backend is backend
            assert callable(module.run_analysis)
            if isinstance(module, GeoInferOwnership):
                assert module.data_source.config, "ownership URL config was not loaded"
                assert module.data_source.arcgis_service_urls, (
                    "ownership URL config contains no usable ArcGIS services"
                )


VALIDATION_CHECKS: tuple[Check, ...] = (
    ("main script contract", check_main_script_contract),
    ("configuration contract", check_configuration_contract),
    ("module structure", check_module_structure),
    ("H3 integration", check_h3_integration),
    ("backend initialization", check_backend_initialization),
    ("data-module initialization", check_data_module_initialization),
)


def run_checks(checks: Sequence[Check] = VALIDATION_CHECKS) -> dict[str, bool]:
    """Run every supplied check and return its exact pass/fail status."""
    results: dict[str, bool] = {}
    for name, check in checks:
        try:
            check()
        except Exception:
            logger.exception("FAIL: %s", name)
            results[name] = False
        else:
            logger.info("PASS: %s", name)
            results[name] = True
    return results


def main(checks: Sequence[Check] = VALIDATION_CHECKS) -> int:
    """Run a validation profile and return a strict process status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_checks(checks)
    passed = sum(results.values())
    logger.info("Cascadia validation: %d/%d checks passed", passed, len(results))
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
