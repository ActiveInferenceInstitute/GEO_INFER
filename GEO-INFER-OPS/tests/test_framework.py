"""Framework integration tests converted to pytest.

Tests that the module structure, imports, and cross-module integration
work correctly. Originally a standalone script, converted to pytest format.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
MINIMUM_MODULE_COUNT = 44


def _discover_module_packages() -> list[str]:
    """Return geo_infer_* packages under GEO-INFER-*/src/, excluding egg-info."""
    packages: list[str] = []
    for src_dir in ROOT.glob("GEO-INFER-*/src"):
        for pkg in src_dir.iterdir():
            if (
                pkg.is_dir()
                and pkg.name.startswith("geo_infer_")
                and "egg-info" not in pkg.name
            ):
                packages.append(pkg.name)
    return sorted(packages)


# --- Tests ---


def test_module_count() -> None:
    """At least 44 GEO-INFER module packages are discoverable."""
    modules = _discover_module_packages()
    assert len(modules) >= MINIMUM_MODULE_COUNT, (
        f"Expected >= {MINIMUM_MODULE_COUNT} modules, found {len(modules)}: {modules}"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "geo_infer_space",
        "geo_infer_place",
        "geo_infer_math",
        "geo_infer_bayes",
        "geo_infer_act",
        "geo_infer_iot",
        "geo_infer_sec",
        "geo_infer_agent",
        "geo_infer_sim",
    ],
)
def test_core_modules_import(module_name: str) -> None:
    """Core modules import without error."""
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")


def test_cross_module_references() -> None:
    """Key cross-module symbols can be imported from their modules."""
    checks: list[tuple[str, str]] = [
        ("geo_infer_space", "PlaceAnalyzer"),
        ("geo_infer_space", "SpatialIndexingInterface"),
        ("geo_infer_iot", "IoTDataIngestion"),
        ("geo_infer_bayes", "GaussianProcess"),
        ("geo_infer_sec", "SecurityFramework"),
    ]
    failures: list[str] = []
    for module_name, symbol in checks:
        try:
            mod = importlib.import_module(module_name)
            if not hasattr(mod, symbol):
                failures.append(f"{module_name} has no attribute {symbol}")
        except ImportError as e:
            failures.append(f"Cannot import {module_name}: {e}")
    assert not failures, "Cross-module reference failures:\n" + "\n".join(failures)


def test_all_module_packages_have_init() -> None:
    """Every geo_infer_* package has an __init__.py."""
    missing = []
    for src_dir in ROOT.glob("GEO-INFER-*/src"):
        if not src_dir.is_dir():
            continue
        for pkg in src_dir.iterdir():
            if not pkg.is_dir() or not pkg.name.startswith("geo_infer_"):
                continue
            if "egg-info" in pkg.name:
                continue
            init = pkg / "__init__.py"
            if not init.exists():
                missing.append(str(pkg))
    assert not missing, "Packages missing __init__.py:\n" + "\n".join(missing)
