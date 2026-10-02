"""Pins the unified runner's explicit extra test-path wiring.

``discover_geo_infer_modules`` only reaches top-level ``GEO-INFER-*/tests``
trees, so nested test estates must be attached deliberately: the cascadia
unit tree (``GEO-INFER-PLACE/locations/cascadia/tests/unit``) joins the PLACE
lanes via ``EXTRA_TEST_PATHS``; the cascadia integration tree joins the
integration lane with its local analytical fixtures. The Crescent City
civic-intel demo test must ride the TEST module's unit lane instead of being
stranded under ``GEO-INFER-TEST/demo/``.
"""

from __future__ import annotations

from geo_infer_test import execution as runner


def _module(name: str) -> runner.Module:
    return runner.module_by_name(name)


def test_cascadia_unit_tree_joins_the_place_unit_lane() -> None:
    """The two cascadia unit test files are resolved into the PLACE unit lane."""
    cascadia = [
        path
        for path in runner.category_test_paths(_module("PLACE"), "unit")
        if "locations" in path.parts and "cascadia" in path.parts
    ]
    assert {path.name for path in cascadia} == {
        "test_import.py",
        "test_geo_infer_integrations.py",
    }


def test_cascadia_integration_tree_joins_its_own_lane() -> None:
    paths = runner.category_test_paths(_module("PLACE"), "integration")
    cascadia = [path for path in paths if "cascadia" in path.parts]
    assert cascadia
    assert all("integration" in path.parts for path in cascadia)


def test_cascadia_tests_are_absent_from_nonmatching_lanes() -> None:
    """Local Cascadia unit/integration fixtures are not performance/system."""
    for category in ("system", "performance"):
        paths = runner.category_test_paths(_module("PLACE"), category)
        assert not any("cascadia" in path.parts for path in paths), category


def test_full_module_file_set_includes_cascadia_unit_tests() -> None:
    """A full ``--module PLACE`` run also picks up the cascadia unit files."""
    files = runner.module_test_files(_module("PLACE"))
    assert {
        path
        for path in files
        if "cascadia" in path.parts and path.name == "test_import.py"
    }


def test_demo_test_file_rides_the_test_module_unit_lane() -> None:
    """The moved civic-intel demo test is collected by the TEST unit lane."""
    demo_test = (
        runner.PROJECT_ROOT
        / "GEO-INFER-TEST"
        / "tests"
        / "unit"
        / "test_crescent_city_civic_intel_demo.py"
    )
    assert demo_test in runner.category_test_paths(_module("TEST"), "unit")
