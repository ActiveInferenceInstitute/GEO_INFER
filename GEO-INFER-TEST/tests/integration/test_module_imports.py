"""
Module Import Validation Tests for GEO-INFER Ecosystem.

Parametrically validates that every GEO-INFER module's Python package
can be imported cleanly. This catches broken __init__.py, missing
dependencies, circular imports, and syntax errors across the ecosystem.
"""

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

# Module name → expected package name mappings
# Most follow the convention: GEO-INFER-FOO → geo_infer_foo
MODULE_PACKAGES = {
    "ACT": "geo_infer_act",
    "AG": "geo_infer_ag",
    "AGENT": "geo_infer_agent",
    "AI": "geo_infer_ai",
    "ANT": "geo_infer_ant",
    "API": "geo_infer_api",
    "APP": "geo_infer_app",
    "ART": "geo_infer_art",
    "BAYES": "geo_infer_bayes",
    "BIO": "geo_infer_bio",
    "CIV": "geo_infer_civ",
    "CLIMATE": "geo_infer_climate",
    "COG": "geo_infer_cog",
    "COMMS": "geo_infer_comms",
    "DATA": "geo_infer_data",
    "ECON": "geo_infer_econ",
    "EDU": "geo_infer_edu",
    "EMERGENCY": "geo_infer_emergency",
    "ENERGY": "geo_infer_energy",
    "EXAMPLES": "geo_infer_examples",
    "FOREST": "geo_infer_forest",
    "GIT": "geo_infer_git",
    "HEALTH": "geo_infer_health",
    "INTRA": "geo_infer_intra",
    "IOT": "geo_infer_iot",
    "LOG": "geo_infer_log",
    "MARINE": "geo_infer_marine",
    "MATH": "geo_infer_math",
    "METAGOV": "geo_infer_metagov",
    "NORMS": "geo_infer_norms",
    "OPS": "geo_infer_ops",
    "ORG": "geo_infer_org",
    "PEP": "geo_infer_pep",
    "PLACE": "geo_infer_place",
    "REQ": "geo_infer_req",
    "RISK": "geo_infer_risk",
    "SEC": "geo_infer_sec",
    "SIM": "geo_infer_sim",
    "SPACE": "geo_infer_space",
    "SPM": "geo_infer_spm",
    "TEST": "geo_infer_test",
    "TIME": "geo_infer_time",
    "TRANSPORT": "geo_infer_transport",
    "WATER": "geo_infer_water",
}


class TestModuleImports:
    """Validate that each module's Python package imports cleanly."""

    @pytest.mark.parametrize("module, package", list(MODULE_PACKAGES.items()))
    def test_module_package_importable(self, module, package):
        """Each package imports from its workspace ``src/`` install."""
        mod = importlib.import_module(package)
        expected_root = REPO_ROOT / f"GEO-INFER-{module}" / "src" / package
        assert mod.__file__ is not None
        assert Path(mod.__file__).resolve().parent == expected_root.resolve(), (
            f"{package} resolved to {mod.__file__}, not the workspace src/ tree"
        )

    @pytest.mark.parametrize("module, package", list(MODULE_PACKAGES.items()))
    def test_module_has_version_or_init(self, module, package):
        """Each package should have a non-empty ``src/<package>/__init__.py``."""
        init_file = REPO_ROOT / f"GEO-INFER-{module}" / "src" / package / "__init__.py"
        assert init_file.is_file(), f"No __init__.py found for {package}"
        content = init_file.read_text()
        assert len(content) > 0, f"{package}/__init__.py is empty"


class TestModulePyprojectConsistency:
    """Validate pyproject.toml fields for each module."""

    @pytest.mark.parametrize("module", list(MODULE_PACKAGES.keys()))
    def test_module_pyproject_exists(self, module):
        """Each module should have pyproject.toml."""
        pyproject = REPO_ROOT / f"GEO-INFER-{module}" / "pyproject.toml"
        if not pyproject.is_file():
            pytest.fail(f"No pyproject.toml for GEO-INFER-{module}")
        content = pyproject.read_text()
        assert "[project]" in content or "[tool." in content, (
            f"pyproject.toml for GEO-INFER-{module} has no [project] or [tool.*] section"
        )

    @pytest.mark.parametrize("module", list(MODULE_PACKAGES.keys()))
    def test_module_pyproject_has_name(self, module):
        """pyproject.toml should declare a project name."""
        pyproject = REPO_ROOT / f"GEO-INFER-{module}" / "pyproject.toml"
        if not pyproject.is_file():
            pytest.fail(f"No pyproject.toml for GEO-INFER-{module}")
        content = pyproject.read_text()
        if "[project]" in content:
            assert "name" in content, (
                f"pyproject.toml for GEO-INFER-{module} missing name field"
            )
