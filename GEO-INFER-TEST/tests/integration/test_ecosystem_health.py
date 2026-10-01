"""
Ecosystem Health Tests for GEO-INFER.

Validates the structural integrity, test coverage, and consistency
of all GEO-INFER modules as a unified ecosystem.
"""

from pathlib import Path

import pytest

# Canonical list of ALL GEO-INFER modules
GEO_INFER_MODULES = [
    "ACT",
    "AG",
    "AGENT",
    "AI",
    "ANT",
    "API",
    "APP",
    "ART",
    "BAYES",
    "BIO",
    "CIV",
    "CLIMATE",
    "COG",
    "COMMS",
    "DATA",
    "ECON",
    "EDU",
    "EMERGENCY",
    "ENERGY",
    "EXAMPLES",
    "FOREST",
    "GIT",
    "HEALTH",
    "INTRA",
    "IOT",
    "LOG",
    "MARINE",
    "MATH",
    "METAGOV",
    "NORMS",
    "OPS",
    "ORG",
    "PEP",
    "PLACE",
    "REQ",
    "RISK",
    "SEC",
    "SIM",
    "SPACE",
    "SPM",
    "TEST",
    "TIME",
    "TRANSPORT",
    "WATER",
]

REPO_ROOT = Path(__file__).resolve().parents[3]  # GEO-INFER repo root


def discover_test_files(tests_dir: Path) -> list[Path]:
    """Return pytest files for both supported naming conventions."""
    return sorted({*tests_dir.rglob("test_*.py"), *tests_dir.rglob("*_test.py")})


# ============================================================================
# Module structure tests
# ============================================================================


class TestModuleDirectoryStructure:
    """Verify each module has the expected directory structure."""

    @pytest.mark.parametrize("module", GEO_INFER_MODULES)
    def test_module_directory_exists(self, module):
        mod_dir = REPO_ROOT / f"GEO-INFER-{module}"
        assert mod_dir.is_dir(), f"Module directory missing: GEO-INFER-{module}"

    @pytest.mark.parametrize("module", GEO_INFER_MODULES)
    def test_module_has_readme(self, module):
        readme = REPO_ROOT / f"GEO-INFER-{module}" / "README.md"
        assert readme.is_file(), f"README.md missing in GEO-INFER-{module}"

    @pytest.mark.parametrize("module", GEO_INFER_MODULES)
    def test_module_has_tests_directory(self, module):
        tests_dir = REPO_ROOT / f"GEO-INFER-{module}" / "tests"
        assert tests_dir.is_dir(), f"tests/ directory missing in GEO-INFER-{module}"

    @pytest.mark.parametrize("module", GEO_INFER_MODULES)
    def test_module_has_test_files(self, module):
        tests_dir = REPO_ROOT / f"GEO-INFER-{module}" / "tests"
        if not tests_dir.is_dir():
            pytest.fail(f"No tests dir for {module}")
        test_files = discover_test_files(tests_dir)
        assert len(test_files) > 0, f"No test files found in GEO-INFER-{module}/tests/"


# ============================================================================
# Ecosystem-level counts and statistics
# ============================================================================


class TestEcosystemStatistics:
    """Verify ecosystem-level test statistics."""

    def test_total_module_count(self):
        """There should be 45 GEO-INFER modules."""
        module_dirs = [
            d
            for d in REPO_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("GEO-INFER-")
        ]
        assert len(module_dirs) >= 45, (
            f"Expected >=45 modules, found {len(module_dirs)}"
        )

    def test_total_test_file_count(self):
        """There should be 200+ test files across the ecosystem."""
        count = 0
        for module in GEO_INFER_MODULES:
            tests_dir = REPO_ROOT / f"GEO-INFER-{module}" / "tests"
            if tests_dir.is_dir():
                count += len(discover_test_files(tests_dir))
        assert count >= 200, f"Expected >=200 test files, found {count}"

    def test_test_discoverer_finds_all_modules(self):
        """TestDiscoverer should find all modules."""
        from geo_infer_test.core.test_discoverer import TestDiscoverer

        discoverer = TestDiscoverer(base_path=REPO_ROOT)
        results = discoverer.discover_all_tests(GEO_INFER_MODULES)
        # Should discover at least 40 modules
        modules_with_tests = [m for m in GEO_INFER_MODULES if m in results]
        assert len(modules_with_tests) >= 40, (
            f"TestDiscoverer found tests for {len(modules_with_tests)} modules, expected >=40"
        )


# ============================================================================
# Source structure validation
# ============================================================================


class TestSourceStructure:
    """Validate source code structure for each module."""

    @pytest.mark.parametrize("module", GEO_INFER_MODULES)
    def test_module_has_src_or_package(self, module):
        """Each module should have a src/ directory or a Python package."""
        mod_dir = REPO_ROOT / f"GEO-INFER-{module}"
        if not mod_dir.is_dir():
            pytest.fail(f"Module {module} directory missing")
        has_src = (mod_dir / "src").is_dir()
        has_package = any(
            (mod_dir / d / "__init__.py").is_file()
            for d in mod_dir.iterdir()
            if d.is_dir() and d.name.startswith("geo_infer")
        )
        assert has_src or has_package, (
            f"GEO-INFER-{module} has neither src/ nor a geo_infer_* package"
        )

    @pytest.mark.parametrize("module", GEO_INFER_MODULES)
    def test_module_packaging_is_pyproject_only(self, module):
        """Each module packages from pyproject.toml with no legacy mirrors."""
        mod_dir = REPO_ROOT / f"GEO-INFER-{module}"
        if not mod_dir.is_dir():
            pytest.fail(f"Module {module} directory missing")
        assert (mod_dir / "pyproject.toml").is_file(), (
            f"GEO-INFER-{module} has no pyproject.toml"
        )
        for retired in ("setup.py", "setup.cfg", "requirements.txt"):
            assert not (mod_dir / retired).exists(), (
                f"GEO-INFER-{module} carries retired packaging file {retired}"
            )
