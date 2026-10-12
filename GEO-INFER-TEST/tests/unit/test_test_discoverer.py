"""
Unit and property-based tests for the TestDiscoverer module.
"""

import tempfile
import shutil
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given, settings, strategies as st

from geo_infer_test.core.test_discoverer import TestDiscoverer as _TestDiscoverer


# ============================================================================
# Helpers
# ============================================================================


def _make_repo(base: Path, modules: list[str], with_tests: bool = True):
    """Create a mini GEO-INFER repo layout under *base*."""
    for mod in modules:
        mod_dir = base / f"GEO-INFER-{mod}"
        mod_dir.mkdir(parents=True, exist_ok=True)
        (mod_dir / "README.md").write_text(f"# {mod}")
        if with_tests:
            tests_dir = mod_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            unit_dir = tests_dir / "unit"
            unit_dir.mkdir(exist_ok=True)
            (unit_dir / f"test_{mod.lower()}.py").write_text(
                dedent(
                    f'''
                    """Tests for {mod}."""
                    import pytest

                    class Test{mod.title().replace("-", "")}Basic:
                        def test_example(self):
                            assert True
                '''
                )
            )
    return base


# ============================================================================
# Standard Tests
# ============================================================================


class TestTestDiscovererBasic:
    """Standard tests for TestDiscoverer."""

    @pytest.fixture
    def repo(self):
        tmp = tempfile.mkdtemp()
        yield _make_repo(Path(tmp), ["AAA", "BBB", "CCC"])
        shutil.rmtree(tmp)

    @pytest.fixture
    def empty_repo(self):
        tmp = tempfile.mkdtemp()
        yield _make_repo(Path(tmp), ["DDD"], with_tests=False)
        shutil.rmtree(tmp)

    def test_discover_finds_modules(self, repo):
        d = _TestDiscoverer(base_path=repo)
        results = d.discover_all_tests(["AAA", "BBB", "CCC"])
        assert "AAA" in results
        assert "BBB" in results
        assert "CCC" in results

    def test_discover_finds_test_files(self, repo):
        d = _TestDiscoverer(base_path=repo)
        results = d.discover_all_tests(["AAA"])
        # Should find at least the test file we created
        aaa = results.get("AAA", {})
        all_files = []
        for test_type_dict in aaa.values():
            if isinstance(test_type_dict, list):
                all_files.extend(test_type_dict)
            elif isinstance(test_type_dict, dict):
                for files in test_type_dict.values():
                    if isinstance(files, list):
                        all_files.extend(files)
        assert len(all_files) >= 1

    def test_general_bucket_lists_only_direct_children(self, repo):
        """GS19-86: "general" holds only tests/ root files, no double-count."""
        aaa_tests = repo / "GEO-INFER-AAA" / "tests"
        integration_dir = aaa_tests / "integration"
        integration_dir.mkdir()
        (integration_dir / "test_integration_thing.py").write_text(
            "def test_integration():\n    assert True\n"
        )
        (aaa_tests / "test_root_thing.py").write_text(
            "def test_root():\n    assert True\n"
        )
        (aaa_tests / "conftest.py").write_text("")

        d = _TestDiscoverer(base_path=repo)
        results = d.discover_all_tests(["AAA"])

        assert results["AAA"]["general"] == ["test_root_thing.py"]
        assert results["AAA"]["unit"] == ["test_aaa.py"]
        assert results["AAA"]["integration"] == ["test_integration_thing.py"]
        stats = d.get_test_statistics()
        assert stats["tests_by_module"]["AAA"] == 3
        assert stats["tests_by_type"]["general"] == 1

    def test_discover_unknown_module(self, repo):
        d = _TestDiscoverer(base_path=repo)
        results = d.discover_all_tests(["NONEXISTENT"])
        # Should still return a dict (possibly empty for unknown)
        assert isinstance(results, dict)

    def test_statistics_structure(self, repo):
        d = _TestDiscoverer(base_path=repo)
        d.discover_all_tests(["AAA", "BBB"])
        stats = d.get_test_statistics()
        assert isinstance(stats, dict)

    def test_validate_test_structure(self, repo):
        d = _TestDiscoverer(base_path=repo)
        d.discover_all_tests(["AAA"])
        validation = d.validate_test_structure()
        assert isinstance(validation, dict)

    def test_analyze_test_file(self, repo):
        test_file = repo / "GEO-INFER-AAA" / "tests" / "unit" / "test_aaa.py"
        d = _TestDiscoverer(base_path=repo)
        analysis = d.analyze_test_file(test_file)
        assert isinstance(analysis, dict)
        # Should detect pytest framework
        assert "framework" in analysis

    def test_find_cross_module_tests(self, repo):
        d = _TestDiscoverer(base_path=repo)
        d.discover_all_tests(["AAA", "BBB", "CCC"])
        cross_tests = d.find_cross_module_tests()
        assert isinstance(cross_tests, (list, dict))

    def test_empty_repo_no_crash(self, empty_repo):
        d = _TestDiscoverer(base_path=empty_repo)
        results = d.discover_all_tests(["DDD"])
        assert isinstance(results, dict)


# ============================================================================
# Parametric Tests
# ============================================================================

_MODULE_NAMES = [
    "SPACE",
    "TIME",
    "AI",
    "BAYES",
    "ACT",
    "AGENT",
    "SEC",
    "APP",
    "API",
    "LOG",
    "DATA",
    "OPS",
    "RISK",
    "ECON",
    "HEALTH",
]


class TestTestDiscovererParametric:
    """Parametric tests exercising many module combinations."""

    @pytest.mark.parametrize("mod", _MODULE_NAMES)
    def test_discover_single_module(self, mod):
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_repo(Path(tmp), [mod])
            d = _TestDiscoverer(base_path=base)
            results = d.discover_all_tests([mod])
            assert mod in results

    @pytest.mark.parametrize("count", [1, 2, 5, 10, 15])
    def test_discover_n_modules(self, count):
        mods = _MODULE_NAMES[:count]
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_repo(Path(tmp), mods)
            d = _TestDiscoverer(base_path=base)
            results = d.discover_all_tests(mods)
            assert len(results) == count

    @pytest.mark.parametrize("mod", _MODULE_NAMES)
    def test_statistics_per_module(self, mod):
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_repo(Path(tmp), [mod])
            d = _TestDiscoverer(base_path=base)
            d.discover_all_tests([mod])
            stats = d.get_test_statistics()
            assert isinstance(stats, dict)


# ============================================================================
# Property-Based Tests (Hypothesis)
# ============================================================================


# Selectors with fixed meaning: ROOT is the repository root, ALL expands to
# every owning module. Hypothesis can draw either from the uppercase alphabet.
RESERVED_SELECTORS = frozenset({"ROOT", "ALL"})


def test_root_selector_never_reads_a_module_directory_named_root(tmp_path):
    """ROOT is the repository root, whose tests are manuscript-only categories."""
    base = _make_repo(tmp_path, ["ROOT"])
    assert (base / "GEO-INFER-ROOT" / "tests" / "unit" / "test_root.py").is_file()
    results = _TestDiscoverer(base_path=base).discover_all_tests(["ROOT"])
    assert "test_root.py" not in str(results)
    assert results.get("ROOT", {}).get("unit") is None


def test_all_selector_expands_to_owning_modules(tmp_path):
    base = _make_repo(tmp_path, ["ALPHA", "BETA"])
    results = _TestDiscoverer(base_path=base).discover_all_tests(["ALL"])
    assert "ALL" not in results
    assert {"ALPHA", "BETA"} <= set(results)


class TestHypothesisTestDiscoverer:
    """Fuzzing tests for TestDiscoverer."""

    @settings(max_examples=200)
    @given(
        st.text(
            min_size=1,
            max_size=15,
            alphabet=st.characters(whitelist_categories=("Lu",)),
        )
    )
    def test_discover_never_crashes(self, module_name):
        """TestDiscoverer should never crash regardless of module name."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Create the module directory
            mod_dir = base / f"GEO-INFER-{module_name}"
            mod_dir.mkdir(parents=True, exist_ok=True)

            d = _TestDiscoverer(base_path=base)
            results = d.discover_all_tests([module_name])
            assert isinstance(results, dict)

    @settings(max_examples=200)
    @given(
        st.lists(
            st.text(
                min_size=1,
                max_size=8,
                alphabet=st.characters(whitelist_categories=("Lu",)),
            ).filter(lambda name: name not in RESERVED_SELECTORS),
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    def test_discover_all_returns_all(self, module_names):
        """All requested owning modules should appear in discovery results.

        ``ROOT`` and ``ALL`` are selectors, not ``GEO-INFER-*`` directories;
        their meaning is pinned separately below.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_repo(Path(tmp), module_names)
            d = _TestDiscoverer(base_path=base)
            results = d.discover_all_tests(module_names)
            for name in module_names:
                assert name in results

    @settings(max_examples=100)
    @given(st.text(min_size=1, max_size=20))
    def test_is_test_file_never_crashes(self, filename):
        """_is_test_file should handle any filename string."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _TestDiscoverer(base_path=Path(tmp))
            result = d._is_test_file(Path(filename))
            assert isinstance(result, bool)
