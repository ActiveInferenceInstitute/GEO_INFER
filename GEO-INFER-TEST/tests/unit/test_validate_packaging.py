"""Unit tests for the packaging / wheel-validation contract helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "GEO-INFER-TEST" / "validate_packaging.py"


def load_packaging_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_validate_packaging", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_pyproject(dir_path: Path) -> Path:
    """Plant a minimal conforming pyproject.toml in a module directory."""
    pyproject = dir_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "geo-infer-sample"',
                'version = "0.2.0"',
                "",
                "[tool.setuptools]",
                'package-dir = {"" = "src"}',
                "",
                "[tool.setuptools.package-data]",
                '"*" = ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return pyproject


def test_valid_distribution_namespace_accepts_platform_prefix(tmp_path, monkeypatch):
    packaging = load_packaging_module()
    assert packaging.valid_distribution_namespace("geo-infer-space") is True
    assert packaging.valid_distribution_namespace("geo-infer-test") is True


def test_valid_distribution_namespace_rejects_bad_prefix(tmp_path, monkeypatch):
    packaging = load_packaging_module()
    assert packaging.valid_distribution_namespace("third-party-pkg") is False
    assert packaging.valid_distribution_namespace("geo-infer") is False


def test_wheel_filename_namespace_validation(tmp_path, monkeypatch):
    packaging = load_packaging_module()
    assert packaging.wheel_filename_is_valid(
        "geo_infer_space-0.2.0-py3-none-any.whl", "geo-infer-space"
    )
    assert not packaging.wheel_filename_is_valid(
        "malicious_pkg-0.2.0-py3-none-any.whl", "geo-infer-space"
    )


def test_validate_module_accepts_conforming_package(tmp_path):
    packaging = load_packaging_module()
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    package_dir = module_dir / "src" / "geo_infer_sample"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("__version__ = '0.2.0'\n")
    _make_pyproject(module_dir)

    report = packaging.ContractReport()
    packaging.validate_module(module_dir, packaging.parse_pyproject(module_dir), report)
    assert report.errors == []


def test_validate_module_rejects_bad_namespace(tmp_path):
    packaging = load_packaging_module()
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    package_dir = module_dir / "src" / "geo_infer_sample"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("__version__ = '0.2.0'\n")
    pyproject = _make_pyproject(module_dir)
    pyproject.write_text(
        pyproject.read_text().replace(
            'name = "geo-infer-sample"', 'name = "bad-package"'
        ),
        encoding="utf-8",
    )

    report = packaging.ContractReport()
    packaging.validate_module(module_dir, packaging.parse_pyproject(module_dir), report)
    assert any("namespace" in e for e in report.errors)


def test_source_traversal_diagnostics_ignore_test_files(tmp_path):
    packaging = load_packaging_module()
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    pkg = module_dir / "src" / "geo_infer_sample"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '0.2.0'\n", encoding="utf-8")
    # A source file that climbs parent directories should be flagged; test files skipped.
    (pkg / "loader.py").write_text(
        "from pathlib import Path\nCONFIG = Path(__file__).parent.parent / 'config.yaml'\n",
        encoding="utf-8",
    )
    (pkg / "test_loader.py").write_text(
        "from pathlib import Path\nCONFIG = Path(__file__).parent.parent / 'config.yaml'\n",
        encoding="utf-8",
    )

    report = packaging.ContractReport()
    packaging.validate_source_traversal(module_dir, report)
    assert any("loader.py" in d for d in report.diagnostics)
    assert not any("test_loader.py" in d for d in report.diagnostics)


def test_validate_all_runs_on_live_monorepo():
    packaging = load_packaging_module()
    report = packaging.validate_all()
    # The live monorepo must already conform to the geo-infer-* namespace.
    assert report.errors == []


def test_release_metadata_rejects_root_member_and_component_drift(tmp_path):
    """Release parity binds runtime literals as well as wheel metadata."""
    packaging = load_packaging_module()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion="0.4.0"\n[tool.uv]\npackage=false\n'
    )
    member = tmp_path / "GEO-INFER-SAMPLE"
    package = member / "src" / "geo_infer_sample"
    (package / "nested").mkdir(parents=True)
    _make_pyproject(member)
    (package / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (package / "nested" / "__init__.py").write_text('__version__ = "1.0.0"\n')
    report = packaging.ContractReport()
    packaging.validate_release_metadata(
        tmp_path, [(member.name, packaging.parse_pyproject(member))], report
    )
    assert len(report.errors) == 3
    assert any("member version" in error for error in report.errors)
    assert any("nested/__init__.py" in error for error in report.errors)


def test_release_metadata_accepts_coherent_virtual_workspace(tmp_path):
    """Aligned metadata is accepted without executing package source."""
    packaging = load_packaging_module()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion="0.4.0"\n[tool.uv]\npackage=false\n'
    )
    member = tmp_path / "GEO-INFER-SAMPLE"
    package = member / "src" / "geo_infer_sample"
    package.mkdir(parents=True)
    project = _make_pyproject(member)
    project.write_text(
        project.read_text().replace('version = "0.2.0"', 'version = "0.4.0"')
    )
    (package / "__init__.py").write_text(
        '__version__ = "0.4.0"\nraise RuntimeError("never import")\n'
    )
    report = packaging.ContractReport()
    packaging.validate_release_metadata(
        tmp_path, [(member.name, packaging.parse_pyproject(member))], report
    )
    assert report.errors == []


def test_aggregate_extras_require_full_expansion_and_marker_parity():
    """A missing backend or weakened platform marker cannot pass aggregate parity."""
    packaging = load_packaging_module()
    project = {
        "project": {
            "name": "geo-infer-sample",
            "optional-dependencies": {
                "backend": ['redis>=4; python_version >= "3.11"'],
                "all": ["geo-infer-sample[backend]", "redis>=4"],
            },
        }
    }
    report = packaging.ContractReport()
    packaging.validate_aggregate_extras(project, "sample", report)
    assert len(report.errors) == 2
    project["project"]["optional-dependencies"]["all"] = project["project"][
        "optional-dependencies"
    ]["backend"].copy()
    report = packaging.ContractReport()
    packaging.validate_aggregate_extras(project, "sample", report)
    assert report.errors == []


def test_citation_version_matches_fleet_majority(tmp_path):
    packaging = load_packaging_module()
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        'cff-version: 1.2.0\nversion: "0.3.0"\ndate-released: "2026-09-17"\n',
        encoding="utf-8",
    )
    report = packaging.ContractReport()
    packaging.validate_citation_version(
        [("GEO-INFER-SAMPLE", {"project": {"version": "0.3.0"}})],
        report,
        citation_path=citation,
    )
    assert report.errors == []
    assert report.warnings == []


def test_citation_version_mismatch_warns_against_fleet_majority(tmp_path):
    packaging = load_packaging_module()
    citation = tmp_path / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\nversion: 9.9.9\n", encoding="utf-8")
    report = packaging.ContractReport()
    packaging.validate_citation_version(
        [
            ("GEO-INFER-A", {"project": {"version": "0.3.0"}}),
            ("GEO-INFER-B", {"project": {"version": "0.3.0"}}),
            ("GEO-INFER-C", {"project": {"version": "0.2.0"}}),
        ],
        report,
        citation_path=citation,
    )
    assert any(
        "CITATION.cff" in warning and "9.9.9" in warning for warning in report.warnings
    )


def test_citation_version_missing_file_warns(tmp_path):
    packaging = load_packaging_module()
    report = packaging.ContractReport()
    packaging.validate_citation_version(
        [("GEO-INFER-SAMPLE", {"project": {"version": "0.3.0"}})],
        report,
        citation_path=tmp_path / "absent" / "CITATION.cff",
    )
    assert any("CITATION.cff" in warning for warning in report.warnings)


def _make_test_parity_module(tmp_path: Path, pyproject_extra: str) -> Path:
    """Plant a module with a src package and a pyproject carrying deps."""
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    package_dir = module_dir / "src" / "geo_infer_sample"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("__version__ = '0.3.0'\n")
    (module_dir / "tests" / "unit").mkdir(parents=True)
    (module_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "geo-infer-sample"',
                'version = "0.3.0"',
                'dependencies = ["numpy>=1.20.0"]',
                "",
                "[project.optional-dependencies]",
                'config = ["pyyaml>=6.0"]',
                "",
                pyproject_extra,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return module_dir


WORKSPACE_PACKAGES = frozenset({"geo_infer_math", "geo_infer_sample"})


def test_pyproject_group_names_resolves_include_group():
    packaging = load_packaging_module()
    pyproject = {
        "dependency-groups": {
            "test": ["pytest>=6.2.0", {"include-group": "lint"}],
            "lint": ["ruff>=0.15.6,<0.16"],
            "cycle": [{"include-group": "cycle"}, "Py_Proj>=3.3.0"],
            "dangling": [{"include-group": "absent"}],
        }
    }
    assert packaging.pyproject_group_names(pyproject, "test") == {"pytest", "ruff"}
    assert packaging.pyproject_group_names(pyproject, "cycle") == {"py-proj"}
    assert packaging.pyproject_group_names(pyproject, "dangling") == set()
    assert packaging.pyproject_group_names(pyproject) == {"pytest", "ruff", "py-proj"}
    assert packaging.pyproject_group_names({}) == set()


def test_test_import_parity_accepts_declared_and_local_imports(tmp_path):
    packaging = load_packaging_module()
    module_dir = _make_test_parity_module(
        tmp_path,
        "\n".join(
            [
                "[dependency-groups]",
                'test = [{include-group = "base"}, "geo-infer-math>=0.3.0"]',
                'base = ["pytest>=6.2.0", "psutil>=5.9.0"]',
            ]
        ),
    )
    tests_dir = module_dir / "tests"
    (tests_dir / "unit" / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tests_dir / "unit" / "test_sample.py").write_text(
        "\n".join(
            [
                "import importlib",
                "import json",
                "import numpy as np",
                "import pytest",
                "import yaml",
                "from geo_infer_sample import __version__",
                "from helpers import VALUE",
                "from . import helpers",
                "",
                "MODULES = ['geo_infer_math', 'geo_infer_sample.core']",
                "",
                "def test_import(name='geo_infer_math'):",
                "    try:",
                "        import psutil",
                "    except ImportError:",
                "        psutil = None",
                "    importlib.import_module(name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = packaging.ContractReport()
    packaging.validate_test_import_parity(
        module_dir,
        packaging.parse_pyproject(module_dir),
        report,
        WORKSPACE_PACKAGES,
    )
    assert report.errors == []


def test_test_import_parity_rejects_guarded_dynamic_and_sibling_imports(tmp_path):
    packaging = load_packaging_module()
    module_dir = _make_test_parity_module(
        tmp_path, '[dependency-groups]\ntest = ["pytest>=6.2.0"]'
    )
    (module_dir / "tests" / "conftest.py").write_text(
        "\n".join(
            [
                "import importlib",
                "import pytest",
                "",
                "SIBLINGS = ['geo_infer_math.core']",
                "",
                "def load(name):",
                "    try:",
                "        import psutil",
                "    except ImportError:",
                "        return None",
                # Spelled in two parts so the test-contract text scan does not
                # flag this fixture line as a real skip.
                "    h3 = pytest.import" + "orskip('h3')",
                "    return importlib.import_module(name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = packaging.ContractReport()
    packaging.validate_test_import_parity(
        module_dir,
        packaging.parse_pyproject(module_dir),
        report,
        WORKSPACE_PACKAGES,
    )
    flagged = sorted(error.split("'")[1] for error in report.errors)
    assert flagged == ["geo_infer_math", "h3", "psutil"]
    assert all("tests/conftest.py" in error for error in report.errors)


def test_test_import_parity_maps_import_aliases_to_distributions(tmp_path):
    packaging = load_packaging_module()
    module_dir = _make_test_parity_module(tmp_path, "")
    (module_dir / "tests" / "test_alias.py").write_text(
        "from sklearn.linear_model import LinearRegression\n", encoding="utf-8"
    )
    report = packaging.ContractReport()
    packaging.validate_test_import_parity(
        module_dir,
        packaging.parse_pyproject(module_dir),
        report,
        WORKSPACE_PACKAGES,
    )
    assert len(report.errors) == 1
    assert "'scikit-learn'" in report.errors[0]
