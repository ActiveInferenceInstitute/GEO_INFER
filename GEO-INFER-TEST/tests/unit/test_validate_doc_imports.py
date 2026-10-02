"""Unit tests for the documentation import-truth checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "GEO-INFER-TEST" / "validate_doc_imports.py"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_validate_doc_imports", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_repo(tmp_path: Path) -> Path:
    """Plant a minimal fixture repo: one real package, two doc pages."""
    repo = tmp_path / "repo"
    package = repo / "GEO-INFER-DEMO" / "src" / "geo_infer_demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'from .core.indexing import SpatialIndex\n\n__all__ = ["SpatialIndex"]\n'
    )
    (package / "core").mkdir()
    (package / "core" / "__init__.py").write_text("")
    (package / "core" / "indexing.py").write_text(
        "class SpatialIndex:\n    pass\n\n\ndef grid_disk(cell: str, k: int):\n    return []\n"
    )
    docs = repo / "GEO-INFER-INTRA" / "docs"
    docs.mkdir(parents=True)
    (docs / "good.md").write_text(
        "```python\nfrom geo_infer_demo import SpatialIndex\n"
        "from geo_infer_demo.core.indexing import grid_disk\n"
        "from geo_infer_demo.core.indexing import SpatialIndex as SI\n```\n"
    )
    (docs / "bad.md").write_text(
        "python from geo_infer_demo import SpatialAnalyzer analyzer = SpatialAnalyzer()\n"
    )
    (docs / "legacy.md").write_text(
        "# Legacy\n\n> **Illustrative example notice.** historical sketch\n\n"
        "```python\nfrom geo_infer_demo import SpatialAnalyzer\n```\n"
    )
    (docs / "multiline.md").write_text(
        "```python\nfrom geo_infer_demo import (\n    Missing,\n    SpatialIndex,\n)\n```\n"
    )
    return repo


def test_phantom_imports_flagged_and_real_imports_pass(tmp_path):
    validator = load_validator()
    repo = _make_repo(tmp_path)
    errors, diagnostics, pages_checked = validator.validate(repo)
    assert any("bad.md" in e and "SpatialAnalyzer" in e for e in errors)
    assert any("multiline.md" in e and "Missing" in e for e in errors)
    assert not any("good.md" in e for e in errors)
    assert any("legacy.md" in e and "exemption" in e for e in errors)
    assert any("legacy.md" in e and "SpatialAnalyzer" in e for e in errors)
    assert diagnostics == []
    assert pages_checked == 4


def test_midline_and_second_statement_imports_are_extracted(tmp_path):
    """Single-physical-line pages and chained statements must not hide phantoms."""
    validator = load_validator()
    repo = _make_repo(tmp_path)
    doc = repo / "GEO-INFER-INTRA" / "docs" / "collapsed.md"
    doc.write_text(
        "python from geo_infer_demo import MissingOne from geo_infer_demo import MissingTwo x = 1\n"
    )
    errors, _, _ = validator.validate(repo)
    text = "\n".join(errors)
    assert "MissingOne" in text and "MissingTwo" in text


def test_single_line_parenthesized_imports_are_extracted(tmp_path):
    validator = load_validator()
    repo = _make_repo(tmp_path)
    doc = repo / "GEO-INFER-INTRA" / "docs" / "paren.md"
    doc.write_text("python from geo_infer_demo import (Ghost, SpatialIndex) x = 1\n")
    errors, _, _ = validator.validate(repo)
    assert any("paren.md" in e and "Ghost" in e for e in errors)
