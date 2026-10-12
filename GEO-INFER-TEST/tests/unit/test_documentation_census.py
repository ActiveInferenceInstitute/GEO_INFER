"""Catalog drift is rejected before expensive manuscript execution."""

import importlib.util
import sys
from pathlib import Path

import pytest

from geo_infer_test.documentation_contracts import module_catalog_census_errors


def seed_catalog(root, text):
    """Create a two-file owning module with independently known inventory."""
    module = root / "GEO-INFER-SPACE"
    (module / "src").mkdir(parents=True)
    (module / "tests" / "unit").mkdir(parents=True)
    for name in ("test_zero.py", "test_missing.py"):
        (module / "tests" / "unit" / name).write_text("# fixture\n")
    (module / "tests" / "helper.py").write_text("# not a test\n")
    cache = module / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "test_stale.py").write_text("# cached artifact\n")
    sections = root / "manuscript" / "sections"
    sections.mkdir(parents=True)
    (sections / "space.md").write_text(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The module has 2 test files.", None),
        ("The module has 1 test files.", "measured count is 2"),
        ("There are 2 test files and 3 test files.", "conflicting"),
        ("See the generated inventory table.", None),
    ],
)
def test_catalog_claims_match_measured_tree(tmp_path, text, expected):
    seed_catalog(tmp_path, text)
    errors = module_catalog_census_errors(tmp_path)
    if expected is None:
        assert errors == []
    else:
        assert len(errors) == 1
        assert expected in errors[0]


def test_claim_for_unknown_module_is_rejected(tmp_path):
    seed_catalog(tmp_path, "See the generated inventory table.")
    (tmp_path / "manuscript" / "sections" / "unknown.md").write_text(
        "There are 1 test files."
    )
    assert (
        "unmeasured module GEO-INFER-UNKNOWN"
        in module_catalog_census_errors(tmp_path)[0]
    )


def test_strict_documentation_cli_rejects_drift(tmp_path, monkeypatch, capsys):
    seed_catalog(tmp_path, "There are 7 test files.")
    script = Path(__file__).resolve().parents[2] / "validate_documentation.py"
    spec = importlib.util.spec_from_file_location("early_docs_gate", script)
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "AUTHORITATIVE_DOCS", ())
    monkeypatch.setattr(sys, "argv", [str(script), "--strict"])
    assert validator.main() == 1
    assert "measured count is 2" in capsys.readouterr().err
