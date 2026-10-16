"""Script-validator contract tests for the repo-contract gates.

The repo-contract scripts (``validate_test_contracts.py``,
``validate_documentation.py``, ``check_coverage_floor.py``) police the
repository itself, yet only ran untested against the live tree.  These tests
load each script as a module, point its repository root at a synthetic tmp
tree, and assert the gate fails on the violation it exists to catch and
passes on a conforming tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "GEO-INFER-TEST"


def _load_script(name: str, tmp_path: Path):
    """Load a validator script and repoint its repo root at ``tmp_path``."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO_ROOT = tmp_path
    if hasattr(module, "ROOT"):
        module.ROOT = tmp_path
    return module


def _seed_contract_tree(root: Path) -> Path:
    """Minimal repository tree that satisfies the strict test contract."""
    (root / "pyproject.toml").write_text('[project]\nname = "seed"\n', encoding="utf-8")
    module_dir = root / "GEO-INFER-SEED"
    (module_dir / "tests").mkdir(parents=True)
    (module_dir / "tests" / "README.md").write_text("# inventory\n", encoding="utf-8")
    (module_dir / "pyproject.toml").write_text(
        '[project]\nname = "geo-infer-seed"\n', encoding="utf-8"
    )
    return module_dir


class TestValidateTestContracts:
    """validate_test_contracts.py must flag forbidden controls on a tmp tree."""

    def test_clean_seed_tree_passes_strict(self, tmp_path):
        module = _load_script("validate_test_contracts", tmp_path)
        module_dir = _seed_contract_tree(tmp_path)
        (module_dir / "tests" / "test_ok.py").write_text(
            '"""Behavior under test."""\n\n'
            "def test_ok() -> None:\n"
            '    """The behavior holds."""\n'
            "    assert True\n",
            encoding="utf-8",
        )
        assert module.validate(strict=True) == []

    def test_forbidden_skip_control_is_flagged(self, tmp_path):
        module = _load_script("validate_test_contracts", tmp_path)
        module_dir = _seed_contract_tree(tmp_path)
        (module_dir / "tests" / "test_bad.py").write_text(
            '"""Behavior under test."""\n\n'
            "import pytest\n\n"
            "def test_bad() -> None:\n"
            '    """Uses a forbidden control."""\n'
            "    pytest." + "skip('nope')\n",
            encoding="utf-8",
        )
        errors = module.validate(strict=True)
        assert errors
        assert all("test_bad.py" in error for error in errors)

    def test_root_tests_tree_is_scanned(self, tmp_path):
        module = _load_script("validate_test_contracts", tmp_path)
        _seed_contract_tree(tmp_path)
        root_tests = tmp_path / "tests"
        root_tests.mkdir()
        (root_tests / "test_root_bad.py").write_text(
            '"""Root behavior."""\n\nimport pytest\n\n'
            "pytest.import" + "orskip('yaml')\n",
            encoding="utf-8",
        )
        errors = module.validate(strict=True)
        assert any("test_root_bad.py" in error for error in errors)


class TestValidateDocumentation:
    """validate_documentation.py link/strict behavior on a tmp doc hub."""

    @staticmethod
    def _docs_module(tmp_path: Path, pages: dict[str, str]):
        module = _load_script("validate_documentation", tmp_path)
        module.AUTHORITATIVE_DOCS = tuple(pages)
        for relative, text in pages.items():
            page = tmp_path / relative
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(text, encoding="utf-8")
        return module

    def test_broken_relative_link_fails(self, tmp_path):
        module = self._docs_module(
            tmp_path,
            {
                "GEO-INFER-INTRA/docs/index.md": "[nope](missing_target.md)\n",
            },
        )
        errors = module.validate_links(
            tuple(tmp_path / relative for relative in module.AUTHORITATIVE_DOCS)
        )
        assert any("missing_target.md" in error for error in errors)

    def test_github_entrypoint_is_authoritative_and_uses_parent_links(self, tmp_path):
        module = _load_script("validate_documentation", tmp_path)
        assert ".github/README.md" in module.AUTHORITATIVE_DOCS
        (tmp_path / "README.md").write_text("# Root overview\n")
        github = tmp_path / ".github" / "README.md"
        github.parent.mkdir()
        github.write_text("[root](../README.md)\n")
        assert module.validate_links((github,)) == []
        github.write_text("[root](README.md)\n[missing](../missing.md)\n")
        assert module.validate_links((github,)) == [
            ".github/README.md:2: missing ../missing.md"
        ]

    def test_strict_flags_missing_authoritative_page(self, tmp_path, monkeypatch):
        module = self._docs_module(
            tmp_path,
            {
                "GEO-INFER-INTRA/docs/index.md": "[ok](./index.md)\n",
                "GEO-INFER-INTRA/docs/overview.md": "missing on disk\n",
            },
        )
        (tmp_path / "GEO-INFER-INTRA/docs/overview.md").unlink()
        monkeypatch.setattr(sys, "argv", ["validate_documentation.py", "--strict"])
        assert module.main() == 1

    def test_stale_current_state_claim_is_flagged(self, tmp_path):
        module = self._docs_module(
            tmp_path,
            {
                "GEO-INFER-INTRA/docs/index.md": "Requires python 3.9+ today.\n",
            },
        )
        errors = module.validate_current_state_language(
            (tmp_path / "GEO-INFER-INTRA/docs/index.md",)
        )
        assert any("python 3.9+" in error for error in errors)

    def test_conforming_hub_passes_strict(self, tmp_path, monkeypatch):
        pages = {
            "GEO-INFER-INTRA/docs/index.md": "[hub](./index.md)\n",
            "GEO-INFER-INTRA/docs/overview.md": "[hub](./index.md)\n",
        }
        module = self._docs_module(tmp_path, pages)
        monkeypatch.setattr(sys, "argv", ["validate_documentation.py", "--strict"])
        assert module.main() == 0


class TestCheckCoverageFloor:
    """check_coverage_floor.py gate paths."""

    def test_no_changed_modules_passes(self, capsys):
        spec = importlib.util.spec_from_file_location(
            "check_coverage_floor_happy", SCRIPT_DIR / "check_coverage_floor.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.main(["--base", "HEAD", "--head", "HEAD"]) == 0
        assert "no module changes" in capsys.readouterr().out
