"""Tests for repository-derived manuscript variables and figure evidence."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "docs" / "manuscript" / "generate_research_artifacts.py"
SPEC = importlib.util.spec_from_file_location(
    "geo_infer_manuscript_research", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MANUSCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MANUSCRIPT
SPEC.loader.exec_module(MANUSCRIPT)

EXCLUDED_MANUSCRIPT_DOCS = MANUSCRIPT.EXCLUDED_MANUSCRIPT_DOCS
TOKEN_RE = MANUSCRIPT.TOKEN_RE
build_variables = MANUSCRIPT.build_variables
collect_inventory = MANUSCRIPT.collect_inventory
generate_figures = MANUSCRIPT.generate_figures
substitute_manuscript_text = MANUSCRIPT.substitute_manuscript_text
write_figure_registry = MANUSCRIPT.write_figure_registry


def test_inventory_counts_live_modules_and_test_files():
    inventory = collect_inventory(REPO_ROOT)

    module_paths = tuple(
        path
        for path in REPO_ROOT.glob("GEO-INFER-*")
        if path.is_dir() and (path / "src").is_dir()
    )
    test_paths = {
        path
        for module in module_paths
        for path in (module / "tests").rglob("test_*.py")
        if path.is_file()
    }
    test_paths.update(
        path
        for path in (REPO_ROOT / "GEO-INFER-TEST" / "tests").rglob("test_*.py")
        if path.is_file()
    )

    assert inventory.module_count == len(module_paths)
    assert inventory.test_files == len(test_paths)
    assert inventory.source_files == sum(
        path.is_file()
        for module in module_paths
        for path in (module / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_every_authored_variable_has_a_generated_value(tmp_path):
    inventory = collect_inventory(REPO_ROOT)
    specs = generate_figures(inventory, tmp_path / "figures")
    variables = build_variables(
        inventory, specs, MANUSCRIPT.VerificationRecord.unmeasured()
    )

    tokens: set[str] = set()
    for source in (REPO_ROOT / "docs" / "manuscript").iterdir():
        if (
            source.suffix not in {".md", ".yaml"}
            or source.name in EXCLUDED_MANUSCRIPT_DOCS
        ):
            continue
        tokens.update(TOKEN_RE.findall(source.read_text(encoding="utf-8")))

    assert tokens
    assert tokens <= variables.keys()
    assert all("{{" not in value and "}}" not in value for value in variables.values())


def test_substitution_fails_closed_for_unknown_tokens():
    resolved, unresolved = substitute_manuscript_text(
        "known={{KNOWN}} unknown={{UNKNOWN}}", {"KNOWN": "measured"}
    )

    assert resolved == "known=measured unknown={{UNKNOWN}}"
    assert unresolved == ("UNKNOWN",)


def test_figure_registry_contains_every_generated_caption(tmp_path):
    inventory = collect_inventory(REPO_ROOT)
    figures_dir = tmp_path / "figures"
    specs = generate_figures(inventory, figures_dir)
    registry_path = figures_dir / "figure_registry.json"
    write_figure_registry(registry_path, specs, inventory)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["schema_version"]
    assert registry["source_hash"] == inventory.source_hash
    assert len(registry["figures"]) == len(specs)
    assert {entry["caption"] for entry in registry["figures"]} == {
        spec.caption for spec in specs
    }
    assert all(
        (figures_dir / entry["filename"]).is_file() for entry in registry["figures"]
    )
