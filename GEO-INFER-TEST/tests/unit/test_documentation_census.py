"""Catalog drift is rejected before expensive manuscript execution."""

import importlib.util
from dataclasses import replace
import sys
from pathlib import Path

import pytest

from geo_infer_test.documentation_contracts import (
    documentation_snippet_errors,
    module_catalog_census_errors,
)


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


def test_strict_documentation_cli_rejects_authored_snippet_regressions(
    tmp_path, monkeypatch, capsys
):
    page = tmp_path / "GEO-INFER-DEMO" / "docs" / "guide.md"
    page.parent.mkdir(parents=True)
    page.write_text("`python\nfrom geo_infer.fake import Missing\n`\n")
    script = Path(__file__).resolve().parents[2] / "validate_documentation.py"
    spec = importlib.util.spec_from_file_location("early_docs_snippet_gate", script)
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "AUTHORITATIVE_DOCS", ())
    monkeypatch.setattr(sys, "argv", [str(script), "--strict"])
    assert validator.main() == 1
    output = capsys.readouterr().err
    assert "guide.md:1: malformed single-backtick" in output
    assert "guide.md:2: nonexistent umbrella package geo_infer" in output


def test_act_agent_contract_reads_the_declared_pymdp_pin():
    script = Path(__file__).resolve().parents[2] / "rewrite_readme_agents.py"
    spec = importlib.util.spec_from_file_location("census_rewriter_pin", script)
    assert spec and spec.loader
    rewriter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rewriter
    spec.loader.exec_module(rewriter)
    module = rewriter.discover_modules()["GEO-INFER-ACT"]
    pin = rewriter.pinned_version(module, "inferactively-pymdp")
    text = rewriter.module_agent_notes(module.path / "AGENTS.md", module)
    assert f"exactly `{pin}`" in text
    changed = replace(module, dependencies=["inferactively-pymdp==9.8.7"])
    control = rewriter.module_agent_notes(module.path / "AGENTS.md", changed)
    assert "exactly `9.8.7`" in control
    assert f"exactly `{pin}`" not in control


def write_guide(root, text, name="guide.md"):
    page = root / "GEO-INFER-DEMO" / "docs" / name
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(text)
    return page


@pytest.mark.parametrize(
    "language", ["", "python", "bash", "text", "mermaid", "yaml", "json"]
)
def test_single_backtick_fence_regression_reports_actual_lines(tmp_path, language):
    write_guide(tmp_path, f"# Guide\n`{language}\nfragment\n`\n")
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 2
    assert "guide.md:2: malformed single-backtick" in errors[0]
    assert "guide.md:4: malformed single-backtick" in errors[1]


@pytest.mark.parametrize(
    "fence, closing",
    [("```python", "```"), ("~~~py", "~~~"), ("````python3", "````"), ("~~~", "~~~")],
)
def test_nonexistent_umbrella_imports_rejected_in_partial_python(
    tmp_path, fence, closing
):
    write_guide(
        tmp_path,
        f"{fence}\n"
        "from geo_infer.spatial.geometry import Point\n"
        "import os, geo_infer.spatial as spatial\n"
        "import geo_infer\n"
        "def intentionally_partial_signature(value: tuple[\n"
        f"{closing}\n",
    )
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 3
    assert all("nonexistent umbrella package geo_infer" in error for error in errors)
    assert [int(error.split(":")[1]) for error in errors] == [2, 3, 4]


def test_literal_payload_and_real_package_imports_are_not_rejected(tmp_path):
    write_guide(
        tmp_path,
        "````markdown\n`python\nfrom geo_infer.spatial import Point\n```\n````\n"
        "~~~bash\nimport geo_infer.fake\n~~~\n"
        "```python\n"
        "# from geo_infer.fake import Ghost\n"
        "literal = 'import geo_infer.fake'\n"
        'multiline = """\nfrom geo_infer.fake import Ghost\n"""\n'
        "from geo_infer_space import Point\n"
        "import geo_infer_data as data, geo_infer_art\n"
        "from actual_package import geo_infer\n"
        "from .geo_infer import local_name\n"
        "def signature_fragment(\n```\n",
    )
    assert documentation_snippet_errors(tmp_path) == []


def test_collapsed_unfenced_example_is_not_hidden_by_markdown_heading(tmp_path):
    write_guide(
        tmp_path,
        "# A collapsed guide python from geo_infer.spatial import Missing setup()\n"
        "python # Example: a collapsed from geo_infer.spatial import Missing setup()\n"
        "from actual_package import geo_infer\n"
        "import os, geo_infer.fake\n",
    )
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 3
    assert "guide.md:1: nonexistent umbrella" in errors[0]
    assert "guide.md:2: nonexistent umbrella" in errors[1]
    assert "guide.md:4: nonexistent umbrella" in errors[2]


def test_all_module_doc_pages_including_signposts_and_history_are_checked(tmp_path):
    for name in ["README.md", "AGENTS.md", "archive/history.md", "guide.md"]:
        write_guide(tmp_path, "```python\nimport geo_infer.fake\n```\n", name)
    ignored = tmp_path / "output" / "generated.md"
    ignored.parent.mkdir()
    ignored.write_text("`python\n")
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 4
    assert any("archive/history.md" in error for error in errors)
    assert any("README.md" in error for error in errors)
    assert any("AGENTS.md" in error for error in errors)
    assert not any("output" in error for error in errors)


def test_explicit_page_selection_checks_only_selected_real_file(tmp_path):
    good = write_guide(tmp_path, "```python\nimport geo_infer_space\n```\n")
    write_guide(tmp_path, "`python\n", "other.md")
    assert documentation_snippet_errors(tmp_path, [good]) == []


def test_fence_info_and_unterminated_fragments_cannot_hide_umbrella_import(tmp_path):
    write_guide(
        tmp_path,
        "```python from geo_infer.fake import Missing\n```\n"
        "~~~python title='Import example'\nimport geo_infer.fake\n",
    )
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 2
    assert "guide.md:1: nonexistent umbrella" in errors[0]
    assert "guide.md:4: nonexistent umbrella" in errors[1]


def test_collapsed_markdown_and_unlabeled_language_prefix_regression(tmp_path):
    write_guide(
        tmp_path,
        '# Collapsed markdown ```mermaid subgraph "partial quote '
        "```python from geo_infer.spatial import Missing\n"
        "```\npython # Example from geo_infer.spatial import Missing # next operation\n```\n",
    )
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 2
    assert "guide.md:1: nonexistent umbrella" in errors[0]
    assert "guide.md:3: nonexistent umbrella" in errors[1]


@pytest.mark.parametrize(
    "info", ["{.python}", "{#example .python}", '{.py title="A Python example"}']
)
def test_pandoc_python_attribute_fence_does_not_hide_umbrella_import(tmp_path, info):
    write_guide(tmp_path, f"```{info}\nfrom geo_infer import Missing\n```\n")
    errors = documentation_snippet_errors(tmp_path)
    assert len(errors) == 1
    assert "guide.md:2: nonexistent umbrella" in errors[0]


def test_python_word_in_non_python_attribute_value_is_literal_payload(tmp_path):
    write_guide(
        tmp_path,
        '```{.text title="A .python class is literal text"}\n'
        "from geo_infer import Missing\n```\n",
    )
    assert documentation_snippet_errors(tmp_path) == []
