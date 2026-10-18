"""Unit tests for SKILL.md semantic validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from geo_infer_test.documentation_contracts import skill_python_syntax_errors

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_PATH = REPO_ROOT / "GEO-INFER-TEST" / "validate_skills.py"


def load_skills_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_validate_skills", SKILLS_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_skill_claim_language_rejects_unscoped_planned_claim():
    skills = load_skills_module()
    content = "\n".join(
        [
            "## Guidelines",
            "- Missing dashboard workflow planned for v0.4.0",
        ]
    )

    errors = skills.validate_skill_claim_language(content, "[TEST]")

    assert errors
    assert "planned" in errors[0]


def test_skill_claim_language_allows_intentional_placeholder_contexts():
    skills = load_skills_module()
    content = "\n".join(
        [
            "- HTML form inputs use standard `placeholder` attributes.",
            "- SQL uses parameterized queries (`:param` placeholders).",
            "- Students fill in the `pass` block in the generated template.",
        ]
    )

    errors = skills.validate_skill_claim_language(content, "[TEST]")

    assert errors == []


def test_skill_claim_language_ignores_code_blocks():
    skills = load_skills_module()
    content = "\n".join(
        [
            "```python",
            "def exercise():",
            "    pass",
            "```",
        ]
    )

    errors = skills.validate_skill_claim_language(content, "[TEST]")

    assert errors == []


def test_cli_help_lists_documented_flags():
    """Every flag advertised in the usage docstring must be accepted by the CLI (GS-292)."""
    import re
    import subprocess

    docstring = Path(SKILLS_PATH).read_text()
    documented = sorted(set(re.findall(r"--[a-z][a-z-]*\b", docstring)))
    assert documented, "usage docstring advertises no flags"

    proc = subprocess.run(
        [sys.executable, str(SKILLS_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"--help exited {proc.returncode}: {proc.stderr}"
    for flag in documented:
        assert flag in proc.stdout, (
            f"docstring advertises {flag} but the parser does not accept it"
        )


def test_cli_accepts_documented_flags_without_argparse_error():
    """Documented flag combinations must parse (exit never 2 from argparse)."""
    import subprocess

    for argv in ([], ["--verbose"], ["--check-xrefs"], ["--verbose", "--check-xrefs"]):
        proc = subprocess.run(
            [sys.executable, str(SKILLS_PATH), *argv],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode in (0, 1), (
            f"{' '.join(argv) or '(no flags)'} exited {proc.returncode}: {proc.stderr}"
        )


def test_warnings_fatal_fails_on_cross_reference_warnings(monkeypatch):
    """--warnings-fatal promotes xref warnings to a non-zero exit (GS-003)."""
    skills = load_skills_module()
    monkeypatch.setattr(skills, "validate_root_skill", lambda verbose=False: [])
    monkeypatch.setattr(skills, "find_module_dirs", lambda: [])
    monkeypatch.setattr(
        skills,
        "validate_cross_references",
        lambda verbose=False: ["[root] broken cross-reference"],
    )
    monkeypatch.setattr(
        sys, "argv", ["validate_skills.py", "--check-xrefs", "--warnings-fatal"]
    )

    assert skills.main() == 1


def test_cross_reference_warnings_stay_advisory_without_flag(monkeypatch):
    """Without --warnings-fatal, xref warnings alone must not fail the run."""
    skills = load_skills_module()
    monkeypatch.setattr(skills, "validate_root_skill", lambda verbose=False: [])
    monkeypatch.setattr(skills, "find_module_dirs", lambda: [])
    monkeypatch.setattr(
        skills,
        "validate_cross_references",
        lambda verbose=False: ["[root] broken cross-reference"],
    )
    monkeypatch.setattr(sys, "argv", ["validate_skills.py", "--check-xrefs"])

    assert skills.main() == 0


@pytest.mark.parametrize(
    "opening, closing",
    [
        ("```python", "```"),
        ("~~~py", "~~~"),
        ("````python3", "````"),
        ("```{.python}", "```"),
        ('~~~{#example .python title="Skill example"}', "~~~"),
    ],
)
def test_explicit_python_skill_fences_reject_actual_missing_parenthesis(
    opening, closing
):
    errors = skill_python_syntax_errors(
        f"# Skill\n{opening}\nmodel = Model(\n{closing}\n", "SKILL.md"
    )
    assert len(errors) == 1
    assert "SKILL.md:3: invalid Python example" in errors[0]


def test_skill_syntax_check_never_executes_and_blocks_parse_independently():
    assert (
        skill_python_syntax_errors(
            "```python\nraise AssertionError('must never execute')\n```\n", "[TEST]"
        )
        == []
    )
    errors = skill_python_syntax_errors(
        "```python\nvalue = (\n```\n```python\n1)\n```\n", "[TEST]"
    )
    assert len(errors) == 2
    assert "[TEST]:2:" in errors[0]
    assert "[TEST]:5:" in errors[1]


def test_skill_only_explicit_python_is_parsed_with_real_markdown_indentation():
    content = (
        "````markdown\n```python\ninvalid = (\n```\n````\n"
        "~~~text\ninvalid = (\n~~~\n"
        "```\npartial_signature(\n```\n"
        "  ~~~python\n  def complete():\n      return 1\n  ~~~\n"
    )
    assert skill_python_syntax_errors(content, "[TEST]", require_example=True) == []


def test_skill_python_fence_must_close_and_required_example_is_explicit():
    assert skill_python_syntax_errors("```python\nx = 1\n", "[TEST]") == [
        "[TEST]:1: unclosed Python code fence"
    ]
    assert skill_python_syntax_errors(
        "```text\npython\n```", "[TEST]", require_example=True
    ) == ["[TEST] No Python code examples found"]


def test_module_and_root_skill_validators_reject_syntax_from_real_files(
    tmp_path, monkeypatch
):
    skills = load_skills_module()
    sections = "\n".join(
        ["## Instructions", "## Examples", "### Integrations", "## Guidelines"]
    )
    body = sections + "\n```python\nmodel = Model(\n```\n" + "\n" * 30
    module = tmp_path / "GEO-INFER-DEMO" / "SKILL.md"
    module.parent.mkdir()
    module.write_text(
        "---\nname: geo-infer-demo\ndescription: Validated instructions\n---\n" + body
    )
    errors = skills.validate_skill_file(module, "geo-infer-demo")
    assert len(errors) == 1
    assert "[geo-infer-demo]" in errors[0] and "invalid Python example" in errors[0]
    (tmp_path / "SKILL.md").write_text(
        "---\nname: geo-infer\ndescription: Instructions\n---\n" + body
    )
    monkeypatch.setattr(skills, "REPO_ROOT", tmp_path)
    assert any(
        "[ROOT]" in error and "invalid Python example" in error
        for error in skills.validate_root_skill()
    )


def test_pandoc_python_example_satisfies_module_example_requirement(tmp_path):
    skills = load_skills_module()
    content = (
        "---\nname: geo-infer-demo\ndescription: Operational example\n---\n"
        "## Instructions\n## Examples\n### Integrations\n## Guidelines\n"
        "```{#example .python}\nassert 1 + 1 == 2\n```\n" + "\n" * 30
    )
    path = tmp_path / "SKILL.md"
    path.write_text(content)
    assert skills.validate_skill_file(path, "geo-infer-demo") == []
