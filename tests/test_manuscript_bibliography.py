"""Bibliography gate tests.

``bibliography.fail_on_missing`` and ``bibliography.fail_on_unused`` are
documented config keys that no renderer code reads.  The generator honours
them, so these tests pin that the gate is real rather than declared.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import ModuleType

import pytest


@pytest.fixture
def bib_tree(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    manuscript = root / "manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "references.bib").write_text(
        "@article{cited_entry,\n  title = {Cited},\n}\n\n"
        "@book{orphan_entry,\n  title = {Orphan},\n}\n",
        encoding="utf-8",
    )
    (manuscript / "config.yaml").write_text(
        "bibliography:\n"
        '  references_path: "manuscript/references.bib"\n'
        "  fail_on_missing: true\n"
        "  fail_on_unused: false\n",
        encoding="utf-8",
    )
    return root


def _body(root: Path, text: str) -> list[Path]:
    body = root / "body.md"
    body.write_text(text, encoding="utf-8")
    return [body]


class TestBibliographyAudit:
    def test_an_orphan_entry_is_reported(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        uncited, undefined = generator.audit_bibliography(
            bib_tree, _body(bib_tree, "As shown [@cited_entry], the result holds.\n")
        )
        assert uncited == ("orphan_entry",)
        assert undefined == ()

    def test_a_citation_with_no_entry_is_reported(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        _uncited, undefined = generator.audit_bibliography(
            bib_tree,
            _body(bib_tree, "See [@cited_entry] and [@orphan_entry] and [@ghost].\n"),
        )
        assert undefined == ("ghost",)

    def test_crossreferences_are_not_bibliography_keys(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        _uncited, undefined = generator.audit_bibliography(
            bib_tree,
            _body(
                bib_tree,
                "See [@fig:module_inventory], [@tbl:evidence] and [@sec:methods]; "
                "[@cited_entry] and [@orphan_entry] cover the rest.\n",
            ),
        )
        assert undefined == ()

    def test_an_email_address_is_not_a_citation(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        _uncited, undefined = generator.audit_bibliography(
            bib_tree,
            _body(
                bib_tree,
                "Contact daniel@activeinference.institute. "
                "[@cited_entry] [@orphan_entry]\n",
            ),
        )
        assert undefined == ()

    def test_policy_is_read_from_config(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        assert generator.bibliography_policy(bib_tree) == {
            "fail_on_missing": True,
            "fail_on_unused": False,
        }
        config = bib_tree / "manuscript" / "config.yaml"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                "fail_on_unused: false", "fail_on_unused: true"
            ),
            encoding="utf-8",
        )
        assert generator.bibliography_policy(bib_tree)["fail_on_unused"] is True


class TestPolicySectionAwareness:
    """The policy read is section-aware and fails closed on a parse miss.

    The line-prefix scan used to accept any two-space-prefixed key anywhere
    in the file and silently fall back to defaults on a miss.  Because
    ``fail_on_unused`` defaults to ``False``, a reindentation of the
    ``bibliography:`` block — or a same-named key under another section —
    quietly disabled the orphaned-citation gate.
    """

    def test_a_reindented_bibliography_block_raises_instead_of_defaulting(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        config = bib_tree / "manuscript" / "config.yaml"
        config.write_text(
            "bibliography:\n"
            '    references_path: "manuscript/references.bib"\n'
            "    fail_on_missing: true\n"
            "    fail_on_unused: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="bibliography block"):
            generator.bibliography_policy(bib_tree)

    def test_a_same_prefixed_key_in_another_section_is_ignored(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        config = bib_tree / "manuscript" / "config.yaml"
        config.write_text(
            "render:\n"
            "  fail_on_unused: true\n"
            "\n"
            "bibliography:\n"
            '  references_path: "manuscript/references.bib"\n'
            "  fail_on_missing: true\n"
            "  fail_on_unused: false\n",
            encoding="utf-8",
        )
        assert generator.bibliography_policy(bib_tree) == {
            "fail_on_missing": True,
            "fail_on_unused": False,
        }

    def test_a_missing_fail_on_unused_key_raises(
        self, generator: ModuleType, bib_tree: Path
    ) -> None:
        config = bib_tree / "manuscript" / "config.yaml"
        config.write_text(
            "bibliography:\n"
            '  references_path: "manuscript/references.bib"\n'
            "  fail_on_missing: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="fail_on_unused"):
            generator.bibliography_policy(bib_tree)


class TestRealBibliography:
    def test_no_citation_is_missing_an_entry(
        self, generator: ModuleType, repo_root: Path
    ) -> None:
        resolved = sorted((repo_root / "output" / "manuscript").glob("*.md"))
        assert resolved, "run scripts/z_generate_manuscript_variables.py first"
        _uncited, undefined = generator.audit_bibliography(repo_root, resolved)
        assert undefined == ()

    def test_the_orphan_is_visible_rather_than_silent(
        self, generator: ModuleType, repo_root: Path
    ) -> None:
        # GI-M3: gelman_bda_2014 sits in references.bib and is cited nowhere.
        # The gate makes that visible on every run; flipping fail_on_unused to
        # true once it is cited makes it fatal.
        resolved = sorted((repo_root / "output" / "manuscript").glob("*.md"))
        uncited, _undefined = generator.audit_bibliography(repo_root, resolved)
        policy = generator.bibliography_policy(repo_root)
        assert policy["fail_on_missing"] is True
        if uncited:
            assert policy["fail_on_unused"] is False, (
                "fail_on_unused is set, so an uncited entry must fail the build "
                f"rather than reach this assertion: {uncited}"
            )


def _settle_source(generator: ModuleType, root: Path, env: dict[str, str]) -> None:
    """Commit authored fixture changes and explicitly settle tracked metadata."""

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env=env,
        )

    git("add", "-A")
    git("commit", "-q", "-m", "bibliography fixture")
    generator.refresh_config_metadata(root, generator.config_metadata_values(root))
    if generator._dirty_file_count(root):
        git("add", "manuscript/config.yaml")
        git("commit", "-q", "-m", "settle bibliography metadata")


def _output_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in (root / "output").rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("problem", ["orphan", "undefined", "chapter-undefined"])
def test_static_citation_failure_precedes_verification_and_output_writes(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    root = generatable_checkout
    if problem == "orphan":
        (root / "manuscript/references.bib").write_text(
            "@book{orphan_entry, title={Uncited}}\n"
        )
        # Operational docs are excluded from the rendered publication. Module
        # chapters are included and are exercised separately below.
        for name in generator.EXCLUDED_MANUSCRIPT_DOCS:
            (root / "manuscript" / name).write_text("[@orphan_entry]\n")
        match = "bibliography entries are never cited: orphan_entry"
    elif problem == "undefined":
        (root / "manuscript/00_abstract.md").write_text("# Abstract\n[@ghost]\n")
        match = "citations with no bibliography entry: ghost"
    else:
        (root / "manuscript/sections").mkdir()
        (root / "manuscript/sections/example.md").write_text("## Example\n[@ghost]\n")
        match = "citations with no bibliography entry: ghost"
    _settle_source(generator, root, git_environment)
    prior = root / "output/data/research_verification.json"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'{"prior": "retained evidence"}\n')
    (root / "output/prior.pdf").write_bytes(b"retained publication bytes\x00")
    before = _output_bytes(root)
    config_before = (root / "manuscript/config.yaml").read_bytes()
    admitted: list[bool] = []

    def unexpected_verification(*args, **kwargs):
        admitted.append(True)
        pytest.fail("static bibliography failure admitted expensive verification")

    monkeypatch.setattr(generator, "run_verification", unexpected_verification)
    with pytest.raises(ValueError, match=match):
        generator.generate(root, verify=True, reuse_verification=False)
    assert admitted == []
    assert _output_bytes(root) == before
    assert (root / "manuscript/config.yaml").read_bytes() == config_before
    assert generator._dirty_file_count(root) == 0


@pytest.mark.parametrize("problem", ["orphan", "undefined", "chapter-undefined"])
def test_false_policy_allows_generation_and_only_final_audit_warns(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    problem: str,
) -> None:
    root = generatable_checkout
    config = root / "manuscript/config.yaml"
    key = "fail_on_unused" if problem == "orphan" else "fail_on_missing"
    config.write_text(config.read_text().replace(f"{key}: true", f"{key}: false"))
    if problem == "orphan":
        (root / "manuscript/references.bib").write_text(
            "@book{orphan_entry, title={Uncited}}\n"
        )
    elif problem == "undefined":
        (root / "manuscript/00_abstract.md").write_text("# Abstract\n[@ghost]\n")
    else:
        (root / "manuscript/sections").mkdir()
        (root / "manuscript/sections/example.md").write_text("## Example\n[@ghost]\n")
    _settle_source(generator, root, git_environment)
    generator.generate(root)
    written = root / "output/manuscript/00_abstract.md"
    assert written.is_file()
    uncited, undefined = generator._audit_publication_bibliography(root, [written])
    assert (uncited, undefined) == (
        (("orphan_entry",), ()) if problem == "orphan" else ((), ("ghost",))
    )
    warnings = capsys.readouterr().err
    assert warnings.count("bibliography entries are never cited") == (
        1 if problem == "orphan" else 0
    )


def test_existing_variable_can_supply_a_valid_bibliography_citation(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
) -> None:
    root = generatable_checkout
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text().replace('license = "MIT"', 'license = "MIT [@cited_entry]"')
    )
    (root / "manuscript/references.bib").write_text(
        "@book{cited_entry, title={Used through existing variable}}\n"
    )
    (root / "manuscript/00_abstract.md").write_text(
        "# Abstract\nLicense: {{PROJECT_LICENSE}}\n"
    )
    _settle_source(generator, root, git_environment)
    generator.generate(root)
    assert (
        "MIT [@cited_entry]" in (root / "output/manuscript/00_abstract.md").read_text()
    )


def test_chapter_only_citation_is_included_without_expanding_resolved_outputs(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
) -> None:
    root = generatable_checkout
    (root / "manuscript/references.bib").write_text(
        "@book{chapter_entry, title={Cited only by the rendered module catalog}}\n"
    )
    chapters = root / "manuscript/sections"
    chapters.mkdir()
    chapter = chapters / "example.md"
    chapter.write_text("## Example\nSee [@chapter_entry].\n")
    _settle_source(generator, root, git_environment)
    manifest = generator.generate(root)
    assert manifest["resolved_manuscript_files"] == ["output/manuscript/00_abstract.md"]
    assert not (root / "output/manuscript/example.md").exists()
    assert generator.check_published_artifacts(root) == ()
    # The generic audit still examines exactly its caller's supplied paths.
    assert generator.audit_bibliography(
        root, [root / "output/manuscript/00_abstract.md"]
    ) == (("chapter_entry",), ())
    before = _output_bytes(root)
    chapter.write_text(chapter.read_text() + "See [@chapter_ghost].\n")
    problems = generator.check_published_artifacts(root)
    assert "citations with no bibliography entry: chapter_ghost" in problems
    assert _output_bytes(root) == before


def test_publication_preflight_keeps_catalog_chapter_text_literal(
    generator: ModuleType, bib_tree: Path
) -> None:
    chapters = bib_tree / "manuscript/sections"
    chapters.mkdir()
    (chapters / "example.md").write_text("{{PROJECT_LICENSE}}\n")
    top_level = _body(bib_tree, "[@cited_entry]\n")
    assert generator._audit_publication_bibliography(
        bib_tree,
        top_level,
        variables={"PROJECT_LICENSE": "[@orphan_entry]"},
    ) == (("orphan_entry",), ())


def test_check_does_not_use_extra_output_markdown_to_rescue_orphan(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
) -> None:
    root = generatable_checkout
    (root / "manuscript/references.bib").write_text(
        "@book{cited_entry, title={Used in authored prose}}\n"
    )
    chapters = root / "manuscript/sections"
    chapters.mkdir()
    chapter = chapters / "example.md"
    chapter.write_text("## Example\n[@cited_entry]\n")
    _settle_source(generator, root, git_environment)
    generator.generate(root)
    assert generator.check_published_artifacts(root) == ()
    chapter.write_text("## Example\nThe reference is no longer used.\n")
    (root / "output/manuscript/extra.md").write_text("[@cited_entry]\n")
    before = _output_bytes(root)
    assert "bibliography entries are never cited: cited_entry" in (
        generator.check_published_artifacts(root)
    )
    assert _output_bytes(root) == before


def test_resolved_audit_also_reads_rendered_catalog_chapters(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = generatable_checkout
    chapters = root / "manuscript/sections"
    chapters.mkdir()
    chapter = chapters / "example.md"
    chapter.write_text("## Example\nNo citation.\n")
    _settle_source(generator, root, git_environment)
    original = generator.write_resolved_manuscript

    def changed_chapter(*args, **kwargs):
        written = original(*args, **kwargs)
        chapter.write_text("## Example\n[@late_chapter_ghost]\n")
        return written

    monkeypatch.setattr(generator, "write_resolved_manuscript", changed_chapter)
    with pytest.raises(
        ValueError, match="citations with no bibliography entry: late_chapter_ghost"
    ):
        generator.generate(root)


def test_generated_citation_still_rejected_by_resolved_audit(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = generatable_checkout
    (root / "manuscript/00_abstract.md").write_text("# Abstract\nNo static citation.\n")
    _settle_source(generator, root, git_environment)
    original = generator.write_resolved_manuscript

    def generated_citation(*args, **kwargs):
        written = original(*args, **kwargs)
        written[0].write_text(written[0].read_text() + "\n[@dynamic_ghost]\n")
        return written

    monkeypatch.setattr(generator, "write_resolved_manuscript", generated_citation)
    with pytest.raises(
        ValueError, match="citations with no bibliography entry: dynamic_ghost"
    ):
        generator.generate(root)
    assert "[@dynamic_ghost]" in (root / "output/manuscript/00_abstract.md").read_text()
