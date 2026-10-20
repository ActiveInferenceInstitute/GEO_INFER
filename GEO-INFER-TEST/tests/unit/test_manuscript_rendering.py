"""Real Git/source-boundary contracts for the optional template renderer."""

from pathlib import Path
import hashlib
import subprocess

import pytest

from geo_infer_test.manuscript_rendering import (
    render_template_pdf,
    template_source_identity,
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


@pytest.fixture
def template_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "template"
    root.mkdir()
    git(root, "init")
    git(root, "config", "core.fsmonitor", "false")
    git(root, "config", "user.name", "Source identity fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    directory = root / "infrastructure/rendering"
    directory.mkdir(parents=True)
    for filename in ("pdf_renderer.py", "config.py", "_pdf_title_page.py"):
        (directory / filename).write_text(
            "# Source identity fixture data; never executed.\n"
        )
    git(root, "add", ".")
    git(root, "commit", "-m", "source identity fixture")
    return root


def test_identity_binds_actual_revision_tree_and_every_source(
    template_checkout: Path,
) -> None:
    identity = template_source_identity(template_checkout)
    assert identity["revision"] == git(template_checkout, "rev-parse", "HEAD")
    assert identity["tree"] == git(template_checkout, "rev-parse", "HEAD^{tree}")
    assert len(identity["source_sha256"]) == 3
    for name, digest in identity["source_sha256"].items():
        assert (
            digest
            == hashlib.sha256((template_checkout / name).read_bytes()).hexdigest()
        )
    assert (
        template_source_identity(
            template_checkout, expected_revision=identity["revision"]
        )
        == identity
    )


@pytest.mark.parametrize("revision", ["main", "1234567", "0" * 40, "g" * 40])
def test_revision_must_be_the_exact_requested_commit(
    template_checkout: Path, revision: str
) -> None:
    with pytest.raises(ValueError, match="revision"):
        template_source_identity(template_checkout, expected_revision=revision)


@pytest.mark.parametrize("filename", ["infrastructure/rendering/config.py", "extra.py"])
def test_dirty_or_untracked_source_is_refused(
    template_checkout: Path, filename: str
) -> None:
    (template_checkout / filename).write_text("# changed source\n")
    with pytest.raises(ValueError, match="clean"):
        template_source_identity(template_checkout)


def test_nested_directory_does_not_masquerade_as_checkout_root(
    template_checkout: Path,
) -> None:
    with pytest.raises(ValueError, match="checkout root"):
        template_source_identity(template_checkout / "infrastructure")


def test_committed_missing_renderer_is_refused(template_checkout: Path) -> None:
    git(template_checkout, "rm", "infrastructure/rendering/pdf_renderer.py")
    git(template_checkout, "commit", "-m", "remove renderer")
    with pytest.raises(ValueError, match="missing regular"):
        template_source_identity(template_checkout)


def test_optional_gitlink_is_recorded_without_claiming_its_contents(
    template_checkout: Path,
) -> None:
    prior = git(template_checkout, "rev-parse", "HEAD")
    git(
        template_checkout,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000," + prior + ",infrastructure/optional_backend",
    )
    (template_checkout / "infrastructure/optional_backend").mkdir()
    git(template_checkout, "commit", "-m", "optional Git link")
    identity = template_source_identity(template_checkout)
    assert identity["gitlinks"] == {"infrastructure/optional_backend": prior}
    assert "infrastructure/optional_backend" not in identity["source_sha256"]


def test_source_paths_preserve_unicode_and_spaces(template_checkout: Path) -> None:
    name = "infrastructure/rendering/résumé notes.py"
    (template_checkout / name).write_text("# source data\n")
    git(template_checkout, "add", ".")
    git(template_checkout, "commit", "-m", "Unicode source data")
    assert name in template_source_identity(template_checkout)["source_sha256"]


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 3601])
def test_invalid_deadlines_fail_before_any_render(
    tmp_path: Path, timeout: float
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        render_template_pdf(tmp_path / "absent", tmp_path / "absent", timeout=timeout)
    assert list(tmp_path.iterdir()) == []
