"""Real Git and producer operations preserve clean manuscript source custody."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType

import pytest


def _git(root: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    ).stdout.strip()


def _stale_source(root: Path, env: dict[str, str]) -> None:
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text().replace('version = "1.2.3"', 'version = "2.0.0"')
    )
    _git(root, env, "add", "pyproject.toml")
    _git(root, env, "commit", "-q", "-m", "update source version")


def _snapshot(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), stat.S_IMODE(path.stat().st_mode)


def _outputs(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "output").rglob("*")
        if path.is_file()
    }


def _cli(generator: ModuleType, root: Path, env: dict[str, str], *args: str):
    return subprocess.run(
        [sys.executable, "-I", str(generator.__file__), "--root", str(root), *args],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )


def test_fresh_clean_generation_preserves_tracked_config_bytes_and_permissions(
    generator: ModuleType, generatable_checkout: Path
):
    root = generatable_checkout
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    before = _snapshot(config)
    source_hash = generator._source_hash(root)
    manifest = generator.generate(root)
    assert _snapshot(config) == before
    assert manifest["dirty_file_count"] == generator._dirty_file_count(root) == 0
    assert not manifest["source_commit"].endswith("-dirty")
    assert manifest["source_hash"] == source_hash == generator._source_hash(root)
    assert (root / "output/data/research_manifest.json").is_file()


def test_stale_clean_generation_rejects_before_source_or_output_writes(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
):
    root = generatable_checkout
    _stale_source(root, git_environment)
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    before = _snapshot(config)
    source_hash = generator._source_hash(root)
    retained = root / "output/data/research_verification.json"
    retained.parent.mkdir(parents=True)
    retained.write_text('{"retained": "prior evidence"}\n')
    outputs = _outputs(root)
    with pytest.raises(
        RuntimeError, match="stale tracked config metadata.*refresh-config"
    ):
        generator.generate(root)
    assert _snapshot(config) == before
    assert generator._source_hash(root) == source_hash
    assert _outputs(root) == outputs
    assert generator._dirty_file_count(root) == 0


def test_failure_after_clean_generation_preflight_preserves_source_config(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
):
    root = generatable_checkout
    abstract = root / "manuscript/00_abstract.md"
    abstract.write_text("# Abstract\n\nMissing citation [@absent-entry].\n")
    _git(root, git_environment, "add", "manuscript/00_abstract.md")
    _git(root, git_environment, "commit", "-q", "-m", "add a bibliography regression")
    generator.refresh_config_metadata(root, generator.config_metadata_values(root))
    if generator._dirty_file_count(root):
        _git(root, git_environment, "add", "manuscript/config.yaml")
        _git(root, git_environment, "commit", "-q", "-m", "settle manuscript metadata")
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    before = _snapshot(config)
    with pytest.raises(ValueError, match="citations with no bibliography entry"):
        generator.generate(root)
    assert (root / "output/data/manuscript_variables.json").is_file()
    assert _snapshot(config) == before
    assert generator._dirty_file_count(root) == 0


def test_explicit_dirty_refresh_stamps_actual_post_refresh_tree(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
):
    root = generatable_checkout
    _stale_source(root, git_environment)
    assert generator._dirty_file_count(root) == 0
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    original, mode = _snapshot(config)
    manifest = generator.generate(root, allow_dirty=True)
    assert config.read_bytes() != original
    assert stat.S_IMODE(config.stat().st_mode) == mode
    assert manifest["dirty_file_count"] == generator._dirty_file_count(root) == 1
    assert manifest["source_commit"].endswith("-dirty")
    assert manifest["source_hash"] == generator._source_hash(root)
    variables = json.loads((root / "output/data/manuscript_variables.json").read_text())
    assert variables["RESEARCH_TREE_DIRTY_FILE_COUNT"] == "1"
    assert variables["RESEARCH_SOURCE_HASH"] == generator._source_hash(root)


def test_refresh_only_cli_and_config_only_commit_reach_a_clean_fixed_point(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
):
    root = generatable_checkout
    source_date = "2026-10-16T12:00:00+00:00"
    config_date = "2026-10-17T12:00:00+00:00"
    _stale_source(
        root,
        {
            **git_environment,
            "GIT_AUTHOR_DATE": source_date,
            "GIT_COMMITTER_DATE": source_date,
        },
    )
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    original, mode = _snapshot(config)
    measured_date = generator._manuscript_source_date(root)
    assert datetime.fromisoformat(measured_date) == datetime.fromisoformat(source_date)
    refreshed = _cli(generator, root, git_environment, "--refresh-config")
    assert refreshed.returncode == 0, refreshed.stderr
    assert "paper.version" in json.loads(refreshed.stdout)["changed_fields"]
    assert config.read_bytes() != original
    assert stat.S_IMODE(config.stat().st_mode) == mode
    assert not (root / "output").exists()
    assert (
        _git(root, git_environment, "status", "--porcelain")
        == "M manuscript/config.yaml"
    )
    _git(root, git_environment, "add", "manuscript/config.yaml")
    _git(
        root,
        {
            **git_environment,
            "GIT_AUTHOR_DATE": config_date,
            "GIT_COMMITTER_DATE": config_date,
        },
        "commit",
        "-q",
        "-m",
        "record generated config only",
    )
    assert datetime.fromisoformat(
        _git(root, git_environment, "show", "-s", "--format=%cI", "HEAD")
    ) == datetime.fromisoformat(config_date)
    before = _snapshot(config)
    again = _cli(generator, root, git_environment, "--refresh-config")
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout) == {"changed_fields": []}
    assert generator._manuscript_source_date(root) == measured_date
    assert generator._dirty_file_count(root) == 0
    assert _snapshot(config) == before
    assert not (root / "output").exists()


@pytest.mark.parametrize(
    "conflict",
    [
        "--check",
        "--verify",
        "--full-validation",
        "--publication",
        "--rerun-verification",
        "--allow-dirty",
    ],
)
def test_refresh_cli_rejects_conflicts_before_any_write(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
    conflict: str,
):
    root = generatable_checkout
    _stale_source(root, git_environment)
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    before = _snapshot(config)
    result = _cli(generator, root, git_environment, "--refresh-config", conflict)
    assert result.returncode == 2
    assert "--refresh-config cannot be combined" in result.stderr
    assert _snapshot(config) == before
    assert generator._dirty_file_count(root) == 0
    assert not (root / "output").exists()


def test_publication_cannot_authorize_dirty_refresh_in_library_or_cli(
    generator: ModuleType,
    generatable_checkout: Path,
    git_environment: dict[str, str],
):
    root = generatable_checkout
    _stale_source(root, git_environment)
    config = root / "manuscript/config.yaml"
    config.chmod(0o640)
    before = _snapshot(config)
    with pytest.raises(ValueError, match="publication requires clean source"):
        generator.generate(root, publication=True, verify=True, allow_dirty=True)
    result = _cli(
        generator, root, git_environment, "--publication", "--verify", "--allow-dirty"
    )
    assert result.returncode == 2
    assert "--publication requires clean source" in result.stderr
    assert _snapshot(config) == before
    assert generator._dirty_file_count(root) == 0
    assert not (root / "output").exists()
