"""Tests for the ``python -m geo_infer_git.main`` bulk-clone entrypoint."""

import subprocess
import sys
from pathlib import Path

import pytest

import geo_infer_git.main as git_main

MODULE_ROOT = Path(__file__).resolve().parents[2]


def test_standalone_clone_scripts_removed() -> None:
    """The package entrypoint is the only bulk-clone implementation."""
    assert not (MODULE_ROOT / "clone_repos.py").exists()
    assert not (MODULE_ROOT / "clone_script.py").exists()


def test_module_entrypoint_help() -> None:
    """``python -m geo_infer_git.main --help`` parses arguments and exits 0."""
    completed = subprocess.run(
        [sys.executable, "-m", "geo_infer_git.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0
    assert "--config-dir" in completed.stdout


def test_main_exits_nonzero_on_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KeyboardInterrupt during cloning exits with code 1."""

    def interrupt(config_dir: str | None = None) -> list[object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(git_main, "load_target_repos_config", interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "geo_infer_git.main",
            "--config-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-clone-users",
            "--no-generate-report",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        git_main.main()
    assert excinfo.value.code == 1
