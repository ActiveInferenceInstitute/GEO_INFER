"""Coverage-gate tests for the generated README/AGENTS signposts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REWRITER_PATH = REPO_ROOT / "GEO-INFER-TEST" / "rewrite_readme_agents.py"


def load_rewriter_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_rewrite_readme_agents_coverage", REWRITER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )


def init_repo(path: Path) -> None:
    git(path, "init", "-q")
    git(path, "config", "core.fsmonitor", "false")


def test_gate_fails_for_tracked_directory_without_signposts(tmp_path):
    init_repo(tmp_path)
    directory = tmp_path / "src" / "widget"
    directory.mkdir(parents=True)
    (directory / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    rewriter = load_rewriter_module()
    with pytest.raises(rewriter.CoverageError, match="src/widget"):
        rewriter.check_doc_coverage(tmp_path)


def test_check_command_fails_for_tracked_directory_without_signposts(
    tmp_path, monkeypatch, capsys
):
    init_repo(tmp_path)
    directory = tmp_path / "src" / "widget"
    directory.mkdir(parents=True)
    (directory / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    rewriter = load_rewriter_module()
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", None)
    monkeypatch.setattr(rewriter, "discover_modules", dict)
    monkeypatch.setattr(sys, "argv", [str(REWRITER_PATH)])
    assert rewriter.main() == 0
    monkeypatch.setattr(sys, "argv", [str(REWRITER_PATH), "--check"])
    assert rewriter.main() == 1
    assert "src/widget" in capsys.readouterr().err


def test_gate_passes_when_tracked_directory_has_both_signposts(tmp_path):
    init_repo(tmp_path)
    directory = tmp_path / "src" / "widget"
    directory.mkdir(parents=True)
    (directory / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (directory / "README.md").write_text("# Widget\n", encoding="utf-8")
    (directory / "AGENTS.md").write_text("# Widget agents\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    rewriter = load_rewriter_module()
    assert rewriter.coverage_gaps(tmp_path) == []


def test_gate_rejects_empty_tracked_set(tmp_path):
    init_repo(tmp_path)
    rewriter = load_rewriter_module()
    with pytest.raises(rewriter.CoverageError, match="no tracked files"):
        rewriter.tracked_directories(tmp_path)


def test_gate_rejects_git_ls_files_failure(tmp_path, monkeypatch):
    rewriter = load_rewriter_module()

    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=128,
            stdout=b"",
            stderr=b"fatal: not a git repository",
        )

    monkeypatch.setattr(rewriter.subprocess, "run", fail)
    with pytest.raises(rewriter.CoverageError, match="git ls-files failed"):
        rewriter.tracked_directories(tmp_path)


def test_gate_skips_rule_based_classes(tmp_path):
    init_repo(tmp_path)
    files = {
        "output/result.json": "{}\n",
        "previews/page.md": "# preview\n",
        "cache/blob.bin": "x\n",
        "raw/input.bin": "x\n",
        "data/payload.csv": "a,b\n",
        ".aii/config.yaml": "version: 1\n",
        "manuscript/sections/ch1.md": "# Chapter\n",
        "locations/california/models/model.bin": "x\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(tmp_path, "add", ".")
    rewriter = load_rewriter_module()
    assert rewriter.coverage_gaps(tmp_path) == []


def test_gate_skips_directories_ignored_by_git(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Root agents\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("config/\n", encoding="utf-8")
    path = tmp_path / "src" / "config" / "keep.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("value: 1\n", encoding="utf-8")
    git(tmp_path, "add", "README.md", "AGENTS.md", ".gitignore")
    git(tmp_path, "add", "-f", "src/config/keep.yaml")
    rewriter = load_rewriter_module()
    assert rewriter.coverage_gaps(tmp_path) == []


def test_repository_coverage_is_current():
    rewriter = load_rewriter_module()
    assert rewriter.coverage_gaps(REPO_ROOT) == []
