"""Actual isolated worker failures retain evidence without certifying a PDF."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from geo_infer_test.manuscript_rendering import render_template_pdf


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def commit_fixture(root: Path) -> str:
    git(root, "init")
    git(root, "config", "user.name", "Native failure fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    # A developer's global ignore rules must not hide worker cache writes.
    git(root, "config", "core.excludesFile", os.devnull)
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "native failure fixture")
    return git(root, "rev-parse", "HEAD")


def render_inputs(
    tmp_path: Path, config_source: str
) -> tuple[Path, Path, dict[str, bytes]]:
    """Create clean Git inputs; selected template code fails before rendering."""
    template = tmp_path / "template"
    directory = template / "infrastructure" / "rendering"
    directory.mkdir(parents=True)
    (directory / "config.py").write_text(config_source)
    for name in ["pdf_renderer.py", "_pdf_title_page.py"]:
        (directory / name).write_text("# Deliberately unreachable source fixture.\n")
    commit_fixture(template)
    repo = tmp_path / "geo-infer"
    repo.mkdir()
    (repo / ".gitignore").write_text("output/\n")
    (repo / "uv.lock").write_text("# Fixture lock bytes; no dependency installation.\n")
    inputs = {
        "output/pdf/_combined_manuscript.md": b"# Native failure fixture\n",
        "output/manuscript/config.yaml": (
            b"paper:\n  cover:\n    image: output/figures/cover.png\n"
            b"    alt: Explicit fixture cover input, not a rendered publication\n"
        ),
        "output/manuscript/preamble.md": b"# Preamble fixture\n",
        "output/manuscript/references.bib": b"% Empty bibliography fixture\n",
        "output/figures/figure_registry.json": b'{"figures": []}\n',
        # No image or PDF success is tested: the worker must fail before it
        # can import or execute the PDF renderer or inspect this fixture data.
        "output/figures/cover.png": b"admission-only cover bytes\n",
    }
    for name, data in inputs.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    commit_fixture(repo)
    inputs["uv.lock"] = (repo / "uv.lock").read_bytes()
    return repo, template, inputs


def failed_attempt(
    repo: Path, template: Path, inputs: dict[str, bytes]
) -> tuple[dict, Path]:
    attempts = list((repo / "output/pdf/template-attempts").iterdir())
    assert len(attempts) == 1
    attempt = attempts[0]
    receipt = json.loads((attempt / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["success"] is False
    assert receipt["source_revision"] == git(repo, "rev-parse", "HEAD")
    assert receipt["source_dirty"] == ""
    assert receipt["require_clean"] is True
    assert receipt["template"]["revision"] == git(template, "rev-parse", "HEAD")
    assert (
        receipt["log_sha256"]
        == hashlib.sha256((attempt / "render.log").read_bytes()).hexdigest()
    )
    assert receipt["elapsed_seconds"] >= 0
    assert "pdf_sha256" not in receipt
    assert not (attempt / "runtime.json").exists()
    assert not (repo / "output/pdf/GEO-INFER_combined.pdf").exists()
    for name, data in inputs.items():
        assert (repo / name).read_bytes() == data
        assert receipt["input_sha256"][name] == hashlib.sha256(data).hexdigest()
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert git(template, "status", "--porcelain", "--untracked-files=all") == ""
    assert not list(template.rglob("*.pyc"))
    return receipt, attempt


def test_actual_worker_import_failure_preserves_native_returncode_and_log(tmp_path):
    repo, template, inputs = render_inputs(
        tmp_path,
        "import sys\n"
        "print('native worker stdout survives', flush=True)\n"
        "print('native worker stderr survives', file=sys.stderr, flush=True)\n"
        "raise RuntimeError('intentional native import failure')\n",
    )
    with pytest.raises(RuntimeError, match=r"template renderer failed \(1\)"):
        render_template_pdf(
            repo, template, python_executable=sys.executable, timeout=30
        )
    receipt, attempt = failed_attempt(repo, template, inputs)
    assert receipt["returncode"] == 1
    assert receipt["process_evidence"]["target_returncode"] == 1
    assert receipt["process_evidence"]["failure_phase"] is None
    log = (attempt / "render.log").read_text()
    assert "native worker stdout survives" in log
    assert "native worker stderr survives" in log
    assert "RuntimeError: intentional native import failure" in log
    assert receipt["command"][0] == sys.executable
    assert receipt["command"][1] == "-I"


def test_actual_missing_interpreter_preserves_failed_launch_receipt(tmp_path):
    repo, template, inputs = render_inputs(
        tmp_path, "raise RuntimeError('not launched')\n"
    )
    missing = tmp_path / "absent-python"
    with pytest.raises(FileNotFoundError):
        render_template_pdf(repo, template, python_executable=missing)
    receipt, attempt = failed_attempt(repo, template, inputs)
    assert receipt.get("returncode") is None
    assert receipt.get("process_evidence") is None
    assert receipt["error"].startswith("FileNotFoundError:")
    assert receipt["command"][0] == str(missing)
    assert (attempt / "render.log").is_file()


def test_actual_worker_deadline_preserves_captured_output_and_failure(tmp_path):
    repo, template, inputs = render_inputs(
        tmp_path,
        "import time\n"
        "print('native worker deadline waiting', flush=True)\n"
        "while True:\n    time.sleep(1)\n",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        render_template_pdf(repo, template, python_executable=sys.executable, timeout=3)
    receipt, attempt = failed_attempt(repo, template, inputs)
    assert receipt.get("returncode") is None
    assert receipt["error"].startswith("TimeoutExpired:")
    assert receipt["process_evidence"]["failure_phase"] in {"target", "census"}
    assert receipt["process_evidence"]["target_returncode"] is None
    assert receipt["process_evidence"]["cleanup_seconds"] >= 0
    assert "native worker deadline waiting" in (attempt / "render.log").read_text()
