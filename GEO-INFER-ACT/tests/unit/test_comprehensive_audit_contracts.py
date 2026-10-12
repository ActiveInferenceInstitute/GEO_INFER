"""Truthful documentation audit receipts and real renderer process cleanup."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from geo_infer_test.process import run_process


@pytest.fixture
def docs_audit(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "verify_comprehensive.py"
    spec = importlib.util.spec_from_file_location("act_documentation_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ACT_ROOT = tmp_path / "act"
    module.REPO_ROOT = tmp_path
    module.ACT_ROOT.mkdir()
    (module.ACT_ROOT / "guide.md").write_text(
        "```mermaid\nflowchart LR\nA --> B\n```\n"
    )
    monkeypatch.setattr(module.shutil, "which", lambda _: sys.executable)
    return module


def test_missing_renderer_fails_with_retained_inventory(
    docs_audit, tmp_path, monkeypatch
):
    monkeypatch.setattr(docs_audit.shutil, "which", lambda _: None)
    output = tmp_path / "audit"
    with pytest.raises(AssertionError, match="mmdc not found"):
        docs_audit.audit_docs_and_mermaid(output)
    result = json.loads((output / "docs_mermaid_audit.json").read_text())
    assert result["mermaid_block_count"] == 1
    assert result["mermaid_results"][0]["status"] == "failed"


def test_renderer_cannot_pass_with_a_stale_svg(docs_audit, tmp_path, monkeypatch):
    output = tmp_path / "audit"
    rendered = output / "mermaid" / "guide_md_1.svg"
    rendered.parent.mkdir(parents=True)
    rendered.write_text("<svg>stale</svg>")
    monkeypatch.setattr(
        "geo_infer_test.process.run_process",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "old", ""),
    )
    with pytest.raises(AssertionError, match="missing or empty SVG"):
        docs_audit.audit_docs_and_mermaid(output)
    assert not rendered.exists()
    assert (output / "mermaid" / "guide_md_1.stdout.log").read_text() == "old"


def test_renderer_timeout_reaps_real_descendants_and_retains_output(
    docs_audit, tmp_path, monkeypatch
):
    renderer = tmp_path / "renderer.py"
    renderer.write_text(
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        "print(child.pid,flush=True)\n"
        "print('renderer diagnostic',file=sys.stderr,flush=True)\n"
        "time.sleep(30)\n"
    )

    def actual_renderer(command, **kwargs):
        return run_process(
            [sys.executable, str(renderer), *command[1:]],
            timeout=0.8,
            cwd=kwargs["cwd"],
        )

    monkeypatch.setattr("geo_infer_test.process.run_process", actual_renderer)
    output = tmp_path / "audit"
    with pytest.raises(AssertionError, match="TimeoutExpired"):
        docs_audit.audit_docs_and_mermaid(output)
    stdout = (output / "mermaid" / "guide_md_1.stdout.log").read_text()
    assert (
        "renderer diagnostic"
        in (output / "mermaid" / "guide_md_1.stderr.log").read_text()
    )
    child_pid = int(stdout.strip())
    if psutil.pid_exists(child_pid):
        assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    result = json.loads((output / "docs_mermaid_audit.json").read_text())
    entry = result["mermaid_results"][0]
    assert entry["status"] == "failed"
    assert entry["error"].startswith("TimeoutExpired:")
