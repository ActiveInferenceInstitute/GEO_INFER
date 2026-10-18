"""Explicit, source-bound rendering with a trusted local docxology/template checkout.

The optional renderer runs in its own interpreter. It does not install template
dependencies, modify the supplied checkout, or substitute for manuscript
verification. The caller first generates the resolved manuscript and combined
Markdown with GEO-INFER's producer and section-order contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from geo_infer_test.process import run_process


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_source_identity(
    template_root: str | Path,
    *,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Identify a clean, trusted template Git checkout before executing its code."""
    root = Path(template_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("template_root must be a Git checkout directory")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("template_root must name the checkout root")
    revision = _git(root, "rev-parse", "HEAD")
    if expected_revision is not None and (
        re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None
        or revision != expected_revision
    ):
        raise ValueError("template revision does not match the requested full commit")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            "template checkout must be clean; preserve active work in another checkout"
        )
    required = (
        "infrastructure/rendering/pdf_renderer.py",
        "infrastructure/rendering/config.py",
        "infrastructure/rendering/_pdf_title_page.py",
    )
    for name in required:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular template renderer source: {name}")
    sources, gitlinks = {}, {}
    for entry in _git(root, "ls-tree", "-rz", "HEAD", "infrastructure").split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, kind, object_id = metadata.split()
        if mode == "160000" and kind == "commit":
            # Optional submodules are recorded as Git links, never mistaken
            # for regular renderer files or claimed as loaded source.
            gitlinks[name] = object_id
            continue
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"template source is missing or symlinked: {name}")
        sources[name] = _sha(path)
    return {
        "revision": revision,
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "source_sha256": sources,
        "gitlinks": gitlinks,
    }


def render_template_pdf(
    repository_root: str | Path,
    template_root: str | Path,
    *,
    python_executable: str | Path | None = None,
    expected_revision: str | None = None,
    require_clean: bool = True,
    timeout: float = 600,
) -> dict[str, Any]:
    """Render prepared GEO-INFER inputs using the actual template PDFRenderer.

    The template dependencies belong to ``python_executable`` (default: the
    caller's interpreter). Every attempt retains its log and receipt. Failed,
    interrupted, or source-changing attempts cannot be recorded as successful.
    ``require_clean=False`` explicitly permits a labelled development render.
    This is execution of trusted local code, not a rendering sandbox.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= 3600
    ):
        raise ValueError(
            "timeout must be a finite positive number no greater than 3600 seconds"
        )
    if not isinstance(require_clean, bool):
        raise ValueError("require_clean must be a boolean")
    repo = Path(repository_root).resolve(strict=True)
    template = Path(template_root).resolve(strict=True)
    identity = template_source_identity(template, expected_revision=expected_revision)
    revision, dirty = (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "status", "--porcelain"),
    )
    if require_clean and dirty:
        raise ValueError("publication rendering requires a clean GEO-INFER checkout")
    required = [
        "output/pdf/_combined_manuscript.md",
        "output/manuscript/config.yaml",
        "output/manuscript/preamble.md",
        "output/manuscript/references.bib",
        "output/figures/figure_registry.json",
        "uv.lock",
    ]
    registry = json.loads((repo / "output/figures/figure_registry.json").read_text())
    for figure in registry["figures"]:
        # The producer owns the registry; require its referenced PNGs as inputs.
        required.append("output/figures/" + figure["filename"])
    configuration = yaml.safe_load((repo / "output/manuscript/config.yaml").read_text())
    cover = configuration["paper"]["cover"]
    if not isinstance(cover.get("alt"), str) or not cover["alt"].strip():
        raise ValueError("the configured cover requires informative alternative text")
    required.append(cover["image"])
    inputs = {}
    for name in required:
        path = repo / name
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(repo)
        ):
            raise ValueError(f"missing or unsafe render input: {name}")
        inputs[name] = _sha(path)
    attempt = repo / "output/pdf/template-attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).with_name("_template_worker.py")
    command = [
        str(python_executable or sys.executable),
        "-I",
        str(worker),
        str(template),
        str(repo),
        str(attempt / "runtime.json"),
    ]
    receipt: dict[str, Any] = {
        "schema": "geo-infer-template-render/v1",
        "status": "failed",
        "success": False,
        "source_revision": revision,
        "source_dirty": dirty,
        "require_clean": require_clean,
        "template": identity,
        "input_sha256": inputs,
        "command": command,
        "worker_sha256": _sha(worker),
        "timeout_seconds": timeout,
        "returncode": None,
    }
    started = time.monotonic()
    try:
        result = run_process(command, cwd=repo, timeout=timeout)
        (attempt / "render.log").write_text(result.stdout + "\n" + result.stderr)
        receipt["process_evidence"] = result.process_evidence
        receipt["returncode"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(
                f"template renderer failed ({result.returncode}); see {attempt / 'render.log'}"
            )
        if (
            template_source_identity(template, expected_revision=identity["revision"])
            != identity
        ):
            raise RuntimeError("template source changed during rendering")
        if (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain")) != (
            revision,
            dirty,
        ):
            raise RuntimeError("GEO-INFER source changed during rendering")
        if any(
            _sha(repo / name) != value
            for name, value in inputs.items()
            if name != "output/pdf/_combined_manuscript.md"
        ):
            raise RuntimeError("render inputs changed during rendering")
        if (
            _sha(repo / "output/pdf/_template_input.md")
            != inputs["output/pdf/_combined_manuscript.md"]
        ):
            raise RuntimeError("template did not retain the exact combined input")
        pdf = repo / "output/pdf/GEO-INFER_combined.pdf"
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("template renderer produced no PDF")
        receipt.update(
            status="passed",
            success=True,
            pdf_sha256=_sha(pdf),
            pdf_bytes=pdf.stat().st_size,
            runtime=json.loads((attempt / "runtime.json").read_text()),
            processed_markdown_sha256=_sha(repo / "output/pdf/_combined_manuscript.md"),
            cover_sha256=inputs[cover["image"]],
            cover_alt=cover["alt"],
        )
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["process_evidence"] = getattr(
            error, "process_evidence", receipt.get("process_evidence")
        )
        observed = (receipt["process_evidence"] or {}).get("target_returncode")
        if receipt["returncode"] is None and observed is not None:
            receipt["returncode"] = observed
        if not (attempt / "render.log").exists():
            (attempt / "render.log").write_text(
                str(getattr(error, "output", "") or "")
                + "\n"
                + str(getattr(error, "stderr", "") or "")
            )
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["log_sha256"] = (
            _sha(attempt / "render.log") if (attempt / "render.log").is_file() else None
        )
        (attempt / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "receipt": str(attempt / "receipt.json")}
