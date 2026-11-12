"""Guards for the one path literal the renderer rewrites without anchoring.

``infrastructure/rendering/_pdf_combined_markdown.py`` rewrites the prefix
``output/figures/`` to ``../figures/`` with an unanchored ``str.replace`` over
the whole combined document, so the rule cannot tell an image target from a
backticked literal in a table cell.  It corrupted three prose pointers into
``../figures/``, a path that does not exist at the repository root, while the
row above each of them still read ``output/data/``.

The manuscript now writes those literals without the trailing slash, which is
outside the rule's match.  That is a real fix but a silent one: the next author
to write ``output/figures/`` in prose gets it rewritten again with a log line
that calls it a success.  These tests make that failure loud and local, at both
ends — the authored source, and the combined document the renderer produced.
"""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
from pathlib import Path

import pytest

# The exact prefix _pdf_combined_markdown.py replaces, and what it becomes.
REWRITTEN_PREFIX = "output/figures/"
REWRITE_RESULT = "../figures/"
# Pandoc image syntax: the only place the rewrite is wanted.
IMAGE_TARGET = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
PUBLISHED = ("[0-9][0-9]_*.md", "S[0-9][0-9]_*.md")


def _published_sources(repo_root: Path) -> list[Path]:
    manuscript = repo_root / "manuscript"
    found: list[Path] = []
    for pattern in PUBLISHED:
        found.extend(sorted(manuscript.glob(pattern)))
    assert found, "no published manuscript sections were found"
    return found


def _outside_image_targets(text: str, needle: str) -> list[str]:
    """Every line holding ``needle`` outside a Pandoc image target."""
    offenders: list[str] = []
    for line in text.split("\n"):
        if needle not in line:
            continue
        stripped = IMAGE_TARGET.sub("", line)
        if needle in stripped:
            offenders.append(line.strip())
    return offenders


class TestFigurePathLiterals:
    def test_no_published_section_writes_the_rewritten_prefix_in_prose(
        self, repo_root: Path
    ) -> None:
        offenders = {
            path.name: lines
            for path in _published_sources(repo_root)
            if (
                lines := _outside_image_targets(
                    path.read_text(encoding="utf-8"), REWRITTEN_PREFIX
                )
            )
        }
        assert not offenders, (
            f"{REWRITTEN_PREFIX!r} outside an image target is rewritten to "
            f"{REWRITE_RESULT!r} by the renderer, which is not a path in this "
            f"repository. Write it without the trailing slash. {offenders}"
        )

    def test_the_combined_document_keeps_the_rewrite_inside_image_targets(
        self, repo_root: Path
    ) -> None:
        combined = repo_root / "output" / "pdf" / "_combined_manuscript.md"
        if not combined.is_file():
            pytest.fail(
                "the combined document has not been rendered (skips become failures)"
            )
        offenders = _outside_image_targets(
            combined.read_text(encoding="utf-8"), REWRITE_RESULT
        )
        assert not offenders, (
            f"the renderer rewrote prose to {REWRITE_RESULT!r}: {offenders}"
        )


def test_default_render_receipt_binds_source_and_pdf_bytes(
    repo_root: Path, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = runpy.run_path(str(repo_root / "scripts/render_manuscript_pdf.py"))
    record = renderer["_record_default_render"]
    monkeypatch.setitem(record.__globals__, "REPO_ROOT", git_repo)
    (git_repo / "docs").mkdir()
    data = git_repo / "output/data"
    data.mkdir(parents=True)
    revision = subprocess.check_output(
        ["git", "-C", str(git_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "source_commit": revision[:7] + "-dirty",
        "source_hash": "measured-input-fingerprint",
        "dirty_file_count": 2,
    }
    (data / "research_manifest.json").write_text(json.dumps(manifest))
    pdf = git_repo / "output/GEO-INFER_combined.pdf"
    pdf.write_bytes(b"%PDF-1.4\nreceipt test bytes\n")
    record(pdf, revision)
    receipt_path = git_repo / "docs/default-render-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt == {
        "schema": "geo-infer-default-render/v1",
        "source_revision": revision,
        **manifest,
        "pdf_path": "output/GEO-INFER_combined.pdf",
        "pdf_bytes": len(pdf.read_bytes()),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "check-ignore",
            "--no-index",
            "--",
            "docs/default-render-receipt.json",
        ],
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 1 and not ignored.stdout
    pdf.write_bytes(pdf.read_bytes() + b"changed\n")
    record(pdf, revision)
    refreshed = json.loads(receipt_path.read_text())
    assert refreshed["pdf_sha256"] != receipt["pdf_sha256"]
    assert refreshed["pdf_bytes"] == pdf.stat().st_size
