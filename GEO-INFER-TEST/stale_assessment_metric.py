#!/usr/bin/env python3
"""DOCS-03 assessment-artifact banner metric for GEO-INFER.

The DOCS-03 ledger row requires every assessment artifact to carry a
visible historical-artifact banner (or be regenerated or deleted), so a
stale point-in-time snapshot cannot read as live guidance.  This script is the
benchmark instrument: it walks the assessment-artifact locations, counts
the tracked ``.md``/``.json`` files missing their banner marker, and
reports the count.

Generated ``README.md``/``AGENTS.md`` files inside the assessment
directories are canonical generator output (current, not stale) and are
excluded, as are untracked files.

Metrics are printed one per line as ``METRIC name=value``; diagnostics as
``ASI key=value``.  The harness exits non-zero only when it cannot
measure; the measured count is data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BANNER_MD = "Historical artifact"
BANNER_JSON = "_historical_artifact"
GENERATED_NAMES = frozenset({"README.md", "AGENTS.md"})
TARGETS = (
    "GEO-INFER-INTRA/assessment_results",
    "GEO-INFER-EXAMPLES/assessment_results",
)
EXTRA_FILES = ("GEO-INFER-EXAMPLES/docs/COMPREHENSIVE_DOCUMENTATION_ANALYSIS.md",)


def _fail(message: str) -> None:
    """Print a harness failure and exit non-zero."""
    print(f"harness failure: {message}", file=sys.stderr)
    raise SystemExit(1)


def _tracked(relative: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _missing_banner(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    if path.suffix == ".json":
        return BANNER_JSON not in head
    return BANNER_MD not in head


def _artifact_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS:
        directory = REPO_ROOT / target
        if not directory.is_dir():
            _fail(f"missing assessment directory: {directory}")
        for name in sorted(os.listdir(directory)):
            path = directory / name
            if path.suffix not in (".md", ".json") or name in GENERATED_NAMES:
                continue
            files.append(path)
    for relative in EXTRA_FILES:
        path = REPO_ROOT / relative
        if path.is_file():
            files.append(path)
    return [path for path in files if _tracked(path.relative_to(REPO_ROOT).as_posix())]


def main() -> int:
    # Retired snapshots are deleted rather than bannered, so an empty
    # inventory is the fully-resolved state, not a measurement failure.
    artifacts = _artifact_files()
    stale = [path for path in artifacts if _missing_banner(path)]

    print(f"METRIC stale_assessment_artifacts={len(stale)}")
    print(f"ASI assessment_artifacts_total={len(artifacts)}")
    for path in sorted(stale):
        print(f"ASI stale_artifact={path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
