# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run python GEO-INFER-ACT/examples/cascadia/study_diagnostics.py STUDY OUTPUT.json
"""Qualify a retained study and export observation-time numerical diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from geo_infer_act.runners.study_diagnostics import diagnostic_document
from geo_infer_act.runners.study_visualization import project_study
from validate_study import validate_study


def export_diagnostics(study: Path, output: Path) -> Path:
    """Reject custody problems before creating a new report."""
    problems = validate_study(study)
    if problems:
        message = "study qualification failed: " + "; ".join(problems)
        raise ValueError(message)
    report = diagnostic_document(project_study(study))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Run the package-owned projection in the maintained ACT environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(export_diagnostics(args.study, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
