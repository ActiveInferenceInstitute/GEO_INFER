# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run python GEO-INFER-ACT/examples/cascadia/study_explorer.py STUDY OUTPUT.html
"""Thin offline visualization entry point; qualification precedes file creation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from geo_infer_act.runners.study_visualization import project_study
from validate_study import validate_study


def render_explorer(study: Path, output: Path) -> Path:
    problems = validate_study(study)
    if problems:
        message = "study qualification failed: " + "; ".join(problems)
        raise ValueError(message)
    view = project_study(study)
    data = view.model_dump_json().replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    content = template.replace("__STUDY_DATA__", data)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(render_explorer(args.study, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
