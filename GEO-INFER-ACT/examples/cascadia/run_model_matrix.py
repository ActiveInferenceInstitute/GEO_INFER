# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: uv run python GEO-INFER-ACT/examples/cascadia/run_model_matrix.py --help
"""Thin CLI for portable, bounded agent/process experiment matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

from geo_infer_act.runners.model_matrix import check_model_matrix, create_model_matrix
from geo_infer_act.runners.model_matrix_design import load_matrix_design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--replay", action="store_true")
    arguments = parser.parse_args()
    if arguments.check is not None:
        if any(
            value is not None
            for value in (arguments.design, arguments.models_root, arguments.output)
        ):
            parser.error("--check excludes design, model and output arguments")
        print(
            check_model_matrix(
                arguments.check, replay=arguments.replay
            ).model_dump_json()
        )
        return
    if arguments.replay or any(
        value is None
        for value in (arguments.design, arguments.models_root, arguments.output)
    ):
        parser.error("execution requires --design, --models-root and --output")
    print(
        create_model_matrix(
            load_matrix_design(arguments.design),
            arguments.models_root,
            arguments.output,
        )
    )


if __name__ == "__main__":
    main()
