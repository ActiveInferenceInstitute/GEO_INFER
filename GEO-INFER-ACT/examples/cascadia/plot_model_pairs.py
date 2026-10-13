# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Thin entry point for integrity-checked paired-model SVG comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from geo_infer_act.runners.model_pair_plot import plot_model_pairs
from geo_infer_act.runners.model_pair_explorer import render_pair_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--html", action="store_true", help="Generate offline interactive HTML"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check existing HTML without writing"
    )
    arguments = parser.parse_args()
    if arguments.check and not arguments.html:
        parser.error("--check requires --html")
    print(
        render_pair_comparison(
            arguments.editions, arguments.output, check=arguments.check
        )
        if arguments.html
        else plot_model_pairs(arguments.editions, arguments.output)
    )


if __name__ == "__main__":
    main()
