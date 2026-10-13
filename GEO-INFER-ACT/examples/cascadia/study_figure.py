# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run python GEO-INFER-ACT/examples/cascadia/study_figure.py OUTCOMES OUTPUT
"""Seed-level arrival and censoring figure; no successful-run-only averages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
from pydantic import Field, TypeAdapter

from study_contract import SeedOutcome


def save_study_figure(path: Path, outcomes: tuple[SeedOutcome, ...]) -> Path:
    """Show each observed arrival or its censoring horizon on the same axis."""
    conditions = tuple(dict.fromkeys(item.condition for item in outcomes))
    with plt.rc_context({"font.size": 10, "figure.dpi": 150}):
        figure, axes = plt.subplots(
            len(conditions), 1, figsize=(8, 2.5 * len(conditions)), squeeze=False
        )
        for ax, condition in zip(axes[:, 0], conditions):
            selected = [item for item in outcomes if item.condition == condition]
            for item in selected:
                step = item.measures.reached_home_step
                arrived = step is not None
                x = step if step is not None else item.measures.steps - 1
                ax.scatter(
                    x,
                    item.process_seed,
                    marker="o" if arrived else ">",
                    color="#176c79" if arrived else "#ab4d20",
                )
            ax.axvline(
                selected[0].measures.home_lower_bound_steps,
                color="#777777",
                linestyle="--",
                label=f"graph lower bound ({selected[0].measures.home_lower_bound_steps} steps)",
            )
            ax.set_title(condition.value.replace("_", " "))
            ax.set_yticks([item.process_seed for item in selected])
            ax.set_ylabel("process seed")
            ax.set_xlabel("first home step; > marks no arrival by last recorded state")
            ax.set_xlim(-1, max(item.measures.steps for item in selected) + 1)
            ax.grid(axis="x", alpha=0.2)
            ax.legend(loc="upper left", frameon=False)
        figure.suptitle(
            "Illustrative model sensitivity: circles arrive, triangles are right-censored"
        )
        figure.tight_layout()
        try:
            figure.savefig(path)
            if path.suffix == ".svg":
                text = path.read_text(encoding="utf-8")
                path.write_text(
                    "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
                    encoding="utf-8",
                )
        finally:
            plt.close(figure)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    outcomes = TypeAdapter[tuple[SeedOutcome, ...]](
        Annotated[tuple[SeedOutcome, ...], Field(min_length=1, max_length=192)]
    ).validate_json(args.outcomes.read_bytes())
    save_study_figure(args.output, outcomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
