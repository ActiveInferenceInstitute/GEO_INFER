"""Plot retained paired-model scores after checking each edition's integrity."""

from __future__ import annotations

from io import StringIO
from collections.abc import Sequence
from pathlib import Path

from geo_infer_act.runners.model_pair_explorer import load_pair_comparison


def plot_model_pairs(editions: Sequence[Path], output: Path) -> Path:
    """Create an SVG with every selected seed and the recorded physical clock."""
    import matplotlib.pyplot as plt

    view = load_pair_comparison(editions)
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "geo-infer-act/model-pair/1",
        }
    ):
        figure, axes = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
        )
        try:
            for index, edition in enumerate(view.editions):
                for run in edition.runs:
                    label = f"{edition.name.replace('_', ' ')}; seed {run.seed}"
                    for axis, values in (
                        (
                            axes[0],
                            [frame.posterior_entropy_nats for frame in run.frames],
                        ),
                        (axes[1], [frame.brier_score for frame in run.frames]),
                    ):
                        axis.plot(
                            view.hours,
                            values,
                            label=label,
                            color=f"C{index % 10}",
                            linestyle="-" if run.seed == view.seeds[0] else "--",
                            alpha=0.85,
                        )
            axes[0].set_ylabel("Posterior entropy (nats)")
            axes[1].set_ylabel("Brier score (full carrier)")
            axes[1].set_xlabel("Elapsed physical hours")
            for axis in axes:
                axis.grid(alpha=0.2)
                axis.set_ylim(bottom=0)
            axes[0].legend(fontsize=8, loc="upper right")
            figure.suptitle(
                "Recorded GNN agent/process comparisons; descriptive simulation"
            )
            with StringIO() as stream:
                figure.savefig(stream, format="svg", metadata={"Date": None})
                rendered = (
                    "\n".join(line.rstrip() for line in stream.getvalue().splitlines())
                    + "\n"
                )
            with output.open("x", encoding="utf-8", newline="\n") as destination:
                destination.write(rendered)
        finally:
            plt.close(figure)
    return output
