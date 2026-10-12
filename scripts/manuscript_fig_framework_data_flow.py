#!/usr/bin/env python3
"""Render the GEO-INFER framework data-flow figure into output/figures/.

One publication figure: how geospatial data flows through the module
layers into the inference models and out as verified claims.  The layers
are the generator's theme classification (imported, not copied); which
modules sit in each layer is therefore a shared editorial fact, while the
arrows are the authored data-flow story the figure exists to tell.

Usage: uv run python scripts/manuscript_fig_framework_data_flow.py
"""

from __future__ import annotations

import textwrap

# Run as ``python scripts/<name>.py``: the script directory is sys.path[0], so
# the sibling helper module imports directly.
from _manuscript_fig_common import (
    CATEGORY_COLOR,
    RC_CONTEXT,
    SOURCE_COLOR,
    SPINE_COLOR,
    SURFACE_COLOR,
    TEST_COLOR,
    import_matplotlib,
    load_generator,
    save_figure,
    text_block,
)

# Authored data-flow story.  Each stage names the theme (by generator
# title prefix) whose modules carry that stage, plus the stages it
# receives from and passes to.  Stages are drawn bottom-up.
STAGES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("Geospatial data", "SPACE · PLACE · TIME · DATA · IOT · API", ()),
    (
        "Domain science layers",
        "CLIMATE · WATER · FOREST · MARINE · ENERGY · TRANSPORT · HEALTH · CIV",
        (0,),
    ),
    (
        "Inference and learning",
        "BAYES · SPM · SIM · COG · ACT · MATH · AG · AI",
        (0, 1),
    ),
    ("Agents and orchestration", "AGENT · ANT · OPS · COMMS · APP", (2,)),
    (
        "Governance and risk",
        "RISK · INSURANCE · METAGOV · NORMS · ECON · SEC · REQ · PEP · ORG",
        (1, 3),
    ),
    ("Verified claims", "TEST · LOG · GIT · INTRA · EXAMPLES", (1, 2, 4)),
)

STAGE_COLORS = (
    SOURCE_COLOR,
    CATEGORY_COLOR,
    SURFACE_COLOR,
    TEST_COLOR,
    SPINE_COLOR,
    "#8a8a8a",
)
# Authored geometry, in inches.  The figure width follows the live
# generator's text block; the stage column is inset from both edges so
# the left rail (skip flows and the re-observation loop) has a clear lane.
STAGE_LEFT_IN = 1.55
STAGE_RIGHT_MARGIN_IN = 0.40
STAGE_HEIGHT_IN = 0.62
STAGE_GAP_IN = 0.28
BOTTOM_MARGIN_IN = 0.16
LABEL_ARROW_IN = 0.08
RAIL_OFFSET_IN = 0.30
FEEDBACK_OFFSET_IN = 0.58
FEEDBACK_LABEL_OFFSET_IN = 0.72
FIG_HEIGHT_IN = 5.6


def _stage_top(stage_index: int, stage_y: list[float]) -> float:
    return stage_y[stage_index] + STAGE_HEIGHT_IN


def main() -> int:
    load_generator()  # fail fast if the shared generator is unreadable
    fig_width_in = text_block()[0]
    stage_right_in = fig_width_in - STAGE_RIGHT_MARGIN_IN
    _matplotlib, plt = import_matplotlib()

    fig, axis = plt.subplots(figsize=(fig_width_in, FIG_HEIGHT_IN))
    axis.set_xlim(0, fig_width_in)
    axis.set_ylim(0, FIG_HEIGHT_IN)
    axis.set_axis_off()

    # Deterministic bottom-up stacking.
    stage_y: list[float] = []
    y = BOTTOM_MARGIN_IN
    for _title, _modules, _feeds in STAGES:
        stage_y.append(y)
        y += STAGE_HEIGHT_IN + STAGE_GAP_IN

    # Feedback arrow from verified claims back to the data stage, drawn on
    # the far-left lane, clear of the stage boxes.
    axis.add_patch(
        plt.Arrow(
            STAGE_LEFT_IN - FEEDBACK_OFFSET_IN,
            stage_y[0] + STAGE_HEIGHT_IN / 2,
            0,
            stage_y[5] + STAGE_HEIGHT_IN / 2 - (stage_y[0] + STAGE_HEIGHT_IN / 2),
            width=0.05,
            color="#8a8a8a",
            alpha=0.75,
        )
    )
    axis.text(
        STAGE_LEFT_IN - FEEDBACK_LABEL_OFFSET_IN,
        (stage_y[0] + stage_y[5]) / 2 + STAGE_HEIGHT_IN / 2,
        "re-observation",
        fontsize=7,
        color="#8a8a8a",
        rotation=90,
        va="center",
        ha="right",
    )

    for index, (title, modules, feeds) in enumerate(STAGES):
        color = STAGE_COLORS[index % len(STAGE_COLORS)]
        axis.add_patch(
            plt.Rectangle(
                (STAGE_LEFT_IN, stage_y[index]),
                stage_right_in - STAGE_LEFT_IN,
                STAGE_HEIGHT_IN,
                facecolor=color,
                alpha=0.16,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        axis.text(
            STAGE_LEFT_IN + 0.10,
            stage_y[index] + STAGE_HEIGHT_IN * 0.70,
            title,
            fontsize=8.5,
            fontweight="bold",
            color=SPINE_COLOR,
            va="center",
        )
        axis.text(
            STAGE_LEFT_IN + 0.10,
            stage_y[index] + STAGE_HEIGHT_IN * 0.32,
            "\n".join(textwrap.wrap(modules, width=76)),
            fontsize=6.0,
            color="#444444",
            va="center",
        )
        for source in feeds:
            start_y = _stage_top(source, stage_y)
            stop_y = stage_y[index]
            if index - source == 1:
                # Adjacent stages: a straight arrow up the centre line.
                x_mid = 0.5 * (STAGE_LEFT_IN + stage_right_in)
                axis.annotate(
                    "",
                    xy=(x_mid, stop_y),
                    xytext=(x_mid, start_y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "linewidth": 1.1,
                        "alpha": 0.85,
                    },
                )
            else:
                # Skipped stages: route around the left of the boxes —
                # up the rail, then into the target's left edge — so no
                # arrow ever crosses a box's text.
                rail_x = STAGE_LEFT_IN - RAIL_OFFSET_IN
                axis.plot(
                    [rail_x, rail_x],
                    [start_y + LABEL_ARROW_IN, stop_y - LABEL_ARROW_IN],
                    color=color,
                    linewidth=1.1,
                    alpha=0.85,
                    solid_capstyle="round",
                )
                axis.annotate(
                    "",
                    xy=(STAGE_LEFT_IN, stop_y + STAGE_HEIGHT_IN * 0.5),
                    xytext=(rail_x, stop_y - LABEL_ARROW_IN),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "linewidth": 1.1,
                        "alpha": 0.85,
                        "connectionstyle": "arc3,rad=0.35",
                    },
                )

    axis.set_title(
        "GEO-INFER framework data flow — geo data to verified claims",
        fontsize=10.5,
    )

    with plt.rc_context({**RC_CONTEXT, "axes.grid": False}):
        path = save_figure(fig, "framework_data_flow.png")
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
