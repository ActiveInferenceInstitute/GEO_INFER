#!/usr/bin/env python3
"""Render the GEO-INFER verification landscape into output/figures/.

One publication figure: test-file counts per module as a horizontal
lollipop chart, split across two panels so the forty-plus module rows
stay legible without claiming a float page.  Counts are measured with the
generator's own test-file definition, so the landscape cannot disagree
with the published evidence inventory.

Usage: uv run --extra dev python scripts/manuscript_fig_verification_landscape.py
"""

from __future__ import annotations


# Run as ``python scripts/<name>.py``: the script directory is sys.path[0], so
# the sibling helper module imports directly.
from _manuscript_fig_common import (
    RC_CONTEXT,
    TEST_COLOR,
    import_matplotlib,
    max_figure_height,
    measured_modules,
    save_figure,
    text_block,
)

ROW_HEIGHT_IN = 0.155
CHROME_IN = 0.95
BAR_BASELINE_ALPHA = 0.25


def main() -> int:
    rows = measured_modules()
    # Most-tested modules first, name as tiebreak — the same ordering the
    # evidence inventory uses for its source counts.
    rows.sort(key=lambda item: (-item[1], item[0]))

    fig_width_in = text_block()[0]
    _matplotlib, plt = import_matplotlib()
    split = (len(rows) + 1) // 2
    panels = ((0, split), (split, len(rows)))
    fig_height = min(
        max_figure_height(),
        max(3.0, CHROME_IN + split * ROW_HEIGHT_IN),
    )

    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height))
    limit = max(count for _name, count, _src in rows) * 1.12 or 1
    for axis, (start, stop) in zip(axes, panels, strict=True):
        panel = rows[start:stop]
        labels = [name.removeprefix("GEO-INFER-") for name, _c, _s in panel]
        counts = [count for _name, count, _s in panel]
        y = list(range(len(panel)))
        axis.hlines(
            y,
            0,
            counts,
            color=TEST_COLOR,
            alpha=BAR_BASELINE_ALPHA,
            linewidth=1.2,
        )
        axis.plot(
            counts,
            y,
            "o",
            color=TEST_COLOR,
            markersize=3.4,
            alpha=0.92,
        )
        for position, count in zip(y, counts, strict=True):
            axis.text(
                count + limit * 0.02,
                position,
                str(count),
                fontsize=6.2,
                color="#222222",
                va="center",
            )
        axis.set_yticks(y, labels, fontsize=6.4)
        axis.set_ylim(len(panel) - 0.5, -0.5)
        axis.set_xlim(0, limit)
        axis.tick_params(axis="x", labelsize=7)
        axis.set_title(f"Modules {start + 1}–{stop} of {len(rows)}", fontsize=8.5)
        axis.set_xlabel("Test files (test_*.py)")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Module", fontsize=8)
    fig.suptitle(
        "Verification landscape — per-module test-file coverage",
        fontsize=10.5,
    )
    fig.tight_layout()

    with plt.rc_context(RC_CONTEXT):
        path = save_figure(fig, "verification_landscape.png")
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
