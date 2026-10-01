#!/usr/bin/env python3
"""Render the GEO-INFER module theme map into output/figures/.

One publication figure: every measured ``GEO-INFER-*`` module drawn as a
chip, grouped into themed blocks.  The theme assignment is the generator's
``MODULE_THEMES`` — imported, not copied, so the map can never disagree
with the manuscript's module table — and the chip set is measured from the
checkout, so a module added without being themed fails the script instead
of dropping silently.

Usage: uv run --extra dev python scripts/manuscript_fig_module_theme_map.py
"""

from __future__ import annotations


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
    measured_modules,
    save_figure,
    text_block,
)

# Layout constants, in inches.  Fixed values, not measurements: the map is
# a classification diagram, so its geometry is authored.
CHIP_HEIGHT_IN = 0.24
CHIP_GAP_IN = 0.05
BLOCK_PAD_IN = 0.10
BLOCK_TITLE_IN = 0.22
BLOCK_GAP_IN = 0.14
THEME_COLOR_ALPHA = 0.14
THEME_COLORS = (
    SOURCE_COLOR,
    TEST_COLOR,
    CATEGORY_COLOR,
    SURFACE_COLOR,
    SPINE_COLOR,
    "#8a8a8a",
)


def _chip_width(text: str) -> float:
    return 0.078 * len(text) + 0.22


def _theme_blocks(
    module_themes: tuple[tuple[str, tuple[str, ...]], ...],
    fig_width_in: float,
) -> tuple[tuple[str, list[list[tuple[str, float]]], float], ...]:
    """Lay each theme's chips into rows that fit the printable width."""
    blocks: list[tuple[str, list[list[tuple[str, float]]], float]] = []
    for title, modules in module_themes:
        rows: list[list[tuple[str, float]]] = [[]]
        x = 2 * BLOCK_PAD_IN
        for name in modules:
            width = _chip_width(name.removeprefix("GEO-INFER-"))
            if x + width > fig_width_in - 2 * BLOCK_PAD_IN and rows[-1]:
                rows.append([])
                x = 2 * BLOCK_PAD_IN
            rows[-1].append((name, width))
            x += width + CHIP_GAP_IN
        rows = [row for row in rows if row]
        height = (
            BLOCK_TITLE_IN + len(rows) * (CHIP_HEIGHT_IN + CHIP_GAP_IN) + BLOCK_PAD_IN
        )
        blocks.append((title, rows, height))
    return tuple(blocks)


def main() -> int:
    generator = load_generator()
    module_themes = generator.MODULE_THEMES
    present = {name for name, _tests, _sources in measured_modules()}
    themed = {name for _title, modules in module_themes for name in modules}
    missing = sorted(present - themed)
    if missing:
        raise SystemExit(
            "measured modules absent from MODULE_THEMES: " + ", ".join(missing)
        )
    stale = sorted(themed - present)
    if stale:
        raise SystemExit("MODULE_THEMES names no live module: " + ", ".join(stale))

    _matplotlib, plt = import_matplotlib()
    fig_width_in = text_block()[0]
    blocks = _theme_blocks(module_themes, fig_width_in)
    module_count = sum(len(row) for _t, rows, _h in blocks for row in rows)

    # Flowing chip layout: within a theme block, chips wrap onto rows that
    # fit the printable width.  Block height follows the wrapped rows, so
    # the layout is a deterministic function of the module set.
    fig_height = (
        sum(height for _t, _rows, height in blocks)
        + (len(blocks) - 1) * BLOCK_GAP_IN
        + 2 * BLOCK_PAD_IN
    )
    fig, axis = plt.subplots(figsize=(fig_width_in, fig_height))
    axis.set_xlim(0, fig_width_in)
    axis.set_ylim(fig_height, 0)
    axis.set_axis_off()

    y = BLOCK_PAD_IN
    for index, (title, rows, height) in enumerate(blocks):
        color = THEME_COLORS[index % len(THEME_COLORS)]
        axis.add_patch(
            plt.Rectangle(
                (BLOCK_PAD_IN, y),
                fig_width_in - 2 * BLOCK_PAD_IN,
                height,
                facecolor=color,
                alpha=THEME_COLOR_ALPHA,
                edgecolor=color,
                linewidth=0.8,
            )
        )
        axis.text(
            BLOCK_PAD_IN + 0.06,
            y + BLOCK_TITLE_IN * 0.7,
            title,
            fontsize=8.5,
            fontweight="bold",
            color=SPINE_COLOR,
            va="center",
        )
        chip_y = y + BLOCK_TITLE_IN
        for row in rows:
            chip_x = 2 * BLOCK_PAD_IN
            for name, width in row:
                axis.add_patch(
                    plt.Rectangle(
                        (chip_x, chip_y),
                        width,
                        CHIP_HEIGHT_IN,
                        facecolor=color,
                        alpha=0.92,
                        edgecolor="none",
                    )
                )
                axis.text(
                    chip_x + width / 2,
                    chip_y + CHIP_HEIGHT_IN / 2,
                    name.removeprefix("GEO-INFER-"),
                    fontsize=7,
                    color="white",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
                chip_x += width + CHIP_GAP_IN
            chip_y += CHIP_HEIGHT_IN + CHIP_GAP_IN
        y += height + BLOCK_GAP_IN

    axis.set_title(
        f"GEO-INFER module theme map — {module_count} modules in {len(blocks)} themes",
        fontsize=10.5,
    )

    with plt.rc_context({**RC_CONTEXT, "axes.grid": False}):
        path = save_figure(fig, "module_theme_map.png")
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
