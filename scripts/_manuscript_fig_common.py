#!/usr/bin/env python3
"""Shared conventions for the standalone manuscript figure scripts.

The scripts in this directory that are named ``manuscript_fig_*.py`` each
render one publication figure into ``output/figures/``.  They mirror the
geometry, raster density, palette, and save discipline of
``docs/manuscript/generate_research_artifacts.py`` — the builder that owns the
core figure set — so a figure from either source is typeset at the same
printed scale and a colour learned in one figure carries to every other
one.  This module holds what all of them share; the figure-specific
drawing code stays in each script.

Determinism contract: no wall-clock, no environment probing beyond the
repository checkout, fixed layout.  Running a script twice on the same
tree produces byte-identical PNGs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

# Root of the GEO-INFER checkout (this file lives in ``scripts/``).
ROOT = Path(__file__).resolve().parent.parent

# Raster density matching the generator: clear of the ~264 DPI floor in
# both render lanes.
FIGURE_DPI = 340
FIGURES_DIR = ROOT / "output" / "figures"

# Shared figure palette, copied verbatim from the generator so the accent
# a reader learns in one figure means the same thing in every other one.
SOURCE_COLOR = "#2f6f9f"
TEST_COLOR = "#d17a2f"
CATEGORY_COLOR = "#5b8e7d"
SURFACE_COLOR = "#6f5b9e"
SPINE_COLOR = "#1b3a5b"

# rc context the generator uses for every non-abstract figure.
RC_CONTEXT: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "font.size": 8,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#666666",
}


def import_matplotlib() -> tuple[Any, Any]:
    """Bind matplotlib to the Agg backend before pyplot exists."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def load_generator() -> ModuleType:
    """Import the artifact generator as a module by file path.

    The manuscript directory is not a package, and the figure scripts must
    read the *same* theme table and the *same* measurement code the
    published figures use — a private copy would drift.
    """
    path = ROOT / "docs" / "manuscript" / "generate_research_artifacts.py"
    spec = importlib.util.spec_from_file_location("geo_infer_artifacts", path)
    if spec is None or spec.loader is None:  # pragma: no cover - structural
        raise RuntimeError(f"cannot load generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def text_block() -> tuple[float, float]:
    """Return the generator's current printable text block, in inches."""
    generator = load_generator()
    return float(generator.TEXT_BLOCK_WIDTH_IN), float(generator.TEXT_BLOCK_HEIGHT_IN)


def max_figure_height() -> float:
    """Return the generator's tallest permitted figure, in inches."""
    return float(load_generator().MAX_FIGURE_HEIGHT_IN)


def measured_modules() -> list[tuple[str, int, int]]:
    """Return ``(name, test_files, source_files)`` per module, sorted by name.

    Test files are counted with the generator's own ``_test_files`` logic,
    restricted to one module so the landscape figure and the registry
    cannot disagree about what a test file is.
    """
    generator = load_generator()
    rows: list[tuple[str, int, int]] = []
    for module_path in generator._module_paths(ROOT):
        tests = tuple(
            path
            for path in generator._python_files(module_path / "tests")
            if path.name.startswith("test_")
        )
        sources = generator._python_files(module_path / "src")
        rows.append((module_path.name, len(tests), len(sources)))
    return sorted(rows)


def save_figure(fig: Any, filename: str) -> Path:
    """Write one figure into ``output/figures/`` and return its path.

    Mirrors the generator's save discipline: tight bounding box, the fixed
    raster density, descriptive PNG metadata — and a hard refusal to emit
    a figure larger than the printable text block, which the typesetter
    would scale down below the legibility floor.  The printable box is
    read from the live generator, so a render-lane repin flows into these
    figures without an edit here.
    """
    path = FIGURES_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        metadata={
            "Title": path.stem.replace("_", " ").title(),
            "Description": filename.replace("_", " "),
            "Source": "GEO-INFER repository",
        },
    )
    header = path.read_bytes()[:24]
    width = int.from_bytes(header[16:20], "big") / FIGURE_DPI
    height = int.from_bytes(header[20:24], "big") / FIGURE_DPI
    text_width, _text_height = text_block()
    tolerance = 1.02
    if width > text_width * tolerance or height > max_figure_height() * tolerance:
        raise ValueError(
            f"{filename} is {width:.2f}in x {height:.2f}in, larger than the "
            f"printable box {text_width:.2f}in x {max_figure_height():.2f}in"
        )
    return path
