#!/usr/bin/env python3
"""GEO-INFER-ART module orchestrator.

Runs one documented end-to-end ART operation on synthetic data: build a
geospatial color scheme with ``ColorPalette.from_color_theory`` (forest-green
blend it with the predefined ``forest`` palette, apply a brightness
adjustment, and generate three deterministic procedural pieces
through ``ProceduralArt`` — a seeded multi-octave noise field, a rule-30
cellular automaton, and a geo-coordinate-seeded field anchored on Crescent
City, CA. Returns image summary statistics only (no file writes; figures are
rendered to in-memory buffers and closed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _image_stats(image: Any) -> dict[str, Any]:
    """Summarize a PIL RGBA image as deterministic scalar statistics."""
    import numpy as np

    arr = np.asarray(image, dtype=np.float64)
    rgb = arr[..., :3]
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return {
        "size_px": [int(image.width), int(image.height)],
        "rgb_mean": [round(float(v), 3) for v in rgb.reshape(-1, 3).mean(axis=0)],
        "rgb_std": [round(float(v), 3) for v in rgb.reshape(-1, 3).std(axis=0)],
        "luminance_mean": round(float(luminance.mean()), 3),
        "luminance_std": round(float(luminance.std()), 3),
        "dark_pixel_fraction": round(float((luminance < 64.0).mean()), 4),
        "bright_pixel_fraction": round(float((luminance > 192.0).mean()), 4),
    }


def _operation() -> dict[str, Any]:
    from geo_infer_art import ColorPalette, ProceduralArt

    # --- Palette construction from color theory --------------------------
    base_palette = ColorPalette.from_color_theory(
        "#2d6a4f", scheme="complementary", n_colors=6
    )
    forest_palette = ColorPalette.get_palette("forest")
    blended = forest_palette.blend_with(base_palette, ratio=0.5)
    dimmed = blended.adjust_brightness(0.6)
    contrast_vs_white = forest_palette.get_contrast_ratio("#ffffff")

    # --- Piece 1: seeded multi-octave noise field -------------------------
    noise_piece = ProceduralArt(
        algorithm="noise_field",
        params={
            "seed": 4217,
            "octaves": 5,
            "persistence": 0.55,
            "lacunarity": 2.0,
            "scale": 80.0,
            "x_influence": 0.31,
            "y_influence": 0.73,
            "color_palette": "forest",
        },
        resolution=(160, 120),
    )
    noise_piece.generate()
    if noise_piece.image is None:
        raise RuntimeError("noise_field generation produced no image")

    # --- Piece 2: rule-30 cellular automata ribbon ------------------------
    ca_piece = ProceduralArt(
        algorithm="cellular_automata",
        params={"rule": 30, "generations": 96, "color_palette": "grayscale"},
        resolution=(192, 96),
    )
    ca_piece.generate()
    if ca_piece.image is None:
        raise RuntimeError("cellular_automata generation produced no image")

    # --- Piece 3: geo-coordinate-seeded generation ------------------------
    # Crescent City, CA — the module's documented geo-seeding entry point
    # derives its RNG seed from the coordinates, so output is deterministic.
    geo_piece = ProceduralArt.from_geo_coordinates(
        lat=41.7558,
        lon=-124.2026,
        algorithm="cellular_automata",
        additional_params={"rule": 110, "generations": 64, "color_palette": "ocean"},
    )
    if geo_piece.image is None:
        raise RuntimeError("from_geo_coordinates generation produced no image")

    palette_colors: list[str] = list(blended.colors)

    return {
        "operation": "geo_palette_and_procedural_art_pipeline",
        "palettes": {
            "base_from_color_theory": list(base_palette.colors),
            "blended": palette_colors,
            "brightness_adjusted": list(dimmed.colors),
            "forest_vs_white_contrast_ratio": round(float(contrast_vs_white), 3),
        },
        "noise_field_piece": {
            "algorithm": noise_piece.algorithm,
            "seed": noise_piece.params["seed"],
            "stats": _image_stats(noise_piece.image),
        },
        "cellular_automata_piece": {
            "algorithm": ca_piece.algorithm,
            "rule": ca_piece.params["rule"],
            "stats": _image_stats(ca_piece.image),
        },
        "geo_seeded_piece": {
            "algorithm": geo_piece.algorithm,
            "geo_coordinates": list(geo_piece.params["geo_coordinates"]),
            "derived_seed": geo_piece.params["seed"],
            "stats": _image_stats(geo_piece.image),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("ART", _operation))
