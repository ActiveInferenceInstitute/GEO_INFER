"""H3 grid resolution policy: resolution suggestion and hard-cap guard.

Pure, dependency-free helpers that mirror the resolution-suggestion and
hard-cap guard idea from the GeoLibre processing registry (``h3-tools.ts``):
pick the finest H3 resolution whose estimated cell count stays within a target,
and refuse a request whose estimated grid would exceed a hard cap so a
pathological input cannot silently allocate an enormous, unbounded grid.

These functions only *estimate* counts and choose resolutions; they never
allocate H3 cells. The caller is responsible for actually building cells after
a guard passes. The authoritative H3 v4 runtime remains ``h3-py`` (see
``geo_infer_space.backends.h3``); this module deliberately adds no H3 import so
it stays importable in any environment.
"""

from __future__ import annotations

import math
from typing import TypedDict

# Official average hexagonal cell area (km^2) at each H3 resolution 0..15.
H3_AVG_AREA_KM2: list[float] = [
    4_357_449.416078381,
    609_788.441794133,
    86_801.780398997,
    12_393.434655088,
    1_770.347654491,
    252.903858182,
    36.129062164,
    5.16129336,
    0.737327598,
    0.105332513,
    0.015047502,
    0.002149643,
    0.000307092,
    0.00004387,
    0.000006267,
    0.000000895,
]

# Soft target used by the auto-suggester (matches GeoLibre's default).
H3_DEFAULT_TARGET_CELLS: int = 10_000
# Finest resolution the auto-suggester will pick.
H3_DEFAULT_MAX_RES: int = 12
# Hard ceiling: an estimated grid larger than this is rejected.
H3_HARD_CELL_CAP: int = 200_000

H3_RESOLUTION_MIN: int = 0
H3_RESOLUTION_MAX: int = 15


class H3HardCapExceededError(ValueError):
    """Raised when an H3 grid estimate exceeds the configured hard cap."""


class ResolutionSuggestion(TypedDict):
    """Result of an H3 resolution-suggestion call."""

    resolution: int
    estimated_cells: float
    within_target: bool


def _nonnegative_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _resolution(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
        raise ValueError(f"{name} must be an integer between 0 and 15")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def estimate_cell_count(area_km2: float, resolution: int) -> float:
    """Estimate the number of H3 cells covering ``area_km2`` at ``resolution``.

    Args:
        area_km2: Area in square kilometres. Must be non-negative.
        resolution: H3 resolution in ``[0, 15]``.

    Returns:
        Estimated cell count (a float; the caller decides how to quantize).

    Raises:
        ValueError: If ``area_km2`` is negative or ``resolution`` is out of
            range.
    """
    _nonnegative_finite(area_km2, "area_km2")
    _resolution(resolution, "resolution")
    return area_km2 / H3_AVG_AREA_KM2[resolution]


def suggest_h3_resolution(
    area_km2: float,
    target_cells: int = H3_DEFAULT_TARGET_CELLS,
    max_res: int = H3_DEFAULT_MAX_RES,
) -> ResolutionSuggestion:
    """Suggest the finest H3 resolution whose cell estimate stays <= target.

    Scans resolutions from ``max_res`` down to 0 and returns the finest one
    whose estimated cell count is at most ``target_cells``. For a very large
    area even resolution 0 may exceed the target; in that case resolution 0 is
    returned with ``estimate_cells`` reported honestly (never fabricated as
    within target).

    Args:
        area_km2: Area in square kilometres.
        target_cells: Target maximum number of cells (default 10,000).
        max_res: Finest resolution to consider (default 12).

    Returns:
        A :class:`ResolutionSuggestion` dict with keys ``resolution`` (int),
        ``estimated_cells`` (float), and ``within_target`` (bool).

    Raises:
        ValueError: If ``area_km2`` is negative, or ``target_cells`` or
            ``max_res`` are outside their valid ranges.
    """
    _nonnegative_finite(area_km2, "area_km2")
    _positive_integer(target_cells, "target_cells")
    _resolution(max_res, "max_res")

    chosen = H3_RESOLUTION_MIN
    within = False
    for res in range(max_res, H3_RESOLUTION_MIN - 1, -1):
        estimated = estimate_cell_count(area_km2, res)
        if estimated <= target_cells:
            chosen = res
            within = True
            break

    return {
        "resolution": chosen,
        "estimated_cells": estimate_cell_count(area_km2, chosen),
        "within_target": within,
    }


def check_cell_budget(
    estimated_cells: float,
    hard_cap: int = H3_HARD_CELL_CAP,
) -> None:
    """Raise :class:`H3HardCapExceededError` if ``estimated_cells`` > ``hard_cap``.

    This is the guard call for a *proposed* grid. It accepts an already-estimated
    count (float) so callers can check before allocating cells.

    Args:
        estimated_cells: Estimated number of cells.
        hard_cap: Maximum allowed cells (default 200,000).

    Raises:
        ValueError: If ``estimated_cells`` is negative.
        H3HardCapExceededError: If ``estimated_cells`` exceeds ``hard_cap``.
    """
    _nonnegative_finite(estimated_cells, "estimated_cells")
    _positive_integer(hard_cap, "hard_cap")
    if estimated_cells > hard_cap:
        raise H3HardCapExceededError(
            f"H3 grid estimate {estimated_cells:g} exceeds hard cap {hard_cap:g}"
        )


def suggest_resolution_with_budget(
    area_km2: float,
    target_cells: int = H3_DEFAULT_TARGET_CELLS,
    max_res: int = H3_DEFAULT_MAX_RES,
    hard_cap: int = H3_HARD_CELL_CAP,
) -> ResolutionSuggestion:
    """Suggest a resolution and enforce the hard cap in one call.

    Convenience wrapper combining :func:`suggest_h3_resolution` and
    :func:`check_cell_budget`. Raises ``H3HardCapExceededError`` before
    returning when even the coarsest relevant resolution would exceed the cap.

    Args:
        area_km2: Area in square kilometres.
        target_cells: Target maximum number of cells.
        max_res: Finest resolution to consider.
        hard_cap: Maximum allowed cells.

    Returns:
        A :class:`ResolutionSuggestion` (same shape as
        :func:`suggest_h3_resolution`).

    Raises:
        ValueError, H3HardCapExceededError: See the composed helpers.
    """
    result = suggest_h3_resolution(area_km2, target_cells=target_cells, max_res=max_res)
    estimated = estimate_cell_count(area_km2, result["resolution"])
    check_cell_budget(estimated, hard_cap=hard_cap)
    return result


__all__ = [
    "H3_AVG_AREA_KM2",
    "H3_DEFAULT_TARGET_CELLS",
    "H3_DEFAULT_MAX_RES",
    "H3_HARD_CELL_CAP",
    "H3HardCapExceededError",
    "ResolutionSuggestion",
    "estimate_cell_count",
    "suggest_h3_resolution",
    "check_cell_budget",
    "suggest_resolution_with_budget",
]
