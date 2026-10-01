#!/usr/bin/env python3
"""Basic Integration Demo — real cross-module data flow on synthetic H3 data.

Demonstrates the documented SPACE → TIME → MATH integration pattern using the
real installed packages on deterministic synthetic data:

1. SPACE (``geo_infer_space``): build an H3 grid over a synthetic study
   region and derive cell centers + pairwise grid distances.
2. TIME (``geo_infer_time``): attach a 90-day synthetic observation series
   to three anchor cells and compute rolling-window statistics and full
   temporal summaries.
3. MATH (``geo_infer_math``): compute global spatial autocorrelation
   (Moran's I and Geary's C) of the per-cell mean values.

Every step asserts its outputs and the script prints one structured JSON
result. Exit code 0 means the full cross-module flow completed.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd


def _build_spatial_canvas() -> dict[str, Any]:
    """SPACE step: build the H3 canvas for a synthetic study region."""
    from geo_infer_space import SpatialIndexingInterface, polygon_to_cells

    indexer = SpatialIndexingInterface()
    region = {
        "type": "Polygon",
        "coordinates": [
            [
                [-123.1, 44.9],
                [-122.9, 44.9],
                [-122.8, 45.1],
                [-123.0, 45.2],
                [-123.2, 45.1],
                [-123.1, 44.9],
            ]
        ],
    }
    cells = polygon_to_cells(region, resolution=7)
    assert len(cells) > 10, f"expected a real grid, got {len(cells)} cells"

    centers = {cell: indexer.cell_to_latlng(cell) for cell in cells[:6]}
    distances = {cell: indexer.get_cell_distance(cells[0], cell) for cell in cells[1:6]}
    return {"cells": cells, "centers": centers, "distances": distances}


def _analyze_temporal_dynamics(
    cells: list[str], centers: dict[str, Any]
) -> dict[str, Any]:
    """TIME step: rolling-window statistics for synthetic per-cell series."""
    from geo_infer_time import TemporalStatistics, TimeSeries

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01", periods=90, freq="D")
    stats_engine = TemporalStatistics()
    series_results: dict[str, Any] = {}

    for cell in cells[:6]:
        lat, lng = centers[cell]
        days = np.arange(90)
        values = (
            10.0
            + 0.02 * days
            + 2.0 * np.sin(2 * np.pi * days / 30.0)
            + rng.normal(0, 0.5, size=90)
        )
        series = TimeSeries(
            data=values,
            timestamps=timestamps,
            spatial_location={"lat": float(lat), "lon": float(lng)},
            metadata={"synthetic": True, "cell": cell},
        )
        frame = series.to_dataframe() if hasattr(series, "to_dataframe") else None
        if frame is not None and isinstance(frame, pd.DataFrame):
            numeric = frame.select_dtypes(include=[np.number]).iloc[:, 0]
        else:
            numeric = pd.Series(values)
        rolling_mean = numeric.rolling(window=7, min_periods=7).mean()
        rolling_std = numeric.rolling(window=7, min_periods=7).std()

        summary = stats_engine.calculate_summary(values=[float(v) for v in values])
        central = summary.get("central_tendency", {})
        dispersion = summary.get("dispersion", {})
        assert "mean" in central, f"summary missing mean: {summary.keys()}"
        series_results[cell] = {
            "series_length": len(values),
            "rolling_window": 7,
            "last_rolling_mean": round(float(rolling_mean.iloc[-1]), 4),
            "last_rolling_std": round(float(rolling_std.iloc[-1]), 4),
            "summary_mean": round(float(central["mean"]), 4),
            "summary_std": round(float(dispersion.get("std", 0.0)), 4),
        }

    assert len(series_results) == 6
    return series_results


def _compute_spatial_autocorrelation(
    cells: list[str], centers: dict[str, Any], per_cell_means: np.ndarray
) -> dict[str, Any]:
    """MATH step: global spatial autocorrelation of per-cell mean values."""
    from geo_infer_math.core.spatial_statistics import GearysC, MoranI

    coords = np.array([centers[cell] for cell in cells[:6]], dtype=float)
    moran = MoranI().compute(values=per_cell_means, coords=coords)
    geary = GearysC(rng=np.random.default_rng(42)).compute(
        values=per_cell_means, coords=coords
    )

    moran_i = float(moran.get("moran_i", moran.get("I", moran.get("statistic", 0.0))))
    assert -1.5 <= moran_i <= 1.5, f"Moran's I out of plausible range: {moran_i}"
    return {
        "moran_i": round(moran_i, 4),
        "moran_p_value": round(float(moran.get("p_value", moran.get("p", 0.0))), 4),
        "gearys_c": round(float(geary.get("C", geary.get("gearys_c", 0.0))), 4),
        "n_locations": len(coords),
    }


def run_demo() -> dict[str, Any]:
    """Run the full SPACE → TIME → MATH cross-module demonstration."""
    spatial = _build_spatial_canvas()
    temporal = _analyze_temporal_dynamics(spatial["cells"], spatial["centers"])

    per_cell_means = np.array(
        [temporal[cell]["summary_mean"] for cell in list(temporal)]
    )
    autocorr = _compute_spatial_autocorrelation(
        list(temporal), spatial["centers"], per_cell_means
    )

    return {
        "pattern": "SPACE -> TIME -> MATH",
        "space": {
            "grid_cells": len(spatial["cells"]),
            "anchor_cell": spatial["cells"][0],
            "distances_from_anchor": spatial["distances"],
        },
        "time": temporal,
        "math": autocorr,
    }


def main() -> int:
    """Entry point: run the demo, print structured JSON, return exit code."""
    try:
        results = run_demo()
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "status": "missing-dependency",
                    "requires": (
                        "geo-infer-space, geo-infer-time and geo-infer-math "
                        "(installed as editable workspace members)"
                    ),
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - demo entrypoint boundary
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps({"status": "ok", **results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
