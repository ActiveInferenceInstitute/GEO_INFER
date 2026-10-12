#!/usr/bin/env python3
"""GEO-INFER-PLACE module orchestrator.

Runs one documented end-to-end PLACE operation on synthetic data: tile a
synthetic Del Norte County study polygon into an H3 canvas with the module's
own H3 operations, derive a synthetic forest-observation dataset from that
canvas (NDVI/EVI measurements, forest-inventory plots, monthly climate), and
run the real ``ForestHealthMonitor.run_analysis`` end-to-end pipeline —
vegetation-index analysis, forest-type health, change detection, mortality
assessment, climate vulnerability, risk scoring, and alert generation. No
network calls: all observation sources are injected as configured callables
and no API clients are constructed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

from geo_infer_examples.orchestration import run_module_orchestrator


def _monthly_dates(start: date, count: int) -> list[str]:
    """Return ``count`` ISO date strings spaced one month apart from ``start``."""
    stamps: list[str] = []
    year, month = start.year, start.month
    for _ in range(count):
        stamps.append(date(year, month, 1).isoformat())
        month += 1
        if month > 12:
            month = 1
            year += 1
    return stamps


class _SyntheticForestIntegrator:
    """In-process integrator stub: no API clients, no network.

    Timber/mortality survey clients are deliberately unconfigured, which
    mirrors the module's own no-client degradation path; the monitor records
    the acquisition error and derives mortality from vegetation stress
    instead. Climate is served by the configured synthetic source because
    this stub exposes no ``noaa_client`` attribute.
    """

    def __init__(self) -> None:
        self.cache_dir: Any = None

    def get_timber_operations(
        self,
        bbox: Any = None,
        time_range: Any = None,
    ) -> dict[str, Any]:
        raise RuntimeError("timber-plan client not configured (synthetic run)")

    def get_tree_mortality_data(
        self,
        bbox: Any = None,
        time_range: Any = None,
    ) -> dict[str, Any]:
        raise RuntimeError("mortality-survey client not configured (synthetic run)")


def _operation() -> dict[str, Any]:
    import numpy as np
    from geo_infer_place.locations.del_norte_county.forest_health_monitor import (
        ForestHealthMonitor,
    )
    from geo_infer_place.utils.h3_operations import (
        cell_area,
        cell_to_latlng,
        compact_cells,
        polygon_to_cells,
    )

    rng = np.random.default_rng(42)

    # --- Synthetic H3 canvas over a fictional Del Norte study plot --------
    study_polygon: dict[str, Any] = {
        "type": "Polygon",
        "coordinates": [
            [
                [-124.15, 41.62],
                [-123.80, 41.62],
                [-123.70, 41.85],
                [-123.95, 42.00],
                [-124.25, 41.88],
                [-124.15, 41.62],
            ]
        ],
    }
    cells = polygon_to_cells(study_polygon, resolution=8)
    if not cells:
        raise RuntimeError("polygon_to_cells returned no cells for the study polygon")
    compacted = compact_cells(cells)
    sample_cell_area_km2 = cell_area(cells[0])

    # Observation sites: centers of the first six canvas cells.
    site_cells = cells[:6]
    site_centers = [cell_to_latlng(cell) for cell in site_cells]

    # --- Synthetic NDVI/EVI measurements (4 sites x 14 monthly scenes) ---
    dates = _monthly_dates(date(2024, 1, 1), 14)
    ndvi_measurements: list[dict[str, Any]] = []
    for site_idx, (lat, lon) in enumerate(site_centers[:4]):
        base_ndvi = 0.55 + 0.10 * float(rng.random())
        for step, stamp in enumerate(dates):
            seasonal = 0.08 * float(np.sin(2.0 * np.pi * step / 12.0))
            ndvi = float(
                np.clip(base_ndvi + seasonal + rng.normal(0.0, 0.03), 0.05, 0.95)
            )
            evi = float(np.clip(0.75 * ndvi + rng.normal(0.0, 0.02), 0.02, 0.90))
            stress = float(np.clip(1.0 - 1.1 * ndvi + rng.normal(0.0, 0.05), 0.0, 1.0))
            ndvi_measurements.append(
                {
                    "date": stamp,
                    "lat": lat,
                    "lon": lon,
                    "ndvi": ndvi,
                    "evi": evi,
                    "moisture_stress": stress,
                    "h3_cell": site_cells[site_idx],
                }
            )

    # --- Synthetic forest inventory plots across three forest types ------
    forest_types = ["Redwood", "Douglas Fir", "Mixed Conifer"]
    health_ratings = ["Good", "Fair", "Poor"]
    age_classes = ["Mature", "Old Growth", "Young"]
    forest_plots: list[dict[str, Any]] = []
    for plot_idx in range(12):
        lat, lon = site_centers[plot_idx % len(site_centers)]
        forest_plots.append(
            {
                "plot_id": f"SYN_{plot_idx:03d}",
                "lat": lat,
                "lon": lon,
                "forest_type": forest_types[plot_idx % len(forest_types)],
                "basal_area_m2_ha": float(rng.uniform(45.0, 130.0)),
                "tree_density_per_ha": float(rng.uniform(220.0, 640.0)),
                "average_height_m": float(rng.uniform(28.0, 78.0)),
                "canopy_cover_percent": float(rng.uniform(55.0, 92.0)),
                "health_rating": health_ratings[int(rng.integers(0, 3))],
                "age_class": age_classes[int(rng.integers(0, 3))],
                "h3_cell": site_cells[plot_idx % len(site_cells)],
            }
        )

    # --- Synthetic monthly climate series (36 months) --------------------
    climate_dates = _monthly_dates(date(2022, 1, 1), 36)
    climate_measurements: list[dict[str, Any]] = []
    for step, stamp in enumerate(climate_dates):
        seasonal_temp = 11.0 + 6.5 * float(np.sin(2.0 * np.pi * (step - 3) / 12.0))
        climate_measurements.append(
            {
                "date": stamp,
                "temperature_c": seasonal_temp + float(rng.normal(0.0, 0.8)),
                "precipitation_mm": float(np.clip(rng.gamma(2.0, 60.0), 0.0, 400.0)),
            }
        )

    sources: dict[str, Callable[[], dict[str, Any]]] = {
        "vegetation_indices": lambda: {"ndvi_measurements": ndvi_measurements},
        "forest_inventory": lambda: {"forest_plots": forest_plots},
        "climate": lambda: {"measurements": climate_measurements},
    }

    config: dict[str, Any] = {
        "location": {
            "bounds": {
                "west": -124.25,
                "south": 41.62,
                "east": -123.70,
                "north": 42.00,
            }
        },
        "spatial": {"h3_resolution": 8},
        "analyses": {
            "forest_health": {
                "vegetation_indices": {
                    "ndvi": {
                        "threshold_healthy": 0.7,
                        "threshold_stressed": 0.4,
                        "threshold_critical": 0.2,
                    }
                },
                "forest_types": forest_types,
                "change_detection": {
                    "baseline_years": [2024],
                    "minimum_change_threshold": 0.1,
                    "time_series_length": 14,
                },
                "data_sources": sources,
            }
        },
    }

    output_dir = Path(tempfile.mkdtemp(prefix="place_orchestrator_"))
    try:
        monitor = ForestHealthMonitor(
            config=config,
            data_integrator=_SyntheticForestIntegrator(),
            spatial_processor=None,
            output_dir=output_dir,
        )
        results = monitor.run_analysis(temporal_range=("2024-01-01", "2025-02-28"))
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    if results.get("status") != "success":
        raise RuntimeError(
            "ForestHealthMonitor.run_analysis did not succeed: "
            f"{results.get('error_message', results.get('status'))}"
        )

    vegetation = results["vegetation_analysis"]
    ndvi_analysis = vegetation["ndvi_analysis"]
    change = results["change_analysis"]
    mortality = results["mortality_analysis"]
    risk = results["risk_assessment"]
    alerts = results["health_alerts"]
    spatial = results["spatial_data"]
    inventory = results["forest_type_analysis"]

    forest_type_summary: dict[str, dict[str, Any]] = {}
    for ftype, summary in inventory.items():
        if isinstance(summary, dict) and "plot_count" in summary:
            forest_type_summary[ftype] = {
                "plot_count": int(summary["plot_count"]),
                "mean_canopy_cover": round(
                    float(summary["structure_metrics"]["mean_canopy_cover"]), 2
                ),
            }

    return {
        "operation": "forest_health_full_analysis_on_synthetic_h3_canvas",
        "h3_canvas": {
            "resolution": 8,
            "region_cells": len(cells),
            "compacted_cells": len(compacted),
            "sample_cell_area_km2": round(sample_cell_area_km2, 4),
            "observation_sites": len(site_cells),
        },
        "synthetic_data": {
            "ndvi_measurements": len(ndvi_measurements),
            "forest_plots": len(forest_plots),
            "climate_measurements": len(climate_measurements),
        },
        "vegetation_analysis": {
            "ndvi_mean": round(float(ndvi_analysis["mean"]), 4),
            "healthy_percent": round(float(ndvi_analysis["healthy_percent"]), 2),
            "stressed_percent": round(float(ndvi_analysis["stressed_percent"]), 2),
            "critical_percent": round(float(ndvi_analysis["critical_percent"]), 2),
            "h3_cells_summarized": len(vegetation["h3_spatial_summary"]),
        },
        "forest_type_health": forest_type_summary,
        "change_detection": {
            "cells_analyzed": len(change.get("h3_cell_changes", [])),
            "significant_changes": int(change.get("significant_changes_count", 0)),
        },
        "tree_mortality": {
            "data_source": mortality.get("data_source"),
            "mortality_rate_percent": mortality.get("mortality_rate_percent"),
            "affected_area_ha": mortality.get("affected_area_ha"),
        },
        "risk_assessment": {
            "overall_risk_score": round(float(risk["overall_risk_score"]), 4),
            "recommendations": risk["recommendations"],
        },
        "alerts": {
            "critical": len(alerts["critical_alerts"]),
            "warnings": len(alerts["warnings"]),
        },
        "spatial_output_cells": len(spatial.get("h3_cells", {})),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("PLACE", _operation))
