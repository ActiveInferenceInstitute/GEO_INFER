#!/usr/bin/env python3
"""GEO-INFER-FOREST module orchestrator.

Runs one documented end-to-end FOREST operation on synthetic data: assess
the fire danger of a forested landscape through the exported convenience
API — ``WildfireRiskAnalyzer.calculate_fire_weather_index`` on a station
observation, ``WildfireRiskAnalyzer.assess_wildfire_risk`` over a
synthetic time-by-space grid, ``FireRiskAssessor.calculate_kbdi`` and
``calculate_angstrom_index`` on a synthetic dry-spell weather series, and
``ForestInventory.estimate_biomass`` plus ``calculate_forest_area`` on the
same grid. All work goes through the real ``geo_infer_forest`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import numpy as np
    import xarray as xr

    from geo_infer_forest import (
        FireRiskAssessor,
        FireWeatherObservation,
        ForestInventory,
        WildfireRiskAnalyzer,
    )

    rng = np.random.default_rng(42)

    # Station observation: hot, dry, windy day in a dry spell.
    observation = FireWeatherObservation(
        observation_id="STN-001",
        location=(45.5, -122.6),
        timestamp="2026-08-14T15:00:00Z",
        temperature_c=34.0,
        relative_humidity=18.0,
        wind_speed_kmh=42.0,
        wind_direction_deg=225.0,
        precipitation_mm=0.0,
    )

    analyzer = WildfireRiskAnalyzer()
    fwi = analyzer.calculate_fire_weather_index(observation)

    # Synthetic 7-day x 4x4 landscape weather grid (fixed seed).
    times = np.arange("2026-08-08", "2026-08-15", dtype="datetime64[D]")
    y = np.arange(4)
    x = np.arange(4)
    temperature = xr.DataArray(
        28.0 + 6.0 * rng.random((7, 4, 4)),
        coords={"time": times, "y": y, "x": x},
        dims=("time", "y", "x"),
        name="temperature",
    )
    precipitation = xr.DataArray(
        np.clip(rng.gamma(shape=0.6, scale=2.0, size=(7, 4, 4)), 0.0, 8.0),
        coords={"time": times, "y": y, "x": x},
        dims=("time", "y", "x"),
        name="precipitation",
    )
    wind_speed = xr.DataArray(
        10.0 + 30.0 * rng.random((7, 4, 4)),
        coords={"time": times, "y": y, "x": x},
        dims=("time", "y", "x"),
        name="wind_speed",
    )
    fuel_load = xr.DataArray(
        40.0 + 80.0 * rng.random((4, 4)),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="fuel_load",
    )

    risk = analyzer.assess_wildfire_risk(
        temperature,
        precipitation,
        fuel_load=fuel_load,
        wind_speed=wind_speed,
    )

    # Drought tracking over the same dry spell.
    assessor = FireRiskAssessor()
    daily_max_temp_c = np.asarray([31.0, 33.5, 32.0, 34.0, 35.0, 33.0, 34.5])
    daily_precip_mm = np.asarray([0.0, 0.0, 1.2, 0.0, 0.0, 3.5, 0.0])
    kbdi = assessor.calculate_kbdi(
        daily_max_temp_c,
        daily_precip_mm,
        mean_annual_precip_mm=1100.0,
    )
    angstrom = assessor.calculate_angstrom_index(
        temperature_c=34.0,
        relative_humidity=18.0,
    )

    # Stand-level inventory on the same landscape grid.
    forest_cover = xr.DataArray(
        55.0 + 40.0 * rng.random((4, 4)),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="forest_cover",
    )
    tree_density = xr.DataArray(
        200.0 + 600.0 * rng.random((4, 4)),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="tree_density",
    )
    inventory = ForestInventory()
    biomass = inventory.estimate_biomass(forest_cover, tree_density)
    forest_area = inventory.calculate_forest_area(forest_cover)

    return {
        "operation": "fire_season_risk_with_forest_inventory",
        "fire_weather_index": {
            "observation_id": fwi["observation_id"],
            "fwi": round(float(fwi["fwi"]), 6),
            "danger_rating": str(fwi["danger_rating"]),
            "ffmc": round(float(fwi["components"]["ffmc"]), 6),
            "isi": round(float(fwi["components"]["isi"]), 6),
        },
        "landscape_wildfire_risk": {
            "mean_risk": round(float(risk["wildfire_risk"].mean()), 6),
            "max_risk": round(float(risk["wildfire_risk"].max()), 6),
            "mean_drought_index": round(float(risk["drought_index"].mean()), 6),
        },
        "drought": {
            "final_kbdi": round(float(kbdi[-1]), 6),
            "peak_kbdi": round(float(kbdi.max()), 6),
            "angstrom_index": round(float(angstrom["angstrom_index"]), 6),
            "angstrom_fire_danger": str(angstrom["fire_danger"]),
        },
        "inventory": {
            "mean_biomass_tons_per_ha": round(float(biomass.mean()), 6),
            "total_forest_area_km2": round(float(forest_area.sum()), 6),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("FOREST", _operation))
