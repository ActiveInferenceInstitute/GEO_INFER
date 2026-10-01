#!/usr/bin/env python3
"""GEO-INFER-ENERGY module orchestrator.

Runs one documented end-to-end ENERGY operation on synthetic data: a
combined wind + solar renewable resource assessment for a single site.
A deterministic year-like set of hourly wind observations is fitted with
a Weibull distribution, hub-height extrapolation and wind power density
are computed, and annual energy production is estimated from the turbine
power curve via ``WindAnalyzer``. In parallel, ``SolarAnalyzer`` computes
solar geometry, clear-sky irradiance, daily insolation, the optimal fixed
panel tilt, tilted-surface irradiance, and a PV system output estimate.
All work goes through the real ``geo_infer_energy`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_energy import SolarAnalyzer, WindAnalyzer

    # Synthetic site: coastal ridge at fixed coordinates; no external data.
    latitude_deg = 37.44
    altitude_m = 120.0
    day_of_year = 180
    solar_noon_hours = 12.0
    anemometer_height_m = 10.0
    hub_height_m = 80.0

    # Deterministic 72-hour synthetic wind record (three diurnal cycles)
    # derived from fixed sinusoids; no random sampling involved.
    hours = np.arange(72)
    wind_speeds = (
        5.5 + 2.5 * np.sin(2.0 * np.pi * hours / 24.0) + 0.8 * np.sin(0.35 * hours)
    )

    wind = WindAnalyzer()
    weibull = wind.fit_weibull(wind_speeds)
    hub_speed = wind.extrapolate_wind_speed(
        weibull["mean_speed"], anemometer_height_m, hub_height_m
    )
    mean_power_density = float(
        wind.wind_power_density(np.array([weibull["mean_speed"]]))[0]
    )
    power_curve_speeds = np.linspace(0.0, 30.0, 31)
    power_curve_kw = wind.turbine_power_curve(power_curve_speeds, rated_power_kw=2000.0)
    aep = wind.annual_energy_production(
        weibull["shape_k"], weibull["scale_c"], rated_power_kw=2000.0
    )

    solar = SolarAnalyzer()
    declination = solar.solar_declination(day_of_year)
    noon_elevation = solar.solar_elevation(latitude_deg, day_of_year, solar_noon_hours)
    noon_ghi = solar.clear_sky_ghi(
        latitude_deg, day_of_year, solar_noon_hours, altitude_m
    )
    insolation = solar.daily_insolation(latitude_deg, day_of_year, altitude_m)
    tilt = solar.optimal_tilt_angle(latitude_deg)
    tilted_factor = solar.tilted_irradiance_factor(
        tilt_deg=tilt,
        azimuth_deg=180.0,
        solar_elevation_deg=noon_elevation,
        solar_azimuth_deg=180.0,
    )
    pv = solar.estimate_pv_output(
        insolation, panel_area_m2=1500.0, efficiency=0.20, performance_ratio=0.80
    )

    return {
        "operation": "wind_solar_site_resource_assessment",
        "site": {
            "latitude_deg": latitude_deg,
            "altitude_m": altitude_m,
            "day_of_year": day_of_year,
        },
        "wind": {
            "samples": int(wind_speeds.size),
            "mean_speed_mps": round(weibull["mean_speed"], 6),
            "std_speed_mps": round(weibull.get("std_speed", 0.0), 6),
            "weibull_shape_k": round(weibull["shape_k"], 6),
            "weibull_scale_c": round(weibull["scale_c"], 6),
            "hub_height_speed_mps": round(hub_speed, 6),
            "mean_power_density_w_m2": round(mean_power_density, 6),
            "power_curve_peak_kw": round(float(power_curve_kw.max()), 6),
            "aep_kwh": round(aep["aep_kwh"], 6),
            "capacity_factor": round(aep["capacity_factor"], 6),
        },
        "solar": {
            "declination_deg": round(declination, 6),
            "noon_elevation_deg": round(noon_elevation, 6),
            "noon_clear_sky_ghi_w_m2": round(noon_ghi, 6),
            "daily_insolation_kwh_m2": round(insolation, 6),
            "optimal_tilt_deg": round(tilt, 6),
            "tilted_irradiance_factor": round(tilted_factor, 6),
            "pv_annual_kwh": round(pv["annual_kwh"], 6),
            "pv_capacity_factor": round(pv["capacity_factor"], 6),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("ENERGY", _operation))
