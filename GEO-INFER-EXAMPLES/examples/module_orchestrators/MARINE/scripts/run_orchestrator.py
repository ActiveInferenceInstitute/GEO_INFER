#!/usr/bin/env python3
"""GEO-INFER-MARINE module orchestrator.

Runs one documented end-to-end MARINE operation on synthetic data: assess the
water quality of a coastal monitoring station — dissolved oxygen saturation,
percent saturation, ocean acidification index, turbidity scoring, and the
composite marine Water Quality Index via ``MarineWaterQuality`` — then
evaluate the adjacent coral reef with ``CoralReefAssessor``: Degree Heating
Weeks from a synthetic SST series, the NOAA-style bleaching alert level,
reef biodiversity metrics, and the composite reef health score. All work goes
through the real ``geo_infer_marine`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    import xarray as xr
    from geo_infer_marine import CoralReefAssessor, MarineWaterQuality

    # Station "reef-lagoon-01": deterministic synthetic measurements.
    temperature_c = xr.DataArray(27.4, name="temperature_c")
    salinity_psu = xr.DataArray(35.1, name="salinity_psu")
    measured_do = xr.DataArray(4.5, name="measured_do")
    ph = xr.DataArray(8.02, name="ph")
    turbidity_ntu = xr.DataArray(3.2, name="turbidity_ntu")

    water_quality = MarineWaterQuality()
    do_saturation = float(
        water_quality.calculate_do_saturation(temperature_c, salinity_psu).values
    )
    do_percent = float(
        water_quality.calculate_do_percent_saturation(
            measured_do, temperature_c, salinity_psu
        ).values
    )
    acidification_index = float(
        water_quality.calculate_ocean_acidification_index(ph).values
    )
    turbidity_score = float(
        water_quality.calculate_turbidity_score(turbidity_ntu).values
    )

    # Station-side 0-100 scores feeding the documented composite WQI weights.
    do_score = xr.DataArray(float(np.clip(do_percent, 0.0, 100.0)))
    ph_score = xr.DataArray(
        float(np.clip(100.0 - acidification_index * 100.0, 0.0, 100.0))
    )
    turb_score = xr.DataArray(turbidity_score)
    wqi = water_quality.composite_marine_wqi(do_score, ph_score, turb_score)
    wqi_value = float(wqi["wqi"].values)
    wqi_class = str(wqi["classification"].values)

    # Weekly sea surface temperature at the reef site vs the MMM climatology.
    sst = xr.DataArray(
        [28.6, 29.2, 29.8, 30.2, 30.8, 31.5, 31.2, 30.6],
        dims=("time",),
        coords={"time": np.arange(8)},
        name="sst",
    )
    climatological_max = xr.DataArray(29.0, name="climatological_max")

    reef = CoralReefAssessor()
    dhw = reef.calculate_degree_heating_weeks(sst, climatological_max, window_weeks=12)
    dhw_final = float(dhw.isel(time=-1).values)
    alert_level = int(reef.classify_bleaching_alert(dhw).isel(time=-1).values)

    species_counts = {
        "Acropora cervicornis": 42,
        "Porites astreoides": 67,
        "Montastraea cavernosa": 31,
        "Siderastrea siderea": 58,
        "Diadema antillarum": 84,
    }
    biodiversity = reef.calculate_reef_biodiversity(species_counts)
    health = reef.assess_reef_health_composite(
        coral_cover_pct=32.0,
        macroalgae_cover_pct=14.0,
        fish_biomass_kg_ha=620.0,
        bleaching_alert_level=alert_level,
    )

    return {
        "operation": "water_quality_assessment_with_reef_health_check",
        "station": {
            "id": "reef-lagoon-01",
            "temperature_c": 27.4,
            "salinity_psu": 35.1,
            "measured_do_mg_l": 4.5,
            "ph": 8.02,
            "turbidity_ntu": 3.2,
        },
        "water_quality": {
            "do_saturation_mg_l": round(do_saturation, 6),
            "do_percent_saturation": round(do_percent, 6),
            "acidification_index": round(acidification_index, 6),
            "turbidity_score": round(turbidity_score, 6),
            "wqi": round(wqi_value, 6),
            "wqi_classification": wqi_class,
        },
        "coral_reef": {
            "degree_heating_weeks": round(dhw_final, 6),
            "bleaching_alert_level": alert_level,
            "biodiversity": {
                "species_richness": int(biodiversity["species_richness"]),
                "total_abundance": int(biodiversity["total_abundance"]),
                "shannon_index": round(float(biodiversity["shannon_index"]), 6),
                "simpson_index": round(float(biodiversity["simpson_index"]), 6),
                "margalef_index": round(float(biodiversity["margalef_index"]), 6),
                "evenness": round(float(biodiversity["evenness"]), 6),
            },
            "health": {
                "composite_score": round(float(health["composite_score"]), 6),
                "classification": str(health["classification"]),
                "coral_score": round(float(health["coral_score"]), 6),
                "algae_score": round(float(health["algae_score"]), 6),
                "fish_score": round(float(health["fish_score"]), 6),
                "thermal_penalty": round(float(health["thermal_penalty"]), 6),
            },
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("MARINE", _operation))
