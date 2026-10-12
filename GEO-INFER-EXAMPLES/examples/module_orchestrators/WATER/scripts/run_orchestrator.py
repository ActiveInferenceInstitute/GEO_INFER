#!/usr/bin/env python3
"""GEO-INFER-WATER module orchestrator.

Runs one documented end-to-end WATER operation on synthetic data: drive a
rainfall-runoff model over a fixed monthly precipitation series, estimate
groundwater recharge from the partitioned infiltration, close the catchment
water balance, and assess the water quality of a synthetic river sample via
the NSF Water Quality Index. All work goes through the real
``geo_infer_water`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    import xarray as xr
    from geo_infer_water import (
        HydrologicalModeler,
        WaterBodyType,
        WaterQualityAssessor,
        WaterSample,
    )

    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    precipitation_mm = np.array([85.0, 62.5, 110.0, 74.0, 48.0, 91.5])
    soil_moisture_fraction = np.array([0.35, 0.40, 0.55, 0.60, 0.30, 0.45])
    evapotranspiration_mm = np.array([18.0, 22.0, 35.0, 48.0, 60.0, 72.5])

    precipitation = xr.DataArray(
        precipitation_mm,
        coords={"time": months},
        dims=["time"],
        name="precipitation",
    )
    soil_moisture = xr.DataArray(
        soil_moisture_fraction,
        coords={"time": months},
        dims=["time"],
        name="soil_moisture",
    )
    evapotranspiration = xr.DataArray(
        evapotranspiration_mm,
        coords={"time": months},
        dims=["time"],
        name="evapotranspiration",
    )

    modeler = HydrologicalModeler(config={"catchment": "Cedar Creek"})
    partition = modeler.rainfall_runoff_model(
        precipitation, soil_moisture=soil_moisture, infiltration_rate=0.5
    )
    recharge = modeler.estimate_groundwater_recharge(
        partition["infiltration"], evapotranspiration=evapotranspiration
    )
    balance = modeler.calculate_water_balance(
        precipitation, evapotranspiration, partition["runoff"]
    )

    assessor = WaterQualityAssessor()
    sample = WaterSample(
        sample_id="WS-2026-0142",
        location=(-122.4194, 37.7749),
        timestamp="2026-06-15T09:00:00+00:00",
        ph=7.2,
        dissolved_oxygen=8.1,
        turbidity=3.5,
        temperature=18.0,
        conductivity=420.0,
        nitrate=2.4,
        e_coli=10.0,
    )
    wqi = assessor.calculate_wqi(sample)

    return {
        "operation": "rainfall_runoff_with_recharge_balance_and_wqi",
        "rainfall_runoff": {
            "total_precipitation_mm": float(precipitation.sum()),
            "total_runoff_mm": round(float(partition["runoff"].sum()), 6),
            "total_infiltration_mm": round(float(partition["infiltration"].sum()), 6),
            "mass_conserved": bool(
                np.isclose(
                    float(partition["runoff"].sum())
                    + float(partition["infiltration"].sum()),
                    float(precipitation.sum()),
                )
            ),
        },
        "groundwater_recharge": {
            "total_recharge_mm": round(float(recharge.sum()), 6),
            "recharge_fraction_of_infiltration": round(
                float(recharge.sum() / partition["infiltration"].sum()), 6
            ),
        },
        "water_balance": {
            "storage_change_mm": round(float(balance["storage_change"].sum()), 6),
            "closure_residual_mm": round(float(balance["closure_residual"].sum()), 6),
        },
        "water_quality": {
            "sample_id": wqi["sample_id"],
            "wqi": round(float(wqi["wqi"]), 6),
            "classification": str(wqi["classification"]),
            "water_body_type": WaterBodyType.RIVER.value,
            "sub_indices": {
                key: round(float(value), 6) for key, value in wqi["sub_indices"].items()
            },
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("WATER", _operation))
