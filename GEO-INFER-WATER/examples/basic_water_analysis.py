"""
Basic water resources analysis example.

Demonstrates rainfall-runoff modeling, water balance closure, watershed
delineation, and water quality assessment using the real GEO-INFER-WATER
APIs.
"""

import numpy as np
import xarray as xr

from geo_infer_water import (
    HydrologicalModeler,
    WaterBalanceModeler,
    WaterQualityAssessor,
    WatershedDelineator,
    WaterSample,
)


def main():
    """Run basic water analysis example."""
    print("GEO-INFER-WATER: Basic Water Resources Analysis")
    print("=" * 50)

    # 1. Rainfall-runoff modeling with mass conservation
    print("\n1. Rainfall-Runoff Modeling")
    print("-" * 30)
    hydrology = HydrologicalModeler()
    precip = xr.DataArray(np.full((5, 5), 100.0), dims=("y", "x"))
    soil_wet = xr.DataArray(np.full((5, 5), 0.8), dims=("y", "x"))
    rr = hydrology.rainfall_runoff_model(
        precip, soil_moisture=soil_wet, infiltration_rate=0.6
    )
    total = rr["runoff"] + rr["infiltration"]
    print(f"   Precipitation:    {float(precip.mean()):.1f} mm")
    print(f"   Runoff:            {float(rr['runoff'].mean()):.1f} mm")
    print(f"   Infiltration:      {float(rr['infiltration'].mean()):.1f} mm")
    print(f"   Mass check (P=R+I): {float(total.mean()):.1f} mm")

    # 2. Water balance closure (canonical owner: WaterBalanceModeler)
    print("\n2. Water Balance")
    print("-" * 30)
    balance_modeler = WaterBalanceModeler()
    p = xr.DataArray(np.full((5, 5), 100.0), dims=("y", "x"))
    et = xr.DataArray(np.full((5, 5), 60.0), dims=("y", "x"))
    runoff = xr.DataArray(np.full((5, 5), 30.0), dims=("y", "x"))
    balance = balance_modeler.water_balance_closure(p, et, runoff)
    print(f"   Storage change:    {float(balance['storage_change'].mean()):.1f} mm")
    print(f"   Closure residual:  {float(balance['closure_residual'].mean()):.1f} mm")

    # 3. Watershed delineation (D8 flow direction + accumulation)
    print("\n3. Watershed Delineation")
    print("-" * 30)
    delineator = WatershedDelineator()
    dem = np.array(
        [
            [9, 8, 7, 8, 9],
            [8, 7, 5, 7, 8],
            [7, 5, 3, 5, 7],
            [8, 6, 4, 6, 8],
            [9, 7, 5, 7, 9],
        ],
        dtype=float,
    )
    dem_da = xr.DataArray(dem, dims=("y", "x"))
    watershed = delineator.full_delineation(dem_da, outlet=(2, 2), cell_size=500.0)
    print(f"   Basin area:        {watershed.attrs['basin_area_km2']:.2f} km2")
    print(f"   Basin cells:       {watershed.attrs['basin_area_cells']}")

    # 4. Water quality assessment
    print("\n4. Water Quality Assessment")
    print("-" * 30)
    assessor = WaterQualityAssessor()
    sample = WaterSample(
        sample_id="wq001",
        location=(-122.0, 38.5),
        timestamp="2024-07-15",
        ph=7.2,
        dissolved_oxygen=8.5,
        turbidity=2.0,
        temperature=18.0,
        nitrate=2.0,
        e_coli=10,
    )
    wqi = assessor.calculate_wqi(sample)
    print(f"   WQI:               {wqi['wqi']:.1f} ({wqi['classification']})")
    print(f"   Sub-indices:       {list(wqi['sub_indices'].keys())}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
