---
name: geo-infer-water
description: Water resource management and hydrological modeling. Use when analyzing watersheds (D8 delineation, flow accumulation), water quality (WQI, pollution tracking, regulatory compliance), rainfall-runoff, water balance, flood/drought screening, or water supply allocation.
difficulty: intermediate
estimated_time: 45min
---

# GEO-INFER-WATER

## Instructions

### Core Capabilities

- **Watershed delineation**: D8 flow direction, topological-sort flow accumulation, upstream basin tracing, stream network extraction, slope calculation, and a full delineation pipeline
- **Water quality**: WQI calculation, pollution hotspot identification (with optional D8 upstream tracing), advection-diffusion plume tracking, trend analysis, risk assessment, regulatory compliance (EPA/WHO/EU), pollutant load estimation
- **Hydrology**: Rainfall-runoff split (mass-conserving, soil-moisture adjusted), Green-Ampt infiltration (physically based, implicit ponding-time solver, mass-exact), groundwater recharge estimate, water balance (delegates to WaterBalanceModeler)
- **Water balance**: Thornthwaite and Hargreaves PET, SCS Curve Number runoff, monthly soil-moisture accounting, canonical water-balance closure
- **Flood & drought**: Equal-weight composite flood-risk screening (extreme precipitation + low elevation + optional soil saturation); drought risk from precipitation deficit
- **Infrastructure**: Priority-weighted water allocation (supply split by demand × priority, capped at demand, surplus redistributed) and capacity-gap assessment

Not implemented: aquifer/well-drawdown modeling, flood-frequency (return-period) analysis, inundation mapping. Groundwater is limited to a trivial recharge estimate. Green-Ampt infiltration is implemented; the remaining three surfaces stay explicitly not-implemented.

### Key Imports

```python
from geo_infer_water import (
    HydrologicalModeler,
    WaterBalanceModeler,
    WaterQualityAssessor,
    WaterInfrastructurePlanner,
    FloodDroughtAnalyzer,
    WatershedDelineator,
)
```

## Examples

```python
import numpy as np
import xarray as xr
from geo_infer_water import HydrologicalModeler, WatershedDelineator, WaterQualityAssessor

# Rainfall-runoff (mass-conserving: runoff + infiltration == precipitation)
hydro = HydrologicalModeler()
precip = xr.DataArray(np.full((5, 5), 100.0), dims=("y", "x"))
result = hydro.rainfall_runoff_model(precip, infiltration_rate=0.6)

# Green-Ampt infiltration (physically based, implicit ponding solver)
ga_precip = xr.DataArray(np.array([[5.0], [40.0], [80.0], [80.0], [3.0]]), dims=("time", "y"))
ga = hydro.green_ampt_infiltration(ga_precip, ks=10.0, suction_head=50.0, delta_theta=0.34, dt=1.0)
# ga['infiltration'] + ga['runoff'] == ga_precip (mass-exact)

# Watershed delineation (D8 flow direction + accumulation)
delineator = WatershedDelineator()
dem = xr.DataArray(np.array([[9, 8, 7], [8, 3, 5], [7, 5, 3]], dtype=float), dims=("y", "x"))
watershed = delineator.full_delineation(dem, outlet=(2, 2), cell_size=500.0)

# Water quality index
assessor = WaterQualityAssessor()
from geo_infer_water import WaterSample
sample = WaterSample("s1", (0.0, 0.0), "2024-07-15", ph=7.2, dissolved_oxygen=8.5,
                     turbidity=2.0, temperature=18.0, nitrate=2.0, e_coli=10)
wqi = assessor.calculate_wqi(sample)
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog WATER` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- `WatershedDelineator` is the canonical watershed module; the older `WatershedAnalyzer` has been removed.
- `rainfall_runoff_model` conserves mass: `runoff + infiltration == precipitation` for any soil moisture.
- `HydrologicalModeler.calculate_water_balance` delegates to `WaterBalanceModeler.water_balance_closure` (single water-balance owner).
- `assess_flood_risk` is an equal-weight screening heuristic, not a calibrated flood model.
- `optimize_water_allocation` splits scarce supply by `demand * priority`, capped at each demand.
- Green-Ampt has two owners with deliberately different numerics:
  `HydrologicalModeler.green_ampt_infiltration` is the spatial/raster
  variant (xarray in/out, implicit Newton ponding solver), while
  `InfiltrationModeler.green_ampt_infiltration` is the point-scale
  variant (numpy dict out, explicit supply-limited stepping). They are
  pinned to stay numerically consistent (start-of-step capacity vs
  mid-step implicit resolution, ~1% cumulative divergence) by
  `tests/unit/test_infiltration.py::test_cross_model_green_ampt_consistency`
  — do not change one without running the other's suite.

### Integrations

- Integrates with CLIMATE for precipitation projections
- Integrates with AG for irrigation water demand
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module WATER`
