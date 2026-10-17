---
name: geo-infer-energy
description: Energy systems analysis and renewable energy siting. Use when assessing solar/wind/hydro potential, computing capacity factors and LCOE, siting renewable sites, forecasting energy demand, planning grid and storage needs, or analyzing carbon emissions of energy systems.
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-data
  recommended:
    - geo-infer-time
    - geo-infer-bayes
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-ENERGY

## Instructions

### Core Capabilities

- **Renewable resource assessment**: solar/wind/hydro potential maps, site suitability scoring, capacity factors, LCOE, storage sizing (`RenewableResourceAssessor`)
- **Solar physics**: declination, clear-sky irradiance, optimal tilt (`SolarAnalyzer`)
- **Wind analysis**: Weibull fitting, power-curve integration (`WindAnalyzer`)
- **Demand forecasting**: trend extrapolation with temperature and population adjustment (`EnergyDemandForecaster`)
- **Grid analysis**: supply-demand balance, reliability and reserve-margin adequacy (`EnergyGridOptimizer`)
- **Infrastructure siting**: facility siting suitability, capacity-gap assessment (`EnergyInfrastructurePlanner`)
- **Emissions**: emission factors by fuel, carbon intensity, renewable impact (`CarbonFootprintAnalyzer`)

### Key Imports

```python
from geo_infer_energy import (
    RenewableResourceAssessor,
    RenewableType,
    SuitabilityClass,
    RenewableSite,
    SolarAnalyzer,
    WindAnalyzer,
    EnergyDemandForecaster,
    EnergyGridOptimizer,
    EnergyInfrastructurePlanner,
    CarbonFootprintAnalyzer,
)
# or equivalently:
from geo_infer_energy.core.renewable_resources import RenewableResourceAssessor, RenewableType
from geo_infer_energy.core.energy_grid import EnergyGridOptimizer
from geo_infer_energy.core.energy_demand import EnergyDemandForecaster
```

## Examples

```python
import numpy as np
import xarray as xr

from geo_infer_energy import RenewableResourceAssessor, RenewableType

assessor = RenewableResourceAssessor()

# 1. Assess solar potential from an irradiance raster (kWh/m^2/day)
irradiance = xr.DataArray(np.full((10, 10), 5.5), dims=("y", "x"))
solar = assessor.assess_solar_potential(irradiance)  # xr.Dataset

# 2. Score site suitability
result = assessor.assess_site_suitability(
    location=(-118.25, 34.05),
    resource_type=RenewableType.SOLAR_PV,
    resource_value=7.0,
)
print(result["suitability_class"], result["final_score"])

# 3. Capacity factor from an hourly resource time series
hours = np.arange(8760)
ghi = np.maximum(0.0, 500 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 4))
cf = assessor.calculate_capacity_factor(
    RenewableType.SOLAR_PV, xr.DataArray(ghi, dims=["time"]), rated_capacity_mw=100
)
print(cf["capacity_factor"], cf["annual_generation_mwh"])

# 4. LCOE (USD/MWh)
lcoe = assessor.calculate_lcoe(
    resource_type=RenewableType.SOLAR_PV,
    capacity_mw=100,
    capacity_factor=cf["capacity_factor"],
)
print(lcoe["lcoe_usd_mwh"], lcoe["competitiveness"])
```

Full runnable scripts live in `examples/` (`renewable_energy_planning.py`, `basic_energy_analysis.py`).

### Agent interfaces

Use `uv run --frozen geo-infer catalog ENERGY` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- All spatial inputs are `xarray` objects; time series use a `time` dimension.
- Coordinates follow (lon, lat) ordering for site locations; GeoJSON-style [lng, lat] elsewhere.
- LCOE benchmarking in development (Alpha)

### Integrations

- Integrates with CLIMATE for renewable resource projections
- Integrates with SPACE for spatial optimization grid
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ENERGY`
