---
name: geo-infer-climate
description: Climate data analysis. Use when loading/validating climate datasets, computing climate indices (SPI, heat index, PDSI-style drought index), detecting extreme events (heatwaves, cold spells, droughts, floods), analyzing temperature trends, fitting IDF precipitation curves, classifying Koppen climate zones, or running simplified SSP scenario projections.
difficulty: intermediate
estimated_time: 45min
---

# GEO-INFER-CLIMATE

## Instructions

### Core Capabilities

- **Data processing**: load (NetCDF/GRIB) and validate climate datasets from CMIP6, ERA5, NCEP, and observational sources
- **Climate indices**: SPI (gamma or normal distribution), heat index (Rothfusz), extreme indices, first-order Palmer-style drought index
- **Extreme events**: heatwave, cold spell, drought, flood, and compound event detection; return period estimation
- **Temperature trends**: OLS linear regression and Mann-Kendall non-parametric trend tests
- **Precipitation analysis**: IDF curve fitting, Gumbel return periods, gamma distribution fitting
- **Downscaling**: bias correction (linear, empirical quantile mapping) and interpolation-based downscaling
- **Projections**: simplified linear-scaling SSP scenario extrapolation (illustrative, not a model emulator)
- **Classification**: Koppen-Geiger climate zone classification

Not implemented: dynamical downscaling, ensemble projection methods, delta-change bias correction, RCP scenarios (SSP only).

### Key Imports

```python
from geo_infer_climate import (
    ClimateDataProcessor,
    ClimateIndicesCalculator,
    DownscalingMethods,
    ClimateProjections,
    ExtremeEventAnalyzer,
    ClimateImpactAssessor,
    ClimateClassifier,
    TemperatureTrendAnalyzer,
    PrecipitationAnalyzer,
)
```

## Examples

```python
import numpy as np
import pandas as pd
import xarray as xr

from geo_infer_climate import ClimateIndicesCalculator, ExtremeEventAnalyzer

calculator = ClimateIndicesCalculator()

precip = xr.DataArray(
    np.random.exponential(50, 240),
    dims=["time"],
    coords={"time": pd.date_range("2000-01-01", periods=240, freq="ME")},
)
spi = calculator.calculate_spi(precip, timescale=3, distribution="gamma")

analyzer = ExtremeEventAnalyzer()
temp = xr.DataArray(np.random.normal(20, 5, 365), dims=["time"])
heatwaves = analyzer.detect_heatwaves(temp, threshold_percentile=90.0, min_duration=3)
print(heatwaves["events_detected"], heatwaves["threshold_temp"])
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog CLIMATE` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- SPI, drought, and bias-correction routines expect a `time` dimension; per-grid-cell computation is handled along `time` regardless of axis position.
- `calculate_pdsi` is a first-order Palmer-style moisture anomaly proxy, not the full Palmer (1965) water-balance system.
- `project_future_climate` is a simplified linear trend extrapolation scaled by SSP factors; treat outputs as illustrative.
- Statistical downscaling is interpolation-only (linear/nearest); no regression or machine-learning downscaling.

### Integrations

- Integrates with WATER for hydrological climate impacts
- Integrates with AG for agricultural climate adaptation
- Integrates with ENERGY for renewable resource projections
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module CLIMATE`
