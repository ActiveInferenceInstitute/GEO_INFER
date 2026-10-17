---
name: geo-infer-ag
description: Precision agriculture and soil health modeling. Use when analyzing soil health, crop water usage (FAO-56), carbon sequestration (IPCC Tier 1), precision farming, or agricultural land management.
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-data
  recommended:
    - geo-infer-bayes
    - geo-infer-time
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-AG

## Instructions

### Core Capabilities

- **Soil health**: USDA-style soil indicator scoring (`SoilHealthModel`)
- **Carbon sequestration**: IPCC Tier 1 / Tier 2 methodology (`CarbonSequestrationModel`)
- **Water usage**: FAO-56 reference-ET crop water requirements (`WaterUsageModel`)
- **Yield prediction**: ML, statistical, and GDD-based process models (`CropYieldModel`)
- **Land management**: field boundary analysis (`FieldBoundaryManager`), sustainability assessment (`SustainabilityAssessment`), seasonal analysis (`SeasonalAnalysis`)

All models follow the same contract: construct with model options, optionally
`fit()` on training data, then call `predict({"field_data": ..., ...})` with a
dict of pandas/geopandas data frames. Required keys per model type are listed
in each model's docstring (`model.required_inputs`).

### Key Imports

```python
from geo_infer_ag.models.soil_health import SoilHealthModel
from geo_infer_ag.models.carbon_sequestration import CarbonSequestrationModel
from geo_infer_ag.models.water_usage import WaterUsageModel
from geo_infer_ag.models.crop_yield import CropYieldModel
from geo_infer_ag.core.field_boundary import FieldBoundaryManager
from geo_infer_ag.core.sustainability import SustainabilityAssessment
from geo_infer_ag.core.seasonal_analysis import SeasonalAnalysis
from geo_infer_ag.api.agricultural_api import AgriculturalAPI
```

## Examples

Water requirements (FAO-56 reference-ET approach):

```python
import pandas as pd
from geo_infer_ag.models.water_usage import WaterUsageModel

model = WaterUsageModel(crop_type="corn", model_type="reference_et")
weather = pd.DataFrame(
    {"temperature": [22.0], "solar_radiation": [22.0], "humidity": [65.0],
     "wind_speed": [2.0], "precipitation": [1.5]},
    index=pd.date_range("2026-06-01", periods=1, freq="D"),
)
fields = pd.DataFrame({"area_ha": [100.0]})
result = model.predict({"field_data": fields, "weather_data": weather})
print(result["summary"]["mean_water_requirement_mm"])          # mm/ha
print(result["summary"]["total_irrigation_requirement_m3"])    # m3
```

Carbon sequestration (IPCC Tier 1 defaults, management/soil modifiers optional):

```python
import pandas as pd
from geo_infer_ag.models.carbon_sequestration import CarbonSequestrationModel

carbon = CarbonSequestrationModel(model_type="tier1", time_horizon=20)
fields = pd.DataFrame({"crop_type": ["corn"], "area_ha": [100.0]})
result = carbon.predict({"field_data": fields})
print(result["summary"]["total_annual_sequestration"])   # t C/yr
print(result["summary"]["total_co2e_sequestration"])     # t CO2e over horizon
```

Yield prediction (fit on historical records, predict on new fields):

```python
import pandas as pd
from geo_infer_ag.models.crop_yield import CropYieldModel

historical = pd.DataFrame({
    "ph": [6.1, 6.4, 6.8, 7.0, 6.3, 6.6],
    "nitrogen_kg_ha": [120.0, 150.0, 170.0, 180.0, 140.0, 165.0],
    "precip_mm": [450.0, 500.0, 540.0, 560.0, 480.0, 530.0],
    "yield": [9.0, 10.2, 11.4, 11.8, 9.8, 11.0],  # t/ha
})
model = CropYieldModel(crop_type="corn", model_type="machine_learning")
model.fit({"field_data": historical}, target_column="yield")
prediction = model.predict({"field_data": historical})
print(prediction["summary"]["mean_yield"])  # t/ha
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog AG` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Models are deterministic on fixed inputs; ML variants need `fit()` first
- Carbon Tier 1 uses crop-specific default rates; pass `management_data`/`soil_data` for Tier 2 modifiers
- Water requirements use FAO-56 crop coefficients with Penman-Monteith reference ET
- Field areas are computed in a true equal-area projection (EPSG:6933), never Web Mercator
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AG`

### Integrations

Cross-module links (the counterpart modules own the implementation; AG itself
contains no H3 code). Note: AG ships an `AgriculturalAPI` HTTP client targeting
`https://api.geo-infer.ag` — that domain does not resolve publicly, so its
network methods (`get_crop_data`, `get_soil_data`, `get_weather_forecast`,
`get_precision_agriculture_data`) raise RuntimeError unless a reachable service
is configured; the offline `analyze_crop_yield` path works without it.

- **CLIMATE** → Precipitation projections for irrigation planning
- **WATER** → Irrigation water demand modeling
- **SPACE** → H3-based field tessellation
- **RISK** → Crop loss risk assessment
- **ECON** → Agricultural market pricing and supply chain

`WaterUsageModel` computes requirements only; it does not produce schedules.
