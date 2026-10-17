---
name: geo-infer-econ
description: Geospatial economics and bioregional market modeling. Use when analyzing spatial economic patterns, bioregional markets, ecosystem-service credit auctions, spatial econometrics (SAR/SEM/SDM/SAC), or supply-demand modeling with geographic context.
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

# GEO-INFER-ECON

## Instructions

### Core Capabilities

- **Bioregional markets**: Ecosystem-service credit markets with double-auction / call-auction clearing (`BioregionalMarketDesign`, `EcosystemServicesMarkets`)
- **Spatial econometrics**: SAR / SEM / SDM / SAC estimation, GWR, Moran's I and Getis-Ord G* diagnostics (`SpatialEconometricsEngine`)
- **Supply-demand**: Consumer demand systems (OLS, SUR, AIDS), market structure and spatial market analysis
- **Impact assessment**: Policy engine and macroeconomic models (growth, business cycles, monetary, fiscal, trade)
- **Logistics integration**: Bridge to LOG module for supply chain analysis

### Key Imports

```python
from geo_infer_econ.bioregional.bioregional_markets import (
    BioregionalMarketDesign, BioregionalAsset, MarketParticipant, EcosystemServicesMarkets
)
from geo_infer_econ.core.econometrics_engine import SpatialEconometricsEngine
from geo_infer_econ.integrations.logistics_integration import LogisticsEconomicAnalyzer
from geo_infer_econ.microeconomics.market_structure import SpatialMarketAnalysis
from geo_infer_econ.macroeconomics import BusinessCycleModels, MonetaryPolicyModels
```

## Examples

Ecosystem-service credit market with call-auction clearing:

```python
import geopandas as gpd
from shapely.geometry import Polygon
from geo_infer_econ.bioregional.bioregional_markets import (
    BioregionalMarketDesign, BioregionalAsset, MarketParticipant, EcosystemServicesMarkets
)

boundary = gpd.GeoDataFrame(
    [{"region_id": "pnw"}],
    geometry=[Polygon([(-120.5, 45.5), (-120.0, 45.5), (-120.0, 45.0), (-120.5, 45.0), (-120.5, 45.5)])],
)
market = BioregionalMarketDesign(bioregion_boundary=boundary)
market.register_asset(BioregionalAsset(
    asset_id="a1", asset_type="forest", location=(45.5, -122.6), area_hectares=100,
    ecological_attributes={}, economic_attributes={}, ownership_type="private",
    management_regime="sustainable", ecosystem_services=["carbon"],
))
market.register_participant(MarketParticipant(
    participant_id="buyer_1", participant_type="buyer", location=(45.5, -122.6),
    assets_owned=[], market_preferences={}, budget_constraints={}, sustainability_goals={},
))
market.register_participant(MarketParticipant(
    participant_id="seller_1", participant_type="seller", location=(45.4, -122.5),
    assets_owned=["a1"], market_preferences={}, budget_constraints={}, sustainability_goals={},
))
credit = market.create_ecosystem_service_credit(
    asset_id="a1", service_type="carbon", quantity=10.0, quality_parameters={"quality": 0.9}
)

esm = EcosystemServicesMarkets(market_design=market)
esm.submit_buy_order("buyer_1", "carbon", quantity=5.0, max_price=100.0, location_preferences={})
esm.submit_sell_order("seller_1", credit.credit_id, min_price=10.0)
trades = esm.clear_market()
print(f"Trades executed: {len(trades)}")
```

Spatial autoregressive model fit and diagnostics:

```python
import numpy as np
from geo_infer_econ.core.econometrics_engine import SpatialEconometricsEngine

n = 16
W = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if abs(i - j) == 1:
            W[i, j] = 1.0
W /= W.sum(axis=1, keepdims=True)  # row-standardized weights

rng = np.random.default_rng(0)
X = np.column_stack([np.ones(n), rng.normal(size=n)])
e = rng.normal(size=n)
y = np.linalg.solve(np.eye(n) - 0.5 * W, X @ np.array([1.0, 2.0]) + e)

engine = SpatialEconometricsEngine({})
engine.fit(X, y, W, model_type="sar")
print("rho:", engine.coefficients_[0])
diagnostics = engine.spatial_diagnostics(engine.residuals, W)
print("Moran's I z-score:", diagnostics["z_morans"])
print("Getis-Ord G* z:", diagnostics["getis_ord_g_star_z"])
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog ECON` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Ecosystem-service credit markets and call-auction clearing are real implementations (`EcosystemServicesMarkets.clear_market`)
- SAC models are estimated by full ML over (rho, beta, lambda, sigma2); standard errors for spatial models are OLS-style approximations and are documented as such in `convergence_info`
- Logistics integration bridges ECON to LOG module (`LogisticsEconomicAnalyzer` degrades gracefully when LOG is unavailable)
- Logger used instead of print() for all library output
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON`

### Integrations

- **LOG** → Supply chain logistics cost modeling (optional, capability-flagged)
- **AG** → Agricultural commodity market analysis
- **TRANSPORT** → Transportation cost for trade flows
- **RISK** → Economic risk and insurance modeling
- **SPACE** → H3 spatial indexing and location analysis (optional, capability-flagged)