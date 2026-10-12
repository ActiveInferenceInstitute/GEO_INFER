---
name: geo-infer-risk
description: Geospatial risk modeling including catastrophe models and exposure analysis. Use when assessing spatial risk, building catastrophe models, analyzing exposure/hazard/vulnerability, or computing portfolio risk metrics.
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-data
  recommended:
    - geo-infer-bayes
    - geo-infer-math
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-RISK

## Instructions

### Core Capabilities

- **Catastrophe models**: Spatial correlation and directed multi-hazard interactions
- **Risk engine**: Moran's I, Geary C, Monte Carlo loss calculation
- **Exposure modeling**: `file://` loading of CSV/JSON/Parquet; other source
  schemes (e.g. `api://`) require a configured data connector and are
  explicitly rejected rather than silently faked
- **Hazard modeling**: Spatial hazard assessment and mapping
- **Vulnerability**: Bayesian uncertainty quantification

### Key Imports

```python
from geo_infer_risk.core.risk_engine import EnhancedRiskEngine
from geo_infer_risk.core.catastrophe_models import (
    EnhancedCatastropheModel,
    MultiHazardInteractionMatrix,
)
from geo_infer_risk.core.exposure_model import EnhancedExposureModel
from geo_infer_risk.core.hazard_model import EnhancedHazardModel
```

## Examples

Every snippet below runs against the current API.

Reproducible catastrophe simulation. The seed lives on the config, and all
draws come from the model's own generator, so a run replays exactly and never
disturbs the caller's `numpy.random` stream:

```python
from geo_infer_risk.core.catastrophe_models import (
    CatastropheConfig,
    EnhancedEarthquakeModel,
)

config = CatastropheConfig(
    simulation_years=50, spatial_correlation=False, random_seed=7
)
model = EnhancedEarthquakeModel(config=config)
model.model_parameters = {"mean_depth": 15.0}
events = model.simulate_events(200)
```

Estimate compound annual exceedance along a directed hazard chain. Zero
off-diagonal interaction recovers independent joint exceedance; positive
interaction raises the downstream conditional probability:

```python
from geo_infer_risk.core import MultiHazardInteractionMatrix

interactions = MultiHazardInteractionMatrix(
    ["earthquake", "fire_following", "flood"],
    [[1.0, 0.5, 0.0], [0.0, 1.0, 0.4], [0.0, 0.0, 1.0]],
)
compound_probability = interactions.compound_exceedance_probability(
    {"earthquake": 0.1, "fire_following": 0.2, "flood": 0.3}
)
```

Risk metrics from an event loss table. `exposure_years` is how many years the
table spans. `calculate_aal` requires it; the exceedance metrics accept it
optionally, and omitting it inflates every per-year figure (a warning says so):

```python
import pandas as pd
from geo_infer_risk.utils.risk_metrics import (
    calculate_aal,
    calculate_pml,
    calculate_annual_aggregate_exceedance_probability,
)

losses = pd.DataFrame(
    {
        "event_id": [event["event_id"] for event in events],
        "hazard_type": ["earthquake"] * len(events),
        "loss": modelled_losses,  # one loss per event
    }
)

aal = calculate_aal(losses, exposure_years=50.0)["total"]
pml_25 = calculate_pml(losses, return_period=25, exposure_years=50.0)
aep = calculate_annual_aggregate_exceedance_probability(
    losses, threshold=5e6, num_years=20_000, random_seed=7, exposure_years=50.0
)
```

`calculate_pml` warns when the requested return period is longer than the
record can resolve; the value is then clamped to the largest observed loss and
understates the tail.

## Reproducibility

Every stochastic entry point in this module takes a `random_seed` and routes it
through `geo_infer_risk.utils.rng.resolve_rng`, which accepts `None`, an `int`,
a `SeedSequence`, a `BitGenerator`, or a `numpy.random.Generator`, and always
returns a `Generator`; a `RandomState` or the `numpy.random` module raises
`TypeError`. Consequences worth knowing:

- Passing an `int` makes a run replayable; `0` is a valid seed.
- Passing a `Generator` threads one stream through a whole pipeline.
- `None` means OS entropy, so results are *not* replayable. Calling
  `np.random.seed(...)` does not make them so: this module never reads the
  process-wide singleton, and never advances it either.
- For independent parallel streams use
  `geo_infer_risk.utils.rng.spawn_rng(seed, n)` rather than `seed`, `seed + 1`,
  ... which carries no independence guarantee.
- At boundaries that accept only an `int` seed, such as scikit-learn's
  `random_state`, use `geo_infer_risk.utils.rng.derive_int_seed`.

## Guidelines

- Production paths require configured data sources and do not fabricate risk inputs.
- Spatial correlation uses Cholesky decomposition
- Directed interaction entries are bounded to `[-1, 1]`; ordered compound
  exceedance uses the configured source-to-target chain
- Risk aggregation uses real Moran's I and Monte Carlo
- Exceedance-probability curves use the Weibull plotting position and
  interpolate loss as a function of exceedance probability; return periods
  beyond the record are clamped, not extrapolated
- `calculate_aal` requires `exposure_years`; pass it to the other per-year
  metrics too (OEP, AEP, and the annualized EP curve), whose fallback treats
  the table as spanning one year
- Simplified conversions, stand-in probabilities, and loss-tail multipliers
  are catalogued in the Honest Capability Register below; source sites point
  here instead of re-explaining themselves
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module RISK`

### Integrations

- **BAYES** → Bayesian uncertainty quantification
- **ECON** → Economic loss and insurance modeling
- **CLIMATE** → Climate-driven hazard projections
- **SPACE** → Spatial correlation of hazards
- **AG** → Crop loss risk assessment

## Honest Capability Register

The simplified conversions and multipliers in this module are deliberate
engineering stand-ins, not calibrated geoscience or actuarial models. They are
listed once here; the inline code comments point back to this register.

- **Magnitude → PGA (GMPE stand-ins)** —
  `EnhancedHazardModel._magnitude_to_intensity` (banded step table,
  `core/hazard_model.py`) and `EnhancedEarthquakeModel._magnitude_to_pga`
  (log-distance attenuation, `core/catastrophe_models.py`). Neither is a
  calibrated ground-motion prediction equation; site effects in the
  catastrophe loss path are fixed soil multipliers (soft 1.3, rock 0.8).
- **Secondary perils (liquefaction, landslide)** —
  `EnhancedHazardModel._calculate_liquefaction_probability` and
  `_calculate_landslide_probability` return magnitude-band probability tables;
  they ignore soil conditions, saturation, and slope.
- **Storm surge** — `EnhancedHurricaneModel._calculate_storm_surge` is linear
  in wind speed above 30 m/s with a ±30% random tidal factor; it includes no
  bathymetry, pressure forcing, or track integration.
- **VaR / CVaR multipliers** — `PropertyInsuranceModel.estimate_losses` uses
  Gaussian quantiles (mean + 1.645σ and mean + 2.063σ);
  `CatastropheInsuranceModel.estimate_losses` applies flat 2.0× and 3.0×
  multipliers to total expected loss. Neither is an empirical
  loss-distribution tail, and the catastrophe multipliers ignore per-peril
  aggregation.
- **Model calibration** — only `cross_validation` against a mean-loss
  baseline is implemented (`EnhancedRiskEngine.calibrate_models`); `bayesian`
  and `maximum_likelihood` are rejected with a documented error rather than
  accepted as options that would hard-fail.
- **Catastrophe base-rate tables** — `CatastropheInsuranceModel` premium
  models (`core/insurance_models.py`) price per-peril coverage from
  hand-set latitude-band base rates: hurricane `_hurricane_model`
  (0.02/0.01/0.005 by latitude band), earthquake `_earthquake_model`
  (0.015/0.01/0.005 by |lat| band), flood `_flood_model`
  (0.025/0.015/0.005 by |lat| band), wildfire `_wildfire_model`
  (0.02/0.015/0.005 by |lat| band). None accounts for real exposure
  drivers (elevation, bathymetry, fuel load, fault proximity). Separately,
  `_estimate_catastrophe_loss` applies flat per-peril expected-loss
  fractions (hurricane 0.01, earthquake 0.008, flood 0.012, wildfire
  0.015; other 0.01) of coverage limit, independent of location or the
  premium rate tables.

