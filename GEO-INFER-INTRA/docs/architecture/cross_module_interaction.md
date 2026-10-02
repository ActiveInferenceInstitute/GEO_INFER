# Cross-Module Interaction

Modules compose through explicit Python values and owning-package interfaces.
TIME owns UTC identity and ordered time axes. SPACE owns H3 cell domains and
movement operators and depends on TIME for observation alignment. DATA owns
transport/storage; IOT owns sensor records; BAYES and ACT own model inference.
There is no reverse TIME dependency on SPACE.

## The SPACE → TIME seam

Long-form records become one bounded `TimeSeries` with caller-specified axes.
The fixture intentionally changes record order and timestamp offset while retaining
state order, an observed zero, and an absent pair.

```python
import h3
import numpy as np
import pandas as pd
from geo_infer_space import H3StateSpace, align_h3_observations

center = h3.latlng_to_cell(41.75, -124.2, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
space = H3StateSpace([neighbor, center])
times = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
records = pd.DataFrame([
    {"cell": center, "timestamp": "2023-12-31T16:00:00-08:00", "value": 2.0},
    {"cell": neighbor, "timestamp": times[0], "value": 0.0},
    {"cell": center, "timestamp": times[1], "value": 4.0},
])
series = align_h3_observations(records, state_space=space, timestamps=times)
assert list(series.data.columns) == [neighbor, center]
assert series.data.iloc[0, 0] == 0.0
assert np.isnan(series.data.iloc[1, 0])
assert series.duration.total_seconds() == 3600
```

## Model boundaries

For BAYES, build numeric coordinates in an explicitly declared order and scale;
filter with an observed-data mask rather than replacing missing pairs with zero.
A matrix of sensor values is not automatically a likelihood or a state posterior.
For ACT, supply stochastic `A` and `B`, preferences, priors, and a model interval;
preserve H3 state order in every matrix. A policy action and an argmax of a sensor
row have different meanings.

General time series may be irregular. Fixed-step ACT inference requires each
model interval; gaps need explicit actions via `action_observation_schedule`.
Adapters must reject duplicate pairs, unknown cells/times, invalid resolutions,
and ambiguous timestamps before inference. Exceptions cannot justify fabricating
fallback observations or silently switching to a different algorithm.

The unchanged GNN wire contract is `gnn-geo-infer/1`. Its companion revision pin
and source-custody procedure live in the
[GNN integration guide](../../../GEO-INFER-TEST/docs/gnn_space_time_2026_09.md).
The [composition contract tests](../../../GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py)
exercise real local HTTP transport, storage bytes, alignment, and ACT inference.
