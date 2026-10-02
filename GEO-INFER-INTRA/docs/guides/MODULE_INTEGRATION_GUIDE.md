# Module Integration Guide

Start an integration with its data contract: coordinate order/CRS, time identity,
state domain/order, units, missingness, duplicate policy, and allocation limits.
Keep domain behavior in owning `src/` packages and orchestration in a thin caller.
Explicit dependencies belong in the importing package's `pyproject.toml`.

## Align real module interfaces

SPACE's public adapter produces TIME's strict model. No universal orchestrator,
event bus, `SpatialAnalyzer`, or domain facade is needed for this pipeline.

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

## Extend the pipeline deliberately

1. Ingest with an actual DATA connector or IOT transport, retaining lineage and
   UTC event time. Test transport behavior locally rather than inventing a response.
2. Store with an owning backend and retrieve through a new instance to verify
   persisted bytes; a successful in-memory call is insufficient storage evidence.
3. Align against an explicit H3 domain/time axis. Aggregate duplicates upstream
   using a documented scientific rule; keep missing observations separate.
4. Build the inference interface the model expects. BAYES numeric inputs need
   declared scales, while ACT needs likelihood/transition matrices and schedules.
5. Compare to a small independent posterior, transition, or analytical oracle.
   Check state permutations, offset equivalence, missing values, and failure paths.

For repeated operations, bound each invocation and preserve the record needed to
reproduce it. An application can add scheduling or services after the synchronous
contract is verified; those deployment choices are not implicit package capabilities.

See [cross-module ownership](../architecture/cross_module_interaction.md),
[environmental integration](ENVIRONMENTAL_MONITORING_INTEGRATION.md), and
[SPACE composition](../../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md).
Run `uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --paths GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py`
from the repository root for the maintained real pipeline regression.
