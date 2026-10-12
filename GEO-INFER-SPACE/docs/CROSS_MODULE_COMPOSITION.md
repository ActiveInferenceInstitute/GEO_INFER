# SPACE and TIME composition

SPACE owns ordered spatial domains and H3 topology. TIME owns aware UTC timestamps,
ordered analytical series and model schedules. SPACE declares TIME as a runtime
dependency; TIME has no dependency on SPACE. DATA and IOT provide observations,
while BAYES and ACT consume explicitly aligned data.

## Align observations without changing their meaning

```python
import h3
import numpy as np
import pandas as pd
from geo_infer_space import H3StateSpace, align_h3_observations

center = h3.latlng_to_cell(41.75, -124.2, 8)
cells = sorted(h3.grid_disk(center, 1), reverse=True)[:2]
space = H3StateSpace(cells)
timestamps = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
observations = pd.DataFrame([
    {"cell": cells[0], "timestamp": "2023-12-31T16:00:00-08:00", "value": 0.0},
    {"cell": cells[1], "timestamp": timestamps[1], "value": 9.0},
])
series = align_h3_observations(
    observations, state_space=space, timestamps=timestamps,
)
frame = series.to_dataframe()
assert tuple(frame.columns) == space.cells
assert frame.iloc[0, 0] == 0.0
assert np.isnan(frame.iloc[0, 1])
assert frame.iloc[1, 1] == 9.0
```

`align_h3_observations(data, *, state_space, timestamps, cell_column="cell",
timestamp_column="timestamp", value_column="value", max_entries=1_000_000)`
returns a `geo_infer_time.TimeSeries`. Column names follow the caller's spatial
state order. Rows follow the supplied unique, increasing time axis, normalized
to UTC without losing pandas nanoseconds. Input records may arrive out of order;
the explicit axes determine where each observation belongs.

The adapter rejects duplicate normalized `(cell, instant)` pairs, observations
outside either axis, nonfinite observed values and allocations above the entry
budget. Missing pairs remain NaN. It never averages duplicates, imputes missing
values, invents time intervals or converts missingness to zero. H3StateSpace
requires canonical, unique cells at one resolution. An empty input table with
valid axes represents an entirely unobserved field.

For IOT records with `h3_index`, use `cell_column="h3_index"`. DATA records with
WGS84 geometries can first obtain cells through
`SpatialIndexingInterface.latlng_to_cell(latitude, longitude, resolution)`.
GeoJSON geometry coordinates use `[longitude, latitude]`; H3 uses latitude
first. Projected geometry must be explicitly transformed to WGS84 before indexing.

## Model boundaries

- **TIME:** `TimeSeries` requires aware, unique increasing timestamps. General
  analytical sampling may be irregular. Fixed-step `inference_schedule` rejects
  missing intervals; `action_observation_schedule` requires explicit prediction
  action histories for gaps. Neither fills observations.
- **BAYES:** Build spatial coordinates from `h3.cell_to_latlng(cell)` in frame
  column order, keeping the documented `[latitude, longitude]` convention. For
  `SpatioTemporalGP`, derive elapsed seconds from the first supplied UTC time,
  select only observed values with an explicit finite mask, and construct
  prediction inputs in `[latitude, longitude, elapsed_seconds]` order. Retain
  the time origin and state order when interpreting predictions.
- **ACT:** `H3StateSpace.transitions()` returns sparse stay/diffuse operators
  indexed `[next_state, current_state]` in `space.cells` order. Columns sum to
  one, omitted neighbors reflect at the boundary, and pentagon degree comes
  from real H3 topology. Bind those operators and the same observation/state
  order to the actual generative model. `dense_transition_tensor` checks an
  explicit allocation budget before creating `[next, current, action]` arrays.
  A numeric argmax is not policy inference.
- **GNN:** Keep the existing interchange schema and companion revision pin.
  State order, physical timestep and source provenance must survive the
  [consumer contract](../../GEO-INFER-ACT/docs/gnn_interchange.md). The aligned
  observation table introduces no new GNN wire format.

## Temporal analytics migration

SPACE's temporal and space-time analyzers normalize explicit aware timestamps
through TIME. Equivalent offset representations use identical UTC hour/day/month
bins; week bins use the ISO week year. Invalid timestamps and invalid or
mixed-resolution cells raise instead of silently changing the neighborhood.

The former `kriging_spatiotemporal` name was removed in 0.4.0. Call
`SpatioTemporalAnalyzer.interpolate_spatiotemporal` with the same arguments.
Its `method` is `softened_inverse_distance`: weights are
`1 / (1 + grid_distance) / (1 + time_difference_hours)` inside the requested
ranges. This estimator does not solve a kriging covariance system and does not
supply posterior uncertainty. Pentagon neighborhoods use `h3.grid_disk`.

See [TIME migration](../../GEO-INFER-TIME/docs/utc_timeseries_migration.md) for
naive timestamp, constructor, IO and precision changes.

## Verification

```bash
uv run python -m pytest GEO-INFER-SPACE/tests/unit/test_spatiotemporal_alignment.py
uv run python -m pytest GEO-INFER-TIME/tests/unit/test_timestamps.py
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME
```

The focused tests check exact values, preserved axes, missingness, UTC offset and
DST identity, bounded input consumption, real pentagon clustering and independent
weighted interpolation values. Module tests are local evidence. Network delivery,
installed-wheel verification and paired GNN acceptance have their own checks.
