# GEO-INFER cross-module integration guide

Composition connects explicit data contracts: source records and provenance in
DATA, ordered spatial states in SPACE, aware instants and temporal analysis in
TIME, and model-specific inference in BAYES or ACT. The workspace contains 45
modules. A module's declared dependencies and public exports are listed in its
[generated inventory](../../README.md); conceptual relationships do not establish
an import dependency or an implemented service.

## Establish axes before analysis

TIME owns `normalize_timestamp` and `normalize_datetime_index`. Inputs must identify
an instant with a timezone; TIME converts them to UTC. `TimeSeries` requires unique,
increasing axes. Local civil time needs an explicit timezone and, at a daylight
saving fold, an explicit choice of instant before construction.

SPACE's `H3StateSpace` preserves caller order. `align_h3_observations` aligns records
to that exact order and a supplied UTC time axis. It rejects duplicate normalized
cell/time pairs, unknown cells or times, mixed H3 resolutions, and nonfinite
observed values. Missing pairs become NaN; measured zero remains zero.
`max_entries` bounds the matrix before allocation.

The example uses real DATA parquet storage, a fresh reader, SPACE alignment, and
TIME analysis. All data are created locally and the numerical expectations are
calculated directly. Each aligned column represents one state; the temporal
analyzer operates on the explicitly selected column.

```python
import asyncio
from tempfile import TemporaryDirectory

import h3
import numpy as np
import pandas as pd
from geo_infer_data.core.storage import LocalFileBackend
from geo_infer_data.models.schemas import DataLineage, DatasetMetadata
from geo_infer_space import H3StateSpace, align_h3_observations
from geo_infer_time import TemporalAnalyzer, TimeSeries, normalize_timestamp

center = h3.latlng_to_cell(41.75, -124.2, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
space = H3StateSpace([neighbor, center])
axis = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
records = pd.DataFrame([
    {"cell": center, "timestamp": "2025-12-31T16:00:00-08:00", "value": 2.0},
    {"cell": center, "timestamp": axis[1], "value": 4.0},
    {"cell": center, "timestamp": axis[2], "value": 6.0},
    {"cell": neighbor, "timestamp": axis[0], "value": 0.0},
])
records["timestamp"] = pd.to_datetime(records["timestamp"].map(normalize_timestamp), utc=True)
metadata = DatasetMetadata(title="Local observation example", lineage=DataLineage(
    source="document fixture", process="local parquet round-trip", created_by="example"))
with TemporaryDirectory() as directory:
    writer = LocalFileBackend({"base_path": directory})
    identifier = asyncio.run(writer.store(records, metadata))
    reader = LocalFileBackend({"base_path": directory})
    restored = asyncio.run(reader.retrieve(identifier, {}))
    pd.testing.assert_frame_equal(restored, records)
    series = align_h3_observations(restored, state_space=space, timestamps=axis, max_entries=6)
assert tuple(series.data.columns) == space.cells
np.testing.assert_array_equal(series.data[center], [2.0, 4.0, 6.0])
assert series.data.loc[axis[0], neighbor] == 0.0
assert series.data.loc[axis[1:], neighbor].isna().all()
assert series.metadata == {"spatial_index": "h3", "crs": "EPSG:4326"}
trend = TemporalAnalyzer().detect_trend(TimeSeries(series.data[[center]]))
assert trend["trend_direction"] == "increasing"
assert np.isclose(trend["slope_per_sample"], 2.0)
assert np.isclose(trend["r_squared"], 1.0)
```

`DataLineage` records the example's origin and transformation. Storage identity,
H3 column order, UTC instants, and observed values survive the round-trip. Recording
provenance does not verify the correctness of an external source; production
callers must establish source custody and validate measurements independently.

## Select an integration pattern

| Pattern | Implement with | Contract to preserve |
| --- | --- | --- |
| Sequential pipeline | Owning DATA connector/storage APIs followed by SPACE/TIME APIs | Record IDs, lineage, CRS, aware instants, units |
| Parallel analyses | Independent analyses of owned copies, joined on declared keys | Identical spatial/time axes and explicit missing-data rules |
| Model feedback | ACT inference with a concrete model and action history | State order, observation order, probabilities, step duration |
| Application orchestration | Thin scripts calling owning packages | Finite deadlines, dependency profiles, retained diagnostics |

BAYES spatiotemporal callers must choose centroid coordinate order explicitly and
use elapsed seconds rather than raw epoch magnitudes. ACT consumes the same state
order as the data and its real transition operators. Neither interpolation nor
an inference model should silently replace absent measurements with zeros.
Observation gaps in fixed-step models require explicit prediction/action histories.

## Verify topology and transitions

`H3StateSpace.transitions()` returns sparse stay and diffuse operators indexed
`[next_state, current_state]`. The diffuse operator uses actual H3 neighbors,
including pentagon degree. Probability crossing the selected domain boundary stays
at the source. A bounded dense representation is available for small models.

```python
import h3
import numpy as np
from geo_infer_space import H3StateSpace

center = h3.latlng_to_cell(41.75, -124.2, 8)
cells = list(reversed(sorted(h3.grid_disk(center, 1))))
space = H3StateSpace(cells)
stay, diffuse = space.transitions()
assert space.cells == tuple(cells)
np.testing.assert_array_equal(stay.toarray(), np.eye(len(cells)))
np.testing.assert_allclose(np.asarray(diffuse.sum(axis=0)).ravel(), 1.0)
assert diffuse.min() >= 0
assert space.dense_transition_tensor(max_entries=2 * len(cells) ** 2).shape == (len(cells), len(cells), 2)
```

For an actual posterior/action trace and an independently calculated Bayesian
update, see the [composition regression](../../GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py)
and [SPACE/TIME interchange guide](../../GEO-INFER-TEST/docs/gnn_space_time_2026_09.md).

## Boundaries, dependencies, and verification

Keep reusable behavior in the owning package under `src/`; scripts assemble
operations. DATA's PostgreSQL, S3, MinIO, Redis, and raster integrations have
separate extras. IOT visualization requires its `visualization` extra. Cascadia
behavior is installed through PLACE's `cascadia` extra. An absent extra should
raise an actionable error at its operation boundary; installing a base package
does not prove a remote service or hardware backend works.

From the repository root, run the real composition tests and the maintained
example gate:

```bash
uv run --no-sync python -m pytest GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py
uv run --no-sync python GEO-INFER-TEST/validate_doc_examples.py
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category integration --workers 2 --timeout 600
```

The runner registers all 45 modules, direct module tests, Cascadia tests, and root
manuscript tests. Receipts retain source custody, selection and execution counts,
logs, JUnit, deadlines, and artifact hashes. Local results, hosted checks, real
services, and hardware acceptance are distinct evidence. See the
[0.4.0 migration guide](../../GEO-INFER-INTRA/docs/releases/0.4.0_migration.md)
for changed signatures and timezone conversion.
