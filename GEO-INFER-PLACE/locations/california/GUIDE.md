# California statewide monitoring

This example asks where a fixed monitoring budget should be directed across
three environmental indicators: mapped fire perimeters, catalogued seismic
activity and year-end drought. Historical geography comes from primary sources.
Sensor readings are explicitly simulated. GNN means **Generalized Notation
Notation** in this workflow.

Read the [methods and source guide](../../../GEO-INFER-INTRA/docs/examples/california_statewide_monitoring.md)
for the measurement windows, inference equations, resource constraints and
interpretation limits.

| Entry | Purpose |
| --- | --- |
| [design.json](design.json) | Strict configuration for geography, temporal selection, sensors and budgets |
| [models/](models/) | Three explicit GNN sources and their independently produced factored artifacts |
| [california_observatory.py](../../examples/california_observatory.py) | Thin, create-only command interface |
| [california package](../../src/geo_infer_place/locations/california/) | Acquisition, geometry, field inference, monitoring and reports |

Install the optional `california` dependencies in a qualified workspace
environment. The example uses `geo_infer_act` and `geo_infer_space`; it does
not import the sibling GNN checkout. Run its exporter in the GNN environment.
Each stage refuses an existing output directory. Keep failed editions for
diagnosis and select a new output namespace for a corrected run.

From the GEO_INFER repository root:

```bash
uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --design GEO-INFER-PLACE/locations/california/design.json \
  --acquire GEO-INFER-PLACE/locations/california/generated/sources

uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --check-sources GEO-INFER-PLACE/locations/california/generated/sources

uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --design GEO-INFER-PLACE/locations/california/design.json \
  --sources GEO-INFER-PLACE/locations/california/generated/sources \
  --prepare GEO-INFER-PLACE/locations/california/generated/landscape

uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --landscape GEO-INFER-PLACE/locations/california/generated/landscape/landscape.json \
  --models GEO-INFER-PLACE/locations/california/models \
  --run GEO-INFER-PLACE/locations/california/generated/experiment

uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --landscape GEO-INFER-PLACE/locations/california/generated/landscape/landscape.json \
  --experiment GEO-INFER-PLACE/locations/california/generated/experiment/experiment.json \
  --report GEO-INFER-PLACE/locations/california/generated/report
```

For a complete edition after acquisition, retain all input bytes and run the
remaining stages through the same thin command interface:

```bash
uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --design GEO-INFER-PLACE/locations/california/design.json \
  --sources GEO-INFER-PLACE/locations/california/generated/sources \
  --models GEO-INFER-PLACE/locations/california/models \
  --workflow GEO-INFER-PLACE/locations/california/generated/edition

uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --check-workflow GEO-INFER-PLACE/locations/california/generated/edition

uv run python GEO-INFER-PLACE/examples/california_observatory.py \
  --check-report GEO-INFER-PLACE/locations/california/generated/edition/report
```

The edition contains `design.json`, copied `sources/` and `models/`, prepared
`landscape/`, sealed `experiment/`, `report/` and `workflow.json`. Output must be
new and outside the input folders. The receipt binds the complete retained file
inventory and the links between stages. An edition may be copied to a new path
and checked offline. This checks supplied-receipt integrity, not source
authenticity or fresh numerical replay. The Python `check_workflow` API also
accepts `current_execution=True` to compare the installed owners and runtime.

Reports now require a sealed aggregate experiment. Reporting rechecks its
execution receipt and recomputes the recorded Brier, log and entropy scores.
`trajectory.csv` records per-round, per-hazard resource expenditure, cell-average
and clipped-area-weighted Brier loss, selected cells and cumulative selected area.
`comparison.csv` and `COMPARISON.md` compare final losses to the initial belief
and rotating strategy for the same seed. Positive reduction or advantage means
lower Brier loss. Seeds are displayed separately; these are descriptive simulation
contrasts without confidence intervals or empirical calibration claims.

Selected-area coverage measures attempted observation, including missing
reports. Repeat visits consume resources but count once in cumulative coverage.
It is neither population coverage nor a count of successful physical sensors.
Primary-source capture logs at most three attempts per transient timeout or
network/service failure, bounded by the original batch deadline. Other failures
are immediate; an incomplete capture never receives a successful source receipt.

When using a separately preserved runtime, set `UV_PROJECT_ENVIRONMENT`,
`UV_PYTHON`, `UV_CACHE_DIR` and `TMPDIR` to its qualified external paths and use
`uv run --no-sync --python "$UV_PYTHON"`. Verify both Python prefixes. A source
receipt checks retained bytes and roster completeness; it does not establish
current service authenticity or convert simulated readings into empirical data.

The test inputs are declared numerical fixtures. Acquisition tests use real
local HTTP and real shapefiles, and refuse incomplete pages, source drift,
duplicate IDs and truncated catalogues. Monitoring tests compare exact policy
costs, exhaustive resource optima, source mutations and target-truth isolation.

```bash
uv run pytest GEO-INFER-PLACE/tests/unit/test_california*.py \
  GEO-INFER-PLACE/tests/unit/test_place_lazy_exports.py -q --no-cov
uv run pytest GEO-INFER-ACT/tests/unit/test_gnn_factored_batch.py \
  GEO-INFER-ACT/tests/unit/test_resource_allocation.py \
  GEO-INFER-SPACE/tests/unit/test_graph_statistics.py -q --no-cov
```

Generated raw captures, simulation editions and media assets remain under
`generated/`. Their absence from Git is not permission to delete them. Preserve
source receipts, failed logs and active runtime ownership when moving storage.
