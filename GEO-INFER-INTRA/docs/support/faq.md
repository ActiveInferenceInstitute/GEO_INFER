# Frequently asked questions

## What is GEO-INFER?

GEO-INFER is a Python 3.11+ geospatial inference monorepo with 45 owning packages.
SPACE supplies spatial operations, TIME temporal contracts and analysis, DATA
transport and storage, and domain modules apply these components. ACT implements
active inference models; using another package does not require describing every
operation as active inference. Start with the [module catalogue](../modules/index.md)
and each module's generated README for current exports and dependencies.

## How do I install and run it?

Use the checkout's root `pyproject.toml`, `uv.lock`, and `.python-version`. The root
is a virtual uv workspace; it is not an installable aggregate distribution.

```bash
uv sync --locked --all-packages --all-extras --all-groups
uv run --no-sync python GEO-INFER-TEST/rewrite_readme_agents.py --check
```

Full extras include hardware-dependent packages. CPU-only CI declares its profile
explicitly; a failed accelerator build is not evidence about CPU behavior. Select
an owning package and its documented extras for a smaller installation. DATA
separates PostgreSQL, S3, MinIO, Redis, and raster extras; IOT separates visualization;
PLACE owns the `cascadia` extra. See the
[installation guide](../getting_started/installation_guide.md) and
[migration guide](../releases/0.4.0_migration.md).

## Why did a timestamp that worked before stop working?

A timestamp must identify an instant: naive datetimes and implicit numeric epochs
are rejected. Use explicit ISO-8601 offsets or timezone-aware datetime objects.
TIME normalizes to UTC. A `TimeSeries` axis must also be unique and increasing;
streams declare their own arrival-order policy. To convert local civil times,
localize them with the source's timezone and resolve ambiguous/nonexistent local
times explicitly before converting to UTC.

```python
from datetime import UTC, datetime
import numpy as np
import pandas as pd
from geo_infer_time import TimeSeries, normalize_timestamp

instant = normalize_timestamp("2026-01-01T00:00:00-08:00")
assert instant == datetime(2026, 1, 1, 8, tzinfo=UTC)
axis = pd.date_range(instant, periods=3, freq="h")
series = TimeSeries(np.array([0.0, 2.0, 4.0]), timestamps=axis)
assert str(series.timestamps.tz) == "UTC"
assert series.duration.total_seconds() == 7200
try:
    normalize_timestamp("2026-01-01T00:00:00")
except ValueError:
    pass
else:
    raise AssertionError("An unspecified timezone must be rejected")
```

## How should I handle missing observations?

Preserve them until a documented analysis or model handles the gap.
`SPACE.align_h3_observations` produces NaN for absent cell/time pairs and retains
observed zero. It checks the supplied H3 state order and time axis before allocation.
It does not invent a measurement. TIME linear trend analysis requires finite
observations, so choose and document a missing-data policy before that operation.
Fixed-step ACT models require explicit prediction/action histories for gaps.

## How do I avoid coordinate and ordering mistakes?

Record the CRS and units. GeoPandas' `set_crs` labels known coordinates; `to_crs`
transforms them. Neither repairs an unknown coordinate system. H3 point functions
use latitude, longitude; GeoJSON coordinates use longitude, latitude. Preserve
`H3StateSpace.cells` through arrays, artifacts, and posterior interpretation.
An H3 cell represents an area; a centroid is an explicit approximation, not the
original sensor location. See the
[integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md).

## How do I check temporal trends and model uncertainty?

Use `TemporalAnalyzer` with a real `TimeSeries`. Its linear slope is per sample;
regular sampling must be established before interpreting that slope per hour or
per day. Fit quality and slope magnitude answer different questions. BAYES and ACT
have model-specific uncertainty APIs; select a concrete likelihood, state space,
transition model, and priors from their current tests rather than assuming a
universal `predict_with_uncertainty` facade.

## Why is a spatial read using the slower path?

DATA can read vectors through GeoPandas or a compatible DuckDB spatial extension.
Provision that extension explicitly and use `require_duckdb=True` when the fast
path itself is the acceptance requirement. Automatic fallback is a convenience,
not a passing DuckDB test. The provisioning CLI and supported options are described
in [DATA's operating guide](../../../GEO-INFER-DATA/SKILL.md).

## How can I improve speed without weakening verification?

Profile the actual operation on representative data. Bound allocations, use real
spatial indexes, and partition work by declared keys. Prefer tiny analytical
fixtures and awaited events over arbitrary sleeps in tests. The unified runner's
`--workers` option runs isolated module processes with bounded concurrency.
Performance claims need retained measurements; enabling an optional GPU package
or detecting a device does not establish acceleration.

## How do I diagnose an import or test failure?

Confirm the selected interpreter, locked environment, owning package, and declared
extra. Run the smallest owning regression before the fleet command. Unknown
modules, invalid deadlines, entirely empty selections, and missing or stale JUnit
are failed runs. Assertion failures are not automatically retried. Retain the
attempt receipt and logs when reporting a failure.

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module TIME --category unit --workers 1 --timeout 600
uv run --no-sync python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
```

## How do I handle private data and operational deployment?

Choose a connector/storage configuration that meets the actual data policy. Local
execution does not prevent configured connectors from making network requests.
Encryption, authentication, access logging, and service availability need direct
verification in the deployed configuration. Keep credentials outside examples,
repositories, URLs, and diagnostic attachments. CPU CI does not establish licensed
data access, advisor authentication, or hardware acceleration.

## How do I update or report a bug?

Read the root [changelog](../../../CHANGELOG.md) and migration notes, preserve local
work, update with `git pull --ff-only`, and synchronize the lock. Report the revision,
interpreter, dependency profile, minimal input, expected result, and retained
receipt through the repository's issue tracker. Remove private source data and
credentials from that reproduction. For further local triage, see the
[support hub](index.md).
