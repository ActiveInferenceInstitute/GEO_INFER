# UTC time series and SPACE composition in 0.4.0

TIME owns timestamp validation. `normalize_timestamp` accepts an aware Python
datetime or an ISO-8601 string with an offset and returns UTC. Its scalar path
uses the standard library; importing it does not load pandas or analysis engines.
`normalize_datetime_index` preserves pandas nanoseconds and requires unique,
increasing UTC instants. Neither function guesses a timezone or sorts input.
TIME provides Python interfaces and transport adapters. The former draft
`api_schema.yaml` was retired because the package implements no HTTP server.
The scalar helper returns Python datetime precision; use the index helper
when nanosecond identity matters.

```python
import pandas as pd
from geo_infer_time import TimeSeries, normalize_timestamp

instant = normalize_timestamp("2023-12-31T16:00:00-08:00")
series = TimeSeries(pd.DataFrame(
    {"temperature": [2.0, 4.0]},
    index=pd.date_range(instant, periods=2, freq="h"),
))
assert series.resample("2h").to_dataframe().iloc[0, 0] == 3.0
```

## Migrate callers

- Replace naive fixtures with `datetime(..., tzinfo=UTC)`, offset-bearing ISO
  strings, or `pd.date_range(..., tz="UTC")`. Local wall-clock readings need
  an explicitly chosen source timezone before calling TIME; daylight-saving
  ambiguity belongs to the source adapter.
- Pass a timezone-preserving `DatetimeIndex`, not an aware Series' `.values`,
  which loses its timezone. An integer index is no longer interpreted as
  nanoseconds since the epoch.
- Aggregate duplicates explicitly before construction. Reversed timestamps
  raise rather than reordering their observations. NaT is invalid; NaN values
  remain supported as missing observations.
- `start_time` and `end_time` return pandas `Timestamp`, a datetime subclass;
  `duration` returns pandas `Timedelta`. These preserve nanoseconds. Convert
  explicitly with `.to_pydatetime()` when a consumer requires microsecond
  precision.
- IO, factories and store query bounds use the same UTC rules. CSV writers
  provide a named time index; CSV/JSON/Parquet readers reject ambiguous time.
  JSON writers serialize UTC index strings directly to preserve nanoseconds,
  including identities pandas would otherwise truncate to microseconds. Metadata
  sidecars are restored on reading unless callers explicitly override them.

## Explicit schedules and missing observations

General analytical series may have irregular intervals. Sequential inference
has a stricter contract: `inference_schedule(..., step_seconds=...)` requires
every model interval, and `action_observation_schedule` requires explicit action
histories for prediction through gaps. Neither contract imputes observations.

SPACE's `align_h3_observations` returns a TIME series with columns in the caller's
`H3StateSpace.cells` order and rows in an explicitly supplied UTC time axis.
Absent pairs remain NaN; zero is observed data. See the
[SPACE composition guide](../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md)
for an executable example and the BAYES/ACT boundary rules.

Linear `TemporalAnalyzer.detect_trend` reports `stable` for numerical constants,
including negative values. `trend_strength` remains the absolute slope per
observation; `slope_per_sample` gives its sign, and `r_squared` describes fit
quality separately. These quantities use sample position, so elapsed-time
rates require an explicit regular grid or a separate elapsed-time regression.
Linear analysis requires at least two finite observations. Forecasting and sample-lag analyses require a complete regular axis;
resample or interpolate explicitly before fitting. Select one value column before
univariate analysis. See [method contracts](method_contracts.md) for holdout,
entropy, configuration, transport unit, and visualization migrations.

## Verification

```bash
uv run python -m pytest GEO-INFER-TIME/tests/unit/test_timestamps.py GEO-INFER-TIME/tests/unit/test_lazy_public_imports.py GEO-INFER-TIME/tests/unit/test_linear_trend_contract.py
uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME
```
