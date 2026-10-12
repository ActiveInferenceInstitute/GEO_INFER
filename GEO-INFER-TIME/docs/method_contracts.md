# TIME method contracts and orchestration

TIME exposes Python data models, analysis engines, validated schedules, and async
transport adapters. Public components load lazily from `geo_infer_time` and
`geo_infer_time.core`; importing timestamp normalization alone does not import
pandas, statsmodels, or matplotlib. No HTTP service is implemented by TIME.

## Data, axis, and ownership

`TimeSeries` owns its tabular buffers and nested metadata. UTC timestamps must be
aware, unique, increasing, and nonmissing; normalized duplicate instants fail.
Value columns must be unique. Missing values remain allowed in the data model:
`resample(..., method="sum")` leaves an absent or entirely missing bin as NaN,
while a measured zero remains zero. `slice` and store query bounds use the same
UTC rules, including rejecting reversed bounds. Empty slices remain empty data.

`TimeSeriesReader`, `TimeSeriesWriter`, `read_timeseries`, and `write_timeseries`
support CSV, JSON, and Parquet. Metadata sidecars round-trip nested metadata and
spatial context; explicit caller metadata overrides the corresponding sidecar
entry. JSON indexes are serialized as UTC strings to preserve nanosecond identity.
Records-oriented JSON includes the timestamp column. An untyped numeric epoch is
rejected by readers. Reading does not sort or merge records.
Overwriting data without writing metadata removes an existing sidecar; this
includes `write_metadata=False`. New observations cannot inherit the previous
series' provenance from a stale sidecar.

`create_timeseries` preserves a nanosecond start. `detect_frequency` returns only
an exact regular cadence; an irregular axis returns None. This replaces the
previous median-interval heuristic, which could label irregular observations as
regular daily data. `fill_gaps` requires a grid retaining every observed timestamp;
choose `TemporalInterpolator.resample_interpolate` when off-grid anchors must be
interpolated onto another grid. Both methods accept `max_points` (default
1,000,000). Fixed-step grids are checked before index allocation; calendar grids
are checked after pandas constructs their index. This is not a universal memory
budget for every pandas operation.

`align_timeseries` supports outer/inner joins and explicit None, linear, forward,
or backward filling. Its default remains forward filling; callers composing
missing observations should pass `fill_method=None` unless imputation is intended.

## Analysis and numerical interpretation

Univariate analytical engines reject a multi-column series. Select a value column
explicitly rather than relying on the first column. Complete finite observations
are required for trend fitting, forecasting, spectral/autocorrelation analysis,
entropy matching, and change-point analysis. They never drop missing samples and
then reinterpret the remaining positions as a regular axis. Rolling statistics
retain supported missing-window behavior, with timestamps accompanying each
surviving statistic.

| Methods | Axis and result contract |
| --- | --- |
| `detect_trend` | Linear, polynomial, and moving-average sample-position trends. Linear `slope_per_sample` and absolute `trend_strength` are value change per sample; `r_squared` is fit quality. Irregular axes do not turn these into elapsed-time rates. |
| `detect_seasonality`, `decompose`, `test_stationarity`, `calculate_autocorrelation`, `detect_periodicity` | Require a regular UTC cadence. Decomposition uses a caller period or a documented cadence mapping; unsupported mappings require an explicit period. Seasonal periods count samples. FFT frequencies are cycles per sample. Constant or underspecified statistical tests return an error or reject input rather than providing a passing inference. |
| `detect_anomalies` | Z-score, IQR, rolling z-score, and deterministic isolation forest. Global z-scores have no additive epsilon changing measurement scale. Threshold applies to the first three methods; isolation uses the fixed 0.05 contamination policy. |
| `detect_change_points` | CUSUM and binary segmentation respect minimum segment length, including the exact two-segment boundary. These are heuristic segmentation methods, not a probabilistic change-point posterior. |
| `calculate_cross_correlation`, `calculate_granger_causality` | Require identical UTC axes and a regular cadence, without positional truncation. Positive cross-correlation lag means series 1 leads series 2. Granger F tests report singular/underspecified designs as errors; statistical predictive association is not causal identification. |
| `validate_forecast` | Requires equal nonempty finite arrays and one ordered confidence interval per observation. MAPE is None if actual values contain zero. Both-zero SMAPE terms are zero; tiny values are not replaced by epsilon. |
| `calculate_rolling_statistics` | Positive sample window and named statistics. Each statistic reports surviving timestamps, values, and latest value; `valid_observations` counts complete windows. |
| `compute_temporal_entropy` | Histogram Shannon entropy uses probability mass, in bits. Sample entropy excludes self-matches with matched template origins; approximate entropy includes them and averages log match frequencies. `embedding_dim`, `tolerance`, and `max_points` are explicit. Default pairwise-work limit is 2,000 observations. Undefined sample matches yield None; a zero entropy remains zero. |

`TemporalStatistics` offers summary, regular/seasonal differencing, Ljung–Box,
Jarque–Bera, Durbin–Watson, Hurst R/S estimation, information criteria, and residual
diagnostics. Invalid counts and nonfinite vectors fail before fitting. Undefined
constant normality, zero residual energy, and zero segment variance are reported
as undefined. Coefficient of variation is None when the mean is zero. Scale-free
diagnostics normalize their computational scale, so tiny residual units do not
change their ratios. Hurst and residual quality classifications are finite-sample
heuristics and statistical diagnostics, not forecast guarantees.

`EventDetector.detect_anomalies` deliberately analyzes observed values and retains
their original timestamps while omitting NaNs; it is a marginal-value detector.
`detect_changepoints` requires a complete univariate series because its windows
are positional. Its two adjacent windows may end exactly at the last observation.

## Forecast configuration and failure behavior

`ForecastingEngine` supports linear, moving-average, ARIMA, and Holt–Winters models.
It requires a complete, univariate, regularly sampled UTC series with at least two
observations. Horizons and moving windows are positive integer sample counts.
Future timestamps extend the observed cadence, including calendar-based UTC
frequencies and explicit custom calendar offsets. An explicit pandas frequency
is verified against every timestamp and retained even for a two-observation
axis; a month-end interval is not replaced by its elapsed-day duration.
`detect_frequency` returns the frequency alias for diagnostics; a custom offset's
alias does not serialize its calendar parameters. Validation requires the
forecast result's explicit `model_type`, refits the
same model and configuration on a chronological prefix, scores the untouched
holdout, and propagates backend failures. It is one holdout, not rolling-origin
cross-validation.

`AdvancedForecastingEngine(config=...)` accepts only `arima_fit_kwargs`,
`smoothing_fit_kwargs`, and `seasonal_period`. Fit dictionaries are passed to the
actual declared statsmodels backend; unknown top-level keys are rejected. This
replaces arbitrary configuration that was stored but ignored. The configuration
is copied deeply, so later caller mutation cannot change a fit. Standalone
`fit_arima_forecast` and `fit_exponential_smoothing_forecast` also accept
`fit_kwargs`; numeric vectors represent sample positions rather than UTC records.
ARIMA results retain convergence status and captured fit diagnostics.
`detect_trend_seasonality(..., period=...)` respects the explicit period, then the
configured period, then supported cadence inference. It requires two full cycles.
A daily input is no longer silently decomposed with a hardcoded 12-sample period.

## Streams and schedules

`ReplayIngestAdapter` is the explicit offline source. WebSocket and Kafka adapters
open actual connections, use finite connection/receive/close deadlines, and never
substitute generated records on failure. Numeric transport timestamps require
`config={"timestamp_unit": "s"}` or `"ms"`; magnitude guessing is removed. Kafka
broker timestamps are explicitly milliseconds even at epoch zero. Kafka manually
commits only after successful downstream processing and acknowledgement. Its
unit tests exercise an HTTP-free Kafka client boundary; a live broker check is
separate service evidence.

`StreamProcessor` normalizes event time and retains its explicit ordering and
watermark policy. Active and late buffers, window history, and sliding evaluation
work have separate limits. `max_window_evaluations` defaults to 10,000; exceeding
it fails before allocating sliding results or firing handlers. Aggregation
callbacks must return finite values. Ingest failures and capacity exhaustion do
not acknowledge Kafka records. Returned history/late snapshots and nested ingest
metadata have separate ownership. Zero watermark delay remains an explicit zero.

`inference_schedule` requires every fixed model step. `action_observation_schedule`
accepts gaps only with exactly the caller-supplied action history needed to reach
the next observation. Both retain nanosecond timestamp identities; model steps
must be representable at microsecond precision. See
[action histories](action_observation_schedule.md) and
[transport migration](streaming_migration.md).

## Visualization and executable surfaces

Install `geo-infer-time[visualization]` to use `TemporalVisualization`; absence
raises an explicit extra error. Internal matplotlib import failures propagate.
Figures use instance-scoped style contexts; invalid styles propagate their
configuration error without substituting another style. Optional timestamps are validated UTC
axes. Forecast plots and dashboards require historical and forecast timestamp
axes together, and forecasts must follow history. `create_dashboard` accepts the
`ForecastingEngine` result directly through its `forecast` and `timestamps` keys, including ARIMA
`lower_bound` and `upper_bound` intervals.
`plot_rolling_statistics(..., rolling_timestamps=...)` can represent missing
windows correctly; omitted rolling timestamps assume a trailing suffix.
Constant dashboard data display an undefined autocorrelation label.

The two example CLIs execute only from `main()`, use seeded tiny fixtures, and
return real numerical output. Backend errors cause CLI failure. Their selected
calls do not establish that every TIME method is universally correct.

## Verification and evidence limits

Run the analytical contracts before the whole module and consumers:

```bash
uv run python -m pytest GEO-INFER-TIME/tests/unit/test_method_contracts.py
uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME
uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
uv run python GEO-INFER-TIME/examples/basic_temporal_analysis.py
uv run python GEO-INFER-TIME/examples/demo_all_methods.py
```

The source inventory records every callable and SHA, distinguishing source review
from numerical oracle evidence and ordinary suite execution. A passing local
suite does not establish live Kafka service delivery, all data distributions,
long-run forecast calibration, or hosted CI. Those require their own receipts.
