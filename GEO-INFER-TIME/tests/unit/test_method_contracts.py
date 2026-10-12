"""Analytical TIME oracles and failure boundaries across public method families."""

from datetime import UTC, datetime, timedelta
import json
import math

import numpy as np
import pandas as pd
import pytest

from geo_infer_time import (
    AdvancedForecastingEngine,
    EventDetector,
    ForecastingEngine,
    ReplayIngestAdapter,
    StreamProcessor,
    TemporalAnalyzer,
    TemporalInterpolator,
    TemporalStatistics,
    TimeSeries,
)
from geo_infer_time.db import InMemoryStore
from geo_infer_time.io import read_timeseries, write_timeseries
from geo_infer_time.utils import create_timeseries, detect_frequency, fill_gaps


def series(values, *, offsets=None, frequency="h", metadata=None):
    if offsets is None:
        index = pd.date_range(
            "2024-01-01", periods=len(values), freq=frequency, tz="UTC"
        )
    else:
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=i)
                for i in offsets
            ]
        )
    return TimeSeries(pd.Series(values, index=index, name="value"), metadata=metadata)


def test_sum_distinguishes_absent_all_missing_and_measured_zero():
    original = series([0, np.nan, np.nan], offsets=[0, 3600, 10800])
    result = original.resample("h", method="sum").to_dataframe()["value"]
    assert result.iloc[0] == 0
    assert result.iloc[1:].isna().all()
    assert len(result) == 4


def test_nested_metadata_and_duplicate_column_ownership():
    metadata = {"source": {"ids": [1]}}
    original = series([1, 2], metadata=metadata)
    metadata["source"]["ids"].append(2)
    copied = original.resample("2h")
    copied.metadata["source"]["ids"].append(3)
    assert original.metadata == {"source": {"ids": [1]}}
    frame = pd.DataFrame(
        [[1, 2]], index=original.timestamps[:1], columns=["cell", "cell"]
    )
    with pytest.raises(ValueError, match="columns must be unique"):
        TimeSeries(frame)


@pytest.mark.parametrize(
    "method",
    [
        "forecast_linear",
        "forecast_moving_average",
        "forecast_arima",
        "forecast_exponential_smoothing",
    ],
)
@pytest.mark.parametrize(
    "values,offsets", [([1, np.nan, 3], None), ([1, 2, 3], [0, 60, 180])]
)
def test_forecasts_never_drop_missing_or_invent_cadence(method, values, offsets):
    with pytest.raises(ValueError, match="finite|regular cadence"):
        getattr(ForecastingEngine(), method)(series(values, offsets=offsets), horizon=1)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_forecast_count_contract(horizon):
    with pytest.raises(ValueError, match="horizon"):
        ForecastingEngine().forecast_linear(series([1, 2, 3]), horizon=horizon)


def test_two_sample_nanosecond_forecast_and_calendar_cadence():
    tiny = series([2, 5], offsets=[0, 0.000000003])
    result = ForecastingEngine().forecast_linear(tiny, horizon=2)
    assert result["forecast"] == pytest.approx([8, 11])
    assert [
        pd.Timestamp(i).value - tiny.timestamps[0].value for i in result["timestamps"]
    ] == [6, 9]
    monthly = ForecastingEngine().forecast_linear(
        series([1, 3, 5], frequency="ME"), horizon=2
    )
    assert monthly["timestamps"] == [
        "2024-04-30T00:00:00+00:00",
        "2024-05-31T00:00:00+00:00",
    ]


def test_explicit_two_sample_calendar_and_custom_cadences_survive_forecasting():
    monthly = series([1, 3], frequency="ME")
    assert detect_frequency(monthly) == "ME"
    assert ForecastingEngine().forecast_linear(monthly, horizon=2)["timestamps"] == [
        "2024-03-31T00:00:00+00:00",
        "2024-04-30T00:00:00+00:00",
    ]
    custom = pd.offsets.CustomBusinessDay(weekmask="Mon Wed Fri")
    axis = pd.date_range("2024-01-01", periods=2, freq=custom, tz="UTC")
    observations = TimeSeries(pd.Series([1, 3], index=axis, name="value"))
    assert ForecastingEngine().forecast_linear(observations, horizon=2)[
        "timestamps"
    ] == ["2024-01-05T00:00:00+00:00", "2024-01-08T00:00:00+00:00"]
    pd.testing.assert_frame_equal(
        fill_gaps(observations).to_dataframe(), observations.to_dataframe()
    )


def test_holdout_uses_same_moving_average_model_with_independent_errors():
    result = ForecastingEngine().validate_forecast(
        series([1, 3, 5, 7, 9, 11]),
        {"model_type": "moving_average", "window": 2},
        validation_split=1 / 3,
    )
    # Training values end in 5, 7: both holdout predictions are 6.
    assert result["mae"] == 4
    assert result["mse"] == 17
    assert result["rmse"] == pytest.approx(math.sqrt(17))
    assert result["mape"] == pytest.approx((3 / 9 + 5 / 11) * 50)
    with pytest.raises(ValueError, match="model_type"):
        ForecastingEngine().validate_forecast(
            series([1, 2, 3, 4]), {"model_type": "unknown"}
        )
    with pytest.raises(ValueError, match="model_type"):
        ForecastingEngine().validate_forecast(series([1, 2, 3, 4]), {"forecast": [99]})


def test_zero_actual_has_undefined_mape_and_exact_smape():
    result = TemporalAnalyzer().validate_forecast([0, 2], [0, 1], [(0, 0), (1, 3)])
    assert result["metrics"]["mape"] is None
    assert result["metrics"]["smape"] == pytest.approx(100 / 3)
    assert result["metrics"]["confidence_coverage"] == 100
    tiny = TemporalAnalyzer().validate_forecast([1e-200, 2e-200], [1e-200, 1e-200])
    assert tiny["metrics"]["mape"] == 25
    assert tiny["metrics"]["smape"] == pytest.approx(100 / 3)


@pytest.mark.parametrize(
    "method", ["calculate_cross_correlation", "calculate_granger_causality"]
)
def test_paired_methods_bind_timestamp_axes(method):
    first = series([1, 3, 2, 5, 7, 4])
    shifted = series([1, 3, 2, 5, 7, 4], offsets=[3600 * i for i in range(1, 7)])
    with pytest.raises(ValueError, match="identical UTC"):
        getattr(TemporalAnalyzer(), method)(first, shifted, max_lag=1)


def test_cross_correlation_matches_independent_centered_dot_product():
    left, right = [2, 5, 1, 4, 8, 3], [4, 1, 2, 8, 5, 3]
    result = TemporalAnalyzer().calculate_cross_correlation(
        series(left), series(right), max_lag=1
    )
    for entry in result["correlations"]:
        lag = entry["lag"]
        a, b = (
            (left[1:], right[:-1])
            if lag < 0
            else (left[:-1], right[1:])
            if lag > 0
            else (left, right)
        )
        da = [x - sum(a) / len(a) for x in a]
        db = [x - sum(b) / len(b) for x in b]
        expected = sum(x * y for x, y in zip(da, db)) / math.sqrt(
            sum(x * x for x in da) * sum(y * y for y in db)
        )
        assert entry["correlation"] == pytest.approx(expected)


def test_granger_matches_declared_statsmodels_f_test():
    from statsmodels.tsa.stattools import grangercausalitytests

    rng = np.random.default_rng(12)
    driver = rng.normal(size=80)
    response = np.zeros(80)
    for i in range(1, 80):
        response[i] = (
            0.3 * response[i - 1] + 0.7 * driver[i - 1] + rng.normal(scale=0.15)
        )
    actual = TemporalAnalyzer().calculate_granger_causality(
        series(driver), series(response), max_lag=2
    )
    expected = grangercausalitytests(np.column_stack([response, driver]), [1, 2])
    for lag in [1, 2]:
        f_value, p_value, _, _ = expected[lag][0]["ssr_ftest"]
        assert actual["tests"][f"series1_causes_series2_lag{lag}"][
            "f_statistic"
        ] == pytest.approx(f_value)
        assert actual["tests"][f"series1_causes_series2_lag{lag}"][
            "p_value"
        ] == pytest.approx(p_value)


def test_exact_two_segments_and_final_change_boundary():
    values = series([0, 0, 10, 10])
    analyzer = TemporalAnalyzer().detect_change_points(
        values, method="binary_segmentation", min_segment_length=2
    )
    assert analyzer["change_points"] == [
        {"index": 2, "mean_before": 0, "mean_after": 10, "magnitude": 10}
    ]
    detector = EventDetector(window_size=2).detect_changepoints(values)
    assert detector["changepoints"][0]["index"] == 2
    assert detector["changepoints"][0]["mean_change"] == 10


def test_entropy_formulas_and_zero_sample_entropy():
    values = series([0, 0, 0, 1, 0, 0, 0, 1])
    result = TemporalAnalyzer().compute_temporal_entropy(
        values, bins=2, method="approximate", embedding_dim=1, tolerance=0
    )
    phi1 = (6 * math.log(6 / 8) + 2 * math.log(2 / 8)) / 8
    phi2 = (4 * math.log(4 / 7) + 2 * math.log(2 / 7) + math.log(1 / 7)) / 7
    assert result["shannon_entropy"]["value"] == pytest.approx(
        -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
    )
    assert result["sample_entropy"]["value"] == pytest.approx(math.log(15 / 7))
    assert result["approximate_entropy"]["value"] == pytest.approx(phi1 - phi2)
    constant = TemporalAnalyzer().compute_temporal_entropy(
        series([3] * 8), method="sample"
    )
    assert constant["sample_entropy"]["value"] == 0
    with pytest.raises(ValueError, match="max_points"):
        TemporalAnalyzer().compute_temporal_entropy(
            values, method="sample", max_points=7
        )


def test_off_grid_anchors_and_gap_aware_fill_count():
    values = series([0, 42, 3], offsets=[0, 420, 600])
    result = TemporalInterpolator().resample_interpolate(values, target_freq="5min")
    # The five-minute value lies between 0 at minute zero and 42 at minute
    # seven: 42 * 5 / 7 = 30. Discarding the off-grid anchor gives 1.5.
    assert result.to_dataframe()["value"].tolist() == pytest.approx([0, 30, 3])
    gap = series([np.nan, 1, np.nan, 3, np.nan, np.nan, 6, np.nan])
    filled = TemporalInterpolator().interpolate_gap_aware(gap, max_gap_size=1)
    assert filled.metadata["filled_count"] == 1
    assert filled.to_dataframe().iloc[2, 0] == 2
    assert filled.to_dataframe().iloc[[0, 4, 5, 7], 0].isna().all()
    with pytest.raises(ValueError, match="Unknown"):
        TemporalInterpolator().interpolate_gap_aware(gap, 1, method="guess")
    with pytest.raises(ValueError, match="discard"):
        fill_gaps(values, freq="5min")
    assert detect_frequency(values) is None


def test_rolling_values_bind_surviving_timestamps():
    original = series([1, 3, np.nan, 7, 9])
    result = TemporalAnalyzer().calculate_rolling_statistics(
        original, window=2, statistics=["mean"]
    )
    assert result["summary"]["valid_observations"] == 2
    assert result["statistics"]["mean"]["values"] == [2, 8]
    assert result["statistics"]["mean"]["timestamps"] == [
        original.timestamps[i].isoformat() for i in [1, 4]
    ]


def test_exact_scaled_diagnostics_and_likelihood():
    stats = TemporalStatistics()
    residuals = [1, -2, 3, -4]
    expected_dw = (9 + 25 + 49) / 30
    assert stats.durbin_watson_test(residuals)["dw_statistic"] == pytest.approx(
        expected_dw
    )
    assert stats.durbin_watson_test([r * 1e-200 for r in residuals])[
        "dw_statistic"
    ] == pytest.approx(expected_dw)
    likelihood = -2 * (math.log(2 * math.pi) + math.log(7.5) + 1)
    actual = stats.information_criteria(residuals, num_params=1)
    assert actual["log_likelihood"] == pytest.approx(likelihood)
    tiny = stats.information_criteria([r * 1e-200 for r in residuals], num_params=1)
    assert tiny["log_likelihood"] == pytest.approx(likelihood - 4 * math.log(1e-200))
    assert stats.ljung_box_test(residuals, lags=2)["lb_statistic"] == pytest.approx(
        stats.ljung_box_test([r * 1e-200 for r in residuals], lags=2)["lb_statistic"]
    )


def test_backend_config_is_applied_owned_and_unknown_rejected():
    config = {
        "smoothing_fit_kwargs": {"optimized": False, "smoothing_level": 0.5},
        "seasonal_period": 4,
    }
    engine = AdvancedForecastingEngine(config)
    config["smoothing_fit_kwargs"]["optimized"] = True
    result = engine.forecast_exponential_smoothing(
        series([1, 2, 3, 4]).to_dataframe()["value"], trend=None, forecast_steps=2
    )
    assert result["model"].params["smoothing_level"] == 0.5
    level = result["model"].params["initial_level"]
    for observation in [1, 2, 3, 4]:
        level = 0.5 * observation + 0.5 * level
    assert result["forecast"].tolist() == pytest.approx([level, level])
    assert engine.config["smoothing_fit_kwargs"]["optimized"] is False
    with pytest.raises(ValueError, match="Unknown forecasting"):
        AdvancedForecastingEngine({"unused": 1})
    cyclic = series([1, 2, 3, 4] * 4).to_dataframe()["value"]
    decomposition = engine.detect_trend_seasonality(cyclic)
    assert decomposition["period"] == 4
    assert decomposition["seasonal"].tolist() == pytest.approx(
        [-1.5, -0.5, 0.5, 1.5] * 4
    )


@pytest.mark.parametrize("suffix", ["csv", "json", "parquet"])
def test_io_metadata_nanosecond_roundtrip_and_store_bounds(tmp_path, suffix):
    original = series(
        [0, 2], offsets=[0.000000001, 0.000000002], metadata={"sensor": {"ids": ["a"]}}
    )
    path = tmp_path / f"observations.{suffix}"
    write_timeseries(original, path)
    restored = read_timeseries(path)
    pd.testing.assert_frame_equal(
        restored.to_dataframe(),
        original.to_dataframe(),
        check_names=False,
        check_dtype=False,
    )
    assert restored.metadata == original.metadata
    store = InMemoryStore()
    store.store("sensor", restored)
    with pytest.raises(ValueError, match="after"):
        store.query("sensor", start=restored.end_time, end=restored.start_time)
    queried = store.query("sensor", start=restored.start_time, end=restored.start_time)
    assert queried.to_dataframe().iloc[0, 0] == 0
    queried.metadata["sensor"]["ids"].append("b")
    assert store.retrieve("sensor").metadata == original.metadata


@pytest.mark.parametrize("write_metadata", [True, False])
def test_overwriting_without_metadata_retires_previous_provenance(
    tmp_path, write_metadata
):
    path = tmp_path / "observations.csv"
    write_timeseries(series([1, 2], metadata={"sensor": "old"}), path)
    assert read_timeseries(path).metadata == {"sensor": "old"}
    write_timeseries(series([3, 4]), path, write_metadata=write_metadata)
    assert not path.with_suffix(".meta.json").exists()
    restored = read_timeseries(path)
    assert restored.metadata == {}
    assert restored.to_dataframe()["value"].tolist() == [3, 4]


def test_json_does_not_convert_numeric_epoch_before_validation(tmp_path):
    path = tmp_path / "numeric.json"
    path.write_text(json.dumps({"value": {"1704067200000": 1}}))
    with pytest.raises(ValueError, match="time column"):
        read_timeseries(path)
    path.write_text(json.dumps([{"timestamp": 1704067200000, "value": 1}]))
    with pytest.raises(TypeError, match="datetime"):
        read_timeseries(path)


@pytest.mark.parametrize(
    "unit,epoch,expected", [("s", 1000, 1000), ("ms", 1000, 1), ("ms", 0, 0)]
)
def test_explicit_numeric_transport_unit(unit, epoch, expected):
    instant, _, _ = ReplayIngestAdapter([], {"timestamp_unit": unit}).parse_record(
        {"timestamp": epoch, "value": 0}
    )
    assert instant == datetime.fromtimestamp(expected, UTC)
    with pytest.raises(ValueError, match="explicit timestamp_unit"):
        ReplayIngestAdapter([]).parse_record({"timestamp": epoch, "value": 0})


def test_stream_snapshot_ownership_finite_aggregation_and_work_budget():
    base = datetime(2024, 1, 1, tzinfo=UTC)
    processor = StreamProcessor(timedelta(seconds=10), watermark_delay=timedelta(0))
    metadata = {"sensor": {"tags": ["a"]}}
    processor.add_data_point(base, 1, metadata)
    metadata["sensor"]["tags"].append("b")
    assert processor.buffer[0]["metadata"]["sensor"]["tags"] == ["a"]
    window = processor.process_window()
    window["count"] = -1
    assert processor.get_recent_windows()[0]["count"] == 1
    assert processor.get_buffer_summary()["watermark_delay_seconds"] == 0
    invalid = StreamProcessor(timedelta(seconds=10), aggregation_func=lambda _: np.nan)
    invalid.add_data_point(base, 1)
    with pytest.raises(ValueError, match="finite"):
        invalid.process_window()
    assert invalid.get_stats()["total_windows"] == 0
    assert invalid.get_recent_windows() == []
    bounded = StreamProcessor(
        timedelta(seconds=10),
        slide_interval=timedelta(microseconds=1),
        max_window_evaluations=5,
    )
    bounded.add_data_point(base, 1)
    bounded.add_data_point(base + timedelta(seconds=1), 9)
    for method in [
        bounded.process_sliding_windows,
        bounded.process_sliding_window_anomaly_alerts,
    ]:
        with pytest.raises(ValueError, match="max_window_evaluations"):
            method(
                min_window_points=2
            ) if method == bounded.process_sliding_window_anomaly_alerts else method()
    assert bounded.get_stats()["anomaly_alerts"] == 0


def test_factory_preserves_nanosecond_start():
    result = create_timeseries(
        [1, 2], start="2024-01-01T00:00:00.000000001Z", freq="ns"
    )
    assert result.timestamps[0].nanosecond == 1
    assert result.timestamps[1].nanosecond == 2


def test_nanosecond_grid_budget_rejected_before_allocation():
    original = series([1, 2], offsets=[0, 1])
    with pytest.raises(ValueError, match="allocation budget"):
        TemporalInterpolator().resample_interpolate(
            original, target_freq="ns", max_points=100
        )
    with pytest.raises(ValueError, match="allocation budget"):
        fill_gaps(original, freq="ns", max_points=100)


def test_zero_mean_cv_and_zero_segment_variance_are_undefined():
    stats = TemporalStatistics()
    assert stats.calculate_summary([-1, 1])["dispersion"]["cv"] is None
    result = stats.residual_diagnostics([1, 1, 2, 2])
    assert result["variance_test"]["variance_ratio"] is None
    assert result["variance_test"]["homoscedastic"] is None
    assert not result["overall"]["residuals_ok"]


def test_visualization_real_axes_and_explicit_missing_extra(monkeypatch):
    import matplotlib.pyplot as plt
    import geo_infer_time.core.visualization as visualization

    viz = visualization.TemporalVisualization()
    original = series([1, 2, 3, 4])
    forecast = ForecastingEngine().forecast_linear(original, horizon=2)
    fig = viz.create_dashboard(
        [1, 2, 3, 4], timestamps=original.timestamps, forecast=forecast
    )
    assert list(fig.axes[3].lines[0].get_xdata()) == list(original.timestamps)
    assert list(fig.axes[3].lines[1].get_xdata()) == [
        pd.Timestamp(t) for t in forecast["timestamps"]
    ]
    plt.close(fig)
    fig = viz.plot_rolling_statistics(
        [1, 2, 3, 4],
        [1.5, 3.5],
        timestamps=original.timestamps,
        rolling_timestamps=original.timestamps[[1, 3]],
    )
    assert list(fig.axes[0].lines[1].get_xdata()) == list(original.timestamps[[1, 3]])
    plt.close(fig)
    with pytest.raises(ValueError, match="supplied together"):
        viz.plot_forecast([1, 2], [3], timestamps_historical=original.timestamps[:2])
    fig = viz.create_dashboard([3, 3, 3])
    assert "undefined" in fig.axes[3].get_title()
    plt.close(fig)
    monkeypatch.setattr(visualization, "HAS_MATPLOTLIB", False)
    with pytest.raises(RuntimeError, match=r"geo-infer-time\[visualization\]"):
        visualization.TemporalVisualization()
    with pytest.raises(RuntimeError, match=r"geo-infer-time\[visualization\]"):
        viz.plot_timeseries([1, 2])


@pytest.mark.parametrize(
    "filename", ["basic_temporal_analysis.py", "demo_all_methods.py"]
)
def test_examples_are_import_safe_and_exit_with_actual_results(filename, capsys):
    import importlib.util
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples" / filename
    spec = importlib.util.spec_from_file_location("time_example", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert capsys.readouterr().out == ""
    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["observations"] > 0


def test_real_arima_random_walk_retains_forecast_intervals():
    actual = ForecastingEngine().forecast_arima(
        series(list(range(24))), horizon=2, order=(0, 1, 0)
    )
    assert actual["forecast"] == pytest.approx([23, 23])
    assert len(actual["lower_bound"]) == len(actual["upper_bound"]) == 2
    assert all(
        low <= prediction <= high
        for low, prediction, high in zip(
            actual["lower_bound"], actual["forecast"], actual["upper_bound"]
        )
    )
    assert isinstance(actual["converged"], bool)
    assert isinstance(actual["fit_warnings"], list)


def test_visualization_invalid_style_and_internal_import_failures_propagate():
    import matplotlib.pyplot as plt
    import geo_infer_time.core.visualization as visualization

    before = dict(plt.rcParams)
    with pytest.raises(OSError):
        visualization.TemporalVisualization(
            style="geo-infer-unavailable-style"
        ).plot_timeseries([1, 2])
    assert dict(plt.rcParams) == before
