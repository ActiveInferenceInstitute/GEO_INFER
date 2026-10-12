"""Regression tests for the risk-engine lifecycle and reproducibility contract."""

from pathlib import Path

import pandas as pd
import pytest
from geo_infer_risk.core.hazard_model import EnhancedHazardModel
from geo_infer_risk.core.risk_engine import EnhancedRiskEngine
from geo_infer_risk.utils.config_loader import load_config_with_defaults
import geo_infer_risk.core.risk_engine as risk_runtime
import geo_infer_risk.core.hazard_model as hazard_runtime
import geo_infer_risk.core.catastrophe_models as catastrophe_runtime


@pytest.mark.parametrize("runtime", [hazard_runtime, catastrophe_runtime])
@pytest.mark.parametrize("interface", ["TemporalAnalyzer", "SpatialIndexingInterface"])
def test_connected_models_propagate_installed_interface_errors(
    monkeypatch, runtime, interface
):
    def failed_interface(*args, **kwargs):
        raise RuntimeError("installed model interface is broken")

    monkeypatch.setattr(runtime, interface, failed_interface)
    with pytest.raises(RuntimeError, match="installed model interface is broken"):
        if runtime is hazard_runtime:
            runtime.EnhancedHazardModel("flood", {})
        else:
            runtime.EnhancedCatastropheModel()


@pytest.mark.parametrize(
    "interface",
    ["TemporalAnalyzer", "SpatialIndexingInterface", "SpatialAnalyticsInterface"],
)
def test_installed_interface_constructor_failures_propagate_before_logging(
    tmp_path, monkeypatch, interface
):
    config = engine_config(tmp_path)

    def failed_interface(*args, **kwargs):
        raise RuntimeError("installed interface is broken")

    monkeypatch.setattr(risk_runtime, interface, failed_interface)
    with pytest.raises(RuntimeError, match="installed interface is broken"):
        EnhancedRiskEngine(config)
    assert not (tmp_path / "outputs" / "risk_engine.log").exists()


def test_time_constructor_is_retained_once(tmp_path, monkeypatch):
    real_analyzer = risk_runtime.TemporalAnalyzer
    constructed = []

    def analyzer():
        actual = real_analyzer()
        constructed.append(actual)
        return actual

    monkeypatch.setattr(risk_runtime, "TemporalAnalyzer", analyzer)
    with EnhancedRiskEngine(engine_config(tmp_path)) as engine:
        assert constructed == [engine.temporal_interface]


def engine_config(tmp_path: Path) -> dict:
    config = load_config_with_defaults()
    config["general"]["output_directory"] = str(tmp_path / "outputs")
    config["general"]["cache_directory"] = str(tmp_path / "cache")
    config["general"]["num_workers"] = 1
    config["general"]["random_seed"] = 17
    config["risk_model"]["random_seed"] = 17
    return config


def configured_hazard() -> EnhancedHazardModel:
    model = EnhancedHazardModel("flood", {"random_seed": 17})
    model.historical_data = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "intensity": [1.0, 2.0, 3.0],
        }
    )
    return model


def test_engine_context_closes_executor_and_rejects_new_work(tmp_path: Path) -> None:
    with EnhancedRiskEngine(engine_config(tmp_path)) as engine:
        assert engine.get_integration_status()["spatial_indexing"] is True
        assert engine._closed is False

    assert engine._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        engine.run_monte_carlo_analysis(num_iterations=1)


def test_engine_event_sampling_is_reproducible_without_global_rng(
    tmp_path: Path,
) -> None:
    first_engine = EnhancedRiskEngine(engine_config(tmp_path / "first"))
    second_engine = EnhancedRiskEngine(engine_config(tmp_path / "second"))
    try:
        first_engine.hazard_models["flood"] = configured_hazard()
        second_engine.hazard_models["flood"] = configured_hazard()

        first_events = [first_engine._generate_random_event() for _ in range(5)]
        second_events = [second_engine._generate_random_event() for _ in range(5)]

        assert first_events == second_events
    finally:
        first_engine.close()
        second_engine.close()


def test_engine_rejects_underspecified_calibration(tmp_path: Path) -> None:
    with EnhancedRiskEngine(engine_config(tmp_path)) as engine:
        with pytest.raises(ValueError, match="at least two"):
            engine.calibrate_models({"samples": []})
        with pytest.raises(ValueError, match="only implemented calibration method"):
            engine.calibrate_models(
                {"samples": [{"loss": 1.0}, {"loss": 2.0}]}, "bayesian"
            )
        with pytest.raises(ValueError, match="only implemented calibration method"):
            engine.calibrate_models(
                {"samples": [{"loss": 1.0}, {"loss": 2.0}]}, "maximum_likelihood"
            )


def test_cross_validation_fits_loss_baseline_parameters(tmp_path: Path) -> None:
    samples = [{"loss": loss} for loss in (10.0, 20.0, 30.0, 40.0)]

    with EnhancedRiskEngine(engine_config(tmp_path)) as engine:
        result = engine.calibrate_models({"samples": samples})

    assert result["method"] == "cross_validation"
    assert result["calibrated_parameters"] == {
        "loss_mean": 25.0,
        "loss_standard_deviation": pytest.approx(12.9099444874),
        "sample_count": 4,
    }
    assert len(result["cross_validation_results"]["folds"]) == 4
    assert all(
        fold["training_sample_count"] == 3
        for fold in result["cross_validation_results"]["folds"]
    )


def test_temporal_analysis_uses_actual_utc_axis_and_forecast(tmp_path):
    """UTC month membership and future values have independent tiny oracles."""
    history = [
        {"timestamp": "2026-01-30T16:00:00-08:00", "value": 2.0},
        {"timestamp": "2026-02-01T00:00:00Z", "value": 5.0},
        {"timestamp": "2026-02-02T00:00:00Z", "value": 8.0},
    ]
    with EnhancedRiskEngine(engine_config(tmp_path)) as engine:
        assert engine.get_integration_status()["temporal_analysis"]
        result = engine.run_enhanced_analysis(loss_history=history, time_horizon=2)
    temporal = result["temporal_analysis"]
    assert temporal["backend"] == "geo_infer_time"
    assert temporal["seasonal_patterns"] == {1: 2.0, 2: 6.5}
    assert temporal["trend_analysis"]["slope_per_sample"] == pytest.approx(3.0)
    scenario = temporal["forecast_scenarios"][0]
    assert scenario["projected_mean"] == pytest.approx(14.0)
    assert scenario["timestamp"] == "2026-02-04T00:00:00+00:00"


@pytest.mark.parametrize(
    "fault", ["naive", "missing_timestamp", "missing_value", "gap", "nan"]
)
def test_temporal_loss_history_errors_propagate(tmp_path, fault):
    history = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 2.0},
        {"timestamp": "2026-01-02T00:00:00Z", "value": 5.0},
        {"timestamp": "2026-01-03T00:00:00Z", "value": 8.0},
    ]
    if fault == "naive":
        history[0]["timestamp"] = "2026-01-01"
    elif fault == "missing_timestamp":
        del history[0]["timestamp"]
    elif fault == "missing_value":
        del history[0]["value"]
    elif fault == "gap":
        history[-1]["timestamp"] = "2026-01-04T00:00:00Z"
    else:
        history[0]["value"] = float("nan")
    with EnhancedRiskEngine(engine_config(tmp_path)) as engine:
        with pytest.raises((ValueError, TypeError)):
            engine.run_enhanced_analysis(loss_history=history, time_horizon=2)
        assert list(engine.active_jobs.values())[-1].status == "failed"
