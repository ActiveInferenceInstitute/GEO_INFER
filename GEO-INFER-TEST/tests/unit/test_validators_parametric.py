"""
High-volume parametric tests for all GEO-INFER-TEST validators.

Uses @pytest.mark.parametrize with hundreds of generated scenarios to
exercise every validator code path, achieving thousands of individual
test cases across DataQuality, Spatial, IoT, Bayesian, Performance,
and QualityController validators.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

from geo_infer_test.core.validators import (
    DataQualityValidator,
    SpatialValidator,
    IoTValidator,
    BayesianValidator,
    PerformanceValidator,
    QualityController,
    run_full_system_test,
)

# ============================================================================
# Helper: generate large parameter sets
# ============================================================================


def _ts(delta_hours: float = 0) -> str:
    """ISO timestamp shifted by *delta_hours* from now."""
    return (datetime.now(UTC) - timedelta(hours=delta_hours)).isoformat()


# ---- DataQualityValidator scenarios ----------------------------------------
_DQ_SCENARIOS = []

# Valid rows
for n in (0, 1, 5, 50):
    _DQ_SCENARIOS.append(
        pytest.param(
            [{"timestamp": _ts(), "value": float(i)} for i in range(n)],
            id=f"valid_{n}rows",
        )
    )

# Missing timestamp
for n in (1, 10, 25):
    _DQ_SCENARIOS.append(
        pytest.param(
            [{"value": float(i)} for i in range(n)],
            id=f"missing_ts_{n}rows",
        )
    )

# Missing value
for n in (1, 10, 25):
    _DQ_SCENARIOS.append(
        pytest.param(
            [{"timestamp": _ts()} for _ in range(n)],
            id=f"missing_val_{n}rows",
        )
    )

# Null values
_DQ_SCENARIOS.append(pytest.param([{"timestamp": None, "value": None}], id="all_null"))
_DQ_SCENARIOS.append(
    pytest.param([{"timestamp": _ts(), "value": None}], id="null_value_only")
)
_DQ_SCENARIOS.append(
    pytest.param([{"timestamp": None, "value": 1.0}], id="null_ts_only")
)

# Non-numeric value fields
for v in ("abc", True, False, [], {}, "123"):
    _DQ_SCENARIOS.append(
        pytest.param(
            [{"timestamp": _ts(), "value": v}],
            id=f"value_type_{type(v).__name__}_{str(v)[:10]}",
        )
    )

# Large numeric values
for v in (0, -1, 1e15, -1e15, float("inf"), float("-inf"), float("nan")):
    _DQ_SCENARIOS.append(
        pytest.param(
            [{"timestamp": _ts(), "value": v}],
            id=f"value_extreme_{v}",
        )
    )

# Bad timestamp formats
for ts in ("not-a-date", "2020/01/01", "01-01-2020", ""):
    _DQ_SCENARIOS.append(
        pytest.param(
            [{"timestamp": ts, "value": 1.0}],
            id=f"bad_ts_{ts[:10]}",
        )
    )

# Extra columns
_DQ_SCENARIOS.append(
    pytest.param(
        [{"timestamp": _ts(), "value": 1.0, "extra1": "x", "extra2": 99}],
        id="extra_columns",
    )
)

# Mix of good and bad rows
_DQ_SCENARIOS.append(
    pytest.param(
        [{"timestamp": _ts(), "value": 1.0}] * 40
        + [{"timestamp": None, "value": None}] * 10,
        id="mixed_40good_10bad",
    )
)

# Single dict input
_DQ_SCENARIOS.append(
    pytest.param(
        {"timestamp": _ts(), "value": 42.0},
        id="single_dict_input",
    )
)

# DataFrame input
_DQ_SCENARIOS.append(
    pytest.param(
        pd.DataFrame({"timestamp": [_ts()] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        id="dataframe_input",
    )
)


class TestDataQualityParametric:
    """Parametric tests for DataQualityValidator."""

    @pytest.mark.parametrize("data", _DQ_SCENARIOS)
    def test_validate_returns_valid_structure(self, data):
        result = DataQualityValidator().validate(data)
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0
        assert "total_records" in result
        assert "validation_errors" in result
        assert isinstance(result["validation_errors"], list)

    @pytest.mark.parametrize("data", _DQ_SCENARIOS)
    def test_valid_records_leq_total(self, data):
        result = DataQualityValidator().validate(data)
        assert result["valid_records"] <= result["total_records"]


# ---- SpatialValidator scenarios --------------------------------------------
_COORD_LATS = [-90.0, -45.0, 0.0, 45.0, 90.0, -91.0, 91.0, -180.0, 180.0, 0.001]
_COORD_LONS = [-180.0, -90.0, 0.0, 90.0, 180.0, -181.0, 181.0, 360.0, -0.001]

_SPATIAL_SCENARIOS = []
for lat in _COORD_LATS:
    for lon in _COORD_LONS:
        _SPATIAL_SCENARIOS.append(
            pytest.param(
                [{"latitude": lat, "longitude": lon}],
                id=f"coord_lat{lat}_lon{lon}",
            )
        )

# Special types in coordinates
for v in (None, "abc", True, [], {}):
    _SPATIAL_SCENARIOS.append(
        pytest.param(
            [{"latitude": v, "longitude": 0.0}],
            id=f"lat_type_{type(v).__name__}",
        )
    )

# Multiple points
_SPATIAL_SCENARIOS.append(
    pytest.param(
        [{"latitude": float(i), "longitude": float(i)} for i in range(-5, 6)],
        id="multi_point_range",
    )
)

# Missing columns
_SPATIAL_SCENARIOS.append(pytest.param([{"latitude": 0.0}], id="missing_longitude"))
_SPATIAL_SCENARIOS.append(pytest.param([{"longitude": 0.0}], id="missing_latitude"))
_SPATIAL_SCENARIOS.append(pytest.param([{"other": "data"}], id="no_spatial_columns"))


class TestSpatialParametric:
    """Parametric tests for SpatialValidator."""

    @pytest.mark.parametrize("data", _SPATIAL_SCENARIOS)
    def test_spatial_returns_structure(self, data):
        result = SpatialValidator().validate(data)
        assert "spatial_validation" in result
        assert "total_records" in result

    def test_duplicate_coordinate_columns_record_unknown_status(self):
        """M9-05: a coordinate coercion failure is recorded, not swallowed.

        Duplicated ``latitude`` columns make ``df["latitude"]`` a DataFrame,
        which ``pd.to_numeric`` rejects with TypeError — exactly the silent
        ``except ... pass`` path this contract forbids.
        """
        df = pd.DataFrame(
            [[0.0, 5.0, 5.0]], columns=["longitude", "latitude", "latitude"]
        )
        result = SpatialValidator().validate(df)
        coords = result["spatial_validation"]["coordinate_validity"]
        assert coords["status"] == "unknown"
        assert "coordinate numeric coercion failed" in coords["reason"]


# ---- IoTValidator scenarios ------------------------------------------------
_IOT_SCENARIOS = []

# Valid sensor readings
for n_sensors in (1, 3, 10, 25):
    rows = []
    for i in range(n_sensors):
        rows.append(
            {
                "sensor_id": f"sensor_{i}",
                "timestamp": _ts(float(i)),
                "radiation_level": float(i * 5),
                "value": float(i * 10),
            }
        )
    _IOT_SCENARIOS.append(pytest.param(rows, id=f"valid_{n_sensors}sensors"))

# Radiation edge cases
for rad in (0.0, 50.0, 100.0, -1.0, 101.0, 999.0, float("nan"), float("inf")):
    _IOT_SCENARIOS.append(
        pytest.param(
            [
                {
                    "sensor_id": "sensor_0",
                    "timestamp": _ts(),
                    "radiation_level": rad,
                    "value": 1.0,
                }
            ],
            id=f"radiation_{rad}",
        )
    )

# String radiation values
for rad_str in ("0", "50.5", "high", ""):
    _IOT_SCENARIOS.append(
        pytest.param(
            [
                {
                    "sensor_id": "sensor_0",
                    "timestamp": _ts(),
                    "radiation_level": rad_str,
                    "value": 1.0,
                }
            ],
            id=f"radiation_str_{rad_str[:5]}",
        )
    )

# Bad sensor IDs
for sid in ("sensor_0", "SENSOR-001", "s", "", "abc_123"):
    _IOT_SCENARIOS.append(
        pytest.param(
            [
                {
                    "sensor_id": sid,
                    "timestamp": _ts(),
                    "radiation_level": 5.0,
                    "value": 1.0,
                }
            ],
            id=f"sensor_id_{sid[:10]}",
        )
    )

# Missing fields
_IOT_SCENARIOS.append(
    pytest.param([{"timestamp": _ts(), "value": 1.0}], id="missing_sensor_id_rad")
)
_IOT_SCENARIOS.append(
    pytest.param([{"sensor_id": "sensor_0", "value": 1.0}], id="missing_timestamp_rad")
)

# Old timestamps
_IOT_SCENARIOS.append(
    pytest.param(
        [
            {
                "sensor_id": "sensor_0",
                "timestamp": "2000-01-01T00:00:00",
                "radiation_level": 5.0,
                "value": 1.0,
            }
        ],
        id="old_timestamp",
    )
)

# Anomaly detection (z-score)
_IOT_SCENARIOS.append(
    pytest.param(
        [
            {
                "sensor_id": f"sensor_{i}",
                "timestamp": _ts(float(i)),
                "radiation_level": 5.0,
                "value": float(i),
            }
            for i in range(20)
        ]
        + [
            {
                "sensor_id": "sensor_99",
                "timestamp": _ts(),
                "radiation_level": 500.0,
                "value": 99.0,
            }
        ],
        id="anomaly_outlier",
    )
)


class TestIoTParametric:
    """Parametric tests for IoTValidator."""

    @pytest.mark.parametrize("data", _IOT_SCENARIOS)
    def test_iot_returns_structure(self, data):
        result = IoTValidator().validate(data)
        assert "sensor_validation" in result
        assert result["total_sensors"] == len(data)

    @pytest.mark.parametrize("data", _IOT_SCENARIOS)
    def test_iot_data_quality_embedded(self, data):
        result = IoTValidator().validate(data)
        dq = result["sensor_validation"]["data_quality"]
        assert "quality_score" in dq
        assert 0.0 <= dq["quality_score"] <= 1.0


# ---- BayesianValidator scenarios -------------------------------------------
_BAYES_SCENARIOS = []

# Convergence states
for conv in (True, False, None, 0, 1, "yes"):
    _BAYES_SCENARIOS.append(
        pytest.param(
            {"converged": conv},
            id=f"converged_{conv}",
        )
    )

# Prediction arrays
for preds in (
    [],
    [0.5],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    list(np.linspace(0, 1, 50)),
    list(np.linspace(0, 1, 100)),
    [float("nan"), 0.5, 0.5],
    [float("inf"), 0.5],
    [-1.0, 0.5, 2.0],
    [0.0] * 20,
    [1.0] * 20,
):
    _BAYES_SCENARIOS.append(
        pytest.param(
            {"converged": True, "predictions": preds},
            id=f"preds_{len(preds)}_{str(preds[:2])}",
        )
    )

# Uncertainty arrays
for unc in (
    [],
    [0.01],
    [0.1, 0.2, 0.3],
    list(np.random.uniform(0, 1, 50)),
    [0.0] * 10,
    [10.0] * 10,
    [float("nan"), 0.1],
):
    _BAYES_SCENARIOS.append(
        pytest.param(
            {"converged": True, "predictions": [0.5, 0.5, 0.5], "uncertainty": unc},
            id=f"unc_{len(unc)}",
        )
    )

# Combined valid
_BAYES_SCENARIOS.append(
    pytest.param(
        {
            "converged": True,
            "predictions": list(np.random.uniform(0, 1, 20)),
            "uncertainty": list(np.random.uniform(0, 0.5, 20)),
            "model_info": {"iterations": 1000, "burn_in": 500},
        },
        id="full_valid_result",
    )
)

# Empty dict
_BAYES_SCENARIOS.append(pytest.param({}, id="empty_dict"))

# Only model info
_BAYES_SCENARIOS.append(
    pytest.param({"model_info": {"iterations": 100}}, id="model_info_only")
)


class TestBayesianParametric:
    """Parametric tests for BayesianValidator."""

    @pytest.mark.parametrize("data", _BAYES_SCENARIOS)
    def test_bayesian_returns_structure(self, data):
        result = BayesianValidator().validate(data)
        assert "inference_validation" in result
        assert "overall_quality" in result
        assert result["overall_quality"] in (
            "excellent",
            "good",
            "acceptable",
            "poor",
            "unknown",
        )

    @pytest.mark.parametrize("data", _BAYES_SCENARIOS)
    def test_bayesian_has_convergence(self, data):
        result = BayesianValidator().validate(data)
        assert "convergence" in result["inference_validation"]


# ---- PerformanceValidator scenarios ----------------------------------------
_PERF_SCENARIOS = []

# Timing variations
for t in (0.0, 0.5, 1.0, 10.0, 29.9, 30.0, 30.1, 60.0, 100.0):
    _PERF_SCENARIOS.append(
        pytest.param(
            {"inference_time": t},
            id=f"time_{t}s",
        )
    )

# Accuracy variations
for a in (0.0, 0.5, 0.84, 0.85, 0.86, 0.95, 1.0):
    _PERF_SCENARIOS.append(
        pytest.param(
            {"accuracy": a},
            id=f"accuracy_{a}",
        )
    )

# Memory variations
for m, label in [
    (1024, "1KB"),
    (1024**2, "1MB"),
    (1024**3, "1GB"),
    (4 * 1024**3, "4GB"),
    (5 * 1024**3, "5GB"),
    (0, "zero"),
]:
    _PERF_SCENARIOS.append(
        pytest.param(
            {"memory_usage": m},
            id=f"memory_{label}",
        )
    )

# Combined metrics
_PERF_SCENARIOS.append(
    pytest.param(
        {"inference_time": 10.0, "accuracy": 0.95, "memory_usage": 1024**3},
        id="combined_good",
    )
)
_PERF_SCENARIOS.append(
    pytest.param(
        {"inference_time": 60.0, "accuracy": 0.5, "memory_usage": 10 * 1024**3},
        id="combined_bad",
    )
)

# Empty dict
_PERF_SCENARIOS.append(pytest.param({}, id="empty"))

# Config threshold overrides
_PERF_CONFIGS = []
for max_t, min_a in [("5s", 0.9), ("60s", 0.5), ("1m", 0.7)]:
    _PERF_CONFIGS.append(
        pytest.param(
            {
                "validation": {
                    "max_inference_time": max_t,
                    "min_prediction_accuracy": min_a,
                }
            },
            {"inference_time": 10.0, "accuracy": 0.8},
            id=f"config_t{max_t}_a{min_a}",
        )
    )


class TestPerformanceParametric:
    """Parametric tests for PerformanceValidator."""

    @pytest.mark.parametrize("metrics", _PERF_SCENARIOS)
    def test_performance_returns_structure(self, metrics):
        result = PerformanceValidator().validate_performance(metrics)
        pv = result["performance_validation"]
        assert "timing_checks" in pv
        assert "throughput_checks" in pv
        assert "resource_checks" in pv
        assert pv["overall_performance"] in ("acceptable", "unacceptable", "unknown")

    @pytest.mark.parametrize("config, metrics", _PERF_CONFIGS)
    def test_performance_with_config(self, config, metrics):
        result = PerformanceValidator(config).validate_performance(metrics)
        assert "performance_validation" in result

    @pytest.mark.parametrize(
        "time_str, expected",
        [
            ("1s", 1.0),
            ("30s", 30.0),
            ("1m", 60.0),
            ("2h", 7200.0),
            ("0.5s", 0.5),
            ("100", 100.0),
        ],
    )
    def test_parse_time_string(self, time_str, expected):
        pv = PerformanceValidator()
        assert pv._parse_time_string(time_str) == expected

    @pytest.mark.parametrize(
        "mem_str, expected",
        [
            ("1KB", 1024.0),
            ("1MB", 1024**2),
            ("1GB", 1024**3),
            ("4GB", 4 * 1024**3),
            ("512MB", 512 * 1024**2),
        ],
    )
    def test_parse_memory_string(self, mem_str, expected):
        pv = PerformanceValidator()
        assert pv._parse_memory_string(mem_str) == expected


# ---- QualityController scenarios -------------------------------------------


def _strip_timing(result):
    """Remove wall-clock fields so two runs compare by content only."""
    if isinstance(result, dict):
        return {
            key: _strip_timing(value)
            for key, value in result.items()
            if key
            not in {
                "validation_timestamp",
                "total_validation_time",
                "validation_time",
                # Recomputed from datetime.now() on every validation call
                # (validators.py _analyze_temporal_patterns): two calls
                # straddling a scheduling gap produce ages whose relative
                # drift exceeds the 1e-6*age tolerance whenever data is
                # recent. They derive from fixed inputs + the wall clock,
                # like the timestamp fields above.
                "newest_measurement_hours_ago",
                "oldest_measurement_hours_ago",
                "mean_age_hours",
            }
        }
    if isinstance(result, list):
        return [_strip_timing(item) for item in result]
    return result


def _parity_equal(left, right, tol: float = 1e-6) -> bool:
    """Structural equality with relative tolerance for clock-derived floats."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _parity_equal(left[key], right[key], tol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _parity_equal(a, b, tol) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= tol * max(1.0, abs(left), abs(right))
    return left == right


_QC_SENSOR_DATA = pd.DataFrame(
    [
        {
            "sensor_id": f"sensor_{i}",
            "timestamp": _ts(float(i)),
            "radiation_level": float(i * 2),
            "value": float(i),
        }
        for i in range(10)
    ]
)

_QC_INFERENCE = {
    "converged": True,
    "predictions": list(np.random.uniform(0, 1, 20)),
    "uncertainty": list(np.random.uniform(0, 0.3, 20)),
}

_QC_PERF = {"inference_time": 5.0, "accuracy": 0.92, "memory_usage": 1024**3}


class TestQualityControllerParametric:
    """Parametric tests for QualityController."""

    @pytest.mark.parametrize("sensor_data", [None, _QC_SENSOR_DATA])
    @pytest.mark.parametrize("inference", [None, _QC_INFERENCE])
    @pytest.mark.parametrize("perf", [None, _QC_PERF])
    def test_qc_combinations(self, sensor_data, inference, perf):
        """Test all combinations of inputs to QualityController."""
        qc = QualityController()
        result = qc.run_comprehensive_validation(
            sensor_data=sensor_data,
            inference_results=inference,
            performance_metrics=perf,
        )
        assert "components_validated" in result
        assert "overall_results" in result
        assert "total_validation_time" in result
        assert result["overall_results"]["system_quality"] in (
            "excellent",
            "good",
            "acceptable",
            "poor",
            "unknown",
        )

    def test_qc_full_pipeline(self):
        """Smoke test with all components active."""
        qc = QualityController()
        result = qc.run_comprehensive_validation(
            sensor_data=_QC_SENSOR_DATA,
            inference_results=_QC_INFERENCE,
            performance_metrics=_QC_PERF,
        )
        assert len(result["components_validated"]) == 3
        assert result["overall_results"]["components_tested"] == 3

    @pytest.mark.parametrize(
        "quality", ["excellent", "good", "acceptable", "poor", "unknown"]
    )
    def test_qc_recommendation_text(self, quality):
        qc = QualityController()
        rec = qc._get_quality_recommendation(quality)
        assert isinstance(rec, str)
        assert len(rec) > 5

    @pytest.mark.parametrize(
        ("sensor_data", "inference", "perf"),
        [
            (None, None, None),
            (_QC_SENSOR_DATA, None, None),
            (None, _QC_INFERENCE, None),
            (None, None, _QC_PERF),
            (_QC_SENSOR_DATA, _QC_INFERENCE, _QC_PERF),
        ],
    )
    def test_run_full_system_test_delegate_parity(self, sensor_data, inference, perf):
        """run_full_system_test must delegate exactly to QualityController."""
        direct = QualityController().run_comprehensive_validation(
            sensor_data=sensor_data,
            inference_results=inference,
            performance_metrics=perf,
        )
        via_export = run_full_system_test(
            sensor_data=sensor_data,
            spatial_results=None,
            inference_results=inference,
            performance_metrics=perf,
        )
        assert _parity_equal(_strip_timing(via_export), _strip_timing(direct))
