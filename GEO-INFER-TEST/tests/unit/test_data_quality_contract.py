"""Independent record-level quality expectations and strict failure boundaries."""

from datetime import UTC, datetime
import warnings

import numpy as np
import pandas as pd
import pytest

from geo_infer_test import DataQualityValidator
from geo_infer_test.models.types import ValidationRule


def test_iso_representations_and_precision_validate_individually():
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-11-01T01:30:00-07:00",
                "2026-11-01T01:30:00-08:00",
                "2026-11-01T09:30:00.000000123Z",
                datetime(2026, 11, 1, 9, 30, tzinfo=UTC),
            ],
            "value": [0.0, -2.0, "3.5", 4],
        },
        index=[9, 3, 9, -1],
    )
    original = frame.copy(deep=True)
    result = DataQualityValidator().validate(frame)
    assert result["quality_score"] == 1.0
    assert result["valid_records"] == 4
    assert result["field_quality"] == {"timestamp": 1.0, "value": 1.0}
    assert result["validation_errors"] == []
    pd.testing.assert_frame_equal(frame, original)


def test_one_bad_timestamp_does_not_invalidate_other_rows():
    result = DataQualityValidator().validate(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "value": 0,
        }
    )
    assert result["quality_score"] == 1.0
    mixed = DataQualityValidator().validate(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 0},
            {"timestamp": "invalid", "value": 2},
            {"timestamp": "2026-01-01T00:01:00.123Z", "value": 3},
            {"timestamp": "2026-01-01T00:02:00+00:00", "value": 4},
        ]
    )
    assert mixed["quality_score"] == 0.75
    assert mixed["valid_records"] == 3
    assert mixed["field_quality"]["timestamp"] == 0.75
    assert mixed["validation_errors"][0]["affected_records"] == 1


def test_overlapping_failures_count_each_record_once():
    result = DataQualityValidator().validate(
        [
            {"timestamp": None, "value": None},
            {"timestamp": "not a timestamp", "value": "not numeric"},
            {"timestamp": "2026-01-01T00:00:00Z", "value": 0.0},
            {"timestamp": "2026-01-01T00:01:00Z", "value": 1.0},
        ]
    )
    assert result["quality_score"] == 0.5
    assert result["valid_records"] == 2
    assert len(result["validation_errors"]) == 3
    assert {
        error["rule"]: error["affected_records"]
        for error in result["validation_errors"]
    } == {
        "no_nulls_in_required_fields": 1,
        "timestamp_format": 2,
        "numeric_values": 2,
    }


@pytest.mark.parametrize("missing", ["timestamp", "value"])
def test_missing_required_columns_are_invalid(missing):
    row = {"timestamp": "2026-01-01T00:00:00Z", "value": 0.0}
    del row[missing]
    result = DataQualityValidator().validate([row, row.copy()])
    assert result["quality_score"] == 0.0
    assert result["valid_records"] == 0
    assert result["validation_errors"][0]["affected_records"] == 2


@pytest.mark.parametrize(
    "value", [float("inf"), -float("inf"), float("nan"), True, False]
)
def test_nonfinite_and_boolean_values_are_invalid(value):
    result = DataQualityValidator().validate(
        {"timestamp": "2026-01-01T00:00:00Z", "value": value}
    )
    assert result["quality_score"] == 0.0
    assert result["valid_records"] == 0
    assert result["field_quality"]["value"] == 0.0


@pytest.mark.parametrize(
    "value", [np.array([1.0]), np.array(1.0), [1.0], {"value": 1.0}]
)
def test_containers_do_not_become_numeric_scalars(value):
    result = DataQualityValidator().validate(
        {"timestamp": "2026-01-01T00:00:00Z", "value": value}
    )
    assert result["valid_records"] == 0
    assert result["quality_score"] == 0.0
    assert result["field_quality"]["value"] == 0.0


@pytest.mark.parametrize("numeric_type", [complex, np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0, 2])
def test_observed_complex_values_do_not_discard_the_imaginary_part(
    numeric_type, imaginary
):
    result = DataQualityValidator().validate(
        {"timestamp": "2026-01-01T00:00:00Z", "value": numeric_type(1 + imaginary * 1j)}
    )
    assert result["valid_records"] == 0
    assert result["quality_score"] == 0.0
    assert result["field_quality"]["value"] == 0.0


@pytest.mark.parametrize("numeric_type", [np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0, 2])
@pytest.mark.parametrize("bound", ["min", "max"])
@pytest.mark.parametrize("data", [[], [{"value": 2}]])
def test_complex_bounds_are_invalid_even_without_records(
    numeric_type, imaginary, bound, data
):
    validator = DataQualityValidator()
    validator.validation_rules = [
        ValidationRule(
            "real range", "value", "range", {bound: numeric_type(1 + imaginary * 1j)}
        )
    ]
    with pytest.raises(ValueError, match="finite number"):
        validator.validate(data)


def test_range_checks_preserve_nondefault_and_duplicate_row_indexes():
    validator = DataQualityValidator()
    validator.validation_rules = [
        ValidationRule("range", "value", "range", {"min": 0, "max": 5})
    ]
    result = validator.validate(pd.DataFrame({"value": [0, 5, 6]}, index=[5, 5, 99]))
    assert result["valid_records"] == 2
    assert result["quality_score"] == pytest.approx(2 / 3)
    assert result["validation_errors"][0]["affected_records"] == 1


def test_field_quality_counts_distinct_failures_across_multiple_rules():
    validator = DataQualityValidator()
    validator.validation_rules = [
        ValidationRule("lower bound", "value", "range", {"min": 0}),
        ValidationRule("upper bound", "value", "range", {"max": 5}),
    ]
    result = validator.validate(pd.DataFrame({"value": [-1, 0, 5, 6]}))
    assert result["valid_records"] == 2
    assert result["quality_score"] == 0.5
    assert result["field_quality"] == {"value": 0.5}


def test_warning_penalty_applies_only_to_rows_without_an_error():
    validator = DataQualityValidator()
    validator.validation_rules = [
        ValidationRule("error", "value", "range", {"min": 0}),
        ValidationRule("warning", "value", "range", {"max": 5}, severity="warning"),
        ValidationRule(
            "warning overlap", "value", "range", {"min": 0}, severity="warning"
        ),
    ]
    result = validator.validate(pd.DataFrame({"value": [-1, 6, 0, 5]}))
    assert result["valid_records"] == 3
    assert result["quality_score"] == 0.625  # One error and one half-weight warning.


@pytest.mark.parametrize(
    "rule, message",
    [
        (ValidationRule("bad", "value", "unrecognized", {}), "rule type"),
        (ValidationRule("bad", "value", "range", {}, severity="fatal"), "severity"),
        (
            ValidationRule("range", "value", "range", {"min": 6, "max": 5}),
            "min must not exceed max",
        ),
        (
            ValidationRule("range", "value", "range", {"min": float("inf")}),
            "finite number",
        ),
        (ValidationRule("range", "value", "range", {"max": True}), "finite number"),
        (
            ValidationRule("timestamp", "timestamp", "format", {"format": "inferred"}),
            "expected iso",
        ),
        (
            ValidationRule("unrecognized", "value", "custom", {}),
            "custom validation rule",
        ),
        (
            ValidationRule(
                "no_nulls_in_required_fields",
                "*",
                "custom",
                {"required_fields": "value"},
            ),
            "column names",
        ),
        (
            ValidationRule(
                "no_nulls_in_required_fields",
                "*",
                "custom",
                {"required_fields": [None]},
            ),
            "column names",
        ),
    ],
)
@pytest.mark.parametrize(
    "data", [[], [{"timestamp": "2026-01-01T00:00:00Z", "value": 0}]]
)
def test_invalid_rule_configuration_fails_even_without_records(rule, message, data):
    validator = DataQualityValidator()
    validator.validation_rules = [rule]
    with pytest.raises(ValueError, match=message):
        validator.validate(data)


def test_empty_dataset_has_no_errors_or_warnings():
    result = DataQualityValidator().validate([])
    assert result["total_records"] == result["valid_records"] == 0
    assert result["quality_score"] == 1.0
    assert result["validation_errors"] == result["warnings"] == []
    assert result["field_quality"] == {"timestamp": 1.0, "value": 1.0}


@pytest.mark.parametrize(
    "error", [RuntimeError("broken parser"), UserWarning("unexpected warning")]
)
def test_unrelated_rule_failures_propagate(monkeypatch, error):
    validator = DataQualityValidator()

    def broken(*args):
        if isinstance(error, Warning):
            warnings.warn(error, stacklevel=2)
        raise error

    monkeypatch.setattr(validator, "_validate_format", broken)
    with pytest.raises(type(error), match=str(error)):
        validator.validate({"timestamp": "2026-01-01T00:00:00Z", "value": 0})


def test_implicit_numeric_epochs_and_nat_are_not_iso_timestamps():
    result = DataQualityValidator().validate(
        pd.DataFrame({"timestamp": [1, np.nan, pd.NaT], "value": [0, 0, 0]})
    )
    assert result["valid_records"] == 0
    assert result["quality_score"] == 0.0


def test_duplicate_columns_are_rejected_before_validation():
    with pytest.raises(ValueError, match="unique"):
        DataQualityValidator().validate(
            pd.DataFrame([[0, 1]], columns=["value", "value"])
        )
