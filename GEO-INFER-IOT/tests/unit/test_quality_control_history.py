"""Behavior tests for the QualityController measurement history.

These cover the path that previously could not run at all: temporal
consistency and window-based outlier detection depend on retained history,
so a controller that keeps no history silently passes everything.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from geo_infer_iot.core.quality_control import QualityController


def _measurement(sensor_id, value, minutes_ago, base=None):
    base = base or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return {
        "sensor_id": sensor_id,
        "value": value,
        "timestamp": (base - timedelta(minutes=minutes_ago)).isoformat(),
    }


@pytest.fixture(name="controller")
def _controller():
    return QualityController()


class TestMeasurementRetention:
    def test_validated_measurements_are_retained(self, controller):
        """Each validated measurement joins its sensor's history."""
        for offset in range(5):
            controller.validate_measurement(_measurement("s1", 10.0, 5 - offset))
        assert len(controller._get_recent_measurements("s1", minutes=60)) == 5

    def test_history_is_per_sensor(self, controller):
        """Sensors do not see each other's measurements."""
        controller.validate_measurement(_measurement("s1", 10.0, 1))
        controller.validate_measurement(_measurement("s2", 99.0, 1))
        s1 = controller._get_recent_measurements("s1", minutes=60)
        assert [entry["value"] for entry in s1] == [10.0]

    def test_unknown_sensor_has_no_history(self, controller):
        """Looking up a sensor never seen returns nothing, not an error."""
        assert controller._get_recent_measurements("never-seen", minutes=60) == []

    def test_history_is_bounded(self):
        """A long-lived controller keeps constant memory per sensor."""
        controller = QualityController({"history_size": 10})
        for offset in range(50):
            controller.validate_measurement(
                _measurement("s1", float(offset), 100 - offset)
            )
        assert len(controller.measurement_history["s1"]) == 10

    def test_window_excludes_older_measurements(self, controller):
        """Only measurements inside the lookback window come back."""
        controller.validate_measurement(_measurement("s1", 1.0, 600))
        controller.validate_measurement(_measurement("s1", 2.0, 5))
        recent = controller._get_recent_measurements("s1", minutes=60)
        assert [entry["value"] for entry in recent] == [2.0]

    def test_unusable_measurements_are_not_retained(self, controller):
        """Values or timestamps the checks cannot use are dropped."""
        controller.validate_measurement({"sensor_id": "s1", "value": None})
        controller.validate_measurement({"sensor_id": "s1", "value": 1.0})
        controller.validate_measurement(
            {
                "sensor_id": "s1",
                "value": float("nan"),
                "timestamp": "2026-01-01T12:00:00+00:00",
            }
        )
        assert controller._get_recent_measurements("s1", minutes=60) == []

    def test_datetime_timestamps_are_accepted(self, controller):
        """An aware datetime timestamp is retained the same as an ISO string."""
        controller.validate_measurement(
            {
                "sensor_id": "s1",
                "value": 5.0,
                "timestamp": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            }
        )
        assert len(controller._get_recent_measurements("s1", minutes=60)) == 1


class TestTimestampContract:
    """Timestamps are timezone-aware UTC end to end; naive input is rejected."""

    @pytest.mark.parametrize(
        "timestamp",
        ["2026-01-01T12:00:00", datetime(2026, 1, 1, 12, 0, 0)],
    )
    def test_naive_timestamps_are_rejected(self, controller, timestamp):
        with pytest.raises(ValueError, match="timezone-aware"):
            controller.validate_measurement(
                {"sensor_id": "s1", "value": 1.0, "timestamp": timestamp}
            )
        # Rejection happens before any state change.
        assert "s1" not in controller.measurement_history

    def test_malformed_timestamp_is_rejected(self, controller):
        with pytest.raises(ValueError, match="ISO-8601"):
            controller.validate_measurement(
                {"sensor_id": "s1", "value": 1.0, "timestamp": "x"}
            )

    def test_non_string_timestamp_is_rejected(self, controller):
        with pytest.raises(TypeError):
            controller.validate_measurement(
                {"sensor_id": "s1", "value": 1.0, "timestamp": 1_700_000_000}
            )

    def test_offset_timestamps_are_normalized_to_utc(self, controller):
        plus_two = timezone(timedelta(hours=2))
        controller.validate_measurement(
            {
                "sensor_id": "s1",
                "value": 1.0,
                "timestamp": "2026-01-01T14:00:00+02:00",
            }
        )
        controller.validate_measurement(
            {
                "sensor_id": "s1",
                "value": 1.0,
                "timestamp": datetime(2026, 1, 1, 14, 30, tzinfo=plus_two),
            }
        )
        stored = [entry["timestamp"] for entry in controller.measurement_history["s1"]]
        assert stored == [
            datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
        ]
        assert all(ts.tzinfo is UTC for ts in stored)

    def test_zulu_suffix_is_accepted(self, controller):
        controller.validate_measurement(
            {"sensor_id": "s1", "value": 1.0, "timestamp": "2026-01-01T12:00:00Z"}
        )
        stored = controller.measurement_history["s1"][0]["timestamp"]
        assert stored == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def test_report_timestamps_are_utc(self, controller):
        result = controller.validate_measurement(
            {"sensor_id": "s1", "value": 1.0, "timestamp": "2026-01-01T12:00:00Z"}
        )
        validated_at = datetime.fromisoformat(result.metadata["validation_timestamp"])
        assert validated_at.utcoffset() == timedelta(0)
        generated_at = controller.get_quality_report()["generated_at"]
        assert datetime.fromisoformat(generated_at).utcoffset() == timedelta(0)


class TestTemporalConsistency:
    def test_stable_series_passes(self, controller):
        """A flat series raises no temporal issue."""
        for offset in range(5):
            controller.validate_measurement(_measurement("s1", 10.0, 10 - offset))
        result = controller._validate_temporal_consistency(_measurement("s1", 10.0, 0))
        assert result.passed

    def test_a_spike_is_reported(self, controller):
        """A jump far beyond the allowed change rate is flagged.

        This is the check that could never fire while history was empty.
        """
        controller.validate_measurement(_measurement("s1", 10.0, 4))
        controller.validate_measurement(_measurement("s1", 10.0, 3))
        controller.validate_measurement(_measurement("s1", 900.0, 2))
        result = controller._validate_temporal_consistency(_measurement("s1", 900.0, 1))
        assert not result.passed
        assert any("change rate" in issue for issue in result.issues)

    def test_measurement_is_not_compared_against_itself(self, controller):
        """The first measurement for a sensor cannot be a temporal violation."""
        result = controller.validate_measurement(_measurement("s1", 10.0, 0))
        assert not any("change rate" in issue for issue in result.issues)
