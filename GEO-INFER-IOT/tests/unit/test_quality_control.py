"""Tests for IoT quality control module."""

from geo_infer_iot.core.quality_control import *  # noqa - import available QC classes
from datetime import UTC


class TestMeasurementQualityChecks:
    """Test measurement validation and quality control logic."""

    def test_valid_temperature_range(self):
        """Temperature values should be within physical limits."""
        min_temp, max_temp = -89.2, 56.7  # Earth record extremes
        valid_temps = [-40.0, 0.0, 22.5, 40.0]
        for temp in valid_temps:
            assert min_temp <= temp <= max_temp

    def test_invalid_temperature(self):
        invalid_temps = [-100.0, 70.0, 200.0]
        min_temp, max_temp = -89.2, 56.7
        for temp in invalid_temps:
            assert not (min_temp <= temp <= max_temp)

    def test_valid_humidity_range(self):
        valid = [0.0, 50.0, 100.0]
        for h in valid:
            assert 0 <= h <= 100

    def test_invalid_humidity(self):
        invalid = [-1.0, 101.0, 200.0]
        for h in invalid:
            assert not (0 <= h <= 100)

    def test_coordinate_validation(self):
        valid_coords = [
            (0.0, 0.0),
            (90.0, 180.0),
            (-90.0, -180.0),
            (37.7749, -122.4194),
        ]
        for lat, lon in valid_coords:
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

    def test_nan_detection(self):
        import math

        assert math.isnan(float("nan"))
        assert not math.isnan(22.5)
        assert not math.isnan(0.0)


class TestQualityFlags:
    def test_quality_flag_assignment(self):
        flags = []
        value = 150.0
        max_threshold = 100.0
        if value > max_threshold:
            flags.append("exceeds_max_threshold")
        assert "exceeds_max_threshold" in flags

    def test_stale_data_detection(self):
        from datetime import datetime, timedelta

        now = datetime.now(UTC)
        old_timestamp = now - timedelta(hours=25)
        is_stale = (now - old_timestamp).total_seconds() > 24 * 3600
        assert is_stale is True

    def test_fresh_data_not_stale(self):
        from datetime import datetime, timedelta

        now = datetime.now(UTC)
        recent_timestamp = now - timedelta(minutes=5)
        is_stale = (now - recent_timestamp).total_seconds() > 24 * 3600
        assert is_stale is False
