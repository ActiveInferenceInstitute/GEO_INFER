"""
Tests for GEO-INFER-TIME stream processing module.

Covers StreamProcessor: add_data_point, process_window, get_recent_windows,
window sliding, data eviction, and various window configurations.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, UTC

from geo_infer_time.core.stream_processing import StreamProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(base, seconds_offset):
    """Quick helper to create a timestamp offset from a base."""
    return base + timedelta(seconds=seconds_offset)


BASE_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


# ===================================================================
# Basic StreamProcessor Tests
# ===================================================================


class TestStreamProcessorInit:
    """Tests for StreamProcessor initialization."""

    def test_default_slide_interval(self):
        """slide_interval defaults to window_size when not specified."""
        sp = StreamProcessor(window_size=timedelta(minutes=5))
        assert sp.slide_interval == timedelta(minutes=5)

    def test_custom_slide_interval(self):
        """slide_interval can be set independently."""
        sp = StreamProcessor(
            window_size=timedelta(minutes=10),
            slide_interval=timedelta(minutes=2),
        )
        assert sp.window_size == timedelta(minutes=10)
        assert sp.slide_interval == timedelta(minutes=2)

    def test_default_aggregation_is_mean(self):
        """Default aggregation function is np.mean."""
        sp = StreamProcessor(window_size=timedelta(minutes=5))
        assert sp.aggregation_func == np.mean

    def test_custom_aggregation(self):
        """Custom aggregation function is accepted."""
        sp = StreamProcessor(window_size=timedelta(minutes=5), aggregation_func=np.sum)
        assert sp.aggregation_func == np.sum

    def test_empty_buffer_at_init(self):
        """Buffer and windows are empty at initialization."""
        sp = StreamProcessor(window_size=timedelta(minutes=5))
        assert len(sp.buffer) == 0
        assert len(sp.windows) == 0


# ===================================================================
# add_data_point Tests
# ===================================================================


class TestAddDataPoint:
    """Tests for adding data points to the stream."""

    @pytest.fixture
    def processor(self):
        return StreamProcessor(window_size=timedelta(seconds=60))

    def test_add_single_point(self, processor):
        """Adding a single point increases buffer size."""
        processor.add_data_point(BASE_TIME, 10.0)
        assert len(processor.buffer) == 1

    def test_add_multiple_points(self, processor):
        """Adding multiple points within window keeps them all."""
        for i in range(10):
            processor.add_data_point(_ts(BASE_TIME, i * 5), float(i))
        # All 10 points are within 45 seconds, window is 60s
        assert len(processor.buffer) == 10

    def test_add_point_with_metadata(self, processor):
        """Metadata is stored with the data point."""
        meta = {"sensor": "A", "quality": "good"}
        processor.add_data_point(BASE_TIME, 42.0, metadata=meta)
        assert processor.buffer[0]["metadata"] == meta

    def test_add_point_default_metadata(self, processor):
        """Metadata defaults to empty dict."""
        processor.add_data_point(BASE_TIME, 1.0)
        assert processor.buffer[0]["metadata"] == {}

    def test_data_point_structure(self, processor):
        """Each buffered point has timestamp, value, metadata keys."""
        processor.add_data_point(BASE_TIME, 99.0, metadata={"tag": "x"})
        point = processor.buffer[0]
        assert "timestamp" in point
        assert "value" in point
        assert "metadata" in point
        assert point["timestamp"] == BASE_TIME
        assert point["value"] == 99.0


# ===================================================================
# Data Eviction Tests
# ===================================================================


class TestDataEviction:
    """Tests for old data eviction from the buffer."""

    def test_eviction_removes_old_points(self):
        """Points older than window_size are removed."""
        sp = StreamProcessor(window_size=timedelta(seconds=10))
        # Add points at t=0, t=5, t=11 (t=0 should be evicted at t=11)
        sp.add_data_point(_ts(BASE_TIME, 0), 1.0)
        sp.add_data_point(_ts(BASE_TIME, 5), 2.0)
        sp.add_data_point(_ts(BASE_TIME, 11), 3.0)
        # t=0 is older than t=11 - 10s = t=1, so evicted
        assert len(sp.buffer) == 2
        assert sp.buffer[0]["value"] == 2.0

    def test_eviction_keeps_boundary_point(self):
        """Point exactly at window boundary is kept (cutoff is strict <)."""
        sp = StreamProcessor(window_size=timedelta(seconds=10))
        sp.add_data_point(_ts(BASE_TIME, 0), 1.0)
        sp.add_data_point(_ts(BASE_TIME, 10), 2.0)
        # cutoff = t=10 - 10s = t=0; buffer[0].timestamp == t=0 is NOT < t=0
        assert len(sp.buffer) == 2

    def test_eviction_all_old_points(self):
        """All points can be evicted if they are too old."""
        sp = StreamProcessor(window_size=timedelta(seconds=5))
        sp.add_data_point(_ts(BASE_TIME, 0), 1.0)
        sp.add_data_point(_ts(BASE_TIME, 1), 2.0)
        # Jump far ahead
        sp.add_data_point(_ts(BASE_TIME, 100), 3.0)
        assert len(sp.buffer) == 1
        assert sp.buffer[0]["value"] == 3.0

    def test_no_eviction_within_window(self):
        """No eviction when all points are within the window."""
        sp = StreamProcessor(window_size=timedelta(minutes=60))
        for i in range(100):
            sp.add_data_point(_ts(BASE_TIME, i), float(i))
        # 100 points spanning 99 seconds, window is 3600 seconds
        assert len(sp.buffer) == 100

    def test_gradual_eviction(self):
        """Points are gradually evicted as the window slides forward."""
        sp = StreamProcessor(window_size=timedelta(seconds=5))
        counts = []
        for i in range(20):
            sp.add_data_point(_ts(BASE_TIME, i), float(i))
            counts.append(len(sp.buffer))
        # Buffer should stabilize around 5-6 points (window = 5 seconds)
        assert max(counts) <= 7  # rough upper bound
        assert counts[-1] <= 6


# ===================================================================
# process_window Tests
# ===================================================================


class TestProcessWindow:
    """Tests for processing the current window."""

    @pytest.fixture
    def loaded_processor(self):
        """Processor with 10 data points loaded."""
        sp = StreamProcessor(window_size=timedelta(seconds=60))
        for i in range(10):
            sp.add_data_point(_ts(BASE_TIME, i * 5), float(i + 1))
        return sp

    def test_process_window_returns_result(self, loaded_processor):
        """process_window returns a dictionary with expected keys."""
        result = loaded_processor.process_window()
        assert result is not None
        assert "window_start" in result
        assert "window_end" in result
        assert "count" in result
        assert "aggregated_value" in result
        assert "min" in result
        assert "max" in result
        assert "std" in result

    def test_process_window_count(self, loaded_processor):
        """Count matches number of points in buffer."""
        result = loaded_processor.process_window()
        assert result["count"] == 10

    def test_process_window_aggregation_mean(self, loaded_processor):
        """Default aggregation (mean) is correct."""
        result = loaded_processor.process_window()
        expected_mean = np.mean([float(i + 1) for i in range(10)])
        assert abs(result["aggregated_value"] - expected_mean) < 1e-10

    def test_process_window_min_max(self, loaded_processor):
        """Min and max are correct."""
        result = loaded_processor.process_window()
        assert result["min"] == 1.0
        assert result["max"] == 10.0

    def test_process_window_empty_buffer(self):
        """process_window returns None for empty buffer."""
        sp = StreamProcessor(window_size=timedelta(seconds=60))
        result = sp.process_window()
        assert result is None

    def test_process_window_single_point(self):
        """process_window works with a single point."""
        sp = StreamProcessor(window_size=timedelta(seconds=60))
        sp.add_data_point(BASE_TIME, 42.0)
        result = sp.process_window()
        assert result is not None
        assert result["count"] == 1
        assert result["aggregated_value"] == 42.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["std"] == 0.0

    def test_process_window_custom_aggregation(self):
        """Custom aggregation function (sum) is used."""
        sp = StreamProcessor(
            window_size=timedelta(seconds=60),
            aggregation_func=np.sum,
        )
        for i in range(5):
            sp.add_data_point(_ts(BASE_TIME, i), 10.0)
        result = sp.process_window()
        assert result["aggregated_value"] == 50.0

    def test_process_window_median_aggregation(self):
        """Custom aggregation function (median) is used."""
        sp = StreamProcessor(
            window_size=timedelta(seconds=60),
            aggregation_func=np.median,
        )
        values = [1.0, 2.0, 3.0, 100.0, 200.0]
        for i, v in enumerate(values):
            sp.add_data_point(_ts(BASE_TIME, i), v)
        result = sp.process_window()
        assert result["aggregated_value"] == 3.0

    def test_process_window_appends_to_windows(self, loaded_processor):
        """Each process_window call appends to the windows list."""
        assert len(loaded_processor.windows) == 0
        loaded_processor.process_window()
        assert len(loaded_processor.windows) == 1
        loaded_processor.process_window()
        assert len(loaded_processor.windows) == 2

    def test_process_window_timestamps_are_iso(self, loaded_processor):
        """Window start/end are ISO format strings."""
        result = loaded_processor.process_window()
        # Should be parseable as datetime
        datetime.fromisoformat(result["window_start"])
        datetime.fromisoformat(result["window_end"])


# ===================================================================
# get_recent_windows Tests
# ===================================================================


class TestGetRecentWindows:
    """Tests for retrieving recent processing windows."""

    @pytest.fixture
    def processor_with_windows(self):
        """Processor with 5 processed windows."""
        sp = StreamProcessor(window_size=timedelta(seconds=10))
        for batch in range(5):
            offset = batch * 10
            for i in range(5):
                sp.add_data_point(_ts(BASE_TIME, offset + i), float(batch * 10 + i))
            sp.process_window()
        return sp

    def test_get_all_recent(self, processor_with_windows):
        """get_recent_windows returns all when count >= total."""
        windows = processor_with_windows.get_recent_windows(count=10)
        assert len(windows) == 5

    def test_get_limited_recent(self, processor_with_windows):
        """get_recent_windows returns only the last N."""
        windows = processor_with_windows.get_recent_windows(count=3)
        assert len(windows) == 3

    def test_get_default_count(self, processor_with_windows):
        """Default count is 10."""
        windows = processor_with_windows.get_recent_windows()
        assert len(windows) == 5  # only 5 exist

    def test_empty_windows(self):
        """get_recent_windows returns empty list when no windows exist."""
        sp = StreamProcessor(window_size=timedelta(seconds=60))
        windows = sp.get_recent_windows()
        assert windows == []

    def test_recent_windows_are_ordered(self, processor_with_windows):
        """Windows are returned in chronological order (oldest first)."""
        windows = processor_with_windows.get_recent_windows(count=5)
        for i in range(len(windows) - 1):
            assert windows[i]["window_start"] <= windows[i + 1]["window_start"]


# ===================================================================
# Window Configuration Tests
# ===================================================================


class TestWindowConfigurations:
    """Tests with various window sizes and slide intervals."""

    def test_small_window(self):
        """Small window (1 second) evicts quickly."""
        sp = StreamProcessor(window_size=timedelta(seconds=1))
        sp.add_data_point(_ts(BASE_TIME, 0), 1.0)
        sp.add_data_point(_ts(BASE_TIME, 0.5), 2.0)
        sp.add_data_point(_ts(BASE_TIME, 2), 3.0)
        # Only the point at t=2 should remain (t=0 and t=0.5 are before t=1)
        assert len(sp.buffer) == 1

    def test_large_window(self):
        """Large window (1 hour) keeps all points."""
        sp = StreamProcessor(window_size=timedelta(hours=1))
        for i in range(100):
            sp.add_data_point(_ts(BASE_TIME, i * 30), float(i))
        # 100 points over 2970 seconds, well within 3600s window
        assert len(sp.buffer) == 100

    def test_minute_window_with_second_slide(self):
        """1-minute window with 10-second slide interval."""
        sp = StreamProcessor(
            window_size=timedelta(minutes=1),
            slide_interval=timedelta(seconds=10),
        )
        # Add 120 seconds of data (1 point per second)
        for i in range(120):
            sp.add_data_point(_ts(BASE_TIME, i), float(i))
        # Buffer should have ~60 points (last minute of data)
        assert 55 <= len(sp.buffer) <= 65

    def test_process_window_after_eviction(self):
        """Process window after some data has been evicted."""
        sp = StreamProcessor(window_size=timedelta(seconds=5))
        # Add 20 points spanning 19 seconds
        for i in range(20):
            sp.add_data_point(_ts(BASE_TIME, i), float(i))
        result = sp.process_window()
        # Only last ~5 seconds of data should be in the window
        assert result["count"] <= 7
        assert result["count"] >= 4


# ===================================================================
# Stress / Sequence Tests
# ===================================================================


class TestStreamSequence:
    """Test realistic streaming sequences."""

    def test_continuous_stream_with_periodic_processing(self):
        """Simulate continuous data feed with periodic window processing."""
        sp = StreamProcessor(window_size=timedelta(seconds=30))
        results = []

        for i in range(100):
            sp.add_data_point(_ts(BASE_TIME, i), np.sin(i / 10.0) * 100)
            # Process every 10 points
            if (i + 1) % 10 == 0:
                result = sp.process_window()
                if result is not None:
                    results.append(result)

        assert len(results) == 10
        # Each result should have valid structure
        for r in results:
            assert r["count"] > 0
            assert r["min"] <= r["aggregated_value"] <= r["max"]

    def test_bursty_data(self):
        """Handle bursty data -- many points at same timestamp."""
        sp = StreamProcessor(window_size=timedelta(seconds=10))
        # Burst of 50 points at the same time
        for i in range(50):
            sp.add_data_point(BASE_TIME, float(i))
        assert len(sp.buffer) == 50

        result = sp.process_window()
        assert result["count"] == 50
        assert result["min"] == 0.0
        assert result["max"] == 49.0

    def test_gaps_in_stream(self):
        """Handle gaps in the data stream."""
        sp = StreamProcessor(window_size=timedelta(seconds=10))
        # Add some data, then a gap, then more data
        for i in range(5):
            sp.add_data_point(_ts(BASE_TIME, i), float(i))

        # Big gap
        sp.add_data_point(_ts(BASE_TIME, 100), 100.0)

        # Old data should be evicted
        assert len(sp.buffer) == 1
        result = sp.process_window()
        assert result["count"] == 1
        assert result["aggregated_value"] == 100.0


class TestStreamProcessorContracts:
    """Validation and state-management contracts for stream processing."""

    @pytest.mark.parametrize(
        ("kwargs", "exception"),
        [
            ({"window_size": timedelta(0)}, ValueError),
            ({"window_size": timedelta(seconds=-1)}, ValueError),
            ({"window_size": 60}, TypeError),
            (
                {
                    "window_size": timedelta(seconds=10),
                    "slide_interval": timedelta(0),
                },
                ValueError,
            ),
            (
                {
                    "window_size": timedelta(seconds=10),
                    "aggregation_func": "mean",
                },
                TypeError,
            ),
        ],
    )
    def test_constructor_rejects_invalid_configuration(self, kwargs, exception):
        with pytest.raises(exception):
            StreamProcessor(**kwargs)

    @pytest.mark.parametrize(
        ("timestamp", "value", "exception"),
        [
            ("2024-01-01", 1.0, TypeError),
            (BASE_TIME, True, TypeError),
            (BASE_TIME, "not-a-number", TypeError),
            (BASE_TIME, np.nan, ValueError),
            (BASE_TIME, np.inf, ValueError),
        ],
    )
    def test_add_data_point_rejects_invalid_values(self, timestamp, value, exception):
        processor = StreamProcessor(timedelta(minutes=1))

        with pytest.raises(exception):
            processor.add_data_point(timestamp, value)

    def test_metadata_is_copied_at_ingress(self):
        processor = StreamProcessor(timedelta(minutes=1))
        metadata = {"sensor": "A"}

        processor.add_data_point(BASE_TIME, 1, metadata)
        metadata["sensor"] = "B"

        assert processor.buffer[0]["metadata"] == {"sensor": "A"}
        assert processor.buffer[0]["value"] == 1.0

    def test_recent_window_count_contract(self):
        processor = StreamProcessor(timedelta(minutes=1))
        processor.add_data_point(BASE_TIME, 1)
        processor.process_window()

        assert processor.get_recent_windows(0) == []
        with pytest.raises(ValueError, match="non-negative"):
            processor.get_recent_windows(-1)
        with pytest.raises(TypeError, match="integer"):
            processor.get_recent_windows(1.5)


class TestWindowingAndEventMethods:
    """Exercise public windowing, late-data, event, and reset methods."""

    @pytest.fixture
    def processor(self):
        processor = StreamProcessor(
            window_size=timedelta(seconds=60),
            slide_interval=timedelta(seconds=20),
        )
        for offset, value in [(0, 1), (10, 2), (30, 3), (50, 4), (60, 5)]:
            processor.add_data_point(_ts(BASE_TIME, offset), value)
        return processor

    def test_tumbling_sliding_and_session_windows(self, processor):
        tumbling = processor.process_tumbling_windows()
        sliding = processor.process_sliding_windows()
        sessions = processor.process_session_windows(timedelta(seconds=15))

        assert [window["count"] for window in tumbling] == [4, 1]
        assert [window["count"] for window in sliding] == [4, 3, 2, 1]
        assert [window["count"] for window in sessions] == [2, 1, 2]

    def test_session_gap_validation(self, processor):
        with pytest.raises(ValueError, match="non-negative"):
            processor.process_session_windows(timedelta(seconds=-1))
        with pytest.raises(TypeError, match="timedelta"):
            processor.process_session_windows(10)

    def test_late_data_watermark_and_flush(self):
        processor = StreamProcessor(timedelta(minutes=1))
        processor.add_data_point(_ts(BASE_TIME, 10), 10)
        processor.add_data_point(_ts(BASE_TIME, 5), 5)

        assert processor.get_watermark() == _ts(BASE_TIME, 10)
        assert [point["value"] for point in processor.get_late_data()] == [5.0]
        assert processor.get_stats()["late_arrivals"] == 1
        assert processor.flush_late_data()[0]["value"] == 5.0
        assert processor.get_late_data() == []

    def test_threshold_events_call_registered_handler(self):
        processor = StreamProcessor(timedelta(minutes=1))
        handled = []
        processor.register_event_handler("threshold_breach", handled.append)
        for offset, value in enumerate([-5, 0, 10]):
            processor.add_data_point(_ts(BASE_TIME, offset), value)

        events = processor.detect_threshold_events(
            lower_threshold=-1, upper_threshold=5
        )

        assert [event["direction"] for event in events] == ["lower", "upper"]
        assert handled == events
        assert processor.get_stats()["events_detected"] == 2

        with pytest.raises(ValueError, match="cannot exceed"):
            processor.detect_threshold_events(lower_threshold=2, upper_threshold=1)

    def test_anomaly_detection_calls_handler_and_validates_threshold(self):
        processor = StreamProcessor(timedelta(minutes=1))
        handled = []
        processor.register_event_handler("anomaly", handled.append)
        for offset, value in enumerate([0] * 20 + [100]):
            processor.add_data_point(_ts(BASE_TIME, offset), value)

        anomalies = processor.detect_anomalies_zscore(3)

        assert len(anomalies) == 1
        assert anomalies[0]["value"] == 100.0
        assert handled == anomalies

        with pytest.raises(ValueError, match="greater than zero"):
            processor.detect_anomalies_zscore(0)

    def test_summary_stats_and_reset(self, processor):
        processor.process_window()
        summary = processor.get_buffer_summary()
        stats = processor.get_stats()

        assert summary["size"] == 5
        assert summary["mean"] == 3.0
        assert stats == {
            "total_points": 5,
            "total_windows": 1,
            "late_arrivals": 0,
            "events_detected": 0,
            "anomaly_alerts": 0,
        }

        processor.reset()
        assert processor.get_buffer_summary() == {
            "size": 0,
            "window_start": None,
            "window_end": None,
            "watermark": None,
            "watermark_delay_seconds": None,
        }
        assert processor.get_stats()["total_points"] == 0
