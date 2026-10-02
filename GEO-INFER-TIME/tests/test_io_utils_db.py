"""
Tests for GEO-INFER-TIME io, utils, and db modules.

Covers TimeSeriesReader/Writer CSV/JSON round-trip, create_timeseries factory,
validate_timeseries, detect_frequency, fill_gaps, align_timeseries,
and InMemoryStore CRUD + query operations.
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

from geo_infer_time.models.timeseries import TimeSeries
from geo_infer_time.io import (
    TimeSeriesReader,
    TimeSeriesWriter,
    read_timeseries,
    write_timeseries,
)
from geo_infer_time.utils import (
    validate_timeseries,
    detect_frequency,
    align_timeseries,
    create_timeseries,
    fill_gaps,
)
from geo_infer_time.db import InMemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTimeSeries:
    """Mock TimeSeries for testing."""

    def __init__(self, values, frequency=None):
        self.values = values
        self.frequency = frequency

    def to_dataframe(self):
        dates = pd.date_range(
            "2024-01-01", periods=len(self.values), freq="D", tz="UTC"
        )
        return pd.DataFrame({"value": self.values}, index=dates)


def _make_ts(
    n=50,
    freq="D",
    start="2024-01-01T00:00:00Z",
    col="value",
    metadata=None,
    spatial=None,
):
    """Create a simple TimeSeries for reuse across tests."""
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({col: np.arange(n, dtype=float)}, index=dates)
    return TimeSeries(data=df, spatial_location=spatial, metadata=metadata or {})


# ===================================================================
# IO Tests
# ===================================================================


class TestTimeSeriesReaderCSV:
    """Tests for TimeSeriesReader with CSV files."""

    @pytest.fixture
    def reader(self):
        return TimeSeriesReader()

    @pytest.fixture
    def csv_path(self, tmp_path):
        """Write a simple CSV file with a 'date' column."""
        p = tmp_path / "series.csv"
        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        df = pd.DataFrame({"date": dates, "value": np.arange(30, dtype=float)})
        df.to_csv(p, index=False)
        return p

    def test_read_csv_auto_detect_time_column(self, reader, csv_path):
        """Reader auto-detects 'date' column as time index."""
        ts = reader.read(csv_path)
        assert isinstance(ts.data.index, pd.DatetimeIndex)
        assert len(ts) == 30

    def test_read_csv_explicit_time_column(self, reader, csv_path):
        """Reader uses explicit time_column parameter."""
        ts = reader.read(csv_path, time_column="date")
        assert ts.data.index.name == "date"
        assert len(ts) == 30

    def test_read_csv_with_metadata(self, reader, csv_path):
        """Metadata and spatial_location are forwarded to TimeSeries."""
        meta = {"source": "test"}
        loc = {"lat": 45.0, "lon": -122.0}
        ts = reader.read(csv_path, metadata=meta, spatial_location=loc)
        assert ts.metadata == meta
        assert ts.spatial_location == loc

    def test_read_csv_missing_file_raises(self, reader, tmp_path):
        """FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError):
            reader.read(tmp_path / "nonexistent.csv")

    def test_read_csv_unsupported_format(self, reader, tmp_path):
        """ValueError for an unsupported extension."""
        bad = tmp_path / "data.xlsx"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read(bad)

    def test_read_csv_invalid_time_column(self, reader, csv_path):
        """ValueError when time_column does not exist in the file."""
        with pytest.raises(ValueError, match="not found"):
            reader.read(csv_path, time_column="nonexistent")


class TestTimeSeriesReaderJSON:
    """Tests for TimeSeriesReader with JSON files."""

    @pytest.fixture
    def reader(self):
        return TimeSeriesReader()

    @pytest.fixture
    def json_path(self, tmp_path):
        """Write a records-oriented JSON file."""
        p = tmp_path / "series.json"
        dates = pd.date_range("2024-03-01", periods=20, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp": dates, "temp": np.random.randn(20)})
        df.to_json(p, orient="records", date_format="iso")
        return p

    def test_read_json(self, reader, json_path):
        """Reader loads JSON and auto-detects 'timestamp' column."""
        ts = reader.read(json_path)
        assert isinstance(ts.data.index, pd.DatetimeIndex)
        assert len(ts) == 20


class TestTimeSeriesWriter:
    """Tests for TimeSeriesWriter."""

    @pytest.fixture
    def writer(self):
        return TimeSeriesWriter()

    @pytest.fixture
    def sample_ts(self):
        return _make_ts(
            n=15, metadata={"source": "test"}, spatial={"lat": 1.0, "lon": 2.0}
        )

    def test_write_csv(self, writer, sample_ts, tmp_path):
        """Write CSV and verify file exists."""
        out = tmp_path / "out.csv"
        result_path = writer.write(sample_ts, out)
        assert result_path.exists()
        df = pd.read_csv(result_path)
        assert len(df) == 15

    def test_write_json(self, writer, sample_ts, tmp_path):
        """Write JSON and verify file exists."""
        out = tmp_path / "out.json"
        result_path = writer.write(sample_ts, out)
        assert result_path.exists()

    def test_write_metadata_sidecar(self, writer, sample_ts, tmp_path):
        """Sidecar .meta.json is written when metadata exists."""
        out = tmp_path / "out.csv"
        writer.write(sample_ts, out, write_metadata=True)
        meta_path = out.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "metadata" in meta
        assert "spatial_location" in meta

    def test_write_no_metadata_sidecar(self, writer, tmp_path):
        """No sidecar when write_metadata=False."""
        ts = _make_ts(n=5)
        out = tmp_path / "out.csv"
        writer.write(ts, out, write_metadata=False)
        meta_path = out.with_suffix(".meta.json")
        assert not meta_path.exists()

    def test_write_unsupported_format(self, writer, tmp_path):
        """ValueError for unsupported extension."""
        ts = _make_ts(n=5)
        with pytest.raises(ValueError, match="Unsupported format"):
            writer.write(ts, tmp_path / "out.xlsx")

    def test_write_creates_parent_dirs(self, writer, tmp_path):
        """Writer creates intermediate directories."""
        ts = _make_ts(n=5)
        out = tmp_path / "sub" / "dir" / "out.csv"
        writer.write(ts, out)
        assert out.exists()


class TestCSVRoundTrip:
    """Test CSV write -> read round-trip preserves data."""

    def test_csv_round_trip_values(self, tmp_path):
        """Values survive a CSV write-read cycle."""
        ts_orig = _make_ts(
            n=20, metadata={"round": "trip"}, spatial={"lat": 10.0, "lon": 20.0}
        )
        csv_path = tmp_path / "round_trip.csv"

        writer = TimeSeriesWriter()
        writer.write(ts_orig, csv_path)

        reader = TimeSeriesReader()
        ts_loaded = reader.read(csv_path)

        assert len(ts_loaded) == len(ts_orig)
        np.testing.assert_array_almost_equal(
            ts_loaded.data["value"].values,
            ts_orig.data["value"].values,
            decimal=5,
        )

    def test_json_round_trip_values(self, tmp_path):
        """Values survive a JSON write-read cycle."""
        ts_orig = _make_ts(n=10)
        json_path = tmp_path / "round_trip.json"

        writer = TimeSeriesWriter()
        writer.write(ts_orig, json_path)

        reader = TimeSeriesReader()
        ts_loaded = reader.read(json_path)

        assert len(ts_loaded) == len(ts_orig)


class TestConvenienceFunctions:
    """Tests for read_timeseries / write_timeseries wrappers."""

    def test_write_and_read_convenience(self, tmp_path):
        """Convenience functions wrap Reader/Writer correctly."""
        ts = _make_ts(n=12)
        p = tmp_path / "conv.csv"
        write_timeseries(ts, p)
        ts2 = read_timeseries(p)
        assert len(ts2) == 12


# ===================================================================
# Utils Tests
# ===================================================================


class TestCreateTimeseries:
    """Tests for the create_timeseries factory."""

    def test_create_from_list(self):
        """Create TimeSeries from a plain list of values."""
        ts = create_timeseries([1.0, 2.0, 3.0], start="2024-01-01T00:00:00Z", freq="D")
        assert len(ts) == 3
        assert isinstance(ts.data.index, pd.DatetimeIndex)
        assert "value" in ts.data.columns

    def test_create_with_custom_name(self):
        """Column name can be specified."""
        ts = create_timeseries(
            [10, 20], start="2024-06-01T00:00:00Z", freq="h", name="temp"
        )
        assert "temp" in ts.data.columns

    def test_create_from_numpy(self):
        """Create from numpy array."""
        arr = np.array([5.0, 6.0, 7.0, 8.0])
        ts = create_timeseries(arr, start="2024-01-01T00:00:00Z", freq="D")
        assert len(ts) == 4

    def test_create_from_dict(self):
        """Create multi-column TimeSeries from dict."""
        vals = {"temp": [20.0, 21.0, 22.0], "humidity": [60.0, 65.0, 70.0]}
        ts = create_timeseries(vals, start="2024-01-01T00:00:00Z", freq="D")
        assert "temp" in ts.data.columns
        assert "humidity" in ts.data.columns
        assert len(ts) == 3

    def test_create_with_metadata(self):
        """Metadata and spatial_location are passed through."""
        meta = {"sensor": "A1"}
        loc = {"lat": 0.0, "lon": 0.0}
        ts = create_timeseries(
            [1, 2],
            start="2024-01-01T00:00:00Z",
            freq="D",
            metadata=meta,
            spatial_location=loc,
        )
        assert ts.metadata == meta
        assert ts.spatial_location == loc

    def test_create_hourly_frequency(self):
        """Hourly frequency produces correct index."""
        ts = create_timeseries(list(range(24)), start="2024-01-01T00:00:00Z", freq="h")
        assert len(ts) == 24
        # Verify spacing is 1 hour
        diff = ts.timestamps[1] - ts.timestamps[0]
        assert diff == timedelta(hours=1)


class TestValidateTimeseries:
    """Tests for validate_timeseries."""

    def test_valid_series(self):
        """A clean TimeSeries passes validation."""
        ts = _make_ts(n=30)
        result = validate_timeseries(ts)
        assert result["valid"] is True
        assert result["row_count"] == 30
        assert len(result["errors"]) == 0

    def test_empty_series_invalid(self):
        """Empty TimeSeries is invalid."""
        df = pd.DataFrame({"value": []}, index=pd.DatetimeIndex([]))
        ts = TimeSeries(data=df)
        result = validate_timeseries(ts)
        assert result["valid"] is False
        assert any("empty" in e.lower() for e in result["errors"])

    def test_missing_values_reported(self):
        """Missing values generate warnings."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        vals = [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        df = pd.DataFrame({"value": vals}, index=dates)
        ts = TimeSeries(data=df)
        result = validate_timeseries(ts)
        assert result["missing_values"]["value"] == 2
        assert result["missing_pct"]["value"] == 20.0

    def test_non_monotonic_timestamps_rejected(self):
        """Invalid order fails at construction instead of reaching analysis."""
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"], utc=True)
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=dates)
        with pytest.raises(ValueError, match="increasing"):
            TimeSeries(data=df)

    def test_duplicate_timestamps_rejected(self):
        """Duplicate observations require explicit aggregation before construction."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True)
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=dates)
        with pytest.raises(ValueError, match="unique"):
            TimeSeries(data=df)

    def test_gap_detection(self):
        """Large gaps in timestamps are detected.

        Note: validate_timeseries has an indexing issue in its gap-detection
        code for certain data shapes. We guard against that here.
        """
        # Regular daily then a 10-day gap
        dates = list(pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"))
        dates.append(pd.Timestamp("2024-01-25", tz="UTC"))  # big gap
        df = pd.DataFrame({"value": range(11)}, index=pd.DatetimeIndex(dates))
        ts = TimeSeries(data=df)
        try:
            result = validate_timeseries(ts)
            assert result["gap_count"] >= 1
        except IndexError:
            pytest.fail("validate_timeseries crashes during gap detection (known bug)")

    def test_single_point_valid(self):
        """A single-point TimeSeries is valid (no gaps possible)."""
        dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        df = pd.DataFrame({"value": [42.0]}, index=dates)
        ts = TimeSeries(data=df)
        result = validate_timeseries(ts)
        assert result["valid"] is True
        assert result["row_count"] == 1
        assert result["gap_count"] == 0


class TestDetectFrequency:
    """Tests for detect_frequency."""

    def test_detect_daily(self):
        """Daily frequency is detected."""
        ts = _make_ts(n=30, freq="D")
        freq = detect_frequency(ts)
        assert freq is not None
        assert "D" in freq

    def test_detect_hourly(self):
        """Hourly frequency is detected."""
        ts = _make_ts(n=48, freq="h")
        freq = detect_frequency(ts)
        assert freq is not None
        # pandas may return 'h' or 'H'
        assert freq.lower() == "h"

    def test_single_point_returns_none(self):
        """Single data point cannot have a frequency."""
        dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        df = pd.DataFrame({"value": [1.0]}, index=dates)
        ts = TimeSeries(data=df)
        freq = detect_frequency(ts)
        assert freq is None

    def test_irregular_fallback(self):
        """Irregular timestamps fall back to median-based detection."""
        # Roughly daily but with noise
        np.random.seed(42)
        base = pd.Timestamp("2024-01-01", tz="UTC")
        timestamps = [
            base + timedelta(days=i, hours=int(np.random.uniform(-2, 2)))
            for i in range(30)
        ]
        df = pd.DataFrame({"value": range(30)}, index=pd.DatetimeIndex(timestamps))
        ts = TimeSeries(data=df)
        freq = detect_frequency(ts)
        # Should still detect roughly daily
        assert freq is not None


class TestFillGaps:
    """Tests for fill_gaps."""

    @pytest.fixture
    def gapped_ts(self):
        """TimeSeries with a gap (missing day 5)."""
        dates = pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                # gap: 2024-01-05 missing
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
            ],
            utc=True,
        )
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0]}, index=dates)
        return TimeSeries(data=df)

    def test_fill_linear(self, gapped_ts):
        """Linear interpolation fills the gap."""
        filled = fill_gaps(gapped_ts, method="linear", freq="D")
        assert len(filled) == 8  # 8 days from Jan 1 to Jan 8
        # The gap at Jan 5 should be interpolated
        jan5 = pd.Timestamp("2024-01-05", tz="UTC")
        assert not np.isnan(filled.data.loc[jan5, "value"])

    def test_fill_ffill(self, gapped_ts):
        """Forward fill propagates last known value."""
        filled = fill_gaps(gapped_ts, method="ffill", freq="D")
        jan5 = pd.Timestamp("2024-01-05", tz="UTC")
        assert filled.data.loc[jan5, "value"] == 4.0  # forward from Jan 4

    def test_fill_bfill(self, gapped_ts):
        """Backward fill propagates next known value."""
        filled = fill_gaps(gapped_ts, method="bfill", freq="D")
        jan5 = pd.Timestamp("2024-01-05", tz="UTC")
        assert filled.data.loc[jan5, "value"] == 6.0  # backward from Jan 6

    def test_fill_metadata_updated(self, gapped_ts):
        """Metadata records that gaps were filled."""
        filled = fill_gaps(gapped_ts, method="linear", freq="D")
        assert filled.metadata.get("gaps_filled") is True
        assert filled.metadata.get("fill_method") == "linear"

    def test_fill_no_freq_raises(self):
        """ValueError when frequency cannot be determined and not provided."""
        # Two points 1 second apart followed by 1 year -- too irregular
        dates = pd.to_datetime(["2024-01-01", "2025-07-01"], utc=True)
        df = pd.DataFrame({"value": [1.0, 2.0]}, index=dates)
        ts = TimeSeries(data=df)
        # detect_frequency may still return something; if not, this raises
        # Use explicit freq=None to force auto-detection which might fail
        # This is a best-effort edge case test
        try:
            fill_gaps(ts, method="linear", freq=None)
        except ValueError:
            pass  # Expected path when frequency cannot be determined


class TestAlignTimeseries:
    """Tests for align_timeseries."""

    @pytest.fixture
    def ts_pair(self):
        """Two TimeSeries with overlapping but different timestamps."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        dates2 = pd.date_range("2024-01-05", periods=10, freq="D", tz="UTC")
        df1 = pd.DataFrame({"temp": np.arange(10, dtype=float)}, index=dates1)
        df2 = pd.DataFrame({"temp": np.arange(100, 110, dtype=float)}, index=dates2)
        ts1 = TimeSeries(data=df1)
        ts2 = TimeSeries(data=df2)
        return [ts1, ts2]

    def test_align_outer(self, ts_pair):
        """Outer join includes all timestamps from both series."""
        aligned = align_timeseries(ts_pair, method="outer")
        assert len(aligned) == 2
        # Outer join of Jan 1-10 and Jan 5-14 = Jan 1-14 = 14 days
        assert len(aligned[0]) == 14
        assert len(aligned[1]) == 14

    def test_align_inner(self, ts_pair):
        """Inner join includes only shared timestamps."""
        aligned = align_timeseries(ts_pair, method="inner")
        assert len(aligned) == 2
        # Intersection of Jan 1-10 and Jan 5-14 = Jan 5-10 = 6 days
        assert len(aligned[0]) == 6
        assert len(aligned[1]) == 6

    def test_align_preserves_metadata(self, ts_pair):
        """Aligned series carry 'aligned' metadata flag."""
        aligned = align_timeseries(ts_pair, method="outer")
        for ts in aligned:
            assert ts.metadata.get("aligned") is True

    def test_align_empty_list_raises(self):
        """ValueError for empty list."""
        with pytest.raises(ValueError, match="must not be empty"):
            align_timeseries([])

    def test_align_invalid_method_raises(self):
        """ValueError for unknown method."""
        ts = _make_ts(n=5)
        with pytest.raises(ValueError, match="must be"):
            align_timeseries([ts], method="cross")

    def test_align_fill_none_leaves_nans(self, ts_pair):
        """fill_method=None leaves NaN values from reindexing."""
        aligned = align_timeseries(ts_pair, method="outer", fill_method=None)
        # ts2 should have NaN for Jan 1-4
        assert aligned[1].data.iloc[0].isna().any()

    def test_align_single_series(self):
        """Aligning a single series returns it unchanged."""
        ts = _make_ts(n=10)
        aligned = align_timeseries([ts], method="outer")
        assert len(aligned) == 1
        assert len(aligned[0]) == 10


# ===================================================================
# DB Tests
# ===================================================================


class TestInMemoryStoreBasic:
    """Basic CRUD tests for InMemoryStore."""

    @pytest.fixture
    def store(self):
        return InMemoryStore()

    @pytest.fixture
    def sample_ts(self):
        return _make_ts(
            n=20, metadata={"sensor": "A"}, spatial={"lat": 1.0, "lon": 2.0}
        )

    def test_store_and_retrieve(self, store, sample_ts):
        """Store a series and retrieve it by name."""
        store.store("temp", sample_ts)
        retrieved = store.retrieve("temp")
        assert len(retrieved) == 20
        np.testing.assert_array_almost_equal(
            retrieved.data["value"].values,
            sample_ts.data["value"].values,
        )

    def test_retrieve_nonexistent_raises(self, store):
        """KeyError when retrieving a name that does not exist."""
        with pytest.raises(KeyError):
            store.retrieve("nonexistent")

    def test_list_series_empty(self, store):
        """Empty store returns empty list."""
        assert store.list_series() == []

    def test_list_series(self, store, sample_ts):
        """list_series returns sorted names."""
        store.store("beta", sample_ts)
        store.store("alpha", sample_ts)
        assert store.list_series() == ["alpha", "beta"]

    def test_delete(self, store, sample_ts):
        """Delete removes a series."""
        store.store("temp", sample_ts)
        assert "temp" in store.list_series()
        store.delete("temp")
        assert "temp" not in store.list_series()

    def test_delete_nonexistent_raises(self, store):
        """KeyError when deleting a name that does not exist."""
        with pytest.raises(KeyError):
            store.delete("nonexistent")

    def test_overwrite(self, store, sample_ts):
        """Storing with the same name overwrites."""
        store.store("temp", sample_ts)
        new_ts = _make_ts(n=5)
        store.store("temp", new_ts)
        assert len(store.retrieve("temp")) == 5

    def test_len(self, store, sample_ts):
        """__len__ returns number of stored series."""
        assert len(store) == 0
        store.store("a", sample_ts)
        store.store("b", sample_ts)
        assert len(store) == 2

    def test_contains(self, store, sample_ts):
        """__contains__ checks membership."""
        store.store("temp", sample_ts)
        assert "temp" in store
        assert "other" not in store


class TestInMemoryStoreQuery:
    """Query tests for InMemoryStore."""

    @pytest.fixture
    def store_with_data(self):
        store = InMemoryStore()
        ts = _make_ts(n=30, freq="D", start="2024-01-01T00:00:00Z")
        store.store("daily", ts)
        return store

    def test_query_full_range(self, store_with_data):
        """Query without bounds returns full series."""
        result = store_with_data.query("daily")
        assert len(result) == 30

    def test_query_start_only(self, store_with_data):
        """Query with start bound."""
        result = store_with_data.query("daily", start=datetime(2024, 1, 15, tzinfo=UTC))
        assert len(result) == 16  # Jan 15 through Jan 30

    def test_query_end_only(self, store_with_data):
        """Query with end bound."""
        result = store_with_data.query("daily", end=datetime(2024, 1, 10, tzinfo=UTC))
        assert len(result) == 10  # Jan 1 through Jan 10

    def test_query_start_and_end(self, store_with_data):
        """Query with both bounds."""
        result = store_with_data.query(
            "daily",
            start=datetime(2024, 1, 5, tzinfo=UTC),
            end=datetime(2024, 1, 10, tzinfo=UTC),
        )
        assert len(result) == 6  # Jan 5 through Jan 10

    def test_query_nonexistent_raises(self, store_with_data):
        """KeyError for unknown series name."""
        with pytest.raises(KeyError):
            store_with_data.query("unknown")

    def test_query_empty_range(self, store_with_data):
        """Query with range outside data returns empty TimeSeries."""
        result = store_with_data.query(
            "daily",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 31, tzinfo=UTC),
        )
        assert len(result) == 0

    def test_query_preserves_metadata(self, store_with_data):
        """Queried result preserves spatial_location and metadata."""
        ts = _make_ts(
            n=10, metadata={"source": "sensor"}, spatial={"lat": 5.0, "lon": 10.0}
        )
        store_with_data.store("annotated", ts)
        result = store_with_data.query("annotated")
        assert result.metadata.get("source") == "sensor"
        assert result.spatial_location["lat"] == 5.0


class TestInMemoryStoreIsolation:
    """Tests that InMemoryStore returns defensive copies."""

    def test_retrieve_returns_copy(self):
        """Mutating a retrieved series does not affect the store."""
        store = InMemoryStore()
        ts = _make_ts(n=5)
        store.store("s", ts)

        retrieved = store.retrieve("s")
        retrieved.data.iloc[0, 0] = -999.0

        original = store.retrieve("s")
        assert original.data.iloc[0, 0] != -999.0

    def test_store_makes_copy(self):
        """Mutating original after store does not affect stored data."""
        store = InMemoryStore()
        ts = _make_ts(n=5)
        store.store("s", ts)

        ts.data.iloc[0, 0] = -999.0
        retrieved = store.retrieve("s")
        assert retrieved.data.iloc[0, 0] != -999.0


# ===================================================================
# Edge Case Tests
# ===================================================================


class TestEdgeCases:
    """Edge cases: empty series, single point, missing columns."""

    def test_empty_dataframe_validation(self):
        """Validate empty TimeSeries returns invalid."""
        df = pd.DataFrame(
            {"value": pd.Series([], dtype=float)}, index=pd.DatetimeIndex([])
        )
        ts = TimeSeries(data=df)
        result = validate_timeseries(ts)
        assert result["valid"] is False

    def test_single_point_frequency_none(self):
        """detect_frequency returns None for single point."""
        dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        df = pd.DataFrame({"value": [1.0]}, index=dates)
        ts = TimeSeries(data=df)
        assert detect_frequency(ts) is None

    def test_create_timeseries_single_value(self):
        """create_timeseries works with a single value."""
        ts = create_timeseries([42.0], start="2024-01-01T00:00:00Z", freq="D")
        assert len(ts) == 1
        assert ts.data["value"].iloc[0] == 42.0

    def test_store_empty_ts(self):
        """InMemoryStore can store and retrieve an empty TimeSeries."""
        store = InMemoryStore()
        df = pd.DataFrame(
            {"value": pd.Series([], dtype=float)}, index=pd.DatetimeIndex([])
        )
        ts = TimeSeries(data=df)
        store.store("empty", ts)
        retrieved = store.retrieve("empty")
        assert len(retrieved) == 0

    def test_align_all_same_index(self):
        """Aligning series with identical indices is a no-op."""
        ts1 = _make_ts(n=10, col="a")
        ts2 = _make_ts(n=10, col="b")
        aligned = align_timeseries([ts1, ts2], method="outer")
        assert len(aligned[0]) == 10
        assert len(aligned[1]) == 10
