"""DATA schema instants obey the shared TIME normalization contract."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
import pandas as pd

from geo_infer_data.models.schemas import TemporalExtent, HealthStatus
from geo_infer_data.utils.caching import CacheEntry
from geo_infer_data.core.pipeline import TransformationEngine
from geo_infer_data.core.storage import AdaptiveDataStorage
from geo_infer_data.utils.indexing import TemporalIndexer


def test_temporal_extent_compares_utc_instants_across_offsets():
    extent = TemporalExtent(
        start="2026-10-01T01:00:00+02:00", end="2026-10-01T00:00:00Z"
    )
    assert extent.start == datetime(2026, 9, 30, 23, tzinfo=UTC)
    assert extent.end.tzinfo is UTC


@pytest.mark.parametrize("value", [datetime(2026, 10, 1), "2026-10-01T00:00:00"])
def test_temporal_extent_rejects_naive_inputs(value):
    with pytest.raises(ValueError, match="timezone-aware"):
        TemporalExtent(start=value, end=datetime.now(UTC))


def test_cache_uses_same_offset_normalization():
    value = datetime(2026, 10, 1, tzinfo=timezone(timedelta(hours=7)))
    cache = CacheEntry("test", None, created_at=value)
    assert cache.created_at == datetime(2026, 9, 30, 17, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        CacheEntry("test", None, created_at=value.replace(tzinfo=None))


def test_health_timestamp_assignment_rejects_naive_without_mutation():
    health = HealthStatus(status="healthy")
    original = health.checked_at
    with pytest.raises(ValueError, match="timezone-aware"):
        health.checked_at = datetime(2026, 10, 1)
    assert health.checked_at == original


@pytest.mark.parametrize("value", [1672531200, 1672531200.0, True])
def test_numeric_instants_require_explicit_unit_conversion(value):
    with pytest.raises(ValueError, match="datetime or an ISO-8601 string"):
        TemporalExtent(start=value, end=datetime.now(UTC))
    health = HealthStatus(status="healthy")
    original = health.checked_at
    with pytest.raises(ValueError, match="datetime or an ISO-8601 string"):
        health.checked_at = value
    assert health.checked_at == original


@pytest.mark.parametrize(
    "field,value", [("start", "2026-01-03T00:00:00Z"), ("end", "2025-12-31T00:00:00Z")]
)
def test_rejected_reversed_extent_assignment_preserves_interval(field, value):
    extent = TemporalExtent(start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z")
    original = extent.model_dump()
    with pytest.raises(ValueError, match="before or equal to end"):
        setattr(extent, field, value)
    assert extent.model_dump() == original


def test_temporal_index_sorts_instants_preserves_duplicates_precision_and_input():
    data = pd.DataFrame(
        {
            "timestamp": [
                "2026-11-01T01:30:00.000000009-08:00",
                "2026-11-01T01:30:00.000000009-07:00",
                "2026-11-01T08:30:00.000000009Z",
            ],
            "identity": ["later", "earlier", "equivalent"],
        }
    )
    original = data.copy(deep=True)
    indexer = TemporalIndexer()
    key = indexer.create_temporal_index(data, "timestamp")
    indexed = indexer.indexes[key]["data"]
    assert indexed.identity.tolist() == ["earlier", "equivalent", "later"]
    assert indexed.timestamp.iloc[0].nanosecond == 9
    assert indexer.query_by_time_point(
        key, "2026-11-01T01:30:00.000000009-07:00"
    ).identity.tolist() == ["earlier", "equivalent"]
    assert indexer.query_by_time_range(
        key, "2026-11-01T09:30:00Z", "2026-11-01T09:30:01Z"
    ).identity.tolist() == ["later"]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.asyncio
async def test_aggregation_uses_utc_bins_with_repeated_equivalent_offsets_and_dst_fold():
    data = pd.DataFrame(
        {
            "timestamp": [
                "2026-11-01T01:30:00-07:00",
                "2026-11-01T08:45:00Z",
                "2026-11-01T01:30:00-08:00",
                "2026-11-01T10:30:00+01:00",
            ],
            "value": [0.0, 4.0, 10.0, 14.0],
        }
    )
    original = data.copy(deep=True)
    result = await TransformationEngine()._temporal_aggregate(
        data, {"frequency": "1h", "aggregation": {"value": "mean"}}, {}
    )
    assert result.value.tolist() == [2.0, 12.0]
    assert result.timestamp.tolist() == list(
        pd.date_range("2026-11-01T08:00:00Z", periods=2, freq="h")
    )
    selected = AdaptiveDataStorage._filter_stored_data(
        data, temporal_range=("2026-11-01T01:00:00-08:00", "2026-11-01T10:00:00Z")
    )
    assert selected.value.tolist() == [10.0, 14.0]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("value", ["2026-01-01", 1672531200, pd.NaT])
@pytest.mark.asyncio
async def test_temporal_operations_reject_untyped_instants_before_mutation(value):
    data = pd.DataFrame({"timestamp": [value], "value": [0.0]})
    indexer = TemporalIndexer()
    with pytest.raises((TypeError, ValueError)):
        indexer.create_temporal_index(data, "timestamp")
    assert indexer.indexes == {}
    with pytest.raises((TypeError, ValueError)):
        AdaptiveDataStorage._filter_stored_data(
            data, temporal_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
        )
    with pytest.raises((TypeError, ValueError)):
        await TransformationEngine()._temporal_aggregate(data, {}, {})


@pytest.mark.parametrize(
    "interval",
    [
        ("2026-01-01", "2026-01-02T00:00:00Z"),
        ("2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"),
    ],
)
def test_invalid_query_intervals_fail_even_without_tabular_rows(interval):
    with pytest.raises(ValueError):
        AdaptiveDataStorage._filter_stored_data({}, temporal_range=interval)
