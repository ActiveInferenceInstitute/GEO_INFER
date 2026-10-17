"""Tracer ownership, bounded work and statistical failure semantics."""

import tracemalloc

import pytest

from geo_infer_test.core.performance_monitor import (
    BenchmarkRunner,
    LoadTester,
    PerformanceMonitor,
)

pytestmark = pytest.mark.unit


def test_external_tracer_survives_multiple_monitors_and_body_failure():
    existing = tracemalloc.is_tracing()
    if not existing:
        tracemalloc.start()
    try:
        first, second = PerformanceMonitor(), PerformanceMonitor()
        with first.section("outer"):
            with pytest.raises(ValueError, match="body"):
                with second.section("inner"):
                    raise ValueError("body")
            assert tracemalloc.is_tracing()
            with pytest.raises(RuntimeError, match="active"):
                first.reset()
        assert tracemalloc.is_tracing()
        assert first.get_all_records()[0]["label"] == "outer"
        assert second.get_all_records()[0]["label"] == "inner"
    finally:
        if not existing:
            tracemalloc.stop()


def test_owned_tracer_survives_overlapping_monitors_until_last_stop():
    existing = tracemalloc.is_tracing()
    first, second = PerformanceMonitor(), PerformanceMonitor()
    first.start("one")
    second.start("two")
    first.stop()
    assert tracemalloc.is_tracing()
    second.stop()
    assert tracemalloc.is_tracing() == existing


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_work_bounds_are_rejected(value):
    with pytest.raises(ValueError):
        BenchmarkRunner(iterations=value)
    with pytest.raises(ValueError):
        LoadTester(concurrency=value)
    with pytest.raises(ValueError):
        LoadTester(total_requests=value)


def test_empty_exception_messages_still_count_as_load_failures():
    def fail():
        raise ValueError()

    result = LoadTester(concurrency=3, total_requests=17).run(fail)
    assert result["completed"] == result["errors"] == 17
    assert result["error_rate"] == 1


def test_nearest_rank_percentiles_and_native_benchmark_results():
    assert LoadTester._percentile(list(range(1, 21)), 0.95) == 19
    assert LoadTester._percentile([3, 1, 2], 0) == 1
    with pytest.raises(ValueError):
        LoadTester._percentile([1], float("nan"))
    result = BenchmarkRunner(iterations=5, warmup=0).run(sum, [1, 2, 3])
    assert result["min_s"] <= result["p95_s"] <= result["p99_s"] <= result["max_s"]
    assert result["operations_per_second"] > 0
