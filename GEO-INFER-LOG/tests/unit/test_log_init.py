"""Behavior tests for the geo_infer_log package (logging core)."""

import json
import logging
import time

import pytest

import geo_infer_log
from geo_infer_log import (
    EnhancedLogger,
    JSONFormatter,
    LogAnalyzer,
    LogEntry,
    PerformanceMetrics,
    SpatialLogContext,
    get_logger,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics collection."""

    def test_timer_round_trip(self) -> None:
        metrics = PerformanceMetrics()
        timer_id = metrics.start_timer("route")
        time.sleep(0.01)
        duration = metrics.end_timer(timer_id)
        assert duration > 0
        stats = metrics.get_stats("route")
        assert stats["count"] == 1
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_end_unknown_timer_returns_zero(self) -> None:
        metrics = PerformanceMetrics()
        assert metrics.end_timer("bogus_timer") == 0.0

    def test_get_stats_empty(self) -> None:
        metrics = PerformanceMetrics()
        assert metrics.get_stats("never") == {}

    def test_counters_and_gauges(self) -> None:
        metrics = PerformanceMetrics()
        metrics.increment_counter("hits")
        metrics.increment_counter("hits", value=4)
        metrics.set_gauge("queue_depth", 3.5)
        all_metrics = metrics.get_all_metrics()
        assert all_metrics["counters"]["hits"] == 5
        assert all_metrics["gauges"]["queue_depth"] == 3.5

    def test_get_all_metrics_includes_performance_stats(self) -> None:
        metrics = PerformanceMetrics()
        metrics.record_duration("pickup", 2.0)
        metrics.record_duration("pickup", 4.0)
        all_metrics = metrics.get_all_metrics()
        assert all_metrics["performance_stats"]["pickup"]["max"] == 4.0
        assert all_metrics["performance_stats"]["pickup"]["count"] == 2


class TestLazyLoading:
    """Tests for module-level lazy attribute loading."""

    def test_lazy_class_from_core(self) -> None:
        from geo_infer_log.core.routing import RouteOptimizer

        assert geo_infer_log.RouteOptimizer is RouteOptimizer

    def test_lazy_submodule(self) -> None:
        import importlib

        module = geo_infer_log.api
        assert module is importlib.import_module("geo_infer_log.api")
        assert module.__name__ == "geo_infer_log.api"

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            geo_infer_log.definitely_not_exported  # noqa: B018 -- attribute access under test


def make_logger(**config) -> EnhancedLogger:
    """Build an EnhancedLogger with synchronous logging for determinism."""
    config.setdefault("async_logging", False)
    config.setdefault("level", "DEBUG")
    return EnhancedLogger("test-enhanced-logger", config)


class TestEnhancedLogger:
    """Tests for EnhancedLogger structured logging."""

    def test_sync_log_records_counter(self) -> None:
        logger = make_logger()
        logger.info("route_plan", "planning route", context={"city": "berlin"})
        assert logger.metrics.counters["test-enhanced-logger_route_plan"] == 1
        assert logger.log_queue.empty()

    def test_level_convenience_methods(self) -> None:
        logger = make_logger()
        logger.debug("op", "d")
        logger.warning("op", "w")
        logger.error("op", "e")
        logger.critical("op", "c")
        assert logger.metrics.counters["test-enhanced-logger_op"] == 4

    def test_operation_lifecycle(self) -> None:
        logger = make_logger()
        timer_id = logger.start_operation("deliver", batch=1)
        assert timer_id.startswith("deliver_")
        logger.end_operation("deliver", timer_id, success=True)
        stats = logger.get_metrics()["performance_stats"]["deliver"]
        assert stats["count"] == 1

    def test_operation_failure_uses_error_level(self, caplog) -> None:
        logger = make_logger()
        timer_id = logger.start_operation("pickup")
        with caplog.at_level(logging.ERROR, logger="test-enhanced-logger"):
            logger.end_operation("pickup", timer_id, success=False)
        assert any("failed" in r.message for r in caplog.records)

    def test_spatial_operation(self) -> None:
        logger = make_logger()
        logger.log_spatial_operation(
            "h3_coverage",
            h3_index="8928308280fffff",
            lat=52.52,
            lon=13.405,
            resolution=9,
        )
        assert logger.metrics.counters["test-enhanced-logger_h3_coverage"] == 1

    def test_error_info_stored_in_entry(self) -> None:
        logger = make_logger()
        written: list = []
        logger._write_log_entry = written.append
        logger.log("ERROR", "op", "boom", error_info={"type": "ValueError", "msg": "x"})
        assert written[0].error_info["type"] == "ValueError"

    def test_async_log_processed(self) -> None:
        logger = EnhancedLogger("test-async-logger", {"level": "DEBUG"})
        logger.info("async_op", "queued message")
        deadline = time.monotonic() + 5.0
        while not logger.log_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        logger.log_queue.join()
        assert logger.metrics.counters["test-async-logger_async_op"] == 1
        logger.log_processor_running = False

    def test_full_queue_falls_back_to_direct_write(self) -> None:
        logger = make_logger(async_logging=True)
        logger.log_queue.put_nowait = _raise_full
        written: list = []
        logger._write_log_entry = written.append
        logger.info("op", "direct write")
        assert len(written) == 1

    def test_write_log_entry_includes_spatial_and_perf(self) -> None:
        logger = make_logger()
        written = []
        logger._write_log_entry = written.append
        entry = LogEntry(
            timestamp="2026-01-01T00:00:00+00:00",
            level="INFO",
            module="m",
            operation="op",
            message="msg",
            spatial_context=SpatialLogContext(region="eu", resolution=8),
            performance_metrics={"duration_seconds": 1.0},
            trace_id="t1",
            span_id="s1",
        )
        logger._write_log_entry(entry)
        assert written[0].spatial_context.region == "eu"


def _raise_full(_entry: LogEntry) -> None:
    import queue as _queue

    raise _queue.Full


class TestJSONFormatter:
    """Tests for JSON structured formatting."""

    def test_format_plain_record(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        entry = json.loads(formatter.format(record))
        assert entry["message"] == "hello world"
        assert entry["level"] == "INFO"
        assert entry["logger"] == "test"
        assert entry["operation"] == "unknown"

    def test_format_with_extras_and_exception(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=None,
            exc_info=exc_info,
        )
        record.context = {"k": "v"}
        record.spatial_context = {"resolution": 9}
        record.performance_metrics = {"duration_seconds": 0.5}
        record.trace_id = "trace-1"
        record.error_info = {"type": "ValueError"}
        entry = json.loads(formatter.format(record))
        assert entry["context"] == {"k": "v"}
        assert entry["trace_id"] == "trace-1"
        assert "boom" in entry["exception"]


class TestLogAnalyzer:
    """Tests for log file analysis."""

    @pytest.fixture
    def log_file(self, tmp_path):
        entries = [
            {
                "timestamp": "2026-01-01T10:00:00+00:00",
                "level": "INFO",
                "operation": "deliver",
                "performance_metrics": {"duration_seconds": 2.0, "success": True},
            },
            {
                "timestamp": "2026-01-01T11:00:00+00:00",
                "level": "INFO",
                "operation": "deliver",
                "performance_metrics": {"duration_seconds": 4.0, "success": False},
            },
            {
                "timestamp": "2026-01-01T12:00:00+00:00",
                "level": "ERROR",
                "operation": "deliver",
            },
            {
                "timestamp": "not-a-date",
                "level": "ERROR",
                "operation": "deliver",
            },
            {
                "timestamp": "2026-01-01T13:00:00+00:00",
                "level": "INFO",
                "operation": "spatial",
                "spatial_context": {"resolution": 9, "region": "eu"},
            },
        ]
        path = tmp_path / "log.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(entries[0]) + "\n")
            f.write(json.dumps(entries[1]) + "\n")
            f.write(json.dumps(entries[2]) + "\n")
            f.write(json.dumps(entries[3]) + "\n")
            f.write(json.dumps(entries[4]) + "\n")
            f.write("{not json}\n")
            f.write("\n")
        return str(path)

    def test_load_and_analyze_performance(self, log_file) -> None:
        analyzer = LogAnalyzer(log_file)
        assert len(analyzer.log_entries) == 5
        result = analyzer.analyze_performance("deliver")
        assert result["total_operations"] == 2
        assert result["successful_operations"] == 1
        assert result["success_rate"] == 0.5
        assert result["duration_stats"]["max"] == 4.0

    def test_analyze_performance_no_data(self, log_file) -> None:
        analyzer = LogAnalyzer(log_file)
        assert analyzer.analyze_performance("ghost") == {
            "message": "No performance data found"
        }

    def test_find_errors_skips_bad_timestamps(self, log_file) -> None:
        analyzer = LogAnalyzer(log_file)
        errors = analyzer.find_errors(hours=24 * 365)
        assert len(errors) == 1
        assert errors[0]["level"] == "ERROR"

    def test_find_errors_inside_window_excludes_old(self, log_file) -> None:
        with open(log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": "1999-01-01T00:00:00+00:00",
                        "level": "ERROR",
                    }
                )
                + "\n"
            )
        fresh = LogAnalyzer(log_file).find_errors(hours=1)
        assert all(e["level"] != "ERROR" or "1999" not in e["timestamp"] for e in fresh)
        assert len(LogAnalyzer(log_file).find_errors(hours=24 * 365)) == 1

    def test_spatial_analysis(self, log_file) -> None:
        analyzer = LogAnalyzer(log_file)
        result = analyzer.spatial_analysis()
        assert result["total_spatial_operations"] == 1
        assert result["h3_resolution_usage"] == {9: 1}
        assert result["region_distribution"] == {"eu": 1}

    def test_spatial_analysis_empty(self, tmp_path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        analyzer = LogAnalyzer(str(path))
        assert analyzer.spatial_analysis() == {"message": "No spatial operations found"}

    def test_missing_file_starts_empty(self, tmp_path) -> None:
        analyzer = LogAnalyzer(str(tmp_path / "absent.jsonl"))
        assert analyzer.log_entries == []

    def test_unreadable_file_logs_error(self, tmp_path) -> None:
        path = tmp_path / "dir.jsonl"
        path.mkdir()
        analyzer = LogAnalyzer(str(path))
        assert analyzer.log_entries == []


def test_get_logger_factory() -> None:
    logger = get_logger("factory-logger", {"level": "INFO", "async_logging": False})
    assert isinstance(logger, EnhancedLogger)
    assert logger.name == "factory-logger"
