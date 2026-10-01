"""Regression tests for GS-278/GS-279 (silent-fabrication fixes).

GS-278: processing_latency_ms must reflect the real ingest latency
instrumented at IoTDataIngestion.ingest_measurement, and the latency
threshold warning must be able to fire.
GS-279: ingest_measurement must swallow only expected input errors;
unexpected internal exceptions propagate and are distinguishable.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from typing import Any
from unittest.mock import patch

import pytest

from geo_infer_iot.core.ingestion import IoTDataIngestion
from geo_infer_iot.performance import PerformanceMetrics, PerformanceMonitor


class _Registry:
    def __init__(self) -> None:
        self.sensors: dict[str, Any] = {}


def _make_ingestion() -> IoTDataIngestion:
    return IoTDataIngestion(registry=_Registry())


def _measurement(value: float = 25.5):
    from geo_infer_iot.core.ingestion import SensorMeasurement

    return SensorMeasurement(
        sensor_id="test_sensor",
        timestamp=datetime.now(UTC),
        variable="temperature",
        value=value,
        unit="celsius",
        latitude=40.7128,
        longitude=-74.0060,
    )


class TestIngestErrorContract:
    """GS-279: expected input errors return False; bugs propagate."""

    def test_malformed_value_returns_false_with_error_recorded(self):
        ingestion = _make_ingestion()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                ingestion.ingest_measurement({"sensor_id": "s1"})
            )
        finally:
            loop.close()
        assert result is False
        # The failure is recorded and distinguishable from success.
        assert ingestion.last_ingest_error is not None
        assert "KeyError" in ingestion.last_ingest_error

    def test_unexpected_internal_exception_propagates(self):
        ingestion = _make_ingestion()

        def boom(_measurement):
            raise RuntimeError("QC subsystem exploded")

        with patch.object(ingestion, "_validate_measurement", boom):
            loop = asyncio.new_event_loop()
            try:
                with pytest.raises(RuntimeError, match="QC subsystem exploded"):
                    loop.run_until_complete(
                        ingestion.ingest_measurement(_measurement())
                    )
            finally:
                loop.close()

    def test_success_clears_last_ingest_error(self):
        ingestion = _make_ingestion()
        loop = asyncio.new_event_loop()
        try:
            assert loop.run_until_complete(ingestion.ingest_measurement(_measurement()))
        finally:
            loop.close()
        assert ingestion.last_ingest_error is None


class _StubIoTSystem:
    """Stub exposing the surface PerformanceMonitor._collect_iot_metrics reads."""

    def __init__(self, latency_ms: float) -> None:
        self.ingestion = _StubIngestion(latency_ms)

    def get_system_status(self) -> dict[str, Any]:
        return {"measurements": 10, "error_count": 1}


class _StubIngestion:
    def __init__(self, latency_ms: float) -> None:
        self.measurements: list[Any] = [object() for _ in range(10)]
        self.processing_tasks: list[Any] = [object(), object()]
        self.last_ingest_latency_ms = latency_ms


class TestLatencyInstrumentation:
    """GS-278: real latency surfaces; the threshold can fire."""

    def test_processing_latency_reflects_measured_ingest_latency(self):
        monitor = PerformanceMonitor({"iot_metrics_interval_seconds": 1})
        monitor._collect_system_metrics()  # seed history
        monitor.iot_system = _StubIoTSystem(latency_ms=42.0)
        monitor._collect_iot_metrics()
        metrics = monitor.get_current_metrics()
        assert metrics is not None
        assert metrics.processing_latency_ms == pytest.approx(42.0)
        assert metrics.queue_size == 2
        assert metrics.error_rate == pytest.approx(0.1)

    def test_latency_threshold_warning_can_fire(self):
        monitor = PerformanceMonitor({"latency_threshold": 10.0})
        monitor.metrics_history.append(PerformanceMetrics())  # seed history
        monitor.iot_system = _StubIoTSystem(latency_ms=500.0)
        monitor._collect_iot_metrics()
        warnings = monitor._check_performance_thresholds(monitor.get_current_metrics())
        assert any("latency" in w.lower() for w in warnings)
