#!/usr/bin/env python3

"""
Tests for the telemetry module: metric collection, event tracking, health monitoring.
"""

import unittest

from geo_infer_agent.api.telemetry import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricType,
    TelemetryService,
    TimerMetric,
)


class TestCounterMetric(unittest.TestCase):
    """Tests for CounterMetric."""

    def test_counter_starts_at_zero(self) -> None:
        """A new counter metric starts with value 0."""
        counter = CounterMetric("requests", "Total requests")
        self.assertEqual(counter.value, 0)
        self.assertEqual(counter.metric_type, MetricType.COUNTER)

    def test_increment_by_default(self) -> None:
        """Incrementing by default adds 1."""
        counter = CounterMetric("requests", "Total requests")
        counter.increment()
        self.assertEqual(counter.value, 1)
        counter.increment()
        self.assertEqual(counter.value, 2)

    def test_increment_by_amount(self) -> None:
        """Incrementing by a specific amount works correctly."""
        counter = CounterMetric("bytes", "Bytes transferred")
        counter.increment(100)
        counter.increment(50)
        self.assertEqual(counter.value, 150)

    def test_to_dict_includes_value(self) -> None:
        """Serialized dict includes counter value."""
        counter = CounterMetric("hits", "Cache hits", agent_id="agent-1")
        counter.increment(5)
        d = counter.to_dict()
        self.assertEqual(d["value"], 5)
        self.assertEqual(d["name"], "hits")
        self.assertEqual(d["agent_id"], "agent-1")
        self.assertEqual(d["type"], MetricType.COUNTER)


class TestGaugeMetric(unittest.TestCase):
    """Tests for GaugeMetric."""

    def test_gauge_set_value(self) -> None:
        """Gauge can be set to an arbitrary value."""
        gauge = GaugeMetric("cpu", "CPU usage")
        gauge.set(72.5)
        self.assertAlmostEqual(gauge.value, 72.5)

    def test_gauge_increment_and_decrement(self) -> None:
        """Gauge supports both increment and decrement."""
        gauge = GaugeMetric("connections", "Active connections")
        gauge.set(10)
        gauge.increment(3)
        self.assertEqual(gauge.value, 13)
        gauge.decrement(5)
        self.assertEqual(gauge.value, 8)

    def test_gauge_to_dict(self) -> None:
        """Gauge serialization preserves value and metadata."""
        gauge = GaugeMetric("mem", "Memory usage", tags={"unit": "percent"})
        gauge.set(55.0)
        d = gauge.to_dict()
        self.assertEqual(d["value"], 55.0)
        self.assertEqual(d["tags"]["unit"], "percent")


class TestHistogramMetric(unittest.TestCase):
    """Tests for HistogramMetric."""

    def test_histogram_records_values(self) -> None:
        """Histogram tracks count, min, max, and mean correctly."""
        hist = HistogramMetric("latency", "Request latency", max_samples=100)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            hist.record(v)

        self.assertEqual(hist.count, 5)
        self.assertAlmostEqual(hist.min, 10.0)
        self.assertAlmostEqual(hist.max, 50.0)
        self.assertAlmostEqual(hist.mean(), 30.0)
        self.assertAlmostEqual(hist.sum, 150.0)

    def test_histogram_mean_empty(self) -> None:
        """Mean returns None for an empty histogram."""
        hist = HistogramMetric("empty", "Empty histogram")
        self.assertIsNone(hist.mean())

    def test_histogram_respects_max_samples(self) -> None:
        """Histogram enforces max_samples limit via deque."""
        hist = HistogramMetric("bounded", "Bounded", max_samples=3)
        for v in [1, 2, 3, 4, 5]:
            hist.record(v)
        # Only last 3 values stored in deque
        self.assertEqual(len(hist.values), 3)
        self.assertEqual(list(hist.values), [3, 4, 5])
        # But count tracks all recordings
        self.assertEqual(hist.count, 5)

    def test_histogram_to_dict(self) -> None:
        """Histogram dict includes summary statistics."""
        hist = HistogramMetric("size", "Response size")
        hist.record(100)
        hist.record(200)
        d = hist.to_dict()
        self.assertEqual(d["count"], 2)
        self.assertEqual(d["min"], 100)
        self.assertEqual(d["max"], 200)
        self.assertAlmostEqual(d["mean"], 150.0)


class TestTimerMetric(unittest.TestCase):
    """Tests for TimerMetric."""

    def test_timer_start_stop_records_duration(self) -> None:
        """Starting and stopping a timer records a non-negative duration."""
        timer = TimerMetric("process_time", "Processing time")
        timer.start()
        duration = timer.stop()
        self.assertGreaterEqual(duration, 0.0)
        # The internal histogram should have one recorded value
        self.assertEqual(timer.histogram.count, 1)

    def test_timer_stop_without_start_raises(self) -> None:
        """Stopping a timer that was never started raises ValueError."""
        timer = TimerMetric("bad_timer", "Not started")
        with self.assertRaises(ValueError):
            timer.stop()

    def test_timer_to_dict(self) -> None:
        """Timer serialization includes running state and histogram."""
        timer = TimerMetric("op", "Operation time")
        d = timer.to_dict()
        self.assertFalse(d["is_running"])
        self.assertIn("histogram", d)


class TestTelemetryService(unittest.TestCase):
    """Tests for TelemetryService metric registration and retrieval."""

    def setUp(self) -> None:
        """Reset the singleton for clean tests."""
        # Force re-initialization by resetting the singleton
        TelemetryService._instance = None
        self.service = TelemetryService()

    def tearDown(self) -> None:
        """Clean up singleton state."""
        TelemetryService._instance = None

    def test_register_counter(self) -> None:
        """Registering a counter creates and returns a CounterMetric."""
        counter = self.service.register_counter("test_counter", "Test")
        self.assertIsInstance(counter, CounterMetric)
        self.assertEqual(counter.name, "test_counter")

    def test_register_gauge(self) -> None:
        """Registering a gauge creates and returns a GaugeMetric."""
        gauge = self.service.register_gauge("test_gauge", "Test")
        self.assertIsInstance(gauge, GaugeMetric)

    def test_register_histogram(self) -> None:
        """Registering a histogram creates and returns a HistogramMetric."""
        hist = self.service.register_histogram("test_hist", "Test")
        self.assertIsInstance(hist, HistogramMetric)

    def test_register_timer(self) -> None:
        """Registering a timer creates and returns a TimerMetric."""
        timer = self.service.register_timer("test_timer", "Test")
        self.assertIsInstance(timer, TimerMetric)

    def test_duplicate_registration_returns_same_metric(self) -> None:
        """Registering the same metric twice returns the existing instance."""
        c1 = self.service.register_counter("dup", "Duplicate")
        c1.increment(10)
        c2 = self.service.register_counter("dup", "Duplicate")
        self.assertIs(c1, c2)
        self.assertEqual(c2.value, 10)

    def test_get_metrics_all(self) -> None:
        """get_metrics returns all registered metrics."""
        self.service.register_counter("c1", "Counter 1")
        self.service.register_gauge("g1", "Gauge 1")
        metrics = self.service.get_metrics()
        self.assertEqual(len(metrics), 2)

    def test_get_metrics_filtered_by_agent(self) -> None:
        """get_metrics with agent_id only returns that agent's metrics."""
        self.service.register_counter("c1", "Counter", agent_id="agent-a")
        self.service.register_counter("c2", "Counter", agent_id="agent-b")
        metrics = self.service.get_metrics(agent_id="agent-a")
        self.assertEqual(len(metrics), 1)

    def test_update_and_get_health(self) -> None:
        """Health status can be set and retrieved per agent."""
        self.service.update_health("agent-1", "healthy", {"uptime": 3600})
        health = self.service.get_health_status("agent-1")
        self.assertEqual(health["agent-1"]["status"], "healthy")
        self.assertEqual(health["agent-1"]["details"]["uptime"], 3600)

    def test_get_health_unknown_agent(self) -> None:
        """Health for an unknown agent returns status 'unknown'."""
        health = self.service.get_health_status("ghost")
        self.assertEqual(health["ghost"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
