"""
DOMAIN-01 Acceptance tests for GEO-INFER-OPS documented features.

These tests exercise real implemented behavior for documented features that
previously lacked focused acceptance tests:

1. is_port_in_use — port availability checking used by start_metrics_server.
2. start_metrics_server port selection — verifies the context manager
   finds an available port and cleans up on exit (integration with
   is_port_in_use).

No mocks, stubs, or placeholders: every assertion exercises actual code paths.
"""

import socket

from geo_infer_ops.core.monitoring import (
    is_port_in_use,
    start_metrics_server,
    record_request,
    record_error,
    get_metric_value,
    reset_metrics,
)


def _kernel_assigned_port() -> int:
    """Return a port the kernel just handed out as free (not a fixed number)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# is_port_in_use
# ---------------------------------------------------------------------------


class TestIsPortInUse:
    """Acceptance: port availability checking works correctly."""

    def test_free_port_returns_false(self):
        """A port with nothing listening returns False."""
        # Find a definitely-free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            free_port = s.getsockname()[1]
        # After closing, the port should be free
        assert is_port_in_use(free_port) is False

    def test_occupied_port_returns_true(self):
        """A port with an active listener returns True."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            s.listen(1)
            occupied_port = s.getsockname()[1]
            assert is_port_in_use(occupied_port) is True

    def test_returns_bool(self):
        """The function always returns a boolean."""
        result = is_port_in_use(8080)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# start_metrics_server port auto-selection
# ---------------------------------------------------------------------------


class TestMetricsServerPortSelection:
    """Acceptance: start_metrics_server uses is_port_in_use to find a free port."""

    def test_yields_a_port(self):
        """The context manager yields a valid port number."""
        with start_metrics_server(port=9094) as port:
            assert isinstance(port, int)
            assert port >= 9094

    def test_cleans_up_after_exit(self):
        """After exiting, the port is released."""
        with start_metrics_server(port=_kernel_assigned_port()) as port:
            assert is_port_in_use(port) is True
        # Exit runs server.shutdown() + server_close() synchronously and the
        # server never accepted a connection (no TIME_WAIT), so the release
        # is observable immediately; no sleep or grace period is needed.
        assert is_port_in_use(port) is False

    def test_shifts_to_next_port_if_occupied(self):
        """If the requested port is busy, the server moves to the next free one."""
        # Occupy a kernel-assigned port so parallel runs never collide on a
        # fixed number.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
            blocker.bind(("localhost", 0))
            blocker.listen(1)
            occupied = blocker.getsockname()[1]
            with start_metrics_server(port=occupied) as port:
                assert port != occupied  # Must have shifted
                assert port > occupied


# ---------------------------------------------------------------------------
# record_request / get_metric_value integration
# ---------------------------------------------------------------------------


class TestMetricRecordingIntegration:
    """Acceptance: metric recording and retrieval work together."""

    def setup_method(self):
        reset_metrics()

    def test_record_and_retrieve(self):
        """A recorded request metric is retrievable via get_metric_value."""
        record_request("test_module", "/api/test", 200, 0.05)
        value = get_metric_value(
            "http_requests_total",
            labels={"module": "test_module", "endpoint": "/api/test", "status": "200"},
        )
        assert value >= 1.0

    def test_record_error_increments(self):
        """Recording an error increments the error counter."""
        record_error("err_module", "timeout")
        errors = get_metric_value(
            "http_errors_total",
            labels={"module": "err_module", "error_type": "timeout"},
        )
        assert errors >= 1.0
