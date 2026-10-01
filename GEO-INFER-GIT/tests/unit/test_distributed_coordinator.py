"""Tests for the GEO-INFER-GIT distributed coordinator lifecycle and bookkeeping.

Regression coverage for GS-215: constructing a DistributedCoordinator must not
bind network ports or spawn threads; background services start only via an
explicit start()/stop() lifecycle.
"""

import socket
import time
from datetime import datetime, timedelta, UTC

from geo_infer_git.core.distributed_coordinator import (
    DistributedCoordinator,
    JobInfo,
    NodeInfo,
)


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` for up to ``timeout`` seconds; slow CI scheduling
    must not fail an assertion that only needs a grace window."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_coordinator(role: str = "worker") -> DistributedCoordinator:
    return DistributedCoordinator(
        role=role,
        discovery_port=_free_udp_port(),
        coordination_port=_free_tcp_port(),
    )


class TestConstructionBindsNoPorts:
    def test_worker_construction_binds_no_ports(self):
        """Constructing a worker must not claim the discovery or coordination port."""
        coordinator = _make_coordinator(role="worker")

        # If the constructor had bound either port, these binds would fail.
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("", coordinator.discovery_port))
        udp.close()
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.bind(("", coordinator.coordination_port))
        tcp.close()

    def test_coordinator_construction_binds_no_ports(self):
        """Even coordinator/master roles must not bind ports at construction time."""
        coordinator = _make_coordinator(role="coordinator")

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("", coordinator.discovery_port))
        udp.close()
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.bind(("", coordinator.coordination_port))
        tcp.close()

    def test_construction_spawns_no_threads(self):
        for role in ("worker", "coordinator", "master"):
            coordinator = _make_coordinator(role=role)
            assert coordinator.discovery_thread is None
            assert coordinator.coordination_thread is None
            assert coordinator.heartbeat_thread is None
            assert coordinator.job_scheduler_thread is None
            assert not coordinator.shutdown_event.is_set()


class TestLifecycle:
    def test_start_spawns_and_stop_joins_threads(self):
        """start() launches services; stop() signals shutdown and joins threads."""
        coordinator = _make_coordinator(role="coordinator")
        coordinator.start()

        threads = (
            coordinator.discovery_thread,
            coordinator.coordination_thread,
            coordinator.heartbeat_thread,
            coordinator.job_scheduler_thread,
        )
        # Threads are created synchronously by start(); aliveness is polled
        # because a freshly started thread may not be observed immediately.
        assert all(thread is not None for thread in threads)
        assert _wait_until(lambda: all(t.is_alive() for t in threads))

        coordinator.stop()

        assert coordinator.shutdown_event.is_set()
        # All managed threads must have exited (bounded join inside stop()).
        # The grace poll covers a thread finishing just after its join
        # timeout; the strict final assertion still holds.
        assert _wait_until(lambda: not any(t.is_alive() for t in threads))
        for thread in threads:
            assert thread is not None
            assert not thread.is_alive()

    def test_stop_without_start_is_a_noop(self):
        coordinator = _make_coordinator(role="coordinator")
        coordinator.stop()
        assert coordinator.shutdown_event.is_set()


class TestDataclassBookkeeping:
    def test_nodeinfo_defaults(self):
        node = NodeInfo(node_id="n1", hostname="h", ip_address="127.0.0.1", port=1)
        assert node.role == "worker"
        assert node.status == "active"
        assert node.capabilities == []
        assert node.current_load == 0.0
        assert node.max_load == 1.0
        assert node.last_heartbeat.tzinfo is not None

    def test_jobinfo_defaults(self):
        job = JobInfo(job_id="j1", job_type="export")
        assert job.status == "pending"
        assert job.priority == 1
        assert job.progress == 0.0
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.assigned_node is None
        assert job.created_at.tzinfo is not None
        assert job.started_at is None
        assert job.completed_at is None

    def test_constructor_registers_self_as_active_node(self):
        coordinator = _make_coordinator()
        assert coordinator.node_id in coordinator.nodes
        assert coordinator.nodes[coordinator.node_id].status == "active"
        assert coordinator.current_node.node_id == coordinator.node_id


class TestMessageDispatch:
    def test_node_register_adds_node(self):
        coordinator = _make_coordinator()
        coordinator._process_message(
            {
                "type": "node_register",
                "node_info": {
                    "node_id": "w1",
                    "hostname": "worker1",
                    "ip_address": "127.0.0.3",
                    "port": 7000,
                    "role": "worker",
                    "capabilities": ["export"],
                },
            }
        )
        assert "w1" in coordinator.nodes
        assert coordinator.nodes["w1"].ip_address == "127.0.0.3"
        assert coordinator.nodes["w1"].capabilities == ["export"]
        assert coordinator.nodes["w1"].status == "active"

    def test_node_unregister_removes_node(self):
        coordinator = _make_coordinator()
        coordinator.register_node(
            NodeInfo(node_id="w1", hostname="h", ip_address="127.0.0.1", port=5)
        )
        coordinator._process_message({"type": "node_unregister", "node_id": "w1"})
        assert "w1" not in coordinator.nodes

    def test_heartbeat_refreshes_node(self):
        coordinator = _make_coordinator()
        coordinator.register_node(
            NodeInfo(node_id="w1", hostname="h", ip_address="127.0.0.1", port=5)
        )
        stale = coordinator.nodes["w1"].last_heartbeat
        coordinator.nodes["w1"].status = "inactive"

        coordinator._process_message({"type": "heartbeat", "node_id": "w1"})

        assert coordinator.nodes["w1"].status == "active"
        assert coordinator.nodes["w1"].last_heartbeat >= stale

    def test_unknown_message_type_does_not_raise(self):
        coordinator = _make_coordinator()
        coordinator._process_message({"type": "bogus_type"})
        coordinator._process_message({})

    def test_job_complete_marks_job_and_frees_node_load(self):
        coordinator = _make_coordinator(role="coordinator")
        node = NodeInfo(node_id="w1", hostname="h", ip_address="127.0.0.1", port=5)
        coordinator.register_node(node)
        job_id = coordinator.submit_job("export")
        job = coordinator.jobs[job_id]
        job.assigned_node = "w1"
        job.status = "running"
        coordinator.nodes["w1"].current_load = 0.1

        coordinator._process_message({"type": "job_complete", "job_id": job_id})

        assert job.status == "completed"
        assert job.progress == 100.0
        assert job.completed_at is not None
        assert coordinator.nodes["w1"].current_load == 0.0

    def test_job_failed_retries_then_fails_permanently(self):
        coordinator = _make_coordinator()
        job_id = coordinator.submit_job("export", priority=2)
        job = coordinator.jobs[job_id]

        # First failure: below max_retries, job is re-queued.
        coordinator._process_message(
            {"type": "job_failed", "job_id": job_id, "error": "boom"}
        )
        assert job.retry_count == 1
        assert job.status == "pending"
        assert job.assigned_node is None
        # The retry is queued on top of the original submission entry.
        assert coordinator.job_queue.qsize() == 2

        # Exhaust remaining retries: job fails permanently.
        coordinator._process_message(
            {"type": "job_failed", "job_id": job_id, "error": "boom"}
        )
        coordinator._process_message(
            {"type": "job_failed", "job_id": job_id, "error": "boom"}
        )
        assert job.retry_count == 3
        assert job.status == "failed"
        assert job.completed_at is not None

    def test_job_failed_for_unknown_job_is_ignored(self):
        coordinator = _make_coordinator()
        coordinator._process_message(
            {"type": "job_failed", "job_id": "missing", "error": "boom"}
        )


class TestHeartbeatTimeouts:
    def test_stale_node_marked_inactive(self):
        coordinator = _make_coordinator()
        coordinator.register_node(
            NodeInfo(
                node_id="w1",
                hostname="h",
                ip_address="127.0.0.1",
                port=5,
                last_heartbeat=_utc_now_minus(45),
            )
        )

        coordinator._check_heartbeats(_utc_now())

        assert coordinator.nodes["w1"].status == "inactive"

    def test_long_dead_node_removed(self):
        coordinator = _make_coordinator()
        coordinator.register_node(
            NodeInfo(
                node_id="w1",
                hostname="h",
                ip_address="127.0.0.1",
                port=5,
                last_heartbeat=_utc_now_minus(90),
            )
        )

        coordinator._check_heartbeats(_utc_now())

        assert "w1" not in coordinator.nodes

    def test_fresh_node_untouched_and_self_never_checked(self):
        coordinator = _make_coordinator()
        coordinator.register_node(
            NodeInfo(node_id="w1", hostname="h", ip_address="127.0.0.1", port=5)
        )
        # Make even the coordinator itself appear stale; it must be exempt.
        coordinator.current_node.last_heartbeat = _utc_now_minus(120)

        coordinator._check_heartbeats(_utc_now())

        assert coordinator.nodes["w1"].status == "active"
        assert coordinator.node_id in coordinator.nodes
        assert coordinator.current_node.status == "active"


class TestJobScheduling:
    def test_schedule_once_assigns_job_to_available_node(self):
        coordinator = _make_coordinator(role="coordinator")
        node = NodeInfo(
            node_id="w1",
            hostname="h",
            ip_address="127.0.0.1",
            port=5,
            capabilities=["export"],
        )
        coordinator.register_node(node)
        job_id = coordinator.submit_job("export")

        coordinator._schedule_once()

        job = coordinator.get_job_status(job_id)
        assert job is not None
        assert job.status == "running"
        assert job.assigned_node == "w1"
        assert job.started_at is not None
        assert coordinator.job_queue.qsize() == 0

    def test_schedule_once_requeues_job_without_suitable_node(self):
        coordinator = _make_coordinator(role="coordinator")
        coordinator.register_node(
            NodeInfo(
                node_id="w1",
                hostname="h",
                ip_address="127.0.0.1",
                port=5,
                capabilities=["render"],
            )
        )
        job_id = coordinator.submit_job(
            "export", metadata={"required_capabilities": ["export"]}
        )

        coordinator._schedule_once()

        job = coordinator.get_job_status(job_id)
        assert job.status == "pending"
        assert job.assigned_node is None
        assert coordinator.job_queue.qsize() == 1

    def test_lowest_load_node_wins(self):
        coordinator = _make_coordinator(role="coordinator")
        busy = NodeInfo(node_id="busy", hostname="h", ip_address="127.0.0.1", port=5)
        idle = NodeInfo(node_id="idle", hostname="h", ip_address="127.0.0.2", port=6)
        busy.current_load = 0.8
        coordinator.register_node(busy)
        coordinator.register_node(idle)

        job_id = coordinator.submit_job("export")
        coordinator._schedule_once()

        assert coordinator.get_job_status(job_id).assigned_node == "idle"

    def test_select_node_for_job_respects_required_capabilities(self):
        coordinator = _make_coordinator()
        job = JobInfo(
            job_id="j1",
            job_type="render",
            metadata={"required_capabilities": ["gpu"]},
        )
        gpu_node = NodeInfo(
            node_id="gpu",
            hostname="h",
            ip_address="127.0.0.1",
            port=5,
            capabilities=["gpu"],
        )
        plain_node = NodeInfo(
            node_id="cpu", hostname="h", ip_address="127.0.0.2", port=6
        )

        assert coordinator._select_node_for_job(job, [plain_node]) is None
        assert coordinator._select_node_for_job(job, [plain_node, gpu_node]) is gpu_node

    def test_cancel_job(self):
        coordinator = _make_coordinator()
        job_id = coordinator.submit_job("export")

        assert coordinator.cancel_job(job_id) is True
        assert coordinator.get_job_status(job_id).status == "cancelled"
        # Cancelling twice (already cancelled) reports failure.
        assert coordinator.cancel_job(job_id) is False
        assert coordinator.cancel_job("missing-job") is False


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_minus(seconds: float) -> datetime:
    return _utc_now() - timedelta(seconds=seconds)
