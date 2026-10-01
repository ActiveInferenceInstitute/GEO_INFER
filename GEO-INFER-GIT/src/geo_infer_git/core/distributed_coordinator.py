#!/usr/bin/env python3

"""
Distributed coordination for large-scale operations across multiple nodes.

This module provides functionality for coordinating operations across multiple
machines or containers, enabling horizontal scaling and distributed processing
of repository operations.
"""

import json
import time
import socket
import threading
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC
import uuid
import hashlib
import queue

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


@dataclass
class NodeInfo:
    """Information about a distributed node."""

    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: str = "worker"  # master, worker, coordinator
    status: str = "active"  # active, inactive, busy, error
    capabilities: list[str] = field(default_factory=list)
    current_load: float = 0.0
    max_load: float = 1.0
    last_heartbeat: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobInfo:
    """Information about a distributed job."""

    job_id: str
    job_type: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    priority: int = 1
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    assigned_node: str | None = None
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    retry_count: int = 0
    max_retries: int = 3
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationMessage:
    """Message for distributed coordination."""

    message_id: str
    message_type: str
    sender_id: str
    recipient_id: str = "*"
    timestamp: datetime = field(default_factory=_utc_now)
    payload: dict[str, Any] = field(default_factory=dict)
    ttl: int = 300  # Time to live in seconds


class DistributedCoordinator:
    """
    Distributed coordinator for managing operations across multiple nodes.

    Provides functionality for:
    - Node discovery and registration
    - Job distribution and load balancing
    - Health monitoring and failover
    - Message passing between nodes
    - Resource allocation and optimization
    """

    def __init__(
        self,
        node_id: str | None = None,
        role: str = "coordinator",
        discovery_port: int = 5555,
        coordination_port: int = 5556,
    ):
        """
        Initialize distributed coordinator.

        Args:
            node_id: Unique identifier for this node
            role: Role of this node (coordinator, worker, master)
            discovery_port: Port for node discovery
            coordination_port: Port for coordination messages
        """
        self.node_id = node_id or self._generate_node_id()
        self.role = role
        self.discovery_port = discovery_port
        self.coordination_port = coordination_port

        # Node management
        self.nodes: dict[str, NodeInfo] = {}
        self.current_node = NodeInfo(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=coordination_port,
            role=role,
            status="active",
        )
        self.nodes[self.node_id] = self.current_node

        # Job management
        self.jobs: dict[str, JobInfo] = {}
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_jobs: dict[str, str] = {}  # job_id -> node_id

        # Communication
        self.message_queue: queue.Queue = queue.Queue()
        self.message_handlers: dict[str, Callable] = {}

        # Synchronization
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Background threads
        self.discovery_thread: threading.Thread | None = None
        self.coordination_thread: threading.Thread | None = None
        self.heartbeat_thread: threading.Thread | None = None
        self.job_scheduler_thread: threading.Thread | None = None

        # Server sockets held by background services (closed on stop() to
        # unblock threads parked in recvfrom/accept).
        self._discovery_socket: socket.socket | None = None
        self._coordination_socket: socket.socket | None = None

        # Initialize message handlers
        self._setup_message_handlers()

        # Background services are NOT started here; call start() explicitly.

    def start(self) -> None:
        """Start background coordination services.

        Spawns the discovery, coordination, heartbeat-monitor, and job-scheduler
        threads. For ``coordinator``/``master`` roles this binds the UDP
        discovery socket and the TCP coordination listener, so call this
        explicitly rather than relying on construction side effects.
        """
        self._start_services()

    def stop(self) -> None:
        """Stop background coordination services and release resources."""
        logger.info("Shutting down distributed coordinator...")

        # Signal shutdown
        self.shutdown_event.set()

        # Close server sockets so threads parked in accept()/recvfrom() wake
        # up. On Linux, close() alone does not interrupt a concurrent blocked
        # accept()/recvfrom(); shutdown(SHUT_RDWR) does for TCP, and an
        # empty datagram to the bound port does for UDP (shutdown() raises
        # ENOTCONN on an unconnected UDP socket).
        coordination_socket = self._coordination_socket
        if coordination_socket is not None:
            try:
                coordination_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                logger.debug("Coordination socket already shut down", exc_info=True)
            try:
                coordination_socket.close()
            except OSError:
                logger.debug("Service socket already closed", exc_info=True)

        discovery_socket = self._discovery_socket
        if discovery_socket is not None:
            try:
                discovery_socket.sendto(b"", ("127.0.0.1", self.discovery_port))
            except OSError:
                logger.debug("Discovery wakeup datagram failed", exc_info=True)
            try:
                discovery_socket.close()
            except OSError:
                logger.debug("Service socket already closed", exc_info=True)

        # Wait for threads to finish
        for thread in (
            self.discovery_thread,
            self.coordination_thread,
            self.heartbeat_thread,
            self.job_scheduler_thread,
        ):
            if thread and thread.is_alive():
                thread.join(timeout=5)

        logger.info("Distributed coordinator shutdown complete")

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        timestamp = str(int(time.time() * 1000000))
        hostname = socket.gethostname()
        unique_id = hashlib.md5(f"{hostname}_{timestamp}".encode()).hexdigest()[:8]
        return f"node_{unique_id}"

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return cast(str, ip)
        except Exception:
            logger.debug(
                "Local IP detection failed; defaulting to 127.0.0.1", exc_info=True
            )
            return "127.0.0.1"

    def _setup_message_handlers(self) -> None:
        """Set up message handlers for coordination."""
        self.message_handlers = {
            "heartbeat": self._handle_heartbeat,
            "node_register": self._handle_node_register,
            "node_unregister": self._handle_node_unregister,
            "job_request": self._handle_job_request,
            "job_complete": self._handle_job_complete,
            "job_failed": self._handle_job_failed,
            "status_request": self._handle_status_request,
            "coordination_message": self._handle_coordination_message,
        }

    def _start_services(self) -> None:
        """Start background coordination services."""
        if self.role in ["coordinator", "master"]:
            # Start discovery service
            self.discovery_thread = threading.Thread(
                target=self._discovery_service, daemon=True
            )
            self.discovery_thread.start()

            # Start coordination service
            self.coordination_thread = threading.Thread(
                target=self._coordination_service, daemon=True
            )
            self.coordination_thread.start()

            # Start heartbeat monitoring
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_monitor, daemon=True
            )
            self.heartbeat_thread.start()

            # Start job scheduler
            self.job_scheduler_thread = threading.Thread(
                target=self._job_scheduler, daemon=True
            )
            self.job_scheduler_thread.start()

    def _discovery_service(self) -> None:
        """Service for discovering and registering nodes."""
        import socket as sock

        discovery_socket: sock.socket | None = None
        try:
            discovery_socket = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
            discovery_socket.setsockopt(sock.SOL_SOCKET, sock.SO_BROADCAST, 1)
            discovery_socket.bind(("", self.discovery_port))
            self._discovery_socket = discovery_socket

            logger.info(f"Discovery service listening on port {self.discovery_port}")

            while not self.shutdown_event.is_set():
                try:
                    data, addr = discovery_socket.recvfrom(1024)
                    message = json.loads(data.decode("utf-8"))

                    if message.get("type") == "node_discovery":
                        self._handle_node_discovery(message, addr)
                except Exception as e:
                    logger.warning(f"Error in discovery service: {e}")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start discovery service: {e}")
        finally:
            if discovery_socket is not None:
                discovery_socket.close()

    def _handle_node_discovery(
        self, message: dict[str, Any], addr: tuple[str, int]
    ) -> None:
        """Handle node discovery message."""
        node_info = message.get("node_info", {})
        node_id = node_info.get("node_id")

        if node_id:
            # Update node information with discovered address
            if node_id in self.nodes:
                self.nodes[node_id].ip_address = addr[0]
                self.nodes[node_id].port = addr[1]
                self.nodes[node_id].last_heartbeat = datetime.now(UTC)

                logger.info(f"Updated node discovery info for {node_id}")
            else:
                # Create new node from discovery
                new_node = NodeInfo(
                    node_id=node_id,
                    hostname=node_info.get("hostname", "unknown"),
                    ip_address=addr[0],
                    port=addr[1],
                    role=node_info.get("role", "worker"),
                    capabilities=node_info.get("capabilities", []),
                    status="active",
                )

                with self.lock:
                    self.nodes[node_id] = new_node

                logger.info(f"Discovered new node {node_id}")

    def _coordination_service(self) -> None:
        """Service for handling coordination messages."""
        import socket as sock

        coord_socket: sock.socket | None = None
        try:
            coord_socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            coord_socket.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
            coord_socket.bind((self.current_node.ip_address, self.coordination_port))
            coord_socket.listen(10)
            self._coordination_socket = coord_socket
            logger.info(
                f"Coordination service listening on {self.current_node.ip_address}:{self.coordination_port}"
            )

            while not self.shutdown_event.is_set():
                try:
                    client_socket, client_addr = coord_socket.accept()

                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client_connection,
                        args=(client_socket, client_addr),
                        daemon=True,
                    )
                    client_thread.start()

                except Exception as e:
                    logger.warning(f"Error accepting connection: {e}")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start coordination service: {e}")
        finally:
            if coord_socket is not None:
                coord_socket.close()

    def _handle_client_connection(self, client_socket: Any, client_addr: Any) -> None:
        """Handle a client connection for coordination."""
        try:
            data = client_socket.recv(4096)
            if data:
                message = json.loads(data.decode("utf-8"))
                self._process_message(message)

        except Exception as e:
            logger.warning(f"Error handling client connection: {e}")
        finally:
            client_socket.close()

    def _check_heartbeats(self, current_time: datetime) -> None:
        """Run one heartbeat sweep: mark stale nodes inactive, drop long-dead ones.

        Nodes whose last heartbeat is older than 30 seconds are marked
        ``inactive``; nodes still unheard from after 60 seconds are removed.
        """
        with self.lock:
            # Check for dead nodes
            dead_nodes = []
            for node_id, node in self.nodes.items():
                if node_id != self.node_id:  # Don't check self
                    time_since_heartbeat = (
                        current_time - node.last_heartbeat
                    ).total_seconds()

                    if time_since_heartbeat > 30:  # 30 second timeout
                        node.status = "inactive"
                        dead_nodes.append(node_id)
                        logger.warning(f"Node {node_id} marked as inactive")

            # Remove dead nodes after longer timeout
            for node_id in dead_nodes:
                if node_id in self.nodes:
                    time_since_inactive = (
                        current_time - self.nodes[node_id].last_heartbeat
                    ).total_seconds()
                    if time_since_inactive > 60:  # 60 second removal timeout
                        del self.nodes[node_id]
                        logger.info(f"Removed dead node {node_id}")

    def _heartbeat_monitor(self) -> None:
        """Monitor node heartbeats and detect failures."""
        while not self.shutdown_event.is_set():
            try:
                self._check_heartbeats(datetime.now(UTC))
                self.shutdown_event.wait(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                time.sleep(10)

    def _schedule_once(self) -> None:
        """Run one scheduling pass: drain the job queue onto available nodes."""
        available_nodes = self._get_available_nodes()

        if available_nodes:
            # Process job queue
            while not self.job_queue.empty():
                try:
                    priority, job_id = self.job_queue.get_nowait()

                    if job_id in self.jobs:
                        job = self.jobs[job_id]

                        # Find suitable node
                        assigned_node = self._select_node_for_job(job, available_nodes)

                        if assigned_node:
                            self._assign_job_to_node(job, assigned_node)
                            available_nodes = self._get_available_nodes()  # Refresh
                        else:
                            # No suitable node, put job back in queue
                            self.job_queue.put((priority, job_id))
                            break

                except queue.Empty:
                    break

    def _job_scheduler(self) -> None:
        """Schedule and distribute jobs to available nodes."""
        while not self.shutdown_event.is_set():
            try:
                self._schedule_once()
                self.shutdown_event.wait(5)  # Schedule every 5 seconds
            except Exception as e:
                logger.error(f"Error in job scheduler: {e}")
                time.sleep(5)

    def _get_available_nodes(self) -> list[NodeInfo]:
        """Get list of available nodes for job assignment."""
        available_nodes = []

        with self.lock:
            for node in self.nodes.values():
                if (
                    node.status == "active"
                    and node.current_load < node.max_load
                    and node.node_id != self.node_id
                ):  # Don't assign to self
                    available_nodes.append(node)

        return available_nodes

    def _select_node_for_job(
        self, job: JobInfo, available_nodes: list[NodeInfo]
    ) -> NodeInfo | None:
        """Select the best node for a job based on load and capabilities."""
        if not available_nodes:
            return None

        # Filter nodes by capabilities if job requires specific capabilities
        required_capabilities = job.metadata.get("required_capabilities", [])
        if required_capabilities:
            suitable_nodes = [
                node
                for node in available_nodes
                if all(cap in node.capabilities for cap in required_capabilities)
            ]
        else:
            suitable_nodes = available_nodes

        if not suitable_nodes:
            return None

        # Select node with lowest load
        selected_node = min(suitable_nodes, key=lambda n: n.current_load)
        return selected_node

    def _assign_job_to_node(self, job: JobInfo, node: NodeInfo) -> None:
        """Assign a job to a specific node."""
        try:
            # Update job status
            job.status = "running"
            job.started_at = datetime.now(UTC)
            job.assigned_node = node.node_id

            # Update node load
            node.current_load += 0.1  # Simple load increment

            # Send job assignment message
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type="job_assignment",
                sender_id=self.node_id,
                recipient_id=node.node_id,
                payload={
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "metadata": job.metadata,
                },
            )

            self._send_message_to_node(node, message)

            logger.info(f"Assigned job {job.job_id} to node {node.node_id}")

        except Exception as e:
            logger.error(
                f"Error assigning job {job.job_id} to node {node.node_id}: {e}"
            )

    def _process_message(self, message: dict[str, Any]) -> None:
        """Process an incoming coordination message."""
        try:
            message_type = cast(str | None, message.get("type"))
            handler = self.message_handlers.get(message_type) if message_type else None

            if handler:
                handler(message)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_heartbeat(self, message: dict[str, Any]) -> None:
        """Handle heartbeat message from a node."""
        node_id = message.get("node_id")
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = datetime.now(UTC)
            self.nodes[node_id].status = "active"

    def _handle_node_register(self, message: dict[str, Any]) -> None:
        """Handle node registration message."""
        node_info = message.get("node_info", {})
        node_id = node_info.get("node_id")

        if node_id:
            node = NodeInfo(
                node_id=node_id,
                hostname=node_info.get("hostname", "unknown"),
                ip_address=node_info.get("ip_address", "unknown"),
                port=node_info.get("port", 0),
                role=node_info.get("role", "worker"),
                capabilities=node_info.get("capabilities", []),
                status="active",
            )

            with self.lock:
                self.nodes[node_id] = node

            logger.info(f"Registered node {node_id}")

    def _handle_node_unregister(self, message: dict[str, Any]) -> None:
        """Handle node unregistration message."""
        node_id = message.get("node_id")

        if node_id and node_id in self.nodes:
            with self.lock:
                del self.nodes[node_id]

            logger.info(f"Unregistered node {node_id}")

    def _handle_job_request(self, message: dict[str, Any]) -> None:
        """Handle job request from a node."""
        job_info = message.get("job_info", {})
        job_id = job_info.get("job_id")

        if job_id and job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = "running"
            job.started_at = datetime.now(UTC)

            logger.info(f"Job {job_id} started execution")

    def _handle_job_complete(self, message: dict[str, Any]) -> None:
        """Handle job completion message."""
        job_id = message.get("job_id")

        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = "completed"
            job.completed_at = datetime.now(UTC)
            job.progress = 100.0

            # Update node load
            if job.assigned_node and job.assigned_node in self.nodes:
                self.nodes[job.assigned_node].current_load -= 0.1

            logger.info(f"Job {job_id} completed successfully")

    def _handle_job_failed(self, message: dict[str, Any]) -> None:
        """Handle job failure message."""
        job_id = message.get("job_id")
        error = message.get("error", "Unknown error")

        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.retry_count += 1

            if job.retry_count >= job.max_retries:
                job.status = "failed"
                job.completed_at = datetime.now(UTC)

                # Update node load
                if job.assigned_node and job.assigned_node in self.nodes:
                    self.nodes[job.assigned_node].current_load -= 0.1

                logger.error(f"Job {job_id} failed permanently: {error}")
            else:
                # Retry the job
                job.status = "pending"
                job.assigned_node = None
                self.job_queue.put((job.priority, job_id))

                logger.warning(
                    f"Job {job_id} failed, retrying (attempt {job.retry_count + 1})"
                )

    def _handle_status_request(self, message: dict[str, Any]) -> None:
        """Handle status request message."""
        sender_id = message.get("sender_id")

        # Send status response
        status_message = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type="status_response",
            sender_id=self.node_id,
            recipient_id=cast(str, sender_id),
            payload={
                "node_status": self.current_node.status,
                "active_jobs": len(
                    [j for j in self.jobs.values() if j.status == "running"]
                ),
                "queue_size": self.job_queue.qsize(),
                "available_nodes": len(self._get_available_nodes()),
            },
        )

        self._send_message_to_node(cast(str, sender_id), status_message)

    def _handle_coordination_message(self, message: dict[str, Any]) -> None:
        """Handle custom coordination message."""
        # Custom message handling can be implemented here
        logger.debug(f"Received coordination message: {message}")

    def _send_message_to_node(
        self, node: NodeInfo | str, message: CoordinationMessage
    ) -> None:
        """Send a message to a specific node."""
        try:
            import socket as sock

            target_ip = node.ip_address if isinstance(node, NodeInfo) else node
            target_port = (
                node.port if isinstance(node, NodeInfo) else self.coordination_port
            )

            message_data = json.dumps(
                {
                    "message_id": message.message_id,
                    "type": message.message_type,
                    "sender_id": message.sender_id,
                    "recipient_id": message.recipient_id,
                    "timestamp": message.timestamp.isoformat(),
                    "payload": message.payload,
                }
            ).encode("utf-8")

            client_socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            try:
                client_socket.settimeout(5)
                client_socket.connect((target_ip, target_port))
                client_socket.send(message_data)
            finally:
                client_socket.close()

        except Exception as e:
            logger.error(f"Error sending message to node {node}: {e}")

    def register_node(self, node_info: NodeInfo) -> None:
        """
        Register a new node in the distributed system.

        Args:
            node_info: Information about the node to register
        """
        with self.lock:
            self.nodes[node_info.node_id] = node_info

        # Send registration message
        message = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type="node_register",
            sender_id=self.node_id,
            payload={
                "node_info": {
                    "node_id": node_info.node_id,
                    "hostname": node_info.hostname,
                    "ip_address": node_info.ip_address,
                    "port": node_info.port,
                    "role": node_info.role,
                    "capabilities": node_info.capabilities,
                }
            },
        )

        self._broadcast_message(message)

        logger.info(f"Registered node {node_info.node_id}")

    def unregister_node(self, node_id: str) -> None:
        """
        Unregister a node from the distributed system.

        Args:
            node_id: ID of the node to unregister
        """
        if node_id in self.nodes:
            with self.lock:
                del self.nodes[node_id]

            # Send unregistration message
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type="node_unregister",
                sender_id=self.node_id,
                payload={"node_id": node_id},
            )

            self._broadcast_message(message)

            logger.info(f"Unregistered node {node_id}")

    def submit_job(
        self,
        job_type: str,
        metadata: dict[str, Any] | None = None,
        priority: int = 1,
        dependencies: list[str] | None = None,
    ) -> str:
        """
        Submit a job for distributed execution.

        Args:
            job_type: Type of job to execute
            metadata: Additional job metadata
            priority: Job priority (higher numbers = higher priority)
            dependencies: List of job IDs this job depends on

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        job = JobInfo(
            job_id=job_id,
            job_type=job_type,
            priority=priority,
            metadata=metadata or {},
            dependencies=dependencies or [],
        )

        with self.lock:
            self.jobs[job_id] = job

        # Add to job queue
        self.job_queue.put((priority, job_id))

        logger.info(f"Submitted job {job_id} of type {job_type}")

        return job_id

    def get_job_status(self, job_id: str) -> JobInfo | None:
        """
        Get the status of a specific job.

        Args:
            job_id: ID of the job

        Returns:
            JobInfo object or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            True if job was cancelled successfully
        """
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status in ["pending", "running"]:
                    job.status = "cancelled"
                    job.completed_at = datetime.now(UTC)

                    # Update node load if job was assigned
                    if job.assigned_node and job.assigned_node in self.nodes:
                        self.nodes[job.assigned_node].current_load -= 0.1

                    logger.info(f"Cancelled job {job_id}")
                    return True

        return False

    def get_cluster_status(self) -> dict[str, Any]:
        """
        Get overall cluster status.

        Returns:
            Dictionary with cluster status information
        """
        with self.lock:
            active_nodes = sum(
                1 for node in self.nodes.values() if node.status == "active"
            )
            total_jobs = len(self.jobs)
            running_jobs = sum(
                1 for job in self.jobs.values() if job.status == "running"
            )
            pending_jobs = sum(
                1 for job in self.jobs.values() if job.status == "pending"
            )
            completed_jobs = sum(
                1 for job in self.jobs.values() if job.status == "completed"
            )
            failed_jobs = sum(1 for job in self.jobs.values() if job.status == "failed")

        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "coordinator_node": self.node_id,
            "total_jobs": total_jobs,
            "running_jobs": running_jobs,
            "pending_jobs": pending_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "queue_size": self.job_queue.qsize(),
            "cluster_health": self._calculate_cluster_health(),
        }

    def _calculate_cluster_health(self) -> str:
        """Calculate overall cluster health."""
        with self.lock:
            if not self.nodes:
                return "empty"

            active_count = sum(
                1 for node in self.nodes.values() if node.status == "active"
            )

            if active_count == 0:
                return "critical"
            elif active_count < len(self.nodes) * 0.5:
                return "degraded"
            else:
                return "healthy"

    def _broadcast_message(self, message: CoordinationMessage) -> None:
        """Broadcast a message to all nodes."""
        with self.lock:
            for node in self.nodes.values():
                if node.node_id != self.node_id:
                    self._send_message_to_node(node, message)


def create_distributed_coordinator(
    role: str = "coordinator", coordinator_host: str = "localhost"
) -> DistributedCoordinator:
    """
    Create a distributed coordinator with appropriate configuration.

    Args:
        role: Role of this node (coordinator, worker, master)
        coordinator_host: Host of the coordinator node

    Returns:
            Configured DistributedCoordinator instance with background
            services started (call ``stop()`` to tear them down).
    """
    # If this is a worker node, try to connect to coordinator
    if role in ["worker"]:
        try:
            logger.info(
                "Configuring worker node for coordinator host %s", coordinator_host
            )
        except Exception as e:
            logger.warning(f"Failed to connect to coordinator: {e}")
            role = "coordinator"  # Fallback to coordinator

    coordinator = DistributedCoordinator(role=role)
    coordinator.start()
    return coordinator
