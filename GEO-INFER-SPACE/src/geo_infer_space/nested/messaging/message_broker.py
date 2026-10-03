"""
Message Broker for H3 Nested Systems.

This module provides a comprehensive message passing system for communication
across boundaries and hierarchies in nested geospatial systems.
"""

import logging
import uuid
from types import TracebackType
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from geo_infer_time import normalize_timestamp

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system."""

    DATA = "data"
    CONTROL = "control"
    STATUS = "status"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    UNICAST = "unicast"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageStatus(Enum):
    """Message delivery status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    PROCESSING = "processing"


@dataclass
class Message:
    """
    Represents a message in the system.
    """

    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType

    # Message content
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Routing information
    source_cell: str | None = None
    target_cell: str | None = None
    routing_path: list[str] = field(default_factory=list)

    # Delivery properties
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: timedelta | None = None
    max_retries: int = 3
    retry_count: int = 0

    # Status tracking
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None
    expires_at: datetime | None = None

    # Response handling
    requires_response: bool = False
    response_timeout: timedelta | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        """Set expiration time if TTL is specified."""
        self.created_at = normalize_timestamp(self.created_at)
        if self.delivered_at is not None:
            self.delivered_at = normalize_timestamp(self.delivered_at)
        if self.expires_at is not None:
            self.expires_at = normalize_timestamp(self.expires_at)
        if self.ttl is not None:
            if not isinstance(self.ttl, timedelta) or self.ttl.total_seconds() < 0:
                raise ValueError("ttl must be a non-negative timedelta")
            self.expires_at = self.created_at + self.ttl

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at:
            return datetime.now(UTC) > self.expires_at
        return False

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "source_cell": self.source_cell,
            "target_cell": self.target_cell,
            "routing_path": self.routing_path,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat()
            if self.delivered_at
            else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
        }


@dataclass
class MessageHandler:
    """
    Represents a message handler.
    """

    handler_id: str
    handler_function: Callable[[Message], Any]
    message_types: set[MessageType] = field(default_factory=set)
    sender_filters: set[str] = field(default_factory=set)
    is_async: bool = False

    def can_handle(self, message: Message) -> bool:
        """Check if this handler can process the message."""
        # Check message type filter
        if self.message_types and message.message_type not in self.message_types:
            return False

        # Check sender filter
        if self.sender_filters and message.sender_id not in self.sender_filters:
            return False

        return True


class H3MessageBroker:
    """
    Comprehensive message broker for H3 nested systems.

    Provides:
    - Message routing across boundaries and hierarchies
    - Multiple delivery patterns (unicast, multicast, broadcast)
    - Priority-based message queuing
    - Reliable delivery with retries
    - Asynchronous and synchronous message handling
    - Message persistence and history
    """

    def __init__(self, broker_id: str | None = None, max_workers: int = 4) -> None:
        """
        Initialize message broker.

        Args:
            broker_id: Unique broker identifier
            max_workers: Maximum number of worker threads
        """
        self.broker_id = broker_id or f"broker_{uuid.uuid4().hex[:8]}"
        self.max_workers = max_workers

        # Message storage
        self.messages: dict[str, Message] = {}
        self.message_queues: dict[str, queue.PriorityQueue] = defaultdict(
            lambda: queue.PriorityQueue()
        )

        # Handlers
        self.handlers: dict[str, MessageHandler] = {}
        self.system_handlers: dict[str, list[str]] = defaultdict(
            list
        )  # system_id -> handler_ids

        # Routing
        self.routing_table: dict[str, str] = {}  # recipient_id -> next_hop
        self.boundary_routes: dict[
            tuple[str, str], list[str]
        ] = {}  # (source, target) -> path

        # Statistics
        self.message_stats: dict[str, int] = defaultdict(int)
        self.delivery_stats: dict[str, list[float]] = defaultdict(
            list
        )  # delivery times

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.worker_threads: list[threading.Thread] = []
        self._lock = threading.RLock()

        # Message history
        self.message_history: deque = deque(maxlen=10000)

        # Created timestamp
        self.created_at = datetime.now(UTC)

    def start(self) -> None:
        """Start the message broker."""
        if self.running:
            return

        self.running = True

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._message_worker, name=f"MessageWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)

        logger.info(
            f"Message broker {self.broker_id} started with {self.max_workers} workers"
        )

    def stop(self) -> None:
        """Stop the message broker."""
        if not self.running:
            return

        self.running = False

        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        logger.info(f"Message broker {self.broker_id} stopped")

    def register_handler(
        self,
        system_id: str,
        handler_function: Callable[[Message], Any],
        message_types: set[MessageType] | None = None,
        sender_filters: set[str] | None = None,
        is_async: bool = False,
    ) -> str:
        """
        Register a message handler for a system.

        Args:
            system_id: System ID that owns this handler
            handler_function: Function to handle messages
            message_types: Types of messages to handle (None for all)
            sender_filters: Sender IDs to accept (None for all)
            is_async: Whether handler is async

        Returns:
            Handler ID
        """
        handler_id = f"handler_{uuid.uuid4().hex[:8]}"

        handler = MessageHandler(
            handler_id=handler_id,
            handler_function=handler_function,
            message_types=message_types or set(),
            sender_filters=sender_filters or set(),
            is_async=is_async,
        )

        with self._lock:
            self.handlers[handler_id] = handler
            self.system_handlers[system_id].append(handler_id)

        logger.debug(f"Registered handler {handler_id} for system {system_id}")
        return handler_id

    def unregister_handler(self, handler_id: str) -> None:
        """Unregister a message handler."""
        with self._lock:
            if handler_id in self.handlers:
                del self.handlers[handler_id]

                # Remove from system handlers
                for handler_list in self.system_handlers.values():
                    if handler_id in handler_list:
                        handler_list.remove(handler_id)
                        break

                logger.debug(f"Unregistered handler {handler_id}")

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        payload: Any,
        message_type: MessageType = MessageType.DATA,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: timedelta | None = None,
        requires_response: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Send a message.

        Args:
            sender_id: Sender system ID
            recipient_id: Recipient system ID
            payload: Message payload
            message_type: Type of message
            priority: Message priority
            ttl: Time to live
            requires_response: Whether response is required
            **kwargs: Additional message properties

        Returns:
            Message ID
        """
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        message = Message(
            message_id=message_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            ttl=ttl,
            requires_response=requires_response,
            **kwargs,
        )

        # Store message
        with self._lock:
            self.messages[message_id] = message
            self.message_stats["sent"] += 1

        # Queue for delivery
        self._queue_message(message)

        logger.debug(f"Queued message {message_id} from {sender_id} to {recipient_id}")
        return message_id

    def broadcast_message(
        self,
        sender_id: str,
        payload: Any,
        message_type: MessageType = MessageType.BROADCAST,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: timedelta | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Broadcast message to all systems.

        Args:
            sender_id: Sender system ID
            payload: Message payload
            message_type: Type of message
            priority: Message priority
            ttl: Time to live
            **kwargs: Additional message properties

        Returns:
            List of message IDs
        """
        message_ids = []

        with self._lock:
            recipients = list(self.system_handlers.keys())

        for recipient_id in recipients:
            if recipient_id != sender_id:  # Don't send to self
                msg_id = self.send_message(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    payload=payload,
                    message_type=message_type,
                    priority=priority,
                    ttl=ttl,
                    **kwargs,
                )
                message_ids.append(msg_id)

        logger.debug(
            f"Broadcast message from {sender_id} to {len(message_ids)} recipients"
        )
        return message_ids

    def multicast_message(
        self,
        sender_id: str,
        recipient_ids: list[str],
        payload: Any,
        message_type: MessageType = MessageType.MULTICAST,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: timedelta | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Multicast message to specific recipients.

        Args:
            sender_id: Sender system ID
            recipient_ids: List of recipient system IDs
            payload: Message payload
            message_type: Type of message
            priority: Message priority
            ttl: Time to live
            **kwargs: Additional message properties

        Returns:
            List of message IDs
        """
        message_ids = []

        for recipient_id in recipient_ids:
            if recipient_id != sender_id:  # Don't send to self
                msg_id = self.send_message(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    payload=payload,
                    message_type=message_type,
                    priority=priority,
                    ttl=ttl,
                    **kwargs,
                )
                message_ids.append(msg_id)

        logger.debug(
            f"Multicast message from {sender_id} to {len(message_ids)} recipients"
        )
        return message_ids

    def send_response(
        self, original_message: Message, payload: Any, status: str = "success"
    ) -> str:
        """
        Send response to a message.

        Args:
            original_message: Original message to respond to
            payload: Response payload
            status: Response status

        Returns:
            Response message ID
        """
        response_payload = {
            "status": status,
            "data": payload,
            "original_message_id": original_message.message_id,
        }

        return self.send_message(
            sender_id=original_message.recipient_id,
            recipient_id=original_message.sender_id,
            payload=response_payload,
            message_type=MessageType.RESPONSE,
            correlation_id=original_message.message_id,
        )

    def get_message_status(self, message_id: str) -> MessageStatus | None:
        """Get status of a message."""
        with self._lock:
            message = self.messages.get(message_id)
            return message.status if message else None

    def get_message_history(
        self,
        system_id: str | None = None,
        message_type: MessageType | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get message history.

        Args:
            system_id: Filter by system ID
            message_type: Filter by message type
            limit: Maximum number of messages

        Returns:
            List of message dictionaries
        """
        history = []
        count = 0

        for record in reversed(self.message_history):
            if count >= limit:
                break

            # Apply filters
            if (
                system_id
                and record.get("sender_id") != system_id
                and record.get("recipient_id") != system_id
            ):
                continue

            if message_type and record.get("message_type") != message_type.value:
                continue

            history.append(record)
            count += 1

        return history

    def get_statistics(self) -> dict[str, Any]:
        """Get broker statistics."""
        with self._lock:
            total_messages = len(self.messages)
            queue_sizes = {
                system_id: q.qsize() for system_id, q in self.message_queues.items()
            }

            # Calculate average delivery times
            avg_delivery_times = {}
            for msg_type, times in self.delivery_stats.items():
                if times:
                    avg_delivery_times[msg_type] = sum(times) / len(times)

        return {
            "broker_id": self.broker_id,
            "total_messages": total_messages,
            "message_stats": dict(self.message_stats),
            "queue_sizes": queue_sizes,
            "average_delivery_times": avg_delivery_times,
            "active_handlers": len(self.handlers),
            "registered_systems": len(self.system_handlers),
            "running": self.running,
            "uptime": (datetime.now(UTC) - self.created_at).total_seconds(),
        }

    def _queue_message(self, message: Message) -> None:
        """Queue message for delivery."""
        # Priority queue uses tuple (priority, timestamp, message)
        # Lower priority values are processed first
        priority_value = -message.priority.value  # Negative for correct ordering
        timestamp = message.created_at.timestamp()

        queue_item = (priority_value, timestamp, message)
        self.message_queues[message.recipient_id].put(queue_item)

    def _message_worker(self) -> None:
        """Worker thread for processing messages."""
        while self.running:
            try:
                # Check all queues for messages
                for msg_queue in list(self.message_queues.values()):
                    try:
                        # Non-blocking get with timeout
                        priority, timestamp, message = msg_queue.get(timeout=0.1)

                        # Process message
                        self._process_message(message)
                        msg_queue.task_done()

                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            except Exception as e:
                logger.error(f"Message worker error: {e}")

    def _process_message(self, message: Message) -> None:
        """Process a single message."""
        try:
            # Check if message has expired
            if message.is_expired():
                message.status = MessageStatus.EXPIRED
                self.message_stats["expired"] += 1
                return

            message.status = MessageStatus.PROCESSING

            # Find handlers for recipient
            handlers = []
            with self._lock:
                handler_ids = self.system_handlers.get(message.recipient_id, [])
                for handler_id in handler_ids:
                    if handler_id in self.handlers:
                        handler = self.handlers[handler_id]
                        if handler.can_handle(message):
                            handlers.append(handler)

            if not handlers:
                logger.warning(
                    f"No handlers found for message {message.message_id} to {message.recipient_id}"
                )
                message.status = MessageStatus.FAILED
                self.message_stats["failed"] += 1
                return

            # Process with handlers
            delivery_start = datetime.now(UTC)

            for handler in handlers:
                try:
                    if handler.is_async:
                        # Submit to thread pool for async processing
                        self.executor.submit(handler.handler_function, message)
                    else:
                        # Process synchronously
                        handler.handler_function(message)

                except Exception as e:
                    logger.error(
                        f"Handler {handler.handler_id} failed to process message {message.message_id}: {e}"
                    )
                    continue

            # Mark as delivered
            message.status = MessageStatus.DELIVERED
            message.delivered_at = datetime.now(UTC)

            # Record delivery time
            delivery_time = (message.delivered_at - delivery_start).total_seconds()
            self.delivery_stats[message.message_type.value].append(delivery_time)

            self.message_stats["delivered"] += 1

            # Add to history
            self.message_history.append(message.to_dict())

            logger.debug(
                f"Delivered message {message.message_id} to {message.recipient_id}"
            )

        except Exception as e:
            logger.error(f"Failed to process message {message.message_id}: {e}")
            message.status = MessageStatus.FAILED
            self.message_stats["failed"] += 1

            # Retry if possible
            if message.can_retry():
                message.retry_count += 1
                message.status = MessageStatus.PENDING
                self._queue_message(message)
                logger.debug(
                    f"Retrying message {message.message_id} (attempt {message.retry_count})"
                )

    def __enter__(self) -> "H3MessageBroker":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.stop()
