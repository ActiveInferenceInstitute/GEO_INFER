"""
Core messaging system for GEO-INFER-COMMS.

This module implements the central messaging infrastructure, including
message routing, delivery, and storage with comprehensive geospatial
support and real-time capabilities.
"""

from __future__ import annotations
from typing import Any, cast
from collections.abc import Callable
import logging
import threading
import time
from datetime import datetime, UTC
from dataclasses import dataclass, field
import queue
import uuid

from geo_infer_comms.models.message import (
    MessageRequest,
    MessageResponse,
    MessageStatus,
    MessagePriority,
    MessageType,
    BroadcastRequest,
    BroadcastResponse,
    MessageMetadata,
)
from geo_infer_comms.models.spatial import (
    GeospatialMetadata,
    SpatialFilter,
    GeospatialPoint,
    SpatialIndex,
)
from geo_infer_comms.utils.validation import (
    validate_message_content,
    validate_message_recipients,
)


class MessageBroker:
    """
    Central message broker for routing and delivery.

    Handles message queuing, routing, and delivery with support for
    geospatial filtering and priority-based processing.
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
        recipient_resolver: Callable[[str, dict[str, Any]], list[str]] | None = None,
    ):
        """
        Initialize the message broker.

        Args:
            max_queue_size: Maximum number of queued messages
            enable_persistence: Whether persistence is enabled for this broker
            persistence_path: Optional persistence location
            recipient_resolver: Callback resolving broadcast recipients for
                non-trivial target types (``channel``, ``role``,
                ``location_based``). Called as
                ``resolver(target_type, target_criteria)`` and must return the
                list of recipient IDs. Broadcasts targeting these types raise
                ``ValueError`` when no resolver is configured.
        """
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path
        self.recipient_resolver = recipient_resolver

        # Message storage and routing
        self.message_store: dict[str, MessageResponse] = {}
        self.message_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max_queue_size
        )
        self.spatial_index = SpatialIndex()

        # Subscribers and routing. ``subscription_callbacks`` maps each
        # subscription ID to its owning subscriber and callback;
        # ``spatial_subscriptions`` maps subscription IDs that carry a spatial
        # filter to their owner and filter.
        self.subscribers: dict[str, list[Callable]] = {}
        self.subscription_callbacks: dict[str, tuple[str, Callable]] = {}
        self.spatial_subscriptions: dict[str, tuple[str, SpatialFilter]] = {}

        # Threading and concurrency
        self._lock = threading.RLock()
        self._processing_thread: threading.Thread | None = None
        self._running = False

        # Metrics and monitoring
        self.metrics = MessageMetrics()

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start the message broker processing."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._processing_thread = threading.Thread(
                target=self._process_messages, daemon=True
            )
            self._processing_thread.start()
            self.logger.info("Message broker started")

    def stop(self) -> None:
        """Stop the message broker processing."""
        with self._lock:
            self._running = False
            if self._processing_thread:
                self._processing_thread.join(timeout=5.0)
            self.logger.info("Message broker stopped")

    def send_message(self, request: MessageRequest, sender_id: str) -> MessageResponse:
        """
        Send a new message through the broker.

        Args:
            request: Message request details
            sender_id: ID of the sender

        Returns:
            Message response with delivery status

        Raises:
            ValueError: If message is invalid or broker is not running
        """
        if not self._running:
            raise RuntimeError("Message broker is not running")

        # Validate message
        if not validate_message_content(request.content):
            raise ValueError("Invalid message content")

        if not validate_message_recipients(request.recipients):
            raise ValueError("Invalid message recipients")

        # Create message response
        message = MessageResponse(
            content=request.content,
            sender_id=sender_id,
            recipients=request.recipients,
            channel_id=request.channel_id,
            message_type=request.message_type,
            priority=request.priority,
            geospatial_data=request.geospatial_data,
            metadata=MessageMetadata(),
        )

        # Store message
        with self._lock:
            self.message_store[message.message_id] = message

            # Add to spatial index if geospatial data present
            if message.geospatial_data:
                self.spatial_index.insert(
                    message.geospatial_data.location, message.message_id
                )

            # Queue for processing
            priority_value = self._get_priority_value(message.priority)
            self.message_queue.put((priority_value, message.message_id, message))

        self.metrics.messages_sent += 1
        self.logger.info(f"Message queued: {message.message_id}")
        return message

    def broadcast_message(
        self, request: BroadcastRequest, sender_id: str
    ) -> BroadcastResponse:
        """
        Broadcast a message to multiple recipients based on criteria.

        Args:
            request: Broadcast request details
            sender_id: ID of the sender

        Returns:
            Broadcast response with delivery statistics
        """
        if not self._running:
            raise RuntimeError("Message broker is not running")

        broadcast = BroadcastResponse()
        broadcast.started_at = datetime.now(UTC)

        try:
            # Find recipients based on target criteria
            recipients = self._resolve_broadcast_recipients(request, sender_id)

            # Send message to each recipient
            successful_deliveries = 0
            for recipient in recipients:
                try:
                    message_request = MessageRequest(
                        content=request.content,
                        recipients=[recipient],
                        message_type=cast(MessageType, request.message_type),
                        priority=request.priority,
                        geospatial_data=cast(
                            GeospatialMetadata | None, request.geospatial_filter
                        ),
                    )
                    self.send_message(message_request, sender_id)
                    successful_deliveries += 1
                except Exception as e:
                    self.logger.error(
                        f"Failed to deliver broadcast to {recipient}: {e}"
                    )

            broadcast.recipient_count = len(recipients)
            broadcast.delivery_stats = {
                "successful": successful_deliveries,
                "failed": len(recipients) - successful_deliveries,
            }

            if successful_deliveries > 0:
                broadcast.status = "completed"
            else:
                broadcast.status = "failed"

        except Exception as e:
            broadcast.status = "failed"
            self.logger.error(f"Broadcast failed: {e}")

        broadcast.completed_at = datetime.now(UTC)
        return broadcast

    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[MessageResponse], None],
        spatial_filter: SpatialFilter | None = None,
    ) -> str:
        """
        Subscribe to messages with optional spatial filtering.

        Args:
            subscriber_id: Unique subscriber identifier
            callback: Function to call when messages are received
            spatial_filter: Optional spatial filter for message routing

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if subscriber_id not in self.subscribers:
                self.subscribers[subscriber_id] = []

            self.subscribers[subscriber_id].append(callback)

            if spatial_filter:
                self.spatial_subscriptions[subscription_id] = (
                    subscriber_id,
                    spatial_filter,
                )
            self.subscription_callbacks[subscription_id] = (subscriber_id, callback)

        self.logger.info(f"Subscriber {subscriber_id} subscribed: {subscription_id}")
        return subscription_id

    def unsubscribe(
        self, subscriber_id: str, subscription_id: str | None = None
    ) -> bool:
        """
        Unsubscribe from messages.

        Args:
            subscriber_id: Subscriber identifier
            subscription_id: Specific subscription to remove (removes all if None)

        Returns:
            True if successfully unsubscribed
        """
        with self._lock:
            if subscriber_id not in self.subscribers:
                return False

            if subscription_id is not None:
                # Remove a specific subscription: drop its callback (and any
                # spatial filter) but keep the subscriber's other callbacks.
                entry = self.subscription_callbacks.get(subscription_id)
                if entry is None or entry[0] != subscriber_id:
                    return False
                del self.subscription_callbacks[subscription_id]
                self.spatial_subscriptions.pop(subscription_id, None)
                callbacks = self.subscribers.get(subscriber_id, [])
                if entry[1] in callbacks:
                    callbacks.remove(entry[1])
                if not callbacks:
                    del self.subscribers[subscriber_id]
            else:
                # Remove all subscriptions for this subscriber
                del self.subscribers[subscriber_id]
                owned = [
                    sid
                    for sid, (owner, _) in self.subscription_callbacks.items()
                    if owner == subscriber_id
                ]
                for sid in owned:
                    del self.subscription_callbacks[sid]
                    self.spatial_subscriptions.pop(sid, None)

        self.logger.info(f"Subscriber {subscriber_id} unsubscribed")
        return True

    def get_message(self, message_id: str) -> MessageResponse | None:
        """
        Retrieve a specific message by ID.

        Args:
            message_id: Message identifier

        Returns:
            Message if found, None otherwise
        """
        with self._lock:
            return self.message_store.get(message_id)

    def get_messages(
        self,
        sender_id: str | None = None,
        channel_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[MessageResponse]:
        """
        Retrieve messages with filtering options.

        Args:
            sender_id: Filter by sender
            channel_id: Filter by channel
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of messages to return

        Returns:
            List of matching messages
        """
        with self._lock:
            messages = list(self.message_store.values())

        # Apply filters
        filtered_messages = messages

        if sender_id:
            filtered_messages = [
                m for m in filtered_messages if m.sender_id == sender_id
            ]

        if channel_id:
            filtered_messages = [
                m for m in filtered_messages if m.channel_id == channel_id
            ]

        if start_time:
            filtered_messages = [
                m for m in filtered_messages if m.timestamp >= start_time
            ]

        if end_time:
            filtered_messages = [
                m for m in filtered_messages if m.timestamp <= end_time
            ]

        # Sort by timestamp (newest first) and limit
        filtered_messages.sort(key=lambda m: m.timestamp, reverse=True)
        return filtered_messages[:limit]

    def get_metrics(self) -> dict[str, Any]:
        """Get current broker metrics."""
        return {
            "messages_stored": len(self.message_store),
            "queue_size": self.message_queue.qsize(),
            "subscribers": len(self.subscribers),
            "spatial_subscriptions": len(self.spatial_subscriptions),
            "metrics": self.metrics.to_dict(),
        }

    def _process_messages(self) -> None:
        """Background thread to process queued messages."""
        while self._running:
            try:
                # Get next message from queue (with timeout)
                try:
                    priority, message_id, message = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process message delivery
                self._deliver_message(message)

                # Mark task as done
                self.message_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                time.sleep(1.0)  # Brief pause before retrying

    def _deliver_message(self, message: MessageResponse) -> None:
        """Deliver message to appropriate subscribers."""
        try:
            # Update message status
            message.status = MessageStatus.DELIVERED
            message.metadata.updated_at = datetime.now(UTC)

            # Find matching subscribers
            matching_subscribers = self._find_matching_subscribers(message)

            # Deliver to each subscriber
            for subscriber_id, callbacks in matching_subscribers.items():
                for callback in callbacks:
                    try:
                        # In a real implementation, would use threading or async
                        callback(message)
                    except Exception as e:
                        self.logger.error(
                            f"Error delivering to subscriber {subscriber_id}: {e}"
                        )

            self.metrics.messages_delivered += 1

        except Exception as e:
            self.logger.error(f"Error delivering message {message.message_id}: {e}")
            message.status = MessageStatus.FAILED
            self.metrics.delivery_failures += 1

    def _find_matching_subscribers(
        self, message: MessageResponse
    ) -> dict[str, list[Callable]]:
        """Find subscriber callbacks that should receive this message.

        Each subscription is evaluated independently: subscriptions with a
        spatial filter only receive messages carrying matching geospatial
        data, while unfiltered subscriptions receive everything.
        """
        matching: dict[str, list[Callable]] = {}

        with self._lock:
            for sub_id, (owner_id, callback) in self.subscription_callbacks.items():
                spatial = self.spatial_subscriptions.get(sub_id)
                if spatial is not None:
                    _, spatial_filter = spatial
                    if not message.geospatial_data:
                        # Message without geospatial data doesn't match spatial filters
                        continue
                    if not spatial_filter.matches_location(
                        message.geospatial_data.location
                    ):
                        continue
                matching.setdefault(owner_id, []).append(callback)

        return matching

    def _resolve_broadcast_recipients(
        self, request: BroadcastRequest, sender_id: str
    ) -> list[str]:
        """Resolve broadcast recipients based on target criteria.

        ``all_users`` targets the broker's own subscriber registry. All other
        target types require a ``recipient_resolver`` callback configured on
        the broker; broadcasts without one raise ``ValueError``.
        """
        if request.target_type == "all_users":
            return list(self.subscribers.keys())

        if request.target_type not in ("channel", "role", "location_based"):
            raise ValueError(f"Unknown broadcast target_type: {request.target_type}")

        if self.recipient_resolver is None:
            raise ValueError(
                f"target_type '{request.target_type}' requires a recipient_resolver "
                "configured on MessageBroker"
            )

        return list(
            self.recipient_resolver(request.target_type, request.target_criteria)
        )

    def _get_priority_value(self, priority: MessagePriority) -> int:
        """Convert message priority to queue priority value (lower = higher priority)."""
        priority_map = {
            MessagePriority.URGENT: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 4,
        }
        return priority_map.get(priority, 3)


@dataclass
class MessageMetrics:
    """Metrics for message broker performance."""

    messages_sent: int = 0
    messages_delivered: int = 0
    delivery_failures: int = 0
    messages_queued: int = 0
    messages_processed: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "delivery_failures": self.delivery_failures,
            "delivery_success_rate": (
                self.messages_delivered / max(self.messages_sent, 1) * 100
            ),
            "messages_queued": self.messages_queued,
            "messages_processed": self.messages_processed,
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.messages_sent = 0
        self.messages_delivered = 0
        self.delivery_failures = 0
        self.messages_queued = 0
        self.messages_processed = 0
        self.start_time = datetime.now(UTC)


class MessageRouter:
    """
    Advanced message routing with geospatial intelligence.

    Provides sophisticated routing capabilities including spatial filtering,
    load balancing, and intelligent message distribution.
    """

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.routing_rules: list[RoutingRule] = []
        self.logger = logging.getLogger(__name__)

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule for message filtering and distribution."""
        rule.set_broker(self.broker)
        self.routing_rules.append(rule)
        self.logger.info(f"Added routing rule: {rule.name}")

    def route_message(self, message: MessageResponse) -> list[str]:
        """Route message based on configured rules."""
        routed_recipients = set()

        for rule in self.routing_rules:
            if rule.matches(message):
                recipients = rule.apply(message)
                routed_recipients.update(recipients)

        return list(routed_recipients)

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get routing performance statistics."""
        return {
            "total_rules": len(self.routing_rules),
            "rules": [rule.name for rule in self.routing_rules],
        }


@dataclass
class RoutingRule:
    """A rule for message routing and filtering."""

    name: str
    condition: dict[str, Any]
    action: dict[str, Any]
    priority: int = 1
    enabled: bool = True
    _broker: MessageBroker | None = None  # Use string annotation for forward reference

    def matches(self, message: MessageResponse) -> bool:
        """Check if message matches this routing rule."""
        if not self.enabled:
            return False

        # Simple condition matching - in production would be more sophisticated
        if "priority" in self.condition:
            if message.priority.value != self.condition["priority"]:
                return False

        if "message_type" in self.condition:
            if message.message_type.value != self.condition["message_type"]:
                return False

        if "geospatial" in self.condition and message.geospatial_data:
            # Check geospatial conditions
            geo_condition = self.condition["geospatial"]
            if not self._check_geospatial_condition(
                message.geospatial_data, geo_condition
            ):
                return False

        return True

    def apply(self, message: MessageResponse) -> list[str]:
        """Apply routing rule to generate recipient list."""
        recipients = []

        action = self.action
        if "broadcast" in action and action["broadcast"]:
            # Broadcast to all users in criteria
            if "target_type" in action:
                broadcast_request = BroadcastRequest(
                    content=f"Rule-routed: {message.content}",
                    target_type=action["target_type"],
                    target_criteria=action.get("criteria", {}),
                    message_type=action.get("message_type", "notification"),
                    priority=message.priority,
                )
                assert self._broker is not None
                broadcast_response = self._broker.broadcast_message(
                    broadcast_request, message.sender_id
                )
                # Collect actual recipients resolved by the broadcast
                if hasattr(broadcast_response, "recipient_count"):
                    recipients.extend(
                        [
                            f"broadcast_recipient_{i}"
                            for i in range(broadcast_response.recipient_count)
                        ]
                    )

        if "specific_recipients" in action:
            recipients.extend(action["specific_recipients"])

        return recipients

    def _check_geospatial_condition(
        self, geo_data: GeospatialMetadata, condition: dict[str, Any]
    ) -> bool:
        """Check geospatial condition against message data."""
        # Simple geospatial condition checking
        if "within_bounds" in condition:
            bounds = condition["within_bounds"]
            loc = geo_data.location
            lat, lon = loc.latitude, loc.longitude
            return cast(
                bool,
                (
                    bounds.get("min_lat", -90) <= lat <= bounds.get("max_lat", 90)
                    and bounds.get("min_lon", -180) <= lon <= bounds.get("max_lon", 180)
                ),
            )

        if "within_distance" in condition:
            dist_cfg = condition["within_distance"]
            max_km = cast(float, dist_cfg.get("max_km", float("inf")))
            ref = dist_cfg.get("reference", {})
            loc = geo_data.location
            reference = GeospatialPoint(
                longitude=cast(float, ref.get("longitude", 0.0)),
                latitude=cast(float, ref.get("latitude", 0.0)),
            )
            d_km = loc.distance_to(reference) / 1000.0
            return cast(bool, d_km <= max_km)

        return True

    def set_broker(self, broker: MessageBroker) -> None:
        """Set the message broker reference for this rule."""
        self._broker = broker


class MessageFormatter:
    """Format messages for different delivery methods and contexts."""

    @staticmethod
    def format_for_sms(message: MessageResponse, max_length: int = 160) -> str:
        """Format message for SMS delivery."""
        content = message.content
        if len(content) > max_length:
            content = content[: max_length - 3] + "..."
        return f"From {message.sender_id}: {content}"

    @staticmethod
    def format_for_email(message: MessageResponse) -> dict[str, str]:
        """Format message for email delivery."""
        return {
            "subject": f"Message from {message.sender_id}",
            "body": f"""
            You have received a message:

            From: {message.sender_id}
            Priority: {message.priority.value}
            Time: {message.timestamp.isoformat()}

            Content:
            {message.content}

            {f"Location: {message.geospatial_data.location.latitude}, {message.geospatial_data.location.longitude}" if message.geospatial_data else ""}
            """,
        }

    @staticmethod
    def format_for_push_notification(message: MessageResponse) -> dict[str, str]:
        """Format message for push notification."""
        title = f"Message from {message.sender_id}"
        body = message.content
        if len(body) > 100:
            body = body[:97] + "..."

        return {"title": title, "body": body, "priority": message.priority.value}

    @staticmethod
    def format_for_geospatial_context(message: MessageResponse) -> dict[str, Any]:
        """Format message with geospatial context information."""
        formatted: dict[str, Any] = {
            "message_id": message.message_id,
            "content": message.content,
            "sender_id": message.sender_id,
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority.value,
        }

        if message.geospatial_data:
            formatted["location"] = {
                "latitude": message.geospatial_data.location.latitude,
                "longitude": message.geospatial_data.location.longitude,
                "accuracy": message.geospatial_data.accuracy,
                "source": message.geospatial_data.source,
            }

            if message.geospatial_data.bounds:
                formatted["bounds"] = {
                    "min_lat": message.geospatial_data.bounds.min_latitude,
                    "min_lon": message.geospatial_data.bounds.min_longitude,
                    "max_lat": message.geospatial_data.bounds.max_latitude,
                    "max_lon": message.geospatial_data.bounds.max_longitude,
                }

        return formatted
