"""
Message Protocols for H3 Nested Systems.

This module defines communication protocols for different types of
message exchanges in nested geospatial systems.
"""

import logging
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Types of communication protocols."""

    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    FIRE_AND_FORGET = "fire_and_forget"
    STREAMING = "streaming"
    BATCH = "batch"
    GOSSIP = "gossip"
    CONSENSUS = "consensus"
    HEARTBEAT = "heartbeat"


class MessageFormat(Enum):
    """Message format types."""

    JSON = "json"
    BINARY = "binary"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    CUSTOM = "custom"


@dataclass
class ProtocolConfig:
    """
    Configuration for a message protocol.
    """

    protocol_type: ProtocolType
    message_format: MessageFormat = MessageFormat.JSON

    # Timing parameters
    timeout: timedelta | None = None
    retry_interval: timedelta | None = None
    max_retries: int = 3

    # Reliability parameters
    require_acknowledgment: bool = False
    guarantee_order: bool = False
    guarantee_delivery: bool = False

    # Batch parameters (for batch protocols)
    batch_size: int | None = None
    batch_timeout: timedelta | None = None

    # Streaming parameters
    buffer_size: int | None = None
    flow_control: bool = False

    # Security parameters
    encryption_required: bool = False
    authentication_required: bool = False

    # Custom parameters
    custom_params: dict[str, Any] = field(default_factory=dict)


class MessageProtocol(ABC):
    """
    Abstract base class for message protocols.
    """

    def __init__(self, protocol_id: str, config: ProtocolConfig) -> None:
        """
        Initialize protocol.

        Args:
            protocol_id: Unique protocol identifier
            config: Protocol configuration
        """
        self.protocol_id = protocol_id
        self.config = config
        self.created_at = datetime.now(UTC)
        self.message_broker: Any = None  # Set externally

        # Protocol state
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.statistics: dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "sessions_created": 0,
            "sessions_closed": 0,
        }

    @abstractmethod
    def send_message(
        self, sender_id: str, recipient_id: str, payload: Any, **kwargs: Any
    ) -> str:
        """
        Send a message using this protocol.

        Args:
            sender_id: Sender system ID
            recipient_id: Recipient system ID
            payload: Message payload
            **kwargs: Protocol-specific parameters

        Returns:
            Message or session ID
        """
        raise RuntimeError(
            "MessageProtocol.send_message requires a concrete protocol implementation"
        )

    @abstractmethod
    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        """
        Handle an incoming message.

        Args:
            message: Incoming message
            **kwargs: Protocol-specific parameters

        Returns:
            Protocol-specific result
        """
        raise RuntimeError(
            "MessageProtocol.handle_message requires a concrete protocol implementation"
        )

    def create_session(
        self, session_id: str, participants: list[str], **kwargs: Any
    ) -> bool:
        """
        Create a protocol session.

        Args:
            session_id: Session identifier
            participants: List of participant system IDs
            **kwargs: Session-specific parameters

        Returns:
            True if session created successfully
        """
        if session_id in self.active_sessions:
            return False

        self.active_sessions[session_id] = {
            "participants": participants,
            "created_at": datetime.now(UTC),
            "last_activity": datetime.now(UTC),
            "message_count": 0,
            "custom_data": kwargs,
        }

        self.statistics["sessions_created"] += 1
        return True

    def close_session(self, session_id: str) -> bool:
        """
        Close a protocol session.

        Args:
            session_id: Session identifier

        Returns:
            True if session closed successfully
        """
        if session_id not in self.active_sessions:
            return False

        del self.active_sessions[session_id]
        self.statistics["sessions_closed"] += 1
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get protocol statistics."""
        return {
            "protocol_id": self.protocol_id,
            "protocol_type": self.config.protocol_type.value,
            "active_sessions": len(self.active_sessions),
            "statistics": self.statistics.copy(),
            "created_at": self.created_at.isoformat(),
        }


class RequestResponseProtocol(MessageProtocol):
    """
    Request-Response protocol implementation.
    """

    def __init__(self, protocol_id: str, config: ProtocolConfig | None = None) -> None:
        if config is None:
            config = ProtocolConfig(
                protocol_type=ProtocolType.REQUEST_RESPONSE,
                timeout=timedelta(seconds=30),
                require_acknowledgment=True,
            )

        super().__init__(protocol_id, config)
        self.pending_requests: dict[str, dict[str, Any]] = {}

    def send_message(
        self, sender_id: str, recipient_id: str, payload: Any, **kwargs: Any
    ) -> str:
        """Send a request message."""
        if not self.message_broker:
            raise RuntimeError("Message broker not set")

        # Create request
        message_id = self.message_broker.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            requires_response=True,
            response_timeout=self.config.timeout,
            **kwargs,
        )

        # Track pending request
        self.pending_requests[message_id] = {
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "sent_at": datetime.now(UTC),
            "timeout_at": datetime.now(UTC)
            + (self.config.timeout or timedelta(seconds=30)),
        }

        self.statistics["messages_sent"] += 1
        return cast(str, message_id)

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        """Handle incoming request or response."""
        if message.message_type.value == "response":
            # Handle response
            if message.correlation_id in self.pending_requests:
                del self.pending_requests[message.correlation_id]

            self.statistics["messages_received"] += 1
            return message.payload

        else:
            # Handle request - send response
            if message.requires_response and self.message_broker:
                response_payload = kwargs.get(
                    "response_payload", {"status": "received"}
                )

                self.message_broker.send_response(message, response_payload)

            self.statistics["messages_received"] += 1
            return message.payload

    def cleanup_expired_requests(self) -> None:
        """Clean up expired requests."""
        now = datetime.now(UTC)
        expired = [
            msg_id
            for msg_id, req in self.pending_requests.items()
            if now > req["timeout_at"]
        ]

        for msg_id in expired:
            del self.pending_requests[msg_id]
            self.statistics["messages_failed"] += 1


class PublishSubscribeProtocol(MessageProtocol):
    """
    Publish-Subscribe protocol implementation.
    """

    def __init__(self, protocol_id: str, config: ProtocolConfig | None = None) -> None:
        if config is None:
            config = ProtocolConfig(
                protocol_type=ProtocolType.PUBLISH_SUBSCRIBE, guarantee_delivery=False
            )

        super().__init__(protocol_id, config)
        self.topics: dict[str, set[str]] = {}  # topic -> subscribers
        self.subscriptions: dict[str, set[str]] = {}  # subscriber -> topics

    def subscribe(self, subscriber_id: str, topic: str) -> bool:
        """
        Subscribe to a topic.

        Args:
            subscriber_id: Subscriber system ID
            topic: Topic name

        Returns:
            True if subscription successful
        """
        if topic not in self.topics:
            self.topics[topic] = set()

        if subscriber_id not in self.subscriptions:
            self.subscriptions[subscriber_id] = set()

        self.topics[topic].add(subscriber_id)
        self.subscriptions[subscriber_id].add(topic)

        return True

    def unsubscribe(self, subscriber_id: str, topic: str) -> bool:
        """
        Unsubscribe from a topic.

        Args:
            subscriber_id: Subscriber system ID
            topic: Topic name

        Returns:
            True if unsubscription successful
        """
        if topic in self.topics:
            self.topics[topic].discard(subscriber_id)

        if subscriber_id in self.subscriptions:
            self.subscriptions[subscriber_id].discard(topic)

        return True

    def send_message(
        self, sender_id: str, recipient_id: str, payload: Any, **kwargs: Any
    ) -> str:
        """Publish a message to a topic."""
        topic = kwargs.get("topic", "default")

        if not self.message_broker:
            raise RuntimeError("Message broker not set")

        # Get subscribers for topic
        subscribers = self.topics.get(topic, set())

        if not subscribers:
            logger.warning(f"No subscribers for topic: {topic}")
            return ""

        # Send to all subscribers
        message_ids = []
        for subscriber_id in subscribers:
            if subscriber_id != sender_id:  # Don't send to self
                msg_id = self.message_broker.send_message(
                    sender_id=sender_id,
                    recipient_id=subscriber_id,
                    payload=payload,
                    metadata={"topic": topic},
                    **kwargs,
                )
                message_ids.append(msg_id)

        self.statistics["messages_sent"] += len(message_ids)
        return f"published_to_{len(message_ids)}_subscribers"

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        """Handle incoming published message."""
        self.statistics["messages_received"] += 1
        return message.payload


class FireAndForgetProtocol(MessageProtocol):
    """
    Fire-and-Forget protocol implementation.
    """

    def __init__(self, protocol_id: str, config: ProtocolConfig | None = None) -> None:
        if config is None:
            config = ProtocolConfig(
                protocol_type=ProtocolType.FIRE_AND_FORGET,
                require_acknowledgment=False,
                max_retries=0,
            )

        super().__init__(protocol_id, config)

    def send_message(
        self, sender_id: str, recipient_id: str, payload: Any, **kwargs: Any
    ) -> str:
        """Send a fire-and-forget message."""
        if not self.message_broker:
            raise RuntimeError("Message broker not set")

        message_id = self.message_broker.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            requires_response=False,
            **kwargs,
        )

        self.statistics["messages_sent"] += 1
        return cast(str, message_id)

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        """Handle incoming fire-and-forget message."""
        self.statistics["messages_received"] += 1
        return message.payload


class StreamingProtocol(MessageProtocol):
    """
    Streaming protocol implementation.
    """

    def __init__(self, protocol_id: str, config: ProtocolConfig | None = None) -> None:
        if config is None:
            config = ProtocolConfig(
                protocol_type=ProtocolType.STREAMING,
                buffer_size=1000,
                flow_control=True,
            )

        super().__init__(protocol_id, config)
        self.streams: dict[str, dict[str, Any]] = {}

    def create_stream(
        self, stream_id: str, sender_id: str, recipient_id: str, **kwargs: Any
    ) -> bool:
        """
        Create a streaming session.

        Args:
            stream_id: Stream identifier
            sender_id: Sender system ID
            recipient_id: Recipient system ID
            **kwargs: Stream parameters

        Returns:
            True if stream created successfully
        """
        if stream_id in self.streams:
            return False

        self.streams[stream_id] = {
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "created_at": datetime.now(UTC),
            "message_count": 0,
            "buffer": [],
            "buffer_size": self.config.buffer_size or 1000,
            "flow_control": self.config.flow_control,
            "active": True,
        }

        return True

    def send_message(
        self, sender_id: str, recipient_id: str, payload: Any, **kwargs: Any
    ) -> str:
        """Send a message in a stream."""
        stream_id = kwargs.get("stream_id")

        if not stream_id or stream_id not in self.streams:
            raise ValueError("Valid stream_id required for streaming protocol")

        stream = self.streams[stream_id]

        if not stream["active"]:
            raise RuntimeError(f"Stream {stream_id} is not active")

        # Add to buffer
        stream["buffer"].append(
            {
                "payload": payload,
                "timestamp": datetime.now(UTC),
                "sequence": stream["message_count"],
            }
        )

        stream["message_count"] += 1

        # Flush buffer if full or flow control disabled
        if len(stream["buffer"]) >= stream["buffer_size"] or not stream["flow_control"]:
            self._flush_stream_buffer(stream_id)

        self.statistics["messages_sent"] += 1
        return f"stream_{stream_id}_seq_{stream['message_count']}"

    def _flush_stream_buffer(self, stream_id: str) -> None:
        """Flush stream buffer."""
        if stream_id not in self.streams or not self.message_broker:
            return

        stream = self.streams[stream_id]

        if not stream["buffer"]:
            return

        # Send buffered messages
        batch_payload = {"stream_id": stream_id, "messages": stream["buffer"].copy()}

        self.message_broker.send_message(
            sender_id=stream["sender_id"],
            recipient_id=stream["recipient_id"],
            payload=batch_payload,
            message_type=self.message_broker.MessageType.DATA,
        )

        # Clear buffer
        stream["buffer"].clear()

    def close_stream(self, stream_id: str) -> bool:
        """Close a stream."""
        if stream_id not in self.streams:
            return False

        # Flush remaining messages
        self._flush_stream_buffer(stream_id)

        # Mark as inactive
        self.streams[stream_id]["active"] = False

        return True

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        """Handle incoming stream message."""
        self.statistics["messages_received"] += 1

        # Extract stream data
        if isinstance(message.payload, dict) and "stream_id" in message.payload:
            return message.payload["messages"]

        return message.payload


class BatchProtocol(MessageProtocol):
    """
    Batch protocol implementation.
    """

    def __init__(self, protocol_id: str, config: ProtocolConfig | None = None) -> None:
        if config is None:
            config = ProtocolConfig(
                protocol_type=ProtocolType.BATCH,
                batch_size=10,
                batch_timeout=timedelta(seconds=5),
            )

        super().__init__(protocol_id, config)
        self.batches: dict[
            tuple[str, str], dict[str, Any]
        ] = {}  # (sender, recipient) -> batch

    def send_message(
        self, sender_id: str, recipient_id: str, payload: Any, **kwargs: Any
    ) -> str:
        """Add message to batch."""
        batch_key = (sender_id, recipient_id)

        if batch_key not in self.batches:
            self.batches[batch_key] = {
                "messages": [],
                "created_at": datetime.now(UTC),
                "last_added": datetime.now(UTC),
            }

        batch = self.batches[batch_key]
        batch["messages"].append(
            {
                "payload": payload,
                "timestamp": datetime.now(UTC),
                "metadata": kwargs.get("metadata", {}),
            }
        )
        batch["last_added"] = datetime.now(UTC)

        # Check if batch should be sent
        should_send = len(batch["messages"]) >= (self.config.batch_size or 10) or (
            self.config.batch_timeout
            and datetime.now(UTC) - batch["created_at"] >= self.config.batch_timeout
        )

        if should_send:
            return self._send_batch(sender_id, recipient_id)

        return f"batched_{len(batch['messages'])}"

    def _send_batch(self, sender_id: str, recipient_id: str) -> str:
        """Send accumulated batch."""
        batch_key = (sender_id, recipient_id)

        if batch_key not in self.batches or not self.message_broker:
            return ""

        batch = self.batches[batch_key]

        if not batch["messages"]:
            return ""

        # Send batch
        batch_payload = {
            "batch_id": f"batch_{datetime.now(UTC).timestamp()}",
            "message_count": len(batch["messages"]),
            "messages": batch["messages"].copy(),
        }

        message_id = self.message_broker.send_message(
            sender_id=sender_id, recipient_id=recipient_id, payload=batch_payload
        )

        # Clear batch
        del self.batches[batch_key]

        self.statistics["messages_sent"] += len(batch["messages"])
        return cast(str, message_id)

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        """Handle incoming batch message."""
        self.statistics["messages_received"] += 1

        # Extract batch data
        if isinstance(message.payload, dict) and "messages" in message.payload:
            return message.payload["messages"]

        return message.payload

    def flush_all_batches(self) -> None:
        """Flush all pending batches."""
        for sender_id, recipient_id in list(self.batches.keys()):
            self._send_batch(sender_id, recipient_id)
