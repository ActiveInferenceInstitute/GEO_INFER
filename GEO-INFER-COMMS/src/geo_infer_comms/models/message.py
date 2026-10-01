"""
Geospatial message data models for the GEO-INFER-COMMS module.

This module defines comprehensive data models for messages, channels, notifications,
and collaboration sessions with full geospatial support and metadata tracking.
"""

from __future__ import annotations
from typing import Any, Literal, cast
from datetime import datetime, UTC
from enum import StrEnum
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
import uuid

from geo_infer_comms.models.spatial import GeospatialMetadata


class MessagePriority(StrEnum):
    """Message priority levels for routing and processing."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageType(StrEnum):
    """Types of messages supported in the system."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    LOCATION = "location"
    ALERT = "alert"
    SENSOR_DATA = "sensor_data"
    COMMAND = "command"
    STATUS = "status"


class MessageStatus(StrEnum):
    """Status of message delivery and processing."""

    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    QUEUED = "queued"
    PROCESSING = "processing"


class ChannelType(StrEnum):
    """Types of communication channels."""

    PUBLIC = "public"
    PRIVATE = "private"
    DIRECT = "direct"
    GROUP = "group"


class ChannelStatus(StrEnum):
    """Status of communication channels."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUSPENDED = "suspended"


class NotificationType(StrEnum):
    """Types of notifications."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    REMINDER = "reminder"


class NotificationStatus(StrEnum):
    """Status of notifications."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    EXPIRED = "expired"


class EventType(StrEnum):
    """Types of events in the system."""

    DATA_UPDATE = "data_update"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"
    SENSOR_TRIGGER = "sensor_trigger"
    GEOSPATIAL_CHANGE = "geospatial_change"


class CollaborationType(StrEnum):
    """Types of collaboration sessions."""

    MEETING = "meeting"
    WORKSHOP = "workshop"
    PLANNING = "planning"
    REVIEW = "review"


class ParticipantRole(StrEnum):
    """Roles of participants in collaboration."""

    HOST = "host"
    MODERATOR = "moderator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class ParticipantStatus(StrEnum):
    """Status of participants in collaboration."""

    ONLINE = "online"
    OFFLINE = "offline"
    AWAY = "away"
    BUSY = "busy"


@dataclass
class MessageMetadata:
    """Metadata associated with messages."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: int = 1
    tags: list[str] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "tags": self.tags,
            "custom_fields": self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageMetadata:
        """Create metadata from dictionary."""
        return cls(
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            custom_fields=data.get("custom_fields", {}),
        )


class MessageRequest(BaseModel):
    """Request model for creating a new message."""

    content: str = Field(..., min_length=1, max_length=10000)
    recipients: list[str] = Field(..., min_length=1)
    channel_id: str | None = None
    message_type: MessageType = MessageType.TEXT
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: dict[str, Any] | None = None
    geospatial_data: GeospatialMetadata | None = None
    expires_at: datetime | None = None

    model_config = ConfigDict(use_enum_values=True)


class MessageResponse(BaseModel):
    """Response model for message data."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    sender_id: str
    recipients: list[str]
    channel_id: str | None = None
    message_type: MessageType
    priority: MessagePriority
    status: MessageStatus = MessageStatus.SENT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    geospatial_data: GeospatialMetadata | None = None
    delivery_stats: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        data = self.model_dump()
        data["metadata"] = self.metadata.to_dict()
        if self.geospatial_data:
            data["geospatial_data"] = self.geospatial_data.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageResponse:
        """Create message from dictionary."""
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = MessageMetadata.from_dict(data["metadata"])
        return cls(**data)


class BroadcastRequest(BaseModel):
    """Request model for broadcasting messages."""

    content: str = Field(..., min_length=1, max_length=10000)
    target_type: Literal["all_users", "channel", "role", "location_based"] = Field(...)
    target_criteria: dict[str, Any] = Field(...)
    message_type: Literal["announcement", "alert", "emergency", "notification"] = (
        "announcement"
    )
    priority: MessagePriority = MessagePriority.NORMAL
    geospatial_filter: dict[str, Any] | None = None
    expires_at: datetime | None = None


class BroadcastResponse(BaseModel):
    """Response model for broadcast operations."""

    broadcast_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["sent", "in_progress", "completed", "failed"] = "sent"
    recipient_count: int = 0
    delivery_stats: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class ChannelRequest(BaseModel):
    """Request model for creating channels."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    type: ChannelType = ChannelType.PUBLIC
    permissions: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    geospatial_bounds: dict[str, Any] | None = None


class ChannelResponse(BaseModel):
    """Response model for channel data."""

    channel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    type: ChannelType
    status: ChannelStatus = ChannelStatus.ACTIVE
    member_count: int = 0
    permissions: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    geospatial_bounds: dict[str, Any] | None = None


class SubscriptionRequest(BaseModel):
    """Request model for channel subscriptions."""

    subscription_type: Literal["all_messages", "mentions_only", "important_only"] = (
        "all_messages"
    )
    notification_preferences: dict[str, Any] = Field(default_factory=dict)


class SubscriptionResponse(BaseModel):
    """Response model for subscription data."""

    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: str
    user_id: str
    subscription_type: str
    status: Literal["active", "paused", "cancelled"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NotificationRequest(BaseModel):
    """Request model for creating notifications."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=2000)
    recipients: list[str] = Field(..., min_length=1)
    notification_type: NotificationType = NotificationType.INFO
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_method: list[Literal["in_app", "email", "sms", "push", "websocket"]] = (
        Field(
            default_factory=lambda: cast(
                "list[Literal['in_app', 'email', 'sms', 'push', 'websocket']]",
                ["in_app"],
            )
        )
    )
    schedule_time: datetime | None = None
    expiry_time: datetime | None = None
    geospatial_context: dict[str, Any] | None = None


class NotificationResponse(BaseModel):
    """Response model for notification data."""

    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    notification_type: NotificationType
    priority: MessagePriority
    status: NotificationStatus = NotificationStatus.PENDING
    delivery_methods: list[str] = Field(default_factory=lambda: ["in_app"])
    recipients: list[str] = Field(default_factory=list)
    delivery_stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    read_at: datetime | None = None
    schedule_time: datetime | None = None
    geospatial_context: dict[str, Any] | None = None


class EventPublishRequest(BaseModel):
    """Request model for publishing events."""

    event_type: str = Field(...)
    payload: dict[str, Any] = Field(...)
    source: str | None = None
    target_channels: list[str] = Field(default_factory=list)
    priority: MessagePriority = MessagePriority.NORMAL
    persistence: bool = False
    geospatial_context: dict[str, Any] | None = None


class EventPublishResponse(BaseModel):
    """Response model for event publishing."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(...)
    payload: dict[str, Any] = Field(...)
    source: str | None = None
    target_channels: list[str] = Field(default_factory=list)
    priority: MessagePriority = MessagePriority.NORMAL
    status: Literal["published", "queued", "failed"] = "published"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delivery_count: int = 0
    geospatial_context: dict[str, Any] | None = None


class EventSubscriptionRequest(BaseModel):
    """Request model for event subscriptions."""

    event_types: list[str] = Field(..., min_length=1)
    filter_criteria: dict[str, Any] = Field(default_factory=dict)
    delivery_mode: Literal["real_time", "batched", "on_demand"] = "real_time"
    callback_url: str | None = None


class EventSubscriptionResponse(BaseModel):
    """Response model for event subscriptions."""

    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_types: list[str]
    filter_criteria: dict[str, Any] = Field(default_factory=dict)
    delivery_mode: Literal["real_time", "batched", "on_demand"] = "real_time"
    callback_url: str | None = None
    status: Literal["active", "paused", "error"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CollaborationSessionRequest(BaseModel):
    """Request model for creating collaboration sessions."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=1000)
    session_type: CollaborationType = CollaborationType.MEETING
    participants: list[str] = Field(..., min_length=1)
    duration: int | None = Field(None, ge=1, le=480)  # minutes
    features: list[
        Literal["screen_share", "whiteboard", "file_share", "voice", "video"]
    ] = Field(default_factory=list)
    geospatial_context: dict[str, Any] | None = None


class Participant(BaseModel):
    """Model for collaboration participants."""

    user_id: str
    name: str
    role: ParticipantRole = ParticipantRole.PARTICIPANT
    status: ParticipantStatus = ParticipantStatus.OFFLINE
    joined_at: datetime | None = None


class CollaborationSessionResponse(BaseModel):
    """Response model for collaboration sessions."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    session_type: CollaborationType
    status: Literal["scheduled", "active", "paused", "ended"] = "scheduled"
    participants: list[Participant] = Field(default_factory=list)
    join_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None
    geospatial_context: dict[str, Any] | None = None


class JoinSessionResponse(BaseModel):
    """Response model for joining collaboration sessions."""

    session_id: str
    participant_id: str
    join_status: Literal["joined", "waiting", "rejected"] = "joined"
    session_info: dict[str, Any] = Field(default_factory=dict)


class StreamRequest(BaseModel):
    """Request model for creating data streams."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=1000)
    stream_type: Literal["data", "video", "audio", "geospatial", "sensor"] = "data"
    source: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    quality_settings: dict[str, Any] = Field(default_factory=dict)
    geospatial_filter: dict[str, Any] | None = None


class StreamResponse(BaseModel):
    """Response model for data streams."""

    stream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    stream_type: str
    status: Literal["active", "paused", "stopped", "error"] = "active"
    connection_url: str | None = None
    viewer_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    geospatial_filter: dict[str, Any] | None = None


class WebSocketInfoResponse(BaseModel):
    """Response model for WebSocket connection information."""

    websocket_url: str
    protocols: list[str] = Field(default_factory=list)
    connection_token: str
    heartbeat_interval: int = 30


# Collection response models
class MessageListResponse(BaseModel):
    """Response model for message lists."""

    messages: list[MessageResponse]
    total_count: int
    pagination: dict[str, Any] = Field(default_factory=dict)


class ChannelListResponse(BaseModel):
    """Response model for channel lists."""

    channels: list[ChannelResponse]
    total_count: int


class NotificationListResponse(BaseModel):
    """Response model for notification lists."""

    notifications: list[NotificationResponse]
    unread_count: int
    total_count: int


class CollaborationSessionListResponse(BaseModel):
    """Response model for collaboration session lists."""

    sessions: list[CollaborationSessionResponse]
    total_count: int


class StreamListResponse(BaseModel):
    """Response model for stream lists."""

    streams: list[StreamResponse]
    total_count: int


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    connections: dict[str, Any] = Field(default_factory=dict)


class Error(BaseModel):
    """Error response model."""

    error: str
    message: str


# Utility functions for model conversion
def message_request_to_response(
    request: MessageRequest, sender_id: str
) -> MessageResponse:
    """Convert message request to response model."""
    return MessageResponse(
        content=request.content,
        sender_id=sender_id,
        recipients=request.recipients,
        channel_id=request.channel_id,
        message_type=request.message_type,
        priority=request.priority,
        geospatial_data=request.geospatial_data,
    )
