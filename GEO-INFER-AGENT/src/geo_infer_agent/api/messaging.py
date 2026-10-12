#!/usr/bin/env python3

"""
Agent-to-agent messaging interface for GEO-INFER-AGENT.

This module provides a communication system for agents to exchange
messages, coordinate activities, and share information within the
GEO-INFER ecosystem.
"""

import asyncio
import logging
from typing import Any, Optional
from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger("geo_infer_agent.api.messaging")


class Message:
    """Represents an agent-to-agent message."""

    def __init__(
        self,
        from_agent_id: str,
        to_agent_id: str,
        content: dict[str, Any],
        message_type: str = "standard",
        priority: int = 1,
        expires_at: datetime | None = None,
    ):
        """
        Initialize a new message.

        Args:
            from_agent_id: ID of the sending agent
            to_agent_id: ID of the receiving agent
            content: Message payload
            message_type: Type of message (standard, request, response, broadcast)
            priority: Message priority (1-10, higher is more important)
            expires_at: When the message expires (None for no expiration)
        """
        self.message_id = str(uuid4())
        self.from_agent_id = from_agent_id
        self.to_agent_id = to_agent_id
        self.content = content
        self.message_type = message_type
        self.priority = max(1, min(10, priority))  # Clamp between 1-10
        self.created_at = datetime.now()
        self.expires_at = expires_at
        self.delivered = False
        self.read = False
        self.replied = False

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert message to a dictionary."""
        return {
            "message_id": self.message_id,
            "from_agent_id": self.from_agent_id,
            "to_agent_id": self.to_agent_id,
            "content": self.content,
            "message_type": self.message_type,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "delivered": self.delivered,
            "read": self.read,
            "replied": self.replied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create a message from a dictionary."""
        msg = cls(
            from_agent_id=data["from_agent_id"],
            to_agent_id=data["to_agent_id"],
            content=data["content"],
            message_type=data["message_type"],
            priority=data["priority"],
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )
        msg.message_id = data["message_id"]
        msg.created_at = datetime.fromisoformat(data["created_at"])
        msg.delivered = data["delivered"]
        msg.read = data["read"]
        msg.replied = data["replied"]
        return msg


class MessagingService:
    """
    Agent-to-agent messaging service.

    This service handles:
    - Direct messages between agents
    - Broadcast messages to groups of agents
    - Message queuing and delivery
    - Pub/sub channels for topic-based communication
    """

    _instance: Optional["MessagingService"] = None
    _initialized: bool = False

    def __new__(cls) -> "MessagingService":
        """Singleton pattern to ensure a single messaging service instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the messaging service."""
        if self._initialized:
            return

        # Message queues by agent ID
        self.message_queues: dict[str, list[Message]] = {}

        # Pub/sub channels and subscribers
        self.channels: dict[str, set[str]] = {}

        # Callbacks for new message handlers
        self.message_callbacks: dict[str, Callable[[Message], None]] = {}

        # Background task for message processing
        self.processing_task: asyncio.Task | None = None
        self.running = False

        self._initialized = True
        logger.info("Messaging service initialized")

    async def start(self) -> None:
        """Start the messaging service."""
        if self.running:
            return

        self.running = True
        self.processing_task = asyncio.create_task(self._process_messages())
        logger.info("Messaging service started")

    async def stop(self) -> None:
        """Stop the messaging service."""
        if not self.running:
            return

        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Messaging service stopped")

    async def send_message(self, message: Message) -> bool:
        """
        Send a message to an agent.

        Args:
            message: The message to send

        Returns:
            True if the message was queued successfully
        """
        if message.is_expired():
            logger.warning(
                f"Attempted to send already expired message: {message.message_id}"
            )
            return False

        # Ensure queue exists for the recipient
        if message.to_agent_id not in self.message_queues:
            self.message_queues[message.to_agent_id] = []

        # Add message to recipient's queue
        self.message_queues[message.to_agent_id].append(message)

        # Sort queue by priority
        self.message_queues[message.to_agent_id].sort(
            key=lambda m: m.priority, reverse=True
        )

        logger.debug(
            f"Message {message.message_id} queued from {message.from_agent_id} to {message.to_agent_id}"
        )
        return True

    async def broadcast_message(
        self,
        from_agent_id: str,
        content: dict[str, Any],
        channel: str,
        priority: int = 1,
        expires_at: datetime | None = None,
    ) -> int:
        """
        Broadcast a message to all subscribers of a channel.

        Args:
            from_agent_id: ID of the sending agent
            content: Message payload
            channel: Channel to broadcast on
            priority: Message priority
            expires_at: When the message expires

        Returns:
            Number of agents the message was sent to
        """
        if channel not in self.channels:
            logger.warning(f"Attempted to broadcast to non-existent channel: {channel}")
            return 0

        sent_count = 0
        for agent_id in self.channels[channel]:
            message = Message(
                from_agent_id=from_agent_id,
                to_agent_id=agent_id,
                content=content,
                message_type="broadcast",
                priority=priority,
                expires_at=expires_at,
            )
            success = await self.send_message(message)
            if success:
                sent_count += 1

        logger.debug(
            f"Broadcast message sent to {sent_count} agents on channel {channel}"
        )
        return sent_count

    def register_agent(self, agent_id: str) -> None:
        """
        Register an agent with the messaging service.

        Args:
            agent_id: ID of the agent to register
        """
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = []
            logger.debug(f"Agent {agent_id} registered with messaging service")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the messaging service.

        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.message_queues:
            del self.message_queues[agent_id]

        # Remove from all channels
        for channel in self.channels:
            if agent_id in self.channels[channel]:
                self.channels[channel].remove(agent_id)

        # Remove callbacks
        if agent_id in self.message_callbacks:
            del self.message_callbacks[agent_id]

        logger.debug(f"Agent {agent_id} unregistered from messaging service")

    def subscribe(self, agent_id: str, channel: str) -> None:
        """
        Subscribe an agent to a channel.

        Args:
            agent_id: ID of the subscribing agent
            channel: Channel to subscribe to
        """
        if channel not in self.channels:
            self.channels[channel] = set()

        self.channels[channel].add(agent_id)
        logger.debug(f"Agent {agent_id} subscribed to channel {channel}")

    def unsubscribe(self, agent_id: str, channel: str) -> None:
        """
        Unsubscribe an agent from a channel.

        Args:
            agent_id: ID of the agent
            channel: Channel to unsubscribe from
        """
        if channel in self.channels and agent_id in self.channels[channel]:
            self.channels[channel].remove(agent_id)
            logger.debug(f"Agent {agent_id} unsubscribed from channel {channel}")

    def register_message_callback(
        self, agent_id: str, callback: Callable[[Message], None]
    ) -> None:
        """
        Register a callback for when an agent receives a message.

        Args:
            agent_id: ID of the agent
            callback: Function to call when a message is received
        """
        self.message_callbacks[agent_id] = callback
        logger.debug(f"Message callback registered for agent {agent_id}")

    async def get_messages(
        self, agent_id: str, mark_as_read: bool = True
    ) -> list[Message]:
        """
        Get all messages for an agent.

        Args:
            agent_id: ID of the agent
            mark_as_read: Whether to mark retrieved messages as read

        Returns:
            List of messages for the agent
        """
        if agent_id not in self.message_queues:
            return []

        # Filter out expired messages
        valid_messages = [
            m for m in self.message_queues[agent_id] if not m.is_expired()
        ]

        # Update queue to remove expired messages
        self.message_queues[agent_id] = valid_messages

        # Mark as delivered
        for message in valid_messages:
            message.delivered = True
            if mark_as_read:
                message.read = True

        logger.debug(f"Retrieved {len(valid_messages)} messages for agent {agent_id}")
        return valid_messages

    async def _process_messages(self) -> None:
        """Background task to process messages."""
        while self.running:
            try:
                # Process message callbacks
                for agent_id, callback in list(self.message_callbacks.items()):
                    if (
                        agent_id in self.message_queues
                        and self.message_queues[agent_id]
                    ):
                        messages = await self.get_messages(agent_id, mark_as_read=False)
                        for message in messages:
                            try:
                                callback(message)
                                message.delivered = True
                                # Callback delivery is at-least-once: retain a
                                # message when the callback fails so it can be
                                # retried, but consume it after success.
                                self.message_queues[agent_id] = [
                                    queued
                                    for queued in self.message_queues[agent_id]
                                    if queued.message_id != message.message_id
                                ]
                            except Exception as e:
                                message.delivered = False
                                logger.error(
                                    f"Error processing message callback for agent {agent_id}: {str(e)}"
                                )

                # Clean up expired messages
                for agent_id in self.message_queues:
                    self.message_queues[agent_id] = [
                        m for m in self.message_queues[agent_id] if not m.is_expired()
                    ]

                await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processing: {str(e)}")
                await asyncio.sleep(1)  # Longer delay on error


# Global instance
messaging_service = MessagingService()
