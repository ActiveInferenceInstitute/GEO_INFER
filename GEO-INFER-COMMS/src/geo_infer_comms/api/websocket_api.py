"""
WebSocket API implementation for GEO-INFER-COMMS.

This module provides real-time WebSocket communication capabilities
for the geospatial communication system, supporting live messaging,
notifications, and event streaming with geospatial context.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, cast
from datetime import datetime, UTC

import jwt as pyjwt
import websockets
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from geo_infer_comms import (
    GeospatialCommunicationSystem,
    MessageResponse,
    EventPublishResponse,
    NotificationResponse,
    GeospatialPoint,
    GeospatialMetadata,
)
from geo_infer_comms.models.message import EventSubscriptionRequest
from geo_infer_comms.utils.validation import VALID_EVENT_TYPES


class WebSocketManager:
    """
    WebSocket connection manager for real-time communication.

    Handles WebSocket connections, message routing, and geospatial
    filtering for real-time communication channels.
    """

    def __init__(self, system: GeospatialCommunicationSystem):
        self.system = system
        self.connections: dict[str, WebSocketConnection] = {}
        self.subscriptions: dict[str, set[str]] = {}  # message_type -> connection_ids

        self.logger = logging.getLogger(__name__)
        # Event loop of the running WebSocket server; captured at
        # start_server so broker/worker threads can hand off broadcasts.
        self.server_loop: asyncio.AbstractEventLoop | None = None
        self.broadcasts_sent = 0

    async def handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a new WebSocket connection."""
        connection_id = f"ws_{id(websocket)}"

        try:
            # Create connection wrapper
            connection = WebSocketConnection(
                connection_id=connection_id, websocket=websocket, manager=self
            )

            # Register connection
            self.connections[connection_id] = connection

            # Send welcome message
            await connection.send_message(
                {
                    "type": "connection_established",
                    "connection_id": connection_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            self.logger.info(f"WebSocket connection established: {connection_id}")

            # Handle messages
            await connection.handle_messages()

        except ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            # Clean up connection
            if connection_id in self.connections:
                del self.connections[connection_id]

            # Remove from subscriptions
            for connection_ids in self.subscriptions.values():
                connection_ids.discard(connection_id)

    def broadcast_message(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Safe to call from both async contexts and plain threads (e.g. a
        message-broker worker thread). When no event loop is running in the
        calling thread, the coroutine is handed to the server's loop via
        ``run_coroutine_threadsafe``; if no server loop is known, a warning
        is logged and the broadcast is dropped.
        """
        self._schedule_broadcast(self._broadcast_to_all(message))

    def _schedule_broadcast(self, coro) -> None:
        """Schedule ``coro`` on the server's event loop, loop-safely."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(coro)
            return
        if self.server_loop is not None:
            asyncio.run_coroutine_threadsafe(coro, self.server_loop)
            return
        self.logger.warning(
            "Broadcast requested with no running event loop and no server "
            "loop captured; message dropped (server not started?)"
        )
        coro.close()

    def _capture_server_loop(self) -> None:
        """Capture the running loop so threads can hand off broadcasts."""
        try:
            self.server_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.server_loop = None

    async def _broadcast_to_all(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connections."""
        disconnected = []

        for connection_id, connection in self.connections.items():
            try:
                await connection.send_message(message)
                self.broadcasts_sent += 1
            except Exception as e:
                self.logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected:
            if connection_id in self.connections:
                del self.connections[connection_id]

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.connections)

    def get_connection_stats(self) -> dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "active_connections": len(self.connections),
            "subscriptions": {k: len(v) for k, v in self.subscriptions.items()},
        }


class WebSocketConnection:
    """
    WebSocket connection wrapper with geospatial capabilities.

    Handles individual WebSocket connections with geospatial filtering,
    subscription management, and message routing.
    """

    def __init__(
        self, connection_id: str, websocket: ServerConnection, manager: WebSocketManager
    ):
        self.connection_id = connection_id
        self.websocket = websocket
        self.manager = manager

        self.geospatial_context: GeospatialMetadata | None = None
        self.subscriptions: set[str] = set()
        self.authenticated = False
        self.user_id: str | None = None

        self.logger = logging.getLogger(__name__)

    async def handle_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    await self._process_message(cast(str, message))
                except Exception as e:
                    self.logger.error(
                        f"Error processing message from {self.connection_id}: {e}"
                    )
                    await self.send_error(f"Message processing error: {str(e)}")

        except ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Connection error for {self.connection_id}: {e}")

    async def _process_message(self, message: str) -> None:
        """Process an incoming WebSocket message."""
        try:
            data = json.loads(message)

            message_type = data.get("type", "unknown")

            if message_type == "authenticate":
                await self._handle_authentication(data)
            elif message_type == "subscribe":
                await self._handle_subscription(data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(data)
            elif message_type == "set_location":
                await self._handle_location_update(data)
            elif message_type == "send_message":
                await self._handle_send_message(data)
            elif message_type == "ping":
                await self._handle_ping(data)
            else:
                await self.send_error(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await self.send_error("Invalid JSON message")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self.send_error(f"Processing error: {str(e)}")

    async def _handle_authentication(self, data: dict[str, Any]) -> None:
        """Handle authentication message.

        When ``COMMS_JWT_SECRET`` is configured and PyJWT is installed, the
        token is validated as an HS256 JWT and invalid tokens are rejected.
        If the secret is configured but PyJWT is unavailable, authentication
        is rejected outright: falling back to hash-derived identity would let
        any token string authenticate. Without JWT validation configured, the
        user ID is derived from a deterministic hash of the token (the token
        itself is not verified).
        """
        token = data.get("token")
        if not token:
            await self.send_error("Authentication token required")
            return

        import os

        secret = os.environ.get("COMMS_JWT_SECRET", "")
        user_id: str | None = None

        if secret:
            try:
                payload = pyjwt.decode(token, secret, algorithms=["HS256"])
            except Exception as e:
                self.logger.warning("Rejected invalid JWT token: %s", e)
                await self.send_error("Invalid authentication token")
                return
            user_id = payload.get("sub", payload.get("user_id"))
            if not user_id:
                await self.send_error("Invalid authentication token")
                return
            user_id = str(user_id)

        if user_id is None:
            import hashlib

            digest = hashlib.sha256(str(token).encode()).hexdigest()[:8]
            user_id = f"user_{digest}"

        self.authenticated = True
        self.user_id = user_id

        await self.send_message(
            {
                "type": "authenticated",
                "user_id": self.user_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        self.logger.info(
            f"WebSocket authenticated: {self.connection_id} as {self.user_id}"
        )

    async def _handle_subscription(self, data: dict[str, Any]) -> None:
        """Handle subscription message."""
        if not self.authenticated:
            await self.send_error("Authentication required")
            return

        event_types = data.get("event_types", [])
        if not event_types:
            await self.send_error("Event types required for subscription")
            return

        # Update subscriptions
        for event_type in event_types:
            if event_type not in self.manager.subscriptions:
                self.manager.subscriptions[event_type] = set()
            self.manager.subscriptions[event_type].add(self.connection_id)

        self.subscriptions.update(event_types)

        await self.send_message(
            {
                "type": "subscribed",
                "event_types": list(self.subscriptions),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        self.logger.info(f"WebSocket subscribed: {self.connection_id} to {event_types}")

    async def _handle_unsubscription(self, data: dict[str, Any]) -> None:
        """Handle unsubscription message."""
        event_types = data.get("event_types", list(self.subscriptions))

        # Remove from subscriptions
        for event_type in event_types:
            if event_type in self.manager.subscriptions:
                self.manager.subscriptions[event_type].discard(self.connection_id)

        self.subscriptions -= set(event_types)

        await self.send_message(
            {
                "type": "unsubscribed",
                "event_types": event_types,
                "remaining_subscriptions": list(self.subscriptions),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    async def _handle_location_update(self, data: dict[str, Any]) -> None:
        """Handle location update message."""
        location_data = data.get("location", {})
        longitude = location_data.get("longitude")
        latitude = location_data.get("latitude")

        if longitude is None or latitude is None:
            await self.send_error("Longitude and latitude required")
            return

        try:
            # Create geospatial point and metadata
            location = GeospatialPoint(longitude=longitude, latitude=latitude)
            self.geospatial_context = GeospatialMetadata(
                location=location,
                accuracy=location_data.get("accuracy", 10.0),
                source="websocket",
            )

            await self.send_message(
                {
                    "type": "location_updated",
                    "location": {
                        "longitude": longitude,
                        "latitude": latitude,
                        "accuracy": self.geospatial_context.accuracy,
                    },
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        except Exception as e:
            await self.send_error(f"Invalid location data: {str(e)}")

    async def _handle_send_message(self, data: dict[str, Any]) -> None:
        """Handle send message request."""
        if not self.authenticated:
            await self.send_error("Authentication required")
            return

        content = data.get("content")
        recipients = data.get("recipients", [])

        if not content or not recipients:
            await self.send_error("Content and recipients required")
            return

        try:
            # Send message through the system
            message = self.manager.system.send_message(
                content=content,
                recipients=recipients,
                geospatial_data=self.geospatial_context,
            )

            await self.send_message(
                {
                    "type": "message_sent",
                    "message_id": message.message_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        except Exception as e:
            await self.send_error(f"Failed to send message: {str(e)}")

    async def _handle_ping(self, data: dict[str, Any]) -> None:
        """Handle ping message."""
        await self.send_message(
            {"type": "pong", "timestamp": datetime.now(UTC).isoformat()}
        )

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message to this WebSocket connection."""
        try:
            message_json = json.dumps(message)
            await self.websocket.send(message_json)
        except Exception as e:
            self.logger.error(f"Error sending message to {self.connection_id}: {e}")
            raise

    async def send_error(self, error_message: str) -> None:
        """Send an error message to this connection."""
        await self.send_message(
            {
                "type": "error",
                "message": error_message,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


class WebSocketServer:
    """
    WebSocket server for real-time communication.

    Provides a complete WebSocket server implementation with
    geospatial filtering and real-time messaging capabilities.
    """

    def __init__(
        self,
        system: GeospatialCommunicationSystem,
        host: str = "0.0.0.0",
        port: int = 8001,
        max_connections: int = 10000,
    ):
        self.system = system
        self.host = host
        self.port = port
        self.max_connections = max_connections

        self.websocket_manager = WebSocketManager(system)
        self.server: Any | None = None

        self.logger = logging.getLogger(__name__)

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        self.server = await websockets.serve(
            self.websocket_manager.handle_connection,
            self.host,
            self.port,
            max_size=2**20,  # 1MB max message size
            max_queue=32,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10,
        )
        self.websocket_manager._capture_server_loop()

        self.logger.info("WebSocket server started")

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket server statistics."""
        return {
            "connections": self.websocket_manager.get_connection_count(),
            "subscriptions": self.websocket_manager.get_connection_stats(),
            "host": self.host,
            "port": self.port,
        }

    def broadcast_system_message(self, message: dict[str, Any]) -> None:
        """Broadcast a system message to all connected clients."""
        self.websocket_manager.broadcast_message(
            {
                "type": "system_message",
                **message,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


class GeospatialWebSocketHandler:
    """
    Advanced geospatial WebSocket handler with spatial filtering.

    Provides sophisticated geospatial message routing and filtering
    for WebSocket connections based on location and spatial context.
    """

    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.spatial_filters: dict[str, dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

    def add_spatial_filter(
        self, connection_id: str, filter_config: dict[str, Any]
    ) -> None:
        """Add a spatial filter for a WebSocket connection."""
        self.spatial_filters[connection_id] = filter_config
        self.logger.info(f"Spatial filter added for connection: {connection_id}")

    def remove_spatial_filter(self, connection_id: str) -> None:
        """Remove spatial filter for a WebSocket connection."""
        if connection_id in self.spatial_filters:
            del self.spatial_filters[connection_id]
            self.logger.info(f"Spatial filter removed for connection: {connection_id}")

    def should_receive_message(
        self, connection_id: str, message: MessageResponse
    ) -> bool:
        """Check if connection should receive a message based on spatial filters."""
        if connection_id not in self.spatial_filters:
            return True  # No filter means receive all

        filter_config = self.spatial_filters[connection_id]

        # Check geospatial filtering
        if message.geospatial_data and "spatial_filter" in filter_config:
            spatial_filter = filter_config["spatial_filter"]

            # Simple distance-based filtering
            max_distance = spatial_filter.get("max_distance_meters")
            if max_distance:
                # Get connection location (would be stored in connection)
                connection_location = getattr(
                    self.websocket_manager.connections.get(connection_id),
                    "geospatial_context",
                    None,
                )

                if connection_location:
                    distance = message.geospatial_data.distance_to(connection_location)
                    if distance > max_distance:
                        return False

            # Bounds-based filtering
            bounds = spatial_filter.get("bounds")
            if bounds and message.geospatial_data.location:
                # Check if message location is within bounds
                if not self._point_in_bounds(message.geospatial_data.location, bounds):
                    return False

        return True

    def _point_in_bounds(self, point: GeospatialPoint, bounds: dict[str, Any]) -> bool:
        """Check if point is within bounds."""
        min_lon = bounds.get("min_longitude", -180)
        max_lon = bounds.get("max_longitude", 180)
        min_lat = bounds.get("min_latitude", -90)
        max_lat = bounds.get("max_latitude", 90)

        return cast(
            bool,
            min_lon <= point.longitude <= max_lon
            and min_lat <= point.latitude <= max_lat,
        )


class RealTimeMessageBroadcaster:
    """
    Real-time message broadcaster for WebSocket connections.

    Handles broadcasting messages to appropriate WebSocket connections
    based on subscriptions, geospatial filters, and connection state.
    """

    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.geospatial_handler = GeospatialWebSocketHandler(websocket_manager)

        self.logger = logging.getLogger(__name__)

    def broadcast_message(self, message: MessageResponse) -> None:
        """Broadcast a message to appropriate WebSocket connections.

        Loop-safe: callable from async contexts or plain worker threads
        (see :meth:`WebSocketManager._schedule_broadcast`).
        """
        self.websocket_manager._schedule_broadcast(
            self._broadcast_message_async(message)
        )

    async def _broadcast_message_async(self, message: MessageResponse) -> None:
        """Asynchronously broadcast message to connections."""
        broadcast_count = 0

        for connection_id, connection in self.websocket_manager.connections.items():
            # Check if connection should receive this message
            if self._should_receive_message(connection_id, message):
                try:
                    # Format message for WebSocket
                    ws_message = self._format_message_for_websocket(message)
                    await connection.send_message(ws_message)
                    broadcast_count += 1

                except Exception as e:
                    self.logger.error(f"Error broadcasting to {connection_id}: {e}")

        self.logger.info(
            f"Message broadcast to {broadcast_count} connections: {message.message_id}"
        )

    def _should_receive_message(
        self, connection_id: str, message: MessageResponse
    ) -> bool:
        """Check if connection should receive this message."""
        # Check geospatial filtering
        if not self.geospatial_handler.should_receive_message(connection_id, message):
            return False

        # Check subscription-based filtering
        connection = self.websocket_manager.connections.get(connection_id)
        if connection and hasattr(connection, "subscriptions"):
            # If connection has specific subscriptions, check them
            if connection.subscriptions:
                # In a real implementation, would check message type against subscriptions
                return True

        return True

    def _format_message_for_websocket(self, message: MessageResponse) -> dict[str, Any]:
        """Format a message for WebSocket transmission."""

        def _enum_value(field: Any) -> Any:
            """Return ``field.value`` for enums, the raw value otherwise
            (models configured with ``use_enum_values=True`` store str)."""
            return field.value if hasattr(field, "value") else field

        formatted: dict[str, Any] = {
            "type": "message",
            "message_id": message.message_id,
            "content": message.content,
            "sender_id": message.sender_id,
            "timestamp": message.timestamp.isoformat(),
            "priority": _enum_value(message.priority),
            "message_type": _enum_value(message.message_type),
        }

        # Add geospatial data if present
        if message.geospatial_data:
            formatted["geospatial_data"] = {
                "location": {
                    "longitude": message.geospatial_data.location.longitude,
                    "latitude": message.geospatial_data.location.latitude,
                    "accuracy": message.geospatial_data.accuracy,
                }
            }

            if message.geospatial_data.bounds:
                formatted["geospatial_data"]["bounds"] = {
                    "min_longitude": message.geospatial_data.bounds.min_longitude,
                    "min_latitude": message.geospatial_data.bounds.min_latitude,
                    "max_longitude": message.geospatial_data.bounds.max_longitude,
                    "max_latitude": message.geospatial_data.bounds.max_latitude,
                }

        return formatted


class WebSocketAPIManager:
    """
    Complete WebSocket API manager.

    Integrates WebSocket server, geospatial filtering, and real-time
    broadcasting for a complete WebSocket communication solution.
    """

    def __init__(self, system: GeospatialCommunicationSystem):
        self.system = system
        self.websocket_server = WebSocketServer(system)
        self.message_broadcaster = RealTimeMessageBroadcaster(
            self.websocket_server.websocket_manager
        )

        # Register message broadcasting with the system
        self._register_broadcasters()

        self.logger = logging.getLogger(__name__)

    def _register_broadcasters(self) -> None:
        """Register broadcasting callbacks at the system's emission points.

        - Messages: subscribed on the system's :class:`MessageBroker`.
        - Events: subscribed on the system's :class:`EventManager` for every
          valid event type.
        - Notifications: registered as the ``websocket`` delivery handler on
          the system's :class:`NotificationManager` (notifications must
          include ``"websocket"`` in their delivery methods to be broadcast).
        """

        def broadcast_message_callback(message: MessageResponse) -> None:
            self.message_broadcaster.broadcast_message(message)

        def broadcast_event_callback(event: EventPublishResponse) -> None:
            self._broadcast_event(event)

        def broadcast_notification_callback(
            notification: NotificationResponse,
        ) -> bool:
            self._broadcast_notification(notification)
            return True

        # Messages: broker publishes dispatch to subscriber callbacks.
        self.system.message_broker.subscribe(
            "websocket_api_broadcaster", broadcast_message_callback
        )

        # Events: subscribe to each valid event type via one subscription
        # request (the manager matches subscribers by event type).
        self.system.event_manager.subscribe_to_events(
            "websocket_api_broadcaster",
            EventSubscriptionRequest(event_types=list(VALID_EVENT_TYPES)),
            broadcast_event_callback,
        )

        # Notifications: deliver via the websocket delivery handler.
        self.system.notification_manager.register_delivery_handler(
            "websocket", broadcast_notification_callback
        )

    def _broadcast_event(self, event: EventPublishResponse) -> None:
        """Broadcast an event to WebSocket connections."""
        event_message = {
            "type": "event",
            "event_id": event.event_id,
            "event_type": event.event_type,
            "payload": event.payload,
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "priority": event.priority.value,
        }

        if event.geospatial_context:
            event_message["geospatial_context"] = event.geospatial_context

        self.websocket_server.websocket_manager.broadcast_message(event_message)

    def _broadcast_notification(self, notification: NotificationResponse) -> None:
        """Broadcast a notification to WebSocket connections."""
        notification_message: dict[str, Any] = {
            "type": "notification",
            "notification_id": notification.notification_id,
            "title": notification.title,
            "content": notification.content,
            "notification_type": notification.notification_type.value,
            "priority": notification.priority.value,
            "timestamp": notification.created_at.isoformat(),
        }

        if notification.geospatial_context:
            notification_message["geospatial_context"] = notification.geospatial_context

        self.websocket_server.websocket_manager.broadcast_message(notification_message)

    async def start(self) -> None:
        """Start the WebSocket API server."""
        await self.websocket_server.start_server()

    async def stop(self) -> None:
        """Stop the WebSocket API server."""
        await self.websocket_server.stop_server()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive WebSocket API statistics."""
        return {
            "websocket_server": self.websocket_server.get_stats(),
            "message_broadcaster": {
                "broadcasts_sent": self.websocket_server.websocket_manager.broadcasts_sent,
            },
        }
