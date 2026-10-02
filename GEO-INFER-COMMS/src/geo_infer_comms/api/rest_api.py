"""
REST API implementation for GEO-INFER-COMMS.

This module provides comprehensive REST API endpoints for the geospatial
communication system, supporting all core functionality with proper
authentication, validation, and geospatial context handling.
"""

from __future__ import annotations
from importlib.metadata import version as distribution_version
import logging
from typing import Any
from datetime import datetime, UTC

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import jwt as pyjwt
import uvicorn

from geo_infer_comms import (
    GeospatialCommunicationSystem,
    MessageRequest,
    MessageResponse,
    ChannelRequest,
    ChannelResponse,
    NotificationRequest,
    NotificationResponse,
    EventPublishRequest,
    EventPublishResponse,
    BroadcastRequest,
    BroadcastResponse,
    GeospatialPoint,
    ChannelType,
    validate_coordinates,
    validate_message_content,
)


class CommunicationAPI:
    """
    REST API server for geospatial communication system.

    Provides comprehensive HTTP endpoints for messaging, notifications,
    channels, events, and geospatial operations with proper authentication
    and validation.
    """

    def __init__(
        self,
        system: GeospatialCommunicationSystem,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_auth: bool = True,
        enable_cors: bool = True,
        cors_origins: list[str] | None = None,
    ):
        self.system = system
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        cors_origins = list(cors_origins) if cors_origins is not None else ["*"]

        # Create FastAPI application
        self.app = FastAPI(
            title="GEO-INFER-COMMS API",
            description="Geospatial Communications Infrastructure API",
            version=distribution_version("geo-infer-comms"),
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware if enabled
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Set up authentication
        self.security = HTTPBearer() if enable_auth else None

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Register API routes
        self._register_routes()

        # Health check endpoint
        @self.app.get("/health")
        async def health_check() -> JSONResponse:
            """System health check endpoint."""
            health = self.system.get_system_health()
            return JSONResponse(status_code=status.HTTP_200_OK, content=health)

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            """API root endpoint."""
            return {
                "name": "GEO-INFER-COMMS API",
                "version": distribution_version("geo-infer-comms"),
                "status": "running",
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _register_routes(self) -> None:
        """Register all API routes."""

        # Message endpoints
        @self.app.post("/messages", response_model=MessageResponse)
        async def send_message(
            request: MessageRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> MessageResponse:
            """Send a new message."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                # Validate message content
                if not validate_message_content(request.content):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid message content",
                    )

                message = self.system.send_message(
                    content=request.content,
                    recipients=request.recipients,
                    channel_id=request.channel_id,
                    message_type=request.message_type,
                    priority=request.priority,
                    geospatial_data=request.geospatial_data,
                )

                return message

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/messages", response_model=list[MessageResponse])
        async def get_messages(
            sender_id: str | None = None,
            channel_id: str | None = None,
            limit: int = 100,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> list[MessageResponse]:
            """Get messages with optional filtering."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                messages = self.system.message_broker.get_messages(
                    sender_id=sender_id, channel_id=channel_id, limit=limit
                )

                return messages

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting messages: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/messages/{message_id}", response_model=MessageResponse)
        async def get_message(
            message_id: str,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> MessageResponse:
            """Get a specific message by ID."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                message = self.system.message_broker.get_message(message_id)
                if not message:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Message not found",
                    )

                return message

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting message: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        # Channel endpoints
        @self.app.post("/channels", response_model=ChannelResponse)
        async def create_channel(
            request: ChannelRequest,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> ChannelResponse:
            """Create a new communication channel."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                channel = self.system.create_channel(
                    name=request.name,
                    channel_type=request.type,
                    description=request.description,
                    permissions=request.permissions,
                    settings=request.settings,
                    geospatial_bounds=request.geospatial_bounds,
                )

                return channel

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error creating channel: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/channels", response_model=list[ChannelResponse])
        async def get_channels(
            channel_type: ChannelType | None = None,
            limit: int = 100,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> list[ChannelResponse]:
            """Get channels with optional filtering."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                channels = self.system.channel_manager.get_channels(
                    channel_type=channel_type, limit=limit
                )

                return channels

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting channels: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/channels/{channel_id}", response_model=ChannelResponse)
        async def get_channel(
            channel_id: str,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> ChannelResponse:
            """Get a specific channel by ID."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                channel = self.system.channel_manager.get_channel(channel_id)
                if not channel:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Channel not found",
                    )

                return channel

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting channel: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        # Notification endpoints
        @self.app.post("/notifications", response_model=NotificationResponse)
        async def create_notification(
            request: NotificationRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> NotificationResponse:
            """Create a new notification."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                notification = self.system.create_notification(
                    title=request.title,
                    content=request.content,
                    recipients=request.recipients,
                    notification_type=request.notification_type,
                    priority=request.priority,
                    delivery_method=request.delivery_method,
                    schedule_time=request.schedule_time,
                    expiry_time=request.expiry_time,
                    geospatial_context=request.geospatial_context,
                )

                return notification

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error creating notification: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/notifications", response_model=list[NotificationResponse])
        async def get_notifications(
            status_filter: str | None = None,
            limit: int = 100,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> list[NotificationResponse]:
            """Get notifications with optional filtering."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                # Convert status string to enum if provided
                notification_status = None
                if status_filter:
                    try:
                        from geo_infer_comms.models.message import NotificationStatus

                        notification_status = NotificationStatus(status_filter)
                    except ValueError as exc:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid notification status: {status_filter}",
                        ) from exc

                notifications = self.system.notification_manager.get_notifications(
                    status=notification_status, limit=limit
                )

                return notifications

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting notifications: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        # Event endpoints
        @self.app.post("/events", response_model=EventPublishResponse)
        async def publish_event(
            request: EventPublishRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> EventPublishResponse:
            """Publish a new event."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                event = self.system.publish_event(
                    event_type=request.event_type,
                    payload=request.payload,
                    source=request.source,
                    target_channels=request.target_channels,
                    priority=request.priority,
                    persistence=request.persistence,
                    geospatial_context=request.geospatial_context,
                )

                return event

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error publishing event: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/events", response_model=list[EventPublishResponse])
        async def get_events(
            event_type: str | None = None,
            source: str | None = None,
            limit: int = 100,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> list[EventPublishResponse]:
            """Get events with optional filtering."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                events = self.system.event_manager.get_events(
                    event_type=event_type, source=source, limit=limit
                )

                return events

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting events: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        # Broadcast endpoints
        @self.app.post("/messages/broadcast", response_model=BroadcastResponse)
        async def broadcast_message(
            request: BroadcastRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> BroadcastResponse:
            """Broadcast a message to multiple recipients."""
            try:
                user_id = (
                    self._validate_credentials(credentials)
                    if self.enable_auth
                    else "system"
                )

                broadcast = self.system.message_broker.broadcast_message(
                    request=request, sender_id=user_id
                )

                return broadcast

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error broadcasting message: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        # Geospatial endpoints
        @self.app.post("/geospatial/distance")
        async def calculate_distance(
            point1: dict[str, float],
            point2: dict[str, float],
            method: str = "haversine",
        ) -> dict[str, Any]:
            """Calculate distance between two geospatial points."""
            try:
                # Validate input points
                if not all(key in point1 for key in ["longitude", "latitude"]):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid point1 format",
                    )

                if not all(key in point2 for key in ["longitude", "latitude"]):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid point2 format",
                    )

                # Create geospatial points
                p1 = GeospatialPoint(
                    longitude=point1["longitude"], latitude=point1["latitude"]
                )
                p2 = GeospatialPoint(
                    longitude=point2["longitude"], latitude=point2["latitude"]
                )

                # Calculate distance
                distance = p1.distance_to(p2, method=method)

                return {
                    "distance_meters": distance,
                    "method": method,
                    "point1": {"longitude": p1.longitude, "latitude": p1.latitude},
                    "point2": {"longitude": p2.longitude, "latitude": p2.latitude},
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error calculating distance: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        @self.app.get("/geospatial/channels/nearby")
        async def get_nearby_channels(
            longitude: float, latitude: float, radius_km: float = 1.0, limit: int = 50
        ) -> dict[str, Any]:
            """Find channels near a specific location."""
            try:
                # Validate coordinates
                if not validate_coordinates(longitude, latitude):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid coordinates",
                    )

                # Create reference point
                location = GeospatialPoint(longitude=longitude, latitude=latitude)

                # Find nearby channels
                channels = self.system.channel_manager.get_channels_by_location(
                    location=location, radius_km=radius_km
                )

                return {
                    "location": {"longitude": longitude, "latitude": latitude},
                    "radius_km": radius_km,
                    "channels_found": len(channels),
                    "channels": channels[:limit],
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error finding nearby channels: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

        # System metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics(
            credentials: HTTPAuthorizationCredentials | None = (
                Depends(self._get_credentials) if self.enable_auth else None
            ),
        ) -> dict[str, Any]:
            """Get comprehensive system metrics."""
            try:
                self._validate_credentials(credentials) if self.enable_auth else None

                metrics = self.system.get_comprehensive_metrics()

                return metrics

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                ) from e

    def _get_credentials(self) -> HTTPBearer | None:
        """Get authentication credentials dependency."""
        return self.security

    def _validate_credentials(
        self, credentials: HTTPAuthorizationCredentials | None
    ) -> str:
        """Validate authentication credentials.

        With ``COMMS_JWT_SECRET`` set and PyJWT installed, the token is
        decoded as an HS256 JWT and invalid tokens are rejected with 401.
        Only when JWT validation is unavailable (PyJWT missing or no secret
        configured) does a deterministic hash fallback derive the user ID.
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        token = credentials.credentials
        if not token or len(token) < 10:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )

        # Decode the token as an HS256 JWT when COMMS_JWT_SECRET is set. A
        # token that fails validation is rejected outright; the hash fallback
        # only applies when JWT validation is not configured (no secret).
        import os

        secret = os.environ.get("COMMS_JWT_SECRET", "")
        if secret:
            try:
                payload = pyjwt.decode(token, secret, algorithms=["HS256"])
            except Exception as e:
                self.logger.warning("Rejected invalid JWT token: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                ) from e
            user_id = payload.get("sub", payload.get("user_id"))
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                )
            return str(user_id)

        # Fallback: derive a deterministic user identifier from the token
        import hashlib

        digest = hashlib.sha256(token.encode()).hexdigest()[:8]
        user_id = f"user_{digest}"

        self.logger.debug("Token validated via hash fallback for %s", user_id)
        return user_id

    def start_server(self) -> None:
        """Start the API server."""
        self.logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(
            self.app, host=self.host, port=self.port, log_level="info", access_log=True
        )

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


def create_api_server(
    system: GeospatialCommunicationSystem, config: dict[str, Any] | None = None
) -> CommunicationAPI:
    """
    Create and configure a communication API server.

    Args:
        system: Geospatial communication system instance
        config: API configuration options

    Returns:
        Configured API server instance
    """
    api_config = config or {}

    return CommunicationAPI(
        system=system,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        enable_auth=api_config.get("enable_auth", True),
        enable_cors=api_config.get("enable_cors", True),
        cors_origins=api_config.get("cors_origins", ["*"]),
    )
