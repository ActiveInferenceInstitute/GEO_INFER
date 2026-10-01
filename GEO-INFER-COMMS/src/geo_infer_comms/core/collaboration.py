"""
Collaboration system for GEO-INFER-COMMS.

This module implements comprehensive real-time collaboration functionality
including session management, participant coordination, shared workspaces,
and geospatial context for multi-user collaborative work.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, cast
import threading
from datetime import datetime, UTC
from dataclasses import dataclass, field
import uuid

from geo_infer_comms.models.message import (
    CollaborationSessionRequest,
    CollaborationSessionResponse,
    CollaborationType,
    Participant,
    ParticipantRole,
    ParticipantStatus,
    JoinSessionResponse,
)
from geo_infer_comms.models.spatial import GeospatialMetadata, GeospatialPoint
from geo_infer_comms.utils.validation import validate_collaboration_session


class CollaborationManager:
    """
    Central collaboration session management system.

    Handles creation, management, and coordination of multi-user
    collaboration sessions with geospatial context and real-time
    participant interaction.
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        max_participants_per_session: int = 100,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
    ):
        self.max_sessions = max_sessions
        self.max_participants_per_session = max_participants_per_session
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Session storage and management
        self.sessions: dict[str, CollaborationSessionResponse] = {}
        self.session_participants: dict[str, dict[str, Participant]] = {}
        self.participant_sessions: dict[str, set[str]] = {}  # user_id -> session_ids

        # Real-time collaboration features
        self.shared_workspaces: dict[str, dict[str, Any]] = {}
        self.session_messages: dict[str, list[dict[str, Any]]] = {}
        self.session_documents: dict[str, dict[str, Any]] = {}

        # Threading and concurrency
        self._lock = threading.RLock()
        self._background_tasks: dict[str, asyncio.Task] = {}

        # Metrics and monitoring
        self.metrics = CollaborationMetrics()

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def create_session(
        self, request: CollaborationSessionRequest, creator_id: str
    ) -> CollaborationSessionResponse:
        """
        Create a new collaboration session.

        Args:
            request: Session creation request
            creator_id: ID of the session creator

        Returns:
            Created collaboration session

        Raises:
            ValueError: If session request is invalid or limit reached
        """
        # Validate request
        if not validate_collaboration_session(request.model_dump()):
            raise ValueError("Invalid collaboration session configuration")

        # Check session limit
        if len(self.sessions) >= self.max_sessions:
            raise ValueError(
                f"Maximum number of sessions ({self.max_sessions}) reached"
            )

        # Create session response
        session = CollaborationSessionResponse(
            name=request.name,
            description=request.description,
            session_type=request.session_type,
            participants=[
                Participant(
                    user_id=creator_id,
                    name=f"User {creator_id}",
                    role=ParticipantRole.HOST,
                    status=ParticipantStatus.ONLINE,
                    joined_at=datetime.now(UTC),
                )
            ],
            geospatial_context=request.geospatial_context,
        )

        # Initialize session data structures
        with self._lock:
            self.sessions[session.session_id] = session
            self.session_participants[session.session_id] = {
                creator_id: session.participants[0]
            }
            self.participant_sessions[creator_id] = {session.session_id}

            # Initialize shared workspace
            self.shared_workspaces[session.session_id] = {
                "documents": {},
                "messages": [],
                "whiteboard": {},
                "shared_files": [],
            }
            self.session_messages[session.session_id] = []
            self.session_documents[session.session_id] = {}

        self.metrics.sessions_created += 1
        self.logger.info(
            f"Collaboration session created: {session.session_id} by {creator_id}"
        )
        return session

    def join_session(
        self,
        session_id: str,
        user_id: str,
        participant_role: ParticipantRole = ParticipantRole.PARTICIPANT,
    ) -> JoinSessionResponse:
        """
        Join an existing collaboration session.

        Args:
            session_id: ID of session to join
            user_id: ID of user joining
            participant_role: Role for the participant

        Returns:
            Join session response

        Raises:
            ValueError: If session not found or user already in session
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status != "active":
            raise ValueError(f"Session is not active: {session.status}")

        # Check participant limit
        if len(session.participants) >= self.max_participants_per_session:
            raise ValueError(
                f"Session is full (max {self.max_participants_per_session} participants)"
            )

        with self._lock:
            # Check if user is already in session
            if user_id in self.session_participants.get(session_id, {}):
                return JoinSessionResponse(
                    session_id=session_id, participant_id=user_id, join_status="joined"
                )

            # Create participant
            participant = Participant(
                user_id=user_id,
                name=f"User {user_id}",
                role=participant_role,
                status=ParticipantStatus.ONLINE,
                joined_at=datetime.now(UTC),
            )

            # Add to session
            session.participants.append(participant)
            self.session_participants[session_id][user_id] = participant

            # Update participant sessions
            if user_id not in self.participant_sessions:
                self.participant_sessions[user_id] = set()
            self.participant_sessions[user_id].add(session_id)

        self.metrics.participants_joined += 1
        self.logger.info(f"User {user_id} joined session {session_id}")

        return JoinSessionResponse(
            session_id=session_id,
            participant_id=user_id,
            join_status="joined",
            session_info=self._get_session_info(session_id),
        )

    def leave_session(self, session_id: str, user_id: str) -> bool:
        """
        Leave a collaboration session.

        Args:
            session_id: ID of session to leave
            user_id: ID of user leaving

        Returns:
            True if successfully left
        """
        with self._lock:
            if session_id not in self.session_participants:
                return False

            if user_id not in self.session_participants[session_id]:
                return False

            # Remove participant from session
            participant = self.session_participants[session_id][user_id]

            # Update participant status
            participant.status = ParticipantStatus.OFFLINE

            # Remove from session after a delay (graceful exit)
            # In a real implementation, might keep for a short time for reconnection

            del self.session_participants[session_id][user_id]

            # Update participant sessions
            if user_id in self.participant_sessions:
                self.participant_sessions[user_id].discard(session_id)

        self.metrics.participants_left += 1
        self.logger.info(f"User {user_id} left session {session_id}")
        return True

    def end_session(self, session_id: str, ended_by: str) -> bool:
        """
        End a collaboration session.

        Args:
            session_id: ID of session to end
            ended_by: ID of user ending the session

        Returns:
            True if successfully ended
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check permissions
        participant = self.session_participants.get(session_id, {}).get(ended_by)
        if not participant or participant.role not in [
            ParticipantRole.HOST,
            ParticipantRole.MODERATOR,
        ]:
            return False

        with self._lock:
            # Update session status
            session.status = "ended"
            session.ended_at = datetime.now(UTC)

            # Update all participants
            for p in session.participants:
                p.status = ParticipantStatus.OFFLINE

        self.metrics.sessions_ended += 1
        self.logger.info(f"Session ended: {session_id} by {ended_by}")
        return True

    def get_session(self, session_id: str) -> CollaborationSessionResponse | None:
        """Get a specific session by ID."""
        with self._lock:
            return self.sessions.get(session_id)

    def get_sessions(
        self,
        session_type: CollaborationType | None = None,
        status: str | None = None,
        participant_id: str | None = None,
        limit: int = 100,
    ) -> list[CollaborationSessionResponse]:
        """Get sessions with filtering."""
        with self._lock:
            sessions = list(self.sessions.values())

        # Apply filters
        filtered_sessions = sessions

        if session_type:
            filtered_sessions = [
                s for s in filtered_sessions if s.session_type == session_type
            ]

        if status:
            filtered_sessions = [s for s in filtered_sessions if s.status == status]

        if participant_id:
            filtered_sessions = [
                s
                for s in filtered_sessions
                if participant_id in self.session_participants.get(s.session_id, {})
            ]

        # Sort by creation time (newest first) and limit
        filtered_sessions.sort(key=lambda s: s.created_at, reverse=True)
        return filtered_sessions[:limit]

    def get_participant_sessions(
        self, user_id: str
    ) -> list[CollaborationSessionResponse]:
        """Get all sessions for a specific participant."""
        session_ids = self.participant_sessions.get(user_id, set())

        sessions = []
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session:
                sessions.append(session)

        return sessions

    def add_session_message(
        self, session_id: str, user_id: str, message: dict[str, Any]
    ) -> bool:
        """Add a message to a session's shared workspace."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check if user is participant
        if user_id not in self.session_participants.get(session_id, {}):
            return False

        with self._lock:
            if session_id not in self.session_messages:
                self.session_messages[session_id] = []

            session_message = {
                **message,
                "user_id": user_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            self.session_messages[session_id].append(session_message)

        return True

    def get_session_messages(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get messages from a session's shared workspace."""
        with self._lock:
            messages = self.session_messages.get(session_id, [])
            return messages[-limit:]  # Return most recent messages

    def update_shared_document(
        self, session_id: str, document_id: str, user_id: str, updates: dict[str, Any]
    ) -> bool:
        """Update a shared document in the session workspace."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check if user is participant
        if user_id not in self.session_participants.get(session_id, {}):
            return False

        with self._lock:
            if session_id not in self.session_documents:
                self.session_documents[session_id] = {}

            if document_id not in self.session_documents[session_id]:
                self.session_documents[session_id][document_id] = {
                    "id": document_id,
                    "created_by": user_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "content": {},
                    "version": 1,
                    "last_modified": datetime.now(UTC).isoformat(),
                }

            document = self.session_documents[session_id][document_id]
            document["content"].update(updates)
            document["version"] += 1
            document["last_modified"] = datetime.now(UTC).isoformat()
            document["last_modified_by"] = user_id

        return True

    def get_shared_document(
        self, session_id: str, document_id: str
    ) -> dict[str, Any] | None:
        """Get a shared document from the session workspace."""
        with self._lock:
            session_docs = self.session_documents.get(session_id, {})
            return session_docs.get(document_id)

    def get_session_statistics(self) -> dict[str, Any]:
        """Get collaboration system statistics."""
        with self._lock:
            active_sessions = len(
                [s for s in self.sessions.values() if s.status == "active"]
            )
            total_participants = sum(
                len(participants) for participants in self.session_participants.values()
            )

            return {
                "total_sessions": len(self.sessions),
                "active_sessions": active_sessions,
                "total_participants": total_participants,
                "shared_documents": sum(
                    len(docs) for docs in self.session_documents.values()
                ),
                "metrics": self.metrics.to_dict(),
            }

    def _get_session_info(self, session_id: str) -> dict[str, Any]:
        """Get detailed information about a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {}

        participants = self.session_participants.get(session_id, {})
        messages = self.session_messages.get(session_id, [])
        documents = self.session_documents.get(session_id, {})

        return {
            "session_id": session_id,
            "name": session.name,
            "type": session.session_type.value,
            "status": session.status,
            "participant_count": len(participants),
            "message_count": len(messages),
            "document_count": len(documents),
            "created_at": session.created_at.isoformat(),
            "geospatial_context": session.geospatial_context,
        }


@dataclass
class CollaborationMetrics:
    """Metrics for collaboration system performance."""

    sessions_created: int = 0
    sessions_ended: int = 0
    participants_joined: int = 0
    participants_left: int = 0
    messages_shared: int = 0
    documents_shared: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "sessions_created": self.sessions_created,
            "sessions_ended": self.sessions_ended,
            "participants_joined": self.participants_joined,
            "participants_left": self.participants_left,
            "messages_shared": self.messages_shared,
            "documents_shared": self.documents_shared,
            "net_sessions": self.sessions_created - self.sessions_ended,
            "net_participants": self.participants_joined - self.participants_left,
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.sessions_created = 0
        self.sessions_ended = 0
        self.participants_joined = 0
        self.participants_left = 0
        self.messages_shared = 0
        self.documents_shared = 0
        self.start_time = datetime.now(UTC)


class RealTimeCollaborationEngine:
    """
    Real-time collaboration engine for live session coordination.

    Provides real-time features including live cursors, shared editing,
    voice/video integration, and geospatial context synchronization.
    """

    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.live_cursors: dict[
            str, dict[str, Any]
        ] = {}  # session_id -> user_id -> cursor_data
        self.shared_editing: dict[
            str, dict[str, Any]
        ] = {}  # session_id -> document_id -> edit_data
        self.voice_channels: dict[
            str, dict[str, Any]
        ] = {}  # session_id -> voice_channel_data

        self.logger = logging.getLogger(__name__)

    def update_live_cursor(
        self, session_id: str, user_id: str, cursor_data: dict[str, Any]
    ) -> None:
        """Update a user's live cursor position in a session."""
        if session_id not in self.live_cursors:
            self.live_cursors[session_id] = {}

        self.live_cursors[session_id][user_id] = {
            **cursor_data,
            "user_id": user_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.logger.debug(
            f"Live cursor updated for user {user_id} in session {session_id}"
        )

    def get_live_cursors(self, session_id: str) -> dict[str, Any]:
        """Get all live cursors for a session."""
        return self.live_cursors.get(session_id, {})

    def start_shared_editing(
        self, session_id: str, document_id: str, user_id: str
    ) -> bool:
        """Start shared editing session for a document."""
        if session_id not in self.shared_editing:
            self.shared_editing[session_id] = {}

        if document_id not in self.shared_editing[session_id]:
            self.shared_editing[session_id][document_id] = {
                "active_editors": set(),
                "edit_history": [],
                "lock_holder": None,
                "version": 1,
            }

        session_data = self.shared_editing[session_id][document_id]
        session_data["active_editors"].add(user_id)

        return True

    def end_shared_editing(
        self, session_id: str, document_id: str, user_id: str
    ) -> bool:
        """End shared editing session for a document."""
        if (
            session_id in self.shared_editing
            and document_id in self.shared_editing[session_id]
        ):
            session_data = self.shared_editing[session_id][document_id]
            session_data["active_editors"].discard(user_id)

            if not session_data["active_editors"]:
                # Clean up if no active editors
                del self.shared_editing[session_id][document_id]

            return True

        return False

    def get_active_editors(self, session_id: str, document_id: str) -> list[str]:
        """Get list of active editors for a document."""
        if (
            session_id in self.shared_editing
            and document_id in self.shared_editing[session_id]
        ):
            return list(self.shared_editing[session_id][document_id]["active_editors"])

        return []

    def create_voice_channel(
        self, session_id: str, channel_config: dict[str, Any]
    ) -> str:
        """Create a voice channel for a session."""
        channel_id = f"voice_{session_id}_{uuid.uuid4().hex[:8]}"

        self.voice_channels[channel_id] = {
            "session_id": session_id,
            "config": channel_config,
            "participants": set(),
            "created_at": datetime.now(UTC).isoformat(),
        }

        self.logger.info(
            f"Voice channel created: {channel_id} for session {session_id}"
        )
        return channel_id

    def join_voice_channel(self, channel_id: str, user_id: str) -> bool:
        """Join a voice channel."""
        if channel_id not in self.voice_channels:
            return False

        self.voice_channels[channel_id]["participants"].add(user_id)
        return True

    def leave_voice_channel(self, channel_id: str, user_id: str) -> bool:
        """Leave a voice channel."""
        if channel_id not in self.voice_channels:
            return False

        self.voice_channels[channel_id]["participants"].discard(user_id)
        return True


class GeospatialCollaborationCoordinator:
    """
    Geospatial coordination for collaboration sessions.

    Provides geospatial context management, location-based participant
    coordination, and spatial data synchronization for collaborative work.
    """

    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.session_locations: dict[
            str, dict[str, GeospatialMetadata]
        ] = {}  # session_id -> user_id -> location
        self.spatial_workspaces: dict[
            str, dict[str, Any]
        ] = {}  # session_id -> spatial_data

        self.logger = logging.getLogger(__name__)

    def update_participant_location(
        self,
        session_id: str,
        user_id: str,
        location: GeospatialPoint,
        accuracy: float = 10.0,
    ) -> None:
        """Update a participant's location in a collaboration session."""
        geospatial_data = GeospatialMetadata(
            location=location,
            accuracy=accuracy,
            source="collaboration",
            timestamp=datetime.now(UTC),
        )

        if session_id not in self.session_locations:
            self.session_locations[session_id] = {}

        self.session_locations[session_id][user_id] = geospatial_data

        self.logger.debug(
            f"Participant location updated: {user_id} in session {session_id}"
        )

    def get_session_participant_locations(
        self, session_id: str
    ) -> dict[str, GeospatialMetadata]:
        """Get all participant locations for a session."""
        return self.session_locations.get(session_id, {})

    def create_spatial_workspace(
        self, session_id: str, workspace_config: dict[str, Any]
    ) -> str:
        """Create a spatial workspace for collaborative geospatial work."""
        workspace_id = f"spatial_{session_id}_{uuid.uuid4().hex[:8]}"

        self.spatial_workspaces[workspace_id] = {
            "session_id": session_id,
            "config": workspace_config,
            "features": [],
            "annotations": {},
            "created_at": datetime.now(UTC).isoformat(),
        }

        self.logger.info(
            f"Spatial workspace created: {workspace_id} for session {session_id}"
        )
        return workspace_id

    def add_spatial_feature(
        self, workspace_id: str, user_id: str, feature: dict[str, Any]
    ) -> bool:
        """Add a spatial feature to a workspace."""
        if workspace_id not in self.spatial_workspaces:
            return False

        feature_data = {
            **feature,
            "added_by": user_id,
            "added_at": datetime.now(UTC).isoformat(),
        }

        self.spatial_workspaces[workspace_id]["features"].append(feature_data)
        return True

    def add_workspace_annotation(
        self, workspace_id: str, user_id: str, annotation: dict[str, Any]
    ) -> bool:
        """Add an annotation to a spatial workspace."""
        if workspace_id not in self.spatial_workspaces:
            return False

        annotation_id = f"annotation_{uuid.uuid4().hex[:8]}"

        if "annotations" not in self.spatial_workspaces[workspace_id]:
            self.spatial_workspaces[workspace_id]["annotations"] = {}

        self.spatial_workspaces[workspace_id]["annotations"][annotation_id] = {
            **annotation,
            "annotation_id": annotation_id,
            "added_by": user_id,
            "added_at": datetime.now(UTC).isoformat(),
        }

        return True

    def get_workspace_features(self, workspace_id: str) -> list[dict[str, Any]]:
        """Get all features in a spatial workspace."""
        if workspace_id in self.spatial_workspaces:
            return cast(
                list[dict[str, Any]],
                self.spatial_workspaces[workspace_id]["features"],
            )
        return []

    def get_workspace_annotations(self, workspace_id: str) -> dict[str, Any]:
        """Get all annotations in a spatial workspace."""
        if workspace_id in self.spatial_workspaces:
            return cast(
                dict[str, Any],
                self.spatial_workspaces[workspace_id]["annotations"],
            )
        return {}


class CollaborationNotificationManager:
    """
    Notification management for collaboration sessions.

    Handles session-specific notifications, participant alerts,
    and real-time updates for collaborative work.
    """

    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.session_notifications: dict[str, list[dict[str, Any]]] = {}

        self.logger = logging.getLogger(__name__)

    def send_session_notification(
        self, session_id: str, notification: dict[str, Any], sender_id: str
    ) -> bool:
        """Send a notification to all participants in a session."""
        session = self.collaboration_manager.sessions.get(session_id)
        if not session:
            return False

        # Check if sender is participant
        if sender_id not in self.collaboration_manager.session_participants.get(
            session_id, {}
        ):
            return False

        session_notification = {
            **notification,
            "session_id": session_id,
            "sender_id": sender_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if session_id not in self.session_notifications:
            self.session_notifications[session_id] = []

        self.session_notifications[session_id].append(session_notification)

        self.logger.info(f"Session notification sent in {session_id} by {sender_id}")
        return True

    def get_session_notifications(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get notifications for a session."""
        notifications = self.session_notifications.get(session_id, [])
        return notifications[-limit:]  # Return most recent

    def clear_session_notifications(self, session_id: str, user_id: str) -> bool:
        """Clear notifications for a user in a session."""
        # In a real implementation, would mark notifications as read for this user
        self.logger.info(
            f"Notifications cleared for user {user_id} in session {session_id}"
        )
        return True


class CollaborationAnalytics:
    """
    Analytics and monitoring for collaboration sessions.

    Provides insights into collaboration patterns, participant engagement,
    and session effectiveness with geospatial analysis.
    """

    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.session_analytics: dict[str, dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

    def record_session_activity(
        self,
        session_id: str,
        activity_type: str,
        user_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an activity in a collaboration session."""
        if session_id not in self.session_analytics:
            self.session_analytics[session_id] = {
                "activities": [],
                "start_time": datetime.now(UTC).isoformat(),
                "participants": set(),
            }

        activity = {
            "activity_type": activity_type,
            "user_id": user_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": details or {},
        }

        self.session_analytics[session_id]["activities"].append(activity)
        self.session_analytics[session_id]["participants"].add(user_id)

    def get_session_analytics(self, session_id: str) -> dict[str, Any]:
        """Get comprehensive analytics for a session."""
        session_data = self.session_analytics.get(session_id, {})
        if not session_data:
            return {"message": "No analytics data available"}

        activities = session_data["activities"]
        participants = session_data["participants"]

        # Calculate metrics
        activity_counts: dict[str, int] = {}
        user_activity: dict[str, int] = {}
        activity_timeline: dict[str, int] = {}

        for activity in activities:
            # Count by type
            activity_type = activity["activity_type"]
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1

            # Count by user
            user_id = activity["user_id"]
            user_activity[user_id] = user_activity.get(user_id, 0) + 1

            # Timeline by hour
            hour = activity["timestamp"][:13]  # YYYY-MM-DDTHH
            activity_timeline[hour] = activity_timeline.get(hour, 0) + 1

        return {
            "session_id": session_id,
            "total_activities": len(activities),
            "unique_participants": len(participants),
            "activity_types": activity_counts,
            "user_activity": user_activity,
            "activity_timeline": activity_timeline,
            "time_range": {
                "start": session_data["start_time"],
                "end": datetime.now(UTC).isoformat(),
            },
        }

    def get_system_analytics(self) -> dict[str, Any]:
        """Get system-wide collaboration analytics."""
        total_sessions = len(self.session_analytics)
        total_activities = sum(
            len(data["activities"]) for data in self.session_analytics.values()
        )

        return {
            "total_sessions_tracked": total_sessions,
            "total_activities": total_activities,
            "average_activities_per_session": total_activities / max(total_sessions, 1),
        }
