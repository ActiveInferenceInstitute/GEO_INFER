"""
Channel management system for GEO-INFER-COMMS.

This module implements comprehensive channel management including
communication groups, permissions, geospatial bounds, and member management
with support for different channel types and access controls.
"""

from __future__ import annotations
from typing import Any, cast
import threading
import logging
from datetime import datetime, UTC
from dataclasses import dataclass, field

from geo_infer_comms.models.message import (
    ChannelRequest,
    ChannelResponse,
    ChannelType,
    ChannelStatus,
    SubscriptionRequest,
    SubscriptionResponse,
    MessageResponse,
)
from geo_infer_comms.models.spatial import GeospatialBounds, GeospatialPoint
from geo_infer_comms.utils.validation import validate_user_id, validate_spatial_bounds


class ChannelManager:
    """
    Central channel management system.

    Handles channel creation, member management, permissions, and
    geospatial filtering for communication channels.
    """

    def __init__(
        self,
        max_channels: int = 1000,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
    ):
        self.max_channels = max_channels
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Channel storage and management
        self.channels: dict[str, ChannelResponse] = {}
        self.channel_members: dict[str, set[str]] = {}
        self.channel_subscriptions: dict[str, dict[str, SubscriptionResponse]] = {}
        self.channel_permissions: dict[str, dict[str, Any]] = {}

        # Geospatial indexing for channels
        self.spatial_channels: dict[str, list[str]] = {}  # location -> channel_ids

        # Threading and concurrency
        self._lock = threading.RLock()

        # Metrics and monitoring
        self.metrics = ChannelMetrics()

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def create_channel(
        self, request: ChannelRequest, creator_id: str
    ) -> ChannelResponse:
        """
        Create a new communication channel.

        Args:
            request: Channel creation request
            creator_id: ID of the user creating the channel

        Returns:
            Created channel response

        Raises:
            ValueError: If channel request is invalid or limit reached
        """
        # Validate request
        if not validate_user_id(creator_id):
            raise ValueError(f"Invalid creator ID: {creator_id}")

        # Check channel limit
        if len(self.channels) >= self.max_channels:
            raise ValueError(
                f"Maximum number of channels ({self.max_channels}) reached"
            )

        # Create channel response
        channel = ChannelResponse(
            name=request.name,
            description=request.description,
            type=request.type,
            permissions=request.permissions,
            geospatial_bounds=request.geospatial_bounds,
        )

        # Initialize channel data structures
        with self._lock:
            self.channels[channel.channel_id] = channel
            self.channel_members[channel.channel_id] = {creator_id}
            self.channel_subscriptions[channel.channel_id] = {}
            self.channel_permissions[channel.channel_id] = request.permissions

            # Set up default permissions for creator
            self._set_default_permissions(channel.channel_id, creator_id)

            # Add to spatial index if geospatial bounds provided
            if request.geospatial_bounds:
                self._add_channel_to_spatial_index(
                    channel.channel_id, request.geospatial_bounds
                )

        self.metrics.channels_created += 1
        self.logger.info(f"Channel created: {channel.channel_id} by {creator_id}")
        return channel

    def get_channel(self, channel_id: str) -> ChannelResponse | None:
        """
        Retrieve a specific channel by ID.

        Args:
            channel_id: Channel identifier

        Returns:
            Channel if found, None otherwise
        """
        with self._lock:
            return self.channels.get(channel_id)

    def get_channels(
        self,
        channel_type: ChannelType | None = None,
        status: ChannelStatus | None = None,
        creator_id: str | None = None,
        limit: int = 100,
    ) -> list[ChannelResponse]:
        """
        Get channels with filtering options.

        Args:
            channel_type: Filter by channel type
            status: Filter by channel status
            creator_id: Filter by creator
            limit: Maximum number of channels to return

        Returns:
            List of matching channels
        """
        with self._lock:
            channels = list(self.channels.values())

        # Apply filters
        filtered_channels = channels

        if channel_type:
            filtered_channels = [c for c in filtered_channels if c.type == channel_type]

        if status:
            filtered_channels = [c for c in filtered_channels if c.status == status]

        if creator_id:
            # In a real implementation, would need to track creator_id
            # For now, filter by channels where user is a member
            filtered_channels = [
                c
                for c in filtered_channels
                if creator_id in self.channel_members.get(c.channel_id, set())
            ]

        # Sort by creation time (newest first) and limit
        filtered_channels.sort(key=lambda c: c.created_at, reverse=True)
        return filtered_channels[:limit]

    def update_channel(
        self, channel_id: str, updates: dict[str, Any], user_id: str
    ) -> bool:
        """
        Update channel properties.

        Args:
            channel_id: ID of channel to update
            updates: Dictionary of properties to update
            user_id: ID of user making the update

        Returns:
            True if successfully updated
        """
        channel = self.channels.get(channel_id)
        if not channel:
            return False

        # Check permissions
        if not self._check_permission(channel_id, user_id, "update"):
            return False

        with self._lock:
            # Apply updates
            for key, value in updates.items():
                if hasattr(channel, key):
                    setattr(channel, key, value)

            # Update geospatial index if bounds changed
            if "geospatial_bounds" in updates:
                # Remove from old spatial index
                if channel.geospatial_bounds:
                    self._remove_channel_from_spatial_index(
                        channel_id, channel.geospatial_bounds
                    )

                # Add to new spatial index
                if updates["geospatial_bounds"]:
                    self._add_channel_to_spatial_index(
                        channel_id, updates["geospatial_bounds"]
                    )

                channel.geospatial_bounds = updates["geospatial_bounds"]

            channel.updated_at = datetime.now(UTC)

        self.logger.info(f"Channel updated: {channel_id} by {user_id}")
        return True

    def delete_channel(self, channel_id: str, user_id: str) -> bool:
        """
        Delete a channel.

        Args:
            channel_id: ID of channel to delete
            user_id: ID of user deleting the channel

        Returns:
            True if successfully deleted
        """
        channel = self.channels.get(channel_id)
        if not channel:
            return False

        # Check permissions
        if not self._check_permission(channel_id, user_id, "delete"):
            return False

        with self._lock:
            # Remove from all data structures
            del self.channels[channel_id]
            del self.channel_members[channel_id]
            del self.channel_subscriptions[channel_id]
            del self.channel_permissions[channel_id]

            # Remove from spatial index
            if channel.geospatial_bounds:
                self._remove_channel_from_spatial_index(
                    channel_id, channel.geospatial_bounds
                )

        self.metrics.channels_deleted += 1
        self.logger.info(f"Channel deleted: {channel_id} by {user_id}")
        return True

    def add_member(self, channel_id: str, user_id: str, added_by: str) -> bool:
        """
        Add a member to a channel.

        Args:
            channel_id: ID of channel
            user_id: ID of user to add
            added_by: ID of user adding the member

        Returns:
            True if successfully added
        """
        if not self._check_permission(channel_id, added_by, "manage_members"):
            return False

        with self._lock:
            if channel_id not in self.channel_members:
                return False

            self.channel_members[channel_id].add(user_id)

        self.logger.info(
            f"Member added to channel {channel_id}: {user_id} by {added_by}"
        )
        return True

    def remove_member(self, channel_id: str, user_id: str, removed_by: str) -> bool:
        """
        Remove a member from a channel.

        Args:
            channel_id: ID of channel
            user_id: ID of user to remove
            removed_by: ID of user removing the member

        Returns:
            True if successfully removed
        """
        if not self._check_permission(channel_id, removed_by, "manage_members"):
            return False

        with self._lock:
            if channel_id not in self.channel_members:
                return False

            self.channel_members[channel_id].discard(user_id)

        self.logger.info(
            f"Member removed from channel {channel_id}: {user_id} by {removed_by}"
        )
        return True

    def get_members(self, channel_id: str) -> list[str]:
        """
        Get list of members in a channel.

        Args:
            channel_id: ID of channel

        Returns:
            List of member user IDs
        """
        with self._lock:
            return list(self.channel_members.get(channel_id, set()))

    def subscribe_to_channel(
        self, channel_id: str, user_id: str, request: SubscriptionRequest
    ) -> SubscriptionResponse | None:
        """
        Subscribe a user to a channel.

        Args:
            channel_id: ID of channel to subscribe to
            user_id: ID of subscribing user
            request: Subscription preferences

        Returns:
            Subscription response if successful, None otherwise
        """
        channel = self.channels.get(channel_id)
        if not channel:
            return None

        # Check if user is member
        if user_id not in self.channel_members.get(channel_id, set()):
            return None

        subscription = SubscriptionResponse(
            channel_id=channel_id,
            user_id=user_id,
            subscription_type=request.subscription_type,
        )

        with self._lock:
            self.channel_subscriptions[channel_id][user_id] = subscription

        self.logger.info(f"User subscribed to channel {channel_id}: {user_id}")
        return subscription

    def unsubscribe_from_channel(self, channel_id: str, user_id: str) -> bool:
        """
        Unsubscribe a user from a channel.

        Args:
            channel_id: ID of channel to unsubscribe from
            user_id: ID of unsubscribing user

        Returns:
            True if successfully unsubscribed
        """
        with self._lock:
            if channel_id in self.channel_subscriptions:
                if user_id in self.channel_subscriptions[channel_id]:
                    del self.channel_subscriptions[channel_id][user_id]
                    return True

        return False

    def check_permission(self, channel_id: str, user_id: str, permission: str) -> bool:
        """
        Check if a user has a specific permission in a channel.

        Args:
            channel_id: ID of channel
            user_id: ID of user to check
            permission: Permission to check

        Returns:
            True if user has permission
        """
        return self._check_permission(channel_id, user_id, permission)

    def set_permissions(
        self, channel_id: str, user_id: str, permissions: dict[str, Any], set_by: str
    ) -> bool:
        """
        Set permissions for a user in a channel.

        Args:
            channel_id: ID of channel
            user_id: ID of user to set permissions for
            permissions: Permission settings
            set_by: ID of user setting permissions

        Returns:
            True if successfully set
        """
        if not self._check_permission(channel_id, set_by, "manage_permissions"):
            return False

        with self._lock:
            if channel_id not in self.channel_permissions:
                return False

            if user_id not in self.channel_permissions[channel_id]:
                self.channel_permissions[channel_id][user_id] = {}

            self.channel_permissions[channel_id][user_id].update(permissions)

        self.logger.info(
            f"Permissions set for user {user_id} in channel {channel_id} by {set_by}"
        )
        return True

    def get_channels_by_location(
        self, location: GeospatialPoint, radius_km: float = 1.0
    ) -> list[ChannelResponse]:
        """
        Find channels near a specific location.

        Args:
            location: Reference location
            radius_km: Search radius in kilometers

        Returns:
            List of nearby channels
        """
        nearby_channels = []

        # Convert radius to approximate degrees (rough approximation)
        radius_degrees = radius_km / 111.0  # 1 degree ≈ 111 km at equator

        min_lon = location.longitude - radius_degrees
        max_lon = location.longitude + radius_degrees
        min_lat = location.latitude - radius_degrees
        max_lat = location.latitude + radius_degrees

        # Check channels within approximate bounds
        for channel in self.channels.values():
            if not channel.geospatial_bounds:
                continue
            try:
                bounds = GeospatialBounds.from_dict(channel.geospatial_bounds)
            except (KeyError, TypeError, ValueError):
                self.logger.warning(
                    f"Channel {channel.channel_id} has invalid geospatial bounds; "
                    "skipping in location query"
                )
                continue
            if (
                bounds.min_longitude <= max_lon
                and bounds.max_longitude >= min_lon
                and bounds.min_latitude <= max_lat
                and bounds.max_latitude >= min_lat
            ):
                nearby_channels.append(channel)

        return nearby_channels

    def get_channel_statistics(self) -> dict[str, Any]:
        """Get channel system statistics."""
        with self._lock:
            total_members = sum(
                len(members) for members in self.channel_members.values()
            )
            total_subscriptions = sum(
                len(subs) for subs in self.channel_subscriptions.values()
            )

            return {
                "total_channels": len(self.channels),
                "total_members": total_members,
                "total_subscriptions": total_subscriptions,
                "spatial_channels": len(
                    [c for c in self.channels.values() if c.geospatial_bounds]
                ),
                "metrics": self.metrics.to_dict(),
            }

    def _check_permission(self, channel_id: str, user_id: str, permission: str) -> bool:
        """Check if user has specific permission in channel."""
        with self._lock:
            if channel_id not in self.channel_permissions:
                return False

            channel_perms = self.channel_permissions[channel_id]

            # Check user-specific permissions
            user_perms = channel_perms.get(user_id, {})

            # Check if user is member (basic permission)
            if permission == "read" and user_id in self.channel_members.get(
                channel_id, set()
            ):
                return True

            # Check explicit permissions
            if permission in user_perms:
                return cast(bool, user_perms[permission])

            # Check role-based permissions
            user_role = user_perms.get("role", "member")
            role_perms = channel_perms.get(f"role_{user_role}", {})

            return cast(bool, role_perms.get(permission, False))

    def _set_default_permissions(self, channel_id: str, user_id: str) -> None:
        """Set default permissions for channel creator."""
        default_permissions = {
            "read": True,
            "write": True,
            "manage_members": True,
            "manage_permissions": True,
            "delete": True,
            "role": "admin",
        }

        if channel_id not in self.channel_permissions:
            self.channel_permissions[channel_id] = {}

        self.channel_permissions[channel_id][user_id] = default_permissions

    def _add_channel_to_spatial_index(
        self, channel_id: str, bounds: dict[str, Any]
    ) -> None:
        """Add channel to spatial index for location-based queries."""
        # Simplified spatial indexing - in production would use proper spatial index
        if validate_spatial_bounds(bounds):
            # Create a key from bounds center
            center_lon = (bounds["min_longitude"] + bounds["max_longitude"]) / 2
            center_lat = (bounds["min_latitude"] + bounds["max_latitude"]) / 2
            key = f"{center_lon:.3f},{center_lat:.3f}"

            if key not in self.spatial_channels:
                self.spatial_channels[key] = []

            if channel_id not in self.spatial_channels[key]:
                self.spatial_channels[key].append(channel_id)

    def _remove_channel_from_spatial_index(
        self, channel_id: str, bounds: dict[str, Any]
    ) -> None:
        """Remove channel from spatial index."""
        center_lon = (bounds["min_longitude"] + bounds["max_longitude"]) / 2
        center_lat = (bounds["min_latitude"] + bounds["max_latitude"]) / 2
        key = f"{center_lon:.3f},{center_lat:.3f}"

        if key in self.spatial_channels:
            try:
                self.spatial_channels[key].remove(channel_id)
                if not self.spatial_channels[key]:
                    del self.spatial_channels[key]
            except ValueError:
                pass


@dataclass
class ChannelMetrics:
    """Metrics for channel system performance."""

    channels_created: int = 0
    channels_deleted: int = 0
    members_added: int = 0
    members_removed: int = 0
    subscriptions_created: int = 0
    subscriptions_removed: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "channels_created": self.channels_created,
            "channels_deleted": self.channels_deleted,
            "members_added": self.members_added,
            "members_removed": self.members_removed,
            "subscriptions_created": self.subscriptions_created,
            "subscriptions_removed": self.subscriptions_removed,
            "net_channels": self.channels_created - self.channels_deleted,
            "net_members": self.members_added - self.members_removed,
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.channels_created = 0
        self.channels_deleted = 0
        self.members_added = 0
        self.members_removed = 0
        self.subscriptions_created = 0
        self.subscriptions_removed = 0
        self.start_time = datetime.now(UTC)


class ChannelPermissionManager:
    """
    Advanced permission management for channels.

    Provides sophisticated permission systems with role-based access control,
    geospatial restrictions, and dynamic permission evaluation.
    """

    def __init__(self, channel_manager: ChannelManager):
        self.channel_manager = channel_manager
        self.permission_templates: dict[str, dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

        # Register default permission templates
        self._register_default_templates()

    def create_permission_template(
        self, template_name: str, permissions: dict[str, Any], description: str = ""
    ) -> None:
        """Create a reusable permission template."""
        self.permission_templates[template_name] = {
            "permissions": permissions,
            "description": description,
            "created_at": datetime.now(UTC),
        }
        self.logger.info(f"Created permission template: {template_name}")

    def apply_permission_template(
        self, channel_id: str, template_name: str, user_id: str, applied_by: str
    ) -> bool:
        """Apply a permission template to a user in a channel."""
        if template_name not in self.permission_templates:
            return False

        template = self.permission_templates[template_name]
        return self.channel_manager.set_permissions(
            channel_id, user_id, template["permissions"], applied_by
        )

    def check_geospatial_permission(
        self, channel_id: str, user_id: str, permission: str, location: GeospatialPoint
    ) -> bool:
        """Check if user has permission at a specific location."""
        channel = self.channel_manager.get_channel(channel_id)
        if not channel or not channel.geospatial_bounds:
            return self.channel_manager.check_permission(
                channel_id, user_id, permission
            )

        # Check if location is within channel bounds
        try:
            bounds = GeospatialBounds.from_dict(channel.geospatial_bounds)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Channel {channel_id} has invalid geospatial bounds: {e}"
            ) from e
        if not bounds.contains_point(location):
            return False

        return self.channel_manager.check_permission(channel_id, user_id, permission)

    def get_effective_permissions(
        self, channel_id: str, user_id: str
    ) -> dict[str, Any]:
        """Get all effective permissions for a user in a channel."""
        with self.channel_manager._lock:
            if channel_id not in self.channel_manager.channel_permissions:
                return {}

            channel_perms = self.channel_manager.channel_permissions[channel_id]
            user_perms = channel_perms.get(user_id, {})

            # Merge with role permissions
            user_role = user_perms.get("role", "member")
            role_perms = channel_perms.get(f"role_{user_role}", {})

            effective = cast(dict[str, Any], role_perms.copy())
            effective.update(cast(dict[str, Any], user_perms))

            return effective

    def _register_default_templates(self) -> None:
        """Register default permission templates."""
        self.permission_templates.update(
            {
                "read_only": {
                    "permissions": {"read": True, "role": "viewer"},
                    "description": "Read-only access to channel",
                },
                "member": {
                    "permissions": {"read": True, "write": True, "role": "member"},
                    "description": "Standard member access",
                },
                "moderator": {
                    "permissions": {
                        "read": True,
                        "write": True,
                        "manage_members": True,
                        "role": "moderator",
                    },
                    "description": "Moderator access with member management",
                },
                "admin": {
                    "permissions": {
                        "read": True,
                        "write": True,
                        "manage_members": True,
                        "manage_permissions": True,
                        "delete": True,
                        "role": "admin",
                    },
                    "description": "Full administrative access",
                },
            }
        )


class ChannelMessageFilter:
    """
    Advanced message filtering for channels.

    Provides sophisticated filtering capabilities including content filtering,
    geospatial filtering, and user-based filtering for channel messages.
    """

    def __init__(self, channel_manager: ChannelManager):
        self.channel_manager = channel_manager
        self.content_filters: dict[str, list[dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__)

    def add_content_filter(
        self, channel_id: str, filter_rule: dict[str, Any], added_by: str
    ) -> bool:
        """Add a content filter rule to a channel."""
        if not self.channel_manager.check_permission(
            channel_id, added_by, "manage_permissions"
        ):
            return False

        if channel_id not in self.content_filters:
            self.content_filters[channel_id] = []

        self.content_filters[channel_id].append(
            {
                **filter_rule,
                "added_by": added_by,
                "added_at": datetime.now(UTC),
            }
        )

        self.logger.info(f"Content filter added to channel {channel_id} by {added_by}")
        return True

    def filter_message(self, message: MessageResponse, channel_id: str) -> bool:
        """
        Check if message passes channel filters.

        Args:
            message: Message to filter
            channel_id: ID of channel

        Returns:
            True if message passes all filters
        """
        filters = self.content_filters.get(channel_id, [])

        for filter_rule in filters:
            if not self._evaluate_filter(message, filter_rule):
                return False

        return True

    def _evaluate_filter(
        self, message: MessageResponse, filter_rule: dict[str, Any]
    ) -> bool:
        """Evaluate a single filter rule against a message."""
        rule_type = filter_rule.get("type", "keyword")

        if rule_type == "keyword":
            content_lower = message.content.lower()

            # Check for blocked keywords
            blocked_keywords = filter_rule.get("blocked_keywords", [])
            for keyword in blocked_keywords:
                if keyword.lower() in content_lower:
                    return False

            # Check for required keywords (if specified)
            required_keywords = filter_rule.get("required_keywords", [])
            if required_keywords:
                if not any(
                    keyword.lower() in content_lower for keyword in required_keywords
                ):
                    return False

        elif rule_type == "length":
            min_length = filter_rule.get("min_length")
            max_length = filter_rule.get("max_length")

            if min_length and len(message.content) < min_length:
                return False

            if max_length and len(message.content) > max_length:
                return False

        elif rule_type == "geospatial":
            if not message.geospatial_data:
                # If geospatial filter requires location data, reject
                return cast(bool, filter_rule.get("require_location", False))

            # Check geospatial constraints
            geo_filter = filter_rule.get("spatial_filter")
            if geo_filter:
                msg_loc = message.geospatial_data.location
                bounds = geo_filter.get("bounds", {})
                if bounds:
                    lat_ok = (
                        bounds.get("min_lat", -90)
                        <= msg_loc.latitude
                        <= bounds.get("max_lat", 90)
                    )
                    lon_ok = (
                        bounds.get("min_lon", -180)
                        <= msg_loc.longitude
                        <= bounds.get("max_lon", 180)
                    )
                    if not (lat_ok and lon_ok):
                        return False

        return True


class ChannelAnalytics:
    """
    Analytics and monitoring for channel activity.

    Provides insights into channel usage, member engagement, and
    communication patterns with geospatial analysis.
    """

    def __init__(self, channel_manager: ChannelManager):
        self.channel_manager = channel_manager
        self.activity_log: list[dict[str, Any]] = []
        self.max_log_entries = 10000

        self.logger = logging.getLogger(__name__)

    def log_activity(
        self,
        channel_id: str,
        activity_type: str,
        user_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log channel activity for analytics."""
        activity_entry = {
            "channel_id": channel_id,
            "activity_type": activity_type,
            "user_id": user_id,
            "timestamp": datetime.now(UTC),
            "details": details or {},
        }

        self.activity_log.append(activity_entry)

        # Trim log if too large
        if len(self.activity_log) > self.max_log_entries:
            self.activity_log = self.activity_log[-self.max_log_entries :]

    def get_channel_activity(
        self,
        channel_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        activity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get activity log for a specific channel."""
        filtered = [
            entry for entry in self.activity_log if entry["channel_id"] == channel_id
        ]

        if start_time:
            filtered = [e for e in filtered if e["timestamp"] >= start_time]

        if end_time:
            filtered = [e for e in filtered if e["timestamp"] <= end_time]

        if activity_types:
            filtered = [e for e in filtered if e["activity_type"] in activity_types]

        return filtered

    def get_channel_analytics(self, channel_id: str) -> dict[str, Any]:
        """Get comprehensive analytics for a channel."""
        activities = self.get_channel_activity(channel_id)

        if not activities:
            return {"message": "No activity data available"}

        # Calculate basic metrics
        activity_counts: dict[str, int] = {}
        user_activity: dict[str, int] = {}
        hourly_activity: dict[int, int] = {}

        for activity in activities:
            # Count by type
            activity_type = activity["activity_type"]
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1

            # Count by user
            user_id = activity["user_id"]
            user_activity[user_id] = user_activity.get(user_id, 0) + 1

            # Count by hour
            hour = activity["timestamp"].hour
            hourly_activity[hour] = hourly_activity.get(hour, 0) + 1

        return {
            "total_activities": len(activities),
            "activity_types": activity_counts,
            "unique_users": len(user_activity),
            "most_active_users": sorted(
                user_activity.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "hourly_distribution": hourly_activity,
            "time_range": {
                "start": min(a["timestamp"] for a in activities),
                "end": max(a["timestamp"] for a in activities),
            },
        }

    def get_system_analytics(self) -> dict[str, Any]:
        """Get system-wide channel analytics."""
        if not self.activity_log:
            return {"message": "No activity data available"}

        # System-wide metrics
        channel_activity: dict[str, int] = {}
        total_activities = len(self.activity_log)

        for entry in self.activity_log:
            channel_id = entry["channel_id"]
            channel_activity[channel_id] = channel_activity.get(channel_id, 0) + 1

        return {
            "total_activities": total_activities,
            "total_channels_with_activity": len(channel_activity),
            "most_active_channels": sorted(
                channel_activity.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "time_range": {
                "start": min(e["timestamp"] for e in self.activity_log),
                "end": max(e["timestamp"] for e in self.activity_log),
            },
        }
