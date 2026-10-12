"""
Notification and alert system for GEO-INFER-COMMS.

This module implements comprehensive notification and alert functionality
with geospatial filtering, multi-channel delivery, and intelligent routing.
"""

from __future__ import annotations
from typing import Any, Literal, cast
from collections.abc import Callable
import logging
import threading
import time
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
import uuid

from geo_infer_comms.models.message import (
    NotificationRequest,
    NotificationResponse,
    NotificationStatus,
    NotificationType,
    MessagePriority,
)
from geo_infer_comms.models.spatial import (
    GeospatialMetadata,
    SpatialFilter,
    GeospatialPoint,
    GeospatialBounds,
)
from geo_infer_comms.utils.validation import (
    validate_notification_type,
    validate_delivery_methods,
    validate_email,
    validate_phone,
    validate_spatial_filter,
)


class NotificationManager:
    """
    Central notification management system.

    Handles notification creation, scheduling, delivery, and tracking
    with support for geospatial filtering and multi-channel delivery.
    """

    def __init__(
        self,
        max_notifications: int = 10000,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
    ):
        self.max_notifications = max_notifications
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Notification storage and tracking
        self.notifications: dict[str, NotificationResponse] = {}
        self.notification_queue: list[NotificationResponse] = []
        self.delivery_handlers: dict[str, Callable] = {}

        # Geospatial filtering
        self.spatial_filters: dict[str, SpatialFilter] = {}

        # Scheduling and threading
        self._scheduler_thread: threading.Thread | None = None
        self._delivery_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.RLock()

        # Metrics and monitoring
        self.metrics = NotificationMetrics()

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Register default delivery handlers
        self._register_default_handlers()

    def start(self) -> None:
        """Start the notification manager."""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Start scheduler thread
            self._scheduler_thread = threading.Thread(
                target=self._process_scheduled_notifications, daemon=True
            )
            self._scheduler_thread.start()

            # Start delivery thread
            self._delivery_thread = threading.Thread(
                target=self._process_notification_delivery, daemon=True
            )
            self._delivery_thread.start()

            self.logger.info("Notification manager started")

    def stop(self) -> None:
        """Stop the notification manager."""
        with self._lock:
            self._running = False
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5.0)
            if self._delivery_thread:
                self._delivery_thread.join(timeout=5.0)
            self.logger.info("Notification manager stopped")

    def create_notification(self, request: NotificationRequest) -> NotificationResponse:
        """
        Create a new notification.

        Args:
            request: Notification creation request

        Returns:
            Created notification response

        Raises:
            ValueError: If notification request is invalid
        """
        if not self._running:
            raise RuntimeError("Notification manager is not running")

        # Validate request
        if not validate_notification_type(request.notification_type):
            raise ValueError(f"Invalid notification type: {request.notification_type}")

        if not validate_delivery_methods(request.delivery_method):
            raise ValueError(f"Invalid delivery methods: {request.delivery_method}")

        # Create notification response
        notification = NotificationResponse(
            title=request.title,
            content=request.content,
            notification_type=request.notification_type,
            priority=request.priority,
            delivery_methods=cast(list[str], request.delivery_method),
            recipients=request.recipients,
            geospatial_context=request.geospatial_context,
        )

        # Store notification
        with self._lock:
            self.notifications[notification.notification_id] = notification

            # Add to queue for processing
            self.notification_queue.append(notification)

        self.metrics.notifications_created += 1
        self.logger.info(f"Notification created: {notification.notification_id}")
        return notification

    def send_notification(self, notification_id: str) -> bool:
        """
        Send a specific notification immediately.

        Args:
            notification_id: ID of notification to send

        Returns:
            True if sent successfully, False otherwise
        """
        notification = self.notifications.get(notification_id)
        if not notification:
            return False

        with self._lock:
            if notification.status != NotificationStatus.PENDING:
                return False

            # Claim the notification under the lock so concurrent scheduler and
            # delivery threads cannot double-send it.
            notification.status = NotificationStatus.SENT
            notification.created_at = datetime.now(UTC)

        try:
            # Deliver notification
            success = self._deliver_notification(notification)

            if success:
                self.metrics.notifications_sent += 1
            else:
                self.metrics.delivery_failures += 1

            return success

        except Exception as e:
            self.logger.error(f"Failed to send notification {notification_id}: {e}")
            notification.status = NotificationStatus.EXPIRED
            self.metrics.delivery_failures += 1
            return False

    def schedule_notification(
        self, request: NotificationRequest, schedule_time: datetime
    ) -> str:
        """
        Schedule a notification for future delivery.

        Args:
            request: Notification request
            schedule_time: When to send the notification

        Returns:
            Notification ID for tracking
        """
        notification = self.create_notification(request)

        # Keep the notification PENDING but withhold delivery until the
        # scheduled time: the delivery thread skips notifications whose
        # schedule_time is still in the future.
        notification.schedule_time = schedule_time
        notification.status = NotificationStatus.PENDING
        self.metrics.scheduled_notifications += 1

        self.logger.info(
            f"Notification scheduled: {notification.notification_id} at {schedule_time}"
        )
        return notification.notification_id

    def mark_as_read(self, notification_id: str, user_id: str) -> bool:
        """
        Mark a notification as read by a user.

        Args:
            notification_id: ID of notification
            user_id: ID of user marking as read

        Returns:
            True if successfully marked as read
        """
        notification = self.notifications.get(notification_id)
        if not notification:
            return False

        if notification.status == NotificationStatus.READ:
            return True

        notification.status = NotificationStatus.READ
        notification.read_at = datetime.now(UTC)

        self.metrics.notifications_read += 1
        self.logger.info(f"Notification marked as read: {notification_id} by {user_id}")
        return True

    def get_notifications(
        self,
        user_id: str | None = None,
        status: NotificationStatus | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[NotificationResponse]:
        """
        Get notifications with filtering.

        Args:
            user_id: Filter by user (if notifications are user-specific)
            status: Filter by status
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number to return

        Returns:
            List of matching notifications
        """
        with self._lock:
            notifications = list(self.notifications.values())

        # Apply filters
        filtered = notifications

        if user_id:
            filtered = [n for n in filtered if user_id in n.recipients]

        if status:
            filtered = [n for n in filtered if n.status == status]

        if start_time:
            filtered = [n for n in filtered if n.created_at >= start_time]

        if end_time:
            filtered = [n for n in filtered if n.created_at <= end_time]

        # Sort by creation time (newest first) and limit
        filtered.sort(key=lambda n: n.created_at, reverse=True)
        return filtered[:limit]

    def add_spatial_filter(self, filter_id: str, spatial_filter: SpatialFilter) -> None:
        """Add a spatial filter for notification targeting."""
        if not validate_spatial_filter(spatial_filter.to_dict()):
            raise ValueError("Invalid spatial filter")

        self.spatial_filters[filter_id] = spatial_filter
        self.logger.info(f"Added spatial filter: {filter_id}")

    def remove_spatial_filter(self, filter_id: str) -> bool:
        """Remove a spatial filter."""
        if filter_id in self.spatial_filters:
            del self.spatial_filters[filter_id]
            self.logger.info(f"Removed spatial filter: {filter_id}")
            return True
        return False

    def register_delivery_handler(self, method: str, handler: Callable) -> None:
        """Register a custom delivery handler for a notification method."""
        self.delivery_handlers[method] = handler
        self.logger.info(f"Registered delivery handler for method: {method}")

    def get_metrics(self) -> dict[str, Any]:
        """Get notification system metrics."""
        return {
            "notifications_stored": len(self.notifications),
            "queue_size": len(self.notification_queue),
            "spatial_filters": len(self.spatial_filters),
            "delivery_handlers": len(self.delivery_handlers),
            "metrics": self.metrics.to_dict(),
        }

    def _process_scheduled_notifications(self) -> None:
        """Background thread to process scheduled notifications."""
        while self._running:
            try:
                current_time = datetime.now(UTC)

                with self._lock:
                    # Check for notifications ready to send
                    ready_ids = [
                        n.notification_id
                        for n in self.notification_queue
                        if n.status == NotificationStatus.PENDING
                        and n.schedule_time
                        and n.schedule_time <= current_time
                    ]

                for notification_id in ready_ids:
                    self.send_notification(notification_id)
                    self._dequeue_notification(notification_id)

                # Brief pause before next check
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error processing scheduled notifications: {e}")
                time.sleep(5.0)  # Longer pause on error

    def _process_notification_delivery(self) -> None:
        """Background thread to process notification delivery."""
        while self._running:
            try:
                current_time = datetime.now(UTC)

                with self._lock:
                    # Process pending notifications that are due (or were never
                    # scheduled); future schedule_time values are withheld
                    # until the scheduler thread picks them up.
                    pending_ids = [
                        n.notification_id
                        for n in self.notification_queue
                        if n.status == NotificationStatus.PENDING
                        and (n.schedule_time is None or n.schedule_time <= current_time)
                    ]

                for notification_id in pending_ids:
                    if self.send_notification(notification_id):
                        self._dequeue_notification(notification_id)

                # Brief pause if no work
                if not pending_ids:
                    time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error processing notification delivery: {e}")
                time.sleep(5.0)  # Longer pause on error

    def _dequeue_notification(self, notification_id: str) -> None:
        """Remove a notification from the processing queue under the lock."""
        with self._lock:
            self.notification_queue = [
                n
                for n in self.notification_queue
                if n.notification_id != notification_id
            ]

    def _deliver_notification(self, notification: NotificationResponse) -> bool:
        """Deliver notification via configured methods."""
        success = True

        for method in notification.delivery_methods:
            try:
                if method == "in_app":
                    success &= self._deliver_in_app(notification)
                elif method == "email":
                    success &= self._deliver_email(notification)
                elif method == "sms":
                    success &= self._deliver_sms(notification)
                elif method == "push":
                    success &= self._deliver_push(notification)
                else:
                    # Try custom handler
                    handler = self.delivery_handlers.get(method)
                    if handler:
                        success &= handler(notification)
                    else:
                        self.logger.warning(f"No handler for delivery method: {method}")
                        success = False

            except Exception as e:
                self.logger.error(f"Failed to deliver notification via {method}: {e}")
                success = False

        return success

    def _deliver_in_app(self, notification: NotificationResponse) -> bool:
        """Deliver notification in-app."""
        # In a real implementation, would send to user's active session
        # For now, just log
        self.logger.info(f"In-app notification: {notification.title}")
        return True

    def _deliver_email(self, notification: NotificationResponse) -> bool:
        """Deliver notification via email."""
        try:
            # Validate addresses, format content, then deliver via configured service
            # (SMTP, SES, etc. — currently logs intended delivery for integration)
            recipients = notification.recipients
            valid_emails = [email for email in recipients if validate_email(email)]

            if not valid_emails:
                return False

            # Create email content
            subject = f"GEO-INFER Notification: {notification.title}"

            # In production, would send actual email
            self.logger.info(f"Email notification sent to {valid_emails}: {subject}")

            return True

        except Exception as e:
            self.logger.error(f"Email delivery failed: {e}")
            return False

    def _deliver_sms(self, notification: NotificationResponse) -> bool:
        """Deliver notification via SMS."""
        try:
            # Validate phone numbers, format for SMS, then deliver via gateway
            # (Twilio, SNS, etc. — currently logs intended delivery for integration)
            recipients = notification.recipients
            valid_phones = [phone for phone in recipients if validate_phone(phone)]

            if not valid_phones:
                return False

            # Format message for SMS
            sms_content = f"{notification.title}: {notification.content}"
            if len(sms_content) > 160:
                sms_content = sms_content[:157] + "..."

            # In production, would send actual SMS
            self.logger.info(f"SMS notification sent to {valid_phones}: {sms_content}")

            return True

        except Exception as e:
            self.logger.error(f"SMS delivery failed: {e}")
            return False

    def _deliver_push(self, notification: NotificationResponse) -> bool:
        """Deliver notification via push notification."""
        try:
            # Deliver via push service (FCM, APNS — currently logs for integration)
            self.logger.info(f"Push notification: {notification.title}")
            return True

        except Exception as e:
            self.logger.error(f"Push delivery failed: {e}")
            return False

    def _register_default_handlers(self) -> None:
        """Register default delivery handlers."""
        self.delivery_handlers.update(
            {
                "in_app": self._deliver_in_app,
                "email": self._deliver_email,
                "sms": self._deliver_sms,
                "push": self._deliver_push,
            }
        )


@dataclass
class NotificationMetrics:
    """Metrics for notification system performance."""

    notifications_created: int = 0
    notifications_sent: int = 0
    notifications_read: int = 0
    delivery_failures: int = 0
    scheduled_notifications: int = 0
    spatial_filters_used: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "notifications_created": self.notifications_created,
            "notifications_sent": self.notifications_sent,
            "notifications_read": self.notifications_read,
            "delivery_failures": self.delivery_failures,
            "delivery_success_rate": (
                self.notifications_sent / max(self.notifications_created, 1) * 100
            ),
            "scheduled_notifications": self.scheduled_notifications,
            "spatial_filters_used": self.spatial_filters_used,
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.notifications_created = 0
        self.notifications_sent = 0
        self.notifications_read = 0
        self.delivery_failures = 0
        self.scheduled_notifications = 0
        self.spatial_filters_used = 0
        self.start_time = datetime.now(UTC)


class AlertSystem:
    """
    Advanced alert system for critical notifications.

    Provides sophisticated alert management with geospatial triggering,
    escalation policies, and multi-channel delivery.
    """

    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.alert_rules: dict[str, AlertRule] = {}
        self.alert_history: list[AlertResponse] = []
        self.max_history = 1000

        self.logger = logging.getLogger(__name__)

    def create_alert_rule(self, rule: AlertRule) -> str:
        """Create a new alert rule."""
        rule_id = f"alert_rule_{uuid.uuid4().hex[:8]}"
        rule.rule_id = rule_id
        self.alert_rules[rule_id] = rule

        self.logger.info(f"Created alert rule: {rule_id}")
        return rule_id

    def trigger_alert(
        self,
        rule_id: str,
        trigger_data: dict[str, Any],
        geospatial_context: GeospatialMetadata | None = None,
    ) -> AlertResponse | None:
        """Trigger an alert based on a rule."""

        rule = self.alert_rules.get(rule_id)
        if not rule:
            self.logger.warning(f"Alert rule not found: {rule_id}")
            return None

        # Check if rule conditions are met
        if not rule.evaluate_conditions(trigger_data):
            return None

        # Create alert notification
        notification_request = NotificationRequest(
            title=rule.alert_title,
            content=rule.alert_content,
            recipients=rule.recipients,
            notification_type=NotificationType.WARNING,
            priority=rule.priority,
            delivery_method=cast(
                "list[Literal['in_app', 'email', 'sms', 'push']]",
                rule.delivery_methods,
            ),
            geospatial_context=geospatial_context.to_dict()
            if geospatial_context
            else None,
        )

        notification = self.notification_manager.create_notification(
            notification_request
        )

        # Create alert response
        alert_response = AlertResponse(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            rule_id=rule_id,
            notification_id=notification.notification_id,
            trigger_data=trigger_data,
            geospatial_context=geospatial_context,
            created_at=datetime.now(UTC),
        )

        # Store in history
        self.alert_history.append(alert_response)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

        self.logger.info(f"Alert triggered: {alert_response.alert_id}")
        rule.update_last_triggered()
        return alert_response

    def get_alert_history(
        self,
        rule_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AlertResponse]:
        """Get alert history with filtering."""
        filtered = self.alert_history

        if rule_id:
            filtered = [a for a in filtered if a.rule_id == rule_id]

        if start_time:
            filtered = [a for a in filtered if a.created_at >= start_time]

        if end_time:
            filtered = [a for a in filtered if a.created_at <= end_time]

        # Sort by creation time (newest first) and limit
        filtered.sort(key=lambda a: a.created_at, reverse=True)
        return filtered[:limit]

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert system statistics."""
        return {
            "total_rules": len(self.alert_rules),
            "total_alerts": len(self.alert_history),
            "rules": list(self.alert_rules.keys()),
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts based on conditions."""

    name: str
    description: str
    conditions: dict[str, Any]
    alert_title: str
    alert_content: str
    recipients: list[str]
    priority: MessagePriority = MessagePriority.HIGH
    delivery_methods: list[str] = field(default_factory=lambda: ["in_app", "email"])
    escalation_policy: dict[str, Any] | None = None
    cooldown_period: int = 300  # seconds
    rule_id: str | None = None
    enabled: bool = True
    last_triggered: datetime | None = None

    def evaluate_conditions(self, trigger_data: dict[str, Any]) -> bool:
        """Evaluate if alert conditions are met."""
        if not self.enabled:
            return False

        # Simple condition evaluation - in production would be more sophisticated
        for condition_key, condition_value in self.conditions.items():
            if condition_key not in trigger_data:
                return False

            trigger_value = trigger_data[condition_key]

            # Simple comparison - in production would support complex expressions
            if isinstance(condition_value, (int, float)):
                if trigger_value != condition_value:
                    return False
            elif isinstance(condition_value, dict):
                # Range check
                if "min" in condition_value and trigger_value < condition_value["min"]:
                    return False
                if "max" in condition_value and trigger_value > condition_value["max"]:
                    return False
                if (
                    "equals" in condition_value
                    and trigger_value != condition_value["equals"]
                ):
                    return False

        # Check cooldown period
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(seconds=self.cooldown_period)
            if datetime.now(UTC) < cooldown_end:
                return False

        return True

    def update_last_triggered(self) -> None:
        """Update the last triggered timestamp."""
        self.last_triggered = datetime.now(UTC)


@dataclass
class AlertResponse:
    """Response from alert triggering."""

    alert_id: str
    rule_id: str
    notification_id: str
    trigger_data: dict[str, Any]
    geospatial_context: GeospatialMetadata | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "notification_id": self.notification_id,
            "trigger_data": self.trigger_data,
            "created_at": self.created_at.isoformat(),
        }
        if self.geospatial_context:
            data["geospatial_context"] = self.geospatial_context.to_dict()
        return data


class NotificationFormatter:
    """Format notifications for different delivery methods and contexts."""

    @staticmethod
    def format_for_sms(
        notification: NotificationResponse, max_length: int = 160
    ) -> str:
        """Format notification for SMS delivery."""
        content = f"{notification.title}: {notification.content}"
        if len(content) > max_length:
            content = content[: max_length - 3] + "..."
        return content

    @staticmethod
    def format_for_email(notification: NotificationResponse) -> dict[str, str]:
        """Format notification for email delivery."""
        return {
            "subject": f"GEO-INFER Alert: {notification.title}",
            "body": f"""
            URGENT NOTIFICATION

            Title: {notification.title}
            Type: {notification.notification_type.value}
            Priority: {notification.priority.value}

            Content:
            {notification.content}

            {f"Geospatial Context: {notification.geospatial_context}" if notification.geospatial_context else ""}

            Sent at: {notification.created_at.isoformat()}
            """,
        }

    @staticmethod
    def format_for_push_notification(
        notification: NotificationResponse,
    ) -> dict[str, str]:
        """Format notification for push notification."""
        title = notification.title
        body = notification.content
        if len(body) > 100:
            body = body[:97] + "..."

        return {
            "title": title,
            "body": body,
            "priority": notification.priority.value,
            "type": notification.notification_type.value,
        }

    @staticmethod
    def format_for_geospatial_context(
        notification: NotificationResponse,
    ) -> dict[str, Any]:
        """Format notification with geospatial context information."""
        formatted: dict[str, Any] = {
            "notification_id": notification.notification_id,
            "title": notification.title,
            "content": notification.content,
            "type": notification.notification_type.value,
            "priority": notification.priority.value,
            "timestamp": notification.created_at.isoformat(),
        }

        if notification.geospatial_context:
            formatted["geospatial_context"] = notification.geospatial_context

        return formatted


class EmergencyAlertSystem:
    """
    Specialized system for emergency alerts and critical notifications.

    Provides high-priority alert management with immediate delivery
    guarantees and escalation procedures.
    """

    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.emergency_contacts: dict[str, dict[str, Any]] = {}
        self.emergency_zones: dict[str, GeospatialBounds] = {}
        self.active_emergencies: dict[str, EmergencyAlert] = {}

        self.logger = logging.getLogger(__name__)

    def register_emergency_contact(
        self, contact_id: str, contact_info: dict[str, Any], priority: int = 1
    ) -> None:
        """Register an emergency contact."""
        self.emergency_contacts[contact_id] = {
            **contact_info,
            "priority": priority,
            "registered_at": datetime.now(UTC),
        }
        self.logger.info(f"Registered emergency contact: {contact_id}")

    def define_emergency_zone(self, zone_id: str, bounds: GeospatialBounds) -> None:
        """Define an emergency zone for spatial alerting."""
        self.emergency_zones[zone_id] = bounds
        self.logger.info(f"Defined emergency zone: {zone_id}")

    def declare_emergency(
        self,
        emergency_type: str,
        location: GeospatialPoint,
        severity: str = "high",
        description: str = "",
    ) -> str:
        """Declare a new emergency situation."""
        emergency_id = f"emergency_{uuid.uuid4().hex[:8]}"

        emergency = EmergencyAlert(
            emergency_id=emergency_id,
            emergency_type=emergency_type,
            location=location,
            severity=severity,
            description=description,
            declared_at=datetime.now(UTC),
        )

        self.active_emergencies[emergency_id] = emergency

        # Trigger immediate alerts
        self._trigger_emergency_alerts(emergency)

        self.logger.info(f"Emergency declared: {emergency_id}")
        return emergency_id

    def resolve_emergency(self, emergency_id: str) -> bool:
        """Resolve an active emergency."""
        if emergency_id not in self.active_emergencies:
            return False

        emergency = self.active_emergencies[emergency_id]
        emergency.resolved_at = datetime.now(UTC)
        emergency.status = "resolved"

        # Send resolution notifications
        self._send_resolution_notifications(emergency)

        self.logger.info(f"Emergency resolved: {emergency_id}")
        return True

    def get_active_emergencies(self) -> list[EmergencyAlert]:
        """Get list of currently active emergencies."""
        return [
            emergency
            for emergency in self.active_emergencies.values()
            if emergency.status == "active"
        ]

    def _trigger_emergency_alerts(self, emergency: EmergencyAlert) -> None:
        """Trigger alerts for an emergency situation."""
        # Find affected emergency zones
        affected_zones = []
        for zone_id, bounds in self.emergency_zones.items():
            if bounds.contains_point(emergency.location):
                affected_zones.append(zone_id)

        # Get emergency contacts
        contacts = list(self.emergency_contacts.values())
        contacts.sort(key=lambda c: c.get("priority", 999))

        for contact in contacts[:10]:  # Limit to top 10 contacts
            # Create emergency notification
            notification_request = NotificationRequest(
                title=f"EMERGENCY: {emergency.emergency_type.upper()}",
                content=f"""
                Emergency Alert

                Type: {emergency.emergency_type}
                Severity: {emergency.severity}
                Location: {emergency.location.latitude}, {emergency.location.longitude}
                Description: {emergency.description}

                Immediate action may be required.
                """,
                recipients=[contact.get("email", "")],
                notification_type=NotificationType.ERROR,
                priority=MessagePriority.URGENT,
                delivery_method=["email", "sms"],
                geospatial_context={
                    "location": {
                        "latitude": emergency.location.latitude,
                        "longitude": emergency.location.longitude,
                    },
                    "emergency_type": emergency.emergency_type,
                    "severity": emergency.severity,
                },
            )

            try:
                self.notification_manager.create_notification(notification_request)
            except Exception as e:
                self.logger.error(f"Failed to send emergency alert: {e}")

    def _send_resolution_notifications(self, emergency: EmergencyAlert) -> None:
        """Send notifications when emergency is resolved."""
        duration_minutes = (
            cast(datetime, emergency.resolved_at) - emergency.declared_at
        ).total_seconds() / 60
        notification_request = NotificationRequest(
            title=f"Emergency Resolved: {emergency.emergency_type}",
            content=f"""
            The following emergency has been resolved:

            Type: {emergency.emergency_type}
            Location: {emergency.location.latitude}, {emergency.location.longitude}
            Duration: {duration_minutes:.1f} minutes
            """,
            recipients=["emergency_contacts"],  # Would be actual contact list
            notification_type=NotificationType.SUCCESS,
            priority=MessagePriority.NORMAL,
            delivery_method=["email"],
        )

        try:
            self.notification_manager.create_notification(notification_request)
        except Exception as e:
            self.logger.error(f"Failed to send resolution notification: {e}")


@dataclass
class EmergencyAlert:
    """Represents an active emergency situation."""

    emergency_id: str
    emergency_type: str
    location: GeospatialPoint
    severity: str
    description: str
    declared_at: datetime
    resolved_at: datetime | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "emergency_id": self.emergency_id,
            "emergency_type": self.emergency_type,
            "location": self.location.to_dict(),
            "severity": self.severity,
            "description": self.description,
            "declared_at": self.declared_at.isoformat(),
            "status": self.status,
        }
        if self.resolved_at:
            data["resolved_at"] = self.resolved_at.isoformat()
        return data
