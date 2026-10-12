"""Tests for notification scheduling, delivery, and alert cooldown."""

import time
from datetime import datetime, timedelta, UTC

from geo_infer_comms.core.notifications import (
    AlertRule,
    AlertSystem,
    NotificationManager,
)
from geo_infer_comms.models.message import (
    NotificationRequest,
    NotificationStatus,
)


def _request(recipients: list, content: str = "notice") -> NotificationRequest:
    return NotificationRequest(
        title="Test notification",
        content=content,
        recipients=recipients,
        notification_type="info",
        priority="normal",
        delivery_method=["in_app"],
    )


class TestScheduledNotifications:
    def test_schedule_notification_records_schedule_time(self) -> None:
        manager = NotificationManager(max_notifications=100, enable_persistence=False)
        manager.start()
        try:
            future = datetime.now(UTC) + timedelta(seconds=5)
            notification_id = manager.schedule_notification(
                _request(["user@example.com"]), future
            )
        finally:
            manager.stop()

        notification = manager.notifications[notification_id]
        assert notification.schedule_time == future
        assert notification.status == NotificationStatus.PENDING

    def test_future_schedule_is_not_delivered_early(self) -> None:
        manager = NotificationManager(max_notifications=100, enable_persistence=False)
        manager.start()
        try:
            future = datetime.now(UTC) + timedelta(seconds=5)
            notification_id = manager.schedule_notification(
                _request(["user@example.com"]), future
            )

            # Allow both background threads several cycles to (incorrectly)
            # deliver the future-scheduled notification.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if manager.notifications[notification_id].status == (
                    NotificationStatus.SENT
                ):
                    break
                time.sleep(0.05)

            assert manager.notifications[notification_id].status == (
                NotificationStatus.PENDING
            )
        finally:
            manager.stop()

    def test_immediate_notification_is_delivered(self) -> None:
        manager = NotificationManager(max_notifications=100, enable_persistence=False)
        manager.start()
        try:
            notification = manager.create_notification(_request(["user@example.com"]))

            deadline = time.time() + 5
            while (
                notification.status == NotificationStatus.PENDING
                and time.time() < deadline
            ):
                time.sleep(0.05)
        finally:
            manager.stop()

        assert notification.status == NotificationStatus.SENT


class TestAlertCooldown:
    def test_trigger_alert_sets_last_triggered_and_engages_cooldown(self) -> None:
        manager = NotificationManager(max_notifications=100, enable_persistence=False)
        manager.start()
        alerts = AlertSystem(manager)
        try:
            rule = AlertRule(
                name="temp",
                description="overheat",
                conditions={"temperature": {"min": 35.0}},
                alert_title="Hot",
                alert_content="Too hot",
                recipients=["ops@example.com"],
                delivery_methods=["in_app"],
                cooldown_period=3600,
            )
            rule_id = alerts.create_alert_rule(rule)

            first = alerts.trigger_alert(rule_id, {"temperature": 40.0})
            second = alerts.trigger_alert(rule_id, {"temperature": 41.0})
        finally:
            manager.stop()

        assert first is not None
        assert rule.last_triggered is not None
        assert second is None, "cooldown period must suppress immediate re-trigger"


class TestGetNotifications:
    def test_user_id_filters_recipients(self) -> None:
        manager = NotificationManager(max_notifications=100, enable_persistence=False)
        manager.start()
        try:
            manager.create_notification(_request(["alice@example.com"], "for alice"))
            manager.create_notification(_request(["bob@example.com"], "for bob"))
        finally:
            manager.stop()

        alice_only = manager.get_notifications(user_id="alice@example.com")
        contents = [n.content for n in alice_only]

        assert contents == ["for alice"]
