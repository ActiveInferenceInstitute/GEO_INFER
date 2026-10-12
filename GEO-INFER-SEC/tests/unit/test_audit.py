"""
Unit tests for audit logging functionality.
"""

import pytest
import logging
from datetime import datetime, timedelta, UTC

from geo_infer_sec import SecurityFramework
from geo_infer_sec.core.audit import (
    AuditLogger,
    AuditEventType,
)


class TestAuditLogger:
    """Test AuditLogger class."""

    @pytest.fixture
    def audit_logger(self, tmp_path) -> AuditLogger:
        """Create an audit logger instance."""
        log_file = tmp_path / "audit.log"
        return AuditLogger(
            log_file=log_file,
            enable_console=False,
            enable_file=True,
        )

    def test_log_event(self, audit_logger: AuditLogger) -> None:
        """Test basic event logging."""
        event = audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            username="testuser",
            action="login",
            result="success",
        )

        assert event is not None
        assert event.event_type == AuditEventType.AUTHENTICATION
        assert event.username == "testuser"
        assert event.result == "success"

    def test_log_authentication(self, audit_logger: AuditLogger) -> None:
        """Test authentication event logging."""
        event = audit_logger.log_authentication(
            username="testuser",
            result="success",
            ip_address="192.168.1.1",
        )

        assert event.event_type == AuditEventType.AUTHENTICATION
        assert event.action == "login"
        assert event.ip_address == "192.168.1.1"

    def test_log_authorization(self, audit_logger: AuditLogger) -> None:
        """Test authorization event logging."""
        event = audit_logger.log_authorization(
            user_id="user123",
            resource="dataset_xyz",
            action="read",
            result="success",
        )

        assert event.event_type == AuditEventType.AUTHORIZATION
        assert event.resource == "dataset_xyz"
        assert event.action == "read"

    def test_log_data_access(self, audit_logger: AuditLogger) -> None:
        """Test data access event logging."""
        event = audit_logger.log_data_access(
            user_id="user123",
            resource="sensor_data",
            action="read",
            result="success",
        )

        assert event.event_type == AuditEventType.DATA_ACCESS
        assert event.resource == "sensor_data"

    def test_get_events_filtered(self, audit_logger: AuditLogger) -> None:
        """Test event retrieval with filtering."""
        # Log multiple events
        audit_logger.log_authentication(username="user1", result="success")
        audit_logger.log_authentication(username="user2", result="failure")
        audit_logger.log_data_access(user_id="user1", resource="data1", action="read")

        # Filter by event type
        auth_events = audit_logger.get_events(event_type=AuditEventType.AUTHENTICATION)
        assert len(auth_events) == 2

        # Filter by user
        user1_events = audit_logger.get_events(user_id="user1")
        assert len(user1_events) >= 1

    def test_get_events_time_range(self, audit_logger: AuditLogger) -> None:
        """Test event retrieval with time range."""
        start_time = datetime.now(UTC).replace(tzinfo=None)

        audit_logger.log_authentication(username="user1", result="success")

        end_time = datetime.now(UTC).replace(tzinfo=None)

        events = audit_logger.get_events(start_time=start_time, end_time=end_time)
        assert len(events) >= 1

    def test_generate_compliance_report(self, audit_logger: AuditLogger) -> None:
        """Test compliance report generation."""
        start_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)

        # Log various events
        audit_logger.log_authentication(username="user1", result="success")
        audit_logger.log_authentication(username="user2", result="failure")
        audit_logger.log_authorization(
            user_id="user1", resource="data1", action="read", result="success"
        )
        audit_logger.log_authorization(
            user_id="user2", resource="data2", action="write", result="denied"
        )

        # Set end_time AFTER logging events so they fall within the range
        end_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=1)

        report = audit_logger.generate_compliance_report(
            start_time=start_time, end_time=end_time, report_type="compliance"
        )

        assert "total_events" in report
        assert "event_counts" in report
        assert "compliance_metrics" in report
        assert report["compliance_metrics"]["failed_authentication_attempts"] >= 1
        assert report["compliance_metrics"]["denied_access_attempts"] >= 1


def test_security_framework_audit_access_records_event(caplog) -> None:
    """Test high-level framework audit access returns and stores an event."""
    framework = SecurityFramework()

    with caplog.at_level(logging.INFO, logger="geo_infer_sec"):
        event = framework.audit_access(
            "user-123", {"resource": "parcel-layer", "action": "read"}
        )

    assert event["user_id"] == "user-123"
    assert event["data_access"]["resource"] == "parcel-layer"
    assert event["status"] == "recorded"
    assert event in framework.audit_log
    assert "timestamp" in event
