"""
Audit logging system for GEO-INFER-SEC.

This module provides comprehensive audit logging for security events,
access attempts, data operations, and compliance tracking.
"""

import logging
import json
from typing import Any
from datetime import datetime, UTC
from dataclasses import dataclass, field, asdict
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(StrEnum):
    """Types of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    SYSTEM_EVENT = "system_event"


class AuditEventSeverity(StrEnum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit log event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str | None = None
    username: str | None = None
    resource: str | None = None
    action: str | None = None
    result: str = "success"  # success, failure, denied
    severity: AuditEventSeverity = AuditEventSeverity.MEDIUM
    ip_address: str | None = None
    user_agent: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data

    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Audit logger for security and compliance events.

    Provides comprehensive audit logging with multiple output formats,
    filtering, and compliance reporting capabilities.
    """

    def __init__(
        self,
        log_file: Path | None = None,
        enable_console: bool = True,
        enable_file: bool = True,
        retention_days: int = 90,
    ) -> None:
        """
        Initialize the audit logger.

        Args:
            log_file: Path to audit log file (if None, uses default location)
            enable_console: Whether to log to console
            enable_file: Whether to log to file
            retention_days: Number of days to retain audit logs
        """
        self.log_file = log_file or Path("audit.log")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.retention_days = retention_days

        # Configure file logging if enabled
        if self.enable_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Event storage (in production, use a database)
        self.events: list[AuditEvent] = []

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str | None = None,
        username: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        result: str = "success",
        severity: AuditEventSeverity = AuditEventSeverity.MEDIUM,
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event
            user_id: User identifier
            username: Username
            resource: Resource being accessed
            action: Action performed
            result: Result of the action (success, failure, denied)
            severity: Event severity level
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional event details
            metadata: Additional metadata

        Returns:
            Created AuditEvent object
        """
        import uuid

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            user_id=user_id,
            username=username,
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            metadata=metadata or {},
        )

        # Store event
        self.events.append(event)

        # Log to console
        if self.enable_console:
            log_message = (
                f"[{event.timestamp.isoformat()}] "
                f"{event.event_type.value.upper()} - "
                f"User: {username or user_id or 'unknown'}, "
                f"Action: {action or 'unknown'}, "
                f"Resource: {resource or 'unknown'}, "
                f"Result: {result.upper()}"
            )
            if severity == AuditEventSeverity.CRITICAL:
                logger.critical(log_message)
            elif severity == AuditEventSeverity.HIGH:
                logger.error(log_message)
            elif severity == AuditEventSeverity.MEDIUM:
                logger.warning(log_message)
            else:
                logger.info(log_message)

        # Log to file
        if self.enable_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log to file: {e}")

        return event

    def log_authentication(
        self,
        username: str,
        user_id: str | None = None,
        result: str = "success",
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Log an authentication event.

        Args:
            username: Username
            user_id: User identifier
            result: Authentication result (success, failure)
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional details

        Returns:
            Created AuditEvent object
        """
        severity = (
            AuditEventSeverity.HIGH
            if result == "failure"
            else AuditEventSeverity.MEDIUM
        )

        return self.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id=user_id,
            username=username,
            action="login",
            result=result,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
        )

    def log_authorization(
        self,
        user_id: str,
        username: str | None = None,
        resource: str = "",
        action: str = "",
        result: str = "success",
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Log an authorization event.

        Args:
            user_id: User identifier
            username: Username
            resource: Resource being accessed
            action: Action attempted
            result: Authorization result (success, denied)
            ip_address: Client IP address
            details: Additional details

        Returns:
            Created AuditEvent object
        """
        severity = (
            AuditEventSeverity.HIGH if result == "denied" else AuditEventSeverity.MEDIUM
        )

        return self.log_event(
            event_type=AuditEventType.AUTHORIZATION,
            user_id=user_id,
            username=username,
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            ip_address=ip_address,
            details=details or {},
        )

    def log_data_access(
        self,
        user_id: str,
        username: str | None = None,
        resource: str = "",
        action: str = "read",
        result: str = "success",
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Log a data access event.

        Args:
            user_id: User identifier
            username: Username
            resource: Resource being accessed
            action: Action performed (read, write, delete)
            result: Access result (success, denied, error)
            ip_address: Client IP address
            details: Additional details (e.g., data volume, spatial bounds)

        Returns:
            Created AuditEvent object
        """
        severity = (
            AuditEventSeverity.CRITICAL
            if result == "denied" and action in ["write", "delete"]
            else AuditEventSeverity.MEDIUM
        )

        return self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            username=username,
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            ip_address=ip_address,
            details=details or {},
        )

    def get_events(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        severity: AuditEventSeverity | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """
        Retrieve audit events with filtering.

        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Filter events after this time
            end_time: Filter events before this time
            severity: Filter by severity level
            limit: Maximum number of events to return

        Returns:
            List of matching AuditEvent objects
        """
        filtered_events = self.events

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]

        # Sort by timestamp (most recent first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]

    def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        report_type: str = "summary",
    ) -> dict[str, Any]:
        """
        Generate a compliance report from audit logs.

        Args:
            start_time: Report start time
            end_time: Report end time
            report_type: Type of report (summary, detailed, compliance)

        Returns:
            Compliance report dictionary
        """
        events = self.get_events(start_time=start_time, end_time=end_time)

        # Count events by type
        event_counts: dict[str, int] = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Count events by result
        result_counts: dict[str, int] = {}
        for event in events:
            result = event.result
            result_counts[result] = result_counts.get(result, 0) + 1

        # Count events by severity
        severity_counts: dict[str, int] = {}
        for event in events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Find critical events
        critical_events = [
            e for e in events if e.severity == AuditEventSeverity.CRITICAL
        ]

        report = {
            "report_type": report_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_events": len(events),
            "event_counts": event_counts,
            "result_counts": result_counts,
            "severity_counts": severity_counts,
            "critical_events_count": len(critical_events),
            "generated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        }

        if report_type == "detailed":
            report["events"] = [e.to_dict() for e in events[:100]]

        if report_type == "compliance":
            # Add compliance-specific metrics
            failed_auth = [
                e
                for e in events
                if e.event_type == AuditEventType.AUTHENTICATION
                and e.result == "failure"
            ]
            denied_access = [
                e
                for e in events
                if e.event_type == AuditEventType.AUTHORIZATION and e.result == "denied"
            ]

            report["compliance_metrics"] = {
                "failed_authentication_attempts": len(failed_auth),
                "denied_access_attempts": len(denied_access),
                "data_access_events": event_counts.get(
                    AuditEventType.DATA_ACCESS.value, 0
                ),
                "data_modification_events": event_counts.get(
                    AuditEventType.DATA_MODIFICATION.value, 0
                ),
            }

        return report
