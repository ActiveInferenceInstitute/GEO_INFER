"""
GEO-INFER-SEC provides security and privacy frameworks for sensitive geospatial information.

This module ensures that geospatial data is collected, processed, shared,
and stored in a manner that protects individual privacy, organizational
security, and complies with relevant regulations.
"""

from datetime import datetime, UTC
import logging
from copy import deepcopy
from typing import Any

from .core.authentication import AuthenticationManager, UserCredentials, TokenInfo
from .core.authorization import GeospatialAccessManager, Role, SpatialPermission
from .core.encryption import GeospatialEncryption
from .core.audit import AuditLogger, AuditEvent, AuditEventType, AuditEventSeverity
from .core.access_control import GeospatialAccessManager as AccessManager
from .models.security_models import SecurityEvent, ThreatLevel
from .utils.security_utils import SecurityUtils

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"
__email__ = "geo-infer@activeinference.institute"

logger = logging.getLogger(__name__)

__all__ = [
    "AuthenticationManager",
    "UserCredentials",
    "TokenInfo",
    "GeospatialAccessManager",
    "AccessManager",
    "Role",
    "SpatialPermission",
    "GeospatialEncryption",
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditEventSeverity",
    "SecurityEvent",
    "ThreatLevel",
    "SecurityUtils",
]


# High-level convenience class
class SecurityFramework:
    """
    High-level security framework for GEO-INFER applications.

    Provides comprehensive security and privacy protection for geospatial data
    processing and analysis workflows.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.audit_log: list[dict[str, Any]] = []
        from .core.cognitive_security import CognitiveSecurityManager

        self.cognitive = CognitiveSecurityManager()
        self.security_utils = SecurityUtils()

    def secure_data_processing(self, data: Any, privacy_level: str = "standard") -> Any:
        """Apply security measures to data processing pipeline."""
        levels = {"standard": 2, "high": 3, "strict": 4}
        if privacy_level not in levels:
            raise ValueError(f"Unsupported privacy level: {privacy_level}")
        protected = deepcopy(data)
        if hasattr(protected, "columns"):
            coordinate_columns = set(protected.columns)
            if {"lat", "lon"}.issubset(coordinate_columns):
                precision = levels[privacy_level] - 1
                protected = self.security_utils.anonymize_spatial_data(
                    protected, lat_col="lat", lon_col="lon", precision=precision
                )
            elif {"latitude", "longitude"}.issubset(coordinate_columns):
                protected = self.security_utils.anonymize_spatial_data(
                    protected,
                    lat_col="latitude",
                    lon_col="longitude",
                    precision=levels[privacy_level] - 1,
                )
        self.audit_access(
            "system",
            {"operation": "secure_data_processing", "privacy_level": privacy_level},
        )
        return protected

    def audit_access(self, user_id: str, data_access: Any) -> dict[str, Any]:
        """Audit data access for security compliance."""
        event = {
            "user_id": user_id,
            "data_access": data_access,
            "status": "recorded",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.audit_log.append(event)
        logger.info(
            "Recorded security audit access event", extra={"audit_event": event}
        )
        return event
