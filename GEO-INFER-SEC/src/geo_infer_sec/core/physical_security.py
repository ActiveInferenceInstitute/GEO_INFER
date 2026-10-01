"""
Physical Security Module for GEO-INFER-SEC

This module provides comprehensive physical security measures including:
- Access control systems with biometric and credential-based authentication
- Surveillance systems with geospatial monitoring capabilities
- Environmental security monitoring and alerting
- Physical infrastructure protection and hardening
- Perimeter security and intrusion detection
- Facility security management and compliance
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import yaml

from ..utils.geospatial_utils import GeoSpatialUtils
from ..utils.security_utils import SecurityUtils
from ..models.security_models import ThreatLevel


class AccessControlType(Enum):
    """Types of access control systems."""

    CARD_READER = "card_reader"
    BIOMETRIC = "biometric"
    KEYPAD = "keypad"
    FACIAL_RECOGNITION = "facial_recognition"
    FINGERPRINT = "fingerprint"
    IRIS_SCAN = "iris_scan"
    PROXIMITY_SENSOR = "proximity_sensor"
    MULTI_FACTOR = "multi_factor"


class SurveillanceType(Enum):
    """Types of surveillance systems."""

    CCTV = "cctv"
    THERMAL_CAMERA = "thermal_camera"
    MOTION_DETECTOR = "motion_detector"
    PERIMETER_SENSOR = "perimeter_sensor"
    DRONE_SURVEILLANCE = "drone_surveillance"
    LIDAR = "lidar"
    RADAR = "radar"
    ACOUSTIC_SENSOR = "acoustic_sensor"


class SecurityZoneType(Enum):
    """Types of security zones."""

    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"


@dataclass
class AccessControlDevice:
    """Represents a physical access control device."""

    device_id: str
    name: str
    device_type: AccessControlType
    location: Point
    zone_id: str
    is_active: bool = True
    last_heartbeat: datetime | None = None
    security_level: int = 1
    backup_power: bool = False
    tamper_detection: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurveillanceDevice:
    """Represents a surveillance device."""

    device_id: str
    name: str
    device_type: SurveillanceType
    location: Point
    zone_id: str
    coverage_area: Polygon | None = None
    is_active: bool = True
    recording_active: bool = False
    detection_sensitivity: float = 0.5
    field_of_view: float | None = None
    range_meters: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityZone:
    """Represents a security zone with specific access requirements."""

    zone_id: str
    name: str
    zone_type: SecurityZoneType
    boundary: Polygon | MultiPolygon
    required_clearance_level: int
    access_hours: dict[str, Any] | None = (
        None  # {"start": "08:00", "end": "18:00", "days": ["mon", "tue", ...]}
    )
    escort_required: bool = False
    two_person_rule: bool = False
    emergency_evacuation: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicalThreat:
    """Represents a detected physical threat."""

    threat_id: str
    threat_type: str
    location: Point
    severity: ThreatLevel
    detected_at: datetime
    detection_method: str
    description: str
    status: str = "active"  # active, investigating, resolved
    assigned_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PhysicalSecurityManager:
    """Comprehensive physical security management system."""

    def __init__(self, config_path: str | None = None):
        """Initialize the physical security manager."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Initialize security components
        self.access_devices: dict[str, AccessControlDevice] = {}
        self.surveillance_devices: dict[str, SurveillanceDevice] = {}
        self.security_zones: dict[str, SecurityZone] = {}
        self.active_threats: dict[str, PhysicalThreat] = {}

        # Initialize monitoring
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.alert_callbacks: list[Callable[[PhysicalThreat], None]] = []

        # Initialize utilities
        self.geo_utils = GeoSpatialUtils()
        self.security_utils = SecurityUtils()

        # Load initial configuration
        self._initialize_security_zones()
        self._initialize_devices()

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return cast(dict[str, Any], yaml.safe_load(f))
        return {
            "monitoring_interval": 5,
            "alert_threshold": 0.7,
            "default_security_level": 1,
            "emergency_contacts": [],
            "logging_level": "INFO",
        }

    def _initialize_security_zones(self) -> None:
        """Initialize default security zones."""
        # Create default zones if none exist
        if not self.security_zones:
            self.add_security_zone(
                SecurityZone(
                    zone_id="public",
                    name="Public Area",
                    zone_type=SecurityZoneType.PUBLIC,
                    boundary=self.geo_utils.create_circle(Point(0, 0), 1000),
                    required_clearance_level=0,
                )
            )

    def _initialize_devices(self) -> None:
        """Initialize devices from configuration."""
        devices_config = self.config.get("devices", {})

        # Initialize access control devices
        for device_config in devices_config.get("access_control", []):
            access_device = AccessControlDevice(**device_config)
            self.access_devices[access_device.device_id] = access_device

        # Initialize surveillance devices
        for device_config in devices_config.get("surveillance", []):
            surv_device = SurveillanceDevice(**device_config)
            self.surveillance_devices[surv_device.device_id] = surv_device

    # Security Zone Management
    def add_security_zone(self, zone: SecurityZone) -> bool:
        """Add a new security zone."""
        try:
            self.security_zones[zone.zone_id] = zone
            self.logger.info(f"Added security zone: {zone.zone_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding security zone: {e}")
            return False

    def get_security_zone(self, zone_id: str) -> SecurityZone | None:
        """Get a security zone by ID."""
        return self.security_zones.get(zone_id)

    def get_zones_for_location(self, location: Point) -> list[SecurityZone]:
        """Get all security zones that contain a given location."""
        zones = []
        for zone in self.security_zones.values():
            if zone.boundary.contains(location):
                zones.append(zone)
        return zones

    def update_zone_boundary(
        self, zone_id: str, new_boundary: Polygon | MultiPolygon
    ) -> bool:
        """Update the boundary of a security zone."""
        if zone_id in self.security_zones:
            self.security_zones[zone_id].boundary = new_boundary
            self.logger.info(f"Updated boundary for zone: {zone_id}")
            return True
        return False

    # Access Control Management
    def add_access_device(self, device: AccessControlDevice) -> bool:
        """Add a new access control device."""
        try:
            self.access_devices[device.device_id] = device
            self.logger.info(f"Added access control device: {device.device_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding access device: {e}")
            return False

    def verify_access_permission(
        self, user_id: str, device_id: str, clearance_level: int
    ) -> tuple[bool, str]:
        """Verify if a user has permission to access a device."""
        device = self.access_devices.get(device_id)
        if not device:
            return False, "Device not found"

        if not device.is_active:
            return False, "Device is inactive"

        # Get zones containing this device
        zones = self.get_zones_for_location(device.location)

        # Check clearance requirements
        for zone in zones:
            if clearance_level < zone.required_clearance_level:
                return False, f"Insufficient clearance level for zone {zone.zone_id}"

        # Check time-based restrictions
        current_time = datetime.now()
        for zone in zones:
            if zone.access_hours:
                if not self._is_within_access_hours(current_time, zone.access_hours):
                    return (
                        False,
                        f"Access outside permitted hours for zone {zone.zone_id}",
                    )

        return True, "Access granted"

    def _is_within_access_hours(
        self, current_time: datetime, access_hours: dict[str, Any]
    ) -> bool:
        """Check if current time is within permitted access hours."""
        # Implementation for time-based access control
        try:
            start_time = datetime.strptime(
                access_hours.get("start", "00:00"), "%H:%M"
            ).time()
            end_time = datetime.strptime(
                access_hours.get("end", "23:59"), "%H:%M"
            ).time()
            current_time_only = current_time.time()

            # Check if current day is allowed
            allowed_days = access_hours.get(
                "days", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            )
            current_day = current_time.strftime("%a").lower()

            if current_day not in allowed_days:
                return False

            # Check if current time is within range
            if start_time <= end_time:  # Same day
                return start_time <= current_time_only <= end_time
            else:  # Crosses midnight
                return current_time_only >= start_time or current_time_only <= end_time
        except Exception:
            self.logger.error("Access-hours configuration is invalid; denying access")
            return False

    # Surveillance Management
    def add_surveillance_device(self, device: SurveillanceDevice) -> bool:
        """Add a new surveillance device."""
        try:
            self.surveillance_devices[device.device_id] = device
            self.logger.info(f"Added surveillance device: {device.device_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding surveillance device: {e}")
            return False

    def get_surveillance_coverage(self, location: Point) -> list[SurveillanceDevice]:
        """Get all surveillance devices that cover a specific location."""
        covering_devices = []
        for device in self.surveillance_devices.values():
            if device.coverage_area and device.coverage_area.contains(location):
                covering_devices.append(device)
        return covering_devices

    def calculate_surveillance_coverage_map(self) -> gpd.GeoDataFrame:
        """Calculate overall surveillance coverage map."""
        coverage_areas = []
        device_info = []

        for device in self.surveillance_devices.values():
            if device.coverage_area and device.is_active:
                coverage_areas.append(device.coverage_area)
                device_info.append(
                    {
                        "device_id": device.device_id,
                        "device_type": device.device_type.value,
                        "name": device.name,
                    }
                )

        if coverage_areas:
            coverage_gdf = gpd.GeoDataFrame(
                device_info, geometry=coverage_areas, crs="EPSG:4326"
            )
            return coverage_gdf
        else:
            return gpd.GeoDataFrame(
                columns=["device_id", "device_type", "name", "geometry"]
            )

    # Threat Detection and Response
    def detect_intrusion(
        self, location: Point, detection_method: str, confidence: float = 1.0
    ) -> PhysicalThreat | None:
        """Detect and register a potential intrusion."""
        threat_id = f"intrusion_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Determine threat severity based on location and zones
        zones = self.get_zones_for_location(location)
        max_security_level = max(
            [zone.required_clearance_level for zone in zones], default=0
        )

        if max_security_level >= 3:
            severity = ThreatLevel.CRITICAL
        elif max_security_level >= 2:
            severity = ThreatLevel.HIGH
        else:
            severity = ThreatLevel.MEDIUM

        threat = PhysicalThreat(
            threat_id=threat_id,
            threat_type="intrusion",
            location=location,
            severity=severity,
            detected_at=datetime.now(),
            detection_method=detection_method,
            description=f"Intrusion detected at {location.x}, {location.y}",
            metadata={"confidence": confidence, "zones": [z.zone_id for z in zones]},
        )

        self.active_threats[threat_id] = threat
        self._trigger_alert(threat)

        return threat

    def detect_unauthorized_access(
        self, device_id: str, user_id: str, attempted_at: datetime
    ) -> PhysicalThreat | None:
        """Detect unauthorized access attempts."""
        device = self.access_devices.get(device_id)
        if not device:
            return None

        threat_id = f"unauthorized_access_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        threat = PhysicalThreat(
            threat_id=threat_id,
            threat_type="unauthorized_access",
            location=device.location,
            severity=ThreatLevel.HIGH,
            detected_at=datetime.now(),
            detection_method="access_control",
            description=f"Unauthorized access attempt by user {user_id} at device {device_id}",
            metadata={
                "device_id": device_id,
                "user_id": user_id,
                "attempted_at": attempted_at.isoformat(),
            },
        )

        self.active_threats[threat_id] = threat
        self._trigger_alert(threat)

        return threat

    def _trigger_alert(self, threat: PhysicalThreat) -> None:
        """Trigger security alerts for detected threats."""
        for callback in self.alert_callbacks:
            try:
                callback(threat)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        # Log the alert
        self.logger.warning(
            f"SECURITY ALERT: {threat.threat_type} at {threat.location} - {threat.description}"
        )

    def add_alert_callback(self, callback: Callable[[PhysicalThreat], None]) -> None:
        """Add a callback function for security alerts."""
        self.alert_callbacks.append(callback)

    # Monitoring and Maintenance
    def start_monitoring(self) -> None:
        """Start continuous security monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Physical security monitoring started")

    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Physical security monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_device_health()
                self._check_zone_integrity()
                self._process_alerts()

                time.sleep(self.config.get("monitoring_interval", 5))
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _check_device_health(self) -> None:
        """Check health status of all devices."""
        current_time = datetime.now()

        # Check access control devices
        for device in self.access_devices.values():
            if device.last_heartbeat:
                time_since_heartbeat = current_time - device.last_heartbeat
                if time_since_heartbeat > timedelta(minutes=5):
                    self.logger.warning(
                        f"Access device {device.device_id} may be offline"
                    )

        # Check surveillance devices
        for surv_device in self.surveillance_devices.values():
            if hasattr(surv_device, "last_heartbeat") and surv_device.last_heartbeat:
                time_since_heartbeat = current_time - surv_device.last_heartbeat
                if time_since_heartbeat > timedelta(minutes=5):
                    self.logger.warning(
                        f"Surveillance device {surv_device.device_id} may be offline"
                    )

    def _check_zone_integrity(self) -> None:
        """Check integrity of security zones by verifying device coverage."""
        for zone_id, zone in self.security_zones.items():
            zone_devices = [
                d
                for d in self.access_devices.values()
                if getattr(d, "zone_id", None) == zone_id
            ]
            if not zone_devices:
                self.logger.warning(
                    f"Security zone {zone_id} has no active access devices"
                )

    def _process_alerts(self) -> None:
        """Process and manage active alerts."""
        # Clean up old resolved threats
        current_time = datetime.now()
        to_remove = []

        for threat_id, threat in self.active_threats.items():
            if threat.status == "resolved":
                age = current_time - threat.detected_at
                if age > timedelta(hours=24):  # Keep resolved threats for 24 hours
                    to_remove.append(threat_id)

        for threat_id in to_remove:
            del self.active_threats[threat_id]

    # Reporting and Analytics
    def generate_security_report(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate a comprehensive security report."""
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_threats": len(self.active_threats),
                "active_devices": len(
                    [d for d in self.access_devices.values() if d.is_active]
                ),
                "security_zones": len(self.security_zones),
                "surveillance_coverage": self._calculate_coverage_percentage(),
            },
            "threat_analysis": self._analyze_threats(start_date, end_date),
            "device_status": self._get_device_status_summary(),
            "zone_analysis": self._analyze_zones(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _calculate_coverage_percentage(self) -> float:
        """Calculate percentage of area under surveillance."""
        # This is a simplified calculation
        total_area = sum([zone.boundary.area for zone in self.security_zones.values()])
        covered_area = sum(
            [
                device.coverage_area.area
                for device in self.surveillance_devices.values()
                if device.coverage_area and device.is_active
            ]
        )

        if total_area > 0:
            return min(100.0, cast(float, (covered_area / total_area) * 100))
        return 0.0

    def _analyze_threats(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Analyze threats within a date range."""
        relevant_threats = [
            threat
            for threat in self.active_threats.values()
            if start_date <= threat.detected_at <= end_date
        ]

        threat_types: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for threat in relevant_threats:
            threat_types[threat.threat_type] = (
                threat_types.get(threat.threat_type, 0) + 1
            )
            severity_counts[threat.severity.value] = (
                severity_counts.get(threat.severity.value, 0) + 1
            )

        return {
            "total_threats": len(relevant_threats),
            "threat_types": threat_types,
            "severity_distribution": severity_counts,
            "resolution_rate": self._calculate_resolution_rate(relevant_threats),
        }

    def _calculate_resolution_rate(self, threats: list[PhysicalThreat]) -> float:
        """Calculate threat resolution rate."""
        if not threats:
            return 0.0

        resolved_count = len([t for t in threats if t.status == "resolved"])
        return (resolved_count / len(threats)) * 100

    def _get_device_status_summary(self) -> dict[str, Any]:
        """Get summary of device status."""
        access_devices = len(self.access_devices)
        active_access = len([d for d in self.access_devices.values() if d.is_active])

        surveillance_devices = len(self.surveillance_devices)
        active_surveillance = len(
            [d for d in self.surveillance_devices.values() if d.is_active]
        )

        return {
            "access_control": {
                "total": access_devices,
                "active": active_access,
                "inactive": access_devices - active_access,
            },
            "surveillance": {
                "total": surveillance_devices,
                "active": active_surveillance,
                "inactive": surveillance_devices - active_surveillance,
            },
        }

    def _analyze_zones(self) -> dict[str, Any]:
        """Analyze security zones."""
        zone_analysis = {}

        for zone in self.security_zones.values():
            # Count devices in zone
            devices_in_zone = 0
            for device in self.access_devices.values():
                if zone.boundary.contains(device.location):
                    devices_in_zone += 1

            # Count surveillance coverage
            surveillance_coverage = 0
            for surv_device in self.surveillance_devices.values():
                if surv_device.coverage_area and zone.boundary.intersects(
                    surv_device.coverage_area
                ):
                    surveillance_coverage += 1

            zone_analysis[zone.zone_id] = {
                "type": zone.zone_type.value,
                "clearance_level": zone.required_clearance_level,
                "access_devices": devices_in_zone,
                "surveillance_coverage": surveillance_coverage,
                "area": zone.boundary.area,
            }

        return zone_analysis

    def _generate_recommendations(self) -> list[str]:
        """Generate security recommendations based on current state."""
        recommendations = []

        # Check for zones without adequate coverage
        for zone in self.security_zones.values():
            devices_in_zone = sum(
                1
                for device in self.access_devices.values()
                if zone.boundary.contains(device.location)
            )
            if devices_in_zone == 0 and zone.zone_type != SecurityZoneType.PUBLIC:
                recommendations.append(
                    f"Zone {zone.zone_id} lacks access control devices"
                )

        # Check for inactive devices
        inactive_devices = [d for d in self.access_devices.values() if not d.is_active]
        if inactive_devices:
            recommendations.append(
                f"{len(inactive_devices)} access control devices are inactive"
            )

        # Check surveillance coverage
        coverage = self._calculate_coverage_percentage()
        if coverage < 80:
            recommendations.append(
                f"Surveillance coverage is only {coverage:.1f}% - consider adding more cameras"
            )

        return recommendations

    def get_active_threats(self) -> list[PhysicalThreat]:
        """Get all active threats."""
        return [
            threat
            for threat in self.active_threats.values()
            if threat.status == "active"
        ]

    def resolve_threat(self, threat_id: str, resolution_notes: str = "") -> bool:
        """Mark a threat as resolved."""
        if threat_id in self.active_threats:
            self.active_threats[threat_id].status = "resolved"
            self.active_threats[threat_id].metadata["resolution_notes"] = (
                resolution_notes
            )
            self.active_threats[threat_id].metadata["resolved_at"] = (
                datetime.now().isoformat()
            )
            return True
        return False
