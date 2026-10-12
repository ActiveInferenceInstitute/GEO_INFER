"""
Digital Security Module for GEO-INFER-SEC

This module provides comprehensive digital security measures including:
- Cybersecurity threat detection and response
- Network security monitoring and protection
- Data protection and encryption management
- Vulnerability assessment and management
- Security information and event management (SIEM)
- Digital forensics and incident response
- API security and secure communications
"""

import importlib.resources
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field

import secrets

from cryptography.fernet import Fernet
import jwt
import yaml

from ..utils.security_utils import SecurityUtils
from ..models.security_models import (
    SecurityEvent,
    ThreatLevel,
    SecurityAlert,
    SecurityEventCategory,
)


class ThreatType(Enum):
    """Types of digital security threats."""

    MALWARE = "malware"
    PHISHING = "phishing"
    RANSOMWARE = "ransomware"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    DDoS = "ddos"
    MAN_IN_MIDDLE = "man_in_middle"
    INSIDER_THREAT = "insider_threat"
    APT = "advanced_persistent_threat"
    ZERO_DAY = "zero_day"


class SecurityEventType(Enum):
    """Types of security events."""

    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    DATA_ACCESS = "data_access"
    API_CALL = "api_call"
    FILE_ACCESS = "file_access"
    NETWORK_CONNECTION = "network_connection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CONFIGURATION_CHANGE = "configuration_change"
    VULNERABILITY_SCAN = "vulnerability_scan"


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA = "ecdsa"
    ChaCha20 = "chacha20"


@dataclass
class DigitalThreat:
    """Represents a detected digital security threat."""

    threat_id: str
    threat_type: ThreatType
    severity: ThreatLevel
    source_ip: str | None = None
    target_system: str | None = None
    detected_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    indicators: list[str] = field(default_factory=list)
    mitigated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DigitalSecurityPolicy:
    """Digital-domain security policy.

    Distinct from the canonical :class:`geo_infer_sec.models.security_models.SecurityPolicy`,
    which models organization-wide policy records; this dataclass is the
    runtime policy shape used by the digital security engine.
    """

    policy_id: str
    name: str
    category: str
    rules: list[dict[str, Any]]
    severity: ThreatLevel
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkConnection:
    """Represents a network connection for monitoring."""

    connection_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    timestamp: datetime
    data_transferred: int = 0
    connection_state: str = "active"
    is_encrypted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VulnerabilityReport:
    """Represents a vulnerability assessment report."""

    scan_id: str
    target_system: str
    scan_date: datetime
    vulnerabilities: list[dict[str, Any]]
    risk_score: float
    recommendations: list[str]
    compliance_status: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)


class DigitalSecurityManager:
    """Comprehensive digital security management system.

    .. note::
        SIMULATION-ONLY threat intelligence. Threat indicators (blocked IPs,
        malware hashes, suspicious domains) are loaded from a configurable
        YAML file (``threat_indicators_file`` config key, defaulting to the
        packaged ``config/threat_indicators.yaml``). The shipped defaults are
        illustrative examples, not live feeds, and the scheduled background
        loops (key rotation, vulnerability scanning) are placeholders that do
        not contact real systems. Do not treat this manager as a production
        SIEM without wiring real feeds and responders.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the digital security manager."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Initialize security components
        self.active_threats: dict[str, DigitalThreat] = {}
        self.security_policies: dict[str, DigitalSecurityPolicy] = {}
        self.security_events: list[SecurityEvent] = []
        self.network_connections: dict[str, NetworkConnection] = {}
        self.vulnerability_reports: dict[str, VulnerabilityReport] = {}

        # Initialize encryption
        self.encryption_keys: dict[str, bytes] = {}
        self.key_rotation_schedule: dict[str, datetime] = {}

        # Initialize monitoring
        self.monitoring_active = False
        self.monitoring_threads: list[threading.Thread] = []
        self.alert_callbacks: list[Callable[[SecurityAlert], None]] = []

        # Initialize security utils
        self.security_utils = SecurityUtils()

        # Threat intelligence
        self.threat_indicators: set[str] = set()
        self.blocked_ips: set[str] = set()
        self.ip_unblock_schedule: dict[str, datetime] = {}
        self.trusted_ips: set[str] = set()

        # Rate limiting
        self.rate_limits: dict[str, dict[str, Any]] = defaultdict(dict)

        # Load initial configuration
        self._initialize_security_policies()
        self._initialize_encryption_keys()
        self._load_threat_intelligence()

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "monitoring_interval": 10,
            "threat_retention_days": 30,
            "max_login_attempts": 5,
            "session_timeout_minutes": 30,
            "encryption_key_rotation_days": 90,
            "vulnerability_scan_interval_hours": 24,
            "threat_intelligence_update_hours": 6,
            "api_rate_limit_per_minute": 100,
            "logging_level": "INFO",
        }

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)

        return default_config

    def _initialize_security_policies(self) -> None:
        """Initialize default security policies."""
        default_policies = [
            {
                "policy_id": "failed_login_attempts",
                "name": "Failed Login Attempts",
                "category": "authentication",
                "rules": [
                    {
                        "condition": "failed_attempts >= 5",
                        "action": "block_ip",
                        "duration": 3600,
                    }
                ],
                "severity": ThreatLevel.HIGH,
            },
            {
                "policy_id": "suspicious_network_traffic",
                "name": "Suspicious Network Traffic",
                "category": "network",
                "rules": [
                    {
                        "condition": "port_scan_detected",
                        "action": "alert",
                        "severity": "high",
                    },
                    {
                        "condition": "unusual_data_volume",
                        "action": "investigate",
                        "threshold": "1GB",
                    },
                ],
                "severity": ThreatLevel.MEDIUM,
            },
            {
                "policy_id": "malware_detection",
                "name": "Malware Detection",
                "category": "endpoint",
                "rules": [
                    {"condition": "signature_match", "action": "quarantine"},
                    {"condition": "behavioral_anomaly", "action": "sandbox"},
                ],
                "severity": ThreatLevel.CRITICAL,
            },
        ]

        for policy_config in default_policies:
            policy = DigitalSecurityPolicy(**cast(dict[str, Any], policy_config))
            self.security_policies[policy.policy_id] = policy

    def _initialize_encryption_keys(self) -> None:
        """Initialize encryption keys."""
        # Generate master key if not exists
        if "master_key" not in self.encryption_keys:
            self.encryption_keys["master_key"] = Fernet.generate_key()

        # Schedule key rotation
        self.key_rotation_schedule["master_key"] = datetime.now() + timedelta(
            days=self.config.get("encryption_key_rotation_days", 90)
        )

    def _load_threat_intelligence(self) -> None:
        """Load threat intelligence indicators from a configurable YAML file.

        SIMULATION-ONLY: the shipped defaults are illustrative examples, not
        live threat-intelligence feeds. Provide ``threat_indicators_file`` in
        the manager config to point at your own indicator file with the shape
        ``{"blocked_ips": [...], "threat_indicators": [...], "trusted_ips": [...]}``.

        Resolution order:
        1. ``threat_indicators_file`` config key
        2. ``GEO_INFER_SEC_THREAT_INDICATORS`` environment variable
        3. packaged ``config/threat_indicators.yaml`` resource
        """
        indicator_file = self.config.get("threat_indicators_file") or os.environ.get(
            "GEO_INFER_SEC_THREAT_INDICATORS"
        )

        if indicator_file:
            path = Path(indicator_file)
            if path.exists():
                with open(path) as f:
                    indicators = yaml.safe_load(f) or {}
            else:
                self.logger.warning(
                    "Threat indicator file not found: %s; starting with empty sets",
                    path,
                )
                indicators = {}
        else:
            packaged = importlib.resources.files("geo_infer_sec").joinpath(
                "config", "threat_indicators.yaml"
            )
            try:
                with importlib.resources.as_file(packaged) as packaged_path:
                    with open(packaged_path) as f:
                        indicators = yaml.safe_load(f) or {}
            except FileNotFoundError:
                self.logger.warning(
                    "Packaged threat indicator resource not found: %s; starting with empty sets",
                    packaged,
                )
                indicators = {}

        self.blocked_ips.update(indicators.get("blocked_ips", []))
        self.threat_indicators.update(indicators.get("threat_indicators", []))
        self.trusted_ips.update(indicators.get("trusted_ips", []))
        self.logger.info(
            "Loaded threat intelligence: %d blocked IPs, %d indicators",
            len(self.blocked_ips),
            len(self.threat_indicators),
        )

    # Threat Detection and Analysis
    def detect_threat(self, event_data: dict[str, Any]) -> DigitalThreat | None:
        """Analyze an event for potential threats."""
        threat_type = self._classify_threat(event_data)
        if not threat_type:
            return None

        threat_id = f"threat_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(str(event_data)) % 10000}"

        threat = DigitalThreat(
            threat_id=threat_id,
            threat_type=threat_type,
            severity=self._calculate_threat_severity(threat_type, event_data),
            source_ip=event_data.get("source_ip"),
            target_system=event_data.get("target_system"),
            description=self._generate_threat_description(threat_type, event_data),
            indicators=self._extract_indicators(event_data),
            metadata=event_data,
        )

        self.active_threats[threat_id] = threat
        self._trigger_alert(threat)

        return threat

    def _classify_threat(self, event_data: dict[str, Any]) -> ThreatType | None:
        """Classify the type of threat based on event data."""
        # Check for malware indicators
        if any(indicator in str(event_data) for indicator in self.threat_indicators):
            return ThreatType.MALWARE

        # Check for failed login attempts
        if event_data.get("event_type") == "login_failure":
            return ThreatType.UNAUTHORIZED_ACCESS

        # Check for SQL injection patterns
        if self._detect_sql_injection(event_data.get("request_data", "")):
            return ThreatType.SQL_INJECTION

        # Check for XSS patterns
        if self._detect_xss(event_data.get("request_data", "")):
            return ThreatType.XSS

        # Check for DDoS patterns
        if self._detect_ddos_pattern(event_data):
            return ThreatType.DDoS

        # Check for data exfiltration
        if self._detect_data_exfiltration(event_data):
            return ThreatType.DATA_BREACH

        return None

    def _detect_sql_injection(self, request_data: str) -> bool:
        """Detect SQL injection patterns."""
        sql_patterns = [
            r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
            r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
            r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
            r"((\%27)|(\'))union",
            r"exec(\s|\+)+(s|x)p\w+",
            r"UNION(?:\s+ALL)?\s+SELECT",
            r"DROP\s+TABLE",
            r"INSERT\s+INTO",
            r"UPDATE\s+\w+\s+SET",
        ]

        for pattern in sql_patterns:
            if re.search(pattern, request_data, re.IGNORECASE):
                return True
        return False

    def _detect_xss(self, request_data: str) -> bool:
        """Detect XSS patterns."""
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
            r"expression\(",
            r"vbscript:",
            r"<img[^>]*onerror[^>]*>",
        ]

        for pattern in xss_patterns:
            if re.search(pattern, request_data, re.IGNORECASE):
                return True
        return False

    def _detect_ddos_pattern(self, event_data: dict[str, Any]) -> bool:
        """Detect DDoS attack patterns."""
        source_ip = event_data.get("source_ip")
        if not source_ip:
            return False

        # Check request rate from same IP
        current_time = datetime.now()
        recent_requests = [
            event
            for event in self.security_events[-1000:]  # Check last 1000 events
            if (
                event.metadata.get("source_ip") == source_ip
                and current_time - event.timestamp < timedelta(minutes=1)
            )
        ]

        # If more than 100 requests per minute from same IP, flag as potential DDoS
        return len(recent_requests) > 100

    def _detect_data_exfiltration(self, event_data: dict[str, Any]) -> bool:
        """Detect potential data exfiltration."""
        # Check for unusual data access patterns
        data_volume = event_data.get("data_volume", 0)
        if data_volume > 1024 * 1024 * 100:  # 100MB threshold
            return True

        # Check for access to sensitive data patterns
        sensitive_patterns = ["credit_card", "ssn", "password", "token", "key"]
        accessed_data = event_data.get("accessed_data", "")

        return any(pattern in accessed_data.lower() for pattern in sensitive_patterns)

    def _calculate_threat_severity(
        self, threat_type: ThreatType, event_data: dict[str, Any]
    ) -> ThreatLevel:
        """Calculate threat severity based on type and context."""
        base_severity = {
            ThreatType.MALWARE: ThreatLevel.CRITICAL,
            ThreatType.RANSOMWARE: ThreatLevel.CRITICAL,
            ThreatType.DATA_BREACH: ThreatLevel.CRITICAL,
            ThreatType.APT: ThreatLevel.CRITICAL,
            ThreatType.ZERO_DAY: ThreatLevel.CRITICAL,
            ThreatType.SQL_INJECTION: ThreatLevel.HIGH,
            ThreatType.UNAUTHORIZED_ACCESS: ThreatLevel.HIGH,
            ThreatType.PHISHING: ThreatLevel.HIGH,
            ThreatType.XSS: ThreatLevel.MEDIUM,
            ThreatType.DDoS: ThreatLevel.MEDIUM,
            ThreatType.MAN_IN_MIDDLE: ThreatLevel.HIGH,
            ThreatType.INSIDER_THREAT: ThreatLevel.HIGH,
        }

        severity = base_severity.get(threat_type, ThreatLevel.MEDIUM)

        # Adjust severity based on context
        if event_data.get("target_system") in ["database", "authentication_server"]:
            if severity == ThreatLevel.MEDIUM:
                severity = ThreatLevel.HIGH
            elif severity == ThreatLevel.HIGH:
                severity = ThreatLevel.CRITICAL

        return severity

    def _generate_threat_description(
        self, threat_type: ThreatType, event_data: dict[str, Any]
    ) -> str:
        """Generate a human-readable threat description."""
        descriptions = {
            ThreatType.MALWARE: f"Malware detected from {event_data.get('source_ip', 'unknown')}",
            ThreatType.SQL_INJECTION: f"SQL injection attempt detected in request to {event_data.get('target_system', 'unknown')}",
            ThreatType.XSS: "Cross-site scripting attempt detected",
            ThreatType.UNAUTHORIZED_ACCESS: f"Unauthorized access attempt from {event_data.get('source_ip', 'unknown')}",
            ThreatType.DDoS: f"Potential DDoS attack from {event_data.get('source_ip', 'unknown')}",
            ThreatType.DATA_BREACH: "Potential data exfiltration detected",
        }

        return descriptions.get(
            threat_type, f"Security threat of type {threat_type.value} detected"
        )

    def _extract_indicators(self, event_data: dict[str, Any]) -> list[str]:
        """Extract threat indicators from event data."""
        indicators = []

        if source_ip := event_data.get("source_ip"):
            indicators.append(f"ip:{source_ip}")

        if user_agent := event_data.get("user_agent"):
            indicators.append(f"user_agent:{user_agent}")

        if url := event_data.get("url"):
            indicators.append(f"url:{url}")

        return indicators

    def _trigger_alert(self, threat: DigitalThreat) -> None:
        """Trigger security alerts for detected threats."""
        alert = SecurityAlert(
            alert_id=f"alert_{threat.threat_id}",
            title=threat.threat_type.name
            if hasattr(threat.threat_type, "name")
            else str(threat.threat_type),
            description=threat.description
            or f"Digital threat detected: {threat.threat_type.value}",
            severity=threat.severity,
            category=SecurityEventCategory.NETWORK_ACTIVITY,
        )

        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        # Log the alert
        self.logger.warning(
            f"SECURITY ALERT: {threat.threat_type.value} - {threat.description}"
        )

        # Auto-mitigation for certain threat types
        self._auto_mitigate_threat(threat)

    def _auto_mitigate_threat(self, threat: DigitalThreat) -> None:
        """Automatically mitigate certain types of threats."""
        if threat.source_ip and threat.severity in [
            ThreatLevel.HIGH,
            ThreatLevel.CRITICAL,
        ]:
            self.block_ip(threat.source_ip, duration_hours=24)

        if threat.threat_type == ThreatType.DDoS and threat.source_ip:
            self.block_ip(threat.source_ip, duration_hours=1)

        if threat.threat_type in [ThreatType.MALWARE, ThreatType.RANSOMWARE]:
            # In a real implementation, this might quarantine affected systems
            self.logger.critical(f"Critical threat detected: {threat.description}")

    # Network Security
    def monitor_network_connection(self, connection: NetworkConnection) -> bool:
        """Monitor a network connection for security issues."""
        self.network_connections[connection.connection_id] = connection

        # Check against blocked IPs
        if (
            connection.source_ip in self.blocked_ips
            or connection.destination_ip in self.blocked_ips
        ):
            self.logger.warning(
                f"Connection blocked: {connection.source_ip} -> {connection.destination_ip}"
            )
            return False

        # Check for suspicious ports
        suspicious_ports = [22, 23, 135, 139, 445, 1433, 3389]  # Common attack vectors
        if connection.destination_port in suspicious_ports:
            self.detect_threat(
                {
                    "event_type": "suspicious_port_access",
                    "source_ip": connection.source_ip,
                    "destination_port": connection.destination_port,
                    "protocol": connection.protocol,
                }
            )

        return True

    def block_ip(self, ip_address: str, duration_hours: int = 24) -> bool:
        """Block an IP address for a specified duration."""
        try:
            self.blocked_ips.add(ip_address)

            # Record when the block should lapse; consumers (or a real
            # scheduler) can expire entries from this mapping.
            self.ip_unblock_schedule[ip_address] = datetime.now() + timedelta(
                hours=duration_hours
            )

            self.logger.info(f"Blocked IP {ip_address} for {duration_hours} hours")
            return True
        except Exception as e:
            self.logger.error(f"Error blocking IP {ip_address}: {e}")
            return False

    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self.logger.info(f"Unblocked IP {ip_address}")
            return True
        return False

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP address is blocked."""
        return ip_address in self.blocked_ips

    # Encryption and Key Management
    def encrypt_data(
        self,
        data: str | bytes,
        key_id: str = "master_key",
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
    ) -> bytes | None:
        """Encrypt data using specified algorithm and key."""
        try:
            if key_id not in self.encryption_keys:
                self.logger.error(f"Encryption key {key_id} not found")
                return None

            key = self.encryption_keys[key_id]

            if isinstance(data, str):
                data = data.encode("utf-8")

            if algorithm == EncryptionAlgorithm.FERNET:
                f = Fernet(key)
                return f.encrypt(data)
            else:
                self.logger.warning(
                    f"Encryption algorithm {algorithm.value} is unsupported"
                )
                return None

        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            return None

    def decrypt_data(
        self,
        encrypted_data: bytes,
        key_id: str = "master_key",
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
    ) -> bytes | None:
        """Decrypt data using specified algorithm and key."""
        try:
            if key_id not in self.encryption_keys:
                self.logger.error(f"Decryption key {key_id} not found")
                return None

            key = self.encryption_keys[key_id]

            if algorithm == EncryptionAlgorithm.FERNET:
                f = Fernet(key)
                return f.decrypt(encrypted_data)
            else:
                self.logger.warning(
                    f"Decryption algorithm {algorithm.value} is unsupported"
                )
                return None

        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return None

    def rotate_encryption_key(self, key_id: str) -> bool:
        """Rotate an encryption key."""
        try:
            # Generate new key
            new_key = Fernet.generate_key()

            # Store old key for decryption of existing data
            old_key_id = f"{key_id}_old_{datetime.now().strftime('%Y%m%d')}"
            self.encryption_keys[old_key_id] = self.encryption_keys[key_id]

            # Update to new key
            self.encryption_keys[key_id] = new_key

            # Schedule next rotation
            self.key_rotation_schedule[key_id] = datetime.now() + timedelta(
                days=self.config.get("encryption_key_rotation_days", 90)
            )

            self.logger.info(f"Rotated encryption key: {key_id}")
            return True

        except Exception as e:
            self.logger.error(f"Key rotation error: {e}")
            return False

    def check_key_rotation_schedule(self) -> None:
        """Check if any keys need rotation."""
        current_time = datetime.now()

        for key_id, rotation_time in self.key_rotation_schedule.items():
            if current_time >= rotation_time:
                self.rotate_encryption_key(key_id)

    # Authentication and Authorization
    def create_jwt_token(
        self, user_id: str, permissions: list[str], expires_in_hours: int = 24
    ) -> str | None:
        """Create a JWT token for user authentication."""
        try:
            payload = {
                "user_id": user_id,
                "permissions": permissions,
                "iat": datetime.now(UTC).replace(tzinfo=None),
                "exp": datetime.now(UTC).replace(tzinfo=None)
                + timedelta(hours=expires_in_hours),
            }

            secret = self.encryption_keys.get(
                "jwt_secret", self.encryption_keys["master_key"]
            )
            token = jwt.encode(payload, secret, algorithm="HS256")

            return token

        except Exception as e:
            self.logger.error(f"JWT token creation error: {e}")
            return None

    def verify_jwt_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode a JWT token."""
        try:
            secret = self.encryption_keys.get(
                "jwt_secret", self.encryption_keys["master_key"]
            )
            payload = jwt.decode(token, secret, algorithms=["HS256"])

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None

    def hash_password(
        self, password: str, salt: bytes | None = None
    ) -> tuple[bytes, bytes]:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_bytes(16)

        password_hash, _ = self.security_utils.hash_password(password, salt)
        return password_hash, salt

    def verify_password(self, password: str, password_hash: bytes, salt: bytes) -> bool:
        """Verify a password against its hash."""
        return self.security_utils.verify_password(password, password_hash, salt)

    # Rate Limiting
    def check_rate_limit(self, identifier: str, limit_per_minute: int = 60) -> bool:
        """Check if an identifier (IP, user, etc.) exceeds rate limit."""
        current_time = datetime.now()

        # Initialize rate limit tracking for identifier
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {"requests": [], "blocked_until": None}

        rate_data = self.rate_limits[identifier]

        # Check if currently blocked
        if rate_data["blocked_until"] and current_time < rate_data["blocked_until"]:
            return False

        # Clean old requests (older than 1 minute)
        minute_ago = current_time - timedelta(minutes=1)
        rate_data["requests"] = [
            req_time for req_time in rate_data["requests"] if req_time > minute_ago
        ]

        # Check rate limit
        if len(rate_data["requests"]) >= limit_per_minute:
            # Block for 5 minutes
            rate_data["blocked_until"] = current_time + timedelta(minutes=5)
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False

        # Add current request
        rate_data["requests"].append(current_time)
        return True

    # Vulnerability Management
    def run_vulnerability_scan(self, target_system: str) -> VulnerabilityReport:
        """Run a vulnerability scan on a target system."""
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        scanner = self.config.get("vulnerability_scanner")
        if not callable(scanner):
            raise RuntimeError(
                "A callable vulnerability_scanner must be configured before scanning"
            )
        vulnerabilities = scanner(target_system)
        if not isinstance(vulnerabilities, list):
            raise TypeError(
                "vulnerability_scanner must return a list of vulnerability records"
            )

        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)

        # Generate recommendations
        recommendations = self._generate_vulnerability_recommendations(vulnerabilities)

        # Check compliance
        compliance_status = self._check_compliance_status(vulnerabilities)

        report = VulnerabilityReport(
            scan_id=scan_id,
            target_system=target_system,
            scan_date=datetime.now(),
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            recommendations=recommendations,
            compliance_status=compliance_status,
        )

        self.vulnerability_reports[scan_id] = report
        return report

    def _calculate_risk_score(self, vulnerabilities: list[dict[str, Any]]) -> float:
        """Calculate overall risk score from vulnerabilities."""
        if not vulnerabilities:
            return 0.0

        total_score = sum(vuln.get("cvss_score", 0) for vuln in vulnerabilities)
        return min(10.0, cast(float, total_score / len(vulnerabilities)))

    def _generate_vulnerability_recommendations(
        self, vulnerabilities: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on vulnerabilities."""
        recommendations = []

        high_severity_count = len(
            [v for v in vulnerabilities if v.get("severity") == "HIGH"]
        )
        if high_severity_count > 0:
            recommendations.append(
                f"Immediately address {high_severity_count} high-severity vulnerabilities"
            )

        patchable_count = len([v for v in vulnerabilities if v.get("patch_available")])
        if patchable_count > 0:
            recommendations.append(
                f"Apply patches for {patchable_count} vulnerabilities"
            )

        exploitable_count = len(
            [v for v in vulnerabilities if v.get("exploit_available")]
        )
        if exploitable_count > 0:
            recommendations.append(
                f"Prioritize {exploitable_count} vulnerabilities with available exploits"
            )

        return recommendations

    def _check_compliance_status(
        self, vulnerabilities: list[dict[str, Any]]
    ) -> dict[str, bool]:
        """Check compliance status based on vulnerabilities."""
        # Simplified compliance checking
        high_severity_vulns = [
            v for v in vulnerabilities if v.get("severity") == "HIGH"
        ]
        critical_severity_vulns = [
            v for v in vulnerabilities if v.get("severity") == "CRITICAL"
        ]

        return {
            "pci_dss": len(critical_severity_vulns) == 0,
            "iso_27001": len(high_severity_vulns) <= 2,
            "nist": len(vulnerabilities) <= 10,
            "sox": len(critical_severity_vulns) == 0 and len(high_severity_vulns) <= 1,
        }

    # Monitoring and Reporting
    def start_monitoring(self) -> None:
        """Start digital security monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True

            # Start different monitoring threads
            threads = [
                threading.Thread(
                    target=self._threat_monitoring_loop, name="ThreatMonitor"
                ),
                threading.Thread(
                    target=self._network_monitoring_loop, name="NetworkMonitor"
                ),
                threading.Thread(target=self._key_rotation_loop, name="KeyRotation"),
                threading.Thread(
                    target=self._vulnerability_scan_loop, name="VulnScanner"
                ),
            ]

            for thread in threads:
                thread.daemon = True
                thread.start()
                self.monitoring_threads.append(thread)

            self.logger.info("Digital security monitoring started")

    def stop_monitoring(self) -> None:
        """Stop digital security monitoring."""
        self.monitoring_active = False

        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5)

        self.monitoring_threads.clear()
        self.logger.info("Digital security monitoring stopped")

    def _threat_monitoring_loop(self) -> None:
        """Main threat monitoring loop."""
        while self.monitoring_active:
            try:
                self._process_security_events()
                self._update_threat_intelligence()
                time.sleep(self.config.get("monitoring_interval", 10))
            except Exception as e:
                self.logger.error(f"Error in threat monitoring: {e}")

    def _network_monitoring_loop(self) -> None:
        """Network monitoring loop."""
        while self.monitoring_active:
            try:
                self._monitor_network_traffic()
                self._check_blocked_ips()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in network monitoring: {e}")

    def _key_rotation_loop(self) -> None:
        """Key rotation monitoring loop."""
        while self.monitoring_active:
            try:
                self.check_key_rotation_schedule()
                time.sleep(3600)  # Check every hour
            except Exception as e:
                self.logger.error(f"Error in key rotation: {e}")

    def _vulnerability_scan_loop(self) -> None:
        """Vulnerability scanning loop."""
        while self.monitoring_active:
            try:
                # In a real implementation, scan different systems periodically
                scan_interval = self.config.get("vulnerability_scan_interval_hours", 24)
                time.sleep(scan_interval * 3600)
            except Exception as e:
                self.logger.error(f"Error in vulnerability scanning: {e}")

    def _process_security_events(self) -> None:
        """Process accumulated security events."""
        # Clean up old events
        retention_days = self.config.get("threat_retention_days", 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        self.security_events = [
            event for event in self.security_events if event.timestamp > cutoff_date
        ]

    def _update_threat_intelligence(self) -> None:
        """Update threat intelligence feeds by refreshing known threat indicators."""
        current_time = datetime.now()
        # Update threat intelligence age tracker
        if not hasattr(self, "_threat_intel_last_updated"):
            self._threat_intel_last_updated = current_time

        # Check if any recent security events suggest new threat patterns
        recent_events = [
            e
            for e in self.security_events
            if (current_time - e.timestamp).total_seconds() < 3600
        ]

        # Aggregate threat indicators from recent events
        threat_ips = set()
        for event in recent_events:
            ip = event.metadata.get("source_ip")
            if ip and event.event_type in (
                "intrusion_attempt",
                "brute_force",
                "unauthorized_access",
            ):
                threat_ips.add(ip)

        # Add newly identified threat IPs to blocked list
        for ip in threat_ips:
            if ip not in self.blocked_ips:
                self.blocked_ips.add(ip)

        self._threat_intel_last_updated = current_time

    def _monitor_network_traffic(self) -> None:
        """Monitor ongoing network traffic."""
        # Clean up old connections
        current_time = datetime.now()
        old_connections = [
            conn_id
            for conn_id, conn in self.network_connections.items()
            if current_time - conn.timestamp > timedelta(hours=1)
        ]

        for conn_id in old_connections:
            del self.network_connections[conn_id]

    def _check_blocked_ips(self) -> None:
        """Check if any blocked IPs should be unblocked based on block duration."""
        if not hasattr(self, "_ip_block_times"):
            self._ip_block_times: dict[str, datetime] = {}

        current_time = datetime.now()
        block_duration_hours = getattr(self, "block_duration_hours", 24)
        ips_to_unblock = []

        for ip in self.blocked_ips:
            block_time = self._ip_block_times.get(ip)
            if (
                block_time
                and (current_time - block_time).total_seconds()
                > block_duration_hours * 3600
            ):
                ips_to_unblock.append(ip)

        for ip in ips_to_unblock:
            self.blocked_ips.discard(ip)
            self._ip_block_times.pop(ip, None)

    # API and Event Management
    def log_security_event(
        self, event_type: SecurityEventType, metadata: dict[str, Any]
    ) -> SecurityEvent:
        """Log a security event."""
        event = SecurityEvent(
            event_id=f"event_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.security_events)}",
            event_type=event_type.value,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        self.security_events.append(event)

        # Check for threats in this event
        self.detect_threat(metadata)

        return event

    def get_security_report(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate a comprehensive security report."""
        # Filter events and threats by date range
        relevant_events = [
            event
            for event in self.security_events
            if start_date <= event.timestamp <= end_date
        ]

        relevant_threats = [
            threat
            for threat in self.active_threats.values()
            if start_date <= threat.detected_at <= end_date
        ]

        # Analyze threat types
        threat_analysis: dict[str, int] = defaultdict(int)
        for threat in relevant_threats:
            threat_analysis[threat.threat_type.value] += 1

        # Analyze event types
        event_analysis: dict[str, int] = defaultdict(int)
        for event in relevant_events:
            event_analysis[event.event_type] += 1

        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(relevant_events),
                "total_threats": len(relevant_threats),
                "blocked_ips": len(self.blocked_ips),
                "active_policies": len(
                    [p for p in self.security_policies.values() if p.active]
                ),
            },
            "threat_analysis": dict(threat_analysis),
            "event_analysis": dict(event_analysis),
            "top_threat_sources": self._get_top_threat_sources(relevant_threats),
            "vulnerability_summary": self._get_vulnerability_summary(),
            "recommendations": self._generate_security_recommendations(),
        }

        return report

    def _get_top_threat_sources(
        self, threats: list[DigitalThreat], top_n: int = 10
    ) -> list[dict[str, Any]]:
        """Get top threat sources by IP."""
        source_counts: dict[str, int] = defaultdict(int)
        for threat in threats:
            if threat.source_ip:
                source_counts[threat.source_ip] += 1

        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        return [{"ip": ip, "threat_count": count} for ip, count in top_sources]

    def _get_vulnerability_summary(self) -> dict[str, Any]:
        """Get summary of vulnerability reports."""
        if not self.vulnerability_reports:
            return {"total_reports": 0}

        latest_report = max(
            self.vulnerability_reports.values(), key=lambda r: r.scan_date
        )

        return {
            "total_reports": len(self.vulnerability_reports),
            "latest_scan_date": latest_report.scan_date.isoformat(),
            "latest_risk_score": latest_report.risk_score,
            "total_vulnerabilities": len(latest_report.vulnerabilities),
        }

    def _generate_security_recommendations(self) -> list[str]:
        """Generate security recommendations."""
        recommendations = []

        # Check threat levels
        high_threats = [
            t for t in self.active_threats.values() if t.severity == ThreatLevel.HIGH
        ]
        critical_threats = [
            t
            for t in self.active_threats.values()
            if t.severity == ThreatLevel.CRITICAL
        ]

        if critical_threats:
            recommendations.append(
                f"Immediately address {len(critical_threats)} critical security threats"
            )

        if high_threats:
            recommendations.append(
                f"Review and mitigate {len(high_threats)} high-severity threats"
            )

        # Check blocked IPs
        if len(self.blocked_ips) > 100:
            recommendations.append(
                "Review blocked IP list - consider implementing more sophisticated filtering"
            )

        # Check key rotation
        overdue_keys = [
            key_id
            for key_id, rotation_time in self.key_rotation_schedule.items()
            if datetime.now() > rotation_time
        ]
        if overdue_keys:
            recommendations.append(
                f"Rotate overdue encryption keys: {', '.join(overdue_keys)}"
            )

        return recommendations

    def add_alert_callback(self, callback: Callable[[SecurityAlert], None]) -> None:
        """Add a callback function for security alerts."""
        self.alert_callbacks.append(callback)

    def get_active_threats(self) -> list[DigitalThreat]:
        """Get all active threats."""
        return [
            threat for threat in self.active_threats.values() if not threat.mitigated
        ]

    def mitigate_threat(self, threat_id: str, mitigation_notes: str = "") -> bool:
        """Mark a threat as mitigated."""
        if threat_id in self.active_threats:
            self.active_threats[threat_id].mitigated = True
            self.active_threats[threat_id].metadata["mitigation_notes"] = (
                mitigation_notes
            )
            self.active_threats[threat_id].metadata["mitigated_at"] = (
                datetime.now().isoformat()
            )
            return True
        return False
