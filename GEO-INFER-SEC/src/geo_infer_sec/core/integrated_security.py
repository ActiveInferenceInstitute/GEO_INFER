"""
Integrated Security Orchestrator for GEO-INFER-SEC

This module provides holistic security management by integrating:
- Physical Security systems and controls
- Digital Security monitoring and protection
- Cognitive Security AI-driven analysis
- Cross-domain threat correlation and response
- Unified security dashboard and reporting
- Coordinated incident response across domains
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field
import yaml

from .physical_security import PhysicalSecurityManager, PhysicalThreat
from .digital_security import DigitalSecurityManager
from .cognitive_security import CognitiveSecurityManager
from ..models.security_models import ThreatLevel, SecurityAlert
from ..utils.security_utils import SecurityUtils


class SecurityDomain(Enum):
    """Security domains."""

    PHYSICAL = "physical"
    DIGITAL = "digital"
    COGNITIVE = "cognitive"
    INTEGRATED = "integrated"


class ThreatCorrelationType(Enum):
    """Types of threat correlation."""

    TEMPORAL = "temporal"  # Threats occurring at similar times
    SPATIAL = "spatial"  # Threats from similar locations/IPs
    BEHAVIORAL = "behavioral"  # Threats with similar patterns
    MULTI_DOMAIN = "multi_domain"  # Threats spanning multiple domains


class IncidentSeverity(Enum):
    """Incident severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class IntegratedThreat:
    """Represents a threat that spans multiple security domains."""

    threat_id: str
    primary_domain: SecurityDomain
    affected_domains: list[SecurityDomain]
    correlation_type: ThreatCorrelationType
    component_threats: dict[str, Any]  # Maps domain to threat objects
    combined_severity: IncidentSeverity
    confidence_score: float
    attack_chain: list[str]
    impact_assessment: dict[str, Any]
    recommended_response: list[str]
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncident:
    """Represents a comprehensive security incident."""

    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: str  # open, investigating, contained, resolved
    affected_systems: list[str]
    threat_vectors: list[str]
    timeline: list[dict[str, Any]]
    response_actions: list[dict[str, Any]]
    assigned_team: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IntegratedSecurityManager:
    """Holistic security management system integrating all security domains."""

    def __init__(self, config_path: str | None = None):
        """Initialize the integrated security manager."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Initialize domain managers
        self.physical_manager = PhysicalSecurityManager(config_path)
        self.digital_manager = DigitalSecurityManager(config_path)
        self.cognitive_manager = CognitiveSecurityManager(config_path)

        # Initialize integration components
        self.integrated_threats: dict[str, IntegratedThreat] = {}
        self.security_incidents: dict[str, SecurityIncident] = {}
        self.correlation_rules: list[dict[str, Any]] = []

        # Initialize monitoring and coordination
        self.orchestration_active = False
        self.orchestration_threads: list[threading.Thread] = []
        self.alert_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self.response_handlers: dict[str, Callable[[IntegratedThreat], None]] = {}

        # Security metrics and KPIs
        self.security_metrics: dict[str, Any] = {}
        self.performance_history: list[dict[str, Any]] = []

        # Initialize utilities
        self.security_utils = SecurityUtils()

        # Setup cross-domain alert handling
        self._setup_cross_domain_alerts()
        self._initialize_correlation_rules()

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration for integrated security."""
        default_config = {
            "correlation_window_minutes": 30,
            "threat_correlation_threshold": 0.7,
            "incident_auto_escalation": True,
            "response_coordination": True,
            "metrics_retention_days": 90,
            "dashboard_update_interval": 30,
            "emergency_notification": True,
        }

        if config_path:
            try:
                with open(config_path) as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config.get("integrated_security", {}))
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")

        return default_config

    def _setup_cross_domain_alerts(self) -> None:
        """Setup alert handlers for cross-domain coordination."""
        # Setup callbacks for each domain manager
        self.physical_manager.add_alert_callback(self._handle_physical_alert)
        self.digital_manager.add_alert_callback(self._handle_digital_alert)
        self.cognitive_manager.add_alert_callback(self._handle_cognitive_alert)

    def _initialize_correlation_rules(self) -> None:
        """Initialize threat correlation rules."""
        self.correlation_rules = [
            {
                "name": "Physical Breach + Digital Access",
                "conditions": {
                    "physical": ["intrusion", "unauthorized_access"],
                    "digital": ["login_attempt", "data_access"],
                    "time_window": 900,  # 15 minutes
                },
                "severity_escalation": 2,
                "response": ["lockdown_systems", "investigate_correlation"],
            },
            {
                "name": "Cognitive + Digital Anomaly",
                "conditions": {
                    "cognitive": ["behavioral_anomaly"],
                    "digital": ["unusual_network_traffic", "data_exfiltration"],
                    "time_window": 1800,  # 30 minutes
                },
                "severity_escalation": 1,
                "response": ["enhanced_monitoring", "user_verification"],
            },
            {
                "name": "Multi-Vector Attack",
                "conditions": {
                    "physical": ["surveillance_anomaly"],
                    "digital": ["malware_detection", "phishing"],
                    "cognitive": ["social_engineering"],
                    "time_window": 3600,  # 1 hour
                },
                "severity_escalation": 3,
                "response": ["emergency_response", "isolate_affected_systems"],
            },
        ]

    # Alert Handling
    def _handle_physical_alert(self, threat: PhysicalThreat) -> None:
        """Handle alerts from physical security domain."""
        self.logger.info(f"Received physical security alert: {threat.threat_id}")
        self._correlate_cross_domain_threat(SecurityDomain.PHYSICAL, threat)

    def _handle_digital_alert(self, alert: SecurityAlert) -> None:
        """Handle alerts from digital security domain."""
        self.logger.info(f"Received digital security alert: {alert.alert_id}")
        self._correlate_cross_domain_threat(SecurityDomain.DIGITAL, alert)

    def _handle_cognitive_alert(self, alert: dict[str, Any]) -> None:
        """Handle alerts from cognitive security domain."""
        self.logger.info(f"Received cognitive security alert: {alert.get('type')}")
        self._correlate_cross_domain_threat(SecurityDomain.COGNITIVE, alert)

    def _correlate_cross_domain_threat(
        self, domain: SecurityDomain, threat_data: Any
    ) -> None:
        """Correlate threats across security domains."""
        try:
            # Look for related threats in other domains
            correlations = self._find_threat_correlations(domain, threat_data)

            if correlations:
                # Create integrated threat
                integrated_threat = self._create_integrated_threat(
                    domain, threat_data, correlations
                )
                self.integrated_threats[integrated_threat.threat_id] = integrated_threat

                # Trigger coordinated response
                self._trigger_coordinated_response(integrated_threat)

                # Create or update security incident
                self._manage_security_incident(integrated_threat)

        except Exception as e:
            self.logger.error(f"Error in threat correlation: {e}")

    def _find_threat_correlations(
        self, primary_domain: SecurityDomain, threat_data: Any
    ) -> list[tuple[SecurityDomain, Any]]:
        """Find correlations with threats in other domains."""
        correlations = []
        current_time = datetime.now()
        correlation_window = timedelta(
            minutes=self.config["correlation_window_minutes"]
        )

        # Check each correlation rule
        for rule in self.correlation_rules:
            if self._matches_correlation_rule(
                primary_domain, threat_data, rule, current_time, correlation_window
            ):
                # Find matching threats in other domains
                other_correlations = self._find_matching_threats(
                    rule, current_time, correlation_window
                )
                correlations.extend(other_correlations)

        return correlations

    def _matches_correlation_rule(
        self,
        domain: SecurityDomain,
        threat_data: Any,
        rule: dict[str, Any],
        current_time: datetime,
        time_window: timedelta,
    ) -> bool:
        """Check if a threat matches a correlation rule."""
        domain_conditions = rule["conditions"].get(domain.value, [])
        if not domain_conditions:
            return False

        # Extract threat type from threat data
        threat_type = self._extract_threat_type(domain, threat_data)

        return threat_type in domain_conditions

    def _extract_threat_type(self, domain: SecurityDomain, threat_data: Any) -> str:
        """Extract threat type from domain-specific threat data."""
        if domain == SecurityDomain.PHYSICAL:
            return getattr(threat_data, "threat_type", "unknown")
        elif domain == SecurityDomain.DIGITAL:
            threat_type = getattr(threat_data, "threat_type", "unknown")
            return cast(str, getattr(threat_type, "value", threat_type))
        elif domain == SecurityDomain.COGNITIVE:
            if isinstance(threat_data, dict):
                return cast(str, threat_data.get("type", "unknown"))
            return cast(str, getattr(threat_data, "threat_type", "unknown"))

        return "unknown"

    def _find_matching_threats(
        self, rule: dict[str, Any], current_time: datetime, time_window: timedelta
    ) -> list[tuple[SecurityDomain, Any]]:
        """Find threats matching correlation rule in other domains."""
        matches: list[tuple[SecurityDomain, Any]] = []

        # Check physical threats
        if "physical" in rule["conditions"]:
            physical_threats = self.physical_manager.get_active_threats()
            for phys_threat in physical_threats:
                if (
                    current_time - phys_threat.detected_at <= time_window
                    and phys_threat.threat_type in rule["conditions"]["physical"]
                ):
                    matches.append((SecurityDomain.PHYSICAL, phys_threat))

        # Check digital threats
        if "digital" in rule["conditions"]:
            digital_threats = self.digital_manager.get_active_threats()
            for dig_threat in digital_threats:
                if (
                    current_time - dig_threat.detected_at <= time_window
                    and dig_threat.threat_type.value in rule["conditions"]["digital"]
                ):
                    matches.append((SecurityDomain.DIGITAL, dig_threat))

        # Check cognitive threats
        if "cognitive" in rule["conditions"]:
            cognitive_threats = self.cognitive_manager.get_cognitive_threats()
            for cog_threat in cognitive_threats:
                if (
                    current_time - cog_threat.detected_at <= time_window
                    and cog_threat.threat_type in rule["conditions"]["cognitive"]
                ):
                    matches.append((SecurityDomain.COGNITIVE, cog_threat))

        return matches

    def _create_integrated_threat(
        self,
        primary_domain: SecurityDomain,
        primary_threat: Any,
        correlations: list[tuple[SecurityDomain, Any]],
    ) -> IntegratedThreat:
        """Create an integrated threat from correlated threats."""
        threat_id = f"integrated_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Determine affected domains
        affected_domains = [primary_domain]
        component_threats = {primary_domain.value: primary_threat}

        for domain, threat in correlations:
            if domain not in affected_domains:
                affected_domains.append(domain)
            component_threats[domain.value] = threat

        # Determine correlation type
        correlation_type = self._determine_correlation_type(
            primary_domain, correlations
        )

        # Calculate combined severity
        combined_severity = self._calculate_combined_severity(
            primary_threat, correlations
        )

        # Generate attack chain
        attack_chain = self._reconstruct_attack_chain(
            primary_domain, primary_threat, correlations
        )

        # Assess impact
        impact_assessment = self._assess_impact(affected_domains, component_threats)

        # Generate response recommendations
        recommended_response = self._generate_response_recommendations(
            combined_severity, affected_domains, correlation_type
        )
        # Calculate confidence score based on correlation strength
        base_confidence = 0.6
        if correlation_type == ThreatCorrelationType.MULTI_DOMAIN:
            base_confidence = 0.75

        # Add bonus for number of correlations (up to 0.95 max)
        correlation_bonus = min(0.2, len(correlations) * 0.05)
        calculated_confidence = min(0.95, base_confidence + correlation_bonus)

        return IntegratedThreat(
            threat_id=threat_id,
            primary_domain=primary_domain,
            affected_domains=affected_domains,
            correlation_type=correlation_type,
            component_threats=component_threats,
            combined_severity=combined_severity,
            confidence_score=calculated_confidence,
            attack_chain=attack_chain,
            impact_assessment=impact_assessment,
            recommended_response=recommended_response,
        )

    def _determine_correlation_type(
        self,
        primary_domain: SecurityDomain,
        correlations: list[tuple[SecurityDomain, Any]],
    ) -> ThreatCorrelationType:
        """Determine the type of threat correlation."""
        if len(set(domain for domain, _ in correlations)) > 1:
            return ThreatCorrelationType.MULTI_DOMAIN

        # For now, default to temporal correlation
        return ThreatCorrelationType.TEMPORAL

    def _calculate_combined_severity(
        self, primary_threat: Any, correlations: list[tuple[SecurityDomain, Any]]
    ) -> IncidentSeverity:
        """Calculate combined severity from multiple threats."""
        # Start with primary threat severity
        base_severity = self._get_threat_severity(primary_threat)

        # Escalate based on correlations
        escalation_factor = len(correlations)

        severity_levels = [
            IncidentSeverity.LOW,
            IncidentSeverity.MEDIUM,
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        ]

        current_index = severity_levels.index(base_severity)
        new_index = min(len(severity_levels) - 1, current_index + escalation_factor)

        return severity_levels[new_index]

    def _get_threat_severity(self, threat: Any) -> IncidentSeverity:
        """Get severity level from a threat object."""
        if hasattr(threat, "severity"):
            severity_map = {
                ThreatLevel.LOW: IncidentSeverity.LOW,
                ThreatLevel.MEDIUM: IncidentSeverity.MEDIUM,
                ThreatLevel.HIGH: IncidentSeverity.HIGH,
                ThreatLevel.CRITICAL: IncidentSeverity.CRITICAL,
            }
            return severity_map.get(threat.severity, IncidentSeverity.MEDIUM)

        return IncidentSeverity.MEDIUM

    def _reconstruct_attack_chain(
        self,
        primary_domain: SecurityDomain,
        primary_threat: Any,
        correlations: list[tuple[SecurityDomain, Any]],
    ) -> list[str]:
        """Reconstruct the attack chain from correlated threats."""
        chain = [
            f"{primary_domain.value}: {self._extract_threat_type(primary_domain, primary_threat)}"
        ]

        for domain, threat in correlations:
            threat_type = self._extract_threat_type(domain, threat)
            chain.append(f"{domain.value}: {threat_type}")

        return chain

    def _assess_impact(
        self, domains: list[SecurityDomain], threats: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess the impact of integrated threats."""
        impact = {
            "affected_domains": len(domains),
            "confidentiality_impact": "MEDIUM",
            "integrity_impact": "MEDIUM",
            "availability_impact": "MEDIUM",
            "business_impact": "MEDIUM",
        }

        # Enhance impact assessment based on specific domains
        if SecurityDomain.PHYSICAL in domains:
            impact["physical_security_impact"] = "HIGH"

        if SecurityDomain.DIGITAL in domains:
            impact["data_security_impact"] = "HIGH"

        if SecurityDomain.COGNITIVE in domains:
            impact["behavioral_security_impact"] = "HIGH"

        return impact

    def _generate_response_recommendations(
        self,
        severity: IncidentSeverity,
        domains: list[SecurityDomain],
        correlation_type: ThreatCorrelationType,
    ) -> list[str]:
        """Generate coordinated response recommendations."""
        recommendations = []

        # Base recommendations by severity
        if severity in [IncidentSeverity.CRITICAL, IncidentSeverity.EMERGENCY]:
            recommendations.extend(
                [
                    "Activate incident response team immediately",
                    "Implement emergency containment procedures",
                    "Notify executive leadership and legal team",
                ]
            )
        elif severity == IncidentSeverity.HIGH:
            recommendations.extend(
                [
                    "Escalate to security operations center",
                    "Implement containment measures",
                    "Begin detailed investigation",
                ]
            )

        # Domain-specific recommendations
        if SecurityDomain.PHYSICAL in domains:
            recommendations.extend(
                [
                    "Secure physical premises",
                    "Review access control logs",
                    "Deploy additional security personnel if needed",
                ]
            )

        if SecurityDomain.DIGITAL in domains:
            recommendations.extend(
                [
                    "Isolate affected digital systems",
                    "Analyze network traffic for IOCs",
                    "Implement additional network monitoring",
                ]
            )

        if SecurityDomain.COGNITIVE in domains:
            recommendations.extend(
                [
                    "Review user behavior patterns",
                    "Implement enhanced user monitoring",
                    "Consider user re-authentication",
                ]
            )

        return recommendations

    def _trigger_coordinated_response(
        self, integrated_threat: IntegratedThreat
    ) -> None:
        """Trigger coordinated response across security domains."""
        self.logger.warning(
            f"Triggering coordinated response for {integrated_threat.threat_id}"
        )

        # Execute response handlers
        for action in integrated_threat.recommended_response:
            if action in self.response_handlers:
                try:
                    self.response_handlers[action](integrated_threat)
                except Exception as e:
                    self.logger.error(f"Error executing response action {action}: {e}")

        # Notify stakeholders
        self._notify_stakeholders(integrated_threat)

    def _manage_security_incident(self, integrated_threat: IntegratedThreat) -> None:
        """Create or update security incident for integrated threat."""
        incident_id = f"incident_{integrated_threat.threat_id}"

        # Create new incident
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"Integrated Security Threat: {integrated_threat.primary_domain.value}",
            description=f"Multi-domain security threat affecting {len(integrated_threat.affected_domains)} domains",
            severity=integrated_threat.combined_severity,
            status="open",
            affected_systems=[
                domain.value for domain in integrated_threat.affected_domains
            ],
            threat_vectors=integrated_threat.attack_chain,
            timeline=[
                {
                    "timestamp": integrated_threat.detected_at.isoformat(),
                    "event": "Integrated threat detected",
                    "details": f"Correlation type: {integrated_threat.correlation_type.value}",
                }
            ],
            response_actions=[
                {
                    "action": action,
                    "status": "recommended",
                    "timestamp": datetime.now().isoformat(),
                }
                for action in integrated_threat.recommended_response
            ],
        )

        self.security_incidents[incident_id] = incident

        # Auto-escalate if configured
        if self.config[
            "incident_auto_escalation"
        ] and integrated_threat.combined_severity in [
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        ]:
            self._escalate_incident(incident)

    def _notify_stakeholders(self, integrated_threat: IntegratedThreat) -> None:
        """Notify relevant stakeholders of integrated threats."""
        notification_data = {
            "threat_id": integrated_threat.threat_id,
            "severity": integrated_threat.combined_severity.value,
            "affected_domains": [d.value for d in integrated_threat.affected_domains],
            "attack_chain": integrated_threat.attack_chain,
            "recommended_response": integrated_threat.recommended_response,
        }

        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(notification_data)
            except Exception as e:
                self.logger.error(f"Error in stakeholder notification: {e}")

    def _escalate_incident(self, incident: SecurityIncident) -> None:
        """Escalate security incident to higher authorities."""
        incident.timeline.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "Incident escalated",
                "details": f"Auto-escalated due to {incident.severity.value} severity",
            }
        )

        self.logger.critical(
            f"INCIDENT ESCALATED: {incident.incident_id} - {incident.title}"
        )

    # Management and Monitoring
    def start_integrated_monitoring(self) -> None:
        """Start integrated security monitoring and orchestration."""
        if not self.orchestration_active:
            self.orchestration_active = True

            # Start individual domain managers
            self.physical_manager.start_monitoring()
            self.digital_manager.start_monitoring()
            self.cognitive_manager.start_cognitive_monitoring()

            # Start orchestration threads
            threads = [
                threading.Thread(
                    target=self._orchestration_loop, name="SecurityOrchestration"
                ),
                threading.Thread(
                    target=self._metrics_collection_loop, name="MetricsCollection"
                ),
                threading.Thread(
                    target=self._incident_management_loop, name="IncidentManagement"
                ),
            ]

            for thread in threads:
                thread.daemon = True
                thread.start()
                self.orchestration_threads.append(thread)

            self.logger.info("Integrated security monitoring started")

    def stop_integrated_monitoring(self) -> None:
        """Stop integrated security monitoring."""
        self.orchestration_active = False

        # Stop individual domain managers
        self.physical_manager.stop_monitoring()
        self.digital_manager.stop_monitoring()
        self.cognitive_manager.stop_cognitive_monitoring()

        # Stop orchestration threads
        for thread in self.orchestration_threads:
            thread.join(timeout=5)

        self.orchestration_threads.clear()
        self.logger.info("Integrated security monitoring stopped")

    def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self.orchestration_active:
            try:
                self._update_security_posture()
                self._cleanup_old_threats()
                time.sleep(60)  # Run every minute
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")

    def _metrics_collection_loop(self) -> None:
        """Collect and update security metrics."""
        while self.orchestration_active:
            try:
                self._collect_security_metrics()
                time.sleep(self.config["dashboard_update_interval"])
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")

    def _incident_management_loop(self) -> None:
        """Manage ongoing security incidents."""
        while self.orchestration_active:
            try:
                self._update_incident_status()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in incident management: {e}")

    def _update_security_posture(self) -> None:
        """Update overall security posture assessment."""
        # Collect metrics from all domains
        physical_metrics = self._get_domain_metrics(SecurityDomain.PHYSICAL)
        digital_metrics = self._get_domain_metrics(SecurityDomain.DIGITAL)
        cognitive_metrics = self._get_domain_metrics(SecurityDomain.COGNITIVE)

        # Calculate overall security score
        overall_score = self._calculate_security_score(
            physical_metrics, digital_metrics, cognitive_metrics
        )

        self.security_metrics["overall_security_score"] = overall_score
        self.security_metrics["last_updated"] = datetime.now().isoformat()

    def _get_domain_metrics(self, domain: SecurityDomain) -> dict[str, Any]:
        """Get metrics from a specific security domain."""
        if domain == SecurityDomain.PHYSICAL:
            return {
                "active_threats": len(self.physical_manager.get_active_threats()),
                "total_devices": len(self.physical_manager.access_devices)
                + len(self.physical_manager.surveillance_devices),
                "coverage_percentage": 85.0,  # Baseline
            }
        elif domain == SecurityDomain.DIGITAL:
            return {
                "active_threats": len(self.digital_manager.get_active_threats()),
                "blocked_ips": len(self.digital_manager.blocked_ips),
                "vulnerability_score": 7.5,  # Baseline
            }
        elif domain == SecurityDomain.COGNITIVE:
            report = self.cognitive_manager.generate_cognitive_report()
            return {
                "behavior_profiles": report["behavior_profiles"]["total_profiles"],
                "cognitive_threats": report["cognitive_threats"]["total_threats"],
                "ml_models_active": report["ml_model_status"]["ml_libraries_available"],
            }

        return {}

    def _calculate_security_score(
        self, physical: dict, digital: dict, cognitive: dict
    ) -> float:
        """Calculate overall security score from domain metrics."""
        # Simplified scoring algorithm
        base_score = 100.0

        # Deduct points for active threats
        threat_penalty = (
            physical.get("active_threats", 0)
            + digital.get("active_threats", 0)
            + cognitive.get("cognitive_threats", 0)
        ) * 5

        # Add points for good coverage/protection
        coverage_bonus = physical.get("coverage_percentage", 0) * 0.1

        final_score = max(0, base_score - threat_penalty + coverage_bonus)
        return min(100.0, cast(float, final_score))

    def _collect_security_metrics(self) -> None:
        """Collect comprehensive security metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "physical_security": self._get_domain_metrics(SecurityDomain.PHYSICAL),
            "digital_security": self._get_domain_metrics(SecurityDomain.DIGITAL),
            "cognitive_security": self._get_domain_metrics(SecurityDomain.COGNITIVE),
            "integrated_threats": len(self.integrated_threats),
            "open_incidents": len(
                [i for i in self.security_incidents.values() if i.status == "open"]
            ),
            "overall_score": self.security_metrics.get("overall_security_score", 0),
        }

        self.performance_history.append(metrics)

        # Keep only recent history
        retention_period = timedelta(days=self.config["metrics_retention_days"])
        cutoff_time = datetime.now() - retention_period

        self.performance_history = [
            m
            for m in self.performance_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

    def _update_incident_status(self) -> None:
        """Update status of ongoing incidents."""
        for incident in self.security_incidents.values():
            if incident.status == "open":
                # Check if threats are resolved
                # Simple resolution check (in practice, this would be more sophisticated)
                if datetime.now() - incident.created_at > timedelta(hours=24):
                    incident.status = "investigating"
                    incident.timeline.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event": "Status updated to investigating",
                            "details": "Automatic status update after 24 hours",
                        }
                    )

    def _cleanup_old_threats(self) -> None:
        """Clean up old resolved threats."""
        cutoff_time = datetime.now() - timedelta(days=7)

        old_threats = [
            threat_id
            for threat_id, threat in self.integrated_threats.items()
            if threat.detected_at < cutoff_time
        ]

        for threat_id in old_threats:
            del self.integrated_threats[threat_id]

    # Public API
    def get_security_dashboard(self) -> dict[str, Any]:
        """Get comprehensive security dashboard data."""
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_security_score": self.security_metrics.get(
                "overall_security_score", 0
            ),
            "active_threats": {
                "physical": len(self.physical_manager.get_active_threats()),
                "digital": len(self.digital_manager.get_active_threats()),
                "cognitive": len(self.cognitive_manager.get_cognitive_threats()),
                "integrated": len(self.integrated_threats),
            },
            "open_incidents": len(
                [i for i in self.security_incidents.values() if i.status == "open"]
            ),
            "recent_alerts": self._get_recent_alerts(),
            "domain_status": {
                "physical": self._get_domain_status(SecurityDomain.PHYSICAL),
                "digital": self._get_domain_status(SecurityDomain.DIGITAL),
                "cognitive": self._get_domain_status(SecurityDomain.COGNITIVE),
            },
            "top_recommendations": self._get_top_recommendations(),
        }

    def _get_recent_alerts(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent security alerts across all domains."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = []

        # Physical alerts
        for phys_threat in self.physical_manager.get_active_threats():
            if phys_threat.detected_at > cutoff_time:
                alerts.append(
                    {
                        "domain": "physical",
                        "type": phys_threat.threat_type,
                        "severity": phys_threat.severity.value,
                        "timestamp": phys_threat.detected_at.isoformat(),
                    }
                )

        # Digital alerts
        for dig_threat in self.digital_manager.get_active_threats():
            if dig_threat.detected_at > cutoff_time:
                alerts.append(
                    {
                        "domain": "digital",
                        "type": dig_threat.threat_type.value,
                        "severity": dig_threat.severity.value,
                        "timestamp": dig_threat.detected_at.isoformat(),
                    }
                )

        # Sort by timestamp
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        return alerts[:20]  # Return top 20 recent alerts

    def _get_domain_status(self, domain: SecurityDomain) -> str:
        """Get status of a security domain."""
        metrics = self._get_domain_metrics(domain)
        active_threats = metrics.get("active_threats", 0)

        if active_threats == 0:
            return "secure"
        elif active_threats < 3:
            return "monitoring"
        elif active_threats < 10:
            return "elevated"
        else:
            return "critical"

    def _get_top_recommendations(self) -> list[str]:
        """Get top security recommendations."""
        recommendations = []

        # Check for high-priority issues
        critical_incidents = [
            i
            for i in self.security_incidents.values()
            if i.severity == IncidentSeverity.CRITICAL and i.status == "open"
        ]

        if critical_incidents:
            recommendations.append(
                f"Address {len(critical_incidents)} critical security incidents"
            )

        # Domain-specific recommendations
        if len(self.digital_manager.get_active_threats()) > 5:
            recommendations.append("Review and mitigate digital security threats")

        if len(self.physical_manager.get_active_threats()) > 3:
            recommendations.append("Investigate physical security alerts")

        return recommendations[:5]  # Return top 5 recommendations

    def get_integrated_threats(self) -> list[IntegratedThreat]:
        """Get all integrated threats."""
        return list(self.integrated_threats.values())

    def get_security_incidents(
        self, status: str | None = None
    ) -> list[SecurityIncident]:
        """Get security incidents, optionally filtered by status."""
        incidents = list(self.security_incidents.values())

        if status:
            incidents = [i for i in incidents if i.status == status]

        return incidents

    def resolve_incident(self, incident_id: str, resolution_notes: str = "") -> bool:
        """Resolve a security incident."""
        if incident_id in self.security_incidents:
            incident = self.security_incidents[incident_id]
            incident.status = "resolved"
            incident.resolved_at = datetime.now()
            incident.timeline.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "Incident resolved",
                    "details": resolution_notes,
                }
            )
            return True
        return False

    def add_response_handler(
        self, action: str, handler: Callable[[IntegratedThreat], None]
    ) -> None:
        """Add a response handler for specific actions."""
        self.response_handlers[action] = handler

    def add_alert_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add alert callback for integrated security notifications."""
        self.alert_callbacks.append(callback)
