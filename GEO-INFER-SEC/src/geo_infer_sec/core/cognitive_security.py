"""
Cognitive Security Module for GEO-INFER-SEC

This module provides AI-driven security capabilities including:
- Behavioral analysis and anomaly detection
- Machine learning threat prediction
- Natural language processing for threat intelligence
- Pattern recognition in security events
- Adaptive security response systems
- Threat hunting with AI assistance
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import threading
import time

# ML and AI libraries
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report as classification_report
    import joblib as joblib

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.getLogger(__name__).debug(
        "Machine learning libraries not available. Installing scikit-learn recommended."
    )

from ..models.security_models import SecurityEvent


class BehaviorType(Enum):
    """Types of behavioral patterns."""

    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ANOMALOUS = "anomalous"
    MALICIOUS = "malicious"


class ThreatHuntingType(Enum):
    """Types of threat hunting activities."""

    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    IOC_BASED = "ioc_based"


class LearningMode(Enum):
    """Machine learning modes."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    HYBRID = "hybrid"


@dataclass
class BehaviorProfile:
    """Represents a behavioral profile for an entity."""

    entity_id: str
    entity_type: str  # user, system, network, etc.
    baseline_metrics: dict[str, float]
    current_metrics: dict[str, float]
    behavior_score: float
    behavior_type: BehaviorType
    last_updated: datetime
    confidence_level: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveThreat:
    """Represents a threat detected through cognitive analysis."""

    threat_id: str
    threat_type: str
    confidence_score: float
    behavioral_indicators: list[str]
    prediction_model: str
    detection_method: str
    affected_entities: list[str]
    risk_score: float
    recommended_actions: list[str]
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatHuntingResult:
    """Results from threat hunting activities."""

    hunt_id: str
    hunt_type: ThreatHuntingType
    hypothesis: str
    findings: list[dict[str, Any]]
    threat_indicators: list[str]
    confidence_level: float
    recommendations: list[str]
    hunt_date: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class CognitiveSecurityManager:
    """AI-driven cognitive security management system."""

    def __init__(self, config_path: str | None = None):
        """Initialize the cognitive security manager."""
        self.logger = logging.getLogger(__name__)

        if not ML_AVAILABLE:
            self.logger.warning(
                "Machine learning libraries not available. Some features will be limited."
            )

        # Initialize data structures
        self.behavior_profiles: dict[str, BehaviorProfile] = {}
        self.cognitive_threats: dict[str, CognitiveThreat] = {}
        self.threat_hunting_results: dict[str, ThreatHuntingResult] = {}
        self.security_events_buffer: deque = deque(maxlen=10000)

        # Initialize ML models
        self.anomaly_detector: Any = None
        self.threat_classifier: Any = None
        self.behavior_analyzer: Any = None
        self.model_scaler: Any = StandardScaler() if ML_AVAILABLE else None

        # Initialize monitoring
        self.monitoring_active = False
        self.analysis_threads: list[threading.Thread] = []
        self.alert_callbacks: list[Callable[[dict[str, Any]], None]] = []

        # Configuration
        self.config = {
            "anomaly_threshold": 0.7,
            "behavior_update_interval": 300,  # 5 minutes
            "threat_prediction_interval": 60,  # 1 minute
            "learning_window_hours": 24,
            "min_events_for_baseline": 100,
            "confidence_threshold": 0.6,
        }

        # Initialize models if ML is available
        if ML_AVAILABLE:
            self._initialize_ml_models()

    def _initialize_ml_models(self) -> None:
        """Initialize machine learning models."""
        try:
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1, random_state=42, n_estimators=100
            )

            # Threat classification model
            self.threat_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )

            # Behavior clustering model
            self.behavior_analyzer = DBSCAN(eps=0.5, min_samples=5)

            self.logger.info("Machine learning models initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")

    # Behavioral Analysis
    def analyze_user_behavior(
        self, user_id: str, events: list[SecurityEvent]
    ) -> BehaviorProfile:
        """Analyze user behavior patterns."""
        if not events:
            return self._create_empty_profile(user_id, "user")

        # Calculate behavioral metrics
        metrics = self._extract_behavioral_metrics(events)

        # Get existing profile or create new one
        existing_profile = self.behavior_profiles.get(user_id)

        if existing_profile and len(events) >= self.config["min_events_for_baseline"]:
            # Update existing profile
            behavior_score = self._calculate_behavior_deviation(
                existing_profile.baseline_metrics, metrics
            )
            behavior_type = self._classify_behavior(behavior_score)

            profile = BehaviorProfile(
                entity_id=user_id,
                entity_type="user",
                baseline_metrics=existing_profile.baseline_metrics,
                current_metrics=metrics,
                behavior_score=behavior_score,
                behavior_type=behavior_type,
                last_updated=datetime.now(),
                confidence_level=min(
                    1.0, len(events) / 100.0
                ),  # Confidence based on sample size
            )
        else:
            # Create baseline profile
            profile = BehaviorProfile(
                entity_id=user_id,
                entity_type="user",
                baseline_metrics=metrics,
                current_metrics=metrics,
                behavior_score=0.0,
                behavior_type=BehaviorType.NORMAL,
                last_updated=datetime.now(),
                confidence_level=min(1.0, len(events) / 100.0),
            )

        self.behavior_profiles[user_id] = profile
        return profile

    def _extract_behavioral_metrics(
        self, events: list[SecurityEvent]
    ) -> dict[str, float]:
        """Extract behavioral metrics from security events."""
        if not events:
            return {}

        # Time-based metrics
        timestamps = [event.timestamp for event in events]
        time_intervals = []

        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            time_intervals.append(interval)

        # Calculate metrics
        metrics = {
            "avg_session_duration": np.mean(time_intervals) if time_intervals else 0,
            "std_session_duration": np.std(time_intervals) if time_intervals else 0,
            "login_frequency": len(
                [e for e in events if e.event_type == "login_success"]
            )
            / max(1, len(events)),
            "failed_login_rate": len(
                [e for e in events if e.event_type == "login_failure"]
            )
            / max(1, len(events)),
            "unique_source_ips": len(
                set(
                    e.metadata.get("source_ip", "")
                    for e in events
                    if e.metadata.get("source_ip")
                )
            ),
            "data_access_volume": sum(e.metadata.get("data_volume", 0) for e in events),
            "privilege_escalation_attempts": len(
                [e for e in events if e.event_type == "privilege_escalation"]
            ),
            "unusual_hour_activity": self._calculate_unusual_hour_activity(events),
            "api_call_frequency": len([e for e in events if e.event_type == "api_call"])
            / max(1, len(events)),
            "file_access_diversity": len(
                set(
                    e.metadata.get("file_path", "")
                    for e in events
                    if e.metadata.get("file_path")
                )
            ),
        }

        return metrics

    def _calculate_unusual_hour_activity(self, events: list[SecurityEvent]) -> float:
        """Calculate the proportion of events occurring during unusual hours."""
        if not events:
            return 0.0

        unusual_hour_count = 0
        for event in events:
            hour = event.timestamp.hour
            # Consider 10 PM to 6 AM as unusual hours
            if hour >= 22 or hour <= 6:
                unusual_hour_count += 1

        return unusual_hour_count / len(events)

    def _calculate_behavior_deviation(
        self, baseline: dict[str, float], current: dict[str, float]
    ) -> float:
        """Calculate behavior deviation score."""
        if not baseline or not current:
            return 0.0

        deviations = []

        for metric, baseline_value in baseline.items():
            if metric in current:
                current_value = current[metric]
                if baseline_value > 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    deviations.append(deviation)

        return cast(float, np.mean(deviations)) if deviations else 0.0

    def _classify_behavior(self, behavior_score: float) -> BehaviorType:
        """Classify behavior based on deviation score."""
        if behavior_score <= 0.3:
            return BehaviorType.NORMAL
        elif behavior_score <= 0.6:
            return BehaviorType.SUSPICIOUS
        elif behavior_score <= 0.8:
            return BehaviorType.ANOMALOUS
        else:
            return BehaviorType.MALICIOUS

    def _create_empty_profile(
        self, entity_id: str, entity_type: str
    ) -> BehaviorProfile:
        """Create an empty behavior profile."""
        return BehaviorProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            baseline_metrics={},
            current_metrics={},
            behavior_score=0.0,
            behavior_type=BehaviorType.NORMAL,
            last_updated=datetime.now(),
            confidence_level=0.0,
        )

    # Anomaly Detection
    def detect_anomalies(self, events: list[SecurityEvent]) -> list[dict[str, Any]]:
        """Detect anomalies in security events using ML."""
        if not ML_AVAILABLE or not events:
            return []

        try:
            # Prepare feature matrix
            features = self._prepare_feature_matrix(events)

            if len(features) < 10:  # Need minimum samples for anomaly detection
                return []

            # Fit and predict anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features)

            anomalies = []
            for i, score in enumerate(anomaly_scores):
                if score == -1:  # Anomaly detected
                    anomalies.append(
                        {
                            "event_index": i,
                            "event": events[i],
                            "anomaly_score": float(
                                self.anomaly_detector.score_samples([features[i]])[0]
                            ),
                            "detected_at": datetime.now(),
                        }
                    )

            return anomalies

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return []

    def _prepare_feature_matrix(self, events: list[SecurityEvent]) -> np.ndarray:
        """Prepare feature matrix for ML algorithms."""
        features = []

        for event in events:
            feature_vector = [
                hash(event.event_type) % 1000,  # Event type hash
                event.timestamp.hour,  # Hour of day
                event.timestamp.weekday(),  # Day of week
                len(str(event.metadata)),  # Metadata size
                event.metadata.get("data_volume", 0),  # Data volume
                len(event.metadata.get("source_ip", "")),  # IP length (proxy for type)
                event.metadata.get("duration", 0),  # Event duration
                len(event.metadata.get("user_agent", "")),  # User agent length
                event.metadata.get("response_code", 200),  # HTTP response code
                len(event.metadata.get("request_data", "")),  # Request data size
            ]
            features.append(feature_vector)

        return np.array(features)

    # Threat Prediction
    def predict_threats(
        self, historical_data: list[SecurityEvent]
    ) -> list[CognitiveThreat]:
        """Predict potential threats using ML models."""
        if not ML_AVAILABLE or not historical_data:
            return []

        threats = []

        try:
            # Analyze patterns in historical data
            patterns = self._analyze_threat_patterns(historical_data)

            for pattern_type, pattern_data in patterns.items():
                if pattern_data["risk_score"] > self.config["confidence_threshold"]:
                    threat = CognitiveThreat(
                        threat_id=f"cognitive_{datetime.now().strftime('%Y%m%d%H%M%S')}_{pattern_type}",
                        threat_type=pattern_type,
                        confidence_score=pattern_data["confidence"],
                        behavioral_indicators=pattern_data["indicators"],
                        prediction_model="pattern_analysis",
                        detection_method="predictive_analytics",
                        affected_entities=pattern_data["entities"],
                        risk_score=pattern_data["risk_score"],
                        recommended_actions=pattern_data["recommendations"],
                    )
                    threats.append(threat)

            return threats

        except Exception as e:
            self.logger.error(f"Error in threat prediction: {e}")
            return []

    def _analyze_threat_patterns(
        self, events: list[SecurityEvent]
    ) -> dict[str, dict[str, Any]]:
        """Analyze patterns in security events to predict threats."""
        patterns = {}

        # Group events by different criteria
        events_by_ip = defaultdict(list)
        events_by_user = defaultdict(list)
        events_by_hour = defaultdict(list)

        for event in events:
            source_ip = event.metadata.get("source_ip", "unknown")
            user_id = event.metadata.get("user_id", "unknown")
            hour = event.timestamp.hour

            events_by_ip[source_ip].append(event)
            events_by_user[user_id].append(event)
            events_by_hour[hour].append(event)

        # Analyze IP-based patterns
        for ip, ip_events in events_by_ip.items():
            if len(ip_events) > 50:  # Threshold for suspicious activity
                failed_logins = len(
                    [e for e in ip_events if e.event_type == "login_failure"]
                )
                if failed_logins > 10:
                    patterns[f"brute_force_{ip}"] = {
                        "confidence": min(1.0, failed_logins / 50.0),
                        "risk_score": min(1.0, failed_logins / 20.0),
                        "indicators": [f"Multiple failed logins from {ip}"],
                        "entities": [ip],
                        "recommendations": [
                            f"Block IP {ip}",
                            "Implement rate limiting",
                        ],
                    }

        # Analyze user-based patterns
        for user, user_events in events_by_user.items():
            if user != "unknown":
                privilege_attempts = len(
                    [e for e in user_events if e.event_type == "privilege_escalation"]
                )
                if privilege_attempts > 3:
                    patterns[f"privilege_abuse_{user}"] = {
                        "confidence": min(1.0, privilege_attempts / 10.0),
                        "risk_score": min(1.0, privilege_attempts / 5.0),
                        "indicators": [
                            f"Multiple privilege escalation attempts by {user}"
                        ],
                        "entities": [user],
                        "recommendations": [
                            f"Review user {user} permissions",
                            "Monitor user activity",
                        ],
                    }

        return patterns

    # Threat Hunting
    def conduct_threat_hunt(
        self,
        hypothesis: str,
        hunt_type: ThreatHuntingType,
        search_criteria: dict[str, Any],
    ) -> ThreatHuntingResult:
        """Conduct AI-assisted threat hunting."""
        hunt_id = f"hunt_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        findings: list[Any] = []
        threat_indicators: list[str] = []

        try:
            # Filter events based on search criteria
            relevant_events = self._filter_events_for_hunt(search_criteria)

            # Apply cognitive analysis to findings
            if hunt_type == ThreatHuntingType.IOC_BASED:
                findings, indicators = self._hunt_by_indicators(
                    relevant_events, search_criteria
                )
            elif hunt_type == ThreatHuntingType.HYPOTHESIS_DRIVEN:
                findings, indicators = self._hunt_by_hypothesis(
                    relevant_events, hypothesis
                )
            elif hunt_type == ThreatHuntingType.PROACTIVE:
                findings, indicators = self._proactive_hunt(relevant_events)
            else:
                findings, indicators = self._reactive_hunt(relevant_events)

            threat_indicators.extend(indicators)

            # Calculate confidence level
            confidence = self._calculate_hunt_confidence(findings)

            # Generate recommendations
            recommendations = self._generate_hunt_recommendations(findings, hunt_type)

            result = ThreatHuntingResult(
                hunt_id=hunt_id,
                hunt_type=hunt_type,
                hypothesis=hypothesis,
                findings=findings,
                threat_indicators=threat_indicators,
                confidence_level=confidence,
                recommendations=recommendations,
            )

            self.threat_hunting_results[hunt_id] = result
            return result

        except Exception as e:
            self.logger.error(f"Error in threat hunting: {e}")
            return ThreatHuntingResult(
                hunt_id=hunt_id,
                hunt_type=hunt_type,
                hypothesis=hypothesis,
                findings=[],
                threat_indicators=[],
                confidence_level=0.0,
                recommendations=[],
            )

    def _filter_events_for_hunt(self, criteria: dict[str, Any]) -> list[SecurityEvent]:
        """Filter security events based on hunt criteria."""
        filtered_events = []

        for event in self.security_events_buffer:
            match = True

            # Apply filters
            if "source_ip" in criteria:
                if event.metadata.get("source_ip") != criteria["source_ip"]:
                    match = False

            if "user_id" in criteria:
                if event.metadata.get("user_id") != criteria["user_id"]:
                    match = False

            if "event_type" in criteria:
                if event.event_type != criteria["event_type"]:
                    match = False

            if "time_range" in criteria:
                start_time, end_time = criteria["time_range"]
                if not (start_time <= event.timestamp <= end_time):
                    match = False

            if match:
                filtered_events.append(event)

        return filtered_events

    def _hunt_by_indicators(
        self, events: list[SecurityEvent], criteria: dict[str, Any]
    ) -> tuple[list[dict], list[str]]:
        """Hunt for threats based on indicators of compromise."""
        findings = []
        indicators = criteria.get("indicators", [])

        for event in events:
            event_str = json.dumps(event.metadata)
            for indicator in indicators:
                if indicator.lower() in event_str.lower():
                    findings.append(
                        {
                            "event_id": event.event_id,
                            "indicator_matched": indicator,
                            "event_data": event.metadata,
                            "timestamp": event.timestamp.isoformat(),
                            "risk_level": "HIGH",
                        }
                    )

        return findings, indicators

    def _hunt_by_hypothesis(
        self, events: list[SecurityEvent], hypothesis: str
    ) -> tuple[list[dict], list[str]]:
        """Hunt for threats based on a hypothesis."""
        findings = []
        indicators = []

        # Simple keyword-based hypothesis testing
        hypothesis_keywords = hypothesis.lower().split()

        for event in events:
            event_str = json.dumps(event.metadata).lower()
            matches = sum(1 for keyword in hypothesis_keywords if keyword in event_str)

            if matches >= len(hypothesis_keywords) * 0.5:  # 50% keyword match threshold
                findings.append(
                    {
                        "event_id": event.event_id,
                        "hypothesis_match_score": matches / len(hypothesis_keywords),
                        "event_data": event.metadata,
                        "timestamp": event.timestamp.isoformat(),
                        "risk_level": "MEDIUM"
                        if matches / len(hypothesis_keywords) > 0.7
                        else "LOW",
                    }
                )
                indicators.append(
                    f"Hypothesis match: {matches}/{len(hypothesis_keywords)} keywords"
                )

        return findings, indicators

    def _proactive_hunt(
        self, events: list[SecurityEvent]
    ) -> tuple[list[dict], list[str]]:
        """Proactive threat hunting using behavioral analysis."""
        findings = []
        indicators = []

        # Look for unusual patterns
        if len(events) > 100:
            # Detect anomalies in the event stream
            anomalies = self.detect_anomalies(events)

            for anomaly in anomalies:
                findings.append(
                    {
                        "event_id": anomaly["event"].event_id,
                        "anomaly_score": anomaly["anomaly_score"],
                        "event_data": anomaly["event"].metadata,
                        "timestamp": anomaly["event"].timestamp.isoformat(),
                        "risk_level": "MEDIUM",
                    }
                )
                indicators.append("Behavioral anomaly detected")

        return findings, indicators

    def _reactive_hunt(
        self, events: list[SecurityEvent]
    ) -> tuple[list[dict], list[str]]:
        """Reactive threat hunting based on recent alerts."""
        findings = []
        indicators = []

        # Look for patterns in recent high-risk events
        recent_events = [e for e in events if (datetime.now() - e.timestamp).days <= 1]

        for event in recent_events:
            if event.metadata.get("severity") in ["HIGH", "CRITICAL"]:
                findings.append(
                    {
                        "event_id": event.event_id,
                        "event_data": event.metadata,
                        "timestamp": event.timestamp.isoformat(),
                        "risk_level": event.metadata.get("severity", "MEDIUM"),
                    }
                )
                indicators.append(f"High-severity event: {event.event_type}")

        return findings, indicators

    def _calculate_hunt_confidence(self, findings: list[dict]) -> float:
        """Calculate confidence level for hunt results."""
        if not findings:
            return 0.0

        # Base confidence on number and quality of findings
        high_risk_count = len([f for f in findings if f.get("risk_level") == "HIGH"])
        medium_risk_count = len(
            [f for f in findings if f.get("risk_level") == "MEDIUM"]
        )

        confidence = (high_risk_count * 0.8 + medium_risk_count * 0.5) / len(findings)
        return min(1.0, confidence)

    def _generate_hunt_recommendations(
        self, findings: list[dict], hunt_type: ThreatHuntingType
    ) -> list[str]:
        """Generate recommendations based on hunt results."""
        recommendations = []

        if not findings:
            recommendations.append(
                "No immediate threats detected. Continue monitoring."
            )
            return recommendations

        high_risk_findings = [f for f in findings if f.get("risk_level") == "HIGH"]

        if high_risk_findings:
            recommendations.append(
                f"Investigate {len(high_risk_findings)} high-risk findings immediately"
            )
            recommendations.append("Consider implementing additional security controls")

        if hunt_type == ThreatHuntingType.IOC_BASED:
            recommendations.append("Update threat intelligence with new indicators")
        elif hunt_type == ThreatHuntingType.HYPOTHESIS_DRIVEN:
            recommendations.append("Refine hypothesis based on findings")

        return recommendations

    # Monitoring and Management
    def start_cognitive_monitoring(self) -> None:
        """Start cognitive security monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True

            # Start analysis threads
            threads = [
                threading.Thread(
                    target=self._behavior_analysis_loop, name="BehaviorAnalysis"
                ),
                threading.Thread(
                    target=self._anomaly_detection_loop, name="AnomalyDetection"
                ),
                threading.Thread(
                    target=self._threat_prediction_loop, name="ThreatPrediction"
                ),
            ]

            for thread in threads:
                thread.daemon = True
                thread.start()
                self.analysis_threads.append(thread)

            self.logger.info("Cognitive security monitoring started")

    def stop_cognitive_monitoring(self) -> None:
        """Stop cognitive security monitoring."""
        self.monitoring_active = False

        for thread in self.analysis_threads:
            thread.join(timeout=5)

        self.analysis_threads.clear()
        self.logger.info("Cognitive security monitoring stopped")

    def _behavior_analysis_loop(self) -> None:
        """Main behavior analysis loop."""
        while self.monitoring_active:
            try:
                self._update_behavior_profiles()
                time.sleep(self.config["behavior_update_interval"])
            except Exception as e:
                self.logger.error(f"Error in behavior analysis: {e}")

    def _anomaly_detection_loop(self) -> None:
        """Main anomaly detection loop."""
        while self.monitoring_active:
            try:
                if len(self.security_events_buffer) > 100:
                    recent_events = list(self.security_events_buffer)[-100:]
                    anomalies = self.detect_anomalies(recent_events)

                    for anomaly in anomalies:
                        self._trigger_cognitive_alert(anomaly)

                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {e}")

    def _threat_prediction_loop(self) -> None:
        """Main threat prediction loop."""
        while self.monitoring_active:
            try:
                if len(self.security_events_buffer) > 200:
                    recent_events = list(self.security_events_buffer)[-200:]
                    threats = self.predict_threats(recent_events)

                    for threat in threats:
                        self.cognitive_threats[threat.threat_id] = threat
                        self._trigger_cognitive_threat_alert(threat)

                time.sleep(self.config["threat_prediction_interval"])
            except Exception as e:
                self.logger.error(f"Error in threat prediction: {e}")

    def _update_behavior_profiles(self) -> None:
        """Update behavior profiles for all entities."""
        # Group recent events by entity
        recent_cutoff = datetime.now() - timedelta(
            hours=self.config["learning_window_hours"]
        )
        recent_events = [
            e for e in self.security_events_buffer if e.timestamp > recent_cutoff
        ]

        events_by_user = defaultdict(list)
        for event in recent_events:
            user_id = event.metadata.get("user_id")
            if user_id:
                events_by_user[user_id].append(event)

        # Update profiles
        for user_id, user_events in events_by_user.items():
            self.analyze_user_behavior(user_id, user_events)

    def _trigger_cognitive_alert(self, anomaly: dict[str, Any]) -> None:
        """Trigger alert for cognitive security detection."""
        alert_data = {
            "type": "cognitive_anomaly",
            "anomaly_score": anomaly["anomaly_score"],
            "event_data": anomaly["event"].metadata,
            "detected_at": datetime.now().isoformat(),
        }

        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in cognitive alert callback: {e}")

    def _trigger_cognitive_threat_alert(self, threat: CognitiveThreat) -> None:
        """Trigger alert for cognitive threat prediction."""
        alert_data = {
            "type": "cognitive_threat",
            "threat_id": threat.threat_id,
            "threat_type": threat.threat_type,
            "confidence_score": threat.confidence_score,
            "risk_score": threat.risk_score,
            "recommended_actions": threat.recommended_actions,
        }

        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in cognitive threat alert callback: {e}")

    # Public API Methods
    def add_security_event(self, event: SecurityEvent) -> None:
        """Add a security event to the cognitive analysis buffer."""
        self.security_events_buffer.append(event)

    def get_behavior_profile(self, entity_id: str) -> BehaviorProfile | None:
        """Get behavior profile for an entity."""
        return self.behavior_profiles.get(entity_id)

    def get_cognitive_threats(self) -> list[CognitiveThreat]:
        """Get all active cognitive threats."""
        return list(self.cognitive_threats.values())

    def get_threat_hunting_results(self) -> list[ThreatHuntingResult]:
        """Get all threat hunting results."""
        return list(self.threat_hunting_results.values())

    def add_alert_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback function for cognitive security alerts."""
        self.alert_callbacks.append(callback)

    def generate_cognitive_report(self) -> dict[str, Any]:
        """Generate a comprehensive cognitive security report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "behavior_profiles": {
                "total_profiles": len(self.behavior_profiles),
                "suspicious_profiles": len(
                    [
                        p
                        for p in self.behavior_profiles.values()
                        if p.behavior_type == BehaviorType.SUSPICIOUS
                    ]
                ),
                "anomalous_profiles": len(
                    [
                        p
                        for p in self.behavior_profiles.values()
                        if p.behavior_type == BehaviorType.ANOMALOUS
                    ]
                ),
                "malicious_profiles": len(
                    [
                        p
                        for p in self.behavior_profiles.values()
                        if p.behavior_type == BehaviorType.MALICIOUS
                    ]
                ),
            },
            "cognitive_threats": {
                "total_threats": len(self.cognitive_threats),
                "high_confidence_threats": len(
                    [
                        t
                        for t in self.cognitive_threats.values()
                        if t.confidence_score > 0.8
                    ]
                ),
                "high_risk_threats": len(
                    [t for t in self.cognitive_threats.values() if t.risk_score > 0.8]
                ),
            },
            "threat_hunting": {
                "total_hunts": len(self.threat_hunting_results),
                "successful_hunts": len(
                    [
                        h
                        for h in self.threat_hunting_results.values()
                        if h.confidence_level > 0.5
                    ]
                ),
            },
            "ml_model_status": {
                "anomaly_detector_trained": self.anomaly_detector is not None,
                "threat_classifier_trained": self.threat_classifier is not None,
                "ml_libraries_available": ML_AVAILABLE,
            },
        }

        return report
