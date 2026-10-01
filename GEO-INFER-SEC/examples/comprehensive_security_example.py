#!/usr/bin/env python3
"""
Comprehensive Security Example for GEO-INFER-SEC

This example demonstrates the integration of Physical, Digital, and Cognitive security
domains in a realistic scenario. It shows how different security threats can be
detected, correlated, and responded to in a coordinated manner.

Scenario: Corporate facility with integrated security systems
- Physical security: Access control, surveillance, perimeter monitoring
- Digital security: Network monitoring, endpoint protection, data loss prevention
- Cognitive security: Behavioral analysis, anomaly detection, threat prediction
"""

import asyncio
import logging
import sys
from datetime import datetime

from geo_infer_sec.core.integrated_security import IntegratedSecurityManager
from geo_infer_sec.core.physical_security import (
    AccessControlDevice,
    SurveillanceDevice,
    SecurityZone,
    AccessControlType,
    SurveillanceType,
    SecurityZoneType,
)
from geo_infer_sec.core.digital_security import (
    SecurityEventType,
)
from shapely.geometry import Point, Polygon


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SecurityDemoEnvironment:
    """Simulated security environment for demonstration."""

    def __init__(self):
        """Initialize the demo environment."""
        self.integrated_manager = IntegratedSecurityManager()
        self.setup_demo_environment()
        self.setup_alert_handlers()

    def setup_demo_environment(self):
        """Setup a realistic demo security environment."""
        logger.info("Setting up comprehensive security demo environment...")

        # Setup physical security zones
        self._setup_physical_zones()

        # Setup security devices
        self._setup_security_devices()

        # Setup digital security policies
        self._setup_digital_policies()

        # Setup response handlers
        self._setup_response_handlers()

    def _setup_physical_zones(self):
        """Setup physical security zones."""
        # Public lobby area
        lobby_zone = SecurityZone(
            zone_id="lobby",
            name="Public Lobby",
            zone_type=SecurityZoneType.PUBLIC,
            boundary=Polygon(
                [
                    (-74.006, 40.712),
                    (-74.005, 40.712),
                    (-74.005, 40.713),
                    (-74.006, 40.713),
                ]
            ),
            required_clearance_level=0,
            access_hours={
                "start": "06:00",
                "end": "22:00",
                "days": ["mon", "tue", "wed", "thu", "fri"],
            },
        )

        # Restricted office area
        office_zone = SecurityZone(
            zone_id="office",
            name="Office Area",
            zone_type=SecurityZoneType.RESTRICTED,
            boundary=Polygon(
                [
                    (-74.005, 40.713),
                    (-74.004, 40.713),
                    (-74.004, 40.714),
                    (-74.005, 40.714),
                ]
            ),
            required_clearance_level=2,
            access_hours={
                "start": "08:00",
                "end": "18:00",
                "days": ["mon", "tue", "wed", "thu", "fri"],
            },
            escort_required=False,
        )

        # High-security server room
        server_zone = SecurityZone(
            zone_id="server_room",
            name="Server Room",
            zone_type=SecurityZoneType.CONFIDENTIAL,
            boundary=Polygon(
                [
                    (-74.004, 40.714),
                    (-74.0035, 40.714),
                    (-74.0035, 40.7145),
                    (-74.004, 40.7145),
                ]
            ),
            required_clearance_level=4,
            two_person_rule=True,
            escort_required=True,
        )

        # Add zones to physical manager
        for zone in [lobby_zone, office_zone, server_zone]:
            self.integrated_manager.physical_manager.add_security_zone(zone)

    def _setup_security_devices(self):
        """Setup physical security devices."""
        # Access control devices
        main_entrance = AccessControlDevice(
            device_id="ac_main_001",
            name="Main Entrance Card Reader",
            device_type=AccessControlType.CARD_READER,
            location=Point(-74.0055, 40.7125),
            zone_id="lobby",
            security_level=2,
            backup_power=True,
            tamper_detection=True,
        )

        office_door = AccessControlDevice(
            device_id="ac_office_001",
            name="Office Area Biometric Scanner",
            device_type=AccessControlType.BIOMETRIC,
            location=Point(-74.0045, 40.7135),
            zone_id="office",
            security_level=3,
            backup_power=True,
            tamper_detection=True,
        )

        server_door = AccessControlDevice(
            device_id="ac_server_001",
            name="Server Room Multi-Factor",
            device_type=AccessControlType.MULTI_FACTOR,
            location=Point(-74.00375, 40.71425),
            zone_id="server_room",
            security_level=5,
            backup_power=True,
            tamper_detection=True,
        )

        # Surveillance devices
        lobby_camera = SurveillanceDevice(
            device_id="cam_lobby_001",
            name="Lobby Security Camera",
            device_type=SurveillanceType.CCTV,
            location=Point(-74.0055, 40.7125),
            coverage_area=Polygon(
                [
                    (-74.006, 40.712),
                    (-74.005, 40.712),
                    (-74.005, 40.713),
                    (-74.006, 40.713),
                ]
            ),
            zone_id="lobby",
            recording_active=True,
            field_of_view=120.0,
            range_meters=50.0,
        )

        perimeter_sensor = SurveillanceDevice(
            device_id="sensor_perimeter_001",
            name="Perimeter Motion Sensor",
            device_type=SurveillanceType.MOTION_DETECTOR,
            location=Point(-74.006, 40.712),
            coverage_area=Polygon(
                [
                    (-74.0065, 40.7115),
                    (-74.0035, 40.7115),
                    (-74.0035, 40.715),
                    (-74.0065, 40.715),
                ]
            ),
            zone_id="lobby",
            detection_sensitivity=0.8,
        )

        # Add devices to managers
        devices = [main_entrance, office_door, server_door]
        for device in devices:
            self.integrated_manager.physical_manager.add_access_device(device)

        surveillance_devices = [lobby_camera, perimeter_sensor]
        for device in surveillance_devices:
            self.integrated_manager.physical_manager.add_surveillance_device(device)

    def _setup_digital_policies(self):
        """Setup digital security policies."""
        # Policies are initialized automatically in the DigitalSecurityManager
        logger.info("Digital security policies initialized")

    def _setup_response_handlers(self):
        """Setup automated response handlers."""

        def lockdown_systems(threat):
            logger.critical(f"LOCKDOWN INITIATED: {threat.threat_id}")
            # In a real system, this would trigger actual lockdown procedures

        def enhanced_monitoring(threat):
            logger.warning(f"ENHANCED MONITORING: {threat.threat_id}")
            # Increase monitoring sensitivity and frequency

        def emergency_response(threat):
            logger.critical(f"EMERGENCY RESPONSE: {threat.threat_id}")
            # Trigger emergency protocols

        # Register response handlers
        self.integrated_manager.add_response_handler(
            "lockdown_systems", lockdown_systems
        )
        self.integrated_manager.add_response_handler(
            "enhanced_monitoring", enhanced_monitoring
        )
        self.integrated_manager.add_response_handler(
            "emergency_response", emergency_response
        )

    def setup_alert_handlers(self):
        """Setup alert handling for the demo."""

        def alert_handler(alert_data):
            logger.warning(f"INTEGRATED ALERT: {alert_data}")

        self.integrated_manager.add_alert_callback(alert_handler)

    async def simulate_security_scenarios(self):
        """Simulate various security scenarios."""
        logger.info("Starting security scenario simulations...")

        # Start integrated monitoring
        self.integrated_manager.start_integrated_monitoring()

        # Wait for systems to initialize
        await asyncio.sleep(2)

        # Scenario 1: Normal operations
        await self.scenario_normal_operations()

        # Scenario 2: Failed login attempts (Digital)
        await self.scenario_failed_logins()

        # Scenario 3: Physical breach attempt
        await self.scenario_physical_breach()

        # Scenario 4: Coordinated attack (Multi-domain)
        await self.scenario_coordinated_attack()

        # Scenario 5: Insider threat (Cognitive detection)
        await self.scenario_insider_threat()

        # Show final dashboard
        await self.show_security_dashboard()

        # Stop monitoring
        self.integrated_manager.stop_integrated_monitoring()

    async def scenario_normal_operations(self):
        """Simulate normal security operations."""
        logger.info("\n=== SCENARIO 1: Normal Operations ===")

        # Normal access events
        normal_events = [
            {
                "event_type": SecurityEventType.LOGIN_SUCCESS,
                "metadata": {
                    "user_id": "john.doe",
                    "source_ip": "192.168.1.100",
                    "device_id": "ac_main_001",
                },
            },
            {
                "event_type": SecurityEventType.DATA_ACCESS,
                "metadata": {
                    "user_id": "jane.smith",
                    "file_path": "/documents/report.pdf",
                    "data_volume": 1024,
                },
            },
        ]

        # Log normal events
        for event_data in normal_events:
            event = self.integrated_manager.digital_manager.log_security_event(
                event_data["event_type"], event_data["metadata"]
            )
            logger.info(f"Normal event logged: {event.event_type}")

        await asyncio.sleep(3)

    async def scenario_failed_logins(self):
        """Simulate failed login attempts scenario."""
        logger.info("\n=== SCENARIO 2: Failed Login Attempts ===")

        # Simulate multiple failed login attempts from same IP
        suspicious_ip = "10.0.0.99"

        for i in range(7):  # Exceeds the threshold of 5
            event = self.integrated_manager.digital_manager.log_security_event(
                SecurityEventType.LOGIN_FAILURE,
                {
                    "user_id": f"attacker_{i}",
                    "source_ip": suspicious_ip,
                    "reason": "invalid_credentials",
                },
            )
            logger.info(
                f"Failed login attempt {i + 1} from {suspicious_ip} ({event.event_id})"
            )
            await asyncio.sleep(1)

        # Wait for threat detection
        await asyncio.sleep(5)

    async def scenario_physical_breach(self):
        """Simulate physical security breach."""
        logger.info("\n=== SCENARIO 3: Physical Security Breach ===")

        # Simulate unauthorized access attempt
        breach_location = Point(-74.00375, 40.71425)  # Near server room

        # Detect intrusion
        threat = self.integrated_manager.physical_manager.detect_intrusion(
            location=breach_location, detection_method="motion_sensor", confidence=0.9
        )

        if threat:
            logger.warning(f"Physical threat detected: {threat.threat_id}")

        await asyncio.sleep(3)

    async def scenario_coordinated_attack(self):
        """Simulate coordinated multi-domain attack."""
        logger.info("\n=== SCENARIO 4: Coordinated Multi-Domain Attack ===")

        # Step 1: Physical reconnaissance (surveillance detection)
        surveillance_location = Point(-74.006, 40.712)
        physical_threat = self.integrated_manager.physical_manager.detect_intrusion(
            location=surveillance_location,
            detection_method="perimeter_sensor",
            confidence=0.7,
        )
        if physical_threat:
            logger.warning(f"Reconnaissance detected: {physical_threat.threat_id}")

        await asyncio.sleep(2)

        # Step 2: Network reconnaissance (digital)
        digital_event = self.integrated_manager.digital_manager.log_security_event(
            SecurityEventType.NETWORK_CONNECTION,
            {
                "source_ip": "203.0.113.15",  # External IP
                "destination_ip": "192.168.1.10",
                "destination_port": 22,  # SSH port scanning
                "protocol": "TCP",
                "suspicious": True,
            },
        )
        logger.info(f"Network reconnaissance logged: {digital_event.event_id}")

        await asyncio.sleep(2)

        # Step 3: Social engineering attempt (cognitive)
        # Simulate unusual user behavior
        for _ in range(3):
            self.integrated_manager.digital_manager.log_security_event(
                SecurityEventType.PRIVILEGE_ESCALATION,
                {
                    "user_id": "compromised.user",
                    "source_ip": "192.168.1.50",
                    "target_privilege": "admin",
                    "attempt_time": datetime.now().isoformat(),
                },
            )
            await asyncio.sleep(1)

        logger.warning("Multi-domain attack simulation complete")
        await asyncio.sleep(5)  # Allow time for correlation

    async def scenario_insider_threat(self):
        """Simulate insider threat detection through behavioral analysis."""
        logger.info("\n=== SCENARIO 5: Insider Threat (Behavioral Anomaly) ===")

        # Simulate unusual behavior patterns for a legitimate user
        suspicious_user = "insider.threat"

        # Unusual access times
        unusual_events = [
            {
                "event_type": SecurityEventType.DATA_ACCESS,
                "metadata": {
                    "user_id": suspicious_user,
                    "source_ip": "192.168.1.25",
                    "file_path": "/confidential/salary_data.xlsx",
                    "data_volume": 50 * 1024 * 1024,  # 50MB - unusual
                    "access_time": "02:30",  # Unusual hour
                },
            },
            {
                "event_type": SecurityEventType.FILE_ACCESS,
                "metadata": {
                    "user_id": suspicious_user,
                    "source_ip": "192.168.1.25",
                    "file_path": "/hr/employee_records/",
                    "access_type": "bulk_download",
                },
            },
        ]

        # Generate events for behavioral analysis
        for event_data in unusual_events:
            event = self.integrated_manager.digital_manager.log_security_event(
                event_data["event_type"], event_data["metadata"]
            )

            # Add to cognitive manager for behavioral analysis
            self.integrated_manager.cognitive_manager.add_security_event(event)

            logger.info(f"Suspicious event from {suspicious_user}: {event.event_type}")
            await asyncio.sleep(2)

        # Trigger behavioral analysis
        recent_events = list(
            self.integrated_manager.cognitive_manager.security_events_buffer
        )
        user_events = [
            e for e in recent_events if e.metadata.get("user_id") == suspicious_user
        ]

        if user_events:
            profile = self.integrated_manager.cognitive_manager.analyze_user_behavior(
                suspicious_user, user_events
            )
            logger.warning(
                f"Behavioral analysis for {suspicious_user}: {profile.behavior_type.value}"
            )

        await asyncio.sleep(3)

    async def show_security_dashboard(self):
        """Display the security dashboard."""
        logger.info("\n=== SECURITY DASHBOARD ===")

        dashboard = self.integrated_manager.get_security_dashboard()

        print(
            f"\n🛡️  Overall Security Score: {dashboard['overall_security_score']:.1f}/100"
        )
        print(f"⚠️  Open Incidents: {dashboard['open_incidents']}")

        print("\n📊 Active Threats by Domain:")
        for domain, count in dashboard["active_threats"].items():
            print(f"   {domain.title()}: {count}")

        print("\n🚨 Domain Status:")
        for domain, status in dashboard["domain_status"].items():
            status_emoji = {
                "secure": "🟢",
                "monitoring": "🟡",
                "elevated": "🟠",
                "critical": "🔴",
            }
            print(
                f"   {domain.title()}: {status_emoji.get(status, '⚪')} {status.title()}"
            )

        if dashboard["top_recommendations"]:
            print("\n💡 Top Recommendations:")
            for i, rec in enumerate(dashboard["top_recommendations"], 1):
                print(f"   {i}. {rec}")

        # Show recent alerts
        if dashboard["recent_alerts"]:
            print(f"\n🚨 Recent Alerts ({len(dashboard['recent_alerts'])}):")
            for alert in dashboard["recent_alerts"][:5]:  # Show top 5
                print(
                    f"   [{alert['domain'].title()}] {alert['type']} - {alert['severity']}"
                )

        # Show integrated threats if any
        integrated_threats = self.integrated_manager.get_integrated_threats()
        if integrated_threats:
            print(f"\n⚡ Integrated Threats ({len(integrated_threats)}):")
            for threat in integrated_threats:
                domains_str = ", ".join([d.value for d in threat.affected_domains])
                print(
                    f"   {threat.threat_id}: {domains_str} - {threat.combined_severity.value}"
                )

        # Show security incidents
        incidents = self.integrated_manager.get_security_incidents()
        if incidents:
            print(f"\n📋 Security Incidents ({len(incidents)}):")
            for incident in incidents:
                print(
                    f"   {incident.incident_id}: {incident.title} - {incident.status}"
                )


async def main():
    """Main demonstration function."""
    print("🔐 GEO-INFER-SEC Comprehensive Security Demonstration")
    print("=" * 60)

    # Initialize demo environment
    demo = SecurityDemoEnvironment()

    try:
        # Run security scenarios
        await demo.simulate_security_scenarios()

        print("\n✅ Security demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Physical Security: Access control, surveillance, zone management")
        print(
            "• Digital Security: Threat detection, network monitoring, vulnerability assessment"
        )
        print(
            "• Cognitive Security: Behavioral analysis, anomaly detection, threat prediction"
        )
        print(
            "• Integrated Security: Cross-domain correlation, unified incident response"
        )

    except KeyboardInterrupt:
        print("\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise
    finally:
        # Cleanup
        try:
            demo.integrated_manager.stop_integrated_monitoring()
        except Exception:
            pass


def run_demo():
    """Run the comprehensive security demonstration."""
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to run demonstration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_demo()
