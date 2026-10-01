"""
Emergency coordination module.

Provides multi-agency incident command and coordination following
ICS (Incident Command System) and NIMS principles.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import zlib

logger = logging.getLogger(__name__)


class IncidentType(Enum):
    """Types of emergency incidents."""

    WILDFIRE = "wildfire"
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    HURRICANE = "hurricane"
    HAZMAT = "hazmat"
    MASS_CASUALTY = "mass_casualty"
    TERRORISM = "terrorism"
    CIVIL_UNREST = "civil_unrest"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class IncidentScale(Enum):
    """ICS incident scale classifications."""

    TYPE_5 = "type_5"  # Local, handled by initial response
    TYPE_4 = "type_4"  # Expanding incident
    TYPE_3 = "type_3"  # Extended attack, multi-discipline
    TYPE_2 = "type_2"  # Complex incident
    TYPE_1 = "type_1"  # Most complex, national significance


@dataclass
class Incident:
    """Represents an emergency incident."""

    incident_id: str
    incident_type: IncidentType
    name: str
    location: dict[str, Any]  # geometry or coordinates
    scale: IncidentScale
    status: str = "active"
    start_time: datetime = field(default_factory=datetime.now)
    description: str = ""
    affected_area: dict[str, Any] | None = None
    priority: int = 1


@dataclass
class Agency:
    """Represents a responding agency."""

    agency_id: str
    name: str
    agency_type: str  # fire, police, medical, public_works, etc.
    jurisdiction: dict[str, Any] | None = None
    contact_info: dict[str, str] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)


@dataclass
class IncidentCommand:
    """Incident Command System structure."""

    incident_commander: str
    command_location: dict[str, Any]
    operations_chief: str | None = None
    planning_chief: str | None = None
    logistics_chief: str | None = None
    finance_chief: str | None = None
    safety_officer: str | None = None
    liaison_officer: str | None = None
    public_info_officer: str | None = None


class EmergencyCoordinator:
    """
    Coordinate multi-agency emergency response following ICS principles.

    Supports incident command structure, mutual aid coordination,
    and situation reporting.
    """

    def __init__(
        self,
        command_structure: str = "ics",
        agencies: list[str] | None = None,
        communication_protocol: str = "secure",
        jurisdiction: dict[str, Any] | None = None,
    ):
        """
        Initialize emergency coordinator.

        Args:
            command_structure: Command structure type ('ics', 'nims', 'custom')
            agencies: Initial list of agencies
            communication_protocol: Communication security level
            jurisdiction: Jurisdictional boundary geometry
        """
        self.command_structure = command_structure
        self.agencies: dict[str, Agency] = {}
        self.communication_protocol = communication_protocol
        self.jurisdiction = jurisdiction
        self._active_incidents: dict[str, Incident] = {}
        self._incident_commands: dict[str, IncidentCommand] = {}
        self._mutual_aid_agreements: dict[str, list[str]] = {}

        # Register initial agencies
        if agencies:
            for agency_name in agencies:
                self.register_agency(
                    Agency(
                        agency_id=f"agency_{agency_name}",
                        name=agency_name,
                        agency_type=agency_name,
                    )
                )

        logger.info(
            f"Initialized EmergencyCoordinator with {command_structure} structure"
        )

    def register_agency(self, agency: Agency) -> None:
        """Register an agency in the coordination system."""
        self.agencies[agency.agency_id] = agency
        logger.info(f"Registered agency: {agency.name}")

    def coordinate(
        self,
        incident: dict[str, Any],
        agencies: list[str],
        resources: dict[str, Any],
        incident_action_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Coordinate response to an incident.

        Args:
            incident: Incident information
            agencies: Responding agencies
            resources: Available resources
            incident_action_plan: Optional IAP

        Returns:
            Coordination plan with assignments
        """
        incident_id = incident.get(
            "id", f"incident_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        # Create incident record
        incident_obj = Incident(
            incident_id=incident_id,
            incident_type=IncidentType(incident.get("type", "other")),
            name=incident.get("name", "Unnamed Incident"),
            location=incident.get("location", {}),
            scale=IncidentScale(incident.get("scale", "type_3")),
            description=incident.get("description", ""),
        )
        self._active_incidents[incident_id] = incident_obj

        # Create coordination plan
        coordination = {
            "incident_id": incident_id,
            "incident_name": incident_obj.name,
            "coordination_start": datetime.now().isoformat(),
            "command_structure": self.command_structure,
            "responding_agencies": agencies,
            "resource_assignments": [],
            "communication_channels": self._assign_channels(incident_obj),
            "operational_period": "12_hours",
            "status": "coordinating",
        }

        # Assign resources to agencies
        for agency_id in agencies:
            agency_resources = self._allocate_resources(
                agency_id, resources, incident_obj
            )
            coordination["resource_assignments"].append(
                {
                    "agency": agency_id,
                    "resources": agency_resources,
                    "sector": self._assign_sector(agency_id, incident_obj),
                }
            )

        # Include IAP if provided
        if incident_action_plan:
            coordination["incident_action_plan"] = incident_action_plan

        logger.info(
            f"Coordinated response for incident {incident_id} with {len(agencies)} agencies"
        )
        return coordination

    def _assign_channels(self, incident: Incident) -> dict[str, str]:
        """Assign communication channels for incident."""
        channels = {
            "command": f"CMD-{incident.incident_id[:8]}",
            "tactical": f"TAC-{incident.incident_id[:8]}",
            "medical": f"MED-{incident.incident_id[:8]}",
            "logistics": f"LOG-{incident.incident_id[:8]}",
        }
        return channels

    @staticmethod
    def _resource_quantity(resource_list: Any) -> int:
        """Count units in a resource entry of any supported shape."""
        if isinstance(resource_list, (list, tuple)):
            return len(resource_list)
        if isinstance(resource_list, dict):
            numeric = [v for v in resource_list.values() if isinstance(v, (int, float))]
            return int(sum(numeric)) if numeric else len(resource_list)
        if isinstance(resource_list, (int, float)):
            return int(resource_list)
        return 1

    def _allocate_resources(
        self, agency_id: str, resources: dict[str, Any], incident: Incident
    ) -> list[dict[str, Any]]:
        """Allocate resources to an agency based on capabilities."""
        allocated = []
        agency_type = agency_id.replace("agency_", "")

        # Resource type mapping by agency
        resource_mapping = {
            "fire": ["engines", "trucks", "personnel"],
            "police": ["patrol_units", "personnel", "barriers"],
            "medical": ["ambulances", "personnel", "supplies"],
            "public_works": ["heavy_equipment", "trucks", "personnel"],
        }

        appropriate_types = resource_mapping.get(agency_type, ["personnel"])

        for resource_type, resource_list in resources.items():
            if resource_type in appropriate_types:
                allocated.append(
                    {
                        "type": resource_type,
                        "quantity": self._resource_quantity(resource_list),
                        "assigned_at": datetime.now().isoformat(),
                    }
                )

        return allocated

    def _assign_sector(self, agency_id: str, incident: Incident) -> str:
        """Assign operational sector to agency."""
        # Deterministic sector assignment: zlib.crc32 is stable across
        # Python processes (unlike hash(), which is salt-randomized).
        sectors = ["Alpha", "Bravo", "Charlie", "Delta"]
        return sectors[zlib.crc32(agency_id.encode("utf-8")) % len(sectors)]

    def establish_command(
        self,
        incident_type: str,
        location: dict[str, Any],
        scale: str,
        command_structure: dict[str, str],
    ) -> dict[str, Any]:
        """
        Establish incident command structure.

        Args:
            incident_type: Type of incident
            location: Command post location
            scale: Incident scale
            command_structure: ICS positions and personnel

        Returns:
            Established command structure
        """
        incident_id = f"incident_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Register the incident in the active-incident registry so that
        # establish_command and coordinate() share one ID space and
        # get_active_incidents() sees command-post incidents.
        try:
            incident_type_enum = IncidentType(incident_type)
        except ValueError as exc:
            valid_types = sorted(t.value for t in IncidentType)
            raise ValueError(
                f"geo_infer_emergency.core.coordinator: unknown incident_type "
                f"{incident_type!r}. Valid types: {valid_types}"
            ) from exc
        try:
            scale_enum = IncidentScale(scale)
        except ValueError as exc:
            valid_scales = sorted(s.value for s in IncidentScale)
            raise ValueError(
                f"geo_infer_emergency.core.coordinator: unknown scale "
                f"{scale!r}. Valid scales: {valid_scales}"
            ) from exc
        self._active_incidents[incident_id] = Incident(
            incident_id=incident_id,
            incident_type=incident_type_enum,
            name=command_structure.get(
                "name", f"{incident_type.replace('_', ' ').title()} Incident"
            ),
            location=location,
            scale=scale_enum,
        )

        # Create incident command
        command = IncidentCommand(
            incident_commander=command_structure.get("incident_commander", ""),
            command_location=location,
            operations_chief=command_structure.get("operations"),
            planning_chief=command_structure.get("planning"),
            logistics_chief=command_structure.get("logistics"),
            finance_chief=command_structure.get("finance"),
            safety_officer=command_structure.get("safety"),
            liaison_officer=command_structure.get("liaison"),
            public_info_officer=command_structure.get("pio"),
        )

        self._incident_commands[incident_id] = command

        result = {
            "incident_id": incident_id,
            "incident_type": incident_type,
            "scale": scale,
            "command_post": location,
            "established_at": datetime.now().isoformat(),
            "command_structure": {
                "incident_commander": command.incident_commander,
                "operations_section": command.operations_chief,
                "planning_section": command.planning_chief,
                "logistics_section": command.logistics_chief,
                "finance_section": command.finance_chief,
                "command_staff": {
                    "safety_officer": command.safety_officer,
                    "liaison_officer": command.liaison_officer,
                    "public_info_officer": command.public_info_officer,
                },
            },
            "status": "established",
        }

        logger.info(
            f"Established command for {incident_type} incident at scale {scale}"
        )
        return result

    def request_mutual_aid(
        self,
        requesting_agency: str,
        resource_needs: list[str],
        duration_hours: int,
        staging_areas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Request mutual aid from neighboring jurisdictions.

        Args:
            requesting_agency: Agency requesting aid
            resource_needs: Types of resources needed
            duration_hours: Expected duration
            staging_areas: Staging area locations

        Returns:
            Mutual aid request and assignments
        """
        request_id = f"mutualaid_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Find available mutual aid partners
        available_partners = self._find_mutual_aid_partners(
            requesting_agency, resource_needs
        )

        assignments_out: list[dict[str, Any]] = []
        request: dict[str, Any] = {
            "request_id": request_id,
            "requesting_agency": requesting_agency,
            "resource_needs": resource_needs,
            "duration_hours": duration_hours,
            "staging_areas": staging_areas,
            "requested_at": datetime.now().isoformat(),
            "status": "pending",
            "potential_providers": available_partners,
            "assignments": assignments_out,
        }

        # Auto-assign based on agreements
        for need in resource_needs:
            for partner in available_partners:
                if need in partner.get("capabilities", []):
                    assignments_out.append(
                        {
                            "resource_type": need,
                            "providing_agency": partner["agency_id"],
                            "staging_area": staging_areas[0] if staging_areas else None,
                            "eta_hours": self._estimate_eta(partner, staging_areas),
                        }
                    )
                    break

        if assignments_out:
            request["status"] = "assigned"

        logger.info(
            f"Mutual aid request {request_id}: {len(assignments_out)} assignments"
        )
        return request

    def _find_mutual_aid_partners(
        self, requesting_agency: str, resource_needs: list[str]
    ) -> list[dict[str, Any]]:
        """Find available mutual aid partners."""
        partners = []
        for agency_id, agency in self.agencies.items():
            if agency_id != requesting_agency:
                partners.append(
                    {
                        "agency_id": agency_id,
                        "name": agency.name,
                        "capabilities": agency.capabilities
                        or resource_needs,  # Default to matching
                    }
                )
        return partners

    def _estimate_eta(
        self, partner: dict[str, Any], staging_areas: list[dict[str, Any]]
    ) -> float:
        """Estimate ETA for mutual aid resources."""
        # Simple estimate based on typical response times
        return 2.0  # hours

    def generate_sitrep(
        self,
        incident: dict[str, Any],
        update_frequency: str = "hourly",
        distribution: list[str] | None = None,
        report_format: str = "ics_209",
    ) -> dict[str, Any]:
        """
        Generate situation report.

        Args:
            incident: Incident to report on
            update_frequency: How often to update
            distribution: Distribution list
            report_format: Report format

        Returns:
            Situation report
        """
        incident_id = incident.get("id", "unknown")

        sitrep = {
            "report_id": f"sitrep_{incident_id}_{datetime.now().strftime('%Y%m%d%H%M')}",
            "incident_id": incident_id,
            "incident_name": incident.get("name", "Unknown"),
            "report_time": datetime.now().isoformat(),
            "format": report_format,
            "update_frequency": update_frequency,
            "distribution": distribution or ["eoc"],
            "current_status": {
                "incident_status": incident.get("status", "active"),
                "percent_contained": incident.get("percent_contained", 0),
                "current_threat": incident.get("threat_level", "moderate"),
                "weather_conditions": incident.get("weather", "unknown"),
            },
            "resources_committed": {
                "personnel": incident.get("personnel_count", 0),
                "engines": incident.get("engines", 0),
                "helicopters": incident.get("helicopters", 0),
                "other": incident.get("other_resources", []),
            },
            "actions_taken": incident.get("actions", []),
            "planned_actions": incident.get("planned_actions", []),
            "casualties": {
                "injuries": incident.get("injuries", 0),
                "fatalities": incident.get("fatalities", 0),
                "missing": incident.get("missing", 0),
            },
            "evacuations": {
                "ordered": incident.get("evacuation_ordered", False),
                "population_affected": incident.get("population_affected", 0),
            },
            "next_update": self._calculate_next_update(update_frequency),
        }

        logger.info(f"Generated sitrep for incident {incident_id}")
        return sitrep

    def _calculate_next_update(self, frequency: str) -> str:
        """Calculate next update time based on frequency."""
        from datetime import timedelta

        frequency_map = {
            "immediate": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
            "hourly": timedelta(hours=1),
            "4hour": timedelta(hours=4),
            "12hour": timedelta(hours=12),
            "daily": timedelta(days=1),
        }

        delta = frequency_map.get(frequency, timedelta(hours=1))
        return (datetime.now() + delta).isoformat()

    def get_active_incidents(self) -> list[dict[str, Any]]:
        """Get list of active incidents."""
        return [
            {
                "incident_id": inc.incident_id,
                "name": inc.name,
                "type": inc.incident_type.value,
                "scale": inc.scale.value,
                "status": inc.status,
                "start_time": inc.start_time.isoformat(),
            }
            for inc in self._active_incidents.values()
        ]
