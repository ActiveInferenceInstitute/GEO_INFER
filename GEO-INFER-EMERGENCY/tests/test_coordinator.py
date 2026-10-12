"""
Unit tests for EmergencyCoordinator and ResourceDeployer.
"""

import pytest
from geo_infer_emergency.core.coordinator import (
    EmergencyCoordinator,
    Incident,
    IncidentType,
    IncidentScale,
    Agency,
)
from geo_infer_emergency.core.resources import (
    ResourceDeployer,
    Resource,
    ResourceType,
    ResourceStatus,
)


class TestEmergencyCoordinator:
    """Test suite for EmergencyCoordinator class."""

    @pytest.fixture
    def coordinator(self):
        """Create an EmergencyCoordinator instance."""
        return EmergencyCoordinator(
            command_structure="ics",
            agencies=["fire", "police", "medical"],
            communication_protocol="secure",
        )

    def test_init_default(self):
        """Test default initialization."""
        coord = EmergencyCoordinator()
        assert coord.command_structure == "ics"
        assert coord.communication_protocol == "secure"

    def test_init_with_agencies(self, coordinator):
        """Test initialization with agencies."""
        assert len(coordinator.agencies) == 3

    def test_register_agency(self, coordinator):
        """Test registering an agency."""
        agency = Agency(
            agency_id="agency_ema",
            name="Emergency Management Agency",
            agency_type="coordination",
        )
        coordinator.register_agency(agency)
        assert "agency_ema" in coordinator.agencies

    def test_coordinate(self, coordinator):
        """Test incident coordination."""
        incident = {
            "id": "inc_001",
            "type": "wildfire",
            "name": "Test Fire",
            "location": {"lat": 34.0, "lon": -118.0},
            "scale": "type_3",
        }

        result = coordinator.coordinate(
            incident=incident,
            agencies=["agency_fire", "agency_police"],
            resources={"engines": [1, 2, 3], "personnel": [1, 2]},
            incident_action_plan=None,
        )

        assert result["incident_id"] == "inc_001"
        assert len(result["responding_agencies"]) == 2
        assert "resource_assignments" in result
        assert "communication_channels" in result

    def test_establish_command(self, coordinator):
        """Test establishing incident command."""
        result = coordinator.establish_command(
            incident_type="wildfire",
            location={"lat": 34.0, "lon": -118.0},
            scale="type_2",
            command_structure={
                "incident_commander": "Chief Smith",
                "operations": "Captain Jones",
                "planning": "Lt. Brown",
            },
        )

        assert result["incident_type"] == "wildfire"
        assert result["scale"] == "type_2"
        assert result["status"] == "established"
        assert result["command_structure"]["incident_commander"] == "Chief Smith"

    def test_establish_command_rejects_unknown_incident_type(self, coordinator):
        """Unknown incident_type raises descriptively instead of coercing to OTHER."""
        before = len(coordinator.get_active_incidents())
        with pytest.raises(ValueError, match="unknown incident_type"):
            coordinator.establish_command(
                incident_type="wildfre",
                location={"lat": 34.0, "lon": -118.0},
                scale="type_2",
                command_structure={"incident_commander": "Chief Doe"},
            )
        assert len(coordinator.get_active_incidents()) == before

    def test_establish_command_rejects_unknown_scale(self, coordinator):
        """Unknown scale raises descriptively instead of coercing to TYPE_3."""
        before = len(coordinator.get_active_incidents())
        with pytest.raises(ValueError, match="unknown scale"):
            coordinator.establish_command(
                incident_type="flood",
                location={"lat": 34.0, "lon": -118.0},
                scale="type9",
                command_structure={"incident_commander": "Chief Doe"},
            )
        assert len(coordinator.get_active_incidents()) == before

    def test_request_mutual_aid(self, coordinator):
        """Test mutual aid request."""
        result = coordinator.request_mutual_aid(
            requesting_agency="agency_fire",
            resource_needs=["engines", "personnel"],
            duration_hours=24,
            staging_areas=[
                {"id": "staging_1", "location": {"lat": 34.0, "lon": -118.0}}
            ],
        )

        assert result["requesting_agency"] == "agency_fire"
        assert "engines" in result["resource_needs"]
        assert "staging_areas" in result

    def test_generate_sitrep(self, coordinator):
        """Test situation report generation."""
        incident = {
            "id": "inc_001",
            "name": "Test Incident",
            "status": "active",
            "percent_contained": 25,
            "personnel_count": 100,
        }

        sitrep = coordinator.generate_sitrep(
            incident=incident,
            update_frequency="hourly",
            distribution=["eoc", "region"],
            report_format="ics_209",
        )

        assert sitrep["incident_id"] == "inc_001"
        assert sitrep["format"] == "ics_209"
        assert "current_status" in sitrep
        assert "resources_committed" in sitrep

    def test_get_active_incidents(self, coordinator):
        """Test getting active incidents."""
        # Create some incidents first
        coordinator.coordinate(
            incident={
                "id": "inc_1",
                "type": "flood",
                "name": "Flood 1",
                "location": {},
                "scale": "type_3",
            },
            agencies=["agency_fire"],
            resources={},
        )

        incidents = coordinator.get_active_incidents()
        assert len(incidents) >= 1

    def test_establish_command_registers_active_incident(self, coordinator):
        """establish_command incidents appear in get_active_incidents."""
        result = coordinator.establish_command(
            incident_type="earthquake",
            location={"lat": 34.0, "lon": -118.0},
            scale="type_1",
            command_structure={"incident_commander": "Chief Doe"},
        )

        incident_ids = {i["incident_id"] for i in coordinator.get_active_incidents()}
        assert result["incident_id"] in incident_ids

    def test_assign_sector_deterministic(self):
        """Sector assignment is stable across processes (crc32, not hash())."""
        import zlib
        from geo_infer_emergency.core.coordinator import (
            Incident as IncidentDataclass,
            IncidentScale,
            IncidentType,
        )

        coordinator = EmergencyCoordinator()
        incident = IncidentDataclass(
            incident_id="inc_x",
            incident_type=IncidentType.WILDFIRE,
            name="Determinism Check",
            location={},
            scale=IncidentScale.TYPE_3,
        )

        expected = ["Alpha", "Bravo", "Charlie", "Delta"][
            zlib.crc32(b"agency_fire") % 4
        ]
        assert coordinator._assign_sector("agency_fire", incident) == expected
        assert coordinator._assign_sector(
            "agency_fire", incident
        ) == EmergencyCoordinator()._assign_sector("agency_fire", incident)

    def test_coordinate_counts_dict_resource_values(self, coordinator):
        """Dict-shaped resource entries are counted, not reported as key counts."""
        plan = coordinator.coordinate(
            incident={
                "id": "inc_res",
                "type": "flood",
                "location": {},
                "scale": "type_3",
            },
            agencies=["agency_fire"],
            resources={"engines": {"engine_1": 2, "engine_2": 3}},
        )

        fire_assignment = [
            a for a in plan["resource_assignments"] if a["agency"] == "agency_fire"
        ][0]
        engines = [r for r in fire_assignment["resources"] if r["type"] == "engines"][0]
        assert engines["quantity"] == 5


class TestResourceDeployer:
    """Test suite for ResourceDeployer class."""

    @pytest.fixture
    def deployer(self):
        """Create a ResourceDeployer instance."""
        return ResourceDeployer(
            resource_types=["engines", "ambulances"],
            optimization_algorithm="greedy_nearest_resource",
            real_time_updates=True,
        )

    def test_init_default(self):
        """Test default initialization."""
        deployer = ResourceDeployer()
        assert "engines" in deployer.resource_types
        assert deployer.optimization_algorithm == "greedy_nearest_resource"

    def test_register_resource(self, deployer):
        """Test registering a resource."""
        resource = Resource(
            resource_id="eng_001",
            resource_type=ResourceType.ENGINE,
            name="Engine 1",
            location={"lat": 34.0, "lon": -118.0},
        )
        deployer.register_resource(resource)
        assert "eng_001" in deployer._resources

    def test_optimize_allocation(self, deployer):
        """Test resource allocation optimization."""
        resources = [
            {
                "id": "eng_1",
                "type": "engine",
                "location": {"lat": 34.0, "lon": -118.0},
                "status": "available",
            },
            {
                "id": "eng_2",
                "type": "engine",
                "location": {"lat": 34.1, "lon": -118.1},
                "status": "available",
            },
        ]

        demand_points = [
            {"id": "fire_1", "location": {"lat": 34.05, "lon": -118.05}},
            {"id": "fire_2", "location": {"lat": 34.15, "lon": -118.15}},
        ]

        result = deployer.optimize_allocation(
            resources=resources,
            demand_points=demand_points,
            constraints={"response_time": 15, "coverage": 0.8},
            objectives=["minimize_response_time", "maximize_coverage"],
        )

        assert "allocations" in result
        assert "metrics" in result
        assert result["metrics"]["total_resources"] == 2

    def test_dynamic_redeploy(self, deployer):
        """Test dynamic redeployment."""
        # Register some resources first
        resource = Resource(
            resource_id="eng_001",
            resource_type=ResourceType.ENGINE,
            name="Engine 1",
            location={"lat": 34.0, "lon": -118.0},
        )
        deployer.register_resource(resource)

        result = deployer.dynamic_redeploy(
            current_positions=[
                {"resource_id": "eng_001", "location": {"lat": 34.0, "lon": -118.0}}
            ],
            pending_incidents=[],
            predicted_demand={"high_risk_areas": [{"lat": 34.5, "lon": -118.5}]},
            strategy="move_up",
        )

        assert result["strategy"] == "move_up"
        assert "redeployments" in result

    def test_manage_staging(self, deployer):
        """Test staging area management."""
        result = deployer.manage_staging(
            staging_areas=[
                {
                    "id": "stage_1",
                    "location": {"lat": 34.0, "lon": -118.0},
                    "capacity": 20,
                }
            ],
            incoming_resources=[{"id": "eng_1"}, {"id": "eng_2"}],
            assignment_queue=[
                {
                    "id": "assign_1",
                    "incident": "inc_1",
                    "resources_needed": 2,
                    "severity": 8,
                }
            ],
            prioritization="incident_severity",
        )

        assert len(result["staging_areas"]) == 1
        assert result["staging_areas"][0]["current_count"] == 2

    def test_track_resources(self, deployer):
        """Test resource tracking."""
        resources = [
            {
                "id": "eng_1",
                "type": "engine",
                "status": "available",
                "location": {"lat": 34.0, "lon": -118.0},
            },
            {"id": "eng_2", "type": "engine", "status": "on_scene"},
            {"id": "eng_3", "type": "engine", "status": "en_route"},
        ]

        result = deployer.track_resources(
            resources=resources,
            update_frequency="real_time",
            metrics=["location", "status", "availability"],
        )

        assert result["summary"]["total"] == 3
        assert result["summary"]["available"] == 1
        assert result["summary"]["on_scene"] == 1

    def test_get_resource_status(self, deployer):
        """Test getting individual resource status."""
        resource = Resource(
            resource_id="eng_001", resource_type=ResourceType.ENGINE, name="Engine 1"
        )
        deployer.register_resource(resource)

        status = deployer.get_resource_status("eng_001")
        assert status["resource_id"] == "eng_001"
        assert status["status"] == "available"


class TestIncident:
    """Test suite for Incident dataclass."""

    def test_create_incident(self):
        """Test creating an incident."""
        incident = Incident(
            incident_id="inc_001",
            incident_type=IncidentType.WILDFIRE,
            name="Test Fire",
            location={"lat": 34.0, "lon": -118.0},
            scale=IncidentScale.TYPE_3,
        )

        assert incident.incident_id == "inc_001"
        assert incident.incident_type == IncidentType.WILDFIRE
        assert incident.status == "active"


class TestResource:
    """Test suite for Resource dataclass."""

    def test_create_resource(self):
        """Test creating a resource."""
        resource = Resource(
            resource_id="eng_001", resource_type=ResourceType.ENGINE, name="Engine 1"
        )

        assert resource.resource_id == "eng_001"
        assert resource.status == ResourceStatus.AVAILABLE
        assert resource.capacity == 1
