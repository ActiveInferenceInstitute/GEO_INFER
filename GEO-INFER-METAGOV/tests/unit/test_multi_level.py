"""Unit tests for multi-level governance framework."""

import pytest

from geo_infer_metagov.core.multi_level import (
    MultiLevelGovernanceFramework,
    GovernanceLevel,
    CoordinationMechanism,
    GovernanceEntity,
    GovernanceStructure,
)


class TestMultiLevelGovernanceFramework:
    """Test suite for MultiLevelGovernanceFramework."""

    @pytest.fixture
    def framework(self):
        """Create a test framework instance."""
        return MultiLevelGovernanceFramework(
            governance_levels=["local", "regional", "national"],
            coordination_mechanisms=["vertical_alignment", "horizontal_integration"],
            domain_coverage=["environmental", "civic", "commercial"],
        )

    def test_unknown_governance_level_raises(self) -> None:
        """Unknown level names must fail loudly instead of coercing to REGIONAL."""
        with pytest.raises(ValueError, match="Unknown governance level 'state'"):
            MultiLevelGovernanceFramework(governance_levels=["local", "state"])

    def test_level_names_case_insensitive(self) -> None:
        framework = MultiLevelGovernanceFramework(
            governance_levels=["LOCAL", "Regional"]
        )
        assert framework.governance_levels == [
            GovernanceLevel.LOCAL,
            GovernanceLevel.REGIONAL,
        ]

    @pytest.fixture
    def spatial_scope(self):
        """Create test spatial scope."""
        return {"name": "Test Region", "area_km2": 50000, "jurisdictions": 5}

    @pytest.fixture
    def stakeholders(self):
        """Create test stakeholder groups."""
        return [
            {"name": "Local Communities"},
            {"name": "Environmental Agencies"},
            {"name": "Water Users"},
            {"name": "Conservation NGOs"},
            {"name": "Business Community"},
        ]

    @pytest.fixture
    def decision_domains(self):
        """Create test decision domains."""
        return [
            "water_management",
            "land_use",
            "environmental_protection",
            "economic_development",
        ]

    def test_framework_initialization(self, framework):
        """Test framework initializes with correct parameters."""
        assert len(framework.governance_levels) == 3
        assert len(framework.coordination_mechanisms) == 2
        assert len(framework.domain_coverage) == 3
        assert len(framework.governance_structures) == 0

    def test_governance_levels_enum(self):
        """Test governance level enum values."""
        assert GovernanceLevel.LOCAL.value == "local"
        assert GovernanceLevel.REGIONAL.value == "regional"
        assert GovernanceLevel.NATIONAL.value == "national"

    def test_coordination_mechanisms_enum(self):
        """Test coordination mechanism enum values."""
        assert CoordinationMechanism.VERTICAL_ALIGNMENT.value == "vertical_alignment"
        assert (
            CoordinationMechanism.HORIZONTAL_INTEGRATION.value
            == "horizontal_integration"
        )

    def test_design_governance_structure(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test governance structure design."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5, 10, 20],
        )

        assert structure is not None
        assert structure.governance_id is not None
        assert len(structure.entities) == 3
        assert structure.spatial_scope["name"] == "Test Region"
        assert len(structure.decision_domains) == 4

    def test_governance_entities_created(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test that governance entities are properly created."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5, 10],
        )

        for entity in structure.entities:
            assert isinstance(entity, GovernanceEntity)
            assert entity.entity_id is not None
            assert entity.governance_level in framework.governance_levels
            assert len(entity.responsibilities) > 0

    def test_reporting_relationships(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test reporting relationships are established."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        assert len(structure.reporting_relationships) > 0
        # Check hierarchical structure
        local_entity = structure.entities[0]
        assert local_entity.entity_id in structure.reporting_relationships

    def test_information_flows_designed(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test information flows between entities."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        assert len(structure.information_flows) > 0
        for flows in structure.information_flows.values():
            assert isinstance(flows, list)

    def test_decision_escalation_rules(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test decision escalation rules are set."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5, 10],
        )

        assert len(structure.decision_escalation_rules) > 0
        for rules in structure.decision_escalation_rules.values():
            assert "implementation_level" in rules
            assert "escalation_condition" in rules

    def test_coordinate_vertical_levels(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test vertical level coordination."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        policy = {"id": "test_policy", "name": "Test Policy"}
        coordination = framework.coordinate_vertical_levels(
            governance_structure=structure, policy_proposal=policy
        )

        assert coordination is not None
        assert "level_approvals" in coordination
        assert "cross_level_conflicts" in coordination
        assert len(coordination["level_approvals"]) == 3

    def test_subsidiarity_principle(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test subsidiarity principle application."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        subsidiarity = framework.apply_subsidiarity_principle(
            governance_structure=structure, decision_domain="water_management"
        )

        assert subsidiarity is not None
        assert "subsidiary_level" in subsidiarity
        assert subsidiarity["subsidiary_level"] in [
            "local",
            "regional",
            "national",
            "watershed",
        ]
        assert "escalation_needed" in subsidiarity

    def test_multiple_governance_structures(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test creating multiple governance structures."""
        structure1 = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        scope2 = {"name": "Test Region 2", "area_km2": 30000, "jurisdictions": 3}
        structure2 = framework.design_governance_structure(
            spatial_scope=scope2,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        assert structure1.governance_id != structure2.governance_id
        assert len(framework.governance_structures) == 2

    def test_governance_structure_persistence(
        self, framework, spatial_scope, stakeholders, decision_domains
    ):
        """Test that governance structures are stored properly."""
        structure = framework.design_governance_structure(
            spatial_scope=spatial_scope,
            stakeholder_groups=stakeholders,
            decision_domains=decision_domains,
            time_horizons=[1, 5],
        )

        retrieved = framework.governance_structures.get(structure.governance_id)
        assert retrieved is not None
        assert retrieved.governance_id == structure.governance_id


class TestGovernanceEntity:
    """Test suite for GovernanceEntity."""

    def test_entity_creation(self):
        """Test governance entity creation."""
        entity = GovernanceEntity(
            entity_id="test_entity",
            name="Test Authority",
            governance_level=GovernanceLevel.LOCAL,
            jurisdiction={"name": "Test"},
            responsibilities=["water_management", "land_use"],
            authority_domain="environmental",
        )

        assert entity.entity_id == "test_entity"
        assert entity.name == "Test Authority"
        assert entity.governance_level == GovernanceLevel.LOCAL
        assert len(entity.responsibilities) == 2

    def test_entity_default_values(self):
        """Test entity default values."""
        entity = GovernanceEntity(
            entity_id="test",
            name="Test",
            governance_level=GovernanceLevel.REGIONAL,
            jurisdiction={},
            responsibilities=[],
            authority_domain="general",
        )

        assert entity.capacity == 1.0
        assert len(entity.stakeholders) == 0
        assert len(entity.resources) == 0


class TestGovernanceStructure:
    """Test suite for GovernanceStructure."""

    def test_structure_creation(self):
        """Test governance structure creation."""
        entities = [
            GovernanceEntity(
                entity_id="local",
                name="Local Authority",
                governance_level=GovernanceLevel.LOCAL,
                jurisdiction={},
                responsibilities=["implementation"],
                authority_domain="local",
            )
        ]

        structure = GovernanceStructure(
            governance_id="test_structure",
            spatial_scope={"name": "Test"},
            governance_levels=[GovernanceLevel.LOCAL],
            entities=entities,
            coordination_mechanisms=[CoordinationMechanism.VERTICAL_ALIGNMENT],
            decision_domains=["water"],
            stakeholder_groups=["communities"],
        )

        assert structure.governance_id == "test_structure"
        assert len(structure.entities) == 1
        assert len(structure.decision_domains) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
