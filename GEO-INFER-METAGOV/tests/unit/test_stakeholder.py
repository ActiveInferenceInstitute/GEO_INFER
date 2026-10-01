"""Unit tests for stakeholder governance coordination."""

import pytest

from geo_infer_metagov.core.stakeholder import (
    StakeholderGovernanceCoordinator,
    Stakeholder,
    GovernancePlatform,
)


class TestStakeholderGovernanceCoordinator:
    """Test suite for StakeholderGovernanceCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a test coordinator instance."""
        return StakeholderGovernanceCoordinator(
            stakeholder_engagement_level="co-production",
            governance_approach="collaborative",
            equity_focus=True,
        )

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes correctly."""
        assert coordinator.engagement_level == "co-production"
        assert coordinator.governance_approach == "collaborative"
        assert coordinator.equity_focus is True
        assert len(coordinator.governance_platforms) == 0

    def test_analyze_stakeholders(self, coordinator):
        """Test stakeholder analysis."""
        analysis = coordinator.analyze_stakeholders(
            governance_domain="water_management",
            spatial_extent={"name": "Test Watershed", "area_km2": 50000},
            stakeholder_categories=["government", "community", "ngo", "business"],
        )

        assert analysis is not None
        assert "stakeholder_groups" in analysis
        assert len(analysis["stakeholder_groups"]) == 4
        assert "power_dynamics" in analysis
        assert "collaboration_potential" in analysis
        assert 0 <= analysis["collaboration_potential"] <= 1.0

    def test_stakeholder_group_creation(self, coordinator):
        """Test stakeholder groups are created with correct attributes."""
        analysis = coordinator.analyze_stakeholders(
            governance_domain="environmental",
            spatial_extent={"name": "Test"},
            stakeholder_categories=["government", "community"],
        )

        for stakeholder in analysis["stakeholder_groups"]:
            assert isinstance(stakeholder, Stakeholder)
            assert stakeholder.stakeholder_id is not None
            assert stakeholder.category is not None
            assert len(stakeholder.interests) > 0
            assert 0 <= stakeholder.influence_level <= 1.0
            assert 0 <= stakeholder.dependence_on_resource <= 1.0
            assert 0 <= stakeholder.decision_power <= 1.0

    def test_power_dynamics_analysis(self, coordinator):
        """Test power dynamics are analyzed."""
        analysis = coordinator.analyze_stakeholders(
            governance_domain="test",
            spatial_extent={},
            stakeholder_categories=["government", "community", "business"],
        )

        power_dynamics = analysis["power_dynamics"]
        assert "total_influence" in power_dynamics
        assert "total_power" in power_dynamics
        assert "power_concentration" in power_dynamics
        assert "power_balance_assessment" in power_dynamics
        assert power_dynamics["power_balance_assessment"] in [
            "balanced",
            "unbalanced",
            "relatively_balanced",
        ]

    def test_conflict_identification(self, coordinator):
        """Test conflicts are identified."""
        analysis = coordinator.analyze_stakeholders(
            governance_domain="test",
            spatial_extent={},
            stakeholder_categories=["government", "community", "business"],
        )

        conflicts = analysis["interest_conflicts"]
        assert isinstance(conflicts, list)

    def test_establish_governance_platform(self, coordinator):
        """Test governance platform establishment."""
        stakeholders = [
            {
                "name": "Group A",
                "category": "government",
                "influence": 0.8,
                "power": 0.7,
            },
            {
                "name": "Group B",
                "category": "community",
                "influence": 0.3,
                "power": 0.2,
            },
        ]

        platform = coordinator.establish_governance_platform(
            participants=stakeholders,
            governance_mechanisms=["participatory_workshops", "consensus_building"],
            decision_domains=["water", "land_use"],
            conflict_resolution_capacity=True,
        )

        assert isinstance(platform, GovernancePlatform)
        assert platform.platform_id is not None
        assert len(platform.stakeholders) == 2
        assert len(platform.governance_mechanisms) == 2
        assert platform.conflict_resolution_capacity is True

    def test_design_participatory_process(self, coordinator):
        """Test participatory process design."""
        stakeholder_groups = [
            {"name": "Group A"},
            {"name": "Group B"},
            {"name": "Group C"},
        ]

        process = coordinator.design_participatory_process(
            stakeholder_groups=stakeholder_groups,
            decision_type="collective_choice",
            equity_principles=["voice", "representation", "influence"],
            transparency_requirements=True,
        )

        assert "process_design" in process
        assert "equity_mechanisms" in process
        assert "timeline" in process
        assert process["process_design"]["transparency"] is True
        assert len(process["equity_mechanisms"]["principles"]) == 3

    def test_multiple_platforms(self, coordinator):
        """Test creating multiple governance platforms."""
        participants1 = [{"name": "Group A", "category": "government"}]
        participants2 = [{"name": "Group B", "category": "community"}]

        platform1 = coordinator.establish_governance_platform(
            participants=participants1,
            governance_mechanisms=["voting"],
            decision_domains=["water"],
            conflict_resolution_capacity=False,
        )

        platform2 = coordinator.establish_governance_platform(
            participants=participants2,
            governance_mechanisms=["consensus"],
            decision_domains=["land"],
            conflict_resolution_capacity=True,
        )

        assert platform1.platform_id != platform2.platform_id
        assert len(coordinator.governance_platforms) == 2


class TestStakeholder:
    """Test suite for Stakeholder class."""

    def test_stakeholder_creation(self):
        """Test stakeholder creation."""
        stakeholder = Stakeholder(
            stakeholder_id="test_stakeholder",
            name="Test Group",
            category="government",
            interests=["regulation", "equity"],
            influence_level=0.8,
            dependence_on_resource=0.5,
            decision_power=0.7,
        )

        assert stakeholder.stakeholder_id == "test_stakeholder"
        assert stakeholder.name == "Test Group"
        assert len(stakeholder.interests) == 2
        assert stakeholder.influence_level == 0.8

    def test_stakeholder_different_categories(self):
        """Test stakeholders of different categories."""
        categories = ["government", "community", "business", "ngo", "indigenous"]

        for category in categories:
            stakeholder = Stakeholder(
                stakeholder_id=f"test_{category}",
                name=f"{category.title()} Group",
                category=category,
                interests=[],
                influence_level=0.5,
                dependence_on_resource=0.5,
                decision_power=0.5,
            )

            assert stakeholder.category == category


class TestGovernancePlatform:
    """Test suite for GovernancePlatform class."""

    def test_platform_creation(self):
        """Test governance platform creation."""
        stakeholders = [
            Stakeholder(
                stakeholder_id="s1",
                name="Group A",
                category="government",
                interests=[],
                influence_level=0.7,
                dependence_on_resource=0.3,
                decision_power=0.8,
            )
        ]

        platform = GovernancePlatform(
            platform_id="test_platform",
            stakeholders=stakeholders,
            governance_mechanisms=["voting", "consensus"],
            decision_domains=["water", "land"],
            conflict_resolution_capacity=True,
            participation_level="co-production",
        )

        assert platform.platform_id == "test_platform"
        assert len(platform.stakeholders) == 1
        assert len(platform.governance_mechanisms) == 2
        assert platform.participation_level == "co-production"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
