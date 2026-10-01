"""Comprehensive tests for all METAGOV modules."""

import pytest

from geo_infer_metagov.core.accountability import AccountabilityFramework
from geo_infer_metagov.core.adaptation import AdaptiveGovernanceSystem
from geo_infer_metagov.core.polycentric import PolycentricGovernanceSystem


class TestAccountabilityFramework:
    """Test suite for accountability framework."""

    @pytest.fixture
    def framework(self):
        return AccountabilityFramework(
            accountability_model="multi_directional",
            transparency_level="full_disclosure",
            public_participation=True,
        )

    def test_framework_initialization(self, framework):
        assert framework.accountability_model == "multi_directional"
        assert framework.transparency_level == "full_disclosure"
        assert framework.public_participation is True

    def test_establish_accountability(self, framework):
        mechanisms = framework.establish_accountability(
            governing_bodies=[{"id": "entity_1", "name": "Authority 1"}],
            stakeholder_groups=[{"id": "sg_1", "name": "Group 1"}],
            accountability_directions=["upward_to_public", "downward_to_users"],
            enforcement_capacity="strong",
        )

        assert mechanisms is not None
        assert mechanisms.enforcement_capacity == "strong"
        assert len(mechanisms.audit_mechanisms) > 0

    def test_implement_transparency(self, framework):
        transparency = framework.implement_transparency(
            information_types=["decisions", "budgets", "outcomes"],
            disclosure_frequency="quarterly",
            accessibility_requirements=["multiple_languages", "digital_access"],
            documentation_standards="comprehensive",
        )

        assert transparency is not None
        assert transparency.disclosure_frequency == "quarterly"
        assert len(transparency.public_access_mechanisms) > 0

    def test_enable_participation(self, framework):
        participation = framework.enable_participation(
            participation_forms=["information_access", "consultation", "co-production"],
            barriers_to_remove=["language", "digital_access"],
            capacity_building="supported",
        )

        assert len(participation["participation_forms"]) == 3
        assert "accessibility_measures" in participation
        assert "participation_channels" in participation


class TestAdaptiveGovernanceSystem:
    """Test suite for adaptive governance system."""

    @pytest.fixture
    def system(self):
        return AdaptiveGovernanceSystem(
            learning_approach="adaptive_management",
            timeframe="multi_year_cycles",
            feedback_mechanisms="real_time",
        )

    def test_system_initialization(self, system):
        assert system.learning_approach == "adaptive_management"
        assert system.timeframe == "multi_year_cycles"
        assert len(system.adaptive_cycles) == 0

    def test_establish_adaptive_cycle(self, system):
        cycle = system.establish_adaptive_cycle(
            governance_domain="natural_resource_management",
            decision_frequency="annual_review",
            learning_mechanisms=["monitoring", "evaluation", "adjustment"],
            stakeholder_participation="continuous",
        )

        assert cycle is not None
        assert cycle.governance_domain == "natural_resource_management"
        assert cycle.monitoring_plan is not None

    def test_monitor_performance(self, system):
        results = system.monitor_performance(
            governance_indicators=["efficiency", "equity", "sustainability"],
            data_sources=["administrative", "stakeholder_feedback"],
            evaluation_periods="annual",
        )

        assert "indicators" in results
        assert "performance_scores" in results
        assert len(results["performance_scores"]) == 3

    def test_adapt_governance(self, system):
        adapted = system.adapt_governance(
            performance_results={"efficiency": 0.6, "equity": 0.7},
            learning_outcomes={
                "lessons": ["improve_transparency", "increase_participation"]
            },
            scenario_changes=[{"type": "environmental", "impact": "high"}],
            adaptation_pathways=[
                {"name": "pathway1", "description": "Improve transparency"},
                {"name": "pathway2", "description": "Increase participation"},
            ],
        )

        assert "adaptations_made" in adapted
        assert "pathways_selected" in adapted
        assert adapted["adaptations_made"] > 0


class TestPolycentricGovernanceSystem:
    """Test suite for polycentric governance system."""

    @pytest.fixture
    def system(self):
        return PolycentricGovernanceSystem(
            governance_model="polycentric",
            coordination_mechanism="network_based",
            redundancy_level="adaptive",
        )

    def test_system_initialization(self, system):
        assert system.governance_model == "polycentric"
        assert system.coordination_mechanism == "network_based"
        assert len(system.polycentric_designs) == 0

    def test_design_polycentric_structure(self, system):
        design = system.design_polycentric_structure(
            governing_bodies=[
                {"id": "body_1", "name": "Authority 1", "level": "local"},
                {"id": "body_2", "name": "Authority 2", "level": "regional"},
            ],
            jurisdictional_overlaps={"body_1": ["body_2"]},
            spatial_scales=["local", "regional", "national"],
            functional_domains=["water", "land", "air"],
            feedback_mechanisms={"monitoring": "continuous", "reporting": "quarterly"},
        )

        assert design is not None
        assert len(design.governing_bodies) == 2
        assert "redundancy_assessment" in design.__dict__

    def test_analyze_authority_relationships(self, system):
        analysis = system.analyze_authority_relationships(
            authorities=[
                {"id": "a1", "name": "Authority 1"},
                {"id": "a2", "name": "Authority 2"},
            ],
            relationships=["coordination", "competition", "subsidiarity"],
            effectiveness_measures=["efficiency", "legitimacy", "equity"],
        )

        assert "authority_count" in analysis
        assert analysis["authority_count"] == 2
        assert "coordination_index" in analysis


class TestIntegrationScenarios:
    """Integration tests for complete governance scenarios."""

    def test_complete_governance_workflow(self):
        """Test a complete governance workflow integration."""
        from geo_infer_metagov import (
            MultiLevelGovernanceFramework,
            StakeholderGovernanceCoordinator,
            InstitutionalDesigner,
            AccountabilityFramework,
        )

        # 1. Create multi-level governance
        mlg = MultiLevelGovernanceFramework()
        governance = mlg.design_governance_structure(
            spatial_scope={"name": "Test Region"},
            stakeholder_groups=[{"name": "Group1"}, {"name": "Group2"}],
            decision_domains=["domain1", "domain2"],
            time_horizons=[1, 5],
        )

        assert governance is not None

        # 2. Analyze stakeholders
        coordinator = StakeholderGovernanceCoordinator()
        analysis = coordinator.analyze_stakeholders(
            governance_domain="test",
            spatial_extent={},
            stakeholder_categories=["government", "community"],
        )

        assert analysis is not None
        assert "stakeholder_groups" in analysis

        # 3. Design institutions
        designer = InstitutionalDesigner()
        inst_design = designer.apply_ostrom_principles(
            principle_set=["clear_boundaries", "monitoring"],
            resource_system={},
            governance_context={},
        )

        assert inst_design is not None

        # 4. Establish accountability
        framework = AccountabilityFramework()
        accountability = framework.establish_accountability(
            governing_bodies=[{"id": "e1"}],
            stakeholder_groups=[{"id": "s1"}],
            accountability_directions=["upward"],
            enforcement_capacity="strong",
        )

        assert accountability is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
