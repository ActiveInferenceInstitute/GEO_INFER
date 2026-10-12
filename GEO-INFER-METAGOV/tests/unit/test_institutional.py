"""Unit tests for institutional design and analysis."""

import pytest

from geo_infer_metagov.core.institutional import (
    InstitutionalDesigner,
    Institution,
    InstitutionalFramework,
)


class TestInstitutionalDesigner:
    """Test suite for InstitutionalDesigner."""

    @pytest.fixture
    def designer(self):
        """Create test designer instance."""
        return InstitutionalDesigner(framework="iad")

    @pytest.fixture
    def institutions(self):
        """Create test institutions."""
        return [
            {"name": "Rule 1", "type": "boundary", "description": "Define boundaries"},
            {"name": "Rule 2", "type": "choice", "description": "Decision rules"},
            {
                "name": "Rule 3",
                "type": "monitoring",
                "description": "Monitor compliance",
            },
        ]

    @pytest.fixture
    def stakeholders(self):
        """Create test stakeholders."""
        return [
            {"name": "Group A", "category": "government"},
            {"name": "Group B", "category": "community"},
            {"name": "Group C", "category": "business"},
        ]

    def test_designer_initialization(self, designer):
        """Test designer initializes correctly."""
        assert designer.framework == InstitutionalFramework.IAD
        assert designer.context_type == "common_pool_resource"
        assert len(designer.institutional_analyses) == 0

    def test_analyze_institutions(self, designer, institutions, stakeholders):
        """Test institutional analysis."""
        analysis = designer.analyze_institutions(
            current_institutions=institutions,
            stakeholder_groups=stakeholders,
            resource_system={"name": "Shared Resource", "domain": "water"},
            decision_outcomes=[
                {"effectiveness": 0.7, "stakeholders": ["Group A", "Group B"]},
                {"effectiveness": 0.6, "stakeholders": ["Group C"]},
            ],
        )

        assert analysis is not None
        assert analysis.governance_domain == "water"
        assert len(analysis.existing_institutions) == 3
        assert len(analysis.recommendations) >= 0

    def test_institutional_effectiveness(self, designer, institutions, stakeholders):
        """Test institutional effectiveness assessment."""
        analysis = designer.analyze_institutions(
            current_institutions=institutions,
            stakeholder_groups=stakeholders,
            resource_system={"name": "Test Resource"},
            decision_outcomes=[{"effectiveness": 0.8, "stakeholders": ["Group A"]}],
        )

        assert len(analysis.institutional_effectiveness) > 0
        for effectiveness in analysis.institutional_effectiveness.values():
            assert 0 <= effectiveness <= 1.0

    def test_design_principles_assessment(self, designer, institutions, stakeholders):
        """Test design principles assessment."""
        analysis = designer.analyze_institutions(
            current_institutions=institutions,
            stakeholder_groups=stakeholders,
            resource_system={"name": "Test"},
            decision_outcomes=[],
        )

        principles = analysis.design_principles_assessment
        assert "clear_boundaries" in principles
        assert "monitoring" in principles
        assert "conflict_resolution" in principles
        assert all(0 <= v <= 1.0 for v in principles.values())

    def test_apply_ostrom_principles(self, designer):
        """Test applying Ostrom's design principles."""
        design = designer.apply_ostrom_principles(
            principle_set=[
                "clear_boundaries",
                "congruence",
                "collective_choice_arrangements",
                "monitoring",
            ],
            resource_system={"name": "Common Pool Resource"},
            governance_context={"scale": "local", "complexity": "high"},
        )

        assert "governance_design" in design
        assert len(design["governance_design"]) == 4
        assert "design_coherence" in design
        assert 0 <= design["design_coherence"] <= 1.0

    def test_ostrom_principle_implementation(self, designer):
        """Test Ostrom principle implementation strategies."""
        design = designer.apply_ostrom_principles(
            principle_set=["clear_boundaries", "monitoring"],
            resource_system={},
            governance_context={},
        )

        for principle_design in design["governance_design"].values():
            assert "description" in principle_design
            assert "implementation_strategy" in principle_design
            assert "expected_outcomes" in principle_design
            assert isinstance(principle_design["implementation_strategy"], list)

    def test_design_coherence_scoring(self, designer):
        """Test design coherence scoring."""
        design1 = designer.apply_ostrom_principles(
            principle_set=["clear_boundaries"],
            resource_system={},
            governance_context={},
        )

        design2 = designer.apply_ostrom_principles(
            principle_set=[
                "clear_boundaries",
                "congruence",
                "collective_choice_arrangements",
                "monitoring",
                "graduated_sanctions",
                "conflict_resolution",
                "right_to_organize",
            ],
            resource_system={},
            governance_context={},
        )

        # More principles should have higher coherence
        assert design2["design_coherence"] >= design1["design_coherence"]

    def test_multiple_analyses(self, designer, institutions, stakeholders):
        """Test creating multiple institutional analyses."""
        designer.analyze_institutions(
            current_institutions=institutions,
            stakeholder_groups=stakeholders,
            resource_system={"name": "Resource 1", "id": "r1"},
            decision_outcomes=[],
        )

        designer.analyze_institutions(
            current_institutions=institutions,
            stakeholder_groups=stakeholders,
            resource_system={"name": "Resource 2", "id": "r2"},
            decision_outcomes=[],
        )

        assert len(designer.institutional_analyses) == 2
        assert "r1" in designer.institutional_analyses
        assert "r2" in designer.institutional_analyses


class TestInstitution:
    """Test suite for Institution class."""

    def test_institution_creation(self):
        """Test institution creation."""
        institution = Institution(
            institution_id="rule_001",
            name="Boundary Rule",
            rule_type="boundary",
            description="Defines user boundaries",
            affected_stakeholders=["Group A", "Group B"],
            enforcement_mechanism="legal",
            effectiveness_rating=0.8,
        )

        assert institution.institution_id == "rule_001"
        assert institution.rule_type == "boundary"
        assert len(institution.affected_stakeholders) == 2
        assert institution.effectiveness_rating == 0.8

    def test_institution_rule_types(self):
        """Test different rule types."""
        rule_types = [
            "boundary",
            "position",
            "choice",
            "information",
            "aggregation",
            "payoff",
            "scope",
        ]

        for rule_type in rule_types:
            institution = Institution(
                institution_id=f"rule_{rule_type}",
                name=f"{rule_type.title()} Rule",
                rule_type=rule_type,
                description=f"Test {rule_type} rule",
                affected_stakeholders=[],
                enforcement_mechanism="informal",
            )

            assert institution.rule_type == rule_type


class TestInstitutionalFramework:
    """Test suite for InstitutionalFramework enum."""

    def test_framework_values(self):
        """Test framework enum values."""
        assert InstitutionalFramework.IAD.value == "iad"
        assert InstitutionalFramework.OSTROM.value == "ostrom"

    def test_framework_enumeration(self):
        """Test framework enumeration."""
        frameworks = list(InstitutionalFramework)
        assert len(frameworks) == 2
        assert InstitutionalFramework.IAD in frameworks
        assert InstitutionalFramework.OSTROM in frameworks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
