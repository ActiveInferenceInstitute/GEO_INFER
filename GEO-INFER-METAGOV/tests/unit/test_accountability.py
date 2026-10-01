"""Unit tests for accountability framework."""

import pytest

from geo_infer_metagov.core.accountability import (
    AccountabilityFramework,
)


class TestAccountabilityFramework:
    """Test suite for AccountabilityFramework."""

    @pytest.fixture
    def framework(self):
        """Create a test framework instance."""
        return AccountabilityFramework(
            accountability_model="multi_directional",
            transparency_level="full_disclosure",
            public_participation=True,
        )

    def test_framework_initialization(self, framework):
        """Test framework initializes correctly."""
        assert framework.accountability_model == "multi_directional"
        assert framework.transparency_level == "full_disclosure"
        assert framework.public_participation is True
        assert len(framework.accountability_systems) == 0
        assert len(framework.transparency_systems) == 0

    def test_establish_accountability(self, framework):
        """Test establishing accountability mechanisms."""
        mechanisms = framework.establish_accountability(
            governing_bodies=[{"id": "entity_1", "name": "Authority 1"}],
            stakeholder_groups=[{"id": "sg_1", "name": "Group 1"}],
            accountability_directions=["upward_to_public", "downward_to_users"],
            enforcement_capacity="strong",
        )

        assert mechanisms is not None
        assert mechanisms.mechanism_id is not None
        assert mechanisms.enforcement_capacity == "strong"
        assert len(mechanisms.audit_mechanisms) > 0
        assert hasattr(mechanisms, "audit_trail_structure")
        assert hasattr(mechanisms, "compliance_framework")

    def test_audit_trail_structure(self, framework):
        """Test audit trail structure generation."""
        mechanisms = framework.establish_accountability(
            governing_bodies=[{"id": "e1"}],
            stakeholder_groups=[{"id": "s1"}],
            accountability_directions=["upward"],
            enforcement_capacity="moderate",
        )

        assert hasattr(mechanisms, "audit_trail_structure")
        trail = mechanisms.audit_trail_structure
        assert "decision_tracking" in trail
        assert "action_tracking" in trail
        assert "stakeholder_interactions" in trail
        assert trail["decision_tracking"]["enabled"] is True

    def test_compliance_framework(self, framework):
        """Test compliance framework creation."""
        mechanisms = framework.establish_accountability(
            governing_bodies=[{"id": "e1"}],
            stakeholder_groups=[{"id": "s1"}],
            accountability_directions=["downward"],
            enforcement_capacity="strong",
        )

        assert hasattr(mechanisms, "compliance_framework")
        compliance = mechanisms.compliance_framework
        assert "compliance_checking" in compliance
        assert "violation_detection" in compliance
        assert "enforcement_actions" in compliance
        assert compliance["enforcement_actions"]["sanctions"] is True  # Strong capacity

    def test_implement_transparency(self, framework):
        """Test transparency system implementation."""
        transparency = framework.implement_transparency(
            information_types=["decisions", "budgets", "outcomes"],
            disclosure_frequency="quarterly",
            accessibility_requirements=["multiple_languages", "digital_access"],
            documentation_standards="comprehensive",
        )

        assert transparency is not None
        assert transparency.system_id is not None
        assert transparency.disclosure_frequency == "quarterly"
        assert len(transparency.public_access_mechanisms) > 0
        assert hasattr(transparency, "transparency_score")
        assert 0 <= transparency.transparency_score <= 1.0

    def test_transparency_score_calculation(self, framework):
        """Test transparency score calculation."""
        # High transparency
        high_transparency = framework.implement_transparency(
            information_types=[
                "decisions",
                "processes",
                "budgets",
                "outcomes",
                "conflicts_of_interest",
            ],
            disclosure_frequency="real_time",
            accessibility_requirements=[
                "multiple_languages",
                "digital_access",
                "traditional_access",
            ],
            documentation_standards="comprehensive",
        )

        # Low transparency
        low_transparency = framework.implement_transparency(
            information_types=["decisions"],
            disclosure_frequency="annual",
            accessibility_requirements=[],
            documentation_standards="minimal",
        )

        assert (
            high_transparency.transparency_score > low_transparency.transparency_score
        )
        assert high_transparency.transparency_score >= 0.7
        assert low_transparency.transparency_score < 0.5

    def test_enable_participation(self, framework):
        """Test enabling public participation."""
        participation = framework.enable_participation(
            participation_forms=["information_access", "consultation", "co-production"],
            barriers_to_remove=["language", "digital_access"],
            capacity_building="supported",
        )

        assert len(participation["participation_forms"]) == 3
        assert "accessibility_measures" in participation
        assert "participation_channels" in participation
        assert "barriers_addressed" in participation
        assert "capacity_building_support" in participation

    def test_multiple_accountability_systems(self, framework):
        """Test creating multiple accountability systems."""
        mechanisms1 = framework.establish_accountability(
            governing_bodies=[{"id": "e1"}],
            stakeholder_groups=[{"id": "s1"}],
            accountability_directions=["upward"],
            enforcement_capacity="strong",
        )

        mechanisms2 = framework.establish_accountability(
            governing_bodies=[{"id": "e2"}],
            stakeholder_groups=[{"id": "s2"}],
            accountability_directions=["downward"],
            enforcement_capacity="moderate",
        )

        assert mechanisms1.mechanism_id != mechanisms2.mechanism_id
        assert len(framework.accountability_systems) == 2

    def test_audit_mechanisms_by_direction(self, framework):
        """Test audit mechanisms vary by accountability direction."""
        upward = framework.establish_accountability(
            governing_bodies=[{"id": "e1"}],
            stakeholder_groups=[{"id": "s1"}],
            accountability_directions=["upward_to_public"],
            enforcement_capacity="strong",
        )

        downward = framework.establish_accountability(
            governing_bodies=[{"id": "e2"}],
            stakeholder_groups=[{"id": "s2"}],
            accountability_directions=["downward_to_users"],
            enforcement_capacity="strong",
        )

        assert "upward_reporting" in upward.audit_mechanisms
        assert "public_reporting" in downward.audit_mechanisms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
