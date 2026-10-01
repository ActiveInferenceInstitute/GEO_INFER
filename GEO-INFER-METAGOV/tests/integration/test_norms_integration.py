"""Integration tests for GEO-INFER-NORMS integration."""

import pytest

from geo_infer_metagov.integrations.normative import NormativeGovernanceIntegration


class TestNormativeIntegration:
    """Test suite for normative governance integration."""

    @pytest.fixture
    def norms_integration(self):
        """Create normative integration instance."""
        return NormativeGovernanceIntegration()

    def test_integration_initialization(self, norms_integration):
        """Test integration initializes correctly."""
        assert norms_integration is not None
        # Should work even if NORMS module not available

    def test_translate_governance_rules_to_norms(self, norms_integration):
        """Test translating governance rules to normative rules."""
        governance_rules = [
            {
                "id": "rule1",
                "type": "boundary",
                "description": "Define user boundaries",
            },
            {"id": "rule2", "type": "choice", "description": "Decision-making rules"},
        ]

        result = norms_integration.translate_governance_rules_to_norms(
            governance_rules, normative_framework="default"
        )

        assert "translated" in result
        if result.get("translated"):
            assert "normative_rules" in result
            assert "translation_quality" in result
            assert 0 <= result["translation_quality"] <= 1.0

    def test_check_compliance_with_norms(self, norms_integration):
        """Test compliance checking."""
        governance_actions = [
            {"id": "action1", "type": "decision", "actor": "entity1"},
            {"id": "action2", "type": "allocation", "actor": "entity2"},
        ]

        normative_rules = [
            {
                "norm_id": "norm1",
                "norm_type": "decision_norm",
                "conditions": ["stakeholder_consultation"],
            }
        ]

        result = norms_integration.check_compliance_with_norms(
            governance_actions, normative_rules
        )

        assert "checked" in result
        if result.get("checked"):
            assert "compliant_actions" in result
            assert "violations" in result
            assert "compliance_rate" in result
            assert 0 <= result["compliance_rate"] <= 1.0

    def test_detect_norm_violations(self, norms_integration):
        """Test norm violation detection."""
        governance_structure = {
            "entities": [
                {
                    "entity_id": "e1",
                    "responsibilities": ["water_management"],
                    "stakeholders": ["s1", "s2"],
                },
                {
                    "entity_id": "e2",
                    "responsibilities": ["land_use"],
                    "stakeholders": [],  # Missing stakeholders
                },
            ]
        }

        normative_rules = [
            {
                "norm_id": "norm1",
                "norm_type": "membership_norm",
                "conditions": ["stakeholders_required"],
            }
        ]

        result = norms_integration.detect_norm_violations(
            governance_structure, normative_rules
        )

        assert "violations_detected" in result
        assert "violations" in result
        assert "violation_count" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
