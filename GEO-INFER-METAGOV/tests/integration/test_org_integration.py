"""Integration tests for GEO-INFER-ORG integration."""

import pytest

from geo_infer_metagov.integrations.organizational import (
    OrganizationalGovernanceIntegration,
)


class TestOrganizationalIntegration:
    """Test suite for organizational governance integration."""

    @pytest.fixture
    def org_integration(self):
        """Create organizational integration instance."""
        return OrganizationalGovernanceIntegration()

    def test_integration_initialization(self, org_integration):
        """Test integration initializes correctly."""
        assert org_integration is not None
        # Should work even if ORG module not available

    def test_map_governance_to_organizational_structure(self, org_integration):
        """Test governance to organizational mapping."""
        governance_entities = [
            {
                "entity_id": "gov1",
                "governance_level": "local",
                "responsibilities": ["water_management", "land_use"],
            },
            {
                "entity_id": "gov2",
                "governance_level": "regional",
                "responsibilities": ["coordination", "planning"],
            },
        ]

        organizational_structure = {
            "roles": [
                {
                    "id": "role1",
                    "level": "local",
                    "responsibilities": ["water_management"],
                },
                {
                    "id": "role2",
                    "level": "regional",
                    "responsibilities": ["coordination"],
                },
            ],
            "units": ["unit1", "unit2"],
        }

        result = org_integration.map_governance_to_organizational_structure(
            governance_entities, organizational_structure
        )

        assert "mapped" in result
        if result.get("mapped"):
            assert "entity_role_mapping" in result
            assert "coverage" in result
            assert "alignment_score" in result
            assert 0 <= result["coverage"] <= 1.0
            assert 0 <= result["alignment_score"] <= 1.0

    def test_assess_organizational_capacity(self, org_integration):
        """Test organizational capacity assessment."""
        governance_entities = [
            {"entity_id": "e1", "responsibilities": ["water_management"]}
        ]

        capacity_data = {
            "staffing_level": 0.7,
            "budget_adequacy": 0.6,
            "expertise_level": 0.8,
            "system_capacity": 0.75,
        }

        result = org_integration.assess_organizational_capacity(
            governance_entities, capacity_data
        )

        assert "capacity_assessed" in result
        if result.get("capacity_assessed"):
            assert "entity_capacity" in result
            assert "overall_capacity" in result
            assert "capacity_gaps" in result
            assert 0 <= result["overall_capacity"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
