"""Integration tests for GEO-INFER-SEC integration."""

import pytest

from geo_infer_metagov.integrations.security import SecurityGovernanceIntegration


class TestSecurityIntegration:
    """Test suite for security governance integration."""

    @pytest.fixture
    def sec_integration(self):
        """Create security integration instance."""
        return SecurityGovernanceIntegration()

    def test_integration_initialization(self, sec_integration):
        """Test integration initializes correctly."""
        assert sec_integration is not None
        # Should work even if SEC module not available

    def test_secure_governance_data(self, sec_integration):
        """Test securing governance data."""
        governance_data = {
            "governance_id": "gov1",
            "entities": [{"id": "e1"}],
            "decisions": [{"id": "d1", "sensitive": True}],
        }

        result = sec_integration.secure_governance_data(
            governance_data, sensitivity_level="high"
        )

        assert "secured" in result
        assert "sensitivity_level" in result
        assert "access_controls" in result
        assert "encryption" in result
        assert "audit_logging" in result

    def test_create_audit_log_entry(self, sec_integration):
        """Test audit log entry creation."""
        log_entry = sec_integration.create_audit_log_entry(
            action="decision_made",
            actor="governance_entity_1",
            governance_entity="entity_1",
            details={"decision_id": "d1", "rationale": "test"},
        )

        assert "timestamp" in log_entry
        assert "action" in log_entry
        assert "actor" in log_entry
        assert "governance_entity" in log_entry
        assert "logged" in log_entry

    def test_configure_access_control(self, sec_integration):
        """Test access control configuration."""
        governance_structure = {
            "entities": [
                {"entity_id": "e1", "governance_level": "local"},
                {"entity_id": "e2", "governance_level": "national"},
            ]
        }

        access_policies = {
            "e1": {"access_level": "public", "allowed_operations": ["read", "view"]},
            "e2": {"access_level": "confidential", "allowed_operations": ["read"]},
        }

        result = sec_integration.configure_access_control(
            governance_structure, access_policies
        )

        assert "configured" in result
        if result.get("configured"):
            assert "policies" in result
            assert "entity_access" in result
            assert len(result["entity_access"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
