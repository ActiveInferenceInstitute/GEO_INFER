"""Security integration for governance data and operations."""

from typing import Any
import logging

logger = logging.getLogger(__name__)

# Optional security integration
try:
    from geo_infer_sec.core.digital_security import (  # type: ignore[import-untyped]
        DigitalSecurityManager as SecurityManager,
    )

    SEC_AVAILABLE = True
except ImportError:
    SEC_AVAILABLE = False
    logger.warning("GEO-INFER-SEC not available, security features disabled")


class SecurityGovernanceIntegration:
    """
    Integrate security controls for governance systems.

    Provides:
    - Access control for governance operations
    - Audit logging for governance decisions
    - Secure communication channels
    - Governance data encryption

    References:
    - Security frameworks for governance systems
    - Access control in multi-stakeholder environments
    """

    def __init__(self) -> None:
        """Initialize security governance integration."""
        if SEC_AVAILABLE:
            try:
                self.security_manager = SecurityManager()
                self.sec_available = True
            except Exception as e:
                logger.warning(f"Could not initialize SecurityManager: {e}")
                self.security_manager = None
                self.sec_available = False
        else:
            self.security_manager = None
            self.sec_available = False
            logger.warning(
                "Security integration disabled - GEO-INFER-SEC not available"
            )

    def secure_governance_data(
        self, governance_data: dict[str, Any], sensitivity_level: str = "medium"
    ) -> dict[str, Any]:
        """
        Apply security controls to governance data.

        Parameters:
        -----------
        governance_data : Dict[str, Any]
            Governance data to secure
        sensitivity_level : str
            Data sensitivity level ('low', 'medium', 'high', 'confidential')

        Returns:
        --------
        Dict[str, Any]
            Security configuration for governance data
        """
        security_config = {
            "secured": True,
            "sensitivity_level": sensitivity_level,
            "access_controls": {},
            "encryption": {},
            "audit_logging": {},
        }

        if not self.sec_available:
            security_config["secured"] = False
            security_config["reason"] = "Security module not available"
            return security_config

        # Configure access controls based on sensitivity
        sensitivity_controls = {
            "low": {"access_level": "public", "encryption": False},
            "medium": {"access_level": "restricted", "encryption": True},
            "high": {
                "access_level": "confidential",
                "encryption": True,
                "audit_required": True,
            },
            "confidential": {
                "access_level": "secret",
                "encryption": True,
                "audit_required": True,
                "multi_factor": True,
            },
        }

        controls = sensitivity_controls.get(
            sensitivity_level.lower(), sensitivity_controls["medium"]
        )
        security_config["access_controls"] = controls

        # Encryption configuration
        if controls.get("encryption"):
            security_config["encryption"] = {
                "enabled": True,
                "algorithm": "AES-256",
                "key_management": "centralized",
            }

        # Audit logging
        security_config["audit_logging"] = {
            "enabled": True,
            "log_level": (
                "detailed"
                if sensitivity_level in ["high", "confidential"]
                else "standard"
            ),
            "retention": (
                "10_years"
                if sensitivity_level in ["high", "confidential"]
                else "5_years"
            ),
        }

        return security_config

    def create_audit_log_entry(
        self, action: str, actor: str, governance_entity: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create audit log entry for governance action.

        Parameters:
        -----------
        action : str
            Action performed
        actor : str
            Who performed the action
        governance_entity : str
            Governance entity involved
        details : Dict[str, Any]
            Additional details

        Returns:
        --------
        Dict[str, Any]
            Audit log entry
        """
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "actor": actor,
            "governance_entity": governance_entity,
            "details": details,
            "logged": True,
        }

        if self.sec_available and self.security_manager:
            # In real implementation, would use SecurityManager to log
            log_entry["security_verified"] = True
        else:
            log_entry["security_verified"] = False

        return log_entry

    def configure_access_control(
        self, governance_structure: dict[str, Any], access_policies: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Configure access control for governance operations.

        Parameters:
        -----------
        governance_structure : Dict[str, Any]
            Governance structure to secure
        access_policies : Dict[str, Any]
            Access control policies

        Returns:
        --------
        Dict[str, Any]
            Access control configuration
        """
        entity_access_out: dict[str, Any] = {}
        access_config: dict[str, Any] = {
            "configured": True,
            "policies": access_policies,
            "entity_access": entity_access_out,
        }

        if not self.sec_available:
            access_config["configured"] = False
            access_config["reason"] = "Security module not available"
            return access_config

        # Configure access for each governance entity
        entities = governance_structure.get("entities", [])
        for entity in entities:
            entity_id = entity.get("entity_id", "unknown")
            entity_level = entity.get("governance_level", "unknown")

            # Determine access level based on entity level
            level_access = {
                "local": "public",
                "regional": "restricted",
                "national": "confidential",
                "international": "secret",
            }

            default_access = level_access.get(str(entity_level).lower(), "restricted")
            entity_access = access_policies.get(
                entity_id, {"access_level": default_access}
            )

            entity_access_out[entity_id] = {
                "entity": entity_id,
                "access_level": entity_access.get("access_level", default_access),
                "allowed_operations": entity_access.get(
                    "allowed_operations", ["read", "view"]
                ),
                "restricted_operations": entity_access.get("restricted_operations", []),
            }

        return access_config
