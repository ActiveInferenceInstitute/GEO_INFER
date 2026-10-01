"""Organizational integration for governance-organization alignment."""

from typing import Any
import logging

logger = logging.getLogger(__name__)

# Optional organizational integration
try:
    from geo_infer_org.core import OrganizationModel  # type: ignore[import-untyped]  # noqa: F401

    ORG_AVAILABLE = True
except ImportError:
    ORG_AVAILABLE = False
    logger.warning("GEO-INFER-ORG not available, organizational features disabled")


class OrganizationalGovernanceIntegration:
    """
    Integrate organizational structures with governance systems.

    Provides:
    - Governance entity to organizational role mapping
    - Organizational capacity assessment
    - Governance-organization alignment checking
    - Organizational learning integration

    References:
    - Organizational design for governance
    - Capacity building for governance institutions
    """

    def __init__(self) -> None:
        """Initialize organizational governance integration."""
        if ORG_AVAILABLE:
            self.org_available = True
        else:
            self.org_available = False
            logger.warning(
                "Organizational integration disabled - GEO-INFER-ORG not available"
            )

    def map_governance_to_organizational_structure(
        self,
        governance_entities: list[dict[str, Any]],
        organizational_structure: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Map governance entities to organizational roles and structure.

        Parameters:
        -----------
        governance_entities : List[Dict[str, Any]]
            Governance entities to map
        organizational_structure : Dict[str, Any]
            Organizational structure definition

        Returns:
        --------
        Dict[str, Any]
            Mapping between governance and organizational structures
        """
        entity_role_mapping_out: dict[str, dict[str, Any]] = {}
        mapping: dict[str, Any] = {
            "mapped": True,
            "entity_role_mapping": entity_role_mapping_out,
            "coverage": 0.0,
            "alignment_score": 0.0,
        }

        if not self.org_available:
            mapping["mapped"] = False
            mapping["reason"] = "Organizational module not available"
            return mapping

        # Extract organizational roles
        org_roles = organizational_structure.get("roles", [])
        _org_units = organizational_structure.get("units", [])

        # Map entities to roles
        for entity in governance_entities:
            entity_id = entity.get("entity_id", "unknown")
            entity_level = entity.get("governance_level", "unknown")
            entity_responsibilities = entity.get("responsibilities", [])

            # Find matching organizational role
            matched_role = None
            for role in org_roles:
                role_level = role.get("level", "")
                role_responsibilities = role.get("responsibilities", [])

                # Check for match
                if entity_level.lower() in role_level.lower():
                    # Check responsibility overlap
                    overlap = len(
                        set(entity_responsibilities) & set(role_responsibilities)
                    )
                    if overlap > 0 or not role_responsibilities:
                        matched_role = role
                        break

            entity_role_mapping_out[entity_id] = {
                "entity": entity,
                "matched_role": matched_role,
                "match_quality": 0.8 if matched_role else 0.3,
            }

        # Calculate coverage
        mapped_count = sum(
            1 for m in entity_role_mapping_out.values() if m["matched_role"] is not None
        )
        mapping["coverage"] = (
            mapped_count / len(governance_entities) if governance_entities else 0.0
        )

        # Calculate alignment score
        if entity_role_mapping_out:
            avg_match_quality = sum(
                float(m["match_quality"]) for m in entity_role_mapping_out.values()
            ) / len(entity_role_mapping_out)
            mapping["alignment_score"] = avg_match_quality

        return mapping

    def assess_organizational_capacity(
        self,
        governance_entities: list[dict[str, Any]],
        organizational_capacity_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Assess organizational capacity for governance functions.

        Parameters:
        -----------
        governance_entities : List[Dict[str, Any]]
            Governance entities requiring capacity
        organizational_capacity_data : Dict[str, Any]
            Organizational capacity information

        Returns:
        --------
        Dict[str, Any]
            Capacity assessment results
        """
        entity_capacity_out: dict[str, Any] = {}
        capacity_gaps_out: list[dict[str, Any]] = []
        assessment: dict[str, Any] = {
            "capacity_assessed": True,
            "entity_capacity": entity_capacity_out,
            "overall_capacity": 0.0,
            "capacity_gaps": capacity_gaps_out,
        }

        if not self.org_available:
            assessment["capacity_assessed"] = False
            assessment["reason"] = "Organizational module not available"
            return assessment

        # Assess capacity for each entity
        for entity in governance_entities:
            entity_id = entity.get("entity_id", "unknown")
            entity_responsibilities = entity.get("responsibilities", [])

            # Get capacity data for this entity's domain
            capacity_factors = {
                "staffing": organizational_capacity_data.get("staffing_level", 0.5),
                "budget": organizational_capacity_data.get("budget_adequacy", 0.5),
                "expertise": organizational_capacity_data.get("expertise_level", 0.5),
                "systems": organizational_capacity_data.get("system_capacity", 0.5),
            }

            # Calculate overall capacity
            entity_capacity = float(
                sum(float(v) for v in capacity_factors.values())
            ) / max(len(capacity_factors), 1)

            entity_capacity_out[entity_id] = {
                "capacity_score": entity_capacity,
                "capacity_factors": capacity_factors,
                "responsibilities": entity_responsibilities,
            }

            # Identify capacity gaps
            if entity_capacity < 0.6:
                capacity_gaps_out.append(
                    {
                        "entity_id": entity_id,
                        "capacity_score": entity_capacity,
                        "gap_severity": "high" if entity_capacity < 0.4 else "medium",
                    }
                )

        # Calculate overall capacity
        if entity_capacity_out:
            assessment["overall_capacity"] = sum(
                float(e["capacity_score"]) for e in entity_capacity_out.values()
            ) / len(entity_capacity_out)

        return assessment

    def check_governance_organization_alignment(
        self,
        governance_structure: dict[str, Any],
        organizational_structure: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Check alignment between governance structure and organizational structure.

        Parameters:
        -----------
        governance_structure : Dict[str, Any]
            Governance structure definition
        organizational_structure : Dict[str, Any]
            Organizational structure definition

        Returns:
        --------
        Dict[str, Any]
            Alignment assessment
        """
        alignment_factors_out: dict[str, float] = {}
        misalignments_out: list[dict[str, Any]] = []
        alignment: dict[str, Any] = {
            "alignment_checked": True,
            "overall_alignment": 0.0,
            "alignment_factors": alignment_factors_out,
            "misalignments": misalignments_out,
        }

        if not self.org_available:
            alignment["alignment_checked"] = False
            alignment["reason"] = "Organizational module not available"
            return alignment

        # Check structural alignment
        gov_levels = governance_structure.get("governance_levels", [])
        org_levels = organizational_structure.get("levels", [])

        level_alignment = len(
            set(str(level).lower() for level in gov_levels)
            & set(str(level).lower() for level in org_levels)
        ) / max(1, len(gov_levels))
        alignment_factors_out["level_alignment"] = level_alignment

        # Check responsibility alignment
        gov_entities = governance_structure.get("entities", [])
        org_roles = organizational_structure.get("roles", [])

        all_gov_responsibilities = set()
        for entity in gov_entities:
            all_gov_responsibilities.update(entity.get("responsibilities", []))

        all_org_responsibilities = set()
        for role in org_roles:
            all_org_responsibilities.update(role.get("responsibilities", []))

        responsibility_alignment = len(
            all_gov_responsibilities & all_org_responsibilities
        ) / max(1, len(all_gov_responsibilities))
        alignment_factors_out["responsibility_alignment"] = responsibility_alignment

        # Identify misalignments
        missing_in_org = all_gov_responsibilities - all_org_responsibilities
        if missing_in_org:
            misalignments_out.append(
                {
                    "type": "missing_organizational_roles",
                    "description": f"Governance responsibilities not covered by organization: {list(missing_in_org)[:5]}",
                    "severity": "high" if len(missing_in_org) > 3 else "medium",
                }
            )

        # Calculate overall alignment
        alignment["overall_alignment"] = (
            level_alignment * 0.4 + responsibility_alignment * 0.6
        )

        return alignment
