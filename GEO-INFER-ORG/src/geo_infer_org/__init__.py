"""
GEO-INFER-ORG: Organizational Modeling

This module provides tools for organizational structure modeling,
governance, team coordination, and collaboration network analysis.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.organization import (
    OrganizationModel,
    OrgUnit,
    Role,
    Resource,
    OrgStructureType,
    RoleLevel,
    OrgMetrics,
)
from .core.governance import (
    VotingEngine,
    ConsensusModel,
    VotingMethod,
    DecisionStatus,
    Vote,
    Proposal,
    VotingResult,
)
from .core.collaboration import (
    CollaborationNetwork,
    TeamFormation,
    CollaborationEdge,
    CollaborationType,
    TeamMember,
    NetworkMetrics,
    TeamFormationResult,
)

__all__ = [
    "OrganizationModel",
    "OrgUnit",
    "Role",
    "Resource",
    "OrgStructureType",
    "RoleLevel",
    "OrgMetrics",
    "VotingEngine",
    "ConsensusModel",
    "VotingMethod",
    "DecisionStatus",
    "Vote",
    "Proposal",
    "VotingResult",
    "CollaborationNetwork",
    "TeamFormation",
    "CollaborationEdge",
    "CollaborationType",
    "TeamMember",
    "NetworkMetrics",
    "TeamFormationResult",
]
