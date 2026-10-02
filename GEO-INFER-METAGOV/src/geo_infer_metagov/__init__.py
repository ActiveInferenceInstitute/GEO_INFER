"""
GEO-INFER-METAGOV: Meta-Governance and Organizational Governance Methods

This module implements comprehensive meta-governance, organizational governance,
and multi-level governance coordination frameworks for autonomous geospatial systems.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"
__license__ = "CC-BY-NC-SA-4.0"

from geo_infer_metagov.core.multi_level import MultiLevelGovernanceFramework
from geo_infer_metagov.core.institutional import InstitutionalDesigner
from geo_infer_metagov.core.stakeholder import StakeholderGovernanceCoordinator
from geo_infer_metagov.core.polycentric import PolycentricGovernanceSystem
from geo_infer_metagov.core.adaptation import AdaptiveGovernanceSystem
from geo_infer_metagov.core.accountability import AccountabilityFramework
from geo_infer_metagov.core.conflict_resolution import (
    ConflictResolver,
    ConflictResolutionMethod,
)
from geo_infer_metagov.core.performance import (
    PerformanceEvaluator,
    PerformanceMetrics,
    PerformanceDimension,
)
from geo_infer_metagov.core.scenarios import ScenarioPlanner, Scenario, ScenarioAnalysis
from geo_infer_metagov.core.advanced_analysis import AdvancedGovernanceAnalyzer

# Integration modules (optional)
try:
    from geo_infer_metagov.integrations.spatial import (
        SpatialGovernanceIntegration as SpatialGovernanceIntegration,
    )
    from geo_infer_metagov.integrations.organizational import (
        OrganizationalGovernanceIntegration as OrganizationalGovernanceIntegration,
    )
    from geo_infer_metagov.integrations.security import (
        SecurityGovernanceIntegration as SecurityGovernanceIntegration,
    )
    from geo_infer_metagov.integrations.normative import (
        NormativeGovernanceIntegration as NormativeGovernanceIntegration,
    )

    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

__all__ = [
    "MultiLevelGovernanceFramework",
    "InstitutionalDesigner",
    "StakeholderGovernanceCoordinator",
    "PolycentricGovernanceSystem",
    "AdaptiveGovernanceSystem",
    "AccountabilityFramework",
    "ConflictResolver",
    "ConflictResolutionMethod",
    "PerformanceEvaluator",
    "PerformanceMetrics",
    "PerformanceDimension",
    "ScenarioPlanner",
    "Scenario",
    "ScenarioAnalysis",
    "AdvancedGovernanceAnalyzer",
]

if INTEGRATIONS_AVAILABLE:
    __all__.extend(
        [
            "SpatialGovernanceIntegration",
            "OrganizationalGovernanceIntegration",
            "SecurityGovernanceIntegration",
            "NormativeGovernanceIntegration",
        ]
    )
