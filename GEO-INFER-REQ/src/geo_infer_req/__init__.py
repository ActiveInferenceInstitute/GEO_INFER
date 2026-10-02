"""
GEO-INFER-REQ: Requirements and Dependencies

This module provides tools for requirements analysis, dependency resolution,
traceability tracking, and requirement validation.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.requirements import (
    RequirementsAnalyzer,
    Requirement,
    RequirementType,
    RequirementStatus,
    PriorityLevel,
    DependencyGraph,
    CompletenessReport,
)
from .core.traceability import (
    TraceabilityManager,
    TraceLink,
    ArtifactType,
    TraceMatrixEntry,
    CoverageReport,
    ImpactReport,
)
from .core.validation import (
    RequirementValidator,
    RequirementSpec,
    ValidationIssue,
    ValidationSeverity,
    ConflictType,
    ConflictDetectionResult,
    ConsistencyReport,
    FeasibilityAssessment,
)

__all__ = [
    "RequirementsAnalyzer",
    "Requirement",
    "RequirementType",
    "RequirementStatus",
    "PriorityLevel",
    "DependencyGraph",
    "CompletenessReport",
    "TraceabilityManager",
    "TraceLink",
    "ArtifactType",
    "TraceMatrixEntry",
    "CoverageReport",
    "ImpactReport",
    "RequirementValidator",
    "RequirementSpec",
    "ValidationIssue",
    "ValidationSeverity",
    "ConflictType",
    "ConflictDetectionResult",
    "ConsistencyReport",
    "FeasibilityAssessment",
]
