"""
GEO-INFER-CIV: Civic Engagement and Participatory Mapping

This module provides tools for community engagement, participatory mapping,
citizen science, and collaborative geospatial decision-making.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.participation import (
    ParticipationAnalyzer,
    ParticipationMethod,
    ParticipantRecord,
    EngagementScore,
    RepresentationReport,
)
from .core.civic_engagement import (
    AttendanceTracker,
    PublicCommentAnalyzer,
    VoterTurnoutModel,
    MeetingRecord,
    MeetingType,
    PublicComment,
    CommentCategory,
    AttendanceTrend as AttendanceTrend,
    CommentAnalysis as CommentAnalysis,
)
from .core.policy_analysis import (
    CostBenefitAnalyzer,
    StakeholderImpactAnalyzer,
    EquityAnalyzer,
    CostBenefitItem,
    StakeholderImpact,
    CostBenefitResult,
    EquityScore,
    ImpactLevel,
    PolicyDomain,
)

__all__ = [
    "ParticipationAnalyzer",
    "ParticipationMethod",
    "ParticipantRecord",
    "EngagementScore",
    "RepresentationReport",
    "AttendanceTracker",
    "PublicCommentAnalyzer",
    "VoterTurnoutModel",
    "MeetingRecord",
    "MeetingType",
    "PublicComment",
    "CommentCategory",
    "CostBenefitAnalyzer",
    "StakeholderImpactAnalyzer",
    "EquityAnalyzer",
    "CostBenefitItem",
    "StakeholderImpact",
    "ImpactLevel",
    "PolicyDomain",
    "CostBenefitResult",
    "EquityScore",
]
