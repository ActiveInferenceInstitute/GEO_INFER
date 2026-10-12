"""
Learning progress tracking and analytics module.

Provides learning analytics, competency tracking, gap identification,
and progress visualization for educational systems.
"""

import hashlib
import json
import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CompetencyLevel(Enum):
    """Competency achievement levels."""

    NOT_STARTED = "not_started"
    EMERGING = "emerging"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    EXEMPLARY = "exemplary"


_COMPETENCY_LEVEL_ORDER: dict[CompetencyLevel, int] = {
    CompetencyLevel.NOT_STARTED: 0,
    CompetencyLevel.EMERGING: 1,
    CompetencyLevel.DEVELOPING: 2,
    CompetencyLevel.PROFICIENT: 3,
    CompetencyLevel.EXEMPLARY: 4,
}
"""Ordinal ranking of competency levels; string values must never be compared lexicographically."""


@dataclass
class LearnerActivity:
    """Represents a learner's activity record."""

    activity_id: str
    activity_type: str  # exercise, reading, video, assessment
    topic: str
    start_time: datetime
    end_time: datetime | None = None
    completion_status: str = "in_progress"  # in_progress, completed, abandoned
    score: float | None = None
    time_spent_minutes: float = 0
    attempts: int = 1


@dataclass
class CompetencyRecord:
    """Tracks competency achievement for a learner."""

    competency_id: str
    competency_name: str
    level: CompetencyLevel
    evidence: list[str] = field(default_factory=list)
    last_assessed: datetime | None = None
    confidence: float = 0.0  # 0 to 1


@dataclass
class LearnerProgress:
    """Complete learning progress for a learner."""

    learner_id: str
    activities: list[LearnerActivity] = field(default_factory=list)
    competencies: dict[str, CompetencyRecord] = field(default_factory=dict)
    total_time_hours: float = 0
    completion_rate: float = 0
    current_streak_days: int = 0
    last_activity_date: datetime | None = None


class ProgressTracker:
    """
    Track and analyze learner progress with competency-based assessment.

    Provides comprehensive learning analytics, gap identification,
    and at-risk learner detection.
    """

    def __init__(
        self,
        competency_framework: str = "geospatial_bok",
        analytics_level: str = "detailed",
        privacy_compliance: str = "ferpa",
    ):
        """
        Initialize progress tracker.

        Args:
            competency_framework: Competency framework to use
            analytics_level: Detail level for analytics ('basic', 'detailed', 'comprehensive')
            privacy_compliance: Privacy standard to comply with ('ferpa', 'gdpr', 'none')
        """
        if privacy_compliance not in ("ferpa", "gdpr", "none"):
            raise ValueError(
                f"Unsupported privacy_compliance: {privacy_compliance!r} "
                "(expected 'ferpa', 'gdpr', or 'none')"
            )
        self.competency_framework = competency_framework
        self.analytics_level = analytics_level
        self.privacy_compliance = privacy_compliance
        self._learner_data: dict[str, LearnerProgress] = {}
        self._competency_definitions = self._load_competency_definitions()
        logger.info(
            f"Initialized ProgressTracker with {competency_framework} framework"
        )

    def _load_competency_definitions(self) -> dict[str, dict]:
        """Load competency definitions from framework."""
        # Geospatial Body of Knowledge competencies
        return {
            "spatial_thinking": {
                "name": "Spatial Thinking",
                "description": "Understanding spatial relationships and patterns",
                "skills": [
                    "spatial_reasoning",
                    "pattern_recognition",
                    "scale_awareness",
                ],
            },
            "data_acquisition": {
                "name": "Data Acquisition",
                "description": "Methods for obtaining geospatial data",
                "skills": ["remote_sensing", "gps", "digitizing", "surveying"],
            },
            "data_management": {
                "name": "Data Management",
                "description": "Managing and organizing spatial data",
                "skills": ["database_design", "data_quality", "metadata", "standards"],
            },
            "spatial_analysis": {
                "name": "Spatial Analysis",
                "description": "Analytical methods for spatial data",
                "skills": ["overlay", "buffer", "interpolation", "network_analysis"],
            },
            "geovisualization": {
                "name": "Geovisualization",
                "description": "Creating visual representations of spatial data",
                "skills": ["cartography", "interactive_mapping", "3d_visualization"],
            },
            "geospatial_programming": {
                "name": "Geospatial Programming",
                "description": "Programming for geospatial applications",
                "skills": [
                    "python_gis",
                    "javascript_mapping",
                    "spatial_sql",
                    "automation",
                ],
            },
        }

    def track_progress(
        self,
        learner_id: str,
        activity_log: list[dict[str, Any]],
        assessments: list[dict[str, Any]] | None = None,
    ) -> LearnerProgress:
        """
        Track learning progress for a learner.

        Args:
            learner_id: Unique learner identifier
            activity_log: List of activity records
            assessments: Optional list of assessment results

        Returns:
            Updated LearnerProgress object
        """
        # Get or create learner progress
        if learner_id not in self._learner_data:
            self._learner_data[learner_id] = LearnerProgress(learner_id=learner_id)

        progress = self._learner_data[learner_id]

        # Process activities
        for activity in activity_log:
            learner_activity = LearnerActivity(
                activity_id=activity.get("id", ""),
                activity_type=activity.get("type", "exercise"),
                topic=activity.get("topic", ""),
                start_time=activity.get("start_time", datetime.now()),
                end_time=activity.get("end_time"),
                completion_status=activity.get("status", "completed"),
                score=activity.get("score"),
                time_spent_minutes=activity.get("duration_minutes", 0),
                attempts=activity.get("attempts", 1),
            )
            progress.activities.append(learner_activity)

        # Process assessments and update competencies
        if assessments:
            for assessment in assessments:
                self._update_competencies(progress, assessment)

        # Calculate summary statistics
        progress.total_time_hours = (
            sum(a.time_spent_minutes for a in progress.activities) / 60
        )

        completed_activities = [
            a for a in progress.activities if a.completion_status == "completed"
        ]
        progress.completion_rate = (
            len(completed_activities) / len(progress.activities)
            if progress.activities
            else 0
        )

        if progress.activities:
            progress.last_activity_date = max(a.start_time for a in progress.activities)

        logger.info(
            f"Updated progress for learner {learner_id}: {len(progress.activities)} activities"
        )
        return progress

    def _update_competencies(
        self, progress: LearnerProgress, assessment: dict[str, Any]
    ) -> None:
        """Update competency records based on assessment results."""
        competency_id = assessment.get("competency", "general")
        score = assessment.get("score", 0)

        # Determine level based on score
        if score >= 0.9:
            level = CompetencyLevel.EXEMPLARY
        elif score >= 0.75:
            level = CompetencyLevel.PROFICIENT
        elif score >= 0.6:
            level = CompetencyLevel.DEVELOPING
        elif score >= 0.4:
            level = CompetencyLevel.EMERGING
        else:
            level = CompetencyLevel.NOT_STARTED

        # Update or create competency record
        if competency_id in progress.competencies:
            record = progress.competencies[competency_id]
            # Update only if the new level is equal or higher (assessments never
            # silently downgrade an achieved level); same-level updates refresh
            # evidence, last_assessed, and confidence.
            if _COMPETENCY_LEVEL_ORDER[level] >= _COMPETENCY_LEVEL_ORDER[record.level]:
                record.level = level
                record.last_assessed = datetime.now()
                record.confidence = min(1.0, record.confidence + 0.1)
                record.evidence.append(assessment.get("id", "assessment"))
        else:
            progress.competencies[competency_id] = CompetencyRecord(
                competency_id=competency_id,
                competency_name=self._competency_definitions.get(competency_id, {}).get(
                    "name", competency_id
                ),
                level=level,
                evidence=[assessment.get("id", "assessment")],
                last_assessed=datetime.now(),
                confidence=0.5,
            )

    def generate_competency_report(
        self,
        learner_id: str,
        competencies: list[str] | None = None,
        visualization: str = "radar_chart",
    ) -> dict[str, Any]:
        """
        Generate competency achievement report.

        Args:
            learner_id: Learner identifier
            competencies: Specific competencies to report (None for all)
            visualization: Visualization type ('radar_chart', 'bar_chart', 'table')

        Returns:
            Report dictionary with competency data and visualization
        """
        if learner_id not in self._learner_data:
            return {"error": "Learner not found"}

        progress = self._learner_data[learner_id]
        target_competencies = competencies or list(self._competency_definitions.keys())

        summary_out: dict[str, int] = {
            "total_competencies": len(target_competencies),
            "proficient_or_above": 0,
            "developing": 0,
            "not_started": 0,
        }
        competencies_out: list[dict[str, Any]] = []
        report: dict[str, Any] = {
            "learner_id": learner_id,
            "generated_at": datetime.now().isoformat(),
            "competencies": competencies_out,
            "summary": summary_out,
            "visualization_type": visualization,
            "visualization_data": {},
        }

        # Collect competency data
        level_values = {
            level: 25 * rank for level, rank in _COMPETENCY_LEVEL_ORDER.items()
        }
        for comp_id in target_competencies:
            if comp_id in progress.competencies:
                record = progress.competencies[comp_id]
                comp_data: dict[str, Any] = {
                    "id": comp_id,
                    "name": record.competency_name,
                    "level": record.level.value,
                    "level_value": level_values[record.level],
                    "confidence": record.confidence,
                    "last_assessed": record.last_assessed.isoformat()
                    if record.last_assessed
                    else None,
                    "evidence_count": len(record.evidence),
                }

                # Update summary
                if record.level in [
                    CompetencyLevel.PROFICIENT,
                    CompetencyLevel.EXEMPLARY,
                ]:
                    summary_out["proficient_or_above"] += 1
                elif record.level == CompetencyLevel.DEVELOPING:
                    summary_out["developing"] += 1
            else:
                comp_data = {
                    "id": comp_id,
                    "name": self._competency_definitions.get(comp_id, {}).get(
                        "name", comp_id
                    ),
                    "level": "not_started",
                    "level_value": 0,
                    "confidence": 0,
                    "last_assessed": None,
                    "evidence_count": 0,
                }
                summary_out["not_started"] += 1

            competencies_out.append(comp_data)

        # Generate visualization data
        if visualization == "radar_chart":
            report["visualization_data"] = {
                "labels": [c["name"] for c in competencies_out],
                "values": [c["level_value"] for c in competencies_out],
            }

        logger.info(f"Generated competency report for {learner_id}")
        return report

    def export_progress(self, learner_id: str, format: str = "json") -> str:
        """Export a learner's progress honouring the privacy policy.

        The exported payload contains the learner's activities summary,
        competency records, and totals. Privacy handling:

        * ``ferpa``: the student identifier is suppressed - the export
          carries a one-way pseudonym (SHA-256 prefix) instead of the
          learner id;
        * ``gdpr``: data-retention metadata is attached (retention
          period and erasure availability);
        * ``none``: the raw identifier is included.

        Args:
            learner_id: Learner identifier to export
            format: Export format ('json')

        Returns:
            Serialized progress export

        Raises:
            ValueError: If the format is unsupported or the learner is
                unknown.
        """
        if format != "json":
            raise ValueError(f"Unsupported export format: {format}")
        if learner_id not in self._learner_data:
            raise ValueError(f"Learner not found: {learner_id}")

        progress = self._learner_data[learner_id]

        if self.privacy_compliance == "ferpa":
            export_id = (
                "learner_" + hashlib.sha256(learner_id.encode("utf-8")).hexdigest()[:16]
            )
            identifier_note = "identifier_suppressed_ferpa"
        else:
            export_id = learner_id
            identifier_note = "identifier_included"

        payload: dict[str, Any] = {
            "learner_id": export_id,
            "identifier_handling": identifier_note,
            "privacy_compliance": self.privacy_compliance,
            "exported_at": datetime.now().isoformat(),
            "total_time_hours": progress.total_time_hours,
            "last_activity_date": (
                progress.last_activity_date.isoformat()
                if progress.last_activity_date
                else None
            ),
            "competencies": {
                name: {"level": record.level.value, "confidence": record.confidence}
                for name, record in progress.competencies.items()
            },
        }

        if self.privacy_compliance == "gdpr":
            payload["data_retention"] = {
                "regulation": "gdpr",
                "retention_period_days": 365,
                "erasure_available": True,
            }

        logger.info(
            f"Exported progress for {identifier_note} learner under {self.privacy_compliance}"
        )
        return json.dumps(payload, indent=2)

    def identify_gaps(
        self,
        learner_progress: LearnerProgress,
        required_competencies: list[str],
        recommendations: bool = True,
    ) -> dict[str, Any]:
        """
        Identify knowledge gaps between current skills and requirements.

        Args:
            learner_progress: Current learner progress
            required_competencies: Required competencies
            recommendations: Whether to include recommendations

        Returns:
            Gap analysis with optional recommendations
        """
        gaps_out: list[dict[str, Any]] = []
        gap_summary_out: dict[str, int] = {
            "total_required": len(required_competencies),
            "met": 0,
            "partially_met": 0,
            "not_met": 0,
        }
        gaps: dict[str, Any] = {
            "learner_id": learner_progress.learner_id,
            "gaps": gaps_out,
            "gap_summary": gap_summary_out,
            "recommendations": [] if recommendations else None,
        }

        for comp_id in required_competencies:
            if comp_id in learner_progress.competencies:
                record = learner_progress.competencies[comp_id]
                if record.level in [
                    CompetencyLevel.PROFICIENT,
                    CompetencyLevel.EXEMPLARY,
                ]:
                    gap_summary_out["met"] += 1
                elif record.level in [
                    CompetencyLevel.DEVELOPING,
                    CompetencyLevel.EMERGING,
                ]:
                    gap_summary_out["partially_met"] += 1
                    gaps_out.append(
                        {
                            "competency": comp_id,
                            "current_level": record.level.value,
                            "required_level": "proficient",
                            "gap_severity": "moderate",
                        }
                    )
                else:
                    gap_summary_out["not_met"] += 1
                    gaps_out.append(
                        {
                            "competency": comp_id,
                            "current_level": record.level.value,
                            "required_level": "proficient",
                            "gap_severity": "high",
                        }
                    )
            else:
                gap_summary_out["not_met"] += 1
                gaps_out.append(
                    {
                        "competency": comp_id,
                        "current_level": "not_started",
                        "required_level": "proficient",
                        "gap_severity": "critical",
                    }
                )

        # Generate recommendations
        if recommendations:
            recs_out: list[dict[str, Any]] = []
            gaps["recommendations"] = recs_out
            for gap in gaps_out:
                recs_out.append(
                    {
                        "competency": gap["competency"],
                        "action": f"Complete exercises and assessments for {gap['competency']}",
                        "priority": "high"
                        if gap["gap_severity"] == "critical"
                        else "medium",
                        "estimated_hours": 10
                        if gap["gap_severity"] == "critical"
                        else 5,
                    }
                )

        logger.info(f"Identified {len(gaps_out)} knowledge gaps")
        return gaps

    def generate_analytics(
        self,
        cohort: list[str],
        metrics: list[str],
        aggregation: str = "weekly",
        visualization: str = "dashboard",
    ) -> dict[str, Any]:
        """
        Generate learning analytics for a cohort.

        Args:
            cohort: List of learner IDs
            metrics: Metrics to calculate
            aggregation: Time aggregation ('daily', 'weekly', 'monthly')
            visualization: Visualization format

        Returns:
            Analytics dashboard data
        """
        metrics_out: dict[str, Any] = {}
        analytics: dict[str, Any] = {
            "cohort_size": len(cohort),
            "aggregation": aggregation,
            "metrics": metrics_out,
            "visualization_type": visualization,
        }

        # Collect data from cohort
        cohort_data = [
            self._learner_data.get(lid) for lid in cohort if lid in self._learner_data
        ]

        for metric in metrics:
            if metric == "completion_rate":
                rates = [p.completion_rate for p in cohort_data if p]
                metrics_out["completion_rate"] = {
                    "average": sum(rates) / len(rates) if rates else 0,
                    "min": min(rates) if rates else 0,
                    "max": max(rates) if rates else 0,
                }

            elif metric == "assessment_scores":
                all_scores = []
                for progress in cohort_data:
                    if progress:
                        scores = [
                            a.score for a in progress.activities if a.score is not None
                        ]
                        all_scores.extend(scores)
                metrics_out["assessment_scores"] = {
                    "average": sum(all_scores) / len(all_scores) if all_scores else 0,
                    "count": len(all_scores),
                }

            elif metric == "time_on_task":
                times = [p.total_time_hours for p in cohort_data if p]
                metrics_out["time_on_task"] = {
                    "average_hours": sum(times) / len(times) if times else 0,
                    "total_hours": sum(times),
                }

            elif metric == "engagement":
                activity_counts = [len(p.activities) for p in cohort_data if p]
                metrics_out["engagement"] = {
                    "average_activities": sum(activity_counts) / len(activity_counts)
                    if activity_counts
                    else 0,
                    "active_learners": len([c for c in activity_counts if c > 0]),
                }

        logger.info(f"Generated analytics for cohort of {len(cohort)} learners")
        return analytics

    def identify_at_risk(
        self,
        cohort: list[str],
        risk_indicators: list[str],
        intervention_recommendations: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Identify learners at risk of failure.

        Args:
            cohort: List of learner IDs
            risk_indicators: Indicators to check
            intervention_recommendations: Whether to include interventions

        Returns:
            List of at-risk learners with risk factors
        """
        at_risk: list[dict[str, Any]] = []

        for learner_id in cohort:
            if learner_id not in self._learner_data:
                continue

            progress = self._learner_data[learner_id]
            risk_factors: list[str] = []
            risk_score = 0

            for indicator in risk_indicators:
                if indicator == "low_engagement":
                    if len(progress.activities) < 3:
                        risk_factors.append("low_engagement")
                        risk_score += 30

                elif indicator == "declining_scores":
                    recent_scores = [
                        a.score for a in progress.activities[-5:] if a.score is not None
                    ]
                    if len(recent_scores) >= 2 and recent_scores[-1] < recent_scores[0]:
                        risk_factors.append("declining_scores")
                        risk_score += 25

                elif indicator == "missed_deadlines":
                    abandoned = [
                        a
                        for a in progress.activities
                        if a.completion_status == "abandoned"
                    ]
                    if len(abandoned) > 2:
                        risk_factors.append("missed_deadlines")
                        risk_score += 35

            if risk_factors:
                at_risk_entry: dict[str, Any] = {
                    "learner_id": learner_id,
                    "risk_score": min(100, risk_score),
                    "risk_factors": risk_factors,
                    "risk_level": "high" if risk_score >= 50 else "moderate",
                }

                if intervention_recommendations:
                    at_risk_entry["interventions"] = self._recommend_interventions(
                        risk_factors
                    )

                at_risk.append(at_risk_entry)

        # Sort by risk score
        at_risk.sort(key=lambda x: int(x["risk_score"] or 0), reverse=True)

        logger.info(f"Identified {len(at_risk)} at-risk learners")
        return at_risk

    def _recommend_interventions(self, risk_factors: list[str]) -> list[dict[str, Any]]:
        """Generate intervention recommendations for risk factors."""
        interventions: list[dict[str, Any]] = []

        intervention_map: dict[str, dict[str, str]] = {
            "low_engagement": {
                "type": "outreach",
                "action": "Schedule one-on-one meeting to discuss challenges",
                "urgency": "high",
            },
            "declining_scores": {
                "type": "academic_support",
                "action": "Provide additional tutoring or practice materials",
                "urgency": "medium",
            },
            "missed_deadlines": {
                "type": "time_management",
                "action": "Review workload and adjust timeline if needed",
                "urgency": "high",
            },
        }

        for factor in risk_factors:
            if factor in intervention_map:
                interventions.append(dict(intervention_map[factor]))

        return interventions
