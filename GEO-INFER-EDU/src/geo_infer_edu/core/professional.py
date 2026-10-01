"""
Professional development module.

Provides continuing education tracking, certification pathways,
and career skill analysis for GIS professionals.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ProfessionalProfile:
    """Professional's profile and credentials."""

    professional_id: str
    name: str
    current_role: str
    years_experience: int
    certifications: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    education: list[dict[str, str]] = field(default_factory=list)
    continuing_education_credits: float = 0


@dataclass
class ContinuingEducationActivity:
    """Continuing education activity record."""

    activity_id: str
    title: str
    activity_type: str  # course, conference, workshop, publication, teaching
    provider: str
    date_completed: datetime
    credits_earned: float
    category: str  # technical, management, ethics, general
    verification: str | None = None


@dataclass
class CertificationPathway:
    """Pathway to achieve a certification."""

    pathway_id: str
    target_certification: str
    requirements: dict[str, Any]
    current_progress: dict[str, float]
    estimated_completion: datetime | None = None
    next_steps: list[str] = field(default_factory=list)


class ProfessionalDevelopment:
    """
    Support continuing education and professional development
    for GIS professionals.
    """

    # GISP certification requirements
    CERTIFICATION_REQUIREMENTS = {
        "gisp": {
            "name": "GIS Professional",
            "education_points": 30,
            "experience_points": 60,
            "contributions_points": 60,
            "total_minimum": 150,
            "recertification_credits": 60,
            "recertification_period_years": 5,
        },
        "esri_technical": {
            "name": "Esri Technical Certification",
            "exam_required": True,
            "experience_years": 2,
            "recertification_period_years": 3,
        },
    }

    def __init__(
        self,
        certification_bodies: list[str] | None = None,
        credit_tracking: bool = True,
        competency_framework: str = "professional",
    ):
        """
        Initialize professional development module.

        Args:
            certification_bodies: Certification organizations to track
            credit_tracking: Enable CE credit tracking
            competency_framework: Professional competency framework
        """
        self.certification_bodies = certification_bodies or ["gisp", "esri", "osgeo"]
        self.credit_tracking = credit_tracking
        self.competency_framework = competency_framework
        self._professionals: dict[str, ProfessionalProfile] = {}
        self._ce_records: dict[str, list[ContinuingEducationActivity]] = {}
        logger.info(
            f"Initialized ProfessionalDevelopment with {self.certification_bodies}"
        )

    def register_professional(
        self, profile_data: dict[str, Any]
    ) -> ProfessionalProfile:
        """Register a professional in the system."""
        profile = ProfessionalProfile(
            professional_id=profile_data.get("id", f"pro_{len(self._professionals)}"),
            name=profile_data.get("name", ""),
            current_role=profile_data.get("role", ""),
            years_experience=profile_data.get("experience_years", 0),
            certifications=profile_data.get("certifications", []),
            skills=profile_data.get("skills", []),
            education=profile_data.get("education", []),
            continuing_education_credits=profile_data.get("ce_credits", 0),
        )
        self._professionals[profile.professional_id] = profile
        self._ce_records[profile.professional_id] = []
        return profile

    def track_continuing_education(
        self,
        professional_id: str,
        activities: list[dict[str, Any]],
        credits_earned: float | None = None,
    ) -> dict[str, Any]:
        """
        Track continuing education activities.

        Args:
            professional_id: Professional's identifier
            activities: List of CE activities
            credits_earned: Override total credits

        Returns:
            CE tracking summary
        """
        if professional_id not in self._professionals:
            return {"error": "Professional not found"}

        profile = self._professionals[professional_id]
        records = self._ce_records[professional_id]

        total_new_credits: float = 0.0
        for activity in activities:
            ce_activity = ContinuingEducationActivity(
                activity_id=activity.get("id", f"ce_{len(records)}"),
                title=activity.get("title", ""),
                activity_type=activity.get("type", "course"),
                provider=activity.get("provider", ""),
                date_completed=activity.get("date", datetime.now()),
                credits_earned=activity.get("credits", 0),
                category=activity.get("category", "technical"),
                verification=activity.get("verification"),
            )
            records.append(ce_activity)
            total_new_credits += float(ce_activity.credits_earned)

        profile.continuing_education_credits += credits_earned or total_new_credits

        summary: dict[str, Any] = {
            "professional_id": professional_id,
            "activities_tracked": len(activities),
            "credits_added": total_new_credits,
            "total_credits": profile.continuing_education_credits,
            "by_category": self._summarize_by_category(records),
        }

        logger.info(f"Tracked {len(activities)} CE activities for {professional_id}")
        return summary

    def _summarize_by_category(
        self, records: list[ContinuingEducationActivity]
    ) -> dict[str, float]:
        """Summarize credits by category."""
        summary: dict[str, float] = {}
        for record in records:
            category = record.category
            summary[category] = summary.get(category, 0) + record.credits_earned
        return summary

    def create_certification_pathway(
        self,
        target_certification: str,
        current_qualifications: dict[str, Any],
        timeline: str = "12_months",
    ) -> CertificationPathway:
        """
        Create a pathway to achieve certification.

        Args:
            target_certification: Target certification
            current_qualifications: Current experience and education
            timeline: Target timeline

        Returns:
            CertificationPathway with steps
        """
        cert_key = target_certification.lower().replace(" ", "_")
        requirements = self.CERTIFICATION_REQUIREMENTS.get(
            cert_key, {"name": target_certification, "total_minimum": 100}
        )

        # Calculate current progress
        progress = {
            "education": min(
                1.0,
                current_qualifications.get("education_points", 0)
                / requirements.get("education_points", 30),
            ),
            "experience": min(
                1.0,
                current_qualifications.get("experience_years", 0)
                / requirements.get("experience_points", 60)
                * 10,
            ),
            "contributions": min(
                1.0,
                current_qualifications.get("contributions_points", 0)
                / requirements.get("contributions_points", 60),
            ),
        }

        overall_progress = sum(progress.values()) / len(progress)

        # Calculate estimated completion
        months = int(timeline.split("_")[0])
        if overall_progress >= 1.0:
            completion_date = datetime.now()
        else:
            remaining_months = int(months * (1 - overall_progress))
            completion_date = datetime.now() + timedelta(days=remaining_months * 30)

        # Generate next steps
        next_steps = []
        if progress["education"] < 1.0:
            next_steps.append("Complete additional GIS coursework or training")
        if progress["experience"] < 1.0:
            next_steps.append("Gain more professional GIS experience")
        if progress["contributions"] < 1.0:
            next_steps.append(
                "Contribute to professional community (publications, presentations)"
            )

        pathway = CertificationPathway(
            pathway_id=f"pathway_{cert_key}_{datetime.now().strftime('%Y%m%d')}",
            target_certification=target_certification,
            requirements=requirements,
            current_progress=progress,
            estimated_completion=completion_date,
            next_steps=next_steps,
        )

        logger.info(f"Created certification pathway for {target_certification}")
        return pathway

    def analyze_career_skills(
        self,
        current_skills: list[str],
        target_role: str,
        job_market_data: dict[str, Any] | None = None,
        recommendations: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze skills for career advancement.

        Args:
            current_skills: Current skill set
            target_role: Target career role
            job_market_data: Optional job market data
            recommendations: Include recommendations

        Returns:
            Skill gap analysis with recommendations
        """
        # Define role skill requirements
        role_skills = {
            "gis_analyst": [
                "spatial_analysis",
                "data_management",
                "cartography",
                "python",
                "sql",
                "arcgis",
            ],
            "geospatial_data_scientist": [
                "spatial_analysis",
                "machine_learning",
                "python",
                "statistics",
                "big_data",
                "cloud_computing",
                "deep_learning",
            ],
            "remote_sensing_specialist": [
                "image_processing",
                "remote_sensing",
                "python",
                "classification",
                "time_series_analysis",
                "gee",
            ],
            "gis_developer": [
                "python",
                "javascript",
                "web_mapping",
                "api_development",
                "database_design",
                "cloud_computing",
                "devops",
            ],
            "gis_manager": [
                "project_management",
                "team_leadership",
                "strategic_planning",
                "budgeting",
                "stakeholder_management",
                "gis_architecture",
            ],
        }

        required_skills = set(
            role_skills.get(target_role.lower().replace(" ", "_"), [])
        )
        current_skill_set = set(s.lower().replace(" ", "_") for s in current_skills)

        # Identify gaps
        skill_gaps = list(required_skills - current_skill_set)
        matching_skills = list(required_skills.intersection(current_skill_set))

        analysis: dict[str, Any] = {
            "target_role": target_role,
            "current_skills": current_skills,
            "required_skills": list(required_skills),
            "matching_skills": matching_skills,
            "skill_gaps": skill_gaps,
            "match_percentage": len(matching_skills) / len(required_skills) * 100
            if required_skills
            else 100,
            "gap_count": len(skill_gaps),
        }

        if recommendations:
            recommendations_out: list[dict[str, Any]] = []
            analysis["recommendations"] = recommendations_out

            # Prioritize skill gaps
            priority_skills = skill_gaps[:3]  # Top 3 gaps

            for skill in priority_skills:
                recommendations_out.append(
                    {
                        "skill": skill,
                        "priority": "high",
                        "suggested_resources": [
                            f"Online course in {skill.replace('_', ' ')}",
                            f"GEO-INFER module on {skill.replace('_', ' ')}",
                            "Hands-on project experience",
                        ],
                        "estimated_learning_hours": 40,
                    }
                )

        logger.info(
            f"Analyzed career skills for {target_role}: {len(skill_gaps)} gaps identified"
        )
        return analysis

    def develop_portfolio(
        self,
        projects: list[dict[str, Any]],
        competencies_demonstrated: dict[str, list[str]],
        format: str = "professional_portfolio",
    ) -> dict[str, Any]:
        """
        Develop professional portfolio.

        Args:
            projects: Completed projects
            competencies_demonstrated: Competencies shown in projects
            format: Portfolio format

        Returns:
            Professional portfolio structure
        """
        portfolio_sections: list[dict[str, Any]] = []
        competency_summary_out: dict[str, Any] = {}
        portfolio: dict[str, Any] = {
            "format": format,
            "generated_at": datetime.now().isoformat(),
            "sections": portfolio_sections,
            "competency_summary": competency_summary_out,
            "project_count": len(projects),
        }

        # Professional summary section
        portfolio_sections.append(
            {
                "section": "professional_summary",
                "title": "Professional Summary",
                "content_type": "text",
                "guidance": "Brief overview of experience and expertise",
            }
        )

        # Skills section
        all_competencies = set()
        for comps in competencies_demonstrated.values():
            all_competencies.update(comps)

        portfolio_sections.append(
            {
                "section": "technical_skills",
                "title": "Technical Skills",
                "content_type": "skills_matrix",
                "skills": list(all_competencies),
            }
        )

        # Project showcase section
        project_entries: list[dict[str, Any]] = []
        for project in projects:
            entry: dict[str, Any] = {
                "title": project.get("title", "Project"),
                "description": project.get("description", ""),
                "technologies": project.get("technologies", []),
                "outcomes": project.get("outcomes", []),
                "competencies": competencies_demonstrated.get(
                    project.get("id", ""), []
                ),
            }
            project_entries.append(entry)

        portfolio_sections.append(
            {
                "section": "project_showcase",
                "title": "Project Showcase",
                "content_type": "projects",
                "projects": project_entries,
            }
        )

        # Competency summary
        for comp in all_competencies:
            evidence_count = sum(
                1 for comps in competencies_demonstrated.values() if comp in comps
            )
            competency_summary_out[comp] = {
                "demonstrated": True,
                "evidence_count": evidence_count,
            }

        logger.info(f"Developed portfolio with {len(projects)} projects")
        return portfolio

    def get_recertification_status(
        self, professional_id: str, certification: str
    ) -> dict[str, Any]:
        """
        Check recertification status.

        Args:
            professional_id: Professional identifier
            certification: Certification to check

        Returns:
            Recertification status and requirements
        """
        if professional_id not in self._professionals:
            return {"error": "Professional not found"}

        records = self._ce_records.get(professional_id, [])

        cert_key = certification.lower().replace(" ", "_")
        req_mapping: dict[str, Any] = self.CERTIFICATION_REQUIREMENTS.get(cert_key, {})
        requirements: Any = req_mapping

        required_credits = float(requirements.get("recertification_credits", 60))
        period_years = float(requirements.get("recertification_period_years", 5))

        # Calculate credits in current period
        period_start = datetime.now() - timedelta(days=period_years * 365)
        period_credits = float(
            sum(r.credits_earned for r in records if r.date_completed >= period_start)
        )

        status: dict[str, Any] = {
            "certification": certification,
            "required_credits": required_credits,
            "earned_credits": period_credits,
            "remaining_credits": max(0.0, required_credits - period_credits),
            "on_track": period_credits >= required_credits * 0.8,
            "period_end": (
                period_start + timedelta(days=period_years * 365)
            ).isoformat(),
            "status": "complete"
            if period_credits >= required_credits
            else "in_progress",
        }

        return status
