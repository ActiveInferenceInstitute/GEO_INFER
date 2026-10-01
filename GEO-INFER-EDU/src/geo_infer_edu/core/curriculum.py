"""
Curriculum design and generation module.

Provides standards-aligned curriculum generation for geospatial education,
including learning objectives, module sequencing, and assessment design.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from enum import Enum

import yaml

from .personalization import compute_skill_gap_pathway

logger = logging.getLogger(__name__)


class EducationLevel(Enum):
    """Educational level classifications."""

    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    INTERMEDIATE = "intermediate"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    PROFESSIONAL = "professional"


class PedagogicalApproach(Enum):
    """Pedagogical methodology approaches."""

    CONSTRUCTIVIST = "constructivist"
    INQUIRY_BASED = "inquiry_based"
    PROJECT_BASED = "project_based"
    COMPETENCY_BASED = "competency_based"
    EXPERIENTIAL = "experiential"


@dataclass
class LearningObjective:
    """Represents a learning objective with Bloom's taxonomy alignment."""

    id: str
    description: str
    bloom_level: str  # remember, understand, apply, analyze, evaluate, create
    competency_area: str
    assessment_criteria: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class CurriculumModule:
    """Represents a curriculum module with content and activities."""

    id: str
    title: str
    description: str
    learning_objectives: list[LearningObjective]
    duration_hours: float
    content_sections: list[dict[str, Any]] = field(default_factory=list)
    activities: list[dict[str, Any]] = field(default_factory=list)
    assessments: list[dict[str, Any]] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)


@dataclass
class Curriculum:
    """Complete curriculum with modules and metadata."""

    id: str
    title: str
    description: str
    level: EducationLevel
    duration_weeks: int
    modules: list[CurriculumModule] = field(default_factory=list)
    standards_alignment: dict[str, list[str]] = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)
    target_competencies: list[str] = field(default_factory=list)


class CurriculumDesigner:
    """
    Design and generate standards-aligned geospatial curricula.

    Supports multiple educational standards (BOK, GISTBOK, NGSS) and
    pedagogical approaches for comprehensive curriculum development.
    """

    # Supported educational standards keys
    _SUPPORTED_STANDARDS = ("bok", "gistbok", "ngss")

    # Standard competency areas from GIS Body of Knowledge
    COMPETENCY_AREAS = [
        "spatial_thinking",
        "data_acquisition",
        "data_management",
        "spatial_analysis",
        "geovisualization",
        "geospatial_programming",
        "project_management",
        "ethics_and_policy",
    ]

    def __init__(
        self,
        standards: list[str] | None = None,
        pedagogical_approach: str = "constructivist",
        assessment_framework: str = "competency_based",
    ):
        """
        Initialize curriculum designer.

        Args:
            standards: Educational standards to align with (e.g., ['bok', 'gistbok', 'ngss'])
            pedagogical_approach: Teaching methodology approach
            assessment_framework: Assessment approach to use
        """
        self.standards = standards or ["bok"]
        unknown = [s for s in self.standards if s not in self._SUPPORTED_STANDARDS]
        if unknown:
            raise ValueError(
                f"Unsupported standard(s): {unknown}. "
                f"Supported standards: {sorted(self._SUPPORTED_STANDARDS)}"
            )
        self.pedagogical_approach = PedagogicalApproach(pedagogical_approach)
        self.assessment_framework = assessment_framework
        self._standards_data = self._load_standards_data()
        logger.info(f"Initialized CurriculumDesigner with standards: {self.standards}")

    def _load_standards_data(self) -> dict[str, dict]:
        """Load educational standards data."""
        # Embedded standards data for GIS BOK
        standards_data = {
            "bok": {
                "name": "Geographic Information Science and Technology Body of Knowledge",
                "competencies": {
                    "AM": "Analytical Methods",
                    "CF": "Conceptual Foundations",
                    "CV": "Cartography and Visualization",
                    "DA": "Design Aspects",
                    "DM": "Data Modeling",
                    "DN": "Data Manipulation",
                    "GC": "Geocomputation",
                    "GD": "Geospatial Data",
                    "OI": "Organizational and Institutional Aspects",
                },
            },
            "gistbok": {
                "name": "UCGIS GIS&T Body of Knowledge (GISTBOK)",
                "competencies": {
                    "AM": "Analytical Methods",
                    "CF": "Conceptual Foundations",
                    "CV": "Cartography and Visualization",
                    "DA": "Design Aspects",
                    "DM": "Data Modeling",
                    "DN": "Data Manipulation",
                    "GC": "Geocomputation",
                    "GD": "Geospatial Data",
                    "GS": "GIS&T and Society",
                    "OI": "Organizational and Institutional Aspects",
                },
            },
            "ngss": {
                "name": "Next Generation Science Standards",
                "competencies": {
                    "SEP": "Science and Engineering Practices",
                    "DCI": "Disciplinary Core Ideas",
                    "CCC": "Crosscutting Concepts",
                },
            },
        }
        return {s: standards_data[s] for s in self.standards}

    def design(
        self,
        topic: str,
        level: str,
        duration: str,
        learning_objectives: list[str] | None = None,
    ) -> Curriculum:
        """
        Design a complete curriculum for the specified topic.

        Args:
            topic: Main curriculum topic (e.g., 'geospatial_analysis')
            level: Education level (e.g., 'undergraduate', 'professional')
            duration: Duration string (e.g., '8_weeks', '16_weeks')
            learning_objectives: Optional list of specific learning objectives

        Returns:
            Curriculum object with complete structure
        """
        # Parse duration
        duration_weeks = int(duration.split("_")[0])
        education_level = EducationLevel(level)

        # Generate curriculum ID
        curriculum_id = f"curriculum_{topic}_{level}"

        # Create base curriculum
        curriculum = Curriculum(
            id=curriculum_id,
            title=f"Introduction to {topic.replace('_', ' ').title()}",
            description=self._generate_description(topic, level),
            level=education_level,
            duration_weeks=duration_weeks,
            target_competencies=self._identify_competencies(topic),
        )

        # Generate learning objectives
        objectives = self._generate_learning_objectives(
            topic, education_level, learning_objectives
        )

        # Generate modules
        curriculum.modules = self.generate_modules(
            topic=topic,
            level=education_level,
            duration_weeks=duration_weeks,
            objectives=objectives,
        )

        # Align with standards
        curriculum.standards_alignment = self._align_with_standards(objectives)

        logger.info(
            f"Designed curriculum '{curriculum.title}' with {len(curriculum.modules)} modules"
        )
        return curriculum

    def _generate_description(self, topic: str, level: str) -> str:
        """Generate curriculum description."""
        topic_title = topic.replace("_", " ").title()
        return (
            f"This curriculum provides a comprehensive introduction to {topic_title} "
            f"designed for {level} learners. Students will develop practical skills "
            f"through hands-on exercises and real-world applications."
        )

    def _identify_competencies(self, topic: str) -> list[str]:
        """Identify relevant competencies for topic."""
        topic_competencies = {
            "geospatial_analysis": [
                "spatial_analysis",
                "data_management",
                "geovisualization",
            ],
            "remote_sensing": [
                "data_acquisition",
                "spatial_analysis",
                "geovisualization",
            ],
            "gis_programming": [
                "geospatial_programming",
                "data_management",
                "spatial_analysis",
            ],
            "cartography": ["geovisualization", "spatial_thinking", "data_management"],
        }
        return topic_competencies.get(topic, ["spatial_thinking", "spatial_analysis"])

    def _generate_learning_objectives(
        self,
        topic: str,
        level: EducationLevel,
        custom_objectives: list[str] | None = None,
    ) -> list[LearningObjective]:
        """Generate learning objectives for curriculum."""
        objectives = []

        # Use custom objectives if provided
        if custom_objectives:
            for i, obj_text in enumerate(custom_objectives):
                objectives.append(
                    LearningObjective(
                        id=f"lo_{i + 1}",
                        description=obj_text,
                        bloom_level="apply",
                        competency_area="spatial_analysis",
                    )
                )
        else:
            # Generate default objectives based on topic
            default_objectives = [
                ("Understand fundamental concepts", "understand", "spatial_thinking"),
                ("Apply analytical techniques", "apply", "spatial_analysis"),
                (
                    "Evaluate results and interpret findings",
                    "evaluate",
                    "spatial_analysis",
                ),
                (
                    "Create solutions for real-world problems",
                    "create",
                    "geospatial_programming",
                ),
            ]

            for i, (desc, bloom, comp) in enumerate(default_objectives):
                objectives.append(
                    LearningObjective(
                        id=f"lo_{i + 1}",
                        description=f"{desc} of {topic.replace('_', ' ')}",
                        bloom_level=bloom,
                        competency_area=comp,
                    )
                )

        return objectives

    def generate_modules(
        self,
        topic: str,
        level: EducationLevel,
        duration_weeks: int,
        objectives: list[LearningObjective],
        hours_per_module: float = 4.0,
    ) -> list[CurriculumModule]:
        """
        Generate curriculum modules from topic and objectives.

        Args:
            topic: Main curriculum topic
            level: Education level
            duration_weeks: Total duration in weeks
            objectives: Learning objectives
            hours_per_module: Hours allocated per module

        Returns:
            List of CurriculumModule objects
        """
        modules = []

        # Standard module structure
        module_templates = [
            ("Introduction and Foundations", "introduce core concepts and terminology"),
            ("Data Fundamentals", "work with geospatial data formats and sources"),
            ("Basic Analysis", "perform fundamental analytical operations"),
            ("Intermediate Techniques", "apply more advanced analytical methods"),
            ("Visualization and Communication", "create effective visualizations"),
            (
                "Integration and Applications",
                "combine techniques for real-world applications",
            ),
            ("Project Development", "develop comprehensive project solutions"),
            ("Review and Assessment", "review concepts and complete final assessment"),
        ]

        # Adjust number of modules to duration
        num_modules = min(duration_weeks, len(module_templates))

        for i in range(num_modules):
            title, description = module_templates[i]

            # Assign objectives to modules
            module_objectives = [
                obj for j, obj in enumerate(objectives) if j % num_modules == i
            ]

            module = CurriculumModule(
                id=f"module_{i + 1}",
                title=f"Module {i + 1}: {title}",
                description=f"In this module, students will {description} for {topic.replace('_', ' ')}.",
                learning_objectives=module_objectives,
                duration_hours=hours_per_module,
                content_sections=self._generate_content_sections(title, topic),
                activities=self._generate_activities(title, level),
                assessments=self._generate_assessments(module_objectives),
                resources=self._generate_resources(topic),
            )
            modules.append(module)

        logger.info(f"Generated {len(modules)} modules for {topic}")
        return modules

    def _generate_content_sections(
        self, title: str, topic: str
    ) -> list[dict[str, Any]]:
        """Generate content sections for a module."""
        return [
            {
                "type": "lecture",
                "title": f"Introduction to {title}",
                "duration_minutes": 30,
                "format": "slides_with_notes",
            },
            {
                "type": "reading",
                "title": f"Background on {topic.replace('_', ' ')}",
                "duration_minutes": 20,
                "format": "text",
            },
            {
                "type": "demonstration",
                "title": "Practical Demonstration",
                "duration_minutes": 30,
                "format": "video_with_code",
            },
        ]

    def _generate_activities(
        self, title: str, level: EducationLevel
    ) -> list[dict[str, Any]]:
        """Generate learning activities for a module."""
        return [
            {
                "type": "hands_on_lab",
                "title": f"Lab: {title} Practice",
                "duration_minutes": 60,
                "individual": True,
                "data_provided": True,
            },
            {
                "type": "group_exercise",
                "title": f"Team Exercise: {title} Application",
                "duration_minutes": 45,
                "group_size": 3,
                "collaborative": True,
            },
        ]

    def _generate_assessments(
        self, objectives: list[LearningObjective]
    ) -> list[dict[str, Any]]:
        """Generate assessments for module objectives."""
        assessments = []
        for obj in objectives:
            assessments.append(
                {
                    "type": "practical",
                    "objective_id": obj.id,
                    "description": f"Demonstrate: {obj.description}",
                    "rubric": True,
                    "weight": 1.0 / len(objectives) if objectives else 1.0,
                }
            )
        return assessments

    def _generate_resources(self, topic: str) -> list[str]:
        """Generate resource list for topic."""
        return [
            f"GEO-INFER-{topic.upper()} Module Documentation",
            "GIS Body of Knowledge Reference",
            "Sample Datasets and Examples",
        ]

    def _align_with_standards(
        self, objectives: list[LearningObjective]
    ) -> dict[str, list[str]]:
        """Align learning objectives with educational standards."""
        alignment = {}
        for standard in self.standards:
            if standard in self._standards_data:
                standard_name = self._standards_data[standard].get("name", standard)
                # Map objectives to standard competencies
                alignment[standard_name] = [
                    f"{obj.competency_area}: {obj.description}" for obj in objectives
                ]
        return alignment

    def align_with_standards(
        self,
        curriculum: Curriculum,
        target_standards: list[str],
        coverage_report: bool = False,
    ) -> dict[str, Any]:
        """
        Align curriculum with specific educational standards.

        Args:
            curriculum: Curriculum to align
            target_standards: Standards to align with
            coverage_report: Whether to generate coverage report

        Returns:
            Alignment mapping and optional coverage report
        """
        mappings_out: dict[str, list[dict[str, Any]]] = {}
        coverage_out: dict[str, dict[str, Any]] = {}
        alignment: dict[str, Any] = {"mappings": mappings_out, "coverage": coverage_out}

        for standard in target_standards:
            standard_objectives: list[dict[str, Any]] = []
            for module in curriculum.modules:
                for obj in module.learning_objectives:
                    standard_objectives.append(
                        {
                            "objective": obj.description,
                            "module": module.title,
                            "competency": obj.competency_area,
                        }
                    )
            mappings_out[standard] = standard_objectives

            if coverage_report:
                coverage_out[standard] = {
                    "objectives_mapped": len(standard_objectives),
                    "modules_covered": len(curriculum.modules),
                    "competencies": list(
                        set(o["competency"] for o in standard_objectives)
                    ),
                }

        return alignment

    def create_learning_pathway(
        self,
        learner_profile: dict[str, Any],
        target_competencies: list[str],
        available_time: str,
        optimization: str = "efficiency",
    ) -> dict[str, Any]:
        """
        Create personalized learning pathway for a learner.

        Args:
            learner_profile: Learner background and current skills
            target_competencies: Desired competencies to achieve
            available_time: Available time (e.g., '10_hours_week')
            optimization: Optimization goal ('efficiency', 'depth', 'breadth')

        Returns:
            Personalized learning pathway
        """
        hours_per_week = int(available_time.split("_")[0])

        core = compute_skill_gap_pathway(
            target_competencies=target_competencies,
            current_skills=learner_profile.get("current_skills", []),
            hours_per_week=hours_per_week,
            hours_per_competency=10 if optimization == "efficiency" else 20,
            resources_for=self._generate_resources,
        )

        pathway = {
            "learner_id": learner_profile.get("id", "anonymous"),
            "target_competencies": target_competencies,
            "skill_gaps": core["skill_gaps"],
            "hours_per_week": hours_per_week,
            "optimization": optimization,
            "recommended_sequence": core["sequence"],
            "estimated_duration_weeks": core["estimated_duration_weeks"],
        }

        logger.info(
            f"Created learning pathway with {len(core['skill_gaps'])} competencies to develop"
        )
        return pathway

    def export_curriculum(self, curriculum: Curriculum, format: str = "yaml") -> str:
        """
        Export curriculum to specified format.

        Args:
            curriculum: Curriculum to export
            format: Export format ('yaml', 'json', 'markdown')

        Returns:
            Exported curriculum string
        """
        curriculum_dict = {
            "id": curriculum.id,
            "title": curriculum.title,
            "description": curriculum.description,
            "level": curriculum.level.value,
            "duration_weeks": curriculum.duration_weeks,
            "modules": [
                {
                    "id": m.id,
                    "title": m.title,
                    "description": m.description,
                    "duration_hours": m.duration_hours,
                    "objectives": [
                        {
                            "id": o.id,
                            "description": o.description,
                            "bloom_level": o.bloom_level,
                        }
                        for o in m.learning_objectives
                    ],
                }
                for m in curriculum.modules
            ],
            "standards_alignment": curriculum.standards_alignment,
        }

        if format == "yaml":
            return yaml.dump(curriculum_dict, default_flow_style=False)
        elif format == "json":
            import json

            return json.dumps(curriculum_dict, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
