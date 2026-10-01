"""
Personalized learning module.

Provides adaptive learning paths, resource recommendations,
and personalized content delivery.
"""

import logging
from typing import Any
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class LearnerProfile:
    """Learner profile with preferences and history."""

    learner_id: str
    learning_style: str = "visual"  # visual, auditory, kinesthetic, reading
    prior_knowledge: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    preferred_pace: str = "moderate"  # slow, moderate, fast
    available_time_hours_week: float = 10
    strengths: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)


@dataclass
class LearningResource:
    """Educational resource with metadata."""

    resource_id: str
    title: str
    resource_type: str  # video, tutorial, exercise, reading, interactive
    topic: str
    difficulty: str
    duration_minutes: int
    format: str
    url: str | None = None
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class LearningPathway:
    """Personalized learning pathway."""

    pathway_id: str
    learner_id: str
    target_competencies: list[str]
    sequence: list[dict[str, Any]]
    estimated_duration_weeks: int
    optimization_strategy: str
    created_at: datetime = field(default_factory=datetime.now)


def compute_skill_gap_pathway(
    target_competencies: list[str],
    current_skills: list[str],
    hours_per_week: float,
    hours_per_competency: float,
    resources_for: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """
    Compute the skill-gap -> sequence -> duration core shared by all
    learning-pathway creators (PersonalizedLearning.create_pathway and
    CurriculumDesigner.create_learning_pathway).

    Args:
        target_competencies: Competencies the learner is aiming for
        current_skills: Competencies the learner already has
        hours_per_week: Weekly study time available to the learner
        hours_per_competency: Estimated study hours per skill gap
        resources_for: Optional callable mapping a competency to its resources

    Returns:
        Dict with ``skill_gaps``, ``sequence``, ``total_estimated_hours`` and
        ``estimated_duration_weeks`` keys.
    """
    current = set(current_skills)
    skill_gaps = [c for c in target_competencies if c not in current]
    total_hours = len(skill_gaps) * hours_per_competency
    sequence = [
        {
            "order": i + 1,
            "competency": skill,
            "estimated_hours": hours_per_competency,
            "resources": resources_for(skill) if resources_for else [],
        }
        for i, skill in enumerate(skill_gaps)
    ]
    return {
        "skill_gaps": skill_gaps,
        "sequence": sequence,
        "total_estimated_hours": total_hours,
        "estimated_duration_weeks": max(1, int(total_hours // hours_per_week)),
    }


class PersonalizedLearning:
    """
    Provide personalized learning experiences through adaptive pathways,
    intelligent recommendations, and spaced repetition.
    """

    def __init__(
        self,
        adaptation_method: str = "knowledge_tracing",
        recommendation_algorithm: str = "collaborative_filtering",
        learning_styles: list[str] | None = None,
    ):
        """
        Initialize personalized learning engine.

        Args:
            adaptation_method: Method for adapting content
            recommendation_algorithm: Algorithm for recommendations
            learning_styles: Supported learning styles
        """
        self.adaptation_method = adaptation_method
        self.recommendation_algorithm = recommendation_algorithm
        self.learning_styles = learning_styles or [
            "visual",
            "auditory",
            "kinesthetic",
            "reading",
        ]
        self._learner_profiles: dict[str, LearnerProfile] = {}
        self._resource_library: dict[str, LearningResource] = {}
        self._mastery_data: dict[str, dict[str, float]] = {}
        logger.info(
            f"Initialized PersonalizedLearning with {adaptation_method} adaptation"
        )

    def register_resource(
        self, resource: LearningResource | dict[str, Any]
    ) -> LearningResource:
        """Register a learning resource supplied by the content owner."""
        if isinstance(resource, dict):
            resource = LearningResource(**resource)
        self._resource_library[resource.resource_id] = resource
        return resource

    def register_learner(self, learner_profile: dict[str, Any]) -> LearnerProfile:
        """Register a new learner with their profile."""
        profile = LearnerProfile(
            learner_id=learner_profile.get(
                "id", f"learner_{len(self._learner_profiles)}"
            ),
            learning_style=learner_profile.get("learning_style", "visual"),
            prior_knowledge=learner_profile.get("prior_knowledge", []),
            interests=learner_profile.get("interests", []),
            preferred_pace=learner_profile.get("pace", "moderate"),
            available_time_hours_week=learner_profile.get("hours_per_week", 10),
            strengths=learner_profile.get("strengths", []),
            challenges=learner_profile.get("challenges", []),
        )
        self._learner_profiles[profile.learner_id] = profile
        # Initialize mastery only for a genuinely new learner: re-registration
        # (e.g. create_pathway for an existing learner) must not wipe the
        # accumulated mastery state used by recommendation/spaced-repetition.
        if profile.learner_id not in self._mastery_data:
            self._mastery_data[profile.learner_id] = {}
        return profile

    def create_pathway(
        self,
        learner_profile: dict[str, Any],
        learning_goals: list[str],
        constraints: dict[str, Any],
        optimization: str = "mastery",
    ) -> LearningPathway:
        """
        Create personalized learning pathway.

        Args:
            learner_profile: Learner background and assessment
            learning_goals: Target competencies
            constraints: Time and other constraints
            optimization: Optimization strategy ('mastery', 'speed', 'engagement')

        Returns:
            Personalized LearningPathway
        """
        # Register or update learner
        profile = self.register_learner(learner_profile)

        # Parse constraints
        time_constraint = constraints.get("time", "20_hours")
        total_hours_budget = (
            int(time_constraint.split("_")[0])
            if isinstance(time_constraint, str)
            else time_constraint
        )

        current_skills = list(profile.prior_knowledge)
        skill_gaps = [c for c in learning_goals if c not in set(current_skills)]
        hours_per_skill = (
            total_hours_budget / len(skill_gaps) if skill_gaps else total_hours_budget
        )

        core = compute_skill_gap_pathway(
            target_competencies=learning_goals,
            current_skills=current_skills,
            hours_per_week=profile.available_time_hours_week,
            hours_per_competency=hours_per_skill,
            resources_for=lambda skill: self._find_resources_for_skill(
                skill, profile.learning_style
            ),
        )
        shared_prerequisites = sorted(set(learning_goals).intersection(current_skills))
        sequence = [
            {
                "order": step["order"],
                "skill": step["competency"],
                "estimated_hours": step["estimated_hours"],
                "resources": step["resources"],
                "assessments": [f"assessment_{step['competency']}"],
                "prerequisites": shared_prerequisites,
            }
            for step in core["sequence"]
        ]

        pathway = LearningPathway(
            pathway_id=f"pathway_{profile.learner_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            learner_id=profile.learner_id,
            target_competencies=learning_goals,
            sequence=sequence,
            estimated_duration_weeks=core["estimated_duration_weeks"],
            optimization_strategy=optimization,
        )

        logger.info(
            f"Created pathway for {profile.learner_id} with {len(sequence)} learning units"
        )
        return pathway

    def _find_resources_for_skill(
        self, skill: str, learning_style: str
    ) -> list[dict[str, Any]]:
        """Find appropriate resources for a skill and learning style."""
        preferred_formats = set(self._get_preferred_formats(learning_style))
        return [
            {
                **asdict(resource),
                "type": resource.resource_type,
                "estimated_minutes": resource.duration_minutes,
            }
            for resource in self._resource_library.values()
            if resource.topic == skill and resource.resource_type in preferred_formats
        ]

    def recommend_resources(
        self,
        learner_id: str,
        current_topic: str,
        resource_types: list[str] | None = None,
        difficulty: str = "appropriate",
    ) -> list[dict[str, Any]]:
        """
        Recommend learning resources for a learner.

        Args:
            learner_id: Learner identifier
            current_topic: Current topic of study
            resource_types: Preferred resource types
            difficulty: Difficulty preference

        Returns:
            List of recommended resources
        """
        profile = self._learner_profiles.get(learner_id)
        if not profile:
            logger.warning(f"Learner {learner_id} not found")
            return []

        resource_type_set = set(resource_types or [])
        preferred = set(self._get_preferred_formats(profile.learning_style))
        recommendations: list[dict[str, Any]] = []
        for resource in self._resource_library.values():
            if resource.topic != current_topic:
                continue
            if resource_type_set and resource.resource_type not in resource_type_set:
                continue
            if difficulty != "appropriate" and resource.difficulty != difficulty:
                continue
            matches_style = resource.resource_type in preferred
            recommendations.append(
                {
                    **asdict(resource),
                    "type": resource.resource_type,
                    "relevance_score": 0.9 if matches_style else 0.7,
                    "estimated_time_minutes": resource.duration_minutes,
                    "matches_style": matches_style,
                }
            )

        # Sort by relevance
        recommendations.sort(
            key=lambda x: float(x["relevance_score"] or 0), reverse=True
        )

        logger.info(
            f"Generated {len(recommendations)} recommendations for {learner_id}"
        )
        return recommendations

    def _get_preferred_formats(self, learning_style: str) -> list[str]:
        """Get preferred resource formats for learning style."""
        style_map = {
            "visual": ["video", "infographic", "diagram", "animation"],
            "auditory": ["podcast", "lecture", "audio", "discussion"],
            "kinesthetic": ["interactive", "simulation", "exercise", "lab"],
            "reading": ["article", "textbook", "documentation", "tutorial"],
        }
        return style_map.get(learning_style, ["video", "tutorial"])

    def deliver_adaptive_content(
        self,
        learner_id: str,
        topic: str,
        format_preference: str | None = None,
        mastery_level: float | None = None,
    ) -> dict[str, Any]:
        """
        Deliver content adapted to learner's current state.

        Args:
            learner_id: Learner identifier
            topic: Topic to deliver
            format_preference: Preferred format
            mastery_level: Current mastery level (0-1)

        Returns:
            Adaptive content package
        """
        profile = self._learner_profiles.get(learner_id)
        if not profile:
            return {"error": "Learner not found"}

        # Determine appropriate difficulty
        mastery = mastery_level or self._mastery_data.get(learner_id, {}).get(topic, 0)

        if mastery < 0.3:
            difficulty = "introductory"
            content_depth = "foundational"
        elif mastery < 0.6:
            difficulty = "intermediate"
            content_depth = "conceptual"
        elif mastery < 0.85:
            difficulty = "advanced"
            content_depth = "applied"
        else:
            difficulty = "expert"
            content_depth = "specialized"

        # Select format
        format_to_use = (
            format_preference or self._get_preferred_formats(profile.learning_style)[0]
        )

        content = {
            "learner_id": learner_id,
            "topic": topic,
            "content_id": f"content_{topic}_{difficulty}",
            "format": format_to_use,
            "difficulty": difficulty,
            "content_depth": content_depth,
            "estimated_time_minutes": self._estimate_time(difficulty),
            "sections": self._generate_content_sections(
                topic, difficulty, content_depth
            ),
            "practice_exercises": self._generate_practice(topic, difficulty),
            "next_steps": self._suggest_next_steps(topic, mastery),
        }

        logger.info(
            f"Delivered adaptive content on {topic} ({difficulty}) to {learner_id}"
        )
        return content

    def _estimate_time(self, difficulty: str) -> int:
        """Estimate time needed based on difficulty."""
        time_map = {
            "introductory": 15,
            "intermediate": 25,
            "advanced": 40,
            "expert": 60,
        }
        return time_map.get(difficulty, 30)

    def _generate_content_sections(
        self, topic: str, difficulty: str, depth: str
    ) -> list[dict[str, Any]]:
        """Generate content sections for adaptive delivery."""
        sections = [
            {
                "section_id": "intro",
                "title": "Introduction",
                "type": "overview",
                "content_summary": f"Introduction to {topic.replace('_', ' ')}",
            },
            {
                "section_id": "concepts",
                "title": "Key Concepts",
                "type": "conceptual",
                "content_summary": f"Core concepts of {topic.replace('_', ' ')}",
            },
            {
                "section_id": "examples",
                "title": "Examples",
                "type": "demonstration",
                "content_summary": "Worked examples and demonstrations",
            },
        ]

        if difficulty in ["advanced", "expert"]:
            sections.append(
                {
                    "section_id": "advanced_topics",
                    "title": "Advanced Topics",
                    "type": "specialized",
                    "content_summary": "Advanced applications and edge cases",
                }
            )

        return sections

    def _generate_practice(self, topic: str, difficulty: str) -> list[dict[str, Any]]:
        """Generate practice exercises."""
        num_exercises = 3 if difficulty in ["introductory", "intermediate"] else 5

        return [
            {
                "exercise_id": f"practice_{topic}_{i}",
                "type": "practice",
                "difficulty": difficulty,
                "estimated_minutes": 5 + (i * 2),
            }
            for i in range(num_exercises)
        ]

    def _suggest_next_steps(self, topic: str, mastery: float) -> list[str]:
        """Suggest next learning steps."""
        if mastery < 0.5:
            return [
                "Review foundational concepts",
                "Complete more practice exercises",
                "Watch explanatory videos",
            ]
        elif mastery < 0.8:
            return [
                "Attempt more challenging exercises",
                "Explore related topics",
                "Apply concepts to a project",
            ]
        else:
            return [
                "Move to advanced topics",
                "Help peers with this topic",
                "Create your own examples",
            ]

    def schedule_review(
        self,
        learner_id: str,
        mastered_topics: list[str],
        retention_model: str = "forgetting_curve",
        review_frequency: str = "optimal",
    ) -> list[dict[str, Any]]:
        """
        Schedule spaced repetition reviews.

        Args:
            learner_id: Learner identifier
            mastered_topics: Topics to schedule for review
            retention_model: Retention model to use
            review_frequency: Frequency strategy

        Returns:
            Review schedule
        """
        schedule: list[dict[str, Any]] = []
        now = datetime.now()

        # Spaced repetition intervals (days)
        intervals = [1, 3, 7, 14, 30, 60, 120]

        for topic in mastered_topics:
            mastery = self._mastery_data.get(learner_id, {}).get(topic, 0.5)

            # Adjust intervals based on mastery
            if mastery >= 0.9:
                topic_intervals = intervals[2:]  # Skip early reviews
            elif mastery >= 0.7:
                topic_intervals = intervals[1:]
            else:
                topic_intervals = intervals

            for i, interval in enumerate(
                topic_intervals[:5]
            ):  # Max 5 reviews scheduled
                schedule.append(
                    {
                        "topic": topic,
                        "review_number": i + 1,
                        "scheduled_date": (now + timedelta(days=interval)).isoformat(),
                        "interval_days": interval,
                        "estimated_minutes": 10,
                        "review_type": "recall" if i < 2 else "application",
                    }
                )

        # Sort by date
        schedule.sort(key=lambda x: x["scheduled_date"])

        logger.info(f"Scheduled {len(schedule)} reviews for {learner_id}")
        return schedule

    def update_mastery(
        self, learner_id: str, topic: str, performance_score: float
    ) -> float:
        """
        Update mastery level based on performance.

        Args:
            learner_id: Learner identifier
            topic: Topic assessed
            performance_score: Score from 0 to 1

        Returns:
            Updated mastery level
        """
        if learner_id not in self._mastery_data:
            self._mastery_data[learner_id] = {}

        current_mastery = self._mastery_data[learner_id].get(topic, 0)

        # Bayesian update with performance
        learning_rate = 0.3
        new_mastery = current_mastery + learning_rate * (
            performance_score - current_mastery
        )
        new_mastery = max(0, min(1, new_mastery))  # Clamp to [0, 1]

        self._mastery_data[learner_id][topic] = new_mastery

        logger.info(
            f"Updated mastery for {learner_id} on {topic}: {current_mastery:.2f} -> {new_mastery:.2f}"
        )
        return new_mastery
