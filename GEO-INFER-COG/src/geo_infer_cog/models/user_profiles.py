"""
User Cognitive Profiles for GEO-INFER-COG

This module defines user cognitive profile models that capture individual
differences in spatial cognition, reasoning styles, and cognitive load
preferences. These profiles enable personalized geospatial interfaces
and decision support systems.

Key Components:
- User cognitive profile schemas and validation
- Spatial expertise and reasoning style classification
- Cognitive load preference modeling
- Adaptive profile learning from user behavior
- Profile-based personalization strategies

Mathematical Foundations:
- Individual differences in spatial cognition (Hegarty et al., 2002)
- Cognitive style theories (Riding & Rayner, 1998)
- Cognitive load theory (Sweller, 1988)
- Adaptive user modeling frameworks
"""

import numpy as np
import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class UserCognitiveProfile:
    """
    Comprehensive user cognitive profile for spatial cognition.

    This class models individual differences in how users perceive, reason about,
    and interact with spatial information, enabling personalized geospatial
    interfaces and decision support systems.

    The profile includes:
    - Spatial expertise and experience levels
    - Cognitive style preferences (visualizer vs. verbalizer)
    - Spatial reasoning approach (qualitative vs. quantitative)
    - Cognitive load tolerance and preferences
    - Learning style and adaptation patterns
    - Demographic and contextual factors
    """

    user_id: str
    profile_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Spatial cognition capabilities
    spatial_expertise: float = 0.5  # 0.0 (novice) to 1.0 (expert)
    spatial_memory_capacity: float = 0.5  # Working memory for spatial information
    spatial_attention_span: float = 0.5  # Attention capacity for spatial tasks
    spatial_reasoning_style: str = (
        "balanced"  # 'qualitative', 'quantitative', 'balanced'
    )

    # Cognitive style preferences
    cognitive_style: str = "balanced"  # 'visualizer', 'verbalizer', 'balanced'
    learning_preference: str = "moderate"  # 'slow', 'moderate', 'fast'
    cognitive_load_preference: str = "moderate"  # 'low', 'moderate', 'high'

    # Behavioral patterns
    navigation_preferences: dict[str, Any] = field(default_factory=dict)
    visualization_preferences: dict[str, Any] = field(default_factory=dict)
    interaction_patterns: dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    task_performance_history: list[dict[str, Any]] = field(default_factory=list)
    adaptation_metrics: dict[str, float] = field(default_factory=dict)

    # Demographic and contextual information
    age_group: str = "adult"  # 'child', 'teen', 'adult', 'senior'
    experience_level: str = (
        "intermediate"  # 'beginner', 'intermediate', 'advanced', 'expert'
    )
    domain_experience: dict[str, float] = field(
        default_factory=dict
    )  # Domain-specific expertise

    def update_from_interaction(
        self, interaction_data: dict[str, Any], outcome: dict[str, Any]
    ) -> None:
        """
        Update profile based on user interaction and outcome.

        Args:
            interaction_data: Details of user interaction with spatial interface
            outcome: Results and performance metrics from the interaction
        """
        # Update spatial expertise based on performance
        performance_score = outcome.get("performance_score", 0.5)
        task_complexity = interaction_data.get("task_complexity", 0.5)

        # Adaptive expertise update
        expertise_change = (performance_score - 0.5) * task_complexity * 0.1
        self.spatial_expertise = max(
            0.0, min(1.0, self.spatial_expertise + expertise_change)
        )

        # Update interaction patterns
        interaction_type = interaction_data.get("interaction_type", "unknown")
        if interaction_type not in self.interaction_patterns:
            self.interaction_patterns[interaction_type] = []

        self.interaction_patterns[interaction_type].append(
            {
                "timestamp": datetime.now().isoformat(),
                "task_complexity": task_complexity,
                "performance_score": performance_score,
                "duration": interaction_data.get("duration", 0),
            }
        )

        # Update task performance history
        self.task_performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "interaction_type": interaction_type,
                "performance_score": performance_score,
                "task_complexity": task_complexity,
                "cognitive_load": outcome.get("cognitive_load", 0.5),
            }
        )

        # Keep history manageable (last 100 interactions)
        if len(self.task_performance_history) > 100:
            self.task_performance_history = self.task_performance_history[-100:]

        self.last_updated = datetime.now()

        logger.info(f"User profile updated for {self.user_id}")

    def calculate_task_suitability(self, task_requirements: dict[str, Any]) -> float:
        """
        Calculate how suitable this user is for a given spatial task.

        Args:
            task_requirements: Requirements and characteristics of the task

        Returns:
            Suitability score (0-1)
        """
        suitability_factors: list[float] = []

        # Spatial expertise match
        required_expertise = task_requirements.get("required_expertise", 0.5)
        expertise_match = 1.0 - abs(self.spatial_expertise - required_expertise)
        suitability_factors.append(expertise_match * 0.4)

        # Cognitive style compatibility
        task_style = task_requirements.get("cognitive_style", "balanced")
        style_compatibility = 1.0 if self.cognitive_style == task_style else 0.7
        suitability_factors.append(style_compatibility * 0.3)

        # Cognitive load compatibility
        task_load = task_requirements.get("cognitive_load", 0.5)
        load_compatibility = 1.0 - abs(self._get_load_preference_score() - task_load)
        suitability_factors.append(load_compatibility * 0.3)

        return sum(suitability_factors)

    def _get_load_preference_score(self) -> float:
        """Convert cognitive load preference to numeric score."""
        preference_scores = {"low": 0.3, "moderate": 0.5, "high": 0.7}
        return preference_scores.get(self.cognitive_load_preference, 0.5)

    def get_personalized_recommendations(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate personalized recommendations for spatial interface use.

        Args:
            context: Current usage context and task information

        Returns:
            Dictionary of personalized recommendations
        """
        recommendations: dict[str, Any] = {
            "visualization_suggestions": [],
            "navigation_aids": [],
            "complexity_adjustments": [],
            "learning_support": [],
        }

        # Visualization recommendations based on cognitive style
        if self.cognitive_style == "visualizer":
            recommendations["visualization_suggestions"].extend(
                [
                    "Use map-based representations",
                    "Highlight spatial relationships visually",
                    "Minimize text-based instructions",
                ]
            )
        elif self.cognitive_style == "verbalizer":
            recommendations["visualization_suggestions"].extend(
                [
                    "Provide detailed textual descriptions",
                    "Use landmark names and directions",
                    "Include explanatory annotations",
                ]
            )

        # Navigation recommendations based on expertise
        if self.spatial_expertise < 0.4:
            recommendations["navigation_aids"].extend(
                [
                    "Provide step-by-step directions",
                    "Use prominent landmarks as reference points",
                    "Include distance estimates and turn indicators",
                ]
            )
        elif self.spatial_expertise > 0.7:
            recommendations["navigation_aids"].extend(
                [
                    "Allow flexible route exploration",
                    "Provide overview maps for context",
                    "Enable advanced spatial analysis tools",
                ]
            )

        # Complexity adjustments based on load preference
        if self.cognitive_load_preference == "low":
            recommendations["complexity_adjustments"].extend(
                [
                    "Simplify spatial representations",
                    "Reduce information density",
                    "Provide progressive disclosure",
                ]
            )
        elif self.cognitive_load_preference == "high":
            recommendations["complexity_adjustments"].extend(
                [
                    "Include detailed spatial information",
                    "Show multiple representation layers",
                    "Enable advanced filtering and analysis",
                ]
            )

        # Learning support based on performance history
        recent_performance = self._get_recent_performance()
        if recent_performance < 0.6:
            recommendations["learning_support"].extend(
                [
                    "Provide additional context and explanations",
                    "Suggest practice exercises",
                    "Include confidence indicators",
                ]
            )

        return recommendations

    def _get_recent_performance(self) -> float:
        """Calculate recent performance average."""
        if not self.task_performance_history:
            return 0.5

        # Get last 10 interactions or all if fewer
        recent_interactions = self.task_performance_history[-10:]

        performance_scores = [
            interaction["performance_score"] for interaction in recent_interactions
        ]
        return float(np.mean(performance_scores))

    def adapt_to_performance_trends(self) -> dict[str, Any]:
        """
        Adapt profile based on performance trends and patterns.

        Returns:
            Dictionary of adaptation recommendations
        """
        adaptations: dict[str, Any] = {
            "expertise_adjustments": [],
            "style_modifications": [],
            "load_optimizations": [],
        }

        if len(self.task_performance_history) < 5:
            return adaptations  # Not enough data for adaptation

        # Analyze performance trends
        recent_performance = self._get_recent_performance()

        # Expertise level adjustment
        if recent_performance > 0.8 and self.spatial_expertise < 0.8:
            adaptations["expertise_adjustments"].append(
                "Consider increasing expertise level"
            )
            self.spatial_expertise = min(1.0, self.spatial_expertise + 0.05)
        elif recent_performance < 0.4 and self.spatial_expertise > 0.2:
            adaptations["expertise_adjustments"].append(
                "Consider reducing expertise expectations"
            )
            self.spatial_expertise = max(0.0, self.spatial_expertise - 0.05)

        # Cognitive style adaptation
        style_performance = self._analyze_style_performance()
        if (
            style_performance.get("visual_performance", 0.5)
            > style_performance.get("verbal_performance", 0.5) + 0.2
        ):
            adaptations["style_modifications"].append(
                "Strengthen visual processing preferences"
            )
            if self.cognitive_style == "balanced":
                self.cognitive_style = "visualizer"
        elif (
            style_performance.get("verbal_performance", 0.5)
            > style_performance.get("visual_performance", 0.5) + 0.2
        ):
            adaptations["style_modifications"].append(
                "Strengthen verbal processing preferences"
            )
            if self.cognitive_style == "balanced":
                self.cognitive_style = "verbalizer"

        # Load preference optimization
        load_analysis = self._analyze_load_patterns()
        optimal_load = load_analysis.get("optimal_load", 0.5)

        if abs(self._get_load_preference_score() - optimal_load) > 0.2:
            adaptations["load_optimizations"].append(
                f"Adjust load preference toward {optimal_load:.2f}"
            )

        return adaptations

    def _analyze_style_performance(self) -> dict[str, float]:
        """Analyze performance across different cognitive styles."""
        style_performance = {"visual_performance": 0.5, "verbal_performance": 0.5}

        if not self.task_performance_history:
            return style_performance

        # Analyze recent interactions for style preferences

        # This would analyze interaction types and outcomes
        # For now, return balanced performance
        return style_performance

    def _analyze_load_patterns(self) -> dict[str, float]:
        """Analyze cognitive load patterns in recent interactions."""
        load_patterns = {"optimal_load": 0.5, "load_sensitivity": 0.5}

        if not self.task_performance_history:
            return load_patterns

        # Analyze relationship between cognitive load and performance
        recent_interactions = self.task_performance_history[-20:]

        loads = [interaction["cognitive_load"] for interaction in recent_interactions]
        performances = [
            interaction["performance_score"] for interaction in recent_interactions
        ]

        if loads and performances:
            # Find load level that maximizes performance
            correlation = np.corrcoef(loads, performances)[0, 1]
            load_patterns["load_sensitivity"] = abs(correlation)

            # Simple optimization - find load with best performance
            best_performance_idx = performances.index(max(performances))
            load_patterns["optimal_load"] = loads[best_performance_idx]

        return load_patterns

    def get_profile_summary(self) -> dict[str, Any]:
        """Get comprehensive profile summary."""
        return {
            "user_id": self.user_id,
            "profile_version": self.profile_version,
            "spatial_capabilities": {
                "expertise_level": self.spatial_expertise,
                "memory_capacity": self.spatial_memory_capacity,
                "attention_span": self.spatial_attention_span,
                "reasoning_style": self.spatial_reasoning_style,
            },
            "cognitive_preferences": {
                "cognitive_style": self.cognitive_style,
                "learning_preference": self.learning_preference,
                "load_preference": self.cognitive_load_preference,
            },
            "performance_metrics": {
                "recent_performance": self._get_recent_performance(),
                "interaction_count": len(self.task_performance_history),
                "adaptation_status": "active"
                if len(self.task_performance_history) > 10
                else "learning",
            },
            "demographics": {
                "age_group": self.age_group,
                "experience_level": self.experience_level,
                "domain_expertise": self.domain_experience,
            },
            "last_updated": self.last_updated.isoformat(),
        }

    def export_profile(self) -> dict[str, Any]:
        """Export profile as dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "profile_version": self.profile_version,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "spatial_expertise": self.spatial_expertise,
            "spatial_memory_capacity": self.spatial_memory_capacity,
            "spatial_attention_span": self.spatial_attention_span,
            "spatial_reasoning_style": self.spatial_reasoning_style,
            "cognitive_style": self.cognitive_style,
            "learning_preference": self.learning_preference,
            "cognitive_load_preference": self.cognitive_load_preference,
            "navigation_preferences": self.navigation_preferences,
            "visualization_preferences": self.visualization_preferences,
            "interaction_patterns": self.interaction_patterns,
            "task_performance_history": self.task_performance_history,
            "adaptation_metrics": self.adaptation_metrics,
            "age_group": self.age_group,
            "experience_level": self.experience_level,
            "domain_experience": self.domain_experience,
        }

    @classmethod
    def import_profile(cls, profile_data: dict[str, Any]) -> "UserCognitiveProfile":
        """Import profile from dictionary data."""
        # Create instance with basic fields
        profile = cls(user_id=profile_data["user_id"])

        # Update all fields from imported data
        for key, value in profile_data.items():
            if hasattr(profile, key):
                if key in ["created_at", "last_updated"]:
                    # Convert ISO strings back to datetime objects
                    setattr(profile, key, datetime.fromisoformat(value))
                else:
                    setattr(profile, key, value)

        return profile


class ProfileManager:
    """
    Manager for user cognitive profiles with learning and adaptation capabilities.

    This class handles:
    - Profile creation and initialization
    - Profile learning from user behavior
    - Profile adaptation and optimization
    - Profile persistence and retrieval
    - Multi-user profile management
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize profile manager.

        Args:
            config: Configuration parameters for profile management
        """
        self.config = config or {}
        self.profiles: dict[
            str, UserCognitiveProfile
        ] = {}  # user_id -> UserCognitiveProfile
        self.profile_learning_enabled = self.config.get("learning_enabled", True)
        self.adaptation_enabled = self.config.get("adaptation_enabled", True)

        # Profile learning parameters
        self.learning_parameters = {
            "expertise_learning_rate": self.config.get("expertise_learning_rate", 0.1),
            "style_adaptation_rate": self.config.get("style_adaptation_rate", 0.05),
            "load_optimization_rate": self.config.get("load_optimization_rate", 0.02),
        }

        logger.info("Profile Manager initialized")

    def create_profile(
        self, user_id: str, initial_assessment: dict[str, Any] | None = None
    ) -> UserCognitiveProfile:
        """
        Create a new user cognitive profile.

        Args:
            user_id: Unique user identifier
            initial_assessment: Initial assessment data (optional)

        Returns:
            Created user profile
        """
        # Initialize profile with defaults
        profile = UserCognitiveProfile(user_id=user_id)

        # Apply initial assessment if provided
        if initial_assessment:
            self._apply_initial_assessment(profile, initial_assessment)

        self.profiles[user_id] = profile

        logger.info(f"Profile created for user {user_id}")
        return profile

    def _apply_initial_assessment(
        self, profile: UserCognitiveProfile, assessment: dict[str, Any]
    ) -> None:
        """Apply initial assessment data to profile."""
        # Update profile based on assessment results
        if "spatial_expertise_score" in assessment:
            profile.spatial_expertise = min(
                1.0, max(0.0, assessment["spatial_expertise_score"])
            )

        if "cognitive_style" in assessment:
            profile.cognitive_style = assessment["cognitive_style"]

        if "preferred_load_level" in assessment:
            load_map = {"light": "low", "normal": "moderate", "heavy": "high"}
            profile.cognitive_load_preference = load_map.get(
                assessment["preferred_load_level"], "moderate"
            )

        # Store assessment metadata
        profile.navigation_preferences["initial_assessment"] = assessment

    def get_profile(self, user_id: str) -> UserCognitiveProfile | None:
        """Retrieve user profile by ID."""
        return self.profiles.get(user_id)

    def update_profile_from_interaction(
        self, user_id: str, interaction_data: dict[str, Any], outcome: dict[str, Any]
    ) -> None:
        """
        Update user profile based on interaction and outcome.

        Args:
            user_id: User identifier
            interaction_data: Interaction details
            outcome: Interaction outcome and performance
        """
        if user_id not in self.profiles:
            # Create profile if it doesn't exist
            self.create_profile(user_id)

        profile = self.profiles[user_id]

        # Update profile with interaction data
        profile.update_from_interaction(interaction_data, outcome)

        # Apply learning and adaptation if enabled
        if self.profile_learning_enabled:
            self._apply_learning(profile, interaction_data, outcome)

        if self.adaptation_enabled:
            self._apply_adaptation(profile)

    def _apply_learning(
        self,
        profile: UserCognitiveProfile,
        interaction_data: dict[str, Any],
        outcome: dict[str, Any],
    ) -> None:
        """Apply learning algorithms to update profile."""
        # Expertise learning based on performance feedback
        performance_score = outcome.get("performance_score", 0.5)
        task_difficulty = interaction_data.get("task_difficulty", 0.5)

        # Adaptive expertise update
        if performance_score > 0.7 and task_difficulty > 0.5:
            # Good performance on difficult task -> increase expertise
            profile.spatial_expertise = min(
                1.0,
                profile.spatial_expertise
                + self.learning_parameters["expertise_learning_rate"] * task_difficulty,
            )
        elif performance_score < 0.4 and task_difficulty < 0.5:
            # Poor performance on easy task -> decrease expertise
            profile.spatial_expertise = max(
                0.0,
                profile.spatial_expertise
                - self.learning_parameters["expertise_learning_rate"]
                * (1 - task_difficulty),
            )

    def _apply_adaptation(self, profile: UserCognitiveProfile) -> None:
        """Apply adaptation strategies to optimize profile."""
        # Get adaptation recommendations
        adaptations = profile.adapt_to_performance_trends()

        # Apply style modifications
        if adaptations["style_modifications"]:
            # Gradually adapt cognitive style based on performance
            style_performance = profile._analyze_style_performance()

            if (
                style_performance.get("visual_performance", 0.5)
                > style_performance.get("verbal_performance", 0.5) + 0.1
            ):
                if profile.cognitive_style != "visualizer":
                    # Move toward visualizer style
                    profile.cognitive_style = "visualizer"

        # Apply load optimizations
        if adaptations["load_optimizations"]:
            load_analysis = profile._analyze_load_patterns()
            optimal_load = load_analysis.get("optimal_load", 0.5)

            # Map numeric load to preference category
            if optimal_load < 0.4:
                profile.cognitive_load_preference = "low"
            elif optimal_load > 0.6:
                profile.cognitive_load_preference = "high"
            else:
                profile.cognitive_load_preference = "moderate"

    def get_user_recommendations(
        self, user_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get personalized recommendations for a user.

        Args:
            user_id: User identifier
            context: Current usage context

        Returns:
            Personalized recommendations
        """
        if user_id not in self.profiles:
            # Create default profile
            profile = self.create_profile(user_id)
        else:
            profile = self.profiles[user_id]

        return profile.get_personalized_recommendations(context)

    def export_all_profiles(self) -> dict[str, Any]:
        """Export all profiles for backup or analysis."""
        return {
            "export_time": datetime.now().isoformat(),
            "profile_count": len(self.profiles),
            "profiles": {
                user_id: profile.export_profile()
                for user_id, profile in self.profiles.items()
            },
        }

    def import_profiles(self, profiles_data: dict[str, Any]) -> int:
        """Import profiles from exported data."""
        imported_count = 0

        for user_id, profile_data in profiles_data.get("profiles", {}).items():
            profile = UserCognitiveProfile.import_profile(profile_data)
            self.profiles[user_id] = profile
            imported_count += 1

        logger.info(f"Imported {imported_count} user profiles")
        return imported_count
