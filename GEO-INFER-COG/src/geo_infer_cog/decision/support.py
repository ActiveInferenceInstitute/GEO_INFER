"""
Decision Support Systems for GEO-INFER-COG

This module implements human-centered decision support systems that integrate
cognitive processing capabilities with spatial decision-making. The systems
provide personalized decision recommendations based on cognitive profiles,
uncertainty quantification, and spatial reasoning.

Key Components:
- SpatialDecisionSupport: Main decision support engine
- Decision alternatives with cognitive weighting
- Uncertainty-aware decision strategies
- User-adapted recommendation systems
- Multi-criteria decision analysis with cognitive factors

Mathematical Foundations:
- Multi-attribute utility theory for decision weighting
- Prospect theory for uncertainty handling (Kahneman & Tversky, 1979)
- Cognitive load theory in decision contexts
- Bayesian decision theory for uncertain environments
- Analytic hierarchy process for multi-criteria decisions
"""

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from ..models.user_profiles import UserCognitiveProfile
from ..utils.rng import resolve_rng

# Module-level monotonic counter backing decision analysis IDs.
_ANALYSIS_ID_SEQUENCE: "itertools.count[int]" = itertools.count(1)

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    """Decision-making strategies supported by the system."""

    COGNITIVE_WEIGHTED = "cognitive_weighted"
    PROSPECT_THEORY = "prospect_theory"
    BAYESIAN_DECISION = "bayesian_decision"
    MULTI_CRITERIA = "multi_criteria"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class DecisionAlternative:
    """Represents a decision alternative with cognitive properties."""

    alternative_id: str
    description: str
    spatial_context: dict[str, Any]
    cognitive_factors: dict[str, float] = field(default_factory=dict)
    uncertainty_measures: dict[str, float] = field(default_factory=dict)
    expected_outcomes: dict[str, Any] = field(default_factory=dict)
    risk_assessment: dict[str, float] = field(default_factory=dict)

    def calculate_cognitive_compatibility(
        self, user_profile: UserCognitiveProfile
    ) -> float:
        """Calculate how compatible this alternative is with user cognitive profile."""
        compatibility = 0.5  # Base compatibility

        # Expertise compatibility
        expertise_required = self.cognitive_factors.get("complexity_level", 0.5)
        expertise_match = 1.0 - abs(user_profile.spatial_expertise - expertise_required)
        compatibility += expertise_match * 0.3

        # Cognitive load compatibility
        load_impact = self.cognitive_factors.get("cognitive_load", 0.5)
        load_preference_score = self._get_load_preference_score(user_profile)
        load_compatibility = 1.0 - abs(load_impact - load_preference_score)
        compatibility += load_compatibility * 0.3

        # Risk tolerance compatibility
        risk_level = self.risk_assessment.get("overall_risk", 0.5)
        risk_tolerance = self._get_risk_tolerance_score(user_profile)
        risk_compatibility = 1.0 - abs(risk_level - risk_tolerance)
        compatibility += risk_compatibility * 0.4

        return min(1.0, max(0.0, compatibility))

    def _get_load_preference_score(self, user_profile: UserCognitiveProfile) -> float:
        """Convert user load preference to numeric score."""
        preference_scores = {"low": 0.3, "moderate": 0.5, "high": 0.7}
        return preference_scores.get(user_profile.cognitive_load_preference, 0.5)

    def _get_risk_tolerance_score(self, user_profile: UserCognitiveProfile) -> float:
        """Convert user risk tolerance to numeric score."""
        # This would be derived from user profile and interaction history
        # For now, use a simple mapping based on expertise and experience
        base_tolerance = 0.5

        if user_profile.spatial_expertise > 0.7:
            base_tolerance += 0.2  # Experts tend to be more risk-tolerant
        elif user_profile.spatial_expertise < 0.4:
            base_tolerance -= 0.2  # Novices tend to be more risk-averse

        return min(1.0, max(0.0, base_tolerance))


@dataclass
class DecisionRecommendation:
    """Represents a decision recommendation with rationale."""

    alternative_id: str
    recommendation_score: float
    confidence_level: float
    primary_rationale: str
    supporting_factors: list[str] = field(default_factory=list)
    risk_warnings: list[str] = field(default_factory=list)
    cognitive_considerations: list[str] = field(default_factory=list)

    def to_display_format(self, user_profile: UserCognitiveProfile) -> dict[str, Any]:
        """Format recommendation for user display."""
        display_format: dict[str, Any] = {
            "alternative_id": self.alternative_id,
            "recommendation_score": self.recommendation_score,
            "confidence_level": self.confidence_level,
            "primary_rationale": self.primary_rationale,
            "display_emphasis": "strong"
            if self.recommendation_score > 0.8
            else "moderate",
            "user_considerations": [],
        }

        # Adapt display based on user profile
        if user_profile.cognitive_style == "visualizer":
            display_format["visual_indicators"] = True
            display_format["text_summary"] = "brief"
        else:
            display_format["visual_indicators"] = False
            display_format["text_summary"] = "detailed"

        # Add cognitive considerations
        if user_profile.cognitive_load_preference == "low":
            display_format["user_considerations"].extend(
                ["Simplified explanation provided", "Reduced cognitive load option"]
            )

        if user_profile.spatial_expertise < 0.5:
            display_format["user_considerations"].extend(
                ["Additional guidance included", "Step-by-step rationale provided"]
            )

        return display_format


class SpatialDecisionSupport:
    """
    Human-centered spatial decision support system.

    This system integrates cognitive processing with decision-making to provide
    personalized, uncertainty-aware spatial decision recommendations. The system
    considers user cognitive profiles, spatial reasoning results, and uncertainty
    quantification to generate optimal decision strategies.
    """

    def __init__(
        self,
        decision_framework: str = "prospect_theory",
        cognitive_bias_mitigation: bool = True,
        spatial_reasoning_model: str = "mental_maps",
        uncertainty_incorporation: str = "bayesian",
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize spatial decision support system.

        Args:
            decision_framework: Decision-making framework ('prospect_theory', 'cognitive_weighted', 'bayesian')
            cognitive_bias_mitigation: Enable cognitive bias detection and mitigation
            spatial_reasoning_model: Spatial reasoning model to use ('mental_maps', 'qualitative', 'quantitative')
            uncertainty_incorporation: Uncertainty handling method ('bayesian', 'fuzzy', 'possibilistic')
            config: Additional configuration parameters
            rng: Optional random generator for analysis-ID suffixes. When
                omitted, a fixed-seed generator is used so the system is
                deterministic by default.
        """
        self.decision_framework = decision_framework
        self.cognitive_bias_mitigation = cognitive_bias_mitigation
        self.spatial_reasoning_model = spatial_reasoning_model
        self.uncertainty_incorporation = uncertainty_incorporation
        self.config = config or {}

        # Resolved via the repo-wide resolve_rng pattern; None resolves to a
        # fixed seed so decision analyses are reproducible by default.
        self._rng = resolve_rng(rng)

        # Decision-making parameters
        self.decision_parameters = {
            "risk_aversion_parameter": self.config.get(
                "risk_aversion", 0.88
            ),  # Prospect theory parameter
            "loss_aversion_parameter": self.config.get(
                "loss_aversion", 2.25
            ),  # Prospect theory parameter
            "cognitive_bias_threshold": self.config.get(
                "cognitive_bias_threshold", 0.7
            ),
            "uncertainty_weight": self.config.get("uncertainty_weight", 0.3),
        }

        # Performance tracking
        self.decision_metrics = {
            "decisions_analyzed": 0,
            "recommendations_generated": 0,
            "user_feedback_incorporated": 0,
            "bias_mitigation_applied": 0,
        }

        logger.info(
            f"Spatial Decision Support initialized with framework: {decision_framework}"
        )

    def analyze_decision(
        self,
        decision_problem: dict[str, Any],
        spatial_alternatives: list[dict[str, Any]],
        decision_criteria: list[str],
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> dict[str, Any]:
        """
        Analyze spatial decision scenario and provide recommendations.

        Args:
            decision_problem: Description of the decision problem
            spatial_alternatives: List of spatial decision alternatives
            decision_criteria: Criteria for decision evaluation
            stakeholder_profiles: Cognitive profiles of decision stakeholders

        Returns:
            Comprehensive decision analysis with recommendations
        """
        start_time = datetime.now()

        try:
            # Step 1: Extract and validate decision alternatives
            alternatives = self._extract_decision_alternatives(spatial_alternatives)
            self.decision_metrics["decisions_analyzed"] += 1

            # Step 2: Apply cognitive bias mitigation if enabled
            if self.cognitive_bias_mitigation:
                alternatives = self._apply_bias_mitigation(
                    alternatives, stakeholder_profiles
                )

            # Step 3: Evaluate alternatives using selected framework
            # Framework selection is driven by the public DecisionStrategy
            # enum; unrecognized values fall back to multi-criteria analysis.
            if self.decision_framework == DecisionStrategy.PROSPECT_THEORY.value:
                evaluations = self._apply_prospect_theory(
                    alternatives, decision_criteria
                )
            elif self.decision_framework == DecisionStrategy.COGNITIVE_WEIGHTED.value:
                evaluations = self._apply_cognitive_weighting(
                    alternatives, stakeholder_profiles, decision_criteria
                )
            elif self.decision_framework == DecisionStrategy.BAYESIAN_DECISION.value:
                evaluations = self._apply_bayesian_decision(
                    alternatives, decision_criteria
                )
            else:
                evaluations = self._apply_multi_criteria_analysis(
                    alternatives, decision_criteria
                )

            # Step 4: Generate personalized recommendations
            recommendations = self._generate_recommendations(
                evaluations, alternatives, stakeholder_profiles, decision_criteria
            )

            # Step 5: Assess decision uncertainty and risk
            uncertainty_assessment = self._assess_decision_uncertainty(
                alternatives, evaluations
            )
            risk_assessment = self._assess_decision_risks(
                alternatives, stakeholder_profiles
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            analysis_result = {
                "analysis_id": f"decision_{next(_ANALYSIS_ID_SEQUENCE)}_{int(self._rng.integers(0, 1000))}",
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "decision_problem": decision_problem,
                "alternatives": alternatives,
                "evaluations": evaluations,
                "recommendations": recommendations,
                "uncertainty_assessment": uncertainty_assessment,
                "risk_assessment": risk_assessment,
                "decision_framework": self.decision_framework,
                "stakeholder_analysis": self._analyze_stakeholder_compatibility(
                    alternatives, stakeholder_profiles
                ),
                "decision_metrics": self.decision_metrics.copy(),
            }

            logger.info(
                f"Decision analysis completed in {processing_time:.3f}s with {len(recommendations)} recommendations"
            )
            return analysis_result

        except Exception as e:
            logger.error(f"Error in decision analysis: {str(e)}")
            raise

    def _extract_decision_alternatives(
        self, spatial_alternatives: list[dict[str, Any]]
    ) -> list[DecisionAlternative]:
        """Extract and structure decision alternatives."""
        alternatives: list[DecisionAlternative] = []

        for alt_data in spatial_alternatives:
            alternative = DecisionAlternative(
                alternative_id=alt_data.get("id", f"alt_{len(alternatives)}"),
                description=alt_data.get("description", ""),
                spatial_context=alt_data.get("spatial_context", {}),
                cognitive_factors=alt_data.get("cognitive_factors", {}),
                uncertainty_measures=alt_data.get("uncertainty_measures", {}),
                expected_outcomes=alt_data.get("expected_outcomes", {}),
                risk_assessment=alt_data.get("risk_assessment", {}),
            )
            alternatives.append(alternative)

        return alternatives

    def _apply_bias_mitigation(
        self,
        alternatives: list[DecisionAlternative],
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> list[DecisionAlternative]:
        """Apply cognitive bias mitigation strategies."""
        mitigated_alternatives = []

        for alternative in alternatives:
            # Detect potential biases
            bias_warnings = self._detect_cognitive_biases(
                alternative, stakeholder_profiles
            )

            if bias_warnings:
                # Apply mitigation strategies
                mitigated_alt = self._mitigate_detected_biases(
                    alternative, bias_warnings
                )
                mitigated_alternatives.append(mitigated_alt)
                self.decision_metrics["bias_mitigation_applied"] += 1
            else:
                mitigated_alternatives.append(alternative)

        return mitigated_alternatives

    def _detect_cognitive_biases(
        self,
        alternative: DecisionAlternative,
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> list[str]:
        """Detect cognitive biases in decision alternative evaluation."""
        biases = []

        # Status quo bias detection
        if alternative.risk_assessment.get("change_magnitude", 0) > 0.8:
            biases.append("status_quo_bias")

        # Confirmation bias detection
        if alternative.cognitive_factors.get("familiarity_boost", 0) > 0.7:
            biases.append("confirmation_bias")

        # Overconfidence bias detection
        if alternative.uncertainty_measures.get("confidence_overestimation", 0) > 0.6:
            biases.append("overconfidence_bias")

        return biases

    def _mitigate_detected_biases(
        self, alternative: DecisionAlternative, bias_warnings: list[str]
    ) -> DecisionAlternative:
        """Apply mitigation strategies for detected biases."""
        # Create a copy of the alternative with mitigated factors
        mitigated = DecisionAlternative(
            alternative_id=alternative.alternative_id,
            description=alternative.description,
            spatial_context=alternative.spatial_context,
            cognitive_factors=alternative.cognitive_factors.copy(),
            uncertainty_measures=alternative.uncertainty_measures.copy(),
            expected_outcomes=alternative.expected_outcomes.copy(),
            risk_assessment=alternative.risk_assessment.copy(),
        )

        # Apply mitigation strategies
        for bias in bias_warnings:
            if bias == "status_quo_bias":
                # Reduce change magnitude assessment
                if "change_magnitude" in mitigated.risk_assessment:
                    mitigated.risk_assessment["change_magnitude"] *= 0.8

            elif bias == "confirmation_bias":
                # Reduce familiarity boost
                if "familiarity_boost" in mitigated.cognitive_factors:
                    mitigated.cognitive_factors["familiarity_boost"] *= 0.7

            elif bias == "overconfidence_bias":
                # Reduce confidence estimates
                for key in mitigated.uncertainty_measures:
                    if "confidence" in key.lower():
                        mitigated.uncertainty_measures[key] *= 0.9

        return mitigated

    def _apply_prospect_theory(
        self, alternatives: list[DecisionAlternative], criteria: list[str]
    ) -> dict[str, float]:
        """Apply prospect theory for decision evaluation."""
        evaluations = {}

        for alternative in alternatives:
            # Calculate prospect theory value
            prospect_value = 0.0

            for criterion in criteria:
                # Get outcome value for this criterion
                outcome_value = alternative.expected_outcomes.get(criterion, 0)

                # Apply prospect theory value function
                # V(x) = x^α if x >= 0, -λ(-x)^β if x < 0
                if outcome_value >= 0:
                    prospect_value += (
                        outcome_value
                        ** self.decision_parameters["risk_aversion_parameter"]
                    )
                else:
                    loss_value = -outcome_value
                    prospect_value -= self.decision_parameters[
                        "loss_aversion_parameter"
                    ] * (
                        loss_value
                        ** self.decision_parameters["risk_aversion_parameter"]
                    )

            # Weight by uncertainty
            uncertainty_factor = alternative.uncertainty_measures.get(
                "overall_uncertainty", 0.5
            )
            prospect_value *= (
                1.0
                - uncertainty_factor * self.decision_parameters["uncertainty_weight"]
            )

            evaluations[alternative.alternative_id] = prospect_value

        return evaluations

    def _apply_cognitive_weighting(
        self,
        alternatives: list[DecisionAlternative],
        stakeholder_profiles: list[UserCognitiveProfile],
        criteria: list[str],
    ) -> dict[str, float]:
        """Apply cognitive weighting based on user profiles."""
        evaluations = {}

        for alternative in alternatives:
            # Calculate weighted score across all stakeholders
            total_weighted_score = 0.0
            total_weight = 0.0

            for profile in stakeholder_profiles:
                # Get compatibility score
                compatibility = alternative.calculate_cognitive_compatibility(profile)

                # Calculate criterion scores
                criterion_scores = []
                for criterion in criteria:
                    score = alternative.expected_outcomes.get(criterion, 0.5)
                    criterion_scores.append(score)

                # Weighted average of criterion scores
                if criterion_scores:
                    avg_score = sum(criterion_scores) / len(criterion_scores)
                    weighted_score = avg_score * compatibility
                    total_weighted_score += weighted_score
                    total_weight += 1.0

            # Normalize by number of stakeholders
            if total_weight > 0:
                evaluations[alternative.alternative_id] = (
                    total_weighted_score / total_weight
                )
            else:
                evaluations[alternative.alternative_id] = 0.5

        return evaluations

    def _apply_bayesian_decision(
        self, alternatives: list[DecisionAlternative], criteria: list[str]
    ) -> dict[str, float]:
        """Apply Bayesian decision theory."""
        evaluations = {}

        for alternative in alternatives:
            # Calculate expected utility using Bayesian approach
            expected_utility = 0.0

            for criterion in criteria:
                # Get prior probability and likelihood
                prior_prob = alternative.expected_outcomes.get(
                    f"{criterion}_prior", 0.5
                )
                likelihood = alternative.expected_outcomes.get(
                    f"{criterion}_likelihood", 0.5
                )

                # Bayesian update: P(criterion|alternative) = P(alternative|criterion) * P(criterion) / P(alternative)
                posterior = (likelihood * prior_prob) / (
                    prior_prob * likelihood
                    + (1 - prior_prob) * (1 - likelihood)
                    + 1e-10
                )

                # Convert to utility (0-1 scale)
                utility = min(1.0, max(0.0, posterior))
                expected_utility += utility

            # Normalize by number of criteria
            if criteria:
                expected_utility /= len(criteria)

            # Adjust for uncertainty
            uncertainty_penalty = alternative.uncertainty_measures.get(
                "overall_uncertainty", 0.0
            )
            expected_utility *= 1.0 - uncertainty_penalty * 0.3

            evaluations[alternative.alternative_id] = expected_utility

        return evaluations

    def _apply_multi_criteria_analysis(
        self, alternatives: list[DecisionAlternative], criteria: list[str]
    ) -> dict[str, float]:
        """Apply multi-criteria decision analysis."""
        evaluations = {}

        # Define criterion weights (could be user-configurable)
        criterion_weights = {criterion: 1.0 / len(criteria) for criterion in criteria}

        for alternative in alternatives:
            # Calculate weighted sum of criterion scores
            weighted_score = 0.0

            for criterion in criteria:
                criterion_score = alternative.expected_outcomes.get(criterion, 0.5)
                weight = criterion_weights.get(criterion, 1.0 / len(criteria))
                weighted_score += criterion_score * weight

            # Adjust for cognitive factors
            cognitive_adjustment = alternative.cognitive_factors.get(
                "cognitive_ease", 1.0
            )
            weighted_score *= cognitive_adjustment

            evaluations[alternative.alternative_id] = weighted_score

        return evaluations

    def _generate_recommendations(
        self,
        evaluations: dict[str, float],
        alternatives: list[DecisionAlternative],
        stakeholder_profiles: list[UserCognitiveProfile],
        criteria: list[str],
    ) -> list[DecisionRecommendation]:
        """Generate decision recommendations based on evaluations."""
        recommendations = []

        # Sort alternatives by evaluation score
        sorted_alternatives = sorted(
            evaluations.items(), key=lambda x: x[1], reverse=True
        )

        for i, (alt_id, score) in enumerate(sorted_alternatives):
            # Find corresponding alternative
            alternative = next(
                alt for alt in alternatives if alt.alternative_id == alt_id
            )

            # Generate recommendation
            recommendation = DecisionRecommendation(
                alternative_id=alt_id,
                recommendation_score=score,
                confidence_level=self._calculate_recommendation_confidence(
                    alternative, score
                ),
                primary_rationale=self._generate_primary_rationale(
                    alternative, score, i == 0
                ),
                supporting_factors=self._generate_supporting_factors(
                    alternative, criteria
                ),
                risk_warnings=self._generate_risk_warnings(alternative),
                cognitive_considerations=self._generate_cognitive_considerations(
                    alternative, stakeholder_profiles
                ),
            )

            recommendations.append(recommendation)
            self.decision_metrics["recommendations_generated"] += 1

        return recommendations

    def _calculate_recommendation_confidence(
        self, alternative: DecisionAlternative, score: float
    ) -> float:
        """Calculate confidence in recommendation."""
        # Base confidence from evaluation score
        base_confidence = min(
            1.0, score + 0.2
        )  # Boost slightly for top recommendations

        # Adjust for alternative uncertainty
        uncertainty_penalty = alternative.uncertainty_measures.get(
            "overall_uncertainty", 0.0
        )
        confidence = base_confidence * (1.0 - uncertainty_penalty * 0.4)

        return min(1.0, max(0.0, confidence))

    def _generate_primary_rationale(
        self, alternative: DecisionAlternative, score: float, is_top_choice: bool
    ) -> str:
        """Generate primary rationale for recommendation."""
        if is_top_choice:
            return f"Highest overall score ({score:.3f}) with optimal cognitive compatibility"

        # Find strongest factor
        max_factor = 0.0
        best_criterion = "overall_compatibility"

        for key, value in alternative.expected_outcomes.items():
            if isinstance(value, (int, float)) and value > max_factor:
                max_factor = value
                best_criterion = key

        return f"Strong performance in {best_criterion} ({max_factor:.3f})"

    def _generate_supporting_factors(
        self, alternative: DecisionAlternative, criteria: list[str]
    ) -> list[str]:
        """Generate supporting factors for recommendation."""
        factors = []

        for criterion in criteria:
            score = alternative.expected_outcomes.get(criterion, 0.5)
            if score > 0.7:
                factors.append(f"Excellent {criterion} score ({score:.2f})")

        # Add cognitive factors
        if alternative.cognitive_factors.get("cognitive_ease", 0) > 0.7:
            factors.append("High cognitive compatibility")

        if alternative.uncertainty_measures.get("overall_uncertainty", 1.0) < 0.3:
            factors.append("Low uncertainty in outcomes")

        return factors[:3]  # Limit to top 3 factors

    def _generate_risk_warnings(self, alternative: DecisionAlternative) -> list[str]:
        """Generate risk warnings for alternative."""
        warnings = []

        risk_level = alternative.risk_assessment.get("overall_risk", 0.5)
        if risk_level > 0.7:
            warnings.append(f"High risk level ({risk_level:.2f})")

        change_magnitude = alternative.risk_assessment.get("change_magnitude", 0.0)
        if change_magnitude > 0.8:
            warnings.append(f"Significant change required ({change_magnitude:.2f})")

        uncertainty_level = alternative.uncertainty_measures.get(
            "overall_uncertainty", 0.0
        )
        if uncertainty_level > 0.6:
            warnings.append(f"High outcome uncertainty ({uncertainty_level:.2f})")

        return warnings

    def _generate_cognitive_considerations(
        self,
        alternative: DecisionAlternative,
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> list[str]:
        """Generate cognitive considerations for recommendation."""
        considerations = []

        # Check compatibility across stakeholders
        compatibilities = [
            alternative.calculate_cognitive_compatibility(profile)
            for profile in stakeholder_profiles
        ]

        avg_compatibility = (
            sum(compatibilities) / len(compatibilities) if compatibilities else 0.5
        )

        if avg_compatibility < 0.6:
            considerations.append(
                f"Moderate stakeholder compatibility ({avg_compatibility:.2f})"
            )

        # Check cognitive load implications
        load_impact = alternative.cognitive_factors.get("cognitive_load", 0.5)
        if load_impact > 0.7:
            considerations.append(f"High cognitive load impact ({load_impact:.2f})")

        return considerations

    def _assess_decision_uncertainty(
        self, alternatives: list[DecisionAlternative], evaluations: dict[str, float]
    ) -> dict[str, Any]:
        """Assess overall uncertainty in decision analysis."""
        uncertainty_measures = []

        for alternative in alternatives:
            uncertainty = alternative.uncertainty_measures.get(
                "overall_uncertainty", 0.5
            )
            uncertainty_measures.append(uncertainty)

        if uncertainty_measures:
            assessment = {
                "mean_uncertainty": float(np.mean(uncertainty_measures)),
                "uncertainty_range": float(
                    np.max(uncertainty_measures) - np.min(uncertainty_measures)
                ),
                "high_uncertainty_alternatives": len(
                    [u for u in uncertainty_measures if u > 0.7]
                ),
                "uncertainty_distribution": {
                    "low": len([u for u in uncertainty_measures if u < 0.3]),
                    "medium": len([u for u in uncertainty_measures if 0.3 <= u <= 0.7]),
                    "high": len([u for u in uncertainty_measures if u > 0.7]),
                },
            }
        else:
            assessment = {
                "mean_uncertainty": 0.0,
                "uncertainty_range": 0.0,
                "high_uncertainty_alternatives": 0,
                "uncertainty_distribution": {"low": 0, "medium": 0, "high": 0},
            }

        return assessment

    def _assess_decision_risks(
        self,
        alternatives: list[DecisionAlternative],
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> dict[str, Any]:
        """Assess risks associated with decision alternatives."""
        risk_measures = []

        for alternative in alternatives:
            risk = alternative.risk_assessment.get("overall_risk", 0.5)
            risk_measures.append(risk)

        if risk_measures:
            assessment = {
                "mean_risk": float(np.mean(risk_measures)),
                "risk_range": float(np.max(risk_measures) - np.min(risk_measures)),
                "high_risk_alternatives": len([r for r in risk_measures if r > 0.7]),
                "risk_tolerance_compatibility": self._assess_risk_tolerance_compatibility(
                    risk_measures, stakeholder_profiles
                ),
            }
        else:
            assessment = {
                "mean_risk": 0.0,
                "risk_range": 0.0,
                "high_risk_alternatives": 0,
                "risk_tolerance_compatibility": 0.5,
            }

        return assessment

    def _assess_risk_tolerance_compatibility(
        self,
        risk_measures: list[float],
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> float:
        """Assess compatibility between alternative risks and stakeholder risk tolerance."""
        if not stakeholder_profiles or not risk_measures:
            return 0.5

        # Calculate average risk tolerance across stakeholders
        avg_risk_tolerance = sum(
            self._get_risk_tolerance_score(profile) for profile in stakeholder_profiles
        ) / len(stakeholder_profiles)

        # Calculate average risk of alternatives
        avg_risk = sum(risk_measures) / len(risk_measures)

        # Compatibility is inverse of difference
        risk_compatibility = 1.0 - abs(avg_risk - avg_risk_tolerance)

        return min(1.0, max(0.0, risk_compatibility))

    def _get_risk_tolerance_score(self, user_profile: UserCognitiveProfile) -> float:
        """Convert user profile to risk tolerance score."""
        # Base tolerance from expertise
        base_tolerance = 0.5 + (user_profile.spatial_expertise * 0.3)

        # Adjust for cognitive load preference (higher load tolerance = higher risk tolerance)
        load_preference_score = self._get_load_preference_score(user_profile)
        base_tolerance += (load_preference_score - 0.5) * 0.2

        return min(1.0, max(0.0, base_tolerance))

    def _get_load_preference_score(self, user_profile: UserCognitiveProfile) -> float:
        """Convert user load preference to numeric score."""
        preference_scores = {"low": 0.3, "moderate": 0.5, "high": 0.7}
        return preference_scores.get(user_profile.cognitive_load_preference, 0.5)

    def _analyze_stakeholder_compatibility(
        self,
        alternatives: list[DecisionAlternative],
        stakeholder_profiles: list[UserCognitiveProfile],
    ) -> dict[str, Any]:
        """Analyze stakeholder compatibility with decision alternatives."""
        analysis: dict[str, Any] = {
            "overall_compatibility": 0.0,
            "stakeholder_preferences": {},
            "consensus_potential": 0.0,
            "conflict_indicators": [],
        }

        if not stakeholder_profiles:
            return analysis

        # Calculate compatibility for each stakeholder
        total_compatibility = 0.0

        for profile in stakeholder_profiles:
            stakeholder_compatibilities = []

            for alternative in alternatives:
                compatibility = alternative.calculate_cognitive_compatibility(profile)
                stakeholder_compatibilities.append(compatibility)
                total_compatibility += compatibility

            # Store stakeholder preference (highest compatibility alternative)
            best_alt_idx = np.argmax(stakeholder_compatibilities)
            analysis["stakeholder_preferences"][profile.user_id] = {
                "preferred_alternative": alternatives[best_alt_idx].alternative_id,
                "compatibility_score": stakeholder_compatibilities[best_alt_idx],
            }

        # Overall compatibility
        total_possible = len(stakeholder_profiles) * len(alternatives)
        analysis["overall_compatibility"] = (
            total_compatibility / total_possible if total_possible > 0 else 0.0
        )

        # Consensus potential (how many stakeholders prefer the same alternative)
        preferred_alternatives = [
            prefs["preferred_alternative"]
            for prefs in analysis["stakeholder_preferences"].values()
        ]

        if preferred_alternatives:
            most_common = max(
                set(preferred_alternatives), key=preferred_alternatives.count
            )
            consensus_ratio = preferred_alternatives.count(most_common) / len(
                preferred_alternatives
            )
            analysis["consensus_potential"] = consensus_ratio

        # Conflict indicators
        if consensus_ratio < 0.6:
            analysis["conflict_indicators"].append("Low consensus among stakeholders")

        if analysis["overall_compatibility"] < 0.6:
            analysis["conflict_indicators"].append(
                "Generally low stakeholder compatibility"
            )

        return analysis

    def get_decision_insights(
        self, decision_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate insights and explanations for decision analysis."""
        insights: dict[str, Any] = {
            "decision_summary": {},
            "key_factors": [],
            "uncertainty_insights": [],
            "stakeholder_insights": [],
            "recommendation_explanations": [],
        }

        # Decision summary
        evaluations = decision_analysis["evaluations"]
        recommendations = decision_analysis["recommendations"]

        if evaluations and recommendations:
            top_recommendation = recommendations[0]
            top_alternative_id = top_recommendation["alternative_id"]
            top_score = evaluations[top_alternative_id]

            insights["decision_summary"] = {
                "top_choice": top_alternative_id,
                "confidence_score": top_recommendation["confidence_level"],
                "evaluation_score": top_score,
                "number_of_alternatives": len(evaluations),
            }

        # Key factors analysis
        uncertainty_assessment = decision_analysis.get("uncertainty_assessment", {})
        if uncertainty_assessment:
            insights["uncertainty_insights"] = [
                f"Average uncertainty: {uncertainty_assessment.get('mean_uncertainty', 0):.3f}",
                f"High uncertainty alternatives: {uncertainty_assessment.get('high_uncertainty_alternatives', 0)}",
            ]

        # Stakeholder insights
        stakeholder_analysis = decision_analysis.get("stakeholder_analysis", {})
        if stakeholder_analysis:
            insights["stakeholder_insights"] = [
                f"Overall stakeholder compatibility: {stakeholder_analysis.get('overall_compatibility', 0):.3f}",
                f"Consensus potential: {stakeholder_analysis.get('consensus_potential', 0):.3f}",
            ]

        # Recommendation explanations
        for recommendation in recommendations[:3]:  # Top 3 recommendations
            explanation = {
                "alternative_id": recommendation["alternative_id"],
                "score": recommendation["recommendation_score"],
                "rationale": recommendation["primary_rationale"],
                "supporting_evidence": recommendation["supporting_factors"],
            }
            insights["recommendation_explanations"].append(explanation)

        return insights

    def get_status(self) -> dict[str, Any]:
        """Get current status of the decision support system."""
        return {
            "system_type": "spatial_decision_support",
            "decision_framework": self.decision_framework,
            "status": "active",
            "decision_metrics": self.decision_metrics,
            "configuration": {
                "cognitive_bias_mitigation": self.cognitive_bias_mitigation,
                "spatial_reasoning_model": self.spatial_reasoning_model,
                "uncertainty_incorporation": self.uncertainty_incorporation,
            },
        }
