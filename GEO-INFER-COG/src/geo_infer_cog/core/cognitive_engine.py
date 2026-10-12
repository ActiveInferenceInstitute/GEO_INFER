"""
Cognitive Processing Engine for GEO-INFER-COG

This module implements the core cognitive processing engine that coordinates
spatial perception, reasoning, memory, and decision-making processes to
model human-like spatial cognition in computational systems.

The engine integrates multiple cognitive components:
- Spatial attention models for focus and prioritization
- Working memory for temporary spatial information
- Long-term memory for spatial knowledge storage
- Reasoning systems for spatial inference and problem-solving
- Decision-making frameworks for spatial choices

Mathematical Foundations:
- Bayesian cognitive models for uncertainty handling
- ACT-R cognitive architecture principles
- Spatial cognition theories (Kitchin & Blades, 2002)
- Attention allocation models (Broadbent, 1958; Treisman, 1969)
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np

from ..models.user_profiles import UserCognitiveProfile
from ..utils.rng import resolve_rng
from .spatial_perception import SpatialPerceptionModel
from .spatial_reasoning import SpatialReasoningEngine
from .spatial_memory import SpatialMemoryModel

# Module-level monotonic counter backing processing IDs.
_PROCESSING_ID_SEQUENCE: "itertools.count[int]" = itertools.count(1)

logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    """Represents the current cognitive state of the processing engine."""

    attention_focus: dict[str, float] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    cognitive_load: float = 0.0
    uncertainty_level: float = 0.0
    decision_confidence: float = 0.0
    spatial_context: dict[str, Any] = field(default_factory=dict)
    temporal_context: dict[str, Any] = field(default_factory=dict)

    def update_attention(self, focus_areas: dict[str, float]) -> None:
        """Update attention focus areas with normalized weights."""
        total_weight = sum(focus_areas.values())
        if total_weight > 0:
            self.attention_focus = {k: v / total_weight for k, v in focus_areas.items()}
        self._update_cognitive_load()

    def add_to_working_memory(
        self, key: str, value: Any, importance: float = 1.0
    ) -> None:
        """Add item to working memory with importance weighting."""
        self.working_memory[key] = {
            "value": value,
            "importance": importance,
            "timestamp": datetime.now(),
            "access_count": 0,
        }
        self._update_cognitive_load()

    def retrieve_from_memory(self, key: str) -> Any | None:
        """Retrieve item from working memory and update access patterns."""
        if key in self.working_memory:
            item = self.working_memory[key]
            item["access_count"] += 1
            item["last_access"] = datetime.now()
            return item["value"]
        return None

    def _update_cognitive_load(self) -> None:
        """Calculate current cognitive load based on memory usage and attention."""
        memory_load = len(self.working_memory) / 10.0  # Normalize to 10 items
        attention_load = len(self.attention_focus) / 5.0  # Normalize to 5 focus areas
        self.cognitive_load = min(1.0, (memory_load + attention_load) / 2.0)

    def get_memory_utilization(self) -> dict[str, float]:
        """Get memory utilization statistics."""
        if not self.working_memory:
            return {"utilization": 0.0, "items": 0}

        total_items = len(self.working_memory)
        accessed_items = sum(
            1
            for item in self.working_memory.values()
            if item.get("access_count", 0) > 0
        )

        return {
            "utilization": self.cognitive_load,
            "items": total_items,
            "accessed_ratio": accessed_items / total_items if total_items > 0 else 0.0,
        }


class CognitiveProcessingEngine:
    """
    Core cognitive processing engine for spatial cognition modeling.

    This engine integrates spatial perception, reasoning, memory, and decision-making
    to provide human-like spatial cognition capabilities for geospatial applications.

    The engine implements a cognitive architecture inspired by:
    - ACT-R (Adaptive Control of Thought-Rational) cognitive architecture
    - Bayesian models of cognition for uncertainty handling
    - Spatial cognition theories for geographic reasoning
    """

    def __init__(
        self,
        cognitive_framework: str = "bayesian_attention",
        spatial_resolution: str = "adaptive",
        temporal_modeling: str = "working_memory",
        uncertainty_handling: str = "probabilistic",
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize the cognitive processing engine.

        Args:
            cognitive_framework: Type of cognitive framework ('bayesian_attention', 'act_r', 'soar')
            spatial_resolution: Spatial resolution strategy ('adaptive', 'fixed', 'hierarchical')
            temporal_modeling: Temporal modeling approach ('working_memory', 'episodic', 'semantic')
            uncertainty_handling: Uncertainty handling method ('probabilistic', 'fuzzy', 'possibilistic')
            config: Additional configuration parameters
            rng: Optional random generator threaded into every component model.
                When omitted, a fixed-seed generator is used so processing is
                deterministic by default.
        """
        self.config = config or {}
        self.cognitive_framework = cognitive_framework
        self.spatial_resolution = spatial_resolution
        self.temporal_modeling = temporal_modeling
        self.uncertainty_handling = uncertainty_handling

        # Shared generator threaded into every component model. Resolved via
        # the repo-wide resolve_rng pattern; None resolves to a fixed seed so
        # the engine is deterministic by default.
        self._rng = resolve_rng(rng)

        # Initialize cognitive state
        self.state = CognitiveState()

        # Initialize component models
        self.perception_model = SpatialPerceptionModel(
            framework=cognitive_framework, resolution=spatial_resolution, rng=self._rng
        )

        self.reasoning_engine = SpatialReasoningEngine(
            reasoning_type="qualitative_spatial",
            uncertainty_method=uncertainty_handling,
            rng=self._rng,
        )

        self.memory_model = SpatialMemoryModel(
            memory_types=["working", "long_term", "episodic"],
            consolidation_strategy="adaptive",
            rng=self._rng,
        )

        # Performance tracking
        self.performance_metrics = {
            "decisions_made": 0,
            "reasoning_chains": 0,
            "memory_operations": 0,
            "perception_updates": 0,
        }

        logger.info(
            f"Cognitive Processing Engine initialized with framework: {cognitive_framework}"
        )

    def process_spatial_input(
        self,
        spatial_data: dict[str, Any],
        context: dict[str, Any] | None = None,
        user_profile: UserCognitiveProfile | None = None,
    ) -> dict[str, Any]:
        """
        Process spatial input through the cognitive pipeline.

        Args:
            spatial_data: Input spatial data (geometry, attributes, metadata)
            context: Additional contextual information
            user_profile: User cognitive profile for personalized processing

        Returns:
            Dictionary containing cognitive processing results
        """
        start_time = datetime.now()

        try:
            # Step 1: Spatial perception and attention allocation
            perception_result = self.perception_model.process_spatial_input(
                spatial_data, context, user_profile
            )
            self.state.update_attention(perception_result.get("attention_weights", {}))
            self.performance_metrics["perception_updates"] += 1

            # Step 2: Working memory update
            for key, value in perception_result.items():
                if key.startswith("spatial_"):
                    self.state.add_to_working_memory(key, value, importance=0.8)

            # Step 3: Spatial reasoning and inference
            reasoning_result = self.reasoning_engine.reason_about_space(
                spatial_data, perception_result, self.state
            )
            self.performance_metrics["reasoning_chains"] += 1

            # Step 4: Memory consolidation and learning
            memory_result = self.memory_model.update_memory(
                perception_result, reasoning_result, self.state
            )
            self.performance_metrics["memory_operations"] += 1

            # Step 5: Decision support integration
            decision_result = self._generate_spatial_decisions(
                reasoning_result, memory_result, user_profile
            )
            self.performance_metrics["decisions_made"] += 1

            # Compile results
            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "processing_id": f"cog_{next(_PROCESSING_ID_SEQUENCE)}_{int(self._rng.integers(0, 1000))}",
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "cognitive_state": self.state.__dict__,
                "perception_result": perception_result,
                "reasoning_result": reasoning_result,
                "memory_result": memory_result,
                "decision_result": decision_result,
                "performance_metrics": self.performance_metrics.copy(),
            }

            logger.info(
                f"Spatial input processed successfully in {processing_time:.3f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Error in cognitive processing: {str(e)}")
            raise

    def _generate_spatial_decisions(
        self,
        reasoning_result: dict[str, Any],
        memory_result: dict[str, Any],
        user_profile: UserCognitiveProfile | None = None,
    ) -> dict[str, Any]:
        """
        Generate spatial decisions based on reasoning and memory.

        Args:
            reasoning_result: Results from spatial reasoning
            memory_result: Results from memory operations
            user_profile: User cognitive profile for personalization

        Returns:
            Dictionary containing decision recommendations
        """
        decisions = []

        # Analyze spatial alternatives if available
        alternatives = reasoning_result.get("spatial_alternatives", [])
        if alternatives:
            for alt in alternatives:
                decision = {
                    "alternative_id": alt.get("id"),
                    "spatial_location": alt.get("geometry"),
                    "cognitive_rationale": alt.get("reasoning_path", []),
                    "confidence_score": self._calculate_decision_confidence(
                        alt, user_profile
                    ),
                    "cognitive_load_impact": self._estimate_cognitive_load(alt),
                    "recommended_action": self._recommend_action(alt, reasoning_result),
                }
                decisions.append(decision)

        # Update overall decision confidence
        if decisions:
            confidences = [d["confidence_score"] for d in decisions]
            self.state.decision_confidence = np.mean(confidences)

        return {
            "decisions": decisions,
            "decision_strategy": "cognitive_weighted" if user_profile else "standard",
            "confidence_distribution": self._analyze_confidence_distribution(decisions),
            "cognitive_factors": self._extract_cognitive_factors(decisions),
        }

    def _calculate_decision_confidence(
        self,
        alternative: dict[str, Any],
        user_profile: UserCognitiveProfile | None = None,
    ) -> float:
        """Calculate confidence score for a spatial decision alternative."""
        base_confidence = alternative.get("confidence", 0.5)

        # Adjust based on cognitive load
        load_penalty = self.state.cognitive_load * 0.2
        confidence = max(0.1, base_confidence - load_penalty)

        # Adjust based on user expertise if available
        if user_profile:
            expertise_bonus = user_profile.spatial_expertise * 0.1
            confidence = min(0.95, confidence + expertise_bonus)

        return cast(float, confidence)

    def _estimate_cognitive_load(self, alternative: dict[str, Any]) -> float:
        """Estimate cognitive load impact of a spatial alternative."""
        # Simple heuristic based on spatial complexity
        geometry = alternative.get("geometry", {})
        if isinstance(geometry, dict):
            complexity_factors = [
                len(geometry.get("coordinates", [])) / 100.0,  # Coordinate complexity
                len(alternative.get("attributes", {})) / 10.0,  # Attribute complexity
                self.state.cognitive_load * 0.3,  # Current load influence
            ]
            return min(1.0, sum(complexity_factors) / 3.0)

        return 0.3  # Default moderate load

    def _recommend_action(
        self, alternative: dict[str, Any], reasoning_result: dict[str, Any]
    ) -> str:
        """Recommend action based on cognitive analysis."""
        confidence = alternative.get("confidence", 0.5)
        reasoning_strength = len(reasoning_result.get("reasoning_path", []))

        if confidence > 0.8 and reasoning_strength > 3:
            return "strong_recommendation"
        elif confidence > 0.6:
            return "moderate_recommendation"
        elif confidence > 0.4:
            return "weak_recommendation"
        else:
            return "requires_further_analysis"

    def _analyze_confidence_distribution(
        self, decisions: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Analyze the distribution of confidence scores across decisions."""
        if not decisions:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        confidences = [d.get("confidence_score", 0.0) for d in decisions]

        return {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        }

    def _extract_cognitive_factors(
        self, decisions: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        """Extract cognitive factors influencing decisions."""
        factors: dict[str, list[str]] = {
            "high_confidence_factors": [],
            "low_confidence_factors": [],
            "cognitive_load_issues": [],
        }

        for decision in decisions:
            confidence = decision.get("confidence_score", 0.5)
            load_impact = decision.get("cognitive_load_impact", 0.5)

            if confidence > 0.7:
                factors["high_confidence_factors"].extend(
                    decision.get("cognitive_rationale", [])
                )

            if confidence < 0.5:
                factors["low_confidence_factors"].append(
                    f"Low confidence in {decision.get('alternative_id', 'unknown')}"
                )

            if load_impact > 0.7:
                factors["cognitive_load_issues"].append(
                    f"High load for {decision.get('alternative_id', 'unknown')}"
                )

        return factors

    def update_cognitive_models(
        self, training_data: dict[str, Any], learning_rate: float = 0.01
    ) -> dict[str, Any]:
        """
        Update cognitive models based on new training data.

        Args:
            training_data: New spatial data and outcomes for model updating
            learning_rate: Learning rate for model parameter updates

        Returns:
            Dictionary containing model update results
        """
        update_results = {}

        try:
            # Update perception model
            perception_updates = self.perception_model.update_model(
                training_data, learning_rate
            )
            update_results["perception"] = perception_updates

            # Update reasoning engine
            reasoning_updates = self.reasoning_engine.update_model(
                training_data, learning_rate
            )
            update_results["reasoning"] = reasoning_updates

            # Update memory model
            memory_updates = self.memory_model.update_model(
                training_data, learning_rate
            )
            update_results["memory"] = memory_updates

            # Update performance metrics
            self.performance_metrics["model_updates"] = (
                self.performance_metrics.get("model_updates", 0) + 1
            )

            logger.info("Cognitive models updated successfully")
            return update_results

        except Exception as e:
            logger.error(f"Error updating cognitive models: {str(e)}")
            raise

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary of the cognitive engine."""
        return {
            "engine_status": "active",
            "cognitive_state": self.state.__dict__,
            "performance_metrics": self.performance_metrics,
            "model_status": {
                "perception_model": self.perception_model.get_status(),
                "reasoning_engine": self.reasoning_engine.get_status(),
                "memory_model": self.memory_model.get_status(),
            },
            "configuration": {
                "cognitive_framework": self.cognitive_framework,
                "spatial_resolution": self.spatial_resolution,
                "temporal_modeling": self.temporal_modeling,
                "uncertainty_handling": self.uncertainty_handling,
            },
        }

    def save_cognitive_state(self, filepath: str) -> None:
        """Save current cognitive state to file.

        Serialization contract: the state is JSON-serialized with
        ``default=str``, so non-JSON-native values (e.g. ``datetime``
        timestamps in working-memory items) are written as ISO-8601 strings
        and are NOT restored as their original types by
        :meth:`load_cognitive_state`. Persisted state is a snapshot for
        inspection/recovery, not a lossless round trip.
        """
        state_data = {
            "cognitive_state": self.state.__dict__,
            "performance_metrics": self.performance_metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        logger.info(f"Cognitive state saved to {filepath}")

    def load_cognitive_state(self, filepath: str) -> None:
        """Load cognitive state from file."""
        with open(filepath) as f:
            state_data = json.load(f)

        # Restore cognitive state
        state_dict = state_data["cognitive_state"]
        self.state = CognitiveState(**state_dict)

        # Restore performance metrics
        self.performance_metrics = state_data["performance_metrics"]

        logger.info(f"Cognitive state loaded from {filepath}")
