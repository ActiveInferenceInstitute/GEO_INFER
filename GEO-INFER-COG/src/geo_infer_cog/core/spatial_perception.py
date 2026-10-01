"""
Spatial Perception Models for GEO-INFER-COG

This module implements spatial perception models that simulate how humans
perceive, attend to, and interpret spatial information. The models are based
on cognitive psychology research and provide the foundation for human-centered
geospatial interfaces.

Key Components:
- Visual attention models for spatial focus
- Spatial relationship perception
- Scale and resolution awareness
- Uncertainty perception in spatial data
- Gestalt principles for spatial grouping

Mathematical Foundations:
- Feature integration theory (Treisman & Gelade, 1980)
- Attentional spotlight model (Posner, 1980)
- Bayesian models of spatial perception
- Signal detection theory for spatial discrimination
"""

import itertools
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..models.user_profiles import UserCognitiveProfile
from ..utils.rng import resolve_rng

# Module-level monotonic counter backing perception processing IDs.
_PERCEPTION_ID_SEQUENCE: "itertools.count[int]" = itertools.count(1)

logger = logging.getLogger(__name__)


@dataclass
class SpatialPercept:
    """Represents a perceived spatial element with cognitive properties."""

    element_id: str
    geometry: dict[str, Any]
    visual_saliency: float = 0.0
    attention_weight: float = 0.0
    perceptual_group: str = ""
    scale_level: str = "medium"
    uncertainty: float = 0.0
    accessibility: float = 1.0

    def calculate_attention_priority(self, task_context: str = "general") -> float:
        """Calculate attention priority based on task context."""
        base_priority = self.visual_saliency * self.accessibility

        # Adjust based on task context
        context_multipliers = {
            "navigation": 1.2 if self.scale_level in ["large", "medium"] else 0.8,
            "search": 1.3 if self.visual_saliency > 0.7 else 0.9,
            "analysis": 1.1 if self.uncertainty < 0.3 else 0.7,
            "planning": 1.0,  # Neutral for planning tasks
        }

        multiplier = context_multipliers.get(task_context, 1.0)
        return min(1.0, base_priority * multiplier)


class AttentionModel:
    """Models spatial attention allocation in human perception."""

    def __init__(self, attention_capacity: float = 1.0, focus_radius: float = 0.5):
        """
        Initialize attention model.

        Args:
            attention_capacity: Total attention capacity (0-1)
            focus_radius: Radius of attentional spotlight (0-1)
        """
        self.attention_capacity = attention_capacity
        self.focus_radius = focus_radius
        self.current_focus = None

    def allocate_attention(
        self, spatial_elements: list[SpatialPercept], task_priority: str = "balanced"
    ) -> dict[str, float]:
        """
        Allocate attention across spatial elements.

        Args:
            spatial_elements: List of spatial percepts to attend to
            task_priority: Priority strategy ('balanced', 'saliency', 'task_relevant')

        Returns:
            Dictionary mapping element IDs to attention weights
        """
        if not spatial_elements:
            return {}

        # Calculate attention priorities
        priorities = {}
        for element in spatial_elements:
            priority = element.calculate_attention_priority(task_priority)
            priorities[element.element_id] = priority

        # Normalize attention weights
        total_priority = sum(priorities.values())
        if total_priority == 0:
            return {
                elem.element_id: 1.0 / len(spatial_elements)
                for elem in spatial_elements
            }

        attention_weights = {
            elem_id: priority / total_priority * self.attention_capacity
            for elem_id, priority in priorities.items()
        }

        return attention_weights


class SpatialPerceptionModel:
    """
    Comprehensive spatial perception model for human-like spatial understanding.

    This model simulates how humans perceive spatial relationships, scales,
    and patterns in geospatial data, providing the foundation for intuitive
    geospatial interfaces.

    The model integrates:
    - Visual perception of spatial layout
    - Scale and resolution awareness
    - Spatial relationship understanding
    - Uncertainty perception and communication
    - Gestalt principles for perceptual grouping
    """

    def __init__(
        self,
        framework: str = "bayesian_attention",
        resolution: str = "adaptive",
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize spatial perception model.

        Args:
            framework: Perception framework ('bayesian_attention', 'gestalt', 'ecological')
            resolution: Resolution strategy ('adaptive', 'fixed', 'hierarchical')
            config: Additional configuration parameters
            rng: Optional random generator for processing-ID suffixes.
                When omitted, a fixed-seed generator is used so the model is
                deterministic by default.
        """
        self.framework = framework
        self.resolution = resolution
        self.config = config or {}

        # Resolved via the repo-wide resolve_rng pattern; None resolves to a
        # fixed seed so perception runs are reproducible by default.
        self._rng = resolve_rng(rng)

        self.attention_model = AttentionModel(
            attention_capacity=self.config.get("attention_capacity", 1.0),
            focus_radius=self.config.get("focus_radius", 0.5),
        )

        # Perceptual parameters
        self.perceptual_parameters = {
            "saliency_threshold": self.config.get("saliency_threshold", 0.3),
            "grouping_strength": self.config.get("grouping_strength", 0.7),
            "scale_sensitivity": self.config.get("scale_sensitivity", 0.8),
            "uncertainty_tolerance": self.config.get("uncertainty_tolerance", 0.4),
        }

        # Performance tracking
        self.perception_metrics = {
            "elements_processed": 0,
            "attention_allocations": 0,
            "grouping_operations": 0,
        }

        logger.info(f"Spatial Perception Model initialized with framework: {framework}")

    def process_spatial_input(
        self,
        spatial_data: dict[str, Any],
        context: dict[str, Any] | None = None,
        user_profile: UserCognitiveProfile | None = None,
    ) -> dict[str, Any]:
        """
        Process spatial input through perceptual modeling pipeline.

        Args:
            spatial_data: Input spatial data (geometries, attributes, metadata)
            context: Contextual information (task, environment, user state)
            user_profile: User cognitive profile for personalization

        Returns:
            Dictionary containing perceptual analysis results
        """
        start_time = datetime.now()

        try:
            # Step 1: Extract and analyze spatial elements
            spatial_elements = self._extract_spatial_elements(spatial_data)
            self.perception_metrics["elements_processed"] += len(spatial_elements)

            # Step 2: Calculate visual saliency for each element
            for element in spatial_elements:
                element.visual_saliency = self._calculate_visual_saliency(
                    element, spatial_data, context
                )

            # Step 3: Apply perceptual grouping (Gestalt principles)
            grouped_elements = self._apply_perceptual_grouping(spatial_elements)
            self.perception_metrics["grouping_operations"] += 1

            # Step 4: Allocate attention based on task and user profile
            task_context = context.get("task_type", "general") if context else "general"
            attention_weights = self.attention_model.allocate_attention(
                grouped_elements, task_context
            )
            self.perception_metrics["attention_allocations"] += 1

            # Step 5: Generate perceptual insights
            perceptual_insights = self._generate_perceptual_insights(
                grouped_elements, attention_weights, context, user_profile
            )

            # Compile results
            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "processing_id": f"percept_{next(_PERCEPTION_ID_SEQUENCE)}_{int(self._rng.integers(0, 1000))}",
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "spatial_elements": [elem.__dict__ for elem in spatial_elements],
                "grouped_elements": [elem.__dict__ for elem in grouped_elements],
                "attention_weights": attention_weights,
                "perceptual_insights": perceptual_insights,
                "perception_metrics": self.perception_metrics.copy(),
            }

            logger.info(
                f"Spatial perception processed successfully in {processing_time:.3f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Error in spatial perception processing: {str(e)}")
            raise

    def _extract_spatial_elements(
        self, spatial_data: dict[str, Any]
    ) -> list[SpatialPercept]:
        """Extract spatial elements from input data."""
        elements = []

        # Handle different spatial data formats
        geometries = spatial_data.get("geometries", [])
        if not geometries and "geometry" in spatial_data:
            geometries = [spatial_data["geometry"]]

        for i, geom in enumerate(geometries):
            element = SpatialPercept(
                element_id=f"spatial_elem_{i}",
                geometry=geom,
                scale_level=self._determine_scale_level(geom),
                uncertainty=self._calculate_element_uncertainty(geom),
            )
            elements.append(element)

        return elements

    def _calculate_visual_saliency(
        self,
        element: SpatialPercept,
        spatial_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Calculate visual saliency of a spatial element."""
        saliency_factors = []

        # Size-based saliency (larger elements are more salient)
        area = self._calculate_element_area(element.geometry)
        size_saliency = min(1.0, area / 1000.0)  # Normalize to reasonable range
        saliency_factors.append(size_saliency)

        # Color/contrast-based saliency (if available)
        if "visual_properties" in element.geometry:
            color_saliency = self._calculate_color_saliency(
                element.geometry["visual_properties"]
            )
            saliency_factors.append(color_saliency)

        # Context-based saliency (relevance to current task)
        context_saliency = self._calculate_context_saliency(element, context)
        saliency_factors.append(context_saliency)

        # Uncertainty-based saliency (novel or uncertain elements)
        uncertainty_saliency = 1.0 - element.uncertainty
        saliency_factors.append(uncertainty_saliency)

        # Combine factors with weights
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each factor
        combined_saliency = sum(s * w for s, w in zip(saliency_factors, weights))

        return min(1.0, max(0.0, combined_saliency))

    def _calculate_element_area(self, geometry: dict[str, Any]) -> float:
        """Calculate approximate area of a spatial element."""
        geom_type = geometry.get("type", "")

        if geom_type == "Point":
            return 1.0  # Points have minimal area
        elif geom_type == "LineString":
            # Approximate area based on line length
            coords = geometry.get("coordinates", [])
            if isinstance(coords[0], list):
                # Multi-line string
                total_length = sum(self._calculate_line_length(line) for line in coords)
            else:
                total_length = self._calculate_line_length(coords)
            return total_length / 10.0  # Normalize line length to area-like measure
        elif geom_type == "Polygon":
            # Simple polygon area calculation
            coords = geometry.get("coordinates", [[]])[0]
            return self._calculate_polygon_area(coords)
        else:
            return 10.0  # Default moderate area

    def _calculate_line_length(self, coordinates: list[list[float]]) -> float:
        """Calculate length of a line given its coordinates."""
        if len(coordinates) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            # Euclidean distance (assuming projected coordinates)
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length += distance

        return total_length

    def _calculate_polygon_area(self, coordinates: list[list[float]]) -> float:
        """Calculate area of a polygon using the shoelace formula."""
        if len(coordinates) < 3:
            return 0.0

        # Shoelace formula
        n = len(coordinates)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        area = abs(area) / 2.0

        return area

    def _calculate_color_saliency(self, visual_props: dict[str, Any]) -> float:
        """Calculate saliency based on visual properties."""
        # Simple color-based saliency (red/orange colors tend to be more salient)
        salient_colors = ["red", "orange", "yellow", "magenta"]
        color = visual_props.get("color", "").lower()

        if color in salient_colors:
            return 0.8
        else:
            return 0.4

    def _calculate_context_saliency(
        self, element: SpatialPercept, context: dict[str, Any] | None = None
    ) -> float:
        """Calculate saliency based on task context."""
        if not context:
            return 0.5

        task_type = context.get("task_type", "general")

        # Adjust saliency based on task relevance
        if task_type == "navigation" and element.scale_level in ["large", "medium"]:
            return 0.9
        elif task_type == "search" and element.visual_saliency > 0.6:
            return 0.8
        elif task_type == "analysis" and element.uncertainty < 0.3:
            return 0.7
        else:
            return 0.5

    def _determine_scale_level(self, geometry: dict[str, Any]) -> str:
        """Determine appropriate scale level for a spatial element."""
        area = self._calculate_element_area(geometry)

        if area < 10:
            return "small"
        elif area < 100:
            return "medium"
        else:
            return "large"

    def _calculate_element_uncertainty(self, geometry: dict[str, Any]) -> float:
        """Calculate uncertainty associated with a spatial element."""
        # Base uncertainty from geometry complexity
        base_uncertainty = min(0.5, len(geometry.get("coordinates", [])) / 100.0)

        # Add uncertainty from coordinate precision (if available)
        if "precision" in geometry:
            precision_uncertainty = 1.0 - (geometry["precision"] / 10.0)
            base_uncertainty = max(base_uncertainty, precision_uncertainty)

        return min(1.0, base_uncertainty)

    def _apply_perceptual_grouping(
        self, elements: list[SpatialPercept]
    ) -> list[SpatialPercept]:
        """Apply Gestalt principles for perceptual grouping."""
        if len(elements) < 2:
            return elements

        # Simple proximity-based grouping
        grouped_elements: list[SpatialPercept] = []
        used_indices = set()

        for i, element in enumerate(elements):
            if i in used_indices:
                continue

            # Find nearby elements for grouping
            nearby_elements = [element]
            used_indices.add(i)

            for j, other_element in enumerate(elements):
                if j in used_indices:
                    continue

                # Check proximity (simple distance-based)
                distance = self._calculate_spatial_distance(
                    element.geometry, other_element.geometry
                )
                if distance < 50.0:  # Threshold for grouping
                    nearby_elements.append(other_element)
                    used_indices.add(j)

            # Create group if multiple elements found
            if len(nearby_elements) > 1:
                group_id = f"group_{len(grouped_elements)}"
                for elem in nearby_elements:
                    elem.perceptual_group = group_id
                    elem.attention_weight = (
                        elem.attention_weight * 1.2
                    )  # Boost grouped elements

            grouped_elements.extend(nearby_elements)

        return grouped_elements

    def _calculate_spatial_distance(
        self, geom1: dict[str, Any], geom2: dict[str, Any]
    ) -> float:
        """Calculate spatial distance between two geometries."""
        # Simple centroid-based distance calculation
        centroid1 = self._calculate_centroid(geom1)
        centroid2 = self._calculate_centroid(geom2)

        if centroid1 and centroid2:
            x1, y1 = centroid1
            x2, y2 = centroid2
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return float("inf")

    def _calculate_centroid(
        self, geometry: dict[str, Any]
    ) -> tuple[float, float] | None:
        """Calculate centroid of a geometry."""
        geom_type = geometry.get("type", "")

        if geom_type == "Point":
            coords = geometry.get("coordinates", [])
            return (coords[0], coords[1]) if coords else None

        elif geom_type in ["LineString", "Polygon"]:
            coords = geometry.get("coordinates", [])
            if not coords:
                return None

            # Simple centroid calculation (average of all coordinates)
            all_x, all_y = [], []
            for coord in coords:
                if isinstance(coord[0], list):  # Handle nested coordinates
                    for sub_coord in coord:
                        all_x.append(sub_coord[0])
                        all_y.append(sub_coord[1])
                else:
                    all_x.append(coord[0])
                    all_y.append(coord[1])

            if all_x and all_y:
                return (sum(all_x) / len(all_x), sum(all_y) / len(all_y))

        return None

    def _generate_perceptual_insights(
        self,
        elements: list[SpatialPercept],
        attention_weights: dict[str, float],
        context: dict[str, Any] | None = None,
        user_profile: UserCognitiveProfile | None = None,
    ) -> dict[str, Any]:
        """Generate perceptual insights and recommendations."""
        insights: dict[str, Any] = {
            "attention_patterns": {},
            "perceptual_groups": {},
            "scale_distribution": {},
            "uncertainty_assessment": {},
            "user_specific_insights": {},
        }

        # Analyze attention patterns
        if attention_weights:
            max_attention = max(attention_weights.values())
            insights["attention_patterns"] = {
                "most_attended": max(
                    attention_weights, key=lambda k: attention_weights[k]
                ),
                "attention_concentration": max_attention,
                "attention_dispersion": len(
                    [w for w in attention_weights.values() if w > 0.1]
                ),
            }

        # Analyze perceptual groups
        groups: dict[str, list[str]] = {}
        for element in elements:
            group_id = element.perceptual_group or element.element_id
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(element.element_id)

        insights["perceptual_groups"] = {
            group_id: {"elements": elements, "size": len(elements)}
            for group_id, elements in groups.items()
        }

        # Analyze scale distribution
        scale_counts: dict[str, int] = {}
        for element in elements:
            scale = element.scale_level
            scale_counts[scale] = scale_counts.get(scale, 0) + 1

        insights["scale_distribution"] = scale_counts

        # Assess uncertainty
        uncertainties = [elem.uncertainty for elem in elements]
        if uncertainties:
            insights["uncertainty_assessment"] = {
                "mean_uncertainty": float(np.mean(uncertainties)),
                "max_uncertainty": float(np.max(uncertainties)),
                "uncertainty_range": float(
                    np.max(uncertainties) - np.min(uncertainties)
                ),
            }

        # Generate user-specific insights
        if user_profile:
            insights["user_specific_insights"] = self._generate_user_insights(
                elements, attention_weights, user_profile
            )

        return insights

    def _generate_user_insights(
        self,
        elements: list[SpatialPercept],
        attention_weights: dict[str, float],
        user_profile: UserCognitiveProfile,
    ) -> dict[str, Any]:
        """Generate insights tailored to user cognitive profile."""
        user_insights: dict[str, Any] = {}

        # Adjust recommendations based on user expertise
        if user_profile.spatial_expertise > 0.7:  # Expert user
            user_insights["expert_recommendations"] = [
                "Consider detailed spatial relationships",
                "Focus on subtle pattern variations",
                "Evaluate uncertainty sources systematically",
            ]
        else:  # Novice user
            user_insights["novice_guidance"] = [
                "Focus on prominent visual elements",
                "Use simplified spatial representations",
                "Seek clarification for uncertain areas",
            ]

        # Adjust based on user cognitive load preferences
        if user_profile.cognitive_load_preference == "low":
            high_load_elements = [
                elem.element_id
                for elem in elements
                if elem.calculate_attention_priority() > 0.8
            ]
            user_insights["load_management"] = (
                f"Consider {len(high_load_elements)} high-complexity elements"
            )

        return user_insights

    def update_model(
        self, training_data: dict[str, Any], learning_rate: float = 0.01
    ) -> dict[str, Any]:
        """Update perception model based on training data.

        Currently applies only ``perception_feedback.saliency_accuracy``
        (adjusting the saliency threshold). Other keys in ``training_data``
        are accepted for forward compatibility but not yet acted upon;
        ``parameters_updated`` in the result reports exactly which
        parameters changed.
        """
        update_results: dict[str, Any] = {
            "parameters_updated": [],
            "performance_improvement": 0.0,
            "training_examples": len(training_data.get("examples", [])),
        }

        # Update perceptual parameters based on feedback
        if "perception_feedback" in training_data:
            feedback = training_data["perception_feedback"]

            # Update saliency threshold
            if "saliency_accuracy" in feedback:
                current_threshold = self.perceptual_parameters["saliency_threshold"]
                target_threshold = 0.3 if feedback["saliency_accuracy"] > 0.8 else 0.5
                new_threshold = current_threshold + learning_rate * (
                    target_threshold - current_threshold
                )
                self.perceptual_parameters["saliency_threshold"] = new_threshold
                update_results["parameters_updated"].append("saliency_threshold")

        return update_results

    def get_status(self) -> dict[str, Any]:
        """Get current status of the perception model."""
        return {
            "model_type": "spatial_perception",
            "framework": self.framework,
            "status": "active",
            "perception_metrics": self.perception_metrics,
            "perceptual_parameters": self.perceptual_parameters,
        }
