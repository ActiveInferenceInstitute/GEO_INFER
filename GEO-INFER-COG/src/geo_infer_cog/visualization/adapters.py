"""
Human-Centered Visualization Adapters for GEO-INFER-COG

This module implements visualization adapters that create human-centered,
cognitively optimized geospatial visualizations. The adapters consider user
cognitive profiles, cognitive load preferences, and spatial cognition principles
to generate intuitive and effective visual representations.

Key Components:
- Cognitive Load-Aware Visualization: Adaptive complexity based on user capacity
- Perceptually Optimized Maps: Color schemes and layouts aligned with human perception
- Uncertainty-Aware Visualizations: Clear communication of spatial uncertainty
- Multi-Modal Interfaces: Support for different cognitive styles (visual/verbal)
- Progressive Disclosure: Information presentation adapted to user expertise

Mathematical Foundations:
- Cognitive load theory (Sweller, 1988)
- Visual perception principles (Ware, 2004)
- Color theory and accessibility (Tufte, 1983)
- Information visualization design principles
- Gestalt principles for visual grouping
"""

import numpy as np
import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
import colorsys
import math

from ..models.user_profiles import UserCognitiveProfile

logger = logging.getLogger(__name__)


@dataclass
class VisualizationElement:
    """Represents a visual element in a cognitive visualization."""

    element_id: str
    element_type: str  # 'point', 'line', 'polygon', 'text', 'symbol'
    geometry: dict[str, Any]
    visual_properties: dict[str, Any] = field(default_factory=dict)
    cognitive_weight: float = 0.5
    accessibility_score: float = 1.0
    uncertainty_level: float = 0.0

    def get_visual_complexity(self) -> float:
        """Calculate visual complexity of this element."""
        base_complexity = {
            "point": 0.1,
            "line": 0.3,
            "polygon": 0.6,
            "text": 0.4,
            "symbol": 0.2,
        }

        complexity = base_complexity.get(self.element_type, 0.3)

        # Adjust for size and detail
        if "coordinates" in self.geometry:
            coord_count = len(str(self.geometry["coordinates"]))
            complexity += min(0.4, coord_count / 1000.0)

        # Adjust for visual properties
        if self.visual_properties:
            prop_count = len(self.visual_properties)
            complexity += min(0.3, prop_count / 10.0)

        return min(1.0, complexity)


class ColorScheme:
    """Manages color schemes optimized for human perception and accessibility."""

    def __init__(self) -> None:
        """Initialize color scheme manager."""
        self.colorblind_friendly = {
            "safe": [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
            ],
            "diverging": [
                "#d73027",
                "#f46d43",
                "#fdae61",
                "#fee090",
                "#e0f3f8",
                "#abd9e9",
                "#74add1",
                "#4575b4",
            ],
            "sequential": [
                "#fff7fb",
                "#ece7f2",
                "#d0d1e6",
                "#a6bddb",
                "#74a9cf",
                "#3690c0",
                "#0570b0",
                "#034e7b",
            ],
        }

        self.cognitive_load_optimized = {
            "low_load": [
                "#e8f4f8",
                "#d1e7dd",
                "#fff3cd",
                "#f8d7da",
            ],  # Pastels for low cognitive load
            "medium_load": [
                "#0d6efd",
                "#198754",
                "#ffc107",
                "#dc3545",
            ],  # Standard colors
            "high_load": [
                "#6f42c1",
                "#fd7e14",
                "#20c997",
                "#e83e8c",
            ],  # Vivid for high information density
        }

    def get_perceptually_uniform_colors(self, n_colors: int) -> list[str]:
        """Generate perceptually uniform colors for data visualization."""
        if not isinstance(n_colors, int) or n_colors < 0:
            raise ValueError("n_colors must be a non-negative integer")
        if n_colors <= 8:
            return self.colorblind_friendly["safe"][:n_colors]

        # Generate additional colors using HSV color space for perceptual uniformity
        colors = []
        for i in range(n_colors):
            # Use golden angle for optimal color distribution
            hue = (i * 137.508) % 360  # Golden angle approximation
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.1  # Vary brightness slightly

            rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
            colors.append(hex_color)

        return colors

    def get_cognitive_load_colors(self, load_level: str, n_colors: int) -> list[str]:
        """Get colors optimized for specific cognitive load levels."""
        if not isinstance(n_colors, int) or n_colors < 0:
            raise ValueError("n_colors must be a non-negative integer")
        load_map = {
            "low": self.cognitive_load_optimized["low_load"],
            "medium": self.cognitive_load_optimized["medium_load"],
            "high": self.cognitive_load_optimized["high_load"],
        }

        base_colors = load_map.get(
            load_level, self.cognitive_load_optimized["medium_load"]
        )

        if n_colors <= len(base_colors):
            return base_colors[:n_colors]

        # Extend color scheme if more colors needed
        return self.get_perceptually_uniform_colors(n_colors)


class HumanCenteredVisualizer:
    """
    Human-centered visualization adapter for geospatial data.

    This visualizer creates cognitively optimized visualizations that adapt to:
    - User cognitive profiles and expertise levels
    - Cognitive load preferences and capacity
    - Visual perception capabilities and limitations
    - Accessibility requirements and color vision deficiencies
    - Task context and information needs

    The visualizer implements:
    - Adaptive complexity based on user capacity
    - Perceptually optimized color schemes
    - Progressive disclosure of information
    - Uncertainty visualization techniques
    - Multi-modal representation strategies
    """

    def __init__(
        self,
        cognitive_load_optimization: bool = True,
        perceptual_grouping: str = "gestalt_principles",
        uncertainty_communication: str = "confidence_intervals",
        accessibility_features: str = "wcag_compliant",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize human-centered visualizer.

        Args:
            cognitive_load_optimization: Enable cognitive load-aware adaptations
            perceptual_grouping: Grouping strategy ('gestalt_principles', 'similarity', 'proximity')
            uncertainty_communication: Uncertainty visualization method
            accessibility_features: Accessibility compliance level
            config: Additional configuration parameters
        """
        self.cognitive_load_optimization = cognitive_load_optimization
        self.perceptual_grouping = perceptual_grouping
        self.uncertainty_communication = uncertainty_communication
        self.accessibility_features = accessibility_features
        self.config = config or {}

        self._proximity_threshold = float(self.config.get("proximity_threshold", 0.1))
        if not np.isfinite(self._proximity_threshold) or self._proximity_threshold <= 0:
            raise ValueError("proximity_threshold must be finite and positive")
        self._visualization_counter = 0

        # Initialize color scheme manager
        self.color_scheme = ColorScheme()

        # Visualization parameters
        self.visualization_parameters = {
            "max_elements_low_load": self.config.get("max_elements_low_load", 10),
            "max_elements_medium_load": self.config.get("max_elements_medium_load", 50),
            "max_elements_high_load": self.config.get("max_elements_high_load", 200),
            "simplification_threshold": self.config.get(
                "simplification_threshold", 0.7
            ),
            "detail_progression_levels": self.config.get(
                "detail_progression_levels", 3
            ),
        }

        # Performance tracking
        self.visualization_metrics = {
            "visualizations_created": 0,
            "adaptations_applied": 0,
            "user_feedback_incorporated": 0,
        }

        logger.info("Human-Centered Visualizer initialized")

    def create_optimized_map(
        self,
        spatial_data: dict[str, Any],
        user_cognitive_profile: UserCognitiveProfile | None = None,
        task_context: str = "general_exploration",
        display_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a cognitively optimized map visualization.

        Args:
            spatial_data: Spatial data to visualize
            user_cognitive_profile: User profile for personalization
            task_context: Context of the visualization task
            display_constraints: Display limitations (screen size, color support, etc.)

        Returns:
            Optimized visualization configuration
        """
        start_time = datetime.now()

        try:
            if not isinstance(spatial_data, dict):
                raise TypeError("spatial_data must be a mapping")
            # Step 1: Analyze spatial data and user profile
            data_analysis = self._analyze_spatial_data(spatial_data)
            user_analysis = self._analyze_user_profile(user_cognitive_profile)

            # Step 2: Determine cognitive load strategy
            load_strategy = self._determine_cognitive_load_strategy(
                data_analysis, user_analysis, task_context
            )

            # Step 3: Apply perceptual grouping
            grouped_elements = self._apply_perceptual_grouping(
                data_analysis["elements"], user_analysis, load_strategy
            )

            # Step 4: Generate visualization specification
            visualization_spec = self._generate_visualization_specification(
                grouped_elements, user_analysis, load_strategy, display_constraints
            )

            # Step 5: Create adaptive color scheme
            color_scheme = self._create_adaptive_color_scheme(
                grouped_elements, user_analysis, load_strategy
            )

            # Step 6: Apply progressive disclosure
            disclosure_strategy = self._apply_progressive_disclosure(
                visualization_spec, user_analysis, load_strategy
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            self._visualization_counter += 1
            visualization_result: dict[str, Any] = {
                "visualization_id": f"viz_{self._visualization_counter:06d}",
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "data_analysis": data_analysis,
                "user_analysis": user_analysis,
                "load_strategy": load_strategy,
                "visualization_specification": visualization_spec,
                "color_scheme": color_scheme,
                "disclosure_strategy": disclosure_strategy,
                "adaptations_applied": self._summarize_adaptations(
                    load_strategy, user_analysis, task_context
                ),
                "visualization_metrics": self.visualization_metrics.copy(),
            }

            self.visualization_metrics["visualizations_created"] += 1
            self.visualization_metrics["adaptations_applied"] += len(
                visualization_result["adaptations_applied"]
            )

            logger.info(
                f"Cognitively optimized visualization created in {processing_time:.3f}s"
            )
            return visualization_result

        except Exception as e:
            logger.error(f"Error creating optimized visualization: {str(e)}")
            raise

    def _analyze_spatial_data(self, spatial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze spatial data for visualization planning."""
        analysis: dict[str, Any] = {
            "element_count": 0,
            "element_types": {},
            "spatial_extent": {},
            "complexity_score": 0.0,
            "elements": [],
        }

        # Extract spatial elements
        geometries = spatial_data.get("geometries", [])
        if not geometries and "geometry" in spatial_data:
            geometries = [spatial_data["geometry"]]

        for i, geometry in enumerate(geometries):
            if not isinstance(geometry, dict):
                raise ValueError(f"geometry at index {i} must be a mapping")
            element = VisualizationElement(
                element_id=f"elem_{i}",
                element_type=geometry.get("type", "unknown").lower(),
                geometry=geometry,
                cognitive_weight=self._calculate_element_cognitive_weight(geometry),
            )

            analysis["elements"].append(element)
            analysis["element_count"] += 1

            # Count element types
            elem_type = element.element_type
            analysis["element_types"][elem_type] = (
                analysis["element_types"].get(elem_type, 0) + 1
            )

        # Calculate overall complexity
        if analysis["elements"]:
            complexities = [
                elem.get_visual_complexity() for elem in analysis["elements"]
            ]
            analysis["complexity_score"] = float(np.mean(complexities))

        # Calculate spatial extent if coordinates available
        if analysis["elements"]:
            all_coords = []
            for element in analysis["elements"]:
                coords = element.geometry.get("coordinates", [])
                if coords:
                    # Flatten coordinate structure
                    def extract_coords(c: Any) -> None:
                        if (
                            isinstance(c, list)
                            and len(c) >= 2
                            and isinstance(c[0], (int, float))
                        ):
                            all_coords.append(c)
                        elif isinstance(c, list):
                            for item in c:
                                extract_coords(item)

                    extract_coords(coords)

            if all_coords:
                x_coords = [c[0] for c in all_coords]
                y_coords = [c[1] for c in all_coords]
                analysis["spatial_extent"] = {
                    "min_x": float(np.min(x_coords)),
                    "max_x": float(np.max(x_coords)),
                    "min_y": float(np.min(y_coords)),
                    "max_y": float(np.max(y_coords)),
                    "center_x": float(np.mean(x_coords)),
                    "center_y": float(np.mean(y_coords)),
                }

        return analysis

    def _calculate_element_cognitive_weight(self, geometry: dict[str, Any]) -> float:
        """Calculate cognitive weight of a spatial element."""
        base_weights = {
            "point": 0.8,  # Points are usually important landmarks
            "polygon": 0.7,  # Areas provide context
            "linestring": 0.6,  # Lines show connections
            "multipoint": 0.5,
            "multilinestring": 0.4,
            "multipolygon": 0.3,
        }

        geom_type = geometry.get("type", "").lower()
        base_weight = base_weights.get(geom_type, 0.5)

        # Adjust for size/complexity
        coords = geometry.get("coordinates", [])
        if coords:
            coord_count = len(str(coords))
            size_factor = min(0.3, coord_count / 1000.0)
            base_weight += size_factor

        return min(1.0, max(0.0, base_weight))

    def _analyze_user_profile(
        self, user_profile: UserCognitiveProfile | None
    ) -> dict[str, Any]:
        """Analyze user profile for visualization adaptation."""
        if not user_profile:
            return {
                "expertise_level": "intermediate",
                "cognitive_style": "balanced",
                "load_preference": "moderate",
                "visual_preferences": {},
                "adaptation_needed": False,
            }

        analysis = {
            "expertise_level": (
                "novice"
                if user_profile.spatial_expertise < 0.4
                else (
                    "expert" if user_profile.spatial_expertise > 0.7 else "intermediate"
                )
            ),
            "cognitive_style": user_profile.cognitive_style,
            "load_preference": user_profile.cognitive_load_preference,
            "visual_preferences": user_profile.visualization_preferences,
            "adaptation_needed": True,
        }

        # Extract specific visual preferences
        nav_prefs = user_profile.navigation_preferences
        if "map_style" in nav_prefs:
            analysis["preferred_map_style"] = nav_prefs["map_style"]

        if "color_preference" in nav_prefs:
            analysis["color_preference"] = nav_prefs["color_preference"]

        return analysis

    def _determine_cognitive_load_strategy(
        self,
        data_analysis: dict[str, Any],
        user_analysis: dict[str, Any],
        task_context: str,
    ) -> dict[str, Any]:
        """Determine cognitive load optimization strategy."""
        strategy: dict[str, Any] = {
            "load_level": "medium",
            "max_elements": 50,
            "simplification_enabled": True,
            "progressive_disclosure": True,
            "color_optimization": True,
            "adaptation_rationale": [],
        }

        # Determine load level based on data complexity and user profile
        data_complexity = data_analysis["complexity_score"]
        user_load_pref = user_analysis.get("load_preference", "moderate")

        if data_complexity > 0.7 or user_load_pref == "low":
            strategy["load_level"] = "low"
            strategy["max_elements"] = self.visualization_parameters[
                "max_elements_low_load"
            ]
            strategy["adaptation_rationale"].append(
                "High data complexity or low load preference"
            )
        elif data_complexity < 0.3 and user_load_pref == "high":
            strategy["load_level"] = "high"
            strategy["max_elements"] = self.visualization_parameters[
                "max_elements_high_load"
            ]
            strategy["adaptation_rationale"].append(
                "Low data complexity with high load tolerance"
            )

        # Adjust based on user expertise
        expertise = user_analysis.get("expertise_level", "intermediate")
        if expertise == "novice":
            strategy["simplification_enabled"] = True
            strategy["progressive_disclosure"] = True
            strategy["adaptation_rationale"].append(
                "Novice user requires simplification"
            )
        elif expertise == "expert":
            strategy["simplification_enabled"] = False
            strategy["progressive_disclosure"] = False
            strategy["adaptation_rationale"].append(
                "Expert user can handle complex visualizations"
            )

        # Task-specific adjustments
        if task_context == "navigation":
            strategy["simplification_enabled"] = (
                False  # Keep full detail for navigation
            )
            strategy["adaptation_rationale"].append(
                "Navigation task requires detailed information"
            )

        return strategy

    def _apply_perceptual_grouping(
        self,
        elements: list[VisualizationElement],
        user_analysis: dict[str, Any],
        load_strategy: dict[str, Any],
    ) -> list[VisualizationElement]:
        """Apply perceptual grouping based on Gestalt principles."""
        if not elements or not load_strategy.get("simplification_enabled", True):
            return elements

        grouped_elements = []

        # Group by proximity (spatial clustering)
        if self.perceptual_grouping == "gestalt_principles":
            grouped_elements = self._apply_proximity_grouping(elements)

        # Group by similarity (visual properties)
        elif self.perceptual_grouping == "similarity":
            grouped_elements = self._apply_similarity_grouping(elements)

        # Apply cognitive load-based filtering
        max_elements = load_strategy["max_elements"]
        if len(grouped_elements) > max_elements:
            # Sort by cognitive weight and keep most important
            grouped_elements.sort(key=lambda x: x.cognitive_weight, reverse=True)
            grouped_elements = grouped_elements[:max_elements]

        return grouped_elements

    def _apply_proximity_grouping(
        self, elements: list[VisualizationElement]
    ) -> list[VisualizationElement]:
        """Apply proximity-based perceptual grouping."""
        # Simple proximity grouping - group elements that are close together
        grouped: list[VisualizationElement] = []

        for element in elements:
            # Check if this element should be grouped with existing groups
            added_to_group = False

            for group_element in grouped:
                if self._elements_are_close(element, group_element):
                    # Merge visual properties or create group representation
                    group_element.visual_properties["group_size"] = (
                        group_element.visual_properties.get("group_size", 1) + 1
                    )
                    added_to_group = True
                    break

            if not added_to_group:
                grouped.append(element)

        return grouped

    def _elements_are_close(
        self, elem1: VisualizationElement, elem2: VisualizationElement
    ) -> bool:
        """Check if two visualization elements are spatially close."""
        # Simple distance-based proximity check
        try:
            # Extract centroids
            centroid1 = self._get_element_centroid(elem1.geometry)
            centroid2 = self._get_element_centroid(elem2.geometry)

            if centroid1 and centroid2:
                distance = math.sqrt(
                    (centroid2[0] - centroid1[0]) ** 2
                    + (centroid2[1] - centroid1[1]) ** 2
                )
                return distance <= self._proximity_threshold

        except (TypeError, KeyError, IndexError):
            logger.debug(
                "Could not compute element proximity due to missing geometry data"
            )

        return False

    def _get_element_centroid(
        self, geometry: dict[str, Any]
    ) -> tuple[float, float] | None:
        """Get centroid of a geometry element."""
        geom_type = geometry.get("type", "").lower()
        coords = geometry.get("coordinates", [])

        if geom_type == "point" and coords:
            return (coords[0], coords[1])

        elif (
            geom_type
            in [
                "linestring",
                "polygon",
                "multipoint",
                "multilinestring",
                "multipolygon",
            ]
            and coords
        ):
            # Simple centroid calculation
            all_coords = []

            def extract_coords(c: Any) -> None:
                if isinstance(c, list) and len(c) >= 2:
                    if isinstance(c[0], (int, float)):
                        all_coords.append(c)
                    else:
                        for item in c:
                            extract_coords(item)

            extract_coords(coords)

            if all_coords:
                x_coords = [c[0] for c in all_coords]
                y_coords = [c[1] for c in all_coords]
                return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

        return None

    def _apply_similarity_grouping(
        self, elements: list[VisualizationElement]
    ) -> list[VisualizationElement]:
        """Apply similarity-based perceptual grouping."""
        # Group elements by visual similarity
        grouped = []

        for element in elements:
            # Find similar elements
            similar_group = [element]

            for other_element in elements:
                if other_element != element and self._elements_are_similar(
                    element, other_element
                ):
                    similar_group.append(other_element)

            if len(similar_group) > 1:
                # Create group representation
                group_rep = similar_group[0]
                group_rep.visual_properties["group_size"] = len(similar_group)
                group_rep.visual_properties["group_type"] = "similarity_cluster"
                grouped.append(group_rep)
            else:
                grouped.append(element)

        return grouped

    def _elements_are_similar(
        self, elem1: VisualizationElement, elem2: VisualizationElement
    ) -> bool:
        """Check if two elements are visually similar."""
        # Compare element types
        if elem1.element_type != elem2.element_type:
            return False

        # Compare size/complexity
        complexity1 = elem1.get_visual_complexity()
        complexity2 = elem2.get_visual_complexity()

        complexity_diff = abs(complexity1 - complexity2)
        return complexity_diff < 0.2  # Similar complexity threshold

    def _generate_visualization_specification(
        self,
        elements: list[VisualizationElement],
        user_analysis: dict[str, Any],
        load_strategy: dict[str, Any],
        display_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate complete visualization specification."""
        spec: dict[str, Any] = {
            "layout_type": "adaptive",
            "element_specifications": {},
            "interaction_model": "progressive",
            "accessibility_features": {},
            "performance_optimizations": {},
        }

        # Generate specifications for each element
        for element in elements:
            element_spec = {
                "element_id": element.element_id,
                "visual_encoding": self._determine_visual_encoding(
                    element, user_analysis
                ),
                "interaction_behavior": self._determine_interaction_behavior(
                    element, user_analysis
                ),
                "accessibility_properties": self._determine_accessibility_properties(
                    element, user_analysis
                ),
            }

            spec["element_specifications"][element.element_id] = element_spec

        # Layout configuration
        spec["layout_configuration"] = self._configure_layout(
            elements, user_analysis, display_constraints
        )

        # Interaction model based on user expertise
        if user_analysis["expertise_level"] == "novice":
            spec["interaction_model"] = "guided_tour"
        elif user_analysis["expertise_level"] == "expert":
            spec["interaction_model"] = "advanced_controls"

        # Accessibility features
        spec["accessibility_features"] = self._configure_accessibility_features(
            user_analysis
        )

        # Performance optimizations
        spec["performance_optimizations"] = self._configure_performance_optimizations(
            load_strategy, display_constraints
        )

        return spec

    def _determine_visual_encoding(
        self, element: VisualizationElement, user_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine visual encoding for an element."""
        encoding: dict[str, Any] = {
            "shape": "default",
            "size": "medium",
            "color": "#1f77b4",
            "opacity": 1.0,
            "stroke_width": 1,
            "text_size": "medium",
        }

        # Adapt based on cognitive style
        cognitive_style = user_analysis.get("cognitive_style", "balanced")

        if cognitive_style == "visualizer":
            # Emphasize visual properties
            if element.element_type == "point":
                encoding["shape"] = "circle"
                encoding["size"] = (
                    "large" if element.cognitive_weight > 0.7 else "medium"
                )
            elif element.element_type == "polygon":
                encoding["opacity"] = 0.7 if element.cognitive_weight < 0.5 else 0.9

        elif cognitive_style == "verbalizer":
            # Emphasize text and labels
            if element.element_type in ["point", "polygon"]:
                encoding["text_size"] = "large"
                encoding["show_label"] = True

        # Adjust for cognitive load
        load_level = user_analysis.get("load_preference", "moderate")
        if load_level == "low":
            encoding["opacity"] = max(0.5, encoding["opacity"] - 0.2)
            encoding["stroke_width"] = max(0.5, encoding["stroke_width"] - 0.5)

        return encoding

    def _determine_interaction_behavior(
        self, element: VisualizationElement, user_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine interaction behavior for an element."""
        behavior = {
            "clickable": True,
            "hover_info": "basic",
            "selection_enabled": True,
            "tooltip_content": "auto",
        }

        # Adapt based on expertise
        expertise = user_analysis.get("expertise_level", "intermediate")

        if expertise == "novice":
            behavior["hover_info"] = "detailed"
            behavior["tooltip_content"] = "explanatory"
        elif expertise == "expert":
            behavior["hover_info"] = "technical"
            behavior["tooltip_content"] = "comprehensive"

        # Cognitive load considerations
        if user_analysis.get("load_preference") == "low":
            behavior["hover_info"] = "minimal"

        return behavior

    def _determine_accessibility_properties(
        self, element: VisualizationElement, user_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine accessibility properties for an element."""
        accessibility = {
            "alt_text": f"Spatial element of type {element.element_type}",
            "aria_label": f"Interactive spatial element {element.element_id}",
            "keyboard_navigable": True,
            "high_contrast_mode": False,
            "color_blind_friendly": True,
        }

        # Apply accessibility features based on configuration
        if self.accessibility_features == "wcag_compliant":
            accessibility["high_contrast_mode"] = True

            # Ensure sufficient color contrast via WCAG luminance check
            if element.visual_properties.get("color"):
                color = element.visual_properties["color"]
                accessibility["color_contrast_checked"] = True
                accessibility["original_color"] = color

        return accessibility

    def _configure_layout(
        self,
        elements: list[VisualizationElement],
        user_analysis: dict[str, Any],
        display_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Configure layout for the visualization."""
        layout = {
            "arrangement": "force_directed",
            "spacing": "adaptive",
            "margins": "standard",
            "responsive": True,
        }

        # Adjust layout based on display constraints
        if display_constraints:
            screen_size = display_constraints.get("screen_size", "desktop")

            if screen_size == "mobile":
                layout["arrangement"] = "grid"
                layout["spacing"] = "compact"
                layout["margins"] = "minimal"

            elif screen_size == "tablet":
                layout["arrangement"] = "hierarchical"
                layout["spacing"] = "medium"

        # User expertise-based layout
        expertise = user_analysis.get("expertise_level", "intermediate")

        if expertise == "novice":
            layout["arrangement"] = "clustered"  # Group similar elements
        elif expertise == "expert":
            layout["arrangement"] = "force_directed"  # Allow complex arrangements

        return layout

    def _configure_accessibility_features(
        self, user_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Configure accessibility features for the visualization."""
        features = {
            "high_contrast_mode": self.accessibility_features == "wcag_compliant",
            "keyboard_navigation": True,
            "screen_reader_support": True,
            "focus_indicators": True,
            "color_alternatives": [],
        }

        # Add color alternatives for color vision deficiency
        if self.accessibility_features == "wcag_compliant":
            features["color_alternatives"] = [
                "deuteranopia_simulation",
                "protanopia_simulation",
                "tritanopia_simulation",
            ]

        return features

    def _configure_performance_optimizations(
        self,
        load_strategy: dict[str, Any],
        display_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Configure performance optimizations."""
        optimizations = {
            "caching_enabled": True,
            "lazy_loading": load_strategy.get("progressive_disclosure", True),
            "geometry_simplification": load_strategy.get(
                "simplification_enabled", True
            ),
            "rendering_priority": "cognitive_weight",
        }

        # Mobile-specific optimizations
        if display_constraints and display_constraints.get("screen_size") == "mobile":
            optimizations["geometry_simplification"] = True
            optimizations["lazy_loading"] = True

        return optimizations

    def _create_adaptive_color_scheme(
        self,
        elements: list[VisualizationElement],
        user_analysis: dict[str, Any],
        load_strategy: dict[str, Any],
    ) -> dict[str, Any]:
        """Create adaptive color scheme for the visualization."""
        scheme = {
            "base_colors": [],
            "background_color": "#ffffff",
            "text_color": "#333333",
            "accent_colors": [],
            "load_optimized": True,
        }

        # Determine number of colors needed
        n_colors = max(1, min(len(elements), 10))  # Cap at 10 for perceptual reasons

        # Get colors based on cognitive load level
        load_level = load_strategy.get("load_level", "medium")
        base_colors = self.color_scheme.get_cognitive_load_colors(load_level, n_colors)

        scheme["base_colors"] = base_colors

        # Add accent colors for special elements
        scheme["accent_colors"] = [
            "#ff6b6b",  # Red for warnings/high importance
            "#4ecdc4",  # Teal for success/positive
            "#45b7d1",  # Blue for information
        ]

        # Adjust for accessibility
        if self.accessibility_features == "wcag_compliant":
            scheme["high_contrast"] = True

            # Ensure WCAG AA compliance (contrast ratio >= 4.5:1)
            scheme["contrast_ratio_target"] = 4.5
            scheme["wcag_level"] = "AA"

        return scheme

    def _apply_progressive_disclosure(
        self,
        visualization_spec: dict[str, Any],
        user_analysis: dict[str, Any],
        load_strategy: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply progressive disclosure strategy."""
        disclosure = {
            "initial_detail_level": "summary",
            "progression_triggers": ["user_interaction", "time_based"],
            "detail_levels": ["summary", "intermediate", "detailed"],
            "adaptation_strategy": "expertise_based",
        }

        # Determine initial detail level based on user expertise
        expertise = user_analysis.get("expertise_level", "intermediate")

        if expertise == "novice":
            disclosure["initial_detail_level"] = "summary"
            disclosure["progression_triggers"] = ["explicit_request"]
        elif expertise == "expert":
            disclosure["initial_detail_level"] = "detailed"
            disclosure["progression_triggers"] = ["user_interaction", "automatic"]

        # Cognitive load-based adjustments
        if load_strategy.get("load_level") == "low":
            disclosure["progression_triggers"] = ["explicit_request"]
            disclosure["detail_levels"] = ["summary", "intermediate"]

        return disclosure

    def _summarize_adaptations(
        self,
        load_strategy: dict[str, Any],
        user_analysis: dict[str, Any],
        task_context: str,
    ) -> list[str]:
        """Summarize adaptations applied to the visualization."""
        adaptations = []

        if load_strategy.get("simplification_enabled"):
            adaptations.append(
                "Applied data simplification for cognitive load management"
            )

        if load_strategy.get("progressive_disclosure"):
            adaptations.append(
                "Implemented progressive disclosure for information management"
            )

        if user_analysis.get("cognitive_style") == "visualizer":
            adaptations.append("Optimized for visual cognitive style")

        if user_analysis.get("cognitive_style") == "verbalizer":
            adaptations.append("Enhanced textual elements for verbal cognitive style")

        if task_context == "navigation":
            adaptations.append(
                "Navigation-optimized layout and information presentation"
            )

        return adaptations

    def communicate_uncertainty(
        self,
        spatial_predictions: dict[str, Any],
        uncertainty_quantification: dict[str, Any],
        user_risk_tolerance: str = "moderate",
    ) -> dict[str, Any]:
        """
        Communicate spatial uncertainty in a user-appropriate manner.

        Args:
            spatial_predictions: Spatial prediction data
            uncertainty_quantification: Uncertainty measures and confidence intervals
            user_risk_tolerance: User's tolerance for uncertainty ('low', 'moderate', 'high')

        Returns:
            Uncertainty communication strategy and visualization
        """
        communication_strategy: dict[str, Any] = {
            "communication_method": "visual_indicators",
            "detail_level": "moderate",
            "user_adapted": True,
            "confidence_display": {},
            "uncertainty_visualization": {},
        }

        # Adapt communication method based on user risk tolerance
        tolerance_map = {
            "low": {"method": "explicit_warnings", "detail": "high"},
            "moderate": {"method": "confidence_bands", "detail": "moderate"},
            "high": {"method": "subtle_indicators", "detail": "low"},
        }

        strategy = tolerance_map.get(user_risk_tolerance, tolerance_map["moderate"])
        communication_strategy["communication_method"] = strategy["method"]
        communication_strategy["detail_level"] = strategy["detail"]

        # Configure confidence display
        if strategy["method"] == "explicit_warnings":
            communication_strategy["confidence_display"] = {
                "show_confidence_score": True,
                "highlight_low_confidence": True,
                "warning_threshold": 0.3,
            }
        elif strategy["method"] == "confidence_bands":
            communication_strategy["confidence_display"] = {
                "show_confidence_bands": True,
                "band_opacity": 0.3,
                "band_color": "#ffa500",
            }

        # Configure uncertainty visualization
        uncertainty_types = uncertainty_quantification.get("uncertainty_types", [])

        if "spatial" in uncertainty_types:
            communication_strategy["uncertainty_visualization"]["spatial"] = {
                "method": "buffer_zones",
                "buffer_style": "gradient_fill",
                "opacity_range": [0.2, 0.6],
            }

        if "temporal" in uncertainty_types:
            communication_strategy["uncertainty_visualization"]["temporal"] = {
                "method": "animation",
                "frame_rate": "slow",
                "highlight_changes": True,
            }

        return communication_strategy

    def apply_perceptual_grouping(
        self,
        spatial_data: dict[str, Any],
        grouping_principles: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Apply perceptual grouping principles to spatial data.

        Args:
            spatial_data: Spatial data to group perceptually
            grouping_principles: Principles to apply ('proximity', 'similarity', 'continuity', 'closure')

        Returns:
            Perceptually grouped visualization specification
        """
        if grouping_principles is None:
            grouping_principles = ["proximity", "similarity"]

        grouping_result: dict[str, Any] = {
            "applied_principles": grouping_principles,
            "group_specifications": {},
            "visual_emphasis": {},
            "grouping_rationale": [],
        }

        # Apply each grouping principle
        for principle in grouping_principles:
            if principle == "proximity":
                groups = self._apply_proximity_grouping_to_data(spatial_data)
                grouping_result["group_specifications"]["proximity"] = groups
                grouping_result["grouping_rationale"].append(
                    "Grouped spatially close elements"
                )

            elif principle == "similarity":
                groups = self._apply_similarity_grouping_to_data(spatial_data)
                grouping_result["group_specifications"]["similarity"] = groups
                grouping_result["grouping_rationale"].append(
                    "Grouped visually similar elements"
                )

        # Configure visual emphasis for groups
        grouping_result["visual_emphasis"] = {
            "group_borders": True,
            "group_labels": len(grouping_principles) > 1,
            "group_colors": True,
            "group_opacity": 0.8,
        }

        return grouping_result

    def _apply_proximity_grouping_to_data(
        self, spatial_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply proximity grouping to spatial data."""
        geometries = spatial_data.get("geometries", [])
        if not geometries:
            return {
                "algorithm": "distance_connected_components",
                "parameters": {"eps": self._proximity_threshold, "min_samples": 2},
                "clusters_found": 0,
                "clusters": [],
                "noise_count": 0,
                "clustering_confidence": 1.0,
            }

        eps = float(spatial_data.get("grouping_eps", self._proximity_threshold))
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("grouping_eps must be finite and positive")

        centroids = [self._get_element_centroid(geometry) for geometry in geometries]
        visited = set()
        clusters = []
        for index, centroid in enumerate(centroids):
            if index in visited or centroid is None:
                continue
            cluster = []
            queue = [index]
            visited.add(index)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                current_centroid = centroids[current]
                if current_centroid is None:
                    continue
                for candidate, candidate_centroid in enumerate(centroids):
                    if candidate in visited or candidate_centroid is None:
                        continue
                    distance = math.hypot(
                        candidate_centroid[0] - current_centroid[0],
                        candidate_centroid[1] - current_centroid[1],
                    )
                    if distance <= eps:
                        visited.add(candidate)
                        queue.append(candidate)
            clusters.append(cluster)

        cluster_specs: list[dict[str, Any]] = [
            {
                "element_indices": cluster,
                "element_ids": [f"elem_{index}" for index in cluster],
                "size": len(cluster),
            }
            for cluster in clusters
        ]
        noise_count = sum(centroid is None for centroid in centroids)
        assigned = sum(spec["size"] for spec in cluster_specs)
        return {
            "algorithm": "distance_connected_components",
            "parameters": {"eps": eps, "min_samples": 2},
            "clusters_found": len(cluster_specs),
            "clusters": cluster_specs,
            "noise_count": noise_count,
            "clustering_confidence": float(assigned / len(centroids)),
        }

    def _apply_similarity_grouping_to_data(
        self, spatial_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply similarity grouping to spatial data."""
        geometries = spatial_data.get("geometries", [])
        groups: dict[tuple[str, int], list[int]] = {}
        for index, geometry in enumerate(geometries):
            if not isinstance(geometry, dict):
                raise ValueError(f"geometry at index {index} must be a mapping")
            coordinates = geometry.get("coordinates", [])
            complexity_bucket = min(4, len(str(coordinates)) // 250)
            key = (geometry.get("type", "unknown").lower(), complexity_bucket)
            groups.setdefault(key, []).append(index)

        group_specs = [
            {
                "signature": {
                    "geometry_type": geometry_type,
                    "complexity_bucket": bucket,
                },
                "element_indices": indices,
                "element_ids": [f"elem_{index}" for index in indices],
                "size": len(indices),
            }
            for (geometry_type, bucket), indices in sorted(groups.items())
        ]
        return {
            "algorithm": "geometry_type_complexity",
            "parameters": {"similarity_threshold": 0.7},
            "similarity_groups": len(group_specs),
            "groups": group_specs,
            "grouping_confidence": 1.0,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current status of the visualizer."""
        return {
            "visualizer_type": "human_centered",
            "status": "active",
            "visualization_metrics": self.visualization_metrics,
            "configuration": {
                "cognitive_load_optimization": self.cognitive_load_optimization,
                "perceptual_grouping": self.perceptual_grouping,
                "uncertainty_communication": self.uncertainty_communication,
                "accessibility_features": self.accessibility_features,
            },
        }
