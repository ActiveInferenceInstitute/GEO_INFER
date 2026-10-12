"""
Spatial Reasoning Engine for GEO-INFER-COG

This module implements computational spatial reasoning capabilities that
model human-like spatial inference and problem-solving. The engine supports
qualitative spatial reasoning, analogical reasoning, and deductive spatial
inference for geospatial decision-making.

Key Components:
- Qualitative spatial reasoning (RCC-8, direction relations)
- Spatial analogy and case-based reasoning
- Deductive spatial inference chains
- Spatial constraint satisfaction
- Bayesian spatial reasoning under uncertainty

Mathematical Foundations:
- Region Connection Calculus (RCC-8) for qualitative spatial relations
- Analogical reasoning frameworks (Gentner, 1983)
- Bayesian networks for spatial inference
- Constraint satisfaction problems for spatial optimization
"""

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..utils.rng import resolve_rng

# Module-level monotonic counter backing reasoning chain IDs.
_REASONING_ID_SEQUENCE: "itertools.count[int]" = itertools.count(1)

# Hard cap on candidate assignments evaluated by the constraint solver. When
# exceeded, the search degrades gracefully with a logged warning instead of
# running an exponential enumeration.
_MAX_EVALUATED_CONSTRAINT_ASSIGNMENTS = 1_000_000

logger = logging.getLogger(__name__)


@dataclass
class SpatialRelation:
    """Represents a qualitative spatial relationship between regions."""

    source_region: str
    target_region: str
    relation_type: str  # 'disconnected', 'externally_connected', 'equal', 'partially_overlapping', 'tangential_proper_part', 'non_tangential_proper_part', 'tangential_proper_part_inverse', 'non_tangential_proper_part_inverse'
    confidence: float = 1.0
    reasoning_path: list[str] = field(default_factory=list)

    def is_consistent_with(self, other_relation: "SpatialRelation") -> bool:
        """Check if this relation is consistent with another relation."""
        # RCC-8 consistency rules
        consistency_matrix = {
            "disconnected": {"disconnected", "externally_connected"},
            "externally_connected": {"externally_connected", "partially_overlapping"},
            "partially_overlapping": {
                "partially_overlapping",
                "tangential_proper_part",
                "non_tangential_proper_part",
            },
            "tangential_proper_part": {
                "tangential_proper_part",
                "non_tangential_proper_part",
            },
            "non_tangential_proper_part": {"non_tangential_proper_part"},
            "tangential_proper_part_inverse": {
                "tangential_proper_part_inverse",
                "non_tangential_proper_part_inverse",
            },
            "non_tangential_proper_part_inverse": {
                "non_tangential_proper_part_inverse"
            },
            "equal": {"equal"},
        }

        if (
            self.target_region == other_relation.source_region
            and self.source_region == other_relation.target_region
        ):
            # Inverse relationship check
            return other_relation.relation_type in consistency_matrix.get(
                self.relation_type, set()
            )

        return True  # No direct conflict


class ReasoningStep:
    """Represents a single step in a spatial reasoning chain."""

    def __init__(
        self,
        step_id: str,
        operation: str,
        input_premises: list[str],
        conclusion: str,
        confidence: float,
        explanation: str = "",
    ):
        self.step_id = step_id
        self.operation = operation
        self.input_premises = input_premises
        self.conclusion = conclusion
        self.confidence = confidence
        self.explanation = explanation
        self.timestamp = datetime.now()


class SpatialReasoningEngine:
    """
    Advanced spatial reasoning engine for human-like geospatial inference.

    This engine implements multiple reasoning strategies:
    - Qualitative spatial reasoning using RCC-8 calculus
    - Analogical reasoning for spatial problem-solving
    - Deductive reasoning chains for spatial inference
    - Constraint-based spatial optimization
    - Bayesian spatial reasoning under uncertainty

    The engine supports reasoning about:
    - Topological relationships between regions
    - Directional relationships (north, south, etc.)
    - Distance relationships (near, far, etc.)
    - Scale relationships (larger, smaller, etc.)
    - Temporal-spatial relationships
    """

    def __init__(
        self,
        reasoning_type: str = "qualitative_spatial",
        uncertainty_method: str = "probabilistic",
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize spatial reasoning engine.

        Args:
            reasoning_type: Type of reasoning ('qualitative_spatial', 'analogical', 'deductive', 'constraint_based')
            uncertainty_method: Uncertainty handling ('probabilistic', 'fuzzy', 'possibilistic')
            config: Additional configuration parameters
            rng: Optional random generator for chain-ID suffixes. When
                omitted, a fixed-seed generator is used so the engine is
                deterministic by default.
        """
        self.reasoning_type = reasoning_type
        self.uncertainty_method = uncertainty_method
        self.config = config or {}

        # Resolved via the repo-wide resolve_rng pattern; None resolves to a
        # fixed seed so reasoning runs are reproducible by default.
        self._rng = resolve_rng(rng)

        # Knowledge base for spatial relations and rules
        self.spatial_knowledge_base: dict[str, Any] = {
            "topological_relations": {},
            "directional_relations": {},
            "distance_relations": {},
            "analogical_cases": [],
            "deductive_rules": [],
        }

        # Reasoning state
        self.current_reasoning_chain: list[ReasoningStep] = []
        self.reasoning_cache: dict[Any, Any] = {}

        # Performance tracking
        self.reasoning_metrics = {
            "reasoning_chains_executed": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_chain_length": 0.0,
        }

        # Initialize reasoning components
        self._initialize_reasoning_components()

        logger.info(f"Spatial Reasoning Engine initialized with type: {reasoning_type}")

    def _initialize_reasoning_components(self) -> None:
        """Initialize reasoning components based on configuration."""
        # Load default spatial reasoning rules
        self._load_default_spatial_rules()

        # Initialize uncertainty handling
        if self.uncertainty_method == "probabilistic":
            self._initialize_bayesian_reasoning()
        elif self.uncertainty_method == "fuzzy":
            self._initialize_fuzzy_reasoning()

    def _load_default_spatial_rules(self) -> None:
        """Load default spatial reasoning rules and relations."""
        # RCC-8 topological relations
        self.spatial_knowledge_base["topological_relations"] = {
            "disconnected": {
                "description": "Regions do not touch or overlap",
                "transitivity": ["disconnected"],
                "inverse": "disconnected",
            },
            "externally_connected": {
                "description": "Regions touch at boundaries but do not overlap",
                "transitivity": ["externally_connected", "disconnected"],
                "inverse": "externally_connected",
            },
            "partially_overlapping": {
                "description": "Regions overlap partially",
                "transitivity": [
                    "partially_overlapping",
                    "tangential_proper_part",
                    "non_tangential_proper_part",
                ],
                "inverse": "partially_overlapping",
            },
            "tangential_proper_part": {
                "description": "One region is completely inside another, touching boundary",
                "transitivity": [
                    "tangential_proper_part",
                    "non_tangential_proper_part",
                ],
                "inverse": "tangential_proper_part_inverse",
            },
            "non_tangential_proper_part": {
                "description": "One region is completely inside another, not touching boundary",
                "transitivity": ["non_tangential_proper_part"],
                "inverse": "non_tangential_proper_part_inverse",
            },
            "equal": {
                "description": "Regions are identical",
                "transitivity": ["equal"],
                "inverse": "equal",
            },
        }

        # Directional relations
        self.spatial_knowledge_base["directional_relations"] = {
            "north": {"opposite": "south", "description": "North of reference region"},
            "south": {"opposite": "north", "description": "South of reference region"},
            "east": {"opposite": "west", "description": "East of reference region"},
            "west": {"opposite": "east", "description": "West of reference region"},
            "northeast": {
                "opposite": "southwest",
                "description": "Northeast of reference region",
            },
            "northwest": {
                "opposite": "southeast",
                "description": "Northwest of reference region",
            },
            "southeast": {
                "opposite": "northwest",
                "description": "Southeast of reference region",
            },
            "southwest": {
                "opposite": "northeast",
                "description": "Southwest of reference region",
            },
        }

        # Distance relations
        self.spatial_knowledge_base["distance_relations"] = {
            "adjacent": {"range": (0, 1), "description": "Immediately next to"},
            "near": {"range": (1, 10), "description": "Close to but not adjacent"},
            "moderate": {"range": (10, 50), "description": "Moderate distance"},
            "far": {"range": (50, float("inf")), "description": "Far from"},
            "very_far": {"range": (100, float("inf")), "description": "Very far from"},
        }

    def _initialize_bayesian_reasoning(self) -> None:
        """Initialize Bayesian reasoning components."""
        # Prior probabilities for spatial relations
        self.relation_priors = {
            "disconnected": 0.4,
            "externally_connected": 0.2,
            "partially_overlapping": 0.15,
            "tangential_proper_part": 0.1,
            "non_tangential_proper_part": 0.1,
            "equal": 0.05,
        }

    def _initialize_fuzzy_reasoning(self) -> None:
        """Initialize fuzzy reasoning components."""
        # Fuzzy membership functions for distance relations
        self.fuzzy_sets = {
            "adjacent": lambda d: max(0, 1 - d / 2),
            "near": lambda d: max(0, min(1, (10 - d) / 8, d / 2)),
            "moderate": lambda d: max(0, min(1, (d - 5) / 20, (50 - d) / 20)),
            "far": lambda d: min(1, d / 50) if d > 25 else 0,
        }

    def reason_about_space(
        self,
        spatial_data: dict[str, Any],
        perception_result: dict[str, Any],
        cognitive_state: Any,
    ) -> dict[str, Any]:
        """
        Perform spatial reasoning on input data.

        Args:
            spatial_data: Input spatial data for reasoning
            perception_result: Results from spatial perception processing
            cognitive_state: Current cognitive state

        Returns:
            Dictionary containing reasoning results and conclusions
        """
        start_time = datetime.now()

        try:
            # Initialize reasoning chain
            chain_id = f"reasoning_{next(_REASONING_ID_SEQUENCE)}_{int(self._rng.integers(0, 1000))}"

            # Step 1: Extract spatial premises from data and perception
            premises = self._extract_spatial_premises(spatial_data, perception_result)
            self._add_reasoning_step(
                chain_id,
                "premise_extraction",
                [],
                str(premises),
                1.0,
                "Extract spatial premises from input data",
            )

            # Step 2: Apply reasoning strategy based on type
            if self.reasoning_type == "qualitative_spatial":
                conclusions = self._qualitative_spatial_reasoning(premises, chain_id)
            elif self.reasoning_type == "analogical":
                conclusions = self._analogical_reasoning(premises, chain_id)
            elif self.reasoning_type == "deductive":
                conclusions = self._deductive_reasoning(premises, chain_id)
            elif self.reasoning_type == "constraint_based":
                conclusions = self._constraint_based_reasoning(premises, chain_id)
            else:
                conclusions = self._default_reasoning(premises, chain_id)

            # Step 3: Validate reasoning consistency
            validation_result = self._validate_reasoning_chain(conclusions, chain_id)

            # Step 4: Generate spatial alternatives and recommendations
            alternatives = self._generate_spatial_alternatives(
                conclusions, spatial_data
            )

            # Update metrics
            chain_length = len(self.current_reasoning_chain)
            self.reasoning_metrics["reasoning_chains_executed"] += 1
            self.reasoning_metrics["average_chain_length"] = (
                self.reasoning_metrics["average_chain_length"]
                * (self.reasoning_metrics["reasoning_chains_executed"] - 1)
                + chain_length
            ) / self.reasoning_metrics["reasoning_chains_executed"]

            if validation_result["valid"]:
                self.reasoning_metrics["successful_inferences"] += 1
            else:
                self.reasoning_metrics["failed_inferences"] += 1

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "reasoning_id": chain_id,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "reasoning_type": self.reasoning_type,
                "premises": premises,
                "conclusions": conclusions,
                "validation_result": validation_result,
                "spatial_alternatives": alternatives,
                "reasoning_chain": [
                    step.__dict__ for step in self.current_reasoning_chain
                ],
                "reasoning_metrics": self.reasoning_metrics.copy(),
                "confidence_score": self._calculate_overall_confidence(conclusions),
            }

            logger.info(
                f"Spatial reasoning completed in {processing_time:.3f}s with {len(conclusions)} conclusions"
            )
            return result

        except Exception as e:
            logger.error(f"Error in spatial reasoning: {str(e)}")
            raise

    def _extract_spatial_premises(
        self, spatial_data: dict[str, Any], perception_result: dict[str, Any]
    ) -> list[SpatialRelation]:
        """Extract spatial premises from input data.

        Geometric premises use RCC-8 relation types. Co-attended regions
        additionally produce auxiliary premises with relation_type
        ``'attention_connected'``; these are non-topological annotations and
        are not part of the RCC-8 knowledge base — downstream composition
        treats them as unknown relations by design.
        """
        premises = []

        # Extract from spatial data geometries
        geometries = spatial_data.get("geometries", [])
        if geometries:
            # Create relations between geometries
            for i, geom1 in enumerate(geometries):
                for geom2 in geometries[i + 1 :]:
                    relation = self._infer_spatial_relation(geom1, geom2)
                    if relation:
                        premises.append(relation)

        # Extract from perception results
        attention_weights = perception_result.get("attention_weights", {})
        if attention_weights:
            # Create attention-based premises
            attended_regions = list(attention_weights.keys())
            for i, region1 in enumerate(attended_regions):
                for region2 in attended_regions[i + 1 :]:
                    attention_relation = SpatialRelation(
                        source_region=region1,
                        target_region=region2,
                        relation_type="attention_connected",
                        confidence=min(
                            attention_weights.get(region1, 0),
                            attention_weights.get(region2, 0),
                        ),
                    )
                    premises.append(attention_relation)

        return premises

    def _region_id(self, geom: dict[str, Any]) -> str:
        """Derive a deterministic region id from geometry content.

        Uses a stable SHA-256 digest of the canonical JSON serialization so
        persisted reasoning results are reproducible across processes (the
        builtin ``hash`` of ``str`` is salted per process).
        """
        canonical = json.dumps(geom, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"region_{int(digest[:8], 16) % 10000}"

    def _infer_spatial_relation(
        self, geom1: dict[str, Any], geom2: dict[str, Any]
    ) -> SpatialRelation | None:
        """Infer spatial relation between two geometries."""
        # Simple relation inference based on bounding boxes
        bbox1 = self._calculate_bounding_box(geom1)
        bbox2 = self._calculate_bounding_box(geom2)

        if not bbox1 or not bbox2:
            return None

        # Check for topological relationships
        relation_type = self._determine_topological_relation(bbox1, bbox2)

        return SpatialRelation(
            source_region=self._region_id(geom1),
            target_region=self._region_id(geom2),
            relation_type=relation_type,
            confidence=0.8,  # Base confidence for geometric inference
        )

    def _calculate_bounding_box(
        self, geometry: dict[str, Any]
    ) -> tuple[float, float, float, float] | None:
        """Calculate bounding box for a geometry."""
        coords = geometry.get("coordinates", [])
        if not coords:
            return None

        # Flatten arbitrarily nested coordinate arrays (Point [x, y],
        # LineString [[x, y], ...], Polygon [[[x, y], ...], ...]) into a list
        # of [x, y] pairs.
        all_coords: list[Any] = []
        stack: list[Any] = [coords]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                if item and isinstance(item[0], (int, float)):
                    all_coords.append(item)
                else:
                    stack.extend(item)

        if not all_coords:
            return None

        x_coords = [c[0] for c in all_coords]
        y_coords = [c[1] for c in all_coords]

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def _determine_topological_relation(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> str:
        """Determine RCC-8 relation between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Check if boxes are equal
        if (
            x1_min == x2_min
            and y1_min == y2_min
            and x1_max == x2_max
            and y1_max == y2_max
        ):
            return "equal"

        # Check if one is completely inside the other (non-tangential proper part)
        if (
            x1_min >= x2_min
            and y1_min >= y2_min
            and x1_max <= x2_max
            and y1_max <= y2_max
        ):
            # Check if they touch boundaries
            touches_boundary = (
                x1_min == x2_min
                or y1_min == y2_min
                or x1_max == x2_max
                or y1_max == y2_max
            )
            return (
                "tangential_proper_part"
                if touches_boundary
                else "non_tangential_proper_part"
            )

        if (
            x2_min >= x1_min
            and y2_min >= y1_min
            and x2_max <= x1_max
            and y2_max <= y1_max
        ):
            touches_boundary = (
                x2_min == x1_min
                or y2_min == y1_min
                or x2_max == x1_max
                or y2_max == y1_max
            )
            return (
                "tangential_proper_part_inverse"
                if touches_boundary
                else "non_tangential_proper_part_inverse"
            )

        # Check for overlap
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        if overlap_x > 0 and overlap_y > 0:
            return "partially_overlapping"

        # Check for external connection (touching boundaries)
        if self._boxes_touch(bbox1, bbox2):
            return "externally_connected"

        # Default to disconnected
        return "disconnected"

    def _boxes_touch(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> bool:
        """Check if two bounding boxes touch at boundaries."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Check if boxes touch (adjacent but not overlapping)
        touches_x = (x1_max == x2_min or x1_min == x2_max) and not (
            y1_max < y2_min or y1_min > y2_max
        )
        touches_y = (y1_max == y2_min or y1_min == y2_max) and not (
            x1_max < x2_min or x1_min > x2_max
        )

        return touches_x or touches_y

    def _add_reasoning_step(
        self,
        chain_id: str,
        operation: str,
        input_premises: list[str],
        conclusion: str,
        confidence: float,
        explanation: str = "",
    ) -> None:
        """Add a step to the current reasoning chain."""
        step = ReasoningStep(
            step_id=f"{chain_id}_step_{len(self.current_reasoning_chain)}",
            operation=operation,
            input_premises=input_premises,
            conclusion=conclusion,
            confidence=confidence,
            explanation=explanation,
        )
        self.current_reasoning_chain.append(step)

    def _qualitative_spatial_reasoning(
        self, premises: list[SpatialRelation], chain_id: str
    ) -> list[SpatialRelation]:
        """Perform qualitative spatial reasoning using RCC-8."""
        conclusions = []

        # Apply RCC-8 composition rules
        for i, premise1 in enumerate(premises):
            for premise2 in premises[i + 1 :]:
                # Look for composition opportunities
                composition = self._compose_relations(premise1, premise2)
                if composition:
                    conclusions.append(composition)

        return conclusions

    def _compose_relations(
        self, rel1: SpatialRelation, rel2: SpatialRelation
    ) -> SpatialRelation | None:
        """Compose two spatial relations if possible."""
        # Simple composition: if rel1 connects A->B and rel2 connects B->C, infer A->C
        if rel1.target_region == rel2.source_region:
            # Compose the relations
            composed_type = self._infer_composed_relation(
                rel1.relation_type, rel2.relation_type
            )

            if composed_type:
                return SpatialRelation(
                    source_region=rel1.source_region,
                    target_region=rel2.target_region,
                    relation_type=composed_type,
                    confidence=min(rel1.confidence, rel2.confidence)
                    * 0.9,  # Slight confidence reduction
                    reasoning_path=[rel1.relation_type, rel2.relation_type],
                )

        return None

    def _infer_composed_relation(self, rel1_type: str, rel2_type: str) -> str | None:
        """Infer composed relation from two relation types."""
        # Simple composition rules for common cases
        composition_rules = {
            ("disconnected", "disconnected"): "disconnected",
            ("externally_connected", "disconnected"): "disconnected",
            ("partially_overlapping", "disconnected"): "disconnected",
            ("disconnected", "externally_connected"): "disconnected",
            ("externally_connected", "externally_connected"): "externally_connected",
            ("partially_overlapping", "externally_connected"): "partially_overlapping",
        }

        return composition_rules.get((rel1_type, rel2_type))

    def _analogical_reasoning(
        self, premises: list[SpatialRelation], chain_id: str
    ) -> list[SpatialRelation]:
        """Perform analogical reasoning on spatial premises."""
        conclusions = []

        # Find analogous situations in knowledge base
        for premise in premises:
            analogies = self._find_spatial_analogies(premise)
            for analogy in analogies:
                analogical_conclusion = self._apply_analogy(premise, analogy)
                if analogical_conclusion:
                    conclusions.append(analogical_conclusion)

        return conclusions

    def _find_spatial_analogies(self, premise: SpatialRelation) -> list[dict[str, Any]]:
        """Find analogous spatial relations in knowledge base."""
        # Simple analogy matching based on relation type
        analogies = []
        for case in self.spatial_knowledge_base["analogical_cases"]:
            if case.get("relation_type") == premise.relation_type:
                analogies.append(case)

        return analogies

    def _apply_analogy(
        self, premise: SpatialRelation, analogy: dict[str, Any]
    ) -> SpatialRelation | None:
        """Apply analogical mapping to generate conclusion."""
        # Create analogical conclusion
        return SpatialRelation(
            source_region=premise.source_region,
            target_region=premise.target_region,
            relation_type=analogy.get("inferred_relation", premise.relation_type),
            confidence=premise.confidence * 0.8,  # Reduced confidence for analogies
            reasoning_path=["analogy", analogy.get("case_id", "unknown")],
        )

    def _deductive_reasoning(
        self, premises: list[SpatialRelation], chain_id: str
    ) -> list[SpatialRelation]:
        """Perform deductive reasoning on spatial premises."""
        conclusions = []

        # Apply deductive rules from knowledge base
        for rule in self.spatial_knowledge_base["deductive_rules"]:
            applicable_premises = self._find_applicable_premises(premises, rule)
            if applicable_premises:
                conclusion = self._apply_deductive_rule(rule, applicable_premises)
                if conclusion:
                    conclusions.append(conclusion)

        return conclusions

    def _find_applicable_premises(
        self, premises: list[SpatialRelation], rule: dict[str, Any]
    ) -> list[SpatialRelation]:
        """Find premises that match a deductive rule."""
        required_relations = rule.get("required_relations", [])
        applicable = []

        for premise in premises:
            if premise.relation_type in required_relations:
                applicable.append(premise)

        return applicable

    def _apply_deductive_rule(
        self, rule: dict[str, Any], premises: list[SpatialRelation]
    ) -> SpatialRelation | None:
        """Apply a deductive rule to generate conclusion."""
        conclusion_type = rule.get("conclusion_relation")

        if conclusion_type:
            # Create conclusion based on first premise's regions
            if premises:
                return SpatialRelation(
                    source_region=premises[0].source_region,
                    target_region=premises[0].target_region,
                    relation_type=conclusion_type,
                    confidence=min(p.confidence for p in premises) * 0.95,
                    reasoning_path=["deduction", rule.get("rule_id", "unknown")],
                )

        return None

    def _constraint_based_reasoning(
        self, premises: list[SpatialRelation], chain_id: str
    ) -> list[SpatialRelation]:
        """Perform constraint-based spatial reasoning."""
        conclusions = []

        # Set up constraint satisfaction problem
        variables, domains, constraints = self._setup_spatial_constraints(premises)

        # Solve constraint satisfaction problem
        solutions = self._solve_spatial_constraints(variables, domains, constraints)

        # Generate conclusions from solutions. The same novel relation can
        # recur across many satisfying assignments; deduplicate so the
        # conclusion set stays proportional to the number of distinct novel
        # (source, target, relation) tuples rather than to the number of
        # solutions.
        seen_conclusions: set = set()
        for solution in solutions:
            conclusion = self._create_constraint_conclusion(solution, premises)
            if conclusion:
                key = (
                    conclusion.source_region,
                    conclusion.target_region,
                    conclusion.relation_type,
                )
                if key not in seen_conclusions:
                    seen_conclusions.add(key)
                    conclusions.append(conclusion)

        return conclusions

    def _setup_spatial_constraints(
        self, premises: list[SpatialRelation]
    ) -> tuple[list[str], dict[str, list[str]], list[dict[str, Any]]]:
        """Set up constraint satisfaction problem for spatial relations.

        Domain variables are restricted to premise pairs plus transitive
        neighbors: only pairs of regions in the same connected component of
        the premise graph get a domain variable. Pairs in different
        components cannot constrain each other, so enumerating them would
        only inflate the search space.
        """
        regions = set()
        for premise in premises:
            regions.add(premise.source_region)
            regions.add(premise.target_region)

        variables = sorted(regions)

        # Union-find over premise edges groups transitively related regions.
        parent = {region: region for region in variables}

        def find(region: str) -> str:
            while parent[region] != region:
                parent[region] = parent[parent[region]]
                region = parent[region]
            return region

        for premise in premises:
            parent[find(premise.source_region)] = find(premise.target_region)

        # Define domains (possible relations for each in-component pair)
        domains: dict[str, list[str]] = {}
        for i, var1 in enumerate(variables):
            for var2 in variables[i + 1 :]:
                if find(var1) == find(var2):
                    domains[f"{var1}_{var2}"] = list(
                        self.spatial_knowledge_base["topological_relations"].keys()
                    )

        # Add constraints from premises (accept either pair ordering)
        constraints = []
        for premise in premises:
            forward = f"{premise.source_region}_{premise.target_region}"
            reverse = f"{premise.target_region}_{premise.source_region}"
            var_pair = forward if forward in domains else reverse
            if var_pair not in domains:
                raise ValueError(
                    "spatial_reasoning: premise pair "
                    f"{premise.source_region}/{premise.target_region} has no "
                    "domain variable; constraint setup is inconsistent"
                )
            constraints.append(
                {
                    "type": "fixed_relation",
                    "variable": var_pair,
                    "value": premise.relation_type,
                }
            )

        return variables, domains, constraints

    def _solve_spatial_constraints(
        self,
        variables: list[str],
        domains: dict[str, list[str]],
        constraints: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Solve the spatial constraint satisfaction problem.

        Backtracking with incremental constraint checking: premise-fixed
        pairs are pinned to their required value, each partial assignment is
        checked as soon as a variable is assigned, and the total number of
        evaluated candidate values is hard-capped (with a logged degradation
        warning) instead of enumerating the full Cartesian product.
        """
        solutions: list[dict[str, str]] = []
        variable_pairs = list(domains.keys())
        if not variable_pairs:
            return solutions

        constraints_by_var: dict[str, list[dict[str, Any]]] = {}
        for constraint in constraints:
            constraints_by_var.setdefault(constraint["variable"], []).append(constraint)

        assignment: dict[str, str] = {}
        evaluated = 0
        degraded = False

        def backtrack(index: int) -> None:
            nonlocal evaluated, degraded
            if degraded:
                return
            if index == len(variable_pairs):
                solutions.append(dict(assignment))
                return
            var = variable_pairs[index]
            var_constraints = constraints_by_var.get(var, [])
            if var_constraints:
                # Fixed constraint: only the required value can succeed.
                candidate_values: list[str] = [var_constraints[0]["value"]]
                if candidate_values[0] not in domains[var]:
                    return  # Unsatisfiable pair: prune the branch.
            else:
                candidate_values = domains[var]
            for value in candidate_values:
                evaluated += 1
                if evaluated > _MAX_EVALUATED_CONSTRAINT_ASSIGNMENTS:
                    degraded = True
                    return
                assignment[var] = value
                if all(
                    assignment.get(c["variable"]) == c["value"] for c in var_constraints
                ):
                    backtrack(index + 1)
                assignment.pop(var, None)

        backtrack(0)

        if degraded:
            logger.warning(
                "spatial_reasoning: constraint search degraded after "
                f"evaluating {evaluated} candidate assignments "
                f"(cap {_MAX_EVALUATED_CONSTRAINT_ASSIGNMENTS}); returning "
                f"{len(solutions)} solutions found so far"
            )

        return solutions

    def _create_constraint_conclusion(
        self, solution: dict[str, str], premises: list[SpatialRelation]
    ) -> SpatialRelation | None:
        """Create conclusion from constraint solution."""
        # Find a novel relation in the solution
        for var_pair, relation in solution.items():
            # Check if this relation wasn't in original premises
            in_premises = any(
                p.source_region in var_pair and p.target_region in var_pair
                for p in premises
            )

            if not in_premises:
                regions = var_pair.split("_")
                if len(regions) == 2:
                    return SpatialRelation(
                        source_region=regions[0],
                        target_region=regions[1],
                        relation_type=relation,
                        confidence=0.7,  # Moderate confidence for inferred relations
                        reasoning_path=["constraint_satisfaction"],
                    )

        return None

    def _default_reasoning(
        self, premises: list[SpatialRelation], chain_id: str
    ) -> list[SpatialRelation]:
        """Default reasoning strategy when specific type not available."""
        conclusions = []

        # Simple consistency checking
        for premise in premises:
            if premise.confidence > 0.8:
                # High confidence premise becomes a conclusion
                conclusions.append(premise)

        return conclusions

    def _validate_reasoning_chain(
        self, conclusions: list[SpatialRelation], chain_id: str
    ) -> dict[str, Any]:
        """Validate the consistency of the reasoning chain."""
        validation_result: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "confidence": 1.0,
        }

        # Check for relation consistency
        for i, conc1 in enumerate(conclusions):
            for conc2 in conclusions[i + 1 :]:
                if not conc1.is_consistent_with(conc2):
                    validation_result["valid"] = False
                    validation_result["issues"].append(
                        f"Inconsistent relations: {conc1.relation_type} vs {conc2.relation_type}"
                    )

        # Calculate overall confidence
        if conclusions:
            confidences = [c.confidence for c in conclusions]
            validation_result["confidence"] = float(np.mean(confidences))

        return validation_result

    def _generate_spatial_alternatives(
        self, conclusions: list[SpatialRelation], spatial_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate spatial alternatives based on reasoning results."""
        alternatives = []

        # Create alternatives for uncertain relations
        for conclusion in conclusions:
            if conclusion.confidence < 0.7:
                # Generate alternative interpretations
                alternative_relations = self._get_alternative_relations(
                    conclusion.relation_type
                )

                for alt_relation in alternative_relations:
                    alternative = {
                        "id": f"alt_{conclusion.source_region}_{conclusion.target_region}_{alt_relation}",
                        "geometry": spatial_data.get("geometry", {}),
                        "original_relation": conclusion.relation_type,
                        "alternative_relation": alt_relation,
                        "confidence": conclusion.confidence * 0.8,
                        "reasoning_path": conclusion.reasoning_path
                        + ["alternative_consideration"],
                    }
                    alternatives.append(alternative)

        return alternatives

    def _get_alternative_relations(self, relation_type: str) -> list[str]:
        """Get alternative possible relations for a given relation type."""
        alternatives = []

        # Get similar relation types
        if relation_type in self.spatial_knowledge_base["topological_relations"]:
            # Return adjacent relations in RCC-8 conceptual neighborhood
            neighborhood = {
                "disconnected": ["externally_connected"],
                "externally_connected": ["disconnected", "partially_overlapping"],
                "partially_overlapping": [
                    "externally_connected",
                    "tangential_proper_part",
                ],
                "tangential_proper_part": [
                    "partially_overlapping",
                    "non_tangential_proper_part",
                ],
                "non_tangential_proper_part": ["tangential_proper_part"],
                "equal": [],
            }

            alternatives = neighborhood.get(relation_type, [])

        return alternatives

    def _calculate_overall_confidence(
        self, conclusions: list[SpatialRelation]
    ) -> float:
        """Calculate overall confidence of reasoning results."""
        if not conclusions:
            return 0.0

        confidences = [c.confidence for c in conclusions]

        # Weight by reasoning chain length
        chain_lengths = [len(c.reasoning_path) for c in conclusions]
        weights = [
            1.0 / (1.0 + length) for length in chain_lengths
        ]  # Prefer shorter chains

        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(
            weights
        )

        return float(weighted_confidence)

    def update_model(
        self, training_data: dict[str, Any], learning_rate: float = 0.01
    ) -> dict[str, Any]:
        """Update reasoning model based on training data."""
        update_results = {
            "rules_updated": 0,
            "knowledge_base_expanded": False,
            "performance_improvement": 0.0,
        }

        # Update analogical cases
        if "analogical_examples" in training_data:
            new_cases = training_data["analogical_examples"]
            self.spatial_knowledge_base["analogical_cases"].extend(new_cases)
            update_results["knowledge_base_expanded"] = True

        # Update deductive rules
        if "deductive_rules" in training_data:
            new_rules = training_data["deductive_rules"]
            self.spatial_knowledge_base["deductive_rules"].extend(new_rules)
            update_results["rules_updated"] = len(new_rules)

        return update_results

    def get_status(self) -> dict[str, Any]:
        """Get current status of the reasoning engine."""
        return {
            "engine_type": "spatial_reasoning",
            "reasoning_type": self.reasoning_type,
            "status": "active",
            "reasoning_metrics": self.reasoning_metrics,
            "knowledge_base_size": {
                "analogical_cases": len(
                    self.spatial_knowledge_base["analogical_cases"]
                ),
                "deductive_rules": len(self.spatial_knowledge_base["deductive_rules"]),
            },
        }
