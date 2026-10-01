"""Multi-level governance framework implementation."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Optional spatial integration
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    logger.warning("GEO-INFER-SPACE not available, spatial features disabled")


class GovernanceLevel(Enum):
    """Governance organizational levels."""

    LOCAL = "local"
    WATERSHED = "watershed"
    REGIONAL = "regional"
    NATIONAL = "national"
    INTERNATIONAL = "international"


class CoordinationMechanism(Enum):
    """Coordination mechanisms between governance levels."""

    VERTICAL_ALIGNMENT = "vertical_alignment"
    HORIZONTAL_INTEGRATION = "horizontal_integration"
    SUBSIDIARITY = "subsidiarity"
    NETWORKED_GOVERNANCE = "networked_governance"
    MARKET_BASED = "market_based"
    CONSENSUS_BUILDING = "consensus_building"


@dataclass
class GovernanceEntity:
    """Represents a governing body or organization."""

    entity_id: str
    name: str
    governance_level: GovernanceLevel
    jurisdiction: dict[str, Any]
    responsibilities: list[str]
    authority_domain: str
    stakeholders: list[str] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)
    capacity: float = 1.0


@dataclass
class GovernanceStructure:
    """Represents a complete governance structure."""

    governance_id: str
    spatial_scope: dict[str, Any]
    governance_levels: list[GovernanceLevel]
    entities: list[GovernanceEntity]
    coordination_mechanisms: list[CoordinationMechanism]
    decision_domains: list[str]
    stakeholder_groups: list[str]
    reporting_relationships: dict[str, list[str]] = field(default_factory=dict)
    information_flows: dict[str, list[str]] = field(default_factory=dict)
    decision_escalation_rules: dict[str, dict[str, str]] = field(default_factory=dict)


class MultiLevelGovernanceFramework:
    """
    Framework for designing and coordinating governance across multiple organizational levels.

    Implements vertical coordination (local-regional-national), horizontal coordination
    (cross-sectoral, cross-jurisdictional), subsidiarity principles, and nested governance
    structures for managing complex geospatial systems.

    References:
    - Benz et al. (2007). Multi-level Governance and Democracy
    - Piattoni, S. (2010). The Multi-level Governance of Regional Policy
    - Termeer, C. J., et al. (2010). Governance capabilities for dealing with conflict
    """

    def __init__(
        self,
        governance_levels: list[str] | None = None,
        coordination_mechanisms: list[str] | None = None,
        domain_coverage: list[str] | None = None,
    ):
        """
        Initialize multi-level governance framework.

        Parameters:
        -----------
        governance_levels : List[str]
            Levels of governance (local, regional, national, international)
        coordination_mechanisms : List[str]
            Coordination approaches between levels
        domain_coverage : List[str]
            Policy domains covered (environmental, civic, commercial, social)
        """
        # Validate governance levels against the GovernanceLevel enum
        default_levels = ["local", "regional", "national"]
        levels_to_use = governance_levels or default_levels

        self.governance_levels: list[GovernanceLevel] = []
        for level in levels_to_use:
            try:
                self.governance_levels.append(GovernanceLevel(level.lower()))
            except ValueError:
                valid = ", ".join(lvl.value for lvl in GovernanceLevel)
                raise ValueError(
                    f"Unknown governance level '{level}'. Valid levels are: {valid}"
                ) from None

        self.coordination_mechanisms = [
            CoordinationMechanism(mech) if isinstance(mech, str) else mech
            for mech in (
                coordination_mechanisms
                or ["vertical_alignment", "horizontal_integration"]
            )
        ]
        self.domain_coverage = domain_coverage or [
            "environmental",
            "civic",
            "commercial",
        ]
        self.governance_structures: dict[str, GovernanceStructure] = {}

    def design_governance_structure(
        self,
        spatial_scope: dict[str, Any],
        stakeholder_groups: list[dict[str, Any]],
        decision_domains: list[str],
        time_horizons: list[int],
    ) -> GovernanceStructure:
        """
        Design comprehensive multi-level governance structure.

        Parameters:
        -----------
        spatial_scope : Dict[str, Any]
            Geographic boundaries and spatial extent
        stakeholder_groups : List[Dict[str, Any]]
            Groups participating in governance
        decision_domains : List[str]
            Areas requiring governance decisions
        time_horizons : List[int]
            Planning horizons in years (e.g., [1, 5, 10, 20])

        Returns:
        --------
        GovernanceStructure
            Designed multi-level governance architecture
        """
        # Validate spatial scope
        spatial_validation = self._validate_spatial_scope(spatial_scope)
        if not spatial_validation.get("valid", True):
            logger.warning(
                f"Spatial scope validation issues: {spatial_validation.get('issues', [])}"
            )

        governance_id = f"mlg_{spatial_scope.get('name', 'region')}_{len(self.governance_structures)}"

        # Create governance entities for each level
        entities = self._create_governance_entities(
            stakeholder_groups=stakeholder_groups,
            decision_domains=decision_domains,
            spatial_scope=spatial_scope,
        )

        # Establish reporting relationships
        reporting_relationships = self._establish_reporting_relationships(entities)

        # Design information flows
        information_flows = self._design_information_flows(
            entities=entities, decision_domains=decision_domains
        )

        # Set decision escalation rules
        decision_escalation = self._set_decision_escalation_rules(
            entities=entities,
            decision_domains=decision_domains,
            time_horizons=time_horizons,
        )

        # Create governance structure
        structure = GovernanceStructure(
            governance_id=governance_id,
            spatial_scope=spatial_scope,
            governance_levels=self.governance_levels,
            entities=entities,
            coordination_mechanisms=self.coordination_mechanisms,
            decision_domains=decision_domains,
            stakeholder_groups=[
                sg.get("name", f"group_{i}") for i, sg in enumerate(stakeholder_groups)
            ],
            reporting_relationships=reporting_relationships,
            information_flows=information_flows,
            decision_escalation_rules=decision_escalation,
        )

        self.governance_structures[governance_id] = structure
        logger.info(f"Governance structure designed: {governance_id}")
        return structure

    def _validate_spatial_scope(self, spatial_scope: Any) -> dict[str, Any]:
        """
        Validate spatial scope using spatial indexing if available.

        Checks:
        - Required fields present
        - Spatial boundaries valid
        - Coordinate systems consistent
        - Area calculations reasonable
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        # Basic validation
        if not isinstance(spatial_scope, dict):
            validation_result["valid"] = False
            validation_result["issues"].append("Spatial scope must be a dictionary")
            return validation_result

        # Check for name
        if "name" not in spatial_scope:
            validation_result["warnings"].append("Spatial scope missing name field")

        # If SPACE module available, perform spatial validation
        if SPACE_AVAILABLE:
            try:
                spatial_indexer = SpatialIndexingInterface()

                # Check for coordinate data
                if "coordinates" in spatial_scope:
                    coords = spatial_scope["coordinates"]
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        lat, lng = coords[0], coords[1]
                        # Validate coordinate ranges
                        if not (-90 <= lat <= 90):
                            validation_result["issues"].append(
                                f"Invalid latitude: {lat}"
                            )
                            validation_result["valid"] = False
                        if not (-180 <= lng <= 180):
                            validation_result["issues"].append(
                                f"Invalid longitude: {lng}"
                            )
                            validation_result["valid"] = False

                        # Try to convert to spatial cell
                        try:
                            cell = spatial_indexer.latlng_to_cell(
                                lat, lng, resolution=9
                            )
                            validation_result["spatial_cell"] = cell
                        except Exception as e:
                            validation_result["warnings"].append(
                                f"Could not convert coordinates to spatial cell: {e}"
                            )

                # Check for polygon data
                if "polygon" in spatial_scope:
                    polygon = spatial_scope["polygon"]
                    try:
                        cells = spatial_indexer.polygon_to_cells(polygon, resolution=9)
                        validation_result["polygon_cells"] = len(cells)
                    except Exception as e:
                        validation_result["warnings"].append(
                            f"Could not process polygon: {e}"
                        )

            except Exception as e:
                logger.warning(f"Error in spatial validation: {e}")
                validation_result["warnings"].append(f"Spatial validation error: {e}")

        # Validate area if present
        if "area_km2" in spatial_scope:
            area = spatial_scope["area_km2"]
            if not isinstance(area, (int, float)) or area <= 0:
                validation_result["issues"].append(f"Invalid area: {area}")
                validation_result["valid"] = False
            elif area > 1e8:  # Very large area (larger than most countries)
                validation_result["warnings"].append(
                    f"Unusually large area: {area} km²"
                )

        return validation_result

    def calculate_performance_metrics(
        self, governance_structure: GovernanceStructure
    ) -> dict[str, Any]:
        """
        Calculate performance metrics for a governance structure.

        Metrics include:
        - Structural efficiency
        - Coordination effectiveness
        - Resource utilization
        - Stakeholder coverage
        - Decision domain coverage

        Returns:
        --------
        Dict[str, Any]
            Performance metrics dictionary
        """
        metrics: dict[str, Any] = {}

        # 1. Structural efficiency
        num_entities = len(governance_structure.entities)
        num_levels = len(governance_structure.governance_levels)
        num_domains = len(governance_structure.decision_domains)

        # Efficiency: balance between entities and levels
        if num_levels > 0:
            entities_per_level = num_entities / num_levels
            # Optimal is around 1-2 entities per level
            structural_efficiency = 1.0 - abs(entities_per_level - 1.5) / 1.5
            structural_efficiency = max(0.0, min(1.0, structural_efficiency))
        else:
            structural_efficiency = 0.0

        metrics["structural_efficiency"] = structural_efficiency

        # 2. Coordination effectiveness
        num_coordination_mechanisms = len(governance_structure.coordination_mechanisms)
        num_information_flows = sum(
            len(flows) for flows in governance_structure.information_flows.values()
        )

        # More coordination mechanisms and information flows generally indicate better coordination
        coordination_score = min(
            1.0, (num_coordination_mechanisms * 0.3 + num_information_flows * 0.1)
        )
        metrics["coordination_effectiveness"] = coordination_score

        # 3. Resource utilization
        total_budget = sum(
            e.resources.get("budget", 0) for e in governance_structure.entities
        )
        avg_capacity = (
            sum(e.capacity for e in governance_structure.entities) / num_entities
            if num_entities > 0
            else 0.0
        )

        resource_utilization = avg_capacity * min(
            1.0, total_budget / 1000000
        )  # Normalize by 1M
        metrics["resource_utilization"] = resource_utilization

        # 4. Stakeholder coverage
        num_stakeholder_groups = len(governance_structure.stakeholder_groups)
        total_stakeholders = sum(
            len(e.stakeholders) for e in governance_structure.entities
        )

        # Coverage is good if stakeholders are distributed across entities
        if num_stakeholder_groups > 0:
            stakeholder_coverage = min(
                1.0, total_stakeholders / (num_stakeholder_groups * num_entities)
            )
        else:
            stakeholder_coverage = 0.0

        metrics["stakeholder_coverage"] = stakeholder_coverage

        # 5. Decision domain coverage
        # Check if entities have responsibilities covering all domains
        all_entity_domains = set()
        for entity in governance_structure.entities:
            all_entity_domains.update(entity.responsibilities)

        if num_domains > 0:
            domain_coverage = (
                len(all_entity_domains & set(governance_structure.decision_domains))
                / num_domains
            )
        else:
            domain_coverage = 0.0

        metrics["decision_domain_coverage"] = domain_coverage

        # 6. Overall performance score
        weights = {
            "structural_efficiency": 0.20,
            "coordination_effectiveness": 0.25,
            "resource_utilization": 0.15,
            "stakeholder_coverage": 0.20,
            "decision_domain_coverage": 0.20,
        }

        overall_score = sum(weights[key] * metrics[key] for key in weights.keys())

        metrics["overall_performance_score"] = overall_score
        metrics["performance_rating"] = (
            "excellent"
            if overall_score >= 0.8
            else "good"
            if overall_score >= 0.6
            else "fair"
            if overall_score >= 0.4
            else "poor"
        )

        return metrics

    def _create_governance_entities(
        self,
        stakeholder_groups: list[dict[str, Any]],
        decision_domains: list[str],
        spatial_scope: dict[str, Any],
    ) -> list[GovernanceEntity]:
        """Create governance entities for each level."""
        entities = []
        level_idx = 0

        for level in self.governance_levels:
            entity_id = f"entity_{level.value}_{spatial_scope.get('name', 'region')}"

            # Allocate stakeholders and domains to this level
            allocated_stakeholders = [
                sg.get("name", f"stakeholder_{i}")
                for i, sg in enumerate(stakeholder_groups)
                if i % len(self.governance_levels) == level_idx
            ]

            allocated_domains = [
                domain
                for i, domain in enumerate(decision_domains)
                if i % len(self.governance_levels) == level_idx
            ]

            entity = GovernanceEntity(
                entity_id=entity_id,
                name=f"{level.value.capitalize()} Authority",
                governance_level=level,
                jurisdiction=spatial_scope,
                responsibilities=allocated_domains,
                authority_domain=", ".join(allocated_domains) or "general",
                stakeholders=allocated_stakeholders,
                resources={
                    "budget": 1000000.0 * (len(self.governance_levels) - level_idx)
                },
                capacity=0.7 + (0.1 * level_idx),
            )

            entities.append(entity)
            level_idx += 1

        return entities

    def _establish_reporting_relationships(
        self, entities: list[GovernanceEntity]
    ) -> dict[str, list[str]]:
        """Establish hierarchical reporting relationships."""
        reporting = {}

        for i, entity in enumerate(entities):
            # Lower levels report to higher levels (local → regional → national)
            if i > 0:
                reporting[entity.entity_id] = [entities[i - 1].entity_id]
            else:
                reporting[entity.entity_id] = []

        return reporting

    def _design_information_flows(
        self, entities: list[GovernanceEntity], decision_domains: list[str]
    ) -> dict[str, list[str]]:
        """Design information flows between governance levels."""
        flows = {}

        for entity in entities:
            # Both upward (monitoring) and downward (directives) flows
            target_entities = [
                e.entity_id for e in entities if e.entity_id != entity.entity_id
            ]
            flows[entity.entity_id] = target_entities

        return flows

    def _set_decision_escalation_rules(
        self,
        entities: list[GovernanceEntity],
        decision_domains: list[str],
        time_horizons: list[int],
    ) -> dict[str, dict[str, str]]:
        """Set rules for escalating decisions to higher governance levels."""
        escalation_rules = {}

        for domain in decision_domains:
            escalation_rules[domain] = {
                "implementation_level": "local",
                "coordination_level": "regional",
                "strategic_level": "national",
                "escalation_condition": "cross_jurisdictional_impact",
                "escalation_time_horizon": str(min(time_horizons)),
            }

        return escalation_rules

    def coordinate_vertical_levels(
        self, governance_structure: GovernanceStructure, policy_proposal: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Coordinate decision-making across vertical governance levels.

        Parameters:
        -----------
        governance_structure : GovernanceStructure
            Multi-level governance architecture
        policy_proposal : Dict[str, Any]
            Policy proposal requiring coordination

        Returns:
        --------
        Dict[str, Any]
            Coordination outcomes with approval status at each level
        """
        coordination_result = {
            "proposal_id": policy_proposal.get("id", "proposal_unknown"),
            "level_approvals": {},
            "cross_level_conflicts": [],
            "coordinated_implementation": {},
        }

        # Evaluate proposal at each governance level
        for entity in governance_structure.entities:
            approval = self._evaluate_at_level(
                entity=entity,
                policy_proposal=policy_proposal,
                coordination_mechanisms=governance_structure.coordination_mechanisms,
            )

            coordination_result["level_approvals"][entity.entity_id] = approval

        # Identify cross-level conflicts
        conflicts = self._identify_conflicts(
            level_approvals=coordination_result["level_approvals"],
            entities=governance_structure.entities,
        )
        coordination_result["cross_level_conflicts"] = conflicts

        # Determine coordinated implementation
        implementation = self._determine_implementation(
            level_approvals=coordination_result["level_approvals"], conflicts=conflicts
        )
        coordination_result["coordinated_implementation"] = implementation

        return coordination_result

    def _evaluate_at_level(
        self,
        entity: GovernanceEntity,
        policy_proposal: dict[str, Any],
        coordination_mechanisms: list[CoordinationMechanism],
    ) -> dict[str, Any]:
        """
        Evaluate policy proposal at specific governance level using multi-criteria decision analysis.

        Uses weighted scoring based on:
        - Entity capacity and resources
        - Stakeholder alignment
        - Policy complexity and scope
        - Resource requirements
        - Coordination mechanism compatibility

        References:
        - Keeney, R. L., & Raiffa, H. (1993). Decisions with Multiple Objectives
        - Saaty, T. L. (1980). The Analytic Hierarchy Process
        """
        # Calculate capacity score (0-1)
        capacity_score = min(1.0, entity.capacity)

        # Calculate resource availability score
        required_budget = policy_proposal.get("budget_required", 0)
        available_budget = entity.resources.get("budget", 0)
        resource_score = (
            min(1.0, available_budget / required_budget) if required_budget > 0 else 1.0
        )

        # Calculate stakeholder consensus (based on number of stakeholders and their alignment)
        # Simulate stakeholder alignment based on entity capacity and policy characteristics
        base_consensus = 0.5 + (capacity_score * 0.3)
        # Adjust for policy complexity
        policy_complexity = policy_proposal.get("complexity", 0.5)
        stakeholder_consensus = base_consensus * (1.0 - policy_complexity * 0.2)
        stakeholder_consensus = max(0.0, min(1.0, stakeholder_consensus))

        # Calculate coordination mechanism alignment
        mechanism_alignment = self._calculate_mechanism_alignment(
            entity, policy_proposal, coordination_mechanisms
        )

        # Calculate domain relevance score
        domain_relevance = self._calculate_domain_relevance(entity, policy_proposal)

        # Weighted decision score
        weights = {
            "capacity": 0.25,
            "resources": 0.20,
            "stakeholder_consensus": 0.25,
            "mechanism_alignment": 0.15,
            "domain_relevance": 0.15,
        }

        decision_score = (
            weights["capacity"] * capacity_score
            + weights["resources"] * resource_score
            + weights["stakeholder_consensus"] * stakeholder_consensus
            + weights["mechanism_alignment"] * mechanism_alignment
            + weights["domain_relevance"] * domain_relevance
        )

        # Determine approval status based on decision score
        if decision_score >= 0.75:
            approval_status = "approved"
        elif decision_score >= 0.50:
            approval_status = "conditional"
        else:
            approval_status = "rejected"

        return {
            "entity_id": entity.entity_id,
            "level": entity.governance_level.value,
            "approval_status": approval_status,
            "decision_score": decision_score,
            "stakeholder_consensus": stakeholder_consensus,
            "resource_availability": resource_score >= 0.5,
            "resource_score": resource_score,
            "capacity_score": capacity_score,
            "coordination_mechanisms_aligned": mechanism_alignment >= 0.5,
            "mechanism_alignment_score": mechanism_alignment,
            "domain_relevance_score": domain_relevance,
            "confidence": min(1.0, decision_score * 1.2),  # Confidence in decision
        }

    def _calculate_mechanism_alignment(
        self,
        entity: GovernanceEntity,
        policy_proposal: dict[str, Any],
        coordination_mechanisms: list[CoordinationMechanism],
    ) -> float:
        """Calculate alignment between coordination mechanisms and policy requirements."""
        if not coordination_mechanisms:
            return 0.5  # Neutral if no mechanisms specified

        # Policy characteristics that affect mechanism choice
        policy_scope = policy_proposal.get("scope", "local")
        requires_consensus = policy_proposal.get("requires_consensus", False)

        alignment_scores = []

        for mechanism in coordination_mechanisms:
            if mechanism == CoordinationMechanism.VERTICAL_ALIGNMENT:
                # Good for hierarchical coordination
                score = 0.8 if policy_scope in ["regional", "national"] else 0.5
            elif mechanism == CoordinationMechanism.HORIZONTAL_INTEGRATION:
                # Good for cross-sectoral coordination
                score = 0.9 if policy_scope == "cross_sectoral" else 0.6
            elif mechanism == CoordinationMechanism.SUBSIDIARITY:
                # Good for local decisions
                score = 0.9 if policy_scope == "local" else 0.5
            elif mechanism == CoordinationMechanism.CONSENSUS_BUILDING:
                # Good when consensus is required
                score = 0.9 if requires_consensus else 0.5
            elif mechanism == CoordinationMechanism.NETWORKED_GOVERNANCE:
                # Good for complex, multi-actor situations
                score = 0.8 if len(entity.stakeholders) > 3 else 0.5
            else:
                score = 0.5  # Default neutral score

            alignment_scores.append(score)

        return (
            sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
        )

    def _calculate_domain_relevance(
        self, entity: GovernanceEntity, policy_proposal: dict[str, Any]
    ) -> float:
        """Calculate how relevant the policy is to the entity's decision domains."""
        policy_domains = policy_proposal.get("domains", [])
        if not policy_domains:
            return 0.5  # Neutral if no domains specified

        # Check overlap between policy domains and entity responsibilities
        entity_domains = set(entity.responsibilities)
        policy_domains_set = set(policy_domains)

        if not entity_domains:
            return 0.3  # Low relevance if entity has no specific domains

        overlap = len(entity_domains & policy_domains_set)
        total_policy_domains = len(policy_domains_set)

        if total_policy_domains == 0:
            return 0.5

        relevance = overlap / total_policy_domains
        return min(1.0, relevance)

    def _identify_conflicts(
        self,
        level_approvals: dict[str, dict[str, Any]],
        entities: list[GovernanceEntity],
    ) -> list[dict[str, Any]]:
        """
        Identify conflicts between governance levels using comprehensive conflict analysis.

        Analyzes:
        - Approval status inconsistencies
        - Decision score disparities
        - Resource allocation conflicts
        - Stakeholder consensus mismatches
        - Jurisdictional overlaps

        References:
        - Axelrod, R. (1984). The Evolution of Cooperation
        - Ostrom, E. (1990). Governing the Commons
        """
        conflicts: list[dict[str, Any]] = []

        if not level_approvals or len(level_approvals) < 2:
            return conflicts

        # Collect approval data
        approval_data: list[dict[str, Any]] = []
        for entity_id, approval in level_approvals.items():
            entity = next((e for e in entities if e.entity_id == entity_id), None)
            if entity:
                approval_data.append(
                    {"entity_id": entity_id, "entity": entity, "approval": approval}
                )

        # 1. Approval status consistency conflict
        statuses = [
            str(a["approval"]["approval_status"])
            if isinstance(a["approval"], dict) and "approval_status" in a["approval"]
            else "unknown"
            for a in approval_data
        ]
        unique_statuses = set(statuses)
        if len(unique_statuses) > 1:
            # Calculate conflict severity based on status disparity
            status_weights = {"approved": 3, "conditional": 2, "rejected": 1}
            status_scores = [status_weights.get(s, 1) for s in statuses]
            max_score = max(status_scores)
            min_score = min(status_scores)
            severity_score = (max_score - min_score) / max_score if max_score > 0 else 0

            conflicts.append(
                {
                    "type": "approval_status_conflict",
                    "description": f"Inconsistent approval status across levels: {', '.join(unique_statuses)}",
                    "severity": "high"
                    if severity_score > 0.5
                    else "medium"
                    if severity_score > 0.2
                    else "low",
                    "severity_score": severity_score,
                    "affected_entities": [a["entity_id"] for a in approval_data],
                    "status_distribution": {
                        s: statuses.count(s) for s in unique_statuses
                    },
                }
            )

        # 2. Decision score disparity conflict
        decision_scores = [
            float(a["approval"].get("decision_score", 0.5))
            if isinstance(a["approval"], dict)
            else 0.5
            for a in approval_data
        ]
        if decision_scores:
            score_variance = self._calculate_variance(decision_scores)
            score_range = max(decision_scores) - min(decision_scores)

            if score_range > 0.3:  # Significant disparity threshold
                conflicts.append(
                    {
                        "type": "decision_score_disparity",
                        "description": f"Large disparity in decision scores across levels (range: {score_range:.2f})",
                        "severity": "high" if score_range > 0.5 else "medium",
                        "severity_score": min(1.0, score_range),
                        "affected_entities": [a["entity_id"] for a in approval_data],
                        "score_range": score_range,
                        "score_variance": score_variance,
                    }
                )

        # 3. Resource allocation conflict
        resource_scores = [
            float(a["approval"].get("resource_score", 1.0))
            if isinstance(a["approval"], dict)
            else 1.0
            for a in approval_data
        ]
        resource_conflicts = [
            i for i, score in enumerate(resource_scores) if score < 0.5
        ]
        if resource_conflicts:
            conflicts.append(
                {
                    "type": "resource_allocation_conflict",
                    "description": f"Insufficient resources at {len(resource_conflicts)} governance level(s)",
                    "severity": "high"
                    if len(resource_conflicts) > len(approval_data) / 2
                    else "medium",
                    "severity_score": len(resource_conflicts) / len(approval_data),
                    "affected_entities": [
                        approval_data[i]["entity_id"] for i in resource_conflicts
                    ],
                    "resource_shortfall": sum(
                        0.5 - score for score in resource_scores if score < 0.5
                    ),
                }
            )

        # 4. Stakeholder consensus mismatch
        consensus_scores = [
            float(a["approval"].get("stakeholder_consensus", 0.5))
            if isinstance(a["approval"], dict)
            else 0.5
            for a in approval_data
        ]
        if consensus_scores:
            consensus_range = max(consensus_scores) - min(consensus_scores)
            if consensus_range > 0.4:  # Significant consensus gap
                conflicts.append(
                    {
                        "type": "stakeholder_consensus_mismatch",
                        "description": f"Large gap in stakeholder consensus across levels (range: {consensus_range:.2f})",
                        "severity": "medium",
                        "severity_score": min(1.0, consensus_range),
                        "affected_entities": [a["entity_id"] for a in approval_data],
                        "consensus_range": consensus_range,
                    }
                )

        # 5. Jurisdictional overlap conflict (if spatial data available)
        if SPACE_AVAILABLE:
            spatial_conflicts = self._identify_spatial_conflicts(entities)
            conflicts.extend(spatial_conflicts)

        return conflicts

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values."""
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _identify_spatial_conflicts(
        self, entities: list[GovernanceEntity]
    ) -> list[dict[str, Any]]:
        """Identify spatial/jurisdictional conflicts using spatial analysis."""
        conflicts = []

        try:
            # Extract spatial boundaries from entities
            entity_boundaries: list[dict[str, Any]] = []
            for entity in entities:
                jurisdiction = entity.jurisdiction
                if isinstance(jurisdiction, dict):
                    # Check for spatial data in jurisdiction
                    if (
                        "bounds" in jurisdiction
                        or "polygon" in jurisdiction
                        or "coordinates" in jurisdiction
                    ):
                        entity_boundaries.append(
                            {"entity_id": entity.entity_id, "boundary": jurisdiction}
                        )

            # Check for overlapping jurisdictions
            if len(entity_boundaries) > 1:
                for i, boundary1 in enumerate(entity_boundaries):
                    for boundary2 in entity_boundaries[i + 1 :]:
                        # Simple overlap detection (can be enhanced with actual spatial operations)
                        b1 = boundary1["boundary"]
                        b2 = boundary2["boundary"]
                        if (
                            isinstance(b1, dict)
                            and isinstance(b2, dict)
                            and self._check_boundary_overlap(b1, b2)
                        ):
                            conflicts.append(
                                {
                                    "type": "jurisdictional_overlap",
                                    "description": f"Overlapping jurisdictions between {boundary1['entity_id']} and {boundary2['entity_id']}",
                                    "severity": "medium",
                                    "severity_score": 0.6,
                                    "affected_entities": [
                                        boundary1["entity_id"],
                                        boundary2["entity_id"],
                                    ],
                                }
                            )
        except Exception as e:
            logger.warning(f"Error in spatial conflict detection: {e}")

        return conflicts

    def _check_boundary_overlap(
        self, boundary1: dict[str, Any], boundary2: dict[str, Any]
    ) -> bool:
        """Check if two boundaries overlap (simplified implementation)."""
        # This is a simplified check - in production, use proper spatial operations
        # Check for common keys that might indicate overlap
        if "name" in boundary1 and "name" in boundary2:
            # If they have the same name or similar area, might overlap
            if boundary1.get("name") == boundary2.get("name"):
                return True

        # Check area overlap (if both have area_km2)
        if "area_km2" in boundary1 and "area_km2" in boundary2:
            area1 = boundary1["area_km2"]
            area2 = boundary2["area_km2"]
            # If areas are very similar, might indicate overlap
            if abs(area1 - area2) / max(area1, area2) < 0.1:
                return True

        return False

    def _determine_implementation(
        self,
        level_approvals: dict[str, dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Determine coordinated implementation strategy using conflict analysis.

        Considers:
        - Approval statuses across levels
        - Conflict severity and types
        - Resource availability
        - Stakeholder consensus levels

        Returns implementation strategy with recommendations.
        """
        if not level_approvals:
            return {
                "implementation_possible": False,
                "required_modifications": 0,
                "recommended_approach": "not_applicable",
                "reason": "No approval data available",
            }

        # Calculate overall approval status
        approval_statuses = [a.get("approval_status") for a in level_approvals.values()]
        all_approved = all(status == "approved" for status in approval_statuses)
        any_rejected = any(status == "rejected" for status in approval_statuses)

        # Calculate average decision score
        decision_scores = [
            a.get("decision_score", 0.0) for a in level_approvals.values()
        ]
        avg_decision_score = (
            sum(decision_scores) / len(decision_scores) if decision_scores else 0.0
        )

        # Analyze conflicts
        high_severity_conflicts = [c for c in conflicts if c.get("severity") == "high"]
        medium_severity_conflicts = [
            c for c in conflicts if c.get("severity") == "medium"
        ]
        total_conflict_severity = sum(c.get("severity_score", 0.0) for c in conflicts)

        # Determine implementation feasibility
        if any_rejected:
            implementation_possible = False
            recommended_approach = "rejected"
            reason = "One or more levels rejected the proposal"
        elif all_approved and len(conflicts) == 0:
            implementation_possible = True
            recommended_approach = "immediate"
            reason = "All levels approved with no conflicts"
        elif all_approved and len(high_severity_conflicts) == 0:
            implementation_possible = True
            recommended_approach = "phased"
            reason = "All levels approved but minor conflicts need resolution"
        elif avg_decision_score >= 0.6 and len(high_severity_conflicts) == 0:
            implementation_possible = True
            recommended_approach = "phased_with_conditions"
            reason = "Moderate approval with manageable conflicts"
        elif avg_decision_score >= 0.5:
            implementation_possible = True
            recommended_approach = "conditional_phased"
            reason = "Conditional approval requires conflict resolution"
        else:
            implementation_possible = False
            recommended_approach = "not_recommended"
            reason = "Low approval scores and/or high conflict severity"

        # Generate recommendations
        recommendations = []
        if high_severity_conflicts:
            recommendations.append(
                "Resolve high-severity conflicts before implementation"
            )
        if medium_severity_conflicts:
            recommendations.append(
                "Address medium-severity conflicts during phased rollout"
            )
        if any(a.get("resource_score", 1.0) < 0.5 for a in level_approvals.values()):
            recommendations.append("Secure additional resources for affected levels")
        if any(
            a.get("stakeholder_consensus", 1.0) < 0.6 for a in level_approvals.values()
        ):
            recommendations.append(
                "Improve stakeholder engagement and consensus building"
            )

        return {
            "implementation_possible": implementation_possible,
            "required_modifications": len(conflicts),
            "recommended_approach": recommended_approach,
            "reason": reason,
            "average_decision_score": avg_decision_score,
            "conflict_summary": {
                "total_conflicts": len(conflicts),
                "high_severity": len(high_severity_conflicts),
                "medium_severity": len(medium_severity_conflicts),
                "total_severity_score": total_conflict_severity,
            },
            "recommendations": recommendations,
            "implementation_timeline": self._estimate_timeline(
                recommended_approach, len(conflicts), avg_decision_score
            ),
        }

    def _estimate_timeline(
        self, approach: str, num_conflicts: int, avg_score: float
    ) -> dict[str, Any]:
        """Estimate implementation timeline based on approach and conflicts."""
        base_timeline: dict[str, dict[str, Any]] = {
            "immediate": {"weeks": 2, "months": 0.5},
            "phased": {"weeks": 8, "months": 2},
            "phased_with_conditions": {"weeks": 12, "months": 3},
            "conditional_phased": {"weeks": 16, "months": 4},
            "not_recommended": {"weeks": 0, "months": 0},
            "rejected": {"weeks": 0, "months": 0},
        }

        base = base_timeline.get(approach, {"weeks": 8, "months": 2})
        timeline: dict[str, Any] = {"weeks": base["weeks"], "months": base["months"]}

        # Adjust for conflicts
        conflict_adjustment = num_conflicts * 2  # 2 weeks per conflict
        timeline["weeks"] = int(timeline["weeks"]) + conflict_adjustment
        timeline["months"] = round(float(timeline["weeks"]) / 4.33, 1)

        # Adjust for decision score (lower score = more time needed)
        if avg_score < 0.6:
            timeline["weeks"] = int(timeline["weeks"]) + 4
            timeline["months"] = round(float(timeline["weeks"]) / 4.33, 1)

        return timeline

    def apply_subsidiarity_principle(
        self, governance_structure: GovernanceStructure, decision_domain: str
    ) -> dict[str, Any]:
        """
        Apply subsidiarity principle to determine appropriate governance level for decision.

        Subsidiarity suggests decisions should be made at the most local level possible,
        escalating only when necessary for coordination or broader impacts.

        Parameters:
        -----------
        governance_structure : GovernanceStructure
            Multi-level governance architecture
        decision_domain : str
            Domain requiring governance decision

        Returns:
        --------
        Dict[str, Any]
            Subsidiarity analysis with recommended decision level
        """
        # Find entities responsible for this domain
        relevant_entities = [
            e
            for e in governance_structure.entities
            if decision_domain in e.responsibilities
        ]

        if not relevant_entities:
            relevant_entities = governance_structure.entities

        # Sort by governance level (local first)
        relevant_entities.sort(
            key=lambda e: list(GovernanceLevel).index(e.governance_level)
        )

        result = {
            "decision_domain": decision_domain,
            "subsidiary_level": relevant_entities[0].governance_level.value
            if relevant_entities
            else None,
            "relevant_entities": [e.entity_id for e in relevant_entities],
            "escalation_needed": len(relevant_entities) > 1,
            "coordination_requirements": self._determine_coordination_requirements(
                relevant_entities
            ),
        }

        return result

    def _determine_coordination_requirements(
        self, entities: list[GovernanceEntity]
    ) -> list[str]:
        """
        Determine coordination requirements based on entity characteristics.

        Analyzes:
        - Number of entities involved
        - Governance level differences
        - Resource distribution
        - Stakeholder overlap
        - Decision domain complexity
        """
        if len(entities) == 1:
            return []

        requirements = []

        # Information sharing is always needed for multi-entity coordination
        requirements.append("information_sharing")

        # Check governance level diversity
        levels = set(e.governance_level for e in entities)
        if len(levels) > 1:
            requirements.append("vertical_coordination")
            requirements.append("hierarchical_alignment")

        # Check resource distribution
        resources = [e.resources.get("budget", 0) for e in entities]
        if resources:
            resource_variance = self._calculate_variance(resources)
            if resource_variance > 1000000:  # Significant variance
                requirements.append("resource_coordination")
                requirements.append("budget_alignment")

        # Check stakeholder overlap
        all_stakeholders = set()
        for entity in entities:
            all_stakeholders.update(entity.stakeholders)

        # If there's significant stakeholder overlap, need alignment
        if len(all_stakeholders) < sum(len(e.stakeholders) for e in entities) * 0.7:
            requirements.append("stakeholder_alignment")
            requirements.append("consensus_building")

        # Check decision domain complexity
        all_domains = set()
        for entity in entities:
            all_domains.update(entity.responsibilities)

        if len(all_domains) > len(entities) * 2:
            requirements.append("domain_coordination")
            requirements.append("jurisdictional_clarification")

        # Add conflict resolution if multiple entities
        if len(entities) > 2:
            requirements.append("conflict_resolution_mechanism")

        return list(set(requirements))  # Remove duplicates
