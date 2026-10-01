"""Institutional design and analysis using IAD framework and Ostrom's principles."""

from dataclasses import dataclass
from typing import Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InstitutionalFramework(Enum):
    """Institutional analysis frameworks."""

    IAD = "iad"  # Institutional Analysis and Development
    OSTROM = "ostrom"  # Ostrom's design principles


@dataclass
class Institution:
    """Represents a governance institution or rule."""

    institution_id: str
    name: str
    rule_type: (
        str  # boundary, position, choice, information, aggregation, payoff, scope
    )
    description: str
    affected_stakeholders: list[str]
    enforcement_mechanism: str
    effectiveness_rating: float = 0.5


@dataclass
class InstitutionalAnalysis:
    """Results of institutional analysis."""

    governance_domain: str
    analysis_framework: InstitutionalFramework
    existing_institutions: list[Institution]
    institutional_effectiveness: dict[str, float]
    design_principles_assessment: dict[str, float]
    recommendations: list[str]


class InstitutionalDesigner:
    """
    Institutional analysis and design using formal frameworks.

    Implements the Institutional Analysis and Development (IAD) framework
    developed by Elinor Ostrom and colleagues, as well as Ostrom's design
    principles for sustainable institutions.

    References:
    - Ostrom, E. (1990). Governing the Commons: The Evolution of Institutions
    - McGinnis, M. D. (2011). An Introduction to IAD and the Language of the Ostrom Workshop
    - Dietz, T., et al. (2003). The Drama of the Commons
    """

    def __init__(
        self, framework: str = "iad", context_type: str = "common_pool_resource"
    ) -> None:
        """
        Initialize institutional designer.

        Parameters:
        -----------
        framework : str
            Institutional analysis framework (iad, ostrom)
        context_type : str
            Context type (common_pool_resource, public_goods, etc.)
        """
        self.framework = InstitutionalFramework(framework)
        self.context_type = context_type
        self.institutional_analyses: dict[str, InstitutionalAnalysis] = {}

    def analyze_institutions(
        self,
        current_institutions: list[dict[str, Any]],
        stakeholder_groups: list[dict[str, Any]],
        resource_system: dict[str, Any],
        decision_outcomes: list[dict[str, Any]],
    ) -> InstitutionalAnalysis:
        """
        Analyze existing institutions using IAD framework.

        Parameters:
        -----------
        current_institutions : List[Dict[str, Any]]
            Current institutional rules
        stakeholder_groups : List[Dict[str, Any]]
            Stakeholder groups affected
        resource_system : Dict[str, Any]
            Resource system being governed
        decision_outcomes : List[Dict[str, Any]]
            Observed governance outcomes

        Returns:
        --------
        InstitutionalAnalysis
            Comprehensive institutional analysis
        """
        # Convert institutions to Institution objects
        institutions = [
            Institution(
                institution_id=f"inst_{i}",
                name=inst.get("name", f"Institution {i}"),
                rule_type=inst.get("type", "unspecified"),
                description=inst.get("description", ""),
                affected_stakeholders=[
                    sg.get("name", f"stakeholder_{j}")
                    for j, sg in enumerate(stakeholder_groups)
                ],
                enforcement_mechanism=inst.get("enforcement", "informal"),
            )
            for i, inst in enumerate(current_institutions)
        ]

        # Assess institutional effectiveness
        effectiveness = self._assess_effectiveness(
            institutions=institutions, outcomes=decision_outcomes
        )

        # Assess design principles
        principles_assessment = self._assess_design_principles(
            institutions=institutions, resource_system=resource_system
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            effectiveness=effectiveness, principles_assessment=principles_assessment
        )

        analysis = InstitutionalAnalysis(
            governance_domain=resource_system.get("domain", "general"),
            analysis_framework=self.framework,
            existing_institutions=institutions,
            institutional_effectiveness=effectiveness,
            design_principles_assessment=principles_assessment,
            recommendations=recommendations,
        )

        self.institutional_analyses[resource_system.get("id", "analysis_0")] = analysis
        logger.info(
            f"Institutional analysis completed for {resource_system.get('domain', 'domain')}"
        )

        return analysis

    def _assess_effectiveness(
        self, institutions: list[Institution], outcomes: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Assess effectiveness of institutions using outcome-based analysis.

        Uses multiple effectiveness indicators:
        - Outcome achievement (did outcomes meet objectives?)
        - Stakeholder satisfaction
        - Resource sustainability
        - Equity of outcomes
        - Compliance rates

        References:
        - Ostrom, E. (2005). Understanding Institutional Diversity
        - Young, O. R. (2002). The Institutional Dimensions of Environmental Change
        """
        effectiveness = {}

        for inst in institutions:
            # Find relevant outcomes for this institution
            relevant_outcomes = [
                outcome
                for outcome in outcomes
                if any(
                    stake in outcome.get("stakeholders", [])
                    for stake in inst.affected_stakeholders
                )
            ]

            if not relevant_outcomes:
                # No outcomes available - use default based on enforcement mechanism
                enforcement_weights = {
                    "legal": 0.6,
                    "formal": 0.5,
                    "informal": 0.4,
                    "social": 0.3,
                }
                effectiveness[inst.institution_id] = enforcement_weights.get(
                    inst.enforcement_mechanism, 0.5
                )
                continue

            # Calculate multiple effectiveness dimensions
            effectiveness_scores = []

            # 1. Direct effectiveness score from outcomes
            direct_scores = [
                outcome.get("effectiveness", 0.5) for outcome in relevant_outcomes
            ]
            if direct_scores:
                avg_direct = sum(direct_scores) / len(direct_scores)
                effectiveness_scores.append(("direct", avg_direct, 0.4))

            # 2. Stakeholder satisfaction
            satisfaction_scores = [
                outcome.get("stakeholder_satisfaction", 0.5)
                for outcome in relevant_outcomes
                if "stakeholder_satisfaction" in outcome
            ]
            if satisfaction_scores:
                avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
                effectiveness_scores.append(("satisfaction", avg_satisfaction, 0.2))

            # 3. Resource sustainability (if applicable)
            sustainability_scores = [
                outcome.get("sustainability", 0.5)
                for outcome in relevant_outcomes
                if "sustainability" in outcome
            ]
            if sustainability_scores:
                avg_sustainability = sum(sustainability_scores) / len(
                    sustainability_scores
                )
                effectiveness_scores.append(("sustainability", avg_sustainability, 0.2))

            # 4. Equity of outcomes
            equity_scores = [
                outcome.get("equity", 0.5)
                for outcome in relevant_outcomes
                if "equity" in outcome
            ]
            if equity_scores:
                avg_equity = sum(equity_scores) / len(equity_scores)
                effectiveness_scores.append(("equity", avg_equity, 0.1))

            # 5. Compliance rate (if available)
            compliance_rates = [
                outcome.get("compliance_rate", 0.5)
                for outcome in relevant_outcomes
                if "compliance_rate" in outcome
            ]
            if compliance_rates:
                avg_compliance = sum(compliance_rates) / len(compliance_rates)
                effectiveness_scores.append(("compliance", avg_compliance, 0.1))

            # Calculate weighted effectiveness
            if effectiveness_scores:
                total_weight = sum(weight for _, _, weight in effectiveness_scores)
                weighted_effectiveness = (
                    sum(score * weight for _, score, weight in effectiveness_scores)
                    / total_weight
                    if total_weight > 0
                    else 0.5
                )
            else:
                # Fallback: use direct scores or default
                weighted_effectiveness = avg_direct if direct_scores else 0.5

            # Adjust for enforcement mechanism strength
            enforcement_adjustment = {
                "legal": 1.0,
                "formal": 0.9,
                "informal": 0.7,
                "social": 0.6,
            }.get(inst.enforcement_mechanism, 0.8)

            final_effectiveness = weighted_effectiveness * enforcement_adjustment
            effectiveness[inst.institution_id] = min(1.0, max(0.0, final_effectiveness))

        return effectiveness

    def _assess_design_principles(
        self, institutions: list[Institution], resource_system: dict[str, Any]
    ) -> dict[str, float]:
        """
        Assess Ostrom's design principles quantitatively based on institutional characteristics.

        Evaluates each principle based on:
        - Presence of relevant institutional rules
        - Rule characteristics and quality
        - Resource system attributes
        - Stakeholder involvement

        References:
        - Ostrom, E. (1990). Governing the Commons
        - Cox, M., et al. (2010). A Review of Design Principles for Community-based Natural Resource Management
        """
        principles = {}

        # 1. Clear Boundaries
        # Check for boundary rules
        boundary_rules = [inst for inst in institutions if inst.rule_type == "boundary"]
        if boundary_rules:
            # Assess quality of boundary rules
            avg_effectiveness = sum(
                inst.effectiveness_rating for inst in boundary_rules
            ) / len(boundary_rules)
            # Check if boundaries are clearly defined in resource system
            has_spatial_boundaries = (
                "bounds" in resource_system or "area_km2" in resource_system
            )
            has_user_boundaries = any(
                "user" in inst.description.lower()
                or "member" in inst.description.lower()
                for inst in boundary_rules
            )
            principles["clear_boundaries"] = min(
                1.0,
                avg_effectiveness * 0.7
                + (0.15 if has_spatial_boundaries else 0)
                + (0.15 if has_user_boundaries else 0),
            )
        else:
            principles["clear_boundaries"] = 0.2  # Low score if no boundary rules

        # 2. Congruence (Rules aligned with local conditions)
        # Check for congruence indicators
        has_local_adaptation = any(
            "local" in inst.description.lower() or "adapt" in inst.description.lower()
            for inst in institutions
        )
        resource_complexity = resource_system.get("complexity", "medium")
        # More complex resources need more sophisticated rules
        complexity_match = {"low": 0.8, "medium": 0.6, "high": 0.4}.get(
            resource_complexity, 0.6
        )

        congruence_rules = [
            inst for inst in institutions if inst.rule_type in ["choice", "scope"]
        ]
        if congruence_rules:
            avg_effectiveness = sum(
                inst.effectiveness_rating for inst in congruence_rules
            ) / len(congruence_rules)
            principles["congruence"] = min(
                1.0,
                avg_effectiveness * 0.6
                + complexity_match * 0.3
                + (0.1 if has_local_adaptation else 0),
            )
        else:
            principles["congruence"] = 0.3

        # 3. Collective Choice Arrangements
        # Check for participation rules
        participation_rules = [
            inst
            for inst in institutions
            if inst.rule_type == "choice"
            and any(
                word in inst.description.lower()
                for word in ["participate", "decision", "vote", "consensus"]
            )
        ]
        if participation_rules:
            avg_effectiveness = sum(
                inst.effectiveness_rating for inst in participation_rules
            ) / len(participation_rules)
            # Check stakeholder involvement
            num_affected_stakeholders = len(
                set(
                    stake
                    for inst in participation_rules
                    for stake in inst.affected_stakeholders
                )
            )
            stakeholder_coverage = min(
                1.0, num_affected_stakeholders / max(1, len(institutions))
            )
            principles["collective_choice_arrangements"] = min(
                1.0, avg_effectiveness * 0.7 + stakeholder_coverage * 0.3
            )
        else:
            principles["collective_choice_arrangements"] = 0.2

        # 4. Monitoring
        # Check for monitoring rules
        monitoring_rules = [
            inst
            for inst in institutions
            if inst.rule_type == "information" or "monitor" in inst.description.lower()
        ]
        if monitoring_rules:
            avg_effectiveness = sum(
                inst.effectiveness_rating for inst in monitoring_rules
            ) / len(monitoring_rules)
            # Check enforcement mechanism strength
            has_formal_monitoring = any(
                inst.enforcement_mechanism in ["legal", "formal"]
                for inst in monitoring_rules
            )
            principles["monitoring"] = min(
                1.0, avg_effectiveness * 0.7 + (0.3 if has_formal_monitoring else 0.1)
            )
        else:
            principles["monitoring"] = 0.2

        # 5. Graduated Sanctions
        # Check for sanction rules
        sanction_rules = [
            inst
            for inst in institutions
            if "sanction" in inst.description.lower()
            or "penalty" in inst.description.lower()
        ]
        if sanction_rules:
            avg_effectiveness = sum(
                inst.effectiveness_rating for inst in sanction_rules
            ) / len(sanction_rules)
            # Check if sanctions are graduated (proportional)
            has_graduation = any(
                "graduat" in inst.description.lower()
                or "proportional" in inst.description.lower()
                for inst in sanction_rules
            )
            principles["graduated_sanctions"] = min(
                1.0, avg_effectiveness * 0.6 + (0.4 if has_graduation else 0.1)
            )
        else:
            principles["graduated_sanctions"] = 0.1  # Very low if no sanctions

        # 6. Conflict Resolution
        # Check for conflict resolution mechanisms
        conflict_rules = [
            inst
            for inst in institutions
            if "conflict" in inst.description.lower()
            or "dispute" in inst.description.lower()
            or "resolution" in inst.description.lower()
        ]
        if conflict_rules:
            avg_effectiveness = sum(
                inst.effectiveness_rating for inst in conflict_rules
            ) / len(conflict_rules)
            # Check accessibility (low-cost)
            has_accessible_process = any(
                "low-cost" in inst.description.lower()
                or "accessible" in inst.description.lower()
                for inst in conflict_rules
            )
            principles["conflict_resolution"] = min(
                1.0, avg_effectiveness * 0.7 + (0.3 if has_accessible_process else 0.1)
            )
        else:
            principles["conflict_resolution"] = 0.2

        # 7. Right to Organize
        # Check for organizational autonomy
        organization_rules = [
            inst
            for inst in institutions
            if "organize" in inst.description.lower()
            or "autonomy" in inst.description.lower()
            or "self-govern" in inst.description.lower()
        ]
        # Also check if external recognition exists
        has_external_recognition = resource_system.get("external_recognition", False)
        if organization_rules or has_external_recognition:
            if organization_rules:
                avg_effectiveness = sum(
                    inst.effectiveness_rating for inst in organization_rules
                ) / len(organization_rules)
            else:
                avg_effectiveness = 0.5
            principles["right_to_organize"] = min(
                1.0,
                avg_effectiveness * 0.6 + (0.4 if has_external_recognition else 0.2),
            )
        else:
            principles["right_to_organize"] = 0.3

        # 8. Nested Enterprises
        # Check for nested governance structures
        nested_rules = [
            inst
            for inst in institutions
            if "nested" in inst.description.lower()
            or "tier" in inst.description.lower()
            or "level" in inst.description.lower()
        ]
        has_multiple_levels = resource_system.get("governance_levels", 1) > 1
        if nested_rules or has_multiple_levels:
            if nested_rules:
                avg_effectiveness = sum(
                    inst.effectiveness_rating for inst in nested_rules
                ) / len(nested_rules)
            else:
                avg_effectiveness = 0.5
            principles["nested_enterprises"] = min(
                1.0, avg_effectiveness * 0.6 + (0.4 if has_multiple_levels else 0.1)
            )
        else:
            principles["nested_enterprises"] = 0.1  # Very low if no nesting

        return principles

    def _generate_recommendations(
        self, effectiveness: dict[str, float], principles_assessment: dict[str, float]
    ) -> list[str]:
        """Generate recommendations for institutional improvement."""
        recommendations = []

        # Identify low-scoring principles
        low_principles = [k for k, v in principles_assessment.items() if v < 0.5]

        if "clear_boundaries" in low_principles:
            recommendations.append("Clarify user and resource boundaries")
        if "congruence" in low_principles:
            recommendations.append("Align rules with local conditions")
        if "monitoring" in low_principles:
            recommendations.append("Strengthen monitoring mechanisms")
        if "conflict_resolution" in low_principles:
            recommendations.append("Establish conflict resolution procedures")

        return recommendations

    def apply_ostrom_principles(
        self,
        principle_set: list[str],
        resource_system: dict[str, Any],
        governance_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply Ostrom's design principles for sustainable institutions.

        Parameters:
        -----------
        principle_set : List[str]
            Subset of Ostrom's 8 principles to apply
        resource_system : Dict[str, Any]
            Resource system characteristics
        governance_context : Dict[str, Any]
            Governance context and constraints

        Returns:
        --------
        Dict[str, Any]
            Designed institutions based on Ostrom's principles
        """
        designed_institutions = {}

        principle_descriptions = {
            "clear_boundaries": "Clearly defined boundaries of resource and users",
            "congruence": "Rules aligned with local conditions and user needs",
            "collective_choice_arrangements": "Users participate in governance decisions",
            "monitoring": "Effective monitoring of resource use and users",
            "graduated_sanctions": "Proportional sanctions for rule violations",
            "conflict_resolution": "Accessible, low-cost conflict resolution",
            "right_to_organize": "External authorities respect right to self-organize",
            "nested_enterprises": "Governance organized in nested tiers",
        }

        for principle in principle_set:
            if principle in principle_descriptions:
                designed_institutions[principle] = {
                    "description": principle_descriptions[principle],
                    "implementation_strategy": self._design_principle_implementation(
                        principle=principle,
                        resource_system=resource_system,
                        governance_context=governance_context,
                    ),
                    "expected_outcomes": self._predict_principle_outcomes(principle),
                }

        return {
            "governance_design": designed_institutions,
            "resource_system": resource_system,
            "design_coherence": self._assess_design_coherence(designed_institutions),
        }

    def _design_principle_implementation(
        self,
        principle: str,
        resource_system: dict[str, Any],
        governance_context: dict[str, Any],
    ) -> list[str]:
        """Design implementation strategy for a principle."""
        strategies = {
            "clear_boundaries": [
                "Map resource extent and use rights",
                "Define membership criteria",
                "Register users in governance system",
            ],
            "congruence": [
                "Conduct local ecological assessment",
                "Align rules with seasonal patterns",
                "Adapt rules based on resource variability",
            ],
            "collective_choice_arrangements": [
                "Establish governance assemblies",
                "Create decision-making committees",
                "Enable user participation in rule-making",
            ],
            "monitoring": [
                "Deploy monitoring systems",
                "Train monitors from local community",
                "Establish regular monitoring schedules",
            ],
            "graduated_sanctions": [
                "Define violation categories",
                "Establish proportional penalties",
                "Document sanction procedures",
            ],
            "conflict_resolution": [
                "Establish arbitration mechanisms",
                "Create mediation procedures",
                "Document dispute resolution processes",
            ],
            "right_to_organize": [
                "Secure legal recognition of organization",
                "Establish collaborative relationships with authorities",
                "Document governance agreements",
            ],
            "nested_enterprises": [
                "Organize governance in tiers",
                "Define inter-tier coordination",
                "Establish information flows between tiers",
            ],
        }

        return strategies.get(
            principle, ["Develop implementation strategy for " + principle]
        )

    def _predict_principle_outcomes(self, principle: str) -> dict[str, Any]:
        """Predict outcomes from implementing a principle."""
        return {
            "sustainability_impact": 0.7
            if principle in ["clear_boundaries", "monitoring"]
            else 0.5,
            "equity_impact": 0.6
            if principle in ["collective_choice_arrangements", "conflict_resolution"]
            else 0.4,
            "efficiency_impact": 0.8
            if principle in ["congruence", "monitoring"]
            else 0.5,
            "implementation_difficulty": 0.6,
        }

    def _assess_design_coherence(self, designed_institutions: dict[str, Any]) -> float:
        """
        Assess how well principles work together (synergy analysis).

        Principles that work well together:
        - Clear boundaries + Monitoring
        - Collective choice + Conflict resolution
        - Monitoring + Graduated sanctions
        - Nested enterprises + Right to organize

        Returns coherence score (0-1).
        """
        num_principles = len(designed_institutions)
        if num_principles == 0:
            return 0.0

        # Base coherence from number of principles
        base_coherence = min(1.0, num_principles / 8.0)

        # Check for synergistic principle pairs
        synergies = 0
        principle_names = set(designed_institutions.keys())

        # Synergy 1: Boundaries + Monitoring
        if "clear_boundaries" in principle_names and "monitoring" in principle_names:
            synergies += 1

        # Synergy 2: Collective choice + Conflict resolution
        if (
            "collective_choice_arrangements" in principle_names
            and "conflict_resolution" in principle_names
        ):
            synergies += 1

        # Synergy 3: Monitoring + Graduated sanctions
        if "monitoring" in principle_names and "graduated_sanctions" in principle_names:
            synergies += 1

        # Synergy 4: Nested enterprises + Right to organize
        if (
            "nested_enterprises" in principle_names
            and "right_to_organize" in principle_names
        ):
            synergies += 1

        # Synergy 5: Congruence + Collective choice
        if (
            "congruence" in principle_names
            and "collective_choice_arrangements" in principle_names
        ):
            synergies += 1

        # Synergy bonus
        synergy_bonus = min(0.3, synergies * 0.06)

        final_coherence = min(1.0, base_coherence + synergy_bonus)
        return final_coherence

    def check_institutional_compatibility(
        self, institutions: list[Institution], resource_system: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Check compatibility between institutions and resource system.

        Analyzes:
        - Rule type coverage
        - Enforcement mechanism consistency
        - Stakeholder alignment
        - Resource system fit

        Returns compatibility assessment.
        """
        compatibility = {
            "overall_compatibility": 0.0,
            "rule_coverage": {},
            "enforcement_consistency": 0.0,
            "stakeholder_alignment": 0.0,
            "resource_fit": 0.0,
            "conflicts": [],
        }

        if not institutions:
            return compatibility

        # 1. Rule type coverage
        rule_types = set(inst.rule_type for inst in institutions)
        expected_rule_types = {
            "boundary",
            "position",
            "choice",
            "information",
            "aggregation",
            "payoff",
            "scope",
        }
        coverage = len(rule_types & expected_rule_types) / len(expected_rule_types)
        compatibility["rule_coverage"] = {
            "coverage_ratio": coverage,
            "covered_types": list(rule_types),
            "missing_types": list(expected_rule_types - rule_types),
        }

        # 2. Enforcement mechanism consistency
        enforcement_mechanisms = [inst.enforcement_mechanism for inst in institutions]
        # Check if mechanisms are consistent (not mixing legal with social)
        legal_count = sum(1 for m in enforcement_mechanisms if m in ["legal", "formal"])
        informal_count = sum(
            1 for m in enforcement_mechanisms if m in ["informal", "social"]
        )
        total = len(enforcement_mechanisms)

        if total > 0:
            # Consistency is higher if one type dominates
            consistency = 1.0 - abs(legal_count - informal_count) / total
        else:
            consistency = 0.5

        compatibility["enforcement_consistency"] = consistency

        # 3. Stakeholder alignment
        all_stakeholders = set()
        for inst in institutions:
            all_stakeholders.update(inst.affected_stakeholders)

        resource_stakeholders = resource_system.get("stakeholders", [])
        if isinstance(resource_stakeholders, list):
            resource_stakeholder_set = set(resource_stakeholders)
        else:
            resource_stakeholder_set = set()

        if resource_stakeholder_set:
            alignment = len(all_stakeholders & resource_stakeholder_set) / len(
                resource_stakeholder_set
            )
        else:
            alignment = 0.5  # Neutral if no stakeholder info

        compatibility["stakeholder_alignment"] = alignment

        # 4. Resource system fit
        resource_type = resource_system.get("type", "common_pool_resource")
        context_type = self.context_type

        # Check if institutions match resource type
        if resource_type == context_type:
            resource_fit = 0.9
        elif resource_type in ["common_pool_resource", "public_goods"]:
            resource_fit = 0.7  # Close match
        else:
            resource_fit = 0.5  # Different types

        compatibility["resource_fit"] = resource_fit

        # 5. Detect rule conflicts
        conflicts = self._detect_rule_conflicts(institutions)
        compatibility["conflicts"] = conflicts

        # Calculate overall compatibility
        weights = {
            "rule_coverage": 0.25,
            "enforcement_consistency": 0.20,
            "stakeholder_alignment": 0.25,
            "resource_fit": 0.20,
            "conflict_penalty": 0.10,
        }

        conflict_penalty = min(1.0, len(conflicts) * 0.2)  # Penalty for conflicts

        overall = (
            weights["rule_coverage"] * coverage
            + weights["enforcement_consistency"] * consistency
            + weights["stakeholder_alignment"] * alignment
            + weights["resource_fit"] * resource_fit
            - weights["conflict_penalty"] * conflict_penalty
        )

        compatibility["overall_compatibility"] = max(0.0, min(1.0, overall))

        return compatibility

    def _detect_rule_conflicts(
        self, institutions: list[Institution]
    ) -> list[dict[str, Any]]:
        """Detect conflicts between institutional rules."""
        # Detect conflicts between institutional rules
        conflicts: list[dict[str, Any]] = []

        # Check for contradictory rules of the same type
        rule_groups: dict[str, list[Institution]] = {}
        for inst in institutions:
            if inst.rule_type not in rule_groups:
                rule_groups[inst.rule_type] = []
            rule_groups[inst.rule_type].append(inst)

        # Check for conflicting boundary definitions
        boundary_rules = rule_groups.get("boundary", [])
        if len(boundary_rules) > 1:
            # Check if boundaries conflict (simplified check)
            for i, rule1 in enumerate(boundary_rules):
                for rule2 in boundary_rules[i + 1 :]:
                    if (
                        "exclusive" in rule1.description.lower()
                        and "exclusive" in rule2.description.lower()
                    ):
                        conflicts.append(
                            {
                                "type": "boundary_conflict",
                                "rule1": rule1.institution_id,
                                "rule2": rule2.institution_id,
                                "description": "Conflicting exclusive boundary definitions",
                            }
                        )

        # Check for conflicting choice rules
        choice_rules = rule_groups.get("choice", [])
        if len(choice_rules) > 1:
            # Check for contradictory decision-making processes
            decision_methods: dict[str, Any] = {}
            for rule in choice_rules:
                method = None
                if "consensus" in rule.description.lower():
                    method = "consensus"
                elif (
                    "majority" in rule.description.lower()
                    or "vote" in rule.description.lower()
                ):
                    method = "majority"
                elif "unanimous" in rule.description.lower():
                    method = "unanimous"

                if method:
                    if method in decision_methods:
                        conflicts.append(
                            {
                                "type": "choice_rule_conflict",
                                "rule1": decision_methods[method].institution_id,
                                "rule2": rule.institution_id,
                                "description": f"Conflicting decision methods: {method}",
                            }
                        )
                    else:
                        decision_methods[method] = rule

        # Check for conflicting payoff rules
        payoff_rules = rule_groups.get("payoff", [])
        if len(payoff_rules) > 1:
            # Check for contradictory cost/benefit distributions
            for i, rule1 in enumerate(payoff_rules):
                for rule2 in payoff_rules[i + 1 :]:
                    if (
                        "equal" in rule1.description.lower()
                        and "proportional" in rule2.description.lower()
                    ) or (
                        "equal" in rule2.description.lower()
                        and "proportional" in rule1.description.lower()
                    ):
                        conflicts.append(
                            {
                                "type": "payoff_conflict",
                                "rule1": rule1.institution_id,
                                "rule2": rule2.institution_id,
                                "description": "Conflicting cost/benefit distribution methods",
                            }
                        )

        return conflicts
