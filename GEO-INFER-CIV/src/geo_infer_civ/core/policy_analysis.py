"""
Policy impact assessment for GEO-INFER-CIV.

Provides cost-benefit scoring, stakeholder impact matrix computation,
and equity analysis for policy evaluation.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ImpactLevel(Enum):
    """Qualitative impact levels for stakeholder analysis."""

    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class PolicyDomain(Enum):
    """Domains of policy action."""

    LAND_USE = "land_use"
    TRANSPORTATION = "transportation"
    HOUSING = "housing"
    ENVIRONMENT = "environment"
    PUBLIC_SAFETY = "public_safety"
    ECONOMIC_DEVELOPMENT = "economic_development"
    EDUCATION = "education"
    HEALTH = "health"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class CostBenefitItem:
    """A single cost or benefit item in the analysis."""

    name: str
    amount: float
    is_benefit: bool
    probability: float = 1.0
    time_horizon_years: int = 1
    category: str | None = None
    description: str | None = None


@dataclass
class StakeholderImpact:
    """Impact assessment for a single stakeholder group."""

    group_name: str
    population_size: int
    impact_level: ImpactLevel
    economic_impact: float = 0.0
    quality_of_life_impact: float = 0.0
    environmental_impact: float = 0.0
    accessibility_impact: float = 0.0


@dataclass
class CostBenefitResult:
    """Result of a cost-benefit analysis."""

    total_costs: float
    total_benefits: float
    net_present_value: float
    benefit_cost_ratio: float
    internal_rate_of_return: float
    payback_period_years: float
    risk_adjusted_npv: float
    category_breakdown: dict[str, dict[str, float]]


@dataclass
class EquityScore:
    """Result of equity analysis across demographics."""

    overall_equity_score: float
    gini_coefficient: float
    impact_distribution: dict[str, float]
    most_impacted_group: str
    least_impacted_group: str
    disparate_impact_flags: list[str]


class CostBenefitAnalyzer:
    """
    Performs cost-benefit analysis for policy proposals.

    Computes net present value, benefit-cost ratios, and risk-adjusted
    metrics for evaluating the financial viability of policies.
    """

    def __init__(self, discount_rate: float = 0.05) -> None:
        """
        Initialize the cost-benefit analyzer.

        Args:
            discount_rate: Annual discount rate for NPV calculations.
        """
        if not (0.0 <= discount_rate < 1.0):
            raise ValueError("discount_rate must be in [0, 1)")
        self._discount_rate = discount_rate
        self._items: list[CostBenefitItem] = []

    def add_item(self, item: CostBenefitItem) -> None:
        """
        Add a cost or benefit item to the analysis.

        Args:
            item: A cost-benefit item.
        """
        self._items.append(item)

    def add_items(self, items: list[CostBenefitItem]) -> None:
        """
        Add multiple cost-benefit items.

        Args:
            items: List of cost-benefit items.
        """
        self._items.extend(items)

    def analyze(self) -> CostBenefitResult:
        """
        Run the cost-benefit analysis on all added items.

        Computes discounted costs and benefits, NPV, BCR,
        approximate IRR, and payback period.

        Returns:
            CostBenefitResult with all computed metrics.

        Raises:
            ValueError: If no items have been added.
        """
        if not self._items:
            raise ValueError("No cost-benefit items to analyze")

        costs = [i for i in self._items if not i.is_benefit]
        benefits = [i for i in self._items if i.is_benefit]

        total_costs = self._sum_discounted(costs)
        total_benefits = self._sum_discounted(benefits)
        risk_adj_costs = self._sum_risk_adjusted(costs)
        risk_adj_benefits = self._sum_risk_adjusted(benefits)

        npv = total_benefits - total_costs
        risk_adjusted_npv = risk_adj_benefits - risk_adj_costs
        bcr = total_benefits / total_costs if total_costs > 0 else float("inf")

        irr = self._approximate_irr()
        payback = self._compute_payback_period()

        # Category breakdown
        categories: dict[str, dict[str, float]] = {}
        for item in self._items:
            cat = item.category or "uncategorized"
            if cat not in categories:
                categories[cat] = {"costs": 0.0, "benefits": 0.0}
            key = "benefits" if item.is_benefit else "costs"
            discounted = item.amount / (
                (1 + self._discount_rate) ** item.time_horizon_years
            )
            categories[cat][key] += discounted

        for cat in categories:
            categories[cat] = {k: round(v, 2) for k, v in categories[cat].items()}

        return CostBenefitResult(
            total_costs=round(total_costs, 2),
            total_benefits=round(total_benefits, 2),
            net_present_value=round(npv, 2),
            benefit_cost_ratio=round(bcr, 4),
            internal_rate_of_return=round(irr, 4),
            payback_period_years=round(payback, 2),
            risk_adjusted_npv=round(risk_adjusted_npv, 2),
            category_breakdown=categories,
        )

    def _sum_discounted(self, items: list[CostBenefitItem]) -> float:
        """Sum items with time-value discounting."""
        total = 0.0
        for item in items:
            discounted = item.amount / (
                (1 + self._discount_rate) ** item.time_horizon_years
            )
            total += discounted
        return total

    def _sum_risk_adjusted(self, items: list[CostBenefitItem]) -> float:
        """Sum items adjusted for probability of occurrence."""
        total = 0.0
        for item in items:
            discounted = item.amount / (
                (1 + self._discount_rate) ** item.time_horizon_years
            )
            total += discounted * item.probability
        return total

    def _approximate_irr(self) -> float:
        """Approximate the internal rate of return using bisection.

        Returns ``nan`` when NPV never crosses zero over the searched
        rate range (pure-cost or pure-benefit item sets have no IRR).
        """

        def npv_at_rate(rate: float) -> float:
            total = 0.0
            for item in self._items:
                disc = item.amount / ((1 + rate) ** item.time_horizon_years)
                total += disc if item.is_benefit else -disc
            return total

        low, high = -0.5, 2.0
        if npv_at_rate(low) * npv_at_rate(high) > 0:
            return float("nan")
        for _ in range(100):
            mid = (low + high) / 2.0
            if npv_at_rate(mid) > 0:
                low = mid
            else:
                high = mid
            if abs(high - low) < 1e-6:
                break

        return (low + high) / 2.0

    def _compute_payback_period(self) -> float:
        """Compute the simple payback period in years.

        Returns ``float("inf")`` when the cumulative net position never
        reaches zero within the modeled horizon.
        """
        costs_by_year: dict[int, float] = {}
        benefits_by_year: dict[int, float] = {}

        for item in self._items:
            year = item.time_horizon_years
            if item.is_benefit:
                benefits_by_year[year] = benefits_by_year.get(year, 0.0) + item.amount
            else:
                costs_by_year[year] = costs_by_year.get(year, 0.0) + item.amount

        max_year = max(
            max(costs_by_year.keys(), default=0),
            max(benefits_by_year.keys(), default=0),
        )

        cumulative_net = 0.0
        for year in range(max_year + 1):
            cumulative_net += benefits_by_year.get(year, 0.0) - costs_by_year.get(
                year, 0.0
            )
            if cumulative_net >= 0 and year > 0:
                # Interpolate
                prev_net = cumulative_net - (
                    benefits_by_year.get(year, 0.0) - costs_by_year.get(year, 0.0)
                )
                yearly_flow = benefits_by_year.get(year, 0.0) - costs_by_year.get(
                    year, 0.0
                )
                if yearly_flow > 0:
                    fraction = -prev_net / yearly_flow
                    return year - 1 + fraction
                return float(year)

        return float("inf")


class StakeholderImpactAnalyzer:
    """
    Computes stakeholder impact matrices for policy proposals.

    Evaluates how different stakeholder groups are affected across
    multiple dimensions (economic, quality of life, environmental, accessibility).
    """

    def __init__(self) -> None:
        self._impacts: list[StakeholderImpact] = []

    def add_impact(self, impact: StakeholderImpact) -> None:
        """
        Add a stakeholder impact assessment.

        Args:
            impact: Impact assessment for a stakeholder group.
        """
        self._impacts.append(impact)

    def add_impacts(self, impacts: list[StakeholderImpact]) -> None:
        """
        Add multiple stakeholder impact assessments.

        Args:
            impacts: List of stakeholder impact assessments.
        """
        self._impacts.extend(impacts)

    def compute_impact_matrix(self) -> dict[str, dict[str, float]]:
        """
        Compute the full stakeholder impact matrix.

        Returns a matrix where rows are stakeholder groups and columns
        are impact dimensions, with values normalized to [-1, 1].

        Returns:
            Nested dictionary of group -> dimension -> normalized score.

        Raises:
            ValueError: If no impacts have been added.
        """
        if not self._impacts:
            raise ValueError("No stakeholder impacts to analyze")

        per_group: dict[str, list[tuple[dict[str, float], int]]] = {}
        for impact in self._impacts:
            scores = {
                "overall_impact": impact.impact_level.value / 2.0,
                "economic": self._normalize_impact(impact.economic_impact),
                "quality_of_life": self._normalize_impact(
                    impact.quality_of_life_impact
                ),
                "environmental": self._normalize_impact(impact.environmental_impact),
                "accessibility": self._normalize_impact(impact.accessibility_impact),
            }
            per_group.setdefault(impact.group_name, []).append(
                (scores, impact.population_size)
            )

        duplicated = sorted(
            name for name, entries in per_group.items() if len(entries) > 1
        )
        if duplicated:
            logger.warning(
                "Duplicate stakeholder group names in impact analysis: %s; "
                "entries are merged with a population-weighted average per "
                "dimension.",
                ", ".join(duplicated),
            )

        matrix: dict[str, dict[str, float]] = {}
        for group, entries in per_group.items():
            total_pop = sum(pop for _, pop in entries)
            if total_pop > 0:
                merged = {
                    key: sum(s[key] * pop for s, pop in entries) / total_pop
                    for key in entries[0][0]
                }
            else:
                merged = {
                    key: sum(s[key] for s, _ in entries) / len(entries)
                    for key in entries[0][0]
                }
            weighted_score = (
                0.30 * merged["economic"]
                + 0.25 * merged["quality_of_life"]
                + 0.25 * merged["environmental"]
                + 0.20 * merged["accessibility"]
            )
            merged["weighted_composite"] = weighted_score
            matrix[group] = {k: round(v, 4) for k, v in merged.items()}

        return matrix

    def compute_aggregate_score(self) -> float:
        """
        Compute a population-weighted aggregate impact score.

        Weights each stakeholder group's impact by their population size.

        Returns:
            Aggregate score in [-1, 1].
        """
        if not self._impacts:
            return 0.0

        total_pop = sum(i.population_size for i in self._impacts)
        if total_pop == 0:
            return 0.0

        weighted_sum = 0.0
        for impact in self._impacts:
            composite = (
                0.30 * self._normalize_impact(impact.economic_impact)
                + 0.25 * self._normalize_impact(impact.quality_of_life_impact)
                + 0.25 * self._normalize_impact(impact.environmental_impact)
                + 0.20 * self._normalize_impact(impact.accessibility_impact)
            )
            weighted_sum += composite * impact.population_size

        return round(weighted_sum / total_pop, 4)

    def find_most_affected(self) -> tuple[str, str]:
        """
        Find the most positively and negatively affected groups.

        Returns:
            Tuple of (most_positive_group, most_negative_group).

        Raises:
            ValueError: If no impacts have been added.
        """
        if not self._impacts:
            raise ValueError("No stakeholder impacts to analyze")

        matrix = self.compute_impact_matrix()
        best_group = max(matrix.keys(), key=lambda g: matrix[g]["weighted_composite"])
        worst_group = min(matrix.keys(), key=lambda g: matrix[g]["weighted_composite"])
        return (best_group, worst_group)

    @staticmethod
    def _normalize_impact(value: float) -> float:
        """Normalize an impact value to [-1, 1] using tanh."""
        return math.tanh(value)


class EquityAnalyzer:
    """
    Performs equity analysis of policy impacts across demographic groups.

    Computes Gini coefficients, disparate impact tests, and overall
    equity scores to evaluate distributional fairness.
    """

    def __init__(self) -> None:
        self._group_impacts: dict[str, float] = {}
        self._group_populations: dict[str, int] = {}

    def set_group_impact(
        self, group_name: str, impact_value: float, population: int
    ) -> None:
        """
        Set the impact value and population for a demographic group.

        Args:
            group_name: Name of the demographic group.
            impact_value: Numeric impact value (higher = more benefit).
            population: Population size of the group (must be positive).

        Raises:
            ValueError: If population is not positive.
        """
        if population <= 0:
            raise ValueError(
                f"Population for '{group_name}' must be positive, got {population}"
            )
        self._group_impacts[group_name] = impact_value
        self._group_populations[group_name] = population

    def analyze(self) -> EquityScore:
        """
        Run the equity analysis.

        Computes Gini coefficient, identifies most/least impacted
        groups, and flags potential disparate impacts.

        Returns:
            EquityScore with equity metrics.

        Raises:
            ValueError: If fewer than 2 groups are defined.
        """
        if len(self._group_impacts) < 2:
            raise ValueError(
                "At least 2 demographic groups required for equity analysis"
            )

        impacts = self._group_impacts
        populations = self._group_populations

        # Sort groups by impact value
        sorted_groups = sorted(impacts.keys(), key=lambda g: impacts[g])

        # Compute Gini coefficient using the absolute impact values
        # Shift values to be non-negative for Gini calculation
        values = [impacts[g] for g in sorted_groups]
        min_val = min(values)
        shifted = [v - min_val for v in values]
        gini = self._compute_gini(shifted)

        # Impact distribution (normalized proportions)
        total_abs_impact = sum(abs(v) for v in impacts.values())
        if total_abs_impact > 0:
            distribution = {
                g: round(abs(v) / total_abs_impact, 4) for g, v in impacts.items()
            }
        else:
            distribution = {g: round(1.0 / len(impacts), 4) for g in impacts}

        most_impacted = max(impacts.keys(), key=lambda g: impacts[g])
        least_impacted = min(impacts.keys(), key=lambda g: impacts[g])

        # Disparate impact flags (4/5ths rule analog)
        flags = self._check_disparate_impact(impacts, populations)

        # Overall equity score: 1.0 = perfect equity, 0.0 = maximum inequality
        overall_equity = max(0.0, 1.0 - gini)

        return EquityScore(
            overall_equity_score=round(overall_equity, 4),
            gini_coefficient=round(gini, 4),
            impact_distribution=distribution,
            most_impacted_group=most_impacted,
            least_impacted_group=least_impacted,
            disparate_impact_flags=flags,
        )

    @staticmethod
    def _compute_gini(values: list[float]) -> float:
        """Compute Gini coefficient from a list of non-negative values."""
        n = len(values)
        if n == 0:
            return 0.0

        total = sum(values)
        if total == 0:
            return 0.0

        sorted_vals = sorted(values)
        cumulative_sum = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumulative_sum += v
            weighted_sum += (2 * (i + 1) - n - 1) * v

        return weighted_sum / (n * total)

    @staticmethod
    def _check_disparate_impact(
        impacts: dict[str, float],
        populations: dict[str, int],
    ) -> list[str]:
        """
        Check for disparate impact using the 4/5ths rule analog on
        per-capita impact rates.

        Each group's per-capita rate is impact / population. If any
        group's per-capita rate is less than 80% of the highest rate,
        flag it.
        """
        flags: list[str] = []
        if not impacts:
            return flags

        rates: dict[str, float] = {
            group: impact / populations[group] for group, impact in impacts.items()
        }

        max_rate = max(rates.values())
        if max_rate <= 0:
            return flags

        threshold = 0.8 * max_rate
        for group, rate in rates.items():
            if rate < threshold:
                flags.append(
                    f"{group}: per-capita impact ({rate:.6f}) is below 80% of "
                    f"highest per-capita impact ({max_rate:.6f})"
                )

        return flags
