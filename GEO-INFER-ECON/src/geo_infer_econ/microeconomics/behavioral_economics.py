"""
Behavioral Economics Module

Implements behavioral economics models including:
- Prospect theory and reference-dependent preferences
- Bounded rationality and cognitive limitations
- Social preferences and fairness concerns
- Mental accounting and framing effects
- Time preferences and hyperbolic discounting
- Nudge analysis and behavioral interventions
"""

import numpy as np
from typing import Any, cast
from dataclasses import dataclass

from ..utils.rng import resolve_rng


@dataclass
class BehavioralParameters:
    """Parameters for behavioral economic models"""

    risk_aversion: float = 0.5
    loss_aversion: float = 2.25
    probability_weighting: str = "prelec"
    time_discount_rate: float = 0.1
    present_bias: float = 1.0
    social_preference_weight: float = 0.5


class ProspectTheory:
    """
    Prospect theory implementation with value function and probability weighting
    """

    def __init__(self, parameters: BehavioralParameters | None = None):
        self.parameters = parameters or BehavioralParameters()
        self.value_function_cache: dict[str, Any] = {}

    def value_function(
        self, outcomes: np.ndarray, reference_point: float = 0.0
    ) -> np.ndarray:
        """
        Prospect theory value function

        Args:
            outcomes: Array of outcome values
            reference_point: Reference point for gains/losses

        Returns:
            Prospect theory values
        """
        gains = outcomes >= reference_point
        losses = outcomes < reference_point

        values = np.zeros_like(outcomes)

        # Gains: v(x) = x^α for x >= 0
        if np.any(gains):
            values[gains] = np.power(outcomes[gains], self.parameters.risk_aversion)

        # Losses: v(x) = -λ*(-x)^β for x < 0
        if np.any(losses):
            loss_values = -(
                outcomes[losses] - reference_point
            )  # Convert to positive losses
            values[losses] = -self.parameters.loss_aversion * np.power(
                loss_values, self.parameters.risk_aversion
            )

        return values

    def probability_weighting(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Probability weighting function (Prelec or other)

        Args:
            probabilities: Array of objective probabilities

        Returns:
            Weighted probabilities
        """
        if self.parameters.probability_weighting == "prelec":
            # Prelec probability weighting: w(p) = exp(-β*(-ln(p))^α)
            alpha = 0.65  # Typical value
            beta = 1.0  # Typical value

            # Avoid log(0)
            probabilities = np.clip(probabilities, 1e-10, 1.0)
            return cast(
                np.ndarray, np.exp(-beta * np.power(-np.log(probabilities), alpha))
            )

        else:
            # Linear probability weighting (expected value)
            return probabilities

    def prospect_value(
        self,
        outcomes: np.ndarray,
        probabilities: np.ndarray,
        reference_point: float = 0.0,
    ) -> float:
        """
        Calculate overall prospect value

        Args:
            outcomes: Array of possible outcomes
            probabilities: Array of outcome probabilities
            reference_point: Reference point

        Returns:
            Prospect theory value
        """
        values = self.value_function(outcomes, reference_point)
        weights = self.probability_weighting(probabilities)

        return float(np.sum(values * weights))


class BoundedRationality:
    """
    Models of bounded rationality and cognitive limitations
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        """
        Initialize bounded rationality models.

        Args:
            rng: Optional random generator for the unrecognised-alternative
                fallback choice. When omitted, a fixed-seed generator is used
                so choices are deterministic by default.
        """
        self.choice_models: dict[str, Any] = {}
        self._rng = resolve_rng(rng)

    def satisficing_model(
        self, alternatives: list[dict[str, Any]], aspiration_level: float
    ) -> dict[str, Any]:
        """
        Satisficing choice model (Simon, 1955)

        Args:
            alternatives: List of alternative options with attributes
            aspiration_level: Minimum acceptable utility level

        Returns:
            Chosen alternative and analysis
        """
        satisficing_alternatives = []

        for alt in alternatives:
            utility = alt.get("utility", 0)
            if utility >= aspiration_level:
                satisficing_alternatives.append(alt)

        if satisficing_alternatives:
            # Choose first satisficing alternative
            chosen = satisficing_alternatives[0]
            return {
                "chosen_alternative": chosen,
                "choice_rule": "satisficing",
                "aspiration_level": aspiration_level,
                "satisficing_alternatives": len(satisficing_alternatives),
            }
        else:
            # If no alternatives meet aspiration, choose best available
            best_alt = max(alternatives, key=lambda x: x.get("utility", 0))
            return {
                "chosen_alternative": best_alt,
                "choice_rule": "satisficing_failed",
                "aspiration_level": aspiration_level,
            }

    def recognition_heuristic(
        self, alternatives: list[str], recognition_memory: dict[str, float]
    ) -> str:
        """
        Recognition heuristic for decision making

        Args:
            alternatives: List of alternative names
            recognition_memory: Recognition strength for each alternative

        Returns:
            Chosen alternative
        """
        # Choose the alternative with highest recognition
        recognized = [alt for alt in alternatives if alt in recognition_memory]

        if recognized:
            return max(recognized, key=lambda x: recognition_memory[x])
        else:
            # Random choice if none recognized
            return cast(str, self._rng.choice(alternatives))


class SocialPreferences:
    """
    Models of social preferences and fairness
    """

    def __init__(self) -> None:
        self.social_utility_functions: dict[str, Any] = {}

    def fehr_schmidt_model(
        self,
        own_payoff: float,
        other_payoff: float,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> float:
        """
        Fehr-Schmidt model of inequity aversion

        Args:
            own_payoff: Own monetary payoff
            other_payoff: Other player's payoff
            alpha: Disadvantageous inequity aversion
            beta: Advantageous inequity aversion

        Returns:
            Social utility including inequity aversion
        """
        if own_payoff >= other_payoff:
            # Advantageous inequity
            inequity_cost = beta * (own_payoff - other_payoff)
        else:
            # Disadvantageous inequity
            inequity_cost = alpha * (other_payoff - own_payoff)

        return own_payoff - inequity_cost

    def calculate_social_welfare(
        self, payoffs: list[float], social_welfare_function: str = "utilitarian"
    ) -> float:
        """
        Calculate social welfare using different welfare functions

        Args:
            payoffs: List of individual payoffs
            social_welfare_function: Type of welfare function

        Returns:
            Social welfare value
        """
        if social_welfare_function == "utilitarian":
            return float(np.sum(payoffs))
        elif social_welfare_function == "egalitarian":
            return float(np.min(payoffs))
        elif social_welfare_function == "rawlsian":
            return float(np.min(payoffs))
        elif social_welfare_function == "nash":
            return float(np.prod(payoffs) ** (1 / len(payoffs)))
        else:
            return float(np.sum(payoffs))  # Default to utilitarian


class TimePreferences:
    """
    Models of time preferences and discounting
    """

    def __init__(self) -> None:
        self.discount_functions: dict[str, Any] = {}

    def hyperbolic_discounting(self, delay: float, k: float = 1.0) -> float:
        """
        Hyperbolic discounting function

        Args:
            delay: Time delay in periods
            k: Discount rate parameter

        Returns:
            Discount factor
        """
        return 1 / (1 + k * delay)

    def quasi_hyperbolic_discounting(
        self, delay: float, beta: float = 0.7, delta: float = 0.9
    ) -> float:
        """
        Quasi-hyperbolic discounting (β-δ model)

        Args:
            delay: Time delay in periods
            beta: Present bias parameter
            delta: Long-run discount rate

        Returns:
            Discount factor
        """
        if delay == 0:
            return 1.0
        elif delay == 1:
            return beta * delta
        else:
            return float(beta * (delta**delay))

    def analyze_time_inconsistency(
        self,
        reward_sizes: list[float],
        delays: list[float],
        discount_params: dict[str, float],
    ) -> dict[str, Any]:
        """
        Analyze time inconsistency in intertemporal choice

        Args:
            reward_sizes: List of reward amounts
            delays: List of time delays
            discount_params: Discounting parameters

        Returns:
            Time inconsistency analysis
        """
        choices = []

        for i, (reward1, delay1) in enumerate(zip(reward_sizes, delays)):
            for j, (reward2, delay2) in enumerate(zip(reward_sizes, delays)):
                if i != j:
                    # Compare two options
                    if delay1 < delay2:
                        # Sooner smaller reward vs later larger reward
                        choice = self._analyze_soon_vs_later(
                            reward1, delay1, reward2, delay2, discount_params
                        )
                    else:
                        # Later larger reward vs sooner smaller reward
                        choice = self._analyze_later_vs_soon(
                            reward2, delay2, reward1, delay1, discount_params
                        )

                    choices.append(choice)

        return {
            "individual_choices": choices,
            "time_inconsistency_score": self._calculate_inconsistency_score(choices),
            "discount_parameters": discount_params,
        }

    def _analyze_soon_vs_later(
        self,
        small_reward: float,
        short_delay: float,
        large_reward: float,
        long_delay: float,
        params: dict[str, float],
    ) -> dict[str, Any]:
        """Analyze choice between sooner small reward and later large reward"""
        # Calculate present values
        pv_small = small_reward * self.hyperbolic_discounting(
            short_delay, params.get("k", 1.0)
        )
        pv_large = large_reward * self.hyperbolic_discounting(
            long_delay, params.get("k", 1.0)
        )

        # Choice based on present values
        chosen = "small_soon" if pv_small > pv_large else "large_later"

        return {
            "option1": {"reward": small_reward, "delay": short_delay, "pv": pv_small},
            "option2": {"reward": large_reward, "delay": long_delay, "pv": pv_large},
            "choice": chosen,
            "type": "soon_vs_later",
        }

    def _analyze_later_vs_soon(
        self,
        large_reward: float,
        long_delay: float,
        small_reward: float,
        short_delay: float,
        params: dict[str, float],
    ) -> dict[str, Any]:
        """Analyze choice between later large reward and sooner small reward"""
        pv_large = large_reward * self.hyperbolic_discounting(
            long_delay, params.get("k", 1.0)
        )
        pv_small = small_reward * self.hyperbolic_discounting(
            short_delay, params.get("k", 1.0)
        )

        chosen = "large_later" if pv_large > pv_small else "small_soon"

        return {
            "option1": {"reward": large_reward, "delay": long_delay, "pv": pv_large},
            "option2": {"reward": small_reward, "delay": short_delay, "pv": pv_small},
            "choice": chosen,
            "type": "later_vs_soon",
        }

    def _calculate_inconsistency_score(self, choices: list[dict[str, Any]]) -> float:
        """Calculate degree of time inconsistency"""
        # Count cases where choice is inconsistent with exponential discounting
        inconsistent_choices = 0

        for choice in choices:
            if choice["type"] == "soon_vs_later" and choice["choice"] == "small_soon":
                inconsistent_choices += 1
            elif choice["type"] == "later_vs_soon" and choice["choice"] == "small_soon":
                inconsistent_choices += 1

        return inconsistent_choices / len(choices) if choices else 0


class MentalAccounting:
    """
    Mental accounting and framing effects
    """

    def __init__(self) -> None:
        self.accounting_frames: dict[str, Any] = {}

    def analyze_framing_effect(
        self, problem_framing: dict[str, Any], choice_options: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze how problem framing affects choices

        Args:
            problem_framing: Description of how the problem is framed
            choice_options: Available choice options

        Returns:
            Framing effect analysis
        """
        # Simplified framing analysis
        # In practice, would use more sophisticated models

        frame_type = problem_framing.get("frame_type", "neutral")

        if frame_type == "gain_frame":
            # Emphasize potential gains
            choice_weights = [1.2, 0.8]  # Bias toward gain options
        elif frame_type == "loss_frame":
            # Emphasize potential losses
            choice_weights = [0.8, 1.2]  # Bias toward avoiding losses
        else:
            choice_weights = [1.0, 1.0]  # Neutral framing

        # Apply framing bias to option utilities
        adjusted_options = []
        for i, option in enumerate(choice_options):
            adjusted_utility = option.get("utility", 0) * choice_weights[i]
            option["adjusted_utility"] = adjusted_utility
            adjusted_options.append(option)

        # Choose option with highest adjusted utility
        chosen = max(adjusted_options, key=lambda x: x.get("adjusted_utility", 0))

        return {
            "chosen_option": chosen,
            "frame_type": frame_type,
            "framing_bias": choice_weights,
            "original_utilities": [opt.get("utility", 0) for opt in choice_options],
        }

    def mental_accounting_segregation(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze how people mentally segregate or integrate financial transactions

        Args:
            transactions: List of financial transactions

        Returns:
            Mental accounting analysis
        """
        # Categorize transactions by mental accounts
        accounts: dict[str, list[dict[str, Any]]] = {
            "income": [],
            "necessary_expenses": [],
            "luxury_expenses": [],
            "savings": [],
            "windfalls": [],
        }

        for transaction in transactions:
            category = transaction.get("category", "misc")
            amount = transaction.get("amount", 0)
            description = transaction.get("description", "")

            # Simple categorization logic
            if category == "income" or "salary" in description.lower():
                accounts["income"].append(transaction)
            elif category in ["food", "rent", "utilities"]:
                accounts["necessary_expenses"].append(transaction)
            elif category in ["entertainment", "vacation"]:
                accounts["luxury_expenses"].append(transaction)
            elif category == "savings":
                accounts["savings"].append(transaction)
            elif amount > 1000:  # Large unexpected income
                accounts["windfalls"].append(transaction)

        # Calculate account balances
        account_balances = {}
        for account_name, transactions_list in accounts.items():
            balance = sum(t.get("amount", 0) for t in transactions_list)
            account_balances[account_name] = balance

        return {
            "mental_accounts": accounts,
            "account_balances": account_balances,
            "total_segregated": len([t for acc in accounts.values() for t in acc]),
        }


class NudgeAnalysis:
    """
    Analysis of nudges and behavioral interventions
    """

    def __init__(self) -> None:
        self.nudge_effectiveness: dict[str, Any] = {}

    def evaluate_nudge_effectiveness(
        self,
        nudge_type: str,
        target_behavior: str,
        population_characteristics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate the effectiveness of different nudge types

        Args:
            nudge_type: Type of nudge ('default', 'social_norm', 'salience')
            target_behavior: Behavior being targeted
            population_characteristics: Characteristics of target population

        Returns:
            Nudge effectiveness analysis
        """
        # Simplified nudge effectiveness model

        effectiveness_scores = {
            "default_option": 0.7,
            "social_norm": 0.6,
            "salience": 0.5,
            "framing": 0.4,
            "commitment_device": 0.8,
        }

        base_effectiveness = effectiveness_scores.get(nudge_type, 0.5)

        # Adjust for population characteristics
        adjustment_factors = []

        if population_characteristics.get("education_level", "medium") == "high":
            adjustment_factors.append(1.1)  # More responsive to nudges

        if population_characteristics.get("age_group", "adult") == "young":
            adjustment_factors.append(1.05)  # Young people more responsive

        if population_characteristics.get("cultural_context") == "individualistic":
            adjustment_factors.append(0.95)  # Less responsive to social nudges

        # Apply adjustments
        adjusted_effectiveness = base_effectiveness * np.prod(adjustment_factors)

        return {
            "nudge_type": nudge_type,
            "target_behavior": target_behavior,
            "base_effectiveness": base_effectiveness,
            "adjusted_effectiveness": adjusted_effectiveness,
            "population_factors": adjustment_factors,
            "expected_impact": self._estimate_behavioral_impact(
                float(adjusted_effectiveness), target_behavior
            ),
        }

    def _estimate_behavioral_impact(
        self, effectiveness: float, behavior: str
    ) -> dict[str, Any]:
        """Estimate the impact of nudge on behavior"""
        # Simplified impact estimation
        impact_multipliers = {
            "saving": 1.5,
            "exercise": 1.3,
            "healthy_eating": 1.2,
            "energy_conservation": 1.4,
            "recycling": 1.6,
        }

        multiplier = impact_multipliers.get(behavior, 1.2)
        estimated_change = effectiveness * multiplier * 10  # Percentage points

        return {
            "estimated_behavior_change": estimated_change,
            "confidence_interval": [estimated_change * 0.8, estimated_change * 1.2],
            "impact_multiplier": multiplier,
        }


class BehavioralEconomicsEngine:
    """
    Main behavioral economics modeling engine
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize the behavioral economics engine.

        Args:
            config: Optional engine configuration.
            rng: Optional random generator threaded into stochastic choice
                models. When omitted, a fixed-seed generator is used so
                behaviour is deterministic by default.
        """
        self.config = config or {}
        self.prospect_theory = ProspectTheory()
        self.bounded_rationality = BoundedRationality(rng=rng)
        self.social_preferences = SocialPreferences()
        self.time_preferences = TimePreferences()
        self.mental_accounting = MentalAccounting()
        self.nudge_analysis = NudgeAnalysis()

    def analyze_behavioral_choice(
        self, choice_problem: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Comprehensive behavioral analysis of a choice problem

        Args:
            choice_problem: Dictionary defining the choice problem

        Returns:
            Behavioral analysis results
        """
        analysis_results = {}

        # Prospect theory analysis
        if "outcomes" in choice_problem and "probabilities" in choice_problem:
            outcomes = np.array(choice_problem["outcomes"])
            probabilities = np.array(choice_problem["probabilities"])

            prospect_value = self.prospect_theory.prospect_value(
                outcomes, probabilities
            )
            analysis_results["prospect_theory"] = {
                "prospect_value": prospect_value,
                "value_function": self.prospect_theory.value_function(outcomes),
                "weighted_probabilities": self.prospect_theory.probability_weighting(
                    probabilities
                ),
            }

        # Social preferences analysis
        if "payoffs" in choice_problem:
            payoffs = choice_problem["payoffs"]
            if len(payoffs) == 2:
                social_utility = self.social_preferences.fehr_schmidt_model(
                    payoffs[0], payoffs[1]
                )
                analysis_results["social_preferences"] = {
                    "social_utility": social_utility,
                    "fairness_concerns": social_utility < payoffs[0],
                }

        # Mental accounting analysis
        if "transactions" in choice_problem:
            mental_accounts = self.mental_accounting.mental_accounting_segregation(
                choice_problem["transactions"]
            )
            analysis_results["mental_accounting"] = mental_accounts

        return analysis_results

    def evaluate_nudge_intervention(
        self, nudge_design: dict[str, Any], target_population: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate a behavioral nudge intervention

        Args:
            nudge_design: Design of the nudge intervention
            target_population: Characteristics of target population

        Returns:
            Nudge evaluation results
        """
        nudge_type = nudge_design.get("type", "default")
        target_behavior = nudge_design.get("target_behavior", "unknown")

        effectiveness = self.nudge_analysis.evaluate_nudge_effectiveness(
            nudge_type, target_behavior, target_population
        )

        return {
            "nudge_effectiveness": effectiveness,
            "implementation_feasibility": self._assess_implementation_feasibility(
                nudge_design
            ),
            "ethical_considerations": self._assess_ethical_considerations(nudge_design),
            "cost_benefit_analysis": self._conduct_cost_benefit_analysis(
                effectiveness, nudge_design
            ),
        }

    def _assess_implementation_feasibility(
        self, nudge_design: dict[str, Any]
    ) -> dict[str, float]:
        """Assess feasibility of implementing the nudge"""
        return {
            "technical_feasibility": 0.9,
            "political_feasibility": 0.7,
            "administrative_feasibility": 0.8,
            "overall_feasibility": 0.8,
        }

    def _assess_ethical_considerations(
        self, nudge_design: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess ethical implications of the nudge"""
        return {
            "autonomy_respect": 0.8,
            "transparency": 0.7,
            "potential_harm": 0.2,
            "overall_ethical_score": 0.7,
        }

    def _conduct_cost_benefit_analysis(
        self, effectiveness: dict[str, Any], nudge_design: dict[str, Any]
    ) -> dict[str, float]:
        """Conduct cost-benefit analysis of nudge"""
        estimated_impact = effectiveness.get("expected_impact", {}).get(
            "estimated_behavior_change", 0
        )

        # Simplified CBA
        benefits = estimated_impact * 1000  # Assume $1000 per percentage point change
        costs = nudge_design.get("implementation_cost", 50000)

        return {
            "estimated_benefits": benefits,
            "implementation_costs": costs,
            "benefit_cost_ratio": benefits / costs if costs > 0 else float("inf"),
            "net_benefits": benefits - costs,
        }
