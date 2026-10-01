"""
Game Theory Module

Implements game theory applications in economics including:
- Strategic form games and Nash equilibrium
- Extensive form games and subgame perfect equilibrium
- Auction theory and mechanism design
- Evolutionary game theory
- Spatial games and location theory
- Bargaining theory and cooperative games
"""

import numpy as np
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass
from scipy.optimize import fsolve
import itertools


@dataclass
class Game:
    """Definition of a strategic form game"""

    players: list[str]
    actions: dict[str, list[Any]]
    payoffs: dict[tuple, dict[str, float]]  # (action_profile) -> {player: payoff}


@dataclass
class ExtensiveFormGame:
    """Definition of an extensive form game"""

    players: list[str]
    tree: dict[str, Any]  # Game tree structure
    payoffs: dict[str, dict[str, float]]  # Terminal node payoffs


class NashEquilibrium:
    """
    Nash equilibrium computation and analysis
    """

    def __init__(self) -> None:
        self.equilibria: list[Any] = []

    def find_nash_equilibrium(self, game: Game) -> list[dict[str, Any]]:
        """
        Find Nash equilibria in strategic form games

        Args:
            game: Strategic form game definition

        Returns:
            List of Nash equilibria
        """
        equilibria = []

        # Generate all possible action profiles
        action_spaces = [game.actions[player] for player in game.players]
        all_profiles = list(itertools.product(*action_spaces))

        for profile in all_profiles:
            profile_dict = dict(zip(game.players, profile))
            profile_tuple = tuple(profile)

            # Check if this is a Nash equilibrium
            is_equilibrium = True

            for i, player in enumerate(game.players):
                # Check if player can improve by unilateral deviation
                current_payoff = game.payoffs[profile_tuple][player]

                for alt_action in game.actions[player]:
                    if alt_action != profile[i]:
                        alt_profile = list(profile)
                        alt_profile[i] = alt_action
                        alt_profile_tuple = tuple(alt_profile)

                        if alt_profile_tuple in game.payoffs:
                            alt_payoff = game.payoffs[alt_profile_tuple][player]

                            if alt_payoff > current_payoff:
                                is_equilibrium = False
                                break

                if not is_equilibrium:
                    break

            if is_equilibrium:
                equilibria.append(
                    {
                        "strategy_profile": profile_dict,
                        "payoffs": game.payoffs[profile_tuple],
                    }
                )

        self.equilibria = equilibria
        return equilibria

    def compute_mixed_strategy_equilibrium(self, game: Game) -> dict[str, Any]:
        """
        Compute mixed strategy Nash equilibrium

        Args:
            game: Strategic form game

        Returns:
            Mixed strategy equilibrium
        """
        # Simplified implementation for 2-player games
        if len(game.players) != 2:
            return {
                "error": "Mixed strategy equilibrium only implemented for 2-player games"
            }

        player1, player2 = game.players
        actions1 = game.actions[player1]
        actions2 = game.actions[player2]

        n1, n2 = len(actions1), len(actions2)

        # Set up equations for mixed strategy equilibrium
        def equations(p: np.ndarray) -> Any:
            # p[0:n1] are probabilities for player 1
            # p[n1:n1+n2] are probabilities for player 2

            p1 = p[:n1]
            p2 = p[n1 : n1 + n2]

            # Constraints: probabilities sum to 1
            eq1 = np.sum(p1) - 1
            eq2 = np.sum(p2) - 1

            equations_list = [eq1, eq2]

            # Indifference conditions for each action
            for i in range(n1 - 1):  # One indifference condition per action except last
                eq = 0
                for j in range(n2):
                    profile1 = [0] * n1
                    profile1[i] = 1
                    profile2 = [0] * n2
                    profile2[j] = 1
                    profile_tuple1 = tuple(profile1 + profile2)

                    profile2_alt = [0] * n2
                    profile2_alt[-1] = 1  # Last action for player 2
                    profile_tuple2 = tuple(profile1 + profile2_alt)

                    if (
                        profile_tuple1 in game.payoffs
                        and profile_tuple2 in game.payoffs
                    ):
                        eq += p2[j] * (
                            game.payoffs[profile_tuple1][player1]
                            - game.payoffs[profile_tuple2][player1]
                        )

                equations_list.append(eq)

            for j in range(n2 - 1):  # Indifference for player 2
                eq = 0
                for i in range(n1):
                    profile1 = [0] * n1
                    profile1[i] = 1
                    profile2 = [0] * n2
                    profile2[j] = 1
                    profile_tuple1 = tuple(profile1 + profile2)

                    profile1_alt = [0] * n1
                    profile1_alt[-1] = 1  # Last action for player 1
                    profile_tuple2 = tuple(profile1_alt + profile2)

                    if (
                        profile_tuple1 in game.payoffs
                        and profile_tuple2 in game.payoffs
                    ):
                        eq += p1[i] * (
                            game.payoffs[profile_tuple1][player2]
                            - game.payoffs[profile_tuple2][player2]
                        )

                equations_list.append(eq)

            return equations_list

        # Initial guess (uniform mixed strategies)
        p0 = np.ones(n1 + n2) / (n1 + n2)

        # Solve system of equations
        solution = fsolve(equations, p0)

        if len(solution) == n1 + n2:
            p1_sol = solution[:n1]
            p2_sol = solution[n1 : n1 + n2]

            # Normalize probabilities
            p1_sol = p1_sol / np.sum(p1_sol) if np.sum(p1_sol) > 0 else p1_sol
            p2_sol = p2_sol / np.sum(p2_sol) if np.sum(p2_sol) > 0 else p2_sol

            return {
                "player1_mixed_strategy": dict(zip(actions1, p1_sol)),
                "player2_mixed_strategy": dict(zip(actions2, p2_sol)),
                "equilibrium_type": "mixed",
            }

        return {"error": "Failed to find mixed strategy equilibrium"}


class AuctionTheory:
    """
    Auction theory and mechanism design
    """

    def __init__(self) -> None:
        self.auction_results: dict[str, Any] = {}

    def analyze_first_price_auction(
        self, values: list[float], n_bidders: int
    ) -> dict[str, Any]:
        """
        Analyze first-price sealed-bid auction

        Args:
            values: List of bidder valuations
            n_bidders: Number of bidders

        Returns:
            Auction analysis results
        """
        # In first-price auction, optimal bid is v_i * (n-1)/n for risk-neutral bidders
        optimal_bids = [v * (n_bidders - 1) / n_bidders for v in values]

        # Expected revenue
        expected_revenue = np.mean(optimal_bids)

        # Winner's curse analysis (simplified)
        winner_payoff = max(optimal_bids) - values[np.argmax(optimal_bids)]

        return {
            "optimal_bids": optimal_bids,
            "expected_revenue": expected_revenue,
            "winner_valuation": values[np.argmax(optimal_bids)],
            "winner_payoff": winner_payoff,
            "auction_format": "first_price_sealed_bid",
        }

    def analyze_second_price_auction(self, values: list[float]) -> dict[str, Any]:
        """
        Analyze second-price sealed-bid auction (Vickrey auction)

        Args:
            values: List of bidder valuations

        Returns:
            Auction analysis results
        """
        # In second-price auction, optimal strategy is to bid true valuation
        optimal_bids = values.copy()

        # Winner pays second-highest bid
        sorted_values = sorted(values, reverse=True)
        winning_price = sorted_values[1] if len(sorted_values) > 1 else 0

        winner_index = np.argmax(values)
        winner_payoff = values[winner_index] - winning_price

        return {
            "optimal_bids": optimal_bids,
            "winning_price": winning_price,
            "winner_index": winner_index,
            "winner_payoff": winner_payoff,
            "auction_format": "second_price_sealed_bid",
        }


class EvolutionaryGames:
    """
    Evolutionary game theory and population dynamics
    """

    def __init__(self) -> None:
        self.dynamics_results: dict[str, Any] = {}

    def replicator_dynamics(
        self,
        payoff_matrix: np.ndarray,
        initial_frequencies: np.ndarray,
        time_steps: int = 100,
    ) -> dict[str, Any]:
        """
        Simulate replicator dynamics for evolutionary games

        Args:
            payoff_matrix: Payoff matrix for 2-player game
            initial_frequencies: Initial strategy frequencies
            time_steps: Number of time steps to simulate

        Returns:
            Replicator dynamics results
        """
        n_strategies = len(payoff_matrix)
        frequencies = [initial_frequencies.copy()]

        dt = 0.1

        for _ in range(time_steps):
            current_freq = frequencies[-1]

            # Average payoffs for each strategy
            avg_payoffs = np.zeros(n_strategies)
            for i in range(n_strategies):
                for j in range(n_strategies):
                    avg_payoffs[i] += payoff_matrix[i, j] * current_freq[j]

            # Overall average payoff
            avg_payoff_total = np.sum(avg_payoffs * current_freq)

            # Replicator dynamics
            new_freq = current_freq.copy()
            for i in range(n_strategies):
                if avg_payoff_total > 0:
                    new_freq[i] = current_freq[i] * (
                        1 + dt * (avg_payoffs[i] - avg_payoff_total) / avg_payoff_total
                    )
                else:
                    new_freq[i] = current_freq[i]

            # Normalize frequencies
            total = np.sum(new_freq)
            if total > 0:
                new_freq = new_freq / total

            frequencies.append(new_freq)

        return {
            "frequency_paths": frequencies,
            "converged_frequencies": frequencies[-1],
            "time_steps": time_steps,
            "equilibrium_type": self._classify_equilibrium(
                frequencies[-1], payoff_matrix
            ),
        }

    def _classify_equilibrium(
        self, frequencies: np.ndarray, payoff_matrix: np.ndarray
    ) -> str:
        """Classify the type of equilibrium reached"""
        # Check if it's a Nash equilibrium of the underlying game
        n = len(frequencies)

        # For 2x2 games, check if frequencies satisfy equilibrium conditions
        if n == 2:
            # Simplified check - would need full equilibrium verification
            if frequencies[0] > 0.9 or frequencies[0] < 0.1:
                return "pure_strategy_equilibrium"
            else:
                return "mixed_strategy_equilibrium"

        return "unknown"


class SpatialGames:
    """
    Spatial game theory and location models
    """

    def __init__(self) -> None:
        self.spatial_results: dict[str, Any] = {}

    def location_game_analysis(
        self, locations: np.ndarray, demand_function: Callable
    ) -> dict[str, Any]:
        """
        Analyze location choice in spatial competition (Hotelling model)

        Args:
            locations: Array of possible location coordinates
            demand_function: Function mapping distance to demand

        Returns:
            Location game analysis
        """
        # Simplified Hotelling duopoly model
        n_locations = len(locations)

        # For two firms, equilibrium is at center
        if n_locations == 2:
            equilibrium_locations = [np.mean(locations)]
        else:
            # Multi-firm case (simplified)
            equilibrium_locations = [locations[i] for i in range(min(3, n_locations))]

        # Calculate market shares
        market_shares = self._calculate_spatial_market_shares(
            equilibrium_locations, locations, demand_function
        )

        return {
            "equilibrium_locations": equilibrium_locations,
            "market_shares": market_shares,
            "model_type": "hotelling",
        }

    def _calculate_spatial_market_shares(
        self,
        firm_locations: list[float],
        consumer_locations: np.ndarray,
        demand_function: Callable,
    ) -> dict[int, float]:
        """Calculate market shares in spatial competition"""
        market_shares = {}

        for i, firm_loc in enumerate(firm_locations):
            shares = []
            for consumer_loc in consumer_locations:
                # Distance to this firm vs others
                dist_to_this = abs(consumer_loc - firm_loc)
                dist_to_others = [
                    abs(consumer_loc - other_loc)
                    for other_loc in firm_locations
                    if other_loc != firm_loc
                ]

                if dist_to_others:
                    min_dist_other = min(dist_to_others)
                    if dist_to_this < min_dist_other:
                        shares.append(1.0)  # Consumer prefers this firm
                    elif dist_to_this == min_dist_other:
                        shares.append(0.5)  # Tie
                    else:
                        shares.append(0.0)  # Consumer prefers other firm

            market_shares[i] = float(np.mean(shares)) if shares else 0.0

        return market_shares


class BargainingTheory:
    """
    Bargaining theory and cooperative games
    """

    def __init__(self) -> None:
        self.bargaining_solutions: dict[str, Any] = {}

    def nash_bargaining_solution(
        self,
        utility_possibilities: np.ndarray,
        disagreement_point: np.ndarray,
        risk_aversion: float = 1.0,
    ) -> dict[str, Any]:
        """
        Compute Nash bargaining solution

        Args:
            utility_possibilities: Array of feasible utility pairs
            disagreement_point: Disagreement point utilities
            risk_aversion: Risk aversion parameter

        Returns:
            Nash bargaining solution
        """
        # Nash bargaining: maximize (u1 - d1)^a * (u2 - d2)^(1-a)
        # For a=0.5, this is (u1-d1)*(u2-d2)

        if risk_aversion == 0.5:
            # Symmetric Nash bargaining
            products = (utility_possibilities[:, 0] - disagreement_point[0]) * (
                utility_possibilities[:, 1] - disagreement_point[1]
            )

            optimal_idx = np.argmax(products)
            optimal_utilities = utility_possibilities[optimal_idx]

        else:
            # Asymmetric Nash bargaining
            products = np.power(
                utility_possibilities[:, 0] - disagreement_point[0], risk_aversion
            ) * np.power(
                utility_possibilities[:, 1] - disagreement_point[1], 1 - risk_aversion
            )

            optimal_idx = np.argmax(products)
            optimal_utilities = utility_possibilities[optimal_idx]

        return {
            "optimal_utilities": optimal_utilities,
            "disagreement_point": disagreement_point,
            "risk_aversion_parameter": risk_aversion,
            "pareto_frontier_point": optimal_idx,
        }


class GameTheoryModels:
    """
    Main game theory modeling class
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.nash_equilibrium = NashEquilibrium()
        self.auction_theory = AuctionTheory()
        self.evolutionary_games = EvolutionaryGames()
        self.spatial_games = SpatialGames()
        self.bargaining_theory = BargainingTheory()

    def analyze_strategic_game(self, game: Game) -> dict[str, Any]:
        """
        Comprehensive analysis of strategic form game

        Args:
            game: Strategic form game definition

        Returns:
            Game analysis results
        """
        # Find pure strategy Nash equilibria
        pure_equilibria = self.nash_equilibrium.find_nash_equilibrium(game)

        # Analyze mixed strategy equilibrium (for 2-player games)
        mixed_equilibrium = None
        if len(game.players) == 2:
            mixed_equilibrium = (
                self.nash_equilibrium.compute_mixed_strategy_equilibrium(game)
            )

        return {
            "pure_strategy_equilibria": pure_equilibria,
            "mixed_strategy_equilibrium": mixed_equilibrium,
            "game_type": "strategic_form",
            "n_players": len(game.players),
            "n_actions": {player: len(game.actions[player]) for player in game.players},
        }

    def analyze_auction_game(
        self,
        auction_type: str,
        valuations: list[float],
        n_bidders: int | None = None,
    ) -> dict[str, Any]:
        """
        Analyze auction game

        Args:
            auction_type: Type of auction ('first_price', 'second_price', 'english', 'dutch')
            valuations: List of bidder valuations
            n_bidders: Number of bidders (for some auction types)

        Returns:
            Auction analysis results
        """
        if auction_type == "first_price":
            if n_bidders is None:
                return {"error": "Number of bidders required for first-price auction"}
            return self.auction_theory.analyze_first_price_auction(
                valuations, n_bidders
            )

        elif auction_type == "second_price":
            return self.auction_theory.analyze_second_price_auction(valuations)

        else:
            return {"error": f"Auction type {auction_type} is unsupported"}

    def analyze_evolutionary_game(
        self, payoff_matrix: np.ndarray, initial_frequencies: np.ndarray
    ) -> dict[str, Any]:
        """
        Analyze evolutionary game dynamics

        Args:
            payoff_matrix: Payoff matrix for the game
            initial_frequencies: Initial strategy frequencies

        Returns:
            Evolutionary dynamics results
        """
        return self.evolutionary_games.replicator_dynamics(
            payoff_matrix, initial_frequencies
        )

    def analyze_location_game(
        self, locations: np.ndarray, demand_function: Callable
    ) -> dict[str, Any]:
        """
        Analyze spatial location game

        Args:
            locations: Array of possible locations
            demand_function: Demand function based on distance

        Returns:
            Location game analysis
        """
        return self.spatial_games.location_game_analysis(locations, demand_function)
