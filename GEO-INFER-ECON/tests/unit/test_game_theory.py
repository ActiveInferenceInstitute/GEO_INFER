"""Tests for game theory module."""

import numpy as np
from geo_infer_econ.microeconomics.game_theory import (
    Game,
    NashEquilibrium,
    AuctionTheory,
    EvolutionaryGames,
    SpatialGames,
    BargainingTheory,
    GameTheoryModels,
)


class TestNashEquilibrium:
    """Tests for Nash equilibrium computation."""

    def setup_method(self) -> None:
        self.ne = NashEquilibrium()

    def test_prisoners_dilemma_equilibrium(self) -> None:
        """Prisoner's dilemma has one Nash equilibrium: (Defect, Defect)."""
        game = Game(
            players=["P1", "P2"],
            actions={"P1": ["cooperate", "defect"], "P2": ["cooperate", "defect"]},
            payoffs={
                ("cooperate", "cooperate"): {"P1": 3.0, "P2": 3.0},
                ("cooperate", "defect"): {"P1": 0.0, "P2": 5.0},
                ("defect", "cooperate"): {"P1": 5.0, "P2": 0.0},
                ("defect", "defect"): {"P1": 1.0, "P2": 1.0},
            },
        )
        equilibria = self.ne.find_nash_equilibrium(game)
        assert len(equilibria) == 1
        assert equilibria[0]["strategy_profile"]["P1"] == "defect"
        assert equilibria[0]["strategy_profile"]["P2"] == "defect"

    def test_coordination_game_multiple_equilibria(self) -> None:
        """Coordination game has two pure-strategy Nash equilibria."""
        game = Game(
            players=["P1", "P2"],
            actions={"P1": ["A", "B"], "P2": ["A", "B"]},
            payoffs={
                ("A", "A"): {"P1": 2.0, "P2": 2.0},
                ("A", "B"): {"P1": 0.0, "P2": 0.0},
                ("B", "A"): {"P1": 0.0, "P2": 0.0},
                ("B", "B"): {"P1": 1.0, "P2": 1.0},
            },
        )
        equilibria = self.ne.find_nash_equilibrium(game)
        assert len(equilibria) == 2

    def test_no_equilibrium_stored(self) -> None:
        """Initially no equilibria are stored."""
        assert self.ne.equilibria == []


class TestAuctionTheory:
    """Tests for auction theory analysis."""

    def setup_method(self) -> None:
        self.auction = AuctionTheory()

    def test_first_price_auction_optimal_bids(self) -> None:
        values = [100.0, 80.0, 60.0]
        result = self.auction.analyze_first_price_auction(values, n_bidders=3)
        # In first-price auction: bid = v * (n-1)/n
        expected_bid_0 = 100.0 * 2 / 3
        assert abs(result["optimal_bids"][0] - expected_bid_0) < 1e-6
        assert result["auction_format"] == "first_price_sealed_bid"

    def test_second_price_auction_truthful_bidding(self) -> None:
        values = [100.0, 80.0, 60.0]
        result = self.auction.analyze_second_price_auction(values)
        assert result["optimal_bids"] == values
        assert result["winning_price"] == 80.0
        assert result["winner_payoff"] == 20.0

    def test_second_price_single_bidder(self) -> None:
        values = [100.0]
        result = self.auction.analyze_second_price_auction(values)
        assert result["winning_price"] == 0


class TestEvolutionaryGames:
    """Tests for evolutionary game dynamics."""

    def setup_method(self) -> None:
        self.evo = EvolutionaryGames()

    def test_replicator_dynamics_hawk_dove(self) -> None:
        payoff_matrix = np.array([[0.0, 3.0], [1.0, 2.0]])
        initial_freq = np.array([0.5, 0.5])
        result = self.evo.replicator_dynamics(
            payoff_matrix, initial_freq, time_steps=200
        )
        assert "converged_frequencies" in result
        assert len(result["converged_frequencies"]) == 2
        assert abs(sum(result["converged_frequencies"]) - 1.0) < 0.01

    def test_replicator_dynamics_dominant_strategy(self) -> None:
        payoff_matrix = np.array([[5.0, 5.0], [1.0, 1.0]])
        initial_freq = np.array([0.3, 0.7])
        result = self.evo.replicator_dynamics(
            payoff_matrix, initial_freq, time_steps=200
        )
        # Strategy 0 dominates, should converge toward high freq
        assert result["converged_frequencies"][0] > 0.5

    def test_frequency_paths_recorded(self) -> None:
        payoff_matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        initial_freq = np.array([0.5, 0.5])
        result = self.evo.replicator_dynamics(
            payoff_matrix, initial_freq, time_steps=10
        )
        assert len(result["frequency_paths"]) == 11  # initial + 10 steps


class TestSpatialGames:
    """Tests for spatial game theory."""

    def setup_method(self) -> None:
        self.sg = SpatialGames()

    def test_location_game_analysis(self) -> None:
        locations = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        def demand_fn(d: float) -> float:
            return max(0.0, 1.0 - d)

        result = self.sg.location_game_analysis(locations, demand_fn)
        assert "equilibrium_locations" in result
        assert "market_shares" in result
        assert result["model_type"] == "hotelling"


class TestBargainingTheory:
    """Tests for bargaining theory."""

    def setup_method(self) -> None:
        self.bt = BargainingTheory()

    def test_nash_bargaining_symmetric(self) -> None:
        utility_possibilities = np.array(
            [
                [0.0, 10.0],
                [2.0, 8.0],
                [5.0, 5.0],
                [8.0, 2.0],
                [10.0, 0.0],
            ]
        )
        disagreement = np.array([0.0, 0.0])
        result = self.bt.nash_bargaining_solution(
            utility_possibilities, disagreement, risk_aversion=0.5
        )
        # Symmetric case: product is maximized at (5, 5)
        assert result["optimal_utilities"][0] == 5.0
        assert result["optimal_utilities"][1] == 5.0

    def test_nash_bargaining_asymmetric(self) -> None:
        utility_possibilities = np.array(
            [
                [1.0, 9.0],
                [3.0, 7.0],
                [5.0, 5.0],
                [7.0, 3.0],
                [9.0, 1.0],
            ]
        )
        disagreement = np.array([0.0, 0.0])
        result = self.bt.nash_bargaining_solution(
            utility_possibilities, disagreement, risk_aversion=0.8
        )
        assert "optimal_utilities" in result


class TestGameTheoryModels:
    """Tests for the main GameTheoryModels facade."""

    def setup_method(self) -> None:
        self.gtm = GameTheoryModels()

    def test_analyze_strategic_game(self) -> None:
        game = Game(
            players=["P1", "P2"],
            actions={"P1": ["A", "B"], "P2": ["A", "B"]},
            payoffs={
                ("A", "A"): {"P1": 2.0, "P2": 2.0},
                ("A", "B"): {"P1": 0.0, "P2": 0.0},
                ("B", "A"): {"P1": 0.0, "P2": 0.0},
                ("B", "B"): {"P1": 1.0, "P2": 1.0},
            },
        )
        result = self.gtm.analyze_strategic_game(game)
        assert "pure_strategy_equilibria" in result
        assert result["n_players"] == 2

    def test_analyze_auction_game_first_price(self) -> None:
        result = self.gtm.analyze_auction_game(
            "first_price", [100.0, 80.0], n_bidders=2
        )
        assert "optimal_bids" in result

    def test_analyze_auction_game_second_price(self) -> None:
        result = self.gtm.analyze_auction_game("second_price", [100.0, 80.0, 60.0])
        assert result["winning_price"] == 80.0

    def test_analyze_evolutionary_game(self) -> None:
        payoff = np.array([[3.0, 0.0], [5.0, 1.0]])
        freq = np.array([0.5, 0.5])
        result = self.gtm.analyze_evolutionary_game(payoff, freq)
        assert "converged_frequencies" in result
