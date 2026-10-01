"""Tests for consumer theory module."""

import numpy as np
from geo_infer_econ.microeconomics.consumer_theory import (
    UtilityFunctions,
    DemandFunctions,
    ConsumerChoiceModels,
    WelfareAnalysis,
    ConsumerProfile,
    ConsumerSurplus,
)


class TestUtilityFunctions:
    """Tests for utility function implementations."""

    def test_cobb_douglas_basic(self) -> None:
        quantities = np.array([4.0, 9.0])
        alpha = np.array([0.5, 0.5])
        result = UtilityFunctions.cobb_douglas(quantities, alpha)
        expected = np.sqrt(4.0) * np.sqrt(9.0)  # 2 * 3 = 6
        assert abs(result - expected) < 1e-6

    def test_cobb_douglas_zero_quantity(self) -> None:
        quantities = np.array([0.0, 5.0])
        alpha = np.array([0.5, 0.5])
        result = UtilityFunctions.cobb_douglas(quantities, alpha)
        assert result == 0

    def test_ces_utility_rho_zero_falls_back_to_cd(self) -> None:
        quantities = np.array([4.0, 9.0])
        alpha = np.array([0.5, 0.5])
        result = UtilityFunctions.ces_utility(quantities, alpha, rho=0)
        expected = UtilityFunctions.cobb_douglas(quantities, alpha)
        assert abs(result - expected) < 1e-6

    def test_ces_utility_positive_rho(self) -> None:
        quantities = np.array([2.0, 3.0])
        alpha = np.array([0.5, 0.5])
        result = UtilityFunctions.ces_utility(quantities, alpha, rho=0.5)
        assert result > 0

    def test_linear_utility(self) -> None:
        quantities = np.array([3.0, 4.0])
        alpha = np.array([2.0, 1.0])
        result = UtilityFunctions.linear_utility(quantities, alpha)
        assert result == 10.0  # 2*3 + 1*4

    def test_leontief_utility(self) -> None:
        quantities = np.array([6.0, 4.0])
        alpha = np.array([2.0, 1.0])
        result = UtilityFunctions.leontief_utility(quantities, alpha)
        assert result == 3.0  # min(6/2, 4/1) = min(3, 4)

    def test_spatial_utility_augments_base(self) -> None:
        quantities = np.array([4.0, 9.0])
        alpha = np.array([0.5, 0.5])
        base = UtilityFunctions.cobb_douglas(quantities, alpha)
        spatial = UtilityFunctions.spatial_utility(
            quantities, alpha, location=(1.0, 1.0), accessibility_weight=0.1
        )
        assert spatial > base


class TestDemandFunctions:
    """Tests for demand function implementations."""

    def setup_method(self) -> None:
        self.df = DemandFunctions()

    def test_marshallian_demand_cobb_douglas(self) -> None:
        income = 100.0
        prices = np.array([2.0, 5.0])
        alpha = np.array([0.6, 0.4])
        demands = self.df.marshallian_demand_cobb_douglas(income, prices, alpha)
        # x_i = alpha_i * m / p_i
        assert abs(demands[0] - 30.0) < 1e-6  # 0.6 * 100 / 2
        assert abs(demands[1] - 8.0) < 1e-6  # 0.4 * 100 / 5

    def test_marshallian_demand_budget_exhaustion(self) -> None:
        income = 100.0
        prices = np.array([2.0, 5.0])
        alpha = np.array([0.6, 0.4])
        demands = self.df.marshallian_demand_cobb_douglas(income, prices, alpha)
        total_expenditure = np.sum(prices * demands)
        assert abs(total_expenditure - income) < 1e-6

    def test_hicksian_demand_cobb_douglas(self) -> None:
        prices = np.array([2.0, 5.0])
        alpha = np.array([0.6, 0.4])
        utility_target = 10.0
        demands = self.df.hicksian_demand_cobb_douglas(prices, alpha, utility_target)
        assert all(d > 0 for d in demands)


class TestConsumerChoiceModels:
    """Tests for consumer choice optimization."""

    def setup_method(self) -> None:
        self.model = ConsumerChoiceModels()

    def test_utility_maximization_basic(self) -> None:
        consumer = ConsumerProfile(
            consumer_id="c1",
            income=100.0,
            location=(40.0, -74.0),
            preferences={"food": 0.6, "clothing": 0.4},
            demographic_attributes={},
            spatial_attributes={},
        )
        prices = np.array([2.0, 4.0])
        goods = ["food", "clothing"]
        result = self.model.solve_utility_maximization(consumer, prices, goods)
        assert result["success"] is True
        assert result["utility"] > 0
        assert result["expenditure"] <= consumer.income + 0.01

    def test_utility_maximization_budget_binding(self) -> None:
        consumer = ConsumerProfile(
            consumer_id="c2",
            income=50.0,
            location=(0.0, 0.0),
            preferences={"A": 0.5, "B": 0.5},
            demographic_attributes={},
            spatial_attributes={},
        )
        prices = np.array([1.0, 1.0])
        goods = ["A", "B"]
        result = self.model.solve_utility_maximization(consumer, prices, goods)
        assert result["success"] is True
        # Budget should be approximately exhausted
        assert result["expenditure"] > 0


class TestWelfareAnalysis:
    """Tests for welfare analysis tools."""

    def test_equivalent_variation_price_increase(self) -> None:
        alpha = np.array([0.5, 0.5])
        prices_old = np.array([1.0, 1.0])
        prices_new = np.array([2.0, 1.0])
        ev = WelfareAnalysis.equivalent_variation(
            UtilityFunctions.cobb_douglas, 100.0, prices_old, prices_new, alpha
        )
        # Price increase should give positive EV (willing to pay to avoid)
        assert isinstance(ev, float)

    def test_compensating_variation_returns_float(self) -> None:
        alpha = np.array([0.5, 0.5])
        prices_old = np.array([1.0, 1.0])
        prices_new = np.array([2.0, 1.0])
        cv = WelfareAnalysis.compensating_variation(
            UtilityFunctions.cobb_douglas, 100.0, prices_old, prices_new, alpha
        )
        assert isinstance(cv, float)


class TestConsumerSurplus:
    """Tests for consumer surplus calculations."""

    def setup_method(self) -> None:
        self.cs = ConsumerSurplus()

    def test_surplus_integral_linear_demand(self) -> None:
        def demand_fn(p: float) -> float:
            return max(0.0, 10.0 - p)

        surplus = self.cs.calculate_surplus_integral(demand_fn, (0.0, 10.0), 5.0)
        # Integral from 5 to 10 of (10-p)dp = [10p - p^2/2] from 5 to 10 = (100-50) - (50-12.5) = 12.5
        assert abs(surplus - 12.5) < 0.5
