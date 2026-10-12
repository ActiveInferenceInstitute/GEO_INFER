"""
Consumer Theory Module

Implements comprehensive consumer theory models including:
- Utility maximization and demand functions
- Consumer choice with spatial considerations
- Welfare analysis and consumer surplus
- Revealed preference analysis
- Spatial consumer behavior modeling
"""

import numpy as np
import pandas as pd
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass
import geopandas as gpd
from scipy.optimize import minimize, minimize_scalar


@dataclass
class ConsumerProfile:
    """Profile of an individual consumer with spatial attributes"""

    consumer_id: str
    income: float
    location: tuple[float, float]  # (lat, lon)
    preferences: dict[str, float]
    demographic_attributes: dict[str, Any]
    spatial_attributes: dict[str, float]  # accessibility, distance to markets, etc.


class UtilityFunctions:
    """
    Collection of utility function implementations for consumer theory
    """

    @staticmethod
    def cobb_douglas(quantities: np.ndarray, alpha: np.ndarray) -> float:
        """
        Cobb-Douglas utility function: U = ∏(x_i^α_i)

        Args:
            quantities: Array of quantities consumed
            alpha: Array of preference parameters (should sum to 1)

        Returns:
            Utility value
        """
        if np.any(quantities <= 0):
            return 0
        return float(np.prod(np.power(quantities, alpha)))

    @staticmethod
    def ces_utility(quantities: np.ndarray, alpha: np.ndarray, rho: float) -> float:
        """
        Constant Elasticity of Substitution (CES) utility function
        U = (∑(α_i * x_i^ρ))^(1/ρ)

        Args:
            quantities: Array of quantities consumed
            alpha: Array of preference parameters
            rho: Substitution parameter

        Returns:
            Utility value
        """
        if rho == 0:
            return UtilityFunctions.cobb_douglas(quantities, alpha)

        ces_sum = np.sum(alpha * np.power(quantities, rho))
        return float(np.power(ces_sum, 1 / rho)) if ces_sum > 0 else 0

    @staticmethod
    def linear_utility(quantities: np.ndarray, alpha: np.ndarray) -> float:
        """
        Linear utility function: U = ∑(α_i * x_i)
        Perfect substitutes case
        """
        return float(np.sum(alpha * quantities))

    @staticmethod
    def leontief_utility(quantities: np.ndarray, alpha: np.ndarray) -> float:
        """
        Leontief utility function: U = min(x_i/α_i)
        Perfect complements case
        """
        return float(np.min(quantities / alpha))

    @staticmethod
    def spatial_utility(
        quantities: np.ndarray,
        alpha: np.ndarray,
        location: tuple[float, float],
        accessibility_weight: float = 0.1,
    ) -> float:
        """
        Spatial utility function incorporating location-based preferences

        Args:
            quantities: Array of quantities consumed
            alpha: Array of preference parameters
            location: Consumer location (lat, lon)
            accessibility_weight: Weight for spatial accessibility component

        Returns:
            Spatial utility value
        """
        base_utility = UtilityFunctions.cobb_douglas(quantities, alpha)

        # Simple accessibility modifier (can be made more sophisticated)
        accessibility_factor = 1 + accessibility_weight * np.sum(location)

        return float(base_utility * accessibility_factor)


class DemandFunctions:
    """
    Implementation of various demand function derivations and estimations
    """

    def __init__(self, utility_function: str = "cobb_douglas"):
        self.utility_function = utility_function
        self.estimated_parameters: dict[str, Any] = {}

    def marshallian_demand_cobb_douglas(
        self, income: float, prices: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """
        Marshallian (uncompensated) demand for Cobb-Douglas utility
        x_i = (α_i * m) / p_i

        Args:
            income: Consumer income
            prices: Array of prices
            alpha: Array of preference parameters

        Returns:
            Array of optimal quantities
        """
        return cast(np.ndarray, (alpha * income) / prices)

    def hicksian_demand_cobb_douglas(
        self, prices: np.ndarray, alpha: np.ndarray, utility_target: float
    ) -> np.ndarray:
        """
        Hicksian (compensated) demand for Cobb-Douglas utility

        Args:
            prices: Array of prices
            alpha: Array of preference parameters
            utility_target: Target utility level

        Returns:
            Array of optimal quantities
        """
        # For Cobb-Douglas: x_i = (α_i/p_i) * (U / ∏(α_j^α_j / p_j^α_j))
        price_index = np.prod(np.power(prices / alpha, alpha))
        expenditure = utility_target * price_index

        return cast(np.ndarray, (alpha * expenditure) / prices)

    def estimate_demand_system(
        self, data: pd.DataFrame, method: str = "ols"
    ) -> dict[str, Any]:
        """
        Estimate demand system from consumer data

        Args:
            data: DataFrame with columns for quantities, prices, income, demographics
            method: Estimation method ('ols', 'sur', 'aids')

        Returns:
            Dictionary with estimated parameters and diagnostics
        """
        if method == "aids":
            return self._estimate_aids_system(data)
        elif method == "sur":
            return self._estimate_sur_system(data)
        else:
            return self._estimate_ols_system(data)

    def _estimate_aids_system(self, data: pd.DataFrame) -> dict[str, Any]:
        """Estimate Almost Ideal Demand System (AIDS).

        The AIDS share equation for good i is

            w_i = alpha_i + sum_j gamma_ij * ln(p_j) + beta_i * ln(m / P)

        where P is the Stone price index, ln(P) = sum_j w_j * ln(p_j),
        computed with observed (approximate) shares. Each share equation is
        estimated by OLS; the adding-up restriction sum_i alpha_i = 1 and
        the symmetry restrictions on gamma are not imposed.

        Expenditure and uncompensated price elasticities follow the standard
        AIDS approximations:

            e_i    = 1 + beta_i / w_i
            eps_ij = -delta_ij + gamma_ij / w_i - beta_i * w_j / w_i

        Args:
            data: DataFrame with `quantity_good_<k>` and `price_<k>` columns
                plus an `income` column, one row per observation.

        Returns:
            Dict with per-good parameters, elasticities, and diagnostics.
        """
        goods = [
            col.replace("quantity_", "")
            for col in data.columns
            if col.startswith("quantity_good_")
        ]
        if len(goods) < 2 or "income" not in data.columns:
            return {"method": "AIDS", "status": "insufficient_data"}

        quantities = data[[f"quantity_{g}" for g in goods]].to_numpy(dtype=float)
        prices = data[[f"price_{g}" for g in goods]].to_numpy(dtype=float)

        expenditure = quantities * prices
        total_expenditure = np.sum(expenditure, axis=1)
        if not np.all(total_expenditure > 0):
            raise ValueError("Total expenditure must be positive in every row")
        shares = expenditure / total_expenditure[:, None]
        mean_shares = np.mean(shares, axis=0)

        # Stone price index using approximate shares
        log_prices = np.log(prices)
        log_price_index = np.sum(shares * log_prices, axis=1)
        log_real_expenditure = np.log(total_expenditure) - log_price_index

        X = np.column_stack([np.ones(len(data)), log_prices, log_real_expenditure])

        k = len(goods)
        parameters: dict[str, dict[str, Any]] = {}
        elasticities: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, Any] = {}

        for i, good in enumerate(goods):
            w = shares[:, i]
            coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
            residuals = w - X @ coef
            tss = float(np.sum((w - np.mean(w)) ** 2))
            r_squared = 1 - np.sum(residuals**2) / tss if tss > 0 else 0.0

            alpha_i = float(coef[0])
            gamma_i = coef[1 : k + 1].astype(float)
            beta_i = float(coef[k + 1])
            mean_share = float(mean_shares[i])

            parameters[good] = {
                "alpha": alpha_i,
                "gamma": gamma_i,
                "beta": beta_i,
                "mean_budget_share": mean_share,
            }
            if mean_share > 0:
                eps = (
                    -np.eye(k)
                    + gamma_i / mean_share
                    - np.outer(np.full(k, beta_i), mean_shares) / mean_share
                )
                elasticities[good] = {
                    "expenditure": 1.0 + beta_i / mean_share,
                    "uncompensated_price": eps.tolist(),
                }
            else:
                elasticities[good] = {
                    "expenditure": float("nan"),
                    "uncompensated_price": np.full((k, k), float("nan")).tolist(),
                }
            diagnostics[good] = {"r_squared": float(r_squared)}

        return {
            "method": "AIDS",
            "goods": goods,
            "parameters": parameters,
            "elasticities": elasticities,
            "diagnostics": diagnostics,
        }

    def _estimate_ols_system(self, data: pd.DataFrame) -> dict[str, Any]:
        """Simple OLS estimation of demand functions"""
        from sklearn.linear_model import LinearRegression

        results = {}

        # Estimate each demand equation separately
        for good in ["good_1", "good_2"]:  # Example goods
            if f"quantity_{good}" in data.columns:
                X = data[["income", f"price_{good}"]].values
                y = data[f"quantity_{good}"].values

                model = LinearRegression()
                model.fit(X, y)

                results[good] = {
                    "coefficients": model.coef_,
                    "intercept": model.intercept_,
                    "r_squared": model.score(X, y),
                }

        return results

    def _estimate_sur_system(self, data: pd.DataFrame) -> dict[str, Any]:
        """Seemingly Unrelated Regression (Zellner two-step GLS).

        Step 1 estimates each equation by OLS to recover the residual
        covariance matrix Sigma; step 2 re-estimates the stacked system by
        GLS, beta = (X' (Sigma^-1 (x) I) X)^-1 X' (Sigma^-1 (x) I) y.
        With identical regressors in every equation the GLS estimator
        collapses to equation-by-equation OLS, so equations here use
        per-good regressors (own price + income) to make the SUR step
        informative.
        """
        goods = [
            col.replace("quantity_", "")
            for col in data.columns
            if col.startswith("quantity_good_")
        ]
        if len(goods) < 2 or "income" not in data.columns:
            return {"method": "SUR", "status": "insufficient_goods"}

        y_list = [data[f"quantity_{g}"].to_numpy(dtype=float) for g in goods]
        X_list = [data[["income", f"price_{g}"]].to_numpy(dtype=float) for g in goods]
        n = len(data)
        k = len(goods)

        # First step: equation-by-equation OLS
        betas: list[np.ndarray] = []
        residuals: list[np.ndarray] = []
        for y_i, X_i in zip(y_list, X_list):
            b = np.linalg.lstsq(X_i, y_i, rcond=None)[0]
            betas.append(b)
            residuals.append(y_i - X_i @ b)

        # Residual covariance matrix Sigma (with small-sample dof correction)
        dof = max(n - X_list[0].shape[1], 1)
        sigma = np.column_stack(residuals).T @ np.column_stack(residuals) / dof
        sigma_inv = np.linalg.inv(sigma)

        # Second step: stacked GLS with Omega = Sigma^-1 (x) I and a
        # block-diagonal regressor matrix (one block per equation)
        y_stack = np.concatenate(y_list)
        X_full = np.zeros((k * n, sum(X_i.shape[1] for X_i in X_list)))
        omega_inv = np.zeros((k * n, k * n))
        col_offset = 0
        for i, X_i in enumerate(X_list):
            X_full[i * n : (i + 1) * n, col_offset : col_offset + X_i.shape[1]] = X_i
            col_offset += X_i.shape[1]
        for i in range(k):
            for j in range(k):
                omega_inv[i * n : (i + 1) * n, j * n : (j + 1) * n] = sigma_inv[
                    i, j
                ] * np.eye(n)
        gls_betas = (
            np.linalg.inv(X_full.T @ omega_inv @ X_full)
            @ X_full.T
            @ omega_inv
            @ y_stack
        )

        system_results: dict[str, dict[str, Any]] = {}
        col_offset = 0
        for i, good in enumerate(goods):
            k_i = X_list[i].shape[1]
            b = gls_betas[col_offset : col_offset + k_i]
            col_offset += k_i
            resid_i = y_list[i] - X_list[i] @ b
            tss = np.sum((y_list[i] - np.mean(y_list[i])) ** 2)
            r_squared = 1 - np.sum(resid_i**2) / tss if tss > 0 else 0.0
            system_results[good] = {
                "coefficients": b,
                "residuals": resid_i,
                "r_squared": float(r_squared),
            }

        return {
            "method": "SUR",
            "system_results": system_results,
            "goods_analyzed": goods,
            "residual_covariance": sigma,
        }


class ConsumerChoiceModels:
    """
    Consumer choice modeling with spatial considerations
    """

    def __init__(self, utility_function: Callable[..., Any] | None = None):
        self.utility_function = (
            utility_function
            if utility_function is not None
            else UtilityFunctions.cobb_douglas
        )
        self.spatial_weights: dict[str, Any] = {}

    def solve_utility_maximization(
        self, consumer: ConsumerProfile, prices: np.ndarray, goods: list[str]
    ) -> dict[str, Any]:
        """
        Solve consumer utility maximization problem

        Args:
            consumer: Consumer profile with income and preferences
            prices: Array of market prices
            goods: List of good names

        Returns:
            Dictionary with optimal quantities, utility, and expenditure
        """

        def objective(quantities: np.ndarray) -> float:
            """Negative utility to minimize"""
            alpha = np.array([consumer.preferences.get(good, 1.0) for good in goods])
            return -self.utility_function(quantities, alpha)

        def budget_constraint(quantities: np.ndarray) -> float:
            """Budget constraint: sum(p_i * x_i) <= income"""
            return float(consumer.income - np.sum(prices * quantities))

        # Non-negativity constraints
        bounds = [(0, None) for _ in goods]

        # Budget constraint
        constraints = {"type": "ineq", "fun": budget_constraint}

        # Initial guess
        x0 = np.ones(len(goods))

        # Solve optimization
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            optimal_quantities = result.x
            optimal_utility = -result.fun
            total_expenditure = np.sum(prices * optimal_quantities)

            return {
                "quantities": dict(zip(goods, optimal_quantities)),
                "utility": optimal_utility,
                "expenditure": total_expenditure,
                "savings": consumer.income - total_expenditure,
                "success": True,
            }
        else:
            return {"success": False, "message": result.message}

    def spatial_consumer_choice(
        self,
        consumer: ConsumerProfile,
        spatial_markets: gpd.GeoDataFrame,
        transport_costs: dict[str, float],
    ) -> dict[str, Any]:
        """
        Model consumer choice with spatial market selection

        Args:
            consumer: Consumer profile with location
            spatial_markets: GeoDataFrame of market locations and prices
            transport_costs: Dictionary of transport cost parameters

        Returns:
            Dictionary with optimal market choice and consumption bundle
        """
        consumer_point = gpd.points_from_xy(
            [consumer.location[1]], [consumer.location[0]]
        )[0]

        best_choice = None
        max_net_utility = -np.inf

        for idx, market in spatial_markets.iterrows():
            # Calculate transport cost
            distance = consumer_point.distance(market.geometry)
            transport_cost = transport_costs.get("per_km", 0.1) * distance

            # Adjust effective income
            effective_income = consumer.income - transport_cost

            if effective_income > 0:
                # Get market prices
                market_prices = np.array(
                    [market.get(f"price_{good}", 1.0) for good in ["good_1", "good_2"]]
                )

                # Solve consumer choice for this market
                temp_consumer = ConsumerProfile(
                    consumer_id=consumer.consumer_id,
                    income=effective_income,
                    location=consumer.location,
                    preferences=consumer.preferences,
                    demographic_attributes=consumer.demographic_attributes,
                    spatial_attributes=consumer.spatial_attributes,
                )

                choice_result = self.solve_utility_maximization(
                    temp_consumer, market_prices, ["good_1", "good_2"]
                )

                if choice_result["success"]:
                    net_utility = choice_result["utility"]

                    if net_utility > max_net_utility:
                        max_net_utility = net_utility
                        best_choice = {
                            "market_id": idx,
                            "market_location": market.geometry,
                            "transport_cost": transport_cost,
                            "consumption": choice_result,
                            "net_utility": net_utility,
                        }

        return best_choice or {"success": False, "message": "No feasible choice"}


class WelfareAnalysis:
    """
    Consumer welfare analysis tools
    """

    @staticmethod
    def consumer_surplus_linear(
        demand_function: Callable, price: float, quantity: float
    ) -> float:
        """
        Calculate consumer surplus for linear demand
        CS = 0.5 * (choke_price - market_price) * quantity
        """
        # This is a simplified implementation
        # In practice, would need to integrate under demand curve
        choke_price = demand_function(0)  # Price where quantity = 0
        return float(0.5 * (choke_price - price) * quantity)

    @staticmethod
    def equivalent_variation(
        utility_function: Callable,
        income: float,
        prices_old: np.ndarray,
        prices_new: np.ndarray,
        alpha: np.ndarray,
    ) -> float:
        """
        Calculate equivalent variation for price change

        Args:
            utility_function: Consumer's utility function
            income: Consumer income
            prices_old: Original prices
            prices_new: New prices
            alpha: Preference parameters

        Returns:
            Equivalent variation amount
        """
        # Calculate utility at original prices
        quantities_old = (alpha * income) / prices_old  # Assuming Cobb-Douglas
        utility_old = utility_function(quantities_old, alpha)

        # Find income needed at new prices to achieve old utility
        def objective(test_income: float) -> float:
            quantities_new = (alpha * test_income) / prices_new
            utility_new = utility_function(quantities_new, alpha)
            return float((utility_new - utility_old) ** 2)

        result = minimize_scalar(objective)
        income_equivalent = result.x if result.success else income

        return income - income_equivalent

    @staticmethod
    def compensating_variation(
        utility_function: Callable,
        income: float,
        prices_old: np.ndarray,
        prices_new: np.ndarray,
        alpha: np.ndarray,
    ) -> float:
        """
        Calculate compensating variation for price change
        """
        # Calculate utility at new prices
        quantities_new = (alpha * income) / prices_new
        utility_new = utility_function(quantities_new, alpha)

        # Find income needed at old prices to achieve new utility
        def objective(test_income: float) -> float:
            quantities_old = (alpha * test_income) / prices_old
            utility_old = utility_function(quantities_old, alpha)
            return float((utility_old - utility_new) ** 2)

        result = minimize_scalar(objective)
        income_compensating = result.x if result.success else income

        return income_compensating - income


class ConsumerSurplus:
    """
    Consumer surplus calculation and analysis
    """

    def __init__(self) -> None:
        self.demand_models: dict[str, Any] = {}

    def calculate_surplus_integral(
        self,
        demand_function: Callable,
        price_range: tuple[float, float],
        market_price: float,
    ) -> float:
        """
        Calculate consumer surplus by integrating under demand curve

        Args:
            demand_function: Function mapping price to quantity demanded
            price_range: (min_price, max_price) for integration
            market_price: Current market price

        Returns:
            Consumer surplus value
        """
        from scipy.integrate import quad

        def integrand(p: float) -> float:
            return float(max(0, demand_function(p)))

        # Integrate from market price to maximum price
        surplus, _ = quad(integrand, market_price, price_range[1])

        return float(surplus)

    def spatial_surplus_analysis(
        self, consumers: list[ConsumerProfile], spatial_markets: gpd.GeoDataFrame
    ) -> dict[str, Any]:
        """
        Analyze consumer surplus across spatial markets

        Args:
            consumers: List of consumer profiles with locations
            spatial_markets: GeoDataFrame of market locations and characteristics

        Returns:
            Dictionary with spatial surplus analysis results
        """
        results: dict[str, Any] = {
            "total_surplus": 0,
            "market_surpluses": {},
            "consumer_surpluses": {},
            "spatial_distribution": {},
        }

        for consumer in consumers:
            # Find nearest markets
            consumer_point = gpd.points_from_xy(
                [consumer.location[1]], [consumer.location[0]]
            )[0]

            # Calculate distances to all markets
            distances = spatial_markets.geometry.distance(consumer_point)
            nearest_market_idx = distances.idxmin()

            # Calculate consumer surplus for nearest market
            # This would involve solving the consumer choice problem
            # and calculating the surplus

            # Baseline calculation
            market_surplus = 100  # Would be calculated based on actual choice

            results["consumer_surpluses"][consumer.consumer_id] = market_surplus
            results["total_surplus"] += market_surplus

            # Aggregate by market
            if nearest_market_idx not in results["market_surpluses"]:
                results["market_surpluses"][nearest_market_idx] = 0
            results["market_surpluses"][nearest_market_idx] += market_surplus

        return results


# Example usage and testing functions
def example_consumer_analysis() -> dict[str, Any]:
    """
    Example usage of consumer theory models
    """
    # Create sample consumer
    consumer = ConsumerProfile(
        consumer_id="consumer_001",
        income=1000.0,
        location=(40.7128, -74.0060),  # NYC coordinates
        preferences={"good_1": 0.6, "good_2": 0.4},
        demographic_attributes={"age": 35, "education": "college"},
        spatial_attributes={"accessibility_index": 0.8},
    )

    # Initialize choice model
    choice_model = ConsumerChoiceModels()

    # Solve utility maximization
    prices = np.array([2.0, 3.0])
    goods = ["good_1", "good_2"]

    result = choice_model.solve_utility_maximization(consumer, prices, goods)

    print("Consumer Choice Results:")
    print(f"Optimal quantities: {result.get('quantities', {})}")
    print(f"Maximum utility: {result.get('utility', 0):.2f}")
    print(f"Total expenditure: {result.get('expenditure', 0):.2f}")

    return result


if __name__ == "__main__":
    # Run example
    example_result = example_consumer_analysis()
