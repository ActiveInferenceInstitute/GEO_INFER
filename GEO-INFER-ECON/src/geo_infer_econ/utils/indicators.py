"""
Economic indicators calculation utilities.
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
import logging


class EconomicIndicators:
    """
    Utility class for calculating various economic indicators and metrics.

    Provides methods for computing standard economic measures and indices.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the EconomicIndicators calculator.

        Args:
            config: Optional configuration for calculations
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

    def calculate_growth_rate(
        self,
        values: pd.Series | np.ndarray,
        periods: int = 1,
        method: str = "simple",
    ) -> pd.Series | np.ndarray:
        """
        Calculate growth rates for time series data.

        Args:
            values: Time series values
            periods: Number of periods for growth calculation
            method: 'simple' or 'compound' growth rate

        Returns:
            Growth rates
        """
        if isinstance(values, pd.Series):
            if method == "simple":
                return values.pct_change(periods=periods).dropna()
            else:  # compound
                return ((values / values.shift(periods)) ** (1 / periods) - 1).dropna()
        else:
            if method == "simple":
                return np.diff(values, n=periods) / values[:-periods]
            else:  # compound
                return (values[periods:] / values[:-periods]) ** (1 / periods) - 1

    def calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for inequality measurement.

        Args:
            values: Array of values (e.g., income, wealth)

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Sort values in ascending order
        sorted_values = np.sort(values)
        n = len(values)

        # Calculate Gini coefficient
        cumulative_values = np.cumsum(sorted_values)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (
            n * cumulative_values[-1]
        ) - (n + 1) / n

        return float(gini)

    def calculate_theil_index(self, values: np.ndarray) -> float:
        """
        Calculate Theil index for inequality measurement.

        Args:
            values: Array of values

        Returns:
            Theil index
        """
        mean_value = np.mean(values)
        # Avoid log(0) by adding small epsilon
        ratio = values / mean_value
        ratio = np.where(ratio <= 0, 1e-10, ratio)

        theil = np.mean(ratio * np.log(ratio))
        return float(theil)

    def calculate_unemployment_rate(
        self,
        unemployed: float | np.ndarray,
        labor_force: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate unemployment rate.

        Args:
            unemployed: Number of unemployed persons
            labor_force: Total labor force

        Returns:
            Unemployment rate (as percentage)
        """
        return (unemployed / labor_force) * 100

    def calculate_inflation_rate(
        self, prices: pd.Series | np.ndarray, base_period: int = 0
    ) -> pd.Series | np.ndarray:
        """
        Calculate inflation rate from price index.

        Args:
            prices: Price index values
            base_period: Base period index

        Returns:
            Inflation rates
        """
        if isinstance(prices, pd.Series):
            base_price = prices.iloc[base_period]
            return ((prices / base_price) - 1) * 100
        else:
            base_price = prices[base_period]
            return ((prices / base_price) - 1) * 100

    def calculate_gdp_per_capita(
        self, gdp: float | np.ndarray, population: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate GDP per capita.

        Args:
            gdp: Gross Domestic Product
            population: Population size

        Returns:
            GDP per capita
        """
        return gdp / population

    def calculate_productivity_growth(
        self, output: np.ndarray, inputs: np.ndarray, method: str = "total_factor"
    ) -> np.ndarray:
        """
        Calculate productivity growth.

        Args:
            output: Output values over time
            inputs: Input values over time (can be multidimensional)
            method: 'labor' or 'total_factor' productivity

        Returns:
            Productivity growth rates
        """
        if method == "labor":
            # Simple labor productivity
            labor_productivity = output / inputs
            return self.calculate_growth_rate(labor_productivity)
        else:
            # Total factor productivity (simplified Solow residual)
            if inputs.ndim == 1:
                # Single input case
                tfp = output / inputs
            else:
                # Multiple inputs - geometric mean
                tfp = output / np.prod(inputs, axis=1)

            return self.calculate_growth_rate(tfp)

    def calculate_economic_complexity_index(
        self, exports_matrix: np.ndarray, countries: list[str], products: list[str]
    ) -> dict[str, float]:
        """
        Calculate Economic Complexity Index (ECI) for countries.

        Args:
            exports_matrix: Matrix of exports (countries x products)
            countries: List of country names
            products: List of product names

        Returns:
            Dictionary with ECI values for each country
        """
        # This is a simplified version of the ECI calculation
        # Full implementation would follow Hidalgo & Hausmann methodology

        # Calculate Revealed Comparative Advantage (RCA)
        total_exports_country = np.sum(exports_matrix, axis=1, keepdims=True)
        total_exports_product = np.sum(exports_matrix, axis=0, keepdims=True)
        total_exports_world = np.sum(exports_matrix)

        # Avoid division by zero
        total_exports_country = np.where(
            total_exports_country == 0, 1e-10, total_exports_country
        )
        total_exports_product = np.where(
            total_exports_product == 0, 1e-10, total_exports_product
        )

        rca = (exports_matrix / total_exports_country) / (
            total_exports_product / total_exports_world
        )

        # Binary matrix (RCA >= 1)
        M = (rca >= 1).astype(int)

        # Calculate diversity and ubiquity
        diversity = np.sum(M, axis=1)  # Number of products with RCA >= 1

        # Simple complexity measure (more sophisticated methods exist)
        complexity_scores = diversity / np.mean(diversity)

        return dict(zip(countries, complexity_scores))

    def calculate_regional_convergence(
        self, regional_data: pd.DataFrame, value_column: str, time_column: str
    ) -> dict[str, Any]:
        """
        Calculate regional convergence indicators.

        Args:
            regional_data: DataFrame with regional economic data
            value_column: Column with economic values (e.g., GDP per capita)
            time_column: Column with time periods

        Returns:
            Dictionary with convergence indicators
        """
        # Beta convergence
        initial_values = regional_data.groupby(regional_data.index)[
            value_column
        ].first()
        final_values = regional_data.groupby(regional_data.index)[value_column].last()

        growth_rates = np.log(final_values / initial_values)
        log_initial = np.log(initial_values)

        # Simple regression for beta convergence
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_initial, growth_rates
        )
        beta_convergence = -slope

        # Sigma convergence
        time_periods = sorted(regional_data[time_column].unique())
        sigma_values = []

        for period in time_periods:
            period_data = regional_data[regional_data[time_column] == period][
                value_column
            ]
            if len(period_data) > 1:
                sigma = np.std(np.log(period_data))
                sigma_values.append(sigma)

        sigma_convergence = len(sigma_values) > 1 and sigma_values[-1] < sigma_values[0]

        return {
            "beta_convergence": beta_convergence,
            "beta_significance": p_value,
            "sigma_convergence": sigma_convergence,
            "initial_sigma": sigma_values[0] if sigma_values else None,
            "final_sigma": sigma_values[-1] if sigma_values else None,
        }

    def calculate_economic_distance(
        self,
        region1_data: dict[str, float],
        region2_data: dict[str, float],
        method: str = "euclidean",
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate economic distance between regions with advanced weighting.

        Args:
            region1_data: Economic indicators for region 1
            region2_data: Economic indicators for region 2
            method: Distance metric ('euclidean', 'manhattan', 'cosine', 'mahalanobis')
            weights: Optional weights for different indicators

        Returns:
            Economic distance measure
        """
        # Get common indicators
        common_indicators = set(region1_data.keys()) & set(region2_data.keys())

        if not common_indicators:
            raise ValueError("No common economic indicators found")

        # Extract values for common indicators
        values1 = np.array([region1_data[ind] for ind in common_indicators])
        values2 = np.array([region2_data[ind] for ind in common_indicators])

        # Apply weights if provided
        if weights:
            weight_values = np.array(
                [weights.get(ind, 1.0) for ind in common_indicators]
            )
            values1 = values1 * weight_values
            values2 = values2 * weight_values

        # Normalize values (robust standardization)
        combined = np.vstack([values1, values2])
        if combined.shape[0] > 1:
            mean_vals = np.mean(combined, axis=0)
            std_vals = np.std(combined, axis=0)
            std_vals = np.where(std_vals == 0, 1, std_vals)  # Avoid division by zero
            normalized = (combined - mean_vals) / std_vals
        else:
            normalized = combined

        norm_values1, norm_values2 = normalized[0], normalized[1]

        if method == "euclidean":
            distance = np.sqrt(np.sum((norm_values1 - norm_values2) ** 2))
        elif method == "manhattan":
            distance = np.sum(np.abs(norm_values1 - norm_values2))
        elif method == "cosine":
            dot_product = np.dot(norm_values1, norm_values2)
            norms = np.linalg.norm(norm_values1) * np.linalg.norm(norm_values2)
            distance = 1 - (dot_product / norms) if norms > 0 else 0
        elif method == "mahalanobis":
            # Mahalanobis distance using covariance of combined data
            if combined.shape[0] > 1:
                cov_matrix = np.cov(combined.T)
                try:
                    inv_cov = np.linalg.inv(cov_matrix)
                    diff = norm_values1 - norm_values2
                    distance = np.sqrt(diff @ inv_cov @ diff)
                except Exception:
                    # Fall back to Euclidean if covariance matrix is singular
                    distance = np.sqrt(np.sum((norm_values1 - norm_values2) ** 2))
            else:
                distance = 0
        else:
            raise ValueError(f"Unknown distance method: {method}")

        return float(distance)

    def calculate_spatial_economic_indicators(
        self, regional_data: pd.DataFrame, spatial_weights: np.ndarray
    ) -> dict[str, Any]:
        """
        Calculate spatial economic indicators including spatial autocorrelation and clustering.

        Args:
            regional_data: DataFrame with regional economic data
            spatial_weights: Spatial weights matrix

        Returns:
            Dictionary with spatial economic indicators
        """
        # Extract economic variables
        variables = [
            col for col in regional_data.columns if col not in ["region_id", "geometry"]
        ]

        spatial_indicators = {}

        for var in variables:
            if var in regional_data.columns:
                values = regional_data[var].values

                # Spatial autocorrelation (Moran's I)
                wy_values = spatial_weights @ values
                n = len(values)
                morans_i = (
                    (n / np.sum(spatial_weights))
                    * (values.T @ wy_values)
                    / (values.T @ values)
                )

                # Local indicators of spatial association (LISA)
                local_morans = values * wy_values / (values.T @ values / n)

                # Spatial clustering (Getis-Ord G*)
                g_stat = np.sum(wy_values) / np.sum(values)

                spatial_indicators[var] = {
                    "morans_i": float(morans_i),
                    "local_morans": local_morans.tolist(),
                    "getis_ord_g": float(g_stat),
                    "spatial_pattern": self._classify_spatial_pattern(morans_i, g_stat),
                }

        return spatial_indicators

    def _classify_spatial_pattern(self, morans_i: float, getis_ord_g: float) -> str:
        """Classify spatial pattern based on autocorrelation and clustering measures."""
        if morans_i > 0.1 and getis_ord_g > 1:
            return "high_high_clustering"
        elif morans_i > 0.1 and getis_ord_g < 1:
            return "low_low_clustering"
        elif morans_i < -0.1 and getis_ord_g > 1:
            return "high_low_outlier"
        elif morans_i < -0.1 and getis_ord_g < 1:
            return "low_high_outlier"
        else:
            return "random_pattern"

    def calculate_economic_resilience(
        self, time_series_data: pd.DataFrame, shock_period: str | None = None
    ) -> dict[str, Any]:
        """
        Calculate economic resilience indicators from time series data.

        Args:
            time_series_data: DataFrame with time series economic data
            shock_period: Optional period when shock occurred

        Returns:
            Dictionary with resilience indicators
        """
        resilience_metrics = {}

        for column in time_series_data.columns:
            if column != "time":
                values = time_series_data[column].values

                # Pre-shock trend
                if shock_period:
                    pre_shock = time_series_data[
                        time_series_data["time"] < shock_period
                    ][column]
                    post_shock = time_series_data[
                        time_series_data["time"] >= shock_period
                    ][column]

                    if len(pre_shock) > 1 and len(post_shock) > 1:
                        # Recovery rate
                        pre_trend = np.polyfit(range(len(pre_shock)), pre_shock, 1)[0]
                        post_trend = np.polyfit(range(len(post_shock)), post_shock, 1)[
                            0
                        ]

                        # Volatility measures
                        pre_volatility = np.std(pre_shock) / np.mean(pre_shock)
                        post_volatility = np.std(post_shock) / np.mean(post_shock)

                        resilience_metrics[column] = {
                            "recovery_rate": float(post_trend - pre_trend),
                            "volatility_change": float(
                                post_volatility - pre_volatility
                            ),
                            "resilience_score": self._calculate_resilience_score(
                                pre_trend, post_trend, pre_volatility, post_volatility
                            ),
                        }
                else:
                    # General resilience measures without specific shock
                    growth_rate = (values[-1] - values[0]) / values[0] / len(values)
                    volatility = np.std(values) / np.mean(values)

                    resilience_metrics[column] = {
                        "growth_rate": float(growth_rate),
                        "volatility": float(volatility),
                        "resilience_score": float(growth_rate / (volatility + 1e-10)),
                    }

        return resilience_metrics

    def _calculate_resilience_score(
        self,
        pre_trend: float,
        post_trend: float,
        pre_volatility: float,
        post_volatility: float,
    ) -> float:
        """Calculate overall resilience score."""
        # Resilience based on trend recovery and volatility control
        trend_recovery = max(0, post_trend / (pre_trend + 1e-10))
        volatility_control = max(0, 1 - (post_volatility / (pre_volatility + 1e-10)))

        return (trend_recovery + volatility_control) / 2

    def calculate_sectoral_composition(
        self, sectoral_data: pd.DataFrame, classification: str = "standard"
    ) -> dict[str, Any]:
        """
        Calculate sectoral composition and diversity indicators.

        Args:
            sectoral_data: DataFrame with sectoral economic data
            classification: Industry classification system

        Returns:
            Dictionary with sectoral composition analysis
        """
        # Calculate shares and diversity measures
        total_value = sectoral_data.iloc[:, 1:].sum().sum()  # Sum all sectoral values

        sectoral_shares = {}
        for col in sectoral_data.columns[1:]:  # Skip region column
            share = sectoral_data[col].sum() / total_value if total_value > 0 else 0
            sectoral_shares[col] = share

        # Herfindahl-Hirschman Index for concentration
        hhi = sum(share**2 for share in sectoral_shares.values())

        # Entropy-based diversity measure
        entropy = -sum(
            share * np.log(share + 1e-10) for share in sectoral_shares.values()
        )

        # Number of effective sectors
        n_effective = np.exp(entropy)

        return {
            "sectoral_shares": sectoral_shares,
            "concentration_hhi": float(hhi),
            "diversity_entropy": float(entropy),
            "effective_sectors": float(n_effective),
            "dominance_index": max(sectoral_shares.values()),
            "classification_system": classification,
        }

    def calculate_trade_integration(
        self, trade_data: pd.DataFrame, regions: list[str]
    ) -> dict[str, Any]:
        """
        Calculate trade integration and connectivity indicators.

        Args:
            trade_data: DataFrame with trade flow data
            regions: List of region identifiers

        Returns:
            Dictionary with trade integration analysis
        """
        # Calculate trade flows and connectivity
        trade_matrix = trade_data.pivot_table(
            index="origin", columns="destination", values="trade_value", fill_value=0
        )

        # Total trade volume
        total_trade = trade_matrix.sum().sum()

        # Trade concentration (HHI of trade partners)
        trade_shares = trade_matrix.sum(axis=1) / total_trade
        trade_hhi = (trade_shares**2).sum()

        # Network centrality measures (simplified)
        # In-degree centrality (imports)
        in_degree = trade_matrix.sum(axis=0)

        # Out-degree centrality (exports)
        out_degree = trade_matrix.sum(axis=1)

        # Trade balance
        trade_balance = out_degree - in_degree

        return {
            "total_trade_volume": float(total_trade),
            "trade_concentration_hhi": float(trade_hhi),
            "import_centrality": in_degree.to_dict(),
            "export_centrality": out_degree.to_dict(),
            "trade_balance": trade_balance.to_dict(),
            "top_trading_partners": self._identify_top_partners(trade_matrix),
        }

    def _identify_top_partners(
        self, trade_matrix: pd.DataFrame, top_n: int = 5
    ) -> dict[str, list[str]]:
        """Identify top trading partners for each region."""
        top_partners = {}

        for region in trade_matrix.index:
            # Get top trading partners for this region
            partner_trade = trade_matrix.loc[region]
            top_partners[region] = partner_trade.nlargest(top_n).index.tolist()

        return top_partners

    def calculate_human_development_index(
        self, data: dict[str, pd.DataFrame], weights: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Calculate Human Development Index (HDI) components and overall index.

        Args:
            data: Dictionary with data for different HDI components
            weights: Optional custom weights for components

        Returns:
            Dictionary with HDI calculations
        """
        # Standard HDI weights
        default_weights = {"health": 1 / 3, "education": 1 / 3, "income": 1 / 3}
        weights = weights or default_weights

        hdi_components: dict[str, float] = {}

        # Health index (life expectancy)
        if "life_expectancy" in data:
            le_data = data["life_expectancy"]
            # Normalize life expectancy (20-85 years)
            health_index = (le_data - 20) / (85 - 20)
            health_index = np.clip(health_index, 0, 1)
            hdi_components["health_index"] = health_index.mean()

        # Education index (expected years of schooling + mean years)
        if "education" in data:
            edu_data = data["education"]
            # Simplified education index
            education_index = np.clip(edu_data / 18, 0, 1)  # Assuming 18 max years
            hdi_components["education_index"] = education_index.mean()

        # Income index (GNI per capita)
        if "income" in data:
            income_data = data["income"]
            # Log transformation for income (PPP $100 - $75,000)
            ln_income = np.log(np.maximum(income_data, 100))
            ln_min, ln_max = np.log(100), np.log(75000)
            income_index = (ln_income - ln_min) / (ln_max - ln_min)
            income_index = np.clip(income_index, 0, 1)
            hdi_components["income_index"] = income_index.mean()

        # Overall HDI
        if len(hdi_components) == 3:
            overall_hdi = sum(
                hdi_components[comp] * weights[comp.replace("_index", "")]
                for comp in hdi_components.keys()
            )
        else:
            overall_hdi = np.nan

        return {
            "hdi_components": hdi_components,
            "overall_hdi": float(overall_hdi),
            "weights_used": weights,
        }
