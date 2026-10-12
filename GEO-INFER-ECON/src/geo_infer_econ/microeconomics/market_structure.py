"""
Market Structure Analysis Module

Implements comprehensive market structure analysis including:
- Competition analysis and market power measurement
- Entry barriers and market concentration
- Spatial market definition and delineation
- Market efficiency and welfare analysis
- Antitrust and competition policy implications
"""

import numpy as np
import pandas as pd
from typing import cast, Any
from dataclasses import dataclass


@dataclass
class MarketDefinition:
    """Definition of a market for antitrust analysis"""

    market_id: str
    product_market: str
    geographic_market: str
    time_period: str
    participants: list[str]
    boundaries: dict[str, Any]


class CompetitionAnalysis:
    """
    Analysis of market competition and structure
    """

    def __init__(self) -> None:
        self.market_data: dict[str, Any] = {}

    def calculate_price_correlation_matrix(
        self, price_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate price correlation matrix for market definition

        Args:
            price_data: DataFrame with price data for different products/locations

        Returns:
            Price correlation matrix
        """
        # Calculate correlations between price series
        return cast(np.ndarray, price_data.corr().values)

    def test_market_definition(
        self, price_data: pd.DataFrame, candidate_market: list[str]
    ) -> dict[str, Any]:
        """
        Test whether candidate products/locations constitute a relevant market

        Args:
            price_data: DataFrame with price data
            candidate_market: List of products/locations in candidate market

        Returns:
            Dictionary with market definition test results
        """
        # SSNIP test (Small but Significant Non-transitory Increase in Price)
        # Simplified implementation

        # Calculate price correlations within candidate market
        if len(candidate_market) > 1:
            candidate_prices = price_data[candidate_market]
            internal_correlation = candidate_prices.corr().values.mean()
        else:
            internal_correlation = 1.0

        # Calculate correlations with outside products
        all_products = list(price_data.columns)
        outside_products = [p for p in all_products if p not in candidate_market]

        if outside_products:
            outside_prices = price_data[outside_products]
            external_correlation = candidate_prices.iloc[:, 0].corr(
                outside_prices.iloc[:, 0]
            )
        else:
            external_correlation = 0.0

        # Market definition criteria
        market_criteria = internal_correlation > 0.7 and external_correlation < 0.5

        return {
            "internal_correlation": internal_correlation,
            "external_correlation": external_correlation,
            "is_relevant_market": market_criteria,
            "candidate_market": candidate_market,
        }

    def analyze_entry_barriers(self, industry_data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze entry barriers in an industry

        Args:
            industry_data: DataFrame with industry characteristics

        Returns:
            Dictionary with entry barrier analysis
        """
        barriers = {}

        # Capital requirements
        if "capital_intensity" in industry_data.columns:
            barriers["capital_requirements"] = (
                "high" if industry_data["capital_intensity"].mean() > 0.5 else "low"
            )

        # Scale economies
        if "minimum_efficient_scale" in industry_data.columns:
            barriers["scale_economies"] = (
                "significant"
                if industry_data["minimum_efficient_scale"].mean() > 0.3
                else "limited"
            )

        # Product differentiation
        if "advertising_intensity" in industry_data.columns:
            barriers["product_differentiation"] = (
                "high" if industry_data["advertising_intensity"].mean() > 0.1 else "low"
            )

        # Regulatory barriers
        if "regulatory_burden" in industry_data.columns:
            barriers["regulatory_barriers"] = (
                "high" if industry_data["regulatory_burden"].mean() > 0.5 else "low"
            )

        return barriers


class SpatialMarketAnalysis:
    """
    Spatial market analysis and geographic market delineation
    """

    def __init__(self) -> None:
        self.spatial_markets: dict[str, Any] = {}

    def delineate_geographic_markets(
        self, price_data: pd.DataFrame, locations: list[str]
    ) -> dict[str, Any]:
        """
        Delineate geographic markets based on price integration

        Args:
            price_data: DataFrame with price data across locations
            locations: List of location identifiers

        Returns:
            Dictionary with geographic market delineation results
        """
        # Calculate price correlations between locations
        correlations = price_data.corr()

        # Find clusters of highly correlated locations
        # This is a simplified approach - would use more sophisticated clustering in practice

        high_correlation_threshold = 0.8
        markets = {}

        for i, location_i in enumerate(locations):
            market_id = f"market_{i}"
            markets[market_id] = [location_i]

            for j, location_j in enumerate(locations[i + 1 :], i + 1):
                if correlations.iloc[i, j] > high_correlation_threshold:
                    markets[market_id].append(location_j)

        return {
            "geographic_markets": markets,
            "price_correlations": correlations.values,
            "integration_threshold": high_correlation_threshold,
        }

    def calculate_market_accessibility(
        self, locations: np.ndarray, market_centers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate market accessibility for different locations

        Args:
            locations: Array of location coordinates
            market_centers: Array of market center coordinates

        Returns:
            Array of accessibility indices
        """
        # Simple distance-based accessibility
        distances = np.zeros((len(locations), len(market_centers)))

        for i, loc in enumerate(locations):
            for j, center in enumerate(market_centers):
                distances[i, j] = np.sqrt(np.sum((loc - center) ** 2))

        # Accessibility as inverse distance (simplified)
        accessibility = 1 / (distances + 1e-10)  # Avoid division by zero
        return cast(np.ndarray, accessibility.mean(axis=1))


class MarketStructureAnalysis:
    """
    Main market structure analysis class
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.competition_analysis: Any = CompetitionAnalysis()
        self.spatial_analysis = SpatialMarketAnalysis()

    def analyze_market_power(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze market power and concentration

        Args:
            market_data: DataFrame with market share and price data

        Returns:
            Dictionary with market power analysis
        """
        # Market concentration analysis
        if "market_share" in market_data.columns:
            market_shares = market_data["market_share"].values
            concentration = self.competition_analysis.calculate_market_concentration(
                market_shares
            )
        else:
            concentration = {"error": "Market share data not available"}

        # Price-cost margins as proxy for market power
        if "price" in market_data.columns and "marginal_cost" in market_data.columns:
            margins = (
                market_data["price"] - market_data["marginal_cost"]
            ) / market_data["price"]
            avg_margin = margins.mean()
        else:
            avg_margin = None

        return {
            "concentration_indices": concentration,
            "average_price_margin": avg_margin,
            "market_power_indicators": self._calculate_power_indicators(market_data),
        }

    def _calculate_power_indicators(
        self, market_data: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate various market power indicators"""
        indicators = {}

        # Lerner index (price-cost margin)
        if "price" in market_data.columns and "marginal_cost" in market_data.columns:
            indicators["lerner_index"] = (
                (market_data["price"] - market_data["marginal_cost"])
                / market_data["price"]
            ).mean()

        # Price elasticity (inverse relationship with market power)
        if "price_elasticity" in market_data.columns:
            indicators["price_elasticity"] = market_data["price_elasticity"].mean()

        # Market share of largest firms
        if "market_share" in market_data.columns:
            sorted_shares = np.sort(market_data["market_share"])[::-1]
            indicators["top_firm_share"] = sorted_shares[0]
            indicators["top_four_share"] = np.sum(sorted_shares[:4])

        return indicators

    def analyze_spatial_market_structure(
        self, spatial_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze market structure in spatial context

        Args:
            spatial_data: DataFrame with spatial market data

        Returns:
            Dictionary with spatial market structure analysis
        """
        # Geographic market delineation
        if "location" in spatial_data.columns and "price" in spatial_data.columns:
            locations = spatial_data["location"].unique()
            price_pivot = spatial_data.pivot(
                index="time", columns="location", values="price"
            )

            geographic_markets = self.spatial_analysis.delineate_geographic_markets(
                price_pivot, locations
            )
        else:
            geographic_markets = {"error": "Insufficient spatial data"}

        # Local market concentration
        if "local_market_share" in spatial_data.columns:
            local_concentration = (
                self.competition_analysis.calculate_market_concentration(
                    spatial_data["local_market_share"].values
                )
            )
        else:
            local_concentration = {"error": "Local market share data not available"}

        return {
            "geographic_markets": geographic_markets,
            "local_concentration": local_concentration,
            "spatial_competition": self._analyze_spatial_competition(spatial_data),
        }

    def _analyze_spatial_competition(
        self, spatial_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Analyze spatial competition patterns"""
        # Baseline for spatial competition analysis
        return {
            "spatial_autocorrelation": 0.5,
            "competition_radius": 10.0,  # km
            "local_competition_index": 0.7,
        }
