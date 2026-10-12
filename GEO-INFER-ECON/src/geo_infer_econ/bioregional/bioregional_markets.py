"""
Bioregional Market Design Module

Implements comprehensive bioregional market mechanisms including:
- Ecosystem services markets and trading systems
- Natural capital accounting and trading
- Biodiversity credits and habitat banking
- Carbon markets and sequestration credits
- Water resource markets and watershed trading
- Local food systems and circular economy markets
"""

import numpy as np
from typing import cast, Any
from dataclasses import dataclass
import geopandas as gpd
from datetime import datetime


@dataclass
class BioregionalAsset:
    """Represents a bioregional asset with ecological and economic attributes"""

    asset_id: str
    asset_type: str  # forest, wetland, grassland, agricultural, etc.
    location: tuple[float, float]  # (lat, lon)
    area_hectares: float
    ecological_attributes: dict[str, float]  # biodiversity, carbon storage, etc.
    economic_attributes: dict[str, float]  # market value, income potential, etc.
    ownership_type: str  # private, public, community, cooperative
    management_regime: str  # conservation, sustainable use, restoration
    ecosystem_services: dict[
        str, float
    ]  # provisioning, regulating, cultural, supporting


@dataclass
class MarketParticipant:
    """Represents a participant in bioregional markets"""

    participant_id: str
    participant_type: str  # landowner, buyer, intermediary, regulator
    location: tuple[float, float]
    assets_owned: list[str]  # list of asset IDs
    market_preferences: dict[str, float]
    budget_constraints: dict[str, float]
    sustainability_goals: dict[str, float]


@dataclass
class EcosystemServiceCredit:
    """Represents a tradeable ecosystem service credit"""

    credit_id: str
    service_type: str  # carbon, biodiversity, water, pollination, etc.
    quantity: float  # units of service
    quality_tier: str  # high, medium, low based on additionality and permanence
    location: tuple[float, float]
    temporal_profile: str  # permanent, temporary, periodic
    verification_status: str  # verified, pending, self-reported
    price_per_unit: float
    expiration_date: datetime | None
    co_benefits: dict[str, float]  # additional ecological benefits


class BioregionalMarketDesign:
    """
    Core engine for designing and operating bioregional markets
    """

    def __init__(self, bioregion_boundary: gpd.GeoDataFrame):
        self.bioregion = bioregion_boundary
        self.assets: dict[str, Any] = {}
        self.participants: dict[str, Any] = {}
        self.credits: dict[str, Any] = {}
        self.market_mechanisms: dict[str, Any] = {}
        self.transaction_history: list[Any] = []

    def register_asset(self, asset: BioregionalAsset) -> bool:
        """Register a bioregional asset in the market system"""
        self.assets[asset.asset_id] = asset
        return True

    def register_participant(self, participant: MarketParticipant) -> bool:
        """Register a market participant"""
        self.participants[participant.participant_id] = participant
        return True

    def create_ecosystem_service_credit(
        self,
        asset_id: str,
        service_type: str,
        quantity: float,
        quality_parameters: dict[str, Any],
    ) -> EcosystemServiceCredit:
        """
        Create ecosystem service credits from bioregional assets

        Args:
            asset_id: ID of the asset generating the service
            service_type: Type of ecosystem service
            quantity: Quantity of service units
            quality_parameters: Parameters for credit quality assessment

        Returns:
            EcosystemServiceCredit object
        """
        asset = self.assets[asset_id]

        # Quality assessment based on asset characteristics
        quality_score = self._assess_credit_quality(
            asset, service_type, quality_parameters
        )
        quality_tier = (
            "high"
            if quality_score > 0.8
            else "medium"
            if quality_score > 0.5
            else "low"
        )

        # Price determination based on quality, location, and market conditions
        base_price = self._determine_base_price(service_type, quality_tier)
        location_multiplier = self._calculate_location_multiplier(
            asset.location, service_type
        )
        price_per_unit = base_price * location_multiplier

        credit = EcosystemServiceCredit(
            credit_id=f"{asset_id}_{service_type}_{datetime.now().isoformat()}",
            service_type=service_type,
            quantity=quantity,
            quality_tier=quality_tier,
            location=asset.location,
            temporal_profile=quality_parameters.get("temporal_profile", "permanent"),
            verification_status="pending",
            price_per_unit=price_per_unit,
            expiration_date=quality_parameters.get("expiration_date"),
            co_benefits=self._calculate_co_benefits(asset, service_type),
        )

        self.credits[credit.credit_id] = credit
        return credit

    def _assess_credit_quality(
        self, asset: BioregionalAsset, service_type: str, parameters: dict[str, Any]
    ) -> float:
        """Assess the quality of ecosystem service credits"""
        quality_factors = {
            "additionality": 0.3,  # Would the service occur without the project?
            "permanence": 0.25,  # How long will the service be maintained?
            "measurability": 0.2,  # How accurately can we measure the service?
            "leakage_risk": 0.15,  # Risk of negative effects elsewhere
            "co_benefits": 0.1,  # Additional ecological benefits
        }

        score = 0
        for factor, weight in quality_factors.items():
            factor_score = parameters.get(factor, 0.5)  # Default to medium quality
            score += factor_score * weight

        return score

    def _determine_base_price(self, service_type: str, quality_tier: str) -> float:
        """Determine base price for ecosystem service credits"""
        base_prices = {
            "carbon": {"high": 50, "medium": 30, "low": 15},
            "biodiversity": {"high": 100, "medium": 60, "low": 30},
            "water_quality": {"high": 80, "medium": 50, "low": 25},
            "pollination": {"high": 120, "medium": 70, "low": 35},
            "flood_control": {"high": 90, "medium": 55, "low": 28},
        }

        return base_prices.get(service_type, {"high": 50, "medium": 30, "low": 15})[
            quality_tier
        ]

    def _calculate_location_multiplier(
        self, location: tuple[float, float], service_type: str
    ) -> float:
        """Calculate location-based price multiplier"""
        # Simple distance-based multiplier (can be made more sophisticated)
        # Closer to population centers or vulnerable areas gets higher prices

        # Urban proximity defaults to 1.0; production deployments should
        # inject actual distance-to-urban calculations via spatial analysis
        urban_proximity = 1.0

        # Ecosystem service specific location factors
        location_factors = {
            "carbon": 1.0,  # Global benefit
            "biodiversity": urban_proximity * 0.8 + 0.2,  # Higher value near cities
            "water_quality": urban_proximity,  # Higher value for urban watersheds
            "pollination": urban_proximity * 0.6 + 0.4,  # Value for agricultural areas
            "flood_control": urban_proximity,  # Higher value protecting urban areas
        }

        return location_factors.get(service_type, 1.0)

    def _calculate_co_benefits(
        self, asset: BioregionalAsset, service_type: str
    ) -> dict[str, float]:
        """Calculate co-benefits of ecosystem service credits"""
        co_benefits = {}

        # Example co-benefit calculations
        if service_type == "carbon":
            co_benefits["biodiversity"] = (
                asset.ecological_attributes.get("biodiversity_index", 0) * 0.5
            )
            co_benefits["water_quality"] = (
                asset.ecological_attributes.get("water_filtration", 0) * 0.3
            )

        elif service_type == "biodiversity":
            co_benefits["carbon"] = (
                asset.ecological_attributes.get("carbon_storage", 0) * 0.4
            )
            co_benefits["recreation"] = (
                asset.ecological_attributes.get("recreation_value", 0) * 0.6
            )

        return co_benefits


class EcosystemServicesMarkets:
    """
    Specialized markets for different ecosystem services
    """

    def __init__(self, market_design: BioregionalMarketDesign):
        self.market_design = market_design
        self.order_book: dict[str, list[Any]] = {"buy": [], "sell": []}
        self.market_clearing_mechanism = "double_auction"

    def submit_buy_order(
        self,
        participant_id: str,
        service_type: str,
        quantity: float,
        max_price: float,
        location_preferences: dict[str, Any],
    ) -> str:
        """Submit a buy order for ecosystem services"""
        order: dict[str, Any] = {
            "order_id": f"buy_{participant_id}_{datetime.now().isoformat()}",
            "participant_id": participant_id,
            "order_type": "buy",
            "service_type": service_type,
            "quantity": quantity,
            "price": max_price,
            "location_preferences": location_preferences,
            "timestamp": datetime.now(),
        }

        self.order_book["buy"].append(order)
        return cast(str, order["order_id"])

    def submit_sell_order(
        self, participant_id: str, credit_id: str, min_price: float
    ) -> str:
        """Submit a sell order for ecosystem service credits"""
        credit = self.market_design.credits[credit_id]

        order: dict[str, Any] = {
            "order_id": f"sell_{participant_id}_{datetime.now().isoformat()}",
            "participant_id": participant_id,
            "order_type": "sell",
            "credit_id": credit_id,
            "service_type": credit.service_type,
            "quantity": credit.quantity,
            "price": max(min_price, credit.price_per_unit),
            "quality_tier": credit.quality_tier,
            "location": credit.location,
            "timestamp": datetime.now(),
        }

        self.order_book["sell"].append(order)
        return cast(str, order["order_id"])

    def clear_market(self) -> list[dict[str, Any]]:
        """Clear the market and execute trades"""
        if self.market_clearing_mechanism == "double_auction":
            return self._double_auction_clearing()
        elif self.market_clearing_mechanism == "call_auction":
            return self._call_auction_clearing()
        else:
            return []

    def _double_auction_clearing(self) -> list[dict[str, Any]]:
        """Double auction market clearing mechanism"""
        transactions = []

        # Sort buy orders by price (descending) and sell orders by price (ascending)
        buy_orders = sorted(
            self.order_book["buy"], key=lambda x: x["price"], reverse=True
        )
        sell_orders = sorted(self.order_book["sell"], key=lambda x: x["price"])

        for buy_order in buy_orders:
            for sell_order in sell_orders:
                # Check if orders can be matched
                if (
                    buy_order["service_type"] == sell_order["service_type"]
                    and buy_order["price"] >= sell_order["price"]
                    and buy_order["quantity"] > 0
                    and sell_order["quantity"] > 0
                ):
                    # Check location preferences
                    if self._check_location_compatibility(buy_order, sell_order):
                        # Execute trade
                        trade_quantity = min(
                            buy_order["quantity"], sell_order["quantity"]
                        )
                        trade_price = (buy_order["price"] + sell_order["price"]) / 2

                        transaction = {
                            "transaction_id": f"trade_{datetime.now().isoformat()}",
                            "buy_order_id": buy_order["order_id"],
                            "sell_order_id": sell_order["order_id"],
                            "service_type": buy_order["service_type"],
                            "quantity": trade_quantity,
                            "price": trade_price,
                            "timestamp": datetime.now(),
                        }

                        transactions.append(transaction)

                        # Update order quantities
                        buy_order["quantity"] -= trade_quantity
                        sell_order["quantity"] -= trade_quantity

                        # Record transaction
                        self.market_design.transaction_history.append(transaction)

        # Remove completed orders
        self.order_book["buy"] = [
            order for order in self.order_book["buy"] if order["quantity"] > 0
        ]
        self.order_book["sell"] = [
            order for order in self.order_book["sell"] if order["quantity"] > 0
        ]

        return transactions

    def _check_location_compatibility(self, buy_order: dict, sell_order: dict) -> bool:
        """Check if buy and sell orders are locationally compatible"""
        location_prefs = buy_order.get("location_preferences", {})

        if not location_prefs:
            return True  # No location preferences

        # Calculate distance between buyer preference and asset location
        max_distance = location_prefs.get("max_distance_km", float("inf"))
        preferred_location = location_prefs.get("preferred_location")

        if preferred_location:
            distance = (
                np.sqrt(
                    (sell_order["location"][0] - preferred_location[0]) ** 2
                    + (sell_order["location"][1] - preferred_location[1]) ** 2
                )
                * 111
            )  # Approximate km per degree

            return bool(distance <= max_distance)

        return True

    def _call_auction_clearing(self) -> list[dict[str, Any]]:
        """Uniform-price call auction: find clearing price at supply/demand intersection."""
        buy_orders = sorted(
            self.order_book["buy"], key=lambda x: x["price"], reverse=True
        )
        sell_orders = sorted(self.order_book["sell"], key=lambda x: x["price"])

        if not buy_orders or not sell_orders:
            return []

        # Build cumulative demand / supply schedules
        demand_qty, supply_qty = 0.0, 0.0
        demand_schedule = []  # (price, cumulative_qty)
        for o in buy_orders:
            demand_qty += o["quantity"]
            demand_schedule.append((o["price"], demand_qty))
        supply_schedule = []
        for o in sell_orders:
            supply_qty += o["quantity"]
            supply_schedule.append((o["price"], supply_qty))

        # Find clearing price: highest price where cumulative demand >= cumulative supply
        clearing_price = sell_orders[0]["price"]
        for bp, d_cum in demand_schedule:
            s_cum = sum(o["quantity"] for o in sell_orders if o["price"] <= bp)
            if s_cum > 0 and d_cum >= s_cum:
                clearing_price = bp
                break

        # Execute all eligible trades at the clearing price
        transactions = []
        remaining_buys = [o.copy() for o in buy_orders if o["price"] >= clearing_price]
        remaining_sells = [
            o.copy() for o in sell_orders if o["price"] <= clearing_price
        ]

        for bo in remaining_buys:
            for so in remaining_sells:
                if bo["quantity"] <= 0 or so["quantity"] <= 0:
                    continue
                if bo["service_type"] != so["service_type"]:
                    continue
                trade_qty = min(bo["quantity"], so["quantity"])
                transactions.append(
                    {
                        "transaction_id": f"call_{datetime.now().isoformat()}",
                        "buy_order_id": bo["order_id"],
                        "sell_order_id": so["order_id"],
                        "service_type": bo["service_type"],
                        "quantity": trade_qty,
                        "price": clearing_price,
                        "timestamp": datetime.now(),
                    }
                )
                bo["quantity"] -= trade_qty
                so["quantity"] -= trade_qty
                self.market_design.transaction_history.append(transactions[-1])

        # Clean fulfilled orders
        self.order_book["buy"] = [
            o for o in self.order_book["buy"] if o["quantity"] > 0
        ]
        self.order_book["sell"] = [
            o for o in self.order_book["sell"] if o["quantity"] > 0
        ]
        return transactions


class BiodiversityMarkets:
    """
    Specialized markets for biodiversity credits and habitat banking
    """

    def __init__(self, market_design: BioregionalMarketDesign):
        self.market_design = market_design
        self.habitat_banks: dict[str, Any] = {}
        self.mitigation_requirements: dict[str, Any] = {}

    def create_habitat_bank(
        self, bank_id: str, asset_ids: list[str], credit_types: list[str]
    ) -> dict[str, Any]:
        """Create a habitat bank for biodiversity credit generation"""
        total_area = sum(
            self.market_design.assets[aid].area_hectares for aid in asset_ids
        )

        # Calculate biodiversity credit potential
        credit_potential = {}
        for credit_type in credit_types:
            total_potential = sum(
                self.market_design.assets[aid].ecological_attributes.get(credit_type, 0)
                for aid in asset_ids
            )
            credit_potential[credit_type] = total_potential

        bank = {
            "bank_id": bank_id,
            "asset_ids": asset_ids,
            "total_area": total_area,
            "credit_types": credit_types,
            "credit_potential": credit_potential,
            "credits_sold": {ct: 0 for ct in credit_types},
            "status": "approved",
        }

        self.habitat_banks[bank_id] = bank
        return bank

    def calculate_mitigation_requirement(
        self,
        impact_location: tuple[float, float],
        impact_area: float,
        habitat_type: str,
    ) -> dict[str, Any]:
        """Calculate biodiversity mitigation requirements for development impacts"""
        # Biodiversity offset ratios based on habitat type and location
        offset_ratios = {
            "wetland": 3.0,  # 3:1 restoration ratio
            "forest": 2.0,  # 2:1 restoration ratio
            "grassland": 1.5,  # 1.5:1 restoration ratio
            "coastal": 4.0,  # 4:1 restoration ratio
        }

        base_ratio = offset_ratios.get(habitat_type, 2.0)

        # Location-based multiplier: higher near known biodiversity hotspots
        # Use inverse-distance weighting to nearest registered asset
        location_multiplier = 1.0
        if self.market_design.assets:
            min_dist = min(
                np.sqrt(
                    (a.location[0] - impact_location[0]) ** 2
                    + (a.location[1] - impact_location[1]) ** 2
                )
                * 111
                for a in self.market_design.assets.values()
            )
            # Closer to existing ecological assets → higher ratio
            location_multiplier = 1.0 + max(0, 1.0 - min_dist / 50)

        required_credits = impact_area * base_ratio * location_multiplier

        return {
            "habitat_type": habitat_type,
            "impact_area": impact_area,
            "required_credits": required_credits,
            "offset_ratio": base_ratio * location_multiplier,
        }

    def match_credits_to_requirements(
        self, requirement_id: str
    ) -> list[dict[str, Any]]:
        """Match available biodiversity credits to mitigation requirements"""
        requirement = self.mitigation_requirements[requirement_id]
        available_credits = []

        # Find suitable habitat banks
        for bank_id, bank in self.habitat_banks.items():
            if requirement["habitat_type"] in bank["credit_types"]:
                available_credits.append(
                    {
                        "bank_id": bank_id,
                        "available_credits": (
                            bank["credit_potential"][requirement["habitat_type"]]
                            - bank["credits_sold"][requirement["habitat_type"]]
                        ),
                        "location": self._get_bank_centroid(bank),
                        "distance_to_impact": self._calculate_distance(
                            self._get_bank_centroid(bank),
                            requirement["impact_location"],
                        ),
                    }
                )

        # Sort by distance and credit availability
        available_credits.sort(
            key=lambda x: (x["distance_to_impact"], -x["available_credits"])
        )

        return available_credits

    def _get_bank_centroid(self, bank: dict[str, Any]) -> tuple[float, float]:
        """Calculate centroid of habitat bank assets"""
        locations = [
            self.market_design.assets[aid].location for aid in bank["asset_ids"]
        ]
        centroid_lat = sum(loc[0] for loc in locations) / len(locations)
        centroid_lon = sum(loc[1] for loc in locations) / len(locations)
        return (centroid_lat, centroid_lon)

    def _calculate_distance(
        self, loc1: tuple[float, float], loc2: tuple[float, float]
    ) -> float:
        """Calculate distance between two locations"""
        return float(
            np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) * 111
        )  # km


class LocalFoodSystems:
    """
    Markets and systems for local and regional food production and distribution
    """

    def __init__(self, market_design: BioregionalMarketDesign):
        self.market_design = market_design
        self.food_producers: dict[str, Any] = {}
        self.food_consumers: dict[str, Any] = {}
        self.distribution_networks: dict[str, Any] = {}

    def optimize_local_food_system(
        self, optimization_objectives: list[str]
    ) -> dict[str, Any]:
        """
        Optimize local food system for multiple objectives using advanced algorithms

        Args:
            optimization_objectives: List of objectives like 'minimize_transport',
                                   'maximize_nutrition', 'minimize_environmental_impact'

        Returns:
            Dictionary with optimization results
        """
        # Multi-objective optimization for local food systems
        # This implements a sophisticated optimization considering multiple dimensions

        # Define objective functions
        objectives = {
            "minimize_transport": self._transport_objective,
            "maximize_nutrition": self._nutrition_objective,
            "minimize_environmental_impact": self._environmental_objective,
            "maximize_economic_efficiency": self._economic_objective,
        }

        # Filter to requested objectives
        active_objectives = {
            k: v for k, v in objectives.items() if k in optimization_objectives
        }

        if not active_objectives:
            return {"error": "No valid optimization objectives specified"}

        # Generate production scenarios
        production_scenarios = self._generate_production_scenarios()

        # Evaluate scenarios against objectives
        scenario_scores: dict[str, dict[str, float]] = {}
        for scenario_name, scenario in production_scenarios.items():
            scores = {}
            for obj_name, obj_func in active_objectives.items():
                scores[obj_name] = obj_func(scenario)
            scenario_scores[scenario_name] = scores

        # Multi-objective optimization (simplified)
        # In practice, would use Pareto optimization or weighted sum
        best_scenario: tuple[str, Any]
        if len(active_objectives) == 1:
            # Single objective - find best scenario
            best_scenario = max(
                scenario_scores.items(), key=lambda x: list(x[1].values())[0]
            )
        else:
            # Multiple objectives - use weighted combination
            weights = {
                obj: 1.0 / len(active_objectives) for obj in active_objectives.keys()
            }
            best_scenario = self._find_pareto_optimal(scenario_scores, weights)

        return {
            "optimal_scenario": best_scenario[0],
            "scenario_scores": scenario_scores,
            "optimization_objectives": optimization_objectives,
            "production_allocation": production_scenarios[best_scenario[0]][
                "allocation"
            ],
            "distribution_plan": production_scenarios[best_scenario[0]]["distribution"],
            "environmental_impact": production_scenarios[best_scenario[0]][
                "environmental"
            ],
            "economic_metrics": production_scenarios[best_scenario[0]]["economic"],
            "nutritional_adequacy": production_scenarios[best_scenario[0]]["nutrition"],
        }

    def _generate_production_scenarios(self) -> dict[str, dict[str, Any]]:
        """Generate different production scenarios for optimization"""
        scenarios = {}

        # Scenario 1: Maximize local production
        scenarios["max_local"] = {
            "allocation": {"local_production": 0.8, "imports": 0.2},
            "distribution": {"direct_to_consumer": 0.6, "local_markets": 0.4},
            "environmental": {"carbon_footprint": 0.3, "water_use": 0.4},
            "economic": {"local_jobs": 100, "economic_multiplier": 1.8},
            "nutrition": {"nutritional_diversity": 0.9, "food_security": 0.95},
        }

        # Scenario 2: Minimize transport costs
        scenarios["min_transport"] = {
            "allocation": {"local_production": 0.9, "imports": 0.1},
            "distribution": {"direct_to_consumer": 0.8, "local_markets": 0.2},
            "environmental": {"carbon_footprint": 0.2, "water_use": 0.5},
            "economic": {"local_jobs": 120, "economic_multiplier": 2.0},
            "nutrition": {"nutritional_diversity": 0.8, "food_security": 0.98},
        }

        # Scenario 3: Maximize nutrition
        scenarios["max_nutrition"] = {
            "allocation": {"local_production": 0.7, "imports": 0.3},
            "distribution": {"direct_to_consumer": 0.5, "local_markets": 0.5},
            "environmental": {"carbon_footprint": 0.4, "water_use": 0.3},
            "economic": {"local_jobs": 80, "economic_multiplier": 1.5},
            "nutrition": {"nutritional_diversity": 0.95, "food_security": 0.90},
        }

        return scenarios

    def _transport_objective(self, scenario: dict[str, Any]) -> float:
        """Objective function for minimizing transport"""
        # Lower transport scores are better
        return float(1.0 / (scenario["environmental"]["carbon_footprint"] + 0.1))

    def _nutrition_objective(self, scenario: dict[str, Any]) -> float:
        """Objective function for maximizing nutrition"""
        return float(
            scenario["nutrition"]["nutritional_diversity"]
            * scenario["nutrition"]["food_security"]
        )

    def _environmental_objective(self, scenario: dict[str, Any]) -> float:
        """Objective function for minimizing environmental impact"""
        # Combined environmental score (lower is better)
        return float(
            1.0
            / (
                scenario["environmental"]["carbon_footprint"]
                + scenario["environmental"]["water_use"]
                + 0.1
            )
        )

    def _economic_objective(self, scenario: dict[str, Any]) -> float:
        """Objective function for maximizing economic efficiency"""
        return float(
            scenario["economic"]["economic_multiplier"]
            * scenario["economic"]["local_jobs"]
            / 100
        )

    def _find_pareto_optimal(
        self, scenario_scores: dict[str, dict[str, float]], weights: dict[str, float]
    ) -> tuple[str, float]:
        """Find Pareto optimal scenario using weighted sum"""
        best_scenario: str = ""
        best_score = -float("inf")

        for scenario_name, scores in scenario_scores.items():
            # Calculate weighted score
            weighted_score = sum(scores[obj] * weights[obj] for obj in weights.keys())

            if weighted_score > best_score:
                best_score = weighted_score
                best_scenario = scenario_name

        return best_scenario, float(best_score)

    def calculate_food_miles(self, producer_id: str, consumer_id: str) -> float:
        """Calculate food miles between producer and consumer"""
        producer_loc = self.food_producers[producer_id]["location"]
        consumer_loc = self.food_consumers[consumer_id]["location"]

        return self._calculate_distance(producer_loc, consumer_loc)

    def _calculate_distance(
        self, loc1: tuple[float, float], loc2: tuple[float, float]
    ) -> float:
        """Calculate distance between two locations"""
        return float(
            np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) * 111
        )  # km


class CarbonMarkets:
    """Cap-and-trade carbon market with allowance allocation and compliance.

    Supports regional carbon trading, tracks emissions against allowances,
    and enables credit generation from sequestration projects.
    """

    def __init__(self, market_design: BioregionalMarketDesign):
        self.market_design = market_design
        self.allowances: dict[str, float] = {}  # participant_id → tonnes CO2
        self.emissions: dict[str, float] = {}  # participant_id → tonnes CO2
        self.price_per_tonne: float = 50.0
        self.trade_history: list[dict[str, Any]] = []
        self.sequestration_projects: dict[str, dict[str, Any]] = {}

    def set_cap(self, total_cap: float, allocation: dict[str, float]) -> dict[str, Any]:
        """Set the emissions cap and allocate allowances.

        Args:
            total_cap: Total allowable emissions (tonnes CO2).
            allocation: Dict mapping participant_id → share (0-1).

        Returns:
            Dict with per-participant allowances.
        """
        self.allowances = {}
        for pid, share in allocation.items():
            self.allowances[pid] = round(total_cap * share, 2)

        return {
            "total_cap": total_cap,
            "participant_allowances": dict(self.allowances),
            "n_participants": len(self.allowances),
        }

    def report_emissions(self, participant_id: str, emissions: float) -> dict[str, Any]:
        """Report emissions for a participant.

        Args:
            participant_id: Participant identifier.
            emissions: Tonnes of CO2 emitted.

        Returns:
            Dict with balance and compliance status.
        """
        self.emissions[participant_id] = (
            self.emissions.get(participant_id, 0) + emissions
        )
        allowance = self.allowances.get(participant_id, 0)
        balance = allowance - self.emissions[participant_id]

        return {
            "participant_id": participant_id,
            "emissions_reported": emissions,
            "cumulative_emissions": self.emissions[participant_id],
            "allowance": allowance,
            "balance": round(balance, 2),
            "compliant": balance >= 0,
        }

    def trade_allowances(
        self,
        seller_id: str,
        buyer_id: str,
        tonnes: float,
        price_per_tonne: float | None = None,
    ) -> dict[str, Any]:
        """Execute an allowance trade between participants.

        Args:
            seller_id: Selling participant.
            buyer_id: Buying participant.
            tonnes: Tonnes of CO2 allowance to transfer.
            price_per_tonne: Agreed price; defaults to market price.

        Returns:
            Dict with trade details and updated balances.
        """
        price = price_per_tonne or self.price_per_tonne
        seller_balance = self.allowances.get(seller_id, 0) - self.emissions.get(
            seller_id, 0
        )

        if seller_balance < tonnes:
            return {
                "error": f"Seller {seller_id} has insufficient surplus ({seller_balance:.2f} t)"
            }

        self.allowances[seller_id] -= tonnes
        self.allowances[buyer_id] = self.allowances.get(buyer_id, 0) + tonnes
        total_cost = round(price * tonnes, 2)

        trade = {
            "seller_id": seller_id,
            "buyer_id": buyer_id,
            "tonnes": tonnes,
            "price_per_tonne": price,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat(),
        }
        self.trade_history.append(trade)
        self.market_design.transaction_history.append(trade)

        return {
            "trade": trade,
            "seller_new_allowance": self.allowances[seller_id],
            "buyer_new_allowance": self.allowances[buyer_id],
        }

    def register_sequestration_project(
        self,
        project_id: str,
        asset_id: str,
        annual_sequestration: float,
        duration_years: int = 20,
    ) -> dict[str, Any]:
        """Register a carbon sequestration project for credit generation.

        Args:
            project_id: Unique project identifier.
            asset_id: ID of the underlying bioregional asset.
            annual_sequestration: Tonnes CO2 sequestered per year.
            duration_years: Project crediting period.

        Returns:
            Dict with project details and projected credits.
        """
        project = {
            "project_id": project_id,
            "asset_id": asset_id,
            "annual_sequestration": annual_sequestration,
            "duration_years": duration_years,
            "total_projected_credits": round(annual_sequestration * duration_years, 2),
            "credits_issued": 0.0,
            "status": "registered",
        }
        self.sequestration_projects[project_id] = project
        return project

    def check_compliance(self) -> dict[str, Any]:
        """Check compliance for all participants.

        Returns:
            Dict with per-participant compliance and summary.
        """
        results: dict[str, dict[str, Any]] = {}
        compliant_count = 0
        total_surplus = 0.0
        total_deficit = 0.0

        for pid, allowance in self.allowances.items():
            emitted = self.emissions.get(pid, 0)
            balance = allowance - emitted
            is_compliant = balance >= 0

            if is_compliant:
                compliant_count += 1
                total_surplus += balance
            else:
                total_deficit += abs(balance)

            results[pid] = {
                "allowance": allowance,
                "emissions": emitted,
                "balance": round(balance, 2),
                "compliant": is_compliant,
            }

        return {
            "participant_results": results,
            "n_compliant": compliant_count,
            "n_non_compliant": len(self.allowances) - compliant_count,
            "total_surplus_tonnes": round(total_surplus, 2),
            "total_deficit_tonnes": round(total_deficit, 2),
        }


class WaterMarkets:
    """Water rights trading and watershed allocation markets.

    Manages water rights registration, priority-based allocation, and
    inter-participant trading within watershed boundaries.
    """

    def __init__(self, market_design: BioregionalMarketDesign):
        self.market_design = market_design
        self.water_rights: dict[str, dict[str, Any]] = {}
        self.allocations: dict[str, float] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.watershed_supply: float = 0.0

    def set_watershed_supply(self, annual_supply_m3: float) -> dict[str, Any]:
        """Set the total available water supply for the watershed.

        Args:
            annual_supply_m3: Total annual water supply in cubic metres.

        Returns:
            Dict with supply summary.
        """
        self.watershed_supply = annual_supply_m3
        return {
            "annual_supply_m3": annual_supply_m3,
            "total_allocated": round(sum(self.allocations.values()), 2),
            "available": round(annual_supply_m3 - sum(self.allocations.values()), 2),
        }

    def register_water_right(
        self,
        right_id: str,
        holder_id: str,
        volume_m3: float,
        priority: int = 1,
        use_type: str = "agricultural",
    ) -> dict[str, Any]:
        """Register a water right.

        Args:
            right_id: Unique right identifier.
            holder_id: Holder participant ID.
            volume_m3: Annual volume entitlement (m³).
            priority: Priority level (1 = highest / senior).
            use_type: Usage category (agricultural, municipal, industrial,
                      environmental).

        Returns:
            Dict with registered right details.
        """
        right = {
            "right_id": right_id,
            "holder_id": holder_id,
            "volume_m3": volume_m3,
            "priority": priority,
            "use_type": use_type,
            "status": "active",
        }
        self.water_rights[right_id] = right
        self.allocations[holder_id] = self.allocations.get(holder_id, 0) + volume_m3

        return right

    def allocate_by_priority(self) -> dict[str, Any]:
        """Allocate water by priority when supply is scarce.

        Rights are filled in priority order (lower number = higher priority).
        When supply is exhausted, remaining rights get proportional shares.

        Returns:
            Dict with per-right allocation and shortfall.
        """
        sorted_rights = sorted(self.water_rights.values(), key=lambda r: r["priority"])
        remaining = self.watershed_supply
        allocations: dict[str, float] = {}
        shortfalls: dict[str, float] = {}

        for right in sorted_rights:
            rid = right["right_id"]
            requested = right["volume_m3"]

            if remaining >= requested:
                allocations[rid] = requested
                remaining -= requested
                shortfalls[rid] = 0.0
            else:
                allocations[rid] = max(remaining, 0.0)
                shortfalls[rid] = round(requested - max(remaining, 0.0), 2)
                remaining = 0.0

        total_allocated = sum(allocations.values())
        total_shortfall = sum(shortfalls.values())

        return {
            "allocations": {k: round(v, 2) for k, v in allocations.items()},
            "shortfalls": shortfalls,
            "total_allocated_m3": round(total_allocated, 2),
            "total_shortfall_m3": round(total_shortfall, 2),
            "supply_utilisation": round(
                total_allocated / max(self.watershed_supply, 1e-10), 4
            ),
        }

    def trade_water_rights(
        self,
        seller_id: str,
        buyer_id: str,
        volume_m3: float,
        price_per_m3: float,
    ) -> dict[str, Any]:
        """Trade water allocation between participants.

        Args:
            seller_id: Selling participant.
            buyer_id: Buying participant.
            volume_m3: Volume to transfer (m³).
            price_per_m3: Price per cubic metre.

        Returns:
            Dict with trade details and updated allocations.
        """
        seller_alloc = self.allocations.get(seller_id, 0)

        if seller_alloc < volume_m3:
            return {
                "error": f"Seller {seller_id} has insufficient allocation "
                f"({seller_alloc:.2f} m³)"
            }

        self.allocations[seller_id] -= volume_m3
        self.allocations[buyer_id] = self.allocations.get(buyer_id, 0) + volume_m3
        total_cost = round(price_per_m3 * volume_m3, 2)

        trade = {
            "seller_id": seller_id,
            "buyer_id": buyer_id,
            "volume_m3": volume_m3,
            "price_per_m3": price_per_m3,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat(),
        }
        self.trade_history.append(trade)
        self.market_design.transaction_history.append(trade)

        return {
            "trade": trade,
            "seller_new_allocation": round(self.allocations[seller_id], 2),
            "buyer_new_allocation": round(self.allocations[buyer_id], 2),
        }

    def watershed_balance(self) -> dict[str, Any]:
        """Get overall watershed water balance.

        Returns:
            Dict with supply, demand, and balance metrics.
        """
        total_demand = sum(r["volume_m3"] for r in self.water_rights.values())
        total_allocated = sum(self.allocations.values())

        return {
            "watershed_supply_m3": self.watershed_supply,
            "total_demand_m3": round(total_demand, 2),
            "total_allocated_m3": round(total_allocated, 2),
            "unallocated_m3": round(self.watershed_supply - total_allocated, 2),
            "demand_to_supply_ratio": round(
                total_demand / max(self.watershed_supply, 1e-10), 4
            ),
            "n_active_rights": len(self.water_rights),
            "n_trades": len(self.trade_history),
        }


# Example usage and testing functions
def example_bioregional_market() -> BioregionalMarketDesign:
    """
    Example usage of bioregional market design
    """
    print("=== Bioregional Market Design Example ===")

    # Create bioregion boundary (baseline)
    bioregion = gpd.GeoDataFrame()

    # Initialize market design
    market = BioregionalMarketDesign(bioregion)

    # Create sample bioregional asset
    forest_asset = BioregionalAsset(
        asset_id="forest_001",
        asset_type="forest",
        location=(45.0, -120.0),
        area_hectares=100.0,
        ecological_attributes={
            "carbon_storage": 500.0,  # tons CO2
            "biodiversity_index": 0.8,
            "water_filtration": 0.9,
        },
        economic_attributes={"market_value": 500000, "annual_income": 10000},
        ownership_type="community",
        management_regime="sustainable_forestry",
        ecosystem_services={
            "carbon_sequestration": 10.0,  # tons CO2/year
            "biodiversity_habitat": 0.8,
            "water_regulation": 0.9,
            "recreation": 0.7,
        },
    )

    # Register asset
    market.register_asset(forest_asset)

    # Create ecosystem service credit
    quality_params = {
        "additionality": 0.9,
        "permanence": 0.8,
        "measurability": 0.85,
        "leakage_risk": 0.1,
        "co_benefits": 0.7,
        "temporal_profile": "permanent",
    }

    carbon_credit = market.create_ecosystem_service_credit(
        asset_id="forest_001",
        service_type="carbon",
        quantity=10.0,
        quality_parameters=quality_params,
    )

    print(f"Created Carbon Credit: {carbon_credit.credit_id}")
    print(f"Quality Tier: {carbon_credit.quality_tier}")
    print(f"Price per Unit: ${carbon_credit.price_per_unit:.2f}")
    print(f"Co-benefits: {carbon_credit.co_benefits}")

    return market


if __name__ == "__main__":
    # Run example
    example_market = example_bioregional_market()
