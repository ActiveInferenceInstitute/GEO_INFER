"""
Natural Capital Accounting Module

Provides SEEA-aligned natural capital accounting including ecosystem asset
valuation, biodiversity credit calculation, carbon stock accounting, and
water resource balance tracking.
"""

from typing import Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference unit values (USD/ha/yr) aligned with SEEA-EA & TEEB databases
# ---------------------------------------------------------------------------
_ASSET_UNIT_VALUES: dict[str, float] = {
    "forest": 3800.0,
    "wetland": 6500.0,
    "grassland": 1200.0,
    "cropland": 2800.0,
    "coastal": 7200.0,
    "freshwater": 5400.0,
    "urban_green": 1600.0,
    "desert": 200.0,
    "tundra": 350.0,
}

# Carbon prices (USD/tCO2e) by market type
_CARBON_PRICES: dict[str, float] = {
    "voluntary": 12.0,
    "compliance": 45.0,
    "social_cost": 80.0,
}


class NaturalCapitalAccounting:
    """SEEA-aligned natural capital accounting and valuation.

    Implements the System of Environmental-Economic Accounting –
    Ecosystem Accounting (SEEA-EA) framework for tracking ecosystem
    assets and their monetary values over time.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize natural capital accounting.

        Args:
            config: Optional configuration overrides. Supported keys:
                - ``discount_rate`` (float): Annual discount rate for NPV (default 0.03).
                - ``time_horizon`` (int): Planning horizon in years (default 25).
                - ``unit_values`` (dict): Override default per-ha unit values.
        """
        self.config = config or {}
        self.discount_rate = float(self.config.get("discount_rate", 0.03))
        self.time_horizon = int(self.config.get("time_horizon", 25))
        self.unit_values = {**_ASSET_UNIT_VALUES, **self.config.get("unit_values", {})}
        logger.info(
            "NaturalCapitalAccounting initialized: discount_rate=%.2f, horizon=%d yr",
            self.discount_rate,
            self.time_horizon,
        )

    def account_assets(self, assets: list[dict[str, Any]]) -> pd.DataFrame:
        """Account for natural capital assets using SEEA-EA framework.

        Each asset dict should contain:
            - ``name`` (str): Asset identifier.
            - ``type`` (str): One of the recognised asset types.
            - ``area_ha`` (float): Area in hectares.
            - ``condition_score`` (float, 0-1): Ecosystem condition index.
            - ``trend`` (float, optional): Annual change rate (default 0).

        Returns:
            DataFrame with columns: name, type, area_ha, condition_score,
            annual_value, npv, trend, projected_value_10yr.
        """
        logger.info("Accounting %d natural capital assets", len(assets))
        rows: list[dict[str, Any]] = []
        for asset in assets:
            a_type = asset.get("type", "forest").lower()
            area = float(asset.get("area_ha", 0.0))
            condition = float(asset.get("condition_score", 1.0))
            trend = float(asset.get("trend", 0.0))
            unit_val = self.unit_values.get(a_type, 1000.0)

            annual_value = round(area * unit_val * condition, 2)
            npv = round(
                annual_value
                * sum(
                    1 / (1 + self.discount_rate) ** t for t in range(self.time_horizon)
                ),
                2,
            )
            projected_condition = min(1.0, max(0.0, condition + trend * 10))
            projected_value = round(area * unit_val * projected_condition, 2)

            rows.append(
                {
                    "name": asset.get("name", f"asset_{len(rows)}"),
                    "type": a_type,
                    "area_ha": area,
                    "condition_score": condition,
                    "annual_value": annual_value,
                    "npv": npv,
                    "trend": trend,
                    "projected_value_10yr": projected_value,
                }
            )
        df = pd.DataFrame(rows)
        logger.info("Total annual value: $%.2f", df["annual_value"].sum())
        return df

    def value_assets(self, assets: pd.DataFrame) -> pd.Series:
        """Compute monetary values for pre-accounted assets DataFrame.

        Expects DataFrame from ``account_assets`` or with columns
        ``area_ha``, ``type``, ``condition_score``.

        Returns:
            Series of annual values (USD) indexed like the input.
        """
        logger.info("Valuing %d assets", len(assets))

        def _row_value(row: pd.Series) -> float:
            a_type = str(row.get("type", "forest")).lower()
            area = float(row.get("area_ha", 0.0))
            condition = float(row.get("condition_score", 1.0))
            unit_val = self.unit_values.get(a_type, 1000.0)
            return float(round(area * unit_val * condition, 2))

        return assets.apply(_row_value, axis=1)


class EcosystemAssetsValuation:
    """Ecosystem assets valuation using benefit-transfer methodology.

    Produces a per-asset breakdown of provisioning, regulating, cultural,
    and supporting service values.
    """

    SERVICE_SHARES: dict[str, dict[str, float]] = {
        "forest": {
            "provisioning": 0.25,
            "regulating": 0.40,
            "cultural": 0.15,
            "supporting": 0.20,
        },
        "wetland": {
            "provisioning": 0.20,
            "regulating": 0.45,
            "cultural": 0.10,
            "supporting": 0.25,
        },
        "grassland": {
            "provisioning": 0.35,
            "regulating": 0.30,
            "cultural": 0.15,
            "supporting": 0.20,
        },
        "coastal": {
            "provisioning": 0.30,
            "regulating": 0.35,
            "cultural": 0.20,
            "supporting": 0.15,
        },
        "freshwater": {
            "provisioning": 0.40,
            "regulating": 0.25,
            "cultural": 0.15,
            "supporting": 0.20,
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.unit_values = {**_ASSET_UNIT_VALUES, **self.config.get("unit_values", {})}
        logger.info("EcosystemAssetsValuation initialized")

    def value_ecosystem_assets(self, assets: list[dict[str, Any]]) -> dict[str, float]:
        """Value ecosystem assets with service-category breakdown.

        Each asset dict requires ``type``, ``area_ha``, and optionally
        ``condition_score`` (0-1, default 1.0).

        Returns:
            Dict with per-service total values and grand total.
        """
        logger.info("Valuing %d ecosystem assets", len(assets))
        totals: dict[str, float] = {
            "provisioning": 0.0,
            "regulating": 0.0,
            "cultural": 0.0,
            "supporting": 0.0,
            "total": 0.0,
        }
        for asset in assets:
            a_type = asset.get("type", "forest").lower()
            area = float(asset.get("area_ha", 0.0))
            condition = float(asset.get("condition_score", 1.0))
            unit_val = self.unit_values.get(a_type, 1000.0)
            total_val = area * unit_val * condition

            shares = self.SERVICE_SHARES.get(
                a_type,
                {
                    "provisioning": 0.25,
                    "regulating": 0.35,
                    "cultural": 0.15,
                    "supporting": 0.25,
                },
            )
            for service, share in shares.items():
                totals[service] += total_val * share
            totals["total"] += total_val

        # Round
        totals = {k: round(v, 2) for k, v in totals.items()}
        logger.info("Ecosystem valuation total: $%.2f", totals["total"])
        return totals


class BiodiversityCredits:
    """Biodiversity credit calculation aligned with emerging frameworks.

    Uses species richness, habitat quality, and area metrics to generate
    tradeable biodiversity units following the Taskforce on Nature-related
    Financial Disclosures (TNFD) approach.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.credit_price = float(self.config.get("credit_price", 25.0))  # USD per unit
        self.baseline_species_per_ha = float(
            self.config.get("baseline_species_per_ha", 50.0)
        )
        logger.info(
            "BiodiversityCredits initialized: price=$%.2f/unit", self.credit_price
        )

    def calculate_credits(self, biodiversity_data: dict[str, Any]) -> float:
        """Calculate biodiversity credits for a site.

        Args:
            biodiversity_data: Dict with keys:
                - ``area_ha`` (float): Site area in hectares.
                - ``species_count`` (int): Observed species count.
                - ``habitat_quality`` (float, 0-1): Habitat condition.
                - ``connectivity_score`` (float, 0-1): Landscape connectivity.
                - ``management_effectiveness`` (float, 0-1): Conservation mgmt score.

        Returns:
            Number of biodiversity credit units (float).
        """
        area = float(biodiversity_data.get("area_ha", 0.0))
        species = int(biodiversity_data.get("species_count", 0))
        quality = float(biodiversity_data.get("habitat_quality", 0.5))
        connectivity = float(biodiversity_data.get("connectivity_score", 0.5))
        management = float(biodiversity_data.get("management_effectiveness", 0.5))

        # Species uplift ratio
        species_ratio = min(2.0, species / max(self.baseline_species_per_ha, 1.0))

        # Composite quality multiplier (geometric mean)
        quality_multiplier = (quality * connectivity * management) ** (1 / 3)

        credits = area * species_ratio * quality_multiplier
        credits = float(round(credits, 2))

        logger.info(
            "Biodiversity credits: %.2f units for %.1f ha (species_ratio=%.2f, quality=%.2f)",
            credits,
            area,
            species_ratio,
            quality_multiplier,
        )
        return credits


class CarbonAccounting:
    """Carbon stock and flow accounting following GHG Protocol Land Sector.

    Tracks above-ground biomass (AGB), below-ground biomass (BGB),
    soil organic carbon (SOC), and dead organic matter across carbon pools.
    """

    # Default carbon densities (tC/ha) by ecosystem type
    CARBON_DENSITIES: dict[str, dict[str, float]] = {
        "forest": {"agb": 120.0, "bgb": 30.0, "soc": 80.0, "dom": 15.0},
        "wetland": {"agb": 20.0, "bgb": 15.0, "soc": 200.0, "dom": 10.0},
        "grassland": {"agb": 8.0, "bgb": 20.0, "soc": 60.0, "dom": 3.0},
        "cropland": {"agb": 5.0, "bgb": 3.0, "soc": 40.0, "dom": 1.0},
        "coastal": {"agb": 15.0, "bgb": 12.0, "soc": 150.0, "dom": 8.0},
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.co2_factor = 3.667  # tCO2 per tC
        self.market = self.config.get("carbon_market", "voluntary")
        self.carbon_price = _CARBON_PRICES.get(self.market, 12.0)
        logger.info(
            "CarbonAccounting initialized: market=%s, price=$%.2f/tCO2e",
            self.market,
            self.carbon_price,
        )

    def account_carbon(self, carbon_data: dict[str, Any]) -> pd.DataFrame:
        """Account for carbon stocks and flows across ecosystem parcels.

        Args:
            carbon_data: Dict with key ``parcels`` — list of dicts each containing:
                - ``name`` (str): Parcel identifier.
                - ``type`` (str): Ecosystem type.
                - ``area_ha`` (float): Area in hectares.
                - ``condition_score`` (float, 0-1): Condition modifier.
                - ``annual_sequestration_rate`` (float, tC/ha/yr, optional): Override.

        Returns:
            DataFrame with carbon stock, CO2e, monetary value per pool per parcel.
        """
        parcels = carbon_data.get("parcels", [])
        logger.info("Carbon accounting for %d parcels", len(parcels))

        rows: list[dict[str, Any]] = []
        for parcel in parcels:
            p_type = parcel.get("type", "forest").lower()
            area = float(parcel.get("area_ha", 0.0))
            condition = float(parcel.get("condition_score", 1.0))
            densities = self.CARBON_DENSITIES.get(
                p_type, {"agb": 50.0, "bgb": 15.0, "soc": 60.0, "dom": 5.0}
            )

            total_c = 0.0
            for pool, density in densities.items():
                stock = round(area * density * condition, 2)
                co2e = round(stock * self.co2_factor, 2)
                value = round(co2e * self.carbon_price, 2)
                total_c += stock
                rows.append(
                    {
                        "parcel": parcel.get("name", "unnamed"),
                        "type": p_type,
                        "pool": pool,
                        "carbon_stock_tC": stock,
                        "co2e_t": co2e,
                        "monetary_value_usd": value,
                    }
                )
            logger.debug("Parcel %s: total %.2f tC", parcel.get("name"), total_c)

        df = pd.DataFrame(rows)
        logger.info(
            "Total carbon stock: %.2f tC, value: $%.2f",
            df["carbon_stock_tC"].sum(),
            df["monetary_value_usd"].sum(),
        )
        return df


class WaterResourceAccounting:
    """Water resource accounting using a simplified water balance model.

    Tracks precipitation, evapotranspiration, runoff, and groundwater
    recharge to produce a water balance sheet for catchment areas.
    """

    # Default hydrological coefficients by land cover
    HYDRO_COEFFICIENTS: dict[str, dict[str, float]] = {
        "forest": {
            "et_fraction": 0.55,
            "runoff_fraction": 0.20,
            "recharge_fraction": 0.25,
        },
        "wetland": {
            "et_fraction": 0.65,
            "runoff_fraction": 0.10,
            "recharge_fraction": 0.25,
        },
        "grassland": {
            "et_fraction": 0.50,
            "runoff_fraction": 0.30,
            "recharge_fraction": 0.20,
        },
        "cropland": {
            "et_fraction": 0.45,
            "runoff_fraction": 0.35,
            "recharge_fraction": 0.20,
        },
        "urban": {
            "et_fraction": 0.15,
            "runoff_fraction": 0.75,
            "recharge_fraction": 0.10,
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.water_price = float(self.config.get("water_price_per_m3", 0.50))  # USD/m³
        logger.info(
            "WaterResourceAccounting initialized: water_price=$%.2f/m³",
            self.water_price,
        )

    def account_water(self, water_data: dict[str, Any]) -> pd.DataFrame:
        """Account for water resources using a catchment water balance.

        Args:
            water_data: Dict with key ``catchments`` — list of dicts each containing:
                - ``name`` (str): Catchment identifier.
                - ``type`` (str): Dominant land cover type.
                - ``area_ha`` (float): Catchment area in hectares.
                - ``annual_precip_mm`` (float): Annual precipitation in mm.
                - ``water_demand_m3`` (float, optional): Annual demand in m³.

        Returns:
            DataFrame with water balance components per catchment.
        """
        catchments = water_data.get("catchments", [])
        logger.info("Water accounting for %d catchments", len(catchments))

        rows: list[dict[str, Any]] = []
        for c in catchments:
            c_type = c.get("type", "grassland").lower()
            area = float(c.get("area_ha", 0.0))
            precip_mm = float(c.get("annual_precip_mm", 800.0))
            demand = float(c.get("water_demand_m3", 0.0))

            coeffs = self.HYDRO_COEFFICIENTS.get(
                c_type,
                {
                    "et_fraction": 0.45,
                    "runoff_fraction": 0.30,
                    "recharge_fraction": 0.25,
                },
            )

            # Convert precip to m³ (1 mm over 1 ha = 10 m³)
            total_precip_m3 = round(precip_mm * area * 10, 2)
            et_m3 = round(total_precip_m3 * coeffs["et_fraction"], 2)
            runoff_m3 = round(total_precip_m3 * coeffs["runoff_fraction"], 2)
            recharge_m3 = round(total_precip_m3 * coeffs["recharge_fraction"], 2)
            available_m3 = round(runoff_m3 + recharge_m3, 2)
            surplus_m3 = round(available_m3 - demand, 2)
            water_value = round(available_m3 * self.water_price, 2)

            rows.append(
                {
                    "catchment": c.get("name", "unnamed"),
                    "type": c_type,
                    "area_ha": area,
                    "total_precip_m3": total_precip_m3,
                    "evapotranspiration_m3": et_m3,
                    "runoff_m3": runoff_m3,
                    "recharge_m3": recharge_m3,
                    "available_m3": available_m3,
                    "demand_m3": demand,
                    "surplus_deficit_m3": surplus_m3,
                    "water_value_usd": water_value,
                }
            )

        df = pd.DataFrame(rows)
        logger.info(
            "Total available water: %.2f m³, value: $%.2f",
            df["available_m3"].sum(),
            df["water_value_usd"].sum(),
        )
        return df
