"""
Ecosystem Services Valuation Module

This module provides ecosystem services valuation and modeling capabilities
using benefit-transfer methodology, unit-value approaches, and flow-based
accounting aligned with TEEB (The Economics of Ecosystems and Biodiversity).
"""

from typing import Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Default per-hectare annual values (USD/ha/yr) from TEEB meta-analysis
DEFAULT_SERVICE_VALUES = {
    "provisioning": {
        "food": 1_385.0,
        "water": 1_808.0,
        "raw_materials": 721.0,
        "genetic_resources": 483.0,
        "medicinal_resources": 181.0,
    },
    "regulating": {
        "air_quality": 449.0,
        "climate_regulation": 2_355.0,
        "water_purification": 1_965.0,
        "erosion_prevention": 576.0,
        "pollination": 1_023.0,
        "pest_control": 417.0,
        "natural_hazard_mitigation": 3_232.0,
    },
    "cultural": {
        "recreation": 1_171.0,
        "tourism": 867.0,
        "aesthetic": 534.0,
        "spiritual": 287.0,
        "educational": 143.0,
    },
    "supporting": {
        "nutrient_cycling": 1_446.0,
        "soil_formation": 576.0,
        "primary_production": 2_010.0,
        "habitat": 3_218.0,
        "biodiversity": 1_867.0,
    },
}


class EcosystemServicesValuation:
    """
    Ecosystem services valuation using benefit-transfer methodology.

    Supports per-hectare unit-value transfer, adjusted for local context
    via purchasing-power-parity (PPP) and biome quality multipliers.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ecosystem services valuation.

        Args:
            config: Optional configuration with keys:
                - ppp_factor: purchasing-power-parity adjustment (default 1.0)
                - discount_rate: for NPV calculations (default 0.03)
                - time_horizon: years for NPV (default 30)
                - service_values: override default per-ha values
        """
        self.config = config or {}
        self.ppp_factor = self.config.get("ppp_factor", 1.0)
        self.discount_rate = self.config.get("discount_rate", 0.03)
        self.time_horizon = self.config.get("time_horizon", 30)
        self.service_values = self.config.get("service_values", DEFAULT_SERVICE_VALUES)
        logger.info(
            "EcosystemServicesValuation initialized (ppp=%.2f, discount=%.3f, horizon=%d yr)",
            self.ppp_factor,
            self.discount_rate,
            self.time_horizon,
        )

    def value_services(self, services: list[dict[str, Any]]) -> dict[str, float]:
        """Value a list of ecosystem services.

        Each service dict should contain:
            - category: one of 'provisioning', 'regulating', 'cultural', 'supporting'
            - type: specific service type (e.g. 'food', 'climate_regulation')
            - area_ha: area in hectares
            - quality_factor: optional biome quality multiplier (0-1, default 1.0)

        Returns:
            Dict mapping service type -> annual value (USD), plus 'total_annual'
            and 'total_npv'.
        """
        logger.info("Valuing %d ecosystem services", len(services))
        results: dict[str, float] = {}
        total_annual = 0.0

        for svc in services:
            category = svc.get("category", "provisioning")
            svc_type = svc.get("type", "food")
            area_ha = float(svc.get("area_ha", 0.0))
            quality = float(svc.get("quality_factor", 1.0))

            base_value = self.service_values.get(category, {}).get(svc_type, 0.0)
            annual_value = base_value * area_ha * quality * self.ppp_factor

            key = f"{category}.{svc_type}"
            results[key] = round(annual_value, 2)
            total_annual += annual_value
            logger.debug(
                "  %s: %.2f USD/yr (%.1f ha × %.0f USD/ha × quality=%.2f × ppp=%.2f)",
                key,
                annual_value,
                area_ha,
                base_value,
                quality,
                self.ppp_factor,
            )

        results["total_annual"] = round(total_annual, 2)
        results["total_npv"] = round(self._compute_npv(total_annual), 2)
        logger.info(
            "Total ecosystem value: %.2f USD/yr, NPV: %.2f USD",
            results["total_annual"],
            results["total_npv"],
        )
        return results

    def estimate_value(self, services: list[dict[str, Any]]) -> dict[str, float]:
        """Alias for value_services for API compatibility."""
        return self.value_services(services)

    def _compute_npv(self, annual_value: float) -> float:
        """Compute net present value of an annual flow."""
        r = self.discount_rate
        if r <= 0:
            return float(annual_value * self.time_horizon)
        factor = (1 - (1 + r) ** (-self.time_horizon)) / r
        return float(annual_value * factor)


class ProvisioningServices:
    """Provisioning ecosystem services valuation (food, water, raw materials)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.unit_values = self.config.get(
            "unit_values", DEFAULT_SERVICE_VALUES["provisioning"]
        )
        logger.info(
            "ProvisioningServices initialized with %d service types",
            len(self.unit_values),
        )

    def value_provisioning(self, data: dict[str, Any]) -> float:
        """Value provisioning services for a region.

        Args:
            data: Dict with:
                - area_ha: total area in hectares
                - land_cover: dict mapping cover type -> fraction (sums to 1)
                - services: list of provisioning types to include
                - quality_factor: biome quality multiplier (default 1.0)

        Returns:
            Total annual provisioning value (USD/yr).
        """
        area_ha = float(data.get("area_ha", 0.0))
        quality = float(data.get("quality_factor", 1.0))
        requested = data.get("services", list(self.unit_values.keys()))

        total = 0.0
        for svc_type in requested:
            base = self.unit_values.get(svc_type, 0.0)
            value = base * area_ha * quality
            total += value
            logger.debug("  Provisioning %s: %.2f USD/yr", svc_type, value)

        logger.info("Total provisioning value: %.2f USD/yr (%.1f ha)", total, area_ha)
        return round(total, 2)


class RegulatingServices:
    """Regulating ecosystem services (air quality, climate, water, etc.)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.unit_values = self.config.get(
            "unit_values", DEFAULT_SERVICE_VALUES["regulating"]
        )
        logger.info(
            "RegulatingServices initialized with %d service types",
            len(self.unit_values),
        )

    def value_regulating(self, data: dict[str, Any]) -> float:
        """Value regulating services for a region.

        Args:
            data: Dict with area_ha, services list, quality_factor.

        Returns:
            Total annual regulating value (USD/yr).
        """
        area_ha = float(data.get("area_ha", 0.0))
        quality = float(data.get("quality_factor", 1.0))
        requested = data.get("services", list(self.unit_values.keys()))

        total = 0.0
        for svc_type in requested:
            base = self.unit_values.get(svc_type, 0.0)
            value = base * area_ha * quality
            total += value
            logger.debug("  Regulating %s: %.2f USD/yr", svc_type, value)

        logger.info("Total regulating value: %.2f USD/yr (%.1f ha)", total, area_ha)
        return round(total, 2)


class CulturalServices:
    """Cultural ecosystem services (recreation, tourism, aesthetics)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.unit_values = self.config.get(
            "unit_values", DEFAULT_SERVICE_VALUES["cultural"]
        )
        logger.info(
            "CulturalServices initialized with %d service types", len(self.unit_values)
        )

    def value_cultural(self, data: dict[str, Any]) -> float:
        """Value cultural services for a region.

        Args:
            data: Dict with area_ha, services list, quality_factor,
                  and optional visitor_count for recreation/tourism.

        Returns:
            Total annual cultural value (USD/yr).
        """
        area_ha = float(data.get("area_ha", 0.0))
        quality = float(data.get("quality_factor", 1.0))
        visitor_multiplier = data.get("visitor_multiplier", 1.0)
        requested = data.get("services", list(self.unit_values.keys()))

        total = 0.0
        for svc_type in requested:
            base = self.unit_values.get(svc_type, 0.0)
            mult = visitor_multiplier if svc_type in ("recreation", "tourism") else 1.0
            value = base * area_ha * quality * mult
            total += value
            logger.debug("  Cultural %s: %.2f USD/yr", svc_type, value)

        logger.info("Total cultural value: %.2f USD/yr (%.1f ha)", total, area_ha)
        return round(total, 2)


class SupportingServices:
    """Supporting ecosystem services (nutrient cycling, soil, habitat)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.unit_values = self.config.get(
            "unit_values", DEFAULT_SERVICE_VALUES["supporting"]
        )
        logger.info(
            "SupportingServices initialized with %d service types",
            len(self.unit_values),
        )

    def value_supporting(self, data: dict[str, Any]) -> float:
        """Value supporting services for a region.

        Args:
            data: Dict with area_ha, services list, quality_factor.

        Returns:
            Total annual supporting value (USD/yr).
        """
        area_ha = float(data.get("area_ha", 0.0))
        quality = float(data.get("quality_factor", 1.0))
        requested = data.get("services", list(self.unit_values.keys()))

        total = 0.0
        for svc_type in requested:
            base = self.unit_values.get(svc_type, 0.0)
            value = base * area_ha * quality
            total += value
            logger.debug("  Supporting %s: %.2f USD/yr", svc_type, value)

        logger.info("Total supporting value: %.2f USD/yr (%.1f ha)", total, area_ha)
        return round(total, 2)


class ServiceFlowModeling:
    """Ecosystem service flow modeling between supply and demand areas.

    Models spatial flows of ecosystem services from providing areas
    to benefiting areas using gravity-model approaches.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.decay_rate = self.config.get("distance_decay_rate", 0.01)
        self.flow_threshold = self.config.get("flow_threshold", 0.01)
        logger.info(
            "ServiceFlowModeling initialized (decay=%.4f, threshold=%.3f)",
            self.decay_rate,
            self.flow_threshold,
        )

    def model_flows(self, flow_data: dict[str, Any]) -> pd.DataFrame:
        """Model ecosystem service flows between supply and demand areas.

        Args:
            flow_data: Dict with:
                - supply_areas: list of {id, lat, lon, capacity}
                - demand_areas: list of {id, lat, lon, demand}
                - service_type: type of service being modeled

        Returns:
            DataFrame with columns: source_id, target_id, flow_value,
            distance_km, flow_fraction.
        """
        supply_areas = flow_data.get("supply_areas", [])
        demand_areas = flow_data.get("demand_areas", [])
        service_type = flow_data.get("service_type", "generic")

        logger.info(
            "Modeling %s flows: %d supply → %d demand areas",
            service_type,
            len(supply_areas),
            len(demand_areas),
        )

        rows = []
        for supply in supply_areas:
            capacity = float(supply.get("capacity", 0.0))
            s_lat, s_lon = float(supply["lat"]), float(supply["lon"])

            for demand in demand_areas:
                d_lat, d_lon = float(demand["lat"]), float(demand["lon"])
                demand_val = float(demand.get("demand", 0.0))

                dist_km = self._haversine(s_lat, s_lon, d_lat, d_lon)
                decay = np.exp(-self.decay_rate * dist_km)
                flow_value = capacity * demand_val * decay

                if flow_value >= self.flow_threshold:
                    rows.append(
                        {
                            "source_id": supply["id"],
                            "target_id": demand["id"],
                            "flow_value": round(flow_value, 4),
                            "distance_km": round(dist_km, 2),
                            "flow_fraction": round(decay, 4),
                        }
                    )

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "source_id",
                    "target_id",
                    "flow_value",
                    "distance_km",
                    "flow_fraction",
                ]
            )

        logger.info(
            "Generated %d service flows (total value: %.2f)",
            len(df),
            df["flow_value"].sum() if not df.empty else 0.0,
        )
        return df

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km."""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )
        return float(R * 2 * np.arcsin(np.sqrt(a)))
