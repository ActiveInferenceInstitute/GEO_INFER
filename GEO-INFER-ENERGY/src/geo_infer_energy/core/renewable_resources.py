"""Renewable resource assessment module."""

import logging
from typing import Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class RenewableType(Enum):
    """Types of renewable energy sources."""

    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"
    ONSHORE_WIND = "onshore_wind"
    OFFSHORE_WIND = "offshore_wind"
    HYDROPOWER = "hydropower"
    GEOTHERMAL = "geothermal"
    BIOMASS = "biomass"
    WAVE = "wave"
    TIDAL = "tidal"


class SuitabilityClass(Enum):
    """Site suitability classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    MARGINAL = "marginal"
    UNSUITABLE = "unsuitable"


@dataclass
class RenewableSite:
    """Renewable energy site data."""

    site_id: str
    name: str
    location: tuple[float, float]
    resource_type: RenewableType
    capacity_mw: float
    capacity_factor: float
    annual_generation_gwh: float | None = None
    lcoe_usd_mwh: float | None = None
    land_area_km2: float | None = None


class RenewableResourceAssessor:
    """
    Comprehensive renewable energy resource assessment system.

    Provides renewable energy analysis including:
    - Solar, wind, hydro, geothermal potential
    - Site suitability assessment
    - Capacity factor estimation
    - LCOE calculation
    - Grid integration analysis
    """

    def __init__(self, config: dict | None = None):
        """Initialize renewable resource assessor.

        Documented contract: ``config`` is accepted and stored for API
        stability but its contents are currently ignored by this assessor
        (reserved argument; only ``WindAnalyzer`` consumes config keys).
        """
        self.config = config or {}

        # Resource-specific parameters
        self.efficiency = {
            RenewableType.SOLAR_PV: 0.20,
            RenewableType.SOLAR_THERMAL: 0.35,
            RenewableType.ONSHORE_WIND: 0.45,
            RenewableType.OFFSHORE_WIND: 0.50,
            RenewableType.HYDROPOWER: 0.85,
            RenewableType.GEOTHERMAL: 0.12,
            RenewableType.BIOMASS: 0.25,
        }

        # Capital costs (USD/kW)
        self.capital_costs = {
            RenewableType.SOLAR_PV: 1000,
            RenewableType.SOLAR_THERMAL: 4500,
            RenewableType.ONSHORE_WIND: 1300,
            RenewableType.OFFSHORE_WIND: 3500,
            RenewableType.HYDROPOWER: 2500,
            RenewableType.GEOTHERMAL: 4000,
            RenewableType.BIOMASS: 3500,
        }

        # Site registry
        self.site_registry: dict[str, RenewableSite] = {}

    def assess_solar_potential(
        self,
        solar_irradiance: xr.DataArray,
        slope: xr.DataArray | None = None,
        aspect: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess solar energy potential.

        Args:
            solar_irradiance: Solar irradiance (kWh/m²/day)
            slope: Optional terrain slope
            aspect: Optional terrain aspect

        Returns:
            Solar potential assessment
        """
        # Base potential from irradiance
        potential = solar_irradiance * 365  # Annual potential

        # Adjust for slope (optimal ~30 degrees)
        if slope is not None:
            optimal_slope = 30.0
            slope_factor = 1 - np.abs(slope - optimal_slope) / 90.0
            potential = potential * slope_factor

        # Adjust for aspect (south-facing optimal in northern hemisphere)
        if aspect is not None:
            # South = 180 degrees
            aspect_factor = np.cos(np.radians(aspect - 180)) * 0.5 + 0.5
            potential = potential * aspect_factor

        return xr.Dataset(
            {
                "solar_potential": potential,
                "annual_energy": potential * 0.2,  # 20% efficiency assumption
            }
        )

    def assess_wind_potential(
        self, wind_speed: xr.DataArray, elevation: xr.DataArray | None = None
    ) -> xr.Dataset:
        """
        Assess wind energy potential.

        Args:
            wind_speed: Wind speed (m/s)
            elevation: Optional elevation data

        Returns:
            Wind potential assessment
        """
        # Wind power is proportional to cube of wind speed
        wind_power = wind_speed**3

        # Adjust for elevation (higher = better typically)
        if elevation is not None:
            elevation_factor = (
                1
                + (elevation - elevation.min())
                / (elevation.max() - elevation.min() + 1e-10)
                * 0.2
            )
            wind_power = wind_power * elevation_factor

        # Documented contract: relative annual energy-potential index — wind
        # speed cubed (plus the elevation uplift) scaled by a constant 50%
        # capacity factor over 8760 hours; no air density, turbine power
        # curve, wake or availability losses are modeled, so the output is a
        # comparable index, not MW/MWh.
        energy_potential = wind_power * 0.5 * 8760

        return xr.Dataset(
            {"wind_power": wind_power, "energy_potential": energy_potential}
        )

    def assess_hydro_potential(
        self, flow_rate: xr.DataArray, head: xr.DataArray
    ) -> xr.Dataset:
        """
        Assess hydroelectric potential.

        Args:
            flow_rate: Water flow rate (m³/s)
            head: Hydraulic head (m)

        Returns:
            Hydro potential assessment
        """
        # Power = density * g * flow * head
        density = 1000  # kg/m³
        g = 9.81  # m/s²
        efficiency = 0.85  # Typical efficiency

        power = density * g * flow_rate * head * efficiency / 1e6  # MW

        # Annual energy
        energy = power * 8760  # MWh/year

        return xr.Dataset({"hydro_power": power, "energy_potential": energy})

    def assess_site_suitability(
        self,
        location: tuple[float, float],
        resource_type: RenewableType,
        resource_value: float,
        constraints: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """
        Assess site suitability for renewable development.

        Args:
            location: (lon, lat)
            resource_type: Type of renewable resource
            resource_value: Resource value (irradiance, wind speed, etc.)
            constraints: Optional constraint flags (protected area, slope, etc.)

        Returns:
            Site suitability assessment
        """
        constraints = constraints or {}

        # Define thresholds by resource type
        thresholds: dict[RenewableType, list[float]] = {
            RenewableType.SOLAR_PV: [3.5, 4.5, 5.5, 6.5],  # kWh/m²/day
            RenewableType.ONSHORE_WIND: [5.0, 6.0, 7.0, 8.0],  # m/s
            RenewableType.OFFSHORE_WIND: [6.0, 7.0, 8.0, 9.0],  # m/s
            RenewableType.HYDROPOWER: [10.0, 50.0, 100.0, 200.0],  # kW potential
            RenewableType.GEOTHERMAL: [100.0, 150.0, 200.0, 250.0],  # °C
        }

        # Score resource potential
        resource_thresholds = thresholds.get(resource_type, [3, 4, 5, 6])

        if resource_value >= resource_thresholds[3]:
            resource_class = SuitabilityClass.EXCELLENT
            resource_score = 1.0
        elif resource_value >= resource_thresholds[2]:
            resource_class = SuitabilityClass.GOOD
            resource_score = 0.8
        elif resource_value >= resource_thresholds[1]:
            resource_class = SuitabilityClass.MODERATE
            resource_score = 0.6
        elif resource_value >= resource_thresholds[0]:
            resource_class = SuitabilityClass.MARGINAL
            resource_score = 0.4
        else:
            resource_class = SuitabilityClass.UNSUITABLE
            resource_score = 0.2

        # Apply constraint penalties
        constraint_score = 1.0
        constraint_issues = []

        if constraints.get("protected_area"):
            constraint_score *= 0
            constraint_issues.append("Protected area - development prohibited")
        if constraints.get("steep_slope"):
            constraint_score *= 0.7
            constraint_issues.append("Steep slope - increased construction cost")
        if constraints.get("poor_access"):
            constraint_score *= 0.8
            constraint_issues.append("Poor road access")
        if constraints.get("grid_distance_km", 0) > 50:
            constraint_score *= 0.7
            constraint_issues.append("Far from grid connection")

        # Final suitability
        final_score = resource_score * constraint_score

        if final_score >= 0.8:
            final_class = SuitabilityClass.EXCELLENT
        elif final_score >= 0.6:
            final_class = SuitabilityClass.GOOD
        elif final_score >= 0.4:
            final_class = SuitabilityClass.MODERATE
        elif final_score >= 0.2:
            final_class = SuitabilityClass.MARGINAL
        else:
            final_class = SuitabilityClass.UNSUITABLE

        return {
            "location": location,
            "resource_type": resource_type.value,
            "resource_value": resource_value,
            "resource_class": resource_class.value,
            "resource_score": resource_score,
            "constraint_score": constraint_score,
            "constraint_issues": constraint_issues,
            "final_score": final_score,
            "suitability_class": final_class.value,
            "development_recommended": final_score >= 0.5,
        }

    def calculate_capacity_factor(
        self,
        resource_type: RenewableType,
        resource_data: xr.DataArray,
        rated_capacity_mw: float = 1.0,
    ) -> dict[str, Any]:
        """
        Calculate capacity factor from resource time series.

        Args:
            resource_type: Type of renewable
            resource_data: Time series of resource. For solar types this is
                expected in W/m2 (1000 W/m2 = full rated power); a series in
                kWh/m2/day (typical documented solar-atlas units, max below
                24) is auto-detected and converted to mean W/m2 by
                multiplying by 1000/24.
            rated_capacity_mw: Rated capacity

        Returns:
            Capacity factor analysis
        """
        if resource_type in [RenewableType.SOLAR_PV, RenewableType.SOLAR_THERMAL]:
            # Solar: capacity factor from irradiance. Expected input units are
            # W/m2 (1000 W/m2 = full power). Auto-detect kWh/m2/day inputs
            # (daily solar-atlas values max out below 24) and convert to mean
            # W/m2, so a 5.5 kWh/m2/day raster yields a meaningful capacity
            # factor instead of a silent ~0.5% one.
            if float(resource_data.max()) <= 24.0:
                resource_data = resource_data * (1000.0 / 24.0)
            power_fraction = resource_data / 1000
            power_fraction = xr.where(power_fraction > 1, 1, power_fraction)
            power_fraction = xr.where(power_fraction < 0, 0, power_fraction)

        elif resource_type in [RenewableType.ONSHORE_WIND, RenewableType.OFFSHORE_WIND]:
            # Wind: cubic relationship with cut-in/cut-out
            cut_in = 3.0  # m/s
            rated = 12.0  # m/s
            cut_out = 25.0  # m/s

            power_fraction = xr.zeros_like(resource_data)
            # Below cut-in: 0
            # Cut-in to rated: cubic ramp
            mask_ramp = (resource_data >= cut_in) & (resource_data < rated)
            power_fraction = xr.where(
                mask_ramp,
                ((resource_data - cut_in) / (rated - cut_in)) ** 3,
                power_fraction,
            )
            # Rated to cut-out: full power
            mask_full = (resource_data >= rated) & (resource_data < cut_out)
            power_fraction = xr.where(mask_full, 1.0, power_fraction)
            # Above cut-out: 0

        else:
            # Default: linear relationship, guarded against all-zero series
            resource_max = float(resource_data.max())
            power_fraction = (
                resource_data / resource_max
                if resource_max > 0
                else xr.zeros_like(resource_data)
            )

        # Calculate statistics
        mean_cf = float(power_fraction.mean())
        max_cf = float(power_fraction.max())
        min_cf = float(power_fraction.min())

        # Hours at different output levels
        hours_total = int(power_fraction.size)
        hours_zero = int((power_fraction < 0.01).sum())
        hours_full = int((power_fraction > 0.95).sum())

        # Annual generation
        annual_generation_mwh = mean_cf * rated_capacity_mw * 8760

        return {
            "resource_type": resource_type.value,
            "rated_capacity_mw": rated_capacity_mw,
            "capacity_factor": mean_cf,
            "capacity_factor_pct": mean_cf * 100,
            "min_cf": min_cf,
            "max_cf": max_cf,
            "hours_analyzed": hours_total,
            "hours_zero_output": hours_zero,
            "hours_full_output": hours_full,
            "annual_generation_mwh": annual_generation_mwh,
            "annual_generation_gwh": annual_generation_mwh / 1000,
        }

    def calculate_lcoe(
        self,
        resource_type: RenewableType,
        capacity_mw: float,
        capacity_factor: float,
        capital_cost_usd_kw: float | None = None,
        discount_rate: float = 0.07,
        lifetime_years: int = 25,
        opex_usd_kw_year: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate Levelized Cost of Energy (LCOE).

        Args:
            resource_type: Type of renewable
            capacity_mw: Installed capacity
            capacity_factor: Expected capacity factor
            capital_cost_usd_kw: Capital cost (default: use standard)
            discount_rate: Discount rate
            lifetime_years: Project lifetime
            opex_usd_kw_year: Annual O&M cost

        Returns:
            LCOE calculation results
        """
        # Get costs
        if capital_cost_usd_kw is None:
            if resource_type not in self.capital_costs:
                raise ValueError(
                    f"No capital cost registered for {resource_type.value!r}; "
                    "pass capital_cost_usd_kw explicitly instead of falling "
                    "back to a default."
                )
            capital_cost_usd_kw = self.capital_costs[resource_type]

        if opex_usd_kw_year is None:
            # Default O&M is ~2% of capital per year
            opex_usd_kw_year = capital_cost_usd_kw * 0.02

        capacity_kw = capacity_mw * 1000

        # Capital cost
        total_capital = capital_cost_usd_kw * capacity_kw

        # Annual generation (kWh)
        annual_generation_kwh = capacity_kw * capacity_factor * 8760

        # Annual O&M
        annual_opex = opex_usd_kw_year * capacity_kw

        # Calculate NPV of costs and generation
        npv_costs = total_capital
        npv_generation = 0.0

        for year in range(1, lifetime_years + 1):
            discount_factor = 1 / (1 + discount_rate) ** year
            npv_costs += annual_opex * discount_factor
            npv_generation += annual_generation_kwh * discount_factor

        # LCOE
        lcoe = npv_costs / npv_generation  # USD/kWh
        lcoe_mwh = lcoe * 1000  # USD/MWh

        return {
            "resource_type": resource_type.value,
            "capacity_mw": capacity_mw,
            "capacity_factor": capacity_factor,
            "capital_cost_usd": total_capital,
            "annual_opex_usd": annual_opex,
            "lifetime_years": lifetime_years,
            "discount_rate": discount_rate,
            "annual_generation_mwh": annual_generation_kwh / 1000,
            "lifetime_generation_gwh": annual_generation_kwh * lifetime_years / 1e6,
            "lcoe_usd_kwh": float(lcoe),
            "lcoe_usd_mwh": float(lcoe_mwh),
            "competitiveness": "Competitive"
            if lcoe_mwh < 50
            else "Moderately competitive"
            if lcoe_mwh < 80
            else "High cost",
        }

    def analyze_storage_requirements(
        self,
        generation_profile: xr.DataArray,
        demand_profile: xr.DataArray,
        renewable_penetration: float = 0.5,
        duration_hours: float = 4.0,
    ) -> dict[str, Any]:
        """
        Analyze storage requirements for renewable integration.

        Args:
            generation_profile: Hourly generation profile
            demand_profile: Hourly demand profile
            renewable_penetration: Target renewable share of annual demand
            duration_hours: Storage duration used to size energy capacity
                from peak deficit

        Returns:
            Storage requirement analysis
        """
        # Scale generation to meet penetration target
        total_demand = float(demand_profile.sum())
        target_generation = total_demand * renewable_penetration
        actual_generation = float(generation_profile.sum())

        scale_factor = (
            target_generation / actual_generation if actual_generation > 0 else 1
        )
        scaled_generation = generation_profile * scale_factor

        # Calculate hourly mismatch against FULL demand: generation is scaled
        # so its annual total meets the penetration target, and deficits are
        # measured against total demand (not demand scaled by penetration,
        # which would double-count the target and undersize storage).
        net_balance = scaled_generation - demand_profile

        surplus = xr.where(net_balance > 0, net_balance, 0)
        deficit = xr.where(net_balance < 0, -net_balance, 0)

        # Storage sizing
        total_surplus = float(surplus.sum())
        total_deficit = float(deficit.sum())
        max_surplus = float(surplus.max())
        max_deficit = float(deficit.max())

        # Documented contract: storage is sized to the worst single-hour
        # deficit power held for `duration_hours`; no round-trip efficiency,
        # state-of-charge dynamics or multi-interval dispatch are modeled.
        storage_power_mw = float(deficit.max())
        storage_energy_mwh = storage_power_mw * duration_hours

        # Calculate curtailment if no storage
        curtailment = total_surplus
        curtailment_pct = (
            curtailment / target_generation * 100 if target_generation > 0 else 0
        )

        return {
            "renewable_penetration": renewable_penetration,
            "total_demand_mwh": total_demand,
            "target_renewable_mwh": target_generation,
            "net_surplus_mwh": total_surplus,
            "net_deficit_mwh": total_deficit,
            "max_hourly_surplus_mw": max_surplus,
            "max_hourly_deficit_mw": max_deficit,
            "recommended_storage": {
                "power_capacity_mw": storage_power_mw,
                "energy_capacity_mwh": storage_energy_mwh,
                "duration_hours": duration_hours,
            },
            "curtailment_without_storage_mwh": curtailment,
            "curtailment_rate_pct": float(curtailment_pct),
        }

    def register_site(self, site: RenewableSite) -> str:
        """Register a renewable energy site."""
        self.site_registry[site.site_id] = site
        logger.info(
            f"Registered renewable site: {site.name} ({site.capacity_mw} MW {site.resource_type.value})"
        )
        return site.site_id

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get summary of registered renewable portfolio."""
        if not self.site_registry:
            return {"error": "No sites registered"}

        sites = list(self.site_registry.values())

        # Aggregate by type
        by_type: dict[str, dict[str, Any]] = {}
        for site in sites:
            rtype = site.resource_type.value
            if rtype not in by_type:
                by_type[rtype] = {"count": 0, "capacity_mw": 0.0, "generation_gwh": 0.0}
            by_type[rtype]["count"] += 1
            by_type[rtype]["capacity_mw"] += site.capacity_mw
            if site.annual_generation_gwh:
                by_type[rtype]["generation_gwh"] += site.annual_generation_gwh

        total_capacity = sum(s.capacity_mw for s in sites)
        total_generation = sum(s.annual_generation_gwh or 0 for s in sites)
        weighted_cf = (
            sum((s.capacity_factor * s.capacity_mw) for s in sites) / total_capacity
            if total_capacity > 0
            else 0
        )

        return {
            "site_count": len(sites),
            "total_capacity_mw": total_capacity,
            "total_generation_gwh": total_generation,
            "weighted_capacity_factor": weighted_cf,
            "by_resource_type": by_type,
        }
