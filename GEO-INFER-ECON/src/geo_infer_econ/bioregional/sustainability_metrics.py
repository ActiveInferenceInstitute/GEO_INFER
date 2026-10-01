"""
Sustainability Metrics Module

Provides sustainability indicators, resilience metrics, regenerative metrics,
wellbeing indicators, and planetary boundaries assessment.
"""

from typing import Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# Planetary boundary definitions (Rockström et al. 2009, updated Steffen et al. 2015)
PLANETARY_BOUNDARIES = {
    "climate_change": {
        "indicator": "CO2 concentration (ppm)",
        "boundary": 350,
        "current_global": 421,
        "unit": "ppm",
    },
    "biodiversity_loss": {
        "indicator": "Extinction rate (E/MSY)",
        "boundary": 10,
        "current_global": 100,
        "unit": "E/MSY",
    },
    "nitrogen_cycle": {
        "indicator": "N fixation (Tg N/yr)",
        "boundary": 62,
        "current_global": 150,
        "unit": "Tg N/yr",
    },
    "phosphorus_cycle": {
        "indicator": "P flow to oceans (Tg P/yr)",
        "boundary": 11,
        "current_global": 22,
        "unit": "Tg P/yr",
    },
    "ozone_depletion": {
        "indicator": "Ozone (DU)",
        "boundary": 276,
        "current_global": 283,
        "unit": "DU",
    },
    "ocean_acidification": {
        "indicator": "Aragonite saturation",
        "boundary": 2.75,
        "current_global": 2.90,
        "unit": "Ω",
    },
    "freshwater_use": {
        "indicator": "Freshwater consumption (km³/yr)",
        "boundary": 4000,
        "current_global": 2600,
        "unit": "km³/yr",
    },
    "land_use_change": {
        "indicator": "Cropland (% ice-free)",
        "boundary": 15,
        "current_global": 12,
        "unit": "%",
    },
    "aerosol_loading": {
        "indicator": "AOD",
        "boundary": 0.25,
        "current_global": 0.30,
        "unit": "AOD",
    },
}


class SustainabilityIndicators:
    """
    Multi-dimensional sustainability indicators calculator.

    Computes composite indices across environmental, social, and economic
    dimensions, aligned with SDG (Sustainable Development Goals) frameworks.
    """

    SDG_DIMENSIONS = {
        "environmental": [
            "clean_water",
            "clean_energy",
            "climate_action",
            "life_on_land",
            "life_below_water",
        ],
        "social": [
            "no_poverty",
            "zero_hunger",
            "health",
            "education",
            "gender_equality",
            "reduced_inequalities",
        ],
        "economic": [
            "decent_work",
            "innovation",
            "responsible_production",
            "partnerships",
        ],
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize sustainability indicators."""
        self.config = config or {}
        self.weights = self.config.get(
            "dimension_weights",
            {"environmental": 0.4, "social": 0.35, "economic": 0.25},
        )
        logger.info("SustainabilityIndicators initialized")

    def calculate_indicators(self, data: dict[str, Any]) -> pd.DataFrame:
        """Calculate sustainability indicators.

        Args:
            data: Dict with:
                - region_id: region identifier
                - indicators: dict mapping indicator_name -> value (0-100 normalized)
                - population: population count
                - area_km2: area

        Returns:
            DataFrame with columns: dimension, indicator, value, score, target, gap.
        """
        region_id = data.get("region_id", "unknown")
        indicators = data.get("indicators", {})
        logger.info("Calculating sustainability indicators for region: %s", region_id)

        rows = []
        dimension_scores = {}

        for dimension, sdg_indicators in self.SDG_DIMENSIONS.items():
            dim_values = []
            for ind in sdg_indicators:
                value = float(indicators.get(ind, 50.0))
                target = 80.0  # SDG target threshold
                gap = target - value
                score = min(value / target, 1.0) * 100

                rows.append(
                    {
                        "region_id": region_id,
                        "dimension": dimension,
                        "indicator": ind,
                        "value": round(value, 2),
                        "score": round(score, 2),
                        "target": target,
                        "gap": round(gap, 2),
                        "on_track": value >= target * 0.75,
                    }
                )
                dim_values.append(score)

            dimension_scores[dimension] = np.mean(dim_values) if dim_values else 0

        # Add composite index row
        composite = sum(dimension_scores.get(d, 0) * w for d, w in self.weights.items())
        rows.append(
            {
                "region_id": region_id,
                "dimension": "composite",
                "indicator": "sustainability_index",
                "value": round(composite, 2),
                "score": round(composite, 2),
                "target": 80.0,
                "gap": round(80.0 - composite, 2),
                "on_track": composite >= 60.0,
            }
        )

        df = pd.DataFrame(rows)
        logger.info("Sustainability index for %s: %.2f", region_id, composite)
        return df


class ResilienceMetrics:
    """Ecological and social resilience metrics."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("ResilienceMetrics initialized")

    def calculate_resilience(self, resilience_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate resilience metrics for a system.

        Args:
            resilience_data: Dict with:
                - diversity_index: Shannon diversity (0-5)
                - connectivity: network connectivity score (0-1)
                - redundancy: functional redundancy (0-1)
                - adaptive_capacity: adaptive capacity score (0-1)
                - recovery_rate: observed recovery rate from disturbances (0-1)
                - disturbance_history: list of {year, severity (0-1), recovery_time_months}

        Returns:
            Dict with composite resilience score and sub-components.
        """
        diversity = float(resilience_data.get("diversity_index", 2.5))
        connectivity = float(resilience_data.get("connectivity", 0.5))
        redundancy = float(resilience_data.get("redundancy", 0.5))
        adaptive = float(resilience_data.get("adaptive_capacity", 0.5))
        recovery = float(resilience_data.get("recovery_rate", 0.5))
        history = resilience_data.get("disturbance_history", [])

        logger.info("Calculating resilience metrics")

        # Normalize diversity to 0-1 (max Shannon ~5 for very diverse systems)
        diversity_norm = min(diversity / 5.0, 1.0)

        # Historical resilience from disturbance history
        historical_resilience = 1.0
        if history:
            recovery_efficiencies = []
            for event in history:
                severity = float(event.get("severity", 0.5))
                recovery_time = float(event.get("recovery_time_months", 12))
                efficiency = (1 - severity) + severity * np.exp(-recovery_time / 24)
                recovery_efficiencies.append(efficiency)
            historical_resilience = float(np.mean(recovery_efficiencies))

        # Composite resilience (weighted geometric mean)
        components: dict[str, Any] = {
            "diversity": diversity_norm,
            "connectivity": connectivity,
            "redundancy": redundancy,
            "adaptive_capacity": adaptive,
            "recovery_rate": recovery,
            "historical_resilience": historical_resilience,
        }

        weights = [0.20, 0.15, 0.15, 0.20, 0.15, 0.15]
        values = list(components.values())
        composite = float(
            np.exp(sum(w * np.log(max(v, 1e-6)) for w, v in zip(weights, values)))
        )

        result: dict[str, Any] = {**{k: round(v, 4) for k, v in components.items()}}
        result["composite_resilience"] = round(composite, 4)
        result["resilience_grade"] = (
            "high" if composite > 0.7 else "medium" if composite > 0.4 else "low"
        )

        logger.info(
            "Composite resilience: %.4f (%s)", composite, result["resilience_grade"]
        )
        return result


class RegenerativeMetrics:
    """Metrics for regenerative capacity of ecosystems."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("RegenerativeMetrics initialized")

    def calculate_regenerative(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate regenerative capacity metrics.

        Args:
            data: Dict with:
                - soil_organic_carbon: SOC in tons/ha
                - vegetation_cover: fraction (0-1)
                - water_retention: water retention capacity (mm)
                - biomass_production: annual NPP (tons C/ha/yr)
                - species_richness: count of species
                - baseline_species_richness: historical baseline

        Returns:
            Dict with regenerative capacity scores.
        """
        soc = float(data.get("soil_organic_carbon", 50))
        veg_cover = float(data.get("vegetation_cover", 0.5))
        water_ret = float(data.get("water_retention", 100))
        npp = float(data.get("biomass_production", 5))
        species = int(data.get("species_richness", 100))
        baseline_species = int(data.get("baseline_species_richness", 150))

        logger.info("Calculating regenerative metrics")

        # Normalize each metric to 0-1 scale
        soc_score = min(soc / 100.0, 1.0)  # 100 t/ha as reference
        veg_score = veg_cover
        water_score = min(water_ret / 300.0, 1.0)  # 300mm as reference
        npp_score = min(npp / 10.0, 1.0)  # 10 tC/ha/yr as reference
        biodiv_score = min(species / max(baseline_species, 1), 1.0)

        # Composite regenerative index
        components: dict[str, Any] = {
            "soil_health": round(soc_score, 4),
            "vegetation_integrity": round(veg_score, 4),
            "hydrological_function": round(water_score, 4),
            "productivity": round(npp_score, 4),
            "biodiversity_intactness": round(biodiv_score, 4),
        }

        composite = float(np.mean(list(components.values())))
        components["regenerative_index"] = round(composite, 4)
        components["is_regenerative"] = composite > 0.6
        components["trajectory"] = (
            "regenerating"
            if composite > 0.6
            else "degrading"
            if composite < 0.3
            else "stable"
        )

        logger.info(
            "Regenerative index: %.4f (%s)", composite, components["trajectory"]
        )
        return components


class WellbeingIndicators:
    """Wellbeing indicators beyond GDP (GPI, HPI, genuine savings)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("WellbeingIndicators initialized")

    def calculate_wellbeing(self, wellbeing_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate wellbeing indicators.

        Args:
            wellbeing_data: Dict with:
                - gdp_per_capita: GDP per capita (USD)
                - gini_coefficient: inequality (0-1)
                - life_expectancy: years
                - education_index: 0-1
                - ecological_footprint: gha per capita
                - biocapacity: gha per capita
                - social_capital: community trust/cohesion (0-1)
                - life_satisfaction: subjective wellbeing (0-10)

        Returns:
            Dict with GPI, HPI, ecological balance, composite wellbeing.
        """
        gdp_pc = float(wellbeing_data.get("gdp_per_capita", 10000))
        gini = float(wellbeing_data.get("gini_coefficient", 0.35))
        life_exp = float(wellbeing_data.get("life_expectancy", 72))
        edu = float(wellbeing_data.get("education_index", 0.7))
        footprint = float(wellbeing_data.get("ecological_footprint", 2.7))
        biocap = float(wellbeing_data.get("biocapacity", 1.6))
        social = float(wellbeing_data.get("social_capital", 0.5))
        satisfaction = float(wellbeing_data.get("life_satisfaction", 6.0))

        logger.info("Calculating wellbeing indicators")

        # Genuine Progress Indicator (simplified)
        inequality_adjustment = 1 - gini
        environmental_cost = max(0, footprint - biocap) * 3000  # USD per overshoot gha
        gpi_per_capita = (gdp_pc * inequality_adjustment) - environmental_cost

        # Happy Planet Index (NEF methodology)
        # HPI = (Experienced wellbeing × Life expectancy) / Ecological Footprint
        hpi = (satisfaction * life_exp) / max(footprint, 0.1)

        # Ecological balance
        ecological_balance = biocap - footprint  # positive = ecological surplus

        # Composite wellbeing (normalize each to 0-1)
        income_score = min(np.log(max(gdp_pc, 1)) / np.log(80000), 1.0)
        health_score = min(life_exp / 85.0, 1.0)
        equality_score = inequality_adjustment
        eco_score = min(biocap / max(footprint, 0.1), 1.0)
        satisfaction_norm = satisfaction / 10.0

        composite = float(
            np.mean(
                [
                    income_score,
                    health_score,
                    equality_score,
                    edu,
                    eco_score,
                    social,
                    satisfaction_norm,
                ]
            )
        )

        result: dict[str, Any] = {
            "gdp_per_capita": gdp_pc,
            "gpi_per_capita": round(gpi_per_capita, 2),
            "happy_planet_index": round(hpi, 2),
            "ecological_balance_gha": round(ecological_balance, 2),
            "ecological_status": "surplus" if ecological_balance > 0 else "deficit",
            "inequality_adjusted_income": round(gdp_pc * inequality_adjustment, 2),
            "composite_wellbeing": round(composite, 4),
            "sub_scores": {
                "income": round(income_score, 4),
                "health": round(health_score, 4),
                "equality": round(equality_score, 4),
                "education": round(edu, 4),
                "ecological": round(eco_score, 4),
                "social_capital": round(social, 4),
                "life_satisfaction": round(satisfaction_norm, 4),
            },
        }

        logger.info(
            "Composite wellbeing: %.4f (GPI: %.2f, HPI: %.2f)",
            composite,
            gpi_per_capita,
            hpi,
        )
        return result


class PlanetaryBoundaries:
    """Planetary boundaries assessment (Rockström/Steffen framework)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.boundaries = self.config.get("boundaries", PLANETARY_BOUNDARIES)
        logger.info(
            "PlanetaryBoundaries initialized with %d boundaries", len(self.boundaries)
        )

    def assess_boundaries(self, boundary_data: dict[str, Any]) -> dict[str, Any]:
        """Assess regional or global status relative to planetary boundaries.

        Args:
            boundary_data: Dict with:
                - measurements: dict mapping boundary_name -> local/regional value
                - scale: 'regional' or 'global'
                - region_id: optional region identifier

        Returns:
            Dict with boundary status, transgression rates, safe operating space.
        """
        measurements = boundary_data.get("measurements", {})
        scale = boundary_data.get("scale", "regional")
        region_id = boundary_data.get("region_id", "global")

        logger.info(
            "Assessing %d planetary boundaries for %s (%s scale)",
            len(measurements),
            region_id,
            scale,
        )

        assessments = {}
        transgressed = []
        safe = []

        for boundary_name, boundary_def in self.boundaries.items():
            limit = float(boundary_def["boundary"])
            current = float(
                measurements.get(
                    boundary_name, boundary_def.get("current_global", limit)
                )
            )

            # For some boundaries, exceeding is bad; for others (ozone, aragonite), going below is bad
            invert = boundary_name in ("ozone_depletion", "ocean_acidification")

            if invert:
                transgression_ratio = limit / max(current, 1e-6)
                is_transgressed = current < limit
            else:
                transgression_ratio = current / max(limit, 1e-6)
                is_transgressed = current > limit

            status = "transgressed" if is_transgressed else "safe"
            if is_transgressed:
                transgressed.append(boundary_name)
            else:
                safe.append(boundary_name)

            assessments[boundary_name] = {
                "indicator": boundary_def["indicator"],
                "boundary_value": limit,
                "current_value": current,
                "unit": boundary_def.get("unit", ""),
                "transgression_ratio": round(transgression_ratio, 3),
                "status": status,
                "margin": round(abs(limit - current), 3),
            }

        n_total = len(self.boundaries)
        result = {
            "region_id": region_id,
            "scale": scale,
            "boundary_assessments": assessments,
            "transgressed_boundaries": transgressed,
            "safe_boundaries": safe,
            "transgression_count": len(transgressed),
            "safe_operating_fraction": round(len(safe) / max(n_total, 1), 3),
            "overall_status": (
                "within_safe_space" if not transgressed else "boundaries_transgressed"
            ),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Planetary boundaries: %d/%d transgressed for %s",
            len(transgressed),
            n_total,
            region_id,
        )
        return result
