"""
Marine ecosystem modeling module.
"""

import logging
from typing import Any
import numpy as np
import xarray as xr
from dataclasses import dataclass
from enum import Enum

from geo_infer_marine.utils.biodiversity import biodiversity_metrics

logger = logging.getLogger(__name__)


class MarineHabitatType(Enum):
    """Types of marine habitats."""

    CORAL_REEF = "coral_reef"
    SEAGRASS = "seagrass"
    MANGROVE = "mangrove"
    KELP_FOREST = "kelp_forest"
    OPEN_OCEAN = "open_ocean"
    DEEP_SEA = "deep_sea"
    ESTUARY = "estuary"


@dataclass
class SpeciesData:
    """Species information for ecosystem modeling."""

    species_id: str
    common_name: str
    scientific_name: str
    trophic_level: float
    habitat_preference: list[MarineHabitatType]
    temperature_range: tuple[float, float]
    depth_range: tuple[float, float]
    conservation_status: str = "LC"  # IUCN status


class MarineEcosystemModeler:
    """
    Model marine ecosystems including coral reefs, fisheries, biodiversity.

    This class provides comprehensive marine ecosystem analysis with:
    - Coral reef health assessment
    - Fisheries stock modeling
    - Biodiversity indices calculation
    - Marine protected area effectiveness
    - Species distribution modeling
    - Climate change impact assessment
    """

    def __init__(self, config: dict | None = None):
        """Initialize marine ecosystem modeler."""
        self.config = config or {}
        self.species_registry: dict[str, SpeciesData] = {}
        self.mpa_registry: dict[str, dict] = {}

    def assess_coral_reef_health(
        self, temperature: xr.DataArray, ph: xr.DataArray | None = None
    ) -> xr.Dataset:
        """
        Assess coral reef health based on temperature and pH.

        Args:
            temperature: Sea surface temperature
            ph: Ocean pH (for acidification assessment)

        Returns:
            Coral reef health assessment
        """
        # Thermal stress (bleaching risk). One-sided: only temperatures
        # above the optimum induce bleaching, matching the hotspot
        # semantics of CoralReefAssessor; cold water is not thermal
        # stress. Capped at 1.0 so the risk stays a normalized
        # probability for very warm SSTs.
        optimal_temp = 26.0  # Optimal coral temperature
        thermal_stress = np.maximum(temperature - optimal_temp, 0)
        bleaching_risk = (thermal_stress / 5.0).clip(min=0.0, max=1.0)

        results: dict[str, Any] = {
            "thermal_stress": thermal_stress,
            "bleaching_risk": bleaching_risk,
        }

        if ph is not None:
            # Ocean acidification stress
            optimal_ph = 8.1
            acidification_stress = optimal_ph - ph
            results["acidification_stress"] = acidification_stress
            results["combined_stress"] = bleaching_risk + acidification_stress

        return xr.Dataset(results)

    def model_fisheries_stock(
        self,
        habitat_quality: xr.DataArray,
        fishing_pressure: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Model fisheries stock based on habitat and fishing pressure.

        Args:
            habitat_quality: Habitat quality index
            fishing_pressure: Optional fishing pressure data

        Returns:
            Fisheries stock assessment
        """
        # Simple stock model
        stock = habitat_quality * 100  # Scale to stock units

        if fishing_pressure is not None:
            # Reduce stock by fishing pressure
            stock = stock * (1 - fishing_pressure / 100.0)
            stock = xr.where(stock < 0, 0, stock)

        return xr.Dataset(
            {"stock_abundance": stock, "habitat_quality": habitat_quality}
        )

    def calculate_biodiversity_indices(
        self, species_counts: dict[str, int], area_km2: float = 1.0
    ) -> dict[str, float]:
        """
        Calculate biodiversity indices from species abundance data.

        Args:
            species_counts: Dictionary of species -> count
            area_km2: Area in square kilometers for density calculation

        Returns:
            Dictionary of biodiversity indices
        """
        metrics = biodiversity_metrics(species_counts)
        density = metrics["species_richness"] / area_km2

        return {
            "species_richness": metrics["species_richness"],
            "shannon_diversity": metrics["shannon"],
            "simpson_diversity": metrics["simpson"],
            "evenness": metrics["evenness"],
            "species_density": float(density),
            "total_abundance": metrics["total_abundance"],
        }

    def register_species(self, species: SpeciesData) -> None:
        """
        Register a species in the ecosystem model.

        Args:
            species: SpeciesData object with species information
        """
        self.species_registry[species.species_id] = species
        logger.info(f"Registered species: {species.common_name}")

    def model_species_distribution(
        self,
        species_id: str,
        temperature: xr.DataArray,
        depth: xr.DataArray,
        habitat_map: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Model species distribution based on environmental conditions.

        Args:
            species_id: ID of the species to model
            temperature: Sea temperature field
            depth: Bathymetry/depth field
            habitat_map: Optional habitat type classification

        Returns:
            Species distribution probability and suitability
        """
        if species_id not in self.species_registry:
            raise ValueError(f"Species {species_id} not registered")

        species = self.species_registry[species_id]

        # Temperature suitability
        temp_min, temp_max = species.temperature_range
        temp_optimal = (temp_min + temp_max) / 2
        temp_range = (temp_max - temp_min) / 2
        temp_suitability = np.exp(-(((temperature - temp_optimal) / temp_range) ** 2))

        # Depth suitability
        depth_min, depth_max = species.depth_range
        depth_optimal = (depth_min + depth_max) / 2
        depth_range = (depth_max - depth_min) / 2
        depth_suitability = np.exp(
            -(((depth - depth_optimal) / (depth_range + 1)) ** 2)
        )

        # Combined suitability
        suitability = temp_suitability * depth_suitability

        # Probability of occurrence (logistic function)
        probability = 1 / (1 + np.exp(-5 * (suitability - 0.5)))

        return xr.Dataset(
            {
                "suitability": suitability,
                "occurrence_probability": probability,
                "temperature_suitability": temp_suitability,
                "depth_suitability": depth_suitability,
            }
        )

    def create_marine_protected_area(
        self,
        mpa_id: str,
        name: str,
        boundary: list[tuple[float, float]],  # List of (lon, lat) points
        protection_level: str = "full",
        target_species: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a marine protected area definition.

        Args:
            mpa_id: Unique identifier for the MPA
            name: Name of the MPA
            boundary: List of boundary coordinates
            protection_level: Protection level (full, partial, seasonal)
            target_species: List of species IDs targeted for protection

        Returns:
            MPA definition dictionary
        """
        mpa = {
            "id": mpa_id,
            "name": name,
            "boundary": boundary,
            "protection_level": protection_level,
            "target_species": target_species or [],
            "area_km2": self._calculate_polygon_area(boundary),
            "status": "active",
        }

        self.mpa_registry[mpa_id] = mpa
        logger.info(f"Created MPA: {name} ({mpa['area_km2']:.2f} km²)")

        return mpa

    def _calculate_polygon_area(self, coords: list[tuple[float, float]]) -> float:
        """Calculate approximate area of a polygon in km²."""
        if len(coords) < 3:
            return 0.0

        # Shoelace formula with latitude correction
        n = len(coords)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[j]

            area += lon1 * lat2
            area -= lon2 * lat1

        area = abs(area) / 2.0

        # Convert to km² (approximate at mid-latitude)
        avg_lat = float(np.mean([c[1] for c in coords]))
        lat_correction = float(np.cos(np.radians(avg_lat)))
        km_per_degree = 111.0  # km

        area_km2 = area * (km_per_degree**2) * lat_correction

        return float(area_km2)

    def assess_mpa_effectiveness(
        self,
        mpa_id: str,
        species_counts_inside: dict[str, int],
        species_counts_outside: dict[str, int],
        time_since_establishment_years: float = 1.0,
    ) -> dict[str, Any]:
        """
        Assess the effectiveness of a marine protected area.

        Args:
            mpa_id: ID of the MPA to assess
            species_counts_inside: Species counts inside MPA
            species_counts_outside: Species counts outside MPA
            time_since_establishment_years: Years since MPA was established

        Returns:
            MPA effectiveness assessment
        """
        if mpa_id not in self.mpa_registry:
            raise ValueError(f"MPA {mpa_id} not found")

        mpa = self.mpa_registry[mpa_id]

        # Calculate biodiversity indices
        bio_inside = self.calculate_biodiversity_indices(species_counts_inside)
        bio_outside = self.calculate_biodiversity_indices(species_counts_outside)

        # Calculate effectiveness metrics
        abundance_ratio = (
            bio_inside["total_abundance"] / bio_outside["total_abundance"]
            if bio_outside["total_abundance"] > 0
            else 0
        )

        richness_ratio = (
            bio_inside["species_richness"] / bio_outside["species_richness"]
            if bio_outside["species_richness"] > 0
            else 0
        )

        # Spillover effect as a proportional biomass transfer
        spillover_index = max(0, (abundance_ratio - 1) * 0.1)

        # Overall effectiveness score (0-100)
        effectiveness_score = min(
            100,
            (
                (abundance_ratio - 1) * 20
                + (richness_ratio - 1) * 30
                + time_since_establishment_years * 2
                + 50  # Base score
            ),
        )

        return {
            "mpa_id": mpa_id,
            "mpa_name": mpa["name"],
            "area_km2": mpa["area_km2"],
            "biodiversity_inside": bio_inside,
            "biodiversity_outside": bio_outside,
            "abundance_ratio": float(abundance_ratio),
            "richness_ratio": float(richness_ratio),
            "spillover_index": float(spillover_index),
            "effectiveness_score": float(effectiveness_score),
            "time_since_establishment": time_since_establishment_years,
            "recommendation": self._get_mpa_recommendation(effectiveness_score),
        }

    def _get_mpa_recommendation(self, score: float) -> str:
        """Get recommendation based on effectiveness score."""
        if score >= 80:
            return "Highly effective - maintain current protection"
        elif score >= 60:
            return "Moderately effective - consider strengthening enforcement"
        elif score >= 40:
            return "Partially effective - review protection measures"
        else:
            return "Low effectiveness - significant management changes needed"

    def assess_climate_change_impact(
        self,
        temperature_change: float,
        sea_level_rise_cm: float,
        ph_change: float,
        time_horizon_years: int = 50,
    ) -> dict[str, Any]:
        """
        Assess climate change impacts on marine ecosystems.

        Args:
            temperature_change: Expected temperature change in °C
            sea_level_rise_cm: Expected sea level rise in cm
            ph_change: Expected ocean pH change (negative = acidification)
            time_horizon_years: Time horizon for assessment

        Returns:
            Climate impact assessment
        """
        # Coral reef impacts
        coral_bleaching_threshold = 1.0  # °C above normal
        coral_bleaching_risk = min(1.0, temperature_change / coral_bleaching_threshold)

        coral_acidification_threshold = -0.3  # pH units
        coral_acidification_risk = min(
            1.0, abs(ph_change) / abs(coral_acidification_threshold)
        )

        coral_survival_probability = max(
            0, 1 - (coral_bleaching_risk + coral_acidification_risk) / 2
        )

        # Habitat loss from sea level rise
        coastal_habitat_loss_pct = min(100, sea_level_rise_cm * 0.5)

        # Species distribution shifts
        poleward_shift_km = temperature_change * 50  # Approximate shift

        # Fisheries productivity
        # Generally decreases with warming in tropical/subtropical regions
        fisheries_change_pct = (
            -temperature_change * 5
            if temperature_change > 0
            else temperature_change * 2
        )

        # Overall ecosystem vulnerability (0-1)
        vulnerability = (
            coral_bleaching_risk * 0.25
            + coral_acidification_risk * 0.25
            + (coastal_habitat_loss_pct / 100) * 0.25
            + min(1, abs(fisheries_change_pct) / 20) * 0.25
        )

        return {
            "time_horizon_years": time_horizon_years,
            "climate_scenarios": {
                "temperature_change_c": temperature_change,
                "sea_level_rise_cm": sea_level_rise_cm,
                "ph_change": ph_change,
            },
            "coral_reef_impacts": {
                "bleaching_risk": float(coral_bleaching_risk),
                "acidification_risk": float(coral_acidification_risk),
                "survival_probability": float(coral_survival_probability),
            },
            "habitat_impacts": {
                "coastal_habitat_loss_pct": float(coastal_habitat_loss_pct),
                "mangrove_vulnerability": "high"
                if sea_level_rise_cm > 30
                else "moderate",
                "seagrass_vulnerability": "high"
                if temperature_change > 2
                else "moderate",
            },
            "species_impacts": {
                "poleward_shift_km": float(poleward_shift_km),
                "local_extinctions_risk": "high"
                if vulnerability > 0.7
                else "moderate"
                if vulnerability > 0.4
                else "low",
            },
            "fisheries_impacts": {
                "productivity_change_pct": float(fisheries_change_pct),
                "catch_potential_trend": "declining"
                if fisheries_change_pct < -5
                else "stable",
            },
            "overall_vulnerability": float(vulnerability),
            "adaptation_priority": "critical"
            if vulnerability > 0.7
            else "high"
            if vulnerability > 0.5
            else "moderate",
        }

    def estimate_blue_carbon(
        self, habitat_area_km2: dict[str, float], condition: str = "healthy"
    ) -> dict[str, Any]:
        """
        Estimate blue carbon storage in marine habitats.

        Args:
            habitat_area_km2: Dictionary of habitat type -> area in km²
            condition: Overall habitat condition (healthy, degraded, severely_degraded)

        Returns:
            Blue carbon storage estimates
        """
        # Carbon storage rates (tonnes CO2e per km² per year)
        storage_rates = {
            "mangrove": 1500,
            "seagrass": 800,
            "salt_marsh": 600,
            "kelp_forest": 400,
            "coral_reef": 200,
        }

        # Condition multipliers
        condition_multipliers = {
            "healthy": 1.0,
            "degraded": 0.6,
            "severely_degraded": 0.3,
        }

        multiplier = condition_multipliers.get(condition, 0.6)

        # Calculate storage by habitat
        storage_by_habitat = {}
        total_storage = 0.0
        total_area = 0.0

        for habitat, area in habitat_area_km2.items():
            rate = float(storage_rates.get(habitat, 100))
            annual_storage = float(area * rate * multiplier)
            storage_by_habitat[habitat] = {
                "area_km2": float(area),
                "annual_storage_tonnes": annual_storage,
                "storage_rate_t_per_km2": rate,
            }
            total_storage += annual_storage
            total_area += float(area)

        # Economic value (approximate $25 per tonne CO2e)
        carbon_value_usd = total_storage * 25

        return {
            "total_area_km2": total_area,
            "total_annual_storage_tonnes": total_storage,
            "storage_by_habitat": storage_by_habitat,
            "habitat_condition": condition,
            "condition_multiplier": multiplier,
            "carbon_value_usd_annual": carbon_value_usd,
            "carbon_value_usd_30yr": carbon_value_usd * 30,
        }
