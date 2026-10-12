"""Coral reef health assessment module.

Implements bleaching index calculation, degree heating weeks (DHW),
and biodiversity metrics specific to coral reef ecosystems.
"""

import logging
from typing import Any, cast

import numpy as np
import xarray as xr

from geo_infer_marine.utils.biodiversity import biodiversity_metrics

logger = logging.getLogger(__name__)


class CoralReefAssessor:
    """Assess coral reef health from environmental parameters.

    Implements NOAA Coral Reef Watch-style thermal stress metrics
    and reef biodiversity assessment tools.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize coral reef assessor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.bleaching_threshold_offset: float = self.config.get(
            "bleaching_threshold_offset", 1.0
        )

    def calculate_degree_heating_weeks(
        self,
        sst: xr.DataArray,
        climatological_max: xr.DataArray,
        window_weeks: int = 12,
    ) -> xr.DataArray:
        """Calculate Degree Heating Weeks (DHW).

        DHW accumulates thermal stress over a rolling window.
        Only positive anomalies above 1C over the maximum monthly
        mean (MMM) climatology are accumulated.

        Bleaching alert thresholds:
        DHW >= 4: Bleaching Watch
        DHW >= 8: Bleaching Alert Level 1
        DHW >= 12: Bleaching Alert Level 2

        Args:
            sst: Sea surface temperature time series (Celsius).
            climatological_max: Maximum Monthly Mean SST (Celsius).
            window_weeks: Accumulation window in weeks (default 12).

        Returns:
            Degree Heating Weeks (C-weeks).
        """
        hotspot = sst - climatological_max
        hotspot = xr.where(hotspot > 0, hotspot, 0)

        stress = xr.where(hotspot >= self.bleaching_threshold_offset, hotspot, 0)

        window_days = window_weeks * 7
        if "time" in stress.dims:
            available_days = int(stress.sizes.get("time", window_days))
            rolling_days = max(1, min(window_days, available_days))
            dhw = stress.rolling(time=rolling_days, min_periods=1).sum() / 7.0
        else:
            dhw = stress / 7.0

        dhw.name = "degree_heating_weeks"
        return cast(xr.DataArray, dhw)

    def classify_bleaching_alert(
        self,
        dhw: xr.DataArray,
    ) -> xr.DataArray:
        """Classify bleaching alert level from DHW.

        NOAA Coral Reef Watch classification:
        0 = No stress (DHW < 1)
        1 = Bleaching Watch (1 <= DHW < 4)
        2 = Bleaching Warning (4 <= DHW < 8)
        3 = Alert Level 1 (8 <= DHW < 12)
        4 = Alert Level 2 (DHW >= 12)

        Args:
            dhw: Degree Heating Weeks values.

        Returns:
            Integer alert level (0-4).
        """
        alert = xr.zeros_like(dhw, dtype=int)
        alert = xr.where(dhw >= 1, 1, alert)
        alert = xr.where(dhw >= 4, 2, alert)
        alert = xr.where(dhw >= 8, 3, alert)
        alert = xr.where(dhw >= 12, 4, alert)
        alert.name = "bleaching_alert_level"
        return cast(xr.DataArray, alert)

    def calculate_reef_biodiversity(
        self,
        species_counts: dict[str, int],
    ) -> dict[str, float]:
        """Calculate reef biodiversity metrics.

        Computes Shannon diversity, Simpson diversity, and Margalef richness
        index for coral reef species assemblages.

        Args:
            species_counts: Dictionary of species name to individual count.

        Returns:
            Dictionary with biodiversity indices.
        """
        if not species_counts:
            return {
                "species_richness": 0,
                "total_abundance": 0,
                "shannon_index": 0.0,
                "simpson_index": 0.0,
                "margalef_index": 0.0,
                "evenness": 0.0,
            }

        metrics = biodiversity_metrics(species_counts)
        richness = metrics["species_richness"]
        total = metrics["total_abundance"]
        margalef = (richness - 1) / np.log(total) if total > 1 else 0.0

        return {
            "species_richness": richness,
            "total_abundance": total,
            "shannon_index": metrics["shannon"],
            "simpson_index": metrics["simpson"],
            "margalef_index": float(margalef),
            "evenness": metrics["evenness"],
        }

    def assess_reef_health_composite(
        self,
        coral_cover_pct: float,
        macroalgae_cover_pct: float,
        fish_biomass_kg_ha: float,
        bleaching_alert_level: int = 0,
    ) -> dict[str, Any]:
        """Calculate composite reef health score.

        Combines live coral cover, macroalgae ratio, fish biomass,
        and thermal stress into a single 0-100 health score.

        Thresholds based on reef monitoring benchmarks:
        - Coral cover: >40% = healthy, 10-40% = impaired, <10% = degraded
        - Macroalgae: <10% = healthy, >30% = degraded
        - Fish biomass: >1000 kg/ha = healthy, <300 = degraded

        Args:
            coral_cover_pct: Live coral cover percentage.
            macroalgae_cover_pct: Macroalgae cover percentage.
            fish_biomass_kg_ha: Fish biomass (kg per hectare).
            bleaching_alert_level: Current bleaching alert (0-4).

        Returns:
            Dictionary with component and composite scores.
        """
        coral_score = min(100.0, (coral_cover_pct / 40.0) * 100.0)
        coral_score = max(0.0, coral_score)

        algae_score = max(0.0, 100.0 - (macroalgae_cover_pct / 30.0) * 100.0)
        algae_score = min(100.0, algae_score)

        fish_score = min(100.0, (fish_biomass_kg_ha / 1000.0) * 100.0)
        fish_score = max(0.0, fish_score)

        thermal_penalty = bleaching_alert_level * 15.0

        composite = (
            0.35 * coral_score
            + 0.20 * algae_score
            + 0.25 * fish_score
            + 0.20 * (100.0 - thermal_penalty)
        )
        composite = max(0.0, min(100.0, composite))

        if composite >= 75:
            classification = "healthy"
        elif composite >= 50:
            classification = "impaired"
        elif composite >= 25:
            classification = "degraded"
        else:
            classification = "critical"

        return {
            "composite_score": float(composite),
            "classification": classification,
            "coral_score": float(coral_score),
            "algae_score": float(algae_score),
            "fish_score": float(fish_score),
            "thermal_penalty": float(thermal_penalty),
        }
