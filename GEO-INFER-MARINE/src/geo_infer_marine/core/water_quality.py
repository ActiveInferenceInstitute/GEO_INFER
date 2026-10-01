"""Marine water quality indices and assessment.

Calculates dissolved oxygen saturation, pH-based ocean acidification
indices, turbidity scoring, and composite marine water quality index.
"""

import logging
from typing import cast

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class MarineWaterQuality:
    """Assess marine water quality using physical and chemical parameters.

    Implements standard oceanographic water quality calculations including
    dissolved oxygen saturation, trophic state, and composite quality indices.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize marine water quality assessor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def calculate_do_saturation(
        self,
        temperature_c: xr.DataArray,
        salinity_psu: xr.DataArray,
    ) -> xr.DataArray:
        """Calculate dissolved oxygen saturation concentration.

        Uses the Weiss (1970) solubility formulation:

        ln(DO_sat) = A1 + A2*(100/T) + A3*ln(T/100) + A4*(T/100)
                     + S * [B1 + B2*(T/100) + B3*(T/100)^2]

        Args:
            temperature_c: Water temperature (Celsius).
            salinity_psu: Salinity (PSU).

        Returns:
            Dissolved oxygen saturation (mg/L).
        """
        t_kelvin = temperature_c + 273.15
        t_scaled = t_kelvin / 100.0

        a1 = -173.4292
        a2 = 249.6339
        a3 = 143.3483
        a4 = -21.8492
        b1 = -0.033096
        b2 = 0.014259
        b3 = -0.001700

        ln_do = (
            a1
            + a2 * (100.0 / t_kelvin)
            + a3 * np.log(t_scaled)
            + a4 * t_scaled
            + salinity_psu * (b1 + b2 * t_scaled + b3 * t_scaled**2)
        )

        do_sat = np.exp(ln_do)
        do_sat = xr.where(do_sat < 0, 0, do_sat)
        do_sat.name = "do_saturation_mg_l"
        return cast(xr.DataArray, do_sat)

    def calculate_do_percent_saturation(
        self,
        measured_do: xr.DataArray,
        temperature_c: xr.DataArray,
        salinity_psu: xr.DataArray,
    ) -> xr.DataArray:
        """Calculate dissolved oxygen as percent of saturation.

        Args:
            measured_do: Measured dissolved oxygen (mg/L).
            temperature_c: Water temperature (Celsius).
            salinity_psu: Salinity (PSU).

        Returns:
            Percent saturation (0-200+).
        """
        do_sat = self.calculate_do_saturation(temperature_c, salinity_psu)
        pct = (measured_do / (do_sat + 1e-10)) * 100.0
        pct.name = "do_percent_saturation"
        return pct

    def calculate_ocean_acidification_index(
        self,
        ph: xr.DataArray,
        reference_ph: float = 8.1,
    ) -> xr.DataArray:
        """Calculate ocean acidification index.

        Aragonite saturation is pH-dependent. This index measures
        deviation from pre-industrial ocean pH as a proxy for
        acidification stress on marine organisms.

        Index = (reference_pH - measured_pH) / 0.3
        where 0.3 is the approximate range of concern.

        Values > 1 indicate severe acidification relative to reference.

        Args:
            ph: Measured ocean pH.
            reference_ph: Pre-industrial reference pH (default 8.1).

        Returns:
            Acidification index (0 = no acidification, >1 = severe).
        """
        index = (reference_ph - ph) / 0.3
        index = xr.where(index < 0, 0, index)
        index.name = "acidification_index"
        return cast(xr.DataArray, index)

    def calculate_turbidity_score(
        self,
        turbidity_ntu: xr.DataArray,
    ) -> xr.DataArray:
        """Score marine turbidity on a 0-100 quality scale.

        Based on typical marine water quality standards:
        < 1 NTU: Excellent (score 90-100)
        1-5 NTU: Good (score 60-90)
        5-25 NTU: Fair (score 30-60)
        25-100 NTU: Poor (score 10-30)
        > 100 NTU: Very poor (score 0-10)

        Uses exponential decay function for smooth scoring.

        Args:
            turbidity_ntu: Turbidity in Nephelometric Turbidity Units.

        Returns:
            Quality score (0-100, higher is better).
        """
        score = 100.0 * np.exp(-0.05 * turbidity_ntu)
        score = xr.where(score > 100, 100, score)
        score = xr.where(score < 0, 0, score)
        score.name = "turbidity_score"
        return cast(xr.DataArray, score)

    def calculate_trophic_state_index(
        self,
        chlorophyll_a: xr.DataArray,
    ) -> xr.DataArray:
        """Calculate Carlson Trophic State Index from chlorophyll-a.

        TSI(Chl) = 9.81 * ln(Chl-a) + 30.6

        Classification:
        < 40: Oligotrophic
        40-50: Mesotrophic
        50-70: Eutrophic
        > 70: Hypereutrophic

        Args:
            chlorophyll_a: Chlorophyll-a concentration (ug/L).

        Returns:
            Trophic state index.
        """
        chl_safe = xr.where(chlorophyll_a > 0.01, chlorophyll_a, 0.01)
        tsi = 9.81 * np.log(chl_safe) + 30.6
        tsi.name = "trophic_state_index"
        return cast(xr.DataArray, tsi)

    def composite_marine_wqi(
        self,
        dissolved_oxygen_score: xr.DataArray,
        ph_score: xr.DataArray,
        turbidity_score: xr.DataArray,
        temperature_score: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """Calculate composite marine Water Quality Index.

        Weighted average of individual parameter scores, each 0-100.
        Weights: DO=0.35, pH=0.25, Turbidity=0.25, Temperature=0.15

        Args:
            dissolved_oxygen_score: DO quality score (0-100).
            ph_score: pH quality score (0-100).
            turbidity_score: Turbidity quality score (0-100).
            temperature_score: Temperature quality score (0-100, optional).

        Returns:
            Dataset with composite WQI and classification.
        """
        if temperature_score is not None:
            wqi = (
                0.35 * dissolved_oxygen_score
                + 0.25 * ph_score
                + 0.25 * turbidity_score
                + 0.15 * temperature_score
            )
        else:
            wqi = (
                0.40 * dissolved_oxygen_score + 0.30 * ph_score + 0.30 * turbidity_score
            )

        classification = xr.full_like(wqi, "poor", dtype="U20")
        classification = xr.where(wqi >= 90, "excellent", classification)
        classification = xr.where((wqi >= 70) & (wqi < 90), "good", classification)
        classification = xr.where((wqi >= 50) & (wqi < 70), "fair", classification)
        classification = xr.where((wqi >= 25) & (wqi < 50), "poor", classification)
        classification = xr.where(wqi < 25, "very_poor", classification)

        return xr.Dataset(
            {
                "wqi": wqi,
                "classification": classification,
            }
        )
