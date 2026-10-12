"""
Climate impact assessment module.
"""

import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ClimateImpactAssessor:
    """
    Assess climate change impacts on various systems.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize impact assessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def assess_agricultural_impact(
        self,
        temperature: xr.DataArray,
        precipitation: xr.DataArray,
        crop_type: str = "wheat",
    ) -> xr.Dataset:
        """
        Assess climate impact on agriculture with a first-order stress model.

        Stress indices compare the input data against fixed per-crop optima.
        Note: precipitation is compared directly against the annual-total
        optimum (mm/year), so input precipitation must be annual totals (or
        totals over the same period as the optima), not daily values. This
        is a documented first-order proxy, not a crop model.

        Args:
            temperature: Temperature data (deg C)
            precipitation: Precipitation totals comparable to the per-crop
                annual optimum (mm)
            crop_type: Type of crop ('wheat', 'corn', 'rice')

        Returns:
            Impact assessment results
        """
        # Temperature-response crop yield model
        optimal_temp = {"wheat": 20, "corn": 25, "rice": 28}.get(crop_type, 22)
        optimal_precip = {"wheat": 500, "corn": 600, "rice": 1000}.get(crop_type, 500)

        # Calculate stress indices
        temp_stress = np.abs(temperature - optimal_temp) / optimal_temp
        precip_stress = np.abs(precipitation - optimal_precip) / (
            optimal_precip + 1e-10
        )

        # Combined impact
        impact = (temp_stress + precip_stress) / 2

        return xr.Dataset(
            {
                "temperature_stress": temp_stress,
                "precipitation_stress": precip_stress,
                "combined_impact": impact,
            }
        )

    def assess_water_resources(
        self,
        precipitation: xr.DataArray,
        temperature: xr.DataArray,
        evapotranspiration: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess climate impact on water resources via a simple water balance.

        When ``evapotranspiration`` is omitted, ET is estimated with a
        crude linear temperature proxy (``ET = 0.5 * T`` mm per time step),
        which is not a physically-based PET estimate; pass measured or
        modelled ET for meaningful results.

        Args:
            precipitation: Precipitation data (mm per time step)
            temperature: Temperature data (deg C); used only when
                ``evapotranspiration`` is omitted
            evapotranspiration: Optional ET data (mm per time step)

        Returns:
            Water resources assessment
        """
        # Calculate water balance
        if evapotranspiration is None:
            # Estimate ET from temperature
            evapotranspiration = temperature * 0.5  # Linear temperature proxy

        water_balance = precipitation - evapotranspiration
        water_deficit = xr.where(water_balance < 0, -water_balance, 0)

        return xr.Dataset(
            {
                "water_balance": water_balance,
                "water_deficit": water_deficit,
                "precipitation": precipitation,
                "evapotranspiration": evapotranspiration,
            }
        )
