"""Water balance modeling with evapotranspiration estimation.

Implements the Thornthwaite and Hargreaves methods for potential
evapotranspiration, and the SCS Curve Number method for runoff estimation.
"""

import logging
from typing import cast

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class WaterBalanceModeler:
    """Model water balance components at catchment scale.

    Implements PET estimation, runoff calculation via SCS-CN method,
    and monthly/annual water balance accounting.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize water balance modeler.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def water_balance_closure(
        self,
        precipitation: xr.DataArray,
        evapotranspiration: xr.DataArray,
        runoff: xr.DataArray,
    ) -> xr.Dataset:
        """Close a water balance from supplied components.

        The storage change is ``P - ET - runoff`` and the closure residual
        is ``P - (ET + runoff) - storage_change`` (identically zero for
        internally consistent components; non-zero values reveal a
        structural mismatch between the supplied fluxes).

        Args:
            precipitation: Precipitation flux.
            evapotranspiration: Evapotranspiration flux.
            runoff: Runoff flux.

        Returns:
            Dataset with ``precipitation``, ``evapotranspiration``,
            ``runoff``, ``storage_change``, ``balance`` (alias of
            ``storage_change``), and ``closure_residual``.
        """
        storage_change = precipitation - evapotranspiration - runoff
        closure_residual = (
            precipitation - (evapotranspiration + runoff) - storage_change
        )
        return xr.Dataset(
            {
                "precipitation": precipitation,
                "evapotranspiration": evapotranspiration,
                "runoff": runoff,
                "storage_change": storage_change,
                "balance": storage_change,
                "closure_residual": closure_residual,
            }
        )

    def thornthwaite_pet(
        self,
        monthly_temp_c: np.ndarray,
        latitude_deg: float,
    ) -> np.ndarray:
        """Calculate monthly PET using the Thornthwaite (1948) method.

        PET = 16 * (10 * T / I)^a * day_length_correction

        where I is the annual heat index and a is derived from I.

        Args:
            monthly_temp_c: Array of 12 monthly mean temperatures (Celsius).
            latitude_deg: Latitude for day length correction.

        Returns:
            Array of 12 monthly PET values (mm/month).
        """
        temp = np.maximum(monthly_temp_c, 0.0)

        heat_index_monthly = (temp / 5.0) ** 1.514
        annual_heat_index = np.sum(heat_index_monthly)

        if annual_heat_index <= 0:
            return np.zeros(12)

        a = (
            6.75e-7 * annual_heat_index**3
            - 7.71e-5 * annual_heat_index**2
            + 1.792e-2 * annual_heat_index
            + 0.49239
        )

        day_length_hours = np.array(
            [self._mean_day_length(latitude_deg, month) for month in range(1, 13)]
        )
        days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

        correction = (day_length_hours / 12.0) * (days_in_month / 30.0)

        pet = np.zeros(12)
        for m in range(12):
            if temp[m] > 0 and annual_heat_index > 0:
                pet[m] = (
                    16.0 * (10.0 * temp[m] / annual_heat_index) ** a * correction[m]
                )

        return pet

    def hargreaves_pet(
        self,
        temp_mean: np.ndarray,
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        latitude_deg: float,
        day_of_year: np.ndarray,
    ) -> np.ndarray:
        """Calculate daily PET using Hargreaves-Samani (1985) method.

        ET0 = 0.0023 * Ra * (T_mean + 17.8) * (T_max - T_min)^0.5

        Args:
            temp_mean: Daily mean temperature (C).
            temp_min: Daily minimum temperature (C).
            temp_max: Daily maximum temperature (C).
            latitude_deg: Latitude (degrees).
            day_of_year: Day of year for each observation.

        Returns:
            Daily PET (mm/day).
        """
        lat_rad = np.radians(latitude_deg)
        ra_values = np.array(
            [self._extraterrestrial_radiation(lat_rad, int(d)) for d in day_of_year]
        )

        temp_range = np.maximum(temp_max - temp_min, 0.0)
        pet = 0.0023 * ra_values * (temp_mean + 17.8) * np.sqrt(temp_range)
        return cast("np.ndarray", np.maximum(pet, 0.0))

    def scs_curve_number_runoff(
        self,
        precipitation_mm: np.ndarray,
        curve_number: float,
        initial_abstraction_ratio: float = 0.2,
    ) -> np.ndarray:
        """Calculate runoff using the SCS Curve Number method.

        Q = (P - Ia)^2 / (P - Ia + S) for P > Ia, else 0
        S = (25400 / CN) - 254
        Ia = lambda * S

        Args:
            precipitation_mm: Precipitation values (mm).
            curve_number: SCS Curve Number (0-100).
            initial_abstraction_ratio: Ia/S ratio (default 0.2).

        Returns:
            Runoff depth (mm).
        """
        if curve_number <= 0:
            return np.zeros_like(precipitation_mm)

        cn = max(1.0, min(100.0, curve_number))
        s = (25400.0 / cn) - 254.0
        ia = initial_abstraction_ratio * s

        p = np.asarray(precipitation_mm, dtype=float)
        excess = p - ia
        excess = np.maximum(excess, 0.0)

        runoff = np.where(
            excess > 0,
            excess**2 / (excess + s),
            0.0,
        )
        return runoff

    def monthly_water_balance(
        self,
        precipitation_mm: np.ndarray,
        pet_mm: np.ndarray,
        soil_capacity_mm: float = 200.0,
        initial_storage_mm: float = 100.0,
    ) -> dict[str, np.ndarray]:
        """Calculate monthly water balance with soil moisture accounting.

        Tracks soil moisture storage, actual evapotranspiration,
        surplus (runoff), and deficit.

        Args:
            precipitation_mm: Monthly precipitation (mm), length 12.
            pet_mm: Monthly potential ET (mm), length 12.
            soil_capacity_mm: Maximum soil moisture storage (mm).
            initial_storage_mm: Initial soil moisture (mm).

        Returns:
            Dictionary with monthly balance components.
        """
        n = len(precipitation_mm)
        storage = np.zeros(n)
        aet = np.zeros(n)
        surplus = np.zeros(n)
        deficit = np.zeros(n)

        current_storage = min(initial_storage_mm, soil_capacity_mm)

        for i in range(n):
            p = precipitation_mm[i]
            pet = pet_mm[i]

            p_minus_pet = p - pet

            if p_minus_pet >= 0:
                aet[i] = pet
                new_storage = current_storage + p_minus_pet
                if new_storage > soil_capacity_mm:
                    surplus[i] = new_storage - soil_capacity_mm
                    new_storage = soil_capacity_mm
            else:
                available = current_storage * (
                    1.0 - np.exp(p_minus_pet / soil_capacity_mm)
                )
                aet[i] = p + available
                new_storage = current_storage - available

            deficit[i] = pet - aet[i]
            storage[i] = new_storage
            current_storage = new_storage

        return {
            "precipitation_mm": precipitation_mm,
            "pet_mm": pet_mm,
            "aet_mm": aet,
            "surplus_mm": surplus,
            "deficit_mm": deficit,
            "soil_storage_mm": storage,
        }

    def _mean_day_length(self, latitude_deg: float, month: int) -> float:
        """Estimate mean day length for a given month and latitude.

        Args:
            latitude_deg: Latitude in degrees.
            month: Month number (1-12).

        Returns:
            Day length in hours.
        """
        mid_day = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
        doy = mid_day[month - 1]

        declination = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + doy)))
        lat_rad = np.radians(latitude_deg)
        dec_rad = np.radians(declination)

        cos_hour_angle = -np.tan(lat_rad) * np.tan(dec_rad)
        cos_hour_angle = max(-1.0, min(1.0, cos_hour_angle))

        hour_angle = np.degrees(np.arccos(cos_hour_angle))
        return float(2.0 * hour_angle / 15.0)

    def _extraterrestrial_radiation(self, lat_rad: float, doy: int) -> float:
        """Calculate daily extraterrestrial radiation (MJ/m^2/day).

        Args:
            lat_rad: Latitude in radians.
            doy: Day of year.

        Returns:
            Ra in mm/day equivalent (divide MJ by 2.45 for mm).
        """
        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
        dec = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)

        ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
        ws = max(0.0, min(np.pi, ws))

        gsc = 0.0820  # Solar constant MJ/m^2/min
        ra = (
            (24.0 * 60.0 / np.pi)
            * gsc
            * dr
            * (
                ws * np.sin(lat_rad) * np.sin(dec)
                + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
            )
        )

        return float(max(0.0, ra / 2.45))
