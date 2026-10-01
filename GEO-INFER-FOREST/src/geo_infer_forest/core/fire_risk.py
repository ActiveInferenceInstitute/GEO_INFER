"""Fire risk assessment using fuel moisture and weather indices.

Implements the Keetch-Byram Drought Index (KBDI) and Angstrom Index
for fire danger assessment based on meteorological conditions.
"""

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class FireRiskAssessor:
    """Assess fire risk using meteorological and fuel moisture indices.

    Implements multiple fire danger rating systems that complement the
    Canadian FWI system in wildfire_risk.py. Focuses on drought-based
    indices and simple weather-based fire danger classification.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize fire risk assessor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def calculate_kbdi(
        self,
        daily_max_temp_c: np.ndarray,
        daily_precip_mm: np.ndarray,
        mean_annual_precip_mm: float = 1200.0,
        initial_kbdi: float = 0.0,
    ) -> np.ndarray:
        """Calculate Keetch-Byram Drought Index over a time series.

        KBDI ranges from 0 (no drought) to 800 (extreme drought).
        It represents soil moisture deficit in hundredths of an inch
        of water, converted here to the standard 0-800 scale.

        Formula follows Keetch & Byram (1968):
        dQ = (800 - Q_prev) * (0.968 * exp(0.0486*T) - 8.30) * dt
             / (1 + 10.88 * exp(-0.00174 * R_annual))
        where T = max temperature (C), R_annual = mean annual precip (mm).

        Args:
            daily_max_temp_c: Array of daily maximum temperatures (Celsius).
            daily_precip_mm: Array of daily precipitation (mm).
            mean_annual_precip_mm: Mean annual precipitation for the location.
            initial_kbdi: Starting KBDI value.

        Returns:
            Array of daily KBDI values.
        """
        n_days = len(daily_max_temp_c)
        kbdi = np.zeros(n_days, dtype=float)

        # Day 0 starts from initial_kbdi and processes that day's weather
        # (rain deficit reduction and temperature drying) like every other day.
        q_prev = initial_kbdi
        for i in range(n_days):
            net_precip = max(0.0, daily_precip_mm[i] - 5.08)
            q_after_rain = max(0.0, q_prev - net_precip * 3.937)

            temp_c = max(0.0, daily_max_temp_c[i])
            temp_f = temp_c * 9.0 / 5.0 + 32.0

            r_annual_inches = mean_annual_precip_mm / 25.4

            numerator = (800.0 - q_after_rain) * (
                0.968 * np.exp(0.0486 * temp_f) - 8.30
            )
            denominator = 1.0 + 10.88 * np.exp(-0.001736 * r_annual_inches)

            dq = numerator / (denominator * 1000.0) if denominator != 0 else 0.0
            dq = max(0.0, dq)

            kbdi[i] = min(800.0, q_after_rain + dq)
            q_prev = kbdi[i]

        return kbdi

    def calculate_angstrom_index(
        self,
        temperature_c: float,
        relative_humidity: float,
    ) -> dict[str, Any]:
        """Calculate Angstrom Fire Danger Index.

        A simple index using temperature and humidity:
        I = (RH / 20) + ((27 - T) / 10)

        Index values:
        < 2.0: Fire conditions likely
        2.0 - 2.5: Fire conditions exist
        > 2.5: Fire conditions unlikely
        > 4.0: Fire conditions very unlikely

        Args:
            temperature_c: Air temperature in Celsius.
            relative_humidity: Relative humidity (0-100).

        Returns:
            Dictionary with index value and danger classification.
        """
        index_value = (relative_humidity / 20.0) + ((27.0 - temperature_c) / 10.0)

        if index_value < 2.0:
            danger = "high"
        elif index_value < 2.5:
            danger = "moderate"
        elif index_value < 4.0:
            danger = "low"
        else:
            danger = "very_low"

        return {
            "angstrom_index": float(index_value),
            "fire_danger": danger,
            "temperature_c": float(temperature_c),
            "relative_humidity": float(relative_humidity),
        }

    def calculate_fuel_moisture(
        self,
        temperature_c: float,
        relative_humidity: float,
        time_lag_hours: float = 1.0,
    ) -> float:
        """Estimate dead fuel moisture content from weather conditions.

        Uses the equilibrium moisture content (EMC) approach based on
        temperature and humidity. The piecewise EMC tables are the Simard
        (1968) formulation (as tabulated in Simard/Jonas USFS references);
        a time-lag factor is applied for time-lag fuel moisture estimation.

        Args:
            temperature_c: Air temperature (Celsius).
            relative_humidity: Relative humidity (0-100).
            time_lag_hours: Fuel time-lag class in hours (1, 10, 100).

        Returns:
            Estimated fuel moisture content (percent dry weight).
        """
        rh = max(0.0, min(100.0, relative_humidity))

        if rh < 10.0:
            emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * temperature_c
        elif rh < 50.0:
            emc = 2.22749 + 0.160107 * rh - 0.01478 * temperature_c
        else:
            emc = (
                21.0606
                + 0.005565 * rh**2
                - 0.00035 * rh * temperature_c
                - 0.483199 * rh
            )

        emc = max(1.0, emc)

        lag_factor = 1.0 + 0.1 * np.log(max(1.0, time_lag_hours))
        fuel_moisture = emc * lag_factor

        return float(max(1.0, fuel_moisture))

    def assess_fire_risk_grid(
        self,
        temperature: xr.DataArray,
        humidity: xr.DataArray,
        wind_speed: xr.DataArray,
        slope: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """Assess fire risk over a spatial grid.

        Combines temperature, humidity, wind, and terrain factors
        into a composite fire risk index (0-1).

        Args:
            temperature: Temperature grid (Celsius).
            humidity: Relative humidity grid (%).
            wind_speed: Wind speed grid (km/h).
            slope: Terrain slope grid (degrees, optional).

        Returns:
            Dataset with component risk factors and composite index.
        """
        temp_risk = (temperature - 15.0) / 30.0
        temp_risk = xr.where(temp_risk < 0, 0, temp_risk)
        temp_risk = xr.where(temp_risk > 1, 1, temp_risk)

        humidity_risk = 1.0 - humidity / 100.0
        humidity_risk = xr.where(humidity_risk < 0, 0, humidity_risk)
        humidity_risk = xr.where(humidity_risk > 1, 1, humidity_risk)

        wind_risk = wind_speed / 60.0
        wind_risk = xr.where(wind_risk < 0, 0, wind_risk)
        wind_risk = xr.where(wind_risk > 1, 1, wind_risk)

        composite = 0.30 * temp_risk + 0.35 * humidity_risk + 0.20 * wind_risk

        if slope is not None:
            slope_risk = slope / 45.0
            slope_risk = xr.where(slope_risk < 0, 0, slope_risk)
            slope_risk = xr.where(slope_risk > 1, 1, slope_risk)
            composite = composite + 0.15 * slope_risk
        else:
            composite = composite / 0.85

        composite = xr.where(composite > 1, 1, composite)
        composite = xr.where(composite < 0, 0, composite)

        return xr.Dataset(
            {
                "fire_risk_index": composite,
                "temperature_risk": temp_risk,
                "humidity_risk": humidity_risk,
                "wind_risk": wind_risk,
            }
        )
