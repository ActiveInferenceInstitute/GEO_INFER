"""
Climate indices calculation module.

Implements calculation of various climate indices including:
- Standardized Precipitation Index (SPI)
- Palmer-style drought index (first-order moisture anomaly)
- Heat indices
- Climate extremes indices
"""

import logging
from typing import cast

import numpy as np
import xarray as xr
from scipy import stats

logger = logging.getLogger(__name__)


class ClimateIndicesCalculator:
    """
    Calculate climate indices from climate data.

    Supports SPI, a first-order Palmer-style drought index, heat indices,
    and climate extremes.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize climate indices calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def calculate_spi(
        self,
        precipitation: xr.DataArray,
        timescale: int = 3,
        distribution: str = "gamma",
    ) -> xr.DataArray:
        """
        Calculate Standardized Precipitation Index (SPI).

        Args:
            precipitation: Precipitation data array with a ``time`` dimension
            timescale: Accumulation timescale in months
            distribution: Distribution type ('gamma' or 'normal')

        Returns:
            SPI values as DataArray
        """
        # Accumulate precipitation over timescale
        if timescale > 1:
            precip_accum = precipitation.rolling(time=timescale, center=False).sum()
        else:
            precip_accum = precipitation

        # Calculate SPI using the requested distribution
        if distribution == "gamma":
            spi = self._spi_gamma(precip_accum)
        elif distribution == "normal":
            spi = self._spi_normal(precip_accum)
        else:
            raise ValueError(
                f"climate_indices.calculate_spi: unsupported distribution "
                f"{distribution!r}; expected 'gamma' or 'normal'"
            )

        spi.name = f"SPI_{timescale}"
        return spi

    def _spi_gamma(self, precip: xr.DataArray) -> xr.DataArray:
        """Calculate SPI using the gamma distribution.

        The gamma distribution is fit independently for every grid cell
        (i.e. along the ``time`` dimension only) using
        :func:`xarray.apply_ufunc`, so the time axis may appear in any
        position and arbitrary leading dimensions (``lat``, ``lon``, ...)
        are handled per cell.

        Zero-precipitation values are handled with the standard Thom (1958)
        mixed-distribution correction.
        """

        def _spi_gamma_1d(values: np.ndarray) -> np.ndarray:
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                return np.full(values.shape, np.nan)

            try:
                alpha, _, beta = stats.gamma.fit(valid, floc=0)
            except Exception as exc:
                logger.warning("Error fitting gamma distribution: %s", exc)
                return np.full(values.shape, np.nan)

            zero_prob = float(np.mean(valid <= 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                cdf = stats.gamma.cdf(values, alpha, loc=0.0, scale=beta)
                prob = zero_prob + (1.0 - zero_prob) * cdf
                prob = np.clip(prob, 1e-9, 1.0 - 1e-9)
                result = stats.norm.ppf(prob)
            result[~np.isfinite(values)] = np.nan
            return result

        spi = xr.apply_ufunc(
            _spi_gamma_1d,
            precip,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="forbidden",
            keep_attrs=True,
        )
        return spi

    def _spi_normal(self, precip: xr.DataArray) -> xr.DataArray:
        """Calculate SPI using normal distribution."""
        mean = precip.mean(dim="time")
        std = precip.std(dim="time")

        spi = (precip - mean) / std
        return spi

    def calculate_heat_index(
        self, temperature: xr.DataArray, humidity: xr.DataArray | None = None
    ) -> xr.DataArray:
        """
        Calculate heat index (apparent temperature).

        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100) if available

        Returns:
            Heat index values
        """
        if humidity is not None:
            # Full heat index calculation with humidity
            hi = self._heat_index_with_humidity(temperature, humidity)
        else:
            # Temperature-only index basis
            hi = temperature.copy()
            hi.name = "heat_index"

        return hi

    def _heat_index_with_humidity(
        self, temp: xr.DataArray, rh: xr.DataArray
    ) -> xr.DataArray:
        """Calculate heat index using temperature and humidity."""
        # Heat index formula (Rothfusz equation approximation)
        hi = temp.copy()

        # Convert to Fahrenheit for calculation
        temp_f = temp * 9 / 5 + 32

        # Heat index calculation
        hi_values = (
            -42.379
            + 2.04901523 * temp_f
            + 10.14333127 * rh
            - 0.22475541 * temp_f * rh
            - 6.83783e-3 * temp_f**2
            - 5.481717e-2 * rh**2
            + 1.22874e-3 * temp_f**2 * rh
            + 8.5282e-4 * temp_f * rh**2
            - 1.99e-6 * temp_f**2 * rh**2
        )

        # Convert back to Celsius
        hi.values = (hi_values - 32) * 5 / 9
        hi.name = "heat_index"

        return hi

    def calculate_extreme_indices(
        self, temperature: xr.DataArray, precipitation: xr.DataArray | None = None
    ) -> xr.Dataset:
        """
        Calculate climate extreme indices.

        Args:
            temperature: Temperature data
            precipitation: Optional precipitation data

        Returns:
            Dataset with extreme indices
        """
        indices = {}

        # Hot days (days above 90th percentile)
        temp_90th = float(temperature.quantile(0.9, dim="time"))
        hot_days = (temperature > temp_90th).sum(dim="time")
        indices["hot_days"] = hot_days

        # Cold days (days below 10th percentile)
        temp_10th = float(temperature.quantile(0.1, dim="time"))
        cold_days = (temperature < temp_10th).sum(dim="time")
        indices["cold_days"] = cold_days

        # Maximum temperature
        indices["max_temp"] = temperature.max(dim="time")

        # Minimum temperature
        indices["min_temp"] = temperature.min(dim="time")

        if precipitation is not None:
            # Heavy precipitation days (above 95th percentile)
            precip_95th = float(precipitation.quantile(0.95, dim="time"))
            heavy_precip_days = (precipitation > precip_95th).sum(dim="time")
            indices["heavy_precip_days"] = heavy_precip_days

            # Total precipitation
            indices["total_precip"] = precipitation.sum(dim="time")

        return xr.Dataset(indices)

    def calculate_pdsi(
        self,
        precipitation: xr.DataArray,
        temperature: xr.DataArray,
        awc: float = 100.0,  # Available water capacity (mm)
    ) -> xr.DataArray:
        """
        Calculate a first-order Palmer-style drought severity index.

        Note: this is NOT the full Palmer (1965) PDSI. The full system
        requires a monthly water-balance bookkeeping model (soil recharge,
        loss, and surplus coefficients derived from the local AWC) and
        self-calibrating climatic constants. This implementation uses a
        simplified moisture-anomaly proxy: Thornthwaite potential
        evapotranspiration, the monthly water balance ``P - PET``, its
        cumulative sum, and a z-score rescaling onto the Palmer scale
        (clipped to [-6, +6]). The ``awc`` parameter is retained for API
        compatibility but does not affect the calculation.

        Args:
            precipitation: Monthly precipitation (mm)
            temperature: Monthly mean temperature (deg C)
            awc: Available water capacity of soil (mm; currently unused)

        Returns:
            Drought index values on the PDSI scale
        """
        # Calculate potential evapotranspiration (Thornthwaite method)
        pet = self._calculate_pet(temperature)

        # Water balance
        water_balance = precipitation - pet

        # Accumulate water balance
        accumulated = water_balance.cumsum(dim="time")

        # Normalize to PDSI scale (-6 to +6)
        mean_balance = accumulated.mean(dim="time")
        std_balance = accumulated.std(dim="time")

        pdsi = (accumulated - mean_balance) / (std_balance + 1e-10) * 2
        pdsi = xr.where(pdsi > 6, 6, pdsi)
        pdsi = xr.where(pdsi < -6, -6, pdsi)

        pdsi.name = "PDSI"
        return cast(xr.DataArray, pdsi)

    def _calculate_pet(self, temperature: xr.DataArray) -> xr.DataArray:
        """Calculate potential evapotranspiration using the Thornthwaite method.

        Implements the unadjusted Thornthwaite (1948) monthly equation:

        - monthly heat index: ``i = (T / 5) ** 1.514`` for ``T > 0``, else 0
        - annual heat index: ``I = sum(i)``
        - exponent: ``a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 0.01791 * I + 0.49239``
        - ``PET = 16 * (10 * T / I) ** a`` mm/month for ``T > 0``, else 0

        The latitude/day-length correction factor is not applied (it is
        fixed at 1.0), so results are the standard unadjusted monthly PET
        estimates. Input is expected to be monthly mean temperature (deg C).
        """
        heat_i = xr.where(temperature > 0, (temperature / 5.0) ** 1.514, 0.0)
        heat_index_sum = heat_i.sum(dim="time")
        a = (
            6.75e-7 * heat_index_sum**3
            - 7.71e-5 * heat_index_sum**2
            + 0.01791 * heat_index_sum
            + 0.49239
        )
        # Guard the division: when I == 0 every month has T <= 0 and PET is 0.
        i_safe = xr.where(heat_index_sum > 0, heat_index_sum, 1.0)
        pet = xr.where(temperature > 0, 16.0 * (10.0 * temperature / i_safe) ** a, 0.0)
        pet.name = "PET"
        return cast(xr.DataArray, pet)
