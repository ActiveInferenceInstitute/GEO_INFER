"""Flood and drought analysis module."""

import logging
import xarray as xr

logger = logging.getLogger(__name__)


class FloodDroughtAnalyzer:
    """Analyze flood and drought risks."""

    def __init__(self, config: dict | None = None):
        """Initialize flood/drought analyzer."""
        self.config = config or {}

    def assess_flood_risk(
        self,
        precipitation: xr.DataArray,
        elevation: xr.DataArray,
        soil_saturation: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess flood risk.

        Flood risk is a simple equal-weight composite of binary hazard
        indicators: the temporal frequency of extreme precipitation (days
        above the 95th percentile) and membership in the low-elevation
        band (below the 20th percentile). When soil saturation is
        supplied, a third saturated-soil indicator (saturation > 0.8)
        is averaged in with the same weight. Each indicator contributes
        0/1 and the composite is the arithmetic mean, so a cell at high
        elevation with frequent extreme rain can score the same as a
        low-elevation cell with moderate rain; the weighting is a
        deliberate first-order screening heuristic, not a calibrated
        flood model.

        Args:
            precipitation: Precipitation data (time, lat, lon)
            elevation: Elevation data (lat, lon)
            soil_saturation: Optional soil saturation data (0-1)

        Returns:
            Flood risk assessment
        """
        # Extreme precipitation
        precip_threshold = precipitation.quantile(0.95, dim="time").drop_vars(
            "quantile"
        )
        extreme_precip = precipitation > precip_threshold

        # Low elevation (flood-prone areas)
        elevation_threshold = elevation.quantile(0.2).drop_vars("quantile")
        low_elevation = elevation < elevation_threshold

        indicators = [
            extreme_precip.astype(float).mean(dim="time"),
            low_elevation.astype(float),
        ]
        if soil_saturation is not None:
            # Saturated soil increases risk
            saturated = soil_saturation > 0.8
            indicators.append(saturated.astype(float))
        flood_risk = sum(indicators) / len(indicators)

        return xr.Dataset(
            {
                "flood_risk": flood_risk,
                "extreme_precipitation": extreme_precip,
                "low_elevation": low_elevation,
            }
        )

    def assess_drought_risk(
        self,
        precipitation: xr.DataArray,
        evapotranspiration: xr.DataArray | None = None,
        soil_moisture: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess drought risk.

        Args:
            precipitation: Precipitation data
            evapotranspiration: Optional ET data
            soil_moisture: Optional soil moisture data

        Returns:
            Drought risk assessment. The ``water_deficit`` variable is
            ``None`` when no ET data is supplied.
        """
        # Low precipitation
        precip_threshold = precipitation.quantile(0.1, dim="time")
        low_precip = precipitation < precip_threshold

        # Water deficit (None when no ET data is supplied)
        water_deficit = None
        if evapotranspiration is not None:
            water_deficit = evapotranspiration - precipitation
            deficit_threshold = water_deficit.quantile(0.9)
            high_deficit = water_deficit > deficit_threshold
        else:
            high_deficit = low_precip

        indicators = [low_precip.astype(float), high_deficit.astype(float)]
        if soil_moisture is not None:
            # Low soil moisture increases risk
            low_moisture = soil_moisture < soil_moisture.quantile(0.2)
            indicators.append(low_moisture.astype(float))
        drought_risk = sum(indicators) / len(indicators)

        return xr.Dataset(
            {
                "drought_risk": drought_risk,
                "low_precipitation": low_precip,
                "water_deficit": water_deficit,
            }
        )
