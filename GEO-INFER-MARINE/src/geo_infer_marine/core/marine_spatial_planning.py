"""
Marine spatial planning module.
"""

import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _normalized(data: xr.DataArray) -> xr.DataArray:
    """Scale a DataArray to [0, 1] by its finite maximum.

    Args:
        data: Input values.

    Returns:
        ``data / max``; uniform 1.0 when the maximum is zero,
        negative, or not finite, so downstream division cannot
        produce NaN priorities.
    """
    max_value = float(data.max())
    if max_value <= 0.0 or not np.isfinite(max_value):
        return xr.ones_like(data)
    return data / max_value


class MarineSpatialPlanner:
    """
    Marine spatial planning (MSP) tools.
    """

    def __init__(self, config: dict | None = None):
        """Initialize marine spatial planner."""
        self.config = config or {}

    def design_mpa_network(
        self,
        biodiversity_data: xr.DataArray,
        threat_data: xr.DataArray | None = None,
        target_coverage: float = 0.3,
    ) -> xr.Dataset:
        """
        Design marine protected area (MPA) network.

        Args:
            biodiversity_data: Biodiversity index
            threat_data: Optional threat/pressure data
            target_coverage: Target MPA coverage; must satisfy
                0 < target_coverage <= 1

        Returns:
            MPA network design
        """
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError(
                f"target_coverage must be in (0, 1], got {target_coverage!r}"
            )

        # Prioritize high biodiversity areas; uniform when data is flat,
        # all-zero or non-finite so no NaN priorities are produced.
        priority = _normalized(biodiversity_data)

        if threat_data is not None:
            # Also consider threat levels
            priority = priority * (1 + _normalized(threat_data))

        # Select top priority areas to meet coverage target
        threshold = priority.quantile(1 - target_coverage)
        mpa_mask = priority >= threshold

        return xr.Dataset(
            {
                "mpa_mask": mpa_mask,
                "priority": priority,
                "coverage": mpa_mask.sum() / mpa_mask.size,
            }
        )

    def optimize_offshore_wind_siting(
        self,
        wind_resource: xr.DataArray,
        depth: xr.DataArray,
        exclusion_zones: xr.DataArray | None = None,
        max_depth: float = 50.0,
    ) -> xr.Dataset:
        """
        Optimize offshore wind farm siting.

        Args:
            wind_resource: Wind speed/power potential
            depth: Water depth
            exclusion_zones: Optional exclusion areas
            max_depth: Maximum viable depth (meters)

        Returns:
            Optimal siting analysis
        """
        # Suitability based on wind and depth. Land/dry cells (negative
        # depth) are excluded rather than rewarded, so suitability stays
        # within [0, 1] and never exceeds the wind-only score.
        wind_suitability = _normalized(wind_resource)
        depth_suitability = xr.where(
            (depth >= 0) & (depth <= max_depth), 1 - depth / max_depth, 0
        )

        suitability = wind_suitability * depth_suitability

        if exclusion_zones is not None:
            suitability = xr.where(exclusion_zones, 0, suitability)

        return xr.Dataset(
            {
                "suitability": suitability,
                "wind_suitability": wind_suitability,
                "depth_suitability": depth_suitability,
            }
        )
