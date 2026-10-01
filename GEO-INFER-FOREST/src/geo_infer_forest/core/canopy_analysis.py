"""Tree canopy analysis using vegetation indices.

Provides NDVI-based canopy cover estimation, leaf area index calculation,
and canopy gap detection for forest monitoring applications.
"""

import logging
from typing import cast

import numpy as np
import xarray as xr
import scipy.ndimage

logger = logging.getLogger(__name__)


class CanopyAnalyzer:
    """Analyze forest canopy structure from remote sensing data.

    Uses vegetation indices (NDVI, EVI) to estimate canopy cover percentage,
    leaf area index, and detect canopy gaps indicating disturbance.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize canopy analyzer.

        Args:
            config: Configuration dictionary with analysis parameters.
        """
        self.config = config or {}
        self.ndvi_forest_threshold: float = self.config.get(
            "ndvi_forest_threshold", 0.4
        )
        self.ndvi_dense_threshold: float = self.config.get("ndvi_dense_threshold", 0.7)

    def calculate_ndvi(
        self,
        red: xr.DataArray,
        nir: xr.DataArray,
    ) -> xr.DataArray:
        """Calculate Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            red: Red band reflectance (0-1 or 0-10000 scaled).
            nir: Near-infrared band reflectance.

        Returns:
            NDVI values in range [-1, 1].
        """
        denominator = nir.astype(float) + red.astype(float)
        ndvi = xr.where(
            denominator != 0,
            (nir.astype(float) - red.astype(float)) / denominator,
            0.0,
        )
        ndvi = xr.where(ndvi > 1.0, 1.0, ndvi)
        ndvi = xr.where(ndvi < -1.0, -1.0, ndvi)
        ndvi.name = "ndvi"
        return cast(xr.DataArray, ndvi)

    def calculate_evi(
        self,
        red: xr.DataArray,
        nir: xr.DataArray,
        blue: xr.DataArray,
        gain: float = 2.5,
        c1: float = 6.0,
        c2: float = 7.5,
        l_soil: float = 1.0,
    ) -> xr.DataArray:
        """Calculate Enhanced Vegetation Index.

        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

        Args:
            red: Red band reflectance.
            nir: Near-infrared band reflectance.
            blue: Blue band reflectance.
            gain: Gain factor (default 2.5).
            c1: Coefficient for atmospheric resistance (red, default 6.0).
            c2: Coefficient for atmospheric resistance (blue, default 7.5).
            l_soil: Canopy background adjustment (default 1.0).

        Returns:
            EVI values.
        """
        r = red.astype(float)
        n = nir.astype(float)
        b = blue.astype(float)
        denominator = n + c1 * r - c2 * b + l_soil
        evi = xr.where(
            denominator != 0,
            gain * (n - r) / denominator,
            0.0,
        )
        evi = xr.where(evi > 1.0, 1.0, evi)
        evi = xr.where(evi < -1.0, -1.0, evi)
        evi.name = "evi"
        return cast(xr.DataArray, evi)

    def estimate_canopy_cover(
        self,
        ndvi: xr.DataArray,
        method: str = "linear",
    ) -> xr.DataArray:
        """Estimate canopy cover percentage from NDVI.

        Uses the relationship between NDVI and fractional vegetation cover (FVC).

        Linear method: FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)
        Squared method: FVC = ((NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil))^2

        Args:
            ndvi: NDVI values.
            method: Estimation method ('linear' or 'squared').

        Returns:
            Canopy cover percentage (0-100).
        """
        ndvi_soil = 0.05
        ndvi_veg = 0.90

        fvc = (ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)
        fvc = xr.where(fvc < 0.0, 0.0, fvc)
        fvc = xr.where(fvc > 1.0, 1.0, fvc)

        if method == "squared":
            fvc = fvc**2

        canopy_cover = fvc * 100.0
        canopy_cover.name = "canopy_cover_pct"
        return cast(xr.DataArray, canopy_cover)

    def estimate_leaf_area_index(
        self,
        ndvi: xr.DataArray,
        k_ext: float = 0.5,
    ) -> xr.DataArray:
        """Estimate Leaf Area Index from NDVI using Beer-Lambert law.

        LAI = -ln(1 - fvc) / k_ext
        where fvc is fractional vegetation cover derived from NDVI.

        Args:
            ndvi: NDVI values.
            k_ext: Light extinction coefficient (default 0.5 for broadleaf forests).

        Returns:
            Leaf Area Index (m^2/m^2).
        """
        fvc = (ndvi - 0.05) / (0.90 - 0.05)
        fvc = xr.where(fvc < 0.01, 0.01, fvc)
        fvc = xr.where(fvc > 0.99, 0.99, fvc)

        lai = -np.log(1.0 - fvc) / k_ext
        lai = xr.where(lai < 0.0, 0.0, lai)
        lai.name = "lai"
        return cast(xr.DataArray, lai)

    def detect_canopy_gaps(
        self,
        ndvi: xr.DataArray,
        gap_threshold: float | None = None,
        min_gap_pixels: int = 1,
    ) -> xr.Dataset:
        """Detect canopy gaps from NDVI data.

        Gaps are areas where NDVI falls below the forest threshold,
        indicating openings in the canopy from disturbance, mortality, or clearing.

        Args:
            ndvi: NDVI values.
            gap_threshold: NDVI threshold below which a pixel is a gap.
                Defaults to self.ndvi_forest_threshold.
            min_gap_pixels: Minimum contiguous gap size in pixels.

        Returns:
            Dataset with gap mask, gap fraction, and gap size statistics.
        """
        threshold = (
            gap_threshold if gap_threshold is not None else self.ndvi_forest_threshold
        )

        gap_mask = ndvi < threshold

        # Filter to contiguous gaps of at least min_gap_pixels via connected
        # component labeling; isolated sub-threshold pixels are not gaps.
        gap_array = np.asarray(gap_mask.values, dtype=bool)
        labels, _ = scipy.ndimage.label(gap_array)
        component_sizes = np.bincount(labels.ravel())
        small_components = np.isin(
            labels, np.nonzero(component_sizes < min_gap_pixels)[0]
        )
        gap_array = gap_array & ~small_components
        gap_mask = xr.DataArray(gap_array, coords=ndvi.coords, dims=ndvi.dims)

        gap_sizes = component_sizes[1:]  # skip background label 0
        gap_sizes = gap_sizes[gap_sizes >= min_gap_pixels]

        total_pixels = float(ndvi.size)
        gap_pixels = float(gap_mask.sum())
        gap_fraction = gap_pixels / total_pixels if total_pixels > 0 else 0.0

        mean_gap_ndvi = float(ndvi.where(gap_mask).mean()) if gap_pixels > 0 else 0.0
        mean_forest_ndvi = (
            float(ndvi.where(~gap_mask).mean())
            if (total_pixels - gap_pixels) > 0
            else 0.0
        )

        return xr.Dataset(
            {
                "gap_mask": gap_mask,
                "ndvi": ndvi,
            },
            attrs={
                "gap_threshold": threshold,
                "gap_fraction": gap_fraction,
                "gap_pixel_count": int(gap_pixels),
                "total_pixel_count": int(total_pixels),
                "mean_gap_ndvi": mean_gap_ndvi,
                "mean_forest_ndvi": mean_forest_ndvi,
                "gap_count": int(gap_sizes.size),
                "mean_gap_size_pixels": float(gap_sizes.mean())
                if gap_sizes.size > 0
                else 0.0,
                "max_gap_size_pixels": int(gap_sizes.max())
                if gap_sizes.size > 0
                else 0,
                "min_gap_pixels": int(min_gap_pixels),
            },
        )

    def classify_canopy_density(
        self,
        ndvi: xr.DataArray,
    ) -> xr.DataArray:
        """Classify canopy density into categories based on NDVI.

        Categories:
            0 = Non-forest (NDVI < 0.2)
            1 = Sparse canopy (0.2 <= NDVI < 0.4)
            2 = Moderate canopy (0.4 <= NDVI < 0.6)
            3 = Dense canopy (0.6 <= NDVI < 0.8)
            4 = Very dense canopy (NDVI >= 0.8)

        Args:
            ndvi: NDVI values.

        Returns:
            Classified canopy density (integer categories 0-4).
        """
        density = xr.zeros_like(ndvi, dtype=int)
        density = xr.where(ndvi >= 0.2, 1, density)
        density = xr.where(ndvi >= 0.4, 2, density)
        density = xr.where(ndvi >= 0.6, 3, density)
        density = xr.where(ndvi >= 0.8, 4, density)
        density.name = "canopy_density_class"
        return cast(xr.DataArray, density)
