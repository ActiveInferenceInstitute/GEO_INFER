"""Climate classification systems.

Implements the Koppen-Geiger climate classification system and
related thermal/moisture zone calculations.
"""

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ClimateClassifier:
    """Classify climate zones using standard classification systems.

    Implements the Koppen-Geiger system which uses temperature and
    precipitation thresholds to categorize climates into main groups
    (A, B, C, D, E) and sub-types.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize climate classifier.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def koppen_geiger_classify(
        self,
        monthly_temp_c: np.ndarray,
        monthly_precip_mm: np.ndarray,
    ) -> dict[str, Any]:
        """Classify climate using Koppen-Geiger system.

        Uses 12 monthly mean temperatures and 12 monthly precipitation
        totals to determine the Koppen-Geiger classification.

        Args:
            monthly_temp_c: Array of 12 monthly mean temperatures (Celsius).
            monthly_precip_mm: Array of 12 monthly precipitation (mm).

        Returns:
            Dictionary with classification code, description, and component info.
        """
        t = np.asarray(monthly_temp_c, dtype=float)
        p = np.asarray(monthly_precip_mm, dtype=float)

        t_ann = float(np.mean(t))
        p_ann = float(np.sum(p))
        t_max = float(np.max(t))
        t_min = float(np.min(t))

        p_summer = p[3:9] if t[6] > t[0] else np.concatenate([p[9:], p[:3]])
        p_winter = p[9:] if t[6] > t[0] else p[3:9]
        if len(p_summer) == 0:
            p_summer = p[:6]
        if len(p_winter) == 0:
            p_winter = p[6:]

        p_summer_total = float(np.sum(p_summer))
        p_winter_total = float(np.sum(p_winter))

        p_driest_month = float(np.min(p))

        # Koppen (B) dryness threshold in mm: 20 * T_ann (2 * T_ann in cm),
        # adjusted for precipitation seasonality because winter
        # precipitation is more effective (less evaporative demand):
        #   - >= 70% of annual precipitation in the winter (cooler) half:
        #     threshold = 20 * T_ann
        #   - >= 70% in the summer (warmer) half:
        #     threshold = 20 * T_ann + 280
        #   - otherwise: threshold = 20 * T_ann + 140
        if p_ann > 0:
            summer_share = p_summer_total / p_ann
            winter_share = p_winter_total / p_ann
        else:
            summer_share = 0.0
            winter_share = 0.0
        if winter_share >= 0.7:
            dry_threshold = 20.0 * t_ann
        elif summer_share >= 0.7:
            dry_threshold = 20.0 * t_ann + 280.0
        else:
            dry_threshold = 20.0 * t_ann + 140.0

        n_warm = int(np.sum(t >= 10))

        if t_min >= 18:
            code, desc = self._classify_tropical(t, p, p_driest_month)
        elif p_ann < dry_threshold:
            code, desc = self._classify_arid(t, p, p_ann, dry_threshold)
        elif t_min >= -3 and t_min < 18 and t_max >= 10:
            code, desc = self._classify_temperate(
                t, p, t_max, t_min, p_summer, p_winter
            )
        elif t_min < -3 and t_max >= 10:
            code, desc = self._classify_continental(
                t, p, t_max, t_min, p_summer, p_winter
            )
        elif t_max < 10:
            code, desc = self._classify_polar(t_max)
        else:
            code, desc = "Cfa", "Humid subtropical"

        return {
            "code": code,
            "description": desc,
            "main_group": code[0],
            "annual_temp_c": float(t_ann),
            "annual_precip_mm": float(p_ann),
            "temp_warmest_month": float(t_max),
            "temp_coldest_month": float(t_min),
            "months_above_10c": n_warm,
        }

    def _classify_tropical(
        self,
        t: np.ndarray,
        p: np.ndarray,
        p_driest: float,
    ) -> tuple:
        """Classify tropical (A) climates."""
        if p_driest >= 60:
            return "Af", "Tropical rainforest"
        elif p_driest >= 100 - float(np.sum(p)) / 25:
            return "Am", "Tropical monsoon"
        else:
            return "Aw", "Tropical savanna"

    def _classify_arid(
        self,
        t: np.ndarray,
        p: np.ndarray,
        p_ann: float,
        threshold: float,
    ) -> tuple:
        """Classify arid (B) climates."""
        t_ann = float(np.mean(t))
        if p_ann < threshold / 2:
            if t_ann >= 18:
                return "BWh", "Hot desert"
            else:
                return "BWk", "Cold desert"
        else:
            if t_ann >= 18:
                return "BSh", "Hot semi-arid"
            else:
                return "BSk", "Cold semi-arid"

    def _classify_temperate(
        self,
        t: np.ndarray,
        p: np.ndarray,
        t_max: float,
        t_min: float,
        p_summer: np.ndarray,
        p_winter: np.ndarray,
    ) -> tuple:
        """Classify temperate (C) climates."""
        p_s_min = float(np.min(p_summer)) if len(p_summer) > 0 else 0
        p_w_min = float(np.min(p_winter)) if len(p_winter) > 0 else 0
        p_w_max = float(np.max(p_winter)) if len(p_winter) > 0 else 1
        p_s_max = float(np.max(p_summer)) if len(p_summer) > 0 else 1

        if p_s_min < 40 and p_s_min < p_w_max / 3:
            precip_code = "s"
        elif p_w_min < p_s_max / 10:
            precip_code = "w"
        else:
            precip_code = "f"

        if t_max >= 22:
            temp_code = "a"
        elif sum(1 for ti in t if ti >= 10) >= 4:
            temp_code = "b"
        else:
            temp_code = "c"

        code = f"C{precip_code}{temp_code}"
        descriptions = {
            "Cfa": "Humid subtropical",
            "Cfb": "Oceanic",
            "Cfc": "Subpolar oceanic",
            "Csa": "Hot-summer Mediterranean",
            "Csb": "Warm-summer Mediterranean",
            "Csc": "Cold-summer Mediterranean",
            "Cwa": "Subtropical highland/monsoon",
            "Cwb": "Subtropical oceanic highland",
            "Cwc": "Cold subtropical highland",
        }
        return code, descriptions.get(code, "Temperate")

    def _classify_continental(
        self,
        t: np.ndarray,
        p: np.ndarray,
        t_max: float,
        t_min: float,
        p_summer: np.ndarray,
        p_winter: np.ndarray,
    ) -> tuple:
        """Classify continental (D) climates."""
        p_s_min = float(np.min(p_summer)) if len(p_summer) > 0 else 0
        p_w_min = float(np.min(p_winter)) if len(p_winter) > 0 else 0
        p_w_max = float(np.max(p_winter)) if len(p_winter) > 0 else 1
        p_s_max = float(np.max(p_summer)) if len(p_summer) > 0 else 1

        if p_s_min < 40 and p_s_min < p_w_max / 3:
            precip_code = "s"
        elif p_w_min < p_s_max / 10:
            precip_code = "w"
        else:
            precip_code = "f"

        if t_max >= 22:
            temp_code = "a"
        elif sum(1 for ti in t if ti >= 10) >= 4:
            temp_code = "b"
        elif t_min < -38:
            temp_code = "d"
        else:
            temp_code = "c"

        code = f"D{precip_code}{temp_code}"
        descriptions = {
            "Dfa": "Hot-summer humid continental",
            "Dfb": "Warm-summer humid continental",
            "Dfc": "Subarctic",
            "Dfd": "Extremely cold subarctic",
            "Dwa": "Hot-summer continental monsoon",
            "Dwb": "Warm-summer continental monsoon",
            "Dwc": "Subarctic monsoon",
            "Dwd": "Extremely cold subarctic monsoon",
        }
        return code, descriptions.get(code, "Continental")

    def _classify_polar(self, t_max: float) -> tuple:
        """Classify polar (E) climates."""
        if t_max >= 0:
            return "ET", "Tundra"
        else:
            return "EF", "Ice cap"

    def classify_grid(
        self,
        monthly_temp: xr.DataArray,
        monthly_precip: xr.DataArray,
    ) -> xr.DataArray:
        """Classify climate for each grid cell.

        Args:
            monthly_temp: Temperature with (time, lat, lon) dimensions.
                Must have 12 time steps (monthly climatology).
            monthly_precip: Precipitation with (time, lat, lon) dimensions.

        Returns:
            DataArray with Koppen-Geiger classification codes.
        """
        if monthly_temp.sizes.get("time", 0) != 12:
            raise ValueError("monthly_temp must have exactly 12 time steps")

        spatial_dims = [d for d in monthly_temp.dims if d != "time"]
        result_shape = [monthly_temp.sizes[d] for d in spatial_dims]
        codes = np.empty(result_shape, dtype="U4")

        for idx in np.ndindex(*result_shape):
            sel = {d: i for d, i in zip(spatial_dims, idx)}
            t_series = monthly_temp.isel(sel).values
            p_series = monthly_precip.isel(sel).values

            if np.any(np.isnan(t_series)) or np.any(np.isnan(p_series)):
                codes[idx] = ""
                continue

            result = self.koppen_geiger_classify(t_series, p_series)
            codes[idx] = result["code"]

        coords = {
            d: monthly_temp.coords[d] for d in spatial_dims if d in monthly_temp.coords
        }
        return xr.DataArray(
            codes, dims=spatial_dims, coords=coords, name="koppen_geiger"
        )
