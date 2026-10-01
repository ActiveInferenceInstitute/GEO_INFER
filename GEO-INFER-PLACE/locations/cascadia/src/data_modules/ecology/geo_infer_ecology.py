"""Cascadia ecological analysis module.

Overlays salmon ESU coverage, EPA ecoregions, spotted owl habitat,
and indigenous territories onto H3 hexagon grids.
"""

from __future__ import annotations

import logging
from typing import Any

from .data_sources import (
    get_esa_listed_salmon_esu_names,
    load_climate_zones,
    load_ecoregion_data,
    load_indigenous_territories,
    load_salmon_esu_data,
)

logger = logging.getLogger(__name__)

# Approximate watershed bounding boxes for ESU presence detection
# Maps watershed name -> (west, south, east, north)
_WATERSHED_BBOX: dict[str, tuple[float, float, float, float]] = {
    "Columbia River": (-124.1, 40.5, -114.5, 49.0),
    "Willamette River": (-123.2, 43.8, -121.5, 45.8),
    "Klamath River": (-124.1, 41.4, -121.0, 43.0),
    "Puget Sound": (-123.5, 46.9, -120.5, 48.8),
    "Fraser River": (-124.0, 49.0, -117.0, 54.0),
    "Snake River": (-117.5, 42.0, -111.0, 47.0),
}

# EPA Level III ecoregion bounding boxes (approximate)
_ECOREGION_BBOX: dict[str, tuple[float, float, float, float]] = {
    "Coast Range": (-124.8, 40.0, -122.5, 49.0),
    "Puget Lowland": (-123.0, 46.8, -121.5, 48.5),
    "Willamette Valley": (-123.5, 43.5, -121.5, 46.0),
    "Cascades": (-122.5, 40.5, -120.5, 49.0),
    "Eastern Cascades Slopes and Foothills": (-121.5, 42.0, -119.5, 48.5),
    "Columbia Plateau": (-120.0, 42.0, -116.0, 49.0),
    "Klamath Mountains": (-124.0, 40.5, -121.0, 42.5),
    "Great Valley (northern extent)": (-122.5, 39.5, -120.0, 41.0),
}


class GeoInferEcology:
    """Ecological analysis module for Cascadia bioregion.

    Integrates salmon ESU data (NOAA NMFS), EPA ecoregions, climate zones,
    and indigenous territory overlays with H3 hexagonal grids.
    """

    def __init__(self) -> None:
        self._salmon_data: dict[str, Any] = {}
        self._ecoregion_data: dict[str, Any] = {}
        self._indigenous_data: dict[str, Any] = {}
        self._climate_data: dict[str, Any] = {}
        self._esa_listed_names: list[str] = []

    def acquire_raw_data(self) -> dict[str, Any]:
        """Load all ecological data sources from config YAML files."""
        self._salmon_data = load_salmon_esu_data()
        self._ecoregion_data = load_ecoregion_data()
        self._indigenous_data = load_indigenous_territories()
        self._climate_data = load_climate_zones()
        self._esa_listed_names = get_esa_listed_salmon_esu_names(self._salmon_data)

        logger.info(
            "Ecology module loaded: %d ESA-listed salmon ESUs, %d ecoregions",
            len(self._esa_listed_names),
            len(self._ecoregion_data.get("ecoregions", [])),
        )
        return {
            "salmon_esu_count": len(self._esa_listed_names),
            "ecoregion_count": len(self._ecoregion_data.get("ecoregions", [])),
            "tribal_nation_count": self._count_tribal_nations(),
        }

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Overlay ecological data onto H3 hexagon grid.

        Args:
            h3_data: Dict mapping H3 cell IDs to property dicts.
                     Each cell dict should contain 'lat' and 'lon' keys.

        Returns:
            Dict mapping H3 cell IDs to enriched property dicts with
            added ecological metrics.
        """
        results: dict[str, Any] = {}
        for cell_id, props in h3_data.items():
            lat = props.get("lat", 0.0)
            lon = props.get("lon", 0.0)
            results[cell_id] = {
                **props,
                "salmon_esu_count": self._salmon_esu_count_at(lat, lon),
                "esa_listed_salmon_present": self._esa_salmon_present(lat, lon),
                "ecoregion_code": self._ecoregion_at(lat, lon),
                "old_growth_probability": self._old_growth_probability(lat, lon),
                "spotted_owl_habitat": self._spotted_owl_habitat(lat, lon),
                "indigenous_territory_overlap": self._indigenous_overlap(lat, lon),
            }
        return results

    # -- Internal spatial helpers -------------------------------------------

    def _point_in_bbox(
        self, lat: float, lon: float, bbox: tuple[float, float, float, float]
    ) -> bool:
        west, south, east, north = bbox
        return south <= lat <= north and west <= lon <= east

    def _salmon_esu_count_at(self, lat: float, lon: float) -> int:
        """Estimate number of salmon ESUs present at this location."""
        count = 0
        for bbox in _WATERSHED_BBOX.values():
            if self._point_in_bbox(lat, lon, bbox):
                count += 1
        return count

    def _esa_salmon_present(self, lat: float, lon: float) -> bool:
        """True if location is within any major salmon-bearing watershed."""
        return self._salmon_esu_count_at(lat, lon) > 0

    def _ecoregion_at(self, lat: float, lon: float) -> str:
        """Return EPA Level III ecoregion name for location."""
        for name, bbox in _ECOREGION_BBOX.items():
            if self._point_in_bbox(lat, lon, bbox):
                return name
        return "Unknown"

    def _old_growth_probability(self, lat: float, lon: float) -> float:
        """Estimate old-growth forest probability (0-1) based on ecoregion and elevation proxy."""
        ecoregion = self._ecoregion_at(lat, lon)
        # Coast Range and Klamath Mountains have highest old-growth probability
        if ecoregion in ("Coast Range", "Klamath Mountains"):
            return 0.35
        if ecoregion == "Cascades":
            return 0.25
        if ecoregion in ("Eastern Cascades Slopes and Foothills",):
            return 0.10
        return 0.05

    def _spotted_owl_habitat(self, lat: float, lon: float) -> bool:
        """True if location falls within Northern Spotted Owl critical habitat range."""
        # NSO critical habitat: Coast Range and Cascades west of crest
        ecoregion = self._ecoregion_at(lat, lon)
        return ecoregion in ("Coast Range", "Klamath Mountains", "Cascades", "Puget Lowland")

    def _indigenous_overlap(self, lat: float, lon: float) -> list[str]:
        """Return names of tribal nations with territories near this location."""
        nearby: list[str] = []
        radius = 0.5  # ~55km approximate radius for territory overlap check

        for state_key in ("washington_state", "oregon_state", "california_tribes"):
            for nation in self._indigenous_data.get(state_key, []):
                t_lat = nation.get("latitude", 0.0)
                t_lon = nation.get("longitude", 0.0)
                if abs(lat - t_lat) <= radius and abs(lon - t_lon) <= radius:
                    nearby.append(nation["name"])
        return nearby

    def _count_tribal_nations(self) -> int:
        total = 0
        for key in ("washington_state", "oregon_state", "california_tribes"):
            total += len(self._indigenous_data.get(key, []))
        return total
