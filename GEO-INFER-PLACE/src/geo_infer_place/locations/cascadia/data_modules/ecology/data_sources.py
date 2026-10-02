"""Data acquisition for Cascadia ecological analysis."""

from __future__ import annotations

from importlib.resources import files

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(str(files("geo_infer_place.locations.cascadia.config")))


def load_salmon_esu_data() -> dict[str, Any]:
    """Load salmon ESU/DPS data from NOAA NMFS YAML."""
    path = CONFIG_DIR / "cascadia_salmon_esus.yaml"
    if not path.exists():
        logger.warning("Salmon ESU data not found at %s", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def load_ecoregion_data() -> dict[str, Any]:
    """Load EPA Level III ecoregion data."""
    path = CONFIG_DIR / "cascadia_ecoregions.yaml"
    if not path.exists():
        logger.warning("Ecoregion data not found at %s", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def load_indigenous_territories() -> dict[str, Any]:
    """Load indigenous territorial data from BIA sources."""
    path = CONFIG_DIR / "cascadia_indigenous_territories.yaml"
    if not path.exists():
        logger.warning("Indigenous territories data not found at %s", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def load_climate_zones() -> dict[str, Any]:
    """Load NOAA climate zone data."""
    path = CONFIG_DIR / "cascadia_climate_zones.yaml"
    if not path.exists():
        logger.warning("Climate zones data not found at %s", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def get_esa_listed_salmon_esu_names(data: dict[str, Any]) -> list[str]:
    """Return names of all ESA-listed salmon ESUs/DPS."""
    listed: list[str] = []
    for species_group in [
        "chinook_salmon",
        "coho_salmon",
        "steelhead",
        "sockeye_salmon",
        "chum_salmon",
        "other_species",
    ]:
        for entry in data.get(species_group, []):
            status = entry.get("esa_status", "")
            if status not in ("Not Listed", "Not Listed (Species of Concern)"):
                listed.append(entry["name"])
    return listed
