"""
Configuration for the Del Norte Dashboard.
"""

from dataclasses import dataclass


@dataclass
class LayerConfig:
    """Configuration for map layers."""

    name: str
    type: str  # 'marker', 'polygon', 'heatmap', 'choropleth', 'raster'
    enabled: bool = True
    color: str = "blue"
    opacity: float = 0.7
    data_source: str | None = None
    update_frequency: str | None = None


# Default bounds for Del Norte County
DEFAULT_BOUNDS = {"north": 42.006, "south": 41.458, "east": -123.536, "west": -124.408}

DEFAULT_CENTER = [41.75, -124.0]

LAYER_CONFIGS = {
    "fire_incidents": LayerConfig("Fire Incidents", "marker", True, "red"),
    "fire_perimeters": LayerConfig("Fire Perimeters", "polygon", False, "red"),
    "weather_stations": LayerConfig("Weather Stations", "marker", True, "blue"),
    "earthquake_activity": LayerConfig(
        "Earthquake Activity", "marker", False, "orange"
    ),
    "h3_forest_health": LayerConfig("Forest Health (H3)", "polygon", True, "green"),
    "climate_risk_zones": LayerConfig(
        "Climate Risk Zones", "choropleth", True, "purple"
    ),
    "tide_levels": LayerConfig("Tide Gauge", "marker", True, "blue"),
    "zoning_overlay": LayerConfig("Zoning Overlay", "polygon", False, "gray"),
    "agricultural_areas": LayerConfig("Agricultural Areas", "polygon", False, "yellow"),
    "conservation_areas": LayerConfig(
        "Conservation Areas", "polygon", True, "darkgreen"
    ),
    "coastal_vulnerability": LayerConfig(
        "Coastal Vulnerability", "heatmap", False, "cyan"
    ),
    "economic_indicators": LayerConfig(
        "Economic Indicators", "choropleth", False, "gold"
    ),
}
