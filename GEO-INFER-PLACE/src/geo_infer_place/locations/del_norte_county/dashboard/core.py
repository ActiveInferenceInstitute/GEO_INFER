"""
Core implementation of the AdvancedDashboard for Del Norte County.
"""

from __future__ import annotations

import importlib.resources
import folium
import folium.plugins
import h3
import json
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from geo_infer_place.utils.integration import DelNorteDataIntegrator
from .analyzers import ClimateAnalyzer, ZoningAnalyzer, AgroEconomicAnalyzer
from .config import LAYER_CONFIGS, DEFAULT_BOUNDS, DEFAULT_CENTER

logger = logging.getLogger(__name__)


class AdvancedDashboard:
    """
    Advanced Geospatial Intelligence Dashboard for Del Norte County.

    Comprehensive dashboard integrating multiple California state data sources
    for real-time analysis of climate, zoning, and agro-economic considerations.
    """

    def __init__(
        self,
        output_dir: str = "./del_norte_dashboard",
        api_keys: dict[str, str] | None = None,
        layer_config: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        # Use centralized data integrator
        self.data_integrator = DelNorteDataIntegrator()

        self.climate_analyzer = ClimateAnalyzer()
        self.zoning_analyzer = ZoningAnalyzer()
        self.agro_economic_analyzer = AgroEconomicAnalyzer()

        # Del Norte County geographic parameters
        self.county_center = DEFAULT_CENTER
        self.county_bounds = DEFAULT_BOUNDS

        self._load_location_config_bounds()

        # Layer configurations
        self.layer_configs = LAYER_CONFIGS

        # Dashboard state
        self.dashboard_data: dict[str, Any] = {}

        # Initialize layer groups
        self.layer_groups = {
            "fire": folium.FeatureGroup(name="🔥 Fire Incidents", show=True),
            "fire_perimeters": folium.FeatureGroup(
                name="🔥 Fire Perimeters", show=False
            ),
            "weather": folium.FeatureGroup(name="🌤️ Weather Data", show=True),
            "earthquake": folium.FeatureGroup(name="🌍 Earthquakes", show=True),
            "forest": folium.FeatureGroup(name="🌲 Forest Health", show=True),
            "climate": folium.FeatureGroup(name="🌡️ Climate Risks", show=False),
            "tides": folium.FeatureGroup(name="🌊 Tide Gauge", show=True),
            "zoning": folium.FeatureGroup(name="🏘️ Zoning", show=False),
            "conservation": folium.FeatureGroup(name="🌿 Conservation", show=True),
            "economic": folium.FeatureGroup(name="💼 Economics", show=False),
            "emergency": folium.FeatureGroup(name="🚨 Emergency Services", show=True),
            "health": folium.FeatureGroup(name="🏥 Public Health", show=False),
            "infrastructure": folium.FeatureGroup(name="🏗️ Infrastructure", show=False),
            "equity": folium.FeatureGroup(name="⚖️ Environmental Justice", show=False),
        }

    def _load_location_config_bounds(self) -> None:
        """Load bounds/center from the packaged Del Norte analysis_config.yaml."""
        try:
            # Explicit environment override wins; otherwise resolve the
            # packaged resource so installed wheels work from any cwd.
            env_override = os.environ.get("GEO_INFER_PLACE_DEL_NORTE_CONFIG")
            if env_override and Path(env_override).exists():
                self._apply_location_bounds(Path(env_override))
                return

            packaged = importlib.resources.files(
                "geo_infer_place.locations.del_norte_county"
            ).joinpath("config/analysis_config.yaml")
            if not packaged.is_file():
                logger.info("Packaged analysis_config.yaml not found")
                return
            with importlib.resources.as_file(packaged) as config_path:
                self._apply_location_bounds(config_path)
        except Exception as e:
            logger.warning(f"Could not load location bounds from config: {e}")

    def _apply_location_bounds(self, config_path: Path) -> None:
        """Apply location.bounds/center from an analysis_config.yaml file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        bounds = cfg.get("location", {}).get("bounds", {})
        if all(k in bounds for k in ["north", "south", "east", "west"]):
            self.county_bounds = bounds
            cy = (bounds["north"] + bounds["south"]) / 2.0
            cx = (bounds["east"] + bounds["west"]) / 2.0
            self.county_center = [cy, cx]
            logger.info("Loaded bounds from analysis_config.yaml")

    def fetch_real_time_data(self) -> dict[str, Any]:
        """Fetch real-time data from all configured sources using the shared integrator."""
        logger.info("Fetching real-time data from California sources...")

        # Map integrator outputs to dashboard data structure
        fire_incidents = self.data_integrator.calfire_client.get_active_incidents()
        fire_perimeters = self.data_integrator.calfire_client.get_fire_perimeters(
            start_year=2015
        )
        weather = self.data_integrator.noaa_client.get_weather_data()
        earthquakes = self.data_integrator.usgs_client.get_earthquakes()
        tides = self.data_integrator.noaa_client.get_tide_gauge_data()

        data: dict[str, Any] = {
            "fire_data": fire_incidents,
            "fire_perimeters": (
                {"success": True, "geojson": fire_perimeters}
                if fire_perimeters
                else {"success": False}
            ),
            "weather_data": weather,
            "earthquake_data": earthquakes,
            "tide_levels": {"success": True, **tides} if tides else {"success": False},
            "fetch_timestamp": datetime.now().isoformat(),
        }

        # Store in dashboard data
        self.dashboard_data.update(data)

        # Persist successfully fetched datasets
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if data.get("fire_perimeters", {}).get("success"):
                self._persist_json(
                    data["fire_perimeters"].get("geojson", {}),
                    f"fire_perimeters_{ts}.geojson",
                )
            if data.get("tide_levels", {}).get("success"):
                # Extract relevant parts for caching
                tide_cache = {
                    "observations": data["tide_levels"]
                    .get("series", {})
                    .get("9419750", {})
                    .get("data", []),
                    "latest": (
                        data["tide_levels"]
                        .get("series", {})
                        .get("9419750", {})
                        .get("data", [])[-1]
                        if data["tide_levels"]
                        .get("series", {})
                        .get("9419750", {})
                        .get("data")
                        else None
                    ),
                }
                self._persist_json(tide_cache, f"tide_levels_{ts}.json")
        except Exception as e:
            logger.warning(f"Failed to persist fetched datasets: {e}")

        return data

    def _persist_json(self, obj: dict[str, Any], filename: str) -> None:
        """Persist a JSON object into the dashboard output directory."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Saved dataset to {filepath}")

    def load_cached_data(self) -> None:
        """Load most recent cached datasets from output_dir into dashboard_data."""
        try:
            perims = sorted(self.output_dir.glob("fire_perimeters_*.geojson"))
            if perims:
                with open(perims[-1], encoding="utf-8") as f:
                    geojson = json.load(f)
                self.dashboard_data["fire_perimeters"] = {
                    "success": True,
                    "geojson": geojson,
                }

            tides = sorted(self.output_dir.glob("tide_levels_*.json"))
            if tides:
                with open(tides[-1], encoding="utf-8") as f:
                    tid = json.load(f)
                # Reconstruct tide data structure expected by dashboard
                self.dashboard_data["tide_levels"] = {
                    "success": True,
                    "series": {
                        "9419750": {
                            "data": tid.get("observations", []),
                        }
                    },
                    "latest": tid.get("latest"),
                }
        except Exception as e:
            logger.warning(f"Failed loading cached datasets: {e}")

    def generate_analysis_panels(self) -> dict[str, str]:
        """Generate HTML panels for different analysis components."""
        panels = {}

        # Climate Analysis Panel
        try:
            climate_data = self.climate_analyzer.generate_climate_projections()
            climate_risks = self.climate_analyzer.calculate_climate_risks()
            panels["climate"] = self._create_climate_panel(climate_data, climate_risks)
        except (RuntimeError, ValueError) as exc:
            panels["climate"] = (
                f"<div class='data-unavailable'>Climate data unavailable: {exc}</div>"
            )

        # Zoning Analysis Panel
        try:
            zoning_data = self.zoning_analyzer.generate_zoning_analysis()
            panels["zoning"] = self._create_zoning_panel(zoning_data)
        except (RuntimeError, ValueError) as exc:
            panels["zoning"] = (
                f"<div class='data-unavailable'>Zoning data unavailable: {exc}</div>"
            )

        # Economic Analysis Panel
        try:
            economic_data = self.agro_economic_analyzer.generate_economic_analysis()
            panels["economic"] = self._create_economic_panel(economic_data)
        except (RuntimeError, ValueError) as exc:
            panels["economic"] = (
                f"<div class='data-unavailable'>Economic data unavailable: {exc}</div>"
            )

        return panels

    def _create_climate_panel(self, climate_data: dict, risks: dict) -> str:
        """Create climate analysis panel HTML."""
        risk_items = ""
        for risk_name, risk_value in risks.items():
            risk_level = (
                "High" if risk_value > 0.5 else "Medium" if risk_value > 0.3 else "Low"
            )
            color = (
                "#ff4444"
                if risk_value > 0.5
                else "#ffaa44"
                if risk_value > 0.3
                else "#44ff44"
            )
            risk_items += f'<div style="margin: 5px 0; padding: 8px; background: {color}20; border-left: 4px solid {color};"><strong>{risk_name.replace("_", " ").title()}:</strong> {risk_level} ({risk_value:.2f})</div>'

        return f'<div style="background: white; padding: 15px; border-radius: 8px; margin: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><h3 style="color: #2c3e50; margin-top: 0;">🌡️ Climate Analysis</h3><div style="margin: 10px 0;"><h4>Climate Risk Assessment</h4>{risk_items}</div><div style="margin: 10px 0;"><h4>Key Insights</h4><ul style="list-style-type: none; padding: 0;"><li>🔥 Fire weather risk is elevated in summer months</li><li>🌊 Coastal flooding risk increasing with sea level rise</li><li>🌡️ Temperature increases affecting forest ecosystems</li><li>💧 Drought risk requires enhanced water management</li></ul></div></div>'

    def _create_zoning_panel(self, zoning_data: dict) -> str:
        """Create zoning analysis panel HTML."""
        zoning_items = ""
        for zone, data in zoning_data["zoning_breakdown"].items():
            zoning_items += f'<div style="margin: 5px 0; padding: 8px; background: {data["color"]}20; border-left: 4px solid {data["color"]};"><strong>{zone.replace("_", " ").title()}:</strong> {data["acres"]:,} acres ({data["percentage"]}%)</div>'

        return f'<div style="background: white; padding: 15px; border-radius: 8px; margin: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><h3 style="color: #2c3e50; margin-top: 0;">🏘️ Zoning & Land Use</h3><div style="margin: 10px 0;"><h4>Land Use Distribution</h4>{zoning_items}</div><div style="margin: 10px 0;"><h4>Development Insights</h4><ul style="list-style-type: none; padding: 0;"><li>🌲 {zoning_data["zoning_breakdown"]["forest_conservation"]["percentage"]}% in forest conservation</li><li>🚜 Agricultural areas support rural economy</li><li>🏠 Limited residential development pressure</li><li>🛡️ Strong environmental protections in place</li></ul></div></div>'

    def _create_economic_panel(self, economic_data: dict) -> str:
        """Create economic analysis panel HTML."""
        sector_items = ""
        for sector, data in economic_data["sector_analysis"].items():
            trend_icon = (
                "📈"
                if data["growth_trend"] > 0
                else "📉"
                if data["growth_trend"] < 0
                else "➡️"
            )
            sector_items += f'<div style="margin: 5px 0; padding: 8px; background: #f8f9fa; border-left: 4px solid #3498db;"><strong>{sector.replace("_", " ").title()}:</strong> {data["employment"]} jobs ({data["employment_share"]}%) {trend_icon}</div>'

        return f'<div style="background: white; padding: 15px; border-radius: 8px; margin: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><h3 style="color: #2c3e50; margin-top: 0;">💼 Economic Analysis</h3><div style="margin: 10px 0;"><h4>Employment by Sector ({economic_data["total_employment"]:,} total jobs)</h4>{sector_items}</div><div style="margin: 10px 0;"><h4>Economic Health Indicators</h4><ul style="list-style-type: none; padding: 0;"><li>📊 Economic Diversity Index: {economic_data["economic_diversity_index"]}</li><li>💰 Total Revenue: ${economic_data["total_revenue"]:,}</li><li>🌾 Agricultural Productivity: Growing</li><li>🏗️ Infrastructure Investment Needed</li></ul></div></div>'

    def create_comprehensive_map(self) -> folium.Map:
        """Create comprehensive interactive map with all layers and controls."""
        m = folium.Map(location=self.county_center, zoom_start=10, tiles=None)

        folium.TileLayer(
            "OpenStreetMap", name="Street Map", attr="OpenStreetMap"
        ).add_to(m)
        folium.TileLayer(
            tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
            attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL",
            name="Terrain",
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri",
            name="Satellite",
            overlay=False,
            control=True,
        ).add_to(m)

        self._add_county_boundary(m)

        if self.dashboard_data:
            self._add_fire_incidents_layer(m)
            self._add_fire_perimeters_layer(m)
            self._add_weather_layer(m)
            self._add_earthquake_layer(m)
            self._add_tide_gauge_layer(m)

        self._add_h3_forest_health_layer(m)
        self._add_climate_risk_zones(m)
        self._add_zoning_overlay(m)
        self._add_conservation_areas(m)
        self._add_economic_indicators(m)
        self._add_emergency_services_layer(m)
        self._add_public_health_layer(m)
        self._add_infrastructure_layer(m)
        self._add_environmental_justice_layer(m)

        for group in self.layer_groups.values():
            group.add_to(m)

        self._add_layer_controls(m)
        self._add_measurement_tools(m)
        self._add_drawing_tools(m)
        self._add_custom_controls(m)

        return m

    def _add_county_boundary(self, m: folium.Map) -> None:
        boundary_coords = [
            [self.county_bounds["north"], self.county_bounds["west"]],
            [self.county_bounds["north"], self.county_bounds["east"]],
            [self.county_bounds["south"], self.county_bounds["east"]],
            [self.county_bounds["south"], self.county_bounds["west"]],
            [self.county_bounds["north"], self.county_bounds["west"]],
        ]
        folium.Polygon(
            locations=boundary_coords,
            popup="<b>Del Norte County</b><br>Area: 1,008 sq mi",
            color="red",
            weight=3,
            fill=False,
        ).add_to(m)

    def _add_fire_incidents_layer(self, m: folium.Map) -> None:
        if (
            "fire_data" in self.dashboard_data
            and self.dashboard_data["fire_data"]["success"]
        ):
            for incident in self.dashboard_data["fire_data"]["incidents"]:
                if incident["lat"] and incident["lon"]:
                    color = (
                        "red"
                        if incident["contained"] < 50
                        else "orange"
                        if incident["contained"] < 100
                        else "green"
                    )
                    folium.Marker(
                        location=[incident["lat"], incident["lon"]],
                        popup=f"<b>{incident['name']}</b><br>Acres: {incident['acres']}<br>Contained: {incident['contained']}%",
                        icon=folium.Icon(color=color, icon="fire"),
                    ).add_to(self.layer_groups["fire"])

    def _add_fire_perimeters_layer(self, m: folium.Map) -> None:
        if "fire_perimeters" in self.dashboard_data and self.dashboard_data[
            "fire_perimeters"
        ].get("success"):
            geojson = self.dashboard_data["fire_perimeters"].get("geojson", {})
            if geojson.get("features"):
                folium.GeoJson(
                    data=geojson,
                    name="Fire Perimeters",
                    style_function=lambda x: {
                        "color": "#d73027",
                        "weight": 2,
                        "fillOpacity": 0.05,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["fire_name", "fire_year"], aliases=["Name", "Year"]
                    ),
                ).add_to(self.layer_groups["fire_perimeters"])

    def _add_weather_layer(self, m: folium.Map) -> None:
        if (
            "weather_data" in self.dashboard_data
            and self.dashboard_data["weather_data"]["success"]
        ):
            w = self.dashboard_data["weather_data"]
            folium.Marker(
                location=[41.7450, -124.1840],
                popup=f"<b>Weather (KCEC)</b><br>Temp: {w.get('temperature')} C<br>Wind: {w.get('wind_speed')} km/h",
                icon=folium.Icon(color="blue", icon="cloud"),
            ).add_to(self.layer_groups["weather"])

    def _add_earthquake_layer(self, m: folium.Map) -> None:
        if (
            "earthquake_data" in self.dashboard_data
            and self.dashboard_data["earthquake_data"]["success"]
        ):
            for eq in self.dashboard_data["earthquake_data"]["earthquakes"]:
                if eq.get("magnitude"):
                    radius = max(5, eq["magnitude"] * 3)
                    color = "red" if eq["magnitude"] >= 4 else "orange"
                    folium.CircleMarker(
                        location=[eq["lat"], eq["lon"]],
                        radius=radius,
                        popup=f"M{eq['magnitude']} - {eq['place']}",
                        color=color,
                        fill=True,
                        fillColor=color,
                    ).add_to(self.layer_groups["earthquake"])

    def _add_tide_gauge_layer(self, m: folium.Map) -> None:
        tide = self.dashboard_data.get("tide_levels", {})
        latest = tide.get("latest")
        if latest:
            level = latest.get("v", latest.get("water_level"))
            ts = latest.get("t", latest.get("time"))
            folium.Marker(
                location=[41.7450, -124.2370],
                popup=f"<b>Tide Gauge</b><br>Level: {level} m<br>Time: {ts}",
                icon=folium.Icon(color="blue", icon="tint"),
            ).add_to(self.layer_groups["tides"])

    def _add_h3_forest_health_layer(self, m: folium.Map) -> None:
        forest_data = self.dashboard_data.get("forest_health", {})
        cells = forest_data.get("spatial_data", {}).get("h3_cells", {})
        for h3_cell, cell_data in cells.items():
            if not isinstance(cell_data, dict) or "health_index" not in cell_data:
                continue
            try:
                h3_boundary = h3.cell_to_boundary(h3_cell)
                health = float(cell_data["health_index"])
            except (TypeError, ValueError, KeyError):
                continue
            color = "green" if health > 0.7 else "orange" if health > 0.5 else "red"
            folium.Polygon(
                locations=[[lat, lon] for lat, lon in h3_boundary],
                popup=f"H3: {h3_cell}<br>Health: {health:.2f}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.4,
            ).add_to(self.layer_groups["forest"])

    def _add_climate_risk_zones(self, m: folium.Map) -> None:
        zones: list[dict[str, Any]] = [
            {
                "name": "High Fire Risk",
                "bounds": [
                    [41.9, -124.3],
                    [42.0, -123.8],
                    [41.8, -123.7],
                    [41.7, -124.2],
                ],
                "color": "red",
            },
            {
                "name": "Coastal Flood",
                "bounds": [
                    [41.85, -124.4],
                    [41.95, -124.3],
                    [41.75, -124.2],
                    [41.65, -124.35],
                ],
                "color": "blue",
            },
        ]
        for z in zones:
            folium.Polygon(
                locations=z["bounds"], popup=z["name"], color=z["color"], fill=True
            ).add_to(self.layer_groups["climate"])

    def _add_zoning_overlay(self, m: folium.Map) -> None:
        areas: list[dict[str, Any]] = [
            {
                "name": "Conservation",
                "bounds": [
                    [41.8, -124.2],
                    [42.0, -123.7],
                    [41.9, -123.6],
                    [41.7, -124.1],
                ],
                "color": "#228B22",
            },
        ]
        for a in areas:
            folium.Polygon(
                locations=a["bounds"], popup=a["name"], color=a["color"], fill=True
            ).add_to(self.layer_groups["zoning"])

    def _add_conservation_areas(self, m: folium.Map) -> None:
        # Redwood Parks
        folium.Polygon(
            locations=[[41.3, -124.1], [41.5, -124.0], [41.4, -123.9], [41.2, -124.0]],
            popup="Redwood National and State Parks",
            color="darkgreen",
            fill=True,
        ).add_to(self.layer_groups["conservation"])

    def _add_economic_indicators(self, m: folium.Map) -> None:
        centers: list[dict[str, Any]] = [
            {"loc": [41.7558, -124.2026], "name": "Crescent City", "emp": 3500}
        ]
        for c in centers:
            folium.CircleMarker(
                location=c["loc"],
                radius=15,
                popup=f"{c['name']}: {c['emp']} jobs",
                color="gold",
                fill=True,
            ).add_to(self.layer_groups["economic"])

    def _add_emergency_services_layer(self, m: folium.Map) -> None:
        facilities: list[dict[str, Any]] = [
            {
                "loc": [41.7586, -124.2031],
                "name": "Sutter Coast Hospital",
                "icon": "plus",
                "color": "green",
            }
        ]
        for f in facilities:
            folium.Marker(
                location=f["loc"],
                popup=f["name"],
                icon=folium.Icon(color=f["color"], icon=f["icon"]),
            ).add_to(self.layer_groups["emergency"])

    def _add_public_health_layer(self, m: folium.Map) -> None:
        folium.Marker(
            location=[41.7520, -124.2010],
            popup="Community Health Center",
            icon=folium.Icon(color="lightgreen", icon="heart"),
        ).add_to(self.layer_groups["health"])

    def _add_infrastructure_layer(self, m: folium.Map) -> None:
        folium.Marker(
            location=[41.7450, -124.2370],
            popup="Crescent City Harbor",
            icon=folium.Icon(color="darkblue", icon="anchor"),
        ).add_to(self.layer_groups["infrastructure"])

    def _add_environmental_justice_layer(self, m: folium.Map) -> None:
        folium.Polygon(
            locations=[[41.80, -124.15], [41.82, -124.12], [41.79, -124.11]],
            popup="Tribal Lands Impact",
            color="purple",
            fill=True,
        ).add_to(self.layer_groups["equity"])

    def _add_layer_controls(self, m: folium.Map) -> None:
        folium.LayerControl().add_to(m)

    def _add_measurement_tools(self, m: folium.Map) -> None:
        folium.plugins.MeasureControl().add_to(m)

    def _add_drawing_tools(self, m: folium.Map) -> None:
        folium.plugins.Draw(export=True).add_to(m)

    def _add_custom_controls(self, m: folium.Map) -> None:
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.LocateControl().add_to(m)
        m.add_child(folium.plugins.MiniMap(toggle_display=True))

    def generate_dashboard_html(self, fetch_data: bool = True) -> str:
        """Generate complete dashboard HTML with panels and map."""
        if fetch_data:
            real_time_data = self.fetch_real_time_data()
        else:
            real_time_data = self.dashboard_data

        panels = self.generate_analysis_panels()
        map_obj = self.create_comprehensive_map()
        map_html = map_obj._repr_html_()

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Del Norte County Geospatial Intelligence Dashboard</title>
            <style>
                body {{ margin: 0; padding: 0; font-family: sans-serif; background-color: #f5f5f5; }}
                .dashboard-header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 20px; text-align: center; }}
                .dashboard-container {{ display: flex; height: calc(100vh - 120px); }}
                .sidebar {{ width: 400px; background: white; overflow-y: auto; box-shadow: 2px 0 10px rgba(0,0,0,0.1); z-index: 1000; }}
                .map-container {{ flex: 1; position: relative; }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>🗺️ Del Norte County Geospatial Intelligence Dashboard</h1>
                <p>Climate • Zoning • Agro-Economics • Policy Support Interface</p>
                <div>
                     <span style="color: {"#27ae60" if real_time_data.get("fire_data", {}).get("success") else "#e74c3c"}">Fire Data</span> •
                     <span style="color: {"#27ae60" if real_time_data.get("weather_data", {}).get("success") else "#e74c3c"}">Weather Data</span>
                </div>
            </div>
            <div class="dashboard-container">
                <div class="sidebar">
                    {panels.get("climate", "")}
                    {panels.get("zoning", "")}
                    {panels.get("economic", "")}
                </div>
                <div class="map-container">
                    {map_html}
                </div>
            </div>
        </body>
        </html>
        """

    def generate_dashboard(
        self, filename: str | None = None, fetch_data: bool = False
    ) -> str:
        """Alias for save_dashboard() — generates HTML without fetching live data by default."""
        return self.save_dashboard(filename=filename, fetch_data=fetch_data)

    def save_dashboard(
        self, filename: str | None = None, fetch_data: bool = True
    ) -> str:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"del_norte_intelligence_dashboard_{ts}.html"

        html_content = self.generate_dashboard_html(fetch_data=fetch_data)

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return str(output_path)
