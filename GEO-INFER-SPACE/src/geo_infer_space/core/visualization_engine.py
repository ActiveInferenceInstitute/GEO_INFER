"""
InteractiveVisualizationEngine: Interactive geospatial dashboard creation.

This module provides comprehensive visualization capabilities for place-based
analysis, including interactive maps with H3 integration, multi-layer overlays,
and dashboard generation adapted from the climate integration example.
"""

import logging
import hashlib
import json
import folium
import h3
import geopandas as gpd
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast
from folium.plugins import MarkerCluster
import numpy as np

logger = logging.getLogger(__name__)


class InteractiveVisualizationEngine:
    """
    Interactive visualization engine for comprehensive place-based dashboards.

    Adapted from the climate integration spatial microbiome example, this engine
    creates sophisticated interactive geospatial visualizations with H3 integration,
    multi-layer overlays, and real-time data integration for place-based analysis.

    Features:
    - H3 hexagonal spatial aggregation and visualization
    - Multi-layer interactive maps with toggle controls
    - Real-time data integration and visualization
    - Professional dashboard generation
    - Clustering and spatial analysis visualization
    - Export capabilities for sharing and embedding

    Example Usage:
        >>> engine = InteractiveVisualizationEngine(location_config, output_dir)
        >>> dashboard = engine.create_comprehensive_dashboard(analysis_results)
        >>> forest_map = engine.create_forest_health_visualization(forest_data)
        >>> coastal_map = engine.create_coastal_resilience_visualization(coastal_data)
    """

    def __init__(
        self, location_config: dict[str, Any], output_dir: Path, h3_resolution: int = 8
    ) -> None:
        """
        Initialize visualization engine.

        Args:
            location_config: Configuration for the location
            output_dir: Output directory for generated visualizations
            h3_resolution: H3 spatial resolution for aggregation
        """
        if not isinstance(location_config, dict):
            raise TypeError("location_config must be a mapping")
        if not isinstance(h3_resolution, int) or not 0 <= h3_resolution <= 15:
            raise ValueError("h3_resolution must be an integer between 0 and 15")
        self.location_config = location_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h3_resolution = h3_resolution

        # Get location center and bounds
        self.location_bounds = location_config.get("location", {}).get("bounds", {})
        required_bounds = ("north", "south", "east", "west")
        if any(key not in self.location_bounds for key in required_bounds):
            raise ValueError(f"location bounds must contain {required_bounds}")
        bounds = {key: float(self.location_bounds[key]) for key in required_bounds}
        if not all(np.isfinite(value) for value in bounds.values()):
            raise ValueError("location bounds must contain finite values")
        if not -90 <= bounds["south"] < bounds["north"] <= 90:
            raise ValueError(
                "location latitude bounds must satisfy -90 <= south < north <= 90"
            )
        if not -180 <= bounds["west"] < bounds["east"] <= 180:
            raise ValueError(
                "location longitude bounds must satisfy -180 <= west < east <= 180"
            )
        self.location_bounds = bounds
        self.center_lat = (
            self.location_bounds.get("north", 42)
            + self.location_bounds.get("south", 41)
        ) / 2
        self.center_lon = (
            self.location_bounds.get("east", -123)
            + self.location_bounds.get("west", -125)
        ) / 2

        logger.info("InteractiveVisualizationEngine initialized")
        logger.info(f"Location center: ({self.center_lat:.3f}, {self.center_lon:.3f})")
        logger.info(f"H3 resolution: {self.h3_resolution}")

    def create_comprehensive_dashboard(
        self, analysis_results: dict[str, Any], dashboard_config: dict | None = None
    ) -> str:
        """
        Create comprehensive interactive dashboard with all analysis results.

        Args:
            analysis_results: Results from comprehensive analysis
            dashboard_config: Optional dashboard configuration

        Returns:
            Path to generated dashboard HTML file
        """
        if not isinstance(analysis_results, dict):
            raise TypeError("analysis_results must be a mapping")
        dashboard_config = dashboard_config or {}
        if not isinstance(dashboard_config, dict):
            raise TypeError("dashboard_config must be a mapping")
        zoom_start = dashboard_config.get("zoom_start", 10)
        if not isinstance(zoom_start, (int, float)) or not np.isfinite(zoom_start):
            raise ValueError("dashboard_config.zoom_start must be finite")
        tiles = dashboard_config.get("tiles", "CartoDB positron")
        if not isinstance(tiles, str) or not tiles:
            raise ValueError("dashboard_config.tiles must be a non-empty string")
        generated_at = dashboard_config.get("generated_at")
        if generated_at is not None and not isinstance(generated_at, str):
            raise ValueError("dashboard_config.generated_at must be a string")
        output_name = dashboard_config.get("output_name")
        if output_name is not None:
            if not isinstance(output_name, str) or not output_name.endswith(".html"):
                raise ValueError(
                    "dashboard_config.output_name must be an .html filename"
                )
            if Path(output_name).name != output_name:
                raise ValueError(
                    "dashboard_config.output_name must not contain directories"
                )
        logger.info("🎨 Creating comprehensive interactive dashboard...")

        # Create base map with professional styling
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=cast(int, zoom_start),
            tiles=tiles,
            attr="© CartoDB, © OpenStreetMap contributors",
        )

        # Add title
        title_html = self._create_dashboard_title(generated_at=generated_at)
        m.get_root().html.add_child(  # type: ignore[attr-defined]
            folium.Element(title_html)
        )

        # Create layer groups for different analysis domains
        layer_groups = self._create_layer_groups()

        # Add domain-specific visualizations
        domain_results = analysis_results.get("domain_results", {})

        if "forest_health" in domain_results:
            self._add_forest_health_layers(
                m, layer_groups, domain_results["forest_health"]
            )

        if "coastal_resilience" in domain_results:
            self._add_coastal_resilience_layers(
                m, layer_groups, domain_results["coastal_resilience"]
            )

        if "fire_risk" in domain_results:
            self._add_fire_risk_layers(m, layer_groups, domain_results["fire_risk"])

        if "community_development" in domain_results:
            self._add_community_development_layers(
                m, layer_groups, domain_results["community_development"]
            )

        # Add integrated results if available
        integrated_results = analysis_results.get("integrated_results", {})
        if integrated_results:
            self._add_integration_layers(m, layer_groups, integrated_results)

        # Add layer control
        for group in layer_groups.values():
            group.add_to(m)
        folium.LayerControl().add_to(m)

        # Save dashboard
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        dashboard_path = self.output_dir / (
            output_name or f"comprehensive_dashboard_{timestamp}.html"
        )
        m.save(str(dashboard_path))

        if dashboard_config.get("write_manifest", True):
            input_digest = hashlib.sha256(
                json.dumps(analysis_results, sort_keys=True, default=str).encode()
            ).hexdigest()
            manifest = {
                "schema_version": "geo-infer-space-visualization/v1",
                "generated_at": generated_at or datetime.now(UTC).isoformat(),
                "input_sha256": input_digest,
                "h3_version": h3.__version__,
                "artifacts": [
                    {
                        "path": dashboard_path.name,
                        "bytes": dashboard_path.stat().st_size,
                    }
                ],
                "accessibility": {
                    "nonempty_html": dashboard_path.stat().st_size > 0,
                    "has_title": "GEO-INFER Place-Based Analysis"
                    in dashboard_path.read_text(encoding="utf-8"),
                },
            }
            dashboard_path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

        logger.info(f"✅ Comprehensive dashboard saved to: {dashboard_path}")
        return str(dashboard_path)

    def create_base_map(self) -> folium.Map:
        """Create a basic folium map for testing."""
        return folium.Map(location=[self.center_lat, self.center_lon], zoom_start=10)

    def _create_dashboard_title(self, generated_at: str | None = None) -> str:
        """Create professional dashboard title."""
        location_name = self.location_config.get("location", {}).get("name", "Location")
        rendered_at = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

        title_html = f"""
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 80px; 
                    background-color: white; border: 2px solid grey; z-index:9999; 
                    font-size: 14px; color: black; font-weight: bold;
                    padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <div style="font-size: 18px; margin-bottom: 5px; color: #2E8B57;">
                🗺️ GEO-INFER Place-Based Analysis
            </div>
            <div style="font-size: 14px; color: #4682B4;">
                📍 {location_name}
            </div>
            <div style="font-size: 11px; color: #666; margin-top: 5px;">
                Interactive Geospatial Dashboard • {rendered_at}
            </div>
        </div>
        """
        return title_html

    def _create_layer_groups(self) -> dict[str, folium.FeatureGroup]:
        """Create layer groups for different analysis domains."""
        layer_groups = {
            "h3_grid": folium.FeatureGroup(name="🔷 H3 Spatial Grid", show=True),
            "forest_health": folium.FeatureGroup(name="🌲 Forest Health", show=True),
            "coastal_resilience": folium.FeatureGroup(
                name="🌊 Coastal Resilience", show=True
            ),
            "fire_risk": folium.FeatureGroup(name="🔥 Fire Risk", show=True),
            "community_development": folium.FeatureGroup(
                name="🏘️ Community Development", show=True
            ),
            "integration": folium.FeatureGroup(
                name="🔗 Cross-Domain Integration", show=False
            ),
        }
        return layer_groups

    def _add_forest_health_layers(
        self, m: folium.Map, layer_groups: dict, forest_data: dict[str, Any]
    ) -> None:
        """Add forest health visualization layers."""
        logger.info("Adding forest health visualization layers...")

        # Create marker cluster for forest monitoring sites
        forest_cluster = MarkerCluster(name="Forest Monitoring Sites")

        # Render only observations supplied by the analysis or configured data source.
        monitoring_sites = self._generate_forest_monitoring_sites(forest_data)

        for site in monitoring_sites:
            # Color code by health status
            if site["health_index"] > 0.7:
                marker_color = "green"
                icon = "leaf"
            elif site["health_index"] > 0.4:
                marker_color = "orange"
                icon = "exclamation-triangle"
            else:
                marker_color = "red"
                icon = "warning"

            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #228B22; margin: 0 0 8px 0;">🌲 Forest Health Site</h4>
                <table style="font-size: 11px; width: 100%;">
                    <tr><td><b>Site ID:</b></td><td>{site["site_id"]}</td></tr>
                    <tr><td><b>Health Index:</b></td><td>{site["health_index"]:.2f}</td></tr>
                    <tr><td><b>NDVI:</b></td><td>{site["ndvi"]:.3f}</td></tr>
                    <tr><td><b>Tree Density:</b></td><td>{site["tree_density"]}/ha</td></tr>
                    <tr><td><b>Species Diversity:</b></td><td>{site["species_diversity"]:.2f}</td></tr>
                    <tr><td><b>Last Survey:</b></td><td>{site["last_survey"]}</td></tr>
                </table>
            </div>
            """

            folium.Marker(
                location=[site["lat"], site["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Health Index: {site['health_index']:.2f}",
                icon=folium.Icon(color=marker_color, icon=icon, prefix="fa"),
            ).add_to(forest_cluster)

        forest_cluster.add_to(layer_groups["forest_health"])

    def _add_coastal_resilience_layers(
        self, m: folium.Map, layer_groups: dict, coastal_data: dict[str, Any]
    ) -> None:
        """Add coastal resilience visualization layers."""
        logger.info("Adding coastal resilience visualization layers...")

        # Create coastal monitoring cluster
        coastal_cluster = MarkerCluster(name="Coastal Monitoring Sites")

        coastal_sites = self._generate_coastal_monitoring_sites(coastal_data)

        for site in coastal_sites:
            # Color code by vulnerability level
            if site["vulnerability"] < 0.3:
                marker_color = "blue"
                icon = "anchor"
            elif site["vulnerability"] < 0.7:
                marker_color = "orange"
                icon = "exclamation-triangle"
            else:
                marker_color = "red"
                icon = "warning"

            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #4682B4; margin: 0 0 8px 0;">🌊 Coastal Monitoring</h4>
                <table style="font-size: 11px; width: 100%;">
                    <tr><td><b>Site ID:</b></td><td>{site["site_id"]}</td></tr>
                    <tr><td><b>Vulnerability:</b></td><td>{site["vulnerability"]:.2f}</td></tr>
                    <tr><td><b>Erosion Rate:</b></td><td>{site["erosion_rate"]:.1f} m/yr</td></tr>
                    <tr><td><b>Sea Level Trend:</b></td><td>{site["sea_level_trend"]:.1f} mm/yr</td></tr>
                    <tr><td><b>Storm Exposure:</b></td><td>{site["storm_exposure"]}</td></tr>
                </table>
            </div>
            """

            folium.Marker(
                location=[site["lat"], site["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Vulnerability: {site['vulnerability']:.2f}",
                icon=folium.Icon(color=marker_color, icon=icon, prefix="fa"),
            ).add_to(coastal_cluster)

        coastal_cluster.add_to(layer_groups["coastal_resilience"])

    def _add_fire_risk_layers(
        self, m: folium.Map, layer_groups: dict, fire_data: dict[str, Any]
    ) -> None:
        """Add fire risk visualization layers."""
        logger.info("Adding fire risk visualization layers...")

        # Create fire monitoring cluster
        fire_cluster = MarkerCluster(name="Fire Risk Monitoring")

        fire_sites = self._generate_fire_monitoring_sites(fire_data)

        for site in fire_sites:
            # Color code by risk level
            if site["risk_level"] < 0.3:
                marker_color = "green"
                icon = "fire-extinguisher"
            elif site["risk_level"] < 0.7:
                marker_color = "orange"
                icon = "exclamation-triangle"
            else:
                marker_color = "red"
                icon = "fire"

            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #DC143C; margin: 0 0 8px 0;">🔥 Fire Risk Site</h4>
                <table style="font-size: 11px; width: 100%;">
                    <tr><td><b>Site ID:</b></td><td>{site["site_id"]}</td></tr>
                    <tr><td><b>Risk Level:</b></td><td>{site["risk_level"]:.2f}</td></tr>
                    <tr><td><b>Fuel Moisture:</b></td><td>{site["fuel_moisture"]:.1f}%</td></tr>
                    <tr><td><b>Fire Weather Index:</b></td><td>{site["fire_weather_index"]:.1f}</td></tr>
                    <tr><td><b>Suppression Distance:</b></td><td>{site["suppression_distance"]:.1f} km</td></tr>
                </table>
            </div>
            """

            folium.Marker(
                location=[site["lat"], site["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Risk Level: {site['risk_level']:.2f}",
                icon=folium.Icon(color=marker_color, icon=icon, prefix="fa"),
            ).add_to(fire_cluster)

        fire_cluster.add_to(layer_groups["fire_risk"])

    def _add_community_development_layers(
        self, m: folium.Map, layer_groups: dict, community_data: dict[str, Any]
    ) -> None:
        """Add community development visualization layers."""
        logger.info("Adding community development visualization layers...")

        # Create community facilities cluster
        community_cluster = MarkerCluster(name="Community Facilities")

        facilities = self._generate_community_facilities(community_data)

        for facility in facilities:
            # Color code by facility type
            facility_colors = {
                "healthcare": "red",
                "education": "blue",
                "emergency": "orange",
                "community": "green",
            }

            facility_icons = {
                "healthcare": "plus",
                "education": "graduation-cap",
                "emergency": "exclamation-triangle",
                "community": "users",
            }

            color = facility_colors.get(facility["type"], "gray")
            icon = facility_icons.get(facility["type"], "info")

            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #4B0082; margin: 0 0 8px 0;">🏘️ {facility["name"]}</h4>
                <table style="font-size: 11px; width: 100%;">
                    <tr><td><b>Type:</b></td><td>{facility["type"].title()}</td></tr>
                    <tr><td><b>Capacity:</b></td><td>{facility["capacity"]}</td></tr>
                    <tr><td><b>Service Area:</b></td><td>{facility["service_area"]} km²</td></tr>
                    <tr><td><b>Accessibility:</b></td><td>{facility["accessibility"]}</td></tr>
                </table>
            </div>
            """

            folium.Marker(
                location=[facility["lat"], facility["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=facility["name"],
                icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            ).add_to(community_cluster)

        community_cluster.add_to(layer_groups["community_development"])

    def _add_integration_layers(
        self, m: folium.Map, layer_groups: dict, integration_data: dict[str, Any]
    ) -> None:
        """Add cross-domain integration visualization layers."""
        logger.info("Adding cross-domain integration layers...")

        # Add H3 hexagonal overlay for integrated analysis
        h3_cells = self._generate_h3_integration_grid(integration_data)

        for h3_cell, cell_data in h3_cells.items():
            # Get H3 cell boundary
            h3_boundary = h3.cell_to_boundary(h3_cell)

            # Color based on integration score
            integration_score = cell_data.get("integration_score", 0)
            if integration_score > 0.7:
                color = "#FF4500"  # High integration (red-orange)
            elif integration_score > 0.4:
                color = "#FFA500"  # Medium integration (orange)
            elif integration_score > 0.2:
                color = "#FFFF00"  # Low integration (yellow)
            else:
                color = "#87CEEB"  # Minimal integration (light blue)

            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #8B4513; margin: 0 0 8px 0;">🔗 H3 Integration Cell</h4>
                <table style="font-size: 11px; width: 100%;">
                    <tr><td><b>H3 Index:</b></td><td>{h3_cell}</td></tr>
                    <tr><td><b>Integration Score:</b></td><td>{integration_score:.3f}</td></tr>
                    <tr><td><b>Domain Count:</b></td><td>{cell_data["domain_count"]}</td></tr>
                    <tr><td><b>Risk Factors:</b></td><td>{cell_data["risk_factors"]}</td></tr>
                </table>
            </div>
            """

            folium.Polygon(
                locations=[[lat, lng] for lat, lng in h3_boundary],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Integration Score: {integration_score:.3f}",
                color="black",
                weight=1,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
            ).add_to(layer_groups["integration"])

    def _records_from_source(
        self,
        analysis_data: dict[str, Any] | None,
        source_key: str,
        data_keys: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Load records from analysis output or an explicitly configured file."""
        if isinstance(analysis_data, dict):
            for key in data_keys:
                records = analysis_data.get(key)
                if isinstance(records, list):
                    return records

        data_path = self.location_config.get("data_paths", {}).get(source_key)
        if data_path is None:
            logger.info(
                "No %s observations supplied; omitting visualization layer", source_key
            )
            return []
        try:
            return cast(
                list[dict[str, Any]],
                gpd.read_file(data_path).to_dict("records"),
            )
        except Exception as exc:
            logger.warning(
                "Unable to read %s data from %s: %s", source_key, data_path, exc
            )
            return []

    def _generate_forest_monitoring_sites(
        self, analysis_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return forest observations from analysis output or an explicit source."""
        return self._records_from_source(
            analysis_data,
            "forest_monitoring",
            ("forest_monitoring_sites", "monitoring_sites", "sites", "observations"),
        )

    def _generate_coastal_monitoring_sites(
        self, analysis_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return coastal observations from analysis output or an explicit source."""
        return self._records_from_source(
            analysis_data,
            "coastal_monitoring",
            ("coastal_monitoring_sites", "monitoring_sites", "sites", "observations"),
        )

    def _generate_fire_monitoring_sites(
        self, analysis_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return fire observations from analysis output or an explicit source."""
        return self._records_from_source(
            analysis_data,
            "fire_monitoring",
            ("fire_monitoring_sites", "monitoring_sites", "sites", "observations"),
        )

    def _generate_community_facilities(
        self, analysis_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return community facilities from analysis output or an explicit source."""
        return self._records_from_source(
            analysis_data,
            "community_facilities",
            ("facilities", "community_facilities", "sites", "observations"),
        )

    def _generate_h3_integration_grid(
        self, integration_data: dict[str, Any]
    ) -> dict[str, dict]:
        """Return H3 integration records explicitly produced by analysis."""
        if not isinstance(integration_data, dict):
            return {}  # type: ignore[unreachable]
        supplied = integration_data.get("h3_cells")
        if isinstance(supplied, dict):
            return {
                cell_id: cell_data
                for cell_id, cell_data in supplied.items()
                if isinstance(cell_id, str)
                and h3.is_valid_cell(cell_id)
                and isinstance(cell_data, dict)
            }

        h3_cells: dict[str, dict[str, Any]] = {}
        domain_datasets = integration_data.get("domain_spatial", {})
        if not isinstance(domain_datasets, dict):
            return h3_cells
        for domain_data in domain_datasets.values():
            if not isinstance(domain_data, dict):
                continue
            for cell_id, cell_data in domain_data.get("h3_cells", {}).items():
                if (
                    not isinstance(cell_id, str)
                    or not h3.is_valid_cell(cell_id)
                    or not isinstance(cell_data, dict)
                ):
                    continue
                score = cell_data.get(
                    "integration_score",
                    cell_data.get(
                        "forest_health_score", cell_data.get("risk_level", 0.0)
                    ),
                )
                try:
                    numeric_score = float(cast(Any, score))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric_score):
                    continue
                output = h3_cells.setdefault(
                    cell_id,
                    {"integration_score": 0.0, "domain_count": 0, "risk_factors": 0},
                )
                output["integration_score"] += numeric_score
                output["domain_count"] += 1
                output["risk_factors"] += int(cell_data.get("risk_factors", 0))

        for cell_data in h3_cells.values():
            cell_data["integration_score"] /= max(cell_data["domain_count"], 1)
        return h3_cells
