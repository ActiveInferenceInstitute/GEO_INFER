#!/usr/bin/env python3
"""
Comprehensive Visualization Module for Cascadia Framework

This module provides all visualization capabilities:
- Interactive H3-based maps
- Static visualizations
- Dashboard generation
- Data export for external tools
- Real-time visualization updates

Separates visualization concerns from main analysis logic.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from datetime import datetime
import os

# Visualization libraries
try:
    import folium
    import folium.plugins as plugins
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    FOLIUM_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Visualization libraries not available")

# Enhanced logging (from parent core package)
from ..enhanced_logging import VisualizationLogger, DataSourceLogger

logger = logging.getLogger(__name__)


class ComprehensiveVisualizationEngine:
    """
    Comprehensive visualization engine for Cascadia framework.

    Provides:
    - Interactive H3-based maps with multiple layers
    - Static visualizations for reports
    - Dashboard generation
    - Data export capabilities
    - Real-time visualization updates
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the comprehensive visualization engine.

        Args:
            output_dir: Directory to store visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.interactive_dir = self.output_dir / "interactive"
        self.static_dir = self.output_dir / "static"
        self.dashboard_dir = self.output_dir / "dashboard"
        self.export_dir = self.output_dir / "export"

        for subdir in [
            self.interactive_dir,
            self.static_dir,
            self.dashboard_dir,
            self.export_dir,
        ]:
            subdir.mkdir(parents=True, exist_ok=True)

        # Initialize loggers
        self.viz_logger = VisualizationLogger("comprehensive_viz")
        self.data_logger = DataSourceLogger("comprehensive_viz")

        # Visualization settings
        self.viz_settings = {
            "default_center": [41.7558, -124.2016],  # Del Norte County
            "default_zoom": 10,
            "tile_layer": "OpenStreetMap",
            "hexagon_opacity": 0.7,
            "max_zoom": 18,
            "min_zoom": 5,
            "color_schemes": {
                "zoning": {
                    "Agricultural": "#90EE90",
                    "Residential": "#FFB6C1",
                    "Commercial": "#FFD700",
                    "Industrial": "#A0522D",
                    "Conservation": "#228B22",
                },
                "current_use": {
                    "Agriculture": "#90EE90",
                    "Forest": "#228B22",
                    "Residential": "#FFB6C1",
                    "Commercial": "#FFD700",
                    "Industrial": "#A0522D",
                    "Open Space": "#98FB98",
                },
                "ownership": {
                    "Private": "#FF6B6B",
                    "Public": "#4ECDC4",
                    "Corporate": "#45B7D1",
                    "Trust": "#96CEB4",
                },
                "improvements": {
                    "High": "#FF0000",
                    "Medium": "#FFA500",
                    "Low": "#FFFF00",
                    "None": "#808080",
                },
            },
        }

        logger.info("Comprehensive Visualization Engine initialized")

    def create_interactive_h3_map(
        self,
        h3_data: dict[str, Any],
        data_sources: dict[str, Any],
        target_hexagons: list[str],
        output_filename: str = "cascadia_interactive_map.html",
        initial_visible_layers: list[str] | None = None,
        include_layers: list[str] | None = None,
        module_status: dict[str, Any] | None = None,
        redevelopment_scores: dict[str, float] | None = None,
    ) -> Path:
        """
        Create an interactive H3-based map with multiple layers.

        Args:
            h3_data: H3-indexed data from fusion
            data_sources: Dictionary of data sources
            target_hexagons: List of target H3 hexagons
            output_filename: Output HTML filename

        Returns:
            Path to the generated HTML file
        """
        if not FOLIUM_AVAILABLE:
            logger.error("Folium not available for interactive maps")
            return None

        start_time = datetime.now()

        self.viz_logger.log_visualization_creation(
            viz_type="Interactive H3 Map",
            data_sources=list(data_sources.keys()),
            hexagon_count=len(target_hexagons),
            layers=list(data_sources.keys()),
            interactive_features=[
                "Layer Toggles",
                "Data Filtering",
                "Hexagon Selection",
                "Export Capabilities",
                "Real-time Analysis",
            ],
        )

        # Create base map
        m = folium.Map(
            location=self.viz_settings["default_center"],
            zoom_start=self.viz_settings["default_zoom"],
            tiles=self.viz_settings["tile_layer"],
            max_zoom=self.viz_settings["max_zoom"],
            min_zoom=self.viz_settings["min_zoom"],
        )

        # Add layer control (we will add layers first with visibility flags)

        # Determine layer inclusion and initial visibility from config and CLI
        initial_visibility: dict[str, bool] = {}
        configured_layers: list[str] = []
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "analysis_config.yaml"
        try:
            if cfg_path.exists():
                import yaml

                with open(cfg_path) as cf:
                    cfg = yaml.safe_load(cf) or {}
                vis_cfg = cfg.get("visualization") or {}
                initial_layer_vis_cfg = vis_cfg.get("initial_layer_visibility") or {}
                if isinstance(initial_layer_vis_cfg, dict):
                    initial_visibility = {str(k): bool(v) for k, v in initial_layer_vis_cfg.items()}
                configured_layers = vis_cfg.get("layers") or []
                if not isinstance(configured_layers, list):
                    configured_layers = []
        except Exception:
            initial_visibility = {}
            configured_layers = []

        # Map config layer names to module keys used internally
        layer_to_module = {
            "zoning_data": "zoning",
            "current_use_data": "current_use",
            "ownership_data": "ownership",
            "improvements_data": "improvements",
            "redevelopment_potential": "redevelopment",
        }
        # Build include set: CLI overrides > config layers > all modules
        if include_layers:
            include_set = set([k for k in include_layers])
        elif configured_layers:
            include_set = set([layer_to_module.get(x, x) for x in configured_layers])
        else:
            include_set = set(
                list(data_sources.keys()) + (["redevelopment"] if redevelopment_scores else [])
            )
        visible_set = set([k for k in (initial_visible_layers or [])])

        # Add redevelopment (score) layer if available
        if redevelopment_scores:
            redevelop_show = False
            # Initial visibility for redevelopment from config key
            if not visible_set:
                redevelop_show = initial_visibility.get("redevelopment_potential", False)
            else:
                redevelop_show = "redevelopment" in visible_set
            if "redevelopment" in include_set:
                self._add_redevelopment_layer(
                    m, redevelopment_scores, target_hexagons, show=redevelop_show
                )

        # Add data layers (respect include list if provided and mapping)
        for module_name, module_data in data_sources.items():
            if include_set and module_name not in include_set:
                continue
            if module_data:
                # Determine visibility: CLI set overrides, otherwise config mapping
                if visible_set:
                    show_flag = module_name in visible_set
                else:
                    # Map internal module to config visibility name
                    config_key = module_name
                    # Configuration uses the canonical module keys (zoning/current_use/...)
                    show_flag = initial_visibility.get(config_key, False)
                self._add_data_layer(m, module_name, module_data, h3_data, show=show_flag)

        # Add H3 hexagon grid layer (sampled for performance if very large)
        grid_hexes = target_hexagons
        try:
            if len(target_hexagons) > 3000:
                # sample every Nth hex to reduce HTML size
                step = max(1, len(target_hexagons) // 2000)
                grid_hexes = target_hexagons[::step]
        except Exception as exc:
            logger.warning("Visualization step failed; continuing without it: %s", exc)
        self._add_h3_grid_layer(m, grid_hexes, h3_data)

        # Add analysis layer (kept hidden by default; redevelopment layer added above is preferred)
        self._add_analysis_layer(m, h3_data, data_sources)

        # Add interactive features
        self._add_interactive_features(m, h3_data, data_sources)

        # Add layer control after layers are added
        folium.LayerControl(collapsed=False).add_to(m)

        # Generate enhanced HTML with custom toggle panel
        layer_names = []
        try:
            # Collect layer names from added overlays in the Leaflet layer control
            layer_names = [
                name.replace("_", " ").title()
                for name in list(data_sources.keys())
                if (not include_set) or (name in include_set)
            ]
            if redevelopment_scores and "redevelopment" in include_set:
                layer_names = ["Redevelopment Potential"] + layer_names
        except Exception as exc:
            logger.warning("Visualization step failed; continuing without it: %s", exc)
        html_content = self._generate_enhanced_html(
            m, h3_data, data_sources, layer_names, module_status
        )

        # Save the map
        output_path = self.interactive_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Log completion
        duration = (datetime.now() - start_time).total_seconds()
        self.viz_logger.log_visualization_creation(
            viz_type="Interactive H3 Map",
            data_sources=list(data_sources.keys()),
            hexagon_count=len(target_hexagons),
            layers=list(data_sources.keys()),
            interactive_features=[
                "Layer Toggles",
                "Data Filtering",
                "Hexagon Selection",
                "Export Capabilities",
                "Real-time Analysis",
            ],
        )

        logger.info(f"Interactive H3 map created: {output_path}")
        return output_path

    def create_static_visualizations(
        self,
        h3_data: dict[str, Any],
        data_sources: dict[str, Any],
        redevelopment_scores: dict[str, float],
    ) -> dict[str, Path]:
        """
        Create static visualizations for reports and presentations.

        Args:
            h3_data: H3-indexed data
            data_sources: Dictionary of data sources
            redevelopment_scores: Dictionary of redevelopment scores

        Returns:
            Dictionary of visualization file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for static visualizations")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_paths = {}

        try:
            # Reduce matplotlib logging noise
            import logging as _logging

            _logging.getLogger("matplotlib").setLevel(_logging.WARNING)

            # Set up plotting style
            plt.style.use("default")
            sns.set_palette("husl")

            # 1. Data coverage visualization
            total_targets = max(1, len(self._get_hexagons_map(h3_data)))
            coverage_fig = self._create_coverage_plot(data_sources, total_targets)
            coverage_path = self.static_dir / "data_coverage.png"
            coverage_fig.savefig(coverage_path, dpi=300, bbox_inches="tight")
            plt.close(coverage_fig)
            viz_paths["coverage_plot"] = coverage_path

            # 2. Redevelopment score distribution
            if redevelopment_scores:
                score_fig = self._create_score_distribution_plot(redevelopment_scores)
                score_path = self.static_dir / "redevelopment_scores.png"
                score_fig.savefig(score_path, dpi=300, bbox_inches="tight")
                plt.close(score_fig)
                viz_paths["score_distribution"] = score_path

            # 3. Module comparison chart
            module_fig = self._create_module_comparison_plot(data_sources, total_targets)
            module_path = self.static_dir / "module_comparison.png"
            module_fig.savefig(module_path, dpi=300, bbox_inches="tight")
            plt.close(module_fig)
            viz_paths["module_comparison"] = module_path

            # 4. Data quality heatmap
            quality_fig = self._create_quality_heatmap(data_sources, total_targets)
            quality_path = self.static_dir / "data_quality.png"
            quality_fig.savefig(quality_path, dpi=300, bbox_inches="tight")
            plt.close(quality_fig)
            viz_paths["quality_heatmap"] = quality_path

            logger.info(f"Static visualizations created: {len(viz_paths)} files")

        except Exception as e:
            logger.error(f"Failed to create static visualizations: {e}")

        return viz_paths

    def create_dashboard(
        self,
        h3_data: dict[str, Any],
        data_sources: dict[str, Any],
        redevelopment_scores: dict[str, float],
        summary: dict[str, Any],
    ) -> Path:
        """
        Create a comprehensive dashboard with all visualizations.

        Args:
            h3_data: H3-indexed data
            data_sources: Dictionary of data sources
            redevelopment_scores: Dictionary of redevelopment scores
            summary: Analysis summary

        Returns:
            Path to the dashboard HTML file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create static visualizations first
        static_viz = self.create_static_visualizations(h3_data, data_sources, redevelopment_scores)

        # Create interactive map
        target_hexes = list(self._get_hexagons_map(h3_data).keys())
        interactive_map = self.create_interactive_h3_map(h3_data, data_sources, target_hexes)

        # Generate dashboard HTML
        dashboard_html = self._generate_dashboard_html(
            static_viz, interactive_map, summary, timestamp
        )

        # Save dashboard
        dashboard_path = self.dashboard_dir / "cascadia_dashboard.html"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html)

        logger.info(f"Comprehensive dashboard created: {dashboard_path}")
        return dashboard_path

    def export_visualization_data(
        self,
        h3_data: dict[str, Any],
        data_sources: dict[str, Any],
        redevelopment_scores: dict[str, float],
    ) -> dict[str, Path]:
        """
        Export visualization data for external tools.

        Args:
            h3_data: H3-indexed data
            data_sources: Dictionary of data sources
            redevelopment_scores: Dictionary of redevelopment scores

        Returns:
            Dictionary of exported file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_paths = {}

        try:
            # Export H3 data as GeoJSON
            h3_geojson = self._convert_h3_to_geojson(h3_data)
            h3_path = self.export_dir / "h3_data.geojson"
            with open(h3_path, "w") as f:
                json.dump(h3_geojson, f, indent=2)
            export_paths["h3_geojson"] = h3_path

            # Export redevelopment scores as CSV
            if redevelopment_scores:
                scores_df = pd.DataFrame(
                    [
                        {"hex_id": hex_id, "score": score}
                        for hex_id, score in redevelopment_scores.items()
                    ]
                )
                scores_path = self.export_dir / "redevelopment_scores.csv"
                scores_df.to_csv(scores_path, index=False)
                export_paths["scores_csv"] = scores_path

            # Export data sources summary
            sources_summary = {
                "timestamp": timestamp,
                "data_sources": list(data_sources.keys()),
                "total_hexagons": len(self._get_hexagons_map(h3_data)),
                "modules": {},
            }

            for module_name, module_data in data_sources.items():
                hex_map = self._get_hexagons_map(module_data)
                sources_summary["modules"][module_name] = {
                    "hexagon_count": len(hex_map),
                    "coverage_percentage": (
                        len(hex_map) / max(1, len(self._get_hexagons_map(h3_data))) * 100.0
                    ),
                }

            summary_path = self.export_dir / "data_sources_summary.json"
            with open(summary_path, "w") as f:
                json.dump(sources_summary, f, indent=2)
            export_paths["sources_summary"] = summary_path

            # Export a simple legend HTML
            try:
                legend_path = self.export_dir / "legend" / "cascadia_layers_legend.html"
                legend_path.parent.mkdir(parents=True, exist_ok=True)
                modules = sorted(list(data_sources.keys()))
                with open(legend_path, "w") as f:
                    f.write(
                        "<html><head><meta charset='utf-8'><title>Cascadia Layers</title></head><body>"
                    )
                    f.write("<h3>Layer Summary</h3><ul>")
                    for m in modules:
                        count = sources_summary["modules"][m]["hexagon_count"]
                        f.write(f"<li>{m}: {count} hexagons</li>")
                    f.write("</ul></body></html>")
                export_paths["legend_html"] = legend_path
            except Exception as exc:
                logger.warning("Visualization step failed; continuing without it: %s", exc)

            logger.info(f"Visualization data exported: {len(export_paths)} files")

        except Exception as e:
            logger.error(f"Failed to export visualization data: {e}")

        return export_paths

    def _get_hexagons_map(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Return a mapping of hex_id -> data from either a flat dict or a dict with 'hexagons'."""
        if isinstance(obj, dict) and "hexagons" in obj and isinstance(obj["hexagons"], dict):
            return obj["hexagons"]
        if isinstance(obj, dict):
            return obj
        return {}

    def _add_data_layer(
        self,
        m: folium.Map,
        module_name: str,
        module_data: dict[str, Any],
        h3_data: dict[str, Any],
        show: bool = False,
    ):
        """Add a data layer to the map."""
        try:
            hexagons = self._get_hexagons_map(module_data)
            if not hexagons:
                return

            # Get color scheme for this module
            color_scheme = self.viz_settings["color_schemes"].get(module_name, {})

            # Create feature group for this layer with initial visibility
            fg = folium.FeatureGroup(name=f"{module_name.replace('_', ' ').title()}", show=show)

            fused_hex_map = self._get_hexagons_map(h3_data)
            for hex_id, hex_data in hexagons.items():
                if fused_hex_map and hex_id not in fused_hex_map:
                    continue
                boundary = self._get_hexagon_boundary(hex_id)
                if boundary:
                    summarized = self._summarize_hex_data(hex_data, module_name)
                    color = self._get_hexagon_color(summarized, color_scheme, module_name)
                    popup_content = self._create_popup_content(hex_id, summarized, module_name)
                    folium.Polygon(
                        locations=boundary,
                        color=color,
                        weight=1,
                        fillColor=color,
                        fillOpacity=self.viz_settings["hexagon_opacity"],
                        popup=folium.Popup(popup_content, max_width=300),
                    ).add_to(fg)

            fg.add_to(m)

        except Exception as e:
            logger.error(f"Failed to add {module_name} layer: {e}")

    def _add_h3_grid_layer(
        self, m: folium.Map, target_hexagons: list[str], h3_data: dict[str, Any]
    ):
        """Add H3 grid layer for reference."""
        try:
            fg = folium.FeatureGroup(name="H3 Grid", show=False)

            for hex_id in target_hexagons:
                boundary = self._get_hexagon_boundary(hex_id)
                if boundary:
                    folium.Polygon(
                        locations=boundary,
                        color="gray",
                        weight=1,
                        fillColor="transparent",
                        fillOpacity=0,
                        popup=f"H3: {hex_id}",
                    ).add_to(fg)

            fg.add_to(m)

        except Exception as e:
            logger.error(f"Failed to add H3 grid layer: {e}")

    def _add_analysis_layer(
        self, m: folium.Map, h3_data: dict[str, Any], data_sources: dict[str, Any]
    ):
        """Add analysis layer with aggregated data."""
        try:
            fg = folium.FeatureGroup(name="Analysis", show=False)

            # Calculate aggregated scores for each hexagon
            for hex_id, hex_data in self._get_hexagons_map(h3_data).items():
                if "analysis" in hex_data:
                    boundary = self._get_hexagon_boundary(hex_id)
                    if boundary:
                        score = hex_data["analysis"].get("redevelopment_score", 0)
                        color = self._get_score_color(score)

                        popup_content = f"""
                        <b>Redevelopment Score:</b> {score:.2f}<br>
                        <b>Data Sources:</b> {len(hex_data.get("sources", []))}<br>
                        <b>Coverage:</b> {hex_data.get("coverage", 0):.1f}%
                        """

                        folium.Polygon(
                            locations=boundary,
                            color=color,
                            weight=2,
                            fillColor=color,
                            fillOpacity=0.8,
                            popup=folium.Popup(popup_content, max_width=300),
                        ).add_to(fg)

            fg.add_to(m)

        except Exception as e:
            logger.error(f"Failed to add analysis layer: {e}")

    def _add_redevelopment_layer(
        self,
        m: "folium.Map",
        scores: dict[str, float],
        target_hexagons: list[str],
        show: bool = True,
    ) -> None:
        """Add a layer visualizing redevelopment scores across hexagons.

        Uses a red-yellow-green color ramp by score.
        """
        try:
            fg = folium.FeatureGroup(name="Redevelopment Potential", show=show)
            for hex_id in target_hexagons:
                score = float(scores.get(hex_id, 0.0))
                boundary = self._get_hexagon_boundary(hex_id)
                if not boundary:
                    continue
                color = self._get_score_color(score)
                popup_content = f"""
                <b>H3:</b> {hex_id}<br>
                <b>Redevelopment Score:</b> {score:.3f}
                """
                folium.Polygon(
                    locations=boundary,
                    color=color,
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.75,
                    popup=folium.Popup(popup_content, max_width=300),
                ).add_to(fg)
            fg.add_to(m)
        except Exception as e:
            logger.error(f"Failed to add redevelopment layer: {e}")

    def _add_interactive_features(
        self, m: folium.Map, h3_data: dict[str, Any], data_sources: dict[str, Any]
    ):
        """Add interactive features to the map."""
        try:
            # Add search functionality
            plugins.Search(
                layer=folium.FeatureGroup(name="Search Results"),
                geom_type="Polygon",
                placeholder="Search hexagons...",
                collapsed=False,
                search_label="name",
            ).add_to(m)

            # Add fullscreen button
            plugins.Fullscreen(
                position="topleft",
                title="Expand me",
                title_cancel="Exit me",
                force_separate_button=True,
            ).add_to(m)

            # Add measure tool
            plugins.MeasureControl(
                position="topleft",
                primary_length_unit="kilometers",
                secondary_length_unit="miles",
                primary_area_unit="sqkilometers",
                secondary_area_unit="acres",
            ).add_to(m)

            # Add minimap
            minimap = plugins.MiniMap(
                tile_layer="OpenStreetMap",
                position="bottomright",
                width=150,
                height=150,
                collapsed_width=25,
                collapsed_height=25,
            )
            m.add_child(minimap)

        except Exception as e:
            logger.error(f"Failed to add interactive features: {e}")

    def _get_hexagon_boundary(self, hex_id: str) -> list[list[float]] | None:
        """Get the boundary coordinates for an H3 hexagon."""
        try:
            import h3

            boundary = h3.cell_to_boundary(hex_id)
            coords = [[lat, lng] for (lat, lng) in boundary]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            return coords
        except Exception as e:
            logger.error(f"Failed to get boundary for hexagon {hex_id}: {e}")
            return None

    def _get_hexagon_color(
        self, hex_data: dict[str, Any], color_scheme: dict[str, str], module_name: str
    ) -> str:
        """Get color for a hexagon based on its data."""
        try:
            if module_name == "zoning":
                zone_type = hex_data.get("zone_type", "Unknown")
                return color_scheme.get(zone_type, "#808080")
            elif module_name == "current_use":
                land_use = hex_data.get("land_use", hex_data.get("crop_type", "Unknown"))
                return color_scheme.get(land_use, "#808080")
            elif module_name == "ownership":
                owner_type = hex_data.get("owner_type", "Unknown")
                return color_scheme.get(owner_type, "#808080")
            elif module_name == "improvements":
                improvement_level = hex_data.get("improvement_level", "Unknown")
                return color_scheme.get(improvement_level, "#808080")
            else:
                # Default color based on data value
                value = hex_data.get("value", 0)
                if isinstance(value, (int, float)):
                    return self._get_value_color(value)
                return "#808080"
        except Exception as e:
            logger.error(f"Failed to get color for hexagon: {e}")
            return "#808080"

    def _get_value_color(self, value: float) -> str:
        """Get color based on numeric value."""
        try:
            if value > 0.8:
                return "#FF0000"  # Red for high values
            elif value > 0.6:
                return "#FFA500"  # Orange for medium-high
            elif value > 0.4:
                return "#FFFF00"  # Yellow for medium
            elif value > 0.2:
                return "#90EE90"  # Light green for medium-low
            else:
                return "#228B22"  # Green for low values
        except Exception:
            return "#808080"  # Gray for unknown

    def _get_score_color(self, score: float) -> str:
        """Get color based on redevelopment score."""
        try:
            if score > 0.8:
                return "#FF0000"  # Red for high redevelopment potential
            elif score > 0.6:
                return "#FFA500"  # Orange
            elif score > 0.4:
                return "#FFFF00"  # Yellow
            elif score > 0.2:
                return "#90EE90"  # Light green
            else:
                return "#228B22"  # Green for low redevelopment potential
        except Exception:
            return "#808080"

    def _create_popup_content(self, hex_id: str, hex_data: dict[str, Any], module_name: str) -> str:
        """Create popup content for a hexagon."""
        try:
            content = f"<b>H3 Hexagon:</b> {hex_id}<br>"
            content += f"<b>Module:</b> {module_name.replace('_', ' ').title()}<br>"

            for key, value in hex_data.items():
                if key != "geometry":
                    if isinstance(value, float):
                        content += f"<b>{key.replace('_', ' ').title()}:</b> {value:.2f}<br>"
                    else:
                        content += f"<b>{key.replace('_', ' ').title()}:</b> {value}<br>"

            return content
        except Exception as e:
            logger.error(f"Failed to create popup content: {e}")
            return f"<b>H3:</b> {hex_id}"

    def _summarize_hex_data(self, hex_data: Any, module_name: str) -> dict[str, Any]:
        """Summarize per-hex data. Accepts a dict or list of dicts and returns a representative dict."""
        try:
            if isinstance(hex_data, dict):
                return hex_data
            if not isinstance(hex_data, list) or len(hex_data) == 0:
                return {}

            # Helper to pick most common value for a key among items
            def most_common(key_candidates: list[str]) -> str:
                from collections import Counter

                values = []
                for item in hex_data:
                    if isinstance(item, dict):
                        for k in key_candidates:
                            if k in item and item[k] not in [None, ""]:
                                values.append(str(item[k]))
                                break
                return Counter(values).most_common(1)[0][0] if values else "Unknown"

            summary: dict[str, Any] = {}
            summary["count"] = len(hex_data)
            if module_name == "zoning":
                summary["zone_type"] = most_common(["zone_type", "zone", "zone_code"])
            elif module_name == "current_use":
                lu = most_common(["land_use", "crop_type"])
                summary["land_use"] = lu
            elif module_name == "ownership":
                summary["owner_type"] = most_common(["owner_type", "owner_category"])
            elif module_name == "improvements":
                # Derive improvement level from building_value if available
                values = [
                    item.get("building_value")
                    for item in hex_data
                    if isinstance(item, dict) and "building_value" in item
                ]
                level = "None"
                if values:
                    try:
                        import numpy as _np

                        v = float(_np.nanmean(values))
                        level = "High" if v >= 100000 else ("Medium" if v >= 25000 else "Low")
                    except Exception:
                        level = "Low"
                summary["improvement_level"] = level
            return summary
        except Exception:
            return {}

    def _generate_enhanced_html(
        self,
        m: folium.Map,
        h3_data: dict[str, Any],
        data_sources: dict[str, Any],
        layer_names: list[str] | None = None,
        module_status: dict[str, Any] | None = None,
    ) -> str:
        """Generate enhanced HTML with custom CSS and JavaScript."""
        try:
            # Get the base HTML
            html = m.get_root().render()

            # Add custom CSS
            custom_css = """
            <style>
            .leaflet-popup-content {
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.4;
            }
            .leaflet-control-layers {
                background: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
            }
            .leaflet-control-layers label {
                margin: 5px 0;
                font-weight: bold;
            }
            .info-panel {
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                z-index: 1000;
                max-width: 300px;
            }
            .info-panel table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }
            .info-panel th, .info-panel td {
                border-bottom: 1px solid #eee;
                padding: 4px 6px;
                text-align: left;
            }
            .status-pill {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 10px;
                color: #fff;
                font-size: 11px;
            }
            .status-real { background: #2ca25f; }
            .status-raw { background: #6c757d; }
            .status-missing { background: #dc3545; }
            .data-summary {
                margin: 10px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }
            </style>
            """

            # Add custom JavaScript
            # Build module status HTML server-side for simplicity
            module_rows = ""
            try:
                if module_status and isinstance(module_status, dict):
                    for mname, stat in module_status.items():
                        hex_cnt = stat.get("output_hexagons", 0)
                        in_feat = stat.get("input_features", 0)
                        cache = stat.get("h3_cache", "")
                        emp = bool(stat.get("empirical_exists", False))
                        raw = bool(stat.get("raw_exists", False))
                        status_lbl = "Real" if emp else ("Raw" if raw else "Missing")
                        status_cls = (
                            "status-real" if emp else ("status-raw" if raw else "status-missing")
                        )
                        safe_cache = cache if isinstance(cache, str) else ""
                        module_rows += f"<tr><td>{mname}</td><td>{hex_cnt}</td><td>{in_feat}</td><td><span class='status-pill {status_cls}'>{status_lbl}</span></td><td title='{safe_cache}'>{safe_cache.split('/')[-1] if safe_cache else ''}</td></tr>"
            except Exception as exc:
                logger.warning("Visualization step failed; continuing without it: %s", exc)

            custom_js = (
                """
            <script>
            // Add data summary panel and layer toggle shortcuts
            function addDataSummary() {
                const summary = document.createElement('div');
                summary.className = 'info-panel';
                summary.innerHTML = `
                    <h4>Cascadia Data Summary</h4>
                    <div class="data-summary">
                         <p><strong>Fused Hexagons:</strong> """
                + str(len(self._get_hexagons_map(h3_data)))
                + """</p>
                        <p><strong>Data Sources:</strong> """
                + str(len(data_sources))
                + """</p>
                        <p><strong>Generated:</strong> """
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + """</p>
                    </div>
                    <div class="data-summary">
                        <p><strong>Module Status</strong></p>
                        <div style="max-height: 200px; overflow-y: auto;">
                          <table>
                            <thead>
                              <tr><th>Module</th><th>Hex</th><th>Features</th><th>Source</th><th>Cache</th></tr>
                            </thead>
                            <tbody>"""
                + module_rows
                + """</tbody>
                          </table>
                        </div>
                    </div>
                    <div class="data-summary">
                        <p><strong>Layer Toggles</strong></p>
                        <div id="layer-toggles"></div>
                    </div>
                `;
                document.body.appendChild(summary);
                try {
                    const controls = document.querySelectorAll('.leaflet-control-layers-overlays label');
                    const container = document.getElementById('layer-toggles');
                    if (controls && container) {
                        controls.forEach((lbl, idx) => {
                            const clone = lbl.cloneNode(true);
                            clone.style.display = 'block';
                            clone.style.marginBottom = '6px';
                            container.appendChild(clone);
                        });
                    }
                } catch (e) {
                    // no-op
                }
            }

            // Initialize when page loads
            window.addEventListener('load', function() {
                addDataSummary();
            });
            </script>
            """
            )

            # Insert custom CSS and JS into the HTML
            html = html.replace("</head>", custom_css + "</head>")
            html = html.replace("</body>", custom_js + "</body>")

            return html

        except Exception as e:
            logger.error(f"Failed to generate enhanced HTML: {e}")
            return m._repr_html_()

    def _create_coverage_plot(self, data_sources: dict[str, Any], total_targets: int):
        """Create data coverage visualization based on per-module hexagon counts."""
        fig, ax = plt.subplots(figsize=(10, 6))

        modules = list(data_sources.keys())
        coverages = []

        for module_name, module_data in data_sources.items():
            hex_map = self._get_hexagons_map(module_data)
            cov = (len(hex_map) / total_targets * 100.0) if total_targets > 0 else 0.0
            coverages.append(cov)

        bars = ax.bar(modules, coverages, color="skyblue", alpha=0.85)
        ax.set_title("Data Coverage by Module", fontsize=14, fontweight="bold")
        ax.set_ylabel("Coverage Percentage (%)", fontsize=12)
        ax.set_xlabel("Module", fontsize=12)
        ax.set_ylim(0, max(100, max(coverages + [0]) * 1.2))

        for bar, coverage in zip(bars, coverages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(1, 0.02 * height),
                f"{coverage:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def _create_score_distribution_plot(self, redevelopment_scores: dict[str, float]):
        """Create redevelopment score distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        scores = list(redevelopment_scores.values())

        ax.hist(scores, bins=20, color="lightcoral", alpha=0.7, edgecolor="black")
        ax.set_title("Redevelopment Score Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Redevelopment Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

        # Add statistics
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        ax.axvline(mean_score, color="red", linestyle="--", label=f"Mean: {mean_score:.3f}")
        ax.axvline(
            median_score,
            color="blue",
            linestyle="--",
            label=f"Median: {median_score:.3f}",
        )
        ax.legend()

        plt.tight_layout()

        return fig

    def _create_module_comparison_plot(self, data_sources: dict[str, Any], total_targets: int):
        """Create module comparison chart with counts and heuristic quality."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        modules = list(data_sources.keys())
        hexagon_counts = []
        quality_scores = []

        for module_name, module_data in data_sources.items():
            hex_map = self._get_hexagons_map(module_data)
            hex_count = len(hex_map)
            # Heuristic quality: attribute diversity across features
            sample_values = []
            for v in list(hex_map.values())[:2000]:
                if isinstance(v, list) and v:
                    sample_values.extend(list(v[0].keys()))
                elif isinstance(v, dict):
                    sample_values.extend(list(v.keys()))
            diverse_attrs = len(set([k for k in sample_values if k != "feature_id"]))
            quality = (
                max(0.0, min(1.0, 0.2 + 0.8 * (diverse_attrs / 10.0))) if hex_count > 0 else 0.0
            )
            hexagon_counts.append(hex_count)
            quality_scores.append(quality)

        bars1 = ax1.bar(modules, hexagon_counts, color="lightgreen", alpha=0.8)
        ax1.set_title("Hexagon Count by Module", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Hexagons", fontsize=10)
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_ylim(0, max(hexagon_counts + [0]) * 1.2 + 1)

        bars2 = ax2.bar(modules, quality_scores, color="lightblue", alpha=0.8)
        ax2.set_title("Quality Score by Module", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Quality Score (0-1)", fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def _create_quality_heatmap(self, data_sources: dict[str, Any], total_targets: int):
        """Create data quality heatmap with computed metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        modules = list(data_sources.keys())
        metrics = ["coverage", "quality", "completeness", "accuracy"]

        heatmap_data = []
        for module_name in modules:
            hex_map = self._get_hexagons_map(data_sources.get(module_name, {}))
            coverage = (len(hex_map) / total_targets) if total_targets > 0 else 0.0
            # Reuse heuristic quality
            sample_values = []
            for v in list(hex_map.values())[:2000]:
                if isinstance(v, list) and v:
                    sample_values.extend(list(v[0].keys()))
                elif isinstance(v, dict):
                    sample_values.extend(list(v.keys()))
            diverse_attrs = len(set([k for k in sample_values if k != "feature_id"]))
            quality = (
                max(0.0, min(1.0, 0.2 + 0.8 * (diverse_attrs / 10.0))) if len(hex_map) > 0 else 0.0
            )
            # Completeness and accuracy indicators derived from coverage and quality
            completeness = min(1.0, coverage * 0.8 + 0.2)
            accuracy = min(1.0, quality * 0.7 + 0.3)
            heatmap_data.append([coverage, quality, completeness, accuracy])

        im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(modules)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(modules)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Score", rotation=270, labelpad=15)
        ax.set_title("Data Quality Heatmap by Module", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def _convert_h3_to_geojson(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Convert H3 data to GeoJSON format."""
        features = []

        for hex_id, hex_data in self._get_hexagons_map(h3_data).items():
            try:
                import h3

                boundary = h3.cell_to_boundary(hex_id)
                coordinates = [[lng, lat] for (lat, lng) in boundary]
                if coordinates and coordinates[0] != coordinates[-1]:
                    coordinates.append(coordinates[0])

                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                    "properties": {"hex_id": hex_id, **hex_data},
                }
                features.append(feature)
            except Exception as e:
                logger.warning(f"Failed to convert hexagon {hex_id}: {e}")

        return {"type": "FeatureCollection", "features": features}

    def _generate_dashboard_html(
        self,
        static_viz: dict[str, Path],
        interactive_map: Path,
        summary: dict[str, Any],
        timestamp: str,
    ) -> str:
        """Generate comprehensive dashboard HTML."""
        # Convert to relative paths from dashboard directory
        # Dashboard is in output/visualizations/dashboard/
        # Interactive map is in output/visualizations/interactive/
        # Static viz is in output/visualizations/static/

        # Get relative path from dashboard to interactive map
        try:
            interactive_rel = os.path.relpath(interactive_map, self.dashboard_dir)
        except ValueError:
            # Fallback if on different drives (Windows)
            interactive_rel = f"../interactive/{interactive_map.name}"

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cascadia Agricultural Analysis Dashboard - {timestamp}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .content {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
                .section {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .full-width {{
                    grid-column: 1 / -1;
                }}
                .viz-container {{
                    text-align: center;
                    margin: 10px 0;
                }}
                .viz-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌲 Cascadia Agricultural Analysis Dashboard</h1>
                <p>Comprehensive agricultural land analysis for the Cascadian bioregion</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{summary.get("total_hexagons", 0):,}</div>
                    <div class="stat-label">Total Hexagons</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get("processed_hexagons", 0):,}</div>
                    <div class="stat-label">Processed Hexagons</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(summary.get("data_sources", []))}</div>
                    <div class="stat-label">Data Sources</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get("h3_resolution", 8)}</div>
                    <div class="stat-label">H3 Resolution</div>
                </div>
            </div>

            <div class="content">
                <div class="section full-width">
                    <h2>🗺️ Interactive Map</h2>
                    <iframe src="{interactive_rel}" width="100%" height="600" frameborder="0"></iframe>
                </div>
        """

        # Add static visualizations with relative paths
        for viz_name, viz_path in static_viz.items():
            if viz_path.exists():
                # Compute relative path from dashboard to static viz
                try:
                    viz_rel = os.path.relpath(viz_path, self.dashboard_dir)
                except ValueError:
                    viz_rel = f"../static/{viz_path.name}"

                html += f"""
                <div class="section">
                    <h3>{viz_name.replace("_", " ").title()}</h3>
                    <div class="viz-container">
                        <img src="{viz_rel}" alt="{viz_name}">
                    </div>
                </div>
                """

        html += """
            </div>
        </body>
        </html>
        """

        return html


def create_comprehensive_visualization_engine(
    output_dir: Path,
) -> ComprehensiveVisualizationEngine:
    """
    Create a comprehensive visualization engine instance.

    Args:
        output_dir: Directory to store visualization outputs

    Returns:
        ComprehensiveVisualizationEngine instance
    """
    return ComprehensiveVisualizationEngine(output_dir)
