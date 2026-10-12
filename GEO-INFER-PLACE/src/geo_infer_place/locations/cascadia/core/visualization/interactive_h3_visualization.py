#!/usr/bin/env python3
"""
Interactive H3-Based Visualization Module for Cascadia Agricultural Analysis Framework

This module provides comprehensive interactive visualization capabilities:
- H3 hexagon-based maps with multiple data layers
- Toggle controls for different data sources
- Real-time data filtering and analysis
- Interactive features for exploration
- Export capabilities for reports and presentations

Based on modern web mapping technologies and geospatial visualization best practices.
"""

import folium
import json
from pathlib import Path
from typing import Any
import logging
from datetime import datetime

# Enhanced logging
from ..enhanced_logging import VisualizationLogger, DataSourceLogger

logger = logging.getLogger(__name__)


class InteractiveH3Visualization:
    """
    Interactive H3-based visualization system for Cascadia framework.

    Provides comprehensive mapping capabilities with:
    - Multiple data layer overlays
    - Interactive controls and filters
    - Real-time data analysis
    - Export and sharing features
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the interactive visualization system.

        Args:
            output_dir: Directory to store visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loggers
        self.viz_logger = VisualizationLogger("interactive_h3_viz")
        self.data_logger = DataSourceLogger("interactive_h3_viz")

        # Visualization settings
        self.viz_settings = {
            "default_center": [41.7558, -124.2016],  # Del Norte County
            "default_zoom": 10,
            "tile_layer": "OpenStreetMap",
            "hexagon_opacity": 0.7,
            "max_zoom": 18,
            "min_zoom": 5,
        }

        # Color schemes for different data types
        self.color_schemes = {
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
            "water_rights": {
                "Senior": "#0000FF",
                "Junior": "#4169E1",
                "Appropriative": "#1E90FF",
                "Riparian": "#00BFFF",
            },
            "ground_water": {
                "High": "#0000FF",
                "Medium": "#4169E1",
                "Low": "#87CEEB",
                "Critical": "#FF0000",
            },
            "surface_water": {
                "Abundant": "#0000FF",
                "Moderate": "#4169E1",
                "Limited": "#87CEEB",
                "Scarce": "#FF0000",
            },
            "power_source": {
                "Grid": "#FFD700",
                "Solar": "#FFA500",
                "Wind": "#87CEEB",
                "Hydro": "#0000FF",
                "None": "#808080",
            },
            "mortgage_debt": {
                "High": "#FF0000",
                "Medium": "#FFA500",
                "Low": "#FFFF00",
                "None": "#90EE90",
            },
        }

        logger.info("Interactive H3 Visualization system initialized")

    def create_comprehensive_map(
        self,
        h3_data: dict[str, Any],
        data_sources: dict[str, Any],
        target_hexagons: list[str],
        output_filename: str = "cascadia_comprehensive_map.html",
    ) -> Path:
        """
        Create a comprehensive interactive map with all data layers.

        Args:
            h3_data: H3-indexed data from fusion
            data_sources: Dictionary of data sources
            target_hexagons: List of target H3 hexagons
            output_filename: Output HTML filename

        Returns:
            Path to the generated HTML file
        """
        self.viz_logger.log_visualization_creation(
            viz_type="Comprehensive H3 Map",
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

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add data layers
        for module_name, module_data in data_sources.items():
            if module_data and "hexagons" in module_data:
                self._add_data_layer(m, module_name, module_data, h3_data)

        # Add H3 hexagon grid layer
        self._add_h3_grid_layer(m, target_hexagons, h3_data)

        # Add analysis layer
        self._add_analysis_layer(m, h3_data, data_sources)

        # Add interactive features
        self._add_interactive_features(m, h3_data, data_sources)

        # Generate HTML with custom CSS and JavaScript
        html_content = self._generate_enhanced_html(m, h3_data, data_sources)

        # Save the map
        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Log completion
        self.viz_logger.log_visualization_creation(
            viz_type="Comprehensive H3 Map",
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

        logger.info(f"Comprehensive map created: {output_path}")
        return output_path

    def _add_data_layer(
        self,
        m: folium.Map,
        module_name: str,
        module_data: dict[str, Any],
        h3_data: dict[str, Any],
    ):
        """Add a data layer to the map."""
        try:
            hexagons = module_data.get("hexagons", {})
            if not hexagons:
                return

            # Get color scheme for this module
            color_scheme = self.color_schemes.get(module_name, {})

            # Create feature group for this layer
            fg = folium.FeatureGroup(name=f"{module_name.replace('_', ' ').title()}")

            for hex_id, hex_data in hexagons.items():
                if hex_id in h3_data.get("hexagons", {}):
                    # Get hexagon boundary
                    boundary = self._get_hexagon_boundary(hex_id)
                    if boundary:
                        # Determine color based on data
                        color = self._get_hexagon_color(
                            hex_data, color_scheme, module_name
                        )

                        # Create popup content
                        popup_content = self._create_popup_content(
                            hex_id, hex_data, module_name
                        )

                        # Add polygon to map
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
            for hex_id, hex_data in h3_data.get("hexagons", {}).items():
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

    def _add_interactive_features(
        self, m: folium.Map, h3_data: dict[str, Any], data_sources: dict[str, Any]
    ):
        """Add interactive features to the map."""
        try:
            # Add search functionality
            folium.plugins.Search(
                layer=folium.FeatureGroup(name="Search Results"),
                geom_type="Polygon",
                placeholder="Search hexagons...",
                collapsed=False,
                search_label="name",
            ).add_to(m)

            # Add fullscreen button
            folium.plugins.Fullscreen(
                position="topleft",
                title="Expand me",
                title_cancel="Exit me",
                force_separate_button=True,
            ).add_to(m)

            # Add measure tool
            folium.plugins.MeasureControl(
                position="topleft",
                primary_length_unit="kilometers",
                secondary_length_unit="miles",
                primary_area_unit="sqkilometers",
                secondary_area_unit="acres",
            ).add_to(m)

            # Add minimap
            minimap = folium.plugins.MiniMap(
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
            coords = [[lat, lng] for lat, lng in boundary]
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
                land_use = hex_data.get("land_use", "Unknown")
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

    def _create_popup_content(
        self, hex_id: str, hex_data: dict[str, Any], module_name: str
    ) -> str:
        """Create popup content for a hexagon."""
        try:
            content = f"<b>H3 Hexagon:</b> {hex_id}<br>"
            content += f"<b>Module:</b> {module_name.replace('_', ' ').title()}<br>"

            for key, value in hex_data.items():
                if key != "geometry":
                    if isinstance(value, float):
                        content += (
                            f"<b>{key.replace('_', ' ').title()}:</b> {value:.2f}<br>"
                        )
                    else:
                        content += (
                            f"<b>{key.replace('_', ' ').title()}:</b> {value}<br>"
                        )

            return content
        except Exception as e:
            logger.error(f"Failed to create popup content: {e}")
            return f"<b>H3:</b> {hex_id}"

    def _generate_enhanced_html(
        self, m: folium.Map, h3_data: dict[str, Any], data_sources: dict[str, Any]
    ) -> str:
        """Generate enhanced HTML with custom CSS and JavaScript."""
        try:
            # Get the base HTML
            html = m._repr_html_()

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
            .data-summary {
                margin: 10px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }
            </style>
            """

            # Add custom JavaScript
            custom_js = (
                """
            <script>
            // Add data summary panel
            function addDataSummary() {
                const summary = document.createElement('div');
                summary.className = 'info-panel';
                summary.innerHTML = `
                    <h4>Cascadia Data Summary</h4>
                    <div class="data-summary">
                        <p><strong>Total Hexagons:</strong> """
                + str(len(h3_data.get("hexagons", {})))
                + """</p>
                        <p><strong>Data Sources:</strong> """
                + str(len(data_sources))
                + """</p>
                        <p><strong>Generated:</strong> """
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + """</p>
                    </div>
                `;
                document.body.appendChild(summary);
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

    def create_layer_specific_map(
        self,
        module_name: str,
        module_data: dict[str, Any],
        target_hexagons: list[str],
        output_filename: str | None = None,
    ) -> Path:
        """
        Create a map focused on a specific data layer.

        Args:
            module_name: Name of the module/layer
            module_data: Data for the specific module
            target_hexagons: List of target H3 hexagons
            output_filename: Optional output filename

        Returns:
            Path to the generated HTML file
        """
        if output_filename is None:
            output_filename = f"{module_name}_map.html"

        self.viz_logger.log_visualization_creation(
            viz_type=f"{module_name.title()} Layer Map",
            data_sources=[module_name],
            hexagon_count=len(target_hexagons),
            layers=[module_name],
            interactive_features=[
                "Layer Focus",
                "Data Filtering",
                "Hexagon Selection",
                "Export Capabilities",
            ],
        )

        # Create base map
        m = folium.Map(
            location=self.viz_settings["default_center"],
            zoom_start=self.viz_settings["default_zoom"],
            tiles=self.viz_settings["tile_layer"],
        )

        # Add the specific layer
        self._add_data_layer(m, module_name, module_data, {})

        # Add H3 grid for reference
        self._add_h3_grid_layer(m, target_hexagons, {})

        # Save the map
        output_path = self.output_dir / output_filename
        m.save(output_path)

        logger.info(f"{module_name} layer map created: {output_path}")
        return output_path

    def export_map_data(
        self, h3_data: dict[str, Any], data_sources: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Export map data for external analysis.

        Args:
            h3_data: H3-indexed data
            data_sources: Dictionary of data sources

        Returns:
            Dictionary with exported data
        """
        try:
            export_data = {
                "metadata": {
                    "generated": datetime.now().isoformat(),
                    "total_hexagons": len(h3_data.get("hexagons", {})),
                    "data_sources": list(data_sources.keys()),
                    "h3_resolution": h3_data.get("h3_resolution", 8),
                },
                "hexagons": h3_data.get("hexagons", {}),
                "data_sources": data_sources,
                "analysis": h3_data.get("analysis", {}),
            }

            # Save to JSON file
            export_path = self.output_dir / "map_data_export.json"
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Map data exported: {export_path}")
            return export_data

        except Exception as e:
            logger.error(f"Failed to export map data: {e}")
            return {}


def create_interactive_h3_visualization(output_dir: Path) -> InteractiveH3Visualization:
    """
    Create an interactive H3 visualization instance.

    Args:
        output_dir: Directory to store visualization outputs

    Returns:
        InteractiveH3Visualization instance
    """
    return InteractiveH3Visualization(output_dir)
