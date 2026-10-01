"""
H3 Visualization module for creating interactive maps and static plots.

This module provides comprehensive visualization capabilities for H3 hexagonal grids
including interactive maps, static plots, animations, and analytical visualizations.
"""

import logging
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as seaborn

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    import folium
    from folium import plugins

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("folium not available. Install with 'uv pip install folium'")

try:
    import plotly.graph_objects as go
    import plotly.express as px  # noqa: F401  # availability probe; binding intentionally unused
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. Install with 'uv pip install plotly'")

from .core import H3Grid, H3Cell, H3Analytics
from .operations import cell_to_boundary


class H3MapVisualizer:
    """
    Interactive map visualizations for H3 grids using Folium and Plotly.

    Provides methods for creating interactive web maps with H3 hexagonal overlays,
    choropleth maps, and analytical visualizations.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize visualizer for an H3Grid.

        Args:
            grid: H3Grid instance to visualize
        """
        self.grid = grid
        self.analytics = H3Analytics(grid)

    def create_folium_map(
        self,
        value_column: str | None = None,
        color_scheme: str = "viridis",
        **kwargs: Any,
    ) -> "folium.Map":
        """
        Create interactive Folium map with H3 grid overlay.

        Args:
            value_column: Column name for choropleth coloring
            color_scheme: Color scheme for visualization
            **kwargs: Additional styling parameters

        Returns:
            Folium map object

        Example:
            >>> grid = H3Grid.from_center(37.7749, -122.4194, 9, k=2)
            >>> viz = H3MapVisualizer(grid)
            >>> m = viz.create_folium_map()
            >>> m.save('h3_map.html')
        """
        if not FOLIUM_AVAILABLE:
            raise ImportError(
                "folium package required. Install with 'uv pip install folium'"
            )

        if not self.grid.cells:
            return folium.Map(location=[0, 0], zoom_start=2)

        # Get grid center and bounds for map setup
        center_lat, center_lng = self.grid.center()
        bounds = self.grid.bounds()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=kwargs.get("zoom_start", 10),
            tiles=kwargs.get("tiles", "OpenStreetMap"),
        )

        # Prepare data for visualization
        if value_column:
            # Create choropleth map
            self._add_choropleth_layer(m, value_column, color_scheme, **kwargs)
        else:
            # Create simple overlay
            self._add_simple_overlay(m, **kwargs)

        # Add grid statistics as control
        self._add_statistics_control(m)

        # Fit map to bounds
        if len(self.grid.cells) > 1:
            southwest = [bounds[0], bounds[1]]
            northeast = [bounds[2], bounds[3]]
            m.fit_bounds([southwest, northeast])

        return m

    def _add_choropleth_layer(
        self,
        m: "folium.Map",
        value_column: str,
        color_scheme: str,
        **kwargs: Any,
    ) -> None:
        """Add choropleth layer to map."""
        # Get values for coloring
        values = []
        for cell in self.grid.cells:
            value = cell.properties.get(value_column, 0)
            values.append(value)

        if not values:
            logger.warning(f"No values found for column '{value_column}'")
            return self._add_simple_overlay(m, **kwargs)

        # Normalize values for coloring
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            normalized_values = [0.5] * len(values)
        else:
            normalized_values = [(v - min_val) / (max_val - min_val) for v in values]

        # Create color map
        if MATPLOTLIB_AVAILABLE:
            colormap = plt.cm.get_cmap(color_scheme)
        else:
            # Fallback color mapping
            colormap = lambda x: (x, 0, 1 - x)  # type: ignore[assignment]

        # Add cells with colors
        for cell, norm_value, actual_value in zip(
            self.grid.cells, normalized_values, values
        ):
            color = colormap(norm_value)
            hex_color = f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"

            # Create polygon
            boundary_coords = [[lat, lng] for lat, lng in cell.boundary]

            # Create popup
            popup_html = self._create_cell_popup(cell, {value_column: actual_value})

            folium.Polygon(
                locations=boundary_coords,
                color=kwargs.get("border_color", "black"),
                weight=kwargs.get("border_weight", 1),
                opacity=kwargs.get("border_opacity", 0.8),
                fillColor=hex_color,
                fillOpacity=kwargs.get("fill_opacity", 0.7),
                popup=folium.Popup(popup_html, max_width=400),
                tooltip=f"{value_column}: {actual_value:.2f}",
            ).add_to(m)

        # Add colorbar legend
        self._add_colorbar_legend(m, min_val, max_val, color_scheme, value_column)

    def _add_simple_overlay(self, m: "folium.Map", **kwargs: Any) -> None:
        """Add simple colored overlay to map."""
        cell_color = kwargs.get("cell_color", "blue")

        for cell in self.grid.cells:
            boundary_coords = [[lat, lng] for lat, lng in cell.boundary]
            popup_html = self._create_cell_popup(cell)

            folium.Polygon(
                locations=boundary_coords,
                color=kwargs.get("border_color", cell_color),
                weight=kwargs.get("border_weight", 2),
                opacity=kwargs.get("border_opacity", 0.8),
                fillColor=cell_color,
                fillOpacity=kwargs.get("fill_opacity", 0.3),
                popup=folium.Popup(popup_html, max_width=400),
                tooltip=f"H3: {cell.index}",
            ).add_to(m)

    def _create_cell_popup(self, cell: H3Cell, extra_data: dict | None = None) -> str:
        """Create HTML popup for cell."""
        popup_html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h4>H3 Cell Information</h4>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td><b>Index:</b></td><td>{cell.index}</td></tr>
                <tr><td><b>Resolution:</b></td><td>{cell.resolution}</td></tr>
                <tr><td><b>Coordinates:</b></td><td>({cell.latitude:.6f}, {cell.longitude:.6f})</td></tr>
                <tr><td><b>Area:</b></td><td>{cell.area_km2:.6f} km²</td></tr>
        """

        # Add custom properties
        if cell.properties:
            popup_html += "<tr><td colspan='2'><b>Properties:</b></td></tr>"
            for key, value in cell.properties.items():
                popup_html += f"<tr><td>{key}:</td><td>{value}</td></tr>"

        # Add extra data
        if extra_data:
            for key, value in extra_data.items():
                popup_html += f"<tr><td><b>{key}:</b></td><td>{value:.4f}</td></tr>"

        popup_html += "</table></div>"
        return popup_html

    def _add_colorbar_legend(
        self,
        m: "folium.Map",
        min_val: float,
        max_val: float,
        color_scheme: str,
        value_column: str,
    ) -> None:
        """Add colorbar legend to map."""
        # Create colorbar HTML
        colorbar_html = f"""
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <p><b>{value_column}</b></p>
            <p>Min: {min_val:.2f}</p>
            <p>Max: {max_val:.2f}</p>
            <p>Scheme: {color_scheme}</p>
        </div>
        """
        m.get_root().html.add_child(  # type: ignore[attr-defined]
            folium.Element(colorbar_html)
        )

    def _add_statistics_control(self, m: "folium.Map") -> None:
        """Add grid statistics control to map."""
        stats = self.analytics.basic_statistics()

        stats_html = f"""
        <div style="position: fixed; 
                    bottom: 10px; left: 10px; width: 250px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
            <h4>Grid Statistics</h4>
            <p><b>Name:</b> {self.grid.name}</p>
            <p><b>Cells:</b> {stats.get("cell_count", 0)}</p>
            <p><b>Total Area:</b> {stats.get("total_area_km2", 0):.2f} km²</p>
            <p><b>Resolutions:</b> {list(stats.get("resolution_distribution", {}).keys())}</p>
        </div>
        """
        m.get_root().html.add_child(  # type: ignore[attr-defined]
            folium.Element(stats_html)
        )

    def create_heatmap(self, value_column: str, **kwargs: Any) -> "folium.Map":
        """
        Create heatmap visualization using cell centroids.

        Args:
            value_column: Column name for heat values
            **kwargs: Additional styling parameters

        Returns:
            Folium map with heatmap layer
        """
        if not FOLIUM_AVAILABLE:
            raise ImportError(
                "folium package required. Install with 'uv pip install folium'"
            )

        # Prepare heat data
        heat_data = []
        for cell in self.grid.cells:
            value = cell.properties.get(value_column, 0)
            if value > 0:  # Only include positive values
                heat_data.append([cell.latitude, cell.longitude, value])

        if not heat_data:
            logger.warning(
                f"No positive values found for heatmap column '{value_column}'"
            )
            return self.create_folium_map(**kwargs)

        # Create base map
        center_lat, center_lng = self.grid.center()
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=kwargs.get("zoom_start", 10),
            tiles=kwargs.get("tiles", "OpenStreetMap"),
        )

        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            radius=kwargs.get("radius", 15),
            blur=kwargs.get("blur", 15),
            max_zoom=kwargs.get("max_zoom", 18),
        ).add_to(m)

        return m


class H3StaticVisualizer:
    """
    Static plot visualizations for H3 grids using Matplotlib and Seaborn.

    Provides methods for creating publication-quality static plots and charts
    for H3 grid analysis and visualization.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize static visualizer for an H3Grid.

        Args:
            grid: H3Grid instance to visualize
        """
        self.grid = grid
        self.analytics = H3Analytics(grid)

    def plot_grid_overview(
        self,
        figsize: tuple[int, int] = (12, 8),
        save_path: str | None = None,
    ) -> Any | None:
        """
        Create comprehensive grid overview plot.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available. Cannot create static plots.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"H3 Grid Overview: {self.grid.name}", fontsize=16, fontweight="bold"
        )

        # 1. Grid spatial distribution
        self._plot_spatial_distribution(axes[0, 0])

        # 2. Resolution distribution
        self._plot_resolution_distribution(axes[0, 1])

        # 3. Area distribution
        self._plot_area_distribution(axes[1, 0])

        # 4. Grid statistics table
        self._plot_statistics_table(axes[1, 1])

        if MATPLOTLIB_AVAILABLE:
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Grid overview saved to {save_path}")

        return fig

    def _plot_spatial_distribution(self, ax: Any) -> None:
        """Plot spatial distribution of cells."""
        if not self.grid.cells:
            ax.text(
                0.5,
                0.5,
                "No cells to display",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Spatial Distribution")
            return

        # Extract coordinates
        lats = [cell.latitude for cell in self.grid.cells]
        lngs = [cell.longitude for cell in self.grid.cells]

        # Create scatter plot
        scatter = ax.scatter(
            lngs, lats, c=range(len(self.grid.cells)), cmap="viridis", alpha=0.7, s=50
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Spatial Distribution")
        ax.grid(True, alpha=0.3)

        # Add colorbar
        if MATPLOTLIB_AVAILABLE:
            plt.colorbar(scatter, ax=ax, label="Cell Index")

    def _plot_resolution_distribution(self, ax: Any) -> None:
        """Plot resolution distribution."""
        stats = self.analytics.resolution_analysis()

        if not stats.get("resolution_counts"):
            ax.text(
                0.5,
                0.5,
                "No resolution data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Resolution Distribution")
            return

        resolutions = list(stats["resolution_counts"].keys())
        counts = list(stats["resolution_counts"].values())

        bars = ax.bar(resolutions, counts, color="skyblue", alpha=0.7, edgecolor="navy")
        ax.set_xlabel("H3 Resolution")
        ax.set_ylabel("Number of Cells")
        ax.set_title("Resolution Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{count}",
                ha="center",
                va="bottom",
            )

    def _plot_area_distribution(self, ax: Any) -> None:
        """Plot area distribution histogram."""
        if not self.grid.cells:
            ax.text(
                0.5,
                0.5,
                "No area data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Area Distribution")
            return

        areas = [cell.area_km2 for cell in self.grid.cells]

        ax.hist(
            areas,
            bins=min(20, len(areas)),
            color="lightgreen",
            alpha=0.7,
            edgecolor="darkgreen",
        )
        ax.set_xlabel("Cell Area (km²)")
        ax.set_ylabel("Frequency")
        ax.set_title("Area Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics text
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_area:.4f} km²\nStd: {std_area:.4f} km²",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _plot_statistics_table(self, ax: Any) -> None:
        """Plot statistics as table."""
        stats = self.analytics.basic_statistics()

        # Prepare table data
        table_data = [
            ["Metric", "Value"],
            ["Total Cells", f"{stats.get('cell_count', 0):,}"],
            ["Total Area", f"{stats.get('total_area_km2', 0):.2f} km²"],
            ["Mean Area", f"{stats.get('mean_area_km2', 0):.4f} km²"],
            ["Std Area", f"{stats.get('std_area_km2', 0):.4f} km²"],
            ["Resolutions", f"{stats.get('unique_resolutions', 0)}"],
            ["Grid Name", f"{self.grid.name}"],
        ]

        # Create table
        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style table
        for i in range(len(table_data)):
            if i == 0:  # Header
                table[(i, 0)].set_facecolor("#4CAF50")
                table[(i, 1)].set_facecolor("#4CAF50")
                table[(i, 0)].set_text_props(weight="bold", color="white")
                table[(i, 1)].set_text_props(weight="bold", color="white")
            else:
                table[(i, 0)].set_facecolor("#f0f0f0")
                table[(i, 1)].set_facecolor("#ffffff")

        ax.axis("off")
        ax.set_title("Grid Statistics")

    def plot_hexagon_grid(
        self,
        value_column: str | None = None,
        figsize: tuple[int, int] = (12, 10),
        save_path: str | None = None,
    ) -> Any | None:
        """
        Plot actual hexagonal grid with proper hexagon shapes.

        Args:
            value_column: Column for color coding
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available. Cannot create hexagon plot.")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        if not self.grid.cells:
            ax.text(
                0.5,
                0.5,
                "No cells to display",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("H3 Hexagonal Grid")
            return fig

        # Prepare colors
        if value_column:
            values = [cell.properties.get(value_column, 0) for cell in self.grid.cells]
            if any(v != 0 for v in values):
                norm = Normalize(vmin=min(values), vmax=max(values))
                cmap = plt.cm.viridis  # type: ignore[attr-defined]
                colors = [cmap(norm(v)) for v in values]
            else:
                colors = ["lightblue"] * len(self.grid.cells)
        else:
            colors = ["lightblue"] * len(self.grid.cells)

        # Plot hexagons
        for cell, color in zip(self.grid.cells, colors):
            # Get boundary coordinates
            boundary = cell_to_boundary(cell.index)

            # Convert to plot coordinates (lng, lat)
            hex_coords = [(lng, lat) for lat, lng in boundary]

            # Create hexagon patch
            hexagon = patches.Polygon(
                hex_coords,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.add_patch(hexagon)

        # Set axis properties
        bounds = self.grid.bounds()
        ax.set_xlim(bounds[1] - 0.01, bounds[3] + 0.01)  # lng bounds with padding
        ax.set_ylim(bounds[0] - 0.01, bounds[2] + 0.01)  # lat bounds with padding

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"H3 Hexagonal Grid: {self.grid.name}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # Add colorbar if using values
        if (
            value_column
            and any(
                cell.properties.get(value_column, 0) != 0 for cell in self.grid.cells
            )
            and MATPLOTLIB_AVAILABLE
        ):
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(value_column)

        if save_path and MATPLOTLIB_AVAILABLE:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Hexagon grid saved to {save_path}")

        return fig

    def plot_connectivity_analysis(
        self,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> Any | None:
        """
        Plot connectivity analysis results.

        Args:
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available. Cannot create connectivity plot.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            f"Connectivity Analysis: {self.grid.name}", fontsize=14, fontweight="bold"
        )

        connectivity = self.analytics.connectivity_analysis()

        if not connectivity:
            for ax in [ax1, ax2]:
                ax.text(
                    0.5,
                    0.5,
                    "No connectivity data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return fig

        # Plot 1: Connectivity metrics
        metrics = [
            "Total Adjacencies",
            "Average Neighbors",
            "Isolated Cells",
            "Connectivity Ratio",
        ]
        values = [
            connectivity.get("total_adjacencies", 0),
            connectivity.get("average_neighbors", 0),
            connectivity.get("isolated_cells", 0),
            connectivity.get("connectivity_ratio", 0),
        ]

        bars = ax1.bar(
            metrics, values, color=["skyblue", "lightgreen", "salmon", "gold"]
        )
        ax1.set_ylabel("Value")
        ax1.set_title("Connectivity Metrics")
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Plot 2: Neighbor distribution
        if "cell_neighbors" in connectivity:
            neighbor_counts = [
                len(neighbors) for neighbors in connectivity["cell_neighbors"].values()
            ]

            if neighbor_counts:
                ax2.hist(
                    neighbor_counts,
                    bins=range(max(neighbor_counts) + 2),
                    color="lightcoral",
                    alpha=0.7,
                    edgecolor="darkred",
                )
                ax2.set_xlabel("Number of Neighbors")
                ax2.set_ylabel("Number of Cells")
                ax2.set_title("Neighbor Count Distribution")
                ax2.grid(True, alpha=0.3, axis="y")

        if MATPLOTLIB_AVAILABLE:
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Connectivity analysis saved to {save_path}")

        return fig


class H3InteractiveVisualizer:
    """
    Interactive visualizations using Plotly for H3 grids.

    Provides advanced interactive plots with zoom, pan, and hover capabilities
    for detailed H3 grid exploration and analysis.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize interactive visualizer for an H3Grid.

        Args:
            grid: H3Grid instance to visualize
        """
        self.grid = grid
        self.analytics = H3Analytics(grid)

    def create_plotly_map(
        self, value_column: str | None = None, **kwargs: Any
    ) -> "go.Figure":
        """
        Create interactive Plotly map with H3 hexagons.

        Args:
            value_column: Column for color coding
            **kwargs: Additional styling parameters

        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly package required. Install with 'uv pip install plotly'"
            )

        if not self.grid.cells:
            fig = go.Figure()
            fig.update_layout(title="Empty H3 Grid")
            return fig

        # Prepare data
        lats, lngs, texts, values = [], [], [], []

        for cell in self.grid.cells:
            lats.append(cell.latitude)
            lngs.append(cell.longitude)

            # Create hover text
            text = f"H3: {cell.index}<br>Resolution: {cell.resolution}<br>Area: {cell.area_km2:.6f} km²"
            if cell.properties:
                for key, val in cell.properties.items():
                    text += f"<br>{key}: {val}"
            texts.append(text)

            # Get value for coloring
            if value_column:
                values.append(cell.properties.get(value_column, 0))
            else:
                values.append(1)

        # Create scatter plot
        if value_column and any(v != 0 for v in values):
            fig = go.Figure(
                data=go.Scattermapbox(
                    lat=lats,
                    lon=lngs,
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("marker_size", 15),
                        color=values,
                        colorscale=kwargs.get("colorscale", "Viridis"),
                        showscale=True,
                        colorbar=dict(title=value_column),
                    ),
                    text=texts,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        else:
            fig = go.Figure(
                data=go.Scattermapbox(
                    lat=lats,
                    lon=lngs,
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("marker_size", 15),
                        color=kwargs.get("marker_color", "blue"),
                    ),
                    text=texts,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        # Update layout
        center_lat, center_lng = self.grid.center()

        fig.update_layout(
            mapbox_style=kwargs.get("mapbox_style", "open-street-map"),
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lng), zoom=kwargs.get("zoom", 10)
            ),
            title=f"Interactive H3 Grid: {self.grid.name}",
            height=kwargs.get("height", 600),
        )

        return fig

    def create_dashboard(self, **kwargs: Any) -> "go.Figure":
        """
        Create comprehensive dashboard with multiple views.

        Args:
            **kwargs: Styling parameters

        Returns:
            Plotly figure with subplots
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly package required. Install with 'uv pip install plotly'"
            )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Spatial Distribution",
                "Resolution Distribution",
                "Area Distribution",
                "Statistics Summary",
            ),
            specs=[
                [{"type": "scattermapbox"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "table"}],
            ],
        )

        if not self.grid.cells:
            fig.update_layout(title="Empty H3 Grid Dashboard")
            return fig

        # 1. Spatial distribution (map)
        lats = [cell.latitude for cell in self.grid.cells]
        lngs = [cell.longitude for cell in self.grid.cells]

        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lngs,
                mode="markers",
                marker=dict(size=8, color="blue"),
                name="H3 Cells",
            ),
            row=1,
            col=1,
        )

        # 2. Resolution distribution
        stats = self.analytics.resolution_analysis()
        if stats.get("resolution_counts"):
            resolutions = list(stats["resolution_counts"].keys())
            counts = list(stats["resolution_counts"].values())

            fig.add_trace(
                go.Bar(x=resolutions, y=counts, name="Resolution Count"), row=1, col=2
            )

        # 3. Area distribution
        areas = [cell.area_km2 for cell in self.grid.cells]
        fig.add_trace(
            go.Histogram(x=areas, name="Area Distribution", nbinsx=20), row=2, col=1
        )

        # 4. Statistics table
        basic_stats = self.analytics.basic_statistics()
        table_data = [
            ["Metric", "Value"],
            ["Total Cells", f"{basic_stats.get('cell_count', 0):,}"],
            ["Total Area (km²)", f"{basic_stats.get('total_area_km2', 0):.2f}"],
            ["Mean Area (km²)", f"{basic_stats.get('mean_area_km2', 0):.4f}"],
            ["Unique Resolutions", f"{basic_stats.get('unique_resolutions', 0)}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=table_data[0], fill_color="lightblue"),
                cells=dict(values=list(zip(*table_data[1:])), fill_color="white"),
            ),
            row=2,
            col=2,
        )

        # Update layout
        center_lat, center_lng = self.grid.center()

        fig.update_layout(
            title=f"H3 Grid Dashboard: {self.grid.name}",
            height=800,
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lng),
                zoom=10,
            ),
        )

        return fig


class H3AnimationVisualizer:
    """
    Animation visualizations for H3 grids showing temporal changes.

    Provides methods for creating animated visualizations of H3 grid evolution,
    temporal analysis, and dynamic spatial patterns.
    """

    def __init__(self, grids: list[H3Grid]) -> None:
        """
        Initialize animation visualizer for multiple H3Grids.

        Args:
            grids: List of H3Grid instances representing time series
        """
        self.grids = grids
        self.timestamps = list(range(len(grids)))

    def create_temporal_animation(
        self,
        value_column: str,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> "go.Figure":
        """
        Create animated visualization showing temporal changes.

        Args:
            value_column: Column name for animation values
            save_path: Optional path to save HTML animation
            **kwargs: Animation parameters

        Returns:
            Plotly animated figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly package required. Install with 'uv pip install plotly'"
            )

        if not self.grids:
            fig = go.Figure()
            fig.update_layout(title="No grids for animation")
            return fig

        # Prepare animation frames
        frames = []

        for i, grid in enumerate(self.grids):
            if not grid.cells:
                continue

            lats = [cell.latitude for cell in grid.cells]
            lngs = [cell.longitude for cell in grid.cells]
            values = [cell.properties.get(value_column, 0) for cell in grid.cells]
            texts = [
                f"Time: {i}<br>H3: {cell.index}<br>{value_column}: {cell.properties.get(value_column, 0)}"
                for cell in grid.cells
            ]

            frame = go.Frame(
                data=[
                    go.Scattermapbox(
                        lat=lats,
                        lon=lngs,
                        mode="markers",
                        marker=dict(
                            size=kwargs.get("marker_size", 15),
                            color=values,
                            colorscale=kwargs.get("colorscale", "Viridis"),
                            cmin=kwargs.get("color_min", min(values) if values else 0),
                            cmax=kwargs.get("color_max", max(values) if values else 1),
                            showscale=True,
                            colorbar=dict(title=value_column),
                        ),
                        text=texts,
                        hovertemplate="%{text}<extra></extra>",
                    )
                ],
                name=f"frame_{i}",
            )
            frames.append(frame)

        # Create initial figure
        if frames:
            fig = go.Figure(data=frames[0].data, frames=frames)
        else:
            fig = go.Figure()

        # Add animation controls
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": kwargs.get("frame_duration", 500),
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": kwargs.get(
                                            "transition_duration", 300
                                        )
                                    },
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Time Step:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{i}"],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i in range(len(frames))
                    ],
                }
            ],
        )

        # Update map layout
        if self.grids and self.grids[0].cells:
            center_lat, center_lng = self.grids[0].center()

            fig.update_layout(
                mapbox_style=kwargs.get("mapbox_style", "open-street-map"),
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lng),
                    zoom=kwargs.get("zoom", 10),
                ),
                title=f"H3 Grid Animation: {value_column}",
                height=kwargs.get("height", 600),
            )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Animation saved to {save_path}")

        return fig
