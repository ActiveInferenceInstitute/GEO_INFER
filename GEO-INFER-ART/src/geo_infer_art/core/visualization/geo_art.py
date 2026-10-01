"""
GeoArt module for artistic visualization of geospatial data.
"""

import logging
import os
import threading
import time
from typing import (
    TYPE_CHECKING,
    Union,
    Any,
)
from collections.abc import Callable, Iterator

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from geo_infer_art.core.aesthetics import ColorPalette
from geo_infer_art.utils.animation import save_animation_with_fallback
from geo_infer_art.utils.validators import validate_file_path, validate_geospatial_data

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from geo_infer_art.core.visualization.map_styling import MapStyle

# Optional imports for advanced features
try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import folium
    from folium.plugins import MarkerCluster  # noqa: F401

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import mayavi.mlab as mlab
    import mayavi  # noqa: F401

    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False


class GeoArt:
    """
    A class for creating artistic visualizations of geospatial data.

    The GeoArt class provides methods for loading, transforming, and
    visualizing geospatial data with artistic elements and aesthetic
    considerations.

    Attributes:
        data: The geospatial data as a GeoDataFrame or raster array
        metadata: Additional information about the data
        crs: Coordinate reference system of the data
    """

    def __init__(
        self,
        data: gpd.GeoDataFrame | np.ndarray | None = None,
        metadata: dict | None = None,
        crs: str | None = "EPSG:4326",
    ):
        """
        Initialize a GeoArt object.

        Args:
            data: Geospatial data as a GeoDataFrame or numpy array (for raster)
            metadata: Additional information about the data
            crs: Coordinate reference system identifier
        """
        self.data = data
        self.metadata = metadata or {}
        self.crs = crs
        self._figure: Figure | None = None
        self._ax: Axes | None = None

    @classmethod
    def load_geojson(cls, file_path: str) -> "GeoArt":
        """
        Load geospatial data from a GeoJSON file.

        Args:
            file_path: Path to the GeoJSON file

        Returns:
            A new GeoArt object with the loaded data

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid GeoJSON
        """
        validate_file_path(file_path, [".geojson", ".json"])

        try:
            data = gpd.read_file(file_path)
            metadata = {
                "source": file_path,
                "type": "vector",
                "features": len(data),
                "attributes": list(data.columns),
            }
            return cls(data=data, metadata=metadata, crs=data.crs)
        except Exception as e:
            raise ValueError(f"Failed to load GeoJSON: {str(e)}") from e

    @classmethod
    def load_raster(cls, file_path: str) -> "GeoArt":
        """
        Load geospatial data from a raster file (e.g., GeoTIFF).

        Args:
            file_path: Path to the raster file

        Returns:
            A new GeoArt object with the loaded data

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid raster
        """
        import rasterio

        validate_file_path(file_path, [".tif", ".tiff", ".jpg", ".png"])

        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                metadata = {
                    "source": file_path,
                    "type": "raster",
                    "shape": data.shape,
                    "bounds": src.bounds,
                    "transform": src.transform,
                }
                return cls(data=data, metadata=metadata, crs=src.crs.to_string())
        except Exception as e:
            raise ValueError(f"Failed to load raster: {str(e)}") from e

    def apply_style(
        self,
        style: str = "default",
        color_palette: str | ColorPalette | None = None,
        line_width: float = 1.0,
        alpha: float = 0.8,
        background_color: str = "white",
        map_style: Union[str, "MapStyle"] | None = None,
        legend: bool = False,
        title: str | None = None,
    ) -> "GeoArt":
        """
        Apply an artistic style to the geospatial data.

        Args:
            style: Name of the style to apply
            color_palette: Color palette name or ColorPalette object
            line_width: Width of lines for vector data
            alpha: Transparency level (0.0 to 1.0)
            background_color: Background color of the visualization
            map_style: MapStyle name or MapStyle object for advanced styling
            legend: Whether to add a legend to the visualization
            title: Custom title for the visualization

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the data is not loaded or the style is invalid
        """
        if self.data is None:
            raise ValueError("No data loaded. Load data first.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not np.isfinite(line_width) or line_width <= 0:
            raise ValueError("line_width must be finite and positive")

        validate_geospatial_data(self.data)

        # Handle color palette
        if isinstance(color_palette, str):
            palette = ColorPalette.get_palette(color_palette)
        elif isinstance(color_palette, ColorPalette):
            palette = color_palette
        elif color_palette is None:
            # Default palettes for different styles
            style_palettes = {
                "default": "viridis",
                "watercolor": "pastel",
                "topographic": "earth",
                "neon": "bright",
                "minimal": "grayscale",
                "blueprint": "blue",
            }
            palette_name = style_palettes.get(style, "viridis")
            palette = ColorPalette.get_palette(palette_name)
        else:
            raise ValueError(f"Unsupported color_palette type: {type(color_palette)}")

        # Handle map_style parameter
        if isinstance(map_style, str):
            from geo_infer_art.core.visualization.map_styling import MapStyle

            map_style_obj = MapStyle(name=map_style)
        elif map_style is not None:
            map_style_obj = map_style
        else:
            map_style_obj = None

        # Keep repeated rendering calls bounded in long-running analysis jobs.
        # Matplotlib otherwise emits a figure-leak warning after 20 figures.
        if len(plt.get_fignums()) >= int(plt.rcParams["figure.max_open_warning"]):
            plt.close("all")

        # Create figure and apply style
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=background_color)

        if isinstance(self.data, gpd.GeoDataFrame):
            # Vector data visualization
            plot_kwargs = {
                "ax": ax,
                "cmap": palette.cmap,
                "linewidth": line_width,
                "alpha": alpha,
            }

            # Add legend if requested
            if legend and len(self.data.columns) > 1:
                # Try to use a categorical column for legend
                categorical_cols = self.data.select_dtypes(include=["object"]).columns
                if len(categorical_cols) > 0:
                    plot_kwargs["legend"] = True
                    plot_kwargs["c"] = categorical_cols[0]

            self.data.plot(**plot_kwargs)

        else:
            # Raster data visualization
            if self.data.ndim == 3 and self.data.shape[0] == 3:
                # RGB image
                rgb = np.dstack([self.data[0], self.data[1], self.data[2]])
                ax.imshow(rgb)
            else:
                # Single band or other raster
                ax.imshow(self.data, cmap=palette.cmap, alpha=alpha)

        # Apply MapStyle if provided
        if map_style_obj is not None:
            map_style_obj.apply_to_axes(ax)

        # Apply style-specific settings
        if style == "watercolor":
            ax.patch.set_alpha(0.3)
            ax.grid(False)
        elif style == "minimal":
            ax.axis("off")
            ax.grid(False)
        elif style == "blueprint":
            ax.set_facecolor("#072448")
            ax.grid(True, color="#1E88E5", alpha=0.3, linestyle="-")

        # Set title
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f"Artistic Visualization - {style.capitalize()}")

        self._figure = fig
        self._ax = ax

        return self

    def save(self, output_path: str, dpi: int = 300) -> str:
        """
        Save the visualization to a file.

        Args:
            output_path: Path where the file should be saved
            dpi: Resolution for the output image

        Returns:
            The path to the saved file

        Raises:
            ValueError: If no visualization has been created
        """
        if self._figure is None:
            raise ValueError("No visualization to save. Apply a style first.")

        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        self._figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(self._figure)
        return output_path

    def show(self) -> None:
        """
        Display the visualization.

        Raises:
            ValueError: If no visualization has been created
        """
        if self._figure is None:
            raise ValueError("No visualization to show. Apply a style first.")

        if "agg" in plt.get_backend().lower() or not plt.isinteractive():
            self._figure.canvas.draw()
            return
        plt.show()

    def create_animation(
        self,
        output_path: str,
        style_sequence: list[str],
        duration: float = 5.0,
        fps: int = 24,
        **kwargs: Any,
    ) -> str:
        """
        Create an animated visualization cycling through different styles.

        Args:
            output_path: Path for the output animation file
            style_sequence: List of style names to cycle through
            duration: Duration of the animation in seconds
            fps: Frames per second
            **kwargs: Additional parameters for apply_style

        Returns:
            Path to the created animation file

        Raises:
            ValueError: If no data is loaded or animation cannot be created
        """
        import matplotlib.animation as animation

        if self.data is None:
            raise ValueError("No data loaded. Load data first.")
        if not style_sequence:
            raise ValueError("style_sequence must contain at least one style")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")

        # Create frames for each style
        frames: list[Any] = []
        for style in style_sequence:
            # Apply style and capture the figure
            self.apply_style(style=style, **kwargs)
            frames.append(self._figure)

        # Create animation
        def animate(frame_num: int) -> Any:
            return frames[frame_num % len(frames)]

        # Calculate number of frames
        num_frames = int(duration * fps)

        # Create the animation
        anim = animation.FuncAnimation(
            frames[0], animate, frames=num_frames, interval=1000 / fps, blit=False
        )

        return save_animation_with_fallback(anim, output_path, fps)

    def add_interactive_elements(self, interactive_type: str = "zoom") -> "GeoArt":
        """
        Add interactive elements to the visualization.

        Args:
            interactive_type: Type of interactivity ("zoom", "pan", "hover", "click")

        Returns:
            Self for method chaining
        """
        if self._figure is None:
            raise ValueError("No visualization to enhance. Apply a style first.")
        assert self._ax is not None

        # Real implementation using matplotlib interactive features and mplcursors
        if interactive_type == "zoom":
            if (
                hasattr(self._figure.canvas, "toolbar")
                and self._figure.canvas.toolbar is not None
            ):
                self._figure.canvas.toolbar.zoom()
            else:
                self._ax.set_navigate(True)
        elif interactive_type == "pan":
            if (
                hasattr(self._figure.canvas, "toolbar")
                and self._figure.canvas.toolbar is not None
            ):
                self._figure.canvas.toolbar.pan()
            else:
                self._ax.set_navigate(True)
        elif interactive_type in ["hover", "click"]:
            try:
                import mplcursors

                hover = interactive_type == "hover"
                cursor = mplcursors.cursor(self._ax, hover=hover)

                @cursor.connect("add")  # type: ignore[misc]
                def on_add(sel: Any) -> None:
                    sel.annotation.get_bbox_patch().set(
                        boxstyle="round,pad=0.5", alpha=0.9, color="white"
                    )

            except ImportError:
                import logging

                logging.getLogger(__name__).warning(
                    "mplcursors not installed. Cannot enable hover/click interactivity."
                )

        return self

    def export_svg(self, output_path: str) -> str:
        """
        Export the visualization as an SVG file.

        Args:
            output_path: Path for the SVG output file

        Returns:
            Path to the exported SVG file

        Raises:
            ValueError: If no visualization has been created
        """
        if self._figure is None:
            raise ValueError("No visualization to export. Apply a style first.")

        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        self._figure.savefig(output_path, format="svg", bbox_inches="tight")
        return output_path

    def get_colorbar(self, label: str = "Value") -> "GeoArt":
        """
        Add a colorbar to the visualization.

        Args:
            label: Label for the colorbar

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no visualization has been created
        """
        if self._figure is None:
            raise ValueError("No visualization to enhance. Apply a style first.")
        assert self._ax is not None

        # Add colorbar for raster data or styled vector data
        if not isinstance(self.data, gpd.GeoDataFrame):
            plt.colorbar(
                self._ax.images[0] if self._ax.images else None,
                ax=self._ax,
                label=label,
            )

        return self

    def set_projection(self, projection: str = "plate_carree") -> "GeoArt":
        """
        Set the map projection for the visualization.

        Args:
            projection: Map projection name (e.g., "plate_carree", "mollweide", "robinson")

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the projection is not supported
        """
        # This would require cartopy for proper map projections
        # For now, just store the preference
        self.metadata["projection"] = projection
        return self

    def add_annotations(self, annotations: list[dict]) -> "GeoArt":
        """
        Add text or graphical annotations to the visualization.

        Args:
            annotations: List of annotation dictionaries with keys:
                        'text', 'x', 'y', 'color', 'fontsize', etc.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no visualization has been created
        """
        if self._figure is None:
            raise ValueError("No visualization to annotate. Apply a style first.")
        assert self._ax is not None

        for annotation in annotations:
            text = annotation.get("text", "")
            x = annotation.get("x", 0)
            y = annotation.get("y", 0)
            color = annotation.get("color", "black")
            fontsize = annotation.get("fontsize", 12)

            self._ax.annotate(
                text,
                xy=(x, y),
                color=color,
                fontsize=fontsize,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        return self

    def apply_filter(self, filter_type: str, **kwargs: Any) -> "GeoArt":
        """
        Apply a spatial or visual filter to the data.

        Args:
            filter_type: Type of filter ("spatial", "attribute", "visual")
            **kwargs: Filter parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the filter type is not supported
        """
        if filter_type == "spatial":
            # Apply spatial filter (e.g., bounding box, buffer)
            if "bounds" in kwargs:
                minx, miny, maxx, maxy = kwargs["bounds"]
                if isinstance(self.data, gpd.GeoDataFrame):
                    self.data = self.data.cx[minx:maxx, miny:maxy]

        elif filter_type == "attribute":
            # Apply attribute filter
            if isinstance(self.data, gpd.GeoDataFrame):
                filtered: gpd.GeoDataFrame = self.data
                for column, value in kwargs.items():
                    if column in filtered.columns:
                        if isinstance(value, (list, tuple)):
                            filtered = filtered[filtered[column].isin(value)]
                        else:
                            filtered = filtered[filtered[column] == value]
                self.data = filtered

        elif filter_type == "visual":
            # Apply visual filter (e.g., opacity, brightness)
            if "opacity" in kwargs:
                opacity = float(kwargs["opacity"])
                opacity = max(0.0, min(1.0, opacity))
                self.metadata["opacity"] = opacity
                if self._figure is not None:
                    for ax in self._figure.get_axes():
                        for child in ax.get_children():
                            if hasattr(child, "set_alpha"):
                                child.set_alpha(opacity)

        return self

    def create_realtime_visualization(
        self,
        data_callback: Callable[[], gpd.GeoDataFrame | np.ndarray],
        update_interval: float = 1.0,
        style: str = "default",
        max_updates: int | None = None,
        output_file: str | None = None,
        **kwargs: Any,
    ) -> "RealtimeVisualization":
        """
        Create a real-time visualization that updates with live data.

        Args:
            data_callback: Function that returns updated geospatial data
            update_interval: Time between updates in seconds
            style: Visualization style to apply
            max_updates: Maximum number of updates (None for unlimited)
            output_file: Optional file to save the final visualization
            **kwargs: Additional styling parameters

        Returns:
            RealtimeVisualization object for managing the live visualization

        Raises:
            ValueError: If no data callback is provided
        """
        if data_callback is None:
            raise ValueError(
                "Data callback function is required for real-time visualization"
            )

        realtime_viz = RealtimeVisualization(
            geo_art=self,
            data_callback=data_callback,
            update_interval=update_interval,
            style=style,
            max_updates=max_updates,
            output_file=output_file,
            **kwargs,
        )

        return realtime_viz

    def create_3d_visualization(
        self,
        elevation_data: np.ndarray | None = None,
        z_column: str | None = None,
        **kwargs: Any,
    ) -> "GeoArt3D":
        """
        Create a 3D visualization of the geospatial data.

        Args:
            elevation_data: Optional elevation data for 3D terrain
            z_column: Column name to use for Z-axis values (for vector data)
            **kwargs: Additional parameters for 3D visualization

        Returns:
            GeoArt3D object for 3D visualization

        Raises:
            ValueError: If 3D visualization libraries are not available
        """
        if not MAYAVI_AVAILABLE and not PLOTLY_AVAILABLE:
            raise ValueError(
                "3D visualization requires either mayavi or plotly. "
                "Install with: uv pip install mayavi or pip install plotly"
            )

        return GeoArt3D(
            geo_art=self, elevation_data=elevation_data, z_column=z_column, **kwargs
        )

    def create_interactive_web_map(
        self,
        output_file: str = "interactive_map.html",
        tiles: str = "OpenStreetMap",
        **kwargs: Any,
    ) -> str:
        """
        Create an interactive web-based map using Folium.

        Args:
            output_file: Path for the HTML output file
            tiles: Map tiles to use ("OpenStreetMap", "Stamen Terrain", etc.)
            **kwargs: Additional parameters for the web map

        Returns:
            Path to the created HTML file

        Raises:
            ValueError: If folium is not available or data is not suitable
        """
        if not FOLIUM_AVAILABLE:
            raise ValueError(
                "Interactive web maps require folium. Install with: uv pip install folium"
            )

        if self.data is None:
            raise ValueError("No data loaded for web map creation")

        # Create base map
        if isinstance(self.data, gpd.GeoDataFrame) and len(self.data) > 0:
            # Get center point from data bounds
            bounds = self.data.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
        else:
            # Default to world center
            center_lat, center_lon = 0, 0

        web_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=kwargs.get("zoom_start", 10),
            tiles=tiles,
        )

        if isinstance(self.data, gpd.GeoDataFrame):
            # Add vector data to map
            for idx, row in self.data.iterrows():
                # Create popup with feature information
                popup_content = f"<b>{row.get('name', f'Feature {idx}')}</b><br>"
                for col in self.data.columns:
                    if col != "geometry" and col in row:
                        popup_content += f"{col}: {row[col]}<br>"

                # Add geometry based on type
                geom = row.geometry
                if geom.geom_type == "Point":
                    folium.CircleMarker(
                        location=[geom.y, geom.x],
                        radius=5,
                        popup=popup_content,
                        color="blue",
                        fill=True,
                        fillColor="blue",
                    ).add_to(web_map)
                elif geom.geom_type in ["LineString", "MultiLineString"]:
                    coords = [(point[1], point[0]) for point in geom.coords]
                    folium.PolyLine(
                        coords, popup=popup_content, color="red", weight=3
                    ).add_to(web_map)
                elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                    coords = [(point[1], point[0]) for point in geom.exterior.coords]
                    folium.Polygon(
                        coords,
                        popup=popup_content,
                        color="green",
                        fill=True,
                        fillOpacity=0.4,
                    ).add_to(web_map)

        # Save the map
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        web_map.save(output_file)
        return output_file

    def create_plotly_visualization(
        self, plot_type: str = "scatter", **kwargs: Any
    ) -> Any:
        """
        Create an interactive Plotly visualization.

        Args:
            plot_type: Type of plot ("scatter", "choropleth", "heatmap", "3d")
            **kwargs: Additional parameters for the plot

        Returns:
            Plotly figure object

        Raises:
            ValueError: If plotly is not available or plot type is not supported
        """
        if not PLOTLY_AVAILABLE:
            raise ValueError(
                "Plotly visualizations require plotly. Install with: uv pip install plotly"
            )

        if self.data is None:
            raise ValueError("No data loaded for plotly visualization")

        if isinstance(self.data, gpd.GeoDataFrame):
            return self._create_plotly_from_geodataframe(plot_type, **kwargs)
        else:
            return self._create_plotly_from_raster(plot_type, **kwargs)

    def _create_plotly_from_geodataframe(self, plot_type: str, **kwargs: Any) -> Any:
        """Create Plotly visualization from GeoDataFrame."""
        if not isinstance(self.data, gpd.GeoDataFrame):
            raise ValueError("Raster data not supported for GeoDataFrame plot")
        if plot_type == "scatter":
            # Create scatter plot on map
            fig = px.scatter_mapbox(
                self.data,
                lat=self.data.geometry.y,
                lon=self.data.geometry.x,
                hover_name=kwargs.get("hover_name", "name"),
                color=kwargs.get("color_column"),
                size=kwargs.get("size_column"),
                zoom=kwargs.get("zoom", 10),
                mapbox_style=kwargs.get("mapbox_style", "carto-positron"),
            )

        elif plot_type == "choropleth":
            # Create choropleth map
            fig = px.choropleth_mapbox(
                self.data,
                geojson=self.data.geometry,
                locations=self.data.index,
                color=kwargs.get("color_column"),
                hover_name=kwargs.get("hover_name", "name"),
                mapbox_style=kwargs.get("mapbox_style", "carto-positron"),
                zoom=kwargs.get("zoom", 10),
            )

        else:
            raise ValueError(f"Unsupported plot type for GeoDataFrame: {plot_type}")

        return fig

    def _create_plotly_from_raster(self, plot_type: str, **kwargs: Any) -> Any:
        """Create Plotly visualization from raster data."""
        if not isinstance(self.data, np.ndarray):
            raise ValueError("GeoDataFrame not supported for raster plot")
        if plot_type == "heatmap":
            # Create heatmap from raster
            fig = go.Figure(data=go.Heatmap(z=self.data))
            fig.update_layout(
                title=kwargs.get("title", "Raster Heatmap"),
                xaxis_title=kwargs.get("x_label", "X"),
                yaxis_title=kwargs.get("y_label", "Y"),
            )

        elif plot_type == "3d":
            # Create 3D surface plot
            x = np.arange(self.data.shape[1])
            y = np.arange(self.data.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = go.Figure(data=[go.Surface(z=self.data, x=X, y=Y)])
            fig.update_layout(
                title=kwargs.get("title", "3D Surface"),
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            )

        else:
            raise ValueError(f"Unsupported plot type for raster: {plot_type}")

        return fig

    def optimize_for_performance(
        self,
        target_resolution: tuple[int, int] | None = None,
        simplify_tolerance: float | None = None,
        cache_data: bool = True,
    ) -> "GeoArt":
        """
        Optimize the visualization for better performance.

        Args:
            target_resolution: Target resolution for raster data
            simplify_tolerance: Tolerance for geometry simplification
            cache_data: Whether to cache processed data

        Returns:
            Optimized GeoArt object

        Raises:
            ValueError: If optimization parameters are invalid
        """
        if self.data is None:
            raise ValueError("No data loaded for optimization")

        # Create optimized copy
        optimized = GeoArt(
            data=self.data.copy(), metadata=self.metadata.copy(), crs=self.crs
        )

        if isinstance(self.data, gpd.GeoDataFrame) and simplify_tolerance:
            # Simplify geometries for better performance
            assert isinstance(optimized.data, gpd.GeoDataFrame)
            optimized.data["geometry"] = optimized.data["geometry"].simplify(
                simplify_tolerance
            )

        if not isinstance(self.data, gpd.GeoDataFrame) and target_resolution:
            # Downsample raster data
            from scipy.ndimage import zoom

            scale_factor = (
                target_resolution[0] / self.data.shape[0],
                target_resolution[1] / self.data.shape[1],
            )
            optimized.data = zoom(self.data, scale_factor, order=1)

        if cache_data:
            # Cache the optimized data
            optimized.metadata["cached"] = True
            optimized.metadata["original_shape"] = (
                self.data.shape
                if not isinstance(self.data, gpd.GeoDataFrame)
                else len(self.data)
            )

        return optimized

    def create_multi_scale_visualization(
        self, scales: list[str] | None = None, **kwargs: Any
    ) -> dict[str, "GeoArt"]:
        """
        Create visualizations at multiple scales.

        Args:
            scales: List of scales to create ("global", "regional", "local")
            **kwargs: Additional parameters for each scale

        Returns:
            Dictionary of GeoArt objects for each scale
        """
        if self.data is None:
            raise ValueError("No data loaded for multi-scale visualization")
        scales = ["global", "regional", "local"] if scales is None else list(scales)
        supported_scales = {"global", "regional", "local"}
        unknown_scales = set(scales) - supported_scales
        if unknown_scales:
            raise ValueError(f"Unsupported scales: {sorted(unknown_scales)}")
        if not scales:
            raise ValueError("scales must contain at least one scale")

        multi_scale_viz = {}

        if isinstance(self.data, gpd.GeoDataFrame):
            # For vector data, create zoomed versions
            bounds = self.data.total_bounds

            for scale in scales:
                if scale == "global":
                    # Show full extent
                    scale_data = self.data.copy()
                    zoom_factor = 1.0
                elif scale == "regional":
                    # Zoom to 50% of extent
                    center_x = (bounds[0] + bounds[2]) / 2
                    center_y = (bounds[1] + bounds[3]) / 2
                    width = (bounds[2] - bounds[0]) * 0.5
                    height = (bounds[3] - bounds[1]) * 0.5

                    scale_data = self.data.cx[
                        center_x - width / 2 : center_x + width / 2,
                        center_y - height / 2 : center_y + height / 2,
                    ]
                    zoom_factor = 2.0
                elif scale == "local":
                    # Zoom to 25% of extent
                    center_x = (bounds[0] + bounds[2]) / 2
                    center_y = (bounds[1] + bounds[3]) / 2
                    width = (bounds[2] - bounds[0]) * 0.25
                    height = (bounds[3] - bounds[1]) * 0.25

                    scale_data = self.data.cx[
                        center_x - width / 2 : center_x + width / 2,
                        center_y - height / 2 : center_y + height / 2,
                    ]
                    zoom_factor = 4.0
                else:
                    continue

                if len(scale_data) > 0:
                    scale_art = GeoArt(data=scale_data, crs=self.crs)
                    scale_art.metadata = {
                        **self.metadata,
                        "scale": scale,
                        "zoom_factor": zoom_factor,
                        "scale_bounds": scale_data.total_bounds,
                    }
                    multi_scale_viz[scale] = scale_art

        return multi_scale_viz

    def apply_custom_algorithm(
        self,
        algorithm_function: Callable,
        algorithm_name: str = "custom",
        **params: Any,
    ) -> "GeoArt":
        """
        Apply a custom algorithm to the geospatial data.

        Args:
            algorithm_function: Custom function to apply (should accept data and return processed data)
            algorithm_name: Name for the custom algorithm
            **params: Parameters to pass to the algorithm function

        Returns:
            GeoArt object with processed data

        Raises:
            ValueError: If algorithm function is invalid
        """
        if self.data is None:
            raise ValueError("No data loaded for custom algorithm")

        if not callable(algorithm_function):
            raise ValueError("Algorithm must be a callable function")

        try:
            # Apply the custom algorithm
            processed_data = algorithm_function(self.data, **params)

            # Create new GeoArt with processed data
            result = GeoArt(data=processed_data, crs=self.crs)
            result.metadata = {
                **self.metadata,
                "custom_algorithm": algorithm_name,
                "algorithm_params": params,
            }

            return result

        except Exception as e:
            raise ValueError(f"Custom algorithm failed: {str(e)}") from e

    def __repr__(self) -> str:
        """Return a string representation of the GeoArt object."""
        if self.data is None:
            return "GeoArt(No data loaded)"

        if isinstance(self.data, gpd.GeoDataFrame):
            return f"GeoArt(Vector data: {len(self.data)} features, CRS: {self.crs})"
        else:
            shape_str = f"{self.data.shape}"
            return f"GeoArt(Raster data: {shape_str}, CRS: {self.crs})"


class RealtimeVisualization:
    """
    A class for managing real-time geospatial visualizations.

    This class handles live updating of visualizations with streaming data,
    providing both matplotlib-based and web-based real-time displays.
    """

    def __init__(
        self,
        geo_art: GeoArt,
        data_callback: Callable[[], gpd.GeoDataFrame | np.ndarray],
        update_interval: float = 1.0,
        style: str = "default",
        max_updates: int | None = None,
        output_file: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize real-time visualization.

        Args:
            geo_art: Base GeoArt object for styling
            data_callback: Function that returns updated data
            update_interval: Time between updates in seconds
            style: Visualization style
            max_updates: Maximum number of updates
            output_file: Optional file to save final visualization
            **kwargs: Additional styling parameters
        """
        self.geo_art = geo_art
        self.data_callback = data_callback
        self.update_interval = update_interval
        self.style = style
        self.max_updates = max_updates
        self.output_file = output_file
        self.kwargs = kwargs

        self.is_running = False
        self.current_data: gpd.GeoDataFrame | np.ndarray | None = None
        self.update_count = 0
        self._thread: threading.Thread | None = None
        self._animation = None

    def start(self, use_threading: bool = True) -> None:
        """
        Start the real-time visualization.

        Args:
            use_threading: Whether to use background threading for updates
        """
        if use_threading:
            self._thread = threading.Thread(target=self._run_updates)
            self._thread.daemon = True
            self._thread.start()
        else:
            self._run_updates()

    def stop(self) -> None:
        """Stop the real-time visualization."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_updates(self) -> None:
        """Run the update loop."""
        self.is_running = True
        self.update_count = 0

        while self.is_running:
            try:
                # Get updated data
                new_data = self.data_callback()

                if new_data is not None:
                    # Update the visualization
                    self.current_data = new_data
                    self._update_visualization()

                    self.update_count += 1

                    # Check if we've reached max updates
                    if self.max_updates and self.update_count >= self.max_updates:
                        break

                # Wait for next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error("Error in real-time update: %s", e)
                break

        self.is_running = False

    def _update_visualization(self) -> None:
        """Update the visualization with new data.

        Re-renders the GeoArt object's figure using the latest data fetched
        by the callback.  If the GeoArt instance already has a matplotlib
        figure, the axes are cleared and redrawn; otherwise a new figure is
        created via the default style.
        """
        if self.current_data is None:
            return

        self.geo_art.data = self.current_data

        if self.geo_art._figure is not None:
            for ax in self.geo_art._figure.get_axes():
                ax.clear()
                if isinstance(self.current_data, gpd.GeoDataFrame):
                    self.current_data.plot(ax=ax)
            self.geo_art._figure.canvas.draw_idle()
        else:
            self.geo_art.apply_style(self.style, **self.kwargs)

    def save_snapshot(self, filename: str | None = None) -> str:
        """
        Save a snapshot of the current visualization.

        Args:
            filename: Optional filename (uses timestamp if not provided)

        Returns:
            Path to saved snapshot
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_snapshot_{timestamp}.png"

        if self.current_data is not None:
            # Create snapshot with current data
            snapshot_art = GeoArt(data=self.current_data, crs=self.geo_art.crs)
            snapshot_art.apply_style(style=self.style, **self.kwargs)
            return snapshot_art.save(filename)

        return ""


class GeoArt3D:
    """
    A class for creating 3D artistic visualizations of geospatial data.

    This class provides methods for creating three-dimensional representations
    of geospatial data using various 3D rendering backends.
    """

    def __init__(
        self,
        geo_art: GeoArt,
        elevation_data: np.ndarray | None = None,
        z_column: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize 3D visualization.

        Args:
            geo_art: Base GeoArt object
            elevation_data: Optional elevation data for terrain
            z_column: Column name for Z-axis values
            **kwargs: Additional 3D parameters
        """
        self.geo_art = geo_art
        self.elevation_data = elevation_data
        self.z_column = z_column
        self.kwargs = kwargs

        self.figure_3d = None

    def create_3d_surface(self, output_file: str | None = None, **kwargs: Any) -> Any:
        """
        Create a 3D surface visualization.

        Args:
            output_file: Optional file to save the 3D visualization
            **kwargs: Additional parameters for 3D surface

        Returns:
            3D figure object

        Raises:
            ValueError: If 3D libraries are not available
        """
        if not PLOTLY_AVAILABLE and not MAYAVI_AVAILABLE:
            raise ValueError("3D visualization requires plotly or mayavi")

        if self.geo_art.data is None:
            raise ValueError("No data loaded for 3D visualization")

        if PLOTLY_AVAILABLE:
            return self._create_plotly_3d_surface(output_file, **kwargs)
        elif MAYAVI_AVAILABLE:
            return self._create_mayavi_3d_surface(output_file, **kwargs)

    def _create_plotly_3d_surface(self, output_file: str | None, **kwargs: Any) -> Any:
        """Create 3D surface using Plotly."""
        assert self.geo_art.data is not None
        if isinstance(self.geo_art.data, gpd.GeoDataFrame):
            # For vector data, create 3D scatter or mesh
            if self.z_column and self.z_column in self.geo_art.data.columns:
                z_values = self.geo_art.data[self.z_column]
            else:
                # Use elevation data or default
                z_values = (
                    self.elevation_data.flatten()
                    if self.elevation_data is not None
                    else np.zeros(len(self.geo_art.data))
                )

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=self.geo_art.data.geometry.x,
                        y=self.geo_art.data.geometry.y,
                        z=z_values,
                        mode="markers",
                        marker=dict(
                            size=5, color=z_values, colorscale="Viridis", opacity=0.8
                        ),
                    )
                ]
            )

        else:
            # For raster data, create 3D surface
            x = np.arange(self.geo_art.data.shape[1])
            y = np.arange(self.geo_art.data.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = go.Figure(
                data=[go.Surface(z=self.geo_art.data, x=X, y=Y, colorscale="Viridis")]
            )

        fig.update_layout(
            title=kwargs.get("title", "3D Geospatial Visualization"),
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        if output_file:
            if output_file.endswith(".html"):
                fig.write_html(output_file)
            else:
                # Save as image
                fig.write_image(output_file)

        return fig

    def _create_mayavi_3d_surface(self, output_file: str | None, **kwargs: Any) -> Any:
        """Create 3D surface using Mayavi."""
        # Implementation for Mayavi 3D visualization
        # This would create a 3D surface plot using mayavi.mlab

        if isinstance(self.geo_art.data, gpd.GeoDataFrame):
            # Convert to mesh for 3D display
            x = self.geo_art.data.geometry.x.values
            y = self.geo_art.data.geometry.y.values

            if self.z_column and self.z_column in self.geo_art.data.columns:
                z = self.geo_art.data[self.z_column].values
            else:
                z = np.zeros(len(x))

            # Create 3D scatter plot
            mlab.figure(size=(800, 600))
            mlab.points3d(x, y, z, scale_factor=0.1)

        else:
            # Create 3D surface from raster
            mlab.figure(size=(800, 600))
            mlab.surf(self.geo_art.data, warp_scale="auto")

        if output_file:
            mlab.savefig(output_file)

        return mlab.gcf()

    def create_3d_animation(self, **kwargs: Any) -> Any:
        """
        Create an animated 3D visualization.

        Rotates the camera around the 3D scene to produce a turntable-style
        animation.  If Mayavi is not available, falls back to a matplotlib
        3D scatter rotation using FuncAnimation.

        Args:
            **kwargs: Parameters for 3D animation.
                frames (int): Number of animation frames (default 36).
                interval (int): Milliseconds between frames (default 100).
                elevation (float): Camera elevation angle (default 30).
                output_file (str): Optional path to save the animation.

        Returns:
            Animation object (matplotlib FuncAnimation or Mayavi scene).
        """
        frames = kwargs.get("frames", 36)
        interval = kwargs.get("interval", 100)
        elevation = kwargs.get("elevation", 30.0)
        output_file = kwargs.get("output_file", None)

        if MAYAVI_AVAILABLE:
            _scene = self._create_mayavi_3d_surface(output_file=None, **kwargs)

            @mlab.animate(delay=interval)  # type: ignore[misc]
            def _rotate() -> Iterator[None]:
                for angle in np.linspace(0, 360, frames, endpoint=False):
                    mlab.view(azimuth=angle, elevation=elevation)
                    yield

            anim = _rotate()
            if output_file:
                mlab.savefig(output_file)
            return anim

        # Fallback: matplotlib 3D rotation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        if isinstance(self.geo_art.data, gpd.GeoDataFrame):
            x = self.geo_art.data.geometry.x.values
            y = self.geo_art.data.geometry.y.values
            z_col = (
                self.z_column
                if self.z_column and self.z_column in self.geo_art.data.columns
                else None
            )
            z = self.geo_art.data[z_col].values if z_col else np.zeros(len(x))
            ax.scatter(x, y, z, s=1)
        else:
            data = np.asarray(self.geo_art.data)
            if data.ndim == 2:
                xg, yg = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
                ax.plot_surface(xg, yg, data, cmap="terrain", alpha=0.8)

        def _update(frame_idx: int) -> None:
            ax.view_init(elev=elevation, azim=frame_idx * (360.0 / frames))

        anim = FuncAnimation(
            fig,
            _update,
            frames=frames,
            interval=interval,  # type: ignore[arg-type]
        )
        if output_file:
            anim.save(output_file, writer="pillow")
        return anim
