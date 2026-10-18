"""
GeoArt module for artistic visualization of geospatial data.
"""

import html
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
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D

from geo_infer_art.core.aesthetics import ColorPalette
from geo_infer_art.utils.animation import save_animation_with_fallback
from geo_infer_art.utils.validators import validate_file_path, validate_geospatial_data

logger = logging.getLogger(__name__)


def _popup_text(value: Any) -> str:
    """Escape label text for both HTML and Folium's JavaScript template literal."""
    return (
        html.escape(str(value))
        .replace("\\", "&#92;")
        .replace("`", "&#96;")
        .replace("$", "&#36;")
    )


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

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import mayavi.mlab as mlab

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
        self._owns_figure = False
        self._colorbar: Colorbar | None = None

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
                data = src.read(masked=True)
                metadata = {
                    "source": file_path,
                    "type": "raster",
                    "shape": data.shape,
                    "bounds": src.bounds,
                    "transform": src.transform,
                    "raster_layout": "band_first",
                }
                return cls(
                    data=data,
                    metadata=metadata,
                    crs=src.crs.to_string() if src.crs else None,
                )
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
        *,
        ax: Axes | None = None,
        column: str | None = None,
        raster_layout: str | None = None,
        raster_band: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
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
            ax: Existing axes for a composed figure. The caller owns its lifecycle.
            column: Vector attribute used for colors and the optional legend.
            raster_layout: ``band_first``, ``channel_last``, or ``auto``. Auto
                honors loaded-raster metadata, then prefers channel-last when
                both ends could contain channels. Specify ambiguous layouts.
            raster_band: Zero-based band to display as a scalar raster.
            vmin: Lower scalar color limit, reusable across comparable panels.
            vmax: Upper scalar color limit, reusable across comparable panels.

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

        for limit in (vmin, vmax):
            if limit is not None and not np.isfinite(limit):
                raise ValueError("Color limits must be finite")
        if vmin is not None and vmax is not None and vmin >= vmax:
            raise ValueError("vmin must be less than vmax")
        if isinstance(self.data, gpd.GeoDataFrame):
            validate_geospatial_data(self.data)
            if column is not None and column not in self.data.columns:
                raise ValueError(f"Unknown color column: {column}")
            image = None
        else:
            image = self._raster_image(raster_layout, raster_band)

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

        # Never close unrelated pyplot figures. Replace only figures we own.
        previous_figure, previous_owned = self._figure, self._owns_figure
        owns_figure = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=background_color)
        else:
            fig = ax.figure

        try:
            if isinstance(self.data, gpd.GeoDataFrame):
                # Vector data visualization
                plot_kwargs = {
                    "ax": ax,
                    "cmap": palette.cmap,
                    "linewidth": line_width,
                    "alpha": alpha,
                }

                # Add legend if requested
                if column is not None:
                    plot_kwargs["column"] = column
                    plot_kwargs["legend"] = legend
                elif legend and len(self.data.columns) > 1:
                    # Try to use a categorical column for legend
                    categorical_cols = self.data.select_dtypes(
                        include=["object"]
                    ).columns
                    if len(categorical_cols) > 0:
                        plot_kwargs["legend"] = True
                        plot_kwargs["column"] = categorical_cols[0]

                if vmin is not None:
                    plot_kwargs["vmin"] = vmin
                if vmax is not None:
                    plot_kwargs["vmax"] = vmax

                self.data.plot(**plot_kwargs)

            else:
                # Raster data visualization
                assert image is not None
                image_kwargs: dict[str, Any] = {"alpha": alpha}
                if image.ndim == 2:
                    image_kwargs.update(cmap=palette.cmap, vmin=vmin, vmax=vmax)
                bounds = self.metadata.get("bounds")
                transform = self.metadata.get("transform")
                if bounds is not None:
                    left, bottom, right, top = bounds
                    image_kwargs["extent"] = (left, right, bottom, top)
                    image_kwargs["origin"] = "upper"
                    if transform is not None and (
                        transform.b or transform.d or transform.a < 0 or transform.e > 0
                    ):
                        height, width = image.shape[:2]
                        image_kwargs["extent"] = (0, width, height, 0)
                        image_kwargs["transform"] = (
                            Affine2D.from_values(
                                transform.a,
                                transform.d,
                                transform.b,
                                transform.e,
                                transform.c,
                                transform.f,
                            )
                            + ax.transData
                        )
                ax.imshow(image, **image_kwargs)
                if bounds is not None:
                    ax.set_xlim(left, right)
                    ax.set_ylim(bottom, top)

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

        except BaseException:
            if owns_figure:
                plt.close(fig)
            raise

        self._figure = fig
        self._ax = ax
        self._colorbar = None
        self._owns_figure = owns_figure or (previous_owned and previous_figure is fig)
        if previous_owned and previous_figure is not fig:
            plt.close(previous_figure)

        return self

    def _raster_image(
        self, layout: str | None = None, band: int | None = None
    ) -> np.ndarray:
        """Normalize native arrays without transposing ambiguous channel-last data."""
        layout = self._raster_layout(layout)
        image = self.data
        if image.ndim == 3:
            if layout == "band_first":
                image = np.moveaxis(image, 0, -1)
            if band is not None:
                if type(band) is not int or not 0 <= band < image.shape[-1]:
                    raise ValueError(
                        "raster_band must be a valid zero-based band index"
                    )
                image = image[..., band]
            elif image.shape[-1] == 1:
                image = image[..., 0]
            elif image.shape[-1] not in (3, 4):
                raise ValueError(
                    "Select raster_band for rasters with other than 1, 3, or 4 bands"
                )
        elif band is not None and (type(band) is not int or band != 0):
            raise ValueError("A 2D raster only has raster_band=0")
        return image

    def _raster_layout(self, layout: str | None) -> str:
        """Resolve the shared plotting/resampling band-axis convention."""
        if not isinstance(self.data, np.ndarray) or self.data.ndim not in (2, 3):
            raise ValueError("Raster data must be a 2D or 3D numpy array")
        if 0 in self.data.shape:
            raise ValueError("Raster dimensions must be nonempty")
        if layout is None or layout == "auto":
            layout = self.metadata.get("raster_layout", "auto")
        if layout not in ("auto", "band_first", "channel_last"):
            raise ValueError("raster_layout must be auto, band_first, or channel_last")
        if layout == "auto":
            layout = (
                "channel_last"
                if self.data.ndim == 2 or self.data.shape[-1] in (1, 3, 4)
                else "band_first"
            )
        return layout

    def close(self) -> None:
        """Release this visualization, closing only an internally created figure."""
        if self._owns_figure and self._figure is not None:
            plt.close(self._figure)
        self._figure = None
        self._ax = None
        self._owns_figure = False
        self._colorbar = None

    def __enter__(self) -> "GeoArt":
        """Return a visualization whose owned figure closes on context exit."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Release owned rendering resources, including on failed exports."""
        self.close()

    def save(
        self,
        output_path: str,
        dpi: int = 300,
        *,
        close: bool = True,
        transparent: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save the visualization to a file.

        Args:
            output_path: Path where the file should be saved
            dpi: Resolution for the output image
            close: Release an owned pyplot figure after saving; caller axes are
                never closed. The figure remains available for further exports.
            transparent: Export transparent axes and figure backgrounds.
            metadata: Native Matplotlib format-specific export metadata.

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

        self._figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
            metadata=metadata,
        )
        if close and self._owns_figure:
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

        num_frames = max(1, round(duration * fps))
        # Render into a separate owner so the current visualization and caller
        # axes remain intact. Animation must update artists on a single canvas;
        # returning a different Figure from the callback never changed its pixels.
        if "ax" in kwargs:
            raise ValueError("Animation owns its canvas; ax is not supported")
        frames = []
        with GeoArt(self.data, dict(self.metadata), self.crs) as renderer:
            for style in style_sequence:
                renderer.apply_style(style=style, **kwargs)
                assert renderer._figure is not None
                renderer._figure.canvas.draw()
                frames.append(np.asarray(renderer._figure.canvas.buffer_rgba()).copy())

        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            ax.set_position([0, 0, 1, 1])
            ax.axis("off")
            artist = ax.imshow(frames[0])

            def animate(frame_num: int) -> tuple[Any, ...]:
                index = min(frame_num * len(frames) // num_frames, len(frames) - 1)
                artist.set_data(frames[index])
                return (artist,)

            anim = animation.FuncAnimation(
                fig, animate, frames=num_frames, interval=1000 / fps, blit=False
            )
            return save_animation_with_fallback(anim, output_path, fps)
        finally:
            plt.close(fig)

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

    def export_svg(
        self,
        output_path: str,
        *,
        metadata: dict[str, Any] | None = None,
        text_as_paths: bool = False,
    ) -> str:
        """
        Export the visualization as an SVG file.

        Args:
            output_path: Path for the SVG output file
            metadata: SVG metadata overrides, such as Title and Description.
            text_as_paths: Convert fonts to paths instead of selectable text.

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

        export_metadata: dict[str, Any] = {"Date": None}
        if self._ax is not None:
            export_metadata["Title"] = self._ax.get_title()
        export_metadata.update(metadata or {})
        with plt.rc_context(
            {
                "svg.fonttype": "path" if text_as_paths else "none",
                "svg.hashsalt": "geo-infer-art",
            }
        ):
            self._figure.savefig(
                output_path, format="svg", bbox_inches="tight", metadata=export_metadata
            )
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

        candidates = [*self._ax.images, *self._ax.collections]
        scalar = next(
            (
                item
                for item in reversed(candidates)
                if item.get_array() is not None and item.get_array().ndim <= 2
            ),
            None,
        )
        if scalar is None:
            raise ValueError(
                "A colorbar requires scalar raster data or a colored vector column"
            )
        if self._colorbar is None:
            self._colorbar = self._figure.colorbar(scalar, ax=self._ax, label=label)
        else:
            self._colorbar.update_normal(scalar)
            self._colorbar.set_label(label)

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
                "Install with: uv sync --package geo-infer-art --extra viz3d "
                "(mayavi) or --extra integrations (plotly)"
            )

        return GeoArt3D(
            geo_art=self, elevation_data=elevation_data, z_column=z_column, **kwargs
        )

    def create_interactive_web_map(
        self,
        output_file: str = "interactive_map.html",
        tiles: str | None = "OpenStreetMap",
        **kwargs: Any,
    ) -> str:
        """
        Create an interactive web-based map using Folium.

        Args:
            output_file: Path for the HTML output file
            tiles: Map tiles to use ("OpenStreetMap", "Stamen Terrain", etc.)
            **kwargs: Native Folium map parameters. ``fit_bounds`` defaults to
                true unless ``zoom_start`` is explicit. Data is transformed to
                EPSG:4326; multipart geometries and polygon holes are retained.

        Returns:
            Path to the created HTML file

        Raises:
            ValueError: If folium is not available or data is not suitable
        """
        if not FOLIUM_AVAILABLE:
            raise ValueError(
                "Interactive web maps require folium. Install with: "
                "uv sync --package geo-infer-art --extra integrations"
            )

        if self.data is None:
            raise ValueError("No data loaded for web map creation")

        data = self._web_geodataframe()
        bounds = data.total_bounds
        center_lat = float((bounds[1] + bounds[3]) / 2)
        center_lon = float((bounds[0] + bounds[2]) / 2)
        options = dict(kwargs)
        fit_bounds = options.pop("fit_bounds", "zoom_start" not in options)
        options.setdefault("zoom_start", 10)
        web_map = folium.Map(location=[center_lat, center_lon], tiles=tiles, **options)
        # GeoJSON preserves polygon interior rings and every multipart geometry.
        # Escape attribute text before passing it to the popup's HTML boundary.
        for idx, row in data.iterrows():
            title = _popup_text(row.get("name", f"Feature {idx}"))
            popup_content = f"<b>{title}</b><br>"
            for col in data.columns:
                if col != data.geometry.name:
                    popup_content += f"{_popup_text(col)}: {_popup_text(row[col])}<br>"
            folium.GeoJson(
                row[data.geometry.name].__geo_interface__,
                popup=folium.Popup(popup_content),
                marker=folium.CircleMarker(radius=5, fill=True),
            ).add_to(web_map)
        if fit_bounds:
            web_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Save the map
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        web_map.save(output_file)
        return output_file

    def _web_geodataframe(self) -> gpd.GeoDataFrame:
        """Return valid geographic features without mutating the source frame."""
        if not isinstance(self.data, gpd.GeoDataFrame):
            raise ValueError("Web maps require vector GeoDataFrame data")
        if self.data.crs is None:
            raise ValueError(
                "Web maps require a CRS to transform coordinates to EPSG:4326"
            )
        data = self.data.loc[self.data.geometry.notna() & ~self.data.geometry.is_empty]
        if data.empty or not data.geometry.is_valid.all():
            raise ValueError("Web maps require nonempty, valid geometries")
        data = data.to_crs("EPSG:4326")
        if not np.isfinite(data.total_bounds).all():
            raise ValueError("Web map coordinates must be finite")
        left, bottom, right, top = data.total_bounds
        if left < -180 or right > 180 or bottom < -90 or top > 90:
            raise ValueError(
                "Web map coordinates must be valid WGS84 longitude and latitude"
            )
        return data

    def create_plotly_visualization(
        self, plot_type: str = "scatter", **kwargs: Any
    ) -> Any:
        """
        Create an interactive Plotly visualization.

        Args:
            plot_type: Type of plot ("scatter", "choropleth", "heatmap", "3d")
            **kwargs: Native Plotly Express map options, including title,
                labels, hover_data, range_color, and map_style. Legacy aliases
                color_column, size_column and mapbox_style remain supported.
                Scalar raster plots accept raster_layout, raster_band, vmin,
                vmax, and colorscale. Map data is transformed to EPSG:4326.

        Returns:
            Plotly figure object

        Raises:
            ValueError: If plotly is not available or plot type is not supported
        """
        if not PLOTLY_AVAILABLE:
            raise ValueError(
                "Plotly visualizations require plotly. Install with: "
                "uv sync --package geo-infer-art --extra integrations"
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
        data = self._web_geodataframe()
        bounds = data.total_bounds
        options = dict(kwargs)
        options["hover_name"] = options.pop(
            "hover_name", "name" if "name" in data else None
        )
        options["color"] = options.pop("color_column", options.pop("color", None))
        options.setdefault("zoom", 10)
        options.setdefault(
            "center",
            {
                "lat": float((bounds[1] + bounds[3]) / 2),
                "lon": float((bounds[0] + bounds[2]) / 2),
            },
        )
        map_style = options.pop(
            "map_style", options.pop("mapbox_style", "carto-positron")
        )
        # Plotly 5.24+ uses MapLibre; retain native Mapbox APIs for the declared
        # older Plotly versions rather than forcing a dependency-floor upgrade.
        modern_maps = hasattr(px, "scatter_map")
        options["map_style" if modern_maps else "mapbox_style"] = map_style
        if plot_type == "scatter":
            if not data.geometry.geom_type.eq("Point").all():
                raise ValueError("Scatter maps require Point geometries")
            options["size"] = options.pop("size_column", options.pop("size", None))
            scatter = px.scatter_map if modern_maps else px.scatter_mapbox
            fig = scatter(data, lat=data.geometry.y, lon=data.geometry.x, **options)
        elif plot_type == "choropleth":
            if not data.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                raise ValueError(
                    "Choropleth maps require Polygon or MultiPolygon geometries"
                )
            # Reset feature ids so duplicate/custom dataframe indices cannot
            # silently join several regions to the same GeoJSON feature.
            data = data.reset_index(drop=True)
            features = data.geometry.__geo_interface__
            locations = [feature["id"] for feature in features["features"]]
            choropleth = px.choropleth_map if modern_maps else px.choropleth_mapbox
            fig = choropleth(
                data,
                geojson=features,
                locations=locations,
                featureidkey="id",
                **options,
            )
        else:
            raise ValueError(f"Unsupported plot type for GeoDataFrame: {plot_type}")

        return fig

    def _create_plotly_from_raster(self, plot_type: str, **kwargs: Any) -> Any:
        """Create Plotly visualization from raster data."""
        if not isinstance(self.data, np.ndarray):
            raise ValueError("GeoDataFrame not supported for raster plot")
        image = self._raster_image(
            kwargs.get("raster_layout"), kwargs.get("raster_band")
        )
        if image.ndim != 2:
            raise ValueError(
                "Heatmap and surface plots require a scalar raster; select raster_band"
            )
        if np.ma.isMaskedArray(image):
            image = image.astype(float).filled(np.nan)
        if plot_type == "heatmap":
            # Create heatmap from raster
            fig = go.Figure(
                data=go.Heatmap(
                    z=image,
                    zmin=kwargs.get("vmin"),
                    zmax=kwargs.get("vmax"),
                    colorscale=kwargs.get("colorscale", "Viridis"),
                )
            )
            fig.update_layout(
                title=kwargs.get("title", "Raster Heatmap"),
                xaxis_title=kwargs.get("x_label", "X"),
                yaxis_title=kwargs.get("y_label", "Y"),
            )

        elif plot_type == "3d":
            # Create 3D surface plot
            x = np.arange(image.shape[1])
            y = np.arange(image.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = go.Figure(
                data=[
                    go.Surface(
                        z=image,
                        x=X,
                        y=Y,
                        cmin=kwargs.get("vmin"),
                        cmax=kwargs.get("vmax"),
                        colorscale=kwargs.get("colorscale", "Viridis"),
                    )
                ]
            )
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
        *,
        raster_layout: str | None = None,
    ) -> "GeoArt":
        """
        Optimize the visualization for better performance.

        Args:
            target_resolution: Positive integer ``(height, width)`` for raster
                data. Bilinear sampling preserves the full pixel extent and
                band axis. Integer arrays retain their native dtype; masked
                arrays use floating interpolation of valid contributions only.
                Cells without valid contributions remain masked. This is not
                an area-average or conservation-preserving resampler.
            simplify_tolerance: Finite nonnegative geometry tolerance in CRS
                units; preserves topology and the active geometry column.
            cache_data: Whether to mark the returned copy as cached and retain
                its original shape. This does not create a disk cache.
            raster_layout: Same layout convention as ``apply_style``; use an
                explicit layout for ambiguous three-dimensional arrays.

        Returns:
            Optimized GeoArt object

        Raises:
            ValueError: If optimization parameters are invalid
        """
        if self.data is None:
            raise ValueError("No data loaded for optimization")
        if target_resolution is not None and (
            not isinstance(target_resolution, (tuple, list))
            or len(target_resolution) != 2
            or any(type(size) is not int or size <= 0 for size in target_resolution)
        ):
            raise ValueError(
                "target_resolution must contain positive integer height and width"
            )
        if simplify_tolerance is not None and (
            isinstance(simplify_tolerance, (bool, np.bool_))
            or not isinstance(simplify_tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(simplify_tolerance)
            or simplify_tolerance < 0
        ):
            raise ValueError("simplify_tolerance must be finite and nonnegative")

        # Create optimized copy
        optimized = GeoArt(
            data=self.data.copy(), metadata=self.metadata.copy(), crs=self.crs
        )

        if isinstance(self.data, gpd.GeoDataFrame) and simplify_tolerance is not None:
            # Simplify geometries for better performance
            assert isinstance(optimized.data, gpd.GeoDataFrame)
            optimized.data = optimized.data.set_geometry(
                optimized.data.geometry.simplify(
                    simplify_tolerance, preserve_topology=True
                )
            )

        if (
            not isinstance(self.data, gpd.GeoDataFrame)
            and target_resolution is not None
        ):
            from scipy.ndimage import zoom

            layout = self._raster_layout(raster_layout)
            spatial_axes = (
                (1, 2) if self.data.ndim == 3 and layout == "band_first" else (0, 1)
            )
            old_height, old_width = (self.data.shape[axis] for axis in spatial_axes)
            factors = [1.0] * self.data.ndim
            for axis, size in zip(spatial_axes, target_resolution):
                factors[axis] = size / self.data.shape[axis]
            options = {
                "order": 1,
                "mode": "nearest",
                "grid_mode": True,
                "prefilter": False,
            }

            def resample(values: np.ndarray) -> np.ndarray:
                # SciPy's complex zoom does not forward grid_mode to its
                # component calls. Use identical pixel coordinates explicitly.
                if np.iscomplexobj(values):
                    return zoom(values.real, factors, **options) + 1j * zoom(
                        values.imag, factors, **options
                    )
                return zoom(values, factors, **options)

            if np.ma.isMaskedArray(self.data):
                valid = (~np.ma.getmaskarray(self.data)).astype(float)
                weights = resample(valid)
                values = resample(
                    self.data.astype(np.result_type(self.data.dtype, float)).filled(0)
                )
                np.divide(values, weights, out=values, where=weights > 0)
                optimized.data = np.ma.array(values, mask=weights <= 0)
            else:
                optimized.data = resample(self.data)
            optimized.metadata["raster_layout"] = layout
            optimized.metadata["shape"] = optimized.data.shape
            transform = self.metadata.get("transform")
            if transform is not None:
                optimized.metadata["transform"] = transform * transform.scale(
                    old_width / target_resolution[1], old_height / target_resolution[0]
                )

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
