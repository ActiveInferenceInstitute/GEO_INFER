"""
Map styling and theming components for advanced cartographic design.
"""

import logging
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from geo_infer_art.core.aesthetics import ColorPalette

logger = logging.getLogger(__name__)


class MapStyle:
    """
    A class for managing advanced map styling and theming.

    MapStyle provides comprehensive styling options for geospatial visualizations,
    including thematic cartography, advanced symbolization, and aesthetic design
    principles.

    Attributes:
        name: Name of the style
        theme: Visual theme (e.g., 'classical', 'modern', 'minimal')
        parameters: Style parameters and settings
    """

    # Predefined map styles
    PREDEFINED_STYLES = {
        "classical": {
            "colors": ["#8B4513", "#D2B48C", "#F5DEB3", "#DEB887"],
            "line_width": 1.5,
            "font_family": "serif",
            "background": "#F5F5DC",
            "grid": True,
            "grid_style": {"color": "#8B7355", "alpha": 0.3, "linestyle": "--"},
        },
        "modern": {
            "colors": ["#2C3E50", "#3498DB", "#E74C3C", "#2ECC71"],
            "line_width": 2.0,
            "font_family": "sans-serif",
            "background": "#ECF0F1",
            "grid": False,
        },
        "minimal": {
            "colors": ["#000000", "#FFFFFF", "#808080"],
            "line_width": 1.0,
            "font_family": "sans-serif",
            "background": "#FFFFFF",
            "grid": False,
        },
        "neon": {
            "colors": ["#FF00FF", "#00FFFF", "#FFFF00", "#FF0000"],
            "line_width": 3.0,
            "font_family": "monospace",
            "background": "#000000",
            "grid": True,
            "grid_style": {"color": "#00FFFF", "alpha": 0.5, "linestyle": ":"},
        },
        "watercolor": {
            "colors": ["#FFB6C1", "#87CEEB", "#98FB98", "#F0E68C"],
            "line_width": 1.2,
            "font_family": "serif",
            "background": "#F8F8FF",
            "grid": False,
        },
        "blueprint": {
            "colors": ["#072448", "#1E88E5", "#54C7EC", "#A7E0E5"],
            "line_width": 1.8,
            "font_family": "monospace",
            "background": "#072448",
            "grid": True,
            "grid_style": {"color": "#1E88E5", "alpha": 0.4, "linestyle": "-"},
        },
        "topographic": {
            "colors": ["#8B4513", "#A0522D", "#CD853F", "#DEB887", "#F5DEB3"],
            "line_width": 1.5,
            "font_family": "serif",
            "background": "#F5F5DC",
            "grid": True,
            "grid_style": {"color": "#8B4513", "alpha": 0.3, "linestyle": "--"},
        },
        "art_nouveau": {
            "colors": ["#800080", "#FF69B4", "#00CED1", "#32CD32"],
            "line_width": 2.2,
            "font_family": "serif",
            "background": "#F0F8FF",
            "grid": False,
        },
    }

    def __init__(
        self,
        name: str = "default",
        theme: str | None = None,
        parameters: dict | None = None,
    ):
        """
        Initialize a MapStyle object.

        Args:
            name: Name of the style
            theme: Visual theme to use
            parameters: Custom style parameters
        """
        self.name = name
        self.theme = theme or name
        self.parameters: dict[str, Any] = parameters or {}

        # Load style configuration
        if name in self.PREDEFINED_STYLES:
            self._load_predefined_style(name)
        else:
            self._create_custom_style()

    def _load_predefined_style(self, style_name: str) -> None:
        """Load a predefined style configuration."""
        style_config = self.PREDEFINED_STYLES[style_name].copy()
        style_config.update(self.parameters)
        self.parameters = style_config

    def _create_custom_style(self) -> None:
        """Create a custom style with default parameters."""
        self.parameters = {
            "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            "line_width": 1.0,
            "font_family": "sans-serif",
            "background": "#ffffff",
            "grid": False,
            **self.parameters,
        }

    @classmethod
    def create_themed_style(
        cls,
        theme: str,
        color_palette: str | ColorPalette | None = None,
        **kwargs: Any,
    ) -> "MapStyle":
        """
        Create a style based on a visual theme.

        Args:
            theme: Visual theme name
            color_palette: Color palette to use
            **kwargs: Additional style parameters

        Returns:
            A new MapStyle object
        """
        # Theme-specific default parameters
        theme_defaults = {
            "classical": {"font_family": "serif", "grid": True},
            "modern": {"font_family": "sans-serif", "grid": False},
            "minimal": {"font_family": "sans-serif", "grid": False, "line_width": 1.0},
            "vintage": {"font_family": "serif", "grid": True, "line_width": 1.5},
            "futuristic": {"font_family": "monospace", "grid": True, "line_width": 2.0},
        }

        parameters = theme_defaults.get(theme, {}).copy()
        parameters.update(kwargs)

        if color_palette:
            if isinstance(color_palette, str):
                palette = ColorPalette.get_palette(color_palette)
            else:
                palette = color_palette
            parameters["colors"] = palette.colors

        return cls(name=theme, theme=theme, parameters=parameters)

    def apply_to_axes(self, ax: plt.Axes, data_bounds: tuple | None = None) -> None:
        """
        Apply the style to matplotlib axes.

        Args:
            ax: Matplotlib axes object
            data_bounds: Optional data bounds (min_x, min_y, max_x, max_y)
        """
        # Set background color
        background = self.parameters.get("background", "#ffffff")
        ax.set_facecolor(background)

        # Configure grid
        if self.parameters.get("grid", False):
            grid_style = self.parameters.get("grid_style", {})
            ax.grid(
                True,
                color=grid_style.get("color", "#000000"),
                alpha=grid_style.get("alpha", 0.3),
                linestyle=grid_style.get("linestyle", "--"),
            )
        else:
            ax.grid(False)

        # Set font family
        font_family = self.parameters.get("font_family", "sans-serif")
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontfamily(font_family)

        # Apply theme-specific styling
        if self.theme == "blueprint":
            self._apply_blueprint_style(ax)
        elif self.theme == "watercolor":
            self._apply_watercolor_style(ax)
        elif self.theme == "neon":
            self._apply_neon_style(ax)
        elif self.theme == "classical":
            self._apply_classical_style(ax)

    def _apply_blueprint_style(self, ax: plt.Axes) -> None:
        """Apply blueprint-style formatting."""
        # Set blueprint colors
        ax.spines["top"].set_color("#1E88E5")
        ax.spines["bottom"].set_color("#1E88E5")
        ax.spines["left"].set_color("#1E88E5")
        ax.spines["right"].set_color("#1E88E5")

        # Set tick colors
        ax.tick_params(axis="x", colors="#1E88E5")
        ax.tick_params(axis="y", colors="#1E88E5")

        # Set label colors
        ax.xaxis.label.set_color("#1E88E5")
        ax.yaxis.label.set_color("#1E88E5")

    def _apply_watercolor_style(self, ax: plt.Axes) -> None:
        """Apply watercolor-style formatting."""
        # Softer, more artistic appearance
        ax.spines["top"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)
        ax.spines["left"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.3)

        # Lighter grid if present
        if self.parameters.get("grid", False):
            ax.grid(True, alpha=0.2, color="#B0C4DE")

    def _apply_neon_style(self, ax: plt.Axes) -> None:
        """Apply neon-style formatting."""
        # Bright, glowing appearance
        ax.spines["top"].set_color("#00FFFF")
        ax.spines["bottom"].set_color("#00FFFF")
        ax.spines["left"].set_color("#00FFFF")
        ax.spines["right"].set_color("#00FFFF")

        # Set tick colors
        ax.tick_params(axis="x", colors="#00FFFF")
        ax.tick_params(axis="y", colors="#00FFFF")

        # Set label colors
        ax.xaxis.label.set_color("#00FFFF")
        ax.yaxis.label.set_color("#00FFFF")

    def _apply_classical_style(self, ax: plt.Axes) -> None:
        """Apply classical-style formatting."""
        # Traditional cartographic appearance
        ax.spines["top"].set_color("#8B4513")
        ax.spines["bottom"].set_color("#8B4513")
        ax.spines["left"].set_color("#8B4513")
        ax.spines["right"].set_color("#8B4513")

        # Set tick colors
        ax.tick_params(axis="x", colors="#8B4513")
        ax.tick_params(axis="y", colors="#8B4513")

        # Set label colors
        ax.xaxis.label.set_color("#8B4513")
        ax.yaxis.label.set_color("#8B4513")

    def get_colormap(self) -> LinearSegmentedColormap:
        """
        Get a matplotlib colormap based on the style colors.

        Returns:
            A LinearSegmentedColormap object
        """
        colors = self.parameters.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c"])
        return LinearSegmentedColormap.from_list(f"{self.name}_cmap", colors)

    def get_color_list(self) -> list[str]:
        """
        Get the list of colors for this style.

        Returns:
            List of color strings
        """
        return cast(
            list[str], self.parameters.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c"])
        )

    def get_line_width(self) -> float:
        """
        Get the default line width for this style.

        Returns:
            Line width as float
        """
        return cast(float, self.parameters.get("line_width", 1.0))

    def get_background_color(self) -> str:
        """
        Get the background color for this style.

        Returns:
            Background color string
        """
        return cast(str, self.parameters.get("background", "#ffffff"))

    def __repr__(self) -> str:
        """Return a string representation of the MapStyle object."""
        return f"MapStyle(name='{self.name}', theme='{self.theme}')"
