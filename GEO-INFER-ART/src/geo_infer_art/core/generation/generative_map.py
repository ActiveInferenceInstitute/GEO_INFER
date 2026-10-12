"""
GenerativeMap module for creating generative art from geospatial data.
"""

import hashlib
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from geo_infer_art.core.aesthetics import ColorPalette

logger = logging.getLogger(__name__)


class GenerativeMap:
    """
    A class for creating generative art from geospatial data.

    The GenerativeMap class provides methods for transforming geospatial data
    into artistic and abstract visualizations using various generative algorithms.

    Attributes:
        data: The underlying data used for generation
        metadata: Additional information about the data source
        image: The generated image as a numpy array
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        metadata: dict | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize a GenerativeMap object.

        Args:
            data: Base data used for generation
            metadata: Information about the data source
            seed: Integer seed or ``np.random.Generator`` for stochastic
                styles and textures; ``None`` draws fresh entropy.
        """
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.data = data
        self.metadata = metadata or {}
        self.image: Image.Image | None = None
        self._figure: Figure | None = None
        self._ax: Axes | None = None
        self._output_resolution: int | None = None

    @classmethod
    def from_elevation(
        cls,
        region: str | np.ndarray | tuple[float, float, float, float],
        resolution: int = 512,
        abstraction_level: float = 0.5,
        style: str = "contour",
        seed: int | np.random.Generator | None = None,
    ) -> "GenerativeMap":
        """
        Create generative art from elevation data.

        Args:
            region: Region name, custom elevation data, or bounding box coordinates
            resolution: Resolution of the output image
            abstraction_level: Level of abstraction (0.0 to 1.0)
            style: Style of the generative art ("contour", "flow", "particles", etc.)
            seed: Integer seed or ``np.random.Generator`` for stochastic styles

        Returns:
            A new GenerativeMap object with generated art

        Raises:
            ValueError: If the region is invalid or the data cannot be retrieved
        """
        if resolution <= 0:
            raise ValueError("Resolution must be a positive integer.")
        if not 0.0 <= abstraction_level <= 1.0:
            raise ValueError("Abstraction level must be between 0 and 1.")

        # Initialize the object
        gen_map = cls(seed=seed)
        gen_map._output_resolution = resolution

        # Load elevation data
        if isinstance(region, str):
            # Get elevation data for a named region
            elevation_data = cls._generate_region_terrain(region, resolution)
            gen_map.metadata["region"] = region
            gen_map.metadata["type"] = "named_region"
        elif isinstance(region, np.ndarray):
            # Use provided elevation data
            elevation_data = region
            gen_map.metadata["region"] = "custom"
            gen_map.metadata["type"] = "elevation_array"
        elif isinstance(region, tuple) and len(region) == 4:
            # Get elevation data for a bounding box (min_lon, min_lat, max_lon, max_lat)
            elevation_data = cls._generate_bbox_terrain(region, resolution)
            gen_map.metadata["region"] = f"bbox_{region}"
            gen_map.metadata["bbox"] = region
            gen_map.metadata["type"] = "bbox"
        else:
            raise ValueError(
                "Invalid region parameter. Expected region name, "
                "elevation array, or bounding box coordinates."
            )

        gen_map.data = elevation_data
        gen_map.metadata["data_type"] = "elevation"
        gen_map.metadata["abstraction_level"] = abstraction_level
        gen_map.metadata["style"] = style

        # Generate art based on style
        if style == "contour":
            gen_map._generate_contour_art(abstraction_level)
        elif style == "flow":
            gen_map._generate_flow_art(abstraction_level)
        elif style == "particles":
            gen_map._generate_particle_art(abstraction_level)
        elif style == "contour_flow":
            gen_map._generate_contour_flow_art(abstraction_level)
        else:
            raise ValueError(
                f"Unsupported style: {style}. Supported styles: "
                "contour, flow, particles, contour_flow"
            )

        return gen_map

    @staticmethod
    def _generate_region_terrain(region: str, resolution: int = 512) -> np.ndarray:
        """
        Generate a procedural terrain field for a named region.

        This is generated relief, not measured elevation: the region's real
        latitude, longitude, and characteristic relief scale parameterise a
        multi-octave noise field. It exists to give the generative-art
        pipeline a plausible, reproducible surface to render, and must not
        be presented as survey or DEM data.

        Output is deterministic for a given ``(region, resolution)`` pair --
        the noise is drawn from a generator seeded from the region name, so
        the same call renders the same artwork across processes.

        Args:
            region: Name of the region (e.g., "grand_canyon", "everest", "alps")
            resolution: Side length of the square output grid

        Returns:
            Generated relief as a 2D numpy array of shape
            ``(resolution, resolution)``

        Raises:
            ValueError: If the region is not one of the supported names
        """
        # lat, lon, characteristic relief scale in metres
        known_regions = {
            "grand_canyon": (
                36.0544,
                -112.2583,
                30,
            ),  # lat, lon, elevation variation scale
            "everest": (27.9881, 86.9250, 100),
            "alps": (45.8333, 6.8667, 40),
            "mariana_trench": (11.3333, 142.2333, -80),
            "sahara": (23.4162, 25.6628, 5),
            "amazon": (-3.4653, -62.2159, 10),
            "great_barrier_reef": (-18.2871, 147.6992, -15),
        }

        if region not in known_regions:
            raise ValueError(
                f"Unknown region: {region}. Supported regions: "
                f"{', '.join(known_regions.keys())}"
            )

        # Get parameters for the region
        lat, lon, scale = known_regions[region]

        # Generate a fractal terrain using Perlin noise
        # (simplified approximation for demonstration)
        x = np.linspace(0, 10, resolution)
        y = np.linspace(0, 10, resolution)
        X, Y = np.meshgrid(x, y)

        # Create multi-scale noise for natural-looking terrain
        elevation = np.zeros((resolution, resolution))
        for octave in range(1, 7):
            frequency = 2**octave
            amplitude = 1.0 / frequency
            elevation += (
                amplitude * np.sin(X * frequency * 0.3) * np.cos(Y * frequency * 0.3)
            )

        # Scale to make the terrain more pronounced for certain regions
        elevation = elevation * scale

        # Seed from the region name so the same region always renders alike.
        seed = int(hashlib.sha256(region.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        elevation += rng.normal(0, abs(scale / 20), elevation.shape)

        return elevation

    @staticmethod
    def _generate_bbox_terrain(
        bbox: tuple[float, float, float, float], resolution: int = 512
    ) -> np.ndarray:
        """
        Generate a procedural terrain field for a bounding box.

        As with :meth:`_generate_region_terrain` this is generated relief,
        not measured elevation. Relief amplitude scales inversely with the
        extent of the box, so large areas render gentler terrain. Output is
        deterministic for a given ``(bbox, resolution)`` pair.

        Args:
            bbox: Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
            resolution: Side length of the square output grid

        Returns:
            Generated relief as a 2D numpy array

        Raises:
            ValueError: If the bounding box is invalid
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Validate bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(
                "Invalid bounding box. Ensure min values are less than max values."
            )

        # Calculate center and scale for terrain generation
        (min_lat + max_lat) / 2
        (min_lon + max_lon) / 2

        # Scale based on the size of the bounding box
        # Larger areas have gentler terrain variation
        scale = 20 * (1 / (max(max_lon - min_lon, max_lat - min_lat) + 0.1))

        # Generate a fractal terrain using Perlin noise
        x = np.linspace(min_lon, max_lon, resolution)
        y = np.linspace(min_lat, max_lat, resolution)
        X, Y = np.meshgrid(x, y)

        # Create multi-scale noise
        elevation = np.zeros((resolution, resolution))
        for octave in range(1, 7):
            frequency = 2**octave
            amplitude = 1.0 / frequency
            elevation += amplitude * np.sin(X * frequency) * np.cos(Y * frequency)

        # Scale the terrain
        elevation = elevation * scale * 50

        # Seed from the bounding box so the same extent always renders alike.
        seed = int(
            hashlib.sha256(repr(tuple(bbox)).encode("utf-8")).hexdigest(), 16
        ) % (2**32)
        rng = np.random.default_rng(seed)
        elevation += rng.normal(0, scale, elevation.shape)

        return elevation

    def _generate_contour_art(self, abstraction_level: float = 0.5) -> None:
        """
        Generate contour-based art from elevation data.

        Args:
            abstraction_level: Level of abstraction (0.0 to 1.0)
        """
        if self.data is None:
            raise ValueError("No data available for generation.")

        # Determine number of contour levels based on abstraction
        # Higher abstraction means fewer contours
        max_contours = 50
        min_contours = 5
        n_contours = int(
            max_contours - abstraction_level * (max_contours - min_contours)
        )

        # Create a figure for the contour plot
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")

        # Get a colormap based on the abstraction level
        if abstraction_level < 0.3:
            cmap_name = "earth"
        elif abstraction_level < 0.7:
            cmap_name = "autumn"
        else:
            cmap_name = "ocean"

        palette = ColorPalette.get_palette(cmap_name)

        # Create contour plot
        ax.contourf(
            self.data,
            levels=n_contours,
            cmap=palette.cmap,
            alpha=0.7,
        )

        # Add contour lines with varying linewidth
        ax.contour(
            self.data,
            levels=n_contours // 2,
            colors="black",
            linewidths=0.5 + abstraction_level,
            alpha=0.6 + 0.4 * abstraction_level,
        )

        # Remove axes for artistic effect
        ax.set_axis_off()

        # Set tight layout
        plt.tight_layout()

        # Store the figure and convert to image
        self._figure = fig
        self._ax = ax

        # Convert matplotlib figure to image array
        self._figure_to_image()

    def _generate_flow_art(self, abstraction_level: float = 0.5) -> None:
        """
        Generate flow-based art from elevation data.

        Args:
            abstraction_level: Level of abstraction (0.0 to 1.0)
        """
        if self.data is None:
            raise ValueError("No data available for generation.")

        # Calculate gradient of the elevation data
        gradient_y, gradient_x = np.gradient(self.data)

        # Normalize gradients
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_x = gradient_x / (magnitude + 1e-8)  # Avoid division by zero
        gradient_y = gradient_y / (magnitude + 1e-8)

        # Determine parameters based on abstraction level
        density = 1.0 - abstraction_level  # Higher abstraction means lower density
        line_width = 0.5 + abstraction_level  # Higher abstraction means thicker lines

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")

        # Get a colormap based on the abstraction level
        if abstraction_level < 0.3:
            cmap_name = "viridis"
        elif abstraction_level < 0.7:
            cmap_name = "sunset"
        else:
            cmap_name = "forest"

        palette = ColorPalette.get_palette(cmap_name)

        # Create stream plot
        n_points = int(30 * density)
        seed_points = self.rng.random((n_points, 2))

        # Scale seed points to data dimensions
        # Streamplot coordinates are indexed from zero through ``shape - 1``.
        # Scaling by ``shape`` occasionally generated an out-of-bounds seed and
        # caused the style path to fail nondeterministically.
        seed_points[:, 0] *= self.data.shape[1] - 1
        seed_points[:, 1] *= self.data.shape[0] - 1

        # Create streamplot
        ax.streamplot(
            np.arange(0, self.data.shape[1]),
            np.arange(0, self.data.shape[0]),
            gradient_x,
            gradient_y,
            color=magnitude,
            linewidth=line_width,
            cmap=palette.cmap,
            density=density,
            arrowsize=0.1,  # Minimal arrow size avoids Matplotlib zero-size warnings
            start_points=seed_points,
        )

        # Remove axes for artistic effect
        ax.set_axis_off()

        # Set aspect ratio and limits
        ax.set_aspect("equal")
        ax.set_xlim(0, self.data.shape[1])
        ax.set_ylim(0, self.data.shape[0])

        # Set tight layout
        plt.tight_layout()

        # Store the figure and convert to image
        self._figure = fig
        self._ax = ax

        # Convert matplotlib figure to image array
        self._figure_to_image()

    def _generate_particle_art(self, abstraction_level: float = 0.5) -> None:
        """
        Generate particle-based art from elevation data.

        Args:
            abstraction_level: Level of abstraction (0.0 to 1.0)
        """
        if self.data is None:
            raise ValueError("No data available for generation.")

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")

        # Determine number of particles based on abstraction
        max_particles = 5000
        min_particles = 200
        n_particles = int(
            max_particles - abstraction_level * (max_particles - min_particles)
        )

        # Get a colormap based on the abstraction level
        if abstraction_level < 0.3:
            cmap_name = "bright"
        elif abstraction_level < 0.7:
            cmap_name = "pastel"
        else:
            cmap_name = "grayscale"

        palette = ColorPalette.get_palette(cmap_name)

        # Scale data to 0-1 range
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        data_range = data_max - data_min
        normalized_data = (
            (self.data - data_min) / data_range if data_range > 0 else self.data * 0
        )

        # Generate random positions weighted by elevation
        # Higher elevations have more particles
        probs = normalized_data.flatten() ** (2 - abstraction_level)
        probs = probs / np.sum(probs)

        indices = self.rng.choice(
            np.arange(normalized_data.size), size=n_particles, p=probs
        )

        y_coords, x_coords = np.unravel_index(indices, normalized_data.shape)

        # Map elevation values to sizes
        sizes = normalized_data[y_coords, x_coords] * 20 + 1

        # Map elevation values to colors
        color_indices = (normalized_data[y_coords, x_coords] * 255).astype(int)
        colors = [
            palette.colors[min(i, len(palette.colors) - 1)] for i in color_indices
        ]

        # Create scatter plot
        ax.scatter(
            x_coords,
            y_coords,
            c=colors,
            s=sizes,
            alpha=0.7,
            edgecolor="none",
        )

        # Remove axes for artistic effect
        ax.set_axis_off()

        # Set aspect ratio and limits
        ax.set_aspect("equal")
        ax.set_xlim(0, self.data.shape[1])
        ax.set_ylim(0, self.data.shape[0])

        # Set tight layout
        plt.tight_layout()

        # Store the figure and convert to image
        self._figure = fig
        self._ax = ax

        # Convert matplotlib figure to image array
        self._figure_to_image()

    def _generate_contour_flow_art(self, abstraction_level: float = 0.5) -> None:
        """
        Generate a combination of contour and flow art from elevation data.

        Args:
            abstraction_level: Level of abstraction (0.0 to 1.0)
        """
        if self.data is None:
            raise ValueError("No data available for generation.")

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")

        # Get color palettes
        palette1 = ColorPalette.get_palette("ocean")
        palette2 = ColorPalette.get_palette("autumn")

        # Determine number of contour levels based on abstraction
        n_contours = int(50 - abstraction_level * 40)

        # Calculate gradient for flow
        gradient_y, gradient_x = np.gradient(self.data)

        # Create contour plot with reduced opacity
        ax.contourf(
            self.data,
            levels=n_contours,
            cmap=palette1.cmap,
            alpha=0.3,
        )

        # Add flow on top of contours
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Normalize gradients
        gradient_x = gradient_x / (magnitude + 1e-8)
        gradient_y = gradient_y / (magnitude + 1e-8)

        # Create streamplot with reduced density
        ax.streamplot(
            np.arange(0, self.data.shape[1]),
            np.arange(0, self.data.shape[0]),
            gradient_x,
            gradient_y,
            color=magnitude,
            linewidth=1.0 + abstraction_level,
            cmap=palette2.cmap,
            density=0.8 - 0.5 * abstraction_level,
            arrowsize=0.1,
        )

        # Remove axes for artistic effect
        ax.set_axis_off()

        # Set aspect ratio and limits
        ax.set_aspect("equal")
        ax.set_xlim(0, self.data.shape[1])
        ax.set_ylim(0, self.data.shape[0])

        # Set tight layout
        plt.tight_layout()

        # Store the figure and convert to image
        self._figure = fig
        self._ax = ax

        # Convert matplotlib figure to image array
        self._figure_to_image()

    def _figure_to_image(self) -> None:
        """Convert the matplotlib figure to a numpy image array."""
        import io

        if self._figure is None:
            return

        # Save figure to a buffer
        buf = io.BytesIO()
        self._figure.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        img = Image.open(buf).convert("RGBA")
        resolution = getattr(self, "_output_resolution", None)
        if resolution is not None:
            img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        self.image = img
        plt.close(self._figure)

    def save(self, output_path: str) -> str:
        """
        Save the generated art to a file.

        Args:
            output_path: Path where the file should be saved

        Returns:
            The path to the saved file

        Raises:
            ValueError: If no image has been generated
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        img = (
            self.image
            if isinstance(self.image, Image.Image)
            else Image.fromarray(self.image)
        )
        img.save(output_path)

        return output_path

    def show(self) -> None:
        """
        Display the generated art.

        Raises:
            ValueError: If no image has been generated
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        figure = plt.figure()
        plt.imshow(self.image)
        plt.axis("off")
        # ``Agg`` is the deterministic headless backend used by CI. Calling
        # ``show`` there emits a backend warning even though rendering works.
        if "agg" in plt.get_backend().lower() or not plt.isinteractive():
            figure.canvas.draw()
            plt.close(figure)
            return
        plt.show()

    def create_animation(
        self,
        output_path: str,
        parameter_sweep: str,
        values: list[float | str],
        duration: float = 5.0,
        fps: int = 24,
    ) -> str:
        """
        Create an animated generative map by varying a parameter.

        Args:
            output_path: Path for the output animation file
            parameter_sweep: Parameter to vary ("abstraction_level" or "style")
            values: List of values to sweep through
            duration: Target total duration of the animation in seconds,
                split evenly across the swept values
            fps: Minimum frame rate; each value is displayed for at least
                1000/fps milliseconds

        Returns:
            Path to the created animation file

        Raises:
            ValueError: If the parameter is unsupported, a sweep value or
                animation setting is invalid
        """
        if self.data is None:
            raise ValueError("No data loaded. Load data first.")

        if parameter_sweep not in ["abstraction_level", "style"]:
            raise ValueError(f"Unsupported parameter for animation: {parameter_sweep}")
        if not values:
            raise ValueError("values must contain at least one entry")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")

        # Create frames for each parameter value
        frames: list[Any] = []
        for value in values:
            if parameter_sweep == "abstraction_level":
                abstraction = float(value)
                if not 0.0 <= abstraction <= 1.0:
                    raise ValueError("Abstraction level must be between 0 and 1.")
                self._generate_contour_art(abstraction_level=abstraction)
            elif parameter_sweep == "style":
                style_generators = {
                    "contour": self._generate_contour_art,
                    "flow": self._generate_flow_art,
                    "particles": self._generate_particle_art,
                    "contour_flow": self._generate_contour_flow_art,
                }
                style = str(value)
                if style not in style_generators:
                    raise ValueError(
                        f"Unsupported style for animation: {style}. Supported styles: "
                        f"{', '.join(style_generators)}"
                    )
                abstraction = float(self.metadata.get("abstraction_level", 0.5))
                style_generators[style](abstraction)

            frames.append(self.image)

        # Each generator call already rendered a PIL image into ``self.image``.
        # Save the frames directly with Pillow: matplotlib's FuncAnimation
        # re-renders only its bound figure every frame, so swept frames would
        # all show the first generated figure.
        if output_path.lower().endswith(".gif"):
            target = output_path
        else:
            target = output_path.rsplit(".", 1)[0] + ".gif"
            logger.warning(
                "ffmpeg export unavailable for create_animation; saving GIF at %s",
                target,
            )
        # ``fps`` sets a floor on the per-frame display time; ``duration``
        # splits the remaining budget evenly across the swept values.
        frame_ms = max(int(1000 / fps), int(duration * 1000 / len(frames)))
        frames[0].save(
            target,
            save_all=True,
            append_images=frames[1:],
            duration=frame_ms,
            loop=0,
        )
        return target

    def apply_texture(
        self, texture_type: str = "noise", **kwargs: Any
    ) -> "GenerativeMap":
        """
        Apply a texture overlay to the generated map.

        Args:
            texture_type: Type of texture ("noise", "pattern", "gradient")
            **kwargs: Texture parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the texture type is not supported
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        # Convert PIL image to numpy array for processing
        img_array = np.array(self.image)

        if texture_type == "noise":
            # Add noise texture
            intensity = kwargs.get("intensity", 0.1)
            noise = self.rng.normal(0, intensity, img_array.shape[:2])
            noise = np.stack([noise] * 3, axis=2) if img_array.ndim == 3 else noise

            # Blend noise with image
            textured = img_array.astype(float) + (noise * 255)
            textured = np.clip(textured, 0, 255).astype(np.uint8)

        elif texture_type == "pattern":
            # Add pattern overlay
            pattern_size = kwargs.get("pattern_size", 20)
            pattern_type = kwargs.get("pattern_type", "dots")

            # Create pattern mask
            if pattern_type == "dots":
                x = np.arange(0, img_array.shape[1], pattern_size)
                y = np.arange(0, img_array.shape[0], pattern_size)
                xx, yy = np.meshgrid(x, y)
                pattern = np.zeros_like(img_array[:, :, 0], dtype=np.uint8)
                for i in range(len(y)):
                    for j in range(len(x)):
                        pattern[
                            i * pattern_size : (i + 1) * pattern_size,
                            j * pattern_size : (j + 1) * pattern_size,
                        ] = 255 * (i + j) % 2

                # Apply pattern
                textured = img_array.copy()
                mask = pattern[:, :, np.newaxis] if img_array.ndim == 3 else pattern
                textured = textured * 0.7 + mask * 0.3

        elif texture_type == "gradient":
            # Add gradient overlay
            gradient_type = kwargs.get("gradient_type", "radial")
            intensity = kwargs.get("intensity", 0.3)

            if gradient_type == "radial":
                # Create radial gradient
                center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2
                y, x = np.ogrid[: img_array.shape[0], : img_array.shape[1]]
                gradient = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                gradient = 1 - (gradient / np.max(gradient))

                # Apply gradient
                textured = (
                    img_array.astype(float) * (1 - intensity)
                    + gradient[:, :, np.newaxis] * intensity * 255
                )
                textured = np.clip(textured, 0, 255).astype(np.uint8)

            else:
                raise ValueError(f"Unsupported gradient type: {gradient_type}")

        else:
            raise ValueError(f"Unsupported texture type: {texture_type}")

        # Update the image
        self.image = Image.fromarray(textured.astype(np.uint8))

        return self

    def blend_with(
        self, other_map: "GenerativeMap", alpha: float = 0.5
    ) -> "GenerativeMap":
        """
        Blend this map with another GenerativeMap.

        Args:
            other_map: Another GenerativeMap to blend with
            alpha: Blending ratio (0.0 = all this map, 1.0 = all other map)

        Returns:
            A new GenerativeMap with blended images

        Raises:
            ValueError: If either map has no generated image or images have different sizes
        """
        if self.image is None:
            raise ValueError("This map has no generated image.")

        if other_map.image is None:
            raise ValueError("Other map has no generated image.")

        self_arr = np.array(self.image)
        other_arr = np.array(other_map.image)

        if self_arr.shape != other_arr.shape:
            raise ValueError("Images must have the same dimensions for blending.")

        # Blend the images
        blended_image = (alpha * other_arr + (1 - alpha) * self_arr).astype(np.uint8)

        # Create new map with blended result
        blended_map = GenerativeMap()
        blended_map.image = Image.fromarray(blended_image)
        blended_map.metadata = {
            **self.metadata,
            "blended_with": other_map.metadata,
            "blend_alpha": alpha,
        }

        return blended_map

    def add_effects(self, effects: list[str], **kwargs: Any) -> "GenerativeMap":
        """
        Apply visual effects to the generated map.

        Args:
            effects: List of effects to apply ("blur", "sharpen", "edge_enhance", "emboss")
            **kwargs: Effect parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If an effect is not supported
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        from PIL import Image, ImageFilter, ImageEnhance

        img = (
            self.image
            if isinstance(self.image, Image.Image)
            else Image.fromarray(self.image)
        )

        for effect in effects:
            if effect == "blur":
                radius = kwargs.get("blur_radius", 1)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))

            elif effect == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)

            elif effect == "edge_enhance":
                img = img.filter(ImageFilter.EDGE_ENHANCE)

            elif effect == "emboss":
                img = img.filter(ImageFilter.EMBOSS)

            elif effect == "smooth":
                img = img.filter(ImageFilter.SMOOTH)

            elif effect == "brightness":
                factor = kwargs.get("brightness_factor", 1.0)
                enhancer: Any = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)

            elif effect == "contrast":
                factor = kwargs.get("contrast_factor", 1.0)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)

            elif effect == "saturation":
                factor = kwargs.get("saturation_factor", 1.0)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(factor)

            else:
                raise ValueError(f"Unsupported effect: {effect}")

        self.image = img

        return self

    def export_multi_format(
        self, base_path: str, formats: list[str] | None = None
    ) -> list[str]:
        """
        Export the map in multiple formats.

        Args:
            base_path: Base path for exported files (without extension)
            formats: List of formats to export ("png", "jpg", "svg", "pdf")

        Returns:
            List of exported file paths
        """
        if formats is None:
            formats = ["png", "jpg", "svg"]

        exported_paths: list[str] = []

        for fmt in formats:
            if fmt.lower() == "svg":
                # Export as SVG (requires matplotlib)
                output_path = f"{base_path}.svg"
                if self._figure is not None:
                    self._figure.savefig(output_path, format="svg", bbox_inches="tight")
                else:
                    # Convert image to SVG-like format
                    output_path = f"{base_path}.png"  # Fallback
            else:
                # Export as image
                output_path = f"{base_path}.{fmt.lower()}"

            # Save the current image
            if self.image is not None:
                directory = os.path.dirname(output_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)

                img = (
                    self.image
                    if isinstance(self.image, Image.Image)
                    else Image.fromarray(self.image)
                )
                img.save(output_path)
                exported_paths.append(output_path)

        return exported_paths

    def __repr__(self) -> str:
        """Return a string representation of the GenerativeMap object."""
        if self.data is None:
            return "GenerativeMap(No data loaded)"

        style = self.metadata.get("style", "unknown")
        region = self.metadata.get("region", "unknown")
        data_type = self.metadata.get("data_type", "unknown")

        return f"GenerativeMap(style='{style}', region='{region}', data_type='{data_type}')"
