"""
ProceduralArt module for creating procedural and algorithmic art from geospatial data.
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from geo_infer_art.core.aesthetics import ColorPalette

logger = logging.getLogger(__name__)


class ProceduralArt:
    """
    A class for creating procedural and algorithmic art from geospatial data.

    The ProceduralArt class provides methods for generating art through
    rule-based algorithms that can be seeded with geospatial parameters.

    Attributes:
        params: Parameters controlling the procedural generation
        image: The generated image as a numpy array
    """

    # Available procedural algorithms
    ALGORITHMS = [
        "l_system",
        "cellular_automata",
        "reaction_diffusion",
        "noise_field",
        "voronoi",
        "fractal_tree",
        "mandelbrot",
        "julia_set",
        "perlin_noise",
        "simplex_noise",
        "wave_function_collapse",
        "marching_squares",
        "space_colonization",
        "boids",
        "particle_system",
        "diffusion_limited_aggregation",
        "turtle_graphics",
        "sierpinski",
        "dragon_curve",
        "hilbert_curve",
        "koch_snowflake",
        "barnsley_fern",
        "ifs_fractal",
    ]

    def __init__(
        self,
        algorithm: str = "noise_field",
        params: dict | None = None,
        resolution: tuple[int, int] = (800, 800),
    ):
        """
        Initialize a ProceduralArt object.

        Args:
            algorithm: Name of the procedural algorithm to use
            params: Parameters for the algorithm
            resolution: Output image resolution (width, height)

        Raises:
            ValueError: If the algorithm is not supported
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Supported algorithms: "
                f"{', '.join(self.ALGORITHMS)}"
            )
        if (
            not isinstance(resolution, tuple)
            or len(resolution) != 2
            or resolution[0] <= 0
            or resolution[1] <= 0
        ):
            raise ValueError("Resolution must be a positive (width, height) tuple.")

        self.algorithm = algorithm
        self.params: dict[str, Any] = params or {}
        self.resolution: tuple[int, int] = resolution
        self.rng: np.random.Generator = np.random.default_rng(self.params.get("seed"))
        self.image: Image.Image | None = None
        self._figure: Figure | None = None

    @classmethod
    def from_geo_coordinates(
        cls,
        lat: float,
        lon: float,
        algorithm: str = "noise_field",
        additional_params: dict | None = None,
    ) -> "ProceduralArt":
        """
        Create procedural art seeded by geographic coordinates.

        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            algorithm: Name of the procedural algorithm to use
            additional_params: Additional parameters for the algorithm

        Returns:
            A new ProceduralArt object with generated art

        Raises:
            ValueError: If coordinates are out of range
        """
        # Validate coordinates
        if not -90 <= lat <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        if not -180 <= lon <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees.")

        # Initialize parameters with geo-coordinates
        params = additional_params or {}

        # Use coordinates to seed parameters
        # Normalize to 0-1 range
        norm_lat = (lat + 90) / 180
        norm_lon = (lon + 180) / 360

        # Base parameters from coordinates
        params.update(
            {
                "seed": int((norm_lat * 1000) + (norm_lon * 10000)),
                "x_influence": norm_lon,
                "y_influence": norm_lat,
                "geo_coordinates": (lat, lon),
            }
        )

        # Create the object
        art = cls(algorithm=algorithm, params=params)

        # Generate the art
        art.generate()

        return art

    @classmethod
    def from_geo_features(
        cls,
        feature_type: str,
        feature_count: int,
        algorithm: str = "l_system",
        additional_params: dict | None = None,
    ) -> "ProceduralArt":
        """
        Create procedural art based on geographic feature statistics.

        Args:
            feature_type: Type of geographic feature ("rivers", "mountains", "coastlines", etc.)
            feature_count: Number of features to simulate
            algorithm: Name of the procedural algorithm to use
            additional_params: Additional parameters for the algorithm

        Returns:
            A new ProceduralArt object with generated art

        Raises:
            ValueError: If the feature type is invalid
        """
        # Validate feature type
        valid_features = ["rivers", "mountains", "coastlines", "urban", "forest"]
        if feature_type not in valid_features:
            raise ValueError(
                f"Invalid feature type: {feature_type}. Valid types: "
                f"{', '.join(valid_features)}"
            )

        # Initialize parameters
        params = additional_params or {}

        # Set parameters based on feature type
        if feature_type == "rivers":
            params.update(
                {
                    "branching_factor": 0.7,
                    "curvature": 0.3,
                    "iteration_depth": min(feature_count, 10),
                    "color_palette": "blue",
                }
            )
            if algorithm == "noise_field":
                algorithm = "l_system"  # Override with better algorithm for rivers

        elif feature_type == "mountains":
            params.update(
                {
                    "roughness": 0.8,
                    "peaks": min(feature_count, 30),
                    "elevation_scale": 10 + min(feature_count, 20),
                    "color_palette": "earth",
                }
            )
            if algorithm == "l_system":
                algorithm = "noise_field"  # Better for mountains

        elif feature_type == "coastlines":
            params.update(
                {
                    "fractal_dimension": 1.2,
                    "jaggedness": 0.6,
                    "water_level": 0.5,
                    "color_palette": "ocean",
                }
            )

        elif feature_type == "urban":
            params.update(
                {
                    "grid_size": max(10, min(feature_count // 10, 50)),
                    "density": min(feature_count / 100, 0.8),
                    "regularity": 0.7,
                    "color_palette": "grayscale",
                }
            )
            if algorithm not in ["cellular_automata", "voronoi"]:
                algorithm = "cellular_automata"  # Better for urban

        elif feature_type == "forest":
            params.update(
                {
                    "tree_count": feature_count,
                    "clustering": 0.6,
                    "variation": 0.3,
                    "color_palette": "forest",
                }
            )
            if algorithm not in ["fractal_tree", "noise_field"]:
                algorithm = "fractal_tree"  # Better for forests

        # Add feature info to params
        params["feature_type"] = feature_type
        params["feature_count"] = feature_count

        # Create the object
        art = cls(algorithm=algorithm, params=params)

        # Generate the art
        art.generate()

        return art

    def generate(self) -> "ProceduralArt":
        """
        Generate the procedural art based on the selected algorithm and parameters.
        """
        # Every generator draws from this instance's Generator; a fixed
        # ``params["seed"]`` reproduces the output, and the process-wide
        # ``numpy.random`` state is never touched.
        self.rng = np.random.default_rng(self.params.get("seed"))

        # Dispatch on the algorithm name; every entry in ALGORITHMS maps to a
        # "_generate_<algorithm>" method on this class.
        generator = getattr(self, f"_generate_{self.algorithm}", None)
        if generator is None or not callable(generator):
            raise ValueError(
                f"Unsupported algorithm: {self.algorithm}. Supported algorithms: "
                f"{', '.join(self.ALGORITHMS)}"
            )
        generator()

        return self

    def _generate_noise_field(self) -> None:
        """
        Generate art using a noise field algorithm (Perlin, Simplex, etc.).
        """
        # Get parameters with defaults
        width, height = self.resolution
        octaves = self.params.get("octaves", 6)
        persistence = self.params.get("persistence", 0.5)
        lacunarity = self.params.get("lacunarity", 2.0)
        scale = self.params.get("scale", 100.0)

        # X and Y influence can be seeded by geo coordinates
        x_influence = self.params.get("x_influence", 1.0)
        y_influence = self.params.get("y_influence", 1.0)

        # Create the noise field

        # Simplified Perlin-like noise for demonstration
        # In production code, you'd use a proper noise library
        x = np.linspace(0, scale * x_influence, width)
        y = np.linspace(0, scale * y_influence, height)
        X, Y = np.meshgrid(x, y)

        # Generate multi-octave noise
        noise = np.zeros((height, width))
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            noise += (
                amplitude * np.sin(X * frequency * 0.1) * np.cos(Y * frequency * 0.1)
            )
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize noise to 0-1
        noise = (noise + max_value) / (2 * max_value)

        # Get color palette
        palette_name = self.params.get("color_palette", "viridis")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the noise field with the color palette
        ax.imshow(
            noise,
            cmap=palette.cmap,
            interpolation="bicubic",
            aspect="auto",
            extent=(0, width, 0, height),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_voronoi(self) -> None:
        """
        Generate art using Voronoi diagrams.
        """
        from scipy.spatial import Voronoi

        # Get parameters with defaults
        width, height = self.resolution
        num_points = self.params.get("num_points", 50)
        point_clustering = self.params.get("point_clustering", 0.0)  # 0.0-1.0
        edge_width = self.params.get("edge_width", 1.0)

        # Generate points
        # If clustering is 0, points are uniform random
        # If clustering is 1, points are clustered around centers
        if point_clustering < 0.1:
            # Uniform random points
            points = self.rng.random((num_points, 2))
            points[:, 0] *= width
            points[:, 1] *= height
        else:
            # Create clustered points
            num_clusters = max(1, int(num_points / 10))
            cluster_centers = self.rng.random((num_clusters, 2))
            cluster_centers[:, 0] *= width
            cluster_centers[:, 1] *= height

            points = np.zeros((num_points, 2))
            for i in range(num_points):
                # Pick a random cluster center
                center_idx = self.rng.integers(0, num_clusters)
                center = cluster_centers[center_idx]

                # Generate point with distance based on clustering parameter
                distance = self.rng.normal(0, width / 10 * (1 - point_clustering))
                angle = self.rng.uniform(0, 2 * np.pi)

                # Calculate the offset from center
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)

                # Ensure point is within bounds
                x = np.clip(center[0] + dx, 0, width)
                y = np.clip(center[1] + dy, 0, height)

                points[i] = [x, y]

        # Add corner points to ensure the diagram covers the full image
        corner_points = np.array(
            [
                [-width, -height],
                [-width, 2 * height],
                [2 * width, -height],
                [2 * width, 2 * height],
            ]
        )
        points = np.vstack([points, corner_points])

        # Create Voronoi diagram
        vor = Voronoi(points)

        # Get color palette
        palette_name = self.params.get("color_palette", "pastel")
        palette = ColorPalette.get_palette(palette_name)

        # Create figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot Voronoi regions with colors
        for i, region_idx in enumerate(vor.point_region):
            if i >= num_points:  # Skip the corner points
                continue

            region = vor.regions[region_idx]

            if -1 not in region and len(region) > 0:  # Skip unbounded regions
                # Get polygon vertices
                polygon = [vor.vertices[v] for v in region]
                if len(polygon) >= 3:  # Need at least 3 points for a polygon
                    # Select color based on position or index
                    color_idx = int(
                        (points[i][0] / width + points[i][1] / height) * 127
                    ) % len(palette.colors)
                    color = palette.colors[color_idx]

                    # Plot polygon
                    poly = plt.Polygon(
                        polygon,
                        fill=True,
                        facecolor=color,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=edge_width,
                    )
                    ax.add_patch(poly)

        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_l_system(self) -> None:
        """
        Generate art using an L-system (Lindenmayer system).
        """
        import math

        # Get parameters with defaults
        width, height = self.resolution
        axiom = self.params.get("axiom", "F")
        rules = self.params.get("rules", {"F": "F+F-F-F+F"})
        iterations = self.params.get("iterations", 3)
        angle = self.params.get("angle", 90)
        line_width = self.params.get("line_width", 1.5)

        # Apply L-system rules
        current = axiom
        for _ in range(iterations):
            next_gen = ""
            for char in current:
                if char in rules:
                    next_gen += rules[char]
                else:
                    next_gen += char
            current = next_gen

        # Get color palette
        palette_name = self.params.get("color_palette", "autumn")
        palette = ColorPalette.get_palette(palette_name)

        # Create figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Set up initial position and direction
        pos = np.array([0.1 * width, 0.1 * height])
        direction = 0  # Angle in degrees

        # Set up stack for storing positions and directions
        stack = []

        # Set up list to store all lines for efficient plotting
        lines = []

        # Track drawing bounds to ensure it fits on canvas
        min_x, max_x = pos[0], pos[0]
        min_y, max_y = pos[1], pos[1]

        # Parse and draw the L-system
        for char in current:
            if char == "F":  # Move forward and draw a line
                # Calculate new position
                rad = math.radians(direction)
                new_pos = pos + np.array([math.cos(rad), math.sin(rad)]) * (
                    width / (iterations * 10)
                )

                # Store line segment
                lines.append([pos[0], pos[1], new_pos[0], new_pos[1]])

                # Update position
                pos = new_pos

                # Update bounds
                min_x = min(min_x, pos[0])
                max_x = max(max_x, pos[0])
                min_y = min(min_y, pos[1])
                max_y = max(max_y, pos[1])

            elif char == "+":  # Turn left
                direction += angle
            elif char == "-":  # Turn right
                direction -= angle
            elif char == "[":  # Push current state onto stack
                stack.append((pos.copy(), direction))
            elif char == "]":  # Pop state from stack
                pos, direction = stack.pop()

        # Calculate scaling to fit the drawing on the canvas
        x_range = max_x - min_x
        y_range = max_y - min_y
        if x_range > 0 and y_range > 0:
            scale = min(0.8 * width / x_range, 0.8 * height / y_range)
            offset_x = width / 2 - (min_x + x_range / 2) * scale
            offset_y = height / 2 - (min_y + y_range / 2) * scale

            # Scale and translate all lines
            scaled_lines = []
            for line in lines:
                x1, y1, x2, y2 = line
                scaled_lines.append(
                    [
                        x1 * scale + offset_x,
                        y1 * scale + offset_y,
                        x2 * scale + offset_x,
                        y2 * scale + offset_y,
                    ]
                )

            # Draw all lines with color gradient
            num_lines = len(scaled_lines)
            for i, line in enumerate(scaled_lines):
                # Calculate color index based on line position
                progress = i / num_lines
                color_idx = int(progress * (len(palette.colors) - 1))
                color = palette.colors[color_idx]

                # Draw the line segment
                ax.plot(
                    [line[0], line[2]],
                    [line[1], line[3]],
                    color=color,
                    linewidth=line_width,
                    alpha=0.8,
                )

        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_cellular_automata(self) -> None:
        """
        Generate art using cellular automata (e.g., Conway's Game of Life).
        """
        # Get parameters with defaults
        width, height = self.resolution
        rule = self.params.get("rule", 30)  # Rule 30 is chaotic and interesting
        generations = self.params.get("generations", height)
        initial_state = self.params.get("initial_state", None)

        # Set up the cellular automaton grid
        cells = np.zeros((generations, width), dtype=np.uint8)

        # Initialize the first row
        if initial_state is None:
            # Default: single cell in the middle
            cells[0, width // 2] = 1
        else:
            # Use provided initial state or random
            if initial_state == "random":
                cells[0] = self.rng.integers(0, 2, width)
            else:
                for i, val in enumerate(initial_state[:width]):
                    cells[0, i] = 1 if val else 0

        # Generate subsequent generations using the specified rule
        for i in range(1, generations):
            for j in range(width):
                # Get the three cells above (with wraparound)
                left = cells[i - 1, (j - 1) % width]
                center = cells[i - 1, j]
                right = cells[i - 1, (j + 1) % width]

                # Convert the three cells to a binary pattern (0-7)
                pattern = (left << 2) | (center << 1) | right

                # Apply the rule
                cells[i, j] = 1 if (rule & (1 << pattern)) != 0 else 0

        # Get color palette
        palette_name = self.params.get("color_palette", "grayscale")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the cellular automaton
        ax.imshow(
            cells,
            cmap=palette.cmap,
            interpolation="nearest",
            aspect="auto",
            extent=(0, width, 0, height),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_reaction_diffusion(self) -> None:
        """
        Generate art using a reaction-diffusion system (e.g., Gray-Scott model).
        """
        # Get parameters with defaults
        width, height = self.resolution
        iterations = self.params.get("iterations", 50)
        du = self.params.get("diffusion_rate_a", 0.16)
        dv = self.params.get("diffusion_rate_b", 0.08)
        f = self.params.get("feed_rate", 0.035)
        k = self.params.get("kill_rate", 0.065)
        dt = self.params.get("time_step", 1.0)
        seed_type = self.params.get("seed_type", "center")

        # Gray-Scott reaction-diffusion implementation
        # Initialize with values for A (activator) and B (inhibitor)
        u = np.ones((height, width), dtype=np.float64)  # Substance U
        v = np.zeros((height, width), dtype=np.float64)  # Substance V

        # Create seed
        if seed_type == "random":
            for _ in range(10):
                y = self.rng.integers(height)
                x = self.rng.integers(width)
                r = self.rng.integers(3, 10)
                y_idxs, x_idxs = np.ogrid[-r : r + 1, -r : r + 1]
                mask = x_idxs**2 + y_idxs**2 <= r**2
                for yy in range(-r, r + 1):
                    for xx in range(-r, r + 1):
                        if mask[yy + r, xx + r]:
                            y_idx = (y + yy) % height
                            x_idx = (x + xx) % width
                            u[y_idx, x_idx] = 0.5
                            v[y_idx, x_idx] = 0.25
        else:  # Default: center
            r = min(width, height) // 4
            center_y, center_x = height // 2, width // 2
            y_idxs, x_idxs = np.ogrid[-r : r + 1, -r : r + 1]
            mask = x_idxs**2 + y_idxs**2 <= r**2
            for y in range(-r, r + 1):
                for x in range(-r, r + 1):
                    if mask[y + r, x + r]:
                        y_idx = (center_y + y) % height
                        x_idx = (center_x + x) % width
                        u[y_idx, x_idx] = 0.5
                        v[y_idx, x_idx] = 0.25

        # Run the simulation
        for _ in range(iterations):
            # Calculate Laplacian
            laplace_u = (
                np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
                - 4 * u
            )

            laplace_v = (
                np.roll(v, 1, axis=0)
                + np.roll(v, -1, axis=0)
                + np.roll(v, 1, axis=1)
                + np.roll(v, -1, axis=1)
                - 4 * v
            )

            # Gray-Scott equations
            uvv = u * v * v
            u_next = u + dt * (du * laplace_u - uvv + f * (1 - u))
            v_next = v + dt * (dv * laplace_v + uvv - (f + k) * v)

            u, v = u_next, v_next

        # Get color palette
        palette_name = self.params.get("color_palette", "ocean")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the result (v is usually more visually interesting)
        ax.imshow(
            v,
            cmap=palette.cmap,
            interpolation="bicubic",
            aspect="auto",
            extent=(0, width, 0, height),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_fractal_tree(self) -> None:
        """
        Generate art using a recursive fractal tree algorithm.
        """
        import math

        # Get parameters with defaults
        width, height = self.resolution
        depth = self.params.get("depth", 9)
        init_angle = self.params.get("initial_angle", 90)  # Degrees
        branch_angle = self.params.get("branch_angle", 25)  # Degrees
        shrink_factor = self.params.get("shrink_factor", 0.7)
        init_length = self.params.get("initial_length", height / 3)
        variation = self.params.get("variation", 0.2)  # Random variation factor
        line_width = self.params.get("line_width", 1.5)

        # Get color palette
        palette_name = self.params.get("color_palette", "forest")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Function to draw a branch recursively
        def draw_branch(
            x: float,
            y: float,
            length: float,
            angle: float,
            branch_depth: int,
            ax: Any,
        ) -> None:
            if branch_depth <= 0:
                return

            # Calculate end point
            angle_rad = math.radians(angle)
            end_x = x + length * math.cos(angle_rad)
            end_y = y + length * math.sin(angle_rad)

            # Determine color based on branch depth
            color_idx = int((depth - branch_depth) / depth * (len(palette.colors) - 1))
            color = palette.colors[color_idx]

            # Draw the branch
            ax.plot(
                [x, end_x],
                [y, end_y],
                color=color,
                linewidth=line_width * (branch_depth / depth),
                alpha=0.8,
            )

            # Add variation to parameters
            left_var = 1.0 + variation * (self.rng.random() - 0.5)
            right_var = 1.0 + variation * (self.rng.random() - 0.5)
            left_angle_var = branch_angle * (
                1.0 + variation * (self.rng.random() - 0.5)
            )
            right_angle_var = branch_angle * (
                1.0 + variation * (self.rng.random() - 0.5)
            )

            # Recursively draw left and right branches
            draw_branch(
                end_x,
                end_y,
                length * shrink_factor * left_var,
                angle + left_angle_var,
                branch_depth - 1,
                ax,
            )

            draw_branch(
                end_x,
                end_y,
                length * shrink_factor * right_var,
                angle - right_angle_var,
                branch_depth - 1,
                ax,
            )

        # Draw the initial trunk
        start_x = width / 2
        start_y = height * 0.1
        draw_branch(start_x, start_y, init_length, init_angle, depth, ax)

        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _figure_to_image(self) -> None:
        """Convert the matplotlib figure to a numpy image array."""
        import io

        if self._figure is None:
            return

        # Save figure to a buffer
        buf = io.BytesIO()
        self._figure.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)

        img = (
            Image.open(buf)
            .convert("RGBA")
            .resize(self.resolution, Image.Resampling.LANCZOS)
        )
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
        if "agg" in plt.get_backend().lower() or not plt.isinteractive():
            figure.canvas.draw()
            plt.close(figure)
            return
        plt.show()

    def _generate_mandelbrot(self) -> None:
        """
        Generate art using the Mandelbrot set.
        """
        # Get parameters with defaults
        width, height = self.resolution
        max_iter = self.params.get("max_iter", 100)
        zoom = self.params.get("zoom", 1.0)
        center_x = self.params.get("center_x", -0.5)
        center_y = self.params.get("center_y", 0.0)

        # Create coordinate grid
        x_min = center_x - 2.0 / zoom
        x_max = center_x + 2.0 / zoom
        y_min = center_y - 2.0 / zoom
        y_max = center_y + 2.0 / zoom

        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        # Compute Mandelbrot set
        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            M[mask] = i
            Z[mask] = Z[mask] ** 2 + C[mask]

        # Normalize and color
        M = M / max_iter

        # Get color palette
        palette_name = self.params.get("color_palette", "viridis")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the Mandelbrot set
        ax.imshow(
            M,
            cmap=palette.cmap,
            interpolation="bilinear",
            extent=(x_min, x_max, y_min, y_max),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_julia_set(self) -> None:
        """
        Generate art using a Julia set.
        """
        # Get parameters with defaults
        width, height = self.resolution
        max_iter = self.params.get("max_iter", 100)
        c_real = self.params.get("c_real", -0.7)
        c_imag = self.params.get("c_imag", 0.27015)
        zoom = self.params.get("zoom", 1.0)

        # Create coordinate grid
        x_min = -2.0 / zoom
        x_max = 2.0 / zoom
        y_min = -2.0 / zoom
        y_max = 2.0 / zoom

        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Julia set constant
        C = complex(c_real, c_imag)

        # Compute Julia set
        M = np.zeros(Z.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            M[mask] = i
            Z[mask] = Z[mask] ** 2 + C

        # Normalize and color
        M = M / max_iter

        # Get color palette
        palette_name = self.params.get("color_palette", "ocean")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the Julia set
        ax.imshow(
            M,
            cmap=palette.cmap,
            interpolation="bilinear",
            extent=(x_min, x_max, y_min, y_max),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_perlin_noise(self) -> None:
        """
        Generate art using Perlin noise.
        """
        # Get parameters with defaults
        width, height = self.resolution
        scale = self.params.get("scale", 100.0)
        octaves = self.params.get("octaves", 6)
        persistence = self.params.get("persistence", 0.5)
        lacunarity = self.params.get("lacunarity", 2.0)

        # X and Y influence can be seeded by geo coordinates
        x_influence = self.params.get("x_influence", 1.0)
        y_influence = self.params.get("y_influence", 1.0)

        # Create the noise field using multiple octaves
        np.zeros((height, width, 3), dtype=np.uint8)

        x = np.linspace(0, scale * x_influence, width)
        y = np.linspace(0, scale * y_influence, height)
        X, Y = np.meshgrid(x, y)

        # Generate multi-octave Perlin-like noise
        noise = np.zeros((height, width))
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            noise += (
                amplitude * np.sin(X * frequency * 0.1) * np.cos(Y * frequency * 0.1)
            )
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize noise to 0-1
        noise = (noise + max_value) / (2 * max_value)

        # Get color palette
        palette_name = self.params.get("color_palette", "viridis")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the noise field with the color palette
        ax.imshow(
            noise,
            cmap=palette.cmap,
            interpolation="bicubic",
            aspect="auto",
            extent=(0, width, 0, height),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    @staticmethod
    def _simplex_noise_2d(
        x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Evaluate classic 2D simplex noise (Gustavson-style simplicial lattice).

        The permutation table is drawn from ``rng``, so a seeded generator
        makes the noise deterministic.

        Returns values in approximately [-1, 1].
        """
        f2 = 0.5 * (np.sqrt(3.0) - 1.0)
        g2 = (3.0 - np.sqrt(3.0)) / 6.0
        gradients = np.array(
            [
                [1.0, 1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [-1.0, -1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )

        # Permutation table (doubled to avoid index wraparound arithmetic).
        perm = rng.permutation(256)
        perm = np.concatenate([perm, perm])

        # Skew input space onto the simplicial lattice.
        s = (x + y) * f2
        i = np.floor(x + s)
        j = np.floor(y + s)
        t = (i + j) * g2
        x0 = x - (i - t)
        y0 = y - (j - t)

        # Corner order of the enclosing simplex triangle.
        i1 = (x0 > y0).astype(np.int64)
        j1 = 1 - i1

        x1 = x0 - i1 + g2
        y1 = y0 - j1 + g2
        x2 = x0 - 1.0 + 2.0 * g2
        y2 = y0 - 1.0 + 2.0 * g2

        ii = i.astype(np.int64) & 255
        jj = j.astype(np.int64) & 255
        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        def _corner(tx: np.ndarray, ty: np.ndarray, gi: np.ndarray) -> np.ndarray:
            """Radial falloff dotted with the corner's gradient direction."""
            fall = 0.5 - tx * tx - ty * ty
            contrib = (fall**4) * (gradients[gi, 0] * tx + gradients[gi, 1] * ty)
            return np.where(fall > 0.0, contrib, 0.0)

        total = _corner(x0, y0, gi0) + _corner(x1, y1, gi1) + _corner(x2, y2, gi2)

        # Standard output scaling for 2D simplex noise.
        return 70.0 * total

    def _generate_simplex_noise(self) -> None:
        """
        Generate art using simplex noise.

        Unlike ``noise_field`` / ``perlin_noise``, which use a trigonometric
        approximation on a square lattice, this evaluates classic 2D simplex
        noise on a skewed simplicial lattice with per-vertex gradients.
        """
        width, height = self.resolution
        scale = self.params.get("scale", 100.0)
        octaves = self.params.get("octaves", 6)
        persistence = self.params.get("persistence", 0.5)
        lacunarity = self.params.get("lacunarity", 2.0)

        # X and Y influence can be seeded by geo coordinates
        x_influence = self.params.get("x_influence", 1.0)
        y_influence = self.params.get("y_influence", 1.0)

        # Same 0.1 spatial attenuation as the Perlin path keeps feature sizes
        # comparable across the two noise algorithms.
        x = np.linspace(0, scale * x_influence * 0.1, width)
        y = np.linspace(0, scale * y_influence * 0.1, height)
        X, Y = np.meshgrid(x, y)

        noise = np.zeros((height, width))
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            noise += amplitude * self._simplex_noise_2d(
                X * frequency, Y * frequency, self.rng
            )
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize noise to 0-1 (simplex output is approximately in [-1, 1])
        noise = np.clip((noise / max_value + 1.0) / 2.0, 0.0, 1.0)

        # Get color palette
        palette_name = self.params.get("color_palette", "viridis")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the noise field with the color palette
        ax.imshow(
            noise,
            cmap=palette.cmap,
            interpolation="bicubic",
            aspect="auto",
            extent=(0, width, 0, height),
        )

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_wave_function_collapse(self) -> None:
        """
        Generate art using Wave Function Collapse algorithm.
        """
        # Simplified implementation of WFC
        # In a full implementation, this would use proper WFC with tile constraints

        width, height = self.resolution
        tile_size = self.params.get("tile_size", 20)
        pattern_complexity = self.params.get("pattern_complexity", 0.7)

        # Create a simple tile-based pattern
        tiles_x = width // tile_size
        tiles_y = height // tile_size

        # Generate a pattern using simple rules
        pattern = self.rng.integers(0, 4, (tiles_y, tiles_x))  # 4 different tile types

        # Apply some smoothing based on neighbors
        for _ in range(3):  # Multiple passes for coherence
            new_pattern = pattern.copy()
            for i in range(1, tiles_y - 1):
                for j in range(1, tiles_x - 1):
                    # Simple rule: prefer tiles similar to neighbors
                    neighbors = [
                        pattern[i - 1, j],
                        pattern[i + 1, j],
                        pattern[i, j - 1],
                        pattern[i, j + 1],
                    ]
                    most_common = np.argmax(np.bincount(neighbors))
                    if self.rng.random() < pattern_complexity:
                        new_pattern[i, j] = most_common
            pattern = new_pattern

        # Create visual representation
        result = np.zeros((height, width, 3), dtype=np.uint8)

        colors = [
            [255, 100, 100],  # Red
            [100, 255, 100],  # Green
            [100, 100, 255],  # Blue
            [255, 255, 100],  # Yellow
        ]

        for i in range(tiles_y):
            for j in range(tiles_x):
                tile_type = pattern[i, j]
                color = colors[tile_type]

                # Fill tile area
                start_y = i * tile_size
                end_y = min((i + 1) * tile_size, height)
                start_x = j * tile_size
                end_x = min((j + 1) * tile_size, width)

                result[start_y:end_y, start_x:end_x] = color

        # Get color palette for additional styling
        palette_name = self.params.get("color_palette", "pastel")
        ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the pattern
        ax.imshow(result, interpolation="nearest")

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_marching_squares(self) -> None:
        """
        Generate art using Marching Squares algorithm for contour generation.
        """
        # Get parameters with defaults
        width, height = self.resolution
        threshold = self.params.get("threshold", 0.5)
        self.params.get("grid_size", 20)

        # Create a scalar field
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        # Create interesting scalar field
        field = (
            np.sin(X) * np.cos(Y)
            + 0.5 * np.sin(X * 2) * np.cos(Y * 2)
            + 0.25 * np.sin(X * 4) * np.cos(Y * 4)
        )

        # Apply threshold
        binary_field = (field > threshold).astype(int)

        # Get color palette
        palette_name = self.params.get("color_palette", "bright")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Plot the binary field
        ax.imshow(
            binary_field,
            cmap=palette.cmap,
            interpolation="nearest",
            alpha=0.8,
        )

        # Add contour lines
        ax.contour(field, levels=[threshold], colors="black", linewidths=1.0, alpha=0.6)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_space_colonization(self) -> None:
        """
        Generate art using Space Colonization algorithm for tree/branch growth.
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_seeds = self.params.get("num_seeds", 10)
        branch_length = self.params.get("branch_length", 15)
        influence_radius = self.params.get("influence_radius", 30)
        kill_distance = self.params.get("kill_distance", 5)

        # Create attraction points (leaves)
        attraction_points = []
        for _ in range(num_seeds):
            x = self.rng.integers(influence_radius, width - influence_radius)
            y = self.rng.integers(influence_radius, height - influence_radius)
            attraction_points.append([x, y])

        # Start with root branches
        branches = []
        root_x = width // 2
        root_y = height - 50
        branches.append([root_x, root_y, root_x, root_y - 20])  # Start with small trunk

        # Space colonization algorithm
        max_iterations = 1000
        for _ in range(max_iterations):
            if not attraction_points:
                break

            # Find closest attraction point to any branch
            closest_point = None
            closest_branch_idx = None
            min_distance = float("inf")

            for branch_idx, branch in enumerate(branches):
                branch_end_x, branch_end_y = branch[2], branch[3]

                for point in attraction_points:
                    distance = np.sqrt(
                        (point[0] - branch_end_x) ** 2 + (point[1] - branch_end_y) ** 2
                    )
                    if distance < min_distance and distance < influence_radius:
                        min_distance = distance
                        closest_point = point
                        closest_branch_idx = branch_idx

            if closest_point is None:
                break

            # Grow branch towards closest point
            assert closest_branch_idx is not None
            branch = branches[closest_branch_idx]
            start_x, start_y, end_x, end_y = branch

            # Calculate direction to attraction point
            dx = closest_point[0] - end_x
            dy = closest_point[1] - end_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 0:
                # Normalize direction
                dx /= distance
                dy /= distance

                # Grow branch
                new_end_x = end_x + dx * branch_length
                new_end_y = end_y + dy * branch_length

                # Create new branch segment
                new_branch = [end_x, end_y, new_end_x, new_end_y]
                branches.append(new_branch)

                # Remove attraction point if close enough
                if distance < kill_distance:
                    attraction_points.remove(closest_point)

        # Get color palette
        palette_name = self.params.get("color_palette", "forest")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Draw all branches
        for i, branch in enumerate(branches):
            start_x, start_y, end_x, end_y = branch

            # Color based on branch generation
            color_idx = min(i * 10, len(palette.colors) - 1)
            color = palette.colors[color_idx]

            # Draw branch with varying thickness
            line_width = max(0.5, 3.0 - i * 0.02)

            ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                color=color,
                linewidth=line_width,
                alpha=0.8,
            )

        # Draw remaining attraction points
        for point in attraction_points:
            ax.scatter(point[0], point[1], c="red", s=2, alpha=0.6)

        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_boids(self) -> None:
        """
        Generate art using Boids flocking algorithm.
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_boids = self.params.get("num_boids", 50)
        max_speed = self.params.get("max_speed", 2.0)
        perception = self.params.get("perception", 50)
        separation_weight = self.params.get("separation_weight", 1.0)
        alignment_weight = self.params.get("alignment_weight", 1.0)
        cohesion_weight = self.params.get("cohesion_weight", 1.0)
        iterations = self.params.get("iterations", 100)

        # Initialize boids with random positions and velocities
        boids = []
        for _ in range(num_boids):
            x = self.rng.uniform(0, width)
            y = self.rng.uniform(0, height)
            vx = self.rng.uniform(-max_speed, max_speed)
            vy = self.rng.uniform(-max_speed, max_speed)
            boids.append([x, y, vx, vy])

        # Run boids simulation
        for _ in range(iterations):
            for i, boid in enumerate(boids):
                x, y, vx, vy = boid

                # Find nearby boids
                nearby_boids = []
                for j, other_boid in enumerate(boids):
                    if i != j:
                        dx = other_boid[0] - x
                        dy = other_boid[1] - y
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance < perception:
                            nearby_boids.append(other_boid)

                if nearby_boids:
                    # Separation: steer to avoid crowding
                    separation_x, separation_y = 0, 0
                    for other in nearby_boids:
                        dx = other[0] - x
                        dy = other[1] - y
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance > 0:
                            separation_x -= dx / distance
                            separation_y -= dy / distance
                    separation_x *= separation_weight
                    separation_y *= separation_weight

                    # Alignment: steer towards average heading
                    avg_vx = sum(b[2] for b in nearby_boids) / len(nearby_boids)
                    avg_vy = sum(b[3] for b in nearby_boids) / len(nearby_boids)
                    alignment_x = (avg_vx - vx) * alignment_weight
                    alignment_y = (avg_vy - vy) * alignment_weight

                    # Cohesion: steer towards average position
                    avg_x = sum(b[0] for b in nearby_boids) / len(nearby_boids)
                    avg_y = sum(b[1] for b in nearby_boids) / len(nearby_boids)
                    cohesion_x = (avg_x - x) * cohesion_weight
                    cohesion_y = (avg_y - y) * cohesion_weight

                    # Update velocity
                    vx += separation_x + alignment_x + cohesion_x
                    vy += separation_y + alignment_y + cohesion_y

                    # Limit speed
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > max_speed:
                        vx = (vx / speed) * max_speed
                        vy = (vy / speed) * max_speed

                    # Update position
                    x = (x + vx) % width
                    y = (y + vy) % height

                    boids[i] = [x, y, vx, vy]

        # Create visualization
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Draw boids trails (simplified - just points for now)
        positions = np.array(boids)[:, :2]
        ax.scatter(positions[:, 0], positions[:, 1], c="cyan", s=2, alpha=0.8)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_particle_system(self) -> None:
        """
        Generate art using a particle system simulation.
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_particles = self.params.get("num_particles", 1000)
        gravity = self.params.get("gravity", 0.1)
        damping = self.params.get("damping", 0.99)
        iterations = self.params.get("iterations", 200)

        # Initialize particles
        particles = []
        for _ in range(num_particles):
            x = self.rng.uniform(0, width)
            y = self.rng.uniform(0, height)
            vx = self.rng.uniform(-2, 2)
            vy = self.rng.uniform(-2, 2)
            life = self.rng.uniform(50, 200)
            particles.append([x, y, vx, vy, life])

        # Run particle simulation
        for _ in range(iterations):
            for i, particle in enumerate(particles):
                x, y, vx, vy, life = particle

                # Apply forces
                vy += gravity  # Gravity
                vx *= damping  # Air resistance
                vy *= damping

                # Update position
                x += vx
                y += vy

                # Boundary conditions
                if x < 0 or x > width:
                    vx *= -0.8
                    x = np.clip(x, 0, width)
                if y < 0 or y > height:
                    vy *= -0.8
                    y = np.clip(y, 0, height)

                # Update life
                life -= 1

                particles[i] = [x, y, vx, vy, life]

        # Get color palette
        palette_name = self.params.get("color_palette", "bright")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Draw particles
        for particle in particles:
            x, y, _, _, life = particle
            if life > 0:
                # Color based on life remaining
                color_idx = int((life / 200) * (len(palette.colors) - 1))
                color = palette.colors[color_idx]
                ax.scatter(x, y, c=color, s=1, alpha=0.6)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_diffusion_limited_aggregation(self) -> None:
        """
        Generate art using Diffusion-Limited Aggregation (DLA).
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_particles = self.params.get("num_particles", 2000)
        stickiness = self.params.get("stickiness", 0.1)
        iterations = self.params.get("iterations", 100)

        # Initialize DLA structure
        center_x, center_y = width // 2, height // 2
        structure = set([(center_x, center_y)])

        # Run DLA simulation
        for _ in range(iterations):
            # Add multiple particles per iteration
            for _ in range(num_particles // iterations):
                # Start particle at random position on edge
                if self.rng.random() < 0.5:
                    x = self.rng.integers(0, width)
                    y = 0 if self.rng.random() < 0.5 else height
                else:
                    x = 0 if self.rng.random() < 0.5 else width
                    y = self.rng.integers(0, height)

                # Move particle until it sticks or goes out of bounds
                stuck = False
                while not stuck and 0 <= x < width and 0 <= y < height:
                    # Random walk
                    dx = self.rng.choice([-1, 0, 1])
                    dy = self.rng.choice([-1, 0, 1])

                    x += dx
                    y += dy

                    # Check if particle sticks to structure
                    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

                    if any(neighbor in structure for neighbor in neighbors):
                        if self.rng.random() < stickiness:
                            structure.add((x, y))
                            stuck = True

        # Get color palette
        palette_name = self.params.get("color_palette", "autumn")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Draw DLA structure
        structure_points = list(structure)
        x_coords = [p[0] for p in structure_points]
        y_coords = [p[1] for p in structure_points]

        # Color based on distance from center
        distances = [
            np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            for x, y in structure_points
        ]
        max_dist = max(distances, default=0.0) or 1.0
        color_indices = [
            int((dist / max_dist) * (len(palette.colors) - 1)) for dist in distances
        ]

        colors = [palette.colors[idx] for idx in color_indices]

        ax.scatter(x_coords, y_coords, c=colors, s=1, alpha=0.8)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_turtle_graphics(self) -> None:
        """
        Generate art using turtle graphics (Logo-style drawing).
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_turtles = self.params.get("num_turtles", 3)
        iterations = self.params.get("iterations", 500)
        angle_step = self.params.get("angle_step", 60)

        # Initialize turtles
        turtles = []
        for i in range(num_turtles):
            x = width // 2 + self.rng.uniform(-50, 50)
            y = height // 2 + self.rng.uniform(-50, 50)
            angle = self.rng.uniform(0, 360)
            color_idx = i % len(ColorPalette.get_palette("bright").colors)
            turtles.append([x, y, angle, color_idx])

        # Get color palette
        palette = ColorPalette.get_palette("bright")

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Run turtle simulation
        for _ in range(iterations):
            for turtle in turtles:
                x, y, angle, t_color = turtle
                color_idx = int(t_color)

                # Move forward
                step_size = 2
                new_x = x + step_size * np.cos(np.radians(angle))
                new_y = y + step_size * np.sin(np.radians(angle))

                # Draw line
                color = palette.colors[color_idx]
                ax.plot([x, new_x], [y, new_y], color=color, linewidth=1, alpha=0.7)

                # Update position and angle
                turtle[0] = new_x
                turtle[1] = new_y
                turtle[2] = (turtle[2] + angle_step + self.rng.uniform(-10, 10)) % 360

                # Boundary conditions
                if not (0 <= turtle[0] < width and 0 <= turtle[1] < height):
                    turtle[0] = width // 2
                    turtle[1] = height // 2

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_sierpinski(self) -> None:
        """
        Generate art using Sierpinski triangle or carpet.
        """
        # Get parameters with defaults
        width, height = self.resolution
        iterations = self.params.get("iterations", 8)
        triangle = self.params.get("triangle", True)

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        if triangle:
            # Sierpinski triangle
            points = np.array([[0, 0], [width, 0], [width // 2, height]])

            # Generate triangle points
            triangle_points: list[Any] = [points]
            for _ in range(iterations):
                new_points: list[Any] = []
                for triangle in triangle_points:
                    # Calculate midpoints
                    p1, p2, p3 = triangle
                    m12 = (p1 + p2) / 2
                    m13 = (p1 + p3) / 2
                    m23 = (p2 + p3) / 2

                    # Add smaller triangles
                    new_points.extend([[p1, m12, m13], [p2, m12, m23], [p3, m13, m23]])
                triangle_points = new_points

            # Draw all triangles
            colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(triangle_points)))
            for i, triangle in enumerate(triangle_points):
                triangle = np.array(triangle)
                ax.fill(triangle[:, 0], triangle[:, 1], color=colors[i], alpha=0.6)

        else:
            # Sierpinski carpet
            carpet = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Remove squares recursively
            def remove_square(x: int, y: int, size: int) -> None:
                if size < 1:
                    return
                # Remove center third
                remove_size = size // 3
                start_x = x + remove_size
                start_y = y + remove_size
                carpet[
                    start_y : start_y + remove_size, start_x : start_x + remove_size
                ] = 0

                # Recurse on remaining squares
                for dx in [0, remove_size * 2]:
                    for dy in [0, remove_size * 2]:
                        remove_square(x + dx, y + dy, remove_size)

            remove_square(0, 0, min(width, height))

            ax.imshow(carpet)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_dragon_curve(self) -> None:
        """
        Generate art using Dragon curve fractal.
        """
        # Get parameters with defaults
        width, height = self.resolution
        iterations = self.params.get("iterations", 12)
        line_length = self.params.get("line_length", 5)

        # Get color palette
        palette = ColorPalette.get_palette("ocean")

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Generate Dragon curve
        def dragon_curve(
            x: float, y: float, length: float, angle: float, depth: int
        ) -> None:
            if depth <= 0:
                return

            # Draw line segment
            end_x = x + length * np.cos(np.radians(angle))
            end_y = y + length * np.sin(np.radians(angle))

            color_idx = (iterations - depth) % len(palette.colors)
            color = palette.colors[color_idx]
            ax.plot([x, end_x], [y, end_y], color=color, linewidth=1, alpha=0.8)

            # Recurse with Dragon curve rule
            dragon_curve(end_x, end_y, length, angle + 90, depth - 1)
            dragon_curve(end_x, end_y, length, angle - 90, depth - 1)

        # Start drawing
        start_x, start_y = width // 2, height // 2
        dragon_curve(start_x, start_y, line_length, 0, iterations)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_hilbert_curve(self) -> None:
        """
        Generate art using Hilbert curve space-filling curve.
        """
        # Get parameters with defaults
        width, height = self.resolution
        order = self.params.get("order", 5)

        # Get color palette
        palette = ColorPalette.get_palette("sunset")

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Generate Hilbert curve points
        def hilbert_curve(
            order: int, x: int, y: int, lg: int, i1: int, i2: int
        ) -> None:
            if order == 0:
                color_idx = (lg // 2) % len(palette.colors)
                color = palette.colors[color_idx]
                ax.scatter(x, y, c=color, s=1, alpha=0.8)
                return

            lg //= 2

            hilbert_curve(order - 1, x + i1 * lg, y + i1 * lg, lg, i1, 1 - i2)
            hilbert_curve(order - 1, x + i2 * lg, y + (1 - i2) * lg, lg, 1 - i2, i2)
            hilbert_curve(
                order - 1, x + (1 - i2) * lg, y + (1 - i1) * lg, lg, 1 - i2, 1 - i1
            )
            hilbert_curve(order - 1, x + (1 - i1) * lg, y + i2 * lg, lg, i1, i2)

        # Calculate curve size
        curve_size = 2**order
        scale = min(width, height) / curve_size

        # Draw Hilbert curve
        hilbert_curve(order, 0, 0, int(scale), 0, 0)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_koch_snowflake(self) -> None:
        """
        Generate art using Koch snowflake fractal.
        """
        # Get parameters with defaults
        width, height = self.resolution
        iterations = self.params.get("iterations", 5)

        # Get color palette
        palette_name = self.params.get("color_palette", "ocean")
        palette = ColorPalette.get_palette(palette_name)

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Generate Koch snowflake
        def koch_curve(x1: float, y1: float, x2: float, y2: float, depth: int) -> None:
            if depth <= 0:
                color_idx = (depth + iterations) % len(palette.colors)
                color = palette.colors[color_idx]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.8)
                return

            # Calculate intermediate points
            dx = x2 - x1
            dy = y2 - y1

            # Koch curve points
            x3 = x1 + dx / 3
            y3 = y1 + dy / 3

            x4 = x1 + 2 * dx / 3
            y4 = y1 + 2 * dy / 3

            # Peak point (60 degree rotation)
            x5 = (
                x3
                + (x4 - x3) * np.cos(np.radians(60))
                - (y4 - y3) * np.sin(np.radians(60))
            )
            y5 = (
                y3
                + (x4 - x3) * np.sin(np.radians(60))
                + (y4 - y3) * np.cos(np.radians(60))
            )

            # Recurse
            koch_curve(x1, y1, x3, y3, depth - 1)
            koch_curve(x3, y3, x5, y5, depth - 1)
            koch_curve(x5, y5, x4, y4, depth - 1)
            koch_curve(x4, y4, x2, y2, depth - 1)

        # Start with equilateral triangle
        size = min(width, height) * 0.8
        center_x, center_y = width // 2, height // 2

        # Triangle vertices
        h = size * np.sqrt(3) / 2
        x1, y1 = center_x, center_y - h / 2
        x2, y2 = center_x - size / 2, center_y + h / 2
        x3, y3 = center_x + size / 2, center_y + h / 2

        # Draw Koch snowflake
        koch_curve(x1, y1, x2, y2, iterations)
        koch_curve(x2, y2, x3, y3, iterations)
        koch_curve(x3, y3, x1, y1, iterations)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_barnsley_fern(self) -> None:
        """
        Generate art using Barnsley fern fractal.
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_points = self.params.get("num_points", 50000)

        # Get color palette
        palette = ColorPalette.get_palette("forest")

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Barnsley fern transformation probabilities and matrices
        transformations = [
            ([0, 0, 0, 0.16, 0, 0], 0.01),  # Stem
            ([0.85, 0.04, -0.04, 0.85, 0, 1.60], 0.85),  # Left branch
            ([0.20, -0.26, 0.23, 0.22, 0, 1.60], 0.07),  # Right branch
            ([-0.15, 0.28, 0.26, 0.24, 0, 0.44], 0.07),  # Right branch 2
        ]

        # Generate fern points
        x: float = 0.0
        y: float = 0.0
        points_x, points_y = [], []

        for _ in range(num_points):
            # Choose random transformation
            r = self.rng.random()
            cumulative_prob: float = 0.0

            for transform, prob in transformations:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    a, b, c, d, e, f = transform
                    new_x = a * x + b * y + e
                    new_y = c * x + d * y + f
                    x, y = new_x, new_y

                    points_x.append(x * 50 + width // 2)
                    points_y.append(height - (y * 50 + height // 3))
                    break

        # Draw fern
        color_idx = 0
        for i in range(0, len(points_x), 100):  # Sample points for coloring
            color = palette.colors[color_idx % len(palette.colors)]
            ax.scatter(points_x[i], points_y[i], c=color, s=0.5, alpha=0.6)
            color_idx += 1

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def _generate_ifs_fractal(self) -> None:
        """
        Generate art using Iterated Function System (IFS) fractals.
        """
        # Get parameters with defaults
        width, height = self.resolution
        num_points = self.params.get("num_points", 20000)
        num_transforms = self.params.get("num_transforms", 4)

        # Get color palette
        palette = ColorPalette.get_palette("bright")

        # Create a figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100, facecolor="black"
        )

        # Generate random IFS transformations
        transforms = []
        for _ in range(num_transforms):
            # Random affine transformation
            a = self.rng.uniform(-0.5, 0.5)
            b = self.rng.uniform(-0.5, 0.5)
            c = self.rng.uniform(-0.5, 0.5)
            d = self.rng.uniform(-0.5, 0.5)
            e = self.rng.uniform(-2, 2)
            f = self.rng.uniform(-2, 2)

            prob = 1.0 / num_transforms
            transforms.append(([a, b, c, d, e, f], prob))

        # Generate IFS points
        x: float = 0.0
        y: float = 0.0
        points_x, points_y = [], []

        for _ in range(num_points):
            # Choose random transformation
            r = self.rng.random()
            cumulative_prob: float = 0.0

            for transform, prob in transforms:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    a, b, c, d, e, f = transform
                    new_x = a * x + b * y + e
                    new_y = c * x + d * y + f
                    x, y = new_x, new_y

                    points_x.append(x * 100 + width // 2)
                    points_y.append(y * 100 + height // 2)
                    break

        # Draw IFS fractal
        for i in range(len(points_x)):
            color_idx = i % len(palette.colors)
            color = palette.colors[color_idx]
            ax.scatter(points_x[i], points_y[i], c=color, s=0.3, alpha=0.4)

        # Remove axes for artistic effect
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        # Store the figure
        self._figure = fig

        # Convert figure to image
        self._figure_to_image()

    def __repr__(self) -> str:
        """Return a string representation of the ProceduralArt object."""
        if self.image is None:
            return f"ProceduralArt(algorithm='{self.algorithm}', not generated)"

        width, height = (
            self.image.size
            if isinstance(self.image, Image.Image)
            else (self.image.shape[1], self.image.shape[0])
        )
        return f"ProceduralArt(algorithm='{self.algorithm}', {width}x{height})"
