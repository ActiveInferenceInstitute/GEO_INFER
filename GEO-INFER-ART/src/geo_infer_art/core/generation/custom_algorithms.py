"""
Custom algorithm framework for creating user-defined procedural art algorithms.
"""

import logging
import os
import inspect
import json
from typing import Any
from collections.abc import Callable
import types

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from geo_infer_art.core.aesthetics import ColorPalette


logger = logging.getLogger(__name__)


class CustomAlgorithmFramework:
    """
    Framework for creating and managing custom procedural art algorithms.

    This class allows users to define their own algorithms for generating art
    from geospatial data, with a consistent interface and validation system.
    """

    def __init__(self) -> None:
        """Initialize the custom algorithm framework."""
        self.registered_algorithms: dict[str, Callable] = {}
        self.algorithm_metadata: dict[str, dict[str, Any]] = {}

    def register_algorithm(
        self,
        name: str,
        algorithm_function: Callable,
        description: str = "",
        parameters: dict | None = None,
        example_usage: str = "",
    ) -> None:
        """
        Register a custom algorithm.

        Args:
            name: Unique name for the algorithm
            algorithm_function: Function that implements the algorithm
            description: Description of what the algorithm does
            parameters: Dictionary of parameter descriptions
            example_usage: Example of how to use the algorithm

        Raises:
            ValueError: If algorithm name already exists or function is invalid
        """
        if name in self.registered_algorithms:
            raise ValueError(f"Algorithm '{name}' already registered")

        if not callable(algorithm_function):
            raise ValueError("Algorithm must be a callable function")

        # Validate function signature
        sig = inspect.signature(algorithm_function)
        required_params = ["data", "params", "width", "height"]

        for param in required_params:
            if param not in sig.parameters:
                raise ValueError(f"Algorithm function must have parameter '{param}'")

        self.registered_algorithms[name] = algorithm_function
        self.algorithm_metadata[name] = {
            "description": description,
            "parameters": parameters or {},
            "example_usage": example_usage,
            "signature": str(sig),
        }

    def unregister_algorithm(self, name: str) -> None:
        """
        Unregister a custom algorithm.

        Args:
            name: Name of the algorithm to remove

        Raises:
            ValueError: If algorithm is not registered
        """
        if name not in self.registered_algorithms:
            raise ValueError(f"Algorithm '{name}' not registered")

        del self.registered_algorithms[name]
        del self.algorithm_metadata[name]

    def get_algorithm_info(self, name: str) -> dict:
        """
        Get information about a registered algorithm.

        Args:
            name: Name of the algorithm

        Returns:
            Dictionary with algorithm information

        Raises:
            ValueError: If algorithm is not registered
        """
        if name not in self.registered_algorithms:
            raise ValueError(f"Algorithm '{name}' not registered")

        return self.algorithm_metadata[name]

    def list_algorithms(self) -> list[str]:
        """List all registered algorithm names."""
        return list(self.registered_algorithms.keys())

    def execute_algorithm(
        self, name: str, data: Any, width: int = 800, height: int = 800, **params: Any
    ) -> Any:
        """
        Execute a registered custom algorithm.

        Args:
            name: Name of the algorithm to execute
            data: Input data for the algorithm
            width: Width of the output image
            height: Height of the output image
            **params: Additional parameters for the algorithm

        Returns:
            Algorithm output (typically a numpy array or matplotlib figure)

        Raises:
            ValueError: If algorithm is not registered or execution fails
        """
        if name not in self.registered_algorithms:
            raise ValueError(f"Algorithm '{name}' not registered")

        algorithm = self.registered_algorithms[name]

        try:
            result = algorithm(data=data, params=params, width=width, height=height)
            return result

        except Exception as e:
            raise ValueError(f"Algorithm '{name}' execution failed: {str(e)}") from e

    def save_algorithms_to_file(self, filepath: str) -> None:
        """
        Save registered algorithms to a JSON file for persistence.

        Args:
            filepath: Path to save the algorithms
        """
        algorithms_data = {}

        for name, func in self.registered_algorithms.items():
            # Get function source code
            try:
                source = inspect.getsource(func)
            except (OSError, TypeError):
                source = "Function source not available"

            algorithms_data[name] = {
                "metadata": self.algorithm_metadata[name],
                "source": source,
                "function_name": getattr(func, "__name__", name),
            }

        with open(filepath, "w") as f:
            json.dump(algorithms_data, f, indent=2)

    def load_algorithms_from_file(self, filepath: str) -> None:
        """
        Load algorithms from a JSON file written by :meth:`save_algorithms_to_file`.

        Every entry must carry ``metadata``, ``source`` and ``function_name``;
        the whole file is validated before any source is executed. Entries
        whose source fails to compile or does not define ``function_name``
        are skipped with a warning.

        Args:
            filepath: Path to the algorithms file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If the file is not a mapping of entries in the current
                format (for example an entry without ``function_name``)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Algorithms file not found: {filepath}")

        with open(filepath) as f:
            algorithms_data = json.load(f)

        if not isinstance(algorithms_data, dict):
            raise ValueError(
                f"Invalid algorithms file {filepath}: expected a JSON object "
                "mapping algorithm names to entries"
            )
        required_keys = ("metadata", "source", "function_name")
        for name, data in algorithms_data.items():
            missing = [
                key
                for key in required_keys
                if not isinstance(data, dict) or key not in data
            ]
            if missing:
                raise ValueError(
                    f"Invalid algorithms file {filepath}: entry '{name}' is "
                    f"missing {', '.join(missing)}"
                )

        for name, data in algorithms_data.items():
            metadata = data["metadata"]
            source = data["source"]
            function_name = data["function_name"]

            # Recreate the function from its saved source
            try:
                module = types.ModuleType(f"geo_infer_art_loaded_{name}")
                exec(compile(source, f"<algorithm:{name}>", "exec"), module.__dict__)

                algorithm_function = getattr(module, function_name, None)
                if not inspect.isfunction(algorithm_function):
                    raise ValueError(
                        f"saved source does not define function '{function_name}'"
                    )

                # Re-register the algorithm
                self.register_algorithm(
                    name=name,
                    algorithm_function=algorithm_function,
                    description=metadata["description"],
                    parameters=metadata["parameters"],
                    example_usage=metadata["example_usage"],
                )
            except Exception as e:
                logger.warning("Could not load algorithm '%s': %s", name, e)


# Example custom algorithms for demonstration


def example_spiral_algorithm(
    data: Any, params: dict, width: int, height: int
) -> Figure:
    """
    Example custom algorithm that creates spiral patterns.

    Args:
        data: Input data (unused in this example)
        params: Parameters including 'spirals', 'colors', etc.
        width: Output width
        height: Output height

    Returns:
        Matplotlib figure
    """
    spirals = params.get("spirals", 3)
    colors = params.get("colors", ["red", "green", "blue"])

    fig, ax = plt.subplots(
        figsize=(width / 100, height / 100), dpi=100, facecolor="black"
    )

    for i in range(spirals):
        # Create spiral pattern
        theta = np.linspace(0, 4 * np.pi, 1000)
        r = theta / (4 * np.pi) * min(width, height) / 4
        x = width // 2 + r * np.cos(theta + i * 2 * np.pi / spirals)
        y = height // 2 + r * np.sin(theta + i * 2 * np.pi / spirals)

        color = colors[i % len(colors)]
        ax.plot(x, y, color=color, linewidth=2, alpha=0.7)

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    return fig


def example_cellular_growth_algorithm(
    data: Any, params: dict, width: int, height: int
) -> Figure:
    """
    Example algorithm simulating cellular growth patterns.

    Args:
        data: Input data (can influence growth patterns)
        params: Parameters including 'seed_points', 'growth_rate' and an
            optional 'seed' (int or ``np.random.Generator``) for placement.
        width: Output width
        height: Output height

    Returns:
        Matplotlib figure
    """
    rng = np.random.default_rng(params.get("seed"))
    seed_points = params.get("seed_points", 5)
    growth_rate = params.get("growth_rate", 1.5)
    max_radius = params.get("max_radius", min(width, height) / 4)

    fig, ax = plt.subplots(
        figsize=(width / 100, height / 100), dpi=100, facecolor="black"
    )

    # Initialize with seed points
    cells = []
    for i in range(seed_points):
        x = rng.uniform(width * 0.2, width * 0.8)
        y = rng.uniform(height * 0.2, height * 0.8)
        cells.append([x, y, 0, i % len(["red", "green", "blue", "yellow"])])

    # Simulate growth
    for _ in range(50):
        new_cells = cells.copy()
        for cell in cells:
            x, y, radius, color_idx = cell
            if radius < max_radius:
                # Grow cell
                new_radius = radius + growth_rate
                new_cells[new_cells.index(cell)] = [x, y, new_radius, color_idx]

        cells = new_cells

    # Draw cells
    colors = ["red", "green", "blue", "yellow"]
    for cell in cells:
        x, y, radius, color_idx = cell
        color = colors[int(color_idx)]

        # Draw cell as circle
        circle = plt.Circle(
            (x, y), radius, color=color, alpha=0.6, edgecolor="white", linewidth=1
        )
        ax.add_patch(circle)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    return fig


def example_fractal_landscape_algorithm(
    data: Any, params: dict, width: int, height: int
) -> Figure:
    """
    Example algorithm creating fractal landscape patterns.

    Args:
        data: Input data (can influence landscape features)
        params: Parameters including 'octaves', 'persistence', etc.
        width: Output width
        height: Output height

    Returns:
        Matplotlib figure
    """
    octaves = params.get("octaves", 6)
    persistence = params.get("persistence", 0.5)
    scale = params.get("scale", 100.0)

    # Create fractal noise
    x = np.linspace(0, scale, width)
    y = np.linspace(0, scale, height)
    X, Y = np.meshgrid(x, y)

    # Generate multi-octave noise
    noise = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        noise += amplitude * np.sin(X * frequency * 0.1) * np.cos(Y * frequency * 0.1)
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2

    # Normalize noise
    noise = (noise + max_value) / (2 * max_value)

    # Get color palette
    palette_name = params.get("color_palette", "earth")
    palette = ColorPalette.get_palette(palette_name)

    # Create visualization
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Plot the landscape
    ax.imshow(noise, cmap=palette.cmap, interpolation="bicubic", aspect="auto")

    # Add contour lines
    contour_levels = np.linspace(0, 1, 11)
    ax.contour(noise, levels=contour_levels, colors="black", linewidths=0.5, alpha=0.3)

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    return fig
