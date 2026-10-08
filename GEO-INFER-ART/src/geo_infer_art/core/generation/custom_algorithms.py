"""
Custom algorithm framework for creating user-defined procedural art algorithms.
"""

import logging
import inspect
import json
import math
from typing import Any
from collections.abc import Callable, Mapping

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

    def __init__(
        self, algorithm_registry: Mapping[str, Callable] | None = None
    ) -> None:
        """Initialize with trusted callables; persisted files contain only keys.

        The builtin keys are ``spiral``, ``cellular_growth`` and
        ``fractal_landscape``. Applications supply their own stable keys here
        before loading a file. A file cannot import modules or define code.
        """
        self._algorithm_registry = {
            "spiral": example_spiral_algorithm,
            "cellular_growth": example_cellular_growth_algorithm,
            "fractal_landscape": example_fractal_landscape_algorithm,
        }
        for key, function in (algorithm_registry or {}).items():
            if not isinstance(key, str) or not key or not callable(function):
                raise ValueError("Registry entries require nonempty keys and callables")
            if (
                key in self._algorithm_registry
                and self._algorithm_registry[key] is not function
            ):
                raise ValueError(f"Cannot replace builtin registry key '{key}'")
            self._algorithm_registry[key] = function
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
        if not isinstance(name, str) or not name:
            raise ValueError("Algorithm name must be a nonempty string")
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
        try:
            sig.bind(data=None, params={}, width=1, height=1)
        except TypeError as exc:
            raise ValueError(
                "Algorithm must accept data, params, width and height as keywords"
            ) from exc

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
        """Save registry references and metadata, never function source.

        Every function must occur in the trusted registry supplied at
        construction. In-memory callables without a registry key cannot be
        persisted. Legacy source-bearing files are unsupported.
        """
        algorithms_data: dict[str, dict[str, Any]] = {}
        for name, func in self.registered_algorithms.items():
            keys = sorted(
                key
                for key, function in self._algorithm_registry.items()
                if function is func
            )
            if not keys:
                raise ValueError(f"Algorithm '{name}' has no trusted registry key")
            algorithms_data[name] = {
                "metadata": self.algorithm_metadata[name],
                "registry_key": keys[0],
            }
        payload = {"schema_version": 1, "algorithms": algorithms_data}
        serialized = json.dumps(payload, indent=2, allow_nan=False)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(serialized)

    def load_algorithms_from_file(self, filepath: str) -> None:
        """Load schema version 1 using only the trusted callable registry.

        Validation is atomic: malformed metadata, duplicate names, unsupported
        versions, source-bearing files and unknown keys raise
        ``ValueError`` without registering any entries. No persisted value is
        passed to import, exec or eval.
        """

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key '{key}'")
                result[key] = value
            return result

        def reject_constant(value: str) -> Any:
            raise ValueError(f"Non-finite JSON value '{value}' is unsupported")

        def finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"Non-finite JSON value '{value}' is unsupported")
            return parsed

        with open(filepath, encoding="utf-8") as f:
            payload = json.load(
                f,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
                parse_float=finite_float,
            )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "algorithms"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["algorithms"], dict)
        ):
            raise ValueError("Expected registry algorithms file with schema_version 1")

        pending = CustomAlgorithmFramework(self._algorithm_registry)
        for name, entry in payload["algorithms"].items():
            if (
                not isinstance(name, str)
                or not name
                or name in self.registered_algorithms
            ):
                raise ValueError(
                    f"Invalid or already registered algorithm name '{name}'"
                )
            if not isinstance(entry, dict) or set(entry) != {
                "registry_key",
                "metadata",
            }:
                raise ValueError(f"Invalid registry entry '{name}'")
            key, metadata = entry["registry_key"], entry["metadata"]
            if not isinstance(key, str) or key not in self._algorithm_registry:
                raise ValueError(f"Unknown registry key for algorithm '{name}'")
            if (
                not isinstance(metadata, dict)
                or not isinstance(metadata.get("description"), str)
                or not isinstance(metadata.get("parameters"), dict)
                or not isinstance(metadata.get("example_usage"), str)
            ):
                raise ValueError(f"Invalid metadata for algorithm '{name}'")
            pending.register_algorithm(
                name,
                self._algorithm_registry[key],
                metadata["description"],
                metadata["parameters"],
                metadata["example_usage"],
            )
        self.registered_algorithms.update(pending.registered_algorithms)
        self.algorithm_metadata.update(pending.algorithm_metadata)


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
            (x, y), radius, facecolor=color, alpha=0.6, edgecolor="white", linewidth=1
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
