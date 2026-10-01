"""Processing algorithm registry (GeoLibre-style).

A small, dependency-free registry that mirrors the shape of GeoLibre's
``ProcessingAlgorithm`` contract (``packages/processing/src/types.ts``):
each algorithm has a stable ``id``, a human ``name`` and ``description``, a
declared ``parameters`` schema, and a ``run(context)`` callable. This gives
GEO-INFER a uniform way to expose spatial operations (H3 grid creation, layer
bounds, feature counts, statistics, ...) through SPACE, API, and APP surfaces
without each module inventing its own tool-dispatch convention.

The registry is deliberately minimal and synchronous so it can be used from
FastAPI endpoints and notebook dashboards alike. A ``ProcessingContext`` wraps
the layers and parameters handed to a run; the return value of ``run`` is
opaque (algorithms may log, mutate the context, or return a result), matching
GeoLibre's ``void``-oriented design while still allowing a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Iterator, Mapping, Sequence
import builtins


@dataclass(frozen=True)
class ParameterSpec:
    """Declared parameter for a processing algorithm."""

    id: str
    label: str
    required: bool = False
    default: Any = None
    type: str = "auto"  # e.g. "layer", "number", "string", "boolean"


@dataclass(frozen=True)
class ProcessingAlgorithm:
    """A registered, runnable processing algorithm."""

    id: str
    name: str
    description: str
    run: Callable[[ProcessingContext], Any] = field(repr=False)
    parameters: Sequence[ParameterSpec] = ()


@dataclass
class ProcessingContext:
    """Execution context passed to an algorithm's ``run`` callable.

    ``layers`` holds the available layer descriptors (opaque dicts with at
    least an ``id``), and ``parameters`` holds the resolved values keyed by
    parameter id. ``log`` collects human-readable progress notes.
    """

    layers: Sequence[Mapping[str, Any]] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Append a progress note to this context's log."""
        self.logs.append(message)


class AlgorithmRegistry:
    """Register and dispatch processing algorithms by stable id."""

    def __init__(self, algorithms: Sequence[ProcessingAlgorithm] | None = None) -> None:
        self._algorithms: dict[str, ProcessingAlgorithm] = {}
        for algorithm in algorithms or ():
            self.register(algorithm)

    def register(self, algorithm: ProcessingAlgorithm) -> None:
        """Register an algorithm, raising on a duplicate id."""
        if algorithm.id in self._algorithms:
            raise ValueError(f"algorithm already registered: {algorithm.id}")
        self._algorithms[algorithm.id] = algorithm

    def get(self, algorithm_id: str) -> ProcessingAlgorithm:
        """Look up an algorithm by id.

        Raises:
            KeyError: If no algorithm is registered under ``algorithm_id``.
        """
        if algorithm_id not in self._algorithms:
            raise KeyError(f"unknown algorithm: {algorithm_id}")
        return self._algorithms[algorithm_id]

    def contains(self, algorithm_id: str) -> bool:
        """Return whether an algorithm id is registered."""
        return algorithm_id in self._algorithms

    def list(self) -> builtins.list[ProcessingAlgorithm]:
        """Return registered algorithms sorted by id."""
        return [self._algorithms[k] for k in sorted(self._algorithms)]

    def run(
        self,
        algorithm_id: str,
        context: ProcessingContext,
    ) -> Any:
        """Run the algorithm ``algorithm_id`` against ``context``.

        Raises:
            KeyError: If the algorithm is unknown.
        """
        algorithm = self.get(algorithm_id)
        return algorithm.run(context)


# -- a couple of reference algorithms (mirroring GeoLibre's registry.ts) ----


def _get_layer(
    context: ProcessingContext,
    param_id: str = "layer",
) -> Mapping[str, Any] | None:
    layer_id = context.parameters.get(param_id)
    for layer in context.layers:
        if layer.get("id") == layer_id:
            return layer
    return None


def build_reference_registry() -> AlgorithmRegistry:
    """Return a registry with a small set of reference algorithms.

    The two reference algorithms mirror GeoLibre's ``calculate-bounds`` and
    ``count-features`` and serve as a template and as registry tests; GEO-INFER
    domain modules register their own real algorithms on top.
    """
    registry = AlgorithmRegistry()

    registry.register(
        ProcessingAlgorithm(
            id="calculate-bounds",
            name="Calculate layer bounds",
            description="Compute the bounding box of a GeoJSON layer",
            parameters=[ParameterSpec(id="layer", label="Layer", required=True)],
            run=lambda ctx: _reference_bounds(ctx),
        )
    )
    registry.register(
        ProcessingAlgorithm(
            id="count-features",
            name="Count features",
            description="Count features in a GeoJSON layer",
            parameters=[ParameterSpec(id="layer", label="Layer", required=True)],
            run=lambda ctx: _reference_count(ctx),
        )
    )
    return registry


def _reference_bounds(context: ProcessingContext) -> Any:
    layer = _get_layer(context)
    if layer is None:
        context.log("Error: layer not found")
        return None
    geojson = layer.get("geojson")
    xs: list[float] = []
    ys: list[float] = []
    for feature in (geojson or {}).get("features", []):
        coords = _walk_coordinates(feature.get("geometry", {}))
        for x, y in coords:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        context.log("Error: layer has no geometries")
        return None
    bounds = [min(xs), min(ys), max(xs), max(ys)]
    context.log(f"Bounds: [{', '.join(f'{v:.6f}' for v in bounds)}]")
    return bounds


def _reference_count(context: ProcessingContext) -> Any:
    layer = _get_layer(context)
    if layer is None:
        context.log("Error: layer not found")
        return None
    features = (layer.get("geojson") or {}).get("features", [])
    count = len(features)
    context.log(f"Feature count: {count}")
    return count


def _walk_coordinates(geometry: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Yield ``(x, y)`` pairs for any GeoJSON geometry."""
    result: list[tuple[float, float]] = []
    _collect(geometry, result)
    return result


def _collect(geometry: Mapping[str, Any], out: list[tuple[float, float]]) -> None:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type in ("Point", "MultiPoint"):
        for item in _flatten_points(coords):
            if _is_point(item):
                out.append((float(item[0]), float(item[1])))
    elif geom_type in (
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    ):
        for position in _iter_positions(coords):
            out.append((float(position[0]), float(position[1])))


def _flatten_points(coords: Any) -> list[Any]:
    if not isinstance(coords, list):
        return []
    if (
        coords
        and isinstance(coords[0], list)
        and isinstance(coords[0][0], (int, float))
    ):
        return [coords]
    flattened: list[Any] = []
    for item in coords:
        flattened.extend(_flatten_points(item) if isinstance(item, list) else [])
    return flattened


def _iter_positions(coords: Any) -> Iterator[list[Any]]:
    if not isinstance(coords, list):
        return
    for ring_or_line in coords:
        if not isinstance(ring_or_line, list):
            continue
        for position in ring_or_line:
            if (
                isinstance(position, list)
                and position
                and isinstance(position[0], (int, float))
            ):
                yield position


def _is_point(item: Any) -> bool:
    return (
        isinstance(item, list)
        and len(item) >= 2
        and isinstance(item[0], (int, float))
        and isinstance(item[1], (int, float))
    )


__all__ = [
    "ParameterSpec",
    "ProcessingAlgorithm",
    "ProcessingContext",
    "AlgorithmRegistry",
    "build_reference_registry",
]
