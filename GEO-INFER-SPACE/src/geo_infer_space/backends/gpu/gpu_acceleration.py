"""Float64 numeric distance kernels with lazy, optional GPU execution.

NumPy is the CPU reference. H3 topology always executes through host H3.
Automatic numeric dispatch falls back with diagnostics; explicit GPU requests
fail if unavailable or if execution fails. Importing this module never imports
an accelerator library; ``get_backend_diagnostics`` reports capability.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import operator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0
DEFAULT_CHUNK_SIZE = 1024
_GPU_BACKENDS = ("cupy", "torch", "jax")


class AcceleratorUnavailableError(RuntimeError):
    """An explicitly requested GPU cannot execute the float64 contract."""


@dataclass(frozen=True)
class _ArrayBackend:
    name: str
    xp: Any
    to_device: Callable[[Any], Any]
    to_host: Callable[[Any], Any]


@dataclass(frozen=True)
class _Probe:
    installed: bool
    backend: _ArrayBackend | None
    reason: str | None = None


_CPU = _ArrayBackend("cpu", np, np.asarray, np.asarray)


@lru_cache(maxsize=3)
def _probe_backend(name: str) -> _Probe:
    """Import on demand and exercise a float64 allocation on a usable GPU."""
    installed = False
    try:
        if importlib.util.find_spec(name) is None:
            return _Probe(False, None, "library is not installed")
        installed = True
        module = importlib.import_module(name)
        if name == "cupy":
            if module.cuda.runtime.getDeviceCount() < 1:
                return _Probe(True, None, "no CUDA device")
            backend = _ArrayBackend(name, module, module.asarray, module.asnumpy)
        elif name == "torch":
            if not module.cuda.is_available():
                return _Probe(True, None, "no CUDA device")
            backend = _ArrayBackend(
                name,
                module,
                lambda value: module.as_tensor(
                    value, dtype=module.float64, device="cuda"
                ),
                lambda value: value.detach().cpu().numpy(),
            )
        else:
            devices = [
                device for device in module.devices() if device.platform == "gpu"
            ]
            if not devices:
                return _Probe(
                    True, None, "no GPU device (CPU-only JAX is not acceleration)"
                )
            if not module.config.x64_enabled:
                return _Probe(True, None, "JAX float64 requires JAX_ENABLE_X64=1")
            xp = importlib.import_module("jax.numpy")
            backend = _ArrayBackend(
                name,
                xp,
                lambda value: module.device_put(
                    np.asarray(value, dtype=np.float64), devices[0]
                ),
                np.asarray,
            )
        sample = backend.to_device(np.array([1.0], dtype=np.float64))
        checked = np.asarray(backend.to_host(backend.xp.sqrt(sample)))
        if checked.dtype != np.float64 or not np.array_equal(checked, [1.0]):
            return _Probe(True, None, "device did not preserve float64")
        return _Probe(True, backend)
    except Exception as exc:
        # Broken binary installs and unavailable drivers must not break CPU use.
        return _Probe(installed, None, f"{type(exc).__name__}: {exc}")


def get_backend_diagnostics(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Report library presence separately from usable float64 GPU execution.

    Probes are cached. Pass refresh=True after a driver/device/config change.
    """
    if refresh:
        _probe_backend.cache_clear()
    result = {}
    for name in _GPU_BACKENDS:
        probe = _probe_backend(name)
        result[name] = {
            "installed": probe.installed,
            "usable": probe.backend is not None,
            "reason": probe.reason,
        }
    return result


def get_available_backends() -> list[str]:
    """Return usable GPU backends in automatic dispatch preference order."""
    return [name for name in _GPU_BACKENDS if _probe_backend(name).backend is not None]


def is_accelerator_available() -> bool:
    """Return whether a GPU can execute the float64 numeric contract."""
    return bool(get_available_backends())


def _validated_points(points: Any, name: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape == (0,):
        arr = arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an (N, 2) array of (lat, lng) coordinates")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} coordinates must be finite")
    if (np.abs(arr[:, 0]) > 90).any() or (np.abs(arr[:, 1]) > 180).any():
        raise ValueError(
            f"{name} latitude/longitude must be within [-90, 90]/[-180, 180]"
        )
    return arr


def _positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _chunk_size(value: Any) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError("chunk_size must be a positive integer") from exc
    if isinstance(value, bool) or result <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return result


def _numeric_block(backend: _ArrayBackend, a: Any, b: Any, radius: float | None) -> Any:
    """Shared array formula; radius=None selects Euclidean distance."""
    xp = backend.xp
    a = backend.to_device(a)
    b = backend.to_device(b)
    if radius is None:
        # Accumulate dimensions without allocating an (N, M, D) tensor.
        result = xp.abs(a[:, None, 0] - b[None, :, 0])
        for column in range(1, a.shape[1]):
            result = xp.hypot(result, a[:, None, column] - b[None, :, column])
    else:
        a = a * (np.pi / 180.0)
        b = b * (np.pi / 180.0)
        dlat = a[:, None, 0] - b[None, :, 0]
        dlon = a[:, None, 1] - b[None, :, 1]
        h = (
            xp.sin(dlat / 2) ** 2
            + xp.cos(a[:, None, 0]) * xp.cos(b[None, :, 0]) * xp.sin(dlon / 2) ** 2
        )
        result = (2 * xp.arcsin(xp.sqrt(xp.clip(h, 0.0, 1.0)))) * radius
    host = np.asarray(backend.to_host(result))
    if host.dtype != np.float64 or not np.isfinite(host).all():
        raise FloatingPointError("distance kernel did not return finite float64 values")
    return host


class _Execution:
    """Per-call backend and fallback state, never global last-call metadata."""

    def __init__(self, requested: str, diagnostics: dict[str, Any] | None) -> None:
        if requested not in ("auto", "cpu", *_GPU_BACKENDS):
            raise ValueError("backend must be auto, cpu, cupy, torch, or jax")
        self.requested = requested
        self.backend = _CPU
        self.diagnostics = diagnostics if diagnostics is not None else {}
        self.diagnostics.clear()
        self.diagnostics.update(
            requested_backend=requested,
            backend="none",
            used_backends=[],
            fallback_reason=None,
        )
        if requested == "cpu":
            return
        reasons = []
        for name in _GPU_BACKENDS if requested == "auto" else (requested,):
            probe = _probe_backend(name)
            if probe.backend is not None:
                self.backend = probe.backend
                return
            reasons.append(f"{name}: {probe.reason}")
        reason = "; ".join(reasons)
        if requested != "auto":
            raise AcceleratorUnavailableError(reason)
        self.diagnostics["fallback_reason"] = reason
        logger.info("Automatic distance execution uses CPU: %s", reason)

    def block(self, a: Any, b: Any, radius: float | None) -> Any:
        try:
            result = _numeric_block(self.backend, a, b, radius)
        except Exception as exc:
            if self.backend.name == "cpu":
                raise
            if self.requested != "auto":
                raise RuntimeError(
                    f"{self.backend.name} distance execution failed"
                ) from exc
            reason = (
                f"{self.backend.name} execution failed: {type(exc).__name__}: {exc}"
            )
            logger.warning("%s; falling back to CPU", reason)
            self.diagnostics["fallback_reason"] = reason
            self.backend = _CPU
            result = _numeric_block(self.backend, a, b, radius)
        used = self.diagnostics["used_backends"]
        if self.backend.name not in used:
            used.append(self.backend.name)
        self.diagnostics["backend"] = used[0] if len(used) == 1 else "mixed"
        return result


def _blocks(
    a: Any, b: Any, chunk_size: int, execution: _Execution, radius: float | None
) -> Iterator[tuple[int, int, Any]]:
    for row in range(0, len(a), chunk_size):
        for col in range(0, len(b), chunk_size):
            yield (
                row,
                col,
                execution.block(
                    a[row : row + chunk_size], b[col : col + chunk_size], radius
                ),
            )


def pairwise_haversine_kernel(
    points_a: Any,
    points_b: Any,
    radius_km: float = EARTH_RADIUS_KM,
    *,
    backend: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    diagnostics: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """Return (N, M) float64 great-circle distances in km.

    Coordinates are finite (lat, longitude) degrees in their geographic bounds.
    The output matrix is retained; intermediate arrays are bounded by chunk_size.
    diagnostics, when supplied, is replaced with actual per-call execution data.
    """
    a = _validated_points(points_a, "points_a")
    b = _validated_points(points_b, "points_b")
    radius = _positive(radius_km, "radius_km")
    size = _chunk_size(chunk_size)
    execution = _Execution(backend, diagnostics)
    out = np.empty((len(a), len(b)), dtype=np.float64)
    for row, col, block in _blocks(a, b, size, execution, radius):
        out[row : row + block.shape[0], col : col + block.shape[1]] = block
    return out


def euclidean_distance_kernel(
    points_a: Any,
    points_b: Any,
    *,
    backend: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    diagnostics: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """Return (N, M) float64 distances for finite (N, D)/(M, D) points, D>=1."""
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] < 1:
        raise ValueError(
            "points_a and points_b must be (N, D) and (M, D) arrays with D>=1"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("coordinates must be finite")
    size = _chunk_size(chunk_size)
    execution = _Execution(backend, diagnostics)
    out = np.empty((len(a), len(b)), dtype=np.float64)
    for row, col, block in _blocks(a, b, size, execution, None):
        out[row : row + block.shape[0], col : col + block.shape[1]] = block
    return out


def _labels(labels: Sequence[int] | None, length: int, name: str) -> list[int]:
    if labels is None:
        return list(range(length))
    if len(labels) != length:
        raise ValueError(f"{name} must contain one integer label per point")
    try:
        if any(isinstance(label, (bool, np.bool_)) for label in labels):
            raise TypeError("boolean label")
        return [operator.index(label) for label in labels]
    except TypeError as exc:
        raise ValueError(f"{name} must contain integer labels") from exc


def gpu_spatial_join_by_distance(
    points_a: Any,
    points_b: Any,
    max_distance_km: float,
    label_offsets_a: Sequence[int] | None = None,
    label_offsets_b: Sequence[int] | None = None,
    radius_km: float = EARTH_RADIUS_KM,
    *,
    backend: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Join points with separation <= a finite positive radius in bounded tiles.

    Pairs and unmatched values are sorted, unique integer labels (row indices
    when labels are absent). A group is matched if any of its points matches.
    Working distance storage is O(chunk_size**2); output can still be O(N*M).
    """
    threshold = _positive(max_distance_km, "max_distance_km")
    radius = _positive(radius_km, "radius_km")
    a = _validated_points(points_a, "points_a")
    b = _validated_points(points_b, "points_b")
    la = _labels(label_offsets_a, len(a), "label_offsets_a")
    lb = _labels(label_offsets_b, len(b), "label_offsets_b")
    size = _chunk_size(chunk_size)
    execution = _Execution(backend, diagnostics)
    pairs: set[tuple[int, int]] = set()
    for row, col, block in _blocks(a, b, size, execution, radius):
        rows, cols = np.nonzero(block <= threshold)
        pairs.update((la[row + int(r)], lb[col + int(c)]) for r, c in zip(rows, cols))
    matched_a = {a_label for a_label, _ in pairs}
    matched_b = {b_label for _, b_label in pairs}
    return sorted(pairs), sorted(set(la) - matched_a), sorted(set(lb) - matched_b)


def spatial_join_kernel(
    cells_a: Sequence[str],
    cells_b: Sequence[str],
    join_type: str = "intersects",
    h3_module: Any = None,
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Host H3 topology join, preserving input order (including pair duplicates).

    'intersects' means identical or distance-1 neighbors at the same resolution;
    contains/within use strict H3 ancestry, not geographic polygon containment;
    each cell's own resolution determines ancestry. Invalid cells raise instead
    of silently disappearing.
    """
    if h3_module is None:
        h3_module = importlib.import_module("h3")
    if join_type not in {"intersects", "contains", "within"}:
        raise ValueError("join_type must be intersects, contains, or within")
    resolutions = {
        cell: h3_module.get_resolution(cell) for cell in (*cells_a, *cells_b)
    }
    matches = []
    for cell_a in cells_a:
        neighbors = (
            set(h3_module.grid_disk(cell_a, 1)) if join_type == "intersects" else set()
        )
        for cell_b in cells_b:
            if join_type == "intersects":
                matched = cell_b in neighbors
            elif join_type == "contains":
                matched = (
                    resolutions[cell_a] < resolutions[cell_b]
                    and h3_module.cell_to_parent(cell_b, resolutions[cell_a]) == cell_a
                )
            else:
                matched = (
                    resolutions[cell_a] > resolutions[cell_b]
                    and h3_module.cell_to_parent(cell_a, resolutions[cell_b]) == cell_b
                )
            if matched:
                matches.append((cell_a, cell_b))
    matched_a = {a for a, _ in matches}
    matched_b = {b for _, b in matches}
    return (
        matches,
        [a for a in dict.fromkeys(cells_a) if a not in matched_a],
        [b for b in dict.fromkeys(cells_b) if b not in matched_b],
    )


def h3_grid_distance_kernel(
    cells_a: Sequence[str], cells_b: Sequence[str], h3_module: Any = None
) -> npt.NDArray[np.int64]:
    """Return host H3 (N, M) int64 distances; incomparable pairs are -1.

    Mixed resolutions, invalid cells, and unsupported paths across pentagons
    retain the documented tolerant sentinel. No accelerator is used or probed.
    """
    if h3_module is None:
        h3_module = importlib.import_module("h3")
    # h3 4.x has no CellIndexError. Tolerant failures arrive in three
    # families: invalid/mixed-resolution cells (H3ValueError, also a
    # ValueError), pentagon-spanning navigation (H3GridNavigationError,
    # RuntimeError family), and failed distance lookups (H3FailedError).
    # Everything else -- e.g. a broken backend raising AttributeError --
    # propagates instead of the silent -1 sentinel.
    tolerant: tuple = (ValueError, KeyError)
    for family_name in ("H3GridNavigationError", "H3FailedError"):
        family = getattr(h3_module, family_name, None)
        if isinstance(family, type) and issubclass(family, BaseException):
            tolerant = tolerant + (family,)
    out = np.full((len(cells_a), len(cells_b)), -1, dtype=np.int64)
    for i, cell_a in enumerate(cells_a):
        for j, cell_b in enumerate(cells_b):
            try:
                out[i, j] = int(h3_module.grid_distance(cell_a, cell_b))
            except tolerant:
                pass
    return out
