"""CPU references, optional GPU numeric execution, and exact host H3 topology.

Failure tests use explicit simulated backends; they do not establish real GPU
hardware correctness. Actual detected GPUs are compared with the CPU reference.
"""

from __future__ import annotations


import numpy as np
import pytest

import h3

from geo_infer_space.backends.gpu.gpu_acceleration import (
    HAS_CUPY,
    HAS_GPU,
    HAS_JAX,
    HAS_TORCH,
    euclidean_distance_kernel,
    get_available_backends,
    gpu_spatial_join_by_distance,
    h3_grid_distance_kernel,
    is_accelerator_available,
    pairwise_haversine_kernel,
    spatial_join_kernel,
)
from geo_infer_space.backends.gpu import gpu_acceleration as gpu_mod


def _make_cells(resolution: int = 9, n: int = 10) -> list[str]:
    return [
        h3.latlng_to_cell(37.0 + i * 0.01, -122.0 + i * 0.01, resolution)
        for i in range(n)
    ]


def _make_parents(resolution: int = 7, n: int = 5) -> list[str]:
    return [
        h3.latlng_to_cell(37.0 + i * 0.1, -122.0 + i * 0.1, resolution)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Kernel-level tests (CPU reference path is always runnable).
# ---------------------------------------------------------------------------
def test_pairwise_haversine_kernel_shape_and_values() -> None:
    a = np.array([[37.0, -122.0], [38.0, -122.5]])
    b = np.array([[37.0, -122.0], [40.0, -120.0]])
    dist = pairwise_haversine_kernel(a, b)
    assert dist.shape == (2, 2)
    # Same point => zero distance
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-9)
    # Distinct points have positive separation.
    assert dist[0, 1] > 0.0
    assert dist[1, 1] > 0.0
    assert dist[1, 0] > 0.0
    assert dist.dtype == np.float64


def test_pairwise_haversine_kernel_is_symmetric() -> None:
    """Reversing the two point sets must yield the transposed distance matrix."""
    a = np.array([[37.0, -122.0], [40.0, -120.0], [34.0, -118.0]])
    b = np.array([[36.0, -115.0], [33.0, -117.0]])
    d1 = pairwise_haversine_kernel(a, b)
    d2 = pairwise_haversine_kernel(b, a)
    np.testing.assert_allclose(d1, d2.T, atol=1e-6)


def test_pairwise_haversine_kernel_matches_reference() -> None:
    """Cross-check the kernel against an independent scalar haversine."""
    a = np.array([[37.7749, -122.4194], [34.0522, -118.2437], [40.7128, -74.0060]])
    b = np.array([[37.7749, -122.4194], [36.1699, -115.1398]])

    result = pairwise_haversine_kernel(a, b)

    def _hav(
        lat1: float, lon1: float, lat2: float, lon2: float, r: float = 6371.0
    ) -> float:
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp1, dp2 = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        hv = np.sin(dp1 / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dp2 / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(hv))

    expected = np.array(
        [
            [
                _hav(a[0, 0], a[0, 1], b[0, 0], b[0, 1]),
                _hav(a[0, 0], a[0, 1], b[1, 0], b[1, 1]),
            ],
            [
                _hav(a[1, 0], a[1, 1], b[0, 0], b[0, 1]),
                _hav(a[1, 0], a[1, 1], b[1, 0], b[1, 1]),
            ],
            [
                _hav(a[2, 0], a[2, 1], b[0, 0], b[0, 1]),
                _hav(a[2, 0], a[2, 1], b[1, 0], b[1, 1]),
            ],
        ]
    )
    # The kernel computes in float32 before casting the result to float64;
    # ~4,000 km distances carry ~1e-7 relative (sub-metre absolute) rounding,
    # so the reference comparison uses float32-appropriate tolerances rather
    # than float64 defaults.
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-4)


def test_pairwise_haversine_kernel_requires_n2() -> None:
    with pytest.raises(ValueError):
        pairwise_haversine_kernel(np.array([37.0, -122.0]), np.array([[37.0, -122.0]]))
    with pytest.raises(ValueError):
        pairwise_haversine_kernel(np.array([[37.0, -122.0]]), np.array([37.0]))


def test_pairwise_haversine_kernel_empty() -> None:
    res = pairwise_haversine_kernel(np.zeros((0, 2)), np.zeros((3, 2)))
    assert res.shape == (0, 3)
    res2 = pairwise_haversine_kernel(np.zeros((2, 2)), np.zeros((0, 2)))
    assert res2.shape == (2, 0)


def test_euclidean_distance_kernel() -> None:
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [1.0, 1.0]])
    dist = euclidean_distance_kernel(a, b)
    assert dist[0, 0] == pytest.approx(0.0)
    assert dist[1, 1] == pytest.approx(np.sqrt((3 - 1) ** 2 + (4 - 1) ** 2))
    assert dist.dtype == np.float64


def test_euclidean_distance_kernel_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        euclidean_distance_kernel(np.zeros((2, 2)), np.zeros((2, 3)))


def test_gpu_spatial_join_by_distance() -> None:
    # Two co-located points and two far-apart points.
    a = np.array([[37.7749, -122.4194], [34.0522, -118.2437], [40.7128, -74.0060]])
    b = np.array([[37.7749, -122.4194], [40.7128, -74.0060]])
    pairs, unmatched_a, unmatched_b = gpu_spatial_join_by_distance(
        a, b, max_distance_km=20.0
    )
    # (0,0) SF~SF match; (2,1) NYC~NYC match; LA (index 1) unmatched.
    assert pairs == [(0, 0), (2, 1)]
    assert unmatched_a == [1]
    assert unmatched_b == []


def test_gpu_spatial_join_by_distance_labels() -> None:
    a = np.array([[37.7749, -122.4194], [37.7750, -122.4195]])
    b = np.array([[37.7749, -122.4194]])
    # Group label offsets collapse both a-points to label 7.
    pairs, _, _ = gpu_spatial_join_by_distance(
        a, b, max_distance_km=1.0, label_offsets_a=[7, 7]
    )
    assert pairs == [(7, 0)]


def test_gpu_spatial_join_by_distance_invalid_radius() -> None:
    with pytest.raises(ValueError):
        gpu_spatial_join_by_distance(
            np.zeros((2, 2)), np.zeros((2, 2)), max_distance_km=0.0
        )


def test_spatial_join_kernel_intersects() -> None:
    parent = _make_parents(resolution=9, n=1)[0]
    # A single cell intersects itself.
    matches, unmatched_a, unmatched_b = spatial_join_kernel(
        [parent], [parent], join_type="intersects", h3_module=h3
    )
    assert (parent, parent) in matches
    assert unmatched_a == []
    assert unmatched_b == []


def test_spatial_join_kernel_contains_and_within() -> None:
    parent = _make_parents(resolution=7, n=1)[0]
    child = h3.cell_to_children(parent)[:3]
    assert child  # parent at res 7 has children

    matches, unmatched_a, unmatched_b = spatial_join_kernel(
        [parent], child, join_type="contains", h3_module=h3
    )
    assert len(matches) == len(child)
    assert all(a == parent for a, _ in matches)

    matches_w, _, _ = spatial_join_kernel(
        child, [parent], join_type="within", h3_module=h3
    )
    assert len(matches_w) == len(child)


def test_spatial_join_kernel_invalid_type() -> None:
    with pytest.raises(ValueError):
        spatial_join_kernel([], [], join_type="bogus", h3_module=h3)


def test_h3_grid_distance_kernel() -> None:
    cells = _make_cells(n=5)
    matrix = h3_grid_distance_kernel(cells, cells, h3_module=h3)
    assert matrix.shape == (5, 5)
    assert (matrix.diagonal() == 0).all()
    assert (matrix >= 0).all()


def test_h3_grid_distance_kernel_mixed_resolution_reports_minus_one() -> None:
    res7 = _make_parents(resolution=7, n=1)[0]
    res9 = _make_cells(resolution=9, n=1)[0]
    matrix = h3_grid_distance_kernel([res7], [res9], h3_module=h3)
    # Different resolutions cannot be compared -> -1
    assert matrix[0, 0] == -1


def test_h3_grid_distance_kernel_same_resolution_finite() -> None:
    cells = _make_cells(n=10)
    matrix = h3_grid_distance_kernel(cells, cells, h3_module=h3)
    assert (matrix >= 0).all()
    assert int(matrix[0, 2]) > 0


# ---------------------------------------------------------------------------
# Availability helpers.
# ---------------------------------------------------------------------------
def test_is_accelerator_available_flag_congruence() -> None:
    """The exported HAS_* flags and the availability helper must agree."""
    assert is_accelerator_available() == HAS_GPU
    backends = set(get_available_backends())
    for name in ("cupy", "torch", "jax"):
        if name == "cupy" and HAS_CUPY:
            assert name in backends
        if name == "torch" and HAS_TORCH:
            assert name in backends
        if name == "jax" and HAS_JAX:
            assert name in backends


def test_module_imports_cleanly() -> None:
    """The module must remain importable even without accelerators."""
    import importlib

    m = importlib.import_module("geo_infer_space.backends.gpu.gpu_acceleration")
    assert hasattr(m, "pairwise_haversine_kernel")
    assert hasattr(m, "spatial_join_kernel")
    assert hasattr(m, "h3_grid_distance_kernel")


# ---------------------------------------------------------------------------
# CPU fallback is authoritative regardless of accelerator presence.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_gpu", [True, False])
def test_kernels_identical_result_when_accelerator_or_not(use_gpu: bool) -> None:
    """Kernels must return identical values on the CPU reference path, and on
    the accelerator path the results are numerically equivalent."""
    a = np.array([[37.0, -122.0], [40.0, -120.0], [34.0, -118.0]])
    b = np.array([[37.0, -122.0], [36.0, -115.0]])
    dist = pairwise_haversine_kernel(a, b, backend="auto" if use_gpu else "cpu")
    assert dist.shape == (3, 2)
    # A known distance (approx): SF -> LA ~ 559 km
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert dist[0, 1] > 400.0
    assert dist[0, 1] < 650.0


def test_accelerator_flags_are_booleans() -> None:
    for flag in (HAS_CUPY, HAS_JAX, HAS_TORCH, HAS_GPU):
        assert isinstance(flag, bool)


def test_backend_metadata() -> None:
    """The GPU backend package exposes the expected public symbols."""
    from geo_infer_space.backends.gpu import (
        EARTH_RADIUS_KM,
        euclidean_distance_kernel as e,
        h3_grid_distance_kernel as h,
        is_accelerator_available as i,
        pairwise_haversine_kernel as p,
        spatial_join_kernel as s,
    )

    assert EARTH_RADIUS_KM == 6371.0
    assert callable(e)
    assert callable(h)
    assert callable(i)
    assert callable(p)
    assert callable(s)


@pytest.mark.parametrize(
    "point", [[[91, 0]], [[0, 181]], [[float("nan"), 0]], [[0, float("inf")]]]
)
def test_haversine_rejects_invalid_coordinates(point) -> None:
    with pytest.raises(ValueError):
        pairwise_haversine_kernel(point, [[0, 0]])


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_haversine_rejects_invalid_earth_radius(radius) -> None:
    with pytest.raises(ValueError):
        pairwise_haversine_kernel([[0, 0]], [[0, 0]], radius_km=radius)


def test_explicit_cpu_does_not_probe_accelerators(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU must not probe accelerator libraries")

    monkeypatch.setattr(gpu_mod, "_probe_backend", unexpected, raising=False)
    diagnostics = {}
    actual = pairwise_haversine_kernel(
        [[0, 0]], [[0, 180]], backend="cpu", diagnostics=diagnostics
    )
    assert actual[0, 0] == pytest.approx(np.pi * 6371)
    assert diagnostics["used_backends"] == ["cpu"]


def test_grouped_unmatched_returns_labels() -> None:
    pairs, unmatched_a, unmatched_b = gpu_spatial_join_by_distance(
        [[0, 0], [0, 90], [0, 100]],
        [[0, 0], [0, -90]],
        1,
        label_offsets_a=[7, 7, 8],
        label_offsets_b=[4, 9],
    )
    assert pairs == [(7, 4)]
    assert unmatched_a == [8]
    assert unmatched_b == [9]


@pytest.mark.parametrize("name", ["cupy", "torch", "jax"])
def test_missing_backend_is_explicit_failure_or_auto_fallback(
    monkeypatch, name
) -> None:
    monkeypatch.setattr(
        gpu_mod, "_probe_backend", lambda _: gpu_mod._Probe(False, None, "absent")
    )
    with pytest.raises(gpu_mod.AcceleratorUnavailableError, match="absent"):
        pairwise_haversine_kernel([[0, 0]], [[1, 0]], backend=name)
    diagnostics = {}
    actual = pairwise_haversine_kernel([[0, 0]], [[1, 0]], diagnostics=diagnostics)
    assert actual[0, 0] == pytest.approx(6371 * np.pi / 180)
    assert diagnostics["backend"] == "cpu"
    assert "absent" in diagnostics["fallback_reason"]


def test_execution_failure_falls_back_only_in_auto(monkeypatch, caplog) -> None:
    def broken(_):
        raise RuntimeError("device disconnected")

    failed = gpu_mod._ArrayBackend("cupy", np, broken, np.asarray)
    monkeypatch.setattr(
        gpu_mod, "_probe_backend", lambda _: gpu_mod._Probe(True, failed)
    )
    diagnostics = {}
    actual = pairwise_haversine_kernel([[0, 0]], [[1, 0]], diagnostics=diagnostics)
    assert actual[0, 0] == pytest.approx(6371 * np.pi / 180)
    assert diagnostics["used_backends"] == ["cpu"]
    assert "device disconnected" in diagnostics["fallback_reason"]
    assert "falling back to CPU" in caplog.text
    with pytest.raises(RuntimeError, match="cupy distance execution failed"):
        pairwise_haversine_kernel([[0, 0]], [[1, 0]], backend="cupy")


def test_success_then_failure_reports_mixed_execution(monkeypatch) -> None:
    calls = []

    def intermittent(value):
        calls.append(value)
        if len(calls) > 2:
            raise RuntimeError("out of device memory")
        return np.asarray(value)

    simulated = gpu_mod._ArrayBackend("cupy", np, intermittent, np.asarray)
    monkeypatch.setattr(
        gpu_mod, "_probe_backend", lambda _: gpu_mod._Probe(True, simulated)
    )
    diagnostics = {}
    actual = pairwise_haversine_kernel(
        [[0, 0], [1, 0]], [[0, 0], [2, 0]], chunk_size=1, diagnostics=diagnostics
    )
    expected = pairwise_haversine_kernel(
        [[0, 0], [1, 0]], [[0, 0], [2, 0]], backend="cpu"
    )
    np.testing.assert_allclose(actual, expected)
    assert diagnostics["backend"] == "mixed"
    assert diagnostics["used_backends"] == ["cupy", "cpu"]


def test_join_tiles_match_full_reference_and_bound_workspace(monkeypatch) -> None:
    rng = np.random.default_rng(42)
    a = rng.uniform([-80, -175], [80, 175], (13, 2))
    b = rng.uniform([-80, -175], [80, 175], (11, 2))
    threshold = 7000
    reference = pairwise_haversine_kernel(a, b, backend="cpu")
    expected = list(zip(*np.nonzero(reference <= threshold)))
    original = gpu_mod._numeric_block
    shapes = []

    def recorded(backend, left, right, radius):
        shapes.append((len(left), len(right)))
        return original(backend, left, right, radius)

    monkeypatch.setattr(gpu_mod, "_numeric_block", recorded)
    actual, unmatched_a, unmatched_b = gpu_spatial_join_by_distance(
        a, b, threshold, backend="cpu", chunk_size=3
    )
    assert actual == expected
    assert unmatched_a == sorted(set(range(len(a))) - {row for row, _ in expected})
    assert unmatched_b == sorted(set(range(len(b))) - {col for _, col in expected})
    assert len(shapes) == 20
    assert max(max(shape) for shape in shapes) <= 3


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_chunk_size(size) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        pairwise_haversine_kernel([], [], backend="cpu", chunk_size=size)


@pytest.mark.parametrize("labels", [[1], [1, 2, 3], [1.5, 2], [True, 2]])
def test_join_rejects_invalid_labels(labels) -> None:
    with pytest.raises(ValueError, match="label_offsets_a"):
        gpu_spatial_join_by_distance(
            [[0, 0], [1, 0]], [[0, 0]], 1, label_offsets_a=labels
        )


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -1, 0])
def test_join_rejects_invalid_threshold(distance) -> None:
    with pytest.raises(ValueError, match="max_distance_km"):
        gpu_spatial_join_by_distance([], [], distance)


def test_empty_join_retains_unmatched_group_labels() -> None:
    assert gpu_spatial_join_by_distance(
        [], [[0, 0], [1, 0]], 1, label_offsets_b=[8, 8], backend="cpu"
    ) == ([], [], [8])


def test_antipodes_poles_and_dateline_are_finite() -> None:
    points = [[90, 0], [-90, 0], [0, 180], [0, -180], [40, 10], [-40, -170]]
    actual = pairwise_haversine_kernel(points, points, backend="cpu", chunk_size=2)
    assert np.isfinite(actual).all()
    assert actual[0, 1] == pytest.approx(np.pi * 6371)
    assert actual[2, 3] == pytest.approx(0, abs=1e-9)
    assert actual[4, 5] == pytest.approx(np.pi * 6371)
    np.testing.assert_allclose(actual, actual.T)


def test_euclidean_chunks_and_large_coordinates() -> None:
    a = np.array([[0, 0, 0], [1e200, 1e200, 0]])
    result = euclidean_distance_kernel(a, a, backend="cpu", chunk_size=1)
    assert result[0, 1] == pytest.approx(np.sqrt(2) * 1e200)
    np.testing.assert_array_equal(result.diagonal(), [0, 0])
    with pytest.raises(ValueError, match="finite"):
        euclidean_distance_kernel([[np.nan]], [[0]], backend="cpu")


def test_host_h3_order_and_invalid_cell() -> None:
    cells = _make_cells(n=4)
    _, unmatched_a, unmatched_b = spatial_join_kernel(cells[::-1], [], h3_module=h3)
    assert unmatched_a == list(dict.fromkeys(cells[::-1]))
    assert unmatched_b == []
    with pytest.raises(ValueError):
        spatial_join_kernel(["invalid"], cells, h3_module=h3)


def test_module_import_and_cpu_execution_do_not_import_accelerators() -> None:
    import subprocess
    import sys

    code = """
import importlib.abc, sys
class DenyAccelerators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'torch', 'cupy'}:
            raise AssertionError('eager accelerator import: ' + fullname)
sys.meta_path.insert(0, DenyAccelerators())
from geo_infer_space.backends.gpu import pairwise_haversine_kernel
from geo_infer_space.backends.h3 import H3Backend
assert pairwise_haversine_kernel([[0, 0]], [[0, 0]], backend='cpu')[0, 0] == 0
assert H3Backend().geodesic_spatial_join([[0, 0]], [[0, 0]], 1, use_gpu=False)['pairs'] == [(0, 0)]
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr


def test_probe_distinguishes_library_from_device(monkeypatch) -> None:
    from types import SimpleNamespace

    gpu_mod._probe_backend.cache_clear()
    monkeypatch.setattr(gpu_mod.importlib.util, "find_spec", lambda _: object())
    modules = {
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
        ),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "jax": SimpleNamespace(devices=lambda: [SimpleNamespace(platform="cpu")]),
    }
    monkeypatch.setattr(gpu_mod.importlib, "import_module", modules.__getitem__)
    try:
        result = gpu_mod.get_backend_diagnostics()
        assert all(
            value["installed"] and not value["usable"] for value in result.values()
        )
        assert gpu_mod.get_available_backends() == []
    finally:
        gpu_mod._probe_backend.cache_clear()


def test_probe_handles_broken_binary_install(monkeypatch) -> None:
    gpu_mod._probe_backend.cache_clear()
    monkeypatch.setattr(gpu_mod.importlib.util, "find_spec", lambda _: object())

    def broken(_):
        raise OSError("missing CUDA runtime")

    monkeypatch.setattr(gpu_mod.importlib, "import_module", broken)
    try:
        result = gpu_mod.get_backend_diagnostics()
        assert all(
            value["installed"] and not value["usable"] for value in result.values()
        )
        assert all(
            "missing CUDA runtime" in value["reason"] for value in result.values()
        )
    finally:
        gpu_mod._probe_backend.cache_clear()


def test_detected_backend_parity_or_diagnosed_cpu_fallback() -> None:
    available = get_available_backends()
    a = np.array([[0, 0], [40, 10], [-40, -170], [89.9, 179.9]])
    b = np.array([[0, 180], [-25, -110], [89.9, -179.9]])
    haversine = pairwise_haversine_kernel(a, b, backend="cpu")
    euclidean = euclidean_distance_kernel(a, b, backend="cpu")
    expected_join = gpu_spatial_join_by_distance(a, b, 5000, backend="cpu")
    if not available:
        details = {}
        np.testing.assert_allclose(
            pairwise_haversine_kernel(a, b, diagnostics=details), haversine
        )
        assert details["used_backends"] == ["cpu"]
        assert details["fallback_reason"]
    for name in available:
        details = {}
        np.testing.assert_allclose(
            pairwise_haversine_kernel(
                a, b, backend=name, chunk_size=2, diagnostics=details
            ),
            haversine,
            rtol=1e-10,
            atol=1e-7,
        )
        assert details["used_backends"] == [name]
        np.testing.assert_allclose(
            euclidean_distance_kernel(a, b, backend=name, chunk_size=2),
            euclidean,
            rtol=1e-12,
        )
        assert (
            gpu_spatial_join_by_distance(a, b, 5000, backend=name, chunk_size=2)
            == expected_join
        )
