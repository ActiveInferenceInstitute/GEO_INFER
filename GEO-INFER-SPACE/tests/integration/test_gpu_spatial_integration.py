"""Numeric backend selection and host H3 topology through H3Backend.

CPU requests must not probe accelerators. Execution metadata describes actual
numeric backends, while exact H3 grid and ancestry operations always use CPU.
"""

from __future__ import annotations


import numpy as np
import pytest

import h3

from geo_infer_space.backends.h3 import H3Backend


@pytest.fixture(scope="module")
def backend() -> H3Backend:
    b = H3Backend()
    assert b.is_available()
    return b


def _cells(n: int = 8, resolution: int = 9) -> list[str]:
    return [
        h3.latlng_to_cell(37.0 + i * 0.01, -122.0 + i * 0.01, resolution)
        for i in range(n)
    ]


def _parent(resolution: int = 7) -> str:
    return h3.latlng_to_cell(37.5, -122.3, resolution)


def test_backend_spatial_join_both_paths(backend: H3Backend) -> None:
    a = _cells(8)
    b = _cells(8)[1:]
    cpu = backend.spatial_join(a, b, join_type="intersects", use_gpu=False)
    gpu = backend.spatial_join(a, b, join_type="intersects", use_gpu=True)

    assert cpu["join_type"] == "intersects"
    assert gpu["join_type"] == "intersects"

    # Both paths must report the same matched pairs and counts.
    cpu_matches = sorted(cpu["matches"])
    gpu_matches = sorted(gpu["matches"])
    assert cpu_matches == gpu_matches
    assert cpu["match_count"] == gpu["match_count"]
    assert sorted(cpu["unmatched_a"]) == sorted(gpu["unmatched_a"])
    assert sorted(cpu["unmatched_b"]) == sorted(gpu["unmatched_b"])

    # Identical cells (a[1] is present in both lists) must intersect.
    assert (a[1], a[1]) in cpu_matches
    assert len(cpu_matches) > 0


def test_backend_spatial_join_contains_within(backend: H3Backend) -> None:
    parent = _parent()
    children = h3.cell_to_children(parent)
    assert children

    cpu_contains = backend.spatial_join(
        [parent], children, join_type="contains", use_gpu=False
    )
    gpu_contains = backend.spatial_join(
        [parent], children, join_type="contains", use_gpu=True
    )
    assert cpu_contains["match_count"] == len(children)
    assert gpu_contains["match_count"] == len(children)

    cpu_within = backend.spatial_join(
        children, [parent], join_type="within", use_gpu=False
    )
    gpu_within = backend.spatial_join(
        children, [parent], join_type="within", use_gpu=True
    )
    assert cpu_within["match_count"] == len(children)
    assert gpu_within["match_count"] == len(children)


def test_backend_spatial_join_invalid_type(backend: H3Backend) -> None:
    with pytest.raises(ValueError):
        backend.spatial_join(_cells(3), _cells(2), join_type="bogus", use_gpu=False)


def test_backend_compute_distance_matrix(backend: H3Backend) -> None:
    a = _cells(6)
    b = _cells(6)
    cpu = backend.compute_distance_matrix(a, b, use_gpu=False)
    gpu = backend.compute_distance_matrix(a, b, use_gpu=True)

    assert cpu["shape"] == [6, 6]
    assert gpu["shape"] == [6, 6]
    cm = np.asarray(cpu["distance_matrix"])
    gm = np.asarray(gpu["distance_matrix"])
    np.testing.assert_array_equal(cm, gm)

    # Diagonal is zero (identical cell comparison).
    np.testing.assert_array_equal(np.diag(cm), np.zeros(6))


def test_backend_compute_distance_matrix_positive_off_diagonal(
    backend: H3Backend,
) -> None:
    cells = _cells(5)
    res = backend.compute_distance_matrix(cells, cells, use_gpu=False)
    m = np.asarray(res["distance_matrix"])
    assert (m >= 0).all()
    # Adjacent cells along a strip will have positive grid distance for some pair.
    assert int(m[0, 4]) > 0


def test_backend_geodesic_distance_matrix(backend: H3Backend) -> None:
    a = _cells(4)
    b = _cells(4)
    cpu = backend.geodesic_distance_matrix(a, b, use_gpu=False)
    gpu = backend.geodesic_distance_matrix(a, b, use_gpu=True)

    assert cpu["units"] == "km"
    assert gpu["units"] == "km"
    assert cpu["shape"] == [4, 4]
    assert gpu["shape"] == [4, 4]

    cm = np.asarray(cpu["distance_matrix"])
    gm = np.asarray(gpu["distance_matrix"])
    np.testing.assert_allclose(cm, gm, atol=1e-6)

    # Self distance is ~0.
    assert cm[0, 0] == pytest.approx(0.0, abs=1e-6)
    # Off-diagonal positive.
    assert cm[0, 1] > 0.0


def test_backend_geodesic_spatial_join_both_paths(backend: H3Backend) -> None:
    sf = (37.7749, -122.4194)
    oak = (37.8044, -122.2712)  # ~19 km from SF
    la = (34.0522, -118.2437)
    nyc = (40.7128, -74.0060)

    cpu = backend.geodesic_spatial_join(
        [sf, la], [oak, nyc], max_distance_km=30.0, use_gpu=False
    )
    gpu = backend.geodesic_spatial_join(
        [sf, la], [oak, nyc], max_distance_km=30.0, use_gpu=True
    )

    assert cpu["pairs"] == gpu["pairs"]
    assert cpu["pair_count"] == gpu["pair_count"]
    assert sorted(cpu["unmatched_a"]) == sorted(gpu["unmatched_a"])
    assert sorted(cpu["unmatched_b"]) == sorted(gpu["unmatched_b"])

    # SF ~ Oakland join at (0, 0); LA and NYC unmatched.
    assert (0, 0) in cpu["pairs"]
    assert 1 in cpu["unmatched_a"]  # LA
    assert 1 in cpu["unmatched_b"]  # NYC


def test_backend_accelerator_state_attribute(backend: H3Backend) -> None:
    """The backend exposes a deterministic accelerator availability state."""
    assert isinstance(backend.accelerator, bool)
    assert isinstance(backend.accelerator_backends, list)
    for item in backend.accelerator_backends:
        assert isinstance(item, str)
    # Both the availability guard and the refresh helper are always callable.
    backend._prepare_accelerator()


def test_backend_package_import_clean(backend: H3Backend) -> None:
    """Force-Python fallback is reachable even if GPU is present."""
    cpu = backend.compute_distance_matrix(_cells(3), _cells(3), use_gpu=False)
    assert cpu["shape"] == [3, 3]


def test_use_gpu_false_never_probes_or_runs_accelerator(backend, monkeypatch) -> None:
    from geo_infer_space.backends.gpu import gpu_acceleration as kernels

    def unexpected(*args, **kwargs):
        raise AssertionError("CPU request must not probe accelerators")

    monkeypatch.setattr(kernels, "_probe_backend", unexpected)
    cells = _cells(2)
    for result in [
        backend.compute_distance_matrix(cells, cells, use_gpu=False),
        backend.spatial_join(cells, cells, use_gpu=True),
        backend.geodesic_distance_matrix(cells, cells, use_gpu=False, chunk_size=1),
        backend.geodesic_spatial_join([(0, 0)], [(0, 0)], 1, use_gpu=False),
    ]:
        assert result["backend"] == "cpu"
        assert result["accelerator"] == []


def test_invalid_centroid_is_not_silently_replaced(backend) -> None:
    with pytest.raises(ValueError):
        backend.geodesic_distance_matrix(["invalid"], _cells(1), use_gpu=False)


def test_empty_centroid_matrix(backend) -> None:
    result = backend.geodesic_distance_matrix([], _cells(2), use_gpu=False)
    assert result["distance_matrix"].shape == (0, 2)
    assert result["accelerator"] == []
    assert result["backend"] == "none"


def test_conflicting_explicit_backend_is_rejected(backend) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        backend.geodesic_spatial_join(
            [(0, 0)], [(0, 0)], 1, use_gpu=False, backend="cupy"
        )
