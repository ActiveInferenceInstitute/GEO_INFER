"""Cross-module parity tests for Moran's I variance (GS-100).

Pins that ``geo_infer_space.core.statistics.SpatialStatistics.moran_i``
computes its randomization-assumption variance with the full Cliff-Ord
connectivity terms — S2 = sum((row_sum_i + col_sum_i)^2) — so that it
matches ``geo_infer_math.core.spatial_statistics.morans_i_variance``
on a shared weight matrix, and that degenerate variances (undefined
for n < 4, or non-positive) surface an explicit ``error`` key instead
of being clamped to a spurious positive value.
"""

import numpy as np
import pytest

from geo_infer_math.core.spatial_statistics import morans_i_variance
from geo_infer_space.core.dispatcher import (
    get_backend_dispatcher,
    reset_dispatcher,
)
from geo_infer_space.core.statistics import SpatialStatistics


@pytest.fixture(autouse=True)
def _reset_global_dispatcher():
    """Isolate the global dispatcher singleton per test."""
    reset_dispatcher()
    yield
    reset_dispatcher()


def _shared_weight_matrix(cells):
    """Rebuild the row-standardized binary adjacency matrix for cells."""
    backend = get_backend_dispatcher().get_backend("h3")
    n = len(cells)
    weights = np.zeros((n, n))
    for i, cell_i in enumerate(cells):
        neighbors = set(backend.get_cell_neighbors(cell_i, k=1))
        for j, cell_j in enumerate(cells):
            if i != j and cell_j in neighbors:
                weights[i, j] = 1.0
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return weights / row_sums


def _adjacent_cells():
    """A central H3 res-8 cell plus its six neighbors."""
    backend = get_backend_dispatcher().get_backend("h3")
    center = backend.latlng_to_cell(37.7749, -122.4194, 8)
    return [center] + list(backend.get_cell_neighbors(center, k=1))


def test_space_variance_matches_math_cliff_ord_variance():
    """SPACE variance equals MATH's Cliff-Ord variance on shared weights."""
    cells = _adjacent_cells()
    values = [3.0, 1.0, 4.0, 1.5, 5.0, 2.5, 0.5]

    stats = SpatialStatistics()
    result = stats.moran_i(cells, values)

    assert "error" not in result, result
    weights = _shared_weight_matrix(cells)
    expected = morans_i_variance(np.asarray(values, dtype=float), weights)
    assert result["variance"] == pytest.approx(expected, abs=1e-6)


def test_variance_undefined_for_three_observations_returns_error():
    """n=3 has a zero-variance denominator; the error key replaces the old
    epsilon-inflated spurious variance."""
    cells = _adjacent_cells()[:3]
    stats = SpatialStatistics()
    result = stats.moran_i(cells, [1.0, 2.0, 3.5])

    assert "error" in result
    assert "variance" not in result
    assert "z_score" not in result
    assert "p_value" not in result


def test_non_positive_variance_returns_error_not_clamp():
    """A genuinely non-positive variance surfaces an error key instead of
    being clamped to a spurious significant z-score.

    For a fully-connected 4-node row-standardized matrix with values
    [0, 0, 0, -1], the exact Cliff-Ord randomization variance is a tiny
    negative float (-9.7e-17); the reference implementation returns it
    as-is, so this input provably has variance <= 0.
    """

    class FixedMatrixStats(SpatialStatistics):
        """Injects a fixed weight matrix through the public moran_i path."""

        def _build_weight_matrix(self, cells, weight_type):  # type: ignore[override]
            n = len(cells)
            weights = np.ones((n, n)) - np.eye(n)
            return weights / weights.sum(axis=1, keepdims=True)

    values = [0.0, 0.0, 0.0, -1.0]
    n = 4
    weights = np.ones((n, n)) - np.eye(n)
    weights = weights / weights.sum(axis=1, keepdims=True)
    assert morans_i_variance(np.asarray(values), weights) < 0

    stats = FixedMatrixStats()
    result = stats.moran_i(_adjacent_cells()[:4], values)

    assert "error" in result
    assert "non-positive" in result["error"]
    assert "z_score" not in result
    assert "p_value" not in result
