"""Regression tests for the weight-matrix backend failure contract (GS-101).

``SpatialStatistics`` must never fabricate a fully-connected weight matrix
when the spatial backend is missing or fails: backend resolution goes
through the dispatcher's explicit ``_resolve_backend_name`` policy, and
unresolvable backends raise a precise ``ValueError`` listing the available
backends instead of silently producing a meaningless all-connected result.
"""

import pytest

from geo_infer_space.core.dispatcher import reset_dispatcher
from geo_infer_space.core.statistics import SpatialStatistics


@pytest.fixture(autouse=True)
def _reset_global_dispatcher():
    """Isolate the global dispatcher singleton per test."""
    reset_dispatcher()
    yield
    reset_dispatcher()


def test_unregistered_backend_raises_not_fully_connected():
    """A backend name that is not registered raises a precise ValueError
    naming the available backends; moran_i never fabricates a p-value."""
    stats = SpatialStatistics(backend="definitely-not-a-backend")
    with pytest.raises(ValueError, match="not available"):
        stats.moran_i(
            ["8928308280fffff", "89283082803ffff", "89283082807ffff"],
            [1.0, 2.0, 3.0],
        )


def test_moran_i_failing_neighbor_lookup_propagates(monkeypatch):
    """A failed neighbor lookup cannot produce a valid analytic weight matrix."""
    stats = SpatialStatistics()

    class FailingBackend:
        def get_cell_neighbors(self, cell, k=1):
            raise ValueError("neighborhood unavailable")

    monkeypatch.setattr(stats.dispatcher, "get_backend", lambda name: FailingBackend())
    cells = ["8928308280fffff", "89283082803ffff", "89283082807ffff", "8928308280bffff"]
    with pytest.raises(ValueError, match="neighborhood unavailable"):
        stats.moran_i(cells, [1.0, 2.0, 3.0, 4.0])


def test_getis_ord_unregistered_backend_raises():
    """An unregistered backend raises before any hotspot output."""
    stats = SpatialStatistics(backend="definitely-not-a-backend")
    with pytest.raises(ValueError, match="not available"):
        stats.getis_ord_g(
            ["8928308280fffff", "89283082803ffff", "89283082807ffff"],
            [1.0, 2.0, 3.0],
        )


def test_nearest_neighbor_unregistered_backend_returns_error_key():
    """nearest_neighbor_index with an unregistered backend surfaces an
    error key, not fabricated distances."""
    stats = SpatialStatistics(backend="definitely-not-a-backend")
    with pytest.raises(ValueError, match="not available"):
        stats.nearest_neighbor_index(["8928308280fffff", "89283082803ffff"])


def test_quadrat_count_unregistered_backend_raises():
    """An unregistered backend cannot produce partial quadrat counts."""
    stats = SpatialStatistics(backend="definitely-not-a-backend")
    with pytest.raises(ValueError, match="not available"):
        stats.quadrat_count(["8928308280fffff", "89283082803ffff", "89283082807ffff"])
