"""Unsupported topology must not masquerade as valid analytic neighborhoods."""

import h3
import pytest

from geo_infer_space.backends.h3.core import H3Cell, H3Grid
from geo_infer_space.backends.h3.analytics import H3DensityAnalyzer, H3SpatialAnalyzer
from geo_infer_space.backends.h3.h3_backend import H3Backend


@pytest.fixture
def grid():
    indexes = list(h3.grid_disk(h3.latlng_to_cell(37.7, -122.4, 9), 1))
    return H3Grid(
        [
            H3Cell(index=c, resolution=9, properties={"value": i + 1.0})
            for i, c in enumerate(indexes)
        ]
    )


@pytest.mark.parametrize("method", ["local_morans", "getis_ord"])
def test_hotspot_topology_failure_propagates(grid, monkeypatch, method):
    analyzer = H3SpatialAnalyzer(grid)

    def unavailable(*args):
        raise h3.H3FailedError("topology unavailable")

    monkeypatch.setattr(h3, "grid_disk", unavailable)
    with pytest.raises(h3.H3FailedError, match="topology unavailable"):
        analyzer.detect_hotspots("value", method=method)


def test_spatial_lag_topology_failure_propagates(grid, monkeypatch):
    analyzer = H3DensityAnalyzer(grid)

    def unavailable(*args):
        raise h3.H3FailedError("topology unavailable")

    monkeypatch.setattr(h3, "grid_disk", unavailable)
    with pytest.raises(h3.H3FailedError, match="topology unavailable"):
        analyzer.analyze_density_patterns("value")


def test_cluster_invalid_domain_is_rejected():
    with pytest.raises(ValueError):
        H3Backend().find_clusters(["not-a-cell"], [1.0])


def test_cluster_topology_failure_propagates(grid, monkeypatch):
    def unavailable(*args):
        raise h3.H3FailedError("topology unavailable")

    monkeypatch.setattr(h3, "grid_disk", unavailable)
    with pytest.raises(h3.H3FailedError, match="topology unavailable"):
        H3Backend().find_clusters(
            [c.index for c in grid], [c.properties["value"] for c in grid]
        )
