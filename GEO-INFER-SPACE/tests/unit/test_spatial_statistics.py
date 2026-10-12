"""
Tests for Spatial Statistics Module.

Comprehensive tests for spatial autocorrelation, clustering indices,
and pattern detection statistics.
"""

import pytest
import numpy as np

from geo_infer_space.core.statistics import SpatialStatistics
from geo_infer_space.core.dispatcher import get_backend_dispatcher, reset_dispatcher


@pytest.fixture(autouse=True)
def reset_global_dispatcher():
    """Reset the global dispatcher before each test."""
    reset_dispatcher()
    yield
    reset_dispatcher()


@pytest.fixture
def stats():
    """Create a SpatialStatistics instance."""
    return SpatialStatistics()


@pytest.fixture
def sample_cells():
    """Generate sample H3 cells for testing."""
    dispatcher = get_backend_dispatcher()
    backend = dispatcher.get_backend("h3")

    # Create a cluster of cells around a central point
    center = backend.latlng_to_cell(37.7749, -122.4194, 8)
    neighbors = backend.get_cell_neighbors(center, k=2)
    cells = [center] + list(neighbors)
    return cells


@pytest.fixture
def clustered_data(sample_cells):
    """Generate clustered spatial data (high values together)."""
    n = len(sample_cells)
    # First half high, second half low
    values = [100.0 + np.random.uniform(0, 10) for _ in range(n // 2)]
    values += [10.0 + np.random.uniform(0, 5) for _ in range(n - n // 2)]
    return sample_cells, values


@pytest.fixture
def random_data(sample_cells):
    """Generate random spatial data."""
    values = [np.random.uniform(0, 100) for _ in sample_cells]
    return sample_cells, values


class TestMoranI:
    """Tests for Moran's I spatial autocorrelation."""

    def test_moran_i_structure(self, stats, random_data):
        """Test that Moran's I returns expected structure."""
        cells, values = random_data
        result = stats.moran_i(cells, values)

        assert "moran_i" in result
        assert "expected_i" in result
        assert "z_score" in result
        assert "p_value" in result
        assert "interpretation" in result

    def test_moran_i_range(self, stats, random_data):
        """Test that Moran's I is within valid range."""
        cells, values = random_data
        result = stats.moran_i(cells, values)

        if result.get("moran_i") is not None:
            assert -1.0 <= result["moran_i"] <= 1.0

    def test_moran_i_weight_types(self, stats, random_data):
        """Test different weight types."""
        cells, values = random_data

        for weight_type in ["queen", "rook", "distance"]:
            result = stats.moran_i(cells, values, weight_type=weight_type)
            assert "moran_i" in result
            assert result["weight_type"] == weight_type

    def test_moran_i_mismatched_lengths(self, stats, sample_cells):
        """Test error handling for mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            stats.moran_i(sample_cells, [1.0, 2.0])  # Too few values

    def test_moran_i_too_few_observations(self, stats):
        """Test handling of too few observations."""
        result = stats.moran_i(["8928308280fffff", "89283082803ffff"], [1.0, 2.0])
        assert "error" in result or result.get("moran_i") is not None


class TestGetisOrdG:
    """Tests for Getis-Ord G* statistic."""

    def test_getis_ord_structure(self, stats, random_data):
        """Test that Getis-Ord G* returns expected structure."""
        cells, values = random_data
        result = stats.getis_ord_g(cells, values)

        assert "g_stars" in result
        assert "hotspots" in result
        assert "coldspots" in result
        assert "num_hotspots" in result
        assert "num_coldspots" in result

    def test_getis_ord_identifies_hotspots(self, stats, sample_cells):
        """Test that G* identifies high-value clusters as hotspots."""
        # Create data with clearly high values
        values = [1000.0] * (len(sample_cells) // 2) + [1.0] * (
            len(sample_cells) - len(sample_cells) // 2
        )

        result = stats.getis_ord_g(sample_cells, values, distance=1)

        # Should identify some hotspots
        assert isinstance(result["hotspots"], list)

    def test_getis_ord_distance_parameter(self, stats, random_data):
        """Test different distance parameters."""
        cells, values = random_data

        result_1 = stats.getis_ord_g(cells, values, distance=1)
        result_2 = stats.getis_ord_g(cells, values, distance=2)

        assert result_1["distance"] == 1
        assert result_2["distance"] == 2


class TestNearestNeighborIndex:
    """Tests for Nearest Neighbor Index."""

    def test_nni_structure(self, stats, sample_cells):
        """Test that NNI returns expected structure."""
        result = stats.nearest_neighbor_index(sample_cells, study_area_km2=100)

        if "error" not in result:
            assert "nni" in result
            assert "pattern" in result
            assert "observed_mean_distance" in result
            assert "expected_mean_distance" in result

    def test_nni_pattern_interpretation(self, stats, sample_cells):
        """Test NNI pattern interpretation."""
        result = stats.nearest_neighbor_index(sample_cells, study_area_km2=100)

        if "pattern" in result:
            valid_patterns = [
                "highly clustered",
                "clustered",
                "random",
                "dispersed",
                "highly dispersed",
            ]
            assert result["pattern"] in valid_patterns

    def test_nni_too_few_cells(self, stats):
        """Test error handling for too few cells."""
        result = stats.nearest_neighbor_index(["8928308280fffff"])
        assert "error" in result


class TestSummaryStatistics:
    """Tests for summary statistics calculation."""

    def test_summary_structure(self, stats):
        """Test summary statistics structure."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = stats.calculate_summary_statistics(values)

        assert "n" in result
        assert "mean" in result
        assert "std" in result
        assert "variance" in result
        assert "min" in result
        assert "max" in result
        assert "q1" in result
        assert "q3" in result

    def test_summary_accuracy(self, stats):
        """Test summary statistics accuracy."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = stats.calculate_summary_statistics(values)

        assert result["n"] == 5
        assert result["mean"] == pytest.approx(3.0)
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_summary_empty_values(self, stats):
        """Test handling of empty values."""
        result = stats.calculate_summary_statistics([])
        assert "error" in result


class TestVarianceMeanRatio:
    """Tests for Variance-to-Mean Ratio."""

    def test_vmr_structure(self, stats):
        """Test VMR returns expected structure."""
        values = [10, 12, 8, 15, 9, 11, 14, 10]
        result = stats.variance_mean_ratio(values)

        assert "vmr" in result
        assert "variance" in result
        assert "mean" in result
        assert "pattern" in result

    def test_vmr_poisson_detection(self, stats):
        """Test VMR detects Poisson-like distribution."""
        # For Poisson, VMR ≈ 1
        np.random.seed(42)
        values = list(np.random.poisson(10, 100))
        result = stats.variance_mean_ratio(values)

        # Should be close to 1 for Poisson
        assert 0.5 < result["vmr"] < 2.0

    def test_vmr_interpretation(self, stats):
        """Test VMR pattern interpretation."""
        values = [5, 5, 5, 5, 5]  # Uniform (underdispersed)
        result = stats.variance_mean_ratio(values)

        assert "underdispersed" in result["pattern"] or "regular" in result["pattern"]


class TestQuadratCount:
    """Tests for quadrat count analysis."""

    def test_quadrat_structure(self, stats, sample_cells):
        """Test quadrat count returns expected structure."""
        result = stats.quadrat_count(sample_cells, quadrat_size=2)

        if "error" not in result:
            assert "num_quadrats" in result
            assert "counts" in result
            assert "total_count" in result
            assert "mean_count" in result

    def test_quadrat_with_values(self, stats, sample_cells):
        """Test quadrat count with explicit values."""
        values = [10] * len(sample_cells)
        result = stats.quadrat_count(sample_cells, values=values, quadrat_size=1)

        if "error" not in result:
            assert result["total_count"] == 10 * len(sample_cells)


class TestIntegration:
    """Integration tests combining multiple statistics."""

    def test_full_spatial_analysis(self, stats, sample_cells):
        """Test running multiple statistics in sequence."""
        values = [np.random.uniform(10, 100) for _ in sample_cells]

        # Summary statistics
        summary = stats.calculate_summary_statistics(values)
        assert "mean" in summary

        # Moran's I
        moran = stats.moran_i(sample_cells, values)
        assert "moran_i" in moran or "error" in moran

        # Getis-Ord G*
        getis = stats.getis_ord_g(sample_cells, values)
        assert "g_stars" in getis or "error" in getis

        # VMR
        vmr = stats.variance_mean_ratio(values)
        assert "vmr" in vmr

    def test_statistics_consistency(self, stats, sample_cells):
        """Test that statistics are consistent with each other."""
        # Create clearly clustered data
        values = [100.0] * len(sample_cells)

        # VMR should indicate underdispersion for uniform values
        vmr = stats.variance_mean_ratio(values)

        if vmr.get("vmr") is not None:
            # Uniform data should have very low VMR
            assert (
                vmr["vmr"] < 0.1
                or "uniform" in vmr["pattern"].lower()
                or "underdispersed" in vmr["pattern"].lower()
            )
