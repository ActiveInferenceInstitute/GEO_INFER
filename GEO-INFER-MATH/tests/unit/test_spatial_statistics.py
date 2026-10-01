"""
Comprehensive tests for the spatial_statistics module.
"""

import numpy as np
import pytest
from geo_infer_math.core.spatial_statistics import (
    MoranI,
    getis_ord_g,
    ripley_k,
    semivariogram,
    spatial_descriptive_statistics,
    spatial_entropy,
    local_indicators_spatial_association,
)


def test_moran_i():
    """Test Moran's I statistic calculation."""
    # Create a simple test case with positive spatial autocorrelation
    # (similar values are close to each other)
    values = np.array([10, 12, 11, 13, 50, 52, 51, 53])
    coords = np.array(
        [
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],  # Cluster of low values
            [10, 10],
            [10, 11],
            [11, 10],
            [11, 11],  # Cluster of high values
        ]
    )

    # Calculate Moran's I
    moran = MoranI()
    result = moran.compute(values, coords)

    # With this pattern, we expect positive spatial autocorrelation
    assert result["I"] > 0

    # The expected I should be -1/(n-1) = -1/7 ≈ -0.143
    assert abs(result["expected_I"] - (-1 / 7)) < 1e-10

    # Check that the p-value is valid
    assert 0 <= result["p_value"] <= 1

    # Test with pre-defined weights matrix
    n = len(values)
    weights = np.zeros((n, n))

    # Define simple weights: adjacent points have weight 1
    for i in range(4):
        for j in range(4):
            if (
                i != j
                and abs(coords[i, 0] - coords[j, 0]) <= 1
                and abs(coords[i, 1] - coords[j, 1]) <= 1
            ):
                weights[i, j] = 1

    for i in range(4, 8):
        for j in range(4, 8):
            if (
                i != j
                and abs(coords[i, 0] - coords[j, 0]) <= 1
                and abs(coords[i, 1] - coords[j, 1]) <= 1
            ):
                weights[i, j] = 1

    # Row-standardize weights
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums[:, np.newaxis]

    moran_with_weights = MoranI(weights)
    result_with_weights = moran_with_weights.compute(values)

    # We should still get positive spatial autocorrelation
    assert result_with_weights["I"] > 0


def test_getis_ord_g():
    """Test Getis-Ord G* statistic calculation."""
    # Create a test case with a hot spot and a cold spot
    values = np.array([10, 11, 12, 13, 50, 51, 52, 53])

    # Create a weights matrix
    n = len(values)
    weights = np.zeros((n, n))

    # Define simple binary weights for two distinct clusters
    for i in range(4):
        for j in range(4):
            if i != j:
                weights[i, j] = 1

    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                weights[i, j] = 1

    # Row-standardize
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    # Calculate Getis-Ord G*
    result = getis_ord_g(values, weights)

    # Check results
    assert "local_g" in result
    assert "z_scores" in result
    assert "global_g" in result

    # The high values should be hot spots (positive z-scores)
    assert np.all(result["z_scores"][4:8] > 0)

    # The low values should be cold spots (negative z-scores)
    assert np.all(result["z_scores"][0:4] < 0)


def test_ripley_k():
    """Test Ripley's K function calculation."""
    # Create a simple point pattern
    points = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],  # Clustered along diagonal
            [10, 10],
            [11, 11],
            [12, 12],
            [13, 13],  # Another cluster
        ]
    )

    # Define distances at which to evaluate K
    distances = [1, 2, 5, 10, 15]

    # Area (assuming a 20x20 study area)
    area = 400

    # Calculate Ripley's K
    result = ripley_k(points, distances, area)

    # Check results
    assert "distances" in result
    assert "k_function" in result
    assert "l_function" in result

    # K should increase with distance
    assert np.all(np.diff(result["k_function"]) >= 0)

    # For a clustered pattern, the L function should be positive at smaller distances
    assert result["l_function"][0] > 0 or result["l_function"][1] > 0


def test_semivariogram():
    """Test semivariogram calculation."""
    # Create a simple dataset with spatial structure
    coords = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
    )

    # Values with spatial trend (increasing from bottom-left to top-right)
    values = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 6])

    # Lag distances (max pairwise distance for this grid is ~4.1, use valid range)
    lag_distances = [1, 2, 3, 4]

    # Calculate semivariogram
    result = semivariogram(coords, values, lag_distances)

    # Check results
    assert "lag_distances" in result
    assert "semivariance" in result
    assert "count" in result

    # For this dataset, semivariance should generally increase with distance
    # (but might not be strictly increasing due to sampling variability)
    assert result["semivariance"][0] < result["semivariance"][-1]


def test_spatial_descriptive_statistics():
    """Test spatial descriptive statistics calculation."""
    # Create test data
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    values = np.array([10, 20, 30, 40])

    # Calculate statistics
    stats = spatial_descriptive_statistics(coords, values)

    # Check basic statistics
    assert stats.mean == 25.0
    assert stats.median == 25.0
    assert abs(stats.stdev - np.std(values, ddof=0)) < 1e-10
    assert abs(stats.variance - np.var(values, ddof=0)) < 1e-10
    assert stats.min_value == 10.0
    assert stats.max_value == 40.0

    # Check centroid
    # For this example, the weighted centroid should be biased towards higher values
    assert stats.centroid[0] > 0.5
    assert stats.centroid[1] > 0.5


def test_spatial_entropy():
    """Test spatial entropy calculation."""
    # Test with uniformly spread values (high entropy)
    np.random.seed(42)
    uniform = np.random.rand(100)
    entropy_uniform = spatial_entropy(uniform)

    # Test with concentrated distribution (low entropy - values cluster in few bins)
    concentrated = np.zeros(100)
    concentrated[45:55] = 1.0
    entropy_concentrated = spatial_entropy(concentrated)

    # Entropy should be higher for uniformly spread distribution
    assert entropy_uniform > entropy_concentrated

    # Test with different number of bins
    entropy_10_bins = spatial_entropy(uniform, bins=10)
    entropy_20_bins = spatial_entropy(uniform, bins=20)

    # More bins should generally increase entropy for uniform distribution
    assert entropy_20_bins >= entropy_10_bins


def test_local_indicators_spatial_association():
    """Test LISA calculation."""
    # Create a simple test case with spatial pattern
    values = np.array([10, 12, 11, 13, 50, 52, 51, 53])

    # Create a weights matrix
    n = len(values)
    weights = np.zeros((n, n))

    # Define weights for two distinct clusters
    for i in range(4):
        for j in range(4):
            if i != j:
                weights[i, j] = 1

    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                weights[i, j] = 1

    # Row-standardize
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    # Calculate LISA
    result = local_indicators_spatial_association(values, weights)

    # Check results
    assert "lisa" in result
    assert "z_scores" in result
    assert "p_values" in result
    assert "classifications" in result

    # Check classifications
    # The clusters should be classified as High-High or Low-Low
    high_vals = values > np.mean(values)

    for i in range(n):
        if result["significant"][i]:
            if high_vals[i]:
                # High values should be in High-High clusters
                assert result["classifications"][i] == 1
            else:
                # Low values should be in Low-Low clusters
                assert result["classifications"][i] == 2


class TestMoranIEdgeCases:
    """Test Moran's I with edge cases."""

    def test_moran_i_with_identical_values(self):
        """Test Moran's I with identical values (should be undefined)."""
        values = np.array([5, 5, 5, 5, 5])
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])

        moran = MoranI()
        result = moran.compute(values, coords)

        # With identical values, Moran's I should be undefined (NaN or specific value)
        assert np.isnan(result["I"]) or result["I"] == 0

    def test_moran_i_with_single_point(self):
        """Test Moran's I with single point."""
        values = np.array([10])
        coords = np.array([[0, 0]])

        moran = MoranI()
        with pytest.raises(ValueError):
            moran.compute(values, coords)

    def test_moran_i_with_two_points(self):
        """Moran's I randomization variance is undefined below 4 locations."""
        values = np.array([10, 20])
        coords = np.array([[0, 0], [1, 0]])

        moran = MoranI()
        with pytest.raises(ValueError, match="at least 4 locations"):
            moran.compute(values, coords)

    def test_moran_i_with_negative_weights(self):
        """Test Moran's I with negative weights matrix."""
        values = np.array([10, 20, 30, 40])
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        # Create weights matrix with some negative values
        weights = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, 1, 0]])

        moran = MoranI(weights)
        result = moran.compute(values, coords)

        # Should handle negative weights gracefully
        assert "I" in result


class TestGetisOrdGEdgeCases:
    """Test Getis-Ord G* with edge cases."""

    def test_getis_ord_g_with_constant_values(self):
        """Test Getis-Ord G* with constant values."""
        values = np.array([10, 10, 10, 10, 10])
        weights = np.eye(5)  # No spatial relationships

        result = getis_ord_g(values, weights)

        # With constant values and no spatial relationships, all z-scores should be 0
        assert np.allclose(result["z_scores"], 0)

    def test_getis_ord_g_with_extreme_values(self):
        """Test Getis-Ord G* with extreme values."""
        values = np.array([0, 0, 0, 1000, 0])
        weights = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0],
            ]
        )

        result = getis_ord_g(values, weights)

        # The point with value 1000 should have high z-score if neighbors also have high values
        # Point 3 has value 1000 and is connected to points 2 and 4 (values 0 and 0)
        # So it should not be a hotspot
        assert result["z_scores"][3] < 1.96  # Not significant at 95% level


class TestRipleyKEdgeCases:
    """Test Ripley's K function with edge cases."""

    def test_ripley_k_with_single_point(self):
        """Test Ripley's K with single point."""
        points = np.array([[0, 0]])
        distances = [1, 2, 3]
        area = 100

        result = ripley_k(points, distances, area)

        # Should handle gracefully, though not meaningful
        assert "k_function" in result
        assert len(result["k_function"]) == len(distances)

    def test_ripley_k_with_identical_points(self):
        """Test Ripley's K with identical points."""
        points = np.array([[0, 0], [0, 0], [0, 0]])
        distances = [1, 2, 3]
        area = 100

        result = ripley_k(points, distances, area)

        # Should handle duplicate points
        assert "k_function" in result


class TestSemivariogramEdgeCases:
    """Test semivariogram with edge cases."""

    def test_semivariogram_with_identical_values(self):
        """Test semivariogram with identical values."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([5, 5, 5, 5])
        lag_distances = [0.5, 1.0, 1.5]

        result = semivariogram(coords, values, lag_distances)

        # Semivariance should be zero for identical values
        assert np.allclose(result["semivariance"], 0, atol=1e-10)

    def test_semivariogram_with_insufficient_pairs(self):
        """Test semivariogram with insufficient point pairs for some lags."""
        coords = np.array([[0, 0], [10, 10]])
        values = np.array([1, 2])
        lag_distances = [1, 5, 10]

        result = semivariogram(coords, values, lag_distances, tolerance=0.1)

        # Should handle gracefully when some lags have no pairs
        assert "semivariance" in result
        assert len(result["semivariance"]) == len(lag_distances)


class TestSpatialEntropyEdgeCases:
    """Test spatial entropy with edge cases."""

    def test_spatial_entropy_with_single_value(self):
        """Test spatial entropy with single value."""
        values = np.array([5])

        entropy = spatial_entropy(values, bins=10)

        # Single value should have zero entropy
        assert entropy == 0

    def test_spatial_entropy_with_identical_values(self):
        """Test spatial entropy with identical values."""
        values = np.array([5, 5, 5, 5, 5])

        entropy = spatial_entropy(values, bins=10)

        # Identical values should have zero entropy
        assert entropy == 0

    def test_spatial_entropy_with_extreme_range(self):
        """Test spatial entropy with extreme value range."""
        values = np.array([0, 1000, 0.001, 999.999])

        entropy = spatial_entropy(values, bins=10)

        # Should handle extreme ranges gracefully
        assert entropy >= 0


class TestInputValidation:
    """Test input validation for spatial statistics functions."""

    def test_moran_i_invalid_inputs(self):
        """Test Moran's I with invalid inputs."""
        moran = MoranI()

        # Test with mismatched dimensions
        values = np.array([1, 2, 3])
        coords = np.array([[0, 0], [1, 0]])  # Different length

        with pytest.raises((ValueError, IndexError)):
            moran.compute(values, coords)

        # Test with NaN values
        values_nan = np.array([1, np.nan, 3])
        coords_valid = np.array([[0, 0], [1, 0], [2, 0]])

        with pytest.raises(ValueError):
            moran.compute(values_nan, coords_valid)

    def test_getis_ord_g_invalid_weights(self):
        """Test Getis-Ord G* with invalid weights matrix."""
        values = np.array([1, 2, 3])
        weights = np.eye(2)  # Wrong size

        with pytest.raises(ValueError):
            getis_ord_g(values, weights)

    def test_ripley_k_invalid_inputs(self):
        """Test Ripley's K with invalid inputs."""
        points = np.array([[0, 0]])  # Single point
        distances = [1, 2, 3]
        area = -100  # Invalid area

        with pytest.raises(ValueError):
            ripley_k(points, distances, area)


class TestNumericalStability:
    """Test numerical stability of spatial statistics functions."""

    def test_moran_i_numerical_stability(self):
        """Test Moran's I numerical stability with large datasets."""
        # Create large dataset
        np.random.seed(42)
        n = 1000
        coords = np.random.rand(n, 2) * 100
        values = np.random.randn(n) * 10 + 50

        moran = MoranI()
        result = moran.compute(values, coords)

        # Results should be finite and reasonable
        assert np.isfinite(result["I"])
        assert -1 <= result["I"] <= 1
        assert 0 <= result["p_value"] <= 1

    def test_getis_ord_g_numerical_stability(self):
        """Test Getis-Ord G* numerical stability."""
        # Create dataset with extreme values
        values = np.array([0, 1e10, 0, 1e-10])
        weights = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

        result = getis_ord_g(values, weights)

        # Results should be finite
        assert np.all(np.isfinite(result["z_scores"]))
        assert np.isfinite(result["global_g"])


class TestPerformance:
    """Test performance of spatial statistics functions."""

    def test_moran_i_performance_large_dataset(self):
        """Test Moran's I performance with large dataset."""
        # Create moderately large dataset
        np.random.seed(42)
        n = 500
        coords = np.random.rand(n, 2) * 100
        values = np.random.randn(n)

        import time

        start_time = time.time()

        moran = MoranI()
        result = moran.compute(values, coords)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert execution_time < 5.0
        assert result["I"] is not None


class TestAPIIntegration:
    """Test API integration and data models."""

    def test_api_request_response_models(self):
        """Test API request/response data models."""
        from geo_infer_math.api.spatial_analysis import (
            DescriptiveStatsRequest,
            DescriptiveStatsResponse,
        )

        # Test request models
        desc_req = DescriptiveStatsRequest(
            data={"features": []}, variables=["value"], statistics=["mean", "std"]
        )
        assert desc_req.data == {"features": []}
        assert desc_req.variables == ["value"]

        # Test response models
        desc_resp = DescriptiveStatsResponse(statistics={"mean": 10.5, "std": 2.1})
        assert desc_resp.statistics["mean"] == 10.5

    def test_api_endpoint_simulation(self):
        """Test API endpoint functionality without HTTP server."""
        from geo_infer_math.api.spatial_analysis import SpatialAnalysisAPI

        api = SpatialAnalysisAPI(verbose=True)

        # Test descriptive statistics endpoint
        test_data = {
            "data": {
                "features": [
                    {
                        "geometry": {"coordinates": [10, 20]},
                        "properties": {"value": 15},
                    },
                    {
                        "geometry": {"coordinates": [11, 21]},
                        "properties": {"value": 25},
                    },
                ]
            },
            "variables": ["value"],
            "statistics": ["mean", "std"],
        }

        result = api.calculate_descriptive_stats(test_data)
        assert "statistics" in result
        assert "mean" in result["statistics"]
        assert "std" in result["statistics"]

    def test_api_error_handling(self):
        """Test API error handling."""
        from werkzeug.exceptions import BadRequest

        from geo_infer_math.api.spatial_analysis import SpatialAnalysisAPI

        api = SpatialAnalysisAPI()

        # Test with invalid data
        invalid_data = {
            "data": {
                "features": [
                    {
                        "geometry": {"coordinates": []},  # Empty coordinates
                        "properties": {"value": 15},
                    }
                ]
            }
        }

        # Invalid data surfaces as an HTTP 400 BadRequest
        with pytest.raises(BadRequest):
            api.calculate_descriptive_stats(invalid_data)


class TestVerifiedStatistics:
    """Tests pinning real implementations over simplified placeholders."""

    def test_nearest_neighbor_p_value_matches_exact_gaussian_tail(self):
        """NN p-value equals 2 * scipy.stats.norm.sf(|z|) on a known case."""
        from scipy.stats import norm
        from geo_infer_math.api.spatial_analysis import SpatialAnalysisAPI

        points = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        api = SpatialAnalysisAPI()
        result = api.point_pattern_analysis(points, method="nearest_neighbor", area=4.0)

        expected_p = 2 * norm.sf(abs(result["z_score"]))
        assert abs(result["p_value"] - expected_p) < 1e-12
        assert 0.0 <= result["p_value"] <= 1.0

    def test_geary_p_value_on_small_grid(self):
        """Geary's C permutational variance yields a valid z and p."""
        from geo_infer_math.core.spatial_statistics import GearysC

        values = np.array([10.0, 12.0, 11.0, 13.0, 50.0, 52.0, 51.0, 53.0])
        coords = np.array(
            [
                [1, 1],
                [1, 2],
                [2, 1],
                [2, 2],
                [10, 10],
                [10, 11],
                [11, 10],
                [11, 11],
            ]
        )
        result = GearysC(rng=0).compute(values, coords)

        assert 0 <= result["p_value"] <= 1
        assert result["var_C"] > 0
        assert (
            abs(result["z_score"] - (result["C"] - 1.0) / np.sqrt(result["var_C"]))
            < 1e-12
        )
        # Strong positive autocorrelation -> C well below 1 -> negative z
        assert result["z_score"] < 0

    def test_morans_i_variance_cliff_ord_known_value(self):
        """Shared Cliff-Ord variance matches an independent loop computation."""
        from geo_infer_math.core.spatial_statistics import morans_i_variance

        values = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 6.0])
        weights = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        n = len(values)
        z = values - values.mean()
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    s0 += weights[i, j]
                    s1 += 0.5 * (weights[i, j] + weights[j, i]) ** 2
        for k in range(n):
            s2 += (weights[k, :].sum() + weights[:, k].sum()) ** 2
        b2 = n * np.sum(z**4) / (np.sum(z**2) ** 2)
        expected = (
            n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * s0**2)
            - b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * s0**2)
        ) / ((n - 1) * (n - 2) * (n - 3) * s0**2) - 1.0 / (n - 1) ** 2

        assert abs(morans_i_variance(values, weights) - expected) < 1e-10
