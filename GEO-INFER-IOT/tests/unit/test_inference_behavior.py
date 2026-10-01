"""Behavioral tests for BayesianSpatialInference (GS19-12).

Covers the success path and every error path of infer_spatial_distribution
(the outer catch previously swallowed without logging) and the exact-normal
z-score contract of get_posterior_map (previously a two-value hardcode).
"""

import logging
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import norm

from geo_infer_iot.core.inference import BayesianSpatialInference

MODULE_LOGGER = "geo_infer_iot.core.inference"


class StubGP:
    """Deterministic Gaussian-process double."""

    def fit(self, coords, values):
        self.fitted_coords = coords
        self.fitted_values = values

    def predict(self, grid, return_std=True):
        return np.full(len(grid), 5.0), np.full(len(grid), 1.0)


class FailingGP:
    def fit(self, coords, values):
        raise RuntimeError("gp exploded")


def _engine():
    return BayesianSpatialInference(
        variable="temperature", spatial_resolution=8, temporal_window="1h"
    )


def _sensor_data(n=4):
    return [
        {"latitude": 40.0 + i, "longitude": -74.0 + i, "value": float(i)}
        for i in range(n)
    ]


def _engine_with_cache(mean=2.0, std=0.5):
    engine = _engine()
    engine.posterior_cache[engine.variable] = {
        "posterior_mean": np.full(2, mean),
        "posterior_std": np.full(2, std),
        "h3_grid": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "sensor_coords": np.zeros((2, 2)),
        "sensor_values": np.zeros(2),
        "timestamp": datetime.now(UTC),
        "update_interval": "15min",
    }
    return engine


class TestInferSpatialDistribution:
    def test_success_path(self):
        engine = _engine()
        engine.gp_model = StubGP()
        result = engine.infer_spatial_distribution(_sensor_data())
        assert result["success"] is True
        assert result["sensor_count"] == 4
        assert result["prediction_points"] > 0
        assert all(p == 5.0 for p in result["posterior_mean"])
        assert all(s == 1.0 for s in result["posterior_std"])

    def test_default_engine_uses_real_bayes_gaussian_process(self):
        from geo_infer_bayes import GaussianProcess

        engine = _engine()
        assert isinstance(engine.gp_model, GaussianProcess)
        data = [
            {"latitude": 40.0 + 0.01 * i, "longitude": -74.0 + 0.01 * i, "value": i}
            for i in range(5)
        ]
        result = engine.infer_spatial_distribution(data)
        assert result["success"] is True
        assert result["sensor_count"] == 5
        assert len(result["posterior_mean"]) == result["prediction_points"] > 0
        assert all(np.isfinite(result["posterior_std"]))

    def test_error_on_insufficient_data(self):
        engine = _engine()
        engine.gp_model = StubGP()
        result = engine.infer_spatial_distribution(_sensor_data(2))
        assert result == {"error": "Insufficient data for spatial inference"}

    def test_error_on_empty_prediction_grid(self):
        engine = _engine()
        engine.gp_model = StubGP()
        engine._generate_h3_prediction_grid = lambda coords: np.array([])
        result = engine.infer_spatial_distribution(_sensor_data())
        assert result == {"error": "Failed to generate prediction grid"}

    def test_gp_failure_returns_error_dict_and_logs(self, caplog):
        engine = _engine()
        engine.gp_model = FailingGP()
        with caplog.at_level(logging.ERROR, logger=MODULE_LOGGER):
            result = engine.infer_spatial_distribution(_sensor_data())
        assert result["error"].startswith("Spatial inference failed:")
        assert any(
            "Spatial inference failed" in record.getMessage()
            for record in caplog.records
        )


class TestPosteriorMapConfidenceBounds:
    def test_bounds_match_exact_normal_quantiles(self):
        engine = _engine_with_cache(mean=2.0, std=0.5)
        result = engine.get_posterior_map([0.68, 0.95])
        bounds = result["confidence_bounds"]
        for ci in (0.68, 0.95):
            z = norm.ppf(0.5 + ci / 2.0)
            assert np.allclose(bounds[ci]["lower"], [2.0 - z * 0.5, 2.0 - z * 0.5])
            assert np.allclose(bounds[ci]["upper"], [2.0 + z * 0.5, 2.0 + z * 0.5])

    def test_default_confidence_intervals(self):
        engine = _engine_with_cache()
        result = engine.get_posterior_map()
        assert set(result["confidence_bounds"]) == {0.68, 0.95}

    def test_non_standard_ci_gets_its_own_quantile(self):
        """Regression: any CI other than 0.95 used to silently get z=1.0."""
        engine = _engine_with_cache(mean=2.0, std=0.5)
        bounds = engine.get_posterior_map([0.50])["confidence_bounds"]
        z = norm.ppf(0.75)
        assert np.allclose(bounds[0.50]["lower"], [2.0 - z * 0.5, 2.0 - z * 0.5])
        assert not np.isclose(bounds[0.50]["lower"][0], 2.0 - 1.0 * 0.5)

    def test_out_of_range_ci_raises(self):
        engine = _engine_with_cache()
        with pytest.raises(ValueError, match="Confidence interval"):
            engine.get_posterior_map([1.5])

    def test_missing_posterior_returns_error(self):
        engine = _engine()
        result = engine.get_posterior_map([0.95])
        assert result == {"error": "No posterior distribution available"}
